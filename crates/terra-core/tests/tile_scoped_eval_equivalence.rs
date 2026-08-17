//! #100 phase-2 equivalence oracle.
//!
//! The tile-scoped suffix walk in `StackEvaluator::rebuild_incremental` must be
//! *indistinguishable* from a whole-field rebuild: bit-identical composed height,
//! zero seam error, and — the point of the exercise — it must recompute only the
//! tiles a localized edit actually reaches, escalating to whole-field exactly when
//! a layer's reach or a basin coupling says it must.
//!
//! Two grounds of truth are used. Most tests hold params fixed and compare the
//! scoped rebuild to a from-scratch `rebuild_all` (the whole-field truth), with
//! `tiles_recomputed` counters proving it genuinely scoped rather than silently
//! recomputing everything. One test applies a real bounded paint edit and compares
//! to a cold `rebuild_all` of the mutated stack, proving no clean tile is carried
//! forward stale.
//!
//! Fields use a small `tile_size` (16) and a modest blur radius so the O(r^2)
//! box blur stays cheap while still reaching exactly one tile; resolutions are
//! deliberately not always multiples of the tile size so partial edge tiles are
//! exercised.

use std::collections::HashSet;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use terra_core::eval::{EvalContext, EvalError, LayerEvalTiming, PreviewQuality, StackEvaluator};
use terra_core::heightfield::{Heightfield, HeightfieldMetrics, TileId};
use terra_core::layer::{
    BlurParams, CoastalParams, Layer, LayerId, LayerKind, LayerStack, PathNode, PathParams,
    PlateauParams, SculptParams,
};
use terra_core::tiling::measure_seams;

const TS: u32 = 16;

fn metrics(res: u32) -> HeightfieldMetrics {
    HeightfieldMetrics {
        width: res,
        height: res,
        world_size_x: res as f32,
        world_size_z: res as f32,
        tile_size: TS,
        halo: 2,
    }
}

/// A blur whose radius reaches exactly one tile (`TS` samples).
fn one_tile_blur(name: &str) -> Layer {
    Layer::new(
        name,
        LayerKind::Blur(BlurParams {
            radius: TS,
            iterations: 1,
        }),
    )
}

/// A SculptBase whose paint buffer varies across the field, so tiles differ and a
/// tile-boundary bug cannot hide behind a flat buffer.
fn sculpt_gradient(res: u32) -> Layer {
    let mut p = SculptParams::filled(res, 0.0);
    for j in 0..res {
        for i in 0..res {
            let v =
                20.0 + (i as f32) * 0.15 + (j as f32) * 0.11 + ((i * 7 + j * 13) % 23) as f32;
            p.samples[(j * res + i) as usize] = v;
        }
    }
    Layer::new("Sculpt", LayerKind::SculptBase(p))
}

fn all_tiles(m: &HeightfieldMetrics) -> Vec<TileId> {
    let mut v = Vec::new();
    for tz in 0..m.tiles_z() {
        for tx in 0..m.tiles_x() {
            v.push(TileId { tx, tz });
        }
    }
    v
}

/// Tiles covering the inclusive sample rectangle `[x0..=x1] x [y0..=y1]`.
fn tiles_in_samples(m: &HeightfieldMetrics, x0: u32, y0: u32, x1: u32, y1: u32) -> Vec<TileId> {
    let ts = m.tile_size;
    let mut v = Vec::new();
    for tz in (y0 / ts)..=(y1 / ts) {
        for tx in (x0 / ts)..=(x1 / ts) {
            v.push(TileId { tx, tz });
        }
    }
    v
}

fn bits(h: &Heightfield) -> Vec<u32> {
    h.to_dense().iter().map(|f| f.to_bits()).collect()
}

/// Tiles whose interior differs between two fields (exact, halo-independent).
fn changed_tiles(a: &Heightfield, b: &Heightfield) -> Vec<TileId> {
    let m = a.metrics;
    let mut set = HashSet::new();
    for j in 0..m.height {
        for i in 0..m.width {
            if a.get(i, j).to_bits() != b.get(i, j).to_bits() {
                set.insert(TileId {
                    tx: i / m.tile_size,
                    tz: j / m.tile_size,
                });
            }
        }
    }
    set.into_iter().collect()
}

fn timing(timings: &[LayerEvalTiming], id: LayerId) -> &LayerEvalTiming {
    timings
        .iter()
        .find(|t| t.layer == id)
        .expect("layer appears in timings")
}

fn wetness_bits(ctx: &EvalContext) -> Vec<u32> {
    ctx.aux_maps
        .wetness
        .as_ref()
        .map(|w| w.data().iter().map(|f| f.to_bits()).collect())
        .unwrap_or_default()
}

fn flat_stack(layers: Vec<Layer>) -> (LayerStack, Vec<LayerId>) {
    let mut stack = LayerStack::new();
    let mut ids = Vec::new();
    for layer in layers {
        ids.push(layer.id());
        stack.push(layer);
    }
    (stack, ids)
}

/// Full rebuild, then whole-field incremental after `mark_dirty_from(base)`.
fn whole_field_control(stack: &LayerStack, base: LayerId, m: HeightfieldMetrics) -> Heightfield {
    let mut eval = StackEvaluator::new();
    let mut ctx = EvalContext::new(m);
    eval.rebuild_all(stack, &mut ctx).expect("rebuild_all");
    eval.mark_dirty_from(stack, base);
    let mut ctx2 = EvalContext::new(m);
    eval.rebuild_incremental(stack, &mut ctx2)
        .expect("whole-field incremental")
}

/// Full rebuild, then tile-scoped incremental after `mark_dirty_from_region`.
fn scoped_rebuild(
    stack: &LayerStack,
    base: LayerId,
    region: &[TileId],
    m: HeightfieldMetrics,
) -> (Heightfield, Vec<LayerEvalTiming>) {
    let mut eval = StackEvaluator::new();
    let mut ctx = EvalContext::new(m);
    eval.rebuild_all(stack, &mut ctx).expect("rebuild_all");
    eval.mark_dirty_from_region(stack, base, region);
    let mut ctx2 = EvalContext::new(m);
    let height = eval
        .rebuild_incremental(stack, &mut ctx2)
        .expect("scoped incremental");
    (height, ctx2.layer_timings)
}

/// The pool of flat, localizable stacks. Each entry names its layers so a failure
/// points at the arm. All arms here are tile-wired or per-texel: SculptBase,
/// Plateau, Coastal, Blur.
fn pool(res: u32) -> Vec<(&'static str, Vec<Layer>)> {
    let plateau = || {
        Layer::new(
            "Plateau",
            LayerKind::Plateau(PlateauParams {
                low: 25.0,
                high: 45.0,
                soft: 6.0,
            }),
        )
    };
    let coastal = || {
        Layer::new(
            "Coastal",
            LayerKind::Coastal(CoastalParams {
                sea_level: 30.0,
                beach_width: 8.0,
                flatten_below: true,
                shelf_depth: 4.0,
            }),
        )
    };
    vec![
        ("sculpt-only", vec![sculpt_gradient(res)]),
        ("sculpt+plateau", vec![sculpt_gradient(res), plateau()]),
        ("sculpt+coastal", vec![sculpt_gradient(res), coastal()]),
        (
            "sculpt+plateau+coastal",
            vec![sculpt_gradient(res), plateau(), coastal()],
        ),
        ("sculpt+blur", vec![sculpt_gradient(res), one_tile_blur("Blur")]),
        (
            "sculpt+blur+plateau",
            vec![sculpt_gradient(res), one_tile_blur("Blur"), plateau()],
        ),
    ]
}

#[test]
fn scoped_matches_whole_field_across_stacks_and_rects() {
    // 72 is not a multiple of 16, so the last tile row/column is a partial edge
    // tile (5x5 grid, last tiles 8 samples wide).
    let res = 72;
    let m = metrics(res);

    let rects: Vec<(&str, Vec<TileId>)> = vec![
        ("single-interior", tiles_in_samples(&m, 34, 34, 34, 34)),
        ("corner-crossing", tiles_in_samples(&m, 28, 28, 36, 36)),
        ("field-edge-partial", tiles_in_samples(&m, res - 3, res - 3, res - 1, res - 1)),
        ("whole-field", all_tiles(&m)),
    ];

    for (stack_name, layers) in pool(res) {
        let (stack, ids) = flat_stack(layers);
        let base = ids[0];

        // One from-scratch whole-field rebuild is the oracle's ground truth; the
        // evaluator is then reused across rects (each scoped rebuild stores a full,
        // clean field, so the next region-mark starts coherent).
        let mut eval = StackEvaluator::new();
        let mut ctx = EvalContext::new(m);
        let truth = eval.rebuild_all(&stack, &mut ctx).expect("rebuild_all");

        for (rect_name, region) in &rects {
            eval.mark_dirty_from_region(&stack, base, region);
            let mut ctx_i = EvalContext::new(m);
            let scoped = eval
                .rebuild_incremental(&stack, &mut ctx_i)
                .expect("scoped incremental");
            assert_eq!(
                bits(&scoped),
                bits(&truth),
                "scoped != from-scratch rebuild for {stack_name} / {rect_name}"
            );
            assert_eq!(
                measure_seams(&scoped),
                0.0,
                "scoped left a seam for {stack_name} / {rect_name}"
            );
        }
    }
}

#[test]
fn scoped_recompute_counts_match_reach() {
    let res = 64; // 4x4 tiles
    let m = metrics(res);

    // A pure per-texel stack: every layer recomputes exactly the edit rect.
    let (stack, ids) = flat_stack(vec![
        sculpt_gradient(res),
        Layer::new(
            "Plateau",
            LayerKind::Plateau(PlateauParams {
                low: 25.0,
                high: 45.0,
                soft: 6.0,
            }),
        ),
    ]);
    let region = tiles_in_samples(&m, 34, 34, 34, 34); // single interior tile
    assert_eq!(region.len(), 1);
    let (_, timings) = scoped_rebuild(&stack, ids[0], &region, m);
    for id in &ids {
        assert_eq!(
            timing(&timings, *id).tiles_recomputed,
            Some(1),
            "per-texel layer should recompute exactly the edit tile"
        );
    }

    // A whole-field rect escalates: the base recomputes whole-field (None).
    let (_, timings) = scoped_rebuild(&stack, ids[0], &all_tiles(&m), m);
    assert_eq!(
        timing(&timings, ids[0]).tiles_recomputed,
        None,
        "a full-field edit escalates the base to whole-field"
    );
}

#[test]
fn stacked_blur_reach_accumulates() {
    let res = 112; // 7x7 = 49 tiles, so a 5x5 expansion does not fill the field
    let m = metrics(res);

    // Two one-tile blurs. A single-tile edit expands 1 -> 9 -> 25: the second blur
    // reaches a tile two rings out, which the first blur alone never touches —
    // cumulative reach, not per-layer.
    let (stack, ids) = flat_stack(vec![
        sculpt_gradient(res),
        one_tile_blur("Blur1"),
        one_tile_blur("Blur2"),
    ]);
    let region = tiles_in_samples(&m, 50, 50, 50, 50); // tile (3,3), interior
    assert_eq!(region, vec![TileId { tx: 3, tz: 3 }]);

    let (scoped, timings) = scoped_rebuild(&stack, ids[0], &region, m);
    assert_eq!(timing(&timings, ids[0]).tiles_recomputed, Some(1), "sculpt");
    assert_eq!(timing(&timings, ids[1]).tiles_recomputed, Some(9), "blur1: 3x3");
    assert_eq!(timing(&timings, ids[2]).tiles_recomputed, Some(25), "blur2: 5x5");

    let control = whole_field_control(&stack, ids[0], m);
    assert_eq!(bits(&scoped), bits(&control));
    assert_eq!(measure_seams(&scoped), 0.0);
}

#[test]
fn basin_dependent_escalates_from_that_layer_up() {
    let res = 96; // 6x6 tiles
    let m = metrics(res);

    // Sculpt -> Blur -> ThermalErosion (basin-coupled, Reach::Full) -> Blur.
    // The scoped walk must recompute sculpt/blur1 tile-scoped, then escalate to
    // whole-field at the erosion and stay there for the blur above it.
    let (stack, ids) = flat_stack(vec![
        sculpt_gradient(res),
        one_tile_blur("Blur1"),
        Layer::new("Thermal", LayerKind::ThermalErosion(Default::default())),
        one_tile_blur("Blur2"),
    ]);
    let region = tiles_in_samples(&m, 34, 34, 34, 34); // tile (2,2)

    let mut eval = StackEvaluator::new();
    let mut ctx = EvalContext::new(m);
    ctx.quality = PreviewQuality::Draft; // keep erosion cheap
    eval.rebuild_all(&stack, &mut ctx).expect("rebuild_all");
    eval.mark_dirty_from_region(&stack, ids[0], &region);
    let mut ctx2 = EvalContext::new(m);
    ctx2.quality = PreviewQuality::Draft;
    let scoped = eval.rebuild_incremental(&stack, &mut ctx2).expect("scoped");
    let t = &ctx2.layer_timings;

    assert!(timing(t, ids[0]).tiles_recomputed.is_some(), "sculpt scoped");
    assert!(timing(t, ids[1]).tiles_recomputed.is_some(), "blur1 scoped");
    assert_eq!(
        timing(t, ids[2]).tiles_recomputed,
        None,
        "basin-coupled erosion escalates to whole-field"
    );
    assert_eq!(
        timing(t, ids[3]).tiles_recomputed,
        None,
        "escalation is sticky: the blur above erosion is whole-field too"
    );

    // Control: same stack, whole-field incremental.
    let mut cval = StackEvaluator::new();
    let mut cctx = EvalContext::new(m);
    cctx.quality = PreviewQuality::Draft;
    cval.rebuild_all(&stack, &mut cctx).expect("rebuild_all");
    cval.mark_dirty_from(&stack, ids[0]);
    let mut cctx2 = EvalContext::new(m);
    cctx2.quality = PreviewQuality::Draft;
    let control = cval.rebuild_incremental(&stack, &mut cctx2).expect("control");
    assert_eq!(bits(&scoped), bits(&control), "scoped != whole-field");
    assert_eq!(measure_seams(&scoped), 0.0);
}

#[test]
fn bounded_paint_edit_matches_cold_rebuild() {
    // The strongest check: a genuine bounded edit, no clean tile carried stale.
    let res = 96; // 6x6 tiles
    let m = metrics(res);

    let base_layer = sculpt_gradient(res);
    let sculpt_id = base_layer.id();
    let (stack, ids) = flat_stack(vec![
        base_layer,
        one_tile_blur("Blur1"),
        one_tile_blur("Blur2"),
    ]);

    // Evaluator S holds the original cache.
    let mut s = StackEvaluator::new();
    let mut sctx = EvalContext::new(m);
    let original = s.rebuild_all(&stack, &mut sctx).expect("rebuild_all");

    // Mutate the sculpt paint inside tile (3,3)'s interior (samples 51..61, clear
    // of the 48/64 tile boundaries so the bilinear footprint stays in the one
    // tile). Both blurs then genuinely scope: 1 -> 9 -> 25 tiles on a 6x6 grid.
    let edit = |p: &mut SculptParams| {
        for j in 51..61 {
            for i in 51..61 {
                p.samples[(j * res + i) as usize] += 60.0;
            }
        }
    };

    let mut mutated = stack.clone();
    if let Some(layer) = mutated.find_mut(sculpt_id) {
        if let LayerKind::SculptBase(p) = &mut layer.kind {
            edit(p);
        }
    }

    // Ground truth: cold rebuild of the mutated stack.
    let mut cold = StackEvaluator::new();
    let mut cctx = EvalContext::new(m);
    let truth = cold.rebuild_all(&mutated, &mut cctx).expect("cold rebuild");

    // The sculpt layer's own changed footprint (self-calibrated): eval the sculpt
    // alone before/after so we mark exactly what the sculpt changed, then let the
    // walk expand for the two blurs. A reach under-expansion would carry a stale
    // tile and diverge from `truth`.
    let sculpt_footprint = {
        let (orig_only, _) = flat_stack(vec![sculpt_gradient(res)]);
        let mut e = StackEvaluator::new();
        let mut c = EvalContext::new(m);
        let a = e.rebuild_all(&orig_only, &mut c).unwrap();
        let mut mut_only = orig_only.clone();
        let mo_id = mut_only.layer_ids()[0];
        if let Some(layer) = mut_only.find_mut(mo_id) {
            if let LayerKind::SculptBase(p) = &mut layer.kind {
                edit(p);
            }
        }
        let mut e2 = StackEvaluator::new();
        let mut c2 = EvalContext::new(m);
        let b = e2.rebuild_all(&mut_only, &mut c2).unwrap();
        changed_tiles(&a, &b)
    };
    assert!(!sculpt_footprint.is_empty(), "the paint edit changed something");
    assert!(
        changed_tiles(&original, &truth).len() > sculpt_footprint.len(),
        "the two blurs must widen the footprint beyond the sculpt's own tiles"
    );

    // Scoped incremental on S (its cache is the original), marking only the
    // sculpt's own footprint and evaluating the mutated stack.
    s.mark_dirty_from_region(&stack, ids[0], &sculpt_footprint);
    let mut sctx2 = EvalContext::new(m);
    let scoped = s
        .rebuild_incremental(&mutated, &mut sctx2)
        .expect("scoped incremental");

    assert_eq!(
        bits(&scoped),
        bits(&truth),
        "a bounded paint edit diverged from a cold rebuild (stale carried tile?)"
    );
    assert_eq!(measure_seams(&scoped), 0.0);
}

#[test]
fn cancelled_scoped_job_publishes_nothing_and_leaves_seeds() {
    let res = 64;
    let m = metrics(res);
    let (stack, ids) = flat_stack(vec![sculpt_gradient(res), one_tile_blur("Blur")]);
    let region = tiles_in_samples(&m, 34, 34, 34, 34);

    let mut eval = StackEvaluator::new();
    let mut ctx = EvalContext::new(m);
    eval.rebuild_all(&stack, &mut ctx).expect("rebuild_all");
    eval.mark_dirty_from_region(&stack, ids[0], &region);

    // A pre-tripped cancellation: the walk must bail before storing anything.
    let mut cancel_ctx = EvalContext::new(m);
    cancel_ctx.set_cancellation_generation(Arc::new(AtomicU64::new(1)), 0);
    let result = eval.rebuild_incremental(&stack, &mut cancel_ctx);
    assert!(
        matches!(result, Err(EvalError::Cancelled)),
        "a cancelled scoped job returns Cancelled"
    );
    assert!(eval.cache.is_dirty(ids[0]), "seeds survive a cancelled job");

    // A subsequent uncancelled rebuild still matches the whole-field control.
    let mut ctx2 = EvalContext::new(m);
    let scoped = eval.rebuild_incremental(&stack, &mut ctx2).expect("rerun");
    let control = whole_field_control(&stack, ids[0], m);
    assert_eq!(bits(&scoped), bits(&control));
}

#[test]
#[ignore = "perf measurement; run with `--ignored --nocapture` to see the numbers"]
fn perf_single_stroke_scoped_is_well_below_whole_field() {
    use std::time::Instant;

    // A SculptBase-over-generators flat stack at 1024^2 with default 256 tiling
    // (4x4 = 16 tiles). Every layer here is tile-wired, so a single-tile stroke
    // recomputes 1/16 of the field instead of all of it.
    let res = 1024;
    let m = HeightfieldMetrics::new(res, res, res as f32, res as f32);
    let (stack, ids) = flat_stack(vec![
        sculpt_gradient(res),
        Layer::new(
            "Plateau",
            LayerKind::Plateau(PlateauParams {
                low: 25.0,
                high: 45.0,
                soft: 6.0,
            }),
        ),
        Layer::new(
            "Coastal",
            LayerKind::Coastal(CoastalParams {
                sea_level: 30.0,
                beach_width: 8.0,
                flatten_below: true,
                shelf_depth: 4.0,
            }),
        ),
    ]);

    // Whole-field incremental (today's behavior).
    let mut w = StackEvaluator::new();
    let mut wctx = EvalContext::new(m);
    w.rebuild_all(&stack, &mut wctx).unwrap();
    w.mark_dirty_from(&stack, ids[0]);
    let mut wctx2 = EvalContext::new(m);
    let t0 = Instant::now();
    let whole = w.rebuild_incremental(&stack, &mut wctx2).unwrap();
    let whole_us = t0.elapsed().as_micros();

    // Tile-scoped incremental: a single-tile stroke on the base.
    let mut s = StackEvaluator::new();
    let mut sctx = EvalContext::new(m);
    s.rebuild_all(&stack, &mut sctx).unwrap();
    let stroke = tiles_in_samples(&m, 300, 300, 300, 300); // one interior tile
    assert_eq!(stroke.len(), 1);
    s.mark_dirty_from_region(&stack, ids[0], &stroke);
    let mut sctx2 = EvalContext::new(m);
    let t1 = Instant::now();
    let scoped = s.rebuild_incremental(&stack, &mut sctx2).unwrap();
    let scoped_us = t1.elapsed().as_micros();

    // The adopted arms recomputed only the expanded (here single) tile set.
    for id in &ids {
        assert_eq!(
            timing(&sctx2.layer_timings, *id).tiles_recomputed,
            Some(1),
            "each tile-wired layer recomputes exactly the stroke tile"
        );
    }
    assert_eq!(bits(&scoped), bits(&whole), "scoped must equal whole-field");
    println!(
        "1024^2 single-stroke: whole-field {whole_us} us, scoped {scoped_us} us ({:.1}x)",
        whole_us as f64 / scoped_us.max(1) as f64
    );
    assert!(
        scoped_us.saturating_mul(2) < whole_us,
        "scoped ({scoped_us} us) should be well below whole-field ({whole_us} us)"
    );
}

#[test]
fn scoped_path_matches_whole_field_including_wetness() {
    // Path is the one wired arm that patches a per-texel aux (wetness). Verify the
    // scoped wetness merge equals the whole-field `merge_wetness_max`, not just the
    // height.
    let res = 96;
    let m = metrics(res);
    let path = Layer::new(
        "Path",
        LayerKind::Path(PathParams {
            nodes: vec![
                PathNode { u: 0.2, v: 0.45, height: -6.0, width: 1.0 },
                PathNode { u: 0.8, v: 0.55, height: -6.0, width: 1.0 },
            ],
            width: 10.0,
            falloff: 6.0,
            carve: true,
            spline: false,
            ..PathParams::default()
        }),
    );
    let (stack, ids) = flat_stack(vec![sculpt_gradient(res), path]);

    // Whole-field ground truth and its wetness aux.
    let mut w = StackEvaluator::new();
    let mut wctx = EvalContext::new(m);
    let truth = w.rebuild_all(&stack, &mut wctx).expect("rebuild_all");
    let truth_wet = wetness_bits(&wctx);
    assert!(!truth_wet.is_empty(), "a carving path must publish wetness");

    // Scoped incremental after a bounded edit on the sculpt below the path (which
    // carves through the edited tile), so the path recomputes tile-scoped and its
    // wetness patch is exercised on a tile with actual carving.
    let mut s = StackEvaluator::new();
    let mut sctx = EvalContext::new(m);
    s.rebuild_all(&stack, &mut sctx).expect("rebuild_all");
    let region = tiles_in_samples(&m, 34, 34, 40, 40);
    s.mark_dirty_from_region(&stack, ids[0], &region);
    let mut sctx2 = EvalContext::new(m);
    let scoped = s.rebuild_incremental(&stack, &mut sctx2).expect("scoped");

    assert!(
        timing(&sctx2.layer_timings, ids[1]).tiles_recomputed.is_some(),
        "the path layer should recompute tile-scoped, not escalate"
    );
    assert_eq!(bits(&scoped), bits(&truth), "scoped path height diverged");
    assert_eq!(
        wetness_bits(&sctx2),
        truth_wet,
        "scoped path wetness diverged from the whole-field merge"
    );
    assert_eq!(measure_seams(&scoped), 0.0);
}

#[test]
fn initial_scope_composes_with_region_marks() {
    let res = 80; // 5x5 tiles
    let m = metrics(res);
    let (stack, ids) = flat_stack(vec![sculpt_gradient(res), one_tile_blur("Blur")]);
    let region = tiles_in_samples(&m, 34, 34, 34, 34);

    let mut eval = StackEvaluator::new();
    let mut ctx = EvalContext::new(m);
    eval.rebuild_all(&stack, &mut ctx).expect("rebuild_all");
    eval.mark_dirty_from_region(&stack, ids[0], &region);

    // An extra initial scope (a disjoint tile) broadens the recompute set but must
    // not change the result: a superset scope recomputes clean tiles to identical
    // values.
    let mut ctx2 = EvalContext::new(m);
    ctx2.initial_scope = Some(vec![TileId { tx: 0, tz: 0 }]);
    let scoped = eval.rebuild_incremental(&stack, &mut ctx2).expect("scoped");

    let control = whole_field_control(&stack, ids[0], m);
    assert_eq!(bits(&scoped), bits(&control));
    assert_eq!(measure_seams(&scoped), 0.0);
}
