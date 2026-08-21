//! Scratch probe (issue diagnosis, not committed): load a real project and time a
//! scoped SculptStrokes stroke edit at Full resolution, with the per-layer
//! breakdown the worker already collects. Answers where a per-stroke eval spends
//! its time — the profiling #112's gate demands before touching the delivery path.
//!
//! Usage: cargo run --release -p terra-io --example sculpt_eval_probe -- <project.json>

use std::collections::HashMap;
use std::time::Instant;

use terra_core::authoring::{stroke_footprint_uv, SculptPoint, SculptStroke, SculptStrokeKind};
use terra_core::heightfield::Heightfield;
use terra_core::layer::LayerKind;
use terra_core::mask::bake_mask_assets;
use terra_core::quality::PreviewQuality;
use terra_core::tiling::{tiles_for_uv_rect, UvRect};
use terra_cpu_eval::{EvalContext, StackEvaluator};

fn print_timings(evaluator_label: &str, us: u128, ctx: &EvalContext) {
    eprintln!("{evaluator_label}: total {:.1} ms", us as f64 / 1000.0);
    for t in &ctx.layer_timings {
        let scope = match t.tiles_recomputed {
            Some(n) => format!("scoped {n} tiles"),
            None => "WHOLE-FIELD/cache".to_string(),
        };
        if t.elapsed_us > 200 {
            eprintln!(
                "    {:>8.2} ms  {:<16} {:<20} [{scope}] {:?}",
                t.elapsed_us as f64 / 1000.0,
                t.layer_kind,
                t.layer_name,
                t.status,
            );
        }
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: sculpt_eval_probe <project.json>");
    let mut doc = terra_io::load_project(std::path::Path::new(&path)).expect("load project");

    let preview_res = doc.preview_resolution.min(8192);
    let metrics = doc
        .metrics
        .at_resolution(preview_res)
        .expect("metrics at preview res");
    eprintln!(
        "project {}x{}  tile_size {}  (Full preview)",
        metrics.width, metrics.height, metrics.tile_size
    );

    let level_steps = doc.level_steps.clone();
    let mask_assets = doc.masks.clone();
    let make_ctx = || {
        let mut ctx = EvalContext::new(metrics);
        ctx.quality = PreviewQuality::Full;
        ctx.level_steps = level_steps.clone();
        ctx.mask_assets = mask_assets.clone();
        ctx.set_aux_hashmap(HashMap::new());
        let seed = Heightfield::zeros(metrics);
        ctx.masks = bake_mask_assets(&mask_assets, &seed, metrics, &HashMap::new());
        ctx
    };

    // Identify the SculptStrokes layer we'll edit.
    let sculpt_id = {
        let stack = doc.preview_eval_stack();
        stack
            .flatten_layers()
            .iter()
            .find(|l| matches!(l.kind, LayerKind::SculptStrokes(_)))
            .map(|l| l.id())
            .expect("project has a SculptStrokes layer")
    };

    // --- Warm the evaluator cache with a cold Full build. ---
    let mut evaluator = StackEvaluator::new();
    {
        let stack = doc.preview_eval_stack();
        evaluator.mark_all_dirty(&stack);
        let mut ctx = make_ctx();
        let t = Instant::now();
        evaluator
            .rebuild_incremental(&stack, &mut ctx)
            .expect("cold full build");
        print_timings("COLD full build", t.elapsed().as_micros(), &ctx);
    }

    // --- Now paint one pinch stroke and time the scoped incremental. ---
    let (u, v, radius_uv) = (0.5f32, 0.5f32, 0.03f32);
    let world_radius = radius_uv * 0.5 * (doc.metrics.world_size_x + doc.metrics.world_size_z);
    let new_stroke = SculptStroke {
        kind: SculptStrokeKind::Pinch,
        points: vec![SculptPoint {
            u,
            v,
            pressure: 1.0,
        }],
        radius_m: world_radius.max(1.0),
        strength: 0.3,
        target_height: 0.0,
        falloff: 1.5,
        enabled: true,
    };
    let footprint = stroke_footprint_uv(&new_stroke, &metrics).expect("stroke footprint");
    if let Some(LayerKind::SculptStrokes(params)) =
        doc.stack.find_mut(sculpt_id).map(|l| &mut l.kind)
    {
        params.strokes.push(new_stroke);
    }

    let rect = UvRect {
        min_u: footprint.min_u,
        min_v: footprint.min_v,
        max_u: footprint.max_u,
        max_v: footprint.max_v,
    };
    let tiles = tiles_for_uv_rect(&metrics, rect);
    eprintln!(
        "\npinch stroke footprint -> {} scope tiles (of {} total)",
        tiles.len(),
        metrics.tiles_x() * metrics.tiles_z()
    );

    let stack = doc.preview_eval_stack();
    evaluator.mark_dirty_from_region(&stack, sculpt_id, &tiles);
    let mut ctx = make_ctx();
    let t = Instant::now();
    evaluator
        .rebuild_incremental(&stack, &mut ctx)
        .expect("scoped stroke rebuild");
    print_timings("SCOPED pinch stroke", t.elapsed().as_micros(), &ctx);
}
