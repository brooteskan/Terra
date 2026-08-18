//! Timing guard for issue #122: the whole-field CPU stroke apply
//! (`authoring::apply_sculpt_strokes`) culls each stroke to its padded bounding
//! box, so its cost scales ~O(Σ stroke footprints) plus an irreducible O(field)
//! floor (two field clones, the reconcile pass, five aux `MaskField`s) — NOT
//! O(strokes × field) as the pre-#122 code did.
//!
//! Usage: `cargo run --release --example sculpt_stamp_cull -- [max_res]`
//!
//! Run it at the commit *before* #122 and *after* to watch the many-small-strokes
//! rows drop by orders of magnitude while the O(field) floor (and the degenerate
//! field-covering single stroke, whose bbox is the whole field) stays put.

use std::hint::black_box;
use std::time::Instant;

use terra_core::authoring::{
    apply_sculpt_strokes, SculptPoint, SculptStroke, SculptStrokeKind, SculptStrokeParams,
};
use terra_core::heightfield::{Heightfield, HeightfieldMetrics};

const WORLD: f32 = 4096.0;

/// `count` small strokes scattered deterministically (a tiny LCG — no `rand`
/// dependency), each ~1% of the world across, a two-point polyline. Every fifth
/// is a Flatten so the footprint-mean scan is exercised too.
fn small_strokes(count: u32) -> Vec<SculptStroke> {
    let mut lcg = 0x9E37_79B9u32;
    let mut next = || {
        lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (lcg >> 8) as f32 / (1u32 << 24) as f32
    };
    let radius = WORLD * 0.01;
    (0..count)
        .map(|k| {
            let u = 0.05 + next() * 0.9;
            let v = 0.05 + next() * 0.9;
            let kind = if k % 5 == 0 {
                SculptStrokeKind::Flatten
            } else {
                SculptStrokeKind::Raise
            };
            SculptStroke {
                kind,
                points: vec![
                    SculptPoint {
                        u,
                        v,
                        pressure: 1.0,
                    },
                    SculptPoint {
                        u: (u + 0.01).min(0.99),
                        v: (v + 0.01).min(0.99),
                        pressure: 0.8,
                    },
                ],
                radius_m: radius,
                strength: 5.0,
                target_height: 8.0,
                falloff: 1.5,
                enabled: true,
            }
        })
        .collect()
}

/// A single stroke whose padded bbox covers the whole field: the cull degenerates
/// to the pre-#122 single-stroke cost (graceful, not wrong).
fn field_covering_stroke() -> Vec<SculptStroke> {
    vec![SculptStroke {
        kind: SculptStrokeKind::Raise,
        points: vec![SculptPoint {
            u: 0.5,
            v: 0.5,
            pressure: 1.0,
        }],
        radius_m: WORLD * 1.5,
        strength: 5.0,
        target_height: 0.0,
        falloff: 1.5,
        enabled: true,
    }]
}

fn ramp(m: HeightfieldMetrics) -> Heightfield {
    let mut h = Heightfield::zeros(m);
    for j in 0..m.height {
        for i in 0..m.width {
            h.set(i, j, (i as f32) * 0.1 + (j as f32) * 0.07);
        }
    }
    h
}

fn time_case(m: HeightfieldMetrics, strokes: Vec<SculptStroke>, label: &str) {
    let p = SculptStrokeParams {
        strokes,
        reconcile: 0.15,
    };
    let h = ramp(m);
    // Warm up caches / branch predictors so the reported time is steady-state.
    black_box(apply_sculpt_strokes(&h, &p));
    let t = Instant::now();
    let r = apply_sculpt_strokes(&h, &p);
    let dt = t.elapsed();
    black_box(&r);
    eprintln!("  {label:<28} {}x{} -> {:?}", m.width, m.height, dt);
}

fn main() {
    let max_res: u32 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(2048);

    for res in [512u32, 1024, 2048] {
        if res > max_res {
            break;
        }
        let m = HeightfieldMetrics::new(res, res, WORLD, WORLD);
        eprintln!("res {res}:");
        for count in [16u32, 64, 256] {
            time_case(m, small_strokes(count), &format!("{count} small strokes"));
        }
        time_case(m, field_covering_stroke(), "1 field-covering stroke");
    }
}
