//! Timing guard for issue #123: the prefix-checkpoint cache
//! (`authoring::apply_sculpt_strokes_cached`) resumes an earlier stamp instead of
//! re-stamping the whole stroke set on every edit. On a few large multi-point
//! strokes — the regime the #122 bbox cull and #110/#121 tile scoping do NOT help
//! (footprints span 10–40 % of the field) — an appended stroke, or a drag growing
//! the last stroke, must cost ~O(one stroke) rather than O(Σ all strokes).
//!
//! Usage: `cargo run --release --example sculpt_prefix_resume -- [max_res]`
//!
//! Reports, per resolution, the wall time and re-stamped stroke count of:
//!   * cold — no entry, stamps all N strokes (the pre-#123 cost of *every* edit),
//!   * append — resume from the tail, stamp the one new stroke,
//!   * drag — resume from the pre-tail, stamp the one growing last stroke,
//!   * edit-mid — edit an interior stroke, stamp the suffix from there.
//!
//! `append` and `drag` should collapse to a small fraction of `cold`; `edit-mid`
//! improves in proportion to how much prefix it can reuse. (The per-stroke stamp
//! cost itself — `distance_to_polyline`'s O(points) per texel — is the separate
//! spatial-acceleration follow-up #123 names as out of scope.)

use std::hint::black_box;
use std::time::Instant;

use terra_core::authoring::{
    apply_sculpt_strokes_cached, SculptPoint, SculptPrefixEntry, SculptStroke, SculptStrokeKind,
    SculptStrokeParams,
};
use terra_core::heightfield::{Heightfield, HeightfieldMetrics};

const WORLD: f32 = 4096.0;

/// A deterministic tiny LCG (no `rand` dependency).
struct Lcg(u32);
impl Lcg {
    fn sample(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 8) as f32 / (1u32 << 24) as f32
    }
}

/// A large multi-point stroke: a wandering polyline of `points` vertices whose
/// brush radius is ~4 % of the world, so each footprint spans a big fraction of
/// the field — the many-points-few-strokes shape from the issue's measurement.
fn big_stroke(lcg: &mut Lcg, points: usize, kind: SculptStrokeKind) -> SculptStroke {
    let mut u = 0.1 + lcg.sample() * 0.8;
    let mut v = 0.1 + lcg.sample() * 0.8;
    let pts = (0..points)
        .map(|_| {
            u = (u + (lcg.sample() - 0.5) * 0.06).clamp(0.02, 0.98);
            v = (v + (lcg.sample() - 0.5) * 0.06).clamp(0.02, 0.98);
            SculptPoint {
                u,
                v,
                pressure: 0.8 + lcg.sample() * 0.2,
            }
        })
        .collect();
    SculptStroke {
        kind,
        points: pts,
        radius_m: WORLD * 0.04,
        strength: 5.0,
        target_height: 8.0,
        falloff: 1.5,
        enabled: true,
    }
}

/// Five large strokes (27–102 points each), every third a Flatten so the coupled
/// footprint-mean path is exercised — mirrors the issue's test project.
fn project_strokes() -> Vec<SculptStroke> {
    let mut lcg = Lcg(0x9E37_79B9);
    let counts = [27usize, 54, 73, 102, 61];
    counts
        .iter()
        .enumerate()
        .map(|(k, &n)| {
            let kind = if k % 3 == 2 {
                SculptStrokeKind::Flatten
            } else {
                SculptStrokeKind::Raise
            };
            big_stroke(&mut lcg, n, kind)
        })
        .collect()
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

/// Time one cached apply, printing wall time and the re-stamped stroke count.
fn time_apply(
    h: &Heightfield,
    p: &SculptStrokeParams,
    entry: &mut Option<SculptPrefixEntry>,
    label: &str,
) {
    let t = Instant::now();
    let (result, stats) = apply_sculpt_strokes_cached(h, p, entry);
    let dt = t.elapsed();
    black_box(&result);
    println!(
        "  {label:<10} {:>8.1?}   restamped {}/{}",
        dt, stats.restamped, stats.stroke_count
    );
}

fn main() {
    let max_res: u32 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(2048);

    for res in [1024u32, 2048, 4096] {
        if res > max_res {
            break;
        }
        let m = HeightfieldMetrics::new(res, res, WORLD, WORLD);
        let h = ramp(m);
        println!("res {res}:");

        // Cold: a fresh entry each time stamps the whole set — the cost every edit
        // paid before #123.
        let mut p = SculptStrokeParams {
            strokes: project_strokes(),
            reconcile: 0.15,
        };
        let mut cold_entry: Option<SculptPrefixEntry> = None;
        time_apply(&h, &p, &mut cold_entry, "cold");

        // Warm one persistent entry, then measure the interactive edits against it.
        let mut entry: Option<SculptPrefixEntry> = None;
        apply_sculpt_strokes_cached(&h, &p, &mut entry);

        // Append: one new large stroke resumes from the tail.
        let mut lcg = Lcg(0x1234_5678);
        p.strokes.push(big_stroke(&mut lcg, 40, SculptStrokeKind::Raise));
        time_apply(&h, &p, &mut entry, "append");

        // Drag: grow the last stroke's polyline one vertex — resumes from pre-tail.
        p.strokes.last_mut().unwrap().points.push(SculptPoint {
            u: 0.5,
            v: 0.5,
            pressure: 1.0,
        });
        time_apply(&h, &p, &mut entry, "drag");

        // Edit an interior stroke — stamps the suffix from that point up.
        let mid = p.strokes.len() / 2;
        p.strokes[mid].strength = 7.5;
        time_apply(&h, &p, &mut entry, "edit-mid");
    }
}
