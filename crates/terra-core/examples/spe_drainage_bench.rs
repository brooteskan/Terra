//! Benchmark: drainage graph rebuild + accumulation cost in the SPE export path.
//! (issue #27)
//!
//! Usage: `cargo run --release --example spe_drainage_bench -- [size] [reps]`
//!
//! For an `n×n` noisy-mountain fixture it reports two things:
//!
//!   - **micro**: one drainage refresh — `build_flow_graph(D8)` (the general
//!     `Vec<Vec<FlowReceiver>>` graph #24 routed SPE through) plus
//!     `accumulate_drainage_area`, split into build vs. accumulate. Once the lean
//!     flat-D8 path exists this section also times it side-by-side, so the binary
//!     is the honest "what did #24's per-cell `Vec` graph cost?" proxy — #24
//!     deleted the old flat path, so the lean path here stands in for it.
//!   - **macro**: `stream_power_erode` end-to-end with export-shaped params
//!     (`drainage_reuse_stride = 1`), run with `refill_each_iter` both ways so the
//!     Priority-Flood cost is separated from the graph cost.
//!
//! Median of `reps` timed runs after one warmup. Compare across a commit boundary
//! (HEAD vs. the lean path) for the macro numbers; the micro numbers compare
//! in-binary once the lean path lands.

use std::hint::black_box;
use std::time::Instant;

use terra_core::geomorph::{
    accumulate_drainage_area, accumulate_drainage_area_d8, build_flow_graph, noisy_mountain,
    priority_flood_fill, D8Drainage, FlowModel, Precipitation,
};
use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
use terra_core::hydro::stream_power_erode;
use terra_core::layer::StreamPowerParams;
use terra_core::mask::MaskField;

const EXPORT_ITERS: u32 = 20;

fn main() {
    let mut args = std::env::args().skip(1);
    let size: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(512);
    let reps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(7);

    let threads = rayon_threads();
    println!("# SPE drainage bench — issue #27");
    println!(
        "size={size}x{size}  reps={reps}  export_iters={EXPORT_ITERS}  rayon_threads={threads}"
    );
    println!();

    let metrics = HeightfieldMetrics::new(size, size, size as f32, size as f32);
    let terrain = noisy_mountain(metrics);
    let filled = priority_flood_fill(&terrain);

    micro(&filled, reps);
    println!();
    macro_spe(&terrain, reps);
}

/// One drainage refresh on a fixed filled surface: general FlowGraph vs. (later)
/// the lean flat-D8 path.
fn micro(filled: &Heightfield, reps: usize) {
    let precip = Precipitation::uniform(1.0);

    // Warmup.
    {
        let g = build_flow_graph(filled, FlowModel::D8);
        black_box(accumulate_drainage_area(&g, &precip));
    }

    let mut gen_build = Vec::with_capacity(reps);
    let mut gen_acc = Vec::with_capacity(reps);
    let mut lean_build = Vec::with_capacity(reps);
    let mut lean_rebuild = Vec::with_capacity(reps);
    let mut lean_acc = Vec::with_capacity(reps);
    let mut checksum = 0.0f64;

    // A pre-built cache to time the buffer-reusing `rebuild` (SPE's steady state).
    let mut cache = D8Drainage::build(filled);

    for _ in 0..reps {
        // General FlowGraph path (what #24 routed SPE through).
        let t0 = Instant::now();
        let g = build_flow_graph(filled, FlowModel::D8);
        let t1 = Instant::now();
        let acc = accumulate_drainage_area(&g, &precip);
        let t2 = Instant::now();
        gen_build.push(ms(t0, t1));
        gen_acc.push(ms(t1, t2));

        // Lean flat-D8 path: fresh build, buffer-reusing rebuild, accumulate.
        let t3 = Instant::now();
        let fresh = D8Drainage::build(filled);
        let t4 = Instant::now();
        cache.rebuild(filled);
        let t5 = Instant::now();
        let acc_d = accumulate_drainage_area_d8(&cache, &precip);
        let t6 = Instant::now();
        lean_build.push(ms(t3, t4));
        lean_rebuild.push(ms(t4, t5));
        lean_acc.push(ms(t5, t6));

        checksum += acc.iter().copied().map(f64::from).sum::<f64>();
        checksum += acc_d.iter().copied().map(f64::from).sum::<f64>();
        black_box((&g, &fresh, &cache));
    }

    let gen_total = median(&gen_build) + median(&gen_acc);
    let lean_total = median(&lean_rebuild) + median(&lean_acc);
    println!("## micro: one D8 drainage refresh (median ms)");
    println!(
        "general FlowGraph:  build={:.3}  accumulate={:.3}  total={:.3}",
        median(&gen_build),
        median(&gen_acc),
        gen_total,
    );
    println!(
        "lean flat-D8:       build={:.3}  rebuild={:.3}  accumulate={:.3}  total(rebuild+acc)={:.3}",
        median(&lean_build),
        median(&lean_rebuild),
        median(&lean_acc),
        lean_total,
    );
    if lean_total > 0.0 {
        println!(
            "-> refresh speedup: {:.2}x  ({:.1}% less time per refresh)",
            gen_total / lean_total,
            (1.0 - lean_total / gen_total) * 100.0,
        );
    }
    black_box(checksum);
}

/// Export-shaped `stream_power_erode`, both `refill_each_iter` modes.
fn macro_spe(terrain: &Heightfield, reps: usize) {
    let hardness = MaskField::zeros(terrain.metrics);

    println!("## macro: stream_power_erode end-to-end (median ms)");
    for &refill in &[true, false] {
        let p = StreamPowerParams {
            iterations: EXPORT_ITERS,
            drainage_reuse_stride: 1,
            refill_each_iter: refill,
            use_dinfinity: false,
            ..StreamPowerParams::default()
        };

        // Warmup.
        black_box(stream_power_erode(terrain, &p, &hardness));

        let mut runs = Vec::with_capacity(reps);
        let mut checksum = 0.0f64;
        for _ in 0..reps {
            let t0 = Instant::now();
            let r = stream_power_erode(terrain, &p, &hardness);
            runs.push(ms(t0, Instant::now()));
            checksum += r.height.get(0, 0) as f64;
            black_box(&r);
        }
        println!("refill_each_iter={refill:<5}  total={:.2}", median(&runs));
        black_box(checksum);
    }
}

fn ms(a: Instant, b: Instant) -> f64 {
    b.duration_since(a).as_secs_f64() * 1e3
}

fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn rayon_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0)
}
