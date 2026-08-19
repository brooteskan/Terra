//! CPU-oracle comparison helpers for GPU-preview parity.

use terra_core::heightfield::Heightfield;

/// Fixed error budget owned by one named GPU preview contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParityTolerance {
    pub max_abs: f32,
    pub normalized_rmse: f32,
}

impl ParityTolerance {
    pub const fn new(max_abs: f32, normalized_rmse: f32) -> Self {
        Self {
            max_abs,
            normalized_rmse,
        }
    }
}

/// Full-field error statistics. `normalized_rmse` uses a CPU-derived scale so
/// the same contract remains meaningful when an authored amplitude changes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParityStats {
    pub max_abs: f32,
    pub rmse: f32,
    pub normalized_rmse: f32,
    pub reference_scale: f32,
    pub worst_index: usize,
}

/// Exact transfer/compositing budget. Approximate kernels deliberately use
/// their own named constants instead of sharing this value.
pub const EXACT_HEIGHT: ParityTolerance = ParityTolerance::new(1.0e-3, 1.0e-5);
/// Supported Constant/Height/Slope mask bake contract.
pub const SIMPLE_MASK: ParityTolerance = ParityTolerance::new(1.0e-3, 1.0e-4);
/// Local blur uses the same bounded neighborhood but may reorder float sums.
pub const BLUR_PREVIEW: ParityTolerance = ParityTolerance::new(2.1, 5.5e-3);
/// Terrace uses the GPU's tracked range, so allow a small range-estimation edge error.
pub const TERRACE_PREVIEW: ParityTolerance = ParityTolerance::new(8.5, 1.0e-1);
/// Thermal is an intentionally reduced preview solver; this is a field-level,
/// metre-scale budget for the small deterministic ratchet fixture.
pub const THERMAL_PREVIEW: ParityTolerance = ParityTolerance::new(3.2, 5.0e-2);
/// Hydraulic preview omits the CPU particle/detail pass. The fixture disables
/// those extensions and bounds the remaining shallow-water approximation.
pub const HYDRAULIC_PREVIEW: ParityTolerance = ParityTolerance::new(3.0, 3.0e-2);
/// RiverCarve D8 preview uses bounded iterative accumulation and an additive
/// gather carve. On the monotone drainage fixture it is exact-height class.
pub const RIVER_CARVE_D8_PREVIEW: ParityTolerance = ParityTolerance::new(1.0e-3, 1.0e-5);
/// The preview intentionally approximates D-infinity routing with D8. The CPU's
/// priority-flood/D-infinity solver remains the export oracle.
pub const RIVER_CARVE_DINFINITY_PREVIEW: ParityTolerance = ParityTolerance::new(6.0, 2.0e-2);
/// Volcanic-island preview is intentionally a reduced massif/shelf model.
pub const VOLCANIC_ISLAND_PREVIEW: ParityTolerance = ParityTolerance::new(220.0, 1.0e-1);
/// GPU shape-family contracts (#126). These are separate named budgets because
/// the preview kernels range from an exact pointwise remap (Plateau) to bounded
/// procedural approximations (notably Dunes and the legacy volcanic island).
pub const MOUNTAINS_PREVIEW: ParityTolerance = ParityTolerance::new(10.0, 5.0e-3);
pub const DUNES_PREVIEW: ParityTolerance = ParityTolerance::new(36.0, 7.8e-1);
pub const CANYONS_PREVIEW: ParityTolerance = ParityTolerance::new(1.0e-3, 1.0e-5);
pub const MESA_PREVIEW: ParityTolerance = ParityTolerance::new(3.0e-3, 1.0e-5);
pub const VOLCANO_PREVIEW: ParityTolerance = ParityTolerance::new(4.0e-3, 1.0e-5);
pub const UPLIFT_PREVIEW: ParityTolerance = ParityTolerance::new(1.0e-1, 2.0e-5);
pub const PLATEAU_PREVIEW: ParityTolerance = ParityTolerance::new(1.0e-3, 1.0e-5);
pub const ARCHIPELAGO_PREVIEW: ParityTolerance = ParityTolerance::new(3.0e-2, 5.0e-6);
pub const ATOLL_PREVIEW: ParityTolerance = ParityTolerance::new(1.0e-3, 1.0e-5);
/// Portable value-noise preview; CPU and GPU use different hash arithmetic.
pub const VALUE_NOISE_PREVIEW: ParityTolerance = ParityTolerance::new(17.0, 2.3e-1);
/// CPU-aligned noise-family previews (#125). The WGSL ports the CPU integer hash,
/// four-way Perlin gradient, octave seed streams, remap, ridged feedback, and domain
/// displacement directly. Residuals are f32 expression-order differences: measured
/// max abs / normalized RMSE were 2.6e-5 / 1.3e-7 (Perlin), 4.5e-5 / 1.7e-7
/// (fBm), 4.1e-5 / 3.7e-7 (ridged), and 7.5e-5 / 3.0e-7 (domain warp) on the
/// non-square parameter fixture. These exact-height-class bounds retain portable
/// headroom without hiding a hash, gradient, octave, or warp wiring regression.
pub const PERLIN_NOISE_PREVIEW: ParityTolerance = ParityTolerance::new(1.0e-3, 1.0e-5);
pub const FBM_VALUE_PREVIEW: ParityTolerance = ParityTolerance::new(1.0e-3, 1.0e-5);
pub const FBM_PERLIN_PREVIEW: ParityTolerance = ParityTolerance::new(1.0e-3, 1.0e-5);
pub const RIDGED_VALUE_PREVIEW: ParityTolerance = ParityTolerance::new(1.0e-3, 1.0e-5);
pub const RIDGED_PERLIN_PREVIEW: ParityTolerance = ParityTolerance::new(1.0e-3, 1.0e-5);
pub const DOMAIN_WARP_PREVIEW: ParityTolerance = ParityTolerance::new(1.0e-3, 1.0e-5);
/// Effect-filter modes retained on GPU after the support audit.
pub const SMOOTH_FILTER_PREVIEW: ParityTolerance = ParityTolerance::new(1.8, 3.0e-2);
pub const INFLATE_FILTER_PREVIEW: ParityTolerance = ParityTolerance::new(2.7, 2.8e-2);
/// Denoise (bilateral) preview (#119). The GPU kernel mirrors the CPU
/// `filter_kernels::bilateral` term-for-term, so the residual is the same class
/// as Smooth: the exp() range/spatial weights diverge only at transcendental
/// edges, and at Draft quality the shared effect-filter dispatch mixes `strength`
/// and floors the pass count per pass while the CPU mixes once at the end (so an
/// authored `iterations: 1` runs two GPU passes against one CPU pass). Across the
/// depth discontinuity the range weight underflows to 0 on both backends, so the
/// deep basin is preserved bit-for-bit; the budget is set by the small-fixture
/// worst texel where the Draft 2-vs-1-pass fold difference lives. Measured worst
/// case is ~1.5 m / 0.030 on the patterned fixture (the shelf/basin fixture, whose
/// passes align at 2, stays well under); this holds portable headroom.
pub const DENOISE_FILTER_PREVIEW: ParityTolerance = ParityTolerance::new(2.2, 3.8e-2);
/// Pointwise Add/Set and bounded greyscale erosion mirror the CPU formulas.
pub const ADD_SET_FILTER_PREVIEW: ParityTolerance = ParityTolerance::new(1.0e-3, 1.0e-5);
pub const DEFLATE_FILTER_PREVIEW: ParityTolerance = ParityTolerance::new(1.0e-3, 1.0e-5);
/// Field-range remaps use an exact GPU reduction; residuals are f32 pow ordering.
pub const CURVE_FILTER_PREVIEW: ParityTolerance = ParityTolerance::new(1.0e-3, 1.0e-5);
pub const CUTOFF_FILTER_PREVIEW: ParityTolerance = ParityTolerance::new(1.0e-3, 1.0e-5);
/// Newly ratcheted pointwise, range-remap, neighbourhood, and procedural filters.
pub const EFFECT_FILTER_EXACT_PREVIEW: ParityTolerance = ParityTolerance::new(1.0e-3, 1.0e-5);
pub const EFFECT_FILTER_SPATIAL_PREVIEW: ParityTolerance = ParityTolerance::new(2.0e-2, 2.0e-4);
pub const EFFECT_FILTER_WARP_PREVIEW: ParityTolerance = ParityTolerance::new(5.0e-2, 1.0e-3);
/// SculptStrokes preview: untouched texels are bit-exact copies; stamped texels
/// diverge only by transcendental edges (`sqrt` vs `hypot`, `pow` vs `powf`). The
/// bit-exact `hash_noise` port keeps Noise strokes in the same budget; the base-3x3
/// pull of Smooth (#114), Pinch (#115), and Coastline (#116) is transcendental-free
/// (nine same-order adds and a divide, plus a multiply or two for Pinch's 1.25 gain
/// and Coastline's lower-and-blend), so none widens the worst case. Flatten (#117)
/// settles toward a footprint mean the reduce/resolve passes sum in an f32 tree
/// rather than the CPU's f64 sequential order; because the out-of-footprint texels
/// contribute exact zeros, that error normalises against the mean and stays below the
/// distance-stamp edge that still dominates. Measured worst case is ~6.5e-5 m over the
/// 16-kind stroke set (unchanged, and located on a distance stamp, not a Flatten);
/// this holds portable headroom.
pub const SCULPT_STROKES_PREVIEW: ParityTolerance = ParityTolerance::new(1.0e-3, 1.0e-5);

/// Canonical documentation table. A unit test keeps the checked-in fidelity
/// document synchronized with these executable contracts.
pub const FIDELITY_MATRIX_MARKDOWN: &str = "\
| Contract | Supported configuration | Max abs (m) | Normalized RMSE |\n\
| --- | --- | ---: | ---: |\n\
| `exact-height` | Flat, Ramp, SculptBase, exact blends, hybrid checkpoint | 0.001 | 0.00001 |\n\
| `authoring.sculpt-strokes` | Per-sample stroke kinds, Smooth/Pinch/Coastline, Flatten, and distance stamps, supported blend/mask | 0.001 | 0.00001 |\n\
| `mask.simple` | One Constant, Height, or Slope Multiply entry without asset operations | 0.001 | 0.0001 |\n\
| `noise.value` | Value noise with a 32-bit seed | 17.0 | 0.23 |\n\
| `noise.perlin` | Perlin, 1-12 octaves, reproducible 32-bit seed stream | 0.001 | 0.00001 |\n\
| `noise.fbm.value` | Value fBm, 1-12 octaves, reproducible 32-bit seed stream | 0.001 | 0.00001 |\n\
| `noise.fbm.perlin` | Perlin fBm, 1-12 octaves, reproducible 32-bit seed stream | 0.001 | 0.00001 |\n\
| `noise.ridged.value` | Value ridged MF, 1-12 octaves, reproducible 32-bit seed stream | 0.001 | 0.00001 |\n\
| `noise.ridged.perlin` | Perlin ridged MF, 1-12 octaves, reproducible 32-bit seed stream | 0.001 | 0.00001 |\n\
| `noise.domain-warp` | Perlin domain warp, 1-12 octaves, reproducible 32-bit seed stream | 0.001 | 0.00001 |\n\
| `filter.blur` | Default outer composite; radius/iteration fixture | 2.1 | 0.0055 |\n\
| `effect.smooth` | Smooth, default outer composite | 1.8 | 0.03 |\n\
| `effect.inflate` | Inflate, default outer composite | 2.7 | 0.028 |\n\
| `effect.denoise` | Denoise (bilateral), default outer composite | 2.2 | 0.038 |\n\
| `effect.add-set` | Add and Set pointwise remaps, default outer composite | 0.001 | 0.00001 |\n\
| `effect.deflate` | Amount-limited greyscale erosion, default outer composite | 0.001 | 0.00001 |\n\
| `effect.curve` | Exact entering-field range reduction, default outer composite | 0.001 | 0.00001 |\n\
| `effect.cutoff` | Exact entering-field range reduction, default outer composite | 0.001 | 0.00001 |\n\
| `effect.pointwise-procedural` | TerraceSimple, Shore, Blocks, ZeroEdge, Squeeze, Perlin-family noise, scatter, Hexagons, and authored absolute border/flatten targets | 0.001 | 0.00001 |\n\
| `effect.spatial` | DirectionalBlur, AngleBlur, Balloon, Crater, TerraceSteep, and radius-one SpikeRemoval | 0.02 | 0.0002 |\n\
| `effect.warp` | Swirl and Distortion with a reproducible 32-bit seed | 0.05 | 0.001 |\n\
| `filter.terrace` | Default outer composite | 8.5 | 0.10 |\n\
| `simulation.thermal` | Non-layered, constant hardness, no weathering extension | 3.2 | 0.05 |\n\
| `simulation.hydraulic` | Base transport, no sources, particles, layers, or post-effects | 3.0 | 0.03 |\n\
| `simulation.river-carve.d8` | D8 routing, no guide mask, bounded bank radius | 0.001 | 0.00001 |\n\
| `simulation.river-carve.d-infinity` | D-infinity authored mode approximated by D8 preview, no guide mask, bounded bank radius | 6.0 | 0.02 |\n\
| `shape.mountains` | Mountains with reproducible 32-bit seed streams | 10.0 | 0.005 |\n\
| `shape.dunes` | Default transport controls, 2-4 octaves, reproducible 32-bit seed stream | 36.0 | 0.78 |\n\
| `shape.canyons` | Canyons with a 32-bit seed | 0.001 | 0.00001 |\n\
| `shape.mesa` | Mesa with a 32-bit seed | 0.003 | 0.00001 |\n\
| `shape.volcano` | Volcano with a 32-bit seed | 0.004 | 0.00001 |\n\
| `shape.uplift` | Uplift with reproducible 32-bit seed streams | 0.1 | 0.00002 |\n\
| `shape.plateau` | Pointwise input remap | 0.001 | 0.00001 |\n\
| `island.archipelago` | Archipelago with reproducible 32-bit seed streams | 0.03 | 0.000005 |\n\
| `island.atoll` | Atoll with reproducible 32-bit seed streams | 0.001 | 0.00001 |\n\
| `island.volcanic-high` | VolcanicHighIsland with a 32-bit seed | 220.0 | 0.10 |";

/// Largest element-wise absolute difference between equally sized slices.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "parity inputs must have equal lengths");
    a.iter()
        .zip(b)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max)
}

/// Compare two complete fields and return diagnostics suitable for a parity
/// failure. Panics on shape mismatch or non-finite data: either is a contract
/// violation, not a numerical tolerance issue.
pub fn field_stats(actual: &Heightfield, reference: &Heightfield) -> ParityStats {
    assert_eq!(
        actual.metrics, reference.metrics,
        "parity fields must have identical metrics"
    );
    let actual = actual.to_dense();
    let reference = reference.to_dense();
    assert_eq!(actual.len(), reference.len());
    assert!(!actual.is_empty(), "parity fields must not be empty");

    let mut max_abs = 0.0f32;
    let mut worst_index = 0usize;
    let mut squared_error = 0.0f64;
    let mut squared_reference = 0.0f64;
    let mut reference_min = f32::INFINITY;
    let mut reference_max = f32::NEG_INFINITY;
    for (index, (&gpu, &cpu)) in actual.iter().zip(&reference).enumerate() {
        assert!(
            gpu.is_finite(),
            "GPU field contains non-finite value at {index}"
        );
        assert!(
            cpu.is_finite(),
            "CPU field contains non-finite value at {index}"
        );
        let error = (gpu - cpu).abs();
        if error > max_abs {
            max_abs = error;
            worst_index = index;
        }
        squared_error += f64::from(error) * f64::from(error);
        squared_reference += f64::from(cpu) * f64::from(cpu);
        reference_min = reference_min.min(cpu);
        reference_max = reference_max.max(cpu);
    }

    let count = actual.len() as f64;
    let rmse = (squared_error / count).sqrt() as f32;
    let reference_rms = (squared_reference / count).sqrt() as f32;
    let reference_scale = (reference_max - reference_min)
        .abs()
        .max(reference_rms)
        .max(1.0);
    ParityStats {
        max_abs,
        rmse,
        normalized_rmse: rmse / reference_scale,
        reference_scale,
        worst_index,
    }
}

pub fn assert_field_parity(
    contract: &str,
    actual: &Heightfield,
    reference: &Heightfield,
    tolerance: ParityTolerance,
) {
    let stats = field_stats(actual, reference);
    let width = actual.metrics.width as usize;
    let x = stats.worst_index % width;
    let y = stats.worst_index / width;
    assert!(
        stats.max_abs <= tolerance.max_abs
            && stats.normalized_rmse <= tolerance.normalized_rmse,
        "{contract} exceeded parity tolerance: stats={stats:?}, tolerance={tolerance:?}, worst=({x},{y})"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use terra_core::analyze::thermal_erode;
    use terra_core::generators::terrace;
    use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
    use terra_core::layer::{TerraceParams, ThermalErosionParams};

    #[test]
    fn cpu_thermal_reference_runs_on_small_field() {
        // The complete GPU/CPU thermal comparison is integration-tested because it
        // requires texture readback. Touching the shared headless device here
        // verifies adapter setup when available while keeping headless test
        // environments portable (the harness returns None instead of failing).
        let _ = terra_test_gpu::headless();

        let metrics = HeightfieldMetrics::new(64, 64, 64.0, 64.0);
        let mut input = Heightfield::zeros(metrics);
        input.set(32, 32, 100.0);
        let (out, _, _) = thermal_erode(
            &input,
            &ThermalErosionParams {
                iterations: 2,
                ..ThermalErosionParams::default()
            },
        );
        assert!(out.get(32, 32) < input.get(32, 32));
    }

    #[test]
    fn cpu_terrace_is_deterministic() {
        let metrics = HeightfieldMetrics::new(64, 64, 64.0, 64.0);
        let values: Vec<f32> = (0..64 * 64).map(|i| i as f32 / 64.0).collect();
        let input = Heightfield::from_dense(metrics, &values);
        let params = TerraceParams::default();
        let first = terrace(&input, &params).to_dense();
        let second = terrace(&input, &params).to_dense();
        assert_eq!(max_abs_diff(&first, &second), 0.0);
        // GPU float arithmetic is checked in integration tests with a small tolerance.
    }

    #[test]
    fn field_stats_are_scale_aware_and_locate_worst_sample() {
        let metrics = HeightfieldMetrics::new(2, 2, 2.0, 2.0);
        let reference = Heightfield::from_dense(metrics, &[10.0, 20.0, 30.0, 40.0]);
        let actual = Heightfield::from_dense(metrics, &[10.0, 22.0, 30.0, 39.0]);
        let stats = field_stats(&actual, &reference);
        assert_eq!(stats.max_abs, 2.0);
        assert_eq!(stats.worst_index, 1);
        assert!(stats.rmse > 1.0 && stats.rmse < 1.2);
        assert!(stats.normalized_rmse < stats.rmse);
    }

    #[test]
    fn fidelity_document_contains_the_executable_contract_matrix() {
        const START: &str = "<!-- BEGIN GENERATED GPU PARITY MATRIX -->";
        const END: &str = "<!-- END GENERATED GPU PARITY MATRIX -->";
        let docs = include_str!("../../../docs/algorithms/fidelity.md");
        let matrix = docs
            .split_once(START)
            .and_then(|(_, tail)| tail.split_once(END).map(|(matrix, _)| matrix))
            .expect("fidelity document must contain generated matrix markers");
        assert_eq!(matrix.trim(), FIDELITY_MATRIX_MARKDOWN.trim());
    }
}
