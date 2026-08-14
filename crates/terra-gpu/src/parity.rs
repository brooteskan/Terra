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
/// Volcanic-island preview is intentionally a reduced massif/shelf model.
pub const VOLCANIC_ISLAND_PREVIEW: ParityTolerance = ParityTolerance::new(220.0, 1.0e-1);
/// Portable value-noise preview; CPU and GPU use different hash arithmetic.
pub const VALUE_NOISE_PREVIEW: ParityTolerance = ParityTolerance::new(17.0, 2.3e-1);
/// Effect-filter modes retained on GPU after the support audit.
pub const SMOOTH_FILTER_PREVIEW: ParityTolerance = ParityTolerance::new(1.8, 3.0e-2);
pub const INFLATE_FILTER_PREVIEW: ParityTolerance = ParityTolerance::new(2.7, 2.8e-2);

/// Canonical documentation table. A unit test keeps the checked-in fidelity
/// document synchronized with these executable contracts.
pub const FIDELITY_MATRIX_MARKDOWN: &str = "\
| Contract | Supported configuration | Max abs (m) | Normalized RMSE |\n\
| --- | --- | ---: | ---: |\n\
| `exact-height` | Flat, Ramp, SculptBase, exact blends, hybrid checkpoint | 0.001 | 0.00001 |\n\
| `mask.simple` | One Constant, Height, or Slope Multiply entry without asset operations | 0.001 | 0.0001 |\n\
| `noise.value` | Value noise with a 32-bit seed | 17.0 | 0.23 |\n\
| `filter.blur` | Default outer composite; radius/iteration fixture | 2.1 | 0.0055 |\n\
| `effect.smooth` | Smooth, default outer composite | 1.8 | 0.03 |\n\
| `effect.inflate` | Inflate, default outer composite | 2.7 | 0.028 |\n\
| `filter.terrace` | Default outer composite | 8.5 | 0.10 |\n\
| `simulation.thermal` | Non-layered, constant hardness, no weathering extension | 3.2 | 0.05 |\n\
| `simulation.hydraulic` | Base transport, no sources, particles, layers, or post-effects | 3.0 | 0.03 |\n\
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
