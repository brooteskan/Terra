//! Context-aware placement coverage estimation.

use super::{bake_distribution_with_context, DistBakeContext};
use crate::heightfield::HeightfieldMetrics;
use crate::mask_ir::PlacementDefinition;

/// Fraction of samples with coverage > 0.05.
pub fn coverage_estimate(
    placement: &PlacementDefinition,
    metrics: HeightfieldMetrics,
    ctx: &DistBakeContext<'_>,
) -> f32 {
    let dist = placement.active_distribution();
    let field = bake_distribution_with_context(&dist, metrics, ctx);
    let total = (metrics.width * metrics.height).max(1) as f32;
    let mut hit = 0u32;
    for j in 0..metrics.height {
        for i in 0..metrics.width {
            if field.get(i, j) > 0.05 {
                hit += 1;
            }
        }
    }
    hit as f32 / total
}
