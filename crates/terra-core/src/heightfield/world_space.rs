//! World-space parameterisation helpers (metres ↔ texels).

use super::HeightfieldMetrics;

/// Convert a world-space length (metres) to approximate texel distance on X.
#[inline]
pub fn metres_to_texels(metres: f32, metrics: HeightfieldMetrics) -> f32 {
    metres / metrics.dx().max(1.0e-6)
}

/// Convert texel distance to world metres using X spacing.
#[inline]
pub fn texels_to_metres(texels: f32, metrics: HeightfieldMetrics) -> f32 {
    texels * metrics.dx()
}

/// Radius in texels for a circular world-space kernel (uses max of dx/dz).
#[inline]
pub fn world_radius_texels(radius_m: f32, metrics: HeightfieldMetrics) -> f32 {
    let spacing = metrics.dx().max(metrics.dz()).max(1.0e-6);
    (radius_m / spacing).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heightfield::HeightfieldMetrics;

    #[test]
    fn resolution_independent_radius() {
        let coarse = HeightfieldMetrics::new(256, 256, 4096.0, 4096.0);
        let fine = HeightfieldMetrics::new(1024, 1024, 4096.0, 4096.0);
        let r_m = 150.0;
        let t_coarse = world_radius_texels(r_m, coarse);
        let t_fine = world_radius_texels(r_m, fine);
        // Fine grid has 4× samples → 4× texel radius for same metres.
        assert!((t_fine / t_coarse - 4.0).abs() < 1e-3);
        assert!((texels_to_metres(t_coarse, coarse) - r_m).abs() < 1e-2);
    }
}
