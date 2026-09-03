//! Tapered separable smoothing primitives used by the sculpt Smooth brush.
//!
//! The terrain is filtered independently of the brush mask, then the filtered
//! target is blended once through that mask. This is important: treating the
//! mask as a closed diffusion boundary preserves total elevation, but it also
//! piles displaced terrace height into circular collars and deforms ordinary
//! slopes where they meet the brush falloff.

/// Radius of the tapered filter support relative to the authored Spread value.
/// Spread behaves like the useful half-width of the transition; the quadratic
/// tail needs twice that distance to decay without a terminal shoulder.
pub const SMOOTH_FILTER_SUPPORT_SCALE: u32 = 2;

#[inline]
pub fn smooth_filter_support_samples(spread_samples: u32) -> u32 {
    spread_samples
        .max(1)
        .saturating_mul(SMOOTH_FILTER_SUPPORT_SCALE)
}

/// Apply one authored Smooth stroke to a dense row-major grid.
///
/// The first pass computes a horizontal tapered average and the second computes
/// its vertical counterpart. Only the final result is blended with the original
/// field using `strength * weights`, so the radial brush falloff cannot become a
/// no-flux wall. Returns whether any output sample changed bit pattern.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_tapered_smooth(
    samples: &mut [f32],
    width: u32,
    height: u32,
    _dx: f32,
    _dz: f32,
    weights: &[f32],
    support: (u32, u32, u32, u32),
    strength: f32,
    spread_samples: u32,
) -> bool {
    let len = (width as usize).saturating_mul(height as usize);
    if width == 0
        || height == 0
        || samples.len() != len
        || weights.len() != len
        || !strength.is_finite()
        || strength <= 0.0
    {
        return false;
    }

    let (i0, i1, j0, j1) = support;
    if i0 > i1 || j0 > j1 || i0 >= width || j0 >= height {
        return false;
    }
    let i1 = i1.min(width - 1);
    let j1 = j1.min(height - 1);
    let filter_support = smooth_filter_support_samples(spread_samples);
    let spread_x = filter_support.min(width.saturating_sub(1).max(1));
    let spread_z = filter_support.min(height.saturating_sub(1).max(1));
    let blend_strength = strength.clamp(0.0, 1.0);
    let oj0 = j0.saturating_sub(spread_z);
    let oj1 = j1.saturating_add(spread_z).min(height - 1);
    let input = samples.to_vec();
    let mut horizontal = input.clone();

    // The quadratic taper reaches zero with zero slope just beyond Spread. A
    // uniform box has a hard terminal sample and leaves visible shoulders at
    // exactly the selected Spread distance.
    for j in oj0..=oj1 {
        for i in i0..=i1 {
            let center_index = index(width, i, j);
            let mut weighted_sum = input[center_index];
            let mut weight_sum = 1.0;
            for offset in 1..=spread_x {
                let kernel_weight = tapered_kernel_weight(offset, spread_x);
                let left = i.saturating_sub(offset);
                let right = i.saturating_add(offset).min(width - 1);
                weighted_sum +=
                    kernel_weight * (input[index(width, left, j)] + input[index(width, right, j)]);
                weight_sum += 2.0 * kernel_weight;
            }
            horizontal[center_index] = weighted_sum / weight_sum;
        }
    }

    let mut changed = false;
    for j in j0..=j1 {
        for i in i0..=i1 {
            let idx = index(width, i, j);
            let brush_weight = weights[idx].clamp(0.0, 1.0);
            if brush_weight <= 0.0 {
                continue;
            }
            let mut weighted_sum = horizontal[idx];
            let mut weight_sum = 1.0;
            for offset in 1..=spread_z {
                let kernel_weight = tapered_kernel_weight(offset, spread_z);
                let down = j.saturating_sub(offset);
                let up = j.saturating_add(offset).min(height - 1);
                weighted_sum += kernel_weight
                    * (horizontal[index(width, i, down)] + horizontal[index(width, i, up)]);
                weight_sum += 2.0 * kernel_weight;
            }
            let blurred = weighted_sum / weight_sum;
            let next = input[idx] + blend_strength * brush_weight * (blurred - input[idx]);
            changed |= next.to_bits() != input[idx].to_bits();
            samples[idx] = next;
        }
    }
    changed
}

#[inline]
fn tapered_kernel_weight(offset: u32, spread: u32) -> f32 {
    let t = 1.0 - offset as f32 / spread.saturating_add(1) as f32;
    t * t
}

#[inline]
fn index(width: u32, i: u32, j: u32) -> usize {
    (j * width + i) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_is_bit_exact_and_zero_strength_is_noop() {
        let mut flat = vec![20.0f32; 17 * 13];
        let before = flat.clone();
        let weights = vec![1.0f32; flat.len()];
        assert!(!apply_tapered_smooth(
            &mut flat,
            17,
            13,
            1.0,
            2.0,
            &weights,
            (2, 14, 2, 10),
            1.0,
            1,
        ));
        assert_eq!(flat, before);
        assert!(!apply_tapered_smooth(
            &mut flat,
            17,
            13,
            1.0,
            2.0,
            &weights,
            (2, 14, 2, 10),
            0.0,
            1,
        ));
        assert_eq!(flat, before);
    }

    #[test]
    fn hard_masked_affine_plane_stays_flat_and_bounded() {
        let (width, height) = (25u32, 19u32);
        let mut plane = Vec::with_capacity((width * height) as usize);
        for j in 0..height {
            for i in 0..width {
                plane.push(7.0 + i as f32 * 0.75 - j as f32 * 0.25);
            }
        }
        let before = plane.clone();
        let mut weights = vec![0.0f32; plane.len()];
        for j in 4..=14 {
            for i in 4..=20 {
                weights[index(width, i, j)] = 1.0;
            }
        }
        apply_tapered_smooth(
            &mut plane,
            width,
            height,
            2.0,
            3.0,
            &weights,
            (4, 20, 4, 14),
            1.0,
            1,
        );
        let error = plane
            .iter()
            .zip(&before)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let min = before.iter().copied().fold(f32::INFINITY, f32::min);
        let max = before.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(error <= 0.83, "hard mask deformed the plane by {error}");
        assert!(plane.iter().all(|&value| value >= min && value <= max));
    }

    #[test]
    fn edge_touching_hard_mask_stays_bounded() {
        let (width, height) = (25u32, 19u32);
        let mut plane = Vec::with_capacity((width * height) as usize);
        for j in 0..height {
            for i in 0..width {
                plane.push(7.0 + i as f32 * 0.75 - j as f32 * 0.25);
            }
        }
        let before = plane.clone();
        let weights = vec![1.0f32; plane.len()];
        apply_tapered_smooth(
            &mut plane,
            width,
            height,
            2.0,
            3.0,
            &weights,
            (0, width - 1, 0, height - 1),
            1.0,
            1,
        );
        let error = plane
            .iter()
            .zip(&before)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let min = before.iter().copied().fold(f32::INFINITY, f32::min);
        let max = before.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(error <= 0.83, "edge mask deformed the plane by {error}");
        assert!(plane.iter().all(|&value| value >= min && value <= max));
    }

    #[test]
    fn tapered_separable_impulse_has_no_cardinal_cross_bias() {
        let (width, height) = (129u32, 129u32);
        let center = width / 2;
        let mut impulse = vec![0.0f32; (width * height) as usize];
        impulse[index(width, center, center)] = 1.0;
        let weights = vec![1.0f32; impulse.len()];

        apply_tapered_smooth(
            &mut impulse,
            width,
            height,
            1.0,
            1.0,
            &weights,
            (0, width - 1, 0, height - 1),
            1.0,
            24,
        );

        // These points lie at almost the same Euclidean radius. The product of
        // identical horizontal and vertical kernels should be close enough to a
        // radial response that grid-aligned risers do not outlive oblique ones.
        let axis = impulse[index(width, center + 20, center)];
        let diagonal = impulse[index(width, center + 14, center + 14)];
        let relative_error = (axis - diagonal).abs() / axis.max(diagonal).max(f32::EPSILON);
        assert!(
            relative_error <= 0.35,
            "multiscale impulse retained cardinal bias: axis={axis}, diagonal={diagonal}, relative error={relative_error}"
        );
    }
}
