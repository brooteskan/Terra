//! Dependency-light single-channel mask storage.

use crate::heightfield::{Heightfield, HeightfieldMetrics};
use serde::{Deserialize, Serialize};

/// Single-channel mask in `[0, 1]`, with the same grid metrics as heightfields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskField {
    pub metrics: HeightfieldMetrics,
    data: Vec<f32>,
}

impl MaskField {
    pub fn new(metrics: HeightfieldMetrics) -> Self {
        let n = (metrics.width * metrics.height) as usize;
        Self {
            metrics,
            data: vec![1.0; n],
        }
    }

    pub fn filled(metrics: HeightfieldMetrics, value: f32) -> Self {
        let n = (metrics.width * metrics.height) as usize;
        Self {
            metrics,
            data: vec![value.clamp(0.0, 1.0); n],
        }
    }

    pub fn ones(metrics: HeightfieldMetrics) -> Self {
        Self::filled(metrics, 1.0)
    }

    pub fn zeros(metrics: HeightfieldMetrics) -> Self {
        Self::filled(metrics, 0.0)
    }

    /// Build from raw samples without `[0, 1]` clamping.
    pub fn from_raw(metrics: HeightfieldMetrics, data: &[f32]) -> Self {
        assert_eq!(data.len(), (metrics.width * metrics.height) as usize);
        Self {
            metrics,
            data: data.to_vec(),
        }
    }

    #[inline]
    fn idx(&self, i: u32, j: u32) -> usize {
        (j * self.metrics.width + i) as usize
    }

    pub fn get(&self, i: u32, j: u32) -> f32 {
        self.data[self.idx(i, j)]
    }

    pub fn set(&mut self, i: u32, j: u32, value: f32) {
        let index = self.idx(i, j);
        self.data[index] = value.clamp(0.0, 1.0);
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Resample onto `target` with normalized nearest-neighbour lookup.
    pub fn resampled_nearest(&self, target: HeightfieldMetrics) -> Self {
        if self.metrics.width == target.width && self.metrics.height == target.height {
            let mut field = self.clone();
            field.metrics = target;
            return field;
        }

        let mut out = Self::zeros(target);
        if self.metrics.width == 0 || self.metrics.height == 0 {
            return out;
        }
        for j in 0..target.height {
            for i in 0..target.width {
                let u = (i as f32 + 0.5) / target.width.max(1) as f32;
                let v = (j as f32 + 0.5) / target.height.max(1) as f32;
                let si = ((u * self.metrics.width as f32) as u32).min(self.metrics.width - 1);
                let sj = ((v * self.metrics.height as f32) as u32).min(self.metrics.height - 1);
                let dst = (j * target.width + i) as usize;
                out.data[dst] = self.get(si, sj);
            }
        }
        out
    }

    /// Owned nearest-neighbour resampling variant.
    pub fn into_resampled_nearest(mut self, target: HeightfieldMetrics) -> Self {
        if self.metrics.width == target.width && self.metrics.height == target.height {
            self.metrics = target;
            self
        } else {
            self.resampled_nearest(target)
        }
    }

    pub fn from_height_range(hf: &Heightfield, min: f32, max: f32) -> Self {
        let metrics = hf.metrics;
        let mut mask = Self::zeros(metrics);
        let span = (max - min).max(1e-6);
        for j in 0..metrics.height {
            for i in 0..metrics.width {
                let value = ((hf.get(i, j) - min) / span).clamp(0.0, 1.0);
                mask.set(i, j, value);
            }
        }
        mask
    }

    /// Scale all samples in place with SIMD-friendly helpers.
    pub fn scale_in_place(&mut self, scale: f32) {
        crate::simd_ops::scale_slice_in_place(self.data_mut(), scale);
        crate::simd_ops::clamp_slice_in_place(self.data_mut(), 0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_resample_matches_target_and_preserves_raw_values() {
        let source_metrics = HeightfieldMetrics::new(2, 2, 20.0, 20.0);
        let source = MaskField::from_raw(source_metrics, &[0.0, 2.0, 4.0, 8.0]);
        let target = HeightfieldMetrics::new(4, 4, 20.0, 20.0);
        let resized = source.resampled_nearest(target);

        assert_eq!(resized.metrics.width, 4);
        assert_eq!(resized.metrics.height, 4);
        assert_eq!(resized.get(0, 0), 0.0);
        assert_eq!(resized.get(3, 0), 2.0);
        assert_eq!(resized.get(0, 3), 4.0);
        assert_eq!(resized.get(3, 3), 8.0);
    }
}
