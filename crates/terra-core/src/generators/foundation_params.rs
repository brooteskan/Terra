//! Persisted parameters for foundation generators.

use crate::gradient_smoothing::apply_tapered_smooth;
use crate::raster::GridDimensions;
use crate::raster::{resample_f32_grid, RasterResizeError, RasterResizeLimits, RasterSemantic};
use serde::{Deserialize, Deserializer, Serialize};

/// Paintable foundation heights in meters (normalized UV grid).

#[derive(Debug, Clone, Serialize)]
pub struct SculptParams {
    /// Width of the normalized UV paint buffer.
    pub width: u32,
    /// Height of the normalized UV paint buffer.
    pub height: u32,
    /// Heights in world meters, row-major, length = width * height.
    pub samples: Vec<f32>,
    /// Fill / reset height when buffer is created or reset.
    pub fill_height: f32,
}

impl Default for SculptParams {
    fn default() -> Self {
        Self::filled(512, 20.0)
    }
}

impl SculptParams {
    pub fn filled(resolution: u32, fill_height: f32) -> Self {
        Self::filled_grid(GridDimensions::square(resolution), fill_height)
    }

    pub fn filled_grid(dimensions: GridDimensions, fill_height: f32) -> Self {
        let n = (dimensions.width as usize).saturating_mul(dimensions.height as usize);
        Self {
            width: dimensions.width,
            height: dimensions.height,
            samples: vec![fill_height; n],
            fill_height,
        }
    }

    pub const fn dimensions(&self) -> GridDimensions {
        GridDimensions::new(self.width, self.height)
    }

    pub fn resized(
        &self,
        dimensions: GridDimensions,
        limits: RasterResizeLimits,
    ) -> Result<Self, RasterResizeError> {
        let samples = resample_f32_grid(
            &self.samples,
            self.dimensions(),
            dimensions,
            RasterSemantic::Height,
            limits,
        )?;
        Ok(Self {
            width: dimensions.width,
            height: dimensions.height,
            samples,
            fill_height: self.fill_height,
        })
    }

    pub fn ensure_buffer(&mut self) {
        let n = (self.width as usize).saturating_mul(self.height as usize);
        if self.samples.len() != n {
            self.samples = vec![self.fill_height; n];
        }
    }

    pub fn reset(&mut self) {
        self.ensure_buffer();
        self.samples.fill(self.fill_height);
    }

    /// Soft circular stamp. `mode`: 0 = raise, 1 = lower, 2 = smooth, 3 = flatten.
    pub fn stamp_circle(
        &mut self,
        u: f32,
        v: f32,
        radius_uv: f32,
        strength: f32,
        mode: u8,
    ) -> bool {
        self.stamp_circle_with_smooth_spread(
            u,
            v,
            radius_uv,
            strength,
            mode,
            crate::authoring::SMOOTH_SPREAD_DEFAULT,
        )
    }

    /// Soft circular stamp with an explicit Smooth stencil spread in samples.
    ///
    /// Non-Smooth modes ignore `smooth_spread_samples`. Keeping the legacy
    /// [`Self::stamp_circle`] entry point preserves callers and persisted
    /// behavior while editor dabs can carry the user-selected Smooth spread.
    pub fn stamp_circle_with_smooth_spread(
        &mut self,
        u: f32,
        v: f32,
        radius_uv: f32,
        strength: f32,
        mode: u8,
        smooth_spread_samples: u32,
    ) -> bool {
        self.ensure_buffer();
        let width = self.width;
        let height = self.height;
        if width == 0 || height == 0 {
            return false;
        }
        let radius = radius_uv.max(1e-6);
        let min_i = ((u - radius) * width as f32).floor().max(0.0) as u32;
        let max_i = ((u + radius) * width as f32).ceil().min(width as f32 - 1.0) as u32;
        let min_j = ((v - radius) * height as f32).floor().max(0.0) as u32;
        let max_j = ((v + radius) * height as f32)
            .ceil()
            .min(height as f32 - 1.0) as u32;

        if mode == 2 {
            if !strength.is_finite() || strength <= 0.0 {
                return false;
            }
            // Use the same tapered target as semantic Shape Layer strokes.
            let mut weights = vec![0.0f32; self.samples.len()];
            for j in min_j..=max_j {
                for i in min_i..=max_i {
                    let x = (i as f32 + 0.5) / width as f32;
                    let y = (j as f32 + 0.5) / height as f32;
                    let d = ((x - u).powi(2) + (y - v).powi(2)).sqrt() / radius;
                    if d > 1.0 {
                        continue;
                    }
                    let idx = (j * width + i) as usize;
                    weights[idx] = (1.0 - d * d).max(0.0);
                }
            }
            return apply_tapered_smooth(
                &mut self.samples,
                width,
                height,
                1.0 / width.max(1) as f32,
                1.0 / height.max(1) as f32,
                &weights,
                (min_i, max_i, min_j, max_j),
                strength,
                smooth_spread_samples.clamp(1, crate::authoring::SMOOTH_SPREAD_MAX),
            );
        }

        if mode == 3 {
            // Flatten: pull the footprint toward its own mean height. This is a
            // lerp toward an average, so it is bounded by the footprint's existing
            // range and settles terrain instead of accumulating (mode 0 raise,
            // which flatten used to fall through to, ran away).
            let mut sum = 0.0f64;
            let mut count = 0.0f64;
            for j in min_j..=max_j {
                for i in min_i..=max_i {
                    let x = (i as f32 + 0.5) / width as f32;
                    let y = (j as f32 + 0.5) / height as f32;
                    let d = ((x - u).powi(2) + (y - v).powi(2)).sqrt() / radius;
                    if d > 1.0 {
                        continue;
                    }
                    sum += self.samples[(j * width + i) as usize] as f64;
                    count += 1.0;
                }
            }
            if count <= 0.0 {
                return false;
            }
            let mean = (sum / count) as f32;
            let mut changed = false;
            for j in min_j..=max_j {
                for i in min_i..=max_i {
                    let x = (i as f32 + 0.5) / width as f32;
                    let y = (j as f32 + 0.5) / height as f32;
                    let d = ((x - u).powi(2) + (y - v).powi(2)).sqrt() / radius;
                    if d > 1.0 {
                        continue;
                    }
                    let falloff = (1.0 - d * d) * strength.clamp(0.0, 1.0);
                    let idx = (j * width + i) as usize;
                    let cur = self.samples[idx];
                    let next = cur + (mean - cur) * falloff;
                    changed |= next.to_bits() != cur.to_bits();
                    self.samples[idx] = next;
                }
            }
            return changed;
        }

        let delta_sign = if mode == 1 { -1.0 } else { 1.0 };
        // strength is meters of peak displacement per stamp
        let peak = strength.max(0.0) * delta_sign;
        let mut changed = false;
        for j in min_j..=max_j {
            for i in min_i..=max_i {
                let x = (i as f32 + 0.5) / width as f32;
                let y = (j as f32 + 0.5) / height as f32;
                let d = ((x - u).powi(2) + (y - v).powi(2)).sqrt() / radius;
                if d <= 1.0 {
                    let amount = (1.0 - d * d) * peak;
                    let sample = &mut self.samples[(j * width + i) as usize];
                    let next = *sample + amount;
                    changed |= next.to_bits() != sample.to_bits();
                    *sample = next;
                }
            }
        }
        changed
    }

    pub fn sample_bilinear(&self, u: f32, v: f32) -> f32 {
        let width = self.width.max(1);
        let height = self.height.max(1);
        let n = (width as usize).saturating_mul(height as usize);
        if self.samples.len() != n {
            return self.fill_height;
        }
        let uf = u.clamp(0.0, 1.0) * (width - 1) as f32;
        let vf = v.clamp(0.0, 1.0) * (height - 1) as f32;
        let i0 = uf.floor() as u32;
        let j0 = vf.floor() as u32;
        let i1 = (i0 + 1).min(width - 1);
        let j1 = (j0 + 1).min(height - 1);
        let tx = uf - i0 as f32;
        let ty = vf - j0 as f32;
        let a = self.samples[(j0 * width + i0) as usize];
        let b = self.samples[(j0 * width + i1) as usize];
        let c = self.samples[(j1 * width + i0) as usize];
        let d = self.samples[(j1 * width + i1) as usize];
        let top = a + (b - a) * tx;
        let bot = c + (d - c) * tx;
        top + (bot - top) * ty
    }

    /// Min/max of the paint buffer (for GPU height-range tracking).
    pub fn sample_range(&self) -> (f32, f32) {
        let n = (self.width.max(1) as usize).saturating_mul(self.height.max(1) as usize);
        if self.samples.len() != n {
            return (self.fill_height, self.fill_height);
        }
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &s in &self.samples {
            lo = lo.min(s);
            hi = hi.max(s);
        }
        if !lo.is_finite() {
            (self.fill_height, self.fill_height)
        } else {
            (lo, hi)
        }
    }
}

#[derive(Deserialize)]
struct SculptParamsWire {
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    #[serde(default)]
    resolution: Option<u32>,
    samples: Vec<f32>,
    fill_height: f32,
}

impl<'de> Deserialize<'de> for SculptParams {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SculptParamsWire::deserialize(deserializer)?;
        let width = wire.width.or(wire.resolution).unwrap_or(512);
        let height = wire.height.or(wire.resolution).unwrap_or(width);
        Ok(Self {
            width,
            height,
            samples: wire.samples,
            fill_height: wire.fill_height,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatParams {
    pub height: f32,
}

impl Default for FlatParams {
    fn default() -> Self {
        Self { height: 0.0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RampParams {
    pub height_min: f32,
    pub height_max: f32,
    /// Angle in radians; 0 = +X.
    pub direction: f32,
}

impl Default for RampParams {
    fn default() -> Self {
        Self {
            height_min: 0.0,
            height_max: 100.0,
            direction: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SculptParams;

    fn peak(p: &SculptParams) -> f32 {
        p.samples.iter().copied().fold(f32::MIN, f32::max)
    }
    fn floor(p: &SculptParams) -> f32 {
        p.samples.iter().copied().fold(f32::MAX, f32::min)
    }

    #[test]
    fn flatten_mode_settles_toward_mean_and_never_runs_away() {
        // Raise a bump, then flatten it. Flatten (mode 3) must pull the footprint
        // toward its mean without ever exceeding the pre-flatten range — the old
        // behavior fell through to raise (mode 0) and grew every stamp.
        let mut p = SculptParams::filled(64, 0.0);
        p.stamp_circle(0.5, 0.5, 0.3, 50.0, 0);
        let (peak0, floor0) = (peak(&p), floor(&p));
        let center0 = p.samples[(32 * 64 + 32) as usize];

        for _ in 0..12 {
            p.stamp_circle(0.5, 0.5, 0.3, 1.0, 3);
        }

        assert!(
            peak(&p) <= peak0 + 1e-3,
            "flatten raised the peak: {peak0} -> {}",
            peak(&p)
        );
        assert!(
            floor(&p) >= floor0 - 1e-3,
            "flatten sank the floor: {floor0} -> {}",
            floor(&p)
        );
        let center1 = p.samples[(32 * 64 + 32) as usize];
        assert!(
            center1 < center0,
            "brush centre was not flattened down: {center0} -> {center1}"
        );
    }

    #[test]
    fn smooth_spread_broadens_foundation_edge_transition() {
        let mut source = SculptParams::filled(129, 0.0);
        for y in 0..source.height {
            for x in source.width / 2..source.width {
                source.samples[(y * source.width + x) as usize] = 10.0;
            }
        }
        let mut narrow = source.clone();
        let mut wide = source.clone();
        assert!(narrow.stamp_circle_with_smooth_spread(0.5, 0.5, 0.45, 1.0, 2, 1));
        assert!(wide.stamp_circle_with_smooth_spread(0.5, 0.5, 0.45, 1.0, 2, 8));

        let row = source.height / 2;
        let transition_width = |params: &SculptParams| {
            (0..params.width)
                .filter(|&x| {
                    let value = params.samples[(row * params.width + x) as usize];
                    value > 1.0e-3 && value < 10.0 - 1.0e-3
                })
                .count()
        };
        assert!(
            transition_width(&wide) >= transition_width(&narrow).saturating_mul(3),
            "foundation Smooth spread did not materially broaden the edge"
        );
    }
}
