//! Layer parameter kinds (split by family).

use serde::{Deserialize, Serialize};

/// Paintable foundation heights in meters (normalized UV grid).

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SculptParams {
    /// Resolution of the paint buffer (square).
    pub resolution: u32,
    /// Heights in world meters, row-major, length = resolutionÂ².
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
        let n = (resolution as usize).saturating_mul(resolution as usize);
        Self {
            resolution,
            samples: vec![fill_height; n],
            fill_height,
        }
    }

    pub fn ensure_buffer(&mut self) {
        let n = (self.resolution as usize).saturating_mul(self.resolution as usize);
        if self.samples.len() != n {
            self.samples = vec![self.fill_height; n];
        }
    }

    pub fn reset(&mut self) {
        self.ensure_buffer();
        for s in &mut self.samples {
            *s = self.fill_height;
        }
    }

    /// Soft circular stamp. `mode`: 0 = raise, 1 = lower, 2 = smooth.
    pub fn stamp_circle(&mut self, u: f32, v: f32, radius_uv: f32, strength: f32, mode: u8) {
        self.ensure_buffer();
        let res = self.resolution;
        if res == 0 {
            return;
        }
        let radius = radius_uv.max(1e-6);
        let min_i = ((u - radius) * res as f32).floor().max(0.0) as u32;
        let max_i = ((u + radius) * res as f32).ceil().min(res as f32 - 1.0) as u32;
        let min_j = ((v - radius) * res as f32).floor().max(0.0) as u32;
        let max_j = ((v + radius) * res as f32).ceil().min(res as f32 - 1.0) as u32;

        if mode == 2 {
            // Smooth: blend toward local neighborhood average.
            let mut updates: Vec<(usize, f32)> = Vec::new();
            for j in min_j..=max_j {
                for i in min_i..=max_i {
                    let x = (i as f32 + 0.5) / res as f32;
                    let y = (j as f32 + 0.5) / res as f32;
                    let d = ((x - u).powi(2) + (y - v).powi(2)).sqrt() / radius;
                    if d > 1.0 {
                        continue;
                    }
                    let falloff = (1.0 - d * d) * strength.clamp(0.0, 1.0);
                    let idx = (j * res + i) as usize;
                    let mut sum = 0.0;
                    let mut count = 0.0;
                    for dj in -1i32..=1 {
                        for di in -1i32..=1 {
                            let ii = i as i32 + di;
                            let jj = j as i32 + dj;
                            if ii < 0 || jj < 0 || ii >= res as i32 || jj >= res as i32 {
                                continue;
                            }
                            sum += self.samples[(jj as u32 * res + ii as u32) as usize];
                            count += 1.0;
                        }
                    }
                    let avg = if count > 0.0 {
                        sum / count
                    } else {
                        self.samples[idx]
                    };
                    let cur = self.samples[idx];
                    updates.push((idx, cur + (avg - cur) * falloff));
                }
            }
            for (idx, val) in updates {
                self.samples[idx] = val;
            }
            return;
        }

        let delta_sign = if mode == 1 { -1.0 } else { 1.0 };
        // strength is meters of peak displacement per stamp
        let peak = strength.max(0.0) * delta_sign;
        for j in min_j..=max_j {
            for i in min_i..=max_i {
                let x = (i as f32 + 0.5) / res as f32;
                let y = (j as f32 + 0.5) / res as f32;
                let d = ((x - u).powi(2) + (y - v).powi(2)).sqrt() / radius;
                if d <= 1.0 {
                    let amount = (1.0 - d * d) * peak;
                    let sample = &mut self.samples[(j * res + i) as usize];
                    *sample += amount;
                }
            }
        }
    }

    pub fn sample_bilinear(&self, u: f32, v: f32) -> f32 {
        let res = self.resolution.max(1);
        let n = (res as usize).saturating_mul(res as usize);
        if self.samples.len() != n {
            return self.fill_height;
        }
        let uf = u.clamp(0.0, 1.0) * (res - 1) as f32;
        let vf = v.clamp(0.0, 1.0) * (res - 1) as f32;
        let i0 = uf.floor() as u32;
        let j0 = vf.floor() as u32;
        let i1 = (i0 + 1).min(res - 1);
        let j1 = (j0 + 1).min(res - 1);
        let tx = uf - i0 as f32;
        let ty = vf - j0 as f32;
        let a = self.samples[(j0 * res + i0) as usize];
        let b = self.samples[(j0 * res + i1) as usize];
        let c = self.samples[(j1 * res + i0) as usize];
        let d = self.samples[(j1 * res + i1) as usize];
        let top = a + (b - a) * tx;
        let bot = c + (d - c) * tx;
        top + (bot - top) * ty
    }

    /// Min/max of the paint buffer (for GPU height-range tracking).
    pub fn sample_range(&self) -> (f32, f32) {
        let res = self.resolution.max(1);
        let n = (res as usize).saturating_mul(res as usize);
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
