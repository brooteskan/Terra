use super::MaskField;
use crate::raster::{
    resample_f32_grid, GridDimensions, RasterResizeError, RasterResizeLimits, RasterSemantic,
};
use serde::{Deserialize, Serialize};

/// Viewport editing tools for a reusable painted mask layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MaskPaintTool {
    #[default]
    Paint,
    Erase,
    Smooth,
    FloodFill,
}

impl MaskPaintTool {
    pub fn label(self) -> &'static str {
        match self {
            Self::Paint => "Paint",
            Self::Erase => "Erase",
            Self::Smooth => "Smooth",
            Self::FloodFill => "Flood Fill",
        }
    }

    pub fn cycle(self) -> Self {
        match self {
            Self::Paint => Self::Erase,
            Self::Erase => Self::Smooth,
            Self::Smooth => Self::FloodFill,
            Self::FloodFill => Self::Paint,
        }
    }
}

/// Dense UV-space paint values stored with a mask asset.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PaintBuffer {
    pub width: u32,
    pub height: u32,
    pub samples: Vec<f32>,
}

impl PaintBuffer {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            samples: vec![0.0; (width * height) as usize],
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
            RasterSemantic::Mask,
            limits,
        )?;
        Ok(Self {
            width: dimensions.width,
            height: dimensions.height,
            samples,
        })
    }

    /// Backwards-compatible paint/erase stamp.
    pub fn stamp_circle(&mut self, u: f32, v: f32, radius_uv: f32, strength: f32, erase: bool) {
        self.edit_circle(
            u,
            v,
            radius_uv,
            strength,
            0.0,
            if erase {
                MaskPaintTool::Erase
            } else {
                MaskPaintTool::Paint
            },
        );
    }

    /// Paint, erase, or smooth a circular brush stamp.
    ///
    /// `hardness` is 0 for a fully soft edge and 1 for a hard-edged brush.
    pub fn edit_circle(
        &mut self,
        u: f32,
        v: f32,
        radius_uv: f32,
        strength: f32,
        hardness: f32,
        tool: MaskPaintTool,
    ) {
        if matches!(tool, MaskPaintTool::FloodFill) {
            // Flood fill is a one-shot op via `flood_fill`, not a circular stamp.
            self.flood_fill(u, v, strength.clamp(0.0, 1.0), 0.08);
            return;
        }
        if self.width == 0
            || self.height == 0
            || self.samples.len() != (self.width * self.height) as usize
        {
            return;
        }
        let radius = radius_uv.max(1e-6);
        let min_i = ((u - radius) * self.width as f32).floor().max(0.0) as u32;
        let max_i = ((u + radius) * self.width as f32)
            .ceil()
            .min(self.width as f32 - 1.0) as u32;
        let min_j = ((v - radius) * self.height as f32).floor().max(0.0) as u32;
        let max_j = ((v + radius) * self.height as f32)
            .ceil()
            .min(self.height as f32 - 1.0) as u32;
        let before = (tool == MaskPaintTool::Smooth).then(|| self.samples.clone());
        let hardness = hardness.clamp(0.0, 1.0);
        for j in min_j..=max_j {
            for i in min_i..=max_i {
                let x = (i as f32 + 0.5) / self.width as f32;
                let y = (j as f32 + 0.5) / self.height as f32;
                let d = ((x - u).powi(2) + (y - v).powi(2)).sqrt() / radius;
                if d <= 1.0 {
                    let edge_t = if hardness >= 1.0 || d <= hardness {
                        0.0
                    } else {
                        (d - hardness) / (1.0 - hardness)
                    };
                    let falloff = 1.0 - edge_t * edge_t * (3.0 - 2.0 * edge_t);
                    let amount = (falloff * strength).clamp(0.0, 1.0);
                    let idx = (j * self.width + i) as usize;
                    let next = match tool {
                        MaskPaintTool::Paint => self.samples[idx] + amount,
                        MaskPaintTool::Erase => self.samples[idx] - amount,
                        MaskPaintTool::FloodFill => self.samples[idx], // handled above
                        MaskPaintTool::Smooth => {
                            let source = before.as_ref().expect("smooth snapshot");
                            let mut sum = 0.0;
                            let mut count = 0.0;
                            for dj in -1..=1 {
                                for di in -1..=1 {
                                    let si = (i as i32 + di).clamp(0, self.width as i32 - 1);
                                    let sj = (j as i32 + dj).clamp(0, self.height as i32 - 1);
                                    sum += source[(sj as u32 * self.width + si as u32) as usize];
                                    count += 1.0;
                                }
                            }
                            source[idx] + (sum / count - source[idx]) * amount
                        }
                    };
                    self.samples[idx] = next.clamp(0.0, 1.0);
                }
            }
        }
    }

    pub fn clear(&mut self) {
        self.samples.fill(0.0);
    }

    pub fn fill(&mut self) {
        self.samples.fill(1.0);
    }

    pub fn flip_x(&mut self) {
        if self.samples.len() != (self.width * self.height) as usize {
            return;
        }
        for j in 0..self.height as usize {
            let row = &mut self.samples[j * self.width as usize..(j + 1) * self.width as usize];
            row.reverse();
        }
    }

    pub fn flip_y(&mut self) {
        if self.samples.len() != (self.width * self.height) as usize {
            return;
        }
        let w = self.width as usize;
        for j in 0..self.height as usize / 2 {
            for i in 0..w {
                self.samples
                    .swap(j * w + i, (self.height as usize - 1 - j) * w + i);
            }
        }
    }

    pub fn rotate_left(&mut self) {
        self.rotate_quarter(false);
    }

    pub fn rotate_right(&mut self) {
        self.rotate_quarter(true);
    }

    fn rotate_quarter(&mut self, clockwise: bool) {
        if self.width == 0
            || self.height == 0
            || self.samples.len() != (self.width * self.height) as usize
        {
            return;
        }
        let old_w = self.width as usize;
        let old_h = self.height as usize;
        let mut rotated = vec![0.0; self.samples.len()];
        for y in 0..old_h {
            for x in 0..old_w {
                let (nx, ny) = if clockwise {
                    (old_h - 1 - y, x)
                } else {
                    (y, old_w - 1 - x)
                };
                rotated[ny * old_h + nx] = self.samples[y * old_w + x];
            }
        }
        self.width = old_h as u32;
        self.height = old_w as u32;
        self.samples = rotated;
    }

    /// Bake an RGBA8 viewport overlay: RGB = `color`, A = paint weight.
    pub fn bake_overlay_rgba(&self, color: [f32; 3]) -> Vec<u8> {
        let r = (color[0].clamp(0.0, 1.0) * 255.0).round() as u8;
        let g = (color[1].clamp(0.0, 1.0) * 255.0).round() as u8;
        let b = (color[2].clamp(0.0, 1.0) * 255.0).round() as u8;
        let mut rgba = Vec::with_capacity(self.samples.len() * 4);
        for &v in &self.samples {
            let a = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            rgba.extend_from_slice(&[r, g, b, a]);
        }
        rgba
    }

    pub fn sample_uv(&self, u: f32, v: f32) -> f32 {
        if self.width == 0
            || self.height == 0
            || self.samples.len() != (self.width * self.height) as usize
        {
            return 0.0;
        }
        let i = (u.clamp(0.0, 1.0) * self.width as f32) as u32;
        let j = (v.clamp(0.0, 1.0) * self.height as f32) as u32;
        self.samples[(j.min(self.height - 1) * self.width + i.min(self.width - 1)) as usize]
    }

    /// 4-connected flood fill from UV seed. Fills cells within `tolerance` of the seed value.
    pub fn flood_fill(&mut self, u: f32, v: f32, value: f32, tolerance: f32) {
        if self.width == 0
            || self.height == 0
            || self.samples.len() != (self.width * self.height) as usize
        {
            return;
        }
        let w = self.width as usize;
        let h = self.height as usize;
        let sx = ((u.clamp(0.0, 1.0) * (self.width - 1) as f32).round() as usize).min(w - 1);
        let sy = ((v.clamp(0.0, 1.0) * (self.height - 1) as f32).round() as usize).min(h - 1);
        let seed = self.samples[sy * w + sx];
        let target = value.clamp(0.0, 1.0);
        let tol = tolerance.max(0.0);
        if (seed - target).abs() <= 1e-5 {
            return;
        }
        let mut stack = vec![(sx, sy)];
        let mut visited = vec![false; w * h];
        while let Some((x, y)) = stack.pop() {
            let idx = y * w + x;
            if visited[idx] {
                continue;
            }
            visited[idx] = true;
            let cur = self.samples[idx];
            if (cur - seed).abs() > tol {
                continue;
            }
            self.samples[idx] = target;
            if x > 0 {
                stack.push((x - 1, y));
            }
            if x + 1 < w {
                stack.push((x + 1, y));
            }
            if y > 0 {
                stack.push((x, y - 1));
            }
            if y + 1 < h {
                stack.push((x, y + 1));
            }
        }
    }

    /// Fill a closed polygon given normalized UV vertices (scanline).
    pub fn fill_polygon(&mut self, points: &[(f32, f32)], value: f32) {
        if points.len() < 3
            || self.width == 0
            || self.height == 0
            || self.samples.len() != (self.width * self.height) as usize
        {
            return;
        }
        let value = value.clamp(0.0, 1.0);
        let w = self.width as i32;
        let h = self.height as i32;
        let mut min_j = h;
        let mut max_j = 0;
        let pix: Vec<(f32, f32)> = points
            .iter()
            .map(|(u, v)| {
                let x = u.clamp(0.0, 1.0) * (self.width - 1) as f32;
                let y = v.clamp(0.0, 1.0) * (self.height - 1) as f32;
                min_j = min_j.min(y.floor() as i32);
                max_j = max_j.max(y.ceil() as i32);
                (x, y)
            })
            .collect();
        min_j = min_j.clamp(0, h - 1);
        max_j = max_j.clamp(0, h - 1);
        let n = pix.len();
        for j in min_j..=max_j {
            let y = j as f32 + 0.5;
            let mut xs = Vec::new();
            for i in 0..n {
                let (x0, y0) = pix[i];
                let (x1, y1) = pix[(i + 1) % n];
                if (y0 <= y && y1 > y) || (y1 <= y && y0 > y) {
                    let t = (y - y0) / (y1 - y0);
                    xs.push(x0 + t * (x1 - x0));
                }
            }
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mut k = 0;
            while k + 1 < xs.len() {
                let x_start = xs[k].ceil() as i32;
                let x_end = xs[k + 1].floor() as i32;
                for i in x_start.max(0)..=x_end.min(w - 1) {
                    self.samples[(j * w + i) as usize] = value;
                }
                k += 2;
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaintStroke {
    pub points: Vec<(f32, f32)>,
    pub radius: f32,
    pub strength: f32,
    pub erase: bool,
}

impl PaintStroke {
    pub fn apply(&self, field: &mut MaskField) {
        let dx = field.metrics.dx();
        let dz = field.metrics.dz();
        let r_samples = (self.radius / dx.min(dz)).max(1.0);
        for &(wx, wz) in &self.points {
            let (ci, cj) = field.metrics.sample_index(wx, wz);
            let r = r_samples as i32;
            for dj in -r..=r {
                for di in -r..=r {
                    let ii = ci as i32 + di;
                    let jj = cj as i32 + dj;
                    if ii < 0
                        || jj < 0
                        || ii >= field.metrics.width as i32
                        || jj >= field.metrics.height as i32
                    {
                        continue;
                    }
                    let dist = ((di * di + dj * dj) as f32).sqrt() / r_samples;
                    if dist > 1.0 {
                        continue;
                    }
                    let falloff = (1.0 - dist * dist) * self.strength;
                    let cur = field.get(ii as u32, jj as u32);
                    let next = if self.erase {
                        cur - falloff
                    } else {
                        cur + falloff
                    };
                    field.set(ii as u32, jj as u32, next);
                }
            }
        }
    }
}

#[cfg(test)]
mod paint_buffer_tests {
    use super::{MaskPaintTool, PaintBuffer};

    #[test]
    fn flood_fill_fills_connected_zeros() {
        let mut buf = PaintBuffer::new(16, 16);
        // Seed a barrier.
        for i in 0..16 {
            buf.samples[(8 * 16 + i) as usize] = 1.0;
        }
        buf.flood_fill(0.25, 0.25, 0.8, 0.05);
        assert!(buf.sample_uv(0.25, 0.25) > 0.7);
        // Below the barrier should stay empty.
        assert!(buf.sample_uv(0.25, 0.75) < 0.1);
    }

    #[test]
    fn polygon_fill_covers_interior() {
        let mut buf = PaintBuffer::new(32, 32);
        buf.fill_polygon(&[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)], 1.0);
        assert!(buf.sample_uv(0.5, 0.5) > 0.9);
        assert!(buf.sample_uv(0.05, 0.05) < 0.1);
    }

    #[test]
    fn world_creator_style_edit_tools_modify_the_same_mask() {
        let mut buf = PaintBuffer::new(32, 32);
        buf.edit_circle(0.5, 0.5, 0.2, 1.0, 0.5, MaskPaintTool::Paint);
        assert!(buf.sample_uv(0.5, 0.5) > 0.9);
        buf.edit_circle(0.5, 0.5, 0.1, 1.0, 1.0, MaskPaintTool::Erase);
        assert!(buf.sample_uv(0.5, 0.5) < 0.1);

        buf.fill();
        buf.samples[16 * 32 + 16] = 0.0;
        buf.edit_circle(0.5, 0.5, 0.15, 1.0, 1.0, MaskPaintTool::Smooth);
        assert!(buf.sample_uv(0.5, 0.5) > 0.5);
        buf.clear();
        assert!(buf.samples.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn mask_actions_flip_and_rotate_rectangular_buffers() {
        let mut buf = PaintBuffer {
            width: 3,
            height: 2,
            samples: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        buf.flip_x();
        assert_eq!(buf.samples, vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
        buf.flip_y();
        assert_eq!(buf.samples, vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        buf.rotate_right();
        assert_eq!((buf.width, buf.height), (2, 3));
        assert_eq!(buf.samples, vec![3.0, 6.0, 2.0, 5.0, 1.0, 4.0]);
        buf.rotate_left();
        assert_eq!((buf.width, buf.height), (3, 2));
        assert_eq!(buf.samples, vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
    }
}
