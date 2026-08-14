//! Biome paint layers (WC Biome Layers) — splat weights + paint-while-sculpt.

use crate::ids::LayerId;
use crate::mask::PaintBuffer;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Stable identity for a biome paint / splat layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BiomeLayerId(pub Uuid);

impl BiomeLayerId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for BiomeLayerId {
    fn default() -> Self {
        Self::new()
    }
}

/// Brush mode for biome paint (WC Biome Layers tools).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BiomePaintTool {
    #[default]
    Paint,
    Erase,
    Smooth,
    Replace,
    Add,
    FloodFill,
    GradientFill,
    PolygonFill,
    Normalize,
    Sample,
    AutoTransition,
    RaisePaint,
    LowerPaint,
    FlattenPaint,
    Raise,
    Lower,
    Flatten,
}

impl BiomePaintTool {
    pub fn label(self) -> &'static str {
        match self {
            BiomePaintTool::Paint => "Paint",
            BiomePaintTool::Erase => "Erase",
            BiomePaintTool::Smooth => "Smooth",
            BiomePaintTool::Replace => "Replace",
            BiomePaintTool::Add => "Add",
            BiomePaintTool::FloodFill => "Flood Fill",
            BiomePaintTool::GradientFill => "Gradient Fill",
            BiomePaintTool::PolygonFill => "Polygon Fill",
            BiomePaintTool::Normalize => "Normalize",
            BiomePaintTool::Sample => "Sample",
            BiomePaintTool::AutoTransition => "Auto Transition",
            BiomePaintTool::RaisePaint => "Raise + Paint",
            BiomePaintTool::LowerPaint => "Lower + Paint",
            BiomePaintTool::FlattenPaint => "Flatten + Paint",
            BiomePaintTool::Raise => "Raise",
            BiomePaintTool::Lower => "Lower",
            BiomePaintTool::Flatten => "Flatten",
        }
    }

    pub fn paints_mask(self) -> bool {
        matches!(
            self,
            BiomePaintTool::Paint
                | BiomePaintTool::Erase
                | BiomePaintTool::Smooth
                | BiomePaintTool::Replace
                | BiomePaintTool::Add
                | BiomePaintTool::FloodFill
                | BiomePaintTool::GradientFill
                | BiomePaintTool::PolygonFill
                | BiomePaintTool::AutoTransition
                | BiomePaintTool::RaisePaint
                | BiomePaintTool::LowerPaint
                | BiomePaintTool::FlattenPaint
        )
    }

    pub fn sculpts(self) -> bool {
        matches!(
            self,
            BiomePaintTool::RaisePaint
                | BiomePaintTool::LowerPaint
                | BiomePaintTool::FlattenPaint
                | BiomePaintTool::Raise
                | BiomePaintTool::Lower
                | BiomePaintTool::Flatten
        )
    }

    pub fn cycle(self) -> Self {
        match self {
            BiomePaintTool::Paint => BiomePaintTool::Erase,
            BiomePaintTool::Erase => BiomePaintTool::Smooth,
            BiomePaintTool::Smooth => BiomePaintTool::Replace,
            BiomePaintTool::Replace => BiomePaintTool::Add,
            BiomePaintTool::Add => BiomePaintTool::Normalize,
            BiomePaintTool::Normalize => BiomePaintTool::RaisePaint,
            BiomePaintTool::RaisePaint => BiomePaintTool::LowerPaint,
            BiomePaintTool::LowerPaint => BiomePaintTool::FlattenPaint,
            BiomePaintTool::FlattenPaint => BiomePaintTool::Raise,
            BiomePaintTool::Raise => BiomePaintTool::Lower,
            BiomePaintTool::Lower => BiomePaintTool::Flatten,
            BiomePaintTool::Flatten => BiomePaintTool::Paint,
            BiomePaintTool::FloodFill
            | BiomePaintTool::GradientFill
            | BiomePaintTool::PolygonFill
            | BiomePaintTool::Sample
            | BiomePaintTool::AutoTransition => BiomePaintTool::Paint,
        }
    }
}

/// Per-biome weight channel inside a splat layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiomeWeightChannel {
    pub biome_id: LayerId,
    pub paint: PaintBuffer,
}

/// Multi-biome splat / placement layer (WC Biome Layers).
///
/// Weights are painted per biome and composited with procedural biome distributions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiomeLayer {
    pub id: BiomeLayerId,
    pub name: String,
    /// When true, painting overwrites lower hierarchical weights; else adds.
    #[serde(default)]
    pub overwrite: bool,
    #[serde(default)]
    pub show_mask: bool,
    #[serde(default)]
    pub show_biome_colors: bool,
    /// Isolate active biome in overlay.
    #[serde(default)]
    pub isolate_active: bool,
    /// Show weight heatmap instead of solid colour.
    #[serde(default)]
    pub show_heatmap: bool,
    #[serde(default = "default_mask_vis")]
    pub mask_visibility: f32,
    pub channels: Vec<BiomeWeightChannel>,
}

fn default_mask_vis() -> f32 {
    0.55
}

impl BiomeLayer {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: BiomeLayerId::new(),
            name: name.into(),
            overwrite: true,
            show_mask: false,
            show_biome_colors: true,
            isolate_active: false,
            show_heatmap: false,
            mask_visibility: 0.55,
            channels: Vec::new(),
        }
    }

    pub fn ensure_channel(&mut self, biome_id: LayerId, w: u32, h: u32) -> &mut BiomeWeightChannel {
        if let Some(idx) = self.channels.iter().position(|c| c.biome_id == biome_id) {
            return &mut self.channels[idx];
        }
        self.channels.push(BiomeWeightChannel {
            biome_id,
            paint: PaintBuffer::new(w, h),
        });
        self.channels.last_mut().unwrap()
    }

    /// Stamp a circular brush into the active biome channel.
    pub fn stamp(
        &mut self,
        biome_id: LayerId,
        u: f32,
        v: f32,
        radius: f32,
        strength: f32,
        erase: bool,
        resolution: u32,
    ) {
        let ch = self.ensure_channel(biome_id, resolution, resolution);
        ch.paint.stamp_circle(u, v, radius, strength, erase);
        if self.overwrite && !erase {
            for other in &mut self.channels {
                if other.biome_id == biome_id {
                    continue;
                }
                other
                    .paint
                    .stamp_circle(u, v, radius, strength * 0.85, true);
            }
        }
    }

    /// Additive stamp that does not suppress sibling channels.
    pub fn stamp_add(
        &mut self,
        biome_id: LayerId,
        u: f32,
        v: f32,
        radius: f32,
        strength: f32,
        resolution: u32,
    ) {
        let prev = self.overwrite;
        self.overwrite = false;
        self.stamp(biome_id, u, v, radius, strength, false, resolution);
        self.overwrite = prev;
    }

    /// Replace: set active channel toward strength and zero siblings under brush.
    pub fn stamp_replace(
        &mut self,
        biome_id: LayerId,
        u: f32,
        v: f32,
        radius: f32,
        strength: f32,
        resolution: u32,
    ) {
        self.ensure_channel(biome_id, resolution, resolution);
        // Erase all, then paint target.
        for other in &mut self.channels {
            other.paint.stamp_circle(u, v, radius, 1.0, true);
        }
        if let Some(ch) = self.channels.iter_mut().find(|c| c.biome_id == biome_id) {
            ch.paint.stamp_circle(u, v, radius, strength, false);
        }
    }

    /// Smooth weights under the brush by averaging with neighbours.
    pub fn smooth_at(&mut self, u: f32, v: f32, radius: f32, strength: f32) {
        let strength = strength.clamp(0.0, 1.0);
        for ch in &mut self.channels {
            smooth_paint_circle(&mut ch.paint, u, v, radius, strength);
        }
    }

    /// Flood-fill the active biome channel from a UV seed.
    pub fn flood_fill(
        &mut self,
        biome_id: LayerId,
        u: f32,
        v: f32,
        value: f32,
        tolerance: f32,
        resolution: u32,
    ) {
        {
            let ch = self.ensure_channel(biome_id, resolution, resolution);
            ch.paint.flood_fill(u, v, value, tolerance);
        }
        if !self.overwrite || value <= 1e-4 {
            return;
        }
        let filled = self
            .channels
            .iter()
            .find(|c| c.biome_id == biome_id)
            .map(|c| c.paint.samples.clone());
        let Some(filled) = filled else {
            return;
        };
        for other in &mut self.channels {
            if other.biome_id == biome_id {
                continue;
            }
            if other.paint.samples.len() != filled.len() {
                continue;
            }
            for (dst, &src) in other.paint.samples.iter_mut().zip(filled.iter()) {
                if src > 0.5 {
                    *dst = (*dst * 0.15).clamp(0.0, 1.0);
                }
            }
        }
    }

    /// Fill a polygon into the active biome channel.
    pub fn fill_polygon(
        &mut self,
        biome_id: LayerId,
        points: &[(f32, f32)],
        value: f32,
        resolution: u32,
    ) {
        {
            let ch = self.ensure_channel(biome_id, resolution, resolution);
            ch.paint.fill_polygon(points, value);
        }
        if self.overwrite && value > 1e-4 {
            for other in &mut self.channels {
                if other.biome_id == biome_id {
                    continue;
                }
                other.paint.fill_polygon(points, 0.0);
            }
        }
    }

    /// Normalize all channel weights at every sample so they sum to 1 (where any > 0).
    pub fn normalize_all(&mut self) {
        if self.channels.is_empty() {
            return;
        }
        let w = self.channels[0].paint.width;
        let h = self.channels[0].paint.height;
        if w == 0 || h == 0 {
            return;
        }
        for j in 0..h {
            for i in 0..w {
                let idx = (j * w + i) as usize;
                let mut sum = 0.0_f32;
                for ch in &self.channels {
                    if ch.paint.width == w && ch.paint.height == h {
                        sum += ch.paint.samples.get(idx).copied().unwrap_or(0.0);
                    }
                }
                if sum > 1e-6 {
                    for ch in &mut self.channels {
                        if ch.paint.width == w
                            && ch.paint.height == h
                            && idx < ch.paint.samples.len()
                        {
                            ch.paint.samples[idx] /= sum;
                        }
                    }
                }
            }
        }
    }

    /// Sample normalised weight for `biome_id` at UV (nearest).
    pub fn weight_at(&self, biome_id: LayerId, u: f32, v: f32) -> f32 {
        let Some(ch) = self.channels.iter().find(|c| c.biome_id == biome_id) else {
            return 0.0;
        };
        sample_paint(&ch.paint, u, v)
    }

    /// Normalized weights across all channels at UV.
    pub fn normalized_weights_at(&self, u: f32, v: f32) -> Vec<(LayerId, f32)> {
        let mut raw: Vec<(LayerId, f32)> = self
            .channels
            .iter()
            .map(|c| (c.biome_id, sample_paint(&c.paint, u, v)))
            .collect();
        let sum: f32 = raw.iter().map(|(_, w)| *w).sum();
        if sum > 1e-6 {
            for (_, w) in &mut raw {
                *w /= sum;
            }
        }
        raw
    }

    /// Bake a colourised placement preview (RGBA). Uses definition/group colours via `color_of`.
    pub fn bake_color_rgba(
        &self,
        width: u32,
        height: u32,
        color_of: &dyn Fn(LayerId) -> [f32; 3],
        isolate: Option<LayerId>,
    ) -> Vec<u8> {
        let n = (width * height) as usize;
        let mut rgba = vec![0u8; n * 4];
        if self.channels.is_empty() || width == 0 || height == 0 {
            return rgba;
        }
        for j in 0..height {
            for i in 0..width {
                let u = if width > 1 {
                    i as f32 / (width - 1) as f32
                } else {
                    0.5
                };
                let v = if height > 1 {
                    j as f32 / (height - 1) as f32
                } else {
                    0.5
                };
                let mut r = 0.0f32;
                let mut g = 0.0f32;
                let mut b = 0.0f32;
                let mut wsum = 0.0f32;
                for ch in &self.channels {
                    if isolate.is_some_and(|id| id != ch.biome_id) {
                        continue;
                    }
                    let w = sample_paint(&ch.paint, u, v);
                    if w <= 1e-5 {
                        continue;
                    }
                    let c = color_of(ch.biome_id);
                    r += c[0] * w;
                    g += c[1] * w;
                    b += c[2] * w;
                    wsum += w;
                }
                let idx = ((j * width + i) * 4) as usize;
                if wsum > 1e-5 {
                    let inv = 1.0 / wsum;
                    let a = wsum.clamp(0.0, 1.0);
                    rgba[idx] = (r * inv * 255.0).round() as u8;
                    rgba[idx + 1] = (g * inv * 255.0).round() as u8;
                    rgba[idx + 2] = (b * inv * 255.0).round() as u8;
                    rgba[idx + 3] = (a * 220.0).round() as u8;
                } else {
                    rgba[idx + 3] = 0;
                }
            }
        }
        rgba
    }
}

fn sample_paint(paint: &PaintBuffer, u: f32, v: f32) -> f32 {
    let w = paint.width.max(1);
    let h = paint.height.max(1);
    let x = (u.clamp(0.0, 1.0) * (w - 1) as f32).round() as u32;
    let y = (v.clamp(0.0, 1.0) * (h - 1) as f32).round() as u32;
    let idx = (y * w + x) as usize;
    paint.samples.get(idx).copied().unwrap_or(0.0)
}

fn smooth_paint_circle(paint: &mut PaintBuffer, u: f32, v: f32, radius: f32, strength: f32) {
    if paint.width == 0 || paint.height == 0 {
        return;
    }
    let radius = radius.max(1e-6);
    let min_i = ((u - radius) * paint.width as f32).floor().max(0.0) as u32;
    let max_i = ((u + radius) * paint.width as f32)
        .ceil()
        .min(paint.width as f32 - 1.0) as u32;
    let min_j = ((v - radius) * paint.height as f32).floor().max(0.0) as u32;
    let max_j = ((v + radius) * paint.height as f32)
        .ceil()
        .min(paint.height as f32 - 1.0) as u32;
    let src = paint.samples.clone();
    for j in min_j..=max_j {
        for i in min_i..=max_i {
            let x = (i as f32 + 0.5) / paint.width as f32;
            let y = (j as f32 + 0.5) / paint.height as f32;
            let d = ((x - u).powi(2) + (y - v).powi(2)).sqrt() / radius;
            if d > 1.0 {
                continue;
            }
            let mut sum = 0.0;
            let mut n = 0.0;
            for dj in -1i32..=1 {
                for di in -1i32..=1 {
                    let ii = i as i32 + di;
                    let jj = j as i32 + dj;
                    if ii < 0 || jj < 0 || ii >= paint.width as i32 || jj >= paint.height as i32 {
                        continue;
                    }
                    sum += src[(jj as u32 * paint.width + ii as u32) as usize];
                    n += 1.0;
                }
            }
            if n > 0.0 {
                let avg = sum / n;
                let idx = (j * paint.width + i) as usize;
                let cur = paint.samples[idx];
                let amt = (1.0 - d * d) * strength;
                paint.samples[idx] = cur * (1.0 - amt) + avg * amt;
            }
        }
    }
}

/// Spatial transform for shape layers (stamp / polygon / path placement).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeTransform {
    pub offset_x: f32,
    pub offset_z: f32,
    #[serde(default = "default_scale")]
    pub scale: f32,
    pub rotation_deg: f32,
    /// Border blend fade (0–1 of extent).
    #[serde(default)]
    pub blend_size: f32,
    #[serde(default)]
    pub blend_roundness: f32,
}

fn default_scale() -> f32 {
    1.0
}

impl Default for ShapeTransform {
    fn default() -> Self {
        Self {
            offset_x: 0.0,
            offset_z: 0.0,
            scale: 1.0,
            rotation_deg: 0.0,
            blend_size: 0.0,
            blend_roundness: 0.0,
        }
    }
}

impl ShapeTransform {
    /// Map a world-space `(x, z)` position (in meters) to local UV space inside
    /// this transform's placed area, plus a soft edge blend weight in \[0, 1\].
    ///
    /// The shape is centred on the terrain, offset by `(offset_x, offset_z)` (meters),
    /// rotated around the terrain centre by `rotation_deg`, then scaled so `scale == 1`
    /// covers the full terrain extent. Returns `None` when the point falls fully outside
    /// the soft bounds (i.e. blend weight would be 0).
    pub fn world_to_local(
        &self,
        x: f32,
        z: f32,
        world_size_x: f32,
        world_size_z: f32,
    ) -> Option<(f32, f32, f32)> {
        let half_x = (world_size_x * 0.5).max(1e-6);
        let half_z = (world_size_z * 0.5).max(1e-6);

        // World position relative to terrain centre.
        let cx = x - half_x;
        let cz = z - half_z;

        // Undo the shape's placement offset.
        let dx = cx - self.offset_x;
        let dz = cz - self.offset_z;

        // Undo rotation (inverse rotate by -rotation_deg).
        let theta = -self.rotation_deg.to_radians();
        let (sin_t, cos_t) = theta.sin_cos();
        let rx = dx * cos_t - dz * sin_t;
        let rz = dx * sin_t + dz * cos_t;

        // Undo scale (scale == 1 maps the full terrain extent into the shape).
        let scale = self.scale.max(1e-6);
        let lx = rx / scale;
        let lz = rz / scale;

        // Normalise into 0..1 UV over the terrain extent.
        let u = (lx / (2.0 * half_x)) + 0.5;
        let v = (lz / (2.0 * half_z)) + 0.5;

        // Distance from centre in normalised [-1, 1] space, per-axis (box-ish falloff).
        let ax = (u - 0.5).abs() * 2.0;
        let az = (v - 0.5).abs() * 2.0;
        let edge = ax.max(az);

        let blend_size = self.blend_size.clamp(0.0, 1.0);
        let roundness = self.blend_roundness.clamp(0.0, 1.0);
        // Roundness blends the box falloff toward a radial (circular) falloff.
        let radial = (ax * ax + az * az).sqrt();
        let shape_edge = edge * (1.0 - roundness) + radial * roundness;

        let outer = 1.0;
        let inner = (1.0 - blend_size).max(0.0);
        let weight = if shape_edge <= inner {
            1.0
        } else if shape_edge >= outer {
            0.0
        } else {
            let t = (outer - shape_edge) / (outer - inner).max(1e-6);
            t.clamp(0.0, 1.0)
        };

        if weight <= 0.0 {
            return None;
        }
        Some((u, v, weight))
    }
}

/// Hole layer — painted pierce mask for caves / cutouts (incomplete; data model only).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoleLayer {
    pub id: BiomeLayerId,
    pub name: String,
    pub paint: PaintBuffer,
    #[serde(default)]
    pub enabled: bool,
}

impl HoleLayer {
    pub fn new(name: impl Into<String>, resolution: u32) -> Self {
        Self {
            id: BiomeLayerId::new(),
            name: name.into(),
            paint: PaintBuffer::new(resolution, resolution),
            enabled: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::LayerId;

    #[test]
    fn world_to_local_centre_is_full_weight() {
        let t = ShapeTransform::default();
        let (u, v, w) = t.world_to_local(50.0, 50.0, 100.0, 100.0).unwrap();
        assert!((u - 0.5).abs() < 1e-5);
        assert!((v - 0.5).abs() < 1e-5);
        assert!((w - 1.0).abs() < 1e-5);
    }

    #[test]
    fn world_to_local_respects_offset_and_blend() {
        let mut t = ShapeTransform {
            offset_x: 10.0,
            ..Default::default()
        };
        t.blend_size = 0.2;
        // Shape centre moved +10m in x; sampling there should land back at local centre.
        let (u, _v, w) = t.world_to_local(60.0, 50.0, 100.0, 100.0).unwrap();
        assert!((u - 0.5).abs() < 1e-5);
        assert!((w - 1.0).abs() < 1e-5);
        // Far outside the shape falls off to None once fully outside soft bounds.
        assert!(t.world_to_local(-1000.0, -1000.0, 100.0, 100.0).is_none());
    }

    #[test]
    fn stamp_and_overwrite() {
        let biome_a = LayerId::new();
        let biome_b = LayerId::new();
        let mut layer = BiomeLayer::new("Splat");
        layer.stamp(biome_a, 0.5, 0.5, 0.1, 1.0, false, 64);
        assert!(layer.weight_at(biome_a, 0.5, 0.5) > 0.5);
        layer.stamp(biome_b, 0.5, 0.5, 0.1, 1.0, false, 64);
        assert!(layer.weight_at(biome_b, 0.5, 0.5) > 0.5);
        // Overwrite should have suppressed A under the brush.
        assert!(layer.weight_at(biome_a, 0.5, 0.5) < 0.5);
    }

    #[test]
    fn normalize_weights_sum_to_one() {
        let a = LayerId::new();
        let b = LayerId::new();
        let mut layer = BiomeLayer::new("Splat");
        layer.overwrite = false;
        layer.stamp(a, 0.5, 0.5, 0.2, 1.0, false, 32);
        layer.stamp(b, 0.5, 0.5, 0.2, 1.0, false, 32);
        layer.normalize_all();
        let weights = layer.normalized_weights_at(0.5, 0.5);
        let sum: f32 = weights.iter().map(|(_, w)| *w).sum();
        assert!((sum - 1.0).abs() < 1e-3, "sum={sum}");
    }

    #[test]
    fn replace_clears_siblings() {
        let a = LayerId::new();
        let b = LayerId::new();
        let mut layer = BiomeLayer::new("Splat");
        layer.overwrite = false;
        layer.stamp(a, 0.5, 0.5, 0.15, 1.0, false, 64);
        layer.stamp_replace(b, 0.5, 0.5, 0.15, 1.0, 64);
        assert!(layer.weight_at(b, 0.5, 0.5) > 0.5);
        assert!(layer.weight_at(a, 0.5, 0.5) < 0.15);
    }

    #[test]
    fn smooth_reduces_local_contrast() {
        let a = LayerId::new();
        let mut layer = BiomeLayer::new("Splat");
        layer.stamp(a, 0.5, 0.5, 0.05, 1.0, false, 64);
        let peak_before = layer.weight_at(a, 0.5, 0.5);
        layer.smooth_at(0.5, 0.5, 0.12, 1.0);
        let peak_after = layer.weight_at(a, 0.5, 0.5);
        // Smoothing a sharp stamp should reduce the peak toward the neighbourhood mean.
        assert!(peak_after < peak_before);
    }

    #[test]
    fn bake_color_rgba_uses_definition_tint() {
        let a = LayerId::new();
        let mut layer = BiomeLayer::new("Splat");
        layer.stamp(a, 0.5, 0.5, 0.2, 1.0, false, 32);
        let rgba = layer.bake_color_rgba(
            32,
            32,
            &|id| {
                if id == a {
                    [1.0, 0.0, 0.0]
                } else {
                    [0.0, 1.0, 0.0]
                }
            },
            None,
        );
        let idx = ((16 * 32 + 16) * 4) as usize;
        assert!(rgba[idx] > 200, "expected red channel");
        assert!(rgba[idx + 3] > 0, "expected alpha");
    }
}
