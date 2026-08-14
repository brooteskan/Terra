//! Tiled heightfield storage in world meters.

mod arena;
mod tile;
mod world_space;

pub use arena::FloatArena;
pub use tile::{HeightTile, TileId};
pub use world_space::{metres_to_texels, texels_to_metres, world_radius_texels};

use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Default interior tile size (samples along one edge).
pub const DEFAULT_TILE_SIZE: u32 = 256;
/// Default ghost/halo width in samples per edge.
pub const DEFAULT_HALO: u32 = 2;
/// Default interactive preview resolution (overridden by WC-style
/// `preview_resolution_for_world_size` when creating projects).
pub const DEFAULT_PREVIEW_RES: u32 = 1024;

/// World-space metrics for a regular grid DEM.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HeightfieldMetrics {
    /// Number of height samples in X.
    pub width: u32,
    /// Number of height samples in Z.
    pub height: u32,
    /// World extent in X (meters).
    pub world_size_x: f32,
    /// World extent in Z (meters).
    pub world_size_z: f32,
    /// Interior tile edge length in samples.
    pub tile_size: u32,
    /// Ghost cell width per edge.
    pub halo: u32,
}

impl HeightfieldMetrics {
    pub fn new(width: u32, height: u32, world_size_x: f32, world_size_z: f32) -> Self {
        Self {
            width,
            height,
            world_size_x,
            world_size_z,
            tile_size: DEFAULT_TILE_SIZE,
            halo: DEFAULT_HALO,
        }
    }

    pub fn preview_default() -> Self {
        Self::new(DEFAULT_PREVIEW_RES, DEFAULT_PREVIEW_RES, 4096.0, 4096.0)
    }

    #[inline]
    pub fn dx(&self) -> f32 {
        self.world_size_x / self.width as f32
    }

    #[inline]
    pub fn dz(&self) -> f32 {
        self.world_size_z / self.height as f32
    }

    /// Cell-center world X for column `i`.
    #[inline]
    pub fn world_x(&self, i: u32) -> f32 {
        (i as f32 + 0.5) * self.dx()
    }

    /// Cell-center world Z for row `j`.
    #[inline]
    pub fn world_z(&self, j: u32) -> f32 {
        (j as f32 + 0.5) * self.dz()
    }

    /// Sample index from world position (clamped).
    pub fn sample_index(&self, x: f32, z: f32) -> (u32, u32) {
        let i = ((x / self.dx()) - 0.5).round() as i32;
        let j = ((z / self.dz()) - 0.5).round() as i32;
        (
            i.clamp(0, self.width as i32 - 1) as u32,
            j.clamp(0, self.height as i32 - 1) as u32,
        )
    }

    pub fn tiles_x(&self) -> u32 {
        self.width.div_ceil(self.tile_size)
    }

    pub fn tiles_z(&self) -> u32 {
        self.height.div_ceil(self.tile_size)
    }

    pub fn tile_count(&self) -> u32 {
        self.tiles_x() * self.tiles_z()
    }
}

/// Tiled heightfield; heights are world meters (Y up).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Heightfield {
    pub metrics: HeightfieldMetrics,
    tiles: Vec<HeightTile>,
}

impl Heightfield {
    pub fn new(metrics: HeightfieldMetrics) -> Self {
        let n = metrics.tile_count() as usize;
        let mut tiles = Vec::with_capacity(n);
        for tz in 0..metrics.tiles_z() {
            for tx in 0..metrics.tiles_x() {
                tiles.push(HeightTile::new(TileId { tx, tz }, &metrics));
            }
        }
        Self { metrics, tiles }
    }

    pub fn filled(metrics: HeightfieldMetrics, value: f32) -> Self {
        let mut hf = Self::new(metrics);
        hf.fill(value);
        hf
    }

    pub fn zeros(metrics: HeightfieldMetrics) -> Self {
        Self::filled(metrics, 0.0)
    }

    pub fn fill(&mut self, value: f32) {
        for tile in &mut self.tiles {
            tile.fill(value);
        }
    }

    pub fn tile(&self, id: TileId) -> Option<&HeightTile> {
        let idx = self.tile_index(id)?;
        Some(&self.tiles[idx])
    }

    pub fn tile_mut(&mut self, id: TileId) -> Option<&mut HeightTile> {
        let idx = self.tile_index(id)?;
        Some(&mut self.tiles[idx])
    }

    pub fn tiles(&self) -> &[HeightTile] {
        &self.tiles
    }

    pub fn tiles_mut(&mut self) -> &mut [HeightTile] {
        &mut self.tiles
    }

    fn tile_index(&self, id: TileId) -> Option<usize> {
        if id.tx >= self.metrics.tiles_x() || id.tz >= self.metrics.tiles_z() {
            return None;
        }
        Some((id.tz * self.metrics.tiles_x() + id.tx) as usize)
    }

    /// Sample height at global sample indices.
    pub fn get(&self, i: u32, j: u32) -> f32 {
        let (tx, lx) = self.local_x(i);
        let (tz, lz) = self.local_z(j);
        self.tile(TileId { tx, tz })
            .map(|t| t.get_interior(lx, lz))
            .unwrap_or(0.0)
    }

    pub fn set(&mut self, i: u32, j: u32, value: f32) {
        let (tx, lx) = self.local_x(i);
        let (tz, lz) = self.local_z(j);
        if let Some(tile) = self.tile_mut(TileId { tx, tz }) {
            tile.set_interior(lx, lz, value);
        }
    }

    fn local_x(&self, i: u32) -> (u32, u32) {
        let ts = self.metrics.tile_size;
        let tx = (i / ts).min(self.metrics.tiles_x() - 1);
        let origin = tx * ts;
        (tx, i - origin)
    }

    fn local_z(&self, j: u32) -> (u32, u32) {
        let ts = self.metrics.tile_size;
        let tz = (j / ts).min(self.metrics.tiles_z() - 1);
        let origin = tz * ts;
        (tz, j - origin)
    }

    /// Flatten interior samples row-major for mesh upload / export.
    pub fn to_dense(&self) -> Vec<f32> {
        let w = self.metrics.width as usize;
        let h = self.metrics.height as usize;
        let mut out = vec![0.0f32; w * h];
        for j in 0..self.metrics.height {
            for i in 0..self.metrics.width {
                out[j as usize * w + i as usize] = self.get(i, j);
            }
        }
        out
    }

    /// Rebuild from dense row-major buffer (must match metrics size).
    pub fn from_dense(metrics: HeightfieldMetrics, data: &[f32]) -> Self {
        assert_eq!(data.len(), (metrics.width * metrics.height) as usize);
        let mut hf = Self::new(metrics);
        let w = metrics.width;
        for j in 0..metrics.height {
            for i in 0..metrics.width {
                hf.set(i, j, data[(j * w + i) as usize]);
            }
        }
        hf.refresh_halos();
        hf
    }

    /// Sample with clamp-to-edge (for halo fill outside the DEM).
    #[inline]
    pub fn get_clamped(&self, i: i32, j: i32) -> f32 {
        let ci = i.clamp(0, self.metrics.width as i32 - 1) as u32;
        let cj = j.clamp(0, self.metrics.height as i32 - 1) as u32;
        self.get(ci, cj)
    }

    /// Copy ghost cells from neighboring tile interiors (full field).
    pub fn refresh_halos(&mut self) {
        let ids: Vec<TileId> = self.tiles.iter().map(|t| t.id).collect();
        self.refresh_halos_for(&ids);
    }

    /// Incremental ghost exchange for a destination tile set (Wave D).
    ///
    /// Only listed destinations have their halo rings rewritten. Callers that
    /// start from changed interior sources must include every destination whose
    /// halo can read those sources. [`crate::tiling::TileScheduler::sync_dirty`]
    /// performs that source-to-destination expansion.
    pub fn refresh_halos_for(&mut self, tile_ids: &[TileId]) {
        if tile_ids.is_empty() {
            return;
        }
        let metrics = self.metrics;
        let halo = metrics.halo as i32;

        // Phase 1: gather samples with shared borrow (no full-field dense alloc).
        let mut plans: Vec<(TileId, u32, u32, u32, u32, Vec<f32>)> =
            Vec::with_capacity(tile_ids.len());
        for &id in tile_ids {
            let Some(tile) = self.tile(id) else {
                continue;
            };
            let (ox, oz) = tile.interior_origin(&metrics);
            let iw = tile.interior_width;
            let ih = tile.interior_height;
            let stride = (iw as i32 + 2 * halo) as u32;
            let stride_z = (ih as i32 + 2 * halo) as u32;
            let mut samples = Vec::with_capacity((stride * stride_z) as usize);
            for gz in -halo..ih as i32 + halo {
                for gx in -halo..iw as i32 + halo {
                    samples.push(self.get_clamped(ox as i32 + gx, oz as i32 + gz));
                }
            }
            plans.push((id, ox, oz, iw, ih, samples));
        }

        // Phase 2: write halos (and refresh interior copies in the halo buffer).
        for (id, _ox, _oz, iw, ih, samples) in plans {
            let Some(tile) = self.tile_mut(id) else {
                continue;
            };
            let mut idx = 0usize;
            for gz in -halo..ih as i32 + halo {
                for gx in -halo..iw as i32 + halo {
                    tile.set_with_halo(gx, gz, samples[idx]);
                    idx += 1;
                }
            }
        }
    }

    pub fn map_mut<F: Fn(f32) -> f32 + Sync>(&mut self, f: F) {
        use rayon::prelude::*;
        self.tiles.par_iter_mut().for_each(|tile| {
            tile.map_interior(|v| f(v));
        });
    }

    pub fn min_max(&self) -> (f32, f32) {
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        for tile in &self.tiles {
            for &v in tile.interior() {
                min_v = min_v.min(v);
                max_v = max_v.max(v);
            }
        }
        if !min_v.is_finite() {
            (0.0, 0.0)
        } else {
            (min_v, max_v)
        }
    }

    /// Shared empty-friendly clone via Arc when immutably shared by cache.
    pub fn into_shared(self) -> Arc<Self> {
        Arc::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_sample_round_trip() {
        let m = HeightfieldMetrics::new(64, 64, 128.0, 128.0);
        let (i, j) = (10u32, 20u32);
        let x = m.world_x(i);
        let z = m.world_z(j);
        let (ii, jj) = m.sample_index(x, z);
        assert_eq!((ii, jj), (i, j));
    }

    #[test]
    fn halo_width_invariant() {
        let m = HeightfieldMetrics::new(128, 128, 256.0, 256.0);
        assert_eq!(m.halo, DEFAULT_HALO);
        let hf = Heightfield::zeros(m);
        for tile in hf.tiles() {
            assert_eq!(tile.halo, DEFAULT_HALO);
            let stride = tile.stride();
            assert_eq!(stride, tile.interior_width + 2 * tile.halo);
        }
    }

    #[test]
    fn dense_round_trip() {
        let m = HeightfieldMetrics::new(32, 24, 64.0, 48.0);
        let mut hf = Heightfield::zeros(m);
        hf.set(3, 5, 12.5);
        hf.set(31, 23, -2.0);
        let dense = hf.to_dense();
        let hf2 = Heightfield::from_dense(m, &dense);
        assert_eq!(hf2.get(3, 5), 12.5);
        assert_eq!(hf2.get(31, 23), -2.0);
    }

    #[test]
    fn tile_count_covers_field() {
        let m = HeightfieldMetrics {
            width: 300,
            height: 300,
            world_size_x: 1000.0,
            world_size_z: 1000.0,
            tile_size: 256,
            halo: 2,
        };
        assert_eq!(m.tiles_x(), 2);
        assert_eq!(m.tiles_z(), 2);
        let hf = Heightfield::zeros(m);
        assert_eq!(hf.tiles().len(), 4);
        // Edge sample on last partial tile
        let mut hf = hf;
        hf.set(299, 299, 7.0);
        assert_eq!(hf.get(299, 299), 7.0);
    }
}
