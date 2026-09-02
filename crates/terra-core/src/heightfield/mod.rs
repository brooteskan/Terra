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

/// Upper bound on samples along one edge (`width`, `height`, `tile_size`).
/// Keeps `width * height` and halo-inflated tile strides inside `u32`/`usize`
/// and keeps every `as i32` index cast non-negative. The UI caps resolution far
/// lower (8192); this is the allocation-sanity ceiling, not a feature promise.
pub const MAX_DIM: u32 = 32_768;

/// Upper bound on halo (ghost) width per tile edge.
pub const MAX_HALO: u32 = 32;

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

/// Why a [`HeightfieldMetrics`] value violates its representation invariants.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum MetricsError {
    #[error("width and height must be 1..={max} samples, got {width}x{height}", max = MAX_DIM)]
    InvalidDimensions { width: u32, height: u32 },
    #[error("tile size must be 1..={max}, got {0}", max = MAX_DIM)]
    InvalidTileSize(u32),
    #[error("halo must be <= {max}, got {0}", max = MAX_HALO)]
    HaloTooLarge(u32),
    #[error("world extent must be finite and positive, got {x} x {z} metres")]
    InvalidWorldExtent { x: f32, z: f32 },
}

impl HeightfieldMetrics {
    pub fn new(width: u32, height: u32, world_size_x: f32, world_size_z: f32) -> Self {
        // No validation here: `HeightfieldMetrics` is a shared carrier and some
        // consumers (e.g. an empty `MaskField`) legitimately use 0x0 metrics as a
        // sentinel. The invariant that matters — a tiled `Heightfield` cannot be
        // built on invalid metrics — is asserted in `Heightfield::new`, and the
        // deserialization/export/import boundaries call `validate` explicitly.
        Self {
            width,
            height,
            world_size_x,
            world_size_z,
            tile_size: DEFAULT_TILE_SIZE,
            halo: DEFAULT_HALO,
        }
    }

    /// Validate every representation invariant. Cheap and allocation-free; call
    /// at each boundary where metrics arrive from deserialization or from a
    /// user-controlled resolution, before any tiling, sampling, or allocation.
    pub fn validate(&self) -> Result<(), MetricsError> {
        if self.width == 0 || self.width > MAX_DIM || self.height == 0 || self.height > MAX_DIM {
            return Err(MetricsError::InvalidDimensions {
                width: self.width,
                height: self.height,
            });
        }
        if self.tile_size == 0 || self.tile_size > MAX_DIM {
            return Err(MetricsError::InvalidTileSize(self.tile_size));
        }
        if self.halo > MAX_HALO {
            return Err(MetricsError::HaloTooLarge(self.halo));
        }
        if !self.world_size_x.is_finite()
            || self.world_size_x <= 0.0
            || !self.world_size_z.is_finite()
            || self.world_size_z <= 0.0
        {
            return Err(MetricsError::InvalidWorldExtent {
                x: self.world_size_x,
                z: self.world_size_z,
            });
        }
        Ok(())
    }

    /// Fallible sibling of [`Self::new`]: the validated metrics, or the first
    /// invariant they violate.
    pub fn try_new(
        width: u32,
        height: u32,
        world_size_x: f32,
        world_size_z: f32,
    ) -> Result<Self, MetricsError> {
        let metrics = Self {
            width,
            height,
            world_size_x,
            world_size_z,
            tile_size: DEFAULT_TILE_SIZE,
            halo: DEFAULT_HALO,
        };
        metrics.validate()?;
        Ok(metrics)
    }

    /// The same world span and halo resampled to a square `resolution`, with the
    /// interior tile size capped to that resolution. This is the single
    /// derivation shared by export, the background eval worker, and interactive
    /// preview; the result is validated before it is returned.
    pub fn at_resolution(&self, resolution: u32) -> Result<Self, MetricsError> {
        let derived = Self {
            width: resolution,
            height: resolution,
            world_size_x: self.world_size_x,
            world_size_z: self.world_size_z,
            tile_size: self.tile_size.min(resolution),
            halo: self.halo,
        };
        derived.validate()?;
        Ok(derived)
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
        debug_assert!(
            metrics.validate().is_ok(),
            "Heightfield::new built from invalid metrics: {metrics:?}"
        );
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

    /// Resident sample bytes across all tiles (interior + halo `f32`s). Excludes
    /// the fixed per-tile/-field struct overhead; used to size in-memory caches.
    pub fn resident_bytes(&self) -> usize {
        self.tiles
            .iter()
            .map(|t| std::mem::size_of_val(t.data()))
            .sum()
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
            tile.map_interior(&f);
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

    fn valid() -> HeightfieldMetrics {
        HeightfieldMetrics {
            width: 16,
            height: 16,
            world_size_x: 16.0,
            world_size_z: 16.0,
            tile_size: 16,
            halo: 2,
        }
    }

    #[test]
    fn representative_valid_metrics_pass_validation() {
        assert!(HeightfieldMetrics::preview_default().validate().is_ok());
        assert!(HeightfieldMetrics::new(64, 64, 128.0, 128.0)
            .validate()
            .is_ok());
        // The historical struct-literal shape used across the crate and fixtures.
        let m = HeightfieldMetrics {
            width: 300,
            height: 300,
            world_size_x: 1000.0,
            world_size_z: 1000.0,
            tile_size: 256,
            halo: 2,
        };
        assert!(m.validate().is_ok());
        // halo == 0 is a supported (render preview) configuration.
        assert!(HeightfieldMetrics { halo: 0, ..valid() }.validate().is_ok());
    }

    #[test]
    fn zero_dimensions_are_rejected() {
        assert_eq!(
            HeightfieldMetrics {
                width: 0,
                ..valid()
            }
            .validate(),
            Err(MetricsError::InvalidDimensions {
                width: 0,
                height: 16
            })
        );
        assert_eq!(
            HeightfieldMetrics {
                height: 0,
                ..valid()
            }
            .validate(),
            Err(MetricsError::InvalidDimensions {
                width: 16,
                height: 0
            })
        );
    }

    #[test]
    fn zero_tile_size_is_rejected_before_div_ceil_would_panic() {
        assert_eq!(
            HeightfieldMetrics {
                tile_size: 0,
                ..valid()
            }
            .validate(),
            Err(MetricsError::InvalidTileSize(0))
        );
    }

    #[test]
    fn oversized_dimensions_tile_and_halo_are_rejected() {
        let over = MAX_DIM + 1;
        assert_eq!(
            HeightfieldMetrics {
                width: over,
                ..valid()
            }
            .validate(),
            Err(MetricsError::InvalidDimensions {
                width: over,
                height: 16
            })
        );
        assert_eq!(
            HeightfieldMetrics {
                tile_size: over,
                ..valid()
            }
            .validate(),
            Err(MetricsError::InvalidTileSize(over))
        );
        assert_eq!(
            HeightfieldMetrics {
                halo: MAX_HALO + 1,
                ..valid()
            }
            .validate(),
            Err(MetricsError::HaloTooLarge(MAX_HALO + 1))
        );
    }

    #[test]
    fn non_finite_and_non_positive_world_extents_are_rejected() {
        // NaN breaks `assert_eq!` (NaN != NaN), so match the variant instead.
        for (x, z) in [
            (0.0, 16.0),
            (-1.0, 16.0),
            (16.0, 0.0),
            (16.0, -1.0),
            (f32::NAN, 16.0),
            (f32::INFINITY, 16.0),
            (16.0, f32::NEG_INFINITY),
        ] {
            let m = HeightfieldMetrics {
                world_size_x: x,
                world_size_z: z,
                ..valid()
            };
            assert!(
                matches!(m.validate(), Err(MetricsError::InvalidWorldExtent { .. })),
                "expected rejection for world extents {x} x {z}"
            );
        }
    }

    #[test]
    fn try_new_agrees_with_new_and_reports_the_first_violation() {
        assert_eq!(
            HeightfieldMetrics::try_new(64, 64, 128.0, 128.0),
            Ok(HeightfieldMetrics::new(64, 64, 128.0, 128.0))
        );
        assert_eq!(
            HeightfieldMetrics::try_new(0, 64, 128.0, 128.0),
            Err(MetricsError::InvalidDimensions {
                width: 0,
                height: 64
            })
        );
    }

    #[test]
    fn at_resolution_caps_tile_size_and_carries_world_span() {
        let base = HeightfieldMetrics {
            width: 1024,
            height: 1024,
            world_size_x: 4096.0,
            world_size_z: 2048.0,
            tile_size: 256,
            halo: 2,
        };
        let down = base.at_resolution(128).expect("128 is valid");
        assert_eq!((down.width, down.height), (128, 128));
        assert_eq!(down.tile_size, 128, "tile size caps to the resolution");
        assert_eq!((down.world_size_x, down.world_size_z), (4096.0, 2048.0));
        assert_eq!(down.halo, 2);
        // Above the base tile size, tile size is unchanged.
        assert_eq!(
            base.at_resolution(512).expect("512 is valid").tile_size,
            256
        );
        // A zero resolution is rejected, not silently produced.
        assert_eq!(
            base.at_resolution(0),
            Err(MetricsError::InvalidDimensions {
                width: 0,
                height: 0
            })
        );
    }
}
