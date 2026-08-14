//! Tile scheduling and ghost exchange for large fields.

mod region;

pub use region::{bounds_from_tiles, rects_from_tiles, SampleRect};

use crate::heightfield::{HeightTile, Heightfield, HeightfieldMetrics, TileId};
use crate::layer::LayerKind;
use std::collections::HashSet;

/// How a process dirty region should expand for incremental recomputation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirtyClass {
    /// Local stencil (blur / single-pass thermal) — pad by stencil radius only.
    Local,
    /// Multi-iteration neighbourhood ops — expand by tile radius from iters.
    Expanding,
    /// Drainage / SPE / amplify — basin-coupled; prefer full field or large expand.
    BasinDependent,
}

/// Classify a layer kind for dirty-region / cache expansion policy.
pub fn dirty_class_for(kind: &LayerKind) -> DirtyClass {
    match kind {
        LayerKind::Blur(_)
        | LayerKind::Coastal(_)
        | LayerKind::EffectFilter(_)
        | LayerKind::Path(_)
        | LayerKind::PolygonHeight(_)
        | LayerKind::Terrace(_)
        | LayerKind::Plateau(_) => DirtyClass::Local,
        LayerKind::ThermalErosion(_)
        | LayerKind::HydraulicErosion(_)
        | LayerKind::DebrisFlow(_)
        | LayerKind::SandSimulation(_)
        | LayerKind::FluidSimulation(_) => DirtyClass::Expanding,
        LayerKind::StreamPowerErosion(_)
        | LayerKind::MultiScaleAmplify(_)
        | LayerKind::RiverCarve(_)
        | LayerKind::RiverNetwork(_) => DirtyClass::BasinDependent,
        _ => DirtyClass::Local,
    }
}

/// Recommended halo (ghost) width for a neighbourhood process.
///
/// `stencil_radius` is the single-pass read radius (thermal/hydraulic ≈ 1).
/// An implementation that skips per-pass halo refresh must both allocate and
/// evolve an expanded domain of at least `stencil_radius * iters_per_batch`.
/// Merely widening storage halos is not sufficient when only tile interiors are
/// evaluated. [`map_tiles_batched`] therefore refreshes halos after every pass.
pub fn recommended_halo(stencil_radius: u32, iters_per_batch: u32) -> u32 {
    let need = stencil_radius.saturating_mul(iters_per_batch.max(1));
    need.max(crate::heightfield::DEFAULT_HALO).min(16)
}

/// Tile Chebyshev expand radius for a dirty class (not sample halo).
pub fn expand_radius_for(class: DirtyClass, stencil: u32, iterations: u32) -> u32 {
    match class {
        DirtyClass::Local => stencil.max(1).saturating_sub(1).max(1),
        DirtyClass::Expanding => {
            // Grow ~1 tile per ~8 iters (halo refresh between batches assumed).
            let batches = (iterations.max(1) + 7) / 8;
            batches.max(1).min(4)
        }
        DirtyClass::BasinDependent => {
            // Conservative: expand several rings; callers may still mark_all.
            ((iterations.max(1) + 3) / 4).max(2).min(8)
        }
    }
}

impl DirtyClass {
    /// Sample-space support radius for cache keys and invalidation expansion.
    ///
    /// Basin-coupled processes are treated as global — callers should prefer
    /// `mark_all` when this returns `None`.
    pub fn support_radius_samples(
        self,
        tile_size_samples: u32,
        stencil: u32,
        iterations: u32,
    ) -> Option<u32> {
        match self {
            Self::BasinDependent => None,
            Self::Local | Self::Expanding => Some(
                expand_radius_for(self, stencil, iterations)
                    .saturating_mul(tile_size_samples.max(1)),
            ),
        }
    }
}

/// Process tiles with neighbor halo refresh between passes.
pub struct TileScheduler {
    pub dirty: Vec<TileId>,
}

impl TileScheduler {
    pub fn new() -> Self {
        Self { dirty: Vec::new() }
    }

    pub fn clear(&mut self) {
        self.dirty.clear();
    }

    pub fn mark_all(&mut self, hf: &Heightfield) {
        self.dirty = hf.tiles().iter().map(|t| t.id).collect();
    }

    pub fn mark_tile(&mut self, id: TileId) {
        if !self.dirty.contains(&id) {
            self.dirty.push(id);
        }
    }

    pub fn mark_tile_and_neighbors(&mut self, hf: &Heightfield, id: TileId) {
        let tx_max = hf.metrics.tiles_x();
        let tz_max = hf.metrics.tiles_z();
        for dj in -1i32..=1 {
            for di in -1i32..=1 {
                let tx = id.tx as i32 + di;
                let tz = id.tz as i32 + dj;
                if tx >= 0 && tz >= 0 && tx < tx_max as i32 && tz < tz_max as i32 {
                    self.mark_tile(TileId {
                        tx: tx as u32,
                        tz: tz as u32,
                    });
                }
            }
        }
    }

    /// Grow the dirty set by Chebyshev radius (for multi-iteration stencils).
    pub fn expand(&mut self, hf: &Heightfield, radius: u32) {
        if radius == 0 || self.dirty.is_empty() {
            return;
        }
        let tx_max = hf.metrics.tiles_x() as i32;
        let tz_max = hf.metrics.tiles_z() as i32;
        let r = radius as i32;
        let mut set: HashSet<TileId> = self.dirty.iter().copied().collect();
        let seed: Vec<TileId> = self.dirty.clone();
        for id in seed {
            for dj in -r..=r {
                for di in -r..=r {
                    let tx = id.tx as i32 + di;
                    let tz = id.tz as i32 + dj;
                    if tx >= 0 && tz >= 0 && tx < tx_max && tz < tz_max {
                        set.insert(TileId {
                            tx: tx as u32,
                            tz: tz as u32,
                        });
                    }
                }
            }
        }
        self.dirty = set.into_iter().collect();
    }

    /// Expand using [`dirty_class_for`] policy; basin-dependent may mark all tiles.
    pub fn expand_for_process(
        &mut self,
        hf: &Heightfield,
        class: DirtyClass,
        stencil: u32,
        iterations: u32,
    ) {
        match class {
            DirtyClass::BasinDependent
                if iterations > 4 || hf.metrics.tiles_x() * hf.metrics.tiles_z() <= 16 =>
            {
                self.mark_all(hf);
            }
            _ => {
                let r = expand_radius_for(class, stencil, iterations);
                self.expand(hf, r);
            }
        }
    }

    /// Refresh ghosts affected by dirty tile interiors and return the remaining
    /// max seam error on edges touching those source tiles.
    ///
    /// A changed interior can feed the halos of any tile in its 8-neighbor
    /// ring, including diagonal corner ghosts. The scheduler expands only the
    /// temporary refresh destinations; `self.dirty` continues to describe the
    /// actual changed interiors.
    pub fn sync_dirty(&self, hf: &mut Heightfield) -> f32 {
        let destinations = halo_destinations_for(hf, &self.dirty);
        hf.refresh_halos_for(&destinations);
        measure_seams_among(hf, &self.dirty)
    }

    /// Ensure halos match neighbors; returns max absolute discontinuity across shared edges.
    pub fn sync_and_measure_seams(&self, hf: &mut Heightfield) -> f32 {
        hf.refresh_halos();
        measure_seams(hf)
    }

    pub fn dirty_bounds(
        &self,
        metrics: &crate::heightfield::HeightfieldMetrics,
        pad: u32,
    ) -> Option<SampleRect> {
        bounds_from_tiles(metrics, &self.dirty, pad)
    }
}

impl Default for TileScheduler {
    fn default() -> Self {
        Self::new()
    }
}

fn halo_destinations_for(hf: &Heightfield, sources: &[TileId]) -> Vec<TileId> {
    let tx_max = hf.metrics.tiles_x() as i32;
    let tz_max = hf.metrics.tiles_z() as i32;
    let mut seen = HashSet::new();
    let mut destinations = Vec::new();

    for &source in sources {
        if source.tx >= hf.metrics.tiles_x() || source.tz >= hf.metrics.tiles_z() {
            continue;
        }
        for dj in -1i32..=1 {
            for di in -1i32..=1 {
                let tx = source.tx as i32 + di;
                let tz = source.tz as i32 + dj;
                if tx < 0 || tz < 0 || tx >= tx_max || tz >= tz_max {
                    continue;
                }
                let id = TileId {
                    tx: tx as u32,
                    tz: tz as u32,
                };
                if seen.insert(id) {
                    destinations.push(id);
                }
            }
        }
    }

    destinations
}

/// Suggest metrics halo capacity for multi-iteration neighbourhood work.
///
/// This does not rebuild tiles or change synchronization cadence. Callers that
/// skip per-pass refresh must also evolve the expanded halo domain.
pub fn suggest_halo(metrics: &HeightfieldMetrics, stencil: u32, iters_per_batch: u32) -> u32 {
    recommended_halo(stencil, iters_per_batch).max(metrics.halo)
}

/// Max |halo − source interior| in both directions and at every configured
/// halo depth across shared tile edges (0 after `refresh_halos`).
pub fn measure_seams(hf: &Heightfield) -> f32 {
    let mut max_err = 0.0f32;
    for tz in 0..hf.metrics.tiles_z() {
        for tx in 0..hf.metrics.tiles_x().saturating_sub(1) {
            max_err = max_err.max(edge_seam_x(hf, tx, tz));
        }
    }
    for tz in 0..hf.metrics.tiles_z().saturating_sub(1) {
        for tx in 0..hf.metrics.tiles_x() {
            max_err = max_err.max(edge_seam_z(hf, tx, tz));
        }
    }
    max_err
}

/// Seam metric restricted to edges touching any tile in `tiles`.
pub fn measure_seams_among(hf: &Heightfield, tiles: &[TileId]) -> f32 {
    let tx_max = hf.metrics.tiles_x();
    let tz_max = hf.metrics.tiles_z();
    let mut x_edges = HashSet::new();
    let mut z_edges = HashSet::new();

    for &id in tiles {
        if id.tx >= tx_max || id.tz >= tz_max {
            continue;
        }
        if id.tx > 0 {
            x_edges.insert((id.tx - 1, id.tz));
        }
        if id.tx + 1 < tx_max {
            x_edges.insert((id.tx, id.tz));
        }
        if id.tz > 0 {
            z_edges.insert((id.tx, id.tz - 1));
        }
        if id.tz + 1 < tz_max {
            z_edges.insert((id.tx, id.tz));
        }
    }

    let mut max_err = 0.0f32;
    for (tx, tz) in x_edges {
        max_err = max_err.max(edge_seam_x(hf, tx, tz));
    }
    for (tx, tz) in z_edges {
        max_err = max_err.max(edge_seam_z(hf, tx, tz));
    }
    max_err
}

fn edge_seam_x(hf: &Heightfield, tx: u32, tz: u32) -> f32 {
    let left = hf.tile(TileId { tx, tz }).unwrap();
    let right = hf.tile(TileId { tx: tx + 1, tz }).unwrap();
    let boundary_x = ((tx + 1) * hf.metrics.tile_size) as i32;
    let origin_z = (tz * hf.metrics.tile_size) as i32;
    let halo = hf.metrics.halo as i32;
    let mut max_err = 0.0f32;
    for lz in 0..left.interior_height.min(right.interior_height) {
        let global_z = origin_z + lz as i32;
        for depth in 1..=halo {
            let left_source = hf.get_clamped(boundary_x - depth, global_z);
            let right_left_halo = right.get_with_halo(-depth, lz as i32);
            max_err = max_err.max((left_source - right_left_halo).abs());

            let right_source = hf.get_clamped(boundary_x + depth - 1, global_z);
            let left_right_halo =
                left.get_with_halo(left.interior_width as i32 + depth - 1, lz as i32);
            max_err = max_err.max((right_source - left_right_halo).abs());
        }
    }
    max_err
}

fn edge_seam_z(hf: &Heightfield, tx: u32, tz: u32) -> f32 {
    let top = hf.tile(TileId { tx, tz }).unwrap();
    let bot = hf.tile(TileId { tx, tz: tz + 1 }).unwrap();
    let origin_x = (tx * hf.metrics.tile_size) as i32;
    let boundary_z = ((tz + 1) * hf.metrics.tile_size) as i32;
    let halo = hf.metrics.halo as i32;
    let mut max_err = 0.0f32;
    for lx in 0..top.interior_width.min(bot.interior_width) {
        let global_x = origin_x + lx as i32;
        for depth in 1..=halo {
            let top_source = hf.get_clamped(global_x, boundary_z - depth);
            let bottom_top_halo = bot.get_with_halo(lx as i32, -depth);
            max_err = max_err.max((top_source - bottom_top_halo).abs());

            let bottom_source = hf.get_clamped(global_x, boundary_z + depth - 1);
            let top_bottom_halo =
                top.get_with_halo(lx as i32, top.interior_height as i32 + depth - 1);
            max_err = max_err.max((bottom_source - top_bottom_halo).abs());
        }
    }
    max_err
}

/// Apply a tile-local stencil that reads halo, writing only interior.
pub fn map_tiles<F>(hf: &mut Heightfield, mut f: F)
where
    F: FnMut(&HeightTile, u32, u32, f32) -> f32,
{
    let snapshot = hf.clone();
    let mut dirty = TileScheduler::new();
    for tile in snapshot.tiles() {
        dirty.mark_tile(tile.id);
        for lz in 0..tile.interior_height {
            for lx in 0..tile.interior_width {
                let v = tile.get_interior(lx, lz);
                let nv = f(tile, lx, lz, v);
                if let Some(t) = hf.tile_mut(tile.id) {
                    t.set_interior(lx, lz, nv);
                }
            }
        }
    }
    dirty.expand(hf, 1);
    dirty.sync_dirty(hf);
}

/// Multi-pass tile stencil with halo refresh after every pass.
///
/// `iters_per_batch` groups passes for scheduling and accounting only; it does
/// not change numerical results. Because this helper evaluates tile interiors
/// rather than an expanded domain, widening halos cannot replace the per-pass
/// `sync_dirty` call.
pub fn map_tiles_batched<F>(
    hf: &mut Heightfield,
    total_iters: u32,
    iters_per_batch: u32,
    mut f: F,
) -> f32
where
    F: FnMut(&HeightTile, u32, u32, f32, u32) -> f32,
{
    let batch = iters_per_batch.max(1);
    let mut max_seam = 0.0f32;
    let mut dirty = TileScheduler::new();
    dirty.mark_all(hf);
    let mut iter = 0u32;
    while iter < total_iters {
        let end = (iter + batch).min(total_iters);
        for pass in iter..end {
            let snapshot = hf.clone();
            for tile in snapshot.tiles() {
                for lz in 0..tile.interior_height {
                    for lx in 0..tile.interior_width {
                        let v = tile.get_interior(lx, lz);
                        let nv = f(tile, lx, lz, v, pass);
                        if let Some(t) = hf.tile_mut(tile.id) {
                            t.set_interior(lx, lz, nv);
                        }
                    }
                }
            }
            max_seam = max_seam.max(dirty.sync_dirty(hf));
        }
        iter = end;
    }
    max_seam
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heightfield::HeightfieldMetrics;

    #[test]
    fn tiled_blur_matches_dense_path() {
        let metrics = HeightfieldMetrics {
            width: 64,
            height: 64,
            world_size_x: 64.0,
            world_size_z: 64.0,
            tile_size: 32,
            halo: 2,
        };
        let mut hf = Heightfield::zeros(metrics);
        for j in 0..64 {
            for i in 0..64 {
                hf.set(i, j, ((i * 13 + j * 7) % 50) as f32);
            }
        }
        hf.refresh_halos();
        let before = measure_seams(&hf);
        assert!(before < 1e-3);

        map_tiles(&mut hf, |tile, lx, lz, _| {
            let mut sum = 0.0;
            let mut c = 0.0;
            for dj in -1..=1 {
                for di in -1..=1 {
                    sum += tile.get_with_halo(lx as i32 + di, lz as i32 + dj);
                    c += 1.0;
                }
            }
            sum / c
        });
        let err = measure_seams(&hf);
        assert!(err < 1e-2, "seam err {err}");
    }

    #[test]
    fn incremental_halo_matches_full() {
        let metrics = HeightfieldMetrics {
            width: 96,
            height: 96,
            world_size_x: 96.0,
            world_size_z: 96.0,
            tile_size: 32,
            halo: 2,
        };
        let mut a = Heightfield::zeros(metrics);
        let mut b = Heightfield::zeros(metrics);
        for j in 0..96 {
            for i in 0..96 {
                let v = (i + j) as f32 * 0.25;
                a.set(i, j, v);
                b.set(i, j, v);
            }
        }
        a.refresh_halos();
        let mut sched = TileScheduler::new();
        sched.mark_tile_and_neighbors(&b, TileId { tx: 1, tz: 1 });
        sched.sync_dirty(&mut b);
        // Compare halo of center tile
        let ta = a.tile(TileId { tx: 1, tz: 1 }).unwrap();
        let tb = b.tile(TileId { tx: 1, tz: 1 }).unwrap();
        for gz in -2i32..34 {
            for gx in -2i32..34 {
                assert!(
                    (ta.get_with_halo(gx, gz) - tb.get_with_halo(gx, gz)).abs() < 1e-5,
                    "mismatch at {gx},{gz}"
                );
            }
        }
    }

    fn patterned_heightfield(metrics: HeightfieldMetrics) -> Heightfield {
        let mut hf = Heightfield::zeros(metrics);
        for j in 0..metrics.height {
            for i in 0..metrics.width {
                hf.set(i, j, (j * metrics.width + i) as f32);
            }
        }
        hf.refresh_halos();
        hf
    }

    fn assert_same_tile_storage(expected: &Heightfield, actual: &Heightfield) {
        assert_eq!(expected.tiles().len(), actual.tiles().len());
        for expected_tile in expected.tiles() {
            let actual_tile = actual.tile(expected_tile.id).unwrap();
            assert_eq!(
                expected_tile.data(),
                actual_tile.data(),
                "tile {:?} differs from a full halo refresh",
                expected_tile.id
            );
        }
    }

    #[test]
    fn dirty_source_refreshes_neighbor_halo_and_metrics_expose_stale_reverse_copy() {
        let metrics = HeightfieldMetrics {
            width: 64,
            height: 32,
            world_size_x: 64.0,
            world_size_z: 32.0,
            tile_size: 32,
            halo: 2,
        };
        let mut hf = patterned_heightfield(metrics);
        let right_id = TileId { tx: 1, tz: 0 };

        hf.set(32, 10, 9999.0);

        let left = hf.tile(TileId { tx: 0, tz: 0 }).unwrap();
        assert_eq!(left.get_with_halo(32, 10), 672.0);
        assert_eq!(measure_seams(&hf), 9327.0);
        assert_eq!(measure_seams_among(&hf, &[right_id]), 9327.0);

        let mut scheduler = TileScheduler::new();
        scheduler.mark_tile(right_id);
        assert_eq!(scheduler.sync_dirty(&mut hf), 0.0);
        assert_eq!(measure_seams(&hf), 0.0);
        assert_eq!(
            hf.tile(TileId { tx: 0, tz: 0 })
                .unwrap()
                .get_with_halo(32, 10),
            9999.0
        );
    }

    #[test]
    fn seam_metrics_cover_both_directions_and_every_halo_depth() {
        let metrics = HeightfieldMetrics {
            width: 64,
            height: 64,
            world_size_x: 64.0,
            world_size_z: 64.0,
            tile_size: 32,
            halo: 3,
        };

        for depth in 1..=metrics.halo {
            let cases = [
                (32 - depth, 10, TileId { tx: 0, tz: 0 }, "left"),
                (32 + depth - 1, 10, TileId { tx: 1, tz: 0 }, "right"),
                (10, 32 - depth, TileId { tx: 0, tz: 0 }, "top"),
                (10, 32 + depth - 1, TileId { tx: 0, tz: 1 }, "bottom"),
            ];

            for (i, j, dirty_id, direction) in cases {
                let mut hf = patterned_heightfield(metrics);
                hf.set(i, j, 20_000.0 + depth as f32);

                assert!(
                    measure_seams(&hf) > 0.0,
                    "full metric missed {direction} source at depth {depth}"
                );
                assert!(
                    measure_seams_among(&hf, &[dirty_id]) > 0.0,
                    "local metric missed {direction} source at depth {depth}"
                );

                let mut scheduler = TileScheduler::new();
                scheduler.mark_tile(dirty_id);
                assert_eq!(scheduler.sync_dirty(&mut hf), 0.0);
                assert_eq!(measure_seams(&hf), 0.0);
            }
        }
    }

    #[test]
    fn dirty_center_refresh_matches_full_for_edges_and_diagonal_corners() {
        let metrics = HeightfieldMetrics {
            width: 96,
            height: 96,
            world_size_x: 96.0,
            world_size_z: 96.0,
            tile_size: 32,
            halo: 3,
        };
        let mut actual = patterned_heightfield(metrics);
        for j in 32..64 {
            for i in 32..64 {
                actual.set(i, j, actual.get(i, j) + 100_000.0);
            }
        }
        let mut expected = actual.clone();
        expected.refresh_halos();

        let mut scheduler = TileScheduler::new();
        scheduler.mark_tile(TileId { tx: 1, tz: 1 });
        assert_eq!(scheduler.sync_dirty(&mut actual), 0.0);
        assert_same_tile_storage(&expected, &actual);
    }

    #[test]
    fn dirty_partial_last_tile_matches_full_and_keeps_outer_clamp() {
        let metrics = HeightfieldMetrics {
            width: 70,
            height: 58,
            world_size_x: 70.0,
            world_size_z: 58.0,
            tile_size: 32,
            halo: 3,
        };
        let mut actual = patterned_heightfield(metrics);
        for j in 32..58 {
            for i in 64..70 {
                actual.set(i, j, actual.get(i, j) + 200_000.0);
            }
        }
        let mut expected = actual.clone();
        expected.refresh_halos();

        let partial_id = TileId { tx: 2, tz: 1 };
        let mut scheduler = TileScheduler::new();
        scheduler.mark_tile(partial_id);
        assert_eq!(scheduler.sync_dirty(&mut actual), 0.0);
        assert_same_tile_storage(&expected, &actual);

        let partial = actual.tile(partial_id).unwrap();
        let last_interior = partial.get_interior(5, 25);
        for depth in 1..=metrics.halo as i32 {
            assert_eq!(partial.get_with_halo(5 + depth, 25 + depth), last_interior);
        }
    }

    #[test]
    fn expand_grows_chebyshev() {
        let metrics = HeightfieldMetrics {
            width: 128,
            height: 128,
            world_size_x: 128.0,
            world_size_z: 128.0,
            tile_size: 32,
            halo: 2,
        };
        let hf = Heightfield::zeros(metrics);
        let mut sched = TileScheduler::new();
        sched.mark_tile(TileId { tx: 2, tz: 2 });
        sched.expand(&hf, 1);
        assert_eq!(sched.dirty.len(), 9);
    }

    fn batched_blur_bits(
        width: u32,
        height: u32,
        tile_size: u32,
        iters_per_batch: u32,
    ) -> Vec<u32> {
        let metrics = HeightfieldMetrics {
            width,
            height,
            world_size_x: width as f32,
            world_size_z: height as f32,
            tile_size,
            halo: 2,
        };
        let mut hf = Heightfield::zeros(metrics);
        for j in 0..height {
            for i in 0..width {
                hf.set(i, j, ((i * 3 + j * 5) % 40) as f32);
            }
        }
        hf.refresh_halos();
        let seam = map_tiles_batched(&mut hf, 4, iters_per_batch, |tile, lx, lz, _, _| {
            let mut sum = 0.0;
            let mut c = 0.0;
            for dj in -1..=1 {
                for di in -1..=1 {
                    sum += tile.get_with_halo(lx as i32 + di, lz as i32 + dj);
                    c += 1.0;
                }
            }
            sum / c
        });
        assert!(seam < 1e-2, "batched seam {seam}");
        assert!(measure_seams(&hf) < 1e-2);
        hf.to_dense().into_iter().map(f32::to_bits).collect()
    }

    fn assert_same_interior_bits(
        expected: &[u32],
        actual: &[u32],
        width: u32,
        tile_size: u32,
        iters_per_batch: u32,
    ) {
        assert_eq!(expected.len(), actual.len());
        if let Some(index) = expected.iter().zip(actual).position(|(a, b)| a != b) {
            let i = index as u32 % width;
            let j = index as u32 / width;
            panic!(
                "interior differs at ({i}, {j}) for tile_size={tile_size}, \
                 iters_per_batch={iters_per_batch}: expected bits {:#010x}, got {:#010x}",
                expected[index], actual[index]
            );
        }
    }

    #[test]
    fn batched_blur_is_tile_and_batch_independent() {
        let reference = batched_blur_bits(64, 64, 64, 1);
        for tile_size in [16, 32, 64] {
            for iters_per_batch in [1, 2, 4] {
                let actual = batched_blur_bits(64, 64, tile_size, iters_per_batch);
                assert_same_interior_bits(&reference, &actual, 64, tile_size, iters_per_batch);
            }
        }
    }

    #[test]
    fn batched_blur_is_independent_with_partial_edge_tiles() {
        let reference = batched_blur_bits(70, 58, 70, 1);
        for tile_size in [16, 32, 70] {
            for iters_per_batch in [1, 2, 4] {
                let actual = batched_blur_bits(70, 58, tile_size, iters_per_batch);
                assert_same_interior_bits(&reference, &actual, 70, tile_size, iters_per_batch);
            }
        }
    }

    #[test]
    fn recommended_halo_scales_with_batch() {
        assert!(recommended_halo(1, 8) >= 8);
        assert_eq!(recommended_halo(1, 1), crate::heightfield::DEFAULT_HALO);
    }

    #[test]
    fn dirty_class_basin_for_amplify() {
        assert_eq!(
            dirty_class_for(&LayerKind::MultiScaleAmplify(Default::default())),
            DirtyClass::BasinDependent
        );
        assert_eq!(
            dirty_class_for(&LayerKind::ThermalErosion(Default::default())),
            DirtyClass::Expanding
        );
    }

    #[test]
    fn basin_dependent_support_radius_is_global() {
        assert_eq!(
            DirtyClass::BasinDependent.support_radius_samples(256, 1, 1),
            None
        );
        assert!(DirtyClass::Local
            .support_radius_samples(256, 2, 1)
            .is_some());
    }

    /// Smoke: dirty tile snapshot can feed a viewport overlay (tx/tz list + bounds).
    #[test]
    fn dirty_tiles_snapshot_for_overlay() {
        let metrics = HeightfieldMetrics {
            width: 64,
            height: 64,
            world_size_x: 64.0,
            world_size_z: 64.0,
            tile_size: 16,
            halo: 2,
        };
        let mut sched = TileScheduler::new();
        sched.mark_tile(TileId { tx: 1, tz: 2 });
        sched.mark_tile(TileId { tx: 2, tz: 2 });
        let ids: Vec<(u32, u32)> = sched.dirty.iter().map(|t| (t.tx, t.tz)).collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&(1, 2)));
        assert!(ids.contains(&(2, 2)));
        let bounds = sched.dirty_bounds(&metrics, 1).expect("dirty bounds");
        assert!(!bounds.is_empty());
        // Overlay grid dimensions match metrics tiling.
        assert_eq!(metrics.tiles_x(), 4);
        assert_eq!(metrics.tiles_z(), 4);
    }
}
