//! Camera-centered clipmap LOD rings + full-world fallback grid.
//!
//! Mesh vertex count is fixed per ring; the terrain vertex shader samples the
//! full-resolution height texture regardless of grid spacing.

use terra_core::{FieldId, LayerId, NormalizedRect, TerrainPyramid, TerrainTileKey};

/// Single world-covering displacement grid (coarsest / full fallback).
#[derive(Debug, Clone)]
pub struct WorldGridConfig {
    pub grid_size: u32,
    /// Optional vertical skirt depth in meters (0 = off). Mesh emission optional.
    pub skirt_depth: f32,
}

impl Default for WorldGridConfig {
    fn default() -> Self {
        Self::for_world(385)
    }
}

impl WorldGridConfig {
    /// Normalize to `4n + 1` vertices (legacy quarter-alignment).
    pub fn for_world(grid_size: u32) -> Self {
        let cells = grid_size.max(9).saturating_sub(1);
        let grid_size = cells.div_ceil(4).saturating_mul(4).saturating_add(1);
        Self {
            grid_size,
            skirt_depth: 0.0,
        }
    }

    /// True when optional skirts should be emitted by the mesh builder.
    pub fn skirts_enabled(&self) -> bool {
        self.skirt_depth > 1e-4
    }

    /// World metres between adjacent vertices when covering `world_extent`.
    pub fn spacing_for_extent(&self, world_extent: f32) -> f32 {
        let cells = self.grid_size.saturating_sub(1).max(1) as f32;
        world_extent / cells
    }
}

/// One nested clipmap ring: fixed vertex grid, world-space vertex spacing.
#[derive(Debug, Clone, Copy)]
pub struct ClipmapRingLevel {
    pub grid_size: u32,
    /// World metres between adjacent grid vertices.
    pub spacing: f32,
}

impl ClipmapRingLevel {
    /// World extent covered by this ring (metres).
    pub fn coverage(&self) -> f32 {
        self.spacing * self.grid_size.saturating_sub(1).max(1) as f32
    }

    /// Snap ring origin so vertices stay aligned across levels (Losasso & Hoppe).
    pub fn snap_origin(
        &self,
        camera_x: f32,
        camera_z: f32,
        world_x: f32,
        world_z: f32,
    ) -> (f32, f32) {
        let half = self.coverage() * 0.5;
        let snap = self.spacing.max(1e-6);
        let mut ox = ((camera_x - half) / snap).floor() * snap;
        let mut oz = ((camera_z - half) / snap).floor() * snap;
        let max_x = (world_x - self.coverage()).max(0.0);
        let max_z = (world_z - self.coverage()).max(0.0);
        ox = ox.clamp(0.0, max_x);
        oz = oz.clamp(0.0, max_z);
        (ox, oz)
    }
}

/// Nested camera-centered rings plus a coarse full-world fallback grid.
#[derive(Debug, Clone)]
pub struct ClipmapConfig {
    /// Finest → coarsest rings drawn around the camera target.
    pub rings: Vec<ClipmapRingLevel>,
    /// Full-world coarse grid when rings do not reach the horizon.
    pub fallback: WorldGridConfig,
    pub skirt_depth: f32,
}

impl Default for ClipmapConfig {
    fn default() -> Self {
        Self::for_world(4096.0, 513)
    }
}

impl ClipmapConfig {
    /// Build 4 nested rings with doubling spacing; `fallback_vertices` is the coarsest grid.
    pub fn for_world(world_extent: f32, fallback_vertices: u32) -> Self {
        Self::for_world_with_height(world_extent, fallback_vertices, 1025)
    }

    /// Prefer an innermost ring whose spacing matches height-tex density.
    pub fn for_world_with_height(
        world_extent: f32,
        fallback_vertices: u32,
        height_tex_res: u32,
    ) -> Self {
        let extent = world_extent.max(1.0);
        let fallback = WorldGridConfig::for_world(fallback_vertices);
        let fallback_spacing = fallback.spacing_for_extent(extent);
        let tex = height_tex_res.max(9);
        let tex_spacing = extent / tex.saturating_sub(1).max(1) as f32;

        // Dense inner ring ≈ height sample spacing; outer rings double.
        let ring_grids = [129u32, 129, 97, 65];
        let mut rings = Vec::with_capacity(ring_grids.len());
        for (i, &requested) in ring_grids.iter().enumerate() {
            let grid_size = WorldGridConfig::for_world(requested).grid_size;
            let spacing = (tex_spacing * 2f32.powi(i as i32)).max(tex_spacing);
            // Never coarser than the full-world fallback spacing for outer rings.
            let spacing = if i + 1 == ring_grids.len() {
                spacing.max(fallback_spacing * 0.5)
            } else {
                spacing.min(fallback_spacing)
            };
            rings.push(ClipmapRingLevel { grid_size, spacing });
        }

        Self {
            rings,
            skirt_depth: fallback.skirt_depth.max(0.0),
            fallback,
        }
    }

    /// Recompute snapped origins for every ring from the camera-centre XZ.
    pub fn ring_origins(
        &self,
        camera_x: f32,
        camera_z: f32,
        world_x: f32,
        world_z: f32,
    ) -> Vec<(f32, f32)> {
        self.rings
            .iter()
            .map(|ring| ring.snap_origin(camera_x, camera_z, world_x, world_z))
            .collect()
    }
}

/// One clipmap draw call (coarse → fine order in [`ClipmapPresentPlan::rings`]).
#[derive(Debug, Clone, Copy)]
pub struct ClipmapRingDraw {
    pub ring_index: usize,
    pub origin_x: f32,
    pub origin_z: f32,
    pub spacing: f32,
    pub grid_size: u32,
    /// Discard fragments whose Chebyshev distance from ring centre is below this
    /// (half-extent of the next-finer coverage). Zero = no hole.
    pub exclude_half_extent: f32,
    /// Soft morph band outside the exclude hole (metres).
    pub morph_width: f32,
}

/// Planned RasterLit geometry for one frame.
#[derive(Debug, Clone)]
pub struct ClipmapPresentPlan {
    /// Small worlds: one dense full-world mesh (no LOD rings).
    pub use_single_grid: bool,
    /// Full-world fallback drawn first (only when rings are active).
    pub draw_fallback: bool,
    pub fallback_spacing: f32,
    pub fallback_grid_size: u32,
    pub fallback_exclude_half_extent: f32,
    /// Coarse → fine rings (fine wins depth; coarse holes prevent overdraw).
    pub rings: Vec<ClipmapRingDraw>,
}

impl ClipmapPresentPlan {
    pub fn build(
        clipmap: &ClipmapConfig,
        camera_x: f32,
        camera_z: f32,
        world_x: f32,
        world_z: f32,
        height_tex_w: u32,
        height_tex_h: u32,
    ) -> Self {
        let extent = world_x.max(world_z).max(1.0);
        let tex = height_tex_w.max(height_tex_h).max(9);
        let tex_spacing = extent / tex.saturating_sub(1).max(1) as f32;
        let fallback_spacing = clipmap.fallback.spacing_for_extent(extent);
        let max_dense = crate::grid::TerrainGrid::max_resolution_for_device_limits();
        let dense_cells = max_dense.saturating_sub(1).max(1) as f32;
        let single_spacing = extent / dense_cells;

        // Prefer a single dense grid when it is at least as fine as the heightfield,
        // or when the world is small enough that rings buy nothing.
        let use_single_grid = single_spacing <= tex_spacing * 1.25
            || clipmap.rings.is_empty()
            || extent <= dense_cells * tex_spacing * 1.1;

        if use_single_grid {
            return Self {
                use_single_grid: true,
                draw_fallback: false,
                fallback_spacing,
                fallback_grid_size: clipmap.fallback.grid_size,
                fallback_exclude_half_extent: 0.0,
                rings: Vec::new(),
            };
        }

        let origins = clipmap.ring_origins(camera_x, camera_z, world_x, world_z);
        // rings[] is fine → coarse; draw order is coarse → fine.
        let mut draws = Vec::with_capacity(clipmap.rings.len());
        for (rev_i, ring) in clipmap.rings.iter().enumerate().rev() {
            let (ox, oz) = origins[rev_i];
            let finer_half = if rev_i > 0 {
                clipmap.rings[rev_i - 1].coverage() * 0.5
            } else {
                0.0
            };
            let morph = (ring.spacing * 2.0).max(tex_spacing);
            draws.push(ClipmapRingDraw {
                ring_index: rev_i,
                origin_x: ox,
                origin_z: oz,
                spacing: ring.spacing,
                grid_size: ring.grid_size,
                exclude_half_extent: finer_half,
                morph_width: morph,
            });
        }

        let outer_half = clipmap
            .rings
            .last()
            .map(|r| r.coverage() * 0.5)
            .unwrap_or(0.0);

        Self {
            use_single_grid: false,
            draw_fallback: true,
            fallback_spacing,
            fallback_grid_size: clipmap.fallback.grid_size,
            fallback_exclude_half_extent: outer_half,
            rings: draws,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentTileSelection {
    pub requested: TerrainTileKey,
    pub resident: Option<TerrainTileKey>,
    pub fallback_levels: u8,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ViewportTilePlan {
    pub selections: Vec<ResidentTileSelection>,
    pub exact_tiles: usize,
    pub fallback_tiles: usize,
    pub missing_tiles: usize,
}

/// Resolve desired visible tiles to their finest resident pages, using ancestors as fallback.
pub fn plan_resident_tiles(
    pyramid: &TerrainPyramid,
    layer: Option<LayerId>,
    field: &FieldId,
    desired_level: u8,
    visible: NormalizedRect,
) -> ViewportTilePlan {
    let desired_level = desired_level.min(pyramid.max_level());
    let Some(level) = pyramid.levels.get(desired_level as usize) else {
        return ViewportTilePlan::default();
    };
    let mut plan = ViewportTilePlan::default();
    for tile in terra_core::RegionSet::from_rect(visible).tiles(&level.metrics) {
        let requested = TerrainTileKey {
            layer,
            field: field.clone(),
            level: desired_level,
            tile,
        };
        let resident = pyramid.best_resident_ancestor(layer, field, desired_level, requested.tile);
        let fallback_levels = resident
            .as_ref()
            .map_or(0, |key| desired_level.saturating_sub(key.level));
        match resident.as_ref() {
            Some(key) if key.level == desired_level => plan.exact_tiles += 1,
            Some(_) => plan.fallback_tiles += 1,
            None => plan.missing_tiles += 1,
        }
        plan.selections.push(ResidentTileSelection {
            requested,
            resident,
            fallback_levels,
        });
    }
    plan
}

/// Project world-space geometric error to screen pixels for LOD prioritization.
pub fn projected_error_px(
    geometric_error_m: f32,
    distance_m: f32,
    fov_y_rad: f32,
    viewport_height_px: u32,
) -> f32 {
    if !geometric_error_m.is_finite()
        || !distance_m.is_finite()
        || distance_m <= 1e-3
        || viewport_height_px == 0
    {
        return f32::INFINITY;
    }
    let half_fov = (fov_y_rad * 0.5).max(1e-4);
    let pixels_per_rad = viewport_height_px as f32 / (2.0 * half_fov.tan());
    (geometric_error_m / distance_m.max(1e-3)) * pixels_per_rad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_grid_uses_quarter_aligned_vertices() {
        for requested in [9, 96, 97, 98, 127, 385] {
            let cfg = WorldGridConfig::for_world(requested);
            assert_eq!((cfg.grid_size - 1) % 4, 0);
        }
    }

    #[test]
    fn default_grid_is_385() {
        assert_eq!(WorldGridConfig::default().grid_size, 385);
    }

    #[test]
    fn clipmap_builds_four_rings() {
        let cfg = ClipmapConfig::for_world(4096.0, 385);
        assert_eq!(cfg.rings.len(), 4);
        assert!(cfg.rings[0].spacing < cfg.rings[3].spacing);
        assert_eq!(cfg.fallback.grid_size, 385);
    }

    #[test]
    fn dense_inner_ring_tracks_height_tex() {
        let cfg = ClipmapConfig::for_world_with_height(8192.0, 513, 2049);
        let tex_spacing = 8192.0 / 2048.0;
        assert!((cfg.rings[0].spacing - tex_spacing).abs() < 1e-3);
        assert!(cfg.rings[0].spacing < cfg.rings[1].spacing);
    }

    #[test]
    fn small_world_uses_single_grid_plan() {
        let cfg = ClipmapConfig::for_world_with_height(512.0, 513, 513);
        let plan = ClipmapPresentPlan::build(&cfg, 256.0, 256.0, 512.0, 512.0, 513, 513);
        assert!(plan.use_single_grid);
        assert!(plan.rings.is_empty());
    }

    #[test]
    fn large_world_plans_coarse_to_fine_rings() {
        // Exceed device-dense single-grid coverage so rings are required.
        let cfg = ClipmapConfig::for_world_with_height(65536.0, 513, 8193);
        let plan = ClipmapPresentPlan::build(&cfg, 32000.0, 32000.0, 65536.0, 65536.0, 8193, 8193);
        assert!(!plan.use_single_grid);
        assert!(plan.draw_fallback);
        assert_eq!(plan.rings.len(), cfg.rings.len());
        // First draw is coarsest ring (highest index).
        assert_eq!(plan.rings[0].ring_index, cfg.rings.len() - 1);
        assert_eq!(plan.rings.last().unwrap().ring_index, 0);
        assert!(plan.rings.last().unwrap().exclude_half_extent <= 1e-4);
        assert!(plan.rings[0].exclude_half_extent > 0.0);
    }

    #[test]
    fn projected_error_scales_with_distance() {
        let err_near = projected_error_px(2.0, 50.0, 1.0, 1080);
        let err_far = projected_error_px(2.0, 500.0, 1.0, 1080);
        assert!(err_near > err_far);
    }

    #[test]
    fn ring_origin_snaps_to_spacing_grid() {
        let ring = ClipmapRingLevel {
            grid_size: 65,
            spacing: 8.0,
        };
        let (ox, oz) = ring.snap_origin(100.3, 200.7, 4096.0, 4096.0);
        assert!((ox % 8.0).abs() < 1e-4);
        assert!((oz % 8.0).abs() < 1e-4);
    }

    #[test]
    fn resident_plan_falls_back_without_leaving_holes() {
        let mut pyramid = TerrainPyramid::new(terra_core::PyramidConfig::new(1024, 4096.0, 4096.0));
        pyramid.publish_resident(
            TerrainTileKey {
                layer: None,
                field: FieldId::Height,
                level: 0,
                tile: terra_core::TileId { tx: 0, tz: 0 },
            },
            terra_core::TilePageHandle {
                slot: 0,
                generation: 1,
            },
            1,
            1,
            0,
        );
        let plan = plan_resident_tiles(
            &pyramid,
            None,
            &FieldId::Height,
            pyramid.max_level(),
            NormalizedRect::new(0.45, 0.45, 0.55, 0.55).unwrap(),
        );
        assert!(!plan.selections.is_empty());
        assert_eq!(plan.missing_tiles, 0);
        assert_eq!(plan.fallback_tiles, plan.selections.len());
        assert!(plan
            .selections
            .iter()
            .all(|selection| selection.fallback_levels > 0));
    }
}
