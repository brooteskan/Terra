//! Camera-centered clipmap LOD rings + full-world fallback grid.
//!
//! Mesh vertex count is fixed per ring; the terrain vertex shader samples the
//! full-resolution height texture regardless of grid spacing.

/// Whether camera-centred geometry is constrained to a finite heightfield.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClipmapTraversalBounds {
    Bounded { world_x: f32, world_z: f32 },
    Infinite,
}

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
        camera_x: f64,
        camera_z: f64,
        bounds: ClipmapTraversalBounds,
    ) -> (f64, f64) {
        let half = f64::from(self.coverage()) * 0.5;
        let snap = f64::from(self.spacing.max(1e-6));
        let mut ox = ((camera_x - half) / snap).floor() * snap;
        let mut oz = ((camera_z - half) / snap).floor() * snap;
        if let ClipmapTraversalBounds::Bounded { world_x, world_z } = bounds {
            let max_x = f64::from((world_x - self.coverage()).max(0.0));
            let max_z = f64::from((world_z - self.coverage()).max(0.0));
            ox = ox.clamp(0.0, max_x);
            oz = oz.clamp(0.0, max_z);
        }
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

    /// Build dyadic rings directly from an Infinite topology. The fallback grid
    /// uses the configured coarsest sample lattice and only spans the active
    /// camera horizon; it never represents a complete world.
    pub fn for_infinite(topology: terra_core::InfiniteTopologyConfig, horizon_m: f64) -> Self {
        let finest = topology.finest_spacing_m.max(1.0e-6) as f32;
        let horizon = horizon_m.max(f64::from(finest));
        let ring_grid_size = WorldGridConfig::for_world(129).grid_size;
        let mut rings = Vec::new();
        for lod in 0..=topology.max_lod.get() {
            let spacing = finest * 2.0f32.powi(i32::from(lod));
            rings.push(ClipmapRingLevel {
                grid_size: ring_grid_size,
                spacing,
            });
            if f64::from(spacing) * f64::from(ring_grid_size - 1) * 0.5 >= horizon {
                break;
            }
        }
        let coarsest_spacing = finest * 2.0f32.powi(i32::from(topology.max_lod.get()));
        let required_cells = ((horizon * 2.0) / f64::from(coarsest_spacing))
            .ceil()
            .clamp(1.0, f64::from(u32::MAX - 1)) as u32;
        let max_grid = crate::grid::TerrainGrid::max_resolution_for_device_limits();
        let fallback = WorldGridConfig::for_world(required_cells.saturating_add(1).min(max_grid));
        Self {
            rings,
            fallback,
            skirt_depth: 0.0,
        }
    }

    /// Recompute snapped origins for every ring from the camera-centre XZ.
    pub fn ring_origins(
        &self,
        camera_x: f64,
        camera_z: f64,
        bounds: ClipmapTraversalBounds,
    ) -> Vec<(f64, f64)> {
        self.rings
            .iter()
            .map(|ring| ring.snap_origin(camera_x, camera_z, bounds))
            .collect()
    }
}

/// One clipmap draw call (coarse → fine order in [`ClipmapPresentPlan::rings`]).
#[derive(Debug, Clone, Copy)]
pub struct ClipmapRingDraw {
    pub ring_index: usize,
    pub origin_x: f64,
    pub origin_z: f64,
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
    pub fallback_origin_x: f64,
    pub fallback_origin_z: f64,
    pub fallback_exclude_half_extent: f32,
    /// Coarse → fine rings (fine wins depth; coarse holes prevent overdraw).
    pub rings: Vec<ClipmapRingDraw>,
}

#[derive(Debug, Clone, Copy)]
pub struct ClipmapPresentInput {
    pub camera_x: f64,
    pub camera_z: f64,
    pub world_x: f32,
    pub world_z: f32,
    pub height_tex_w: u32,
    pub height_tex_h: u32,
    pub traversal_bounds: ClipmapTraversalBounds,
}

impl ClipmapPresentPlan {
    pub fn build(clipmap: &ClipmapConfig, input: ClipmapPresentInput) -> Self {
        let extent = input.world_x.max(input.world_z).max(1.0);
        let tex = input.height_tex_w.max(input.height_tex_h).max(9);
        let tex_spacing = extent / tex.saturating_sub(1).max(1) as f32;
        let fallback_spacing = clipmap.fallback.spacing_for_extent(extent);
        let max_dense = crate::grid::TerrainGrid::max_resolution_for_device_limits();
        let dense_cells = max_dense.saturating_sub(1).max(1) as f32;
        let single_spacing = extent / dense_cells;

        // Prefer a single dense grid when it is at least as fine as the heightfield,
        // or when the world is small enough that rings buy nothing.
        let use_single_grid = input.traversal_bounds != ClipmapTraversalBounds::Infinite
            && (single_spacing <= tex_spacing * 1.25
                || clipmap.rings.is_empty()
                || extent <= dense_cells * tex_spacing * 1.1);

        if use_single_grid {
            return Self {
                use_single_grid: true,
                draw_fallback: false,
                fallback_spacing,
                fallback_grid_size: clipmap.fallback.grid_size,
                fallback_origin_x: 0.0,
                fallback_origin_z: 0.0,
                fallback_exclude_half_extent: 0.0,
                rings: Vec::new(),
            };
        }

        let origins = clipmap.ring_origins(input.camera_x, input.camera_z, input.traversal_bounds);
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
        let fallback_extent = f64::from(fallback_spacing)
            * f64::from(clipmap.fallback.grid_size.saturating_sub(1).max(1));
        let fallback_snap = f64::from(fallback_spacing.max(1.0e-6));
        let (fallback_origin_x, fallback_origin_z) = if input.traversal_bounds
            == ClipmapTraversalBounds::Infinite
        {
            (
                ((input.camera_x - fallback_extent * 0.5) / fallback_snap).floor() * fallback_snap,
                ((input.camera_z - fallback_extent * 0.5) / fallback_snap).floor() * fallback_snap,
            )
        } else {
            (0.0, 0.0)
        };

        Self {
            use_single_grid: false,
            draw_fallback: true,
            fallback_spacing,
            fallback_grid_size: clipmap.fallback.grid_size,
            fallback_origin_x,
            fallback_origin_z,
            fallback_exclude_half_extent: outer_half,
            rings: draws,
        }
    }
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
        let plan = ClipmapPresentPlan::build(
            &cfg,
            ClipmapPresentInput {
                camera_x: 256.0,
                camera_z: 256.0,
                world_x: 512.0,
                world_z: 512.0,
                height_tex_w: 513,
                height_tex_h: 513,
                traversal_bounds: ClipmapTraversalBounds::Bounded {
                    world_x: 512.0,
                    world_z: 512.0,
                },
            },
        );
        assert!(plan.use_single_grid);
        assert!(plan.rings.is_empty());
    }

    #[test]
    fn large_world_plans_coarse_to_fine_rings() {
        // Exceed device-dense single-grid coverage so rings are required.
        let cfg = ClipmapConfig::for_world_with_height(65536.0, 513, 8193);
        let plan = ClipmapPresentPlan::build(
            &cfg,
            ClipmapPresentInput {
                camera_x: 32000.0,
                camera_z: 32000.0,
                world_x: 65536.0,
                world_z: 65536.0,
                height_tex_w: 8193,
                height_tex_h: 8193,
                traversal_bounds: ClipmapTraversalBounds::Bounded {
                    world_x: 65536.0,
                    world_z: 65536.0,
                },
            },
        );
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
    fn ring_origin_snaps_to_spacing_grid() {
        let ring = ClipmapRingLevel {
            grid_size: 65,
            spacing: 8.0,
        };
        let (ox, oz) = ring.snap_origin(
            100.3,
            200.7,
            ClipmapTraversalBounds::Bounded {
                world_x: 4096.0,
                world_z: 4096.0,
            },
        );
        assert!((ox % 8.0).abs() < 1e-4);
        assert!((oz % 8.0).abs() < 1e-4);
    }

    #[test]
    fn infinite_ring_origin_is_unclamped_across_zero() {
        let ring = ClipmapRingLevel {
            grid_size: 65,
            spacing: 8.0,
        };
        let (ox, oz) = ring.snap_origin(-3.0, 5.0, ClipmapTraversalBounds::Infinite);
        assert!(ox < 0.0);
        assert!(oz < 0.0);
        assert_eq!(ox.rem_euclid(8.0), 0.0);
        assert_eq!(oz.rem_euclid(8.0), 0.0);
    }

    #[test]
    fn infinite_ring_origins_stay_on_each_dyadic_lattice() {
        let cfg = ClipmapConfig {
            rings: vec![
                ClipmapRingLevel {
                    grid_size: 129,
                    spacing: 2.0,
                },
                ClipmapRingLevel {
                    grid_size: 129,
                    spacing: 4.0,
                },
                ClipmapRingLevel {
                    grid_size: 65,
                    spacing: 8.0,
                },
            ],
            fallback: WorldGridConfig::for_world(65),
            skirt_depth: 0.0,
        };
        let origins = cfg.ring_origins(-17.25, 9.5, ClipmapTraversalBounds::Infinite);
        for (ring, (x, z)) in cfg.rings.iter().zip(origins) {
            assert_eq!(x.rem_euclid(f64::from(ring.spacing)), 0.0);
            assert_eq!(z.rem_euclid(f64::from(ring.spacing)), 0.0);
        }
    }

    #[test]
    fn infinite_config_uses_topology_spacing_and_horizon_fallback() {
        let topology = terra_core::InfiniteTopologyConfig {
            origin: terra_core::WorldPosition::ORIGIN,
            tile_size: 256,
            finest_spacing_m: 1.0,
            max_lod: terra_core::Lod::try_new(12).unwrap(),
        };
        let cfg = ClipmapConfig::for_infinite(topology, 16_384.0);
        assert_eq!(cfg.rings.first().unwrap().spacing, 1.0);
        assert_eq!(cfg.rings.last().unwrap().spacing, 256.0);
        assert!(cfg.rings.last().unwrap().coverage() * 0.5 >= 16_384.0);
        assert_eq!(cfg.fallback.grid_size, 9);
        assert_eq!(cfg.fallback.spacing_for_extent(32_768.0), 4096.0);
    }
}
