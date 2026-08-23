use super::{TerrainPyramid, TerrainTileKey};
use crate::fields::FieldId;
use crate::heightfield::TileId;
use glam::{Mat4, Vec3, Vec4};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use terra_world::{InfiniteTopology, Lod, TileAddress, WorldError, WorldPosition, WorldRect};
use thiserror::Error;

/// Camera-driven terrain demand policy. Tile and node limits bound both the
/// production request set and the amount of hierarchy work performed in one plan.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerrainDemandConfig {
    pub target_error_px: f32,
    pub coarsen_error_px: f32,
    pub max_demand_tiles: usize,
    pub max_visited_nodes: usize,
}

impl Default for TerrainDemandConfig {
    fn default() -> Self {
        Self {
            target_error_px: 2.0,
            coarsen_error_px: 1.5,
            max_demand_tiles: 256,
            max_visited_nodes: 4096,
        }
    }
}

/// Backend-neutral view data used by the planner. Height bounds are global and
/// deliberately conservative: per-tile bounds may tighten culling later without
/// changing the planner's residency-free authority boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerrainDemandView {
    pub eye: Vec3,
    pub view_proj: Mat4,
    pub fov_y: f32,
    pub near: f32,
    pub viewport_width_px: u32,
    pub viewport_height_px: u32,
    pub min_height: f32,
    pub max_height: f32,
}

/// Camera state for an unbounded topology. X/Z positions remain authoritative
/// `f64` world coordinates; `relative_view_proj` observes positions translated
/// so the eye is at the origin before they are narrowed for clip-space tests.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InfiniteTerrainDemandView {
    pub eye_world: WorldPosition,
    pub coverage_center_world: WorldPosition,
    pub eye_height: f32,
    pub relative_view_proj: Mat4,
    pub fov_y: f32,
    pub near: f32,
    pub viewport_width_px: u32,
    pub viewport_height_px: u32,
    pub min_height: f32,
    pub max_height: f32,
}

/// Persistent-world radii used to derive a finite sparse planning window.
/// Work limits remain in [`TerrainDemandConfig`] and are intentionally transient.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InfiniteTerrainDemandConfig {
    pub preview_radius_m: f64,
    pub horizon_m: f64,
}

/// Conservative world-space approximation error indexed only by LOD. This is
/// O(topology depth), never O(visited area), and can later be tightened by a
/// certified sparse metadata source without changing planner traversal.
#[derive(Debug, Clone, PartialEq)]
pub struct InfiniteTerrainErrorModel {
    errors_by_lod: Vec<f32>,
}

impl InfiniteTerrainErrorModel {
    pub fn try_new(
        topology: &InfiniteTopology,
        errors_by_lod: Vec<f32>,
    ) -> Result<Self, TerrainDemandError> {
        let required = usize::from(topology.config().max_lod.get()) + 1;
        if errors_by_lod.len() < required
            || errors_by_lod
                .iter()
                .take(required)
                .any(|error| !error.is_finite() || *error < 0.0)
        {
            return Err(TerrainDemandError::InvalidInfiniteErrorModel);
        }
        Ok(Self {
            errors_by_lod: errors_by_lod[..required].to_vec(),
        })
    }

    /// Safe initial model when generated sparse tiles have no certified local
    /// envelope yet. The configured height span bounds any parent approximation.
    pub fn from_height_span(
        topology: &InfiniteTopology,
        min_height: f32,
        max_height: f32,
    ) -> Result<Self, TerrainDemandError> {
        if !min_height.is_finite() || !max_height.is_finite() {
            return Err(TerrainDemandError::InvalidInfiniteErrorModel);
        }
        let span = (max_height - min_height).abs().max(1.0e-3);
        Self::try_new(
            topology,
            vec![span; usize::from(topology.config().max_lod.get()) + 1],
        )
    }

    fn error(&self, address: TileAddress) -> f32 {
        self.errors_by_lod[usize::from(address.lod.get())]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TerrainDemandClass {
    CoarseCoverage,
    FallbackAncestor,
    Refinement,
}

/// One immutable demand fact. It contains no page handle, residency bit, cache
/// slot, or publication state; those remain owned by the GPU atlas/cache path.
#[derive(Debug, Clone, PartialEq)]
pub struct TerrainTileDemand {
    pub key: TerrainTileKey,
    pub class: TerrainDemandClass,
    pub projected_error_px: f32,
    pub distance_m: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TerrainDemandPlan {
    pub tiles: Vec<TerrainTileDemand>,
    pub visited_nodes: usize,
    pub culled_nodes: usize,
    pub node_budget_exhausted: bool,
    pub tile_budget_exhausted: bool,
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum TerrainDemandError {
    #[error("geometric-error metadata has {actual} entries; pyramid requires {required}")]
    MetadataLength { required: usize, actual: usize },
    #[error("invalid demand configuration")]
    InvalidConfig,
    #[error("invalid demand view")]
    InvalidView,
    #[error("invalid Infinite demand configuration")]
    InvalidInfiniteConfig,
    #[error("invalid Infinite geometric-error model")]
    InvalidInfiniteErrorModel,
    #[error(
        "mandatory Infinite coarse coverage requires {required} tiles and nodes; limits are {tile_limit} tiles and {node_limit} nodes"
    )]
    InsufficientCoverageBudget {
        required: usize,
        tile_limit: usize,
        node_limit: usize,
    },
    #[error("Infinite demand spatial calculation failed: {0}")]
    Spatial(WorldError),
}

impl From<WorldError> for TerrainDemandError {
    fn from(value: WorldError) -> Self {
        Self::Spatial(value)
    }
}

/// Convert local child-versus-parent errors into a conservative top-down error
/// envelope once per immutable pyramid. Summing the local error with the largest
/// descendant envelope follows the triangle inequality and prevents a feature
/// lost across several coarse levels from being hidden by a zero-error ancestor.
pub fn conservative_geometric_errors(
    pyramid: &TerrainPyramid,
    local_errors: &[f32],
) -> Result<Vec<f32>, TerrainDemandError> {
    let required = pyramid.metadata_len() as usize;
    if local_errors.len() < required {
        return Err(TerrainDemandError::MetadataLength {
            required,
            actual: local_errors.len(),
        });
    }
    let mut errors = local_errors[..required]
        .iter()
        .map(|error| {
            if error.is_finite() && *error >= 0.0 {
                *error
            } else {
                f32::MAX
            }
        })
        .collect::<Vec<_>>();
    for level in (0..pyramid.max_level()).rev() {
        let metrics = pyramid.level_metrics(level).expect("valid pyramid level");
        for tz in 0..metrics.tiles_z() {
            for tx in 0..metrics.tiles_x() {
                let tile = TileId { tx, tz };
                let parent_index = pyramid
                    .tile_metadata_index(level, tile)
                    .expect("valid parent tile") as usize;
                let descendant_error = pyramid
                    .covering_child_tiles(level, tile)
                    .into_iter()
                    .flat_map(|range| range.iter())
                    .filter_map(|child| pyramid.tile_metadata_index(level + 1, child))
                    .map(|index| errors[index as usize])
                    .max_by(f32::total_cmp)
                    .unwrap_or(0.0);
                errors[parent_index] = (errors[parent_index] + descendant_error).min(f32::MAX);
            }
        }
    }
    Ok(errors)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct NodeKey {
    level: u8,
    tz: u32,
    tx: u32,
}

impl NodeKey {
    fn new(level: u8, tile: TileId) -> Self {
        Self {
            level,
            tz: tile.tz,
            tx: tile.tx,
        }
    }

    fn tile(self) -> TileId {
        TileId {
            tx: self.tx,
            tz: self.tz,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct NodeCandidate {
    key: NodeKey,
    projected_error_px: f32,
    distance_m: f32,
}

#[derive(Debug, Clone, Copy)]
struct InfiniteNodeCandidate {
    address: TileAddress,
    projected_error_px: f32,
    distance_m: f32,
}

/// Stateful only for threshold hysteresis. The retained keys describe the last
/// refinement decision and are not a mirror of current or intended residency.
#[derive(Debug, Default, Clone)]
pub struct TerrainDemandPlanner {
    previously_refined: BTreeSet<NodeKey>,
    previously_refined_infinite: BTreeSet<TileAddress>,
}

impl TerrainDemandPlanner {
    pub fn reset(&mut self) {
        self.previously_refined.clear();
        self.previously_refined_infinite.clear();
    }

    pub fn plan(
        &mut self,
        pyramid: &TerrainPyramid,
        geometric_errors: &[f32],
        view: TerrainDemandView,
        config: TerrainDemandConfig,
    ) -> Result<TerrainDemandPlan, TerrainDemandError> {
        validate_inputs(pyramid, geometric_errors, view, config)?;

        let mut demands = BTreeMap::<NodeKey, TerrainTileDemand>::new();
        let mut refined = BTreeSet::new();
        let mut queued = BTreeSet::new();
        let mut frontier = VecDeque::new();
        let mut visited_nodes = 0usize;
        let mut culled_nodes = 0usize;
        let mut node_budget_exhausted = false;
        let mut tile_budget_exhausted = false;

        let root = pyramid
            .level_metrics(0)
            .expect("pyramid always has level zero");
        for tz in 0..root.tiles_z() {
            for tx in 0..root.tiles_x() {
                let key = NodeKey::new(0, TileId { tx, tz });
                queued.insert(key);
                frontier.push_back(key);
            }
        }

        while let Some(key) = frontier.pop_front() {
            if visited_nodes >= config.max_visited_nodes {
                node_budget_exhausted = true;
                break;
            }
            visited_nodes += 1;

            let Some(bounds) = tile_bounds(pyramid, key, view.min_height, view.max_height) else {
                continue;
            };
            if aabb_outside_clip(bounds, view.view_proj) {
                culled_nodes += 1;
                continue;
            }

            let distance_m = distance_to_aabb(view.eye, bounds);
            let projected_error_px = node_error(pyramid, geometric_errors, key)
                * pixel_projection_scale(view)
                / distance_m.max(view.near);
            let selected = if key.level == 0 {
                true
            } else if self.previously_refined.contains(&key) {
                projected_error_px >= config.coarsen_error_px
            } else {
                projected_error_px > config.target_error_px
            };
            if !selected {
                continue;
            }

            let missing_ancestors = missing_parent_chain(pyramid, key, &demands);
            let needed = 1usize.saturating_add(missing_ancestors.len());
            if demands.len().saturating_add(needed) > config.max_demand_tiles {
                tile_budget_exhausted = true;
                continue;
            }
            insert_ancestor_demands(
                pyramid,
                geometric_errors,
                view,
                &mut demands,
                missing_ancestors,
            );
            if key.level > 0 {
                refined.insert(key);
                for ancestor in parent_chain(pyramid, key) {
                    if let Some(demand) = demands.get_mut(&ancestor) {
                        if ancestor.level > 0 {
                            demand.class = TerrainDemandClass::FallbackAncestor;
                        }
                    }
                }
            }
            demands.entry(key).or_insert_with(|| TerrainTileDemand {
                key: terrain_key(pyramid, key),
                class: if key.level == 0 {
                    TerrainDemandClass::CoarseCoverage
                } else {
                    TerrainDemandClass::Refinement
                },
                projected_error_px,
                distance_m,
            });

            let Some(children) = pyramid.covering_child_tiles(key.level, key.tile()) else {
                continue;
            };
            let mut candidates = Vec::new();
            for tile in children.iter() {
                let child = NodeKey::new(key.level + 1, tile);
                if queued.contains(&child) {
                    continue;
                }
                let Some(child_bounds) =
                    tile_bounds(pyramid, child, view.min_height, view.max_height)
                else {
                    continue;
                };
                let child_distance = distance_to_aabb(view.eye, child_bounds);
                candidates.push(NodeCandidate {
                    key: child,
                    projected_error_px: node_error(pyramid, geometric_errors, child)
                        * pixel_projection_scale(view)
                        / child_distance.max(view.near),
                    distance_m: child_distance,
                });
            }
            candidates.sort_by(compare_candidates);
            for candidate in candidates {
                queued.insert(candidate.key);
                frontier.push_back(candidate.key);
            }
        }

        self.previously_refined = refined;
        let mut tiles: Vec<_> = demands.into_values().collect();
        tiles.sort_by(|a, b| {
            let (a_level, a_tile) = pyramid
                .level_and_tile(a.key.address)
                .expect("demand address belongs to bounded pyramid");
            let (b_level, b_tile) = pyramid
                .level_and_tile(b.key.address)
                .expect("demand address belongs to bounded pyramid");
            a_level
                .cmp(&b_level)
                .then_with(|| a.class.cmp(&b.class))
                .then_with(|| b.projected_error_px.total_cmp(&a.projected_error_px))
                .then_with(|| a.distance_m.total_cmp(&b.distance_m))
                .then_with(|| a_tile.tz.cmp(&b_tile.tz))
                .then_with(|| a_tile.tx.cmp(&b_tile.tx))
        });
        Ok(TerrainDemandPlan {
            tiles,
            visited_nodes,
            culled_nodes,
            node_budget_exhausted,
            tile_budget_exhausted,
        })
    }

    /// Plan one finite camera-centred window over the signed sparse hierarchy.
    /// Mandatory horizon coverage is established before any visible refinement,
    /// and every admitted refinement is closed over its Euclidean parent chain.
    pub fn plan_infinite(
        &mut self,
        topology: &InfiniteTopology,
        errors: &InfiniteTerrainErrorModel,
        view: InfiniteTerrainDemandView,
        config: TerrainDemandConfig,
        infinite: InfiniteTerrainDemandConfig,
    ) -> Result<TerrainDemandPlan, TerrainDemandError> {
        validate_infinite_inputs(topology, errors, view, config, infinite)?;

        let coverage_rect = centered_world_rect(view.coverage_center_world, infinite.horizon_m)?;
        let coverage_range =
            topology.addresses_intersecting(coverage_rect, topology.config().max_lod)?;
        let coverage_count = coverage_range.checked_len()?;
        if coverage_count > config.max_demand_tiles || coverage_count > config.max_visited_nodes {
            return Err(TerrainDemandError::InsufficientCoverageBudget {
                required: coverage_count,
                tile_limit: config.max_demand_tiles,
                node_limit: config.max_visited_nodes,
            });
        }

        let mut demands = BTreeMap::<TileAddress, TerrainTileDemand>::new();
        let mut refined = BTreeSet::new();
        let mut queued = BTreeSet::new();
        let mut frontier = VecDeque::new();
        let mut visited_nodes = 0usize;
        let mut culled_nodes = 0usize;
        let mut node_budget_exhausted = false;
        let mut tile_budget_exhausted = false;

        let mut coverage = coverage_range.iter().collect::<Vec<_>>();
        coverage.sort_by(|a, b| compare_infinite_addresses_by_distance(topology, view, *a, *b));
        for address in &coverage {
            visited_nodes += 1;
            let candidate = infinite_candidate(topology, errors, view, *address)?;
            demands.insert(
                *address,
                TerrainTileDemand {
                    key: TerrainTileKey::height(*address),
                    class: TerrainDemandClass::CoarseCoverage,
                    projected_error_px: candidate.projected_error_px,
                    distance_m: candidate.distance_m,
                },
            );
        }

        let mut first_children = Vec::new();
        for address in coverage {
            if address.lod == Lod::FINEST {
                continue;
            }
            for child in topology.children(address)? {
                if queued.insert(child)
                    && tile_intersects_radius(
                        topology,
                        child,
                        view.coverage_center_world,
                        infinite.preview_radius_m,
                    )?
                {
                    first_children.push(infinite_candidate(topology, errors, view, child)?);
                }
            }
        }
        first_children.sort_by(compare_infinite_candidates);
        frontier.extend(first_children);

        while let Some(candidate) = frontier.pop_front() {
            if visited_nodes >= config.max_visited_nodes {
                node_budget_exhausted = true;
                break;
            }
            visited_nodes += 1;
            let address = candidate.address;
            let bounds = infinite_tile_relative_bounds(topology, address, view)?;
            if aabb_outside_clip(bounds, view.relative_view_proj) {
                culled_nodes += 1;
                continue;
            }

            let selected = if self.previously_refined_infinite.contains(&address) {
                candidate.projected_error_px >= config.coarsen_error_px
            } else {
                candidate.projected_error_px > config.target_error_px
            };
            if !selected {
                continue;
            }

            let ancestors = infinite_parent_chain(topology, address)?;
            let missing = ancestors
                .iter()
                .copied()
                .filter(|ancestor| !demands.contains_key(ancestor))
                .collect::<Vec<_>>();
            let needed = 1usize.saturating_add(missing.len());
            if demands.len().saturating_add(needed) > config.max_demand_tiles {
                tile_budget_exhausted = true;
                continue;
            }

            for ancestor in missing {
                let ancestor_candidate = infinite_candidate(topology, errors, view, ancestor)?;
                demands.insert(
                    ancestor,
                    TerrainTileDemand {
                        key: TerrainTileKey::height(ancestor),
                        class: if ancestor.lod == topology.config().max_lod {
                            TerrainDemandClass::CoarseCoverage
                        } else {
                            TerrainDemandClass::FallbackAncestor
                        },
                        projected_error_px: ancestor_candidate.projected_error_px,
                        distance_m: ancestor_candidate.distance_m,
                    },
                );
            }
            for ancestor in &ancestors {
                if ancestor.lod != topology.config().max_lod {
                    if let Some(demand) = demands.get_mut(ancestor) {
                        demand.class = TerrainDemandClass::FallbackAncestor;
                    }
                }
            }
            refined.insert(address);
            demands.entry(address).or_insert_with(|| TerrainTileDemand {
                key: TerrainTileKey::height(address),
                class: TerrainDemandClass::Refinement,
                projected_error_px: candidate.projected_error_px,
                distance_m: candidate.distance_m,
            });

            if address.lod != Lod::FINEST {
                let mut children = Vec::new();
                for child in topology.children(address)? {
                    if queued.insert(child)
                        && tile_intersects_radius(
                            topology,
                            child,
                            view.coverage_center_world,
                            infinite.preview_radius_m,
                        )?
                    {
                        children.push(infinite_candidate(topology, errors, view, child)?);
                    }
                }
                children.sort_by(compare_infinite_candidates);
                frontier.extend(children);
            }
        }

        self.previously_refined_infinite = refined;
        let mut tiles = demands.into_values().collect::<Vec<_>>();
        tiles.sort_by(|a, b| {
            b.key
                .address
                .lod
                .cmp(&a.key.address.lod)
                .then_with(|| a.class.cmp(&b.class))
                .then_with(|| b.projected_error_px.total_cmp(&a.projected_error_px))
                .then_with(|| a.distance_m.total_cmp(&b.distance_m))
                .then_with(|| a.key.address.coord.z.cmp(&b.key.address.coord.z))
                .then_with(|| a.key.address.coord.x.cmp(&b.key.address.coord.x))
        });
        Ok(TerrainDemandPlan {
            tiles,
            visited_nodes,
            culled_nodes,
            node_budget_exhausted,
            tile_budget_exhausted,
        })
    }
}

fn validate_inputs(
    pyramid: &TerrainPyramid,
    geometric_errors: &[f32],
    view: TerrainDemandView,
    config: TerrainDemandConfig,
) -> Result<(), TerrainDemandError> {
    let required = pyramid.metadata_len() as usize;
    if geometric_errors.len() < required {
        return Err(TerrainDemandError::MetadataLength {
            required,
            actual: geometric_errors.len(),
        });
    }
    if !config.target_error_px.is_finite()
        || !config.coarsen_error_px.is_finite()
        || config.target_error_px <= 0.0
        || config.coarsen_error_px < 0.0
        || config.coarsen_error_px >= config.target_error_px
        || config.max_demand_tiles == 0
        || config.max_visited_nodes == 0
    {
        return Err(TerrainDemandError::InvalidConfig);
    }
    if !view.eye.is_finite()
        || !view.view_proj.is_finite()
        || !view.fov_y.is_finite()
        || !view.near.is_finite()
        || !view.min_height.is_finite()
        || !view.max_height.is_finite()
        || view.fov_y <= 0.0
        || view.fov_y >= std::f32::consts::PI
        || view.near <= 0.0
        || view.viewport_width_px == 0
        || view.viewport_height_px == 0
    {
        return Err(TerrainDemandError::InvalidView);
    }
    Ok(())
}

fn validate_infinite_inputs(
    topology: &InfiniteTopology,
    errors: &InfiniteTerrainErrorModel,
    view: InfiniteTerrainDemandView,
    config: TerrainDemandConfig,
    infinite: InfiniteTerrainDemandConfig,
) -> Result<(), TerrainDemandError> {
    if !config.target_error_px.is_finite()
        || !config.coarsen_error_px.is_finite()
        || config.target_error_px <= 0.0
        || config.coarsen_error_px < 0.0
        || config.coarsen_error_px >= config.target_error_px
        || config.max_demand_tiles == 0
        || config.max_visited_nodes == 0
    {
        return Err(TerrainDemandError::InvalidConfig);
    }
    if !view.relative_view_proj.is_finite()
        || !view.eye_height.is_finite()
        || !view.fov_y.is_finite()
        || !view.near.is_finite()
        || !view.min_height.is_finite()
        || !view.max_height.is_finite()
        || view.fov_y <= 0.0
        || view.fov_y >= std::f32::consts::PI
        || view.near <= 0.0
        || view.viewport_width_px == 0
        || view.viewport_height_px == 0
    {
        return Err(TerrainDemandError::InvalidView);
    }
    if !infinite.preview_radius_m.is_finite()
        || !infinite.horizon_m.is_finite()
        || infinite.preview_radius_m <= 0.0
        || infinite.horizon_m < infinite.preview_radius_m
    {
        return Err(TerrainDemandError::InvalidInfiniteConfig);
    }
    let required = usize::from(topology.config().max_lod.get()) + 1;
    if errors.errors_by_lod.len() < required {
        return Err(TerrainDemandError::InvalidInfiniteErrorModel);
    }
    Ok(())
}

fn centered_world_rect(
    center: WorldPosition,
    radius_m: f64,
) -> Result<WorldRect, TerrainDemandError> {
    let min = WorldPosition::try_new(center.x_m() - radius_m, center.z_m() - radius_m)?;
    let max = WorldPosition::try_new(center.x_m() + radius_m, center.z_m() + radius_m)?;
    Ok(WorldRect::try_new(min, max)?)
}

fn tile_intersects_radius(
    topology: &InfiniteTopology,
    address: TileAddress,
    center: WorldPosition,
    radius_m: f64,
) -> Result<bool, TerrainDemandError> {
    let rect = topology.tile_extent(address)?.world;
    let nearest_x = center.x_m().clamp(rect.min().x_m(), rect.max().x_m());
    let nearest_z = center.z_m().clamp(rect.min().z_m(), rect.max().z_m());
    let dx = nearest_x - center.x_m();
    let dz = nearest_z - center.z_m();
    Ok(dx.mul_add(dx, dz * dz) <= radius_m * radius_m)
}

fn infinite_tile_relative_bounds(
    topology: &InfiniteTopology,
    address: TileAddress,
    view: InfiniteTerrainDemandView,
) -> Result<(Vec3, Vec3), TerrainDemandError> {
    let rect = topology.tile_extent(address)?.world;
    let min_y = view.min_height.min(view.max_height) - view.eye_height;
    let max_y = view
        .min_height
        .max(view.max_height)
        .max(view.min_height.min(view.max_height) + 1.0e-3)
        - view.eye_height;
    let min = Vec3::new(
        relative_f64_to_f32(rect.min().x_m() - view.eye_world.x_m())?,
        min_y,
        relative_f64_to_f32(rect.min().z_m() - view.eye_world.z_m())?,
    );
    let max = Vec3::new(
        relative_f64_to_f32(rect.max().x_m() - view.eye_world.x_m())?,
        max_y,
        relative_f64_to_f32(rect.max().z_m() - view.eye_world.z_m())?,
    );
    Ok((min, max))
}

fn relative_f64_to_f32(value: f64) -> Result<f32, TerrainDemandError> {
    if !value.is_finite() || value.abs() > f64::from(f32::MAX) {
        return Err(TerrainDemandError::Spatial(WorldError::ArithmeticOverflow));
    }
    Ok(value as f32)
}

fn infinite_candidate(
    topology: &InfiniteTopology,
    errors: &InfiniteTerrainErrorModel,
    view: InfiniteTerrainDemandView,
    address: TileAddress,
) -> Result<InfiniteNodeCandidate, TerrainDemandError> {
    let bounds = infinite_tile_relative_bounds(topology, address, view)?;
    let distance_m = distance_to_aabb(Vec3::ZERO, bounds);
    Ok(InfiniteNodeCandidate {
        address,
        projected_error_px: errors.error(address) * infinite_pixel_projection_scale(view)
            / distance_m.max(view.near),
        distance_m,
    })
}

fn infinite_pixel_projection_scale(view: InfiniteTerrainDemandView) -> f32 {
    view.viewport_height_px as f32 / (2.0 * (view.fov_y * 0.5).tan())
}

fn compare_infinite_candidates(a: &InfiniteNodeCandidate, b: &InfiniteNodeCandidate) -> Ordering {
    b.projected_error_px
        .total_cmp(&a.projected_error_px)
        .then_with(|| a.distance_m.total_cmp(&b.distance_m))
        .then_with(|| a.address.cmp(&b.address))
}

fn compare_infinite_addresses_by_distance(
    topology: &InfiniteTopology,
    view: InfiniteTerrainDemandView,
    a: TileAddress,
    b: TileAddress,
) -> Ordering {
    let distance = |address| {
        let rect = topology
            .tile_extent(address)
            .expect("validated Infinite coverage address")
            .world;
        let nearest_x = view
            .coverage_center_world
            .x_m()
            .clamp(rect.min().x_m(), rect.max().x_m());
        let nearest_z = view
            .coverage_center_world
            .z_m()
            .clamp(rect.min().z_m(), rect.max().z_m());
        let dx = nearest_x - view.coverage_center_world.x_m();
        let dz = nearest_z - view.coverage_center_world.z_m();
        dx.mul_add(dx, dz * dz)
    };
    distance(a)
        .total_cmp(&distance(b))
        .then_with(|| a.coord.z.cmp(&b.coord.z))
        .then_with(|| a.coord.x.cmp(&b.coord.x))
}

fn infinite_parent_chain(
    topology: &InfiniteTopology,
    address: TileAddress,
) -> Result<Vec<TileAddress>, TerrainDemandError> {
    let mut parents = Vec::new();
    let mut child = address;
    while child.lod < topology.config().max_lod {
        child = topology.parent(child)?;
        parents.push(child);
    }
    Ok(parents)
}

fn terrain_key(pyramid: &TerrainPyramid, key: NodeKey) -> TerrainTileKey {
    TerrainTileKey::new(
        None,
        FieldId::Height,
        pyramid
            .address(key.level, key.tile())
            .expect("demand node belongs to bounded pyramid"),
    )
}

fn node_error(pyramid: &TerrainPyramid, errors: &[f32], key: NodeKey) -> f32 {
    let index = pyramid
        .tile_metadata_index(key.level, key.tile())
        .expect("visited node belongs to pyramid") as usize;
    let error = errors[index];
    if error.is_finite() && error >= 0.0 {
        error
    } else {
        f32::MAX
    }
}

fn pixel_projection_scale(view: TerrainDemandView) -> f32 {
    view.viewport_height_px as f32 / (2.0 * (view.fov_y * 0.5).tan())
}

fn compare_candidates(a: &NodeCandidate, b: &NodeCandidate) -> Ordering {
    b.projected_error_px
        .total_cmp(&a.projected_error_px)
        .then_with(|| a.distance_m.total_cmp(&b.distance_m))
        .then_with(|| a.key.cmp(&b.key))
}

fn parent_chain(pyramid: &TerrainPyramid, key: NodeKey) -> BTreeSet<NodeKey> {
    let mut parents = BTreeSet::new();
    let mut frontier = vec![key];
    while let Some(child) = frontier.pop() {
        let Some(range) = pyramid.covering_parent_tiles(child.level, child.tile()) else {
            continue;
        };
        for tile in range.iter() {
            let parent = NodeKey::new(child.level - 1, tile);
            if parents.insert(parent) {
                frontier.push(parent);
            }
        }
    }
    parents
}

fn missing_parent_chain(
    pyramid: &TerrainPyramid,
    key: NodeKey,
    demands: &BTreeMap<NodeKey, TerrainTileDemand>,
) -> Vec<NodeKey> {
    parent_chain(pyramid, key)
        .into_iter()
        .filter(|parent| !demands.contains_key(parent))
        .collect()
}

fn insert_ancestor_demands(
    pyramid: &TerrainPyramid,
    errors: &[f32],
    view: TerrainDemandView,
    demands: &mut BTreeMap<NodeKey, TerrainTileDemand>,
    ancestors: Vec<NodeKey>,
) {
    for ancestor in ancestors {
        let Some(bounds) = tile_bounds(pyramid, ancestor, view.min_height, view.max_height) else {
            continue;
        };
        let distance_m = distance_to_aabb(view.eye, bounds);
        demands.insert(
            ancestor,
            TerrainTileDemand {
                key: terrain_key(pyramid, ancestor),
                class: if ancestor.level == 0 {
                    TerrainDemandClass::CoarseCoverage
                } else {
                    TerrainDemandClass::FallbackAncestor
                },
                projected_error_px: node_error(pyramid, errors, ancestor)
                    * pixel_projection_scale(view)
                    / distance_m.max(view.near),
                distance_m,
            },
        );
    }
}

fn tile_bounds(
    pyramid: &TerrainPyramid,
    key: NodeKey,
    min_height: f32,
    max_height: f32,
) -> Option<(Vec3, Vec3)> {
    let metrics = pyramid.level_metrics(key.level)?;
    let extent = pyramid.tile_extent(key.level, key.tile())?;
    let min_y = min_height.min(max_height);
    let max_y = min_height.max(max_height).max(min_y + 1.0e-3);
    let min = Vec3::new(
        extent.origin_x as f32 / metrics.width as f32 * metrics.world_size_x,
        min_y,
        extent.origin_z as f32 / metrics.height as f32 * metrics.world_size_z,
    );
    let max = Vec3::new(
        extent.origin_x.saturating_add(extent.width) as f32 / metrics.width as f32
            * metrics.world_size_x,
        max_y,
        extent.origin_z.saturating_add(extent.height) as f32 / metrics.height as f32
            * metrics.world_size_z,
    );
    Some((min, max))
}

fn distance_to_aabb(point: Vec3, bounds: (Vec3, Vec3)) -> f32 {
    let delta = (bounds.0 - point).max(Vec3::ZERO) + (point - bounds.1).max(Vec3::ZERO);
    delta.length()
}

fn aabb_outside_clip(bounds: (Vec3, Vec3), view_proj: Mat4) -> bool {
    let mut clip = [Vec4::ZERO; 8];
    let mut index = 0;
    for x in [bounds.0.x, bounds.1.x] {
        for y in [bounds.0.y, bounds.1.y] {
            for z in [bounds.0.z, bounds.1.z] {
                clip[index] = view_proj * Vec4::new(x, y, z, 1.0);
                index += 1;
            }
        }
    }
    clip.iter().all(|p| p.x < -p.w)
        || clip.iter().all(|p| p.x > p.w)
        || clip.iter().all(|p| p.y < -p.w)
        || clip.iter().all(|p| p.y > p.w)
        || clip.iter().all(|p| p.z < 0.0)
        || clip.iter().all(|p| p.z > p.w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PyramidConfig;
    use terra_world::{InfiniteTopologyConfig, TileCoord};

    fn view(eye: Vec3, target: Vec3, viewport_height_px: u32, fov_y: f32) -> TerrainDemandView {
        let aspect = 16.0 / 9.0;
        let projection = Mat4::perspective_rh(fov_y, aspect, 1.0, 100_000.0);
        TerrainDemandView {
            eye,
            view_proj: projection * Mat4::look_at_rh(eye, target, Vec3::Y),
            fov_y,
            near: 1.0,
            viewport_width_px: (viewport_height_px as f32 * aspect) as u32,
            viewport_height_px,
            min_height: -100.0,
            max_height: 500.0,
        }
    }

    fn uniform_errors(pyramid: &TerrainPyramid, error: f32) -> Vec<f32> {
        let mut errors = vec![0.0; pyramid.metadata_len() as usize];
        for level in 1..=pyramid.max_level() {
            let metrics = pyramid.level_metrics(level).unwrap();
            for tz in 0..metrics.tiles_z() {
                for tx in 0..metrics.tiles_x() {
                    let index = pyramid
                        .tile_metadata_index(level, TileId { tx, tz })
                        .unwrap() as usize;
                    errors[index] = error;
                }
            }
        }
        errors
    }

    fn deepest(pyramid: &TerrainPyramid, plan: &TerrainDemandPlan) -> u8 {
        plan.tiles
            .iter()
            .filter_map(|demand| pyramid.topology().level_index(demand.key.address))
            .max()
            .unwrap_or(0)
    }

    fn infinite_topology() -> InfiniteTopology {
        InfiniteTopology::try_new(InfiniteTopologyConfig {
            origin: WorldPosition::ORIGIN,
            tile_size: 16,
            finest_spacing_m: 1.0,
            max_lod: Lod::try_new(3).unwrap(),
        })
        .unwrap()
    }

    fn infinite_view(center_x: f64, center_z: f64) -> InfiniteTerrainDemandView {
        let eye_world = WorldPosition::try_new(center_x, center_z + 96.0).unwrap();
        let center_world = WorldPosition::try_new(center_x, center_z).unwrap();
        let eye_height = 96.0;
        let target_from_eye = Vec3::new(0.0, -eye_height, -96.0);
        let projection = Mat4::perspective_rh(1.0, 16.0 / 9.0, 1.0, 10_000.0);
        InfiniteTerrainDemandView {
            eye_world,
            coverage_center_world: center_world,
            eye_height,
            relative_view_proj: projection * Mat4::look_at_rh(Vec3::ZERO, target_from_eye, Vec3::Y),
            fov_y: 1.0,
            near: 1.0,
            viewport_width_px: 1920,
            viewport_height_px: 1080,
            min_height: -64.0,
            max_height: 256.0,
        }
    }

    fn infinite_config() -> InfiniteTerrainDemandConfig {
        InfiniteTerrainDemandConfig {
            preview_radius_m: 96.0,
            horizon_m: 160.0,
        }
    }

    fn infinite_errors(topology: &InfiniteTopology) -> InfiniteTerrainErrorModel {
        InfiniteTerrainErrorModel::try_new(topology, vec![500.0; 4]).unwrap()
    }

    #[test]
    fn moving_closer_requests_finer_detail() {
        let pyramid = TerrainPyramid::new(PyramidConfig::new(512, 4096.0, 4096.0));
        let errors = uniform_errors(&pyramid, 10.0);
        let target = Vec3::new(2048.0, 0.0, 2048.0);
        let mut far_planner = TerrainDemandPlanner::default();
        let far = far_planner
            .plan(
                &pyramid,
                &errors,
                view(Vec3::new(2048.0, 10_000.0, 10_000.0), target, 1080, 1.0),
                TerrainDemandConfig::default(),
            )
            .unwrap();
        let mut near_planner = TerrainDemandPlanner::default();
        let near = near_planner
            .plan(
                &pyramid,
                &errors,
                view(Vec3::new(2048.0, 1200.0, 3500.0), target, 1080, 1.0),
                TerrainDemandConfig::default(),
            )
            .unwrap();
        assert!(deepest(&pyramid, &near) > deepest(&pyramid, &far));
    }

    #[test]
    fn viewport_height_and_narrower_fov_increase_detail() {
        let pyramid = TerrainPyramid::new(PyramidConfig::new(512, 4096.0, 4096.0));
        let errors = uniform_errors(&pyramid, 2.0);
        let eye = Vec3::new(2048.0, 2500.0, 5000.0);
        let target = Vec3::new(2048.0, 0.0, 2048.0);
        let mut planner = TerrainDemandPlanner::default();
        let short = planner
            .plan(
                &pyramid,
                &errors,
                view(eye, target, 480, 1.2),
                TerrainDemandConfig::default(),
            )
            .unwrap();
        planner.reset();
        let tall = planner
            .plan(
                &pyramid,
                &errors,
                view(eye, target, 1440, 1.2),
                TerrainDemandConfig::default(),
            )
            .unwrap();
        planner.reset();
        let narrow = planner
            .plan(
                &pyramid,
                &errors,
                view(eye, target, 480, 0.6),
                TerrainDemandConfig::default(),
            )
            .unwrap();
        assert!(deepest(&pyramid, &tall) >= deepest(&pyramid, &short));
        assert!(deepest(&pyramid, &narrow) >= deepest(&pyramid, &short));
    }

    #[test]
    fn sub_pixel_error_stays_coarse_and_offscreen_branch_is_absent() {
        let pyramid = TerrainPyramid::new(PyramidConfig::new(512, 4096.0, 4096.0));
        let tiny = uniform_errors(&pyramid, 0.0001);
        let mut planner = TerrainDemandPlanner::default();
        let coarse = planner
            .plan(
                &pyramid,
                &tiny,
                view(
                    Vec3::new(2048.0, 1500.0, 3500.0),
                    Vec3::new(2048.0, 0.0, 2048.0),
                    1080,
                    1.0,
                ),
                TerrainDemandConfig::default(),
            )
            .unwrap();
        assert_eq!(deepest(&pyramid, &coarse), 0);

        let errors = uniform_errors(&pyramid, 100.0);
        planner.reset();
        let away = planner
            .plan(
                &pyramid,
                &errors,
                view(
                    Vec3::new(2048.0, 1000.0, 5000.0),
                    Vec3::new(2048.0, 1000.0, 10_000.0),
                    1080,
                    1.0,
                ),
                TerrainDemandConfig::default(),
            )
            .unwrap();
        assert!(away.tiles.is_empty());
        assert!(away.culled_nodes > 0);
    }

    #[test]
    fn repeated_view_is_stable_and_hysteresis_prevents_threshold_thrash() {
        let pyramid = TerrainPyramid::new(PyramidConfig::new(16, 1024.0, 1024.0));
        let errors = uniform_errors(&pyramid, 1.0);
        let camera = view(
            Vec3::new(512.0, 500.0, 1200.0),
            Vec3::new(512.0, 0.0, 512.0),
            1080,
            1.0,
        );
        let config = TerrainDemandConfig {
            target_error_px: 1.0,
            coarsen_error_px: 0.5,
            ..TerrainDemandConfig::default()
        };
        let mut planner = TerrainDemandPlanner::default();
        let first = planner.plan(&pyramid, &errors, camera, config).unwrap();
        let second = planner.plan(&pyramid, &errors, camera, config).unwrap();
        assert_eq!(first, second);

        let mut lowered = errors.clone();
        for value in &mut lowered {
            *value *= 0.75;
        }
        let held = planner.plan(&pyramid, &lowered, camera, config).unwrap();
        assert_eq!(deepest(&pyramid, &held), deepest(&pyramid, &first));
    }

    #[test]
    fn demand_is_coarse_first_unique_and_bounded_by_visited_nodes() {
        let mut pyramid_config = PyramidConfig::new(16_384, 32_768.0, 32_768.0);
        pyramid_config.tile_size = 128;
        let pyramid = TerrainPyramid::new(pyramid_config);
        let errors = uniform_errors(&pyramid, 1000.0);
        let config = TerrainDemandConfig {
            max_demand_tiles: 32,
            max_visited_nodes: 48,
            ..TerrainDemandConfig::default()
        };
        let mut planner = TerrainDemandPlanner::default();
        let plan = planner
            .plan(
                &pyramid,
                &errors,
                view(
                    Vec3::new(16_384.0, 1000.0, 17_000.0),
                    Vec3::new(16_384.0, 0.0, 16_384.0),
                    1080,
                    0.5,
                ),
                config,
            )
            .unwrap();
        assert!(plan.tiles.len() <= config.max_demand_tiles);
        assert!(plan.visited_nodes <= config.max_visited_nodes);
        assert!(plan.tiles.windows(2).all(|pair| {
            pyramid.topology().level_index(pair[0].key.address)
                <= pyramid.topology().level_index(pair[1].key.address)
        }));
        let unique: BTreeSet<_> = plan.tiles.iter().map(|demand| demand.key.address).collect();
        assert_eq!(unique.len(), plan.tiles.len());
        let finest = pyramid.level_metrics(pyramid.max_level()).unwrap();
        assert!(plan.visited_nodes < (finest.tiles_x() * finest.tiles_z()) as usize);
    }

    #[test]
    fn descendant_envelope_exposes_a_feature_lost_below_zero_error_ancestors() {
        let pyramid = TerrainPyramid::new(PyramidConfig::new(512, 4096.0, 4096.0));
        let mut local = vec![0.0; pyramid.metadata_len() as usize];
        let finest = pyramid.max_level();
        let feature = TileId { tx: 0, tz: 0 };
        local[pyramid.tile_metadata_index(finest, feature).unwrap() as usize] = 100.0;
        let conservative = conservative_geometric_errors(&pyramid, &local).unwrap();
        assert!(
            conservative[pyramid
                .tile_metadata_index(1, TileId { tx: 0, tz: 0 })
                .unwrap() as usize]
                > 0.0
        );

        let mut planner = TerrainDemandPlanner::default();
        let plan = planner
            .plan(
                &pyramid,
                &conservative,
                view(
                    Vec3::new(512.0, 500.0, 1000.0),
                    Vec3::new(512.0, 0.0, 512.0),
                    1080,
                    1.0,
                ),
                TerrainDemandConfig::default(),
            )
            .unwrap();
        assert_eq!(deepest(&pyramid, &plan), finest);
    }

    #[test]
    fn infinite_planning_crosses_zero_and_closes_every_refinement_over_parents() {
        let topology = infinite_topology();
        let errors = infinite_errors(&topology);
        let mut planner = TerrainDemandPlanner::default();
        let plan = planner
            .plan_infinite(
                &topology,
                &errors,
                infinite_view(0.0, 0.0),
                TerrainDemandConfig::default(),
                infinite_config(),
            )
            .unwrap();

        assert!(plan
            .tiles
            .iter()
            .any(|demand| demand.key.address.coord.x < 0));
        assert!(plan
            .tiles
            .iter()
            .any(|demand| demand.key.address.coord.x >= 0));
        assert!(plan
            .tiles
            .iter()
            .any(|demand| demand.key.address.coord.z < 0));
        assert!(plan
            .tiles
            .iter()
            .any(|demand| demand.key.address.coord.z >= 0));

        let addresses = plan
            .tiles
            .iter()
            .map(|demand| demand.key.address)
            .collect::<BTreeSet<_>>();
        for refinement in plan
            .tiles
            .iter()
            .filter(|demand| demand.class == TerrainDemandClass::Refinement)
        {
            let mut address = refinement.key.address;
            while address.lod < topology.config().max_lod {
                address = topology.parent(address).unwrap();
                assert!(addresses.contains(&address), "missing parent {address:?}");
            }
        }
        assert!(plan
            .tiles
            .windows(2)
            .all(|pair| { pair[0].key.address.lod >= pair[1].key.address.lod }));
    }

    #[test]
    fn infinite_planning_work_is_independent_of_distance_from_fixed_origin() {
        let topology = infinite_topology();
        let errors = infinite_errors(&topology);
        let mut near_planner = TerrainDemandPlanner::default();
        let near = near_planner
            .plan_infinite(
                &topology,
                &errors,
                infinite_view(0.0, 0.0),
                TerrainDemandConfig::default(),
                infinite_config(),
            )
            .unwrap();
        let mut far_planner = TerrainDemandPlanner::default();
        let far = far_planner
            .plan_infinite(
                &topology,
                &errors,
                infinite_view(10_000_000.0, -10_000_000.0),
                TerrainDemandConfig::default(),
                infinite_config(),
            )
            .unwrap();

        assert_eq!(near.visited_nodes, far.visited_nodes);
        assert_eq!(near.tiles.len(), far.tiles.len());
    }

    #[test]
    fn infinite_planning_respects_budgets_after_mandatory_coverage() {
        let topology = infinite_topology();
        let errors = infinite_errors(&topology);
        let config = TerrainDemandConfig {
            max_demand_tiles: 24,
            max_visited_nodes: 28,
            ..TerrainDemandConfig::default()
        };
        let mut planner = TerrainDemandPlanner::default();
        let plan = planner
            .plan_infinite(
                &topology,
                &errors,
                infinite_view(0.0, 0.0),
                config,
                InfiniteTerrainDemandConfig {
                    preview_radius_m: 64.0,
                    horizon_m: 96.0,
                },
            )
            .unwrap();
        assert!(plan.tiles.len() <= config.max_demand_tiles);
        assert!(plan.visited_nodes <= config.max_visited_nodes);
        assert!(plan
            .tiles
            .iter()
            .take_while(|demand| demand.key.address.lod == topology.config().max_lod)
            .all(|demand| demand.class == TerrainDemandClass::CoarseCoverage));
    }

    #[test]
    fn infinite_planning_rejects_a_budget_that_cannot_cover_the_horizon() {
        let topology = infinite_topology();
        let errors = infinite_errors(&topology);
        let mut planner = TerrainDemandPlanner::default();
        let error = planner
            .plan_infinite(
                &topology,
                &errors,
                infinite_view(0.0, 0.0),
                TerrainDemandConfig {
                    max_demand_tiles: 1,
                    max_visited_nodes: 1,
                    ..TerrainDemandConfig::default()
                },
                infinite_config(),
            )
            .unwrap_err();
        assert!(matches!(
            error,
            TerrainDemandError::InsufficientCoverageBudget { .. }
        ));
    }

    #[test]
    fn infinite_replanning_drops_irrelevant_demand_and_hysteresis_state() {
        let topology = infinite_topology();
        let errors = infinite_errors(&topology);
        let mut planner = TerrainDemandPlanner::default();
        let first = planner
            .plan_infinite(
                &topology,
                &errors,
                infinite_view(-256.0, -256.0),
                TerrainDemandConfig::default(),
                infinite_config(),
            )
            .unwrap();
        let second = planner
            .plan_infinite(
                &topology,
                &errors,
                infinite_view(1_000_000.0, 1_000_000.0),
                TerrainDemandConfig::default(),
                infinite_config(),
            )
            .unwrap();
        let second_addresses = second
            .tiles
            .iter()
            .map(|demand| demand.key.address)
            .collect::<BTreeSet<_>>();
        assert!(first
            .tiles
            .iter()
            .all(|demand| !second_addresses.contains(&demand.key.address)));
        assert!(planner.previously_refined_infinite.len() <= second.visited_nodes);
    }

    #[test]
    fn infinite_signed_quadrants_produce_expected_finest_coordinates() {
        let topology = infinite_topology();
        for (x, z) in [(-32.0, -32.0), (-32.0, 32.0), (32.0, -32.0), (32.0, 32.0)] {
            let address = topology
                .address_at_world(WorldPosition::try_new(x, z).unwrap(), Lod::FINEST)
                .unwrap();
            assert_eq!(address.coord.x.signum(), (x as i64).signum());
            assert_eq!(address.coord.z.signum(), (z as i64).signum());
        }
        assert_eq!(
            topology
                .address_at_world(WorldPosition::try_new(-0.001, -0.001).unwrap(), Lod::FINEST,)
                .unwrap()
                .coord,
            TileCoord { x: -1, z: -1 }
        );
    }
}
