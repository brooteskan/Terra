use super::{TerrainPyramid, TerrainTileKey};
use crate::fields::FieldId;
use crate::heightfield::TileId;
use glam::{Mat4, Vec3, Vec4};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
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

/// Stateful only for threshold hysteresis. The retained keys describe the last
/// refinement decision and are not a mirror of current or intended residency.
#[derive(Debug, Default, Clone)]
pub struct TerrainDemandPlanner {
    previously_refined: BTreeSet<NodeKey>,
}

impl TerrainDemandPlanner {
    pub fn reset(&mut self) {
        self.previously_refined.clear();
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
                key: terrain_key(key),
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
            a.key
                .level
                .cmp(&b.key.level)
                .then_with(|| a.class.cmp(&b.class))
                .then_with(|| b.projected_error_px.total_cmp(&a.projected_error_px))
                .then_with(|| a.distance_m.total_cmp(&b.distance_m))
                .then_with(|| a.key.tile.tz.cmp(&b.key.tile.tz))
                .then_with(|| a.key.tile.tx.cmp(&b.key.tile.tx))
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

fn terrain_key(key: NodeKey) -> TerrainTileKey {
    TerrainTileKey {
        layer: None,
        field: FieldId::Height,
        level: key.level,
        tile: key.tile(),
    }
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
                key: terrain_key(ancestor),
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

    fn deepest(plan: &TerrainDemandPlan) -> u8 {
        plan.tiles
            .iter()
            .map(|demand| demand.key.level)
            .max()
            .unwrap_or(0)
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
        assert!(deepest(&near) > deepest(&far));
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
        assert!(deepest(&tall) >= deepest(&short));
        assert!(deepest(&narrow) >= deepest(&short));
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
        assert_eq!(deepest(&coarse), 0);

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
        assert_eq!(deepest(&held), deepest(&first));
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
        assert!(plan
            .tiles
            .windows(2)
            .all(|pair| pair[0].key.level <= pair[1].key.level));
        let unique: BTreeSet<_> = plan
            .tiles
            .iter()
            .map(|demand| NodeKey::new(demand.key.level, demand.key.tile))
            .collect();
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
        assert_eq!(deepest(&plan), finest);
    }
}
