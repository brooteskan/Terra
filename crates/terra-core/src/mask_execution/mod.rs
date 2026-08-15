//! Context-aware mask baking and distribution execution.

mod bake;
mod coverage;
mod dist_nodes;
mod distribution;

pub use bake::{bake_mask_assets, bake_mask_assets_resolved};
pub use coverage::coverage_estimate;
pub use dist_nodes::{apply_effect_public, bake_dist_node_base, bake_dist_nodes, DistBakeContext};
pub use distribution::{bake_distribution, bake_distribution_with_context};
