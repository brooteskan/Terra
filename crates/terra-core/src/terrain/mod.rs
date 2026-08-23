//! Final-output tile streaming: the resolution ladder, the GPU-mirrored
//! residency cache, and progressive refinement.

mod cache;
mod demand;
mod domain;
mod geometric_error;
mod pyramid;
mod refinement;
mod runtime;
mod tile;
mod work;

pub use cache::{
    ResidentTile, TerrainCacheKey, TileCacheError, TileCacheEviction, TileCacheInsert,
    TileCacheStats, TilePageHandle, TileResidencyCache,
};
pub use demand::{
    conservative_geometric_errors, InfiniteTerrainDemandConfig, InfiniteTerrainDemandView,
    InfiniteTerrainErrorModel, TerrainDemandClass, TerrainDemandConfig, TerrainDemandError,
    TerrainDemandPlan, TerrainDemandPlanner, TerrainDemandView, TerrainTileDemand,
};
pub use domain::{
    TerrainDomainError, TerrainEvaluationDomain, TerrainEvaluationSpace, TerrainSampleExtent,
    TerrainWorldTransform,
};
pub use geometric_error::measure_tile_geometric_error;
pub use pyramid::{
    PyramidConfig, TerrainLevel, TerrainPyramid, TerrainTileExtent, TerrainTileRange,
};
pub use refinement::{EditorRefinementState, RefinementController, RefinementTimings};
pub use runtime::{TerrainRuntime, TerrainRuntimeConfig, TerrainRuntimeTopology};
pub use tile::TerrainTileKey;
pub use work::{
    TerrainContentStamp, TerrainTileWorkBudget, TerrainTileWorkKey, TerrainTileWorkLease,
    TerrainTileWorkRequest, TerrainTileWorkScheduler, TerrainTileWorkSource, TerrainTileWorkStats,
};
