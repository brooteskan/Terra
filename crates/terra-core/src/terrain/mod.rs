//! Final-output tile streaming: the resolution ladder, the GPU-mirrored
//! residency cache, and progressive refinement.

mod cache;
mod demand;
mod pyramid;
mod refinement;
mod runtime;
mod tile;

pub use cache::{
    ResidentTile, TerrainCacheKey, TileCacheError, TileCacheEviction, TileCacheInsert,
    TileCacheStats, TilePageHandle, TileResidencyCache,
};
pub use demand::{
    conservative_geometric_errors, TerrainDemandClass, TerrainDemandConfig, TerrainDemandError,
    TerrainDemandPlan, TerrainDemandPlanner, TerrainDemandView, TerrainTileDemand,
};
pub use pyramid::{
    PyramidConfig, TerrainLevel, TerrainPyramid, TerrainTileExtent, TerrainTileRange,
};
pub use refinement::{EditorRefinementState, RefinementController, RefinementTimings};
pub use runtime::TerrainRuntime;
pub use tile::TerrainTileKey;
