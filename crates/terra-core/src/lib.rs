//! Terra core: backend-neutral heightfields, layer stacks, masks, and terrain algorithms.
//!
//! This crate must remain free of `wgpu` and UI crates.
//!
//! Prefer importing from submodules (`terra_core::layer::LayerKind`, …).
//! Crate-root re-exports below are a curated convenience surface for the app
//! binary and are not an invitation to grow a flat global namespace.

pub mod analyze;
pub mod authoring;
pub mod authoring_coordinate;
pub mod biome_definition;
pub mod biome_paint;
pub mod climate;
pub mod command;
pub mod contextual_create;
pub mod deps;
pub mod document;
pub mod domain;
pub mod field_data;
pub mod fields;
pub mod filter_params;
pub mod generators;
pub mod geology;
pub mod geomorph;
pub mod heightfield;
pub mod hydro;
pub mod ids;
pub mod invalidation;
pub mod landscape_blueprint;
pub mod landscape_evolution;
pub mod landscape_style;
pub mod layer;
pub mod layer_reach;
pub mod mask;
pub mod mask_execution;
pub mod mask_field;
pub mod mask_ir;
pub mod mask_types;
pub mod material_schema;
pub mod matter_sim;
pub mod noise;
pub mod operation_placement;
pub mod quality;
pub mod raster;
pub mod realism_benchmark;
pub mod rebuild_feedback;
pub mod rebuild_state;
pub mod scatter;
pub mod shape_history;
pub mod shape_object;
pub mod simd_ops;
pub mod simulation_scenario;
pub mod sparse_paint;
pub mod spatial_kernels;
pub mod surface;
pub mod terrain;
pub mod terrain_plan;
pub mod terrain_recipe;
#[doc(hidden)]
pub mod test_fixtures;
pub mod tiling;
pub mod volumetric;
pub mod world_archetype;
pub mod world_rules;

pub use fields::{
    fields_invalidated_by, height_dependents, shared_physical_fields, AuxMaps, FieldId,
    IterativeFieldGuard,
};
pub use geomorph::{
    analyze_terrain, bake_debug_field, GeomorphAnalysis, GeomorphDebugField, GeomorphOptions,
};
/// The cancellation primitive backing eval cancellation. Re-exported so callers
/// and fixtures have one canonical path and sibling crates need no direct
/// `terra-jobs` dependency.
pub use terra_jobs::CancelToken;
pub use terra_world::{
    BoundedLevel, BoundedTopology, BoundedTopologyConfig, InfiniteTopology, InfiniteTopologyConfig,
    Lod, SampleCoord, SampleExtent, SampleSpacing, SampleWorldTransform, SpatialDomain,
    TileAddress, TileAddressRange, TileCoord, TileExtent, WorldBounds, WorldError, WorldPosition,
    WorldRect,
};
pub use terrain_recipe::{
    build_terrain_recipe_from_stack, recipe_matches_stack, RecipeItem, RecipeItemKind,
    RecipeRebuildStatus,
};

pub use authoring_coordinate::{
    AuthoringBrushStamp, AuthoringCoordinateError, AuthoringPoint, BoundedUv, TerrainSurfaceHit,
};
pub use biome_definition::{
    blend_height_deltas, normalize_weights, BiomeDefinition, BiomeDefinitionId, BiomeLibrary,
    BiomeOverlapPolicy, BiomePlacementRules, PlacementCombineMode,
};
pub use biome_paint::{
    BiomeLayer, BiomeLayerId, BiomePaintTool, BiomeWeightChannel, HoleLayer, ShapeTransform,
};
pub use document::{EditorSession, MaskPaintStrokeUndo, PaintStrokeUndo, TerrainDocument};
pub use domain::{
    authoring_order_is_arbitrary, behavioural_differences, classify_in_context,
    classify_layer_kind, evaluation_eval_stage_order, incomplete_project_diagnostics,
    workflow_stage_metadata_order, world_eval_outline, DomainBiomeRef, DomainLayerRef,
    DomainParent, DomainRole, DomainView, SoftDiagnostic,
};
pub use heightfield::{HeightTile, Heightfield, HeightfieldMetrics, MetricsError, TileId};
pub use landscape_blueprint::{
    preview_resolution_for_world_size, ArchetypeId, EvalStage, LandscapeBlueprint,
};
pub use landscape_evolution::{
    asymmetric_belt, evaluate_landscape_evolution, synthesise_uplift, BoundaryMode,
    EvolutionSolverMode, LandscapeEvolutionInput, LandscapeEvolutionOperator,
    LandscapeEvolutionOutput, LandscapeEvolutionParams, UpliftMode,
};
pub use landscape_style::{LandscapeStyle, LandscapeStyleParams};
pub use layer::{
    biome_destination_section, is_shape_kind, AccentCategory, BlendMode, BuildStatus, CachePolicy,
    GroupEvalMode, GroupInputMode, Layer, LayerCapabilities, LayerGroup, LayerId,
    LayerInstanceMeta, LayerKind, LayerStack, LayerTypeMeta, LayerTypeRegistry, MaskCompatibility,
    OperationCategory, StackCategory, WorkflowStage,
};
pub use matter_sim::{
    diagnose_matter_sim, outputs_for_consumer, sync_scenario_outputs_from_matter,
    MatterAdvancedParams, MatterArtistControls, MatterArtistSource, MatterOutputConsumer,
    MatterSimConfig, MatterType,
};
pub use realism_benchmark::{
    validate_benchmark_structure, BenchmarkExpectations, RealismBenchmark,
};
pub use shape_object::{ShapeKind, ShapeObject, ShapeObjectId, ShapeObjectStore};
pub use simulation_scenario::{
    diagnose_scenario, layer_kind_is_scenario_compatible, MatterSource, MatterSourceKind,
    OutputApplicationSettings, OutputInfluence, ScenarioPass, ScenarioPassKind, ScenarioQuality,
    ScenarioResultState, ScenarioScope, ScenarioSnapshot, ScenarioTimeline, SimulationDomain,
    SimulationScenario, SimulationScenarioCommand, SimulationScenarioId, SimulationScenarioLibrary,
};
pub use sparse_paint::{
    PaintPage, PaintPageCoord, PaintStrokeId, SparsePaintChannelKey, SparsePaintStore,
};
pub use terrain::{
    conservative_geometric_errors, measure_tile_geometric_error, EditorRefinementState,
    InfiniteTerrainDemandConfig, InfiniteTerrainDemandView, InfiniteTerrainErrorModel,
    PyramidConfig, RefinementController, RefinementTimings, ResidentTile, TerrainCacheKey,
    TerrainContentStamp, TerrainDemandClass, TerrainDemandConfig, TerrainDemandError,
    TerrainDemandPlan, TerrainDemandPlanner, TerrainDemandPlannerStats, TerrainDemandView,
    TerrainDomainError, TerrainEvaluationDomain, TerrainEvaluationSpace, TerrainLevel,
    TerrainPyramid, TerrainRuntime, TerrainRuntimeConfig, TerrainRuntimeTopology,
    TerrainSampleExtent, TerrainTileDemand, TerrainTileExtent, TerrainTileKey, TerrainTileRange,
    TerrainTileWorkBudget, TerrainTileWorkKey, TerrainTileWorkLease, TerrainTileWorkRequest,
    TerrainTileWorkScheduler, TerrainTileWorkSource, TerrainTileWorkStats, TerrainWorldTransform,
    TileCacheError, TileCacheEviction, TileCacheInsert, TileCacheStats, TilePageHandle,
    TileResidencyCache,
};
pub use world_archetype::{
    alpine_world, badlands_world, blank_world_design, build_world, coastal_world, desert_world,
    dune_field_world, old_mountains_world, river_valley_world, tropical_island_world,
    young_mountains_world, WorldTemplate,
};
pub use world_rules::{
    beach_preset, builtin_world_rule_presets, cliff_preset, coastal_wetness_preset,
    diagnose_world_rule, high_altitude_rock_preset, placement_from_conditions, riverbank_preset,
    snowline_preset, underwater_sand_preset, world_rule_preset_by_name, WorldRule,
    WorldRuleCommand, WorldRuleEffect, WorldRuleEffectKind, WorldRuleId, WorldRuleLibrary,
    WorldRulePhase, WorldRuleScope,
};
