//! Phase 2 geomorphology / hydrology foundation.
//!
//! Shared terrain-analysis and drainage primitives for later erosion, canyon,
//! river, sediment, and geological filters.

mod accumulation;
mod debug;
mod depression;
mod derivatives;
mod drainage;
mod fixtures;
mod routing;
mod streams;
mod watersheds;

pub use accumulation::{
    accumulate_discharge, accumulate_drainage_area, accumulate_drainage_area_d8, Precipitation,
    PrecipitationSource,
};
pub use debug::{bake_debug_field, GeomorphDebugField};
pub(crate) use depression::priority_flood_fill_from_outlets;
pub use depression::{
    handle_depressions, priority_flood_fill, BreachParams, DepressionMode, DepressionResult,
    PreserveBasinsParams,
};
pub use derivatives::{
    aspect_radians, cavity_openness, convexity_concavity, gaussian_curvature, gradient_components,
    laplacian, mean_curvature, multi_radius_roughness, plan_curvature, profile_curvature,
    ridge_valley_likelihood, slope_magnitude, DerivativeOptions, DerivativeSet,
};
pub use drainage::{
    distance_to_ridge, distance_to_valley, drainage_analysis, drainage_density, ridge_mask,
    valley_mask, DrainageAnalysis,
};
pub use fixtures::{
    closed_depression, cone, noisy_mountain, plane, single_basin, single_valley, two_basins,
    SyntheticKind,
};
pub use routing::{
    build_flow_graph, flow_direction_d8, flow_direction_dinfinity, D8Drainage, FlowGraph,
    FlowModel, FlowReceiver, D8_OFFSETS, NO_FLOW,
};
pub use streams::{
    channel_width_estimate, distance_to_channel, extract_streams, strahler_order,
    StreamExtractParams, StreamNetwork,
};
pub use watersheds::{
    discover_outlets, local_contributing_area, watershed_boundaries, watersheds_from_graph,
    WatershedOptions, WatershedResult,
};

use crate::heightfield::Heightfield;
use crate::mask::MaskField;

/// Full geomorph analysis pass over a DEM (CPU oracle).
#[derive(Debug, Clone)]
pub struct GeomorphAnalysis {
    pub routed_height: Heightfield,
    pub fill_delta: MaskField,
    pub graph: FlowGraph,
    pub drainage_area: Vec<f32>,
    pub discharge: Vec<f32>,
    pub watersheds: WatershedResult,
    pub streams: StreamNetwork,
    pub drainage: DrainageAnalysis,
    pub derivatives: DerivativeSet,
}

/// Options for [`analyze_terrain`].
#[derive(Debug, Clone)]
pub struct GeomorphOptions {
    pub depression: DepressionMode,
    pub flow_model: FlowModel,
    pub precipitation: Precipitation,
    pub streams: StreamExtractParams,
    pub derivatives: DerivativeOptions,
    pub watershed: WatershedOptions,
}

impl Default for GeomorphOptions {
    fn default() -> Self {
        Self {
            depression: DepressionMode::Fill,
            flow_model: FlowModel::D8,
            precipitation: Precipitation::uniform(1.0),
            streams: StreamExtractParams::default(),
            derivatives: DerivativeOptions::default(),
            watershed: WatershedOptions::default(),
        }
    }
}

/// Run the Phase 2 geomorph pipeline on `height`.
pub fn analyze_terrain(height: &Heightfield, opts: &GeomorphOptions) -> GeomorphAnalysis {
    let dep = handle_depressions(height, opts.depression);
    let graph = build_flow_graph(&dep.height, opts.flow_model);
    let drainage_area = accumulate_drainage_area(&graph, &opts.precipitation);
    let discharge = accumulate_discharge(&graph, &opts.precipitation, height.metrics);
    let watersheds = watersheds_from_graph(&graph, &dep.height, &opts.watershed);
    let streams = extract_streams(&graph, &drainage_area, height.metrics, &opts.streams);
    let drainage = drainage_analysis(&dep.height, &graph, &streams, &drainage_area);
    let derivatives = DerivativeSet::compute(&dep.height, &opts.derivatives);

    GeomorphAnalysis {
        routed_height: dep.height,
        fill_delta: dep.fill_delta,
        graph,
        drainage_area,
        discharge,
        watersheds,
        streams,
        drainage,
        derivatives,
    }
}
