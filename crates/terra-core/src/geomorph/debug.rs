//! Opt-in debug visualisations for geomorph fields.
//!
//! These are intentionally **not** part of the default artist viewport chrome.
//! Select via command palette / debug menu.

use crate::heightfield::Heightfield;
use crate::mask::MaskField;

use super::depression::DepressionMode;
use super::derivatives::DerivativeOptions;
use super::routing::FlowModel;
use super::{
    analyze_terrain, GeomorphOptions, Precipitation, StreamExtractParams, WatershedOptions,
};

/// Every Phase 2 analysis product available as a debug overlay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GeomorphDebugField {
    GradientMagnitude,
    Slope,
    Aspect,
    Laplacian,
    ProfileCurvature,
    PlanCurvature,
    MeanCurvature,
    GaussianCurvature,
    Convexity,
    Concavity,
    Roughness,
    Cavity,
    RidgeLikelihood,
    ValleyLikelihood,
    FillDelta,
    FlowDirection,
    FlowAccumulation,
    Discharge,
    WatershedId,
    WatershedBoundary,
    ChannelMask,
    ChannelWidth,
    StreamOrder,
    StreamHierarchy,
    DistanceToChannel,
    RidgeMask,
    ValleyMask,
    DistanceToRidge,
    DistanceToValley,
    DrainageDensity,
}

impl GeomorphDebugField {
    pub const ALL: &'static [GeomorphDebugField] = &[
        Self::GradientMagnitude,
        Self::Slope,
        Self::Aspect,
        Self::Laplacian,
        Self::ProfileCurvature,
        Self::PlanCurvature,
        Self::MeanCurvature,
        Self::GaussianCurvature,
        Self::Convexity,
        Self::Concavity,
        Self::Roughness,
        Self::Cavity,
        Self::RidgeLikelihood,
        Self::ValleyLikelihood,
        Self::FillDelta,
        Self::FlowDirection,
        Self::FlowAccumulation,
        Self::Discharge,
        Self::WatershedId,
        Self::WatershedBoundary,
        Self::ChannelMask,
        Self::ChannelWidth,
        Self::StreamOrder,
        Self::StreamHierarchy,
        Self::DistanceToChannel,
        Self::RidgeMask,
        Self::ValleyMask,
        Self::DistanceToRidge,
        Self::DistanceToValley,
        Self::DrainageDensity,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::GradientMagnitude => "Gradient |∇h|",
            Self::Slope => "Slope",
            Self::Aspect => "Aspect",
            Self::Laplacian => "Laplacian",
            Self::ProfileCurvature => "Profile curvature",
            Self::PlanCurvature => "Plan curvature",
            Self::MeanCurvature => "Mean curvature",
            Self::GaussianCurvature => "Gaussian curvature",
            Self::Convexity => "Convexity",
            Self::Concavity => "Concavity",
            Self::Roughness => "Roughness",
            Self::Cavity => "Cavity / openness",
            Self::RidgeLikelihood => "Ridge likelihood",
            Self::ValleyLikelihood => "Valley likelihood",
            Self::FillDelta => "Depression fill Δ",
            Self::FlowDirection => "Flow direction",
            Self::FlowAccumulation => "Flow accumulation",
            Self::Discharge => "Discharge Q",
            Self::WatershedId => "Watershed ID",
            Self::WatershedBoundary => "Watershed boundary",
            Self::ChannelMask => "Channel mask",
            Self::ChannelWidth => "Channel width",
            Self::StreamOrder => "Stream order (Strahler)",
            Self::StreamHierarchy => "Stream hierarchy",
            Self::DistanceToChannel => "Distance to channel",
            Self::RidgeMask => "Ridge mask",
            Self::ValleyMask => "Valley mask",
            Self::DistanceToRidge => "Distance to ridge",
            Self::DistanceToValley => "Distance to valley",
            Self::DrainageDensity => "Drainage density",
        }
    }

    pub fn command_id(self) -> &'static str {
        match self {
            Self::GradientMagnitude => "debug.geomorph.gradient",
            Self::Slope => "debug.geomorph.slope",
            Self::Aspect => "debug.geomorph.aspect",
            Self::Laplacian => "debug.geomorph.laplacian",
            Self::ProfileCurvature => "debug.geomorph.profile_curvature",
            Self::PlanCurvature => "debug.geomorph.plan_curvature",
            Self::MeanCurvature => "debug.geomorph.mean_curvature",
            Self::GaussianCurvature => "debug.geomorph.gaussian_curvature",
            Self::Convexity => "debug.geomorph.convexity",
            Self::Concavity => "debug.geomorph.concavity",
            Self::Roughness => "debug.geomorph.roughness",
            Self::Cavity => "debug.geomorph.cavity",
            Self::RidgeLikelihood => "debug.geomorph.ridge_likelihood",
            Self::ValleyLikelihood => "debug.geomorph.valley_likelihood",
            Self::FillDelta => "debug.geomorph.fill_delta",
            Self::FlowDirection => "debug.geomorph.flow_direction",
            Self::FlowAccumulation => "debug.geomorph.flow_accumulation",
            Self::Discharge => "debug.geomorph.discharge",
            Self::WatershedId => "debug.geomorph.watershed_id",
            Self::WatershedBoundary => "debug.geomorph.watershed_boundary",
            Self::ChannelMask => "debug.geomorph.channel_mask",
            Self::ChannelWidth => "debug.geomorph.channel_width",
            Self::StreamOrder => "debug.geomorph.stream_order",
            Self::StreamHierarchy => "debug.geomorph.stream_hierarchy",
            Self::DistanceToChannel => "debug.geomorph.distance_to_channel",
            Self::RidgeMask => "debug.geomorph.ridge_mask",
            Self::ValleyMask => "debug.geomorph.valley_mask",
            Self::DistanceToRidge => "debug.geomorph.distance_to_ridge",
            Self::DistanceToValley => "debug.geomorph.distance_to_valley",
            Self::DrainageDensity => "debug.geomorph.drainage_density",
        }
    }
}

/// Bake a normalised \[0,1\] debug field for 2D preview.
pub fn bake_debug_field(
    height: &Heightfield,
    field: GeomorphDebugField,
    opts: Option<&GeomorphOptions>,
) -> MaskField {
    let default_opts = GeomorphOptions {
        depression: DepressionMode::Fill,
        flow_model: FlowModel::D8,
        precipitation: Precipitation::uniform(1.0),
        streams: StreamExtractParams::default(),
        derivatives: DerivativeOptions {
            radius_m: 0.0,
            roughness_radii_m: vec![height.metrics.dx() * 4.0, height.metrics.dx() * 16.0],
            openness_sectors: 8,
        },
        watershed: WatershedOptions::default(),
    };
    let opts = opts.unwrap_or(&default_opts);
    let analysis = analyze_terrain(height, opts);
    let d = &analysis.derivatives;

    let raw = match field {
        GeomorphDebugField::GradientMagnitude => d.gradient_mag.clone(),
        GeomorphDebugField::Slope => d.slope.clone(),
        GeomorphDebugField::Aspect => d.aspect.clone(),
        GeomorphDebugField::Laplacian => d.laplacian.clone(),
        GeomorphDebugField::ProfileCurvature => d.profile_curvature.clone(),
        GeomorphDebugField::PlanCurvature => d.plan_curvature.clone(),
        GeomorphDebugField::MeanCurvature => d.mean_curvature.clone(),
        GeomorphDebugField::GaussianCurvature => d.gaussian_curvature.clone(),
        GeomorphDebugField::Convexity => d.convexity.clone(),
        GeomorphDebugField::Concavity => d.concavity.clone(),
        GeomorphDebugField::Roughness => d.roughness.clone(),
        GeomorphDebugField::Cavity => d.cavity.clone(),
        GeomorphDebugField::RidgeLikelihood => analysis.drainage.ridge_likelihood.clone(),
        GeomorphDebugField::ValleyLikelihood => analysis.drainage.valley_likelihood.clone(),
        GeomorphDebugField::FillDelta => analysis.fill_delta.clone(),
        GeomorphDebugField::FlowDirection => analysis.graph.direction_mask.clone(),
        GeomorphDebugField::FlowAccumulation => {
            vec_to_mask_log(&analysis.drainage_area, height.metrics)
        }
        GeomorphDebugField::Discharge => vec_to_mask_log(&analysis.discharge, height.metrics),
        GeomorphDebugField::WatershedId => analysis.watersheds.id_mask.clone(),
        GeomorphDebugField::WatershedBoundary => analysis.watersheds.boundaries.clone(),
        GeomorphDebugField::ChannelMask => analysis.streams.channel_mask.clone(),
        GeomorphDebugField::ChannelWidth => normalize_mask(&analysis.streams.channel_width),
        GeomorphDebugField::StreamOrder => analysis.streams.order_normalised.clone(),
        GeomorphDebugField::StreamHierarchy => analysis.streams.hierarchy.clone(),
        GeomorphDebugField::DistanceToChannel => analysis.streams.distance_to_channel.clone(),
        GeomorphDebugField::RidgeMask => analysis.drainage.ridge_mask.clone(),
        GeomorphDebugField::ValleyMask => analysis.drainage.valley_mask.clone(),
        GeomorphDebugField::DistanceToRidge => analysis.drainage.distance_to_ridge.clone(),
        GeomorphDebugField::DistanceToValley => analysis.drainage.distance_to_valley.clone(),
        GeomorphDebugField::DrainageDensity => analysis.drainage.drainage_density.clone(),
    };

    normalize_mask(&raw)
}

fn vec_to_mask_log(data: &[f32], metrics: crate::heightfield::HeightfieldMetrics) -> MaskField {
    let mut out = MaskField::zeros(metrics);
    let max_ln = data
        .iter()
        .copied()
        .fold(0.0f32, f32::max)
        .max(0.0)
        .ln_1p()
        .max(1e-6);
    for j in 0..metrics.height {
        for i in 0..metrics.width {
            let v = data[(j * metrics.width + i) as usize].max(0.0).ln_1p() / max_ln;
            out.set(i, j, v.clamp(0.0, 1.0));
        }
    }
    out
}

fn normalize_mask(mask: &MaskField) -> MaskField {
    let data = mask.data();
    let min_v = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max_v = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let span = (max_v - min_v).max(1e-6);
    let mut out = MaskField::zeros(mask.metrics);
    for j in 0..mask.metrics.height {
        for i in 0..mask.metrics.width {
            let v = (mask.get(i, j) - min_v) / span;
            out.set(i, j, v.clamp(0.0, 1.0));
        }
    }
    out
}
