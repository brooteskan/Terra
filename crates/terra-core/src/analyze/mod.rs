//! Terrain analysis masks and CPU erosion references.

pub mod aeolian;
mod erosion;
pub mod hydraulic_core;
mod level_schedule;
mod level_step;
pub mod mass_wasting;
mod metrics;
pub mod terrain_amplification;
pub mod terrain_stats;

pub use aeolian::{
    transport_steps_for_quality, AeolianResult, AeolianState, AeolianTransportParams,
};
pub use erosion::{
    apply_particle_erosion, apply_sediment_material_ids, apply_sediment_softness, hydraulic_erode,
    hydraulic_erode_with_fields, hydraulic_erode_with_hardness, hydraulic_erode_with_strata,
    hydraulic_erode_with_strata_ex, thermal_erode, thermal_erode_with_hardness,
    thermal_erode_with_strata, thermal_erode_with_strata_ex, HydraulicResult, SEDIMENT_MATERIAL_ID,
};
pub use hydraulic_core::{
    apply_transport_model, clamp_timestep_cfl, generate_rainfall_field, params_from_effect_filter,
    sanitize_field, HydraulicErosionCore, HydraulicOutputFlags, HydraulicOutputs, LayeredMaterials,
    MassDiagnostics, RainMapSource, TerrainHydroState, TransportModelExt,
};
pub use level_schedule::{HighDetailMode, LevelStepSettings};
pub use level_step::{
    amplify_sim_levels, author_sim_levels, default_sim_levels, downsample_height, draft_sim_levels,
    hydraulic_erode_leveled, hydraulic_erode_leveled_with_fields,
    hydraulic_erode_leveled_with_hardness, multi_scale_amplify, thermal_erode_leveled,
    thermal_erode_leveled_with_hardness, AmplifyResult, SimLevel,
};
pub use mass_wasting::{
    debris_flow_erode, mud_settle_mass, sediment_fill_soft_mass, talus_apron,
    thermal_erode_layered, thermal_erode_mass, DebrisFlowMassLedger, DebrisFlowResult,
    MassWastingState, ThermalResult,
};
pub use metrics::*;
pub use terrain_amplification::{
    amplify_terrain, AmplificationBands, TerrainAmplificationParams, TerrainAmplificationResult,
};
pub use terrain_stats::{Histogram, HypsometricCurve, StreamOrderDistribution, TerrainStatistics};
