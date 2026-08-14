//! Layer parameter kinds (split by family).

use crate::mask::MaskSource;
use serde::{Deserialize, Serialize};

/// Shared hydraulic-family transport bias (one solver, many artist presets).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransportModel {
    /// General-purpose sheet / rill hydraulic erosion.
    #[default]
    Hydraulic,
    /// Broader smoother drainage, lower incision, more lateral slip + deposition.
    SoftFlows,
    /// Stronger channel incision with ridge protection from ridge/valley analysis.
    RidgedFlows,
    /// Narrow-channel / particle footprint, high trajectory definition.
    ThinFlows,
    /// Wide particle/path kernel, larger capacity and deposition footprint.
    WideFlows,
    /// Capacity / transport / deposition / loose-sediment forward.
    SedimentFlows,
    /// Layered bedrock + soft sediment with independent erodibility.
    HydraulicSediment,
}

impl TransportModel {
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Hydraulic => "Hydraulic",
            Self::SoftFlows => "Soft Flows",
            Self::RidgedFlows => "Ridged Flows",
            Self::ThinFlows => "Thin Flows",
            Self::WideFlows => "Wide Flows",
            Self::SedimentFlows => "Sediment Flows",
            Self::HydraulicSediment => "Hydraulic Sediment",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalErosionParams {
    pub talus_angle_deg: f32,
    pub iterations: u32,
    pub strength: f32,
    /// Constant bedrock hardness K ∈ \[0,1\]. Soft (0) erodes fully; hard (1) resists.
    #[serde(default = "default_soft_hardness")]
    pub hardness: f32,
    /// Optional painted / procedural hardness override (`None` / default = use constant).
    #[serde(default)]
    pub hardness_source: MaskSource,
    /// Max loose material (meters) that can weather / move per cell step.
    #[serde(default = "default_material_amount")]
    pub material_amount: f32,
    /// Yang-style bedrock→loose weathering rate (K_th). 0 disables detach.
    #[serde(default = "default_weathering_rate")]
    pub weathering_rate: f32,
    /// Downhill transport hop count (cells) for loose debris before settling.
    #[serde(default = "default_transport_distance")]
    pub transport_distance: f32,
    /// Track bedrock vs loose debris separately (default on).
    #[serde(default = "default_true")]
    pub layered_materials: bool,
    /// 0 = quality default schedule length; else clamp multilevel count.
    #[serde(default)]
    pub level_count: u32,
    /// Skip this many coarse levels (WC Start Level).
    #[serde(default)]
    pub start_level: u32,
    /// Scales fine-level effect (WC Level Step Strength).
    #[serde(default = "default_level_step_strength")]
    pub level_step_strength: f32,
    /// Per-level strength bars (empty = use scalar `level_step_strength` only).
    #[serde(default)]
    pub level_step_curve: Vec<f32>,
}

fn default_soft_hardness() -> f32 {
    0.0
}

fn default_material_amount() -> f32 {
    8.0
}

fn default_weathering_rate() -> f32 {
    1.0
}

fn default_transport_distance() -> f32 {
    1.0
}

fn default_level_step_strength() -> f32 {
    1.0
}

fn default_particle_density() -> f32 {
    0.06
}

fn default_particle_lifetime() -> u32 {
    32
}

fn default_particle_inertia() -> f32 {
    0.2
}

fn default_particle_radius() -> u32 {
    2
}

fn default_bedrock_hardness() -> f32 {
    0.55
}

fn default_sediment_layer_hardness() -> f32 {
    0.08
}

fn default_incision_bias() -> f32 {
    1.0
}

fn default_flow_footprint() -> f32 {
    1.0
}

fn default_hydro_flow_threshold() -> f32 {
    0.15
}

fn default_true() -> bool {
    true
}

impl Default for ThermalErosionParams {
    fn default() -> Self {
        Self {
            talus_angle_deg: 35.0,
            iterations: 40,
            strength: 0.5,
            hardness: 0.0,
            hardness_source: MaskSource::None,
            material_amount: default_material_amount(),
            weathering_rate: default_weathering_rate(),
            transport_distance: default_transport_distance(),
            layered_materials: true,
            level_count: 0,
            start_level: 0,
            level_step_strength: 1.0,
            level_step_curve: Vec::new(),
        }
    }
}

/// Jain, Beneš & Cordonnier 2024 debris-flow erosion for steep / low-drainage terrain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebrisFlowParams {
    pub iterations: u32,
    /// Explicit forward-Euler time step (years-scale units).
    #[serde(default = "default_debris_dt")]
    pub dt: f32,
    /// Critical talus angle for thermal trigger \(E_{th}\).
    #[serde(default = "default_debris_talus")]
    pub talus_angle_deg: f32,
    /// Thermal detach coefficient \(k_{th}\) (Jain Eq. 8).
    #[serde(default = "default_thermal_k")]
    pub thermal_k: f32,
    /// Debris abrasion coefficient (Jain Eq. 12).
    #[serde(default = "default_abrasion_k")]
    pub abrasion_k: f32,
    #[serde(default = "default_abrasion_exp_q")]
    pub abrasion_exp_q: f32,
    #[serde(default = "default_abrasion_exp_s")]
    pub abrasion_exp_s: f32,
    /// Yield threshold factor controls deposit slope (Jain \(c\), Table 1).
    #[serde(default = "default_yield_stress")]
    pub yield_stress: f32,
    #[serde(default = "default_threshold_exp_q")]
    pub threshold_exp_q: f32,
    #[serde(default = "default_threshold_exp_s")]
    pub threshold_exp_s: f32,
    /// Coupled fluvial stream-power coefficient (0 = debris-only).
    #[serde(default = "default_fluvial_k")]
    pub fluvial_k: f32,
    #[serde(default = "default_fluvial_deposition")]
    pub fluvial_deposition: f32,
    #[serde(default = "default_fluvial_m")]
    pub fluvial_m: f32,
    #[serde(default = "default_fluvial_n")]
    pub fluvial_n: f32,
    /// Hillslope Laplacian creep coefficient.
    #[serde(default = "default_hillslope_k")]
    pub hillslope_k: f32,
    #[serde(default = "default_precipitation")]
    pub precipitation: f32,
    #[serde(default = "default_soft_hardness")]
    pub hardness: f32,
    #[serde(default)]
    pub hardness_source: MaskSource,
    #[serde(default)]
    pub initial_debris_thickness: f32,
    #[serde(default)]
    pub initial_sediment_thickness: f32,
    /// Cap debris deposit height per step (meters).
    #[serde(default = "default_max_deposit")]
    pub max_deposit_per_step: f32,
    /// Recompute Priority-Flood drainage every N iters (1 = every).
    #[serde(default = "default_drainage_reuse_stride")]
    pub drainage_reuse_stride: u32,
    /// Always refill depressions when refreshing drainage.
    #[serde(default = "default_true")]
    pub refill_depressions: bool,
    #[serde(default)]
    pub seed: u64,
}

fn default_debris_dt() -> f32 {
    5.0
}
fn default_debris_talus() -> f32 {
    30.0
}
fn default_thermal_k() -> f32 {
    0.005
}
fn default_abrasion_k() -> f32 {
    0.001
}
fn default_abrasion_exp_q() -> f32 {
    0.08
}
fn default_abrasion_exp_s() -> f32 {
    1.0
}
fn default_yield_stress() -> f32 {
    0.3
}
fn default_threshold_exp_q() -> f32 {
    0.25
}
fn default_threshold_exp_s() -> f32 {
    1.0
}
fn default_fluvial_k() -> f32 {
    0.0007
}
fn default_fluvial_deposition() -> f32 {
    0.01
}
fn default_fluvial_m() -> f32 {
    0.4
}
fn default_fluvial_n() -> f32 {
    1.0
}
fn default_hillslope_k() -> f32 {
    0.04
}
fn default_precipitation() -> f32 {
    1.0
}
fn default_max_deposit() -> f32 {
    2.5
}

impl Default for DebrisFlowParams {
    fn default() -> Self {
        Self {
            iterations: 48,
            dt: default_debris_dt(),
            talus_angle_deg: default_debris_talus(),
            thermal_k: default_thermal_k(),
            abrasion_k: default_abrasion_k(),
            abrasion_exp_q: default_abrasion_exp_q(),
            abrasion_exp_s: default_abrasion_exp_s(),
            yield_stress: default_yield_stress(),
            threshold_exp_q: default_threshold_exp_q(),
            threshold_exp_s: default_threshold_exp_s(),
            fluvial_k: default_fluvial_k(),
            fluvial_deposition: default_fluvial_deposition(),
            fluvial_m: default_fluvial_m(),
            fluvial_n: default_fluvial_n(),
            hillslope_k: default_hillslope_k(),
            precipitation: default_precipitation(),
            hardness: 0.0,
            hardness_source: MaskSource::None,
            initial_debris_thickness: 0.0,
            initial_sediment_thickness: 0.0,
            max_deposit_per_step: default_max_deposit(),
            drainage_reuse_stride: 1,
            refill_depressions: true,
            seed: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HydraulicErosionParams {
    pub iterations: u32,
    pub rainfall: f32,
    pub evaporation: f32,
    /// Optional spatial rainfall multiplier consumed inside the water solve.
    #[serde(default)]
    pub rainfall_source: MaskSource,
    /// Optional shape-protection field. One protects authored terrain from incision.
    #[serde(default)]
    pub protection_source: MaskSource,
    /// Strength applied to the protection field before erodibility is evaluated.
    #[serde(default)]
    pub protection_strength: f32,
    pub capacity: f32,
    pub erosion: f32,
    pub deposition: f32,
    pub timestep: f32,
    /// Constant hardness K âˆˆ \[0,1\] modulating erodibility as `(1-K)`.
    #[serde(default = "default_soft_hardness")]
    pub hardness: f32,
    #[serde(default)]
    pub hardness_source: MaskSource,
    /// Amplify deposition at steepâ†’flat slope breaks (alluvial fans). 0 = off.
    #[serde(default)]
    pub fan_boost: f32,
    /// Prefer depositing along high-flow / low-slope corridors (floodplain fill). 0 = off.
    #[serde(default)]
    pub floodplain_bias: f32,
    /// Post-hydraulic thermal bank-slip strength (0 = off). Builds scree/talus after undercutting.
    #[serde(default)]
    pub bank_slip: f32,
    /// Soften hardness (and tag sediment material IDs when present) in deposited zones. 0 = off.
    #[serde(default)]
    pub sediment_softness: f32,
    /// Fine-scale inertial droplets per terrain cell. Zero disables the detail pass.
    #[serde(default = "default_particle_density")]
    pub particle_density: f32,
    /// Maximum advection steps per detail droplet.
    #[serde(default = "default_particle_lifetime")]
    pub particle_lifetime: u32,
    /// Momentum retained by a droplet while following the height gradient.
    #[serde(default = "default_particle_inertia")]
    pub particle_inertia: f32,
    /// Radius in cells of the erosion brush carried by each droplet.
    #[serde(default = "default_particle_radius")]
    pub particle_radius: u32,
    /// Shared-core transport preset (Soft/Ridged/Thin/Wide/Sediment/…).
    #[serde(default)]
    pub transport_model: TransportModel,
    /// When true, erode soft loose sediment before harder bedrock.
    #[serde(default)]
    pub layered_materials: bool,
    /// Bedrock hardness K ∈ [0,1] for layered mode.
    #[serde(default = "default_bedrock_hardness")]
    pub bedrock_hardness: f32,
    /// Loose-sediment hardness K ∈ [0,1] for layered mode.
    #[serde(default = "default_sediment_layer_hardness")]
    pub sediment_hardness: f32,
    /// Initial loose-sediment thickness (meters) when layered mode starts.
    #[serde(default)]
    pub initial_sediment_thickness: f32,
    /// Ridge/divide protection strength from ridge–valley analysis (Ridged Flows).
    #[serde(default)]
    pub ridge_protection: f32,
    /// Lateral bank smoothing bias folded into bank-slip (Soft/Wide Flows).
    #[serde(default)]
    pub lateral_smoothing: f32,
    /// Channel incision multiplier (Ridged/Thin raise; Soft lowers).
    #[serde(default = "default_incision_bias")]
    pub incision_bias: f32,
    /// Hydraulic / particle footprint scale (Thin < 1, Wide > 1).
    #[serde(default = "default_flow_footprint")]
    pub flow_footprint: f32,
    /// Optional flow-accumulation threshold hint for filter→core bridging.
    #[serde(default = "default_hydro_flow_threshold")]
    pub flow_threshold: f32,
    /// Emit erosion mask aux (default on).
    #[serde(default = "default_true")]
    pub output_erosion: bool,
    /// Emit deposition mask aux (default on).
    #[serde(default = "default_true")]
    pub output_deposition: bool,
    /// Emit flow / flux aux (default on).
    #[serde(default = "default_true")]
    pub output_flow: bool,
    /// Emit water-depth aux (default on).
    #[serde(default = "default_true")]
    pub output_water: bool,
    /// Emit sediment aux (default on).
    #[serde(default = "default_true")]
    pub output_sediment: bool,
    /// Emit channel mask aux (default on).
    #[serde(default = "default_true")]
    pub output_channels: bool,
    /// 0 = quality default schedule length; else clamp multilevel count.
    #[serde(default)]
    pub level_count: u32,
    /// Skip this many coarse levels (WC Start Level).
    #[serde(default)]
    pub start_level: u32,
    /// Scales fine-level effect (WC Level Step Strength).
    #[serde(default = "default_level_step_strength")]
    pub level_step_strength: f32,
    /// Per-level strength bars (empty = scalar only).
    #[serde(default)]
    pub level_step_curve: Vec<f32>,
}

impl Default for HydraulicErosionParams {
    fn default() -> Self {
        Self {
            iterations: 60,
            rainfall: 0.02,
            evaporation: 0.01,
            capacity: 0.1,
            erosion: 0.3,
            rainfall_source: MaskSource::None,
            protection_source: MaskSource::None,
            protection_strength: 0.0,
            deposition: 0.3,
            timestep: 0.2,
            hardness: 0.0,
            hardness_source: MaskSource::None,
            fan_boost: 0.0,
            floodplain_bias: 0.0,
            bank_slip: 0.0,
            sediment_softness: 0.0,
            particle_density: default_particle_density(),
            particle_lifetime: default_particle_lifetime(),
            particle_inertia: default_particle_inertia(),
            particle_radius: default_particle_radius(),
            transport_model: TransportModel::Hydraulic,
            layered_materials: false,
            bedrock_hardness: default_bedrock_hardness(),
            sediment_hardness: default_sediment_layer_hardness(),
            initial_sediment_thickness: 0.0,
            ridge_protection: 0.0,
            lateral_smoothing: 0.0,
            incision_bias: 1.0,
            flow_footprint: 1.0,
            flow_threshold: default_hydro_flow_threshold(),
            output_erosion: true,
            output_deposition: true,
            output_flow: true,
            output_water: true,
            output_sediment: true,
            output_channels: true,
            level_count: 0,
            start_level: 0,
            level_step_strength: 1.0,
            level_step_curve: Vec::new(),
        }
    }
}

impl HydraulicErosionParams {
    /// Deposition-forward defaults for alluvial fan / river-valley presets.
    pub fn depositional() -> Self {
        Self {
            iterations: 90,
            rainfall: 0.045,
            evaporation: 0.012,
            capacity: 0.12,
            erosion: 0.4,
            rainfall_source: MaskSource::None,
            protection_source: MaskSource::None,
            protection_strength: 0.0,
            deposition: 0.65,
            timestep: 0.22,
            hardness: 0.0,
            hardness_source: MaskSource::None,
            fan_boost: 1.4,
            floodplain_bias: 0.9,
            bank_slip: 0.45,
            sediment_softness: 0.55,
            particle_density: 0.1,
            particle_lifetime: default_particle_lifetime(),
            particle_inertia: default_particle_inertia(),
            particle_radius: 3,
            transport_model: TransportModel::SedimentFlows,
            layered_materials: true,
            bedrock_hardness: 0.5,
            sediment_hardness: 0.1,
            initial_sediment_thickness: 0.25,
            ridge_protection: 0.0,
            lateral_smoothing: 0.35,
            incision_bias: 0.85,
            flow_footprint: 1.2,
            flow_threshold: default_hydro_flow_threshold(),
            output_erosion: true,
            output_deposition: true,
            output_flow: true,
            output_water: true,
            output_sediment: true,
            output_channels: true,
            level_count: 0,
            start_level: 0,
            level_step_strength: 1.0,
            level_step_curve: Vec::new(),
        }
    }

    pub fn with_transport_model(model: TransportModel) -> Self {
        let mut p = Self::default();
        p.transport_model = model;
        p
    }
}

/// Stream-power erosion (SPE) parameters.
///
/// Incision follows \(E = K\,A^{m}\,S^{n}\) modulated by softness \(1-K_{\mathrm{hard}}\).
/// Drainage uses Priority-Flood + D8/D∞ on the CPU export oracle; Draft/Medium GPU
/// runs an approximate multi-pass D8 SPE.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamPowerParams {
    pub iterations: u32,
    /// Erodibility coefficient \(K\) in the stream-power law.
    ///
    /// Acts on **grid-relative** D8 slope (drop per 1 / √2 cells) and world-m²
    /// drainage area — the convention baked into [`crate::hydro::spe_increment`]
    /// on the `hydro` path. This is **not** numerically comparable with
    /// [`crate::landscape_evolution::LandscapeEvolutionParams`]'s `k`, which acts
    /// on world-metric slope and rain-scaled discharge. Do not silently unify the
    /// two: it would retune every saved project and needs a versioned document
    /// migration (station D1's territory).
    pub k: f32,
    /// Drainage-area exponent \(m\) (classic ~0.5).
    pub m: f32,
    /// Local-slope exponent \(n\) (classic ~1).
    pub n: f32,
    /// Optional tectonic uplift per iteration (meters). Usually 0 when uplift is a prior layer.
    pub uplift_rate: f32,
    /// Heights cannot cut below this floor (meters).
    pub base_level: f32,
    /// Incision time-step / scale factor.
    pub dt: f32,
    /// Prefer Dâˆž accumulation when true; else D8.
    pub use_dinfinity: bool,
    /// Re-run Priority-Flood every iteration (costlier, fewer trapped pits).
    pub refill_each_iter: bool,
    /// Recompute D8/Dâˆž drainage every N SPE iterations (1 = every iter).
    /// Draft preview may raise this to skip barriers; Full/Export keeps 1.
    #[serde(default = "default_drainage_reuse_stride")]
    pub drainage_reuse_stride: u32,
    /// Constant bedrock hardness \(K \in [0,1]\); soft rock incises faster.
    #[serde(default = "default_soft_hardness")]
    pub hardness: f32,
    #[serde(default)]
    pub hardness_source: MaskSource,
    /// 0 = off; >0 applies a light ridge-distance valley prime before SPE (Dendry-inspired).
    #[serde(default)]
    pub dendritic_seed: f32,
    /// Accumulation threshold used when baking the stream-order aux map.
    #[serde(default = "default_stream_threshold")]
    pub stream_threshold: f32,
    /// 0 = quality default schedule length; else clamp multilevel count.
    #[serde(default)]
    pub level_count: u32,
    /// Skip this many coarse levels (WC Start Level).
    #[serde(default)]
    pub start_level: u32,
    /// Scales fine-level effect (WC Level Step Strength).
    #[serde(default = "default_level_step_strength")]
    pub level_step_strength: f32,
    /// Per-level strength bars (empty = scalar only).
    #[serde(default)]
    pub level_step_curve: Vec<f32>,
}

fn default_stream_threshold() -> f32 {
    40.0
}

fn default_drainage_reuse_stride() -> u32 {
    1
}

impl Default for StreamPowerParams {
    fn default() -> Self {
        Self {
            iterations: 24,
            k: 0.08,
            m: 0.5,
            n: 1.0,
            uplift_rate: 0.0,
            base_level: 0.0,
            dt: 1.0,
            use_dinfinity: false,
            refill_each_iter: false,
            drainage_reuse_stride: 1,
            hardness: 0.0,
            hardness_source: MaskSource::None,
            dendritic_seed: 0.0,
            stream_threshold: 40.0,
            level_count: 0,
            start_level: 0,
            level_step_strength: 1.0,
            level_step_curve: Vec::new(),
        }
    }
}

/// Multi-scale geomorphology amplify (Phase I).
///
/// Coarseâ†’fine `SimLevel` schedule applies wavelength-scaled thermal polish,
/// light SPE incision, and optional deposition. Hard ridges and optional lock
/// masks resist carve so large drainage structure is preserved while soft
/// areas receive coherent fine detail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiScaleAmplifyParams {
    /// 0 = auto schedule from field resolution; else clamp level count.
    #[serde(default)]
    pub level_count: u32,
    /// Thermal talus polish strength (scaled per level).
    pub thermal_strength: f32,
    pub thermal_iters: u32,
    pub talus_angle_deg: f32,
    /// SPE erodibility scale (multiplies a small base \(K\)).
    pub spe_strength: f32,
    pub spe_iters: u32,
    /// Light hydraulic deposition at fine levels (fans/settle). 0 = off.
    pub deposition_strength: f32,
    /// Soft-area detail gain (>1 adds more amplify delta where soft).
    pub detail_boost: f32,
    /// Constant hardness \(K \in [0,1]\).
    #[serde(default = "default_soft_hardness")]
    pub hardness: f32,
    #[serde(default)]
    pub hardness_source: MaskSource,
    /// Optional ridge/feature lock mask (1 = preserve input height more strongly).
    #[serde(default)]
    pub ridge_lock: MaskSource,
    /// How strongly `ridge_lock` / hardness retarget toward the pre-amplify surface.
    pub lock_strength: f32,
}

impl Default for MultiScaleAmplifyParams {
    fn default() -> Self {
        Self {
            level_count: 0,
            thermal_strength: 0.35,
            thermal_iters: 12,
            talus_angle_deg: 34.0,
            spe_strength: 0.45,
            spe_iters: 6,
            deposition_strength: 0.25,
            detail_boost: 1.25,
            hardness: 0.0,
            hardness_source: MaskSource::None,
            ridge_lock: MaskSource::None,
            lock_strength: 0.85,
        }
    }
}
