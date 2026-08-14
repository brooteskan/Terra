//! Unified hydraulic erosion core (Beneš / Šťava / Mei-lite family).
//!
//! One shared solver powers Hydraulic, Soft/Ridged/Thin/Wide Flows, Sediment Flows,
//! and Hydraulic Sediment. Presets bias transport / erodibility / particle kernels —
//! they do not fork independent erosion implementations or post-apply ridge noise.

use crate::geomorph::ridge_valley_likelihood;
use crate::heightfield::{Heightfield, HeightfieldMetrics};
use crate::layer::{
    EffectFilterKind, EffectFilterParams, FractalNoiseType, HydraulicErosionParams, NoiseParams,
    TransportModel,
};
use crate::mask::{MaskField, MaskSource};
use crate::noise::fbm;

use super::erosion::{apply_particle_erosion, hydraulic_erode_with_fields, HydraulicResult};

/// How rainfall is authored for every hydraulic-family operator.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RainMapSource {
    /// Uniform rate (relative units, multiplied by `HydraulicErosionParams::rainfall`).
    Constant(f32),
    /// Procedural fBm rainfall field.
    Noise {
        seed: u64,
        frequency: f32,
        scale: f32,
        offset: f32,
    },
    /// Painted / mask library entry (via [`MaskSource`]).
    Painted(MaskSource),
    /// Artist distribution stack resolved externally into a field.
    Distribution,
    /// Explicit external map (already baked).
    External,
}

impl Default for RainMapSource {
    fn default() -> Self {
        Self::Constant(1.0)
    }
}

impl RainMapSource {
    pub fn from_mask_source(source: &MaskSource) -> Self {
        match source {
            MaskSource::None => Self::Constant(1.0),
            MaskSource::Constant(v) => Self::Constant((*v).max(0.0)),
            MaskSource::Noise { seed, frequency } => Self::Noise {
                seed: *seed,
                frequency: *frequency,
                scale: 1.0,
                offset: 0.0,
            },
            MaskSource::Painted { .. }
            | MaskSource::Named(_)
            | MaskSource::LayerOutput { .. }
            | MaskSource::Rainfall => Self::Painted(source.clone()),
            _ => Self::Painted(source.clone()),
        }
    }
}

/// Extension helpers for [`TransportModel`].
pub trait TransportModelExt {
    fn from_effect_filter(kind: EffectFilterKind) -> Option<TransportModel>;
}

impl TransportModelExt for TransportModel {
    fn from_effect_filter(kind: EffectFilterKind) -> Option<TransportModel> {
        match kind {
            EffectFilterKind::SoftFlows => Some(Self::SoftFlows),
            EffectFilterKind::RidgedFlows => Some(Self::RidgedFlows),
            EffectFilterKind::ThinFlows => Some(Self::ThinFlows),
            EffectFilterKind::WideFlows => Some(Self::WideFlows),
            EffectFilterKind::SedimentFlows => Some(Self::SedimentFlows),
            EffectFilterKind::HydraulicSediment => Some(Self::HydraulicSediment),
            _ => None,
        }
    }
}

/// Layered terrain materials for hydraulic sediment transport.
#[derive(Debug, Clone)]
pub struct LayeredMaterials {
    /// Bedrock height (meters). Surface = bedrock + loose sediment.
    pub bedrock: Vec<f32>,
    /// Loose sediment thickness (meters).
    pub loose_sediment: Vec<f32>,
    /// Bedrock hardness K ∈ [0,1].
    pub bedrock_hardness: f32,
    /// Loose-sediment hardness K ∈ [0,1] (typically soft).
    pub sediment_hardness: f32,
}

impl LayeredMaterials {
    pub fn from_height(
        height: &[f32],
        initial_sediment: f32,
        bedrock_k: f32,
        sediment_k: f32,
    ) -> Self {
        let sed = initial_sediment.max(0.0);
        let bedrock = height.iter().map(|h| (h - sed).max(0.0)).collect();
        let loose_sediment = vec![sed; height.len()];
        Self {
            bedrock,
            loose_sediment,
            bedrock_hardness: bedrock_k.clamp(0.0, 1.0),
            sediment_hardness: sediment_k.clamp(0.0, 1.0),
        }
    }

    pub fn surface_at(&self, idx: usize) -> f32 {
        self.bedrock[idx] + self.loose_sediment[idx]
    }

    pub fn sync_surface(&self, height: &mut [f32]) {
        for (i, h) in height.iter_mut().enumerate() {
            *h = self.bedrock[i] + self.loose_sediment[i];
        }
    }
}

/// Full per-cell hydro / sediment state maintained by the core.
#[derive(Debug, Clone)]
pub struct TerrainHydroState {
    pub metrics: HeightfieldMetrics,
    /// Surface height = bedrock + loose sediment (authoring convenience).
    pub height: Vec<f32>,
    pub bedrock: Vec<f32>,
    pub loose_sediment: Vec<f32>,
    pub water_depth: Vec<f32>,
    /// Outflow flux magnitude proxy (Σf) per cell after last step.
    pub water_flux: Vec<f32>,
    /// Approximate velocity magnitude from flux / depth.
    pub water_velocity: Vec<f32>,
    /// Suspended sediment concentration (load / max(water, ε)).
    pub sediment_concentration: Vec<f32>,
    /// Suspended sediment load (meters of height-equivalent).
    pub suspended_sediment: Vec<f32>,
    /// Effective erosion resistance K ∈ [0,1].
    pub erosion_resistance: Vec<f32>,
}

impl TerrainHydroState {
    pub fn from_heightfield(
        input: &Heightfield,
        materials: Option<&LayeredMaterials>,
        initial_water: Option<&MaskField>,
        initial_sediment: Option<&MaskField>,
    ) -> Self {
        let metrics = input.metrics;
        let n = (metrics.width * metrics.height) as usize;
        let height = input.to_dense();
        let (bedrock, loose_sediment, resistance) = if let Some(m) = materials {
            let mut res = vec![m.bedrock_hardness; n];
            for i in 0..n {
                if m.loose_sediment[i] > 1e-5 {
                    res[i] = m.sediment_hardness;
                }
            }
            (m.bedrock.clone(), m.loose_sediment.clone(), res)
        } else {
            (height.clone(), vec![0.0f32; n], vec![0.0f32; n])
        };
        let water_depth = initial_water
            .map(|f| f.data().to_vec())
            .unwrap_or_else(|| vec![0.0f32; n]);
        let suspended_sediment = initial_sediment
            .map(|f| f.data().to_vec())
            .unwrap_or_else(|| vec![0.0f32; n]);
        Self {
            metrics,
            height,
            bedrock,
            loose_sediment,
            water_depth,
            water_flux: vec![0.0f32; n],
            water_velocity: vec![0.0f32; n],
            sediment_concentration: vec![0.0f32; n],
            suspended_sediment,
            erosion_resistance: resistance,
        }
    }

    pub fn surface_heightfield(&self) -> Heightfield {
        Heightfield::from_dense(self.metrics, &self.height)
    }
}

/// Optional diagnostic / downstream aux outputs.
#[derive(Debug, Clone, Default)]
pub struct HydraulicOutputFlags {
    pub erosion_mask: bool,
    pub deposition_mask: bool,
    pub flow: bool,
    pub water: bool,
    pub sediment: bool,
    pub channels: bool,
}

impl HydraulicOutputFlags {
    pub fn all() -> Self {
        Self {
            erosion_mask: true,
            deposition_mask: true,
            flow: true,
            water: true,
            sediment: true,
            channels: true,
        }
    }

    pub fn from_params(p: &HydraulicErosionParams) -> Self {
        Self {
            erosion_mask: p.output_erosion,
            deposition_mask: p.output_deposition,
            flow: p.output_flow,
            water: p.output_water,
            sediment: p.output_sediment,
            channels: p.output_channels,
        }
    }
}

/// Bundled optional outputs from a core solve.
#[derive(Debug, Clone, Default)]
pub struct HydraulicOutputs {
    pub erosion: Option<MaskField>,
    pub deposition: Option<MaskField>,
    pub flow: Option<MaskField>,
    pub water: Option<MaskField>,
    pub sediment: Option<MaskField>,
    pub channels: Option<MaskField>,
    pub rainfall: Option<MaskField>,
    pub bedrock: Option<MaskField>,
    pub loose_sediment: Option<MaskField>,
    pub velocity: Option<MaskField>,
}

/// Mass / stability diagnostics for a solve.
#[derive(Debug, Clone, Copy, Default)]
pub struct MassDiagnostics {
    pub height_delta_sum: f32,
    pub suspended_sum: f32,
    pub eroded_sum: f32,
    pub deposited_sum: f32,
    pub residual: f32,
    pub nan_or_inf_cells: u32,
    pub clamped_negative_cells: u32,
}

impl MassDiagnostics {
    pub fn from_result(input: &Heightfield, result: &HydraulicResult) -> Self {
        let h0 = input.to_dense();
        let h1 = result.height.to_dense();
        let mut nan_or_inf = 0u32;
        let mut clamped = 0u32;
        let mut dh = 0.0f32;
        for (&a, &b) in h1.iter().zip(h0.iter()) {
            if !a.is_finite() {
                nan_or_inf += 1;
            }
            if a < 0.0 {
                clamped += 1;
            }
            dh += a - b;
        }
        let sed: f32 = result.sediment_raw.data().iter().copied().sum();
        let eroded: f32 = result.erosion_raw.data().iter().copied().sum();
        let deposited: f32 = result.deposition_raw.data().iter().copied().sum();
        for &v in result.water_raw.data() {
            if !v.is_finite() {
                nan_or_inf += 1;
            }
        }
        for &v in result.sediment_raw.data() {
            if !v.is_finite() {
                nan_or_inf += 1;
            }
        }
        Self {
            height_delta_sum: dh,
            suspended_sum: sed,
            eroded_sum: eroded,
            deposited_sum: deposited,
            residual: dh + sed,
            nan_or_inf_cells: nan_or_inf,
            clamped_negative_cells: clamped,
        }
    }

    pub fn approximately_conserved(self, eps_scale: f32) -> bool {
        let scale = self.eroded_sum.max(self.deposited_sum).max(1.0);
        self.residual.abs() < scale * eps_scale + 1e-2 && self.nan_or_inf_cells == 0
    }
}

/// CFL-style timestep clamp for the Mei-lite pipe model.
pub fn clamp_timestep_cfl(timestep: f32, dx: f32, max_speed: f32) -> f32 {
    let dx = dx.max(1e-6);
    let vmax = max_speed.max(1e-3);
    let cfl = 0.45 * dx / vmax;
    timestep.clamp(1e-4, cfl.min(1.0))
}

/// Sanitize a dense field in place (NaN/Inf → 0, negatives → 0).
///
/// For non-negative quantities (water depth, sediment load). For terrain height,
/// use [`sanitize_finite`], which preserves genuine sub-zero (underwater) terrain.
pub fn sanitize_field(data: &mut [f32]) -> u32 {
    let mut hits = 0u32;
    for v in data.iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
            hits += 1;
        } else if *v < 0.0 {
            *v = 0.0;
            hits += 1;
        }
    }
    hits
}

/// Sanitize a dense height field in place (NaN/Inf → 0 only).
///
/// Unlike [`sanitize_field`], negative samples are preserved: below-sea-level
/// bathymetry is valid terrain, not garbage to clip.
pub fn sanitize_finite(data: &mut [f32]) -> u32 {
    let mut hits = 0u32;
    for v in data.iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
            hits += 1;
        }
    }
    hits
}

/// Build a proper rainfall field for hydraulic operators.
pub fn generate_rainfall_field(
    metrics: HeightfieldMetrics,
    source: &RainMapSource,
    external: Option<&MaskField>,
    baked_mask: Option<&MaskField>,
) -> MaskField {
    let w = metrics.width as usize;
    let h = metrics.height as usize;
    let n = w * h;
    let data = match source {
        RainMapSource::Constant(v) => vec![(*v).max(0.0); n],
        RainMapSource::Noise {
            seed,
            frequency,
            scale,
            offset,
        } => {
            let params = NoiseParams {
                seed: *seed,
                frequency: (*frequency).max(1e-6),
                amplitude: 1.0,
                octaves: 4,
                lacunarity: 2.0,
                persistence: 0.5,
                ..NoiseParams::default()
            };
            let mut out = vec![0.0f32; n];
            let dx = metrics.dx();
            let dz = metrics.dz();
            for j in 0..h {
                for i in 0..w {
                    let x = i as f32 * dx;
                    let z = j as f32 * dz;
                    let nval = fbm(FractalNoiseType::OpenSimplex, x, z, &params)
                        .mul_add(*scale, *offset)
                        .max(0.0);
                    out[j * w + i] = nval;
                }
            }
            out
        }
        RainMapSource::Painted(_) | RainMapSource::Distribution => baked_mask
            .map(|m| m.data().iter().map(|v| (*v).max(0.0)).collect::<Vec<_>>())
            .unwrap_or_else(|| vec![1.0; n]),
        RainMapSource::External => external
            .map(|m| m.data().iter().map(|v| (*v).max(0.0)).collect::<Vec<_>>())
            .unwrap_or_else(|| vec![1.0; n]),
    };
    MaskField::from_raw(metrics, &data)
}

/// Apply transport-model biases onto a base parameter set.
pub fn apply_transport_model(
    base: &HydraulicErosionParams,
    model: TransportModel,
) -> HydraulicErosionParams {
    let mut p = base.clone();
    p.transport_model = model;
    match model {
        TransportModel::Hydraulic => {}
        TransportModel::SoftFlows => {
            p.erosion = (p.erosion * 0.75).max(0.18);
            p.deposition = (p.deposition * 1.35).min(2.0);
            p.capacity *= 0.9;
            p.rainfall = (p.rainfall * 1.15).max(0.022);
            p.floodplain_bias = p.floodplain_bias.max(0.85);
            p.fan_boost = p.fan_boost.max(0.55);
            p.bank_slip = p.bank_slip.max(0.2).min(0.35);
            p.sediment_softness = p.sediment_softness.max(0.35);
            p.particle_density *= 0.45;
            p.particle_radius = p.particle_radius.max(3);
            p.particle_inertia = (p.particle_inertia * 0.6).min(0.35);
            p.lateral_smoothing = p.lateral_smoothing.max(0.55);
            // Baked into erosion above — keep GPU scale near 1.
            p.incision_bias = 0.75;
        }
        TransportModel::RidgedFlows => {
            p.erosion = (p.erosion * 1.45).min(1.5);
            p.deposition *= 0.55;
            p.capacity *= 1.15;
            p.rainfall = (p.rainfall * 1.1).max(0.022);
            p.floodplain_bias *= 0.35;
            p.bank_slip = (p.bank_slip * 0.4).min(0.25);
            p.particle_density = p.particle_density.max(0.1);
            p.particle_radius = 1.max(p.particle_radius / 2);
            p.particle_inertia = p.particle_inertia.max(0.35);
            p.protection_strength = p.protection_strength.max(0.75);
            p.incision_bias = 1.35;
            p.ridge_protection = p.ridge_protection.max(0.85);
            p.lateral_smoothing *= 0.25;
        }
        TransportModel::ThinFlows => {
            p.erosion = (p.erosion * 1.15).min(1.2);
            p.deposition *= 0.75;
            p.capacity *= 0.7;
            p.floodplain_bias *= 0.2;
            p.fan_boost *= 0.35;
            p.particle_density = p.particle_density.max(0.14);
            p.particle_lifetime = p.particle_lifetime.max(48);
            p.particle_inertia = p.particle_inertia.max(0.55);
            p.particle_radius = 1;
            p.lateral_smoothing = 0.05;
            p.incision_bias = 1.2;
            p.flow_footprint = 0.35;
        }
        TransportModel::WideFlows => {
            p.erosion *= 0.9;
            p.deposition = (p.deposition * 1.25).min(2.0);
            p.capacity = (p.capacity * 1.55).min(0.4);
            p.floodplain_bias = p.floodplain_bias.max(0.7);
            p.fan_boost = p.fan_boost.max(0.8);
            p.particle_density = p.particle_density.max(0.1);
            p.particle_radius = p.particle_radius.max(5);
            p.particle_inertia = p.particle_inertia.clamp(0.15, 0.55);
            p.lateral_smoothing = p.lateral_smoothing.max(0.4);
            p.flow_footprint = 1.6;
            p.incision_bias = 0.75;
        }
        TransportModel::SedimentFlows => {
            p.capacity = (p.capacity * 1.6).min(0.45);
            p.deposition = (p.deposition * 1.55).min(2.0);
            p.erosion *= 0.85;
            p.fan_boost = p.fan_boost.max(1.2);
            p.floodplain_bias = p.floodplain_bias.max(1.0);
            p.sediment_softness = p.sediment_softness.max(0.65);
            p.bank_slip = p.bank_slip.max(0.4);
            p.particle_density = p.particle_density.max(0.09);
            p.particle_radius = p.particle_radius.max(3);
            p.layered_materials = true;
            p.sediment_hardness = p.sediment_hardness.min(0.15);
            p.bedrock_hardness = p.bedrock_hardness.max(0.45);
        }
        TransportModel::HydraulicSediment => {
            p.layered_materials = true;
            p.bedrock_hardness = p.bedrock_hardness.max(0.55);
            p.sediment_hardness = p.sediment_hardness.min(0.12);
            p.initial_sediment_thickness = p.initial_sediment_thickness.max(0.35);
            p.capacity = (p.capacity * 1.25).min(0.35);
            p.deposition = (p.deposition * 1.35).min(2.0);
            p.sediment_softness = p.sediment_softness.max(0.7);
            p.fan_boost = p.fan_boost.max(1.0);
            p.floodplain_bias = p.floodplain_bias.max(0.75);
            p.bank_slip = p.bank_slip.max(0.35);
        }
    }
    p
}

/// Build params for an EffectFilter amount/strength using a transport preset.
pub fn params_from_effect_filter(fp: &EffectFilterParams) -> Option<HydraulicErosionParams> {
    let model = TransportModel::from_effect_filter(fp.kind)?;
    let amount = fp.amount.max(0.25);
    let strength = fp.strength.clamp(0.0, 2.0);
    let iters = ((amount * 4.5 + 8.0) * strength).round() as u32;
    let base = HydraulicErosionParams {
        iterations: iters.clamp(8, 64),
        rainfall: (0.018 + amount * 0.0025).clamp(0.01, 0.08),
        evaporation: 0.012,
        capacity: (0.08 + amount * 0.004).clamp(0.05, 0.25),
        erosion: (0.22 + amount * 0.02).clamp(0.15, 0.85) * strength,
        deposition: (0.28 + amount * 0.015).clamp(0.2, 1.2),
        timestep: 0.2,
        particle_density: match model {
            TransportModel::ThinFlows => 0.12,
            TransportModel::WideFlows => 0.09,
            _ => 0.05,
        },
        particle_radius: match model {
            TransportModel::ThinFlows => 1,
            TransportModel::WideFlows => fp.radius.max(4),
            TransportModel::SoftFlows => fp.radius.max(2),
            _ => 2,
        },
        particle_inertia: match model {
            TransportModel::ThinFlows => 0.6,
            TransportModel::WideFlows => 0.25,
            _ => 0.2,
        },
        flow_threshold: fp.flow_threshold,
        ..HydraulicErosionParams::default()
    };
    Some(apply_transport_model(&base, model))
}

/// Unified hydraulic erosion entry point.
#[derive(Debug, Clone)]
pub struct HydraulicErosionCore {
    pub params: HydraulicErosionParams,
    pub rain: RainMapSource,
    pub outputs: HydraulicOutputFlags,
}

impl Default for HydraulicErosionCore {
    fn default() -> Self {
        Self {
            params: HydraulicErosionParams::default(),
            rain: RainMapSource::default(),
            outputs: HydraulicOutputFlags::all(),
        }
    }
}

impl HydraulicErosionCore {
    pub fn new(params: HydraulicErosionParams) -> Self {
        let rain = RainMapSource::from_mask_source(&params.rainfall_source);
        let outputs = HydraulicOutputFlags::from_params(&params);
        Self {
            params,
            rain,
            outputs,
        }
    }

    pub fn with_model(model: TransportModel) -> Self {
        let params = apply_transport_model(&HydraulicErosionParams::default(), model);
        Self::new(params)
    }

    pub fn from_effect_filter(fp: &EffectFilterParams) -> Option<Self> {
        let params = params_from_effect_filter(fp)?;
        Some(Self::new(params))
    }

    /// Run the shared solver and package optional outputs.
    pub fn simulate(
        &self,
        input: &Heightfield,
        hardness: &MaskField,
        baked_rainfall: Option<&MaskField>,
        protection: Option<&MaskField>,
        initial_water: Option<&MaskField>,
        initial_sediment: Option<&MaskField>,
    ) -> (
        HydraulicResult,
        HydraulicOutputs,
        MassDiagnostics,
        TerrainHydroState,
    ) {
        let mut p = self.params.clone();
        let dx = input.metrics.dx().max(1e-6);
        p.timestep = clamp_timestep_cfl(p.timestep, dx, 4.0);

        let rainfall =
            generate_rainfall_field(input.metrics, &self.rain, baked_rainfall, baked_rainfall);

        let mut protection_field = protection.cloned();
        if p.ridge_protection > 1e-4 {
            let (ridge, _valley) = ridge_valley_likelihood(input, 0.0);
            let strength = p.ridge_protection.clamp(0.0, 1.0);
            let mut combined = MaskField::zeros(input.metrics);
            for j in 0..input.metrics.height {
                for i in 0..input.metrics.width {
                    let base = protection
                        .map(|m| m.get(i, j))
                        .unwrap_or(0.0)
                        .clamp(0.0, 1.0);
                    let r = ridge.get(i, j) * strength;
                    combined.set(i, j, (base + r * (1.0 - base)).clamp(0.0, 1.0));
                }
            }
            p.protection_strength = p.protection_strength.max(strength);
            protection_field = Some(combined);
        }

        let layered = if p.layered_materials {
            Some(LayeredMaterials::from_height(
                &input.to_dense(),
                p.initial_sediment_thickness,
                p.bedrock_hardness,
                p.sediment_hardness,
            ))
        } else {
            None
        };

        let mut hardness_eff = hardness.clone();
        if let Some(ref layers) = layered {
            for j in 0..input.metrics.height {
                for i in 0..input.metrics.width {
                    let idx = j as usize * input.metrics.width as usize + i as usize;
                    let k = if layers.loose_sediment[idx] > 1e-5 {
                        layers.sediment_hardness
                    } else {
                        layers.bedrock_hardness.max(hardness.get(i, j))
                    };
                    hardness_eff.set(i, j, k.clamp(0.0, 1.0));
                }
            }
        }

        // Transport-model knobs are already baked into erosion/deposition/bank_slip.
        // Keep incision_bias / flow_footprint as relative hints for GPU uniforms only.
        p.bank_slip = (p.bank_slip + p.lateral_smoothing * 0.35).clamp(0.0, 1.0);

        let mut result = hydraulic_erode_with_fields(
            input,
            &p,
            &hardness_eff,
            Some(&rainfall),
            protection_field.as_ref(),
            initial_water,
            initial_sediment,
        );

        if p.particle_density > 0.0 {
            result = apply_particle_erosion(result, &p, &hardness_eff);
        }

        // Sanitize oracle fields. Height keeps valid sub-zero bathymetry;
        // only water/sediment (below) are clamped non-negative.
        let mut height = result.height.to_dense();
        sanitize_finite(&mut height);
        result.height = Heightfield::from_dense(input.metrics, &height);
        sanitize_field(result.water_raw.data_mut());
        sanitize_field(result.sediment_raw.data_mut());

        let diagnostics = MassDiagnostics::from_result(input, &result);

        let mut state = TerrainHydroState::from_heightfield(
            &result.height,
            layered.as_ref(),
            Some(&result.water_raw),
            Some(&result.sediment_raw),
        );
        // Flux / velocity proxies from wetness + water depth.
        let wet = result.wetness.data();
        for i in 0..state.water_flux.len() {
            let w = result.water_raw.data()[i].max(0.0);
            let flux = wet[i].max(0.0);
            state.water_flux[i] = flux;
            state.water_velocity[i] = flux / w.max(1e-4);
            state.suspended_sediment[i] = result.sediment_raw.data()[i].max(0.0);
            state.sediment_concentration[i] = state.suspended_sediment[i] / w.max(1e-4);
            if let Some(ref layers) = layered {
                let dep = result.deposition_raw.data()[i].max(0.0);
                let eroded = result.erosion_raw.data()[i].max(0.0);
                let loose = (layers.loose_sediment[i] + dep - eroded * 0.35).max(0.0);
                state.loose_sediment[i] = loose;
                state.bedrock[i] = (state.height[i] - loose).max(0.0);
                state.erosion_resistance[i] = if loose > 1e-5 {
                    p.sediment_hardness
                } else {
                    p.bedrock_hardness
                };
            }
        }

        let outputs = self.package_outputs(&result, &state, &rainfall);
        (result, outputs, diagnostics, state)
    }

    /// Convenience: height-only apply for EffectFilter morphology path.
    pub fn apply_height(&self, input: &Heightfield, hardness: Option<&MaskField>) -> Heightfield {
        let k = hardness.cloned().unwrap_or_else(|| {
            MaskField::filled(input.metrics, self.params.hardness.clamp(0.0, 1.0))
        });
        let (result, _, _, _) = self.simulate(input, &k, None, None, None, None);
        result.height
    }

    fn package_outputs(
        &self,
        result: &HydraulicResult,
        state: &TerrainHydroState,
        rainfall: &MaskField,
    ) -> HydraulicOutputs {
        let flags = &self.outputs;
        let mut out = HydraulicOutputs {
            rainfall: Some(rainfall.clone()),
            ..HydraulicOutputs::default()
        };
        if flags.erosion_mask {
            out.erosion = Some(result.erosion.clone());
        }
        if flags.deposition_mask {
            out.deposition = Some(result.deposition.clone());
        }
        if flags.flow {
            out.flow = Some(normalize_dense(state.metrics, &state.water_flux));
        }
        if flags.water {
            out.water = Some(result.water_raw.clone());
            out.velocity = Some(normalize_dense(state.metrics, &state.water_velocity));
        }
        if flags.sediment {
            out.sediment = Some(result.sediment.clone());
            out.loose_sediment = Some(MaskField::from_raw(state.metrics, &state.loose_sediment));
            out.bedrock = Some(MaskField::from_raw(state.metrics, &state.bedrock));
        }
        if flags.channels {
            out.channels = Some(channel_mask_from_flux(state.metrics, &state.water_flux));
        }
        out
    }
}

fn normalize_dense(metrics: HeightfieldMetrics, data: &[f32]) -> MaskField {
    let max_v = data.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
    let norm: Vec<f32> = data.iter().map(|v| (*v / max_v).clamp(0.0, 1.0)).collect();
    MaskField::from_raw(metrics, &norm)
}

fn channel_mask_from_flux(metrics: HeightfieldMetrics, flux: &[f32]) -> MaskField {
    let max_v = flux.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
    let mut out = vec![0.0f32; flux.len()];
    for (i, &f) in flux.iter().enumerate() {
        let t = f / max_v;
        out[i] = if t > 0.35 {
            ((t - 0.35) / 0.65).clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
    MaskField::from_raw(metrics, &out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heightfield::HeightfieldMetrics;

    fn hill(res: u32) -> Heightfield {
        let m = HeightfieldMetrics::new(res, res, res as f32 * 8.0, res as f32 * 8.0);
        let mut hf = Heightfield::zeros(m);
        for j in 0..res {
            for i in 0..res {
                let dx = i as f32 - res as f32 * 0.5;
                let dz = j as f32 - res as f32 * 0.5;
                // Dome: water drains off the peak so hydraulic incision lowers the summit.
                hf.set(i, j, 120.0 - (dx * dx + dz * dz).sqrt() * 0.55);
            }
        }
        hf
    }

    #[test]
    fn soft_flows_preset_carves_basin() {
        let hf = hill(40);
        let core = HydraulicErosionCore::with_model(TransportModel::SoftFlows);
        let out = core.apply_height(&hf, None);
        let max_dh = hf
            .to_dense()
            .iter()
            .zip(out.to_dense().iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_dh > 0.1,
            "soft flows should reshape drainage, max_dh={max_dh}"
        );
        assert!(out.get(2, 2) <= hf.get(2, 2) + 0.75);
    }

    #[test]
    fn hydraulic_preserves_sub_zero_bathymetry() {
        // Island dome ringed by a deep sub-zero (underwater) shelf. Hydraulic
        // erosion sanitizes non-finite samples but must keep negative terrain;
        // before this fix `sanitize_field` clamped the whole seabed to 0.
        let m = HeightfieldMetrics::new(40, 40, 320.0, 320.0);
        let mut hf = Heightfield::filled(m, -200.0);
        for j in 0..40 {
            for i in 0..40 {
                let r = ((i as f32 - 20.0).powi(2) + (j as f32 - 20.0).powi(2)).sqrt();
                if r < 8.0 {
                    hf.set(i, j, 120.0 - r * 4.0); // land above water
                }
            }
        }
        let core = HydraulicErosionCore::new(HydraulicErosionParams::default());
        let out = core.apply_height(&hf, None);
        let min = out.to_dense().iter().copied().fold(f32::INFINITY, f32::min);
        assert!(
            min < -150.0,
            "hydraulic erosion flattened bathymetry: min={min}"
        );
    }

    #[test]
    fn ridged_does_not_raise_divides_artificially() {
        let hf = hill(40);
        let core = HydraulicErosionCore::with_model(TransportModel::RidgedFlows);
        let out = core.apply_height(&hf, None);
        let max_dh = hf
            .to_dense()
            .iter()
            .zip(out.to_dense().iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_dh > 0.1, "ridged flows should incise, max_dh={max_dh}");
        // Allow toe deposition; forbid morphology-style fake ridge boosts.
        assert!(out.get(2, 2) <= hf.get(2, 2) + 0.75);
    }

    #[test]
    fn rainfall_noise_field_is_non_uniform() {
        let m = HeightfieldMetrics::new(32, 32, 320.0, 320.0);
        let field = generate_rainfall_field(
            m,
            &RainMapSource::Noise {
                seed: 7,
                frequency: 0.02,
                scale: 1.0,
                offset: 0.2,
            },
            None,
            None,
        );
        let min = field.data().iter().cloned().fold(f32::INFINITY, f32::min);
        let max = field.data().iter().cloned().fold(0.0f32, f32::max);
        assert!(max > min + 1e-3, "noise rainfall should vary");
        assert!(min >= 0.0);
    }

    #[test]
    fn mass_diagnostics_finite() {
        let hf = hill(32);
        let core = HydraulicErosionCore::with_model(TransportModel::Hydraulic);
        let k = MaskField::filled(hf.metrics, 0.0);
        let (_, _, diag, _) = core.simulate(&hf, &k, None, None, None, None);
        assert_eq!(diag.nan_or_inf_cells, 0);
    }
}
