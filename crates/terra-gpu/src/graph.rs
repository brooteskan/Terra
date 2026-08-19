//! Compile a layer stack into per-layer executable GPU plans.
//!
//! Artist UI remains layer-based. `compile_gpu_graph` produces one plan slot per
//! flattened layer — the executable kernel, its dirty-region policy, and the
//! per-iteration halo the kernel actually reaches — plus the CPU-resume boundary
//! (`cpu_from`). `GpuTerrainEngine::evaluate` indexes this plan directly and never
//! re-derives kernels mid-walk, so planning has a single authority.

use crate::effect_filter::{effect_filter_gpu_spec, EffectFilterGpuScope};
use terra_core::fields::FieldId;
use terra_core::layer::{
    BlendMode, DuneParams, EffectFilterKind, FractalNoiseType, IslandArchetype, IslandParams,
    Layer, LayerKind, LayerStack, MountainParams, MultiScaleAmplifyParams, PathParams,
    PolygonHeightParams, ProceduralGenerator, ProceduralShapeParams, RiverCarveParams,
    SculptStrokeKind, StreamPowerParams, TransportModel, UpliftParams,
};
use terra_core::mask::{MaskAsset, MaskCombine, MaskSource};

/// Max per-iteration blur radius the Blur kernel executes (`shaders/blur.wgsl`).
pub const BLUR_MAX_RADIUS: u32 = 8;
/// Max per-iteration reach the EffectFilter kernel executes.
pub const EFFECT_FILTER_MAX_RADIUS: u32 = 16;
/// Largest full-quality RiverCarve bank radius represented by the gather shader.
/// Wider authored banks remain on the CPU oracle instead of being silently clipped.
pub const RIVER_CARVE_MAX_RADIUS: u32 = 40;
/// Upper bound executed by the interactive noise shader. Larger authored
/// fractals remain on the CPU oracle rather than silently dropping octaves.
const NOISE_MAX_OCTAVES: u32 = 12;

/// Concrete executable pipeline selected by the support planner. The engine
/// consumes this value, so a configuration cannot be advertised merely because
/// its broad layer family appears in a second, independent match table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuKernel {
    Fill,
    Ramp,
    Noise,
    Shape,
    Sculpt,
    SculptStrokes,
    Path,
    PolygonHeight,
    ProceduralShape,
    HeightmapSample,
    Blur,
    EffectFilter,
    Terrace,
    Thermal,
    Hydraulic,
    RiverCarve,
    StreamPower,
    MultiScaleAmplify,
}

impl GpuKernel {
    pub fn matches_layer_kind(self, kind: &LayerKind) -> bool {
        matches!(
            (self, kind),
            (Self::Fill, LayerKind::Flat(_))
                | (Self::Ramp, LayerKind::Ramp(_))
                | (Self::Sculpt, LayerKind::SculptBase(_))
                | (Self::SculptStrokes, LayerKind::SculptStrokes(_))
                | (Self::Path, LayerKind::Path(_))
                | (Self::PolygonHeight, LayerKind::PolygonHeight(_))
                | (Self::ProceduralShape, LayerKind::ProceduralShape(_))
                | (
                    Self::HeightmapSample,
                    LayerKind::ImportHeightmap(_) | LayerKind::Stamp2d(_)
                )
                | (
                    Self::Noise,
                    LayerKind::NoiseValue(_)
                        | LayerKind::NoisePerlin(_)
                        | LayerKind::Fbm(_)
                        | LayerKind::Ridged(_)
                        | LayerKind::DomainWarp(_)
                )
                | (
                    Self::Shape,
                    LayerKind::Mountains(_)
                        | LayerKind::Dunes(_)
                        | LayerKind::Canyons(_)
                        | LayerKind::Mesa(_)
                        | LayerKind::Volcano(_)
                        | LayerKind::Island(_)
                        | LayerKind::Plateau(_)
                        | LayerKind::Uplift(_)
                )
                | (Self::Blur, LayerKind::Blur(_))
                | (Self::EffectFilter, LayerKind::EffectFilter(_))
                | (Self::Terrace, LayerKind::Terrace(_))
                | (Self::Thermal, LayerKind::ThermalErosion(_))
                | (Self::Hydraulic, LayerKind::HydraulicErosion(_))
                | (Self::RiverCarve, LayerKind::RiverCarve(_))
                | (Self::StreamPower, LayerKind::StreamPowerErosion(_))
                | (Self::MultiScaleAmplify, LayerKind::MultiScaleAmplify(_))
        )
    }
}

/// Dirty policy for a pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDirtyPolicy {
    /// Local ops — honor SampleRect + halo.
    Local,
    /// Basin / global coupling — full field (or level-step coarse).
    FullField,
}

/// Executable plan for one GPU-supported layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuLayerPlan {
    /// Concrete pipeline the engine dispatches for this layer.
    pub kernel: GpuKernel,
    /// Whether the layer couples the whole field or honors a local dirty rect.
    pub dirty_policy: GpuDirtyPolicy,
    /// Per-iteration kernel reach in texels, matching the executed pipeline's
    /// clamp; the engine multiplies this by the executed iteration count to size
    /// the dirty-region halo.
    pub halo_texels: u32,
}

/// Compiled interactive GPU plan: one slot per flattened layer plus the CPU
/// resume boundary.
#[derive(Debug, Clone, Default)]
pub struct GpuComputeGraph {
    /// One entry per `stack.flatten_layers()` index. `Some` = this layer runs on
    /// the GPU with the given plan; `None` = disabled or not GPU-supported.
    pub plans: Vec<Option<GpuLayerPlan>>,
    /// First flat index that is enabled but not GPU-supported (None = fully GPU).
    pub cpu_from: Option<usize>,
}

impl GpuComputeGraph {
    pub fn fully_gpu(&self) -> bool {
        self.cpu_from.is_none()
    }
}

fn gpu_mask_supported(layer: &Layer, assets: &[MaskAsset]) -> bool {
    if !layer.common.masks.nodes.is_empty() {
        return false;
    }
    let [entry] = layer.common.masks.entries.as_slice() else {
        return layer.common.masks.entries.is_empty();
    };
    if entry.combine != MaskCombine::Multiply {
        return false;
    }
    assets
        .iter()
        .find(|asset| asset.id == entry.mask.id)
        .is_some_and(|asset| {
            asset.ops.is_empty()
                && matches!(
                    asset.source,
                    MaskSource::Constant(_) | MaskSource::Height { .. } | MaskSource::Slope { .. }
                )
        })
}

/// Exact mode IDs implemented by `shaders/blend.wgsl`.
///
/// Returning `None` is deliberate: unsupported equations must select the CPU
/// oracle instead of being approximated by a different GPU blend operation.
pub(crate) fn gpu_blend_mode(mode: BlendMode) -> Option<u32> {
    match mode {
        BlendMode::Normal | BlendMode::Replace | BlendMode::Interpolate => Some(0),
        BlendMode::Add => Some(1),
        BlendMode::Subtract => Some(2),
        BlendMode::Multiply => Some(3),
        BlendMode::Min => Some(4),
        BlendMode::Max => Some(5),
        BlendMode::Overlay => Some(6),
        BlendMode::HeightBlend
        | BlendMode::SmoothMaximum
        | BlendMode::SmoothMinimum
        | BlendMode::SmoothUnion
        | BlendMode::SmoothSubtraction => None,
    }
}

fn inplace_composite_supported(layer: &Layer) -> bool {
    layer.common.opacity == 1.0
        && layer.common.masks.is_empty()
        && matches!(
            layer.common.blend,
            BlendMode::Normal | BlendMode::Replace | BlendMode::Interpolate
        )
}

fn seed_supported(seed: u64) -> bool {
    seed <= u64::from(u32::MAX)
}

/// The shader carries a 32-bit seed. CPU fractals derive a fresh `u64` seed for
/// every octave before canonicalizing it, so merely checking the authored base
/// seed would still let a near-`u32::MAX` stream wrap differently on the GPU.
fn seed_stream_supported(seed: u64, octaves: u32, stride: u64) -> bool {
    if octaves.max(1) > NOISE_MAX_OCTAVES {
        return false;
    }
    let last_octave = u64::from(octaves.max(1) - 1);
    last_octave
        .checked_mul(stride)
        .and_then(|offset| seed.checked_add(offset))
        .is_some_and(seed_supported)
}

fn mountain_seed_streams_supported(p: &MountainParams) -> bool {
    seed_stream_supported(p.base.seed, p.base.octaves, 9173)
        && seed_stream_supported(p.base.seed, p.base.octaves.clamp(2, 4), 9173)
        && seed_stream_supported(p.base.seed ^ 0xC0DE, 4, 9173)
}

fn uplift_seed_streams_supported(p: &UpliftParams) -> bool {
    seed_stream_supported(p.seed, 3, 9173)
        && seed_stream_supported(p.seed ^ 0xC0FFEE, p.detail_octaves.clamp(1, 8), 1013)
}

fn island_seed_streams_supported(p: &IslandParams) -> bool {
    seed_supported(p.seed)
        && (p.archetype == IslandArchetype::VolcanicHighIsland
            || (seed_stream_supported(p.seed ^ 0x51AD_E771, 5, 9173)
                && seed_stream_supported(p.seed ^ 0xD37A_11ED, 3, 1013)))
}

/// The current preview shader is the bounded procedural dune approximation. It
/// intentionally represents the shipped transport controls; authoring a
/// different aeolian relaxation still resumes at the CPU oracle.
fn dune_preview_config_supported(p: &DuneParams) -> bool {
    let defaults = DuneParams::default();
    (2..=4).contains(&p.base.octaves)
        && seed_stream_supported(p.base.seed, p.base.octaves, 1013)
        && p.wind_strength == defaults.wind_strength
        && p.sand_supply == defaults.sand_supply
        && p.transport_length == defaults.transport_length
        && p.avalanche_angle == defaults.avalanche_angle
        && p.iterations == defaults.iterations
}

fn fractal_noise_supported(noise: FractalNoiseType) -> bool {
    matches!(noise, FractalNoiseType::Value | FractalNoiseType::Perlin)
}

fn path_config_supported(p: &PathParams) -> bool {
    let scalars_finite = p.width.is_finite()
        && p.falloff.is_finite()
        && p.noise_strength.is_finite()
        && p.noise_scale.is_finite()
        && p.height_offset.is_finite()
        && p.profile.is_finite();
    let nodes_finite = p.nodes.iter().all(|node| {
        node.u.is_finite()
            && node.v.is_finite()
            && node.height.is_finite()
            && node.width.is_finite()
    });
    scalars_finite && nodes_finite && (p.noise_strength.abs() <= 1.0e-5 || seed_supported(p.seed))
}

fn polygon_height_config_supported(p: &PolygonHeightParams) -> bool {
    p.height.is_finite()
        && p.falloff.is_finite()
        && p.points
            .iter()
            .all(|point| point[0].is_finite() && point[1].is_finite())
}

fn procedural_shape_config_supported(p: &ProceduralShapeParams) -> bool {
    match p.generator {
        ProceduralGenerator::Mountain => mountain_seed_streams_supported(&p.mountain),
        ProceduralGenerator::Hills | ProceduralGenerator::Plateau => {
            fractal_noise_supported(p.hills.noise)
                && seed_stream_supported(p.hills.base.seed, p.hills.base.octaves, 1013)
        }
        ProceduralGenerator::Mesa => seed_supported(p.mesa.seed),
        ProceduralGenerator::Volcano => seed_supported(p.volcano.seed),
        ProceduralGenerator::Canyon => seed_supported(p.canyon.seed),
        ProceduralGenerator::Noise => seed_stream_supported(p.noise.seed, p.noise.octaves, 1013),
        ProceduralGenerator::Crater => {
            p.crater.kind == EffectFilterKind::Crater
                && effect_filter_gpu_spec(&p.crater).is_some_and(|spec| {
                    spec.scope == EffectFilterGpuScope::LocalPointwise && !spec.needs_height_range
                })
        }
        // The CPU Dunes generator evolves a globally coupled aeolian field.
        ProceduralGenerator::Dunes => false,
    }
}

/// Whether the GPU stamp path can reproduce a single sculpt-stroke kind. Most
/// supported kinds are pure per-sample maps of the running height (plus the
/// distance-to-polyline SDF); `Smooth`, `Pinch`, and `Coastline` additionally read a
/// clamped 3x3 of the layer input (`src`), which the kernel samples directly — `Pinch`
/// is `Smooth`'s pull at a 1.25 overdrive, `Coastline` a lower-and-blend toward that
/// mean under a weight gate (#114, #115, #116). `Flatten` needs a per-stroke
/// footprint-mean reduction over the running field; the reduce/resolve passes now
/// precompute that scalar and the stamp path segments the run around it, so it too
/// previews on the GPU (#117). The aux-only kinds (`Uplift` / `Hardness` /
/// `Sediment` / `Protect` / `EncourageErosion`) are supported because their height
/// contribution is a per-sample function even though the GPU preview drops the aux
/// they would publish — the aux gate in `compile_gpu_graph` handles any downstream
/// consumer. Every kind is now GPU-previewable; the hook is retained so a future
/// kind the stamp path cannot reproduce can be excluded here (and localised in the
/// CPU oracle to match).
fn stroke_kind_gpu_supported(_kind: SculptStrokeKind) -> bool {
    true
}

fn thermal_config_supported(p: &terra_core::layer::ThermalErosionParams) -> bool {
    !p.layered_materials
        && p.weathering_rate == 0.0
        && matches!(p.hardness_source, MaskSource::None)
        && p.level_count == 0
        && p.start_level == 0
        && p.level_step_strength == 1.0
        && p.level_step_curve.is_empty()
}

fn hydraulic_config_supported(p: &terra_core::layer::HydraulicErosionParams) -> bool {
    !p.layered_materials
        && p.particle_density == 0.0
        && matches!(p.transport_model, TransportModel::Hydraulic)
        && matches!(p.rainfall_source, MaskSource::None)
        && matches!(p.protection_source, MaskSource::None)
        && matches!(p.hardness_source, MaskSource::None)
        && p.fan_boost == 0.0
        && p.floodplain_bias == 0.0
        && p.bank_slip == 0.0
        && p.sediment_softness == 0.0
        && p.level_count == 0
        && p.start_level == 0
        && p.level_step_strength == 1.0
        && p.level_step_curve.is_empty()
}

/// The current RiverCarve preview is height-only and uses bounded iterative D8
/// accumulation. D-infinity is an explicitly parity-bounded preview approximation,
/// but effective guide masks and banks wider than the gather kernel can represent
/// must stay on the CPU oracle.
fn river_carve_config_supported(p: &RiverCarveParams) -> bool {
    let guide_is_inert = matches!(p.guide, MaskSource::None) || p.guide_boost.max(0.0) <= 1.0e-6;
    let max_bank_radius = p.width.max(1.0) * 4.0 * (1.0 + p.bank_smooth.max(0.0) * 0.75);
    p.accumulation_threshold.is_finite()
        && p.accumulation_threshold >= 1.0e-3
        && p.depth.is_finite()
        && p.width.is_finite()
        && p.bank_smooth.is_finite()
        && p.guide_boost.is_finite()
        && guide_is_inert
        && max_bank_radius <= RIVER_CARVE_MAX_RADIUS as f32
}

/// Height-only stream-power preview. Drainage is the shared bounded iterative
/// D8 approximation; D-infinity-authored configurations use that approximation
/// under their own parity contract. Features requiring Priority-Flood, aux-driven
/// hardness, dendritic preprocessing, or authored multilevel controls stay CPU-only.
fn stream_power_config_supported(p: &StreamPowerParams) -> bool {
    p.k.is_finite()
        && p.k >= 0.0
        && p.m.is_finite()
        && p.m >= 0.0
        && p.n.is_finite()
        && p.n >= 0.0
        && p.dt.is_finite()
        && p.dt >= 0.0
        && p.uplift_rate.is_finite()
        && p.base_level.is_finite()
        && p.hardness.is_finite()
        && p.dendritic_seed.is_finite()
        && p.dendritic_seed <= 1.0e-6
        && p.stream_threshold.is_finite()
        && !p.refill_each_iter
        && matches!(p.hardness_source, MaskSource::None)
        && p.level_count == 0
        && p.start_level == 0
        && p.level_step_strength == 1.0
        && p.level_step_curve.is_empty()
}

/// Height-only multi-scale preview. The executor mirrors the CPU coarse-to-fine
/// schedule but currently carries only uniform hardness and ridge-lock values;
/// field-sourced masks and inherited hardness remain on the CPU oracle.
fn multi_scale_amplify_config_supported(p: &MultiScaleAmplifyParams) -> bool {
    let hardness_supported = match p.hardness_source {
        MaskSource::None => true,
        MaskSource::Constant(value) => value.is_finite(),
        _ => false,
    };
    let ridge_lock_supported = match p.ridge_lock {
        MaskSource::None => true,
        MaskSource::Constant(value) => value.is_finite(),
        _ => false,
    };
    hardness_supported
        && ridge_lock_supported
        && p.thermal_strength.is_finite()
        && p.talus_angle_deg.is_finite()
        && p.spe_strength.is_finite()
        && p.deposition_strength.is_finite()
        && p.detail_boost.is_finite()
        && p.hardness.is_finite()
        && p.lock_strength.is_finite()
}

fn gpu_plan_for_layer(layer: &Layer, mask_assets: &[MaskAsset]) -> Option<GpuLayerPlan> {
    use LayerKind::*;
    if !gpu_mask_supported(layer, mask_assets) {
        return None;
    }
    let (kernel, dirty_policy, halo_texels) = match &layer.kind {
        Flat(_) if gpu_blend_mode(layer.common.blend).is_some() => {
            (GpuKernel::Fill, GpuDirtyPolicy::Local, 0)
        }
        Ramp(_) if gpu_blend_mode(layer.common.blend).is_some() => {
            (GpuKernel::Ramp, GpuDirtyPolicy::Local, 0)
        }
        SculptBase(_) if gpu_blend_mode(layer.common.blend).is_some() => {
            (GpuKernel::Sculpt, GpuDirtyPolicy::Local, 0)
        }
        // Height preview for the supported stroke kinds. A Smooth, Pinch, or Coastline
        // stroke reads a 3x3 of the layer input, so an upstream edit reaches one texel
        // further through the stamp; a non-zero reconcile reads a 3x3 of the stamped
        // result and adds its own texel. The two compose (the base read feeds
        // reconcile's stamped read), so the plan halo is their sum — kept in agreement
        // with the CPU `SculptStrokes` intrinsic reach. Flatten adds no per-texel
        // neighbourhood read (its footprint mean is a precomputed scalar), so it does
        // not widen the stamp halo. Flatten stays tile-scoped like the CPU (its #110
        // footprint fixpoint keeps a self-edit recompute bit-exact), so it takes the
        // same `Local` policy — not `FullField`. The aux this layer would publish is
        // dropped here; `compile_gpu_graph` demotes the plan when a downstream layer
        // consumes it.
        SculptStrokes(p)
            if gpu_blend_mode(layer.common.blend).is_some()
                && p.strokes
                    .iter()
                    .filter(|s| s.enabled)
                    .all(|stroke| stroke_kind_gpu_supported(stroke.kind)) =>
        {
            let reads_base_neighborhood = p.strokes.iter().any(|stroke| {
                stroke.enabled
                    && matches!(
                        stroke.kind,
                        SculptStrokeKind::Smooth
                            | SculptStrokeKind::Pinch
                            | SculptStrokeKind::Coastline
                    )
            });
            let halo = u32::from(reads_base_neighborhood) + u32::from(p.reconcile > 0.0);
            (GpuKernel::SculptStrokes, GpuDirtyPolicy::Local, halo)
        }
        SculptStrokes(_) => return None,
        Path(p) if path_config_supported(p) && gpu_blend_mode(layer.common.blend).is_some() => {
            (GpuKernel::Path, GpuDirtyPolicy::Local, 0)
        }
        PolygonHeight(p)
            if polygon_height_config_supported(p)
                && gpu_blend_mode(layer.common.blend).is_some() =>
        {
            (GpuKernel::PolygonHeight, GpuDirtyPolicy::Local, 0)
        }
        ProceduralShape(p)
            if procedural_shape_config_supported(p)
                && gpu_blend_mode(layer.common.blend).is_some() =>
        {
            (GpuKernel::ProceduralShape, GpuDirtyPolicy::Local, 0)
        }
        ImportHeightmap(p)
            if p.height_scale.is_finite()
                && p.height_offset.is_finite()
                && gpu_blend_mode(layer.common.blend).is_some() =>
        {
            (GpuKernel::HeightmapSample, GpuDirtyPolicy::Local, 0)
        }
        Stamp2d(p)
            if p.heightmap.height_scale.is_finite()
                && p.heightmap.height_offset.is_finite()
                && layer.common.shape_transform.as_ref().is_none_or(|t| {
                    t.offset_x.is_finite()
                        && t.offset_z.is_finite()
                        && t.scale.is_finite()
                        && t.rotation_deg.is_finite()
                        && t.blend_size.is_finite()
                        && t.blend_roundness.is_finite()
                })
                && gpu_blend_mode(layer.common.blend).is_some() =>
        {
            (GpuKernel::HeightmapSample, GpuDirtyPolicy::Local, 0)
        }
        Path(_) | PolygonHeight(_) | ProceduralShape(_) | ImportHeightmap(_) | Stamp2d(_) => {
            return None;
        }
        NoiseValue(p) if seed_supported(p.seed) && gpu_blend_mode(layer.common.blend).is_some() => {
            (GpuKernel::Noise, GpuDirtyPolicy::Local, 2)
        }
        NoisePerlin(p)
            if seed_stream_supported(p.seed, p.octaves, 1013)
                && gpu_blend_mode(layer.common.blend).is_some() =>
        {
            (GpuKernel::Noise, GpuDirtyPolicy::Local, 2)
        }
        Fbm(p)
            if fractal_noise_supported(p.noise)
                && seed_stream_supported(p.base.seed, p.base.octaves, 1013)
                && gpu_blend_mode(layer.common.blend).is_some() =>
        {
            (GpuKernel::Noise, GpuDirtyPolicy::Local, 2)
        }
        Ridged(p)
            if fractal_noise_supported(p.noise)
                && seed_stream_supported(p.base.seed, p.base.octaves, 9173)
                && gpu_blend_mode(layer.common.blend).is_some() =>
        {
            (GpuKernel::Noise, GpuDirtyPolicy::Local, 2)
        }
        DomainWarp(p)
            if seed_stream_supported(p.base.seed, p.base.octaves, 1013)
                && seed_stream_supported(p.base.seed, 2, 1)
                && gpu_blend_mode(layer.common.blend).is_some() =>
        {
            // The displacement samples procedural noise, not the entering height;
            // a bounded upstream edit still propagates only through same-texel blend.
            (GpuKernel::Noise, GpuDirtyPolicy::Local, 2)
        }
        Mountains(p)
            if mountain_seed_streams_supported(p)
                && gpu_blend_mode(layer.common.blend).is_some() =>
        {
            (GpuKernel::Shape, GpuDirtyPolicy::Local, 2)
        }
        Dunes(p)
            if dune_preview_config_supported(p) && gpu_blend_mode(layer.common.blend).is_some() =>
        {
            (GpuKernel::Shape, GpuDirtyPolicy::Local, 2)
        }
        Canyons(p) if seed_supported(p.seed) && gpu_blend_mode(layer.common.blend).is_some() => {
            (GpuKernel::Shape, GpuDirtyPolicy::Local, 2)
        }
        Mesa(p) if seed_supported(p.seed) && gpu_blend_mode(layer.common.blend).is_some() => {
            (GpuKernel::Shape, GpuDirtyPolicy::Local, 2)
        }
        Volcano(p) if seed_supported(p.seed) && gpu_blend_mode(layer.common.blend).is_some() => {
            (GpuKernel::Shape, GpuDirtyPolicy::Local, 2)
        }
        Uplift(p)
            if uplift_seed_streams_supported(p) && gpu_blend_mode(layer.common.blend).is_some() =>
        {
            (GpuKernel::Shape, GpuDirtyPolicy::Local, 2)
        }
        Plateau(_) if gpu_blend_mode(layer.common.blend).is_some() => {
            (GpuKernel::Shape, GpuDirtyPolicy::Local, 0)
        }
        Island(p)
            if island_seed_streams_supported(p) && gpu_blend_mode(layer.common.blend).is_some() =>
        {
            (GpuKernel::Shape, GpuDirtyPolicy::Local, 2)
        }
        Flat(_) | Ramp(_) | NoiseValue(_) | NoisePerlin(_) | Fbm(_) | Ridged(_) | Mountains(_)
        | Dunes(_) | Canyons(_) | DomainWarp(_) | SculptBase(_) | Mesa(_) | Volcano(_)
        | Island(_) | Plateau(_) | Uplift(_) => return None,
        Blur(p) if inplace_composite_supported(layer) => (
            GpuKernel::Blur,
            GpuDirtyPolicy::Local,
            p.radius.clamp(1, BLUR_MAX_RADIUS),
        ),
        EffectFilter(p) if inplace_composite_supported(layer) => {
            let spec = effect_filter_gpu_spec(p)?;
            let (dirty_policy, halo_texels) = match spec.scope {
                EffectFilterGpuScope::LocalPointwise => (GpuDirtyPolicy::Local, 0),
                EffectFilterGpuScope::LocalExpanding { halo_per_pass } => {
                    (GpuDirtyPolicy::Local, halo_per_pass)
                }
                EffectFilterGpuScope::FullField => (GpuDirtyPolicy::FullField, 0),
            };
            (GpuKernel::EffectFilter, dirty_policy, halo_texels)
        }
        Blur(_) | EffectFilter(_) => return None,
        Terrace(_) if inplace_composite_supported(layer) => {
            (GpuKernel::Terrace, GpuDirtyPolicy::Local, 4)
        }
        Terrace(_) => return None,
        ThermalErosion(p) if thermal_config_supported(p) && inplace_composite_supported(layer) => {
            (GpuKernel::Thermal, GpuDirtyPolicy::FullField, 0)
        }
        HydraulicErosion(p)
            if hydraulic_config_supported(p) && inplace_composite_supported(layer) =>
        {
            (GpuKernel::Hydraulic, GpuDirtyPolicy::FullField, 0)
        }
        RiverCarve(p) if river_carve_config_supported(p) && inplace_composite_supported(layer) => {
            (GpuKernel::RiverCarve, GpuDirtyPolicy::FullField, 0)
        }
        StreamPowerErosion(p)
            if stream_power_config_supported(p) && inplace_composite_supported(layer) =>
        {
            (GpuKernel::StreamPower, GpuDirtyPolicy::FullField, 0)
        }
        MultiScaleAmplify(p)
            if multi_scale_amplify_config_supported(p) && inplace_composite_supported(layer) =>
        {
            (GpuKernel::MultiScaleAmplify, GpuDirtyPolicy::FullField, 0)
        }
        ThermalErosion(_)
        | HydraulicErosion(_)
        | RiverCarve(_)
        | StreamPowerErosion(_)
        | MultiScaleAmplify(_) => {
            return None;
        }
        // These CPU operations either modify height without a GPU kernel or publish
        // observable auxiliary fields that the GPU preview cannot currently produce.
        Coastal(_) | Materials(_) | Biomes(_) | Vegetation(_) => return None,
        _ => return None,
    };
    Some(GpuLayerPlan {
        kernel,
        dirty_policy,
        halo_texels,
    })
}

/// Whether this complete layer configuration can run on the GPU preview path.
pub fn layer_gpu_supported(layer: &Layer, mask_assets: &[MaskAsset]) -> bool {
    gpu_plan_for_layer(layer, mask_assets).is_some()
}

fn layer_consumes_wetness(layer: &Layer, mask_assets: &[MaskAsset]) -> bool {
    if layer.kind.required_fields().contains(&FieldId::Wetness)
        || layer.kind.optional_fields().contains(&FieldId::Wetness)
    {
        return true;
    }

    let parameter_source_uses_wetness = match &layer.kind {
        LayerKind::ThermalErosion(p) => matches!(p.hardness_source, MaskSource::Wetness),
        LayerKind::DebrisFlow(p) => matches!(p.hardness_source, MaskSource::Wetness),
        LayerKind::HydraulicErosion(p) => matches!(
            (&p.rainfall_source, &p.protection_source, &p.hardness_source),
            (MaskSource::Wetness, _, _) | (_, MaskSource::Wetness, _) | (_, _, MaskSource::Wetness)
        ),
        LayerKind::StreamPowerErosion(p) => {
            matches!(p.hardness_source, MaskSource::Wetness)
        }
        LayerKind::MultiScaleAmplify(p) => matches!(
            (&p.hardness_source, &p.ridge_lock),
            (MaskSource::Wetness, _) | (_, MaskSource::Wetness)
        ),
        LayerKind::RiverCarve(p) => matches!(p.guide, MaskSource::Wetness),
        _ => false,
    };
    if parameter_source_uses_wetness {
        return true;
    }

    layer.common.masks.entries.iter().any(|entry| {
        mask_assets
            .iter()
            .find(|asset| asset.id == entry.mask.id)
            .is_some_and(|asset| matches!(asset.source, MaskSource::Wetness))
    })
}

/// Compile the preview stack into per-layer GPU plans.
///
/// Records a plan slot for every flattened layer (`None` = disabled or
/// unsupported) and sets `cpu_from` to the first enabled layer that cannot run on
/// the GPU preview path. Supported layers above that boundary still receive a plan
/// so the engine's speculative suffix walk keeps them live on the GPU.
pub fn compile_gpu_graph(stack: &LayerStack, mask_assets: &[MaskAsset]) -> GpuComputeGraph {
    let layers: Vec<&Layer> = stack.flatten_layers();
    let mut plans: Vec<Option<GpuLayerPlan>> = layers
        .iter()
        .map(|layer| {
            layer
                .common
                .enabled
                .then(|| gpu_plan_for_layer(layer, mask_assets))
                .flatten()
        })
        .collect();

    // A `SculptStrokes` GPU plan is a height-only preview; it does not reproduce the
    // aux maps the CPU eval publishes (protection / uplift / hardness / sediment /
    // edit-region). If any *enabled* later layer consumes that aux, previewing the
    // stroke on the GPU would let the downstream result silently diverge from the
    // authoritative CPU eval, so demote the plan to force a CPU resume at the stroke
    // layer — the same reason `Coastal` / `Materials` stay off the GPU today (#113).
    for i in 0..layers.len() {
        if plans[i].is_none() || !matches!(layers[i].kind, LayerKind::SculptStrokes(_)) {
            continue;
        }
        let downstream_consumer = layers[i + 1..]
            .iter()
            .any(|l| l.common.enabled && l.kind.consumes_sculpt_aux());
        if downstream_consumer {
            plans[i] = None;
        }
    }

    // Carved paths publish wetness on the CPU. The GPU path is deliberately a
    // height kernel, so keep it only when no enabled suffix layer can observe that
    // omitted field. Raise-only paths publish no auxiliary data and need no gate.
    for i in 0..layers.len() {
        let LayerKind::Path(params) = &layers[i].kind else {
            continue;
        };
        if plans[i].is_none() || !params.carve {
            continue;
        }
        if layers[i + 1..]
            .iter()
            .any(|layer| layer.common.enabled && layer_consumes_wetness(layer, mask_assets))
        {
            plans[i] = None;
        }
    }

    // MultiScaleAmplify's GPU path is height-only. `MaskSource::None` inherits a
    // previously published hardness field on the CPU, so retain the plan only when
    // the prefix proves no such field exists. Likewise, do not advertise a fully-GPU
    // suffix that would consume hardness/erosion/deposition the preview omits.
    for i in 0..layers.len() {
        let LayerKind::MultiScaleAmplify(params) = &layers[i].kind else {
            continue;
        };
        if plans[i].is_none() {
            continue;
        }
        let inherits_hardness = matches!(params.hardness_source, MaskSource::None)
            && layers[..i].iter().any(|layer| {
                layer.common.enabled && layer.kind.produced_fields().contains(&FieldId::Hardness)
            });
        let downstream_aux_consumer = layers[i + 1..]
            .iter()
            .any(|layer| layer.common.enabled && layer.kind.consumes_sculpt_aux());
        if inherits_hardness || downstream_aux_consumer {
            plans[i] = None;
        }
    }

    // First enabled layer with no executable plan is the CPU-resume boundary.
    let cpu_from = layers
        .iter()
        .zip(&plans)
        .position(|(layer, plan)| layer.common.enabled && plan.is_none());

    GpuComputeGraph { plans, cpu_from }
}

/// Expand a dirty rect by halo, clamped to field bounds.
pub fn expand_dirty_rect(
    rect: (u32, u32, u32, u32),
    halo: u32,
    width: u32,
    height: u32,
) -> (u32, u32, u32, u32) {
    let (x, y, w, h) = rect;
    let x0 = x.saturating_sub(halo);
    let y0 = y.saturating_sub(halo);
    let x1 = (x + w).saturating_add(halo).min(width);
    let y1 = (y + h).saturating_add(halo).min(height);
    (x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use terra_core::layer::{
        BiomesParams, BlurParams, CanyonParams, CoastalParams, DomainWarpParams, DuneParams,
        EffectFilterKind, EffectFilterParams, FbmParams, FlatParams, FractalNoiseType,
        HydraulicErosionParams, ImportHeightmapParams, IslandParams, LandscapeEvolutionParams,
        Layer, LayerKind, LayerStack, LayerTypeRegistry, MaterialsParams, MesaParams,
        MountainParams, MultiScaleAmplifyParams, NoiseParams, PlateauParams, RiverCarveParams,
        SculptStroke, SculptStrokeKind, SculptStrokeParams, Stamp2dParams, Stamp3dParams,
        StreamPowerParams, TerraceParams, ThermalErosionParams, UpliftParams, VegetationParams,
        VolcanoParams,
    };
    use terra_core::mask::{
        bake_distribution, bake_mask_assets, DistributionEntry, MaskId, MaskOp, MaskRef,
    };

    fn single_layer_graph(layer: Layer) -> GpuComputeGraph {
        let mut stack = LayerStack::new();
        stack.push(layer);
        compile_gpu_graph(&stack, &[])
    }

    #[test]
    fn heightmap_asset_support_gate_is_explicit() {
        let import = Layer::new(
            "import",
            LayerKind::ImportHeightmap(ImportHeightmapParams::default()),
        );
        let stamp = Layer::new("stamp", LayerKind::Stamp2d(Stamp2dParams::default()));
        let stamp3d = Layer::new("stamp3d", LayerKind::Stamp3d(Stamp3dParams::default()));
        assert_eq!(
            gpu_plan_for_layer(&import, &[]).map(|plan| plan.kernel),
            Some(GpuKernel::HeightmapSample)
        );
        assert_eq!(
            gpu_plan_for_layer(&stamp, &[]).map(|plan| plan.kernel),
            Some(GpuKernel::HeightmapSample)
        );
        assert!(!layer_gpu_supported(&stamp3d, &[]));

        let mut invalid = import;
        let LayerKind::ImportHeightmap(params) = &mut invalid.kind else {
            unreachable!();
        };
        params.height_scale = f32::NAN;
        assert!(!layer_gpu_supported(&invalid, &[]));
    }

    fn masked_flat(asset: &MaskAsset) -> Layer {
        let mut layer = Layer::new("masked", LayerKind::Flat(FlatParams::default()));
        layer.common.masks.push(MaskRef::new(asset.id));
        layer
    }

    fn inplace_layers() -> Vec<Layer> {
        vec![
            Layer::new("blur", LayerKind::Blur(BlurParams::default())),
            Layer::new(
                "effect filter",
                LayerKind::EffectFilter(EffectFilterParams::default()),
            ),
            Layer::new("terrace", LayerKind::Terrace(TerraceParams::default())),
            Layer::new(
                "thermal",
                LayerKind::ThermalErosion(ThermalErosionParams {
                    layered_materials: false,
                    weathering_rate: 0.0,
                    ..ThermalErosionParams::default()
                }),
            ),
            Layer::new(
                "hydraulic",
                LayerKind::HydraulicErosion(HydraulicErosionParams {
                    layered_materials: false,
                    particle_density: 0.0,
                    ..HydraulicErosionParams::default()
                }),
            ),
            Layer::new(
                "river carve",
                LayerKind::RiverCarve(RiverCarveParams::default()),
            ),
            Layer::new(
                "stream power",
                LayerKind::StreamPowerErosion(StreamPowerParams::default()),
            ),
        ]
    }

    /// The CPU spatial-dependency class must never be *more permissive* than the
    /// GPU dirty policy for the same layer: whatever the GPU refuses to localize,
    /// the CPU sub-region recompute (#100) must also refuse. Ranks: Local < Expanding
    /// < BasinDependent on the CPU side; the GPU's `FullField` demands the CPU be
    /// fully basin-coupled, while GPU `Local` demands nothing (the CPU may be
    /// stricter — `Terrace` deliberately is, using the exact field range where the
    /// shader only approximates it).
    #[test]
    fn cpu_reach_is_never_more_permissive_than_gpu_policy() {
        use terra_core::invalidation::DirtyClass;

        fn cpu_rank(c: DirtyClass) -> u8 {
            match c {
                DirtyClass::Local => 0,
                DirtyClass::Expanding => 1,
                DirtyClass::BasinDependent => 2,
            }
        }
        fn gpu_required_rank(p: GpuDirtyPolicy) -> u8 {
            match p {
                GpuDirtyPolicy::Local => 0,
                GpuDirtyPolicy::FullField => 2,
            }
        }

        let mut candidates = inplace_layers();
        candidates.push(Layer::new("flat", LayerKind::Flat(FlatParams::default())));
        candidates.push(Layer::new(
            "noise",
            LayerKind::NoiseValue(NoiseParams::default()),
        ));
        // SculptStrokes: GPU localizes it (Local), CPU is Local too — never more
        // permissive. This holds for Flatten as well: it stays tile-scoped (Local) on
        // both sides via its #110 footprint fixpoint.
        candidates.push(strokes_layer(SculptStrokeKind::Raise, 0.15));
        candidates.push(strokes_layer(SculptStrokeKind::Flatten, 0.15));

        let mut saw_full_field = false;
        for layer in &candidates {
            let Some(plan) = gpu_plan_for_layer(layer, &[]) else {
                continue;
            };
            let cpu = layer.kind.spatial_dependency();
            assert!(
                cpu_rank(cpu) >= gpu_required_rank(plan.dirty_policy),
                "{:?}: CPU {cpu:?} is more permissive than GPU {:?}",
                layer.kind,
                plan.dirty_policy
            );
            if plan.dirty_policy == GpuDirtyPolicy::FullField {
                saw_full_field = true;
                assert_eq!(
                    cpu,
                    DirtyClass::BasinDependent,
                    "{:?}: GPU FullField requires CPU BasinDependent",
                    layer.kind
                );
            }
        }
        assert!(
            saw_full_field,
            "expected at least one FullField GPU plan to exercise the constraint"
        );

        // Terrace: GPU localizes it, CPU is deliberately stricter.
        let terrace = Layer::new("terrace", LayerKind::Terrace(TerraceParams::default()));
        assert_eq!(
            gpu_plan_for_layer(&terrace, &[]).map(|p| p.dirty_policy),
            Some(GpuDirtyPolicy::Local)
        );
        assert_eq!(
            terrace.kind.spatial_dependency(),
            DirtyClass::BasinDependent
        );
    }

    #[test]
    fn compiles_generator_filter_stack_as_fully_gpu() {
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "base",
            LayerKind::Flat(FlatParams { height: 12.0 }),
        ));
        stack.push(Layer::new(
            "smooth",
            LayerKind::EffectFilter(EffectFilterParams {
                kind: EffectFilterKind::Smooth,
                ..EffectFilterParams::default()
            }),
        ));
        stack.push(Layer::new("flat", LayerKind::Flat(FlatParams::default())));
        let g = compile_gpu_graph(&stack, &[]);
        assert!(
            g.fully_gpu(),
            "expected fully GPU graph, cpu_from={:?}",
            g.cpu_from
        );
        assert!(g.plans.iter().filter(|p| p.is_some()).count() >= 2);
    }

    #[test]
    fn expand_dirty_respects_bounds() {
        let r = expand_dirty_rect((10, 10, 20, 20), 8, 100, 100);
        assert_eq!(r, (2, 2, 36, 36));
        let edge = expand_dirty_rect((0, 0, 5, 5), 8, 100, 100);
        assert_eq!(edge.0, 0);
        assert_eq!(edge.1, 0);
    }

    /// #125 admits the shader's Value/Perlin fractal modes without substituting
    /// OpenSimplex, which remains a CPU boundary for #130.
    #[test]
    fn fractal_noise_support_is_explicit_without_substitution() {
        for make_kind in [
            |params| LayerKind::Fbm(params),
            |params| LayerKind::Ridged(params),
        ] {
            for noise in [
                FractalNoiseType::Value,
                FractalNoiseType::Perlin,
                FractalNoiseType::OpenSimplex,
            ] {
                let layer = Layer::new(
                    "fractal",
                    make_kind(FbmParams {
                        noise,
                        ..FbmParams::default()
                    }),
                );
                let supported = matches!(noise, FractalNoiseType::Value | FractalNoiseType::Perlin);
                assert_eq!(layer_gpu_supported(&layer, &[]), supported, "{noise:?}");
                let graph = single_layer_graph(layer);
                assert_eq!(graph.fully_gpu(), supported, "{noise:?}");
                assert_eq!(graph.cpu_from, (!supported).then_some(0), "{noise:?}");
            }
        }
    }

    #[test]
    fn noise_family_defaults_compile_to_the_noise_kernel() {
        let layers = [
            Layer::new("perlin", LayerKind::NoisePerlin(NoiseParams::default())),
            Layer::new("fbm", LayerKind::Fbm(FbmParams::default())),
            Layer::new("ridged", LayerKind::Ridged(FbmParams::default())),
            Layer::new(
                "domain warp",
                LayerKind::DomainWarp(DomainWarpParams::default()),
            ),
        ];
        for layer in layers {
            let plan = gpu_plan_for_layer(&layer, &[])
                .unwrap_or_else(|| panic!("{} should compile", layer.common.name));
            assert_eq!(plan.kernel, GpuKernel::Noise, "{}", layer.common.name);
            assert_eq!(plan.dirty_policy, GpuDirtyPolicy::Local);
            assert_eq!(plan.halo_texels, 2);
            assert!(plan.kernel.matches_layer_kind(&layer.kind));
        }
    }

    /// Revert check for #48: semantic CPU operations cannot compile as GPU no-ops.
    #[test]
    fn semantic_noops_and_coastal_are_cpu_boundaries() {
        let layers = [
            Layer::new("coastal", LayerKind::Coastal(CoastalParams::default())),
            Layer::new(
                "materials",
                LayerKind::Materials(MaterialsParams::default()),
            ),
            Layer::new("biomes", LayerKind::Biomes(BiomesParams::default())),
            Layer::new(
                "vegetation",
                LayerKind::Vegetation(VegetationParams::default()),
            ),
        ];
        for layer in layers {
            assert!(!layer_gpu_supported(&layer, &[]));
            let graph = single_layer_graph(layer.clone());
            assert_eq!(graph.cpu_from, Some(0));
            assert_eq!(graph.plans.len(), 1);
            assert!(graph.plans[0].is_none());
        }
    }

    #[test]
    fn public_support_query_and_graph_agree_for_every_builtin_default() {
        let registry = LayerTypeRegistry::builtin();
        for meta in registry.all() {
            let layer = registry.create(meta.type_id).expect("registered factory");
            let supported = layer_gpu_supported(&layer, &[]);
            let graph = single_layer_graph(layer.clone());
            assert_eq!(
                graph.fully_gpu(),
                supported,
                "support disagreement for {}",
                meta.type_id
            );
            assert_eq!(graph.cpu_from, (!supported).then_some(0));
            assert_eq!(graph.plans.len(), 1, "{}", meta.type_id);
            if supported {
                let plan = graph.plans[0].expect("supported builtin retains a plan");
                assert!(
                    plan.kernel.matches_layer_kind(&layer.kind),
                    "planner selected {:?} for {}",
                    plan.kernel,
                    meta.type_id
                );
            }
        }
    }

    #[test]
    fn categorical_effect_filter_configs_are_explicitly_supported_or_exempted() {
        for &kind in EffectFilterKind::ALL {
            let layer = Layer::new(
                kind.label(),
                LayerKind::EffectFilter(EffectFilterParams {
                    kind,
                    ..EffectFilterParams::default()
                }),
            );
            let supported = matches!(
                kind,
                EffectFilterKind::Smooth
                    | EffectFilterKind::Inflate
                    | EffectFilterKind::Denoise
                    | EffectFilterKind::AddSet
                    | EffectFilterKind::Deflate
                    | EffectFilterKind::Curve
                    | EffectFilterKind::Cutoff
                    | EffectFilterKind::TerraceSimple
                    | EffectFilterKind::Shore
                    | EffectFilterKind::Blocks
                    | EffectFilterKind::ZeroEdge
                    | EffectFilterKind::Squeeze
                    | EffectFilterKind::DirectionalBlur
                    | EffectFilterKind::AngleBlur
                    | EffectFilterKind::Swirl
                    | EffectFilterKind::Crater
                    | EffectFilterKind::Distortion
                    | EffectFilterKind::Balloon
                    | EffectFilterKind::NoisePerlin
                    | EffectFilterKind::NoiseValue
                    | EffectFilterKind::NoiseWhite
                    | EffectFilterKind::NoiseWave
                    | EffectFilterKind::ScatterDetail
                    | EffectFilterKind::NoiseBillow
                    | EffectFilterKind::NoiseRidged
                    | EffectFilterKind::Ridged
                    | EffectFilterKind::Rugged
                    | EffectFilterKind::Hexagons
                    | EffectFilterKind::TerraceSteep
            );
            let plan = gpu_plan_for_layer(&layer, &[]);
            assert_eq!(plan.is_some(), supported, "{}", kind.label());
            if let Some(plan) = plan {
                assert_eq!(plan.kernel, GpuKernel::EffectFilter, "{}", kind.label());
                assert!(plan.kernel.matches_layer_kind(&layer.kind));
                let expected_policy = match kind {
                    EffectFilterKind::Curve
                    | EffectFilterKind::Cutoff
                    | EffectFilterKind::TerraceSimple
                    | EffectFilterKind::ZeroEdge
                    | EffectFilterKind::Squeeze
                    | EffectFilterKind::Swirl
                    | EffectFilterKind::Distortion
                    | EffectFilterKind::Hexagons
                    | EffectFilterKind::TerraceSteep => GpuDirtyPolicy::FullField,
                    _ => GpuDirtyPolicy::Local,
                };
                assert_eq!(plan.dirty_policy, expected_policy, "{}", kind.label());
            }
        }
    }

    fn strokes_layer(kind: SculptStrokeKind, reconcile: f32) -> Layer {
        Layer::new(
            "strokes",
            LayerKind::SculptStrokes(SculptStrokeParams {
                strokes: vec![SculptStroke {
                    kind,
                    ..SculptStroke::default()
                }],
                reconcile,
            }),
        )
    }

    #[test]
    fn sculpt_strokes_supported_kinds_compile_to_a_local_stamp_plan() {
        // Supported kinds preview on the GPU. The plan halo is the base-neighborhood
        // read (Smooth, Pinch, and Coastline read a 3x3 of the layer input: +1) plus the
        // reconcile 3x3 relax (non-zero reconcile: +1). Per-sample kinds have no base
        // read, so they carry only the reconcile texel (1) or none (0); Smooth, Pinch,
        // and Coastline carry both (2) or their lone base read (1).
        for (kind, reconcile, halo) in [
            (SculptStrokeKind::Raise, 0.15, 1),
            (SculptStrokeKind::Lower, 0.0, 0),
            (SculptStrokeKind::Ridge, 0.2, 1),
            (SculptStrokeKind::Valley, 0.2, 1),
            (SculptStrokeKind::Inflate, 0.2, 1),
            (SculptStrokeKind::Terrace, 0.2, 1),
            (SculptStrokeKind::Noise, 0.2, 1),
            (SculptStrokeKind::HeightStamp, 0.0, 0),
            (SculptStrokeKind::PlateauStamp, 0.2, 1),
            (SculptStrokeKind::CraterStamp, 0.2, 1),
            (SculptStrokeKind::MountainStamp, 0.2, 1),
            (SculptStrokeKind::Uplift, 0.2, 1),
            (SculptStrokeKind::Smooth, 0.2, 2),
            (SculptStrokeKind::Smooth, 0.0, 1),
            (SculptStrokeKind::Pinch, 0.2, 2),
            (SculptStrokeKind::Pinch, 0.0, 1),
            (SculptStrokeKind::Coastline, 0.2, 2),
            (SculptStrokeKind::Coastline, 0.0, 1),
            // Flatten's footprint mean is a precomputed scalar (no base 3x3), so it
            // carries only the reconcile texel (1) or none (0) — #117.
            (SculptStrokeKind::Flatten, 0.2, 1),
            (SculptStrokeKind::Flatten, 0.0, 0),
        ] {
            let layer = strokes_layer(kind, reconcile);
            let plan = gpu_plan_for_layer(&layer, &[])
                .unwrap_or_else(|| panic!("{kind:?} should compile to a GPU plan"));
            assert_eq!(plan.kernel, GpuKernel::SculptStrokes, "{kind:?}");
            assert!(plan.kernel.matches_layer_kind(&layer.kind), "{kind:?}");
            assert_eq!(plan.dirty_policy, GpuDirtyPolicy::Local, "{kind:?}");
            assert_eq!(plan.halo_texels, halo, "{kind:?}");
            assert!(layer_gpu_supported(&layer, &[]), "{kind:?}");
        }
    }

    #[test]
    fn sculpt_strokes_reduction_kind_previews_on_gpu() {
        // Flatten's per-stroke footprint-mean reduction now runs on the GPU: the
        // reduce/resolve passes precompute each target and the stamp path segments
        // the run around it (#117). A Flatten layer compiles fully GPU, and it
        // composes with per-sample kinds in one plan (the stroke set no longer has to
        // split across the GPU/CPU boundary at a Flatten).
        let kind = SculptStrokeKind::Flatten;
        let layer = strokes_layer(kind, 0.15);
        let plan = gpu_plan_for_layer(&layer, &[])
            .unwrap_or_else(|| panic!("{kind:?} should compile to a GPU plan"));
        assert_eq!(plan.kernel, GpuKernel::SculptStrokes, "{kind:?}");
        // Flatten stays tile-scoped like the CPU (its #110 footprint fixpoint keeps a
        // self-edit recompute bit-exact), so it takes the same Local policy.
        assert_eq!(plan.dirty_policy, GpuDirtyPolicy::Local, "{kind:?}");
        assert!(layer_gpu_supported(&layer, &[]), "{kind:?}");
        assert_eq!(single_layer_graph(layer).cpu_from, None, "{kind:?}");

        // Flatten interleaved with a per-sample kind still compiles: the segmentation
        // measures the Flatten against the running field the Raise already wrote.
        let mixed = Layer::new(
            "mixed",
            LayerKind::SculptStrokes(SculptStrokeParams {
                strokes: vec![
                    SculptStroke {
                        kind: SculptStrokeKind::Raise,
                        ..SculptStroke::default()
                    },
                    SculptStroke {
                        kind: SculptStrokeKind::Flatten,
                        ..SculptStroke::default()
                    },
                ],
                reconcile: 0.15,
            }),
        );
        assert!(layer_gpu_supported(&mixed, &[]));
    }

    #[test]
    fn sculpt_strokes_require_a_supported_blend() {
        let mut layer = strokes_layer(SculptStrokeKind::Raise, 0.15);
        layer.common.blend = BlendMode::SmoothMaximum;
        assert!(gpu_blend_mode(layer.common.blend).is_none());
        assert!(!layer_gpu_supported(&layer, &[]));
        assert_eq!(single_layer_graph(layer).cpu_from, Some(0));
    }

    #[test]
    fn sculpt_strokes_are_demoted_when_a_downstream_layer_consumes_their_aux() {
        // Raise stroke (GPU-capable) followed by a consumer of the aux it cannot
        // reproduce on the GPU: the stroke plan is demoted so the preview resumes on
        // the CPU at the stroke layer, not silently diverging past it.
        let mut stack = LayerStack::new();
        stack.push(Layer::new("base", LayerKind::Flat(FlatParams::default())));
        stack.push(strokes_layer(SculptStrokeKind::Raise, 0.15));
        stack.push(Layer::new(
            "evolve",
            LayerKind::LandscapeEvolution(LandscapeEvolutionParams::default()),
        ));
        let graph = compile_gpu_graph(&stack, &[]);
        assert!(graph.plans[0].is_some(), "base flat stays on GPU");
        assert!(graph.plans[1].is_none(), "stroke layer demoted by consumer");
        assert_eq!(graph.cpu_from, Some(1));

        // A disabled consumer does not run, so it must not demote the preview.
        let mut with_disabled = LayerStack::new();
        with_disabled.push(strokes_layer(SculptStrokeKind::Raise, 0.15));
        let mut disabled_consumer = Layer::new(
            "evolve",
            LayerKind::LandscapeEvolution(LandscapeEvolutionParams::default()),
        );
        disabled_consumer.common.enabled = false;
        with_disabled.push(disabled_consumer);
        let graph = compile_gpu_graph(&with_disabled, &[]);
        assert!(graph.plans[0].is_some(), "no enabled consumer downstream");
        assert_eq!(graph.cpu_from, None);
    }

    #[test]
    fn sculpt_strokes_stay_on_gpu_above_a_non_consuming_filter() {
        // Raise stroke then a Blur (does not read sculpt aux): fully GPU, live.
        let mut stack = LayerStack::new();
        stack.push(Layer::new("base", LayerKind::Flat(FlatParams::default())));
        stack.push(strokes_layer(SculptStrokeKind::Raise, 0.15));
        stack.push(Layer::new("blur", LayerKind::Blur(BlurParams::default())));
        let graph = compile_gpu_graph(&stack, &[]);
        assert!(graph.fully_gpu(), "cpu_from={:?}", graph.cpu_from);
        assert!(graph.plans.iter().all(|p| p.is_some()));
    }

    #[test]
    fn ignored_or_truncated_generator_configs_force_cpu_fallback() {
        let high_seed = NoiseParams {
            seed: u64::from(u32::MAX) + 1,
            ..NoiseParams::default()
        };
        assert!(layer_gpu_supported(
            &Layer::new("low seed", LayerKind::NoiseValue(NoiseParams::default())),
            &[]
        ));
        let mut high_seed_mountains = MountainParams::default();
        high_seed_mountains.base.seed = u64::from(u32::MAX) + 1;
        let mut custom_transport_dunes = DuneParams::default();
        custom_transport_dunes.iterations += 1;
        let cases = [
            Layer::new("high seed", LayerKind::NoiseValue(high_seed)),
            Layer::new(
                "high-seed mountains",
                LayerKind::Mountains(high_seed_mountains),
            ),
            Layer::new(
                "custom dune transport",
                LayerKind::Dunes(custom_transport_dunes),
            ),
            Layer::new(
                "layered thermal",
                LayerKind::ThermalErosion(ThermalErosionParams::default()),
            ),
            Layer::new(
                "particle hydraulic",
                LayerKind::HydraulicErosion(HydraulicErosionParams::default()),
            ),
        ];
        for layer in cases {
            assert!(!layer_gpu_supported(&layer, &[]), "{}", layer.common.name);
            assert_eq!(single_layer_graph(layer).cpu_from, Some(0));
        }
    }

    #[test]
    fn river_carve_support_is_full_field_and_semantics_preserving() {
        let default = Layer::new(
            "D-infinity river",
            LayerKind::RiverCarve(RiverCarveParams::default()),
        );
        let plan = gpu_plan_for_layer(&default, &[]).expect("default RiverCarve should compile");
        assert_eq!(plan.kernel, GpuKernel::RiverCarve);
        assert_eq!(plan.dirty_policy, GpuDirtyPolicy::FullField);
        assert_eq!(plan.halo_texels, 0);

        let mut guided = RiverCarveParams::default();
        guided.guide = MaskSource::Wetness;
        assert!(!layer_gpu_supported(
            &Layer::new("guided", LayerKind::RiverCarve(guided)),
            &[]
        ));

        let inert_guide = RiverCarveParams {
            guide: MaskSource::Wetness,
            guide_boost: 0.0,
            ..RiverCarveParams::default()
        };
        assert!(layer_gpu_supported(
            &Layer::new("inert guide", LayerKind::RiverCarve(inert_guide)),
            &[]
        ));

        let too_wide = RiverCarveParams {
            bank_smooth: 3.0,
            ..RiverCarveParams::default()
        };
        assert!(!layer_gpu_supported(
            &Layer::new("wide banks", LayerKind::RiverCarve(too_wide)),
            &[]
        ));

        let invalid_threshold = RiverCarveParams {
            accumulation_threshold: 0.0,
            ..RiverCarveParams::default()
        };
        assert!(!layer_gpu_supported(
            &Layer::new(
                "invalid threshold",
                LayerKind::RiverCarve(invalid_threshold)
            ),
            &[]
        ));
    }

    #[test]
    fn stream_power_support_is_full_field_and_semantics_preserving() {
        let default = Layer::new(
            "stream power",
            LayerKind::StreamPowerErosion(StreamPowerParams::default()),
        );
        let plan = gpu_plan_for_layer(&default, &[]).expect("default StreamPower should compile");
        assert_eq!(plan.kernel, GpuKernel::StreamPower);
        assert_eq!(plan.dirty_policy, GpuDirtyPolicy::FullField);
        assert_eq!(plan.halo_texels, 0);

        let sourced_hardness = StreamPowerParams {
            hardness_source: MaskSource::Hardness,
            ..StreamPowerParams::default()
        };
        let dendritic = StreamPowerParams {
            dendritic_seed: 0.25,
            ..StreamPowerParams::default()
        };
        let refill = StreamPowerParams {
            refill_each_iter: true,
            ..StreamPowerParams::default()
        };
        let authored_levels = StreamPowerParams {
            level_count: 1,
            ..StreamPowerParams::default()
        };
        let invalid_k = StreamPowerParams {
            k: f32::NAN,
            ..StreamPowerParams::default()
        };
        for (name, params) in [
            ("sourced hardness", sourced_hardness),
            ("dendritic seed", dendritic),
            ("refill each iter", refill),
            ("authored levels", authored_levels),
            ("invalid k", invalid_k),
        ] {
            let layer = Layer::new(name, LayerKind::StreamPowerErosion(params));
            assert!(!layer_gpu_supported(&layer, &[]), "{name}");
            assert_eq!(single_layer_graph(layer).cpu_from, Some(0), "{name}");
        }

        let mut partial = default;
        partial.common.opacity = 0.5;
        assert!(!layer_gpu_supported(&partial, &[]));
    }

    #[test]
    fn multi_scale_amplify_support_is_full_field_and_aux_safe() {
        let default = Layer::new(
            "multi scale",
            LayerKind::MultiScaleAmplify(MultiScaleAmplifyParams::default()),
        );
        let plan = gpu_plan_for_layer(&default, &[]).expect("default amplify should compile");
        assert_eq!(plan.kernel, GpuKernel::MultiScaleAmplify);
        assert_eq!(plan.dirty_policy, GpuDirtyPolicy::FullField);
        assert_eq!(plan.halo_texels, 0);

        for params in [
            MultiScaleAmplifyParams {
                hardness_source: MaskSource::Hardness,
                ..MultiScaleAmplifyParams::default()
            },
            MultiScaleAmplifyParams {
                ridge_lock: MaskSource::Slope {
                    min_deg: 2.0,
                    max_deg: 20.0,
                },
                ..MultiScaleAmplifyParams::default()
            },
            MultiScaleAmplifyParams {
                detail_boost: f32::NAN,
                ..MultiScaleAmplifyParams::default()
            },
        ] {
            assert!(!layer_gpu_supported(
                &Layer::new("unsupported amplify", LayerKind::MultiScaleAmplify(params)),
                &[]
            ));
        }

        let mut inherited = LayerStack::new();
        inherited.push(Layer::new(
            "thermal",
            LayerKind::ThermalErosion(ThermalErosionParams {
                layered_materials: false,
                weathering_rate: 0.0,
                ..ThermalErosionParams::default()
            }),
        ));
        inherited.push(default.clone());
        let graph = compile_gpu_graph(&inherited, &[]);
        assert!(graph.plans[0].is_some());
        assert!(graph.plans[1].is_none());
        assert_eq!(graph.cpu_from, Some(1));

        let mut consumed = LayerStack::new();
        consumed.push(default);
        consumed.push(Layer::new(
            "stream power",
            LayerKind::StreamPowerErosion(StreamPowerParams::default()),
        ));
        let graph = compile_gpu_graph(&consumed, &[]);
        assert!(graph.plans[0].is_none());
        assert_eq!(graph.cpu_from, Some(0));
    }

    #[test]
    fn shape_family_defaults_and_all_island_archetypes_compile_to_shape_kernel() {
        let mut islands = [
            IslandParams::default(),
            IslandParams::archipelago(),
            IslandParams::atoll(),
        ];
        islands[0].archetype = IslandArchetype::VolcanicHighIsland;
        let mut layers = vec![
            Layer::new("mountains", LayerKind::Mountains(MountainParams::default())),
            Layer::new("dunes", LayerKind::Dunes(DuneParams::default())),
            Layer::new("canyons", LayerKind::Canyons(CanyonParams::default())),
            Layer::new("mesa", LayerKind::Mesa(MesaParams::default())),
            Layer::new("volcano", LayerKind::Volcano(VolcanoParams::default())),
            Layer::new("uplift", LayerKind::Uplift(UpliftParams::default())),
            Layer::new("plateau", LayerKind::Plateau(PlateauParams::default())),
        ];
        layers.extend(
            islands
                .into_iter()
                .map(|p| Layer::new("island", LayerKind::Island(p))),
        );

        for layer in layers {
            let name = layer.common.name.clone();
            let graph = single_layer_graph(layer);
            assert!(graph.fully_gpu(), "{name}: cpu_from={:?}", graph.cpu_from);
            assert_eq!(graph.plans[0].unwrap().kernel, GpuKernel::Shape, "{name}");
        }
    }

    #[test]
    fn graph_reports_first_unsupported_flattened_index() {
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "noise",
            LayerKind::NoiseValue(NoiseParams::default()),
        ));
        stack.push(Layer::new(
            "materials",
            LayerKind::Materials(MaterialsParams::default()),
        ));
        stack.push(Layer::new("flat", LayerKind::Flat(FlatParams::default())));

        let graph = compile_gpu_graph(&stack, &[]);
        assert_eq!(graph.cpu_from, Some(1));
        // Per-index plans: layers below AND above the CPU boundary keep a plan
        // (the speculative suffix runs them); only the unsupported owner is None.
        assert!(graph.plans[0].is_some());
        assert!(graph.plans[1].is_none());
        assert!(graph.plans[2].is_some());
    }

    /// Revert check for #50: in-place kernels only implement the default outer
    /// composite, so richer LayerCommon settings must begin CPU fallback at the owner.
    #[test]
    fn inplace_kernels_require_default_outer_composite() {
        let mask = MaskAsset::new(MaskId::new(), "constant", MaskSource::Constant(0.5));
        for default_layer in inplace_layers() {
            assert!(
                layer_gpu_supported(&default_layer, &[]),
                "default {} should remain GPU-supported",
                default_layer.common.name
            );

            let mut cases = Vec::new();
            let mut partial = default_layer.clone();
            partial.common.opacity = 0.5;
            cases.push((partial, Vec::new(), "partial opacity"));

            let mut masked = default_layer.clone();
            masked.common.masks.push(MaskRef::new(mask.id));
            cases.push((masked, vec![mask.clone()], "mask"));

            let mut additive = default_layer.clone();
            additive.common.blend = BlendMode::Add;
            cases.push((additive, Vec::new(), "non-replacement blend"));

            for (layer, assets, reason) in cases {
                assert!(
                    !layer_gpu_supported(&layer, &assets),
                    "{} unexpectedly supports {reason}",
                    layer.common.name
                );
                let mut stack = LayerStack::new();
                stack.push(Layer::new("prefix", LayerKind::Flat(FlatParams::default())));
                stack.push(layer);
                let graph = compile_gpu_graph(&stack, &assets);
                assert_eq!(graph.cpu_from, Some(1), "{reason}");
                assert!(graph.plans[0].is_some(), "{reason}");
                assert!(graph.plans[1].is_none(), "{reason}");
            }
        }
    }

    /// Revert check for #50: only equations actually implemented in blend.wgsl
    /// may be advertised; no authored blend is substituted with a nearby equation.
    #[test]
    fn blend_support_maps_exact_equations_or_falls_back() {
        for (mode, shader_mode) in [
            (BlendMode::Normal, 0),
            (BlendMode::Replace, 0),
            (BlendMode::Interpolate, 0),
            (BlendMode::Add, 1),
            (BlendMode::Subtract, 2),
            (BlendMode::Multiply, 3),
            (BlendMode::Min, 4),
            (BlendMode::Max, 5),
            (BlendMode::Overlay, 6),
        ] {
            assert_eq!(gpu_blend_mode(mode), Some(shader_mode));
            let mut layer = Layer::new("generator", LayerKind::Flat(FlatParams::default()));
            layer.common.blend = mode;
            assert!(layer_gpu_supported(&layer, &[]), "{mode:?}");
        }

        for mode in [
            BlendMode::HeightBlend,
            BlendMode::SmoothMaximum,
            BlendMode::SmoothMinimum,
            BlendMode::SmoothUnion,
            BlendMode::SmoothSubtraction,
        ] {
            assert_eq!(gpu_blend_mode(mode), None, "{mode:?}");
            let mut layer = Layer::new("generator", LayerKind::Flat(FlatParams::default()));
            layer.common.blend = mode;
            assert!(!layer_gpu_supported(&layer, &[]), "{mode:?}");
            assert_eq!(single_layer_graph(layer).cpu_from, Some(0), "{mode:?}");
        }
    }

    #[test]
    fn simple_single_entry_masks_are_the_only_gpu_supported_contract() {
        for source in [
            MaskSource::Constant(0.5),
            MaskSource::Height {
                min: 10.0,
                max: 20.0,
            },
            MaskSource::Slope {
                min_deg: 5.0,
                max_deg: 35.0,
            },
        ] {
            let asset = MaskAsset::new(MaskId::new(), "supported", source);
            let layer = masked_flat(&asset);
            assert!(layer_gpu_supported(&layer, std::slice::from_ref(&asset)));
        }

        let empty = Layer::new("empty", LayerKind::Flat(FlatParams::default()));
        assert!(layer_gpu_supported(&empty, &[]));
    }

    #[test]
    fn complex_or_unproven_masks_start_cpu_fallback_at_the_owner() {
        let mut cases = Vec::new();

        let missing = MaskAsset::new(MaskId::new(), "missing", MaskSource::Constant(0.5));
        cases.push((masked_flat(&missing), Vec::new(), "missing asset"));

        for source in [
            MaskSource::Curvature {
                min: -1.0,
                max: 1.0,
            },
            MaskSource::Noise {
                seed: 0x1_0000_0001,
                frequency: 0.05,
            },
        ] {
            let asset = MaskAsset::new(MaskId::new(), "unproven", source);
            cases.push((masked_flat(&asset), vec![asset], "unproven source"));
        }

        let mut operated = MaskAsset::new(MaskId::new(), "operated", MaskSource::Constant(0.2));
        operated.ops.push(MaskOp::Invert);
        cases.push((masked_flat(&operated), vec![operated], "asset operation"));

        let combined = MaskAsset::new(MaskId::new(), "combined", MaskSource::Constant(0.5));
        let mut non_multiply = masked_flat(&combined);
        non_multiply.common.masks.entries[0].combine = MaskCombine::Subtract;
        cases.push((non_multiply, vec![combined], "non-Multiply combine"));

        for (layer, assets, reason) in cases {
            assert!(!layer_gpu_supported(&layer, &assets), "{reason}");
            let mut stack = LayerStack::new();
            stack.push(layer);
            let graph = compile_gpu_graph(&stack, &assets);
            assert_eq!(graph.cpu_from, Some(0), "{reason}");
            assert!(graph.plans[0].is_none(), "{reason}");
        }
    }

    #[test]
    fn ordered_distribution_fixture_is_non_commutative_and_cpu_bound() {
        let metrics = terra_core::heightfield::HeightfieldMetrics::new(2, 2, 2.0, 2.0);
        let first = MaskAsset::new(MaskId::new(), "first", MaskSource::Constant(0.8));
        let second = MaskAsset::new(MaskId::new(), "second", MaskSource::Constant(0.25));
        let assets = vec![first.clone(), second.clone()];
        let baked = bake_mask_assets(
            &assets,
            &terra_core::heightfield::Heightfield::zeros(metrics),
            metrics,
            &HashMap::new(),
        );

        let mut layer = masked_flat(&first);
        layer.common.masks.entries.push(DistributionEntry {
            mask: MaskRef::new(second.id),
            combine: MaskCombine::Subtract,
        });
        let oracle = bake_distribution(&layer.common.masks, &baked, metrics);
        assert!((oracle.get(0, 0) - 0.55).abs() < 1.0e-6);
        assert!(!layer_gpu_supported(&layer, &assets));

        let mut stack = LayerStack::new();
        stack.push(layer);
        assert_eq!(compile_gpu_graph(&stack, &assets).cpu_from, Some(0));
    }

    #[test]
    fn asset_operation_fixture_changes_the_cpu_mask_and_requires_fallback() {
        let metrics = terra_core::heightfield::HeightfieldMetrics::new(2, 2, 2.0, 2.0);
        let mut asset = MaskAsset::new(MaskId::new(), "invert", MaskSource::Constant(0.2));
        asset.ops.push(MaskOp::Invert);
        let baked = bake_mask_assets(
            std::slice::from_ref(&asset),
            &terra_core::heightfield::Heightfield::zeros(metrics),
            metrics,
            &HashMap::new(),
        );
        let layer = masked_flat(&asset);
        let oracle = bake_distribution(&layer.common.masks, &baked, metrics);
        assert!((oracle.get(0, 0) - 0.8).abs() < 1.0e-6);
        assert!(!layer_gpu_supported(&layer, std::slice::from_ref(&asset)));
    }
}
