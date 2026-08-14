//! Compile a layer stack into an ordered GPU compute pass list.
//!
//! Artist UI remains layer-based; this IR is the internal execution plan for
//! `GpuTerrainEngine` (deps, dirty policy, fusion hooks).

use terra_core::layer::{
    BlendMode, EffectFilterKind, IslandArchetype, Layer, LayerId, LayerKind, LayerStack,
    TransportModel,
};
use terra_core::mask::{MaskAsset, MaskCombine, MaskSource};
use terra_core::tiling::DirtyClass;

/// Kind of GPU work a compiled pass performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPassKind {
    Generator,
    Filter,
    Simulation,
    MaskBake,
    Blend,
}

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
    Blur,
    EffectFilter,
    Terrace,
    Thermal,
    Hydraulic,
    RiverCarve,
}

impl GpuKernel {
    pub fn matches_layer_kind(self, kind: &LayerKind) -> bool {
        matches!(
            (self, kind),
            (Self::Fill, LayerKind::Flat(_))
                | (Self::Ramp, LayerKind::Ramp(_))
                | (Self::Sculpt, LayerKind::SculptBase(_))
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

/// One executable GPU pass corresponding to a flattened layer (or mask prep).
#[derive(Debug, Clone)]
pub struct GpuPass {
    pub layer_id: LayerId,
    pub kind: GpuPassKind,
    pub kernel: GpuKernel,
    pub dirty_policy: GpuDirtyPolicy,
    /// Flattened index into `stack.flatten_layers()`.
    pub flat_index: usize,
    /// Approximate kernel halo in texels for dirty expansion.
    pub halo_texels: u32,
}

/// Compiled interactive GPU graph.
#[derive(Debug, Clone, Default)]
pub struct GpuComputeGraph {
    pub passes: Vec<GpuPass>,
    /// First flat index that is not GPU-supported (None = fully GPU).
    pub cpu_from: Option<usize>,
}

impl GpuComputeGraph {
    pub fn fully_gpu(&self) -> bool {
        self.cpu_from.is_none()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GpuLayerPlan {
    pub(crate) kind: GpuPassKind,
    pub(crate) kernel: GpuKernel,
    pub(crate) dirty_policy: GpuDirtyPolicy,
    pub(crate) halo_texels: u32,
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

pub(crate) fn gpu_plan_for_layer(layer: &Layer, mask_assets: &[MaskAsset]) -> Option<GpuLayerPlan> {
    use LayerKind::*;
    if !gpu_mask_supported(layer, mask_assets) {
        return None;
    }
    let (kernel, kind, dirty_policy, halo_texels) = match &layer.kind {
        Flat(_) if gpu_blend_mode(layer.common.blend).is_some() => (
            GpuKernel::Fill,
            GpuPassKind::Generator,
            GpuDirtyPolicy::Local,
            0,
        ),
        Ramp(_) if gpu_blend_mode(layer.common.blend).is_some() => (
            GpuKernel::Ramp,
            GpuPassKind::Generator,
            GpuDirtyPolicy::Local,
            0,
        ),
        SculptBase(_) if gpu_blend_mode(layer.common.blend).is_some() => (
            GpuKernel::Sculpt,
            GpuPassKind::Generator,
            GpuDirtyPolicy::Local,
            0,
        ),
        NoiseValue(p) if seed_supported(p.seed) && gpu_blend_mode(layer.common.blend).is_some() => {
            (
                GpuKernel::Noise,
                GpuPassKind::Generator,
                GpuDirtyPolicy::Local,
                2,
            )
        }
        Island(p)
            if p.archetype == IslandArchetype::VolcanicHighIsland
                && seed_supported(p.seed)
                && gpu_blend_mode(layer.common.blend).is_some() =>
        {
            (
                GpuKernel::Shape,
                GpuPassKind::Generator,
                GpuDirtyPolicy::Local,
                2,
            )
        }
        Flat(_) | Ramp(_) | NoiseValue(_) | NoisePerlin(_) | Mountains(_) | Dunes(_)
        | Canyons(_) | DomainWarp(_) | SculptBase(_) | Mesa(_) | Volcano(_) | Island(_)
        | Plateau(_) | Uplift(_) => return None,
        Fbm(_) | Ridged(_) => return None,
        Blur(p) if inplace_composite_supported(layer) => (
            GpuKernel::Blur,
            GpuPassKind::Filter,
            GpuDirtyPolicy::Local,
            p.radius.max(1).min(16),
        ),
        EffectFilter(p)
            if matches!(p.kind, EffectFilterKind::Smooth | EffectFilterKind::Inflate)
                && inplace_composite_supported(layer) =>
        {
            (
                GpuKernel::EffectFilter,
                GpuPassKind::Filter,
                GpuDirtyPolicy::Local,
                p.radius.max(1).min(16),
            )
        }
        Blur(_) | EffectFilter(_) => return None,
        Terrace(_) if inplace_composite_supported(layer) => (
            GpuKernel::Terrace,
            GpuPassKind::Filter,
            GpuDirtyPolicy::Local,
            4,
        ),
        Terrace(_) => return None,
        ThermalErosion(p) if thermal_config_supported(p) && inplace_composite_supported(layer) => (
            GpuKernel::Thermal,
            GpuPassKind::Simulation,
            GpuDirtyPolicy::FullField,
            0,
        ),
        HydraulicErosion(p)
            if hydraulic_config_supported(p) && inplace_composite_supported(layer) =>
        {
            (
                GpuKernel::Hydraulic,
                GpuPassKind::Simulation,
                GpuDirtyPolicy::FullField,
                0,
            )
        }
        ThermalErosion(_) | HydraulicErosion(_) | RiverCarve(_) => return None,
        // These CPU operations either modify height without a GPU kernel or publish
        // observable auxiliary fields that the GPU preview cannot currently produce.
        Coastal(_) | Materials(_) | Biomes(_) | Vegetation(_) => return None,
        _ => return None,
    };
    Some(GpuLayerPlan {
        kind,
        kernel,
        dirty_policy,
        halo_texels,
    })
}

/// Whether this complete layer configuration can run on the GPU preview path.
pub fn layer_gpu_supported(layer: &Layer, mask_assets: &[MaskAsset]) -> bool {
    gpu_plan_for_layer(layer, mask_assets).is_some()
}

/// Compile the preview stack into a GPU pass list.
///
/// Stops recording passes at the first unsupported layer (sets `cpu_from`).
pub fn compile_gpu_graph(stack: &LayerStack, mask_assets: &[MaskAsset]) -> GpuComputeGraph {
    let layers: Vec<&Layer> = stack.flatten_layers();
    let mut graph = GpuComputeGraph::default();
    for (flat_index, layer) in layers.iter().enumerate() {
        if !layer.common.enabled {
            continue;
        }
        let Some(plan) = gpu_plan_for_layer(layer, mask_assets) else {
            graph.cpu_from = Some(flat_index);
            break;
        };
        graph.passes.push(GpuPass {
            layer_id: layer.id(),
            kind: plan.kind,
            kernel: plan.kernel,
            dirty_policy: plan.dirty_policy,
            flat_index,
            halo_texels: plan.halo_texels,
        });
    }
    graph
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

/// Map pass dirty policy to core `DirtyClass`.
pub fn dirty_class_for(policy: GpuDirtyPolicy) -> DirtyClass {
    match policy {
        GpuDirtyPolicy::Local => DirtyClass::Local,
        GpuDirtyPolicy::FullField => DirtyClass::BasinDependent,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use terra_core::layer::{
        BiomesParams, BlurParams, CoastalParams, DomainWarpParams, DuneParams, EffectFilterKind,
        EffectFilterParams, FbmParams, FlatParams, FractalNoiseType, HydraulicErosionParams, Layer,
        LayerKind, LayerStack, LayerTypeRegistry, MaterialsParams, MountainParams, NoiseParams,
        PlateauParams, RiverCarveParams, TerraceParams, ThermalErosionParams, VegetationParams,
    };
    use terra_core::mask::{
        bake_distribution, bake_mask_assets, DistributionEntry, MaskId, MaskOp, MaskRef,
    };

    fn single_layer_graph(layer: Layer) -> GpuComputeGraph {
        let mut stack = LayerStack::new();
        stack.push(layer);
        compile_gpu_graph(&stack, &[])
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
        ]
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
        assert!(g.passes.len() >= 2);
    }

    #[test]
    fn expand_dirty_respects_bounds() {
        let r = expand_dirty_rect((10, 10, 20, 20), 8, 100, 100);
        assert_eq!(r, (2, 2, 36, 36));
        let edge = expand_dirty_rect((0, 0, 5, 5), 8, 100, 100);
        assert_eq!(edge.0, 0);
        assert_eq!(edge.1, 0);
    }

    /// Revert check for #48: support depends on the authored fractal noise family.
    #[test]
    fn fractal_noise_support_rejects_open_simplex_without_substitution() {
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
                assert!(!layer_gpu_supported(&layer, &[]));
                let graph = single_layer_graph(layer);
                assert!(!graph.fully_gpu());
                assert_eq!(graph.cpu_from, Some(0));
            }
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
            assert!(graph.passes.is_empty());
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
            if supported {
                assert_eq!(graph.passes.len(), 1, "{}", meta.type_id);
                assert!(
                    graph.passes[0].kernel.matches_layer_kind(&layer.kind),
                    "planner selected {:?} for {}",
                    graph.passes[0].kernel,
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
            let supported = matches!(kind, EffectFilterKind::Smooth | EffectFilterKind::Inflate);
            let plan = gpu_plan_for_layer(&layer, &[]);
            assert_eq!(plan.is_some(), supported, "{}", kind.label());
            if let Some(plan) = plan {
                assert_eq!(plan.kernel, GpuKernel::EffectFilter, "{}", kind.label());
                assert!(plan.kernel.matches_layer_kind(&layer.kind));
            }
        }
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
        let cases = [
            Layer::new("high seed", LayerKind::NoiseValue(high_seed)),
            Layer::new(
                "domain warp",
                LayerKind::DomainWarp(DomainWarpParams::default()),
            ),
            Layer::new("mountains", LayerKind::Mountains(MountainParams::default())),
            Layer::new("dunes", LayerKind::Dunes(DuneParams::default())),
            Layer::new("plateau", LayerKind::Plateau(PlateauParams::default())),
            Layer::new(
                "D-infinity river",
                LayerKind::RiverCarve(RiverCarveParams::default()),
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
        assert_eq!(graph.passes.len(), 1);
        assert_eq!(graph.passes[0].flat_index, 0);
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
                assert_eq!(graph.passes.len(), 1, "{reason}");
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
            assert!(graph.passes.is_empty(), "{reason}");
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
