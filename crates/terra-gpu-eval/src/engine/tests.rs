use super::*;
use std::collections::HashMap;
use terra_core::heightfield::HeightfieldMetrics;
use terra_core::layer::{
    BindingSource, BlendMode, BlurParams, CoastalParams, DomainWarpParams, EffectFilterParams,
    FbmParams, FlatParams, FractalNoiseType, GroupInputMode, ImportHeightmapParams, IslandParams,
    Layer, LayerGroup, LayerKind, LayerStack, MaterialsParams, MultiScaleAmplifyParams,
    NamedOutputDecl, NoiseParams, ParamBinding, RiverCarveParams, SculptParams, StackNode,
    Stamp2dParams, StreamPowerParams, ThermalErosionParams, VoronoiParams,
};
use terra_core::layer::{BrushDab, BrushEditable, SculptStrokeKind};
use terra_core::mask::{
    bake_mask_assets, DistributionEntry, MaskAsset, MaskCombine, MaskId, MaskOp, MaskRef,
    MaskSource,
};
use terra_core::quality::PreviewQuality;
use terra_core::terrain_plan::{PlanInvalidation, TerrainPlanCache};
use terra_core::test_fixtures::{untitled6_document, Untitled6Variant};
use terra_core::tiling::UvRect;
use terra_cpu_eval::{EvalContext, StackEvaluator};

fn cpu_oracle(stack: &LayerStack, metrics: HeightfieldMetrics) -> Heightfield {
    let mut evaluator = StackEvaluator::new();
    let mut ctx = EvalContext::new(metrics);
    evaluator
        .rebuild_all(stack, &mut ctx)
        .expect("CPU stack oracle")
}

fn cpu_mask_oracle(
    stack: &LayerStack,
    metrics: HeightfieldMetrics,
    assets: &[MaskAsset],
) -> Heightfield {
    let mut evaluator = StackEvaluator::new();
    let mut ctx = EvalContext::new(metrics);
    ctx.masks = bake_mask_assets(
        assets,
        &Heightfield::zeros(metrics),
        metrics,
        &HashMap::new(),
    );
    ctx.mask_assets = assets.to_vec();
    evaluator
        .rebuild_all(stack, &mut ctx)
        .expect("CPU mask oracle")
}

fn masked_probe_stack(source: MaskSource) -> (LayerStack, MaskAsset, LayerId) {
    let resolution = 32;
    let samples = (0..resolution)
        .flat_map(|j| (0..resolution).map(move |i| i as f32 * 0.5 + j as f32 * 0.25))
        .collect();
    let sculpt = SculptParams {
        width: resolution,
        height: resolution,
        samples,
        fill_height: 0.0,
    };
    let asset = MaskAsset::new(MaskId::new(), "probe mask", source);
    let mut probe = Layer::new("unit probe", LayerKind::Flat(FlatParams { height: 1.0 }));
    probe.common.blend = BlendMode::Add;
    let mut binding = MaskRef::new(asset.id);
    binding.strength = 0.65;
    binding.invert = true;
    probe.common.masks.push(binding);

    let mut stack = LayerStack::new();
    let base = Layer::new("authored base", LayerKind::SculptBase(sculpt));
    let base_id = base.id();
    stack.push(base);
    stack.push(probe);
    (stack, asset, base_id)
}

/// Spatially varying sculpt data shared by execution and dirty-region fixtures.
fn varied_sculpt(resolution: u32) -> SculptParams {
    let mut sculpt = SculptParams::filled(resolution, 20.0);
    for y in 0..resolution {
        for x in 0..resolution {
            let fx = x as f32;
            let fy = y as f32;
            sculpt.samples[(y * resolution + x) as usize] =
                20.0 + 12.0 * (fx * 0.35).sin() + 10.0 * (fy * 0.27).cos();
        }
    }
    sculpt
}

/// A single raised brush dab shared by cache and dirty-region fixtures.
fn raise_strokes(u: f32, v: f32, strength: f32) -> SculptStrokeParams {
    SculptStrokeParams {
        strokes: vec![terra_core::layer::SculptStroke {
            kind: SculptStrokeKind::Raise,
            points: vec![terra_core::layer::SculptPoint {
                u,
                v,
                pressure: 1.0,
            }],
            radius_m: 60.0,
            strength,
            target_height: 0.0,
            falloff: 1.5,
            enabled: true,
        }],
        reconcile: 0.15,
    }
}

#[path = "tests/cache_reuse.rs"]
mod cache_reuse;
#[path = "tests/compilation_fallback.rs"]
mod compilation_fallback;
#[path = "tests/dirty_regions.rs"]
mod dirty_regions;
#[path = "tests/dirty_regions_smooth.rs"]
mod dirty_regions_smooth;
#[path = "tests/execution_resources.rs"]
mod execution_resources;
#[path = "tests/masks_composition.rs"]
mod masks_composition;
#[path = "tests/output_identity.rs"]
mod output_identity;
#[path = "tests/refinement_performance.rs"]
mod refinement_performance;
