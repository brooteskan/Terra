use std::collections::HashMap;

use terra_core::biome_paint::ShapeTransform;
use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
use terra_core::layer::{
    BindingSource, BlendMode, BlurParams, CanyonParams, DomainWarpParams, DuneParams,
    EffectFilterKind, EffectFilterParams, FbmParams, FlatParams, FractalNoiseType,
    HydraulicErosionParams, ImportHeightmapParams, IslandParams, LandscapeEvolutionParams, Layer,
    LayerGroup, LayerKind, LayerStack, MesaParams, MountainParams, MultiScaleAmplifyParams,
    NoiseParams, ParamBinding, PathNode, PathParams, PlateauParams, PolygonHeightMode,
    PolygonHeightParams, ProceduralGenerator, ProceduralShapeParams, RampParams, RiverCarveParams,
    SculptParams, SculptPoint, SculptStroke, SculptStrokeKind, SculptStrokeParams, StackNode,
    Stamp2dParams, StreamPowerParams, TerraceParams, ThermalErosionParams, UpliftParams,
    VolcanoParams, VoronoiParams,
};
use terra_core::mask::{bake_mask_assets, MaskAsset, MaskId, MaskRef, MaskSource};
use terra_core::quality::PreviewQuality;
use terra_cpu_eval::{EvalContext, StackEvaluator};
use terra_gpu::parity::{
    assert_field_parity, ADD_SET_FILTER_PREVIEW, ARCHIPELAGO_PREVIEW, ATOLL_PREVIEW, BLUR_PREVIEW,
    CANYONS_PREVIEW, CURVE_FILTER_PREVIEW, CUTOFF_FILTER_PREVIEW, DEFLATE_FILTER_PREVIEW,
    DENOISE_FILTER_PREVIEW, DOMAIN_WARP_PREVIEW, DUNES_PREVIEW, EFFECT_FILTER_EXACT_PREVIEW,
    EFFECT_FILTER_SPATIAL_PREVIEW, EFFECT_FILTER_WARP_PREVIEW, EXACT_HEIGHT, FBM_PERLIN_PREVIEW,
    FBM_VALUE_PREVIEW, HEIGHTMAP_SAMPLE_PREVIEW, HYDRAULIC_PREVIEW, INFLATE_FILTER_PREVIEW,
    MESA_PREVIEW, MOUNTAINS_PREVIEW, MULTI_SCALE_AMPLIFY_PREVIEW, PATH_HEIGHT_PREVIEW,
    PERLIN_NOISE_PREVIEW, PLATEAU_PREVIEW, POLYGON_HEIGHT_PREVIEW, RIDGED_PERLIN_PREVIEW,
    RIDGED_VALUE_PREVIEW, RIVER_CARVE_D8_PREVIEW, RIVER_CARVE_DINFINITY_PREVIEW,
    SCULPT_STROKES_PREVIEW, SIMPLE_MASK, SMOOTH_FILTER_PREVIEW, STREAM_POWER_D8_PREVIEW,
    STREAM_POWER_DINFINITY_PREVIEW, TERRACE_PREVIEW, THERMAL_PREVIEW, UPLIFT_PREVIEW,
    VALUE_NOISE_PREVIEW, VOLCANIC_ISLAND_PREVIEW, VOLCANO_PREVIEW, VORONOI_REGIONS_PREVIEW,
};
use terra_gpu_eval::GpuTerrainEngine;

const QUALITY: PreviewQuality = PreviewQuality::Draft;

struct TempHeightmap(std::path::PathBuf);

impl TempHeightmap {
    fn new() -> Self {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("terra-heightmap-{unique}.png"));
        let image = image::ImageBuffer::from_fn(7, 5, |x, y| {
            image::Luma([((x * 7000 + y * 9000) % 65536) as u16])
        });
        image.save(&path).expect("write heightmap fixture");
        Self(path)
    }

    fn path(&self) -> String {
        self.0.to_string_lossy().into_owned()
    }
}

impl Drop for TempHeightmap {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn cpu_oracle(
    stack: &LayerStack,
    assets: &[MaskAsset],
    metrics: HeightfieldMetrics,
) -> Heightfield {
    let mut evaluator = StackEvaluator::new();
    let mut context = EvalContext::new(metrics);
    context.quality = QUALITY;
    context.mask_assets = assets.to_vec();
    context.masks = bake_mask_assets(
        assets,
        &Heightfield::zeros(metrics),
        metrics,
        &HashMap::new(),
    );
    evaluator
        .rebuild_all(stack, &mut context)
        .expect("CPU parity oracle")
}

fn gpu_eval(stack: &LayerStack, assets: &[MaskAsset], metrics: HeightfieldMetrics) -> Heightfield {
    let gpu = terra_test_gpu::headless_required();
    let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
    gpu_eval_with_engine(&mut engine, gpu, stack, assets, metrics)
}

fn gpu_eval_with_engine(
    engine: &mut GpuTerrainEngine,
    gpu: &terra_test_gpu::TestGpu,
    stack: &LayerStack,
    assets: &[MaskAsset],
    metrics: HeightfieldMetrics,
) -> Heightfield {
    engine.mark_all_dirty(stack);
    let result = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            stack,
            assets,
            metrics,
            QUALITY,
            true,
            None,
        )
        .expect("GPU parity evaluation");
    assert!(
        result.fully_gpu,
        "fixture unexpectedly selected CPU fallback"
    );
    assert_eq!(result.resume_cpu_from, None);
    result.cpu.expect("GPU-required readback")
}

fn patterned_sculpt(width: u32, height: u32) -> SculptParams {
    let samples = (0..height)
        .flat_map(|y| {
            (0..width).map(move |x| {
                let ramp = x as f32 * 0.7 + y as f32 * 0.35;
                let checker = if (x / 3 + y / 2) % 2 == 0 { 8.0 } else { -3.0 };
                20.0 + ramp + checker
            })
        })
        .collect();
    SculptParams {
        width,
        height,
        samples,
        fill_height: 0.0,
    }
}

#[path = "parity_matrix/generators_assets.rs"]
mod generators_assets;
#[path = "parity_matrix/masks_filters.rs"]
mod masks_filters;
#[path = "parity_matrix/shapes_sculpt.rs"]
mod shapes_sculpt;
#[path = "parity_matrix/simulations.rs"]
mod simulations;
