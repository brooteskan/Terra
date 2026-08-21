use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
use terra_core::layer::{
    BlendMode, BlurParams, DomainWarpParams, EffectFilterParams, FlatParams, ImportHeightmapParams,
    Layer, LayerId, LayerKind, LayerStack, NoiseParams, SculptParams,
};
use terra_core::mask::{MaskAsset, MaskId, MaskRef, MaskSource};
use terra_core::quality::PreviewQuality;
use terra_gpu_eval::{GpuEvaluationIntent, GpuTerrainEngine};

const QUALITY: PreviewQuality = PreviewQuality::Draft;
const WIDTH: u32 = 47;
const HEIGHT: u32 = 31;
const PARTITIONS: [(u32, u32, u32, u32); 3] = [
    (0, 0, 7, 6),   // corner
    (41, 11, 6, 9), // right edge
    (19, 13, 9, 7), // center
];

#[derive(Clone, Copy)]
enum Equality {
    Exact(&'static str),
    Tolerance { max_abs: f32, reason: &'static str },
}

#[derive(Clone, Copy)]
enum FixtureKind {
    Pointwise,
    LocalHalo,
    Warp,
    Mask,
    Asset,
}

impl FixtureKind {
    const ALL: [Self; 5] = [
        Self::Pointwise,
        Self::LocalHalo,
        Self::Warp,
        Self::Mask,
        Self::Asset,
    ];

    const fn name(self) -> &'static str {
        match self {
            Self::Pointwise => "local-pointwise-zero-reach",
            Self::LocalHalo => "local-filter-bounded-halo",
            Self::Warp => "seeded-procedural-warp",
            Self::Mask => "height-mask-rebake",
            Self::Asset => "heightmap-asset-sampling",
        }
    }

    const fn equality(self) -> Equality {
        match self {
            Self::Pointwise => Equality::Exact(
                "pointwise kernels execute the same arithmetic for full and regional dispatch",
            ),
            Self::Warp => Equality::Exact(
                "the seeded warp samples coordinates only and is bit-stable across cache histories",
            ),
            Self::Asset => Equality::Exact(
                "the immutable source raster is sampled identically when its contribution cache is reused",
            ),
            Self::LocalHalo => Equality::Tolerance {
                max_abs: 0.125,
                reason: "regional blur reuses rounded warm intermediates; this remains far tighter than its 2.1 m GPU/CPU preview budget",
            },
            Self::Mask => Equality::Tolerance {
                max_abs: 1.0e-3,
                reason: "derived mask rebakes are compared at the evaluator's exact-height tolerance",
            },
        }
    }
}

struct TempHeightmap(std::path::PathBuf);

impl TempHeightmap {
    fn new() -> Self {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("terra-c2-g2-{unique}.png"));
        let image = image::ImageBuffer::from_fn(9, 7, |x, y| {
            image::Luma([((x * 5_911 + y * 8_123 + x * y * 379) % 65_536) as u16])
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

fn metrics() -> HeightfieldMetrics {
    HeightfieldMetrics::new(WIDTH, HEIGHT, 329.0, 124.0)
}

fn base_sculpt() -> SculptParams {
    let samples = (0..HEIGHT)
        .flat_map(|y| {
            (0..WIDTH).map(move |x| {
                14.0 + x as f32 * 0.19
                    + y as f32 * 0.31
                    + if (x * 3 + y * 5) % 11 == 0 {
                        3.75
                    } else {
                        -1.25
                    }
            })
        })
        .collect();
    SculptParams {
        width: WIDTH,
        height: HEIGHT,
        samples,
        fill_height: 0.0,
    }
}

fn edit_height(x: u32, y: u32) -> f32 {
    38.0 + x as f32 * 0.47 - y as f32 * 0.13 + ((x ^ y) % 7) as f32
}

fn sculpt_mut(stack: &mut LayerStack, id: LayerId) -> &mut SculptParams {
    let layer = stack.find_mut(id).expect("fixture source layer");
    let LayerKind::SculptBase(params) = &mut layer.kind else {
        panic!("fixture source changed kind");
    };
    params
}

fn apply_partition(stack: &mut LayerStack, id: LayerId, rect: (u32, u32, u32, u32)) {
    let params = sculpt_mut(stack, id);
    for y in rect.1..rect.1 + rect.3 {
        for x in rect.0..rect.0 + rect.2 {
            params.samples[(y * WIDTH + x) as usize] = edit_height(x, y);
        }
    }
}

fn build_fixture(kind: FixtureKind, asset_path: &str) -> (LayerStack, Vec<MaskAsset>, LayerId) {
    let mut stack = LayerStack::new();
    let source = Layer::new("editable source", LayerKind::SculptBase(base_sculpt()));
    let source_id = source.id();
    stack.push(source);
    let mut assets = Vec::new();

    match kind {
        FixtureKind::Pointwise => stack.push(Layer::new(
            "pointwise add/set",
            LayerKind::EffectFilter(EffectFilterParams::add_set()),
        )),
        FixtureKind::LocalHalo => stack.push(Layer::new(
            "bounded blur",
            LayerKind::Blur(BlurParams {
                radius: 4,
                iterations: 1,
            }),
        )),
        FixtureKind::Warp => {
            // Highest base seed whose three-octave 1013-stride stream still fits
            // the shader's documented 32-bit transport.
            let seed = u64::from(u32::MAX) - 2 * 1_013;
            let mut warp = Layer::new(
                "seed-boundary warp",
                LayerKind::DomainWarp(DomainWarpParams {
                    base: NoiseParams {
                        seed,
                        octaves: 3,
                        amplitude: 9.0,
                        frequency: 0.021,
                        ..NoiseParams::default()
                    },
                    warp_strength: 17.0,
                    warp_frequency: 0.013,
                }),
            );
            warp.common.blend = BlendMode::Add;
            stack.push(warp);
        }
        FixtureKind::Mask => {
            let mask = MaskAsset::new(
                MaskId::new(),
                "entering height",
                MaskSource::Height {
                    min: 15.0,
                    max: 35.0,
                },
            );
            let mut masked = Layer::new("masked add", LayerKind::Flat(FlatParams { height: 6.0 }));
            masked.common.blend = BlendMode::Add;
            masked.common.masks.push(MaskRef::new(mask.id));
            stack.push(masked);
            assets.push(mask);
        }
        FixtureKind::Asset => {
            let mut imported = Layer::new(
                "sampled asset",
                LayerKind::ImportHeightmap(ImportHeightmapParams {
                    path: asset_path.to_owned(),
                    height_scale: 12.0,
                    height_offset: -3.0,
                }),
            );
            imported.common.blend = BlendMode::Add;
            stack.push(imported);
        }
    }
    (stack, assets, source_id)
}

fn evaluate(
    engine: &mut GpuTerrainEngine,
    gpu: &terra_test_gpu::TestGpu,
    stack: &LayerStack,
    assets: &[MaskAsset],
    intent: GpuEvaluationIntent,
) -> terra_gpu_eval::GpuEvalResult {
    engine
        .evaluate_with_intent(
            &gpu.device,
            &gpu.queue,
            stack,
            assets,
            metrics(),
            QUALITY,
            true,
            None,
            intent,
        )
        .expect("required GPU determinism evaluation")
}

fn complete_height(
    engine: &mut GpuTerrainEngine,
    gpu: &terra_test_gpu::TestGpu,
    stack: &LayerStack,
    assets: &[MaskAsset],
) -> Heightfield {
    let result = evaluate(engine, gpu, stack, assets, GpuEvaluationIntent::Complete);
    assert!(
        result.fully_gpu,
        "fixture unexpectedly selected CPU fallback"
    );
    assert_eq!(result.resume_cpu_from, None);
    assert!(
        !result.freshness.is_deferred(),
        "complete evaluation was deferred"
    );
    result.cpu.expect("required GPU readback")
}

fn compare(name: &str, lhs: &Heightfield, rhs: &Heightfield, equality: Equality) {
    let lhs = lhs.to_dense();
    let rhs = rhs.to_dense();
    assert_eq!(lhs.len(), rhs.len(), "{name}: shape mismatch");
    match equality {
        Equality::Exact(reason) => {
            assert_eq!(lhs, rhs, "{name}: expected exact equality: {reason}")
        }
        Equality::Tolerance { max_abs, reason } => {
            let (index, error) = lhs
                .iter()
                .zip(&rhs)
                .enumerate()
                .map(|(index, (a, b))| (index, (*a - *b).abs()))
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap_or((0, 0.0));
            assert!(
                error <= max_abs,
                "{name}: max_abs {error} at ({},{}), tolerance {max_abs}: {reason}",
                index % WIDTH as usize,
                index / WIDTH as usize,
            );
        }
    }
}

#[test]
fn gpu_required_determinism_and_partition_equivalence_matrix() {
    let gpu = terra_test_gpu::headless_required();
    let heightmap = TempHeightmap::new();

    for kind in FixtureKind::ALL {
        let (initial, assets, source_id) = build_fixture(kind, &heightmap.path());
        let mut final_stack = initial.clone();
        for rect in PARTITIONS {
            apply_partition(&mut final_stack, source_id, rect);
        }

        let mut cold_a_engine = GpuTerrainEngine::new(&gpu.device, WIDTH);
        cold_a_engine.mark_all_dirty(&final_stack);
        let cold_a = complete_height(&mut cold_a_engine, gpu, &final_stack, &assets);

        let mut cold_b_engine = GpuTerrainEngine::new(&gpu.device, WIDTH);
        cold_b_engine.mark_all_dirty(&final_stack);
        let cold_b = complete_height(&mut cold_b_engine, gpu, &final_stack, &assets);
        compare(
            &format!("{}.cold", kind.name()),
            &cold_a,
            &cold_b,
            kind.equality(),
        );

        cold_a_engine.set_dirty_rect(None);
        cold_a_engine.mark_all_dirty(&final_stack);
        let repeated = complete_height(&mut cold_a_engine, gpu, &final_stack, &assets);
        compare(
            &format!("{}.repeated-warm", kind.name()),
            &cold_a,
            &repeated,
            kind.equality(),
        );

        let mut warm_engine = GpuTerrainEngine::new(&gpu.device, WIDTH);
        warm_engine.mark_all_dirty(&initial);
        let _ = complete_height(&mut warm_engine, gpu, &initial, &assets);
        warm_engine.set_dirty_rect(None);
        warm_engine.mark_dirty(source_id);
        let warm = complete_height(&mut warm_engine, gpu, &final_stack, &assets);
        compare(
            &format!("{}.warm-full", kind.name()),
            &cold_a,
            &warm,
            kind.equality(),
        );

        let mut partitioned_stack = initial.clone();
        let mut partitioned_engine = GpuTerrainEngine::new(&gpu.device, WIDTH);
        partitioned_engine.mark_all_dirty(&partitioned_stack);
        let mut partitioned =
            complete_height(&mut partitioned_engine, gpu, &partitioned_stack, &assets);
        for rect in PARTITIONS {
            apply_partition(&mut partitioned_stack, source_id, rect);
            partitioned_engine.set_dirty_rect(Some(rect));
            partitioned_engine.mark_dirty(source_id);
            partitioned =
                complete_height(&mut partitioned_engine, gpu, &partitioned_stack, &assets);
        }
        compare(
            &format!("{}.partitioned", kind.name()),
            &cold_a,
            &partitioned,
            kind.equality(),
        );
    }
}

#[test]
fn gpu_required_full_field_partition_mode_is_explicitly_deferred() {
    let gpu = terra_test_gpu::headless_required();
    let mut stack = LayerStack::new();
    let source = Layer::new("editable source", LayerKind::SculptBase(base_sculpt()));
    let source_id = source.id();
    stack.push(source);
    stack.push(Layer::new(
        "full-field curve",
        LayerKind::EffectFilter(EffectFilterParams::curve()),
    ));

    let mut engine = GpuTerrainEngine::new(&gpu.device, WIDTH);
    engine.mark_all_dirty(&stack);
    let _ = complete_height(&mut engine, gpu, &stack, &[]);
    apply_partition(&mut stack, source_id, PARTITIONS[2]);
    engine.set_dirty_rect(Some(PARTITIONS[2]));
    engine.mark_dirty(source_id);
    let local = evaluate(
        &mut engine,
        gpu,
        &stack,
        &[],
        GpuEvaluationIntent::InteractiveLocal,
    );
    assert!(
        local.freshness.is_deferred(),
        "full-field kernels must explicitly defer partitioned interactive execution"
    );
    assert!(
        !local.fully_gpu,
        "a deferred full-field suffix must not be reported as a complete GPU result"
    );

    engine.set_dirty_rect(None);
    let completed = complete_height(&mut engine, gpu, &stack, &[]);
    let mut oracle_engine = GpuTerrainEngine::new(&gpu.device, WIDTH);
    oracle_engine.mark_all_dirty(&stack);
    let oracle = complete_height(&mut oracle_engine, gpu, &stack, &[]);
    compare(
        "full-field.explicit-completion",
        &completed,
        &oracle,
        Equality::Tolerance {
            max_abs: 1.0e-3,
            reason: "range-reduced full-field remaps use the evaluator exact-height tolerance",
        },
    );
}
