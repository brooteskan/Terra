use std::collections::HashMap;

use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
use terra_core::layer::{
    HydraulicErosionParams, Layer, LayerKind, LayerStack, MultiScaleAmplifyParams,
    RiverCarveParams, SculptParams, StreamPowerParams, ThermalErosionParams,
};
use terra_core::mask::bake_mask_assets;
use terra_core::quality::PreviewQuality;
use terra_cpu_eval::{EvalContext, StackEvaluator};
use terra_gpu::parity::{
    assert_field_parity, HYDRAULIC_PREVIEW, MULTI_SCALE_AMPLIFY_PREVIEW, RIVER_CARVE_D8_PREVIEW,
    STREAM_POWER_D8_PREVIEW, THERMAL_PREVIEW,
};
use terra_gpu_eval::{GpuSimulationStateReadback, GpuTerrainEngine};

const WIDTH: u32 = 17;
const HEIGHT: u32 = 11;
const QUALITY: PreviewQuality = PreviewQuality::Full;

#[derive(Clone, Copy, Debug)]
enum EdgeField {
    Flat,
    EpsilonSlope,
    IsolatedSpike,
    ClosedBasin,
    BoundaryRamp,
    NegativeElevation,
}

impl EdgeField {
    const ALL: [Self; 6] = [
        Self::Flat,
        Self::EpsilonSlope,
        Self::IsolatedSpike,
        Self::ClosedBasin,
        Self::BoundaryRamp,
        Self::NegativeElevation,
    ];

    fn samples(self) -> Vec<f32> {
        let center_x = (WIDTH / 2) as i32;
        let center_y = (HEIGHT / 2) as i32;
        (0..HEIGHT)
            .flat_map(|y| {
                (0..WIDTH).map(move |x| match self {
                    Self::Flat => 24.0,
                    Self::EpsilonSlope => -3.0 + x as f32 * 1.0e-5 + y as f32 * 5.0e-6,
                    Self::IsolatedSpike => {
                        if x as i32 == center_x && y as i32 == center_y {
                            220.0
                        } else {
                            8.0
                        }
                    }
                    Self::ClosedBasin => {
                        let dx = (x as i32 - center_x).abs() as f32;
                        let dy = (y as i32 - center_y).abs() as f32;
                        -18.0 + (dx + dy) * 4.0
                    }
                    Self::BoundaryRamp => 70.0 - x as f32 * 2.25 + y as f32 * 0.17,
                    Self::NegativeElevation => -140.0 + x as f32 * 0.8 - y as f32 * 0.35,
                })
            })
            .collect()
    }
}

fn metrics() -> HeightfieldMetrics {
    HeightfieldMetrics::new(WIDTH, HEIGHT, 51.0, 22.0)
}

fn stack_for(samples: &[f32], simulation: LayerKind) -> LayerStack {
    let mut stack = source_stack(samples);
    stack.push(Layer::new("simulation", simulation));
    stack
}

fn source_stack(samples: &[f32]) -> LayerStack {
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "edge input",
        LayerKind::SculptBase(SculptParams {
            width: WIDTH,
            height: HEIGHT,
            samples: samples.to_vec(),
            fill_height: 0.0,
        }),
    ));
    stack
}

fn cpu_oracle(stack: &LayerStack, field_metrics: HeightfieldMetrics) -> Heightfield {
    let mut evaluator = StackEvaluator::new();
    let mut context = EvalContext::new(field_metrics);
    context.quality = PreviewQuality::Draft;
    context.masks = bake_mask_assets(
        &[],
        &Heightfield::zeros(field_metrics),
        field_metrics,
        &HashMap::new(),
    );
    evaluator
        .rebuild_all(stack, &mut context)
        .expect("CPU simulation edge oracle")
}

fn gpu_eval(stack: &LayerStack) -> (Heightfield, GpuSimulationStateReadback) {
    let gpu = terra_test_gpu::headless_required();
    let mut engine = GpuTerrainEngine::new(&gpu.device, WIDTH);
    engine.mark_all_dirty(stack);
    let result = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            stack,
            &[],
            metrics(),
            QUALITY,
            true,
            None,
        )
        .expect("required GPU simulation edge evaluation");
    assert!(result.fully_gpu, "simulation fixture selected CPU fallback");
    assert!(!result.freshness.is_deferred());
    let height = result.cpu.expect("GPU simulation height readback");
    let state = engine
        .readback_simulation_state(&gpu.device, &gpu.queue)
        .expect("GPU simulation auxiliary readback");
    (height, state)
}

fn gpu_height(stack: &LayerStack) -> Heightfield {
    let gpu = terra_test_gpu::headless_required();
    let mut engine = GpuTerrainEngine::new(&gpu.device, WIDTH);
    engine.mark_all_dirty(stack);
    engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            stack,
            &[],
            metrics(),
            QUALITY,
            true,
            None,
        )
        .expect("required GPU input-boundary evaluation")
        .cpu
        .expect("GPU input-boundary readback")
}

fn assert_finite(name: &str, field: &Heightfield) {
    for (index, value) in field.to_dense().iter().enumerate() {
        assert!(
            value.is_finite(),
            "{name} contains non-finite state at ({},{})",
            index % WIDTH as usize,
            index / WIDTH as usize,
        );
    }
}

fn assert_nonnegative_finite(name: &str, field: &Heightfield) {
    assert_finite(name, field);
    for (index, value) in field.to_dense().iter().enumerate() {
        assert!(
            *value >= 0.0,
            "{name} contains negative state {value} at ({},{})",
            index % WIDTH as usize,
            index / WIDTH as usize,
        );
    }
}

fn sum(values: &[f32]) -> f64 {
    values.iter().map(|value| f64::from(*value)).sum()
}

fn assert_mass_close(name: &str, before: f64, after: f64, tolerance: f64) {
    let error = (after - before).abs();
    assert!(
        error <= tolerance,
        "{name} mass ledger drifted by {error}: before={before}, after={after}, tolerance={tolerance}"
    );
}

fn thermal(strength: f32, hardness: f32) -> LayerKind {
    LayerKind::ThermalErosion(ThermalErosionParams {
        iterations: 1,
        strength,
        hardness,
        layered_materials: false,
        weathering_rate: 0.0,
        ..ThermalErosionParams::default()
    })
}

fn hydraulic(
    rainfall: f32,
    evaporation: f32,
    erosion: f32,
    deposition: f32,
    hardness: f32,
) -> LayerKind {
    LayerKind::HydraulicErosion(HydraulicErosionParams {
        iterations: 1,
        rainfall,
        evaporation,
        erosion,
        deposition,
        capacity: 0.35,
        timestep: 0.2,
        hardness,
        particle_density: 0.0,
        layered_materials: false,
        ..HydraulicErosionParams::default()
    })
}

fn assert_hydraulic_state_finite(state: &GpuSimulationStateReadback) {
    assert_eq!(
        state.invalid_state_bits, 0,
        "hydraulic shader observed invalid raw state before its stability clamps"
    );
    assert_nonnegative_finite("hydraulic.water-a", &state.water_a);
    assert_nonnegative_finite("hydraulic.water-b", &state.water_b);
    assert_nonnegative_finite("hydraulic.sediment-a", &state.sediment_a);
    assert_nonnegative_finite("hydraulic.sediment-b", &state.sediment_b);
    assert_nonnegative_finite("hydraulic.hardness", &state.hardness);
    assert_nonnegative_finite("hydraulic.rainfall", &state.rainfall);
    assert_nonnegative_finite("hydraulic.loose-sediment", &state.loose_sediment);
    assert_finite("simulation.redistribution", &state.redistribution);
}

#[test]
fn gpu_required_thermal_edges_are_finite_and_mass_conservative() {
    for edge in EdgeField::ALL {
        let input = edge.samples();
        let entering = gpu_height(&source_stack(&input));
        let stack = stack_for(&input, thermal(1.0, 0.0));
        let (gpu, state) = gpu_eval(&stack);
        assert_finite(&format!("thermal.{edge:?}.height"), &gpu);
        assert_nonnegative_finite(&format!("thermal.{edge:?}.hardness"), &state.hardness);
        assert_nonnegative_finite(
            &format!("thermal.{edge:?}.redistribution"),
            &state.redistribution,
        );
        assert_mass_close(
            &format!("thermal.{edge:?}"),
            sum(&entering.to_dense()),
            sum(&gpu.to_dense()),
            2.5e-3,
        );
    }

    let input = EdgeField::IsolatedSpike.samples();
    let entering = gpu_height(&source_stack(&input));
    for (name, simulation) in [
        ("zero-effect", thermal(0.0, 0.0)),
        ("full-hardness", thermal(1.0, 1.0)),
    ] {
        let (gpu, _) = gpu_eval(&stack_for(&input, simulation));
        assert_eq!(
            gpu.to_dense(),
            entering.to_dense(),
            "thermal {name} endpoint changed height"
        );
    }
}

#[test]
fn gpu_required_hydraulic_edges_have_closed_mass_and_water_ledgers() {
    const RAIN: f32 = 0.08;
    const EVAP: f32 = 0.15;
    for edge in EdgeField::ALL {
        let input = edge.samples();
        let entering = gpu_height(&source_stack(&input));
        let stack = stack_for(&input, hydraulic(RAIN, EVAP, 0.4, 0.55, 0.0));
        let (gpu, state) = gpu_eval(&stack);
        assert_finite(&format!("hydraulic.{edge:?}.height"), &gpu);
        assert_hydraulic_state_finite(&state);

        // One Full-quality authored step writes the final water/sediment state to B.
        // The shader has closed boundaries: rainfall is the only water source and
        // evaporation the only sink; terrain plus suspended sediment is conservative.
        let sediment = state.sediment_b.to_dense();
        assert_mass_close(
            &format!("hydraulic.{edge:?}.terrain+sediment"),
            sum(&entering.to_dense()),
            sum(&gpu.to_dense()) + sum(&sediment),
            5.0e-3,
        );
        let expected_water = f64::from(RAIN * (1.0 - EVAP)) * f64::from(WIDTH * HEIGHT);
        assert_mass_close(
            &format!("hydraulic.{edge:?}.water"),
            expected_water,
            sum(&state.water_b.to_dense()),
            5.0e-3,
        );
    }

    let input = EdgeField::BoundaryRamp.samples();
    let entering = gpu_height(&source_stack(&input));
    for (name, simulation) in [
        ("zero-effect", hydraulic(0.0, 0.0, 0.0, 0.0, 0.0)),
        ("full-hardness", hydraulic(0.08, 0.0, 1.0, 1.0, 1.0)),
    ] {
        let (gpu, state) = gpu_eval(&stack_for(&input, simulation));
        assert_eq!(
            gpu.to_dense(),
            entering.to_dense(),
            "hydraulic {name} endpoint changed terrain"
        );
        assert_hydraulic_state_finite(&state);
    }
}

fn river(depth: f32) -> LayerKind {
    LayerKind::RiverCarve(RiverCarveParams {
        accumulation_threshold: 3.0,
        depth,
        width: 1.5,
        bank_smooth: 0.4,
        use_dinfinity: false,
        ..RiverCarveParams::default()
    })
}

fn stream_power(k: f32, hardness: f32) -> LayerKind {
    LayerKind::StreamPowerErosion(StreamPowerParams {
        iterations: 1,
        k,
        m: 0.5,
        n: 1.0,
        dt: 0.6,
        uplift_rate: 0.0,
        base_level: -500.0,
        hardness,
        use_dinfinity: false,
        ..StreamPowerParams::default()
    })
}

#[test]
fn gpu_required_open_sink_and_amplify_paths_account_for_numerical_edges() {
    for edge in [EdgeField::ClosedBasin, EdgeField::BoundaryRamp] {
        let input = edge.samples();
        let entering = gpu_height(&source_stack(&input));
        let stack = stack_for(&input, river(3.0));
        let (gpu, state) = gpu_eval(&stack);
        assert_finite(&format!("river.{edge:?}.height"), &gpu);
        assert_nonnegative_finite(&format!("river.{edge:?}.accum-a"), &state.water_a);
        assert_nonnegative_finite(&format!("river.{edge:?}.accum-b"), &state.water_b);
        let exported = sum(&entering.to_dense()) - sum(&gpu.to_dense());
        assert!(
            exported >= 0.0,
            "river carve imported terrain mass: {exported}"
        );
    }
    let river_input = EdgeField::BoundaryRamp.samples();
    let river_entering = gpu_height(&source_stack(&river_input));
    let (river_zero, _) = gpu_eval(&stack_for(&river_input, river(0.0)));
    assert_eq!(
        river_zero.to_dense(),
        river_entering.to_dense(),
        "zero-depth river changed height"
    );
    for edge in [
        EdgeField::Flat,
        EdgeField::EpsilonSlope,
        EdgeField::IsolatedSpike,
        EdgeField::NegativeElevation,
    ] {
        let input = edge.samples();
        let entering = gpu_height(&source_stack(&input));
        let (gpu, state) = gpu_eval(&stack_for(&input, stream_power(0.01, 0.0)));
        assert_finite(&format!("stream-power.{edge:?}.height"), &gpu);
        assert_nonnegative_finite(&format!("stream-power.{edge:?}.accum-a"), &state.water_a);
        assert_nonnegative_finite(&format!("stream-power.{edge:?}.accum-b"), &state.water_b);
        let exported = sum(&entering.to_dense()) - sum(&gpu.to_dense());
        assert!(
            exported >= -2.0e-3,
            "stream power imported mass: {exported}"
        );
    }
    let stream_input = EdgeField::BoundaryRamp.samples();
    let stream_entering = gpu_height(&source_stack(&stream_input));
    for (name, simulation) in [
        ("zero-effect", stream_power(0.0, 0.0)),
        ("full-hardness", stream_power(0.02, 1.0)),
    ] {
        let (gpu, _) = gpu_eval(&stack_for(&stream_input, simulation));
        assert_eq!(
            gpu.to_dense(),
            stream_entering.to_dense(),
            "stream-power {name} changed height"
        );
    }
    for edge in [EdgeField::IsolatedSpike, EdgeField::ClosedBasin] {
        let input = edge.samples();
        let stack = stack_for(
            &input,
            LayerKind::MultiScaleAmplify(MultiScaleAmplifyParams {
                level_count: 1,
                thermal_iters: 1,
                spe_iters: 1,
                thermal_strength: 0.65,
                spe_strength: 0.35,
                deposition_strength: 0.0,
                hardness: 0.2,
                ..MultiScaleAmplifyParams::default()
            }),
        );
        let (gpu, state) = gpu_eval(&stack);
        assert_finite(&format!("multi-scale.{edge:?}.height"), &gpu);
        assert_hydraulic_state_finite(&state);
    }
}

fn square_stack(side: u32, simulation: LayerKind) -> (LayerStack, HeightfieldMetrics) {
    let center = (side as f32 - 1.0) * 0.5;
    let samples = (0..side)
        .flat_map(|y| {
            (0..side).map(move |x| {
                180.0 - y as f32 * 3.0 + (x as f32 - center).abs() * 1.5 + x as f32 * 0.01
            })
        })
        .collect();
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "square parity input",
        LayerKind::SculptBase(SculptParams {
            width: side,
            height: side,
            samples,
            fill_height: 0.0,
        }),
    ));
    stack.push(Layer::new("simulation", simulation));
    (
        stack,
        HeightfieldMetrics::new(side, side, side as f32 * 10.0, side as f32 * 10.0),
    )
}

fn gpu_height_for(stack: &LayerStack, field_metrics: HeightfieldMetrics) -> Heightfield {
    let gpu = terra_test_gpu::headless_required();
    let mut engine = GpuTerrainEngine::new(&gpu.device, field_metrics.width);
    engine.mark_all_dirty(stack);
    let result = engine
        .evaluate(
            &gpu.device,
            &gpu.queue,
            stack,
            &[],
            field_metrics,
            PreviewQuality::Draft,
            true,
            None,
        )
        .expect("required GPU square parity evaluation");
    assert!(result.fully_gpu);
    result.cpu.expect("GPU square parity readback")
}

#[test]
fn gpu_required_simulation_edge_suite_retains_cpu_comparison() {
    for (name, simulation, tolerance) in [
        ("thermal", thermal(0.65, 0.2), THERMAL_PREVIEW),
        (
            "hydraulic",
            hydraulic(0.05, 0.1, 0.35, 0.5, 0.25),
            HYDRAULIC_PREVIEW,
        ),
        ("river-carve", river(2.0), RIVER_CARVE_D8_PREVIEW),
        (
            "stream-power",
            stream_power(0.003, 0.2),
            STREAM_POWER_D8_PREVIEW,
        ),
    ] {
        let (stack, field_metrics) = square_stack(12, simulation);
        let cpu = cpu_oracle(&stack, field_metrics);
        let gpu = gpu_height_for(&stack, field_metrics);
        assert_field_parity(&format!("simulation-edge.{name}"), &gpu, &cpu, tolerance);
    }

    let side = 128u32;
    let center = (side as f32 - 1.0) * 0.5;
    let samples = (0..side)
        .flat_map(|y| {
            (0..side).map(move |x| {
                let basin = 220.0 - y as f32 * 0.45 + (x as f32 - center).abs() * 0.22;
                let ripple = ((x * 7 + y * 3) % 11) as f32 * 0.15;
                basin + ripple
            })
        })
        .collect();
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "multi-scale parity input",
        LayerKind::SculptBase(SculptParams {
            width: side,
            height: side,
            samples,
            fill_height: 0.0,
        }),
    ));
    stack.push(Layer::new(
        "multi-scale",
        LayerKind::MultiScaleAmplify(MultiScaleAmplifyParams {
            level_count: 2,
            thermal_iters: 2,
            spe_iters: 1,
            thermal_strength: 0.65,
            spe_strength: 0.35,
            deposition_strength: 0.0,
            hardness: 0.2,
            ..MultiScaleAmplifyParams::default()
        }),
    ));
    let field_metrics = HeightfieldMetrics::new(side, side, 512.0, 384.0);
    let cpu = cpu_oracle(&stack, field_metrics);
    let gpu = gpu_height_for(&stack, field_metrics);
    assert_field_parity(
        "simulation-edge.multi-scale-amplify",
        &gpu,
        &cpu,
        MULTI_SCALE_AMPLIFY_PREVIEW,
    );
}
