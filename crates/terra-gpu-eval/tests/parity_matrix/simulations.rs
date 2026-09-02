use super::*;

#[test]
fn gpu_required_simulation_previews_have_bounded_full_field_error() {
    let metrics = HeightfieldMetrics::new(24, 24, 48.0, 36.0);
    for (name, kind, tolerance) in [
        (
            "thermal",
            LayerKind::ThermalErosion(ThermalErosionParams {
                iterations: 2,
                layered_materials: false,
                weathering_rate: 0.0,
                ..ThermalErosionParams::default()
            }),
            THERMAL_PREVIEW,
        ),
        (
            "hydraulic",
            LayerKind::HydraulicErosion(HydraulicErosionParams {
                iterations: 2,
                particle_density: 0.0,
                layered_materials: false,
                ..HydraulicErosionParams::default()
            }),
            HYDRAULIC_PREVIEW,
        ),
    ] {
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "pattern",
            LayerKind::SculptBase(patterned_sculpt(24, 24)),
        ));
        stack.push(Layer::new(name, kind));
        let cpu = cpu_oracle(&stack, &[], metrics);
        let gpu = gpu_eval(&stack, &[], metrics);
        assert_field_parity(name, &gpu, &cpu, tolerance);
    }
}

#[test]
fn gpu_required_river_carve_d8_and_dinfinity_previews_are_bounded() {
    const RES: u32 = 12;
    let metrics = HeightfieldMetrics::new(RES, RES, 120.0, 120.0);
    let center = (RES as f32 - 1.0) * 0.5;
    let samples: Vec<f32> = (0..RES)
        .flat_map(|y| {
            (0..RES).map(move |x| {
                // An open, monotone V-shaped drainage basin avoids depression-fill
                // ambiguity while exercising channel convergence and overlapping banks.
                180.0 - y as f32 * 3.0 + (x as f32 - center).abs() * 1.5 + x as f32 * 0.01
            })
        })
        .collect();

    for use_dinfinity in [false, true] {
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "drainage basin",
            LayerKind::SculptBase(SculptParams {
                width: RES,
                height: RES,
                samples: samples.clone(),
                fill_height: 0.0,
            }),
        ));
        stack.push(Layer::new(
            "river carve",
            LayerKind::RiverCarve(RiverCarveParams {
                accumulation_threshold: 3.0,
                depth: 2.0,
                width: 1.5,
                bank_smooth: 0.4,
                use_dinfinity,
                ..RiverCarveParams::default()
            }),
        ));
        let cpu = cpu_oracle(&stack, &[], metrics);
        let gpu = gpu_eval(&stack, &[], metrics);
        let (contract, tolerance) = if use_dinfinity {
            (
                "simulation.river-carve.d-infinity",
                RIVER_CARVE_DINFINITY_PREVIEW,
            )
        } else {
            ("simulation.river-carve.d8", RIVER_CARVE_D8_PREVIEW)
        };
        assert_field_parity(contract, &gpu, &cpu, tolerance);
    }
}

#[test]
fn gpu_required_stream_power_d8_and_dinfinity_previews_are_bounded() {
    const RES: u32 = 12;
    let metrics = HeightfieldMetrics::new(RES, RES, 120.0, 120.0);
    let center = (RES as f32 - 1.0) * 0.5;
    let samples: Vec<f32> = (0..RES)
        .flat_map(|y| {
            (0..RES).map(move |x| {
                180.0 - y as f32 * 3.0 + (x as f32 - center).abs() * 1.5 + x as f32 * 0.01
            })
        })
        .collect();

    for use_dinfinity in [false, true] {
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "drainage basin",
            LayerKind::SculptBase(SculptParams {
                width: RES,
                height: RES,
                samples: samples.clone(),
                fill_height: 0.0,
            }),
        ));
        stack.push(Layer::new(
            "stream power",
            LayerKind::StreamPowerErosion(StreamPowerParams {
                iterations: 3,
                k: 0.002,
                m: 0.5,
                n: 1.0,
                dt: 0.75,
                uplift_rate: 0.05,
                base_level: 100.0,
                hardness: 0.2,
                use_dinfinity,
                ..StreamPowerParams::default()
            }),
        ));
        let cpu = cpu_oracle(&stack, &[], metrics);
        let gpu = gpu_eval(&stack, &[], metrics);
        let (contract, tolerance) = if use_dinfinity {
            (
                "simulation.stream-power.d-infinity",
                STREAM_POWER_DINFINITY_PREVIEW,
            )
        } else {
            ("simulation.stream-power.d8", STREAM_POWER_D8_PREVIEW)
        };
        assert_field_parity(contract, &gpu, &cpu, tolerance);
        assert!(
            gpu.to_dense().iter().all(|height| height.is_finite()),
            "{contract} produced a non-finite height"
        );
        assert!(
            gpu.to_dense().iter().all(|height| *height >= 100.0),
            "{contract} crossed the authored base level"
        );
        assert!(
            gpu.to_dense()
                .iter()
                .zip(&samples)
                .any(|(after, before)| after < before),
            "{contract} must incise at least one texel"
        );
    }
}

#[test]
fn gpu_required_multi_scale_amplify_preview_is_bounded_across_two_levels() {
    const RES: u32 = 128;
    let metrics = HeightfieldMetrics::new(RES, RES, 512.0, 384.0);
    let center = (RES as f32 - 1.0) * 0.5;
    let samples: Vec<f32> = (0..RES)
        .flat_map(|y| {
            (0..RES).map(move |x| {
                let basin = 220.0 - y as f32 * 0.45 + (x as f32 - center).abs() * 0.22;
                let ripple = ((x * 7 + y * 3) % 11) as f32 * 0.15;
                basin + ripple
            })
        })
        .collect();
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "open basin",
        LayerKind::SculptBase(SculptParams {
            width: RES,
            height: RES,
            samples,
            fill_height: 0.0,
        }),
    ));
    stack.push(Layer::new(
        "multi scale",
        LayerKind::MultiScaleAmplify(MultiScaleAmplifyParams {
            level_count: 2,
            thermal_iters: 2,
            spe_iters: 1,
            hardness: 0.2,
            ..MultiScaleAmplifyParams::default()
        }),
    ));

    let cpu = cpu_oracle(&stack, &[], metrics);
    let gpu = gpu_eval(&stack, &[], metrics);
    assert_field_parity(
        "simulation.multi-scale-amplify",
        &gpu,
        &cpu,
        MULTI_SCALE_AMPLIFY_PREVIEW,
    );
    assert!(gpu.to_dense().iter().all(|height| height.is_finite()));
}
