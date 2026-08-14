//! CPU-only numerical stability contract for the erosion simulation family.
//!
//! This complements the qualitative fixtures with executable invariants for
//! finite state, conservative transfers, boundary semantics, degenerate limits,
//! and typed material continuity through `StackEvaluator`.

use std::collections::HashMap;

use terra_core::analyze::{
    debris_flow_erode, thermal_erode_layered, thermal_erode_with_hardness, HydraulicErosionCore,
    MassWastingState,
};
use terra_core::eval::{EvalContext, StackEvaluator};
use terra_core::fields::{keys, AuxMaps};
use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
use terra_core::hydro::stream_power_erode;
use terra_core::landscape_evolution::{
    BoundaryMode, EvolutionSolverMode, LandscapeEvolutionInput, LandscapeEvolutionOperator,
    LandscapeEvolutionOutput, LandscapeEvolutionParams, UpliftMode,
};
use terra_core::layer::{
    DebrisFlowParams, FlatParams, HydraulicErosionParams, Layer, LayerId, LayerKind, LayerStack,
    StreamPowerParams, ThermalErosionParams, TransportModel,
};
use terra_core::mask::MaskField;

fn metrics(resolution: u32) -> HeightfieldMetrics {
    HeightfieldMetrics::new(
        resolution,
        resolution,
        resolution as f32 * 2.0,
        resolution as f32 * 2.0,
    )
}

fn flat(resolution: u32, height: f32) -> Heightfield {
    Heightfield::filled(metrics(resolution), height)
}

fn epsilon_slope(resolution: u32) -> Heightfield {
    let m = metrics(resolution);
    let mut height = Heightfield::filled(m, 10.0);
    for j in 0..resolution {
        for i in 0..resolution {
            height.set(i, j, 10.0 + i as f32 * 1.0e-5 + j as f32 * 5.0e-6);
        }
    }
    height
}

fn spike_cliff(resolution: u32) -> Heightfield {
    let m = metrics(resolution);
    let mut height = Heightfield::filled(m, 8.0);
    let center = resolution / 2;
    height.set(center, center, 48.0);
    for j in 0..resolution {
        for i in 0..resolution / 4 {
            height.set(i, j, 30.0);
        }
    }
    height
}

fn closed_basin(resolution: u32) -> Heightfield {
    let m = metrics(resolution);
    let center = (resolution - 1) as f32 * 0.5;
    let mut height = Heightfield::zeros(m);
    for j in 0..resolution {
        for i in 0..resolution {
            let radius = (i as f32 - center).hypot(j as f32 - center);
            height.set(i, j, 5.0 + radius * 0.8);
        }
    }
    height
}

fn edge_ramp(resolution: u32, edge: usize) -> Heightfield {
    let m = metrics(resolution);
    let mut height = Heightfield::zeros(m);
    for j in 0..resolution {
        for i in 0..resolution {
            let coordinate = match edge {
                0 => i,
                1 => resolution - 1 - i,
                2 => j,
                3 => resolution - 1 - j,
                _ => unreachable!("edge index must be 0..4"),
            };
            height.set(i, j, 50.0 - coordinate as f32 * 0.7);
        }
    }
    height
}

fn bathymetry(resolution: u32) -> Heightfield {
    let m = metrics(resolution);
    let mut height = Heightfield::zeros(m);
    for j in 0..resolution {
        for i in 0..resolution {
            let coast = i as f32 - resolution as f32 * 0.45;
            height.set(i, j, coast * 0.75 + (j as f32 * 0.2).sin());
        }
    }
    height
}

fn issue_cone_ramp() -> Heightfield {
    let m = HeightfieldMetrics::new(32, 32, 64.0, 64.0);
    let mut height = Heightfield::filled(m, 5.0);
    for j in 0..32 {
        for i in 0..32 {
            let x = i as f32 - 16.0;
            let z = j as f32 - 16.0;
            height.set(
                i,
                j,
                5.0 + (18.0 - x.hypot(z)).max(0.0) * 2.0 + i as f32 * 0.02,
            );
        }
    }
    height
}

fn sum(values: &[f32]) -> f64 {
    values.iter().copied().map(f64::from).sum()
}

fn height_sum(height: &Heightfield) -> f64 {
    sum(&height.to_dense())
}

fn mass_tolerance(input: &Heightfield, moved_material: f64) -> f64 {
    let dense = input.to_dense();
    let max_abs_height = dense.iter().copied().map(f32::abs).fold(0.0f32, f32::max) as f64;
    let terrain_scale = dense.len() as f64 * max_abs_height;
    16.0 * f64::from(f32::EPSILON) * (terrain_scale + moved_material.abs()).max(1.0)
}

fn assert_mass_close(label: &str, actual: f64, expected: f64, input: &Heightfield, moved: f64) {
    let tolerance = mass_tolerance(input, moved);
    let residual = actual - expected;
    assert!(
        residual.abs() <= tolerance,
        "{label}: residual {residual} exceeds scale-aware tolerance {tolerance}"
    );
}

fn assert_height_finite(label: &str, height: &Heightfield) {
    assert!(
        height.to_dense().iter().all(|value| value.is_finite()),
        "{label} contains a non-finite height"
    );
}

fn assert_mask_finite(label: &str, field: &MaskField) {
    assert!(
        field.data().iter().all(|value| value.is_finite()),
        "{label} contains a non-finite value"
    );
}

fn assert_physical_mask(label: &str, field: &MaskField) {
    assert_mask_finite(label, field);
    assert!(
        field.data().iter().all(|value| *value >= 0.0),
        "{label} contains a negative physical quantity"
    );
}

fn assert_normalized_mask(label: &str, field: &MaskField) {
    assert_mask_finite(label, field);
    assert!(
        field.data().iter().all(|value| (0.0..=1.0).contains(value)),
        "{label} contains a value outside [0, 1]"
    );
}

fn relief(height: &Heightfield) -> f32 {
    let (minimum, maximum) = height.min_max();
    maximum - minimum
}

#[test]
fn thermal_classical_and_layered_close_mass_and_degenerate_limits() {
    let input = spike_cliff(20);
    let hardness = MaskField::zeros(input.metrics);

    for (name, strength, talus, iterations) in [
        ("ui-min", 0.0, 5.0, 1),
        ("default", 0.5, 35.0, 40),
        ("ui-max", 1.0, 60.0, 200),
    ] {
        let params = ThermalErosionParams {
            strength,
            talus_angle_deg: talus,
            iterations,
            layered_materials: false,
            ..ThermalErosionParams::default()
        };
        let (height, erosion, deposition) = thermal_erode_with_hardness(&input, &params, &hardness);
        assert_height_finite(name, &height);
        assert_normalized_mask(&format!("{name} classical erosion"), &erosion);
        assert_normalized_mask(&format!("{name} classical deposition"), &deposition);
        assert_mass_close(
            &format!("{name} classical thermal"),
            height_sum(&height),
            height_sum(&input),
            &input,
            height_sum(&input).abs(),
        );
        if strength == 0.0 {
            assert_eq!(height.to_dense(), input.to_dense());
        }
    }

    for input in [spike_cliff(20), bathymetry(20)] {
        let initial = MassWastingState::from_height(&input, 0.25, 0.5);
        let params = ThermalErosionParams {
            iterations: 32,
            strength: 0.7,
            layered_materials: true,
            ..ThermalErosionParams::default()
        };
        let result = thermal_erode_layered(&input, &params, &hardness, Some(&initial));
        assert_height_finite("layered thermal", &result.height);
        for (name, field) in [
            ("thermal bedrock", &result.bedrock),
            ("thermal debris", &result.loose_debris),
            ("thermal sediment", &result.sediment),
            ("thermal erosion raw", &result.erosion_raw),
            ("thermal deposition raw", &result.deposition_raw),
        ] {
            assert_physical_mask(name, field);
        }
        for idx in 0..initial.base.len() {
            let reconstructed = initial.base[idx]
                + result.bedrock.data()[idx]
                + result.loose_debris.data()[idx]
                + result.sediment.data()[idx];
            assert!(
                (reconstructed - result.height.to_dense()[idx]).abs() <= 2.0e-5,
                "layered thermal surface mismatch at {idx}"
            );
        }
        let moved = sum(result.erosion_raw.data()) + sum(result.deposition_raw.data());
        assert_mass_close(
            "layered thermal inventory",
            height_sum(&result.height),
            height_sum(&input),
            &input,
            moved,
        );
    }

    let full_hardness = MaskField::filled(input.metrics, 1.0);
    let params = ThermalErosionParams {
        layered_materials: false,
        ..ThermalErosionParams::default()
    };
    let (height, _, _) = thermal_erode_with_hardness(&input, &params, &full_hardness);
    assert_eq!(height.to_dense(), input.to_dense());
}

#[test]
fn hydraulic_models_particles_and_ui_edges_close_the_material_ledger() {
    let input = closed_basin(18);
    let hardness = MaskField::zeros(input.metrics);
    let models = [
        TransportModel::Hydraulic,
        TransportModel::SoftFlows,
        TransportModel::RidgedFlows,
        TransportModel::ThinFlows,
        TransportModel::WideFlows,
        TransportModel::SedimentFlows,
        TransportModel::HydraulicSediment,
    ];

    for model in models {
        let mut core = HydraulicErosionCore::with_model(model);
        core.params.iterations = 16;
        core.params.particle_density = 0.0;
        let (result, outputs, diagnostics, state) =
            core.simulate(&input, &hardness, None, None, None, None);
        assert_height_finite(&format!("{model:?} height"), &result.height);
        for (name, field) in [
            ("water", &result.water_raw),
            ("suspended sediment", &result.sediment_raw),
            ("hydraulic erosion", &result.erosion_raw),
            ("hydraulic deposition", &result.deposition_raw),
        ] {
            assert_physical_mask(&format!("{model:?} {name}"), field);
        }
        assert_eq!(diagnostics.nan_or_inf_cells, 0, "{model:?}");
        assert!(state
            .water_depth
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0));
        assert!(state
            .suspended_sediment
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0));
        for field in [
            outputs.erosion.as_ref(),
            outputs.deposition.as_ref(),
            outputs.flow.as_ref(),
            outputs.water.as_ref(),
            outputs.sediment.as_ref(),
            outputs.channels.as_ref(),
            outputs.rainfall.as_ref(),
            outputs.bedrock.as_ref(),
            outputs.loose_sediment.as_ref(),
            outputs.velocity.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            assert_mask_finite(&format!("{model:?} packaged output"), field);
        }
        let final_inventory = height_sum(&result.height) + sum(result.sediment_raw.data());
        let moved = diagnostics.eroded_sum.abs() as f64 + diagnostics.deposited_sum.abs() as f64;
        assert_mass_close(
            &format!("{model:?} closed hydraulic boundary"),
            final_inventory,
            height_sum(&input),
            &input,
            moved,
        );
    }

    for (name, iterations, rainfall, erosion, particles) in [
        ("ui-min", 1, 0.0, 0.0, 0.0),
        ("default", 60, 0.02, 0.3, 0.06),
        ("ui-max", 200, 0.2, 1.0, 0.5),
    ] {
        let params = HydraulicErosionParams {
            iterations,
            rainfall,
            erosion,
            particle_density: particles,
            ..HydraulicErosionParams::default()
        };
        let core = HydraulicErosionCore::new(params);
        let (result, _, diagnostics, _) = core.simulate(
            &flat(12, 12.0),
            &MaskField::zeros(metrics(12)),
            None,
            None,
            None,
            None,
        );
        assert_height_finite(name, &result.height);
        assert_physical_mask(&format!("{name} water"), &result.water_raw);
        assert_physical_mask(&format!("{name} sediment"), &result.sediment_raw);
        assert_eq!(diagnostics.nan_or_inf_cells, 0, "{name}");
    }

    let hard = MaskField::filled(input.metrics, 1.0);
    let params = HydraulicErosionParams {
        particle_density: 0.0,
        hardness: 1.0,
        ..HydraulicErosionParams::default()
    };
    let (result, _, _, _) =
        HydraulicErosionCore::new(params).simulate(&input, &hard, None, None, None, None);
    assert_eq!(sum(result.erosion_raw.data()), 0.0);
}

#[test]
fn debris_flow_ledger_closes_for_cone_basins_and_every_edge() {
    let cone = issue_cone_ramp();
    let hardness = MaskField::zeros(cone.metrics);
    for iterations in [1, 2, 3, 12] {
        let params = DebrisFlowParams {
            iterations,
            dt: 2.0,
            ..DebrisFlowParams::default()
        };
        let result = debris_flow_erode(&cone, &params, &hardness, None);
        let ledger = result.mass_ledger;
        let tolerance = mass_tolerance(&cone, ledger.mobilized_sum);
        assert!(ledger.residual.abs() <= tolerance, "{ledger:?}");
        assert!((ledger.mobilized_sum - ledger.deposited_sum).abs() <= tolerance);
        assert_eq!(ledger.in_flight_sum, 0.0);
        assert_eq!(ledger.exported_sum, 0.0);
        assert_eq!(sum(result.erosion_raw.data()), ledger.eroded_sum);
        assert_eq!(sum(result.deposition_raw.data()), ledger.deposited_sum);
        assert_height_finite("debris cone", &result.height);
        for (name, field) in [
            ("debris bedrock", &result.bedrock),
            ("debris layer", &result.debris),
            ("fine sediment", &result.sediment),
            ("debris erosion", &result.erosion_raw),
            ("debris deposition", &result.deposition_raw),
        ] {
            assert_physical_mask(name, field);
        }
    }

    let basin = closed_basin(20);
    let basin_result = debris_flow_erode(
        &basin,
        &DebrisFlowParams {
            iterations: 4,
            dt: 2.0,
            ..DebrisFlowParams::default()
        },
        &MaskField::zeros(basin.metrics),
        None,
    );
    assert!(sum(basin_result.deposition_raw.data()) > 0.0);
    assert_eq!(basin_result.mass_ledger.exported_sum, 0.0);

    for edge in 0..4 {
        let ramp = edge_ramp(18, edge);
        let result = debris_flow_erode(
            &ramp,
            &DebrisFlowParams {
                iterations: 2,
                dt: 2.0,
                ..DebrisFlowParams::default()
            },
            &MaskField::zeros(ramp.metrics),
            None,
        );
        assert!(sum(result.deposition_raw.data()) > 0.0, "edge {edge}");
        assert!(
            result.mass_ledger.residual.abs()
                <= mass_tolerance(&ramp, result.mass_ledger.mobilized_sum)
        );
    }

    let flat = flat(16, 10.0);
    let result = debris_flow_erode(
        &flat,
        &DebrisFlowParams::default(),
        &MaskField::zeros(flat.metrics),
        None,
    );
    assert_eq!(result.height.to_dense(), flat.to_dense());
}

#[test]
fn debris_receiver_weighting_is_balanced_through_the_public_solver() {
    const RESOLUTION: u32 = 5;
    const SEEDS: u64 = 1_024;
    let m = HeightfieldMetrics::new(RESOLUTION, RESOLUTION, 10.0, 10.0);
    let mut input = Heightfield::zeros(m);
    input.set(2, 2, 20.0);
    let mut hardness = MaskField::filled(m, 1.0);
    hardness.set(2, 2, 0.0);
    let receivers = [(1u32, 2u32), (3, 2), (2, 1), (2, 3)];
    let mut counts = [0usize; 4];

    for seed in 0..SEEDS {
        let params = DebrisFlowParams {
            iterations: 1,
            dt: 2.0,
            talus_angle_deg: 15.0,
            thermal_k: 0.05,
            abrasion_k: 0.0,
            fluvial_k: 0.0,
            fluvial_deposition: 0.0,
            hillslope_k: 0.0,
            precipitation: 0.0,
            max_deposit_per_step: 100.0,
            seed,
            ..DebrisFlowParams::default()
        };
        let result = debris_flow_erode(&input, &params, &hardness, None);
        let (direction, amount) = receivers
            .iter()
            .enumerate()
            .map(|(index, &(i, j))| (index, result.deposition_raw.get(i, j)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        assert!(
            amount > 0.0,
            "seed {seed} did not settle at a cardinal receiver"
        );
        counts[direction] += 1;
    }

    let expected = SEEDS as f64 / 4.0;
    let tolerance = SEEDS as f64 * 0.03;
    for (direction, count) in ["left", "right", "up", "down"].into_iter().zip(counts) {
        assert!(
            (count as f64 - expected).abs() <= tolerance,
            "{direction} receiver count {count} is outside {expected} +/- {tolerance}"
        );
    }
}

#[test]
fn stream_power_reports_removed_material_and_respects_degenerate_limits() {
    let input = edge_ramp(20, 0);
    let soft = MaskField::zeros(input.metrics);
    for (name, iterations, k) in [
        ("ui-min", 1, 0.0),
        ("default", 24, 0.08),
        ("ui-max", 80, 0.5),
    ] {
        let params = StreamPowerParams {
            iterations,
            k,
            uplift_rate: 0.0,
            dendritic_seed: 0.0,
            ..StreamPowerParams::default()
        };
        let result = stream_power_erode(&input, &params, &soft);
        assert_height_finite(name, &result.height);
        for (field_name, field) in [
            ("flow direction", &result.flow_direction),
            ("flow accumulation", &result.flow_accumulation),
            ("stream order", &result.stream_order),
            ("incision", &result.spe_incision),
        ] {
            assert_mask_finite(&format!("{name} {field_name}"), field);
        }
        let removed = height_sum(&input) - height_sum(&result.height);
        let reported = sum(result.spe_incision.data());
        assert_mass_close(
            &format!("{name} stream-power removed material"),
            removed,
            reported,
            &input,
            reported,
        );
        if k == 0.0 {
            assert_eq!(result.height.to_dense(), input.to_dense());
            assert_eq!(reported, 0.0);
        }
    }

    let hard = MaskField::filled(input.metrics, 1.0);
    let result = stream_power_erode(&input, &StreamPowerParams::default(), &hard);
    assert_eq!(result.height.to_dense(), input.to_dense());
    assert_eq!(sum(result.spe_incision.data()), 0.0);
}

fn landscape_seed() -> Heightfield {
    let m = HeightfieldMetrics::new(20, 20, 200.0, 200.0);
    let mut height = Heightfield::filled(m, 10.0);
    for j in 0..m.height {
        for i in 0..m.width {
            height.set(i, j, 10.0 + i as f32 * 1.0e-5 + j as f32 * 5.0e-6);
        }
    }
    height.set(10, 10, 38.0);
    height.set(8, 8, -2.0);
    height.set(9, 8, -1.0);
    height
}

fn evaluate_landscape(
    seed: &Heightfield,
    solver: EvolutionSolverMode,
    boundary: BoundaryMode,
    outlet_mask: Option<&MaskField>,
) -> LandscapeEvolutionOutput {
    let params = LandscapeEvolutionParams {
        solver,
        boundary,
        uplift_mode: UpliftMode::Uniform,
        uplift: 0.8,
        uplift_noise: 0.0,
        geological_age: 0.8,
        hillslope_diffusion: 0.15,
        constraint_preservation: 0.8,
        fixed_point_iters: 2,
        iterations: 9,
        time_scale: 300_000.0,
        dt: 5_000.0,
        base_level: 0.0,
        ..LandscapeEvolutionParams::default()
    };
    LandscapeEvolutionOperator::new(params).evaluate(LandscapeEvolutionInput {
        elevation: seed,
        painted_uplift: None,
        precipitation: None,
        erodibility: None,
        lithology_hardness: None,
        outlet_mask,
        protection: None,
    })
}

fn rim_coords(metrics: HeightfieldMetrics) -> Vec<(u32, u32)> {
    let mut cells = Vec::new();
    for j in 0..metrics.height {
        for i in 0..metrics.width {
            if i == 0 || j == 0 || i + 1 == metrics.width || j + 1 == metrics.height {
                cells.push((i, j));
            }
        }
    }
    cells
}

fn assert_same_bits(actual: &Heightfield, expected: &Heightfield, cells: &[(u32, u32)]) {
    for &(i, j) in cells {
        assert_eq!(
            actual.get(i, j).to_bits(),
            expected.get(i, j).to_bits(),
            "({i}, {j})"
        );
    }
}

fn any_changed(actual: &Heightfield, expected: &Heightfield, cells: &[(u32, u32)]) -> bool {
    cells
        .iter()
        .any(|&(i, j)| actual.get(i, j).to_bits() != expected.get(i, j).to_bits())
}

fn assert_landscape_finite(label: &str, output: &LandscapeEvolutionOutput) {
    assert_height_finite(&format!("{label} elevation"), &output.elevation);
    assert_height_finite(&format!("{label} tectonic base"), &output.tectonic_base);
    for (name, field) in [
        ("flow direction", &output.flow_direction),
        ("flow accumulation", &output.flow_accumulation),
        ("stream order", &output.stream_order),
        ("erosion rate", &output.erosion_rate),
        ("incision", &output.incision),
        ("uplift", &output.uplift_field),
        ("discharge", &output.discharge),
    ] {
        assert_mask_finite(&format!("{label} {name}"), field);
    }
    if let Some(deposition) = output.deposition.as_ref() {
        assert_physical_mask(&format!("{label} deposition"), deposition);
    }
}

#[test]
fn landscape_boundary_modes_separate_outlets_from_elevation_locks() {
    let seed = landscape_seed();
    let rim = rim_coords(seed.metrics);
    let submerged = [(8u32, 8u32), (9, 8)];
    let authored = [(5u32, 5u32), (14, 12)];
    let mut outlet_mask = MaskField::zeros(seed.metrics);
    for &(i, j) in &authored {
        outlet_mask.set(i, j, 1.0);
    }

    for solver in [EvolutionSolverMode::Fast, EvolutionSolverMode::Accurate] {
        let fixed = evaluate_landscape(&seed, solver, BoundaryMode::Fixed, None);
        assert_landscape_finite(&format!("{solver:?} fixed"), &fixed);
        assert_same_bits(&fixed.elevation, &seed, &rim);
        assert_same_bits(&fixed.tectonic_base, &seed, &rim);

        let open = evaluate_landscape(&seed, solver, BoundaryMode::OpenDrainage, None);
        assert_landscape_finite(&format!("{solver:?} open"), &open);
        assert!(any_changed(&open.elevation, &seed, &rim));
        assert!(rim.iter().any(|&(i, j)| open.uplift_field.get(i, j) > 0.0));

        let sea = evaluate_landscape(&seed, solver, BoundaryMode::SeaLevel, None);
        assert_landscape_finite(&format!("{solver:?} sea"), &sea);
        assert_same_bits(&sea.elevation, &seed, &submerged);
        assert!(any_changed(&sea.elevation, &seed, &rim));

        let masked =
            evaluate_landscape(&seed, solver, BoundaryMode::OutletMask, Some(&outlet_mask));
        assert_landscape_finite(&format!("{solver:?} outlet mask"), &masked);
        assert_same_bits(&masked.elevation, &seed, &authored);
        assert!(any_changed(&masked.elevation, &seed, &rim));

        let missing = evaluate_landscape(&seed, solver, BoundaryMode::OutletMask, None);
        assert_eq!(missing.elevation.to_dense(), sea.elevation.to_dense());
        let empty = MaskField::zeros(seed.metrics);
        let empty_result =
            evaluate_landscape(&seed, solver, BoundaryMode::OutletMask, Some(&empty));
        assert!(any_changed(&empty_result.elevation, &seed, &rim));
    }
}

#[test]
fn landscape_zero_effect_full_resistance_and_ui_max_stay_bounded() {
    let seed = landscape_seed();
    for solver in [EvolutionSolverMode::Fast, EvolutionSolverMode::Accurate] {
        let zero = LandscapeEvolutionParams {
            solver,
            erosion: 0.0,
            hillslope_diffusion: 0.0,
            fixed_point_iters: 2,
            iterations: 8,
            ..LandscapeEvolutionParams::default()
        };
        let zero_output =
            LandscapeEvolutionOperator::new(zero.clone()).evaluate(LandscapeEvolutionInput {
                elevation: &seed,
                painted_uplift: None,
                precipitation: None,
                erodibility: None,
                lithology_hardness: None,
                outlet_mask: None,
                protection: None,
            });
        assert_landscape_finite(&format!("{solver:?} zero erosion"), &zero_output);
        assert!(sum(zero_output.incision.data()) <= 1.0e-5);
        assert!(relief(&zero_output.elevation) < 1_000.0);

        let resistant = LandscapeEvolutionParams {
            solver,
            terrain_resistance: 1.0,
            hillslope_diffusion: 0.0,
            fixed_point_iters: 2,
            iterations: 8,
            ..LandscapeEvolutionParams::default()
        };
        let resistant_output =
            LandscapeEvolutionOperator::new(resistant).evaluate(LandscapeEvolutionInput {
                elevation: &seed,
                painted_uplift: None,
                precipitation: Some(&MaskField::filled(seed.metrics, 3.0)),
                erodibility: None,
                lithology_hardness: Some(&MaskField::zeros(seed.metrics)),
                outlet_mask: None,
                protection: None,
            });
        assert_landscape_finite(&format!("{solver:?} full resistance"), &resistant_output);
        assert!(sum(resistant_output.incision.data()) <= 1.0e-5);

        let ui_max = LandscapeEvolutionParams {
            solver,
            uplift: 2.0,
            erosion: 2.0,
            rainfall: 3.0,
            river_incision: 2.0,
            geological_age: 1.0,
            terrain_resistance: 0.0,
            fixed_point_iters: 6,
            iterations: 24,
            ..LandscapeEvolutionParams::default()
        };
        let max_output =
            LandscapeEvolutionOperator::new(ui_max).evaluate(LandscapeEvolutionInput {
                elevation: &epsilon_slope(20),
                painted_uplift: None,
                precipitation: None,
                erodibility: None,
                lithology_hardness: None,
                outlet_mask: None,
                protection: None,
            });
        assert_landscape_finite(&format!("{solver:?} UI max"), &max_output);
        assert!(relief(&max_output.elevation) < 10_000.0);
    }
}

struct StackRun {
    evaluator: StackEvaluator,
    context: EvalContext,
    height: Heightfield,
    hydraulic: LayerId,
    debris: LayerId,
    thermal: LayerId,
}

fn run_handoff_stack(base_height: f32, active: bool) -> StackRun {
    let m = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
    let hydraulic = HydraulicErosionParams {
        iterations: if active { 4 } else { 1 },
        rainfall: if active { 0.02 } else { 0.0 },
        evaporation: 0.0,
        capacity: if active { 0.1 } else { 0.0 },
        erosion: if active { 0.2 } else { 0.0 },
        deposition: if active { 0.2 } else { 0.0 },
        particle_density: 0.0,
        transport_model: TransportModel::HydraulicSediment,
        layered_materials: true,
        initial_sediment_thickness: 1.25,
        level_count: 1,
        ..HydraulicErosionParams::default()
    };
    let debris = DebrisFlowParams {
        iterations: if active { 2 } else { 1 },
        thermal_k: if active { 0.01 } else { 0.0 },
        abrasion_k: 0.0,
        fluvial_k: 0.0,
        fluvial_deposition: 0.0,
        hillslope_k: 0.0,
        precipitation: 0.0,
        initial_debris_thickness: 0.0,
        initial_sediment_thickness: 0.0,
        ..DebrisFlowParams::default()
    };
    let thermal = ThermalErosionParams {
        iterations: if active { 2 } else { 1 },
        strength: if active { 0.2 } else { 0.0 },
        weathering_rate: if active { 0.2 } else { 0.0 },
        material_amount: if active { 1.0 } else { 0.0 },
        layered_materials: true,
        level_count: 1,
        ..ThermalErosionParams::default()
    };

    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "Base",
        LayerKind::Flat(FlatParams {
            height: base_height,
        }),
    ));
    let hydraulic_layer = Layer::new("Hydraulic", LayerKind::HydraulicErosion(hydraulic));
    let hydraulic_id = hydraulic_layer.id();
    stack.push(hydraulic_layer);
    let debris_layer = Layer::new("Debris", LayerKind::DebrisFlow(debris));
    let debris_id = debris_layer.id();
    stack.push(debris_layer);
    let thermal_layer = Layer::new("Thermal", LayerKind::ThermalErosion(thermal));
    let thermal_id = thermal_layer.id();
    stack.push(thermal_layer);

    let mut evaluator = StackEvaluator::new();
    evaluator.cache.disable_disk();
    let mut context = EvalContext::new(m);
    let height = evaluator
        .rebuild_all(&stack, &mut context)
        .expect("stack evaluates");
    StackRun {
        evaluator,
        context,
        height,
        hydraulic: hydraulic_id,
        debris: debris_id,
        thermal: thermal_id,
    }
}

fn field<'a>(aux: &'a HashMap<String, MaskField>, key: &str) -> &'a MaskField {
    aux.get(key).unwrap_or_else(|| panic!("missing {key}"))
}

fn assert_canonical_inventory(aux: &HashMap<String, MaskField>) {
    assert!(aux.contains_key(keys::BEDROCK_HEIGHT));
    assert!(aux.contains_key(keys::SEDIMENT_THICKNESS));
    assert!(!aux.contains_key(keys::SEDIMENT_DEPTH));
    assert!(!aux.contains_key(keys::LOOSE_SEDIMENT));
    assert_physical_mask("stack bedrock", field(aux, keys::BEDROCK_HEIGHT));
    assert_physical_mask("stack sediment", field(aux, keys::SEDIMENT_THICKNESS));
    if let Some(debris) = aux.get(keys::DEBRIS_DEPTH) {
        assert_physical_mask("stack debris", debris);
    }
}

#[test]
fn stack_evaluator_preserves_canonical_inventories_and_sub_zero_surfaces() {
    for base_height in [10.0, -10.0] {
        let run = run_handoff_stack(base_height, false);
        for id in [run.hydraulic, run.debris, run.thermal] {
            let checkpoint = run.evaluator.cache.get(id).expect("layer checkpoint");
            assert_canonical_inventory(&checkpoint.aux);
            assert_height_finite("stack checkpoint", &checkpoint.height);
            assert!(
                (sum(field(&checkpoint.aux, keys::SEDIMENT_THICKNESS).data()) - 320.0).abs()
                    < 1.0e-4
            );
            assert!(checkpoint
                .height
                .to_dense()
                .iter()
                .all(|height| (*height - base_height).abs() < 1.0e-5));
        }
        assert_canonical_inventory(&run.context.aux);
        assert_height_finite("stack output", &run.height);

        let typed = AuxMaps::from_hashmap(&run.context.aux);
        let state = MassWastingState::with_optional_layers(
            &run.height,
            typed.bedrock_height.as_ref(),
            typed.get(keys::DEBRIS_DEPTH),
            typed.sediment_thickness.as_ref(),
            0.0,
            0.0,
        );
        for (actual, expected) in state.sync_surface().into_iter().zip(run.height.to_dense()) {
            assert!((actual - expected).abs() < 1.0e-5);
        }
    }

    let active = run_handoff_stack(10.0, true);
    assert_height_finite("active authored stack", &active.height);
    assert_canonical_inventory(&active.context.aux);
}
