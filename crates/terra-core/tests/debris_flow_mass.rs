//! Regression fixtures for issue #63: debris-flow transport must conserve its load.

use terra_core::analyze::{debris_flow_erode, DebrisFlowResult};
use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
use terra_core::layer::DebrisFlowParams;
use terra_core::mask::MaskField;

fn issue_cone_ramp() -> Heightfield {
    let metrics = HeightfieldMetrics::new(32, 32, 64.0, 64.0);
    let mut height = Heightfield::filled(metrics, 5.0);
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

fn mass_tolerance(initial_surface_sum: f64) -> f64 {
    // Solver storage and transfers are f32. Diagnostics are summed as f64 so this
    // tolerance covers only f32 operation ordering, not visible percentage drift.
    (initial_surface_sum.abs() * 5.0e-6).max(1.0e-3)
}

fn sum(values: &[f32]) -> f64 {
    values.iter().copied().map(f64::from).sum()
}

fn assert_conservative(result: &DebrisFlowResult) {
    let ledger = result.mass_ledger;
    let tolerance = mass_tolerance(ledger.initial_surface_sum);
    assert!(
        ledger.residual.abs() <= tolerance,
        "surface residual {} exceeds tolerance {tolerance}: {ledger:?}",
        ledger.residual
    );
    assert!(
        (ledger.mobilized_sum - ledger.deposited_sum).abs() <= tolerance,
        "transport residual exceeds tolerance {tolerance}: {ledger:?}"
    );
    assert_eq!(ledger.in_flight_sum, 0.0);
    assert_eq!(ledger.exported_sum, 0.0);
    assert_eq!(
        sum(result.erosion_raw.data()),
        ledger.eroded_sum,
        "erosion raw mask must be the ledger source of truth"
    );
    assert_eq!(
        sum(result.deposition_raw.data()),
        ledger.deposited_sum,
        "deposition raw mask must be the ledger source of truth"
    );
}

fn assert_finite_non_negative(result: &DebrisFlowResult) {
    for (name, values) in [
        ("bedrock", result.bedrock.data()),
        ("debris", result.debris.data()),
        ("sediment", result.sediment.data()),
        ("erosion_raw", result.erosion_raw.data()),
        ("deposition_raw", result.deposition_raw.data()),
    ] {
        assert!(
            values.iter().all(|value| value.is_finite()),
            "{name} contains a non-finite value"
        );
        assert!(
            values.iter().all(|value| *value >= 0.0),
            "{name} contains a negative value"
        );
    }
    assert!(result.height.to_dense().iter().all(|v| v.is_finite()));
    let ledger_values = [
        result.mass_ledger.initial_surface_sum,
        result.mass_ledger.eroded_sum,
        result.mass_ledger.mobilized_sum,
        result.mass_ledger.deposited_sum,
        result.mass_ledger.in_flight_sum,
        result.mass_ledger.exported_sum,
        result.mass_ledger.final_surface_sum,
        result.mass_ledger.residual,
    ];
    assert!(ledger_values.iter().all(|value| value.is_finite()));
    assert!(ledger_values[..7].iter().all(|value| *value >= 0.0));
}

#[test]
fn issue_63_cone_ramp_conserves_at_each_iteration_count() {
    let input = issue_cone_ramp();
    let hardness = MaskField::filled(input.metrics, 0.0);
    for iterations in [1, 2, 3, 12] {
        let params = DebrisFlowParams {
            iterations,
            dt: 2.0,
            ..DebrisFlowParams::default()
        };
        let result = debris_flow_erode(&input, &params, &hardness, None);
        assert_conservative(&result);
        assert_finite_non_negative(&result);
    }
}

#[test]
fn flat_terrain_is_an_exact_no_op() {
    let metrics = HeightfieldMetrics::new(24, 24, 48.0, 48.0);
    let input = Heightfield::filled(metrics, 10.0);
    let hardness = MaskField::filled(metrics, 0.0);
    let result = debris_flow_erode(&input, &DebrisFlowParams::default(), &hardness, None);

    assert_eq!(result.height.to_dense(), input.to_dense());
    assert_eq!(result.mass_ledger.eroded_sum, 0.0);
    assert_eq!(result.mass_ledger.mobilized_sum, 0.0);
    assert_eq!(result.mass_ledger.deposited_sum, 0.0);
    assert_conservative(&result);
    assert_finite_non_negative(&result);
}

#[test]
fn closed_basin_settles_terminal_load_in_domain() {
    let metrics = HeightfieldMetrics::new(33, 33, 66.0, 66.0);
    let mut input = Heightfield::zeros(metrics);
    for j in 0..33 {
        for i in 0..33 {
            let x = i as f32 - 16.0;
            let z = j as f32 - 16.0;
            input.set(i, j, 5.0 + x.hypot(z) * 1.5);
        }
    }
    let hardness = MaskField::filled(metrics, 0.0);
    let params = DebrisFlowParams {
        iterations: 3,
        dt: 2.0,
        ..DebrisFlowParams::default()
    };
    let result = debris_flow_erode(&input, &params, &hardness, None);
    let mut center_deposition = 0.0f32;
    for j in 15..=17 {
        for i in 15..=17 {
            center_deposition += result.deposition_raw.get(i, j);
        }
    }

    assert!(
        center_deposition > 0.0,
        "closed basin should settle load near its terminal low point"
    );
    assert_conservative(&result);
    assert_finite_non_negative(&result);
}

#[test]
fn edge_draining_ramp_settles_at_closed_boundary() {
    let metrics = HeightfieldMetrics::new(32, 32, 64.0, 64.0);
    let mut input = Heightfield::zeros(metrics);
    for j in 0..32 {
        for i in 0..32 {
            input.set(i, j, 70.0 - i as f32 * 2.0);
        }
    }
    let hardness = MaskField::filled(metrics, 0.0);
    let params = DebrisFlowParams {
        iterations: 2,
        dt: 2.0,
        ..DebrisFlowParams::default()
    };
    let result = debris_flow_erode(&input, &params, &hardness, None);
    let edge_deposition: f32 = (0..32).map(|j| result.deposition_raw.get(31, j)).sum();

    assert!(
        edge_deposition > 0.0,
        "edge-draining load should settle on the closed boundary"
    );
    assert_conservative(&result);
    assert_finite_non_negative(&result);
}

#[test]
fn debris_flow_is_deterministic_for_a_fixed_seed() {
    let input = issue_cone_ramp();
    let hardness = MaskField::filled(input.metrics, 0.0);
    let params = DebrisFlowParams {
        iterations: 6,
        dt: 2.0,
        seed: 0x63,
        ..DebrisFlowParams::default()
    };
    let first = debris_flow_erode(&input, &params, &hardness, None);
    let second = debris_flow_erode(&input, &params, &hardness, None);

    assert_eq!(first.height.to_dense(), second.height.to_dense());
    assert_eq!(first.bedrock.data(), second.bedrock.data());
    assert_eq!(first.debris.data(), second.debris.data());
    assert_eq!(first.sediment.data(), second.sediment.data());
    assert_eq!(first.erosion.data(), second.erosion.data());
    assert_eq!(first.deposition.data(), second.deposition.data());
    assert_eq!(first.debris_erosion.data(), second.debris_erosion.data());
    assert_eq!(
        first.debris_deposition.data(),
        second.debris_deposition.data()
    );
    assert_eq!(first.slide_path.data(), second.slide_path.data());
    assert_eq!(first.instability.data(), second.instability.data());
    assert_eq!(
        first.flow_accumulation.data(),
        second.flow_accumulation.data()
    );
    assert_eq!(first.erosion_raw.data(), second.erosion_raw.data());
    assert_eq!(first.deposition_raw.data(), second.deposition_raw.data());
    assert_eq!(first.mass_ledger, second.mass_ledger);
}

#[test]
fn drainage_reuse_local_hops_remain_conservative() {
    let input = issue_cone_ramp();
    let hardness = MaskField::filled(input.metrics, 0.0);
    let params = DebrisFlowParams {
        iterations: 4,
        dt: 2.0,
        drainage_reuse_stride: 2,
        refill_depressions: false,
        ..DebrisFlowParams::default()
    };
    let result = debris_flow_erode(&input, &params, &hardness, None);

    assert!(
        result.mass_ledger.eroded_sum >= result.mass_ledger.mobilized_sum,
        "local bedrock-to-debris weathering is erosion but not transported material"
    );
    assert_conservative(&result);
    assert_finite_non_negative(&result);
}
