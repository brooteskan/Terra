//! Phase 3 landscape evolution fixtures: uplift modes + multi-age evolution.
//!
//! Asserts coherent drainage organisation rather than "noise with scratches".

use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
use terra_core::landscape_evolution::{
    asymmetric_belt, synthesise_uplift, BoundaryMode, EvolutionSolverMode, LandscapeEvolutionInput,
    LandscapeEvolutionOperator, LandscapeEvolutionParams, UpliftMode,
};
use terra_core::mask::MaskField;

fn metrics() -> HeightfieldMetrics {
    HeightfieldMetrics::new(64, 64, 4000.0, 4000.0)
}

fn flat_seed(m: HeightfieldMetrics, h: f32) -> Heightfield {
    Heightfield::from_dense(m, &vec![h; (m.width * m.height) as usize])
}

fn mean_valley_accumulation(height: &Heightfield, acc: &MaskField) -> f32 {
    let dense = height.to_dense();
    let mut sorted = dense.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = sorted[sorted.len() / 4];
    let mut sum = 0.0f32;
    let mut n = 0u32;
    for j in 0..height.metrics.height {
        for i in 0..height.metrics.width {
            if height.get(i, j) <= q {
                sum += acc.get(i, j);
                n += 1;
            }
        }
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f32
    }
}

fn relief(height: &Heightfield) -> f32 {
    let (lo, hi) = height.min_max();
    hi - lo
}

fn max_abs_delta(a: &Heightfield, b: &Heightfield) -> f32 {
    a.to_dense()
        .iter()
        .zip(b.to_dense())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

fn max_abs_delta_interior(a: &Heightfield, b: &Heightfield, margin: u32) -> f32 {
    let m = a.metrics;
    let mut max_delta = 0.0f32;
    for j in margin..m.height.saturating_sub(margin) {
        for i in margin..m.width.saturating_sub(margin) {
            max_delta = max_delta.max((a.get(i, j) - b.get(i, j)).abs());
        }
    }
    max_delta
}

fn plateau_cone_seed() -> Heightfield {
    let m = HeightfieldMetrics::new(24, 24, 240.0, 240.0);
    let mut data = vec![20.0f32; (m.width * m.height) as usize];
    let cx = (m.width - 1) as f32 * 0.5;
    let cy = (m.height - 1) as f32 * 0.5;
    let radius = 7.0f32;
    for j in 0..m.height {
        for i in 0..m.width {
            let d = (i as f32 - cx).hypot(j as f32 - cy);
            let cone = (1.0 - d / radius).max(0.0) * 18.0;
            data[(j * m.width + i) as usize] += cone;
        }
    }
    Heightfield::from_dense(m, &data)
}

fn evaluate_with_inputs(
    seed: &Heightfield,
    p: &LandscapeEvolutionParams,
    hardness: Option<&MaskField>,
    precipitation: Option<&MaskField>,
) -> terra_core::landscape_evolution::LandscapeEvolutionOutput {
    LandscapeEvolutionOperator::new(p.clone()).evaluate(LandscapeEvolutionInput {
        elevation: seed,
        painted_uplift: None,
        precipitation,
        erodibility: None,
        lithology_hardness: hardness,
        outlet_mask: None,
        protection: None,
    })
}

fn mean_channel_weight(order: &MaskField) -> f32 {
    let d = order.data();
    if d.is_empty() {
        return 0.0;
    }
    d.iter().sum::<f32>() / d.len() as f32
}

fn run_evo(
    seed: &Heightfield,
    uplift_mode: UpliftMode,
    age: f32,
    solver: EvolutionSolverMode,
    painted: Option<&MaskField>,
) -> terra_core::landscape_evolution::LandscapeEvolutionOutput {
    let mut p = LandscapeEvolutionParams::default();
    p.uplift_mode = uplift_mode;
    p.geological_age = age;
    p.solver = solver;
    p.uplift = 0.9;
    p.erosion = 0.85;
    p.rainfall = 1.4;
    p.river_incision = 0.9;
    p.drainage_scale = 0.6;
    p.hillslope_diffusion = 0.1;
    p.fixed_point_iters = 6;
    p.iterations = 24;
    p.boundary = BoundaryMode::SeaLevel;
    p.base_level = 0.0;
    p.uplift_noise = 0.04;
    p.time_scale = 1.2e6;
    p.dt = 5_000.0;
    let op = LandscapeEvolutionOperator::new(p);
    op.evaluate(LandscapeEvolutionInput {
        elevation: seed,
        painted_uplift: painted,
        precipitation: None,
        erodibility: None,
        lithology_hardness: None,
        outlet_mask: None,
        protection: None,
    })
}

#[test]
fn uplift_plateau_uniform_preserves_large_form() {
    let m = metrics();
    let seed = flat_seed(m, 20.0);
    let young = run_evo(
        &seed,
        UpliftMode::Uniform,
        0.15,
        EvolutionSolverMode::Fast,
        None,
    );
    let mature = run_evo(
        &seed,
        UpliftMode::Uniform,
        0.85,
        EvolutionSolverMode::Fast,
        None,
    );
    assert!(
        relief(&young.elevation) > 5.0,
        "young uplifted plateau should gain relief"
    );
    assert!(
        mean_valley_accumulation(&mature.elevation, &mature.flow_accumulation)
            > mean_valley_accumulation(&young.elevation, &young.flow_accumulation) * 0.9,
        "mature drainage should organise valleys"
    );
    // Large form: tectonic base remains higher-amplitude than fine scratches.
    assert!(relief(&mature.tectonic_base) > relief(&seed) * 0.5);
}

#[test]
fn linear_mountain_belt_forms_organised_ridges() {
    let m = metrics();
    let seed = flat_seed(m, 10.0);
    let out = run_evo(
        &seed,
        UpliftMode::LinearBelt,
        0.7,
        EvolutionSolverMode::Fast,
        None,
    );
    assert!(relief(&out.elevation) > 15.0);
    // Belt centre (along mid UV) should be higher than corners on average.
    let mut centre = 0.0f32;
    let mut corner = 0.0f32;
    let mut nc = 0u32;
    let mut nk = 0u32;
    for j in 20..44 {
        for i in 20..44 {
            centre += out.elevation.get(i, j);
            nc += 1;
        }
    }
    for &(i, j) in &[(2u32, 2), (61, 2), (2, 61), (61, 61)] {
        corner += out.elevation.get(i, j);
        nk += 1;
    }
    assert!(centre / nc as f32 > corner / nk as f32 + 2.0);
    assert!(
        mean_valley_accumulation(&out.elevation, &out.flow_accumulation) > 0.04
            || mean_channel_weight(&out.stream_order) > 0.02,
        "valleys / channels should organise under a linear belt"
    );
    assert!(
        out.incision.data().iter().copied().sum::<f32>() > 1.0,
        "fluvial incision should accumulate"
    );
}

#[test]
fn radial_dome_produces_outward_drainage() {
    let m = metrics();
    let seed = flat_seed(m, 5.0);
    let mut p = LandscapeEvolutionParams::default();
    p.uplift_mode = UpliftMode::Radial;
    p.uplift_center_u = 0.5;
    p.uplift_center_v = 0.5;
    p.uplift_falloff = 0.4;
    p.geological_age = 0.65;
    p.solver = EvolutionSolverMode::Fast;
    p.time_scale = 8.0e5;
    p.fixed_point_iters = 4;
    p.hillslope_diffusion = 0.1;
    let op = LandscapeEvolutionOperator::new(p);
    let out = op.evaluate(LandscapeEvolutionInput {
        elevation: &seed,
        painted_uplift: None,
        precipitation: None,
        erodibility: None,
        lithology_hardness: None,
        outlet_mask: None,
        protection: None,
    });
    let centre = out.elevation.get(32, 32);
    let rim = (out.elevation.get(4, 32)
        + out.elevation.get(60, 32)
        + out.elevation.get(32, 4)
        + out.elevation.get(32, 60))
        * 0.25;
    assert!(
        centre > rim + 3.0,
        "radial dome peak should exceed rim (c={centre} rim={rim})"
    );
}

#[test]
fn asymmetric_uplift_biases_relief() {
    let m = metrics();
    let painted = asymmetric_belt(m, 1.0, 0.0, 0.9); // horizontal belt, bias across V
    let seed = flat_seed(m, 8.0);
    let out = run_evo(
        &seed,
        UpliftMode::Painted,
        0.6,
        EvolutionSolverMode::Fast,
        Some(&painted),
    );
    assert!(relief(&out.elevation) > 8.0);
    // Bias is across the belt normal (V): compare top vs bottom halves.
    let mut top = 0.0f32;
    let mut bottom = 0.0f32;
    for j in 0..m.height / 2 {
        for i in 0..m.width {
            top += out.elevation.get(i, j);
        }
    }
    for j in m.height / 2..m.height {
        for i in 0..m.width {
            bottom += out.elevation.get(i, j);
        }
    }
    assert!(
        (top - bottom).abs() > 30.0,
        "asymmetric uplift should skew relief (top={top} bottom={bottom})"
    );
    // Painted uplift field itself must be asymmetric.
    let mut up_top = 0.0f32;
    let mut up_bot = 0.0f32;
    for j in 0..m.height / 2 {
        for i in 0..m.width {
            up_top += painted.get(i, j);
        }
    }
    for j in m.height / 2..m.height {
        for i in 0..m.width {
            up_bot += painted.get(i, j);
        }
    }
    assert!((up_top - up_bot).abs() > 1e-3);
}

#[test]
fn geological_age_increases_incision_not_noise() {
    let m = metrics();
    let seed = flat_seed(m, 15.0);
    let young = run_evo(
        &seed,
        UpliftMode::LinearBelt,
        0.12,
        EvolutionSolverMode::Fast,
        None,
    );
    let mid = run_evo(
        &seed,
        UpliftMode::LinearBelt,
        0.45,
        EvolutionSolverMode::Fast,
        None,
    );
    let old = run_evo(
        &seed,
        UpliftMode::LinearBelt,
        0.9,
        EvolutionSolverMode::Fast,
        None,
    );
    let young_inc: f32 = young.incision.data().iter().sum();
    let mid_inc: f32 = mid.incision.data().iter().sum();
    let old_inc: f32 = old.incision.data().iter().sum();
    assert!(
        mid_inc >= young_inc * 0.75,
        "greater age should not reduce organised incision"
    );
    assert!(
        old_inc >= mid_inc * 0.75,
        "mature age should continue stream incision"
    );
    // Stream network weight should not collapse with age (organisation, not noise).
    let yw = mean_channel_weight(&young.stream_order);
    let ow = mean_channel_weight(&old.stream_order);
    assert!(
        ow + mean_valley_accumulation(&old.elevation, &old.flow_accumulation)
            >= (yw + mean_valley_accumulation(&young.elevation, &young.flow_accumulation)) * 0.5,
        "mature drainage should remain coherent (young_w={yw} old_w={ow})"
    );
    // Large forms preserved: tectonic base still carries belt-scale relief.
    assert!(relief(&old.tectonic_base) > 10.0);
}

#[test]
fn accurate_and_fast_share_high_level_params() {
    let m = HeightfieldMetrics::new(48, 48, 3000.0, 3000.0);
    let seed = flat_seed(m, 12.0);
    let fast = run_evo(
        &seed,
        UpliftMode::Radial,
        0.5,
        EvolutionSolverMode::Fast,
        None,
    );
    let accurate = run_evo(
        &seed,
        UpliftMode::Radial,
        0.5,
        EvolutionSolverMode::Accurate,
        None,
    );
    // Both should create meaningful relief and drainage — not identical, but same family.
    assert!(relief(&fast.elevation) > 3.0);
    assert!(relief(&accurate.elevation) > 3.0);
    assert!(
        mean_valley_accumulation(&fast.elevation, &fast.flow_accumulation) > 0.02
            || mean_channel_weight(&fast.stream_order) > 0.01
    );
    assert!(
        mean_valley_accumulation(&accurate.elevation, &accurate.flow_accumulation) > 0.02
            || mean_channel_weight(&accurate.stream_order) > 0.01
            || accurate.incision.data().iter().copied().sum::<f32>() > 0.5
    );
}

#[test]
fn synthesise_uplift_modes_are_smooth() {
    let m = metrics();
    let mut p = LandscapeEvolutionParams::default();
    p.uplift_mode = UpliftMode::Procedural;
    p.uplift_noise = 0.1;
    let field = synthesise_uplift(m, &p, None);
    // Neighbour differences should stay modest (geological smoothness).
    let mut max_grad = 0.0f32;
    let margin = 4u32;
    for j in margin..m.height - margin {
        for i in margin..m.width - margin {
            let g = (field.get(i + 1, j) - field.get(i - 1, j))
                .abs()
                .max((field.get(i, j + 1) - field.get(i, j - 1)).abs());
            max_grad = max_grad.max(g);
        }
    }
    let mut peak = 0.0f32;
    for j in margin..m.height - margin {
        for i in margin..m.width - margin {
            peak = peak.max(field.get(i, j));
        }
    }
    let peak = peak.max(1e-8);
    assert!(
        max_grad / peak < 0.45,
        "procedural uplift must stay smooth at geological scale (g/peak={})",
        max_grad / peak
    );
}

#[test]
fn evolution_is_deterministic() {
    let m = HeightfieldMetrics::new(40, 40, 2000.0, 2000.0);
    let seed = flat_seed(m, 10.0);
    let a = run_evo(
        &seed,
        UpliftMode::LinearBelt,
        0.4,
        EvolutionSolverMode::Fast,
        None,
    );
    let b = run_evo(
        &seed,
        UpliftMode::LinearBelt,
        0.4,
        EvolutionSolverMode::Fast,
        None,
    );
    assert_eq!(a.elevation.to_dense(), b.elevation.to_dense());
}

#[test]
fn zero_erosion_has_explicit_bounded_limits_in_both_solvers() {
    let seed = plateau_cone_seed();
    for solver in [EvolutionSolverMode::Fast, EvolutionSolverMode::Accurate] {
        let mut p = LandscapeEvolutionParams::default();
        p.solver = solver;
        p.erosion = 0.0;
        p.hillslope_diffusion = 0.0;
        p.fixed_point_iters = 2;

        let out = evaluate_with_inputs(&seed, &p, None, None);
        assert!(
            out.elevation.to_dense().iter().all(|v| v.is_finite()),
            "{solver:?} zero-erosion output must stay finite"
        );
        let incision: f32 = out.incision.data().iter().copied().sum();
        assert!(
            incision <= 1e-5,
            "{solver:?} zero erosion must produce no incision, got {incision}"
        );

        match solver {
            EvolutionSolverMode::Fast => assert!(
                max_abs_delta_interior(&out.elevation, &out.tectonic_base, 1) <= 1e-4,
                "Fast zero erosion must publish the finite tectonic seed"
            ),
            EvolutionSolverMode::Accurate => {
                let (_, tectonic_max) = out.tectonic_base.min_max();
                let max_uplift = out
                    .uplift_field
                    .data()
                    .iter()
                    .copied()
                    .fold(0.0f32, f32::max);
                let (_, actual_max) = out.elevation.min_max();
                let uplift_budget = max_uplift * p.dt.max(1.0) * p.accurate_steps() as f32;
                assert!(
                    actual_max <= tectonic_max + uplift_budget + 1.0,
                    "Accurate zero erosion exceeded uplift-only budget: {actual_max}"
                );
            }
        }
    }
}

#[test]
fn zero_uplift_overrides_legacy_rate_without_disabling_incision() {
    let seed = plateau_cone_seed();
    let mut p = LandscapeEvolutionParams::default();
    p.uplift = 0.0;
    p.uplift_mode = UpliftMode::Uniform;
    p.hillslope_diffusion = 0.0;
    p.fixed_point_iters = 2;

    let out = evaluate_with_inputs(&seed, &p, None, None);
    assert!(out.uplift_field.data().iter().all(|&v| v == 0.0));
    assert!(
        max_abs_delta(&out.tectonic_base, &seed) <= 1e-6,
        "zero uplift must not alter the tectonic base"
    );
    assert!(out.elevation.to_dense().iter().all(|v| v.is_finite()));
}

#[test]
fn full_resistance_remains_no_incision_after_spatial_softening() {
    let seed = plateau_cone_seed();
    let m = seed.metrics;
    let lithology = MaskField::zeros(m);
    let precipitation = MaskField::filled(m, 2.5);
    for solver in [EvolutionSolverMode::Fast, EvolutionSolverMode::Accurate] {
        let mut p = LandscapeEvolutionParams::default();
        p.solver = solver;
        p.terrain_resistance = 1.0;
        p.hillslope_diffusion = 0.0;
        p.fixed_point_iters = 2;

        let out = evaluate_with_inputs(&seed, &p, Some(&lithology), Some(&precipitation));
        let incision: f32 = out.incision.data().iter().copied().sum();
        assert!(
            incision <= 1e-5,
            "{solver:?} full resistance leaked incision: {incision}"
        );
        if solver == EvolutionSolverMode::Fast {
            assert!(
                max_abs_delta_interior(&out.elevation, &out.tectonic_base, 1) <= 1e-4,
                "Fast full resistance must use the finite no-advection result"
            );
        }
    }
}

#[test]
fn mixed_hardness_terminates_characteristics_without_extreme_relief() {
    let seed = plateau_cone_seed();
    let m = seed.metrics;
    let mut hardness = MaskField::zeros(m);
    for j in 0..m.height {
        for i in m.width / 2..m.width {
            hardness.set(i, j, 1.0);
        }
    }
    let mut p = LandscapeEvolutionParams::default();
    p.hillslope_diffusion = 0.0;
    p.fixed_point_iters = 2;

    let out = evaluate_with_inputs(&seed, &p, Some(&hardness), None);
    assert!(out.elevation.to_dense().iter().all(|v| v.is_finite()));
    let mut hard_incision = 0.0f32;
    for j in 0..m.height {
        for i in m.width / 2..m.width {
            hard_incision += out.incision.get(i, j);
        }
    }
    assert!(
        hard_incision <= 1e-5,
        "non-advecting cells accumulated incision: {hard_incision}"
    );

    let (_, seed_max) = out.tectonic_base.min_max();
    let max_uplift = out
        .uplift_field
        .data()
        .iter()
        .copied()
        .fold(0.0f32, f32::max);
    let (_, actual_max) = out.elevation.min_max();
    assert!(
        actual_max <= seed_max + max_uplift * p.evolution_time() + 1.0,
        "mixed hardness exceeded the finite-time uplift envelope: {actual_max}"
    );
}

#[test]
fn default_fast_plateau_cone_retains_incision_and_bounded_relief() {
    let seed = plateau_cone_seed();
    let mut p = LandscapeEvolutionParams::default();
    p.hillslope_diffusion = 0.0;
    p.fixed_point_iters = 2;
    let out = evaluate_with_inputs(&seed, &p, None, None);
    let values = out.elevation.to_dense();
    assert!(values.iter().all(|v| v.is_finite()));
    assert!(
        out.incision.data().iter().copied().sum::<f32>() > 0.1,
        "default Fast path lost qualitative incision"
    );
    assert!(
        relief(&out.elevation) > 5.0 && relief(&out.elevation) < 1_000.0,
        "default Fast relief left its qualitative range: {}",
        relief(&out.elevation)
    );
}

fn boundary_ramp_seed() -> Heightfield {
    let m = HeightfieldMetrics::new(20, 20, 200.0, 200.0);
    let mut data = vec![0.0f32; (m.width * m.height) as usize];
    for j in 0..m.height {
        for i in 0..m.width {
            data[(j * m.width + i) as usize] = 10.0 + i as f32 * 0.37 + j as f32 * 0.19;
        }
    }
    data[(8 * m.width + 8) as usize] = -2.0;
    data[(8 * m.width + 9) as usize] = -1.0;
    Heightfield::from_dense(m, &data)
}

fn boundary_case(
    seed: &Heightfield,
    boundary: BoundaryMode,
    solver: EvolutionSolverMode,
    outlet_mask: Option<&MaskField>,
) -> terra_core::landscape_evolution::LandscapeEvolutionOutput {
    let mut p = LandscapeEvolutionParams::default();
    p.boundary = boundary;
    p.solver = solver;
    p.uplift_mode = UpliftMode::Uniform;
    p.uplift = 0.8;
    p.uplift_noise = 0.0;
    p.geological_age = 0.8;
    p.hillslope_diffusion = 0.15;
    p.constraint_preservation = 0.8;
    p.fixed_point_iters = 2;
    p.iterations = 9;
    p.time_scale = 300_000.0;
    p.dt = 5_000.0;
    p.base_level = 0.0;

    LandscapeEvolutionOperator::new(p).evaluate(LandscapeEvolutionInput {
        elevation: seed,
        painted_uplift: None,
        precipitation: None,
        erodibility: None,
        lithology_hardness: None,
        outlet_mask,
        protection: None,
    })
}

fn rim_coords(m: HeightfieldMetrics) -> Vec<(u32, u32)> {
    let mut coords = Vec::new();
    for j in 0..m.height {
        for i in 0..m.width {
            if i == 0 || j == 0 || i + 1 == m.width || j + 1 == m.height {
                coords.push((i, j));
            }
        }
    }
    coords
}

fn assert_same_bits_at(actual: &Heightfield, expected: &Heightfield, cells: &[(u32, u32)]) {
    for &(i, j) in cells {
        assert_eq!(
            actual.get(i, j).to_bits(),
            expected.get(i, j).to_bits(),
            "cell ({i},{j}) changed"
        );
    }
}

fn any_changed_at(actual: &Heightfield, expected: &Heightfield, cells: &[(u32, u32)]) -> bool {
    cells
        .iter()
        .any(|&(i, j)| actual.get(i, j).to_bits() != expected.get(i, j).to_bits())
}

#[test]
fn boundary_modes_separate_routing_outlets_from_elevation_locks() {
    let seed = boundary_ramp_seed();
    let m = seed.metrics;
    let rim = rim_coords(m);
    let submerged = [(8u32, 8u32), (9, 8)];
    let authored = [(5u32, 5u32), (14, 12)];
    let mut outlet_mask = MaskField::zeros(m);
    for &(i, j) in &authored {
        outlet_mask.set(i, j, 1.0);
    }

    for solver in [EvolutionSolverMode::Fast, EvolutionSolverMode::Accurate] {
        let fixed = boundary_case(&seed, BoundaryMode::Fixed, solver, None);
        assert_same_bits_at(&fixed.elevation, &seed, &rim);
        assert_same_bits_at(&fixed.tectonic_base, &seed, &rim);
        assert!(any_changed_at(&fixed.elevation, &seed, &[(10, 10)]));

        let open = boundary_case(&seed, BoundaryMode::OpenDrainage, solver, None);
        assert!(
            rim.iter().any(|&(i, j)| open.uplift_field.get(i, j) > 0.0),
            "{solver:?} OpenDrainage rim must receive uplift"
        );
        assert!(
            any_changed_at(&open.elevation, &seed, &rim),
            "{solver:?} OpenDrainage rim remained fixed"
        );
        assert!(any_changed_at(&open.tectonic_base, &seed, &rim));

        let sea = boundary_case(&seed, BoundaryMode::SeaLevel, solver, None);
        assert_same_bits_at(&sea.elevation, &seed, &submerged);
        assert_same_bits_at(&sea.tectonic_base, &seed, &submerged);
        assert!(
            rim.iter().any(|&(i, j)| sea.uplift_field.get(i, j) > 0.0),
            "{solver:?} above-sea routing rim was elevation-locked"
        );
        assert!(any_changed_at(&sea.elevation, &seed, &rim));

        let masked = boundary_case(&seed, BoundaryMode::OutletMask, solver, Some(&outlet_mask));
        assert_same_bits_at(&masked.elevation, &seed, &authored);
        assert_same_bits_at(&masked.tectonic_base, &seed, &authored);
        assert!(rim
            .iter()
            .any(|&(i, j)| masked.uplift_field.get(i, j) > 0.0));
        assert!(any_changed_at(&masked.elevation, &seed, &rim));
    }
}

#[test]
fn outlet_mask_missing_and_empty_fallbacks_do_not_invent_locks() {
    let seed = boundary_ramp_seed();
    let m = seed.metrics;
    let empty = MaskField::zeros(m);

    for solver in [EvolutionSolverMode::Fast, EvolutionSolverMode::Accurate] {
        let sea = boundary_case(&seed, BoundaryMode::SeaLevel, solver, None);
        let missing = boundary_case(&seed, BoundaryMode::OutletMask, solver, None);
        assert_eq!(sea.elevation.to_dense(), missing.elevation.to_dense());
        assert_eq!(
            sea.tectonic_base.to_dense(),
            missing.tectonic_base.to_dense()
        );

        let empty_out = boundary_case(&seed, BoundaryMode::OutletMask, solver, Some(&empty));
        assert!(empty_out.uplift_field.data().iter().all(|&v| v > 0.0));
        assert!(any_changed_at(&empty_out.elevation, &seed, &rim_coords(m)));
    }
}
