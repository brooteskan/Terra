use terra_core::authoring::{
    apply_sculpt_strokes, SculptPoint, SculptStroke, SculptStrokeKind, SculptStrokeParams,
};
use terra_core::{Heightfield, HeightfieldMetrics};

fn smooth(strength: f32) -> SculptStroke {
    SculptStroke {
        kind: SculptStrokeKind::Smooth,
        points: vec![SculptPoint {
            u: 0.5,
            v: 0.5,
            pressure: 1.0,
        }],
        radius_m: 100.0,
        strength,
        target_height: 0.0,
        falloff: 1.5,
        enabled: true,
    }
}

fn smooth_with_spread(strength: f32, spread_samples: u32) -> SculptStroke {
    SculptStroke {
        target_height: spread_samples as f32,
        ..smooth(strength)
    }
}

fn apply(input: &Heightfield, strokes: Vec<SculptStroke>) -> Heightfield {
    apply_sculpt_strokes(
        input,
        &SculptStrokeParams {
            strokes,
            // Smooth supplies its own constrained edge transition and must not be
            // fed through the legacy height-mean reconcile pass.
            reconcile: 0.15,
        },
    )
    .height
}

fn max_abs_delta(a: &Heightfield, b: &Heightfield) -> f32 {
    a.to_dense()
        .into_iter()
        .zip(b.to_dense())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn crease_energy(h: &Heightfield) -> f64 {
    let m = h.metrics;
    let j = m.height / 2;
    (1..m.width - 1)
        .map(|i| {
            let second = h.get(i - 1, j) - 2.0 * h.get(i, j) + h.get(i + 1, j);
            f64::from(second * second)
        })
        .sum()
}

#[test]
fn flat_and_zero_strength_are_exact_noops() {
    let metrics = HeightfieldMetrics::new(65, 49, 320.0, 240.0);
    let flat = Heightfield::filled(metrics, 20.0);
    assert_eq!(apply(&flat, vec![smooth(1.0)]).to_dense(), flat.to_dense());

    let mut varied = Heightfield::zeros(metrics);
    for j in 0..metrics.height {
        for i in 0..metrics.width {
            varied.set(i, j, (i as f32 * 0.31).sin() * 8.0 + j as f32 * 0.2);
        }
    }
    assert_eq!(
        apply(&varied, vec![smooth(0.0)]).to_dense(),
        varied.to_dense()
    );
}

#[test]
fn affine_plane_is_not_materially_deformed_with_rectangular_cells() {
    let metrics = HeightfieldMetrics::new(65, 49, 640.0, 240.0);
    let mut plane = Heightfield::zeros(metrics);
    for j in 0..metrics.height {
        for i in 0..metrics.width {
            plane.set(i, j, 13.0 + 0.35 * i as f32 - 0.2 * j as f32);
        }
    }
    let out = apply(&plane, vec![smooth(1.0)]);
    let plane_error = max_abs_delta(&plane, &out);
    assert!(
        plane_error <= 8.0e-2,
        "Smooth changed an affine plane by {plane_error}m"
    );
}

#[test]
fn crease_is_reduced_without_range_overshoot() {
    let metrics = HeightfieldMetrics::new(65, 65, 320.0, 320.0);
    let mut crease = Heightfield::zeros(metrics);
    let center = metrics.width as f32 * 0.5;
    for j in 0..metrics.height {
        for i in 0..metrics.width {
            crease.set(
                i,
                j,
                30.0 + (i as f32 - center).abs() * 1.5 + j as f32 * 0.05,
            );
        }
    }
    let before_energy = crease_energy(&crease);
    let before = crease.to_dense();
    let before_min = before.iter().copied().fold(f32::INFINITY, f32::min);
    let before_max = before.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let out = apply(&crease, vec![smooth(1.0)]);
    let after = out.to_dense();

    assert!(crease_energy(&out) < before_energy);
    assert!(after.iter().all(|&h| h >= before_min - 1.0e-5));
    assert!(after.iter().all(|&h| h <= before_max + 1.0e-5));

    // Symmetric brush conductance closes the diffusion at the radial support.
    // Corners are therefore exact, not merely close.
    assert_eq!(out.get(0, 0).to_bits(), crease.get(0, 0).to_bits());
    assert_eq!(
        out.get(metrics.width - 1, metrics.height - 1).to_bits(),
        crease.get(metrics.width - 1, metrics.height - 1).to_bits()
    );
}

#[test]
fn strength_scales_magnitude_and_repeated_strokes_converge_without_overshoot() {
    let metrics = HeightfieldMetrics::new(65, 65, 320.0, 320.0);
    let mut crease = Heightfield::zeros(metrics);
    for j in 0..metrics.height {
        for i in 0..metrics.width {
            crease.set(i, j, (i as f32 - 32.0).abs() * 2.0);
        }
    }
    let weak = apply(&crease, vec![smooth(0.25)]);
    let medium = apply(&crease, vec![smooth(0.5)]);
    let strong = apply(&crease, vec![smooth(1.0)]);
    let weak_delta = max_abs_delta(&crease, &weak);
    let medium_delta = max_abs_delta(&crease, &medium);
    let strong_delta = max_abs_delta(&crease, &strong);
    assert!(weak_delta > 0.0 && weak_delta < medium_delta && medium_delta < strong_delta);

    let repeated = apply(&crease, vec![smooth(1.0); 32]);
    let repeated_dense = repeated.to_dense();
    let before = crease.to_dense();
    let before_min = before.iter().copied().fold(f32::INFINITY, f32::min);
    let before_max = before.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(repeated_dense.iter().all(|&h| h >= before_min - 1.0e-5));
    assert!(repeated_dense.iter().all(|&h| h <= before_max + 1.0e-5));
    assert!(crease_energy(&repeated) < crease_energy(&strong));
}

#[test]
fn smooth_visibly_rounds_terrace_corners_at_normal_brush_strength() {
    let metrics = HeightfieldMetrics::new(129, 65, 320.0, 160.0);
    let mut ramp = Heightfield::zeros(metrics);
    for j in 0..metrics.height {
        for i in 0..metrics.width {
            ramp.set(i, j, i as f32 * 0.8 + j as f32 * 0.03);
        }
    }
    let terrace = SculptStroke {
        kind: SculptStrokeKind::Terrace,
        points: vec![SculptPoint {
            u: 0.5,
            v: 0.5,
            pressure: 1.0,
        }],
        radius_m: 120.0,
        strength: 8.0,
        target_height: 0.0,
        falloff: 1.5,
        enabled: true,
    };
    let terraced = apply(&ramp, vec![terrace.clone()]);
    let rounded = apply(&ramp, vec![terrace, smooth(0.4)]);
    let before = crease_energy(&terraced);
    let after = crease_energy(&rounded);
    let displacement = max_abs_delta(&terraced, &rounded);
    let terraced_dense = terraced.to_dense();
    let rounded_dense = rounded.to_dense();
    let terraced_min = terraced_dense.iter().copied().fold(f32::INFINITY, f32::min);
    let terraced_max = terraced_dense
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    assert!(
        after < before * 0.72,
        "Smooth left Terrace corners effectively unchanged: energy {before} -> {after}, max displacement {displacement}m"
    );
    assert!(
        displacement >= 0.25,
        "normal Smooth strength moved terrain by only {displacement}m"
    );
    assert!(
        rounded_dense.iter().all(|&h| h >= terraced_min - 1.0e-5),
        "Smooth overshot below the Terrace range"
    );
    assert!(
        rounded_dense.iter().all(|&h| h <= terraced_max + 1.0e-5),
        "Smooth overshot above the Terrace range"
    );
}

#[test]
fn spread_control_broadens_a_sharp_transition() {
    let metrics = HeightfieldMetrics::new(129, 65, 320.0, 160.0);
    let mut step = Heightfield::zeros(metrics);
    for j in 0..metrics.height {
        for i in metrics.width / 2..metrics.width {
            step.set(i, j, 10.0);
        }
    }

    let narrow = apply(&step, vec![smooth_with_spread(1.0, 1)]);
    let wide = apply(&step, vec![smooth_with_spread(1.0, 8)]);
    let center = metrics.height / 2;
    let transition_width = |height: &Heightfield| {
        (0..metrics.width)
            .filter(|&i| {
                let value = height.get(i, center);
                value > 1.0e-3 && value < 10.0 - 1.0e-3
            })
            .count()
    };
    let narrow_width = transition_width(&narrow);
    let wide_width = transition_width(&wide);

    assert!(
        wide_width >= narrow_width.saturating_mul(2),
        "Spread did not materially broaden the edge: narrow={narrow_width}, wide={wide_width} samples"
    );
    assert!(
        (wide.get(48, center) - step.get(48, center)).abs() > 1.0e-3,
        "wide Spread did not reach 16 samples from the original edge"
    );
    assert!(
        (narrow.get(48, center) - step.get(48, center)).abs() <= 1.0e-3,
        "narrow Spread materially reached the wide transition band (narrow={narrow_width}, wide={wide_width})"
    );
}

#[test]
fn hundred_sample_spread_is_continuous_without_new_terraces() {
    let metrics = HeightfieldMetrics::new(257, 257, 256.0, 256.0);
    let mut step = Heightfield::zeros(metrics);
    for j in 0..metrics.height {
        for i in metrics.width / 2..metrics.width {
            step.set(i, j, 10.0);
        }
    }

    let wide = apply(
        &step,
        vec![SculptStroke {
            radius_m: 220.0,
            ..smooth_with_spread(1.0, 100)
        }],
    );
    let center = metrics.height / 2;
    let row: Vec<f32> = (0..metrics.width).map(|i| wide.get(i, center)).collect();
    let transition_width = row
        .iter()
        .filter(|&&value| value > 1.0e-3 && value < 10.0 - 1.0e-3)
        .count();
    let reversal_depth = row
        .windows(2)
        .map(|pair| (pair[0] - pair[1]).max(0.0))
        .fold(0.0f32, f32::max);
    let maximum_local_rise = row
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .fold(0.0f32, f32::max);

    assert!(
        transition_width >= 50,
        "Spread 100 produced only a {transition_width}-sample transition"
    );
    assert!(
        reversal_depth <= 1.0e-3,
        "Spread 100 introduced a {reversal_depth}m slope reversal/terrace band"
    );
    assert!(
        maximum_local_rise <= 0.5,
        "Spread 100 retained a {maximum_local_rise}m one-sample terrace riser inside its broad transition"
    );
}

#[test]
fn off_center_terrace_edge_has_no_brush_boundary_collar() {
    let metrics = HeightfieldMetrics::new(257, 129, 256.0, 128.0);
    let mut step = Heightfield::zeros(metrics);
    // Put the riser well off the brush centre so its broadened transition meets
    // the circular falloff. A closed diffusion boundary piles the displaced
    // height against that falloff and creates the concentric collars visible in
    // the editor, even though a centred one-dimensional fixture stays monotone.
    for j in 0..metrics.height {
        for i in 190..metrics.width {
            step.set(i, j, 10.0);
        }
    }

    let stroke = SculptStroke {
        radius_m: 96.0,
        target_height: 48.0,
        falloff: 1.5,
        ..smooth(1.0)
    };
    let out = apply(&step, vec![stroke]);
    let center = metrics.height / 2;
    let row: Vec<f32> = (0..metrics.width).map(|i| out.get(i, center)).collect();
    let reversal_depth = row
        .windows(2)
        .map(|pair| (pair[0] - pair[1]).max(0.0))
        .fold(0.0f32, f32::max);

    assert!(
        reversal_depth <= 5.0e-2,
        "Smooth formed a {reversal_depth}m collar where the off-centre terrace edge met the brush falloff"
    );
}
