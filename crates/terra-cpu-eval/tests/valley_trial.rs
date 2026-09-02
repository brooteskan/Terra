//! Valley + river usability trial (plan: mountain valley, main river, tributary creeks).
//!
//! Mirrors the "River Valley" UI preset and tries Path / RiverNetwork tributary polish.
//! Writes diagnostic PNGs under `target/valley_trial/` when `VALLEY_TRIAL_OUT=1`
//! or when run with `--ignored` via the explicit export test.
//!
//! Run:
//! ```text
//! cargo test -p terra-core --test valley_trial -- --nocapture
//! cargo test -p terra-core --test valley_trial export_valley_diagnostics -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use terra_core::fields::keys;
use terra_core::geomorph::{accumulate_drainage_area, build_flow_graph, FlowModel, Precipitation};
use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
use terra_core::layer::{
    CoastalParams, HydraulicErosionParams, Layer, LayerKind, LayerStack, MaterialsParams, PathNode,
    PathParams, RiverCarveParams, RiverNetworkParams, RiverNode, SculptParams, StreamPowerParams,
    UpliftParams,
};
use terra_core::mask::MaskField;
use terra_core::quality::PreviewQuality;
use terra_cpu_eval::{EvalContext, StackEvaluator};

fn out_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("target")
        .join("valley_trial")
}

fn metrics(res: u32) -> HeightfieldMetrics {
    HeightfieldMetrics::new(res, res, 4000.0, 4000.0)
}

/// Same layer kinds as the UI "River Valley" preset (iters scaled for fixture time).
fn river_valley_stack(spe_iters: u32, hydro_iters: u32) -> LayerStack {
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "Base",
        LayerKind::SculptBase(SculptParams::filled(512, 45.0)),
    ));
    stack.push(Layer::new(
        "Uplift",
        LayerKind::Uplift(UpliftParams {
            amplitude: 360.0,
            corridor_width: 0.4,
            ..UpliftParams::default()
        }),
    ));
    stack.push(Layer::new(
        "Stream Power",
        LayerKind::StreamPowerErosion(StreamPowerParams {
            iterations: spe_iters,
            k: 0.08,
            dendritic_seed: 0.45,
            level_step_strength: 1.1,
            ..StreamPowerParams::default()
        }),
    ));
    let mut fill = HydraulicErosionParams::depositional();
    fill.iterations = hydro_iters;
    stack.push(Layer::new("Valley Fill", LayerKind::HydraulicErosion(fill)));
    stack.push(Layer::new(
        "River Carve",
        LayerKind::RiverCarve(RiverCarveParams {
            accumulation_threshold: 35.0,
            depth: 32.0,
            width: 6.0,
            ..RiverCarveParams::default()
        }),
    ));
    stack.push(Layer::new(
        "Coastal Edge",
        LayerKind::Coastal(CoastalParams {
            sea_level: 30.0,
            ..CoastalParams::default()
        }),
    ));
    stack.push(Layer::new(
        "Materials",
        LayerKind::Materials(MaterialsParams::default()),
    ));
    stack
}

/// Attempt A: denser dendritic SPE + lower carve threshold (more side channels).
fn attempt_a_tuned(spe_iters: u32, hydro_iters: u32) -> LayerStack {
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "Base",
        LayerKind::SculptBase(SculptParams::filled(512, 45.0)),
    ));
    stack.push(Layer::new(
        "Uplift",
        LayerKind::Uplift(UpliftParams {
            amplitude: 360.0,
            corridor_width: 0.4,
            ..UpliftParams::default()
        }),
    ));
    stack.push(Layer::new(
        "Stream Power",
        LayerKind::StreamPowerErosion(StreamPowerParams {
            iterations: spe_iters.saturating_add(4),
            k: 0.1,
            dendritic_seed: 0.75,
            level_step_strength: 1.1,
            ..StreamPowerParams::default()
        }),
    ));
    let mut fill = HydraulicErosionParams::depositional();
    fill.iterations = hydro_iters;
    stack.push(Layer::new("Valley Fill", LayerKind::HydraulicErosion(fill)));
    stack.push(Layer::new(
        "River Carve",
        LayerKind::RiverCarve(RiverCarveParams {
            accumulation_threshold: 18.0,
            depth: 40.0,
            width: 5.0,
            ..RiverCarveParams::default()
        }),
    ));
    stack.push(Layer::new(
        "Coastal Edge",
        LayerKind::Coastal(CoastalParams {
            sea_level: 30.0,
            ..CoastalParams::default()
        }),
    ));
    stack.push(Layer::new(
        "Materials",
        LayerKind::Materials(MaterialsParams::default()),
    ));
    stack
}

/// Attempt B: River Valley + multi-spring RiverNetwork (mountain creeks).
fn attempt_b_network(spe_iters: u32, hydro_iters: u32) -> LayerStack {
    let mut stack = river_valley_stack(spe_iters, hydro_iters);
    stack.push(Layer::new(
        "Creek Network",
        LayerKind::RiverNetwork(RiverNetworkParams {
            springs: vec![
                RiverNode {
                    u: 0.22,
                    v: 0.18,
                    flow: 1.2,
                    width: 1.0,
                },
                RiverNode {
                    u: 0.78,
                    v: 0.20,
                    flow: 1.1,
                    width: 0.95,
                },
                RiverNode {
                    u: 0.35,
                    v: 0.12,
                    flow: 1.0,
                    width: 0.85,
                },
                RiverNode {
                    u: 0.62,
                    v: 0.14,
                    flow: 1.0,
                    width: 0.9,
                },
            ],
            auto_generate: true,
            max_length: 400,
            // World is 4 km; cell ≈ 40 m at 96² — radius must exceed one cell.
            carve_depth: 35.0,
            valley_width: 90.0,
            seed: 7,
        }),
    ));
    stack
}

/// Attempt C: authored Path corridor from ridge into valley axis.
fn attempt_c_paths(spe_iters: u32, hydro_iters: u32) -> LayerStack {
    let mut stack = river_valley_stack(spe_iters, hydro_iters);
    stack.push(Layer::new(
        "Creek Path N",
        LayerKind::Path(PathParams {
            nodes: vec![
                PathNode {
                    u: 0.25,
                    v: 0.12,
                    height: 0.0,
                    width: 0.7,
                },
                PathNode {
                    u: 0.38,
                    v: 0.32,
                    height: 0.0,
                    width: 0.9,
                },
                PathNode {
                    u: 0.48,
                    v: 0.48,
                    height: 0.0,
                    width: 1.1,
                },
            ],
            width: 8.0,
            falloff: 14.0,
            height_offset: -10.0,
            carve: true,
            ..PathParams::default()
        }),
    ));
    stack.push(Layer::new(
        "Creek Path S",
        LayerKind::Path(PathParams {
            nodes: vec![
                PathNode {
                    u: 0.78,
                    v: 0.15,
                    height: 0.0,
                    width: 0.6,
                },
                PathNode {
                    u: 0.65,
                    v: 0.35,
                    height: 0.0,
                    width: 0.85,
                },
                PathNode {
                    u: 0.55,
                    v: 0.50,
                    height: 0.0,
                    width: 1.0,
                },
            ],
            width: 7.0,
            falloff: 12.0,
            height_offset: -9.0,
            carve: true,
            seed: 3,
            ..PathParams::default()
        }),
    ));
    stack
}

struct TrialResult {
    name: &'static str,
    height: Heightfield,
    flow: Option<MaskField>,
    wetness: Option<MaskField>,
    channel_frac: f32,
    valley_flow: f32,
    relief: f32,
}

fn eval_stack(stack: &LayerStack, res: u32, quality: PreviewQuality) -> TrialResult {
    let m = metrics(res);
    let mut eval = StackEvaluator::new();
    let mut ctx = EvalContext::new(m);
    ctx.quality = quality;
    let height = eval.rebuild_all(stack, &mut ctx).expect("rebuild");
    let flow = ctx.aux.get(keys::FLOW_ACCUMULATION).cloned();
    let wetness = ctx.aux.get(keys::WETNESS).cloned();
    let dense = height.to_dense();
    let min_h = dense.iter().copied().fold(f32::INFINITY, f32::min);
    let max_h = dense.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let relief = max_h - min_h;

    let acc = flow.as_ref().map(|f| f.data().to_vec()).unwrap_or_else(|| {
        let graph = build_flow_graph(&height, FlowModel::D8);
        accumulate_drainage_area(&graph, &Precipitation::uniform(1.0))
    });
    let max_acc = acc.iter().copied().fold(1.0f32, f32::max);
    let threshold = max_acc * 0.08;
    let channel_cells = acc.iter().filter(|&&a| a >= threshold).count();
    let channel_frac = channel_cells as f32 / acc.len() as f32;

    let mut heights: Vec<f32> = dense.clone();
    heights.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = heights[heights.len() / 4];
    let mut sum = 0.0f32;
    let mut n = 0u32;
    for (h, &a) in dense.iter().zip(acc.iter()) {
        if *h <= q {
            sum += a / max_acc;
            n += 1;
        }
    }
    let valley_flow = if n == 0 { 0.0 } else { sum / n as f32 };

    TrialResult {
        name: "",
        height,
        flow,
        wetness,
        channel_frac,
        valley_flow,
        relief,
    }
}

fn write_height_png(hf: &Heightfield, path: &std::path::Path) {
    let w = hf.metrics.width;
    let h = hf.metrics.height;
    let data = hf.to_dense();
    let min = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let span = (max - min).max(1e-5);
    let mut img = image::GrayImage::new(w, h);
    for j in 0..h {
        for i in 0..w {
            let t = ((data[(j * w + i) as usize] - min) / span * 255.0).clamp(0.0, 255.0) as u8;
            img.put_pixel(i, j, image::Luma([t]));
        }
    }
    let _ = img.save(path);
}

fn write_mask_png(mask: &MaskField, path: &std::path::Path, log_scale: bool) {
    let w = mask.metrics.width;
    let h = mask.metrics.height;
    let data = mask.data();
    let max = data.iter().copied().fold(1e-5f32, f32::max);
    let mut img = image::GrayImage::new(w, h);
    for j in 0..h {
        for i in 0..w {
            let v = data[(j * w + i) as usize];
            let t = if log_scale {
                (v.max(0.0).ln_1p() / max.ln_1p() * 255.0).clamp(0.0, 255.0)
            } else {
                (v / max * 255.0).clamp(0.0, 255.0)
            } as u8;
            img.put_pixel(i, j, image::Luma([t]));
        }
    }
    let _ = img.save(path);
}

fn export_trial(label: &str, result: &TrialResult) {
    let dir = out_dir().join(label);
    let _ = std::fs::create_dir_all(&dir);
    write_height_png(&result.height, &dir.join("height.png"));
    if let Some(flow) = &result.flow {
        write_mask_png(flow, &dir.join("flow.png"), true);
    } else {
        let graph = build_flow_graph(&result.height, FlowModel::D8);
        let acc = accumulate_drainage_area(&graph, &Precipitation::uniform(1.0));
        let mut field = MaskField::zeros(result.height.metrics);
        for j in 0..result.height.metrics.height {
            for i in 0..result.height.metrics.width {
                field.set(i, j, acc[(j * result.height.metrics.width + i) as usize]);
            }
        }
        write_mask_png(&field, &dir.join("flow.png"), true);
    }
    if let Some(w) = &result.wetness {
        write_mask_png(w, &dir.join("wetness.png"), false);
    }
    let report = format!(
        "label={label}\nrelief_m={:.1}\nchannel_frac={:.4}\nvalley_flow_score={:.4}\nhas_flow_aux={}\nhas_wetness_aux={}\n",
        result.relief,
        result.channel_frac,
        result.valley_flow,
        result.flow.is_some(),
        result.wetness.is_some(),
    );
    let _ = std::fs::write(dir.join("metrics.txt"), report);
}

#[test]
fn baseline_river_valley_produces_relief_and_channels() {
    let stack = river_valley_stack(12, 20);
    let r = eval_stack(&stack, 128, PreviewQuality::Draft);
    println!(
        "baseline relief={:.1}m channel_frac={:.3} valley_flow={:.3} flow_aux={}",
        r.relief,
        r.channel_frac,
        r.valley_flow,
        r.flow.is_some()
    );
    assert!(r.relief > 70.0, "valley should have meaningful relief");
    assert!(
        r.channel_frac > 0.01,
        "RiverCarve/SPE should leave a channel network"
    );
    assert!(r.flow.is_some(), "SPE/RiverCarve should publish flow aux");
    // Note: valley_flow (accum in lowest height quartile) is often weak after Coastal
    // flattens the outlet — channels exist but are not synonymous with "lowest cells".
}

#[test]
fn attempt_a_tuned_changes_drainage_pattern() {
    let base = eval_stack(&river_valley_stack(12, 20), 128, PreviewQuality::Draft);
    let tuned = eval_stack(&attempt_a_tuned(12, 20), 128, PreviewQuality::Draft);
    let a = base.height.to_dense();
    let b = tuned.height.to_dense();
    let mean_abs: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f32>()
        / a.len() as f32;
    println!(
        "A base_channel={:.3} tuned_channel={:.3} mean_abs_delta={:.3} base_valley={:.3} tuned_valley={:.3}",
        base.channel_frac, tuned.channel_frac, mean_abs, base.valley_flow, tuned.valley_flow
    );
    assert!(
        mean_abs > 0.5,
        "retuning SPE/RiverCarve should change the DEM (got {mean_abs})"
    );
    // Product finding: lowering accumulation_threshold does NOT reliably raise
    // channel_frac under a relative (max*0.08) definition — network shape changes.
}

#[test]
fn attempt_b_river_network_carves_additional_tracks() {
    let base = eval_stack(&river_valley_stack(10, 16), 96, PreviewQuality::Draft);
    let net = eval_stack(&attempt_b_network(10, 16), 96, PreviewQuality::Draft);
    let a = base.height.to_dense();
    let b = net.height.to_dense();
    let mean_abs: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f32>()
        / a.len() as f32;
    println!(
        "B mean_abs_delta={mean_abs:.4} channel_frac={:.3}",
        net.channel_frac
    );
    assert!(
        mean_abs > 0.01,
        "RiverNetwork springs should carve measurable tracks (got {mean_abs})"
    );
}

#[test]
fn attempt_c_path_carves_authored_corridors() {
    let base = eval_stack(&river_valley_stack(10, 16), 96, PreviewQuality::Draft);
    let paths = eval_stack(&attempt_c_paths(10, 16), 96, PreviewQuality::Draft);
    let a = base.height.to_dense();
    let b = paths.height.to_dense();
    let mean_abs: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f32>()
        / a.len() as f32;
    println!("C mean_abs_delta={mean_abs:.4}");
    assert!(
        mean_abs > 0.05,
        "Path creek stamps should carve measurable corridors (got {mean_abs})"
    );
}

#[test]
#[ignore = "writes PNGs under target/valley_trial; run explicitly"]
fn export_valley_diagnostics() {
    let dir = out_dir();
    let _ = std::fs::create_dir_all(&dir);

    let cases: [(&str, LayerStack); 4] = [
        ("01_baseline", river_valley_stack(16, 28)),
        ("02_attempt_a_tuned", attempt_a_tuned(16, 28)),
        ("03_attempt_b_network", attempt_b_network(14, 24)),
        ("04_attempt_c_paths", attempt_c_paths(14, 24)),
    ];

    let mut summary = String::from("# Valley + river trial report\n\n");
    summary.push_str("Resolution: 192² Draft-quality eval (UI Full would be higher).\n\n");

    for (label, stack) in cases {
        let mut r = eval_stack(&stack, 192, PreviewQuality::Draft);
        r.name = label;
        export_trial(label, &r);
        summary.push_str(&format!(
            "## {label}\n- relief: {:.1} m\n- channel_frac: {:.4}\n- valley_flow: {:.4}\n- flow_aux: {}\n- wetness_aux: {}\n\n",
            r.relief,
            r.channel_frac,
            r.valley_flow,
            r.flow.is_some(),
            r.wetness.is_some(),
        ));
        println!(
            "exported {label}: relief={:.1} channel={:.3} valley_flow={:.3}",
            r.relief, r.channel_frac, r.valley_flow
        );
    }

    summary.push_str(
        "## Verdict notes\n\
         - Baseline uses the River Valley UI recipe (Uplift→SPE→fill→RiverCarve→Coast).\n\
         - Tributaries are drainage-emergent via SPE+RiverCarve; Path/RiverNetwork can add tracks but lack viewport authoring and confluence graphs.\n\
         - Wetness aux is typically absent after RiverCarve (carve only lowers DEM / publishes flow).\n\
         - See PNGs beside this report for height + log-flow overlays.\n",
    );
    let _ = std::fs::write(dir.join("REPORT.md"), summary);
}
