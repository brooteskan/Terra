//! Regression test for issues #98 / #101.
//!
//! A superseding edit must interrupt an in-flight long CPU eval *mid-fill*
//! instead of waiting it out. The stack is the #98 repro (a GPU-unsupported
//! `VoronoiRegions` generator plus a Flatten Shape layer); at a resolution where
//! a Full eval takes seconds, a follow-up job submitted while that eval is still
//! filling must return in a small fraction of the uncancelled time.
//!
//! Timing bounds are self-calibrated against this machine's own uncancelled
//! eval, so a slow debug CI runner cannot produce a false failure: the follow-up
//! is a 256² Draft (sub-millisecond of real compute) and the cancel latency is
//! roughly one 1024-wide row-chunk per rayon worker, both far under the
//! `t_full / 2` bound.

use std::collections::HashMap;
use std::thread::sleep;
use std::time::{Duration, Instant};

use terra_core::eval::{EvalWorkRequest, EvalWorker, EvalWorkerEvent, PreviewQuality};
use terra_core::generators::VoronoiParams;
use terra_core::heightfield::HeightfieldMetrics;
use terra_core::layer::{Layer, LayerKind, LayerStack};
use terra_core::shape_history::{create_shape_layer, stamp_stroke, ShapeTool};
use terra_core::CancelToken;

/// The #98 repro stack: a Voronoi base (no GPU kernel) plus a Flatten Shape
/// layer that contributes real per-pixel work.
fn voronoi_flatten_stack() -> LayerStack {
    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "Voronoi",
        LayerKind::VoronoiRegions(VoronoiParams::default()),
    ));
    let mut flat = create_shape_layer("Flatten");
    if let LayerKind::SculptStrokes(p) = &mut flat.kind {
        stamp_stroke(
            p,
            ShapeTool::Flatten.stroke_kind(),
            0.5,
            0.5,
            200.0,
            4.0,
            0.0,
            false,
        );
    }
    stack.push(flat);
    stack
}

fn request(
    token: u64,
    quality: PreviewQuality,
    preview_res: u32,
    metrics: HeightfieldMetrics,
) -> EvalWorkRequest {
    EvalWorkRequest {
        token,
        quality,
        stack: voronoi_flatten_stack(),
        masks: Vec::new(),
        base_metrics: metrics,
        level_steps: terra_core::analyze::LevelStepSettings::default(),
        preview_res,
        export_res: preview_res,
        aux: HashMap::new(),
        strata: None,
        mask_reference: None,
        dirty_from: None,
        dirty_region: None,
        mark_all_dirty: true,
    }
}

/// Poll until the worker reports completion of `token`, returning the elapsed
/// wait. Stale events for superseded tokens are ignored, matching how the UI
/// consumes worker events. Panics on failure, disconnect, or deadline.
fn wait_for_completion(worker: &mut EvalWorker, token: u64, deadline: Duration) -> Duration {
    let start = Instant::now();
    loop {
        while let Some(event) = worker.try_recv_event() {
            match event {
                EvalWorkerEvent::Completed(result) if result.token == token => {
                    return start.elapsed();
                }
                // A superseded job that still managed to publish — ignore it.
                EvalWorkerEvent::Completed(_) => {}
                EvalWorkerEvent::Failed(failure) if failure.token == token => {
                    panic!("token {token} failed: {}", failure.error);
                }
                EvalWorkerEvent::Failed(_) => {}
                EvalWorkerEvent::Disconnected => panic!("worker disconnected"),
            }
        }
        assert!(
            start.elapsed() < deadline,
            "timed out waiting for token {token}"
        );
        sleep(Duration::from_millis(2));
    }
}

#[test]
fn superseding_job_interrupts_in_flight_cpu_eval() {
    // 1024² keeps a debug-build Full eval in the multi-second range (per the #98
    // table, ~1/4 of the 2048² cost) while staying quick enough for CI.
    let metrics = HeightfieldMetrics::new(1024, 1024, 4096.0, 4096.0);
    let mut worker = EvalWorker::spawn();

    // Baseline: how long an uncancelled Full eval of this stack actually takes.
    worker
        .submit(request(1, PreviewQuality::Full, 1024, metrics))
        .expect("submit baseline");
    let t_full = wait_for_completion(&mut worker, 1, Duration::from_secs(120));

    // Submit a second Full job, let it get well into the fill, then supersede it
    // with a small Draft follow-up. The Draft can only return promptly if the
    // stale Full eval was interrupted mid-fill.
    worker
        .submit(request(2, PreviewQuality::Full, 1024, metrics))
        .expect("submit superseded");
    // At most a quarter into the baseline: job 2 is provably still filling, and
    // it cannot have completed before we supersede it.
    sleep((t_full / 4).min(Duration::from_millis(200)));

    let follow_up_start = Instant::now();
    worker
        .submit(request(3, PreviewQuality::Draft, 256, metrics))
        .expect("submit follow-up");
    let _ = wait_for_completion(&mut worker, 3, Duration::from_secs(120));
    let follow_up = follow_up_start.elapsed();

    // Discriminator: without mid-fill cancellation the follow-up would queue
    // behind the ~t_full stale eval. Skip the ratio when the baseline is too
    // fast to distinguish (e.g. an unexpectedly quick release-mode runner).
    if t_full >= Duration::from_millis(500) {
        assert!(
            follow_up < t_full / 2,
            "follow-up took {follow_up:?}, not a small fraction of the \
             uncancelled eval {t_full:?} — the stale eval was not interrupted"
        );
    }
}

#[test]
fn voronoi_regions_reports_cancellation() {
    let metrics = HeightfieldMetrics::new(64, 64, 512.0, 512.0);
    let params = VoronoiParams::default();

    // A live token fills to completion.
    assert!(
        terra_core::generators::voronoi_regions(metrics, &CancelToken::never(), &params).is_some()
    );

    // A pre-cancelled token stops before completing.
    let (token, flag) = CancelToken::flag();
    flag.cancel();
    assert!(terra_core::generators::voronoi_regions(metrics, &token, &params).is_none());
}
