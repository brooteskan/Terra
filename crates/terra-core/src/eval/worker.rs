//! Background CPU evaluation worker (Gaea-style engine/UI split).
//!
//! All CPU stack eval (Draft refine fallback, Medium, Full) runs off the UI thread.
//! Interactive Draft prefers GPU present; jobs are cancelled by bumping `current_token`.

use super::{EvalContext, EvalError, PreviewQuality, StackEvaluator};
use crate::heightfield::{Heightfield, HeightfieldMetrics};
use crate::layer::{LayerId, LayerStack};
use crate::mask::{bake_mask_assets, MaskAsset, MaskField};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct EvalWorkRequest {
    pub token: u64,
    pub quality: PreviewQuality,
    pub stack: LayerStack,
    pub masks: Vec<MaskAsset>,
    pub base_metrics: HeightfieldMetrics,
    pub level_steps: crate::analyze::LevelStepSettings,
    pub preview_res: u32,
    pub export_res: u32,
    pub aux: HashMap<String, MaskField>,
    /// Depth-aware Materials strata (not representable in the aux HashMap).
    pub strata: Option<Vec<crate::layer::Stratum>>,
    /// Prior composed height for baking height/slope/curvature masks.
    /// Falls back to zeros only on the first build of a generation.
    pub mask_reference: Option<std::sync::Arc<Heightfield>>,
    /// When set, only this layer and above are dirty (suffix rebuild).
    pub dirty_from: Option<LayerId>,
    pub mark_all_dirty: bool,
}

#[derive(Debug)]
pub struct EvalWorkResult {
    pub token: u64,
    pub quality: PreviewQuality,
    pub height: Heightfield,
    pub aux: HashMap<String, MaskField>,
    pub strata: Option<Vec<crate::layer::Stratum>>,
    pub eval_us: u64,
    pub layer_timings: Vec<super::LayerEvalTiming>,
}

#[derive(Debug)]
pub struct EvalWorkFailure {
    pub token: u64,
    pub quality: PreviewQuality,
    pub error: EvalError,
}

#[derive(Debug)]
pub enum EvalWorkerEvent {
    Completed(EvalWorkResult),
    Failed(EvalWorkFailure),
    /// The worker result channel closed, normally because its thread terminated.
    Disconnected,
}

impl EvalWorkerEvent {
    fn token(&self) -> Option<u64> {
        match self {
            Self::Completed(result) => Some(result.token),
            Self::Failed(failure) => Some(failure.token),
            Self::Disconnected => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("evaluation worker is disconnected")]
pub struct EvalWorkerSubmitError;

enum WorkerMsg {
    // Boxed so the common `Shutdown` and idle channel slots aren't sized to the large
    // `EvalWorkRequest` payload (clippy::large_enum_variant).
    Job(Box<EvalWorkRequest>),
    Shutdown,
}

/// Owns a dedicated thread with its own [`StackEvaluator`] and layer cache.
pub struct EvalWorker {
    tx: Sender<WorkerMsg>,
    rx: Receiver<EvalWorkerEvent>,
    /// Shared cancel / generation id — worker skips jobs with older tokens.
    pub current_token: Arc<AtomicU64>,
    _handle: JoinHandle<()>,
    /// True while a job may still be running (best-effort).
    pub busy: bool,
    disconnected_reported: bool,
}

impl EvalWorker {
    pub fn spawn() -> Self {
        let (job_tx, job_rx) = mpsc::channel::<WorkerMsg>();
        let (result_tx, result_rx) = mpsc::channel::<EvalWorkerEvent>();
        let current_token = Arc::new(AtomicU64::new(0));
        let token_flag = Arc::clone(&current_token);

        let handle = thread::Builder::new()
            .name("terra-eval-worker".into())
            .spawn(move || {
                let mut evaluator = StackEvaluator::new();
                while let Ok(msg) = job_rx.recv() {
                    match msg {
                        WorkerMsg::Shutdown => break,
                        WorkerMsg::Job(job) => {
                            let live = token_flag.load(Ordering::Acquire);
                            if job.token != live {
                                continue;
                            }
                            let result =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    run_cpu_job(&mut evaluator, &job, &token_flag)
                                }));
                            let result = match result {
                                Ok(result) => result,
                                Err(payload) => {
                                    // A job-level panic may leave cache state partially updated.
                                    // Start subsequent requests from a fresh evaluator, but carry
                                    // the disk spill store (and its private root) across so pinned
                                    // bakes written before the panic survive the restart as a
                                    // declared handoff (B1-D8) rather than filesystem coincidence.
                                    let salvaged_disk = evaluator.cache.take_disk();
                                    evaluator = StackEvaluator::new();
                                    evaluator.cache.set_disk(salvaged_disk);
                                    Err(EvalError::Panicked(super::panic_payload_message(payload)))
                                }
                            };
                            publish_job_result(&result_tx, job.token, job.quality, result);
                        }
                    }
                }
            })
            .expect("spawn terra-eval-worker");

        Self {
            tx: job_tx,
            rx: result_rx,
            current_token,
            _handle: handle,
            busy: false,
            disconnected_reported: false,
        }
    }

    pub fn set_token(&self, token: u64) {
        self.current_token.store(token, Ordering::Release);
    }

    pub fn submit(&mut self, request: EvalWorkRequest) -> Result<(), EvalWorkerSubmitError> {
        self.set_token(request.token);
        self.tx
            .send(WorkerMsg::Job(Box::new(request)))
            .map_err(|_| EvalWorkerSubmitError)?;
        self.busy = true;
        Ok(())
    }

    /// Non-blocking poll for one worker event.
    ///
    /// Disconnection is emitted once so a UI polling loop cannot flood logs.
    pub fn try_recv_event(&mut self) -> Option<EvalWorkerEvent> {
        match self.rx.try_recv() {
            Ok(event) => {
                if event
                    .token()
                    .is_some_and(|token| token == self.current_token.load(Ordering::Acquire))
                {
                    self.busy = false;
                }
                Some(event)
            }
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) if !self.disconnected_reported => {
                self.busy = false;
                self.disconnected_reported = true;
                Some(EvalWorkerEvent::Disconnected)
            }
            Err(TryRecvError::Disconnected) => None,
        }
    }

    pub fn shutdown(&self) {
        let _ = self.tx.send(WorkerMsg::Shutdown);
    }

    /// Replace the worker thread and its evaluator/cache after an unexpected disconnect.
    pub fn restart(&mut self) {
        self.shutdown();
        *self = Self::spawn();
    }
}

fn publish_job_result(
    result_tx: &Sender<EvalWorkerEvent>,
    token: u64,
    quality: PreviewQuality,
    result: Result<EvalWorkResult, EvalError>,
) {
    let event = match result {
        Ok(result) => EvalWorkerEvent::Completed(result),
        Err(EvalError::Cancelled) => return,
        Err(error) => EvalWorkerEvent::Failed(EvalWorkFailure {
            token,
            quality,
            error,
        }),
    };
    let _ = result_tx.send(event);
}

impl Drop for EvalWorker {
    fn drop(&mut self) {
        let _ = self.tx.send(WorkerMsg::Shutdown);
    }
}

fn run_cpu_job(
    evaluator: &mut StackEvaluator,
    job: &EvalWorkRequest,
    token_flag: &Arc<AtomicU64>,
) -> Result<EvalWorkResult, EvalError> {
    let t0 = Instant::now();
    if job.token != token_flag.load(Ordering::Acquire) {
        return Err(EvalError::Cancelled);
    }

    let res = job.quality.resolution(job.preview_res, job.export_res);
    let metrics = HeightfieldMetrics {
        width: res,
        height: res,
        world_size_x: job.base_metrics.world_size_x,
        world_size_z: job.base_metrics.world_size_z,
        tile_size: job.base_metrics.tile_size.min(res),
        halo: job.base_metrics.halo,
    };

    if job.mark_all_dirty {
        evaluator.mark_all_dirty(&job.stack);
    } else if let Some(id) = job.dirty_from {
        evaluator.mark_dirty_from(&job.stack, id);
    }

    let mut ctx = EvalContext::new(metrics);
    ctx.set_cancellation_generation(Arc::clone(token_flag), job.token);
    ctx.quality = job.quality;
    ctx.level_steps = job.level_steps.clone();
    ctx.mask_assets = job.masks.clone();
    ctx.set_aux_hashmap(job.aux.clone());
    if let Some(strata) = &job.strata {
        ctx.aux_maps.strata = Some(strata.clone());
    }
    // Bake masks against the prior composed DEM, not zeros.
    let reference;
    let reference_ref: &Heightfield = match &job.mask_reference {
        Some(prev)
            if prev.metrics.width == metrics.width && prev.metrics.height == metrics.height =>
        {
            prev.as_ref()
        }
        Some(prev) => {
            // Resolution changed between qualities — resample nearest for mask bake.
            reference = resample_height_nearest(prev.as_ref(), metrics);
            &reference
        }
        None => {
            reference = Heightfield::zeros(metrics);
            &reference
        }
    };
    ctx.masks = bake_mask_assets(&job.masks, reference_ref, metrics, &ctx.aux);

    // Cooperative cancel between layers.
    let hf = {
        if job.token != token_flag.load(Ordering::Acquire) {
            return Err(EvalError::Cancelled);
        }
        evaluator.rebuild_incremental(&job.stack, &mut ctx)?
    };

    if job.token != token_flag.load(Ordering::Acquire) {
        return Err(EvalError::Cancelled);
    }

    ctx.sync_aux_hashmap();
    Ok(EvalWorkResult {
        token: job.token,
        quality: job.quality,
        height: hf,
        aux: ctx.aux,
        strata: ctx.aux_maps.strata.clone(),
        eval_us: t0.elapsed().as_micros() as u64,
        layer_timings: ctx.layer_timings,
    })
}

fn resample_height_nearest(src: &Heightfield, dst: HeightfieldMetrics) -> Heightfield {
    let mut out = Heightfield::zeros(dst);
    if src.metrics.width == 0 || src.metrics.height == 0 {
        return out;
    }
    for j in 0..dst.height {
        for i in 0..dst.width {
            let u = (i as f32 + 0.5) / dst.width as f32;
            let v = (j as f32 + 0.5) / dst.height as f32;
            let si = ((u * src.metrics.width as f32) as u32).min(src.metrics.width - 1);
            let sj = ((v * src.metrics.height as f32) as u32).min(src.metrics.height - 1);
            out.set(i, j, src.get(si, sj));
        }
    }
    out.refresh_halos();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::{BlendMode, EffectFilterParams, FlatParams, Layer, LayerKind};

    #[test]
    fn worker_produces_heightfield() {
        let mut worker = EvalWorker::spawn();
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "Flat",
            LayerKind::Flat(FlatParams { height: 12.0 }),
        ));
        let token = 1;
        worker
            .submit(EvalWorkRequest {
                token,
                quality: PreviewQuality::Draft,
                stack,
                masks: Vec::new(),
                base_metrics: HeightfieldMetrics::preview_default(),
                level_steps: crate::analyze::LevelStepSettings::default(),
                preview_res: 256,
                export_res: 1024,
                aux: HashMap::new(),
                strata: None,
                mask_reference: None,
                dirty_from: None,
                mark_all_dirty: true,
            })
            .expect("submit worker job");
        let mut result = None;
        for _ in 0..200 {
            if let Some(event) = worker.try_recv_event() {
                match event {
                    EvalWorkerEvent::Completed(r) if r.token == token => {
                        result = Some(r);
                        break;
                    }
                    EvalWorkerEvent::Failed(failure) => {
                        panic!("worker failed unexpectedly: {}", failure.error)
                    }
                    EvalWorkerEvent::Completed(_) | EvalWorkerEvent::Disconnected => {}
                }
            }
            thread::sleep(std::time::Duration::from_millis(5));
        }
        let r = result.expect("worker result");
        assert_eq!(r.token, token);
        assert!(r.height.metrics.width > 0);
    }

    #[test]
    fn worker_height_mask_uses_layer_input_not_previous_frame() {
        use crate::mask::{MaskAsset, MaskId, MaskRef, MaskSource};

        let metrics = HeightfieldMetrics::new(32, 32, 320.0, 320.0);
        // Deliberately stale prior DEM. Point-of-use evaluation must follow the
        // Base layer below the mask consumer instead of this previous frame.
        let reference = Heightfield::zeros(metrics);

        let mask_id = MaskId::new();
        let asset = MaskAsset {
            id: mask_id,
            name: "High".into(),
            source: MaskSource::Height {
                min: 50.0,
                max: 100.0,
            },
            ops: Vec::new(),
            paint: None,
            display_color: crate::mask::default_mask_display_color(),
        };

        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "Base",
            LayerKind::Flat(FlatParams { height: 100.0 }),
        ));
        let mut raise = Layer::new("Raise", LayerKind::Flat(FlatParams { height: 40.0 }));
        raise.common.blend = BlendMode::Add;
        raise.common.masks.push(MaskRef {
            id: mask_id,
            strength: 1.0,
            invert: false,
        });
        stack.push(raise);

        let mut worker = EvalWorker::spawn();
        let token = 2;
        worker
            .submit(EvalWorkRequest {
                token,
                quality: PreviewQuality::Full,
                stack,
                masks: vec![asset],
                base_metrics: metrics,
                level_steps: crate::analyze::LevelStepSettings::default(),
                preview_res: 32,
                export_res: 32,
                aux: HashMap::new(),
                strata: None,
                mask_reference: Some(std::sync::Arc::new(reference)),
                dirty_from: None,
                mark_all_dirty: true,
            })
            .expect("submit worker job");
        let mut result = None;
        for _ in 0..400 {
            if let Some(event) = worker.try_recv_event() {
                match event {
                    EvalWorkerEvent::Completed(r) if r.token == token => {
                        result = Some(r);
                        break;
                    }
                    EvalWorkerEvent::Failed(failure) => {
                        panic!("worker failed unexpectedly: {}", failure.error)
                    }
                    EvalWorkerEvent::Completed(_) | EvalWorkerEvent::Disconnected => {}
                }
            }
            thread::sleep(std::time::Duration::from_millis(5));
        }
        let r = result.expect("worker result");
        assert!((r.height.get(24, 16) - 140.0).abs() < 1.0e-4);
        assert!((r.height.get(8, 16) - 140.0).abs() < 1.0e-4);
    }

    #[test]
    fn draft_aux_mask_is_resampled_before_full_crater_composite() {
        use crate::mask::{MaskAsset, MaskId, MaskRef, MaskSource};

        let source_metrics = HeightfieldMetrics::new(512, 512, 1024.0, 1024.0);
        let mask_id = MaskId::new();
        let asset = MaskAsset::new(mask_id, "Wetness", MaskSource::Wetness);
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "Base",
            LayerKind::Flat(FlatParams { height: 100.0 }),
        ));
        let mut crater = Layer::new(
            "Crater",
            LayerKind::EffectFilter(EffectFilterParams::crater()),
        );
        crater.common.masks.push(MaskRef::new(mask_id));
        stack.push(crater);

        let mut worker = EvalWorker::spawn();
        worker
            .submit(EvalWorkRequest {
                token: 40,
                quality: PreviewQuality::Draft,
                stack: stack.clone(),
                masks: vec![asset.clone()],
                base_metrics: source_metrics,
                level_steps: crate::analyze::LevelStepSettings::default(),
                preview_res: 1024,
                export_res: 1024,
                aux: HashMap::from([("wetness".into(), MaskField::filled(source_metrics, 0.75))]),
                strata: None,
                mask_reference: None,
                dirty_from: None,
                mark_all_dirty: true,
            })
            .expect("submit Draft worker job");

        let draft = wait_for_result(&mut worker, 40);
        assert_eq!(draft.height.metrics.width, 512);

        worker
            .submit(EvalWorkRequest {
                token: 41,
                quality: PreviewQuality::Full,
                stack,
                masks: vec![asset],
                base_metrics: source_metrics,
                level_steps: crate::analyze::LevelStepSettings::default(),
                preview_res: 1024,
                export_res: 1024,
                aux: draft.aux,
                strata: draft.strata,
                mask_reference: Some(Arc::new(draft.height)),
                dirty_from: None,
                mark_all_dirty: false,
            })
            .expect("submit Full worker job");

        let full = wait_for_result(&mut worker, 41);
        assert_eq!(full.height.metrics.width, 1024);
        assert!(full.height.get(1023, 1023).is_finite());
        let wetness = full.aux.get("wetness").expect("wetness aux");
        assert_eq!(wetness.metrics.width, 1024);
        assert_eq!(wetness.metrics.height, 1024);
        assert_eq!(wetness.get(1023, 1023), 0.75);
    }

    fn wait_for_result(worker: &mut EvalWorker, token: u64) -> EvalWorkResult {
        for _ in 0..2_000 {
            if let Some(event) = worker.try_recv_event() {
                match event {
                    EvalWorkerEvent::Completed(result) if result.token == token => return result,
                    EvalWorkerEvent::Failed(failure) if failure.token == token => {
                        panic!("worker failed unexpectedly: {}", failure.error)
                    }
                    EvalWorkerEvent::Disconnected => panic!("worker disconnected unexpectedly"),
                    EvalWorkerEvent::Completed(_) | EvalWorkerEvent::Failed(_) => {}
                }
            }
            thread::sleep(std::time::Duration::from_millis(2));
        }
        panic!("worker did not complete token {token}")
    }

    #[test]
    fn persistent_worker_evaluator_reuses_clean_stack() {
        let metrics = HeightfieldMetrics::new(32, 32, 320.0, 320.0);
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "Base",
            LayerKind::Flat(FlatParams { height: 25.0 }),
        ));
        let token = 7;
        let live = Arc::new(AtomicU64::new(token));
        let mut evaluator = StackEvaluator::new();
        let mut request = EvalWorkRequest {
            token,
            quality: PreviewQuality::Full,
            stack,
            masks: Vec::new(),
            base_metrics: metrics,
            level_steps: crate::analyze::LevelStepSettings::default(),
            preview_res: 32,
            export_res: 32,
            aux: HashMap::new(),
            strata: None,
            mask_reference: None,
            dirty_from: None,
            mark_all_dirty: true,
        };
        let first = run_cpu_job(&mut evaluator, &request, &live).expect("first build");
        assert_eq!(
            first
                .layer_timings
                .iter()
                .filter(|timing| timing.status == super::super::LayerEvalStatus::Computed)
                .count(),
            1
        );

        request.mark_all_dirty = false;
        let second = run_cpu_job(&mut evaluator, &request, &live).expect("cached build");
        assert_eq!(
            second
                .layer_timings
                .iter()
                .filter(|timing| timing.status == super::super::LayerEvalStatus::CacheHit)
                .count(),
            1
        );
        assert_eq!(second.height.get(16, 16), 25.0);
    }

    #[test]
    fn evaluation_context_observes_superseding_generation() {
        let generation = Arc::new(AtomicU64::new(11));
        let mut ctx = EvalContext::new(HeightfieldMetrics::new(8, 8, 80.0, 80.0));
        ctx.set_cancellation_generation(Arc::clone(&generation), 11);
        assert!(ctx.check_cancelled().is_ok());
        generation.store(12, Ordering::Release);
        assert!(matches!(ctx.check_cancelled(), Err(EvalError::Cancelled)));
    }

    #[test]
    fn non_cancellation_error_becomes_a_contextual_failure_event() {
        let (tx, rx) = mpsc::channel();
        publish_job_result(
            &tx,
            17,
            PreviewQuality::Medium,
            Err(EvalError::Io("broken input".into())),
        );
        let event = rx.try_recv().expect("failure event");
        let EvalWorkerEvent::Failed(failure) = event else {
            panic!("expected failure event");
        };
        assert_eq!(failure.token, 17);
        assert_eq!(failure.quality, PreviewQuality::Medium);
        assert!(failure.error.to_string().contains("broken input"));
    }

    #[test]
    fn cancellation_does_not_become_a_failure_event() {
        let (tx, rx) = mpsc::channel();
        publish_job_result(&tx, 18, PreviewQuality::Draft, Err(EvalError::Cancelled));
        assert!(matches!(rx.try_recv(), Err(TryRecvError::Empty)));
    }

    #[test]
    fn layer_panic_becomes_failure_and_worker_accepts_later_job() {
        use crate::mask::{MaskAsset, MaskId, MaskRef, MaskSource};

        let metrics = HeightfieldMetrics::new(128, 128, 128.0, 128.0);
        let malformed: MaskField = serde_json::from_value(serde_json::json!({
            "metrics": metrics,
            "data": []
        }))
        .expect("deserialize malformed test field");
        let mask_id = MaskId::new();
        let asset = MaskAsset::new(mask_id, "Wetness", MaskSource::Wetness);
        let mut crater = Layer::new(
            "Crater",
            LayerKind::EffectFilter(EffectFilterParams::crater()),
        );
        crater.common.masks.push(MaskRef::new(mask_id));
        let mut crashing_stack = LayerStack::new();
        crashing_stack.push(Layer::new(
            "Base",
            LayerKind::Flat(FlatParams { height: 10.0 }),
        ));
        crashing_stack.push(crater);

        let mut worker = EvalWorker::spawn();
        worker
            .submit(EvalWorkRequest {
                token: 50,
                quality: PreviewQuality::Full,
                stack: crashing_stack,
                masks: vec![asset],
                base_metrics: metrics,
                level_steps: crate::analyze::LevelStepSettings::default(),
                preview_res: 128,
                export_res: 128,
                aux: HashMap::from([("wetness".into(), malformed)]),
                strata: None,
                mask_reference: None,
                dirty_from: None,
                mark_all_dirty: true,
            })
            .expect("submit panicking job");

        let mut failure = None;
        for _ in 0..1_000 {
            if let Some(event) = worker.try_recv_event() {
                match event {
                    EvalWorkerEvent::Failed(found) if found.token == 50 => {
                        failure = Some(found);
                        break;
                    }
                    EvalWorkerEvent::Disconnected => panic!("worker must contain layer panic"),
                    _ => {}
                }
            }
            thread::sleep(std::time::Duration::from_millis(2));
        }
        let failure = failure.expect("panic failure event");
        assert!(matches!(failure.error, EvalError::LayerPanicked { .. }));

        let mut valid_stack = LayerStack::new();
        valid_stack.push(Layer::new(
            "Recovered",
            LayerKind::Flat(FlatParams { height: 23.0 }),
        ));
        worker
            .submit(EvalWorkRequest {
                token: 51,
                quality: PreviewQuality::Full,
                stack: valid_stack,
                masks: Vec::new(),
                base_metrics: metrics,
                level_steps: crate::analyze::LevelStepSettings::default(),
                preview_res: 128,
                export_res: 128,
                aux: HashMap::new(),
                strata: None,
                mask_reference: None,
                dirty_from: None,
                mark_all_dirty: true,
            })
            .expect("worker should accept a later job");

        let recovered = wait_for_result(&mut worker, 51);
        assert_eq!(recovered.height.get(127, 127), 23.0);
    }

    /// B1-D8 revert check: a pinned bake written by one job survives the worker's
    /// evaluator restart (triggered by a process-level panic in a later job) via
    /// the declared disk handoff, so the restarted evaluator adopts it as a cache
    /// hit. Reverting the `take_disk` / `set_disk` salvage drops the spill and the
    /// base recomputes (`Computed`).
    #[test]
    fn worker_restart_salvages_a_pinned_bake() {
        let metrics = HeightfieldMetrics::new(128, 128, 128.0, 128.0);
        let mut base = Layer::new("Base", LayerKind::Flat(FlatParams { height: 10.0 }));
        base.common.cached = true;
        let base_id = base.id();
        let mut base_stack = LayerStack::new();
        base_stack.push(base);

        let mut worker = EvalWorker::spawn();

        // Job A: bake the pinned base — it spills to the worker evaluator's root.
        worker
            .submit(EvalWorkRequest {
                token: 80,
                quality: PreviewQuality::Full,
                stack: base_stack.clone(),
                masks: Vec::new(),
                base_metrics: metrics,
                level_steps: crate::analyze::LevelStepSettings::default(),
                preview_res: 128,
                export_res: 128,
                aux: HashMap::new(),
                strata: None,
                mask_reference: None,
                dirty_from: None,
                mark_all_dirty: true,
            })
            .expect("submit bake job");
        let baked = wait_for_result(&mut worker, 80);
        assert_eq!(baked.height.get(0, 0), 10.0);

        // Job B: a malformed aux (claims 64x64 but is empty) panics while resampling
        // aux into the 128x128 job — outside the per-layer catch — so the worker
        // restarts its evaluator. `mark_all_dirty:false` keeps the base bake on disk.
        let malformed: MaskField = serde_json::from_value(serde_json::json!({
            "metrics": HeightfieldMetrics::new(64, 64, 64.0, 64.0),
            "data": []
        }))
        .expect("deserialize malformed test field");
        let mut junk_stack = LayerStack::new();
        junk_stack.push(Layer::new(
            "Junk",
            LayerKind::Flat(FlatParams { height: 1.0 }),
        ));
        worker
            .submit(EvalWorkRequest {
                token: 81,
                quality: PreviewQuality::Full,
                stack: junk_stack,
                masks: Vec::new(),
                base_metrics: metrics,
                level_steps: crate::analyze::LevelStepSettings::default(),
                preview_res: 128,
                export_res: 128,
                aux: HashMap::from([("junk".into(), malformed)]),
                strata: None,
                mask_reference: None,
                dirty_from: None,
                mark_all_dirty: false,
            })
            .expect("submit restarting job");
        let mut restarted = false;
        for _ in 0..1_000 {
            if let Some(event) = worker.try_recv_event() {
                match event {
                    EvalWorkerEvent::Failed(found) if found.token == 81 => {
                        restarted = true;
                        break;
                    }
                    EvalWorkerEvent::Disconnected => panic!("worker restart must not disconnect"),
                    _ => {}
                }
            }
            thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(
            restarted,
            "the malformed aux should fail the job at the process level"
        );

        // Job C: the restarted evaluator adopts the salvaged base spill — CacheHit,
        // not a recompute.
        worker
            .submit(EvalWorkRequest {
                token: 82,
                quality: PreviewQuality::Full,
                stack: base_stack,
                masks: Vec::new(),
                base_metrics: metrics,
                level_steps: crate::analyze::LevelStepSettings::default(),
                preview_res: 128,
                export_res: 128,
                aux: HashMap::new(),
                strata: None,
                mask_reference: None,
                dirty_from: None,
                mark_all_dirty: false,
            })
            .expect("worker accepts the follow-up job");
        let reused = wait_for_result(&mut worker, 82);
        assert_eq!(reused.height.get(0, 0), 10.0);
        let base_timing = reused
            .layer_timings
            .iter()
            .find(|timing| timing.layer == base_id)
            .expect("base appears in timings");
        assert_eq!(
            base_timing.status,
            super::super::LayerEvalStatus::CacheHit,
            "the salvaged spill must be reused after the restart, not recomputed"
        );
    }

    #[test]
    fn disconnection_is_reported_only_once_and_rejects_new_work() {
        let mut worker = EvalWorker::spawn();
        worker.shutdown();
        let mut disconnected = false;
        for _ in 0..200 {
            if matches!(worker.try_recv_event(), Some(EvalWorkerEvent::Disconnected)) {
                disconnected = true;
                break;
            }
            thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(disconnected, "worker should report result-channel closure");
        assert!(worker.try_recv_event().is_none());

        let request = EvalWorkRequest {
            token: 19,
            quality: PreviewQuality::Draft,
            stack: LayerStack::new(),
            masks: Vec::new(),
            base_metrics: HeightfieldMetrics::preview_default(),
            level_steps: crate::analyze::LevelStepSettings::default(),
            preview_res: 256,
            export_res: 1024,
            aux: HashMap::new(),
            strata: None,
            mask_reference: None,
            dirty_from: None,
            mark_all_dirty: true,
        };
        assert_eq!(worker.submit(request), Err(EvalWorkerSubmitError));
        assert!(!worker.busy);

        worker.restart();
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "Recovered",
            LayerKind::Flat(FlatParams { height: 9.0 }),
        ));
        worker
            .submit(EvalWorkRequest {
                token: 20,
                quality: PreviewQuality::Draft,
                stack,
                masks: Vec::new(),
                base_metrics: HeightfieldMetrics::preview_default(),
                level_steps: crate::analyze::LevelStepSettings::default(),
                preview_res: 128,
                export_res: 128,
                aux: HashMap::new(),
                strata: None,
                mask_reference: None,
                dirty_from: None,
                mark_all_dirty: true,
            })
            .expect("restarted worker accepts work");
        let recovered = wait_for_result(&mut worker, 20);
        assert_eq!(recovered.height.get(0, 0), 9.0);
    }
}
