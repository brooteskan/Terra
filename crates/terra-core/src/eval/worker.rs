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
use std::sync::Arc;
use std::time::Instant;
use terra_jobs::{CancelToken, JobError, JobEvent, LatestWins};

// The worker thread, its channels, stale-skip, and sleep-based polling now live
// in `terra-jobs`; only the tests still touch the raw mpsc / thread primitives.
#[cfg(test)]
use std::sync::mpsc::{self, Sender, TryRecvError};
#[cfg(test)]
use std::thread;

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
    /// Spatial scope of a suffix rebuild, in normalized UV. Consulted only
    /// alongside `dirty_from`: `Some(rect)` maps to a tile set at the job's own
    /// resolution and drives a tile-scoped `mark_dirty_from_region` (#100 phase
    /// 4); `None` keeps whole-field per-layer marks, exactly as before. UV rather
    /// than texels/tiles because the worker recomputes its resolution independently.
    pub dirty_region: Option<crate::tiling::UvRect>,
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

/// The value a CPU eval job produces: the request's quality paired with the
/// evaluation outcome. Domain failures (a cancelled generation, a layer panic
/// surfaced as `EvalError`) live *inside* this `T`; only a process-level panic
/// in the body escapes as a `terra_jobs` [`JobError`].
type JobOutput = (PreviewQuality, Result<EvalWorkResult, EvalError>);

/// Owns a dedicated thread with its own [`StackEvaluator`] and layer cache.
///
/// A thin wrapper over a [`LatestWins`] executor: the thread loop, channel
/// plumbing, dequeue-time stale-skip, per-job panic containment, and
/// disconnect-once bookkeeping all live in `terra-jobs`. This layer keeps the
/// eval-specific surface — the generation token the app owns, the best-effort
/// `busy` flag, and translating a job outcome into an [`EvalWorkerEvent`]
/// (dropping cancelled results, which publish nothing).
pub struct EvalWorker {
    inner: LatestWins<EvalWorkRequest, JobOutput>,
    /// Shared cancel / generation id — the executor skips jobs with older tokens.
    /// The same `Arc` the executor reads, so `set_token` supersedes in-flight work.
    pub current_token: Arc<AtomicU64>,
    /// True while a job may still be running (best-effort).
    pub busy: bool,
}

impl EvalWorker {
    pub fn spawn() -> Self {
        let current_token = Arc::new(AtomicU64::new(0));
        // The job body reads the live generation directly through this handle, as
        // it did before the executor existed, so `run_cpu_job` keeps its exact
        // signature and its cooperative between-layer cancel checks.
        let job_token = Arc::clone(&current_token);
        let inner = LatestWins::spawn(
            "terra-eval-worker",
            Arc::clone(&current_token),
            StackEvaluator::new,
            move |evaluator: &mut StackEvaluator, job: &EvalWorkRequest, _cancel: &CancelToken| {
                Ok((job.quality, run_cpu_job(evaluator, job, &job_token)))
            },
            |mut evaluator: StackEvaluator| {
                // A job-level panic may leave cache state partially updated. Start
                // subsequent requests from a fresh evaluator, but carry the disk
                // spill store (and its private root) across so pinned bakes written
                // before the panic survive the restart as a declared handoff
                // (B1-D8) rather than filesystem coincidence.
                let salvaged_disk = evaluator.cache.take_disk();
                let mut fresh = StackEvaluator::new();
                fresh.cache.set_disk(salvaged_disk);
                fresh
            },
        );

        Self {
            inner,
            current_token,
            busy: false,
        }
    }

    pub fn set_token(&self, token: u64) {
        self.current_token.store(token, Ordering::Release);
    }

    pub fn submit(&mut self, request: EvalWorkRequest) -> Result<(), EvalWorkerSubmitError> {
        let token = request.token;
        self.inner
            .submit(request, token)
            .map_err(|_| EvalWorkerSubmitError)?;
        self.busy = true;
        Ok(())
    }

    /// Non-blocking poll for one worker event.
    ///
    /// A cancelled job publishes nothing — that policy lives here, keeping the
    /// executor generic — so this loops past any suppressed result to return the
    /// next real event rather than stalling the caller's drain loop. Disconnection
    /// is emitted once so a UI polling loop cannot flood logs.
    pub fn try_recv_event(&mut self) -> Option<EvalWorkerEvent> {
        loop {
            let event = match self.inner.try_recv()? {
                JobEvent::Completed {
                    token,
                    value: (quality, result),
                } => match job_result_event(token, quality, result) {
                    Some(event) => event,
                    None => continue, // cancelled result: suppressed
                },
                JobEvent::Failed {
                    token,
                    request,
                    error,
                } => {
                    // The job body always returns `Ok`, so a `JobError` here is a
                    // process-level panic. Map it to `EvalError::Panicked` with the
                    // failed request's quality, exactly as the old panic path did.
                    let eval_error = match error {
                        JobError::Panicked(message) => EvalError::Panicked(message),
                        JobError::Cancelled => continue,
                    };
                    match job_result_event(token, request.quality, Err(eval_error)) {
                        Some(event) => event,
                        None => continue,
                    }
                }
                JobEvent::Disconnected => EvalWorkerEvent::Disconnected,
            };

            // Clear `busy` on the live generation's events, and always on
            // disconnect. Suppressed results never reach here, so they leave
            // `busy` untouched — as when they were filtered before the channel.
            if matches!(event, EvalWorkerEvent::Disconnected)
                || event
                    .token()
                    .is_some_and(|token| token == self.current_token.load(Ordering::Acquire))
            {
                self.busy = false;
            }
            return Some(event);
        }
    }

    pub fn shutdown(&self) {
        self.inner.shutdown();
    }

    /// Replace the worker thread and its evaluator/cache after an unexpected disconnect.
    pub fn restart(&mut self) {
        self.shutdown();
        *self = Self::spawn();
    }
}

/// Translate a job outcome into a worker event, or `None` when it publishes
/// nothing. A cancelled evaluation is dropped (the app superseded it); every
/// other error becomes a contextual [`EvalWorkFailure`].
fn job_result_event(
    token: u64,
    quality: PreviewQuality,
    result: Result<EvalWorkResult, EvalError>,
) -> Option<EvalWorkerEvent> {
    match result {
        Ok(result) => Some(EvalWorkerEvent::Completed(result)),
        Err(EvalError::Cancelled) => None,
        Err(error) => Some(EvalWorkerEvent::Failed(EvalWorkFailure {
            token,
            quality,
            error,
        })),
    }
}

#[cfg(test)]
fn publish_job_result(
    result_tx: &Sender<EvalWorkerEvent>,
    token: u64,
    quality: PreviewQuality,
    result: Result<EvalWorkResult, EvalError>,
) {
    if let Some(event) = job_result_event(token, quality, result) {
        let _ = result_tx.send(event);
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
    let metrics = job.base_metrics.at_resolution(res)?;

    if job.mark_all_dirty {
        evaluator.mark_all_dirty(&job.stack);
    } else if let Some(id) = job.dirty_from {
        // Soundness invariant: cache seed marks persist until a job actually
        // stores clean tiles (only `insert`/`insert_baked` clear them), so a
        // cancelled or superseded job after this point can never lose dirt — the
        // next job's marks union into the seeds these leave behind.
        match job.dirty_region {
            Some(region) => {
                let tiles = crate::tiling::tiles_for_uv_rect(&metrics, region);
                evaluator.mark_dirty_from_region(&job.stack, id, &tiles);
            }
            None => evaluator.mark_dirty_from(&job.stack, id),
        }
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
                dirty_region: None,
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
                dirty_region: None,
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
                dirty_region: None,
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
                dirty_region: None,
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
            dirty_region: None,
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

    /// #100 phase 4: a request's `dirty_region` (normalized UV) is mapped to tiles
    /// at the job's resolution and drives a tile-scoped `mark_dirty_from_region`, so
    /// a bounded edit recomputes only the reached tiles yet stays bit-identical to a
    /// whole-field rebuild of the edited stack. (Reach *expansion* across a coupled
    /// downstream pass is covered exhaustively at the evaluator level in
    /// `tile_scoped_eval_equivalence`; this asserts the worker request wiring on a
    /// per-texel suffix where scoped and whole-field are bit-exact.)
    #[test]
    fn scoped_dirty_region_request_matches_whole_field_and_recomputes_bounded_tiles() {
        use crate::layer::{CoastalParams, PlateauParams, SculptParams};
        use crate::tiling::UvRect;

        // 128^2 over 32-sample tiles = a 4x4 grid. A SculptBase with a varying paint
        // buffer (so tiles genuinely differ) feeding two per-texel passes.
        let res = 128u32;
        let ts = 32u32;
        let base_metrics = HeightfieldMetrics {
            width: res,
            height: res,
            world_size_x: res as f32,
            world_size_z: res as f32,
            tile_size: ts,
            halo: 2,
        };

        let mut sculpt = SculptParams::filled(res, 0.0);
        for j in 0..res {
            for i in 0..res {
                sculpt.samples[(j * res + i) as usize] = 30.0 + i as f32 * 0.1 + j as f32 * 0.07;
            }
        }
        let base = Layer::new("Sculpt", LayerKind::SculptBase(sculpt));
        let base_id = base.id();
        let plateau = Layer::new(
            "Plateau",
            LayerKind::Plateau(PlateauParams {
                low: 25.0,
                high: 45.0,
                soft: 6.0,
            }),
        );
        let plateau_id = plateau.id();
        let coastal = Layer::new(
            "Coastal",
            LayerKind::Coastal(CoastalParams {
                sea_level: 30.0,
                beach_width: 8.0,
                flatten_below: true,
                shelf_depth: 4.0,
            }),
        );
        let coastal_id = coastal.id();
        let mut stack = LayerStack::new();
        stack.push(base);
        stack.push(plateau);
        stack.push(coastal);

        let make = |token: u64,
                    stack: LayerStack,
                    dirty_from: Option<LayerId>,
                    dirty_region: Option<UvRect>,
                    mark_all_dirty: bool| EvalWorkRequest {
            token,
            quality: PreviewQuality::Full,
            stack,
            masks: Vec::new(),
            base_metrics,
            level_steps: crate::analyze::LevelStepSettings::default(),
            preview_res: res,
            export_res: res,
            aux: HashMap::new(),
            strata: None,
            mask_reference: None,
            dirty_from,
            dirty_region,
            mark_all_dirty,
        };

        // Full build populates the persistent worker cache.
        let live = Arc::new(AtomicU64::new(1));
        let mut evaluator = StackEvaluator::new();
        run_cpu_job(
            &mut evaluator,
            &make(1, stack.clone(), None, None, true),
            &live,
        )
        .expect("full build");

        // Edit paint samples [40,56) x [40,56): well inside tile (1,1)'s interior
        // and clear of its 32/64 boundaries, so the base's bilinear paint footprint
        // stays within the one tile (a boundary-hugging edit would spill a sample
        // into the neighbour and expose a carry, but that is a too-tight-region
        // authoring bug, not what this test isolates).
        let mut edited_stack = stack.clone();
        if let Some(layer) = edited_stack.find_mut(base_id) {
            if let LayerKind::SculptBase(p) = &mut layer.kind {
                for j in 40..56u32 {
                    for i in 40..56u32 {
                        p.samples[(j * res + i) as usize] += 25.0;
                    }
                }
            }
        }

        // Scoped incremental over the edited tile: the UV rect for tile (1,1) spans
        // u,v in [0.25, 0.5]. The token must match the live generation.
        live.store(2, Ordering::Release);
        let region = UvRect::from_center_radius(0.375, 0.375, 0.125);
        let scoped = run_cpu_job(
            &mut evaluator,
            &make(2, edited_stack.clone(), Some(base_id), Some(region), false),
            &live,
        )
        .expect("scoped incremental");

        // Whole-field control: a cold evaluator rebuilds the edited stack.
        let control_live = Arc::new(AtomicU64::new(9));
        let mut control_eval = StackEvaluator::new();
        let control = run_cpu_job(
            &mut control_eval,
            &make(9, edited_stack, None, None, true),
            &control_live,
        )
        .expect("control build");

        let scoped_bits: Vec<u32> = scoped
            .height
            .to_dense()
            .iter()
            .map(|f| f.to_bits())
            .collect();
        let control_bits: Vec<u32> = control
            .height
            .to_dense()
            .iter()
            .map(|f| f.to_bits())
            .collect();
        assert_eq!(
            scoped_bits, control_bits,
            "a scoped dirty_region rebuild diverged from whole-field"
        );

        // Every per-texel layer recomputed exactly the one edited tile — the UV
        // scope reached mark_dirty_from_region rather than escalating to whole-field.
        let recomputed = |id: LayerId| -> Option<u32> {
            scoped
                .layer_timings
                .iter()
                .find(|t| t.layer == id)
                .and_then(|t| t.tiles_recomputed)
        };
        assert_eq!(
            recomputed(base_id),
            Some(1),
            "base recomputes the one edit tile"
        );
        assert_eq!(recomputed(plateau_id), Some(1), "plateau stays tile-scoped");
        assert_eq!(recomputed(coastal_id), Some(1), "coastal stays tile-scoped");
    }

    /// Perf (#100 phase 4): on the #98 Voronoi + Flatten stack over a sculpt base,
    /// a scoped-Full resubmit — the ladder-clobber scenario the app's
    /// straight-to-Full policy enables — recomputes only a few tiles and is far
    /// cheaper than the whole-field Full resubmit it replaces. This is the worker
    /// path (`run_cpu_job` + `dirty_region`) the acceptance measures. Ignored (a
    /// timing measurement); run with `--ignored --nocapture` to see the numbers.
    #[test]
    #[ignore = "perf measurement; run with `--ignored --nocapture` to see the numbers"]
    fn perf_scoped_full_resubmit_is_well_below_whole_field() {
        use crate::authoring::SculptStrokeKind;
        use crate::layer::{SculptParams, VoronoiParams};
        use crate::shape_history::{create_shape_layer, stamp_stroke};
        use crate::tiling::UvRect;
        use std::time::Instant;

        // 512^2 over 128-sample tiles = a 4x4 grid, Full quality.
        let res = 512u32;
        let ts = 128u32;
        let base_metrics = HeightfieldMetrics {
            width: res,
            height: res,
            world_size_x: res as f32,
            world_size_z: res as f32,
            tile_size: ts,
            halo: 2,
        };

        let mut sculpt = SculptParams::filled(res, 0.0);
        for j in 0..res {
            for i in 0..res {
                sculpt.samples[(j * res + i) as usize] = 30.0 + i as f32 * 0.05 + j as f32 * 0.03;
            }
        }
        let base = Layer::new("Sculpt", LayerKind::SculptBase(sculpt));
        let base_id = base.id();
        let voronoi = Layer::new(
            "Voronoi",
            LayerKind::VoronoiRegions(VoronoiParams::default()),
        );
        let mut flatten = create_shape_layer("Flatten");
        if let LayerKind::SculptStrokes(p) = &mut flatten.kind {
            stamp_stroke(
                p,
                SculptStrokeKind::Flatten,
                0.5,
                0.5,
                40.0,
                4.0,
                0.0,
                false,
            );
        }
        let mut stack = LayerStack::new();
        stack.push(base);
        stack.push(voronoi);
        stack.push(flatten);

        let make = |token: u64,
                    dirty_from: Option<LayerId>,
                    dirty_region: Option<UvRect>,
                    mark_all_dirty: bool| EvalWorkRequest {
            token,
            quality: PreviewQuality::Full,
            stack: stack.clone(),
            masks: Vec::new(),
            base_metrics,
            level_steps: crate::analyze::LevelStepSettings::default(),
            preview_res: res,
            export_res: res,
            aux: HashMap::new(),
            strata: None,
            mask_reference: None,
            dirty_from,
            dirty_region,
            mark_all_dirty,
        };

        let live = Arc::new(AtomicU64::new(1));
        let mut evaluator = StackEvaluator::new();
        run_cpu_job(&mut evaluator, &make(1, None, None, true), &live).expect("build");

        // Baseline: a whole-field-suffix Full resubmit (dirty_region None) — the
        // Full rung that rebuilds the whole field despite a one-tile edit today.
        live.store(2, Ordering::Release);
        let t0 = Instant::now();
        run_cpu_job(&mut evaluator, &make(2, Some(base_id), None, false), &live).expect("whole");
        let whole_us = t0.elapsed().as_micros();

        // Scoped: the same Full resubmit carrying a one-tile UV scope.
        live.store(3, Ordering::Release);
        let region = UvRect::from_center_radius(0.3, 0.3, 0.02);
        let t1 = Instant::now();
        let scoped = run_cpu_job(
            &mut evaluator,
            &make(3, Some(base_id), Some(region), false),
            &live,
        )
        .expect("scoped");
        let scoped_us = t1.elapsed().as_micros();

        let recomputed: u32 = scoped
            .layer_timings
            .iter()
            .filter_map(|t| t.tiles_recomputed)
            .sum();
        println!(
            "Voronoi+Flatten Full resubmit @ {res}^2 ({} tiles): whole-field {whole_us} us, \
             scoped {scoped_us} us ({:.1}x, {recomputed} tile-recomputes)",
            base_metrics.tile_count(),
            whole_us as f64 / scoped_us.max(1) as f64
        );
        assert!(
            scoped_us.saturating_mul(2) < whole_us,
            "scoped Full ({scoped_us} us) should be well below whole-field ({whole_us} us)"
        );
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
                dirty_region: None,
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
                dirty_region: None,
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
                dirty_region: None,
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
                dirty_region: None,
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
                dirty_region: None,
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
            dirty_region: None,
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
                dirty_region: None,
                mark_all_dirty: true,
            })
            .expect("restarted worker accepts work");
        let recovered = wait_for_result(&mut worker, 20);
        assert_eq!(recovered.height.get(0, 0), 9.0);
    }
}
