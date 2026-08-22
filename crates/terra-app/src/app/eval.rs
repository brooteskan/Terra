use std::time::{Duration, Instant};

use crate::logging::OperationContext;
use crate::ui::{Preview2dMode, TerrainPreviewFreshness};
use terra_core::layer::{LayerId, LayerKind};
use terra_core::mask::bake_mask_assets;
use terra_core::quality::PreviewQuality;
use terra_core::tiling::UvRect;
use terra_cpu_eval::EvalWorkRequest;
use terra_gpu::{GpuError, GpuPyramidContentIdentity};
use terra_gpu_eval::{GpuEvaluationIntent, GpuPreviewFreshness, GpuRefinementStep};

use super::frame_trace::{EvaluationTraceId, FrameTraceEventKind};
use super::logical_frame::{EditGeneration, FrameDeadlineKind, FrameIdentity, FrameRequestReason};
use super::refinement_job::{RefinementJob, RefinementPublicationState};
use super::{quality_stage_progress, DeferredFullField, TerraApp};

/// Interactive Full preview ceiling — same 1 m footing as WC (world metres ≈ samples),
/// capped at export-class 8192² so extreme worlds stay bounded.
const INTERACTIVE_PREVIEW_CAP: u32 = 8192;

/// Convert a resolution-independent edit footprint only after the evaluation
/// quality has selected its actual texture dimensions.
fn uv_to_texel_rect(region: UvRect, width: u32, height: u32) -> (u32, u32, u32, u32) {
    if width == 0 || height == 0 {
        return (0, 0, 0, 0);
    }
    if !(region.min_u.is_finite()
        && region.min_v.is_finite()
        && region.max_u.is_finite()
        && region.max_v.is_finite())
    {
        return (0, 0, width, height);
    }
    let x0 = ((region.min_u.clamp(0.0, 1.0) * width as f32).floor() as u32).min(width - 1);
    let x1 = ((region.max_u.clamp(0.0, 1.0) * width as f32).ceil() as u32).clamp(x0 + 1, width);
    let y0 = ((region.min_v.clamp(0.0, 1.0) * height as f32).floor() as u32).min(height - 1);
    let y1 = ((region.max_v.clamp(0.0, 1.0) * height as f32).ceil() as u32).clamp(y0 + 1, height);
    (x0, y0, x1 - x0, y1 - y0)
}

/// Correlate GPU work with the logical frame that requested it while keeping
/// publication authority tied to the captured evaluation token. A logical frame
/// may predate a project reset and therefore carry an older edit generation.
fn gpu_evaluation_trace_context(
    logical_frame: Option<FrameIdentity>,
    publication_generation: u64,
    evaluation: EvaluationTraceId,
) -> terra_gpu_eval::GpuEvaluationTraceContext {
    terra_gpu_eval::GpuEvaluationTraceContext {
        frame_id: logical_frame.unwrap_or_default().id.get(),
        generation: publication_generation,
        evaluation_id: evaluation.get(),
    }
}

fn terrain_tile_work_budget(
    state: terra_core::EditorRefinementState,
    max_pages: usize,
) -> terra_core::TerrainTileWorkBudget {
    let budget_us = state.terrain_budget_us();
    let max_items = if budget_us == u64::MAX {
        128
    } else {
        (budget_us / 250).clamp(1, 32) as usize
    };
    terra_core::TerrainTileWorkBudget::new(budget_us, max_items, max_pages.max(1))
}

impl TerraApp {
    pub(crate) fn note_refinement_activity(&mut self) {
        let now = Instant::now();
        self.last_refine = now;
        self.logical_frames.schedule_deadline(
            FrameDeadlineKind::OptionalRefinement,
            EditGeneration::new(self.eval_token),
            now + Duration::from_millis(super::REFINE_INTERVAL_MS as u64),
        );
    }

    pub(crate) fn request_rebuild(&mut self) {
        self.eval_token = self.scheduler.request_rebuild();
        self.eval_worker.set_token(self.eval_token);
        self.supersede_gpu_refinement();
        self.worker_refine_pending = false;
        self.deferred_full_field = None;
        self.logical_frames
            .clear_deadline(FrameDeadlineKind::FullFieldRefinement);
        self.ui_state.terrain_preview_freshness = if self.pending_gpu_dirty_region.is_some() {
            TerrainPreviewFreshness::LastCompleteStale
        } else {
            TerrainPreviewFreshness::Current
        };
        self.ui_state.refining = true;
        self.ui_state.quality = PreviewQuality::Draft;
        self.ui_state.build_progress = Some(0.0);
        let preview_stack = self.session.document.preview_eval_stack();
        self.ui_state.refining_layer_name = self
            .session
            .document
            .selected
            .and_then(|id| preview_stack.find(id))
            .map(|layer| layer.common.name.clone())
            .or_else(|| Some("terrain".into()));
        self.scheduler.quality = PreviewQuality::Draft;
        let now = Instant::now();
        self.last_edit = now;
        self.pending_eval = true;
        self.pending_eval_immediate = false;
        self.force_draft = true;
        self.ui_state.profile.gen_id = self.eval_token;
        let generation = EditGeneration::new(self.eval_token);
        self.logical_frames.discard_stale_deadlines(generation);
        self.logical_frames.schedule_deadline(
            FrameDeadlineKind::InteractiveEvaluation,
            generation,
            now + Duration::from_millis(
                self.session.rebuild_feedback.prefs.edit_debounce_ms.max(1),
            ),
        );
        self.logical_frames.schedule_deadline(
            FrameDeadlineKind::OptionalRefinement,
            generation,
            now + Duration::from_millis(super::POST_INPUT_REFINE_GRACE_MS),
        );
        self.request_app_frame(FrameRequestReason::RequiredEvaluation);
    }

    /// Interactive structural edit (add filter/layer): keep current viewport resolution
    /// and present supported work immediately through the GPU. Unsupported CPU work is
    /// submitted to the existing eval worker, leaving last-good content on screen until
    /// the worker publishes its result.
    pub(crate) fn request_rebuild_immediate(&mut self) {
        // Cancel any in-flight CPU job so a late Full result cannot pop the viewport.
        self.eval_token = self.eval_token.wrapping_add(1);
        self.scheduler.current_token = self.eval_token;
        self.eval_worker.set_token(self.eval_token);
        self.supersede_gpu_refinement();
        self.worker_refine_pending = false;
        self.deferred_full_field = None;
        self.logical_frames
            .clear_deadline(FrameDeadlineKind::FullFieldRefinement);
        self.ui_state.terrain_preview_freshness = TerrainPreviewFreshness::Current;
        // Structural edits may use a complete GPU intent at the currently visible
        // resolution, but an unsupported CPU suffix must still enter through the
        // non-blocking Draft worker path.
        self.force_draft = true;
        self.pending_eval = true;
        self.pending_eval_immediate = true;
        self.ui_state.refining = false;
        self.ui_state.build_progress = None;
        self.ui_state.profile.gen_id = self.eval_token;
        self.last_edit = Instant::now();

        // Match the resolution already on screen (renderer tex or last_height).
        if let Some(r) = self.renderer.as_ref() {
            let w = r.heights.tex_size.0.max(1);
            let preview = self
                .session
                .document
                .preview_resolution
                .min(INTERACTIVE_PREVIEW_CAP);
            let medium = PreviewQuality::Medium.resolution(preview, preview);
            self.scheduler.quality = if w >= preview {
                PreviewQuality::Full
            } else if w >= medium {
                PreviewQuality::Medium
            } else {
                PreviewQuality::Draft
            };
            self.ui_state.quality = self.scheduler.quality;
        }

        let now = Instant::now();
        let generation = EditGeneration::new(self.eval_token);
        self.logical_frames.discard_stale_deadlines(generation);
        self.logical_frames.schedule_deadline(
            FrameDeadlineKind::InteractiveEvaluation,
            generation,
            now,
        );
        self.logical_frames.schedule_deadline(
            FrameDeadlineKind::OptionalRefinement,
            generation,
            now + Duration::from_millis(super::POST_INPUT_REFINE_GRACE_MS),
        );
        self.request_app_frame(FrameRequestReason::RequiredEvaluation);
    }

    /// Submit the current Medium/Full snapshot to the CPU worker without blocking the UI.
    pub(crate) fn enqueue_refine_job(&mut self) {
        self.enqueue_async_eval(self.scheduler.quality);
    }

    /// Create an isolated GPU job for optional Medium/Full work. Required
    /// interactive evaluation continues to use `run_eval_step_with_intent`.
    pub(crate) fn begin_gpu_refinement(&mut self, quality: PreviewQuality) -> bool {
        if self.refinement_job.is_some()
            || !matches!(quality, PreviewQuality::Medium | PreviewQuality::Full)
        {
            return false;
        }
        let Some(gpu) = self.gpu.as_ref() else {
            return false;
        };
        let preview_stack = self.session.document.preview_eval_stack();
        let plan = match self
            .terrain_plan_cache
            .acquire(&preview_stack, &self.session.document.masks)
        {
            Ok(plan) => plan.clone(),
            Err(_) => return false,
        };
        let invalidation = self.pending_plan_invalidation.clone().unwrap_or_default();
        let revision = self.terrain_plan_cache.structure_revision();
        let preview = self
            .session
            .document
            .preview_resolution
            .min(INTERACTIVE_PREVIEW_CAP);
        let resolution = quality.resolution(preview, self.session.document.export_resolution);
        let Ok(metrics) = self.session.document.metrics.at_resolution(resolution) else {
            return false;
        };

        // A full-field present may currently share the engine's ping texture.
        // Snapshot it into the renderer's local double buffer before resumable
        // encoding mutates engine scratch across logical frames.
        if let (Some(engine), Some(renderer)) = (self.gpu_engine.as_ref(), self.renderer.as_mut()) {
            let (width, height) = renderer.heights.tex_size;
            if width > 0 && height > 0 {
                let current_metrics = self
                    .session
                    .document
                    .metrics
                    .at_resolution(width)
                    .unwrap_or(self.session.document.metrics);
                renderer.present_gpu_height_region(
                    engine.output_texture(),
                    terra_render::HeightPresentGeom {
                        width,
                        height,
                        world_size: renderer.heights.world_size,
                        height_range: renderer.heights.height_range,
                        dx: current_metrics.dx(),
                        dz: current_metrics.dz(),
                    },
                    None,
                );
            }
        }

        let origin = self.logical_frames.active_identity().unwrap_or_default();
        let evaluation = self.frame_trace.next_evaluation_id();
        let Some(engine) = self.gpu_engine.as_mut() else {
            return false;
        };
        engine.set_evaluation_trace_context(gpu_evaluation_trace_context(
            Some(origin),
            self.eval_token,
            evaluation,
        ));
        let engine_job = match engine.begin_compiled_refinement(
            &gpu.device,
            &preview_stack,
            &self.session.document.masks,
            &plan,
            revision,
            &invalidation,
            metrics,
            quality,
        ) {
            Ok(job) => job,
            Err(error) => {
                log::debug!(
                    target: "terra_app::evaluation",
                    "resumable GPU refinement unavailable: {error}"
                );
                return false;
            }
        };
        self.next_refinement_job_id = self.next_refinement_job_id.wrapping_add(1).max(1);
        let job = RefinementJob {
            id: self.next_refinement_job_id,
            origin,
            generation: super::logical_frame::EditGeneration::new(self.eval_token),
            target_quality: quality,
            evaluation,
            engine: engine_job,
            publication: RefinementPublicationState::Preparing,
            started_at: Instant::now(),
        };
        let progress = job.engine.progress();
        let submission_depth = self
            .gpu_engine
            .as_ref()
            .map_or(usize::from(progress.submissions_in_flight), |engine| {
                engine.refinement_submissions_in_flight(&job.engine)
            })
            .min(u8::MAX as usize) as u8;
        self.frame_trace.record_refinement(
            Instant::now(),
            FrameTraceEventKind::RefinementJobCreated,
            Some(origin),
            self.logical_frames.active_phase(),
            evaluation,
            quality,
            job.id,
            progress.completed_units,
            progress.total_units,
            submission_depth,
            None,
        );
        self.refinement_job = Some(job);
        self.ui_state.refining = true;
        self.ui_state.quality = quality;
        self.ui_state.build_progress = Some(super::quality_in_flight_progress(quality, 0.0));
        true
    }

    /// Advance at most one safe unit. The engine itself enforces a depth-one
    /// refinement queue; this method supplies generation and publication gates.
    pub(crate) fn advance_gpu_refinement(&mut self) -> bool {
        let Some(mut job) = self.refinement_job.take() else {
            return false;
        };
        if !job.is_fresh(self.eval_token) || self.input.has_pending() || self.pending_eval {
            let progress = job.engine.progress();
            if let Some(engine) = self.gpu_engine.as_mut() {
                engine.abandon_compiled_refinement(job.engine);
            }
            self.frame_trace.record_refinement(
                Instant::now(),
                FrameTraceEventKind::RefinementSuperseded,
                Some(job.origin),
                self.logical_frames.active_phase(),
                job.evaluation,
                job.target_quality,
                job.id,
                progress.completed_units,
                progress.total_units,
                progress.submissions_in_flight,
                Some(job.started_at.elapsed()),
            );
            return false;
        }
        let Some(gpu) = self.gpu.as_ref() else {
            return false;
        };
        let progress_before = job.engine.progress();
        let unit_started = Instant::now();
        let step = match self.gpu_engine.as_mut().map(|engine| {
            engine.advance_compiled_refinement(&gpu.device, &gpu.queue, &mut job.engine)
        }) {
            Some(Ok(step)) => step,
            Some(Err(error)) => {
                let progress = job.engine.progress();
                if let Some(engine) = self.gpu_engine.as_mut() {
                    engine.abandon_compiled_refinement(job.engine);
                }
                self.frame_trace.record_refinement(
                    Instant::now(),
                    FrameTraceEventKind::RefinementFailed,
                    Some(job.origin),
                    self.logical_frames.active_phase(),
                    job.evaluation,
                    job.target_quality,
                    job.id,
                    progress.completed_units,
                    progress.total_units,
                    progress.submissions_in_flight,
                    Some(job.started_at.elapsed()),
                );
                log::warn!(target: "terra_app::evaluation", "GPU refinement failed: {error}");
                if !self.worker_refine_pending {
                    self.enqueue_async_eval(job.target_quality);
                }
                return false;
            }
            None => return false,
        };
        let progress = job.engine.progress();
        let submission_depth = self
            .gpu_engine
            .as_ref()
            .map_or(usize::from(progress.submissions_in_flight), |engine| {
                engine.refinement_submissions_in_flight(&job.engine)
            })
            .min(u8::MAX as usize) as u8;
        let kind = match step {
            GpuRefinementStep::Submitted { .. } => {
                job.publication = RefinementPublicationState::SubmissionInFlight;
                FrameTraceEventKind::RefinementSubmissionQueued
            }
            GpuRefinementStep::ReadyToPublish => {
                job.publication = RefinementPublicationState::ReadyToPublish;
                FrameTraceEventKind::RefinementCandidateCompleted
            }
            GpuRefinementStep::Progressed
                if progress_before.submissions_in_flight == 1
                    && progress.submissions_in_flight == 0 =>
            {
                job.publication = RefinementPublicationState::Preparing;
                FrameTraceEventKind::RefinementSubmissionCompleted
            }
            GpuRefinementStep::Progressed | GpuRefinementStep::AwaitingGpu => {
                job.publication = if progress.submissions_in_flight == 0 {
                    RefinementPublicationState::Preparing
                } else {
                    RefinementPublicationState::SubmissionInFlight
                };
                FrameTraceEventKind::RefinementUnitProgress
            }
        };
        self.frame_trace.record_refinement(
            Instant::now(),
            kind,
            Some(job.origin),
            self.logical_frames.active_phase(),
            job.evaluation,
            job.target_quality,
            job.id,
            progress.completed_units,
            progress.total_units,
            submission_depth,
            Some(unit_started.elapsed()),
        );

        if !matches!(step, GpuRefinementStep::ReadyToPublish) {
            let elapsed = job.started_at.elapsed().as_secs_f32();
            self.ui_state.build_progress = Some(super::quality_in_flight_progress(
                job.target_quality,
                elapsed,
            ));
            self.refinement_job = Some(job);
            return !matches!(step, GpuRefinementStep::AwaitingGpu);
        }

        // No input can interleave with this event-loop turn. Recheck the token
        // immediately before consuming and publishing the candidate.
        if !job.is_fresh(self.eval_token) || self.input.has_pending() {
            if let Some(engine) = self.gpu_engine.as_mut() {
                engine.abandon_compiled_refinement(job.engine);
            }
            return false;
        }
        let target_quality = job.target_quality;
        let refinement_evaluation = job.evaluation;
        let refinement_origin = job.origin;
        let result = {
            let Some(engine) = self.gpu_engine.as_mut() else {
                return false;
            };
            match engine.publish_compiled_refinement(job.engine) {
                Ok(result) => result,
                Err(error) => {
                    log::warn!(target: "terra_app::evaluation", "refinement publication failed: {error}");
                    return false;
                }
            }
        };
        let pyramid_candidate = result
            .output_identity
            .map(|output| (output, result.width, result.height));
        if let Some(output) = result.output_identity {
            self.frame_trace.record_evaluation_output(
                Instant::now(),
                Some(refinement_origin),
                self.logical_frames.active_phase(),
                refinement_evaluation,
                output,
            );
        }
        if let (Some(engine), Some(renderer)) = (self.gpu_engine.as_ref(), self.renderer.as_mut()) {
            let result_metrics = self
                .session
                .document
                .metrics
                .at_resolution(result.width)
                .unwrap_or(self.session.document.metrics);
            let dx = result_metrics.dx();
            let dz = result_metrics.dz();
            let geom = terra_render::HeightPresentGeom {
                width: result.width,
                height: result.height,
                world_size: result.world_size,
                height_range: result.height_range,
                dx,
                dz,
            };
            if let Some(output) = result.output_identity {
                let record = renderer.present_gpu_height_shared_traced(
                    engine.output_texture(),
                    engine.output_texture_view(),
                    geom,
                    None,
                    output,
                    terra_render::TerrainPresentationExpectations {
                        plan_revision: output.plan_revision,
                        generation: output.generation,
                        extent: (result.width, result.height),
                    },
                );
                self.frame_trace.record_presentation(
                    Instant::now(),
                    Some(refinement_origin),
                    self.logical_frames.active_phase(),
                    refinement_evaluation,
                    record,
                );
            } else {
                renderer.present_gpu_height_shared(
                    engine.output_texture(),
                    engine.output_texture_view(),
                    geom,
                    None,
                );
            }
        }
        if let Some((output, width, height)) = pyramid_candidate {
            self.materialize_gpu_pyramid(output, width, height);
        }
        self.scheduler.quality = target_quality;
        self.ui_state.quality = target_quality;
        self.ui_state.profile.quality = match target_quality {
            PreviewQuality::Medium => "Medium",
            PreviewQuality::Full => "Final (viewport)",
            _ => unreachable!("refinement target is Medium or Full"),
        };
        self.ui_state.profile.gpu = self
            .gpu_engine
            .as_ref()
            .map(|engine| engine.last_eval_stats())
            .unwrap_or_default();
        self.ui_state.refining = target_quality.next_refine().is_some();
        self.ui_state.build_progress = self
            .ui_state
            .refining
            .then_some(quality_stage_progress(target_quality));
        self.ui_state.draft_displayed = matches!(target_quality, PreviewQuality::Medium);
        self.last_complete_generation = job.generation;
        self.last_accepted_evaluation_id = job.evaluation.get();
        self.last_eval_gpu_supported = true;
        self.pending_plan_invalidation = None;
        self.needs_height_upload = false;
        self.note_refinement_activity();
        self.frame_trace.record_refinement(
            Instant::now(),
            FrameTraceEventKind::RefinementPublished,
            Some(job.origin),
            self.logical_frames.active_phase(),
            job.evaluation,
            target_quality,
            job.id,
            progress.total_units,
            progress.total_units,
            0,
            Some(job.started_at.elapsed()),
        );
        true
    }

    pub(crate) fn supersede_gpu_refinement(&mut self) {
        let Some(job) = self.refinement_job.take() else {
            return;
        };
        let progress = job.engine.progress();
        if let Some(engine) = self.gpu_engine.as_mut() {
            engine.abandon_compiled_refinement(job.engine);
        }
        self.frame_trace.record_refinement(
            Instant::now(),
            FrameTraceEventKind::RefinementSuperseded,
            Some(job.origin),
            self.logical_frames.active_phase(),
            job.evaluation,
            job.target_quality,
            job.id,
            progress.completed_units,
            progress.total_units,
            progress.submissions_in_flight,
            Some(job.started_at.elapsed()),
        );
    }

    /// Offload CPU stack eval to the worker — never block the UI thread.
    pub(crate) fn enqueue_async_eval(&mut self, quality: PreviewQuality) {
        let preview_stack = self.session.document.preview_eval_stack();
        self.ui_state.refining_layer_name = self
            .session
            .document
            .selected
            .and_then(|id| preview_stack.find(id))
            .map(|layer| layer.common.name.clone())
            .or_else(|| Some("terrain".into()));
        // Ladder policy (a): a bounded sculpt scope over an already-Full worker
        // cache is submitted straight at Full, skipping the Draft/Medium CPU rungs
        // that would otherwise clobber the Full-res checkpoints it reuses.
        let quality = self.straight_to_full_quality(quality);

        // Copy — not drain — the dirty accumulators. `LatestWins` drops a job that
        // is superseded before it is dequeued *without running its body*, so a
        // drained scope on such a job would be lost, leaving stale tiles. Instead
        // the accumulators are cleared only when a fresh result is consumed (see the
        // Completed arm in `about_to_wait`); a re-carried scope is idempotent (region
        // unions, `dirty_from`/`mark_all` are supersets), so at worst a later job
        // over-recomputes. This is what makes the issue's seed-persistence invariant
        // hold against dequeue-skip, not just against in-body cancel.
        let mark_all_dirty = self.worker_mark_all_dirty;
        let dirty_from = self.worker_dirty_from;
        let dirty_region = self.worker_dirty_region;

        let request = EvalWorkRequest {
            token: self.eval_token,
            quality,
            stack: preview_stack,
            masks: self.session.document.masks.clone(),
            base_metrics: self.session.document.metrics,
            level_steps: self.session.document.level_steps.clone(),
            preview_res: self
                .session
                .document
                .preview_resolution
                .min(INTERACTIVE_PREVIEW_CAP),
            export_res: self.session.document.export_resolution,
            aux: self.scheduler.last_aux.clone(),
            strata: self.scheduler.last_strata.clone(),
            mask_reference: self.scheduler.last_good.clone(),
            dirty_from,
            dirty_region,
            mark_all_dirty,
        };
        match self.eval_worker.submit(request) {
            Ok(()) => {
                self.worker_refine_pending = true;
                self.ui_state.refining = true;
            }
            Err(error) => {
                // Nothing was drained, so there is nothing to restore. The restarted
                // worker's cache is empty (missing == dirty), so a whole-field mark
                // is the correct reset; drop the now-meaningless scope and cache res.
                self.eval_worker.restart();
                self.eval_worker.set_token(self.eval_token);
                self.worker_mark_all_dirty = true;
                self.worker_dirty_from = None;
                self.worker_dirty_region = None;
                self.worker_cache_res = None;
                self.handle_evaluation_failure_details(
                    self.eval_token,
                    quality,
                    None,
                    true,
                    format!("evaluation worker submission failed: {error}"),
                );
            }
        }
    }

    /// Ladder policy (a) (#100 phase 4): if a bounded sculpt scope is pending and
    /// the worker's persistent cache is already at Full resolution, promote the
    /// submit to Full and skip the Draft/Medium CPU rungs.
    ///
    /// The refine ladder resubmits at each rung's resolution, and the single-slot
    /// layer cache is keyed only by dimension — so a Draft rung after a Full stroke
    /// overwrites the Full-res checkpoints the next stroke's scope would reuse, and
    /// the Full rung then rebuilds whole-field despite a perfect one-tile scope.
    /// Submitting scoped-Full directly avoids the clobber: a Full eval of a few
    /// tiles is comparable to a whole-field Draft eval, immediate coarse feedback
    /// still comes from the GPU present, and the cache stays untouched. On promotion
    /// the scheduler/UI quality is synced to Full so the lifecycle refine loop
    /// settles there (`next_refine(Full) == None`) instead of re-queuing Medium.
    ///
    /// The gate is deliberately conservative: it fires only with a bounded scope
    /// (`dirty_region` set, no `mark_all`, a `dirty_from` suffix), an already-Full
    /// cache (`worker_cache_res`), and a scope mapping to at most a quarter of the
    /// Full tile grid. A cold cache or first stroke fails the gate and runs today's
    /// Draft→Medium→Full ladder unchanged.
    fn straight_to_full_quality(&mut self, requested: PreviewQuality) -> PreviewQuality {
        if matches!(requested, PreviewQuality::Full | PreviewQuality::Export) {
            return requested;
        }
        if self.worker_mark_all_dirty || self.worker_dirty_from.is_none() {
            return requested;
        }
        let Some(region) = self.worker_dirty_region else {
            return requested;
        };
        let full_res = self
            .session
            .document
            .preview_resolution
            .min(INTERACTIVE_PREVIEW_CAP);
        // Only worthwhile once the cache holds Full-res checkpoints to reuse.
        if self.worker_cache_res != Some(full_res) {
            return requested;
        }
        let Ok(full_metrics) = self.session.document.metrics.at_resolution(full_res) else {
            return requested;
        };
        let budget = (full_metrics.tile_count() / 4).max(1);
        let tiles = terra_core::tiling::tiles_for_uv_rect(&full_metrics, region);
        if tiles.len() as u32 > budget {
            return requested;
        }
        self.scheduler.quality = PreviewQuality::Full;
        self.ui_state.quality = PreviewQuality::Full;
        PreviewQuality::Full
    }

    pub(crate) fn evaluation_log_context(&self, token: u64, quality: PreviewQuality) -> String {
        let current = token == self.eval_token;
        let operation = OperationContext::evaluation(token, quality)
            .with_layer(
                current
                    .then_some(self.ui_state.refining_layer_name.as_deref())
                    .flatten(),
            )
            .with_project_path(current.then_some(self.project_path.as_deref()).flatten())
            .to_string();
        match (
            self.logical_frames.active_identity(),
            self.logical_frames.active_phase(),
        ) {
            (Some(identity), Some(phase)) => format!(
                "{operation}; logical_frame={}; edit_generation={}; phase={}",
                identity.id.get(),
                identity.generation.get(),
                phase.label()
            ),
            _ => operation,
        }
    }

    pub(crate) fn handle_evaluation_failure_details(
        &mut self,
        token: u64,
        quality: PreviewQuality,
        layer_name: Option<String>,
        worker_restarted: bool,
        message: impl std::fmt::Display,
    ) {
        let message = message.to_string();
        let context = self.evaluation_log_context(token, quality);
        log::error!(target: "terra_app::evaluation", "{message}; {context}");
        if token != self.eval_token {
            return;
        }
        let layer_name = layer_name.or_else(|| self.ui_state.refining_layer_name.clone());
        self.worker_refine_pending = false;
        self.ui_state.refining = false;
        self.ui_state.build_progress = None;
        self.ui_state.refining_layer_name = None;
        self.ui_state.status = if worker_restarted {
            "Terrain evaluation failed; worker restarted".into()
        } else {
            "Terrain evaluation failed; last good preview retained".into()
        };
        self.ui_state.evaluation_failure = Some(crate::ui::EvaluationFailureStatus {
            layer_name,
            quality,
            message,
            worker_restarted,
        });
    }

    /// The pyramid level matching `last_height`, as `(level_index, resolution)`.
    /// `None` when there is no `last_height` or its resolution is not a pyramid
    /// level — e.g. an interactive Export-quality result whose size exceeds every
    /// level. Both the upload side and the renderer sync read this one authority,
    /// so a fabricated level can never be stamped on pages the shader then samples
    /// at a different level (the divergent-fallback bug this replaces).
    pub(crate) fn streamed_level_for_last_height(&self) -> Option<(u8, u32)> {
        let width = self.last_height.as_ref()?.metrics.width;
        self.terrain_runtime
            .pyramid
            .levels
            .iter()
            .find(|level| level.resolution == width)
            .map(|level| (level.index, level.resolution))
    }

    pub(crate) fn streamed_level_for_current_output(&self) -> Option<(u8, u32)> {
        self.gpu_height_pyramid
            .as_ref()
            .and_then(|pyramid| {
                pyramid
                    .descriptor()
                    .level(pyramid.source_level())
                    .map(|level| (level.index, level.resolution))
            })
            .or_else(|| self.streamed_level_for_last_height())
    }

    pub(crate) fn queue_final_tile_uploads(&mut self) {
        self.clear_terrain_tile_work();
        self.gpu_height_pyramid = None;
        self.clear_terrain_demand();
        if self.tile_atlas.is_none() {
            return;
        }
        let Some((level, _res)) = self.streamed_level_for_last_height() else {
            // No pyramid level matches this result's resolution: don't stamp pages
            // at a fabricated level. Streaming stays off and the monolithic path
            // (normalized, correct at any resolution) presents.
            return;
        };
        let Some(height) = self.last_height.as_ref() else {
            return;
        };
        self.next_cpu_tile_content_revision =
            self.next_cpu_tile_content_revision.wrapping_add(1).max(1);
        let stamp = terra_core::TerrainContentStamp {
            document_revision: self.eval_token,
            plan_revision: self.terrain_plan_cache.structure_revision().get(),
            output_revision: self.terrain_runtime.output_revision(),
            content_revision: self.next_cpu_tile_content_revision,
        };
        if let (Some(atlas), Some(gpu)) = (self.tile_atlas.as_mut(), self.gpu.as_ref()) {
            atlas.configure_hierarchy(&gpu.device, &gpu.queue, &self.terrain_runtime.pyramid);
        }
        let requests = height
            .tiles()
            .iter()
            .map(|tile| terra_core::TerrainTileWorkRequest {
                key: terra_core::TerrainTileWorkKey {
                    tile: terra_core::TerrainTileKey {
                        layer: None,
                        field: terra_core::FieldId::Height,
                        level,
                        tile: tile.id,
                    },
                    plan_revision: stamp.plan_revision,
                    output_revision: stamp.output_revision,
                },
                content: stamp,
                source: terra_core::TerrainTileWorkSource::CpuHeight,
                class: terra_core::TerrainDemandClass::CoarseCoverage,
                visible: false,
                projected_error_px: 0.0,
                distance_m: f32::INFINITY,
                estimated_us: 250,
            });
        self.terrain_tile_scheduler.reconcile(stamp, requests);
        self.ui_state
            .profile
            .update_terrain_tile_work(self.terrain_tile_scheduler.stats());
    }

    fn materialize_gpu_pyramid(
        &mut self,
        output: terra_gpu::output_identity::GpuTerrainOutputIdentity,
        width: u32,
        height: u32,
    ) {
        if !output.is_current_complete_final()
            || output.intent != GpuEvaluationIntent::Complete
            || output.generation != self.eval_token
            || self.tile_atlas.is_none()
        {
            return;
        }
        let Some(gpu) = self.gpu.as_ref() else {
            return;
        };
        if self.gpu_pyramid_materializer.is_none() {
            self.gpu_pyramid_materializer =
                Some(terra_gpu::GpuHeightPyramidMaterializer::new(&gpu.device));
        }
        let Some(engine) = self.gpu_engine.as_ref() else {
            return;
        };
        let materialized = self
            .gpu_pyramid_materializer
            .as_ref()
            .expect("initialized above")
            .materialize(
                &gpu.device,
                &gpu.queue,
                &self.terrain_runtime.pyramid,
                engine.output_texture(),
                (width, height),
                GpuPyramidContentIdentity {
                    output_revision: self.terrain_runtime.output_revision(),
                    output: output.output,
                    generation: output.generation,
                    plan_revision: output.plan_revision,
                },
            );
        match materialized {
            Ok(pyramid) => {
                let stamp = pyramid.identity().content_stamp();
                let error_readback = pyramid.begin_error_readback(&gpu.device, &gpu.queue);
                if let Some(atlas) = self.tile_atlas.as_mut() {
                    atlas.configure_hierarchy(&gpu.device, &gpu.queue, pyramid.descriptor());
                }
                self.clear_terrain_demand();
                self.gpu_pyramid_error_readback = Some(error_readback);
                self.gpu_height_pyramid = Some(pyramid);
                self.clear_terrain_tile_work();
                let root_metrics = self
                    .terrain_runtime
                    .pyramid
                    .level_metrics(0)
                    .expect("materialized root level");
                // Bootstrap the fallback chain while the compact metadata copy is
                // mapping. The completed demand plan replaces this queue.
                let mut requests = Vec::new();
                for tz in 0..root_metrics.tiles_z() {
                    for tx in 0..root_metrics.tiles_x() {
                        requests.push(terra_core::TerrainTileWorkRequest {
                            key: terra_core::TerrainTileWorkKey {
                                tile: terra_core::TerrainTileKey {
                                    layer: None,
                                    field: terra_core::FieldId::Height,
                                    level: 0,
                                    tile: terra_core::TileId { tx, tz },
                                },
                                plan_revision: stamp.plan_revision,
                                output_revision: stamp.output_revision,
                            },
                            content: stamp,
                            source: terra_core::TerrainTileWorkSource::GpuCompiledPlan,
                            class: terra_core::TerrainDemandClass::CoarseCoverage,
                            visible: true,
                            projected_error_px: f32::MAX,
                            distance_m: 0.0,
                            estimated_us: 250,
                        });
                    }
                }
                self.terrain_tile_scheduler.reconcile(stamp, requests);
                self.ui_state
                    .profile
                    .update_terrain_tile_work(self.terrain_tile_scheduler.stats());
            }
            Err(error) => {
                log::warn!(target: "terra_app::evaluation", "GPU pyramid materialization skipped: {error}");
            }
        }
    }

    fn clear_terrain_demand(&mut self) {
        self.gpu_pyramid_error_readback = None;
        self.gpu_pyramid_planning_metadata = None;
        self.latest_terrain_demand = None;
        self.terrain_demand_planner.reset();
    }

    /// Poll the one-shot geometric-error metadata transfer and, once ready,
    /// refresh the camera-visible demand consumed by the current upload adapter.
    pub(crate) fn refresh_terrain_demand(&mut self) -> bool {
        let metadata_result = match (self.gpu.as_ref(), self.gpu_pyramid_error_readback.as_mut()) {
            (Some(gpu), Some(readback)) => Some(readback.poll(&gpu.device)),
            _ => None,
        };
        if let Some(result) = metadata_result {
            match result {
                Ok(Some(metadata)) => {
                    self.gpu_pyramid_error_readback = None;
                    let live = self.gpu_height_pyramid.as_ref().is_some_and(|pyramid| {
                        pyramid.identity() == metadata.identity
                            && metadata.identity.output_revision
                                == self.terrain_runtime.output_revision()
                    });
                    if live {
                        let mut metadata = metadata;
                        let descriptor = self
                            .gpu_height_pyramid
                            .as_ref()
                            .expect("live pyramid checked above")
                            .descriptor();
                        match terra_core::conservative_geometric_errors(
                            descriptor,
                            &metadata.geometric_errors,
                        ) {
                            Ok(errors) => metadata.geometric_errors = errors,
                            Err(error) => {
                                log::warn!(target: "terra_app::evaluation", "terrain demand metadata rejected: {error}");
                                return false;
                            }
                        }
                        self.gpu_pyramid_planning_metadata = Some(metadata);
                        self.latest_terrain_demand = None;
                        self.terrain_demand_planner.reset();
                    }
                }
                Ok(None) => return false,
                Err(error) => {
                    self.gpu_pyramid_error_readback = None;
                    log::warn!(target: "terra_app::evaluation", "terrain demand metadata skipped: {error}");
                    return false;
                }
            }
        }

        let Some(pyramid) = self.gpu_height_pyramid.as_ref() else {
            return false;
        };
        let Some(metadata) = self.gpu_pyramid_planning_metadata.as_ref() else {
            return false;
        };
        if metadata.identity != pyramid.identity()
            || metadata.identity.output_revision != self.terrain_runtime.output_revision()
        {
            return false;
        }
        let Some(renderer) = self.renderer.as_ref() else {
            return false;
        };
        let (width, height) = renderer.size();
        let aspect = width as f32 / height.max(1) as f32;
        let (min_height, max_height) = renderer.heights.height_range;
        let view = terra_core::TerrainDemandView {
            eye: renderer.camera.eye(),
            view_proj: renderer.camera.view_proj(aspect),
            fov_y: renderer.camera.fov_y,
            near: renderer.camera.near,
            viewport_width_px: width,
            viewport_height_px: height,
            min_height,
            max_height,
        };
        let max_pages = self
            .tile_atlas
            .as_ref()
            .map_or(1usize, |atlas| atlas.max_pages() as usize);
        let config = terra_core::TerrainDemandConfig {
            max_demand_tiles: max_pages,
            max_visited_nodes: max_pages.saturating_mul(32).max(64),
            ..terra_core::TerrainDemandConfig::default()
        };
        let plan = match self.terrain_demand_planner.plan(
            pyramid.descriptor(),
            &metadata.geometric_errors,
            view,
            config,
        ) {
            Ok(plan) => plan,
            Err(error) => {
                log::warn!(target: "terra_app::evaluation", "terrain demand planning skipped: {error}");
                return false;
            }
        };
        let identity = pyramid.identity();
        let stamp = identity.content_stamp();
        let mut requests = Vec::new();
        for demand in &plan.tiles {
            let resident_current = self
                .tile_atlas
                .as_ref()
                .is_some_and(|atlas| atlas.is_current(&demand.key, stamp));
            if !resident_current {
                requests.push(terra_core::TerrainTileWorkRequest {
                    key: terra_core::TerrainTileWorkKey {
                        tile: demand.key.clone(),
                        plan_revision: stamp.plan_revision,
                        output_revision: stamp.output_revision,
                    },
                    content: stamp,
                    source: terra_core::TerrainTileWorkSource::GpuCompiledPlan,
                    class: demand.class,
                    visible: true,
                    projected_error_px: demand.projected_error_px,
                    distance_m: demand.distance_m,
                    estimated_us: 250,
                });
            }
        }
        if let Some(atlas) = self.tile_atlas.as_ref() {
            self.terrain_tile_scheduler
                .set_capacity(atlas.max_pages() as usize);
        }
        self.terrain_tile_scheduler.reconcile(stamp, requests);
        self.ui_state
            .profile
            .update_terrain_tile_work(self.terrain_tile_scheduler.stats());
        let changed = self.latest_terrain_demand.as_ref() != Some(&plan);
        self.latest_terrain_demand = Some(plan);
        changed || !self.terrain_tile_scheduler.is_empty()
    }

    pub(crate) fn upload_pending_terrain_tiles(&mut self) -> usize {
        if self.tile_atlas.is_none()
            || self.gpu.is_none()
            || (self.last_height.is_none() && self.gpu_height_pyramid.is_none())
        {
            self.clear_terrain_tile_work();
            return 0;
        }
        let Some(live_stamp) = self.terrain_tile_scheduler.live_content() else {
            return 0;
        };
        let budget = terrain_tile_work_budget(
            self.terrain_runtime.refinement.state(),
            self.tile_atlas.as_ref().unwrap().max_pages() as usize,
        );
        let published_frame = self
            .renderer
            .as_ref()
            .map_or(0, terra_render::TerrainRenderer::global_frame_index);
        let mut uploaded = self.poll_compiled_tile_jobs(live_stamp, published_frame);
        let leases = self.terrain_tile_scheduler.dequeue_budgeted(budget);
        for lease in leases {
            if lease.request.content != live_stamp {
                self.terrain_tile_scheduler.fail(lease);
                continue;
            }
            let key = lease.request.key.tile.clone();
            if self
                .tile_atlas
                .as_ref()
                .unwrap()
                .is_current(&key, live_stamp)
            {
                self.terrain_tile_scheduler
                    .skip_in_flight_as_resident(lease);
                continue;
            }
            if lease.request.source == terra_core::TerrainTileWorkSource::GpuCompiledPlan {
                let preview_stack = self.session.document.preview_eval_stack();
                let plan = self
                    .terrain_plan_cache
                    .acquire(&preview_stack, &self.session.document.masks)
                    .ok()
                    .cloned();
                let started_engine = plan.and_then(|plan| {
                    let slice =
                        terra_gpu_eval::GpuCompiledTileProducer::analyze(&preview_stack, &plan)
                            .map_err(|error| {
                                log::debug!("compiled tile deferred to pyramid: {error:?}");
                            })
                            .ok()?;
                    let domain = terra_core::TerrainEvaluationDomain::for_tile(
                        &self.terrain_runtime.pyramid,
                        key.clone(),
                        self.tile_atlas.as_ref().unwrap().halo(),
                        slice.operation_halo,
                        live_stamp,
                    )
                    .map_err(|error| {
                        log::warn!("compiled tile domain rejected: {error:?}");
                    })
                    .ok()?;
                    let invalidation = self.pending_plan_invalidation.clone().unwrap_or_default();
                    match self.compiled_tile_producer.begin(
                        &self.gpu.as_ref().unwrap().device,
                        &self.gpu.as_ref().unwrap().queue,
                        &preview_stack,
                        &self.session.document.masks,
                        &plan,
                        &invalidation,
                        PreviewQuality::Full,
                        domain,
                    ) {
                        Ok(engine) => Some(engine),
                        Err(error) => {
                            log::debug!("compiled tile deferred to pyramid: {error:?}");
                            None
                        }
                    }
                });
                if let Some(engine) = started_engine {
                    self.compiled_tile_jobs
                        .push(super::CompiledTileWorkJob { lease, engine });
                    continue;
                }
                // Explicit complete-field fallback for unsupported/global work.
                let Some(pyramid) = self
                    .gpu_height_pyramid
                    .as_ref()
                    .filter(|pyramid| pyramid.identity().content_stamp() == live_stamp)
                else {
                    self.terrain_tile_scheduler.fail(lease);
                    continue;
                };
                let result = self
                    .tile_atlas
                    .as_mut()
                    .unwrap()
                    .publish_pyramid_tile_current_at_frame(
                        &self.gpu.as_ref().unwrap().device,
                        &self.gpu.as_ref().unwrap().queue,
                        pyramid,
                        key.clone(),
                        live_stamp,
                        published_frame,
                    );
                match result {
                    Ok(_) => {
                        uploaded += 1;
                        if key.level == 0 {
                            let _ = self.tile_atlas.as_mut().unwrap().pin(&key);
                        }
                        self.terrain_tile_scheduler.complete(lease, live_stamp);
                    }
                    Err(error) => {
                        log::warn!("terrain tile fallback upload failed: {error}");
                        self.terrain_tile_scheduler.fail(lease);
                    }
                }
                continue;
            }
            let result = match lease.request.source {
                terra_core::TerrainTileWorkSource::CpuHeight => {
                    let Some(tile) = self
                        .last_height
                        .as_ref()
                        .and_then(|height| height.tile(key.tile))
                    else {
                        self.terrain_tile_scheduler.fail(lease);
                        continue;
                    };
                    self.tile_atlas
                        .as_mut()
                        .unwrap()
                        .upload_height_tile_current_at_frame(
                            &self.gpu.as_ref().unwrap().queue,
                            key.clone(),
                            tile,
                            live_stamp,
                            published_frame,
                        )
                }
                terra_core::TerrainTileWorkSource::GpuPyramid => {
                    let Some(pyramid) = self
                        .gpu_height_pyramid
                        .as_ref()
                        .filter(|pyramid| pyramid.identity().content_stamp() == live_stamp)
                    else {
                        self.terrain_tile_scheduler.fail(lease);
                        continue;
                    };
                    self.tile_atlas
                        .as_mut()
                        .unwrap()
                        .publish_pyramid_tile_current_at_frame(
                            &self.gpu.as_ref().unwrap().device,
                            &self.gpu.as_ref().unwrap().queue,
                            pyramid,
                            key.clone(),
                            live_stamp,
                            published_frame,
                        )
                }
                terra_core::TerrainTileWorkSource::GpuCompiledPlan => unreachable!(),
            };
            match result {
                Ok(_) => {
                    uploaded += 1;
                    if key.level == 0 {
                        let _ = self.tile_atlas.as_mut().unwrap().pin(&key);
                    }
                    self.terrain_tile_scheduler.complete(lease, live_stamp);
                }
                Err(error) => {
                    log::warn!("terrain tile upload failed: {error}");
                    self.terrain_tile_scheduler.fail(lease);
                }
            }
            if let Some(atlas) = self.tile_atlas.as_ref() {
                self.ui_state.profile.update_tile_cache(
                    atlas.residency().stats(),
                    self.terrain_tile_scheduler.len(),
                );
            }
        }
        if uploaded > 0 {
            self.sync_tile_stream_to_renderer();
        }
        self.ui_state
            .profile
            .update_terrain_tile_work(self.terrain_tile_scheduler.stats());
        uploaded
    }

    fn poll_compiled_tile_jobs(
        &mut self,
        live_stamp: terra_core::TerrainContentStamp,
        published_frame: u64,
    ) -> usize {
        let mut retained = Vec::new();
        let mut uploaded = 0;
        for mut work in std::mem::take(&mut self.compiled_tile_jobs) {
            let fresh = self
                .terrain_tile_scheduler
                .lease_is_live(work.lease.id, live_stamp)
                && work.lease.request.content == live_stamp;
            if !fresh {
                self.compiled_tile_producer.cancel(work.engine);
                continue;
            }
            if !self
                .compiled_tile_producer
                .poll(&self.gpu.as_ref().unwrap().device, &mut work.engine)
            {
                retained.push(work);
                continue;
            }
            let key = work.lease.request.key.tile.clone();
            let result = self
                .tile_atlas
                .as_mut()
                .unwrap()
                .publish_evaluated_tile_current_at_frame(
                    &self.gpu.as_ref().unwrap().device,
                    &self.gpu.as_ref().unwrap().queue,
                    work.engine.output_texture_view().expect("completed tile"),
                    work.engine.domain(),
                    live_stamp,
                    published_frame,
                );
            self.compiled_tile_producer.recycle(work.engine);
            match result {
                Ok(_) => {
                    uploaded += 1;
                    if key.level == 0 {
                        let _ = self.tile_atlas.as_mut().unwrap().pin(&key);
                    }
                    self.terrain_tile_scheduler.complete(work.lease, live_stamp);
                }
                Err(error) => {
                    log::warn!("compiled terrain tile publication failed: {error}");
                    self.terrain_tile_scheduler.fail(work.lease);
                }
            }
        }
        self.compiled_tile_jobs = retained;
        uploaded
    }

    pub(crate) fn clear_terrain_tile_work(&mut self) {
        for work in std::mem::take(&mut self.compiled_tile_jobs) {
            self.compiled_tile_producer.cancel(work.engine);
        }
        self.terrain_tile_scheduler.clear();
    }

    pub(crate) fn sync_tile_stream_to_renderer(&mut self) {
        if self.tile_atlas.is_none() {
            return;
        }
        let Some(content) = self.terrain_tile_scheduler.live_content() else {
            if let Some(renderer) = self.renderer.as_mut() {
                renderer.set_use_tile_stream(false);
            }
            return;
        };
        let Some((level, level_res)) = self.streamed_level_for_current_output() else {
            if let Some(renderer) = self.renderer.as_mut() {
                renderer.set_use_tile_stream(false);
            }
            return;
        };
        let root_required = self.gpu_height_pyramid.is_some();
        // A new GPU revision initially queues only root coverage while the
        // geometric-error readback and camera demand are still pending. Enabling
        // the stream at that point replaces the complete monolithic terrain with
        // its coarse root ancestor for several frames, which presents as a
        // whole-terrain flash after a brush stroke. Keep the revision boundary on
        // the monolithic output until the current camera-demand set (including any
        // asynchronous compiled-tile jobs) is fully resident, then switch once.
        let gpu_demand_ready = !root_required
            || (self.gpu_pyramid_error_readback.is_none()
                && self.latest_terrain_demand.is_some()
                && self.terrain_tile_scheduler.is_empty()
                && self.compiled_tile_jobs.is_empty());
        if !gpu_demand_ready {
            if let Some(renderer) = self.renderer.as_mut() {
                renderer.set_use_tile_stream(false);
            }
            return;
        }
        let root_current = if root_required {
            let root_metrics = self.terrain_runtime.pyramid.level_metrics(0);
            root_metrics.is_some_and(|metrics| {
                let atlas = self.tile_atlas.as_ref().expect("atlas checked below");
                (0..metrics.tiles_z()).all(|tz| {
                    (0..metrics.tiles_x()).all(|tx| {
                        atlas.is_current(
                            &terra_core::TerrainTileKey {
                                layer: None,
                                field: terra_core::FieldId::Height,
                                level: 0,
                                tile: terra_core::TileId { tx, tz },
                            },
                            content,
                        )
                    })
                })
            })
        } else {
            true
        };
        if !root_current {
            if let Some(renderer) = self.renderer.as_mut() {
                renderer.set_use_tile_stream(false);
            }
            return;
        }
        let resources = {
            let Some(atlas) = self.tile_atlas.as_ref() else {
                return;
            };
            terra_render::TerrainTileStreamResources {
                atlas_view: atlas.create_texture_view(),
                physical_page_table: atlas.page_table_buffer_cloned(),
                virtual_page_table: atlas.virtual_page_table_buffer_cloned(),
                level_table: atlas.level_table_buffer_cloned(),
                tile_size: atlas.tile_size(),
                halo: atlas.halo(),
                max_pages: atlas.max_pages(),
                level_count: atlas.level_count(),
                target_level: level,
                target_resolution: level_res,
                content,
                transition_frames: 8,
                terminal_fallback: if root_required {
                    terra_render::TerrainTerminalFallback::RootRequired
                } else {
                    terra_render::TerrainTerminalFallback::MonolithicMigration
                },
                enable: true,
            }
        };
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        renderer.set_tile_stream_resources(resources);
    }

    /// GPU residency half of the output-revision boundary. `TerrainPyramid` is only
    /// a resolution ladder and has no resident-page records; this retires pending
    /// uploads, the atlas page table + its `TileResidencyCache` policy mirror, and
    /// the renderer's streaming flag. Presentation stays continuous via the
    /// shader's monolithic page-miss fallback until `upload_pending_terrain_tiles` →
    /// `sync_tile_stream_to_renderer` re-enable streaming for the new revision.
    pub(crate) fn retire_streamed_residency(&mut self) {
        self.clear_terrain_tile_work();
        self.ui_state
            .profile
            .update_terrain_tile_work(self.terrain_tile_scheduler.stats());
        self.gpu_height_pyramid = None;
        self.clear_terrain_demand();
        if let (Some(atlas), Some(gpu)) = (self.tile_atlas.as_mut(), self.gpu.as_ref()) {
            atlas.clear(&gpu.queue);
            self.ui_state
                .profile
                .update_tile_cache(atlas.residency().stats(), 0);
        }
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.set_use_tile_stream(false);
        }
    }

    /// Single door for edit-driven revision advancement: bump the output revision
    /// and retire both sides of tile residency in the same step. Callers that
    /// advance the revision via `reconfigure` instead call
    /// [`Self::retire_streamed_residency`] directly.
    pub(crate) fn advance_output_revision(&mut self) {
        self.terrain_runtime.advance_output_revision();
        self.retire_streamed_residency();
    }

    pub(crate) fn mark_dirty_from(&mut self, id: LayerId) {
        let preview = self.session.document.preview_eval_stack();
        self.pending_plan_edits
            .push(terra_core::terrain_plan::TerrainEditClass::Parameters {
                owner: terra_core::deps::NodeRef::Layer(id),
            });
        self.scheduler.evaluator.mark_dirty_from(&preview, id);
        self.advance_output_revision();
        // No spatial footprint (param/structural edit): whole-field suffix.
        self.track_worker_dirty_from(&preview, id, None);
        if let Some(gpu) = self.gpu_engine.as_mut() {
            gpu.mark_dirty_from(&preview, id);
        }
    }

    pub(crate) fn mark_dirty_from_stage(&mut self, id: LayerId) {
        let preview = self.session.document.preview_eval_stack();
        self.pending_plan_edits
            .push(terra_core::terrain_plan::TerrainEditClass::Parameters {
                owner: terra_core::deps::NodeRef::Layer(id),
            });
        self.scheduler.evaluator.mark_dirty_from_stage(&preview, id);
        self.advance_output_revision();
        self.track_worker_dirty_from(&preview, id, None);
        if let Some(gpu) = self.gpu_engine.as_mut() {
            // GPU path still uses suffix dirty; stage-aware CPU cache is the main win.
            gpu.mark_dirty_from(&preview, id);
        }
    }

    /// Mirror a suffix dirty from `id` onto the worker accumulators, folding in an
    /// optional spatial `footprint` (normalized UV).
    ///
    /// `footprint`:
    /// - `Some(rect)` on the *first* pending edit seeds the bounded scope; on a
    ///   later edit whose scope is still bounded it unions in; but once any
    ///   footprint-less edit has escalated the region to `None`, it stays `None`
    ///   (whole-field suffix) until the accumulators are cleared — never narrowed
    ///   back to a rect, which would drop the escalated dirt.
    /// - `None` escalates the region to `None`: an edit with no known footprint
    ///   dirties the whole suffix, exactly as before this scope existed.
    pub(crate) fn track_worker_dirty_from(
        &mut self,
        stack: &terra_core::layer::LayerStack,
        id: LayerId,
        footprint: Option<UvRect>,
    ) {
        if self.worker_mark_all_dirty {
            return;
        }
        let had_pending = self.worker_dirty_from.is_some();
        let ids = stack.layer_ids();
        let Some(next_index) = ids.iter().position(|candidate| *candidate == id) else {
            self.worker_mark_all_dirty = true;
            self.worker_dirty_from = None;
            self.worker_dirty_region = None;
            return;
        };
        let replace = self
            .worker_dirty_from
            .and_then(|current| ids.iter().position(|candidate| *candidate == current))
            .is_none_or(|current_index| next_index < current_index);
        if replace {
            self.worker_dirty_from = Some(id);
        }
        match footprint {
            // First bounded edit since the last clear: seed the scope. A later
            // bounded edit unions in. A bounded edit *after* an escalation keeps the
            // escalated whole-field `None` (do not narrow).
            Some(rect) => {
                if !had_pending {
                    self.worker_dirty_region = Some(rect);
                } else if let Some(existing) = self.worker_dirty_region {
                    self.worker_dirty_region = Some(existing.union(rect));
                }
            }
            None => self.worker_dirty_region = None,
        }
    }

    /// Mirror a stage-scoped UI-evaluator dirty ([`StackEvaluator::mark_dirty_from_eval_stage`])
    /// onto the background worker's dirty accumulators.
    ///
    /// The worker request carries only a single suffix layer (`dirty_from`) or a
    /// whole-field flag — it has no stage field — so a stage dirty is mirrored by
    /// dirtying from the earliest in-stack layer at or after `stage`. Every
    /// stage-dirtied layer has an index at least that earliest one's, so the suffix
    /// is a superset of the stage set: never an under-dirty, at worst a bounded
    /// over-recompute of clean lower-stage layers sitting above it. Without this,
    /// explicit scenario (re)builds dirty only the UI-thread evaluator and the
    /// worker reuses its stale checkpoint for the (worker-authoritative) sim layers.
    pub(crate) fn track_worker_dirty_from_eval_stage(
        &mut self,
        stack: &terra_core::layer::LayerStack,
        stage: terra_core::EvalStage,
    ) {
        let min_order = stage.order();
        let earliest = stack.layer_ids().into_iter().find(|id| {
            stack
                .find(*id)
                .is_some_and(|layer| layer.kind.eval_stage().order() >= min_order)
        });
        if let Some(id) = earliest {
            // Stage (re)builds have no spatial footprint: whole-field suffix.
            self.track_worker_dirty_from(stack, id, None);
        }
    }

    pub(crate) fn mark_all_layers_dirty(&mut self) {
        let preview = self.session.document.preview_eval_stack();
        self.pending_plan_edits
            .push(terra_core::terrain_plan::TerrainEditClass::Structure);
        self.scheduler.evaluator.mark_all_dirty(&preview);
        let metrics = self.session.document.metrics;
        self.terrain_runtime
            .reconfigure(terra_core::PyramidConfig::new(
                self.session.document.preview_resolution,
                metrics.world_size_x,
                metrics.world_size_z,
            ));
        // reconfigure() advances the output revision; retire the streamed side too.
        self.retire_streamed_residency();
        self.worker_mark_all_dirty = true;
        self.worker_dirty_from = None;
        self.worker_dirty_region = None;
        if let Some(gpu) = self.gpu_engine.as_mut() {
            gpu.mark_all_dirty(&preview);
        }
    }

    /// After Shape history edits: mark downstream simulation layers outdated
    /// without scheduling an immediate sim rebuild.
    pub(crate) fn mark_shape_dependents_outdated(&mut self) {
        let Some(shape_id) = self
            .ui_state
            .shape_session_layer
            .or(self.session.document.selected)
        else {
            return;
        };
        self.mark_dirty_from(shape_id);
        let now_ms = self.last_edit.elapsed().as_millis().saturating_add(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0),
        ) as u64;
        let feedback = terra_core::rebuild_feedback::apply_upstream_change(
            &mut self.session,
            terra_core::deps::NodeRef::Layer(shape_id),
            "Shape / region geometry edited",
            now_ms,
        );
        if !feedback.is_empty() {
            self.ui_state.status = feedback.format_updating().replace('\n', " Â· ");
            self.ui_state.affected_feedback = Some(feedback.format_updating());
        }
    }

    /// Bind / create Simulation Layers for unbound scenario passes (reuse existing solvers).
    pub(crate) fn ensure_scenario_layers(
        &mut self,
        id: terra_core::simulation_scenario::SimulationScenarioId,
    ) {
        use terra_core::layer::{Layer, StackCategory};
        use terra_core::simulation_scenario::layer_kind_is_scenario_compatible;

        let Some(scenario) = self.session.document.simulation_scenarios.get(id).cloned() else {
            return;
        };
        for pass in &scenario.passes {
            if !pass.enabled {
                continue;
            }
            if let Some(lid) = pass.layer_id {
                if self.session.document.stack.find(lid).is_some() {
                    continue;
                }
            }
            // Prefer an existing compatible unbound layer of the same kind.
            let existing = self
                .session
                .document
                .stack
                .flatten_layers()
                .into_iter()
                .find(|l| {
                    layer_kind_is_scenario_compatible(&l.kind)
                        && terra_core::simulation_scenario::ScenarioPassKind::from_layer_kind(
                            &l.kind,
                        ) == Some(pass.kind)
                })
                .map(|l| l.id());
            if let Some(lid) = existing {
                if let Some(s) = self.session.document.simulation_scenarios.get_mut(id) {
                    let _ = s.bind_pass_layer(pass.id, lid);
                }
                continue;
            }
            // Create a new Simulation Layer via the existing framework.
            let kind = pass.kind.default_layer_kind();
            let layer = Layer::new(pass.name.clone(), kind);
            let lid = layer.id();
            self.session.document.stack.push_into_category(layer);
            // Ensure it landed in Simulation category when possible.
            let _ = StackCategory::Simulation;
            if let Some(s) = self.session.document.simulation_scenarios.get_mut(id) {
                let _ = s.bind_pass_layer(pass.id, lid);
            }
        }
    }

    /// When a biome group's DistNode stack is edited by hand, freeze Placement rules.
    pub(crate) fn mark_biome_placement_custom_for_group(
        doc: &mut terra_core::document::TerrainDocument,
        group_id: terra_core::layer::LayerId,
    ) {
        let stack = doc
            .stack
            .find_group(group_id)
            .map(|g| g.masks.clone())
            .unwrap_or_default();
        if let Some(def) = doc.biome_library.by_group_mut(group_id) {
            def.placement.mark_mask_stack_custom(stack);
        }
    }

    #[cfg(test)]
    pub(crate) fn run_eval_step(&mut self) {
        self.run_eval_step_with_intent(GpuEvaluationIntent::Complete);
    }

    /// Settle a globally coupled suffix at the quality already presented by the
    /// local prefix. Downgrading here would replace a resident Full texture with
    /// Draft immediately after the gesture, causing the viewport to disappear and
    /// rebuild in coarse squares.
    pub(crate) fn complete_deferred_full_field(&mut self) {
        self.run_eval_step_with_intent(GpuEvaluationIntent::Complete);
        self.note_refinement_activity();
    }

    pub(crate) fn run_eval_step_with_intent(&mut self, intent: GpuEvaluationIntent) {
        profiling::scope!("eval_step");
        let t0 = Instant::now();
        let trace_id = self.frame_trace.next_evaluation_id();
        let logical_frame_identity = self.logical_frames.active_identity();
        self.frame_trace.record(
            t0,
            FrameTraceEventKind::EvaluationRequested,
            logical_frame_identity,
            self.logical_frames.active_phase(),
            Some(trace_id),
            Some(self.scheduler.quality),
            Some(intent),
            None,
        );
        self.ui_state.profile.first_visible_preview_us = 0;
        self.ui_state.profile.settled_authoritative_us = 0;
        let preview = self
            .session
            .document
            .preview_resolution
            .min(INTERACTIVE_PREVIEW_CAP);
        let export = self.session.document.export_resolution;
        if self.force_draft {
            let full_resolution = PreviewQuality::Full.resolution(preview, export);
            let selected_is_bounded_sculpt = self
                .session
                .document
                .selected
                .and_then(|selected| self.session.document.stack.find(selected))
                .is_some_and(|layer| {
                    matches!(
                        layer.kind,
                        LayerKind::SculptBase(_) | LayerKind::SculptStrokes(_)
                    )
                });
            let retained_full = self.renderer.as_ref().is_some_and(|renderer| {
                renderer.heights.tex_size.0 == full_resolution
                    && renderer.heights.tex_size.1 == full_resolution
            });
            self.scheduler.quality = if selected_is_bounded_sculpt
                && self.pending_gpu_dirty_region.is_some()
                && self.last_eval_gpu_supported
                && retained_full
            {
                // Both Foundation and Shape Layer edits carry bounded footprints
                // and can update the already-valid Full realization regionally.
                // Keeping it resident avoids the Draft→Full allocation cliff.
                PreviewQuality::Full
            } else {
                PreviewQuality::Draft
            };
            self.force_draft = false;
        }
        let base = self.session.document.metrics;
        let quality = self.scheduler.quality;
        self.frame_trace.record(
            Instant::now(),
            FrameTraceEventKind::EvaluationStarted,
            logical_frame_identity,
            self.logical_frames.active_phase(),
            Some(trace_id),
            Some(quality),
            Some(intent),
            Some(t0.elapsed()),
        );
        // Global + active region â€” interactive preview must include world.global.
        let preview_stack = self.session.document.preview_eval_stack();
        self.ui_state.refining_layer_name = self
            .session
            .document
            .selected
            .and_then(|id| preview_stack.find(id))
            .map(|layer| layer.common.name.clone())
            .or_else(|| Some("terrain".into()));
        let res = quality.resolution(preview, export);
        let token = self.eval_token;
        let metrics = match base.at_resolution(res) {
            Ok(metrics) => metrics,
            Err(error) => {
                // Loads validate metrics, so this is only reachable if the live
                // document's base metrics were corrupted in memory. Surface it as
                // an evaluation failure instead of panicking in tile arithmetic.
                self.handle_evaluation_failure_details(token, quality, None, false, error);
                return;
            }
        };
        let operation_context = self.evaluation_log_context(token, quality);
        // Interactive evaluation must never force a GPU readback/Wait on the UI thread.
        let want_cpu = false;

        let plan_acquire_started = Instant::now();
        let edits = std::mem::take(&mut self.pending_plan_edits);
        let plan_update = if edits.is_empty() {
            self.terrain_plan_cache
                .acquire(&preview_stack, &self.session.document.masks)
                .map(|_| self.pending_plan_invalidation.clone().unwrap_or_default())
        } else {
            self.terrain_plan_cache
                .update(&preview_stack, &self.session.document.masks, &edits)
        };
        let plan_invalidation = match plan_update {
            Ok(invalidation) => {
                self.pending_plan_invalidation = Some(invalidation.clone());
                invalidation
            }
            Err(diagnostics) => {
                log::debug!(
                    target: "terra_app::evaluation",
                    "compiled terrain plan requires CPU fallback ({}); {operation_context}",
                    diagnostics
                        .first()
                        .map(ToString::to_string)
                        .unwrap_or_else(|| "unknown plan diagnostic".to_string())
                );
                self.last_eval_gpu_supported = false;
                self.ui_state.profile.path = "async CPU";
                if !self.worker_refine_pending {
                    self.enqueue_async_eval(quality);
                }
                self.ui_state.profile.eval_us = t0.elapsed().as_micros() as u64;
                return;
            }
        };
        self.frame_trace.record(
            Instant::now(),
            FrameTraceEventKind::PlanAcquired,
            logical_frame_identity,
            self.logical_frames.active_phase(),
            Some(trace_id),
            Some(quality),
            Some(intent),
            Some(plan_acquire_started.elapsed()),
        );
        let plan_revision = self.terrain_plan_cache.structure_revision();
        let compiled_plan = self
            .terrain_plan_cache
            .current_plan()
            .expect("successful plan acquisition is current")
            .clone();

        let mut pyramid_candidate = None;
        let mut used_gpu = false;
        let mut eval_completed = false;
        if token == self.eval_token {
            if let (Some(engine), Some(renderer), Some(gpu)) = (
                self.gpu_engine.as_mut(),
                self.renderer.as_mut(),
                self.gpu.as_ref(),
            ) {
                if intent == GpuEvaluationIntent::InteractiveLocal {
                    if let Some(region) = self.pending_gpu_dirty_region {
                        engine.set_dirty_rect(Some(uv_to_texel_rect(
                            region,
                            metrics.width,
                            metrics.height,
                        )));
                    }
                } else {
                    // A suffix completion is globally coupled even though the edit that
                    // triggered it was bounded. Do not leak the prior local rect into it.
                    engine.set_dirty_rect(None);
                }
                let gpu_eval_started = Instant::now();
                engine.set_evaluation_trace_context(gpu_evaluation_trace_context(
                    logical_frame_identity,
                    token,
                    trace_id,
                ));
                match engine.evaluate_compiled_with_intent(
                    &gpu.device,
                    &gpu.queue,
                    &preview_stack,
                    &self.session.document.masks,
                    &compiled_plan,
                    plan_revision,
                    &plan_invalidation,
                    metrics,
                    quality,
                    want_cpu,
                    intent,
                ) {
                    Ok(result) => {
                        self.frame_trace.record_evaluation_submission(
                            Instant::now(),
                            logical_frame_identity,
                            self.logical_frames.active_phase(),
                            trace_id,
                            quality,
                            intent,
                            gpu_eval_started.elapsed(),
                            engine.last_eval_stats(),
                        );
                        if let Some(output) = result.output_identity {
                            self.frame_trace.record_evaluation_output(
                                Instant::now(),
                                logical_frame_identity,
                                self.logical_frames.active_phase(),
                                trace_id,
                                output,
                            );
                        }
                        if token != self.eval_token {
                            // Stale generation â€” discard.
                            self.frame_trace.record_candidate_refusal(
                                Instant::now(),
                                logical_frame_identity,
                                self.logical_frames.active_phase(),
                                trace_id,
                                quality,
                                intent,
                                result.output_identity,
                                terra_render::TerrainPresentationDecisionCode::RefusedStaleGeneration,
                            );
                        } else {
                            let dx = metrics.dx();
                            let dz = metrics.dz();
                            // Snapshot dirty tiles before take clears the scheduler.
                            let dirty_ids: Vec<(u32, u32)> =
                                engine.dirty_tiles().iter().map(|t| (t.tx, t.tz)).collect();
                            self.ui_state.dirty_tile_ids = dirty_ids;
                            self.ui_state.dirty_tile_grid = (metrics.tiles_x(), metrics.tiles_z());
                            // Only skip present when evaluate failed to seed (no filter work ran).
                            let skip_present = !result.did_eval;
                            if !skip_present {
                                let region = engine.take_dirty_region(1);
                                // Full-field updates bind the engine texture directly (WC path).
                                // Partial rects still copy through the double-buffer.
                                let full_field = region.is_none_or(|r| {
                                    r.x == 0
                                        && r.y == 0
                                        && r.w == result.width
                                        && r.h == result.height
                                });
                                let presentation_record = if full_field {
                                    let geom = terra_render::HeightPresentGeom {
                                        width: result.width,
                                        height: result.height,
                                        world_size: result.world_size,
                                        height_range: result.height_range,
                                        dx,
                                        dz,
                                    };
                                    result.output_identity.map(|output| {
                                        renderer.present_gpu_height_shared_traced(
                                            engine.output_texture(),
                                            engine.output_texture_view(),
                                            geom,
                                            None,
                                            output,
                                            terra_render::TerrainPresentationExpectations {
                                                plan_revision: plan_revision.get(),
                                                generation: token,
                                                extent: (result.width, result.height),
                                            },
                                        )
                                    })
                                } else if let Some(region) = region {
                                    let geom = terra_render::HeightPresentGeom {
                                        width: result.width,
                                        height: result.height,
                                        world_size: result.world_size,
                                        height_range: result.height_range,
                                        dx,
                                        dz,
                                    };
                                    result.output_identity.map(|output| {
                                        renderer.present_gpu_height_region_traced(
                                            engine.output_texture(),
                                            geom,
                                            Some(region),
                                            output,
                                            terra_render::TerrainPresentationExpectations {
                                                plan_revision: plan_revision.get(),
                                                generation: token,
                                                extent: (result.width, result.height),
                                            },
                                        )
                                    })
                                } else {
                                    None
                                };
                                if let Some(record) = presentation_record {
                                    self.frame_trace.record_presentation(
                                        Instant::now(),
                                        logical_frame_identity,
                                        self.logical_frames.active_phase(),
                                        trace_id,
                                        record,
                                    );
                                }
                                self.ui_state.profile.upload_us = renderer.last_upload_us;
                                self.ui_state.profile.terrain_grid_size =
                                    renderer.last_grid_resolution;
                                // GPU present owns the viewport — don't let a stale CPU
                                // upload from a prior job clobber it on the next redraw.
                                self.needs_height_upload = false;
                                self.last_complete_generation =
                                    super::logical_frame::EditGeneration::new(token);
                                self.last_accepted_evaluation_id = trace_id.get();
                                self.frame_trace.record(
                                    Instant::now(),
                                    FrameTraceEventKind::CandidateAccepted,
                                    logical_frame_identity,
                                    self.logical_frames.active_phase(),
                                    Some(trace_id),
                                    Some(quality),
                                    Some(intent),
                                    Some(t0.elapsed()),
                                );
                            } else {
                                self.frame_trace.record_candidate_refusal(
                                    Instant::now(),
                                    logical_frame_identity,
                                    self.logical_frames.active_phase(),
                                    trace_id,
                                    quality,
                                    intent,
                                    result.output_identity,
                                    terra_render::TerrainPresentationDecisionCode::RefusedNoOutput,
                                );
                                let _ = engine.take_dirty_region(1);
                            }
                            self.ui_state.profile.tex_w = result.width;
                            self.ui_state.profile.tex_h = result.height;
                            self.ui_state.profile.tiles_x = metrics.tiles_x();
                            self.ui_state.profile.tiles_z = metrics.tiles_z();
                            self.ui_state.quality = quality;
                            self.ui_state.profile.quality = match quality {
                                PreviewQuality::Draft => "Draft (fast)",
                                PreviewQuality::Medium => "Medium",
                                PreviewQuality::Full => "Final (viewport)",
                                PreviewQuality::Export => "Export quality",
                            };
                            self.ui_state.profile.gpu_fallback = result.cpu_fallback.clone();
                            if let Some(output) = result.output_identity {
                                pyramid_candidate = Some((output, result.width, result.height));
                            }
                            self.ui_state.profile.first_visible_preview_us =
                                t0.elapsed().as_micros() as u64;
                            if result.fully_gpu
                                && result.cpu_fallback.is_none()
                                && !result.freshness.is_deferred()
                            {
                                self.ui_state.profile.settled_authoritative_us =
                                    t0.elapsed().as_micros() as u64;
                            }

                            // Interactive path: GPU present is authoritative for the frame.
                            // Never sync-evaluate CPU on the UI thread — that hangs the app
                            // when stacks include unsupported layers (painted masks, SPE, …).
                            let full_field_deferred = result.freshness.is_deferred();
                            let needs_cpu_suffix = result.resume_cpu_from.is_some();
                            let completing_deferred_suffix = if full_field_deferred {
                                None
                            } else {
                                self.deferred_full_field.as_ref().map(|pending| {
                                    (pending.layer_name.clone(), pending.deferred_layers)
                                })
                            };
                            // `fully_gpu` also encodes whether a globally coupled
                            // suffix is already settled. A deferred suffix remains
                            // GPU-capable, so keep the resident Full-quality local
                            // path armed for rapid follow-up dabs.
                            self.last_eval_gpu_supported =
                                result.resume_cpu_from.is_none() && result.cpu_fallback.is_none();
                            if !full_field_deferred && result.cpu_fallback.is_none() {
                                self.pending_plan_invalidation = None;
                            }
                            // want_cpu=false, so the GPU engine never returns a CPU prefix
                            // (`result.cpu` is always None) and the UI thread never runs a CPU
                            // stack eval — every hybrid resume routes to the async worker below.
                            debug_assert!(
                                result.cpu.is_none(),
                                "interactive eval must not carry a CPU prefix (want_cpu=false)"
                            );
                            if result.resume_cpu_from == Some(0)
                                && !result.fully_gpu
                                && !result.did_eval
                            {
                                // Seed failed — keep last-good, refine async at current quality.
                                self.ui_state.profile.path = "GPU→async CPU";
                                used_gpu = true;
                                eval_completed = true;
                                if !self.worker_refine_pending {
                                    self.enqueue_async_eval(quality);
                                }
                            } else {
                                self.ui_state.profile.path = if full_field_deferred {
                                    "GPU (FullField deferred)"
                                } else if needs_cpu_suffix {
                                    "GPU*"
                                } else {
                                    "GPU"
                                };
                                if let Some(hf) = result.cpu {
                                    self.scheduler.last_good =
                                        Some(std::sync::Arc::new(hf.clone()));
                                    self.last_height = Some(hf);
                                    self.preview_dirty = true;
                                }
                                used_gpu = true;
                                eval_completed = true;
                                if full_field_deferred {
                                    if let GpuPreviewFreshness::Deferred {
                                        from_layer,
                                        deferred_layers,
                                        ..
                                    } = result.freshness
                                    {
                                        let layer_name = preview_stack
                                            .find(from_layer)
                                            .map(|layer| layer.common.name.clone())
                                            .unwrap_or_else(|| "global layer".into());
                                        self.deferred_full_field = Some(DeferredFullField {
                                            generation: token,
                                            layer_name: layer_name.clone(),
                                            deferred_layers,
                                            settle_at: None,
                                        });
                                        self.ui_state.terrain_preview_freshness =
                                            TerrainPreviewFreshness::Deferred {
                                                layer_name,
                                                deferred_layers,
                                                settling: false,
                                            };
                                    }
                                    // Local generators are live while an expensive
                                    // full-field suffix waits for mouse-up refinement.
                                    // This is not a CPU fallback.
                                    self.ui_state.refining = true;
                                    self.ui_state.build_progress =
                                        Some(quality_stage_progress(quality).max(0.15));
                                } else if needs_cpu_suffix {
                                    self.deferred_full_field = None;
                                    if let Some((layer_name, _)) = completing_deferred_suffix {
                                        self.ui_state.terrain_preview_freshness =
                                            TerrainPreviewFreshness::RefiningSuffix {
                                                layer_name,
                                                quality,
                                            };
                                    }
                                    // Unsupported layers need CPU bake at *this* quality
                                    // (not Draft), then lifecycle advances Draft→Medium→Full.
                                    self.ui_state.profile.path = "GPU→async CPU";
                                    self.ui_state.refining = true;
                                    self.ui_state.build_progress =
                                        Some(quality_stage_progress(quality).max(0.15));
                                    if !self.worker_refine_pending {
                                        self.enqueue_async_eval(quality);
                                    }
                                } else {
                                    self.deferred_full_field = None;
                                    self.ui_state.terrain_preview_freshness =
                                        TerrainPreviewFreshness::Current;
                                    // GPU finished this quality — keep climbing toward Full.
                                    self.ui_state.refining = quality.next_refine().is_some();
                                    if !self.ui_state.refining {
                                        self.ui_state.build_progress = None;
                                    }
                                }
                            }
                            if used_gpu && !eval_completed {
                                eval_completed = true;
                            }
                            if used_gpu && intent == GpuEvaluationIntent::InteractiveLocal {
                                // A current-token GPU result consumed this accumulated scope.
                                // A later edit has a newer token and therefore cannot reach here.
                                self.pending_gpu_dirty_region = None;
                            }
                        }
                    }
                    Err(error) => {
                        self.ui_state.profile.gpu_fallback = None;
                        match &error {
                            GpuError::RequiresCpu(reason) => log::debug!(
                                target: "terra_app::evaluation",
                                "GPU path requires CPU fallback ({:?}: {}); {operation_context}",
                                reason.code,
                                reason.user_message()
                            ),
                            GpuError::Wgpu(_) => log::error!(
                                target: "terra_app::evaluation",
                                "GPU evaluation failed: {error}; {operation_context}"
                            ),
                            GpuError::SourceAsset(_) => log::warn!(
                                target: "terra_app::evaluation",
                                "GPU source asset failed: {error}; {operation_context}"
                            ),
                            GpuError::StalePlan { .. } => log::debug!(
                                target: "terra_app::evaluation",
                                "discarding stale GPU terrain plan: {error}; {operation_context}"
                            ),
                        }
                        // GPU path failed — async CPU, keep last-good on screen.
                        self.last_eval_gpu_supported = false;
                        self.ui_state.profile.path = "async CPU";
                        if !self.worker_refine_pending {
                            self.enqueue_async_eval(quality);
                        }
                        used_gpu = true; // skip sync fallback below
                        eval_completed = true;
                    }
                }
            }
        }

        if let Some((output, width, height)) = pyramid_candidate {
            self.materialize_gpu_pyramid(output, width, height);
        }

        if !used_gpu {
            // Last resort when no GPU engine: still avoid blocking if a worker exists.
            if !self.worker_refine_pending {
                self.enqueue_async_eval(quality);
                eval_completed = true;
                self.ui_state.profile.path = "async CPU";
            }
        }

        let elapsed = t0.elapsed().as_micros() as u64;
        self.ui_state.profile.eval_us = elapsed;
        self.ui_state.profile.plan = self.terrain_plan_cache.stats().snapshot();
        self.ui_state.profile.gpu = self
            .gpu_engine
            .as_ref()
            .map(|engine| engine.last_eval_stats())
            .unwrap_or_default();
        self.ui_state.profile.cpu_worker = self.eval_worker.stats();
        if eval_completed {
            self.ui_state.quality = quality;
            self.ui_state.build_progress = Some(quality_stage_progress(quality));
            self.ui_state.draft_displayed =
                matches!(quality, PreviewQuality::Draft | PreviewQuality::Medium);
            if matches!(quality, PreviewQuality::Full | PreviewQuality::Export) {
                self.ui_state.draft_displayed = false;
            }
        }
    }

    /// When a climate overlay is selected and aux is missing, bake from the
    /// current height + Biomes layer params (no full stack re-eval / GPU readback).
    pub(crate) fn ensure_climate_overlay_aux(&mut self) {
        self.ensure_climate_aux(false);
    }

    /// Bake climate aux for lit 3D (snow/temp/rain) whenever a Biomes layer is present.
    pub(crate) fn ensure_climate_for_lit(&mut self) {
        self.ensure_climate_aux(true);
    }

    pub(crate) fn ensure_climate_aux(&mut self, for_lit: bool) {
        if !for_lit {
            let key = match self.ui_state.preview_mode {
                Preview2dMode::Temperature => "temperature",
                Preview2dMode::Rainfall => "rainfall",
                Preview2dMode::Snow => "snow",
                Preview2dMode::SoilMoisture => "soil_moisture",
                Preview2dMode::Biome => "biomes",
                _ => return,
            };
            if self.scheduler.last_aux.contains_key(key) {
                return;
            }
        } else if self.scheduler.last_aux.contains_key("snow")
            && self.scheduler.last_aux.contains_key("temperature")
            && self.scheduler.last_aux.contains_key("rainfall")
        {
            return;
        }
        let Some(hf) = self.last_height.as_ref() else {
            return;
        };
        let biomes_params = self
            .session
            .document
            .stack
            .flatten_layers()
            .into_iter()
            .find_map(|layer| match &layer.kind {
                LayerKind::Biomes(p) if layer.common.enabled && p.use_climate => Some(p.clone()),
                _ => None,
            });
        let Some(params) = biomes_params else {
            return;
        };
        let wetness = self.scheduler.last_aux.get("wetness");
        let sediment = self.scheduler.last_aux.get("sediment");
        let maps = terra_core::surface::bake_biomes_climate(hf, &params, wetness, sediment);
        self.scheduler
            .last_aux
            .insert("temperature".into(), maps.temperature);
        self.scheduler
            .last_aux
            .insert("rainfall".into(), maps.rainfall);
        self.scheduler
            .last_aux
            .insert("humidity".into(), maps.humidity);
        self.scheduler
            .last_aux
            .insert("aridity".into(), maps.aridity);
        self.scheduler.last_aux.insert("snow".into(), maps.snow);
        self.scheduler
            .last_aux
            .insert("soil_moisture".into(), maps.soil_moisture);
        self.scheduler
            .last_aux
            .insert("wind_exposure".into(), maps.wind_exposure);
        self.scheduler.last_aux.insert("biomes".into(), maps.biomes);
        self.preview_dirty = true;
        if for_lit {
            // Lit path will upload on next height present; don't force re-upload loop.
        } else {
            self.needs_height_upload = true;
        }
    }

    pub(crate) fn refresh_2d_preview(&mut self) {
        if self.ui_state.preview_mode != self.last_preview_mode {
            self.last_preview_mode = self.ui_state.preview_mode;
            self.preview_dirty = true;
        }
        if self.ui_state.force_preview_refresh {
            self.ui_state.force_preview_refresh = false;
            self.preview_dirty = true;
        }
        self.ensure_climate_overlay_aux();
        if !self.preview_dirty {
            return;
        }
        let Some(hf) = self.last_height.as_ref() else {
            self.ui_state.preview_rgba = None;
            self.preview_dirty = false;
            return;
        };
        let metrics = hf.metrics;
        // Opt-in Phase 2 geomorph debug overlays (command palette only).
        if let Some(field) = self.ui_state.geomorph_debug_field {
            let mask = terra_core::bake_debug_field(hf, field, None);
            let mut rgba = Vec::with_capacity(mask.data().len() * 4);
            for &value in mask.data() {
                let value = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
                rgba.extend_from_slice(&[value, value, value, 255]);
            }
            self.ui_state.preview_rgba = Some((metrics.width, metrics.height, rgba));
            self.preview_dirty = false;
            return;
        }
        // Coloured biome placement overlay â€” bypass grayscale path.
        if self.ui_state.preview_mode == Preview2dMode::Biome {
            if let Some(layer) = self.session.document.selected_placement_layer() {
                if !layer.channels.is_empty() {
                    let doc = &self.session.document;
                    let isolate = if layer.isolate_active {
                        doc.active_biome
                    } else {
                        None
                    };
                    let rgba = layer.bake_color_rgba(
                        metrics.width,
                        metrics.height,
                        &|gid| {
                            doc.biome_library
                                .by_group(gid)
                                .map(|d| d.color)
                                .or_else(|| doc.stack.find_group(gid).map(|g| g.preview_color))
                                .unwrap_or([0.45, 0.55, 0.4])
                        },
                        isolate,
                    );
                    self.ui_state.preview_rgba = Some((metrics.width, metrics.height, rgba));
                    self.preview_dirty = false;
                    return;
                }
            }
        }
        let values = match self.ui_state.preview_mode {
            Preview2dMode::Height => {
                let (min, max) = hf.min_max();
                let span = (max - min).max(1e-6);
                hf.to_dense()
                    .into_iter()
                    .map(|value| (value - min) / span)
                    .collect()
            }
            Preview2dMode::Slope => terra_core::analyze::slope_degrees(hf).data().to_vec(),
            Preview2dMode::Flow => {
                let Some(flow) = self.scheduler.last_aux.get("flow_accumulation") else {
                    self.ui_state.preview_rgba = None;
                    return;
                };
                let max = flow.data().iter().copied().fold(1.0f32, f32::max).ln_1p();
                flow.data()
                    .iter()
                    .copied()
                    .map(|value| value.max(0.0).ln_1p() / max)
                    .collect()
            }
            Preview2dMode::Mask | Preview2dMode::Masks => {
                let baked = bake_mask_assets(
                    &self.session.document.masks,
                    hf,
                    metrics,
                    &self.scheduler.last_aux,
                );
                self.ui_state
                    .selected_mask
                    .and_then(|id| baked.get(&id))
                    .or_else(|| baked.values().next())
                    .map(|mask| mask.data().to_vec())
                    .or_else(|| {
                        self.scheduler
                            .last_aux
                            .get("materials")
                            .map(|mask| mask.data().to_vec())
                    })
                    .unwrap_or_else(|| vec![0.0; (metrics.width * metrics.height) as usize])
            }
            Preview2dMode::Material | Preview2dMode::Biome | Preview2dMode::VegetationDensity => {
                let key = match self.ui_state.preview_mode {
                    Preview2dMode::Material => "materials",
                    Preview2dMode::Biome => "biomes",
                    Preview2dMode::VegetationDensity => "vegetation",
                    _ => unreachable!(),
                };
                self.scheduler
                    .last_aux
                    .get(key)
                    .map(|field| field.data().to_vec())
                    .unwrap_or_else(|| vec![0.0; (metrics.width * metrics.height) as usize])
            }
            Preview2dMode::Water
            | Preview2dMode::Sediment
            | Preview2dMode::Hardness
            | Preview2dMode::Erosion
            | Preview2dMode::Deposition
            | Preview2dMode::StreamOrder
            | Preview2dMode::SpeIncision
            | Preview2dMode::Temperature
            | Preview2dMode::Rainfall
            | Preview2dMode::Snow
            | Preview2dMode::SoilMoisture
            | Preview2dMode::Overhang => {
                let key = match self.ui_state.preview_mode {
                    Preview2dMode::Water => "wetness",
                    Preview2dMode::Sediment => "sediment",
                    Preview2dMode::Hardness => "hardness",
                    Preview2dMode::Erosion => "erosion",
                    Preview2dMode::Deposition => "deposition",
                    Preview2dMode::StreamOrder => "stream_order",
                    Preview2dMode::SpeIncision => "spe_incision",
                    Preview2dMode::Temperature => "temperature",
                    Preview2dMode::Rainfall => "rainfall",
                    Preview2dMode::Snow => "snow",
                    Preview2dMode::SoilMoisture => "soil_moisture",
                    Preview2dMode::Overhang => "overhang_mask",
                    _ => unreachable!(),
                };
                self.scheduler
                    .last_aux
                    .get(key)
                    .map(|field| {
                        let data = field.data();
                        let max_v = data.iter().copied().fold(1e-6f32, f32::max);
                        data.iter().map(|&v| (v / max_v).clamp(0.0, 1.0)).collect()
                    })
                    .unwrap_or_else(|| vec![0.0; (metrics.width * metrics.height) as usize])
            }
            // TODO: render colorized 2D previews for these diagnostics. The 3D viewport
            // continues to own their shader presentation; retain a height preview here.
            Preview2dMode::Lit
            | Preview2dMode::Unlit
            | Preview2dMode::Curvature
            | Preview2dMode::Convexity
            | Preview2dMode::Concavity
            | Preview2dMode::Normals
            | Preview2dMode::Wireframe
            | Preview2dMode::AmbientOcclusion => {
                let (min, max) = hf.min_max();
                let span = (max - min).max(1e-6);
                hf.to_dense()
                    .into_iter()
                    .map(|value| (value - min) / span)
                    .collect()
            }
        };
        let mut rgba = Vec::with_capacity(values.len() * 4);
        for value in values {
            let value = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
            rgba.extend_from_slice(&[value, value, value, 255]);
        }
        self.ui_state.preview_rgba = Some((metrics.width, metrics.height, rgba));
        self.preview_dirty = false;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Instant;

    use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
    use terra_core::layer::{
        FlatParams, Layer, LayerKind, LayerStack, SculptStrokeKind, StreamPowerParams,
    };
    use terra_core::quality::PreviewQuality;
    use terra_core::shape_history::{create_shape_layer, ShapeTool};
    use terra_core::test_fixtures::{untitled6_document, Untitled6Variant};
    use terra_core::tiling::UvRect;
    use terra_core::PyramidConfig;
    use terra_gpu::GpuTileAtlas;
    use terra_gpu_eval::{GpuEvaluationIntent, GpuTerrainEngine};
    use terra_render::{GpuContext, HeightPresentGeom, TerrainRenderer};

    use crate::app::frame_trace::{EvaluationTraceId, FrameTraceEventKind};
    use crate::app::logical_frame::{
        EditGeneration, FrameIdentity, FrameRequestReason, LogicalFrameId,
    };
    use crate::ui::PanelAction;

    use super::{
        gpu_evaluation_trace_context, terrain_tile_work_budget, uv_to_texel_rect,
        DeferredFullField, TerraApp,
    };

    fn flat(height: f32) -> Layer {
        Layer::new("Flat", LayerKind::Flat(FlatParams { height }))
    }

    fn rect(u: f32, v: f32, r: f32) -> UvRect {
        UvRect::from_center_radius(u, v, r)
    }

    #[test]
    fn gpu_dirty_uv_is_converted_at_each_evaluation_resolution() {
        let region = UvRect {
            min_u: 0.25,
            min_v: 0.5,
            max_u: 0.251,
            max_v: 0.502,
        };
        assert_eq!(uv_to_texel_rect(region, 512, 512), (128, 256, 1, 2));
        assert_eq!(uv_to_texel_rect(region, 4096, 4096), (1024, 2048, 5, 9));
        assert_eq!(uv_to_texel_rect(region, 512, 256), (128, 128, 1, 1));
    }

    #[test]
    fn invalid_gpu_dirty_uv_escalates_to_whole_field() {
        let region = UvRect {
            min_u: f32::NAN,
            min_v: 0.0,
            max_u: 1.0,
            max_v: 1.0,
        };
        assert_eq!(uv_to_texel_rect(region, 512, 256), (0, 0, 512, 256));
    }

    #[test]
    fn tile_work_budgets_follow_every_refinement_state() {
        use terra_core::EditorRefinementState::*;
        assert_eq!(terrain_tile_work_budget(Interactive, 256).max_items, 8);
        assert_eq!(terrain_tile_work_budget(Settling, 256).max_items, 20);
        assert_eq!(terrain_tile_work_budget(Refining, 256).max_items, 32);
        assert_eq!(terrain_tile_work_budget(Converged, 256).max_items, 4);
        assert_eq!(terrain_tile_work_budget(Export, 256).max_items, 128);
    }

    #[test]
    fn production_scheduler_choice_controls_first_uploaded_page() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let context = GpuContext {
            device: gpu.device.clone(),
            queue: gpu.queue.clone(),
            surface_format: wgpu::TextureFormat::Rgba8Unorm,
        };
        let metrics = HeightfieldMetrics {
            tile_size: 16,
            halo: 1,
            ..HeightfieldMetrics::new(32, 32, 320.0, 320.0)
        };
        let mut config = PyramidConfig::new(32, 320.0, 320.0);
        config.tile_size = 16;
        config.halo = 1;
        let mut app = TerraApp::default();
        app.session.document.metrics = metrics;
        app.terrain_runtime.reconfigure(config);
        app.last_height = Some(Heightfield::filled(metrics, 7.0));
        app.tile_atlas = Some(GpuTileAtlas::new(&context.device, 16, 1, 8).unwrap());
        app.gpu = Some(context);

        let level = app.streamed_level_for_last_height().unwrap().0;
        let stamp = terra_core::TerrainContentStamp {
            document_revision: app.eval_token,
            plan_revision: app.terrain_plan_cache.structure_revision().get(),
            output_revision: app.terrain_runtime.output_revision(),
            content_revision: app.eval_token,
        };
        let make_request = |tx, class, visible, error| terra_core::TerrainTileWorkRequest {
            key: terra_core::TerrainTileWorkKey {
                tile: terra_core::TerrainTileKey {
                    layer: None,
                    field: terra_core::FieldId::Height,
                    level,
                    tile: terra_core::TileId { tx, tz: 0 },
                },
                plan_revision: stamp.plan_revision,
                output_revision: stamp.output_revision,
            },
            content: stamp,
            source: terra_core::TerrainTileWorkSource::CpuHeight,
            class,
            visible,
            projected_error_px: error,
            distance_m: tx as f32,
            // Converged has a 1 ms budget, so exactly one request can dispatch.
            estimated_us: 1_000,
        };
        let fine = make_request(0, terra_core::TerrainDemandClass::Refinement, true, 100.0);
        let coarse = make_request(1, terra_core::TerrainDemandClass::CoarseCoverage, true, 1.0);
        let coarse_key = coarse.key.tile.clone();
        let fine_key = fine.key.tile.clone();
        app.terrain_tile_scheduler.reconcile(stamp, [fine, coarse]);

        assert_eq!(app.upload_pending_terrain_tiles(), 1);
        let atlas = app.tile_atlas.as_ref().unwrap();
        assert!(atlas.is_current(&coarse_key, stamp));
        assert!(!atlas.is_current(&fine_key, stamp));

        // A newer CPU result in the same document/output revision receives a
        // distinct content identity and must replace, not cache-skip, old pages.
        app.last_height = Some(Heightfield::filled(metrics, 9.0));
        app.queue_final_tile_uploads();
        let newer = app.terrain_tile_scheduler.live_content().unwrap();
        assert_ne!(newer.content_revision, stamp.content_revision);
        assert!(app.upload_pending_terrain_tiles() > 0);
        assert!(app
            .tile_atlas
            .as_ref()
            .unwrap()
            .is_current(&coarse_key, newer));
    }

    #[test]
    fn gpu_output_generation_is_independent_of_logical_frame_generation() {
        let logical_frame = FrameIdentity {
            id: LogicalFrameId::new(7),
            generation_at_start: EditGeneration::new(0),
            generation: EditGeneration::new(0),
        };
        let context =
            gpu_evaluation_trace_context(Some(logical_frame), 2, EvaluationTraceId::new(11));

        assert_eq!(context.frame_id, 7);
        assert_eq!(context.generation, 2);
        assert_eq!(context.evaluation_id, 11);
        assert_eq!(logical_frame.generation.get(), 0);
    }

    /// #169 project-entry regression: project initialization can advance the
    /// publication token while the logical frame that opened it still has the
    /// pre-entry generation. The first GPU output must use publication authority
    /// without losing the older frame generation used for trace correlation.
    #[test]
    fn first_project_entry_gpu_presentation_uses_evaluation_token() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let context = GpuContext {
            device: gpu.device.clone(),
            queue: gpu.queue.clone(),
            surface_format: wgpu::TextureFormat::Rgba8Unorm,
        };
        let resolution = 64;
        let mut stack = LayerStack::new();
        stack.push(flat(12.0));

        let mut app = TerraApp::default();
        app.session.document.metrics =
            HeightfieldMetrics::new(resolution, resolution, 640.0, 640.0);
        app.session.document.preview_resolution = resolution;
        app.session.document.stack = stack;
        app.renderer = Some(TerrainRenderer::new_headless(
            &context, resolution, resolution,
        ));
        app.gpu_engine = Some(GpuTerrainEngine::new(&context.device, resolution));
        app.gpu = Some(context);

        let frame_started = Instant::now();
        app.logical_frames.request(
            EditGeneration::new(app.eval_token),
            FrameRequestReason::UiActions,
        );
        let project_entry_frame = app
            .logical_frames
            .begin(frame_started, 0, 0)
            .expect("project-entry logical frame");
        assert_eq!(project_entry_frame.generation.get(), 0);

        app.request_rebuild();
        app.request_rebuild();
        let publication_generation = app.eval_token;
        assert_eq!(publication_generation, 2);
        assert_eq!(
            app.logical_frames
                .active_identity()
                .expect("active project-entry frame")
                .generation
                .get(),
            0
        );

        app.run_eval_step_with_intent(GpuEvaluationIntent::Complete);

        let output = app
            .gpu_engine
            .as_ref()
            .and_then(GpuTerrainEngine::last_output_identity)
            .expect("first project-entry GPU output identity");
        assert_eq!(output.generation, publication_generation);
        assert_eq!(output.frame_id, project_entry_frame.id.get());

        let output_event = app
            .frame_trace
            .events()
            .iter()
            .rev()
            .find(|event| event.kind == FrameTraceEventKind::EvaluationOutputSelected)
            .expect("evaluation output trace event");
        assert_eq!(output_event.generation.get(), 0);
        assert_eq!(
            output_event
                .output_identity
                .expect("traced output identity")
                .generation,
            publication_generation
        );
        assert_eq!(app.frame_trace.first_violation(), None);
        assert_eq!(
            app.renderer
                .as_ref()
                .and_then(TerrainRenderer::last_terrain_presentation_record)
                .expect("first GPU presentation record")
                .shadow_diagnostic,
            None
        );
    }

    /// #173 app-path acceptance: a complete GPU result is materialized and
    /// published without creating a dense CPU heightfield or waking the worker.
    /// The compact geometric-error metadata transfer drives demand but must not
    /// mutate atlas residency until the upload consumer runs.
    #[test]
    fn complete_gpu_output_streams_from_bounded_demand_without_height_readback() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let context = GpuContext {
            device: gpu.device.clone(),
            queue: gpu.queue.clone(),
            surface_format: wgpu::TextureFormat::Rgba8Unorm,
        };
        let resolution = 128;
        let mut config = PyramidConfig::new(resolution, 640.0, 640.0);
        config.tile_size = 16;
        config.halo = 1;

        let mut stack = LayerStack::new();
        stack.push(flat(19.0));
        let mut app = TerraApp::default();
        app.session.document.metrics =
            HeightfieldMetrics::new(resolution, resolution, 640.0, 640.0);
        app.session.document.preview_resolution = resolution;
        app.session.document.stack = stack;
        app.scheduler.quality = PreviewQuality::Full;
        app.terrain_runtime.reconfigure(config);
        app.renderer = Some(TerrainRenderer::new_headless(
            &context, resolution, resolution,
        ));
        app.gpu_engine = Some(GpuTerrainEngine::new(&context.device, resolution));
        app.tile_atlas =
            Some(GpuTileAtlas::new(&context.device, 16, 1, 32).expect("test tile atlas"));
        app.gpu = Some(context);

        let worker_before = app.eval_worker.stats();
        app.run_eval_step_with_intent(GpuEvaluationIntent::Complete);

        let pyramid = app.gpu_height_pyramid.as_ref().unwrap_or_else(|| {
            panic!(
                "accepted GPU output must own a pyramid; output={:?}, fallback={:?}, failure={:?}",
                app.gpu_engine
                    .as_ref()
                    .and_then(GpuTerrainEngine::last_output_identity),
                app.ui_state.profile.gpu_fallback,
                app.ui_state.evaluation_failure
            )
        });
        assert_eq!(
            pyramid
                .descriptor()
                .level(pyramid.source_level())
                .unwrap()
                .resolution,
            resolution
        );
        assert!(
            app.last_height.is_none(),
            "GPU path must not publish CPU height"
        );
        assert_eq!(app.eval_worker.stats(), worker_before);
        assert_eq!(
            app.gpu_engine
                .as_ref()
                .unwrap()
                .last_eval_stats()
                .readback_bytes,
            0
        );
        assert!(!app.terrain_tile_scheduler.is_empty());
        assert!(app
            .terrain_tile_scheduler
            .queued_requests()
            .all(|pending| matches!(
                pending.source,
                terra_core::TerrainTileWorkSource::GpuCompiledPlan
            )));

        let residency_before = app.tile_atlas.as_ref().unwrap().residency().stats();
        app.gpu.as_ref().unwrap().device.poll(wgpu::Maintain::Wait);
        assert!(app.refresh_terrain_demand());
        let demand = app.latest_terrain_demand.as_ref().expect("camera demand");
        assert!(!demand.tiles.is_empty());
        assert!(demand.tiles.len() <= app.tile_atlas.as_ref().unwrap().max_pages() as usize);
        assert!(demand
            .tiles
            .windows(2)
            .all(|pair| pair[0].key.level <= pair[1].key.level));

        // Exercise the production camera adapter with constructed measured-error
        // metadata: the same immutable pyramid demands deeper tiles when close.
        app.gpu_pyramid_planning_metadata
            .as_mut()
            .unwrap()
            .geometric_errors
            .iter_mut()
            .for_each(|error| *error = 10.0);
        {
            let camera = &mut app.renderer.as_mut().unwrap().camera;
            camera.target = glam::Vec3::new(320.0, 19.0, 320.0);
            camera.distance = 10_000.0;
            camera.yaw = 0.7;
            camera.pitch = 0.7;
        }
        app.latest_terrain_demand = None;
        app.terrain_demand_planner.reset();
        assert!(app.refresh_terrain_demand());
        let far_level = app
            .latest_terrain_demand
            .as_ref()
            .unwrap()
            .tiles
            .iter()
            .map(|demand| demand.key.level)
            .max()
            .unwrap();
        app.renderer.as_mut().unwrap().camera.distance = 500.0;
        assert!(app.refresh_terrain_demand());
        let near_level = app
            .latest_terrain_demand
            .as_ref()
            .unwrap()
            .tiles
            .iter()
            .map(|demand| demand.key.level)
            .max()
            .unwrap();
        assert!(near_level > far_level);
        assert_eq!(
            app.tile_atlas.as_ref().unwrap().residency().stats(),
            residency_before,
            "planning demand must not publish or mirror residency"
        );

        let mut uploaded = app.upload_pending_terrain_tiles();
        for _ in 0..4 {
            if uploaded > 0 {
                break;
            }
            app.gpu.as_ref().unwrap().device.poll(wgpu::Maintain::Wait);
            uploaded += app.upload_pending_terrain_tiles();
        }
        assert!(uploaded > 0);
        assert!(
            !app.terrain_tile_scheduler.is_empty(),
            "bounded upload should leave part of the camera demand pending"
        );
        assert!(
            !app.renderer.as_ref().unwrap().tile_stream_enabled(),
            "a partial GPU demand set must not replace the complete monolithic terrain"
        );
        let atlas = app.tile_atlas.as_ref().unwrap();
        let rows = atlas.read_page_table_blocking(
            &app.gpu.as_ref().unwrap().device,
            &app.gpu.as_ref().unwrap().queue,
        );
        let live_revision = app.terrain_runtime.output_revision();
        assert!(rows.iter().any(|row| {
            row.valid == 1
                && (u64::from(row.output_revision_lo) | (u64::from(row.output_revision_hi) << 32))
                    == live_revision
        }));
    }

    /// #150 app-path ratchet: the production-shaped document resolves the real
    /// shape target and applies the real panel action without CPU fallback.
    #[test]
    fn untitled6_mouse_down_preview_is_gpu_only_and_visible() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let context = GpuContext {
            device: gpu.device.clone(),
            queue: gpu.queue.clone(),
            surface_format: wgpu::TextureFormat::Rgba8Unorm,
        };
        let (document, ids) = untitled6_document(96, Untitled6Variant::ProductionTopology);
        let mut app = TerraApp::default();
        app.session.document = document;
        app.session.document.selected = Some(ids.base);
        app.scheduler.quality = PreviewQuality::Draft;
        app.renderer = Some(TerrainRenderer::new_headless(&context, 96, 96));
        app.gpu_engine = Some(GpuTerrainEngine::new(&context.device, 96));
        app.gpu = Some(context);
        app.run_eval_step_with_intent(GpuEvaluationIntent::Complete);
        assert_eq!(app.ui_state.profile.path, "GPU");
        assert!(app.last_eval_gpu_supported);
        assert!(app.ui_state.profile.gpu_fallback.is_none());
        assert!(app.ui_state.evaluation_failure.is_none());

        let plan_before = app.terrain_plan_cache.stats().snapshot();
        let worker_before = app.eval_worker.stats();
        let cpu_published_before = app.ui_state.profile.cpu_published;
        let target = app
            .ensure_shape_history_target(ShapeTool::Raise)
            .expect("resolve the selected Base sculpt target");
        assert_eq!(target, ids.base);

        for u in [0.47, 0.50, 0.53] {
            app.apply_actions(vec![PanelAction::PaintSculptStamp {
                layer: target,
                u,
                v: 0.5,
                radius: 0.04,
                strength: 5.0,
                stroke_kind: SculptStrokeKind::Raise,
                target_height: 0.0,
            }]);
            app.last_paint_uv = Some((u, 0.5));
            app.run_eval_step_with_intent(GpuEvaluationIntent::InteractiveLocal);

            assert_eq!(app.ui_state.profile.path, "GPU");
            assert!(app.last_eval_gpu_supported);
            assert!(app.ui_state.profile.gpu_fallback.is_none());
            assert!(app.ui_state.evaluation_failure.is_none());
            let stats = app.gpu_engine.as_ref().unwrap().last_eval_stats();
            assert_eq!(stats.readback_bytes, 0);
            assert_eq!(stats.operations_deferred, 0);
            assert!(stats.operations_dispatched > 0);
        }

        let plan_after = app.terrain_plan_cache.stats().snapshot();
        let worker_after = app.eval_worker.stats();
        assert_eq!(app.ui_state.profile.path, "GPU");
        assert_eq!(
            (app.ui_state.profile.tex_w, app.ui_state.profile.tex_h),
            (128, 128),
            "Draft quality clamps the 96-sample fixture to its 128-sample floor"
        );
        assert_eq!(plan_after.plan_compiles, plan_before.plan_compiles);
        assert_eq!(
            plan_after.authored_tree_walks,
            plan_before.authored_tree_walks
        );
        assert_eq!(plan_after.dependency_builds, plan_before.dependency_builds);
        assert_eq!(worker_after, worker_before);
        assert_eq!(app.ui_state.profile.cpu_published, cpu_published_before);

        // #151 renderer seam: after the shared cold present and consecutive warm
        // local dabs, the accumulated regional frame must exactly match a full
        // presentation of the engine's complete current texture. The readbacks
        // below belong only to the test oracle; engine stats above/below must stay 0.
        let regional_target = gpu.target(96, 96, wgpu::TextureFormat::Rgba8Unorm);
        app.renderer
            .as_mut()
            .expect("headless renderer")
            .render_to_view(&regional_target.view, 96, 96);
        let regional_frame = gpu.read_rgba8(&regional_target);

        let geom = {
            let heights = &app.renderer.as_ref().expect("headless renderer").heights;
            HeightPresentGeom {
                width: heights.tex_size.0,
                height: heights.tex_size.1,
                world_size: heights.world_size,
                height_range: heights.height_range,
                dx: heights.world_size.0 / heights.tex_size.0.max(1) as f32,
                dz: heights.world_size.1 / heights.tex_size.1.max(1) as f32,
            }
        };
        let oracle_context = GpuContext {
            device: gpu.device.clone(),
            queue: gpu.queue.clone(),
            surface_format: wgpu::TextureFormat::Rgba8Unorm,
        };
        let mut oracle = TerrainRenderer::new_headless(&oracle_context, 96, 96);
        let engine = app.gpu_engine.as_ref().expect("GPU engine");
        oracle.present_gpu_height_shared(
            engine.output_texture(),
            engine.output_texture_view(),
            geom,
            None,
        );
        oracle.camera = app
            .renderer
            .as_ref()
            .expect("headless renderer")
            .camera
            .clone();
        let full_target = gpu.target(96, 96, wgpu::TextureFormat::Rgba8Unorm);
        oracle.render_to_view(&full_target.view, 96, 96);
        let full_frame = gpu.read_rgba8(&full_target);

        let mut differing_pixels = 0u32;
        let mut first_mismatch = None;
        for y in 0..regional_frame.height() {
            for x in 0..regional_frame.width() {
                let regional = regional_frame.get(x, y);
                let full = full_frame.get(x, y);
                if regional != full {
                    differing_pixels += 1;
                    if first_mismatch.is_none() {
                        first_mismatch = Some((x, y, regional, full));
                    }
                }
            }
        }
        assert_eq!(
            differing_pixels, 0,
            "warm regional frame differs from full engine-texture present in \
             {differing_pixels} pixel(s); first mismatch {first_mismatch:?}"
        );
        assert_eq!(
            app.gpu_engine
                .as_ref()
                .unwrap()
                .last_eval_stats()
                .readback_bytes,
            0,
            "renderer oracle must not alter the engine's no-readback result"
        );
        assert_eq!(app.eval_worker.stats(), worker_before);
        assert_eq!(app.ui_state.profile.cpu_published, cpu_published_before);

        for quality in [PreviewQuality::Medium, PreviewQuality::Full] {
            app.scheduler.quality = quality;
            app.run_eval_step_with_intent(GpuEvaluationIntent::Complete);
            assert_eq!(app.ui_state.profile.path, "GPU", "{quality:?}");
            assert!(app.last_eval_gpu_supported, "{quality:?}");
            assert!(app.ui_state.profile.gpu_fallback.is_none(), "{quality:?}");
            assert!(app.ui_state.evaluation_failure.is_none(), "{quality:?}");
            assert_eq!(app.eval_worker.stats(), worker_before, "{quality:?}");
            assert_eq!(
                app.ui_state.profile.cpu_published, cpu_published_before,
                "{quality:?}"
            );
            let stats = app.gpu_engine.as_ref().unwrap().last_eval_stats();
            assert_eq!(stats.readback_bytes, 0, "{quality:?}");
            assert_eq!(stats.operations_deferred, 0, "{quality:?}");
        }

        app.apply_actions(vec![PanelAction::PaintSculptStamp {
            layer: target,
            u: 0.56,
            v: 0.5,
            radius: 0.04,
            strength: 1.0,
            stroke_kind: SculptStrokeKind::Raise,
            target_height: 0.0,
        }]);
        assert!(app.force_draft, "the ordinary edit path requests Draft");
        app.run_eval_step_with_intent(GpuEvaluationIntent::InteractiveLocal);
        assert_eq!(
            app.scheduler.quality,
            PreviewQuality::Full,
            "a bounded Base edit keeps an already-valid Full realization resident"
        );
        let stats = app.gpu_engine.as_ref().unwrap().last_eval_stats();
        assert!(!stats.cold_execution);
        assert!(stats.dirty_texels < u64::from(stats.resolution).pow(2));
        assert_eq!(stats.readback_bytes, 0);
    }

    /// The normal Raise workflow targets a non-destructive Shape Layer, not the
    /// Foundation raster. It must retain the Full texture across repeated complete
    /// publications and across 8/16/32-stroke GPU buffer growth boundaries.
    #[test]
    fn rapid_raise_shape_layer_strokes_stay_full_across_publications() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let context = GpuContext {
            device: gpu.device.clone(),
            queue: gpu.queue.clone(),
            surface_format: wgpu::TextureFormat::Rgba8Unorm,
        };
        let resolution = 128;
        let shape = create_shape_layer("Raise strokes");
        let shape_id = shape.id();
        let mut stack = LayerStack::new();
        stack.push(shape);

        let mut app = TerraApp::default();
        app.session.document.metrics =
            HeightfieldMetrics::new(resolution, resolution, 1280.0, 1280.0);
        app.session.document.preview_resolution = resolution;
        app.session.document.stack = stack;
        app.session.document.selected = Some(shape_id);
        app.scheduler.quality = PreviewQuality::Full;
        app.ui_state.quality = PreviewQuality::Full;
        app.renderer = Some(TerrainRenderer::new_headless(
            &context, resolution, resolution,
        ));
        app.gpu_engine = Some(GpuTerrainEngine::new(&context.device, resolution));
        app.gpu = Some(context);

        app.run_eval_step_with_intent(GpuEvaluationIntent::Complete);
        assert_eq!(app.scheduler.quality, PreviewQuality::Full);
        assert!(
            app.last_eval_gpu_supported,
            "initial Shape Layer stack did not stay GPU-capable: path={}, fallback={:?}, failure={:?}",
            app.ui_state.profile.path,
            app.ui_state.profile.gpu_fallback,
            app.ui_state.evaluation_failure
        );

        for stroke in 0..72 {
            let frame_started = Instant::now();
            app.logical_frames.request(
                EditGeneration::new(app.eval_token),
                FrameRequestReason::Input,
            );
            app.logical_frames
                .begin(frame_started, 3, 1)
                .expect("stroke logical frame");
            app.frame_trace
                .note_input_receipt(frame_started, EditGeneration::new(app.eval_token));
            app.logical_frames
                .transition(super::super::logical_frame::FramePhase::ApplicationUpdate);
            app.mouse_pressed = Some(winit::event::MouseButton::Left);
            let u = 0.35 + (stroke % 10) as f32 * 0.03;
            let v = 0.40 + (stroke / 10) as f32 * 0.05;
            app.last_paint_uv = None;
            app.apply_actions(vec![PanelAction::PaintSculptStamp {
                layer: shape_id,
                u,
                v,
                radius: 0.04,
                strength: 5.0,
                stroke_kind: SculptStrokeKind::Raise,
                target_height: 0.0,
            }]);
            app.logical_frames
                .update_generation(EditGeneration::new(app.eval_token));
            app.logical_frames
                .transition(super::super::logical_frame::FramePhase::RequiredInteractiveWork);
            app.run_eval_step_with_intent(GpuEvaluationIntent::InteractiveLocal);

            assert_eq!(
                app.frame_trace.first_violation(),
                None,
                "stroke {} emitted a transition violation: {:#?}",
                stroke + 1,
                app.renderer
                    .as_ref()
                    .and_then(TerrainRenderer::last_terrain_presentation_record)
            );

            assert_eq!(
                app.scheduler.quality,
                PreviewQuality::Full,
                "rapid stroke {} demoted the resident texture",
                stroke + 1
            );
            assert_eq!(
                (app.ui_state.profile.tex_w, app.ui_state.profile.tex_h),
                (resolution, resolution),
                "rapid stroke {} changed evaluation resolution",
                stroke + 1
            );
            assert!(
                app.last_eval_gpu_supported,
                "Shape Layer edits must remain on the GPU"
            );
            assert!(app.ui_state.profile.gpu_fallback.is_none());
            app.logical_frames
                .transition(super::super::logical_frame::FramePhase::PresentationRequest);
            app.mouse_pressed = None;
            app.frame_trace
                .note_release(Instant::now(), EditGeneration::new(app.eval_token));
            app.logical_frames.complete(Instant::now());

            // A background publication switches the renderer back to the engine's
            // shared Full texture. The next regional stroke must retain Full and
            // establish a renderer-local baseline without a Draft replacement.
            if matches!(stroke, 7 | 15 | 31 | 63 | 71) {
                app.logical_frames.request(
                    EditGeneration::new(app.eval_token),
                    FrameRequestReason::OptionalRefinement,
                );
                app.logical_frames
                    .begin(Instant::now(), 0, 0)
                    .expect("refinement publication frame");
                app.logical_frames
                    .transition(super::super::logical_frame::FramePhase::OptionalRefinement);
                app.run_eval_step_with_intent(GpuEvaluationIntent::Complete);
                assert_eq!(
                    app.scheduler.quality,
                    PreviewQuality::Full,
                    "complete publication after stroke {} demoted the resident texture",
                    stroke + 1
                );
                assert_eq!(
                    (app.ui_state.profile.tex_w, app.ui_state.profile.tex_h),
                    (resolution, resolution)
                );
                assert!(app.last_eval_gpu_supported);
                app.logical_frames.complete(Instant::now());
            }
        }
        assert_eq!(
            app.frame_trace.first_violation(),
            None,
            "the warm Shape Layer path must not emit identity/baseline false positives"
        );
    }

    #[test]
    fn full_field_suffix_deadline_starts_once_after_gesture_end() {
        let start = std::time::Instant::now();
        let mut pending = DeferredFullField {
            generation: 7,
            layer_name: "Rivers".into(),
            deferred_layers: 3,
            settle_at: None,
        };
        pending.hold_during_gesture();
        assert_eq!(pending.settle_at, None);

        let released = start + std::time::Duration::from_millis(20);
        assert!(pending.arm_after_gesture(released));
        let deadline = released + std::time::Duration::from_millis(75);
        assert_eq!(pending.settle_at, Some(deadline));
        assert!(!pending.arm_after_gesture(released + std::time::Duration::from_millis(40)));
        assert_eq!(
            pending.settle_at,
            Some(deadline),
            "idle ticks must not debounce"
        );
        assert!(!pending.ready(7, deadline - std::time::Duration::from_millis(1)));
        assert!(pending.ready(7, deadline));
        assert!(
            !pending.ready(8, deadline),
            "stale generations never publish"
        );
    }

    /// #100 phase 4 (loss-proof transport): `enqueue_async_eval` must *copy* the
    /// dirty accumulators, not drain them — a job that is stale-skipped on dequeue
    /// runs no body and applies no marks, so a drained scope would be lost. The
    /// accumulators are cleared only when a fresh result is consumed.
    #[test]
    fn enqueue_copies_dirty_accumulators_instead_of_draining() {
        let mut app = TerraApp::default();
        let layer = flat(10.0);
        let id = layer.id();
        app.session.document.stack = LayerStack::new();
        app.session.document.stack.push(layer);

        app.worker_mark_all_dirty = false;
        app.worker_dirty_from = Some(id);
        let region = rect(0.4, 0.4, 0.05);
        app.worker_dirty_region = Some(region);

        app.enqueue_async_eval(PreviewQuality::Draft);

        assert!(app.worker_refine_pending, "the job was submitted");
        assert_eq!(
            app.worker_dirty_from,
            Some(id),
            "dirty_from must survive submit (copied, not drained)"
        );
        assert_eq!(
            app.worker_dirty_region,
            Some(region),
            "the UV scope must survive submit for re-carry on a dequeue-skip"
        );
        assert!(!app.worker_mark_all_dirty);
    }

    /// A bounded sculpt footprint seeds `worker_dirty_region`; a second bounded edit
    /// unions in; a later footprint-less edit escalates it to whole-field (`None`),
    /// and — the correctness-critical part — a bounded edit *after* the escalation
    /// does not narrow it back.
    #[test]
    fn track_worker_dirty_from_unions_and_escalates_region() {
        let mut app = TerraApp::default();
        let layer = flat(1.0);
        let id = layer.id();
        let mut stack = LayerStack::new();
        stack.push(layer);

        app.worker_mark_all_dirty = false;
        app.worker_dirty_from = None;
        app.worker_dirty_region = None;

        let a = rect(0.2, 0.2, 0.03);
        let b = rect(0.8, 0.8, 0.03);
        app.track_worker_dirty_from(&stack, id, Some(a));
        assert_eq!(app.worker_dirty_from, Some(id));
        assert_eq!(app.worker_dirty_region, Some(a), "first bounded edit seeds");

        app.track_worker_dirty_from(&stack, id, Some(b));
        assert_eq!(
            app.worker_dirty_region,
            Some(a.union(b)),
            "a second bounded edit unions the scope"
        );

        app.track_worker_dirty_from(&stack, id, None);
        assert_eq!(
            app.worker_dirty_region, None,
            "a footprint-less edit escalates the scope to whole-field"
        );

        app.track_worker_dirty_from(&stack, id, Some(a));
        assert_eq!(
            app.worker_dirty_region, None,
            "a bounded edit after escalation must not narrow the whole-field scope"
        );
    }

    /// Escalation table: every footprint-less trigger drops `worker_dirty_region` to
    /// `None`, matching pre-#111 whole-field behavior.
    #[test]
    fn footprintless_triggers_escalate_region_to_none() {
        // Param / structural suffix edit via mark_dirty_from.
        {
            let mut app = TerraApp::default();
            let layer = flat(5.0);
            let id = layer.id();
            app.session.document.stack = LayerStack::new();
            app.session.document.stack.push(layer);
            app.worker_mark_all_dirty = false;
            app.worker_dirty_from = Some(id);
            app.worker_dirty_region = Some(rect(0.5, 0.5, 0.02));
            app.mark_dirty_from(id);
            assert_eq!(app.worker_dirty_region, None, "param edit escalates");
        }
        // Whole-field invalidation (mask paint commit / undo / resolution change).
        {
            let mut app = TerraApp::default();
            app.session.document.stack = LayerStack::new();
            app.session.document.stack.push(flat(5.0));
            app.worker_mark_all_dirty = false;
            app.worker_dirty_region = Some(rect(0.5, 0.5, 0.02));
            app.mark_all_layers_dirty();
            assert!(app.worker_mark_all_dirty);
            assert_eq!(app.worker_dirty_region, None, "mark_all clears the scope");
        }
        // Stage (re)build (scenario / World Rule).
        {
            let mut app = TerraApp::default();
            let layer = flat(5.0);
            let id = layer.id();
            app.session.document.stack = LayerStack::new();
            app.session.document.stack.push(layer);
            app.worker_mark_all_dirty = false;
            app.worker_dirty_from = Some(id);
            app.worker_dirty_region = Some(rect(0.5, 0.5, 0.02));
            let preview = app.session.document.preview_eval_stack();
            app.track_worker_dirty_from_eval_stage(&preview, terra_core::EvalStage::Blueprint);
            assert_eq!(app.worker_dirty_region, None, "stage dirty escalates");
        }
    }

    /// Ladder policy (a): the straight-to-Full gate promotes a bounded small-scope
    /// submit to Full once the worker cache is Full-res, and syncs the scheduler/UI
    /// quality so the refine loop settles at Full; a cold cache declines.
    #[test]
    fn straight_to_full_gate_promotes_only_over_a_full_cache() {
        let mut app = TerraApp::default();
        let layer = flat(3.0);
        let id = layer.id();
        app.session.document.stack = LayerStack::new();
        app.session.document.stack.push(layer);
        let full_res = app
            .session
            .document
            .preview_resolution
            .min(super::INTERACTIVE_PREVIEW_CAP);

        app.worker_mark_all_dirty = false;
        app.worker_dirty_from = Some(id);
        app.worker_dirty_region = Some(rect(0.5, 0.5, 0.02));

        // Cold cache: gate declines, quality unchanged.
        app.worker_cache_res = None;
        app.scheduler.quality = PreviewQuality::Draft;
        assert_eq!(
            app.straight_to_full_quality(PreviewQuality::Draft),
            PreviewQuality::Draft,
            "a cold worker cache keeps today's Draft-first ladder"
        );
        assert_eq!(app.scheduler.quality, PreviewQuality::Draft);

        // Full cache + small scope: gate promotes and syncs quality.
        app.worker_cache_res = Some(full_res);
        assert_eq!(
            app.straight_to_full_quality(PreviewQuality::Draft),
            PreviewQuality::Full,
            "a bounded scope over a Full cache submits straight at Full"
        );
        assert_eq!(app.scheduler.quality, PreviewQuality::Full);
        assert_eq!(app.ui_state.quality, PreviewQuality::Full);
    }

    /// The gate declines when there is no bounded scope (whole-field pending), even
    /// over a Full cache — otherwise a param edit would wrongly skip the ladder.
    #[test]
    fn straight_to_full_gate_declines_without_a_bounded_scope() {
        let mut app = TerraApp::default();
        let layer = flat(3.0);
        let id = layer.id();
        app.session.document.stack = LayerStack::new();
        app.session.document.stack.push(layer);
        let full_res = app
            .session
            .document
            .preview_resolution
            .min(super::INTERACTIVE_PREVIEW_CAP);
        app.worker_cache_res = Some(full_res);
        app.worker_mark_all_dirty = false;
        app.worker_dirty_from = Some(id);
        app.worker_dirty_region = None; // whole-field suffix

        assert_eq!(
            app.straight_to_full_quality(PreviewQuality::Draft),
            PreviewQuality::Draft,
            "no bounded scope: the ladder runs normally"
        );
    }

    /// Revert check for #33: restoring a fresh-layer CPU shortcut in
    /// `request_rebuild_immediate` will populate the cache and replace last-good before
    /// this test reaches the worker assertions.
    #[test]
    fn add_layer_keeps_last_good_until_async_cpu_evaluation_publishes() {
        let mut app = TerraApp::default();
        app.session.document.stack = LayerStack::new();

        let layer = Layer::new(
            "Unsupported SPE",
            LayerKind::StreamPowerErosion(StreamPowerParams::default()),
        );
        let layer_id = layer.id();
        app.session.document.stack.push(layer);
        app.session.document.selected = Some(layer_id);

        let height = Heightfield::filled(HeightfieldMetrics::new(32, 32, 320.0, 320.0), 7.0);
        let last_good = Arc::new(height.clone());
        app.last_height = Some(height);
        app.scheduler.last_good = Some(Arc::clone(&last_good));

        app.request_rebuild_immediate();

        assert!(
            app.scheduler.evaluator.cache.get(layer_id).is_none(),
            "the add-layer event path must not evaluate or cache CPU layer output"
        );
        assert!(
            Arc::ptr_eq(
                app.scheduler
                    .last_good
                    .as_ref()
                    .expect("seeded last-good must remain available"),
                &last_good,
            ),
            "last-good viewport content must remain authoritative while CPU work is pending"
        );
        assert!(
            app.pending_eval,
            "the add-layer edit must queue Draft evaluation"
        );
        assert!(
            app.force_draft,
            "the queued evaluation must start at Draft quality"
        );

        // This is the same non-blocking step the lifecycle runs after the edit debounce.
        // With no GPU engine, it must submit the existing worker rather than evaluate here.
        app.run_eval_step();

        assert!(
            app.worker_refine_pending,
            "CPU fallback must be owned by EvalWorker"
        );
        assert_eq!(app.ui_state.profile.path, "async CPU");
        assert!(
            app.scheduler.evaluator.cache.get(layer_id).is_none(),
            "submitting the worker must not synchronously mutate the UI-thread cache"
        );
        assert!(
            Arc::ptr_eq(
                app.scheduler
                    .last_good
                    .as_ref()
                    .expect("last-good must remain while the worker runs"),
                &last_good,
            ),
            "the viewport must retain last-good until the worker result is consumed"
        );
    }

    #[test]
    fn current_worker_failure_clears_pending_state_and_preserves_last_good() {
        let mut app = TerraApp::default();
        app.eval_token = 23;
        app.worker_refine_pending = true;
        app.ui_state.refining = true;
        app.ui_state.build_progress = Some(0.5);
        app.ui_state.refining_layer_name = Some("Hydraulic Erosion".into());
        let last_good = Arc::new(Heightfield::filled(
            HeightfieldMetrics::new(8, 8, 80.0, 80.0),
            12.0,
        ));
        app.scheduler.last_good = Some(Arc::clone(&last_good));

        app.handle_evaluation_failure_details(
            23,
            PreviewQuality::Full,
            None,
            false,
            "synthetic worker failure",
        );

        assert!(!app.worker_refine_pending);
        assert!(!app.ui_state.refining);
        assert_eq!(app.ui_state.build_progress, None);
        assert_eq!(app.ui_state.refining_layer_name, None);
        assert!(app.ui_state.status.contains("last good preview"));
        let failure = app
            .ui_state
            .evaluation_failure
            .as_ref()
            .expect("persistent evaluation failure");
        assert_eq!(failure.layer_name.as_deref(), Some("Hydraulic Erosion"));
        assert_eq!(failure.quality, PreviewQuality::Full);
        assert!(failure.message.contains("synthetic worker failure"));
        assert!(!failure.worker_restarted);
        assert!(Arc::ptr_eq(
            app.scheduler.last_good.as_ref().expect("last good"),
            &last_good
        ));
    }

    #[test]
    fn stale_worker_failure_does_not_cancel_current_generation() {
        let mut app = TerraApp::default();
        app.eval_token = 31;
        app.worker_refine_pending = true;
        app.ui_state.refining = true;
        app.ui_state.build_progress = Some(0.25);

        app.handle_evaluation_failure_details(
            30,
            PreviewQuality::Draft,
            None,
            false,
            "stale failure",
        );

        assert!(app.worker_refine_pending);
        assert!(app.ui_state.refining);
        assert_eq!(app.ui_state.build_progress, Some(0.25));
        assert!(app.ui_state.evaluation_failure.is_none());
    }

    #[test]
    fn restarted_worker_failure_is_actionable_and_preserves_last_good() {
        let mut app = TerraApp::default();
        app.eval_token = 44;
        let last_good = Arc::new(Heightfield::filled(
            HeightfieldMetrics::new(8, 8, 80.0, 80.0),
            6.0,
        ));
        app.scheduler.last_good = Some(Arc::clone(&last_good));

        app.handle_evaluation_failure_details(
            44,
            PreviewQuality::Medium,
            Some("Crater".into()),
            true,
            "worker disconnected",
        );

        let failure = app
            .ui_state
            .evaluation_failure
            .as_ref()
            .expect("persistent failure");
        assert_eq!(failure.layer_name.as_deref(), Some("Crater"));
        assert!(failure.worker_restarted);
        assert!(app.ui_state.status.contains("worker restarted"));
        assert!(Arc::ptr_eq(
            app.scheduler.last_good.as_ref().expect("last good"),
            &last_good
        ));
    }
}
