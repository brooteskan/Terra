//! Bounded logical-frame tracing for the Base-brush/refinement workload.
//!
//! Recording is deliberately allocation-light and never performs I/O. The
//! benchmark/reporting path may snapshot and serialize records after a run.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use terra_core::quality::PreviewQuality;
use terra_gpu_eval::GpuEvaluationIntent;

use super::logical_frame::{EditGeneration, FrameIdentity, FramePhase, LogicalFrameId};

const EVENT_CAPACITY: usize = 2_048;
const SAMPLE_CAPACITY: usize = 256;
const VIOLATION_TRACE_EVENT_LIMIT: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct EvaluationTraceId(u64);

impl EvaluationTraceId {
    pub(crate) const fn new(value: u64) -> Self {
        Self(value)
    }

    pub(crate) const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FrameTraceEventKind {
    OsInputReceipt,
    SnapshotSealed,
    ToolUpdateComplete,
    StrokeRelease,
    FollowUpPressReceipt,
    FollowUpPressSealed,
    EvaluationRequested,
    EvaluationStarted,
    PlanAcquired,
    QueueSubmitted,
    CandidateAccepted,
    CandidateRefused,
    PresentationRequested,
    SurfacePresented,
    GpuEvaluationResolved,
    GpuPresentationResolved,
    RefinementJobCreated,
    RefinementUnitProgress,
    RefinementSubmissionQueued,
    RefinementSubmissionCompleted,
    RefinementSuperseded,
    RefinementCandidateCompleted,
    RefinementPublished,
    RefinementFailed,
    HeartbeatOverBudget,
    UiEffectsQueued,
    UiEffectsApplied,
    ResizeCaptured,
    ResizeApplied,
    SurfaceRecoveryRequested,
    DeviceLost,
    ShutdownRequested,
    FrameAborted,
    TerrainPresentationTransition,
    OutsideRegionProbeResolved,
    EvaluationOutputSelected,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct FrameTraceEvent {
    pub(crate) at: Instant,
    pub(crate) kind: FrameTraceEventKind,
    pub(crate) frame: LogicalFrameId,
    pub(crate) generation: EditGeneration,
    pub(crate) evaluation: Option<EvaluationTraceId>,
    pub(crate) phase: Option<FramePhase>,
    pub(crate) quality: Option<PreviewQuality>,
    pub(crate) intent: Option<GpuEvaluationIntent>,
    pub(crate) duration: Option<Duration>,
    pub(crate) gpu_stats: Option<terra_gpu_eval::GpuEvalStats>,
    pub(crate) refinement_job: Option<u64>,
    pub(crate) completed_units: usize,
    pub(crate) total_units: usize,
    pub(crate) refinement_submission_depth: u8,
    pub(crate) output_identity: Option<terra_gpu::output_identity::GpuTerrainOutputIdentity>,
    pub(crate) presentation: Option<terra_render::TerrainPresentationRecord>,
    pub(crate) diagnostic: Option<terra_render::TerrainTransitionDiagnosticCode>,
    pub(crate) candidate_decision: Option<terra_render::TerrainPresentationDecisionCode>,
    pub(crate) presentation_mode: Option<terra_render::TerrainPresentationMode>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct LatencySummary {
    pub(crate) count: usize,
    pub(crate) p50_us: u64,
    pub(crate) p95_us: u64,
    pub(crate) max_us: u64,
}

#[derive(Debug)]
pub(crate) struct FrameTraceRecorder {
    next_evaluation: u64,
    events: VecDeque<FrameTraceEvent>,
    input_to_visible_us: VecDeque<u64>,
    refinement_us: VecDeque<u64>,
    release_to_follow_up_press_us: VecDeque<u64>,
    pending_input_receipt: Option<(Instant, EditGeneration)>,
    pending_release: Option<(Instant, EditGeneration)>,
    verbose: bool,
    orphaned_events: u64,
    first_violation: Option<terra_render::TerrainTransitionDiagnosticCode>,
}

impl Default for FrameTraceRecorder {
    fn default() -> Self {
        Self {
            next_evaluation: 0,
            events: VecDeque::with_capacity(EVENT_CAPACITY),
            input_to_visible_us: VecDeque::with_capacity(SAMPLE_CAPACITY),
            refinement_us: VecDeque::with_capacity(SAMPLE_CAPACITY),
            release_to_follow_up_press_us: VecDeque::with_capacity(SAMPLE_CAPACITY),
            pending_input_receipt: None,
            pending_release: None,
            verbose: cfg!(test)
                || std::env::var("TERRA_FRAME_TRACE")
                    .is_ok_and(|value| value.eq_ignore_ascii_case("verbose")),
            orphaned_events: 0,
            first_violation: None,
        }
    }
}

impl FrameTraceRecorder {
    pub(crate) fn next_evaluation_id(&mut self) -> EvaluationTraceId {
        self.next_evaluation = self
            .next_evaluation
            .checked_add(1)
            .expect("evaluation trace id exhausted");
        EvaluationTraceId(self.next_evaluation)
    }

    pub(crate) fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose
            || std::env::var("TERRA_FRAME_TRACE")
                .is_ok_and(|value| value.eq_ignore_ascii_case("verbose"));
    }

    pub(crate) const fn orphaned_events(&self) -> u64 {
        self.orphaned_events
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record(
        &mut self,
        now: Instant,
        kind: FrameTraceEventKind,
        identity: Option<FrameIdentity>,
        phase: Option<FramePhase>,
        evaluation: Option<EvaluationTraceId>,
        quality: Option<PreviewQuality>,
        intent: Option<GpuEvaluationIntent>,
        duration: Option<Duration>,
    ) -> bool {
        let Some(identity) = identity else {
            self.orphaned_events = self.orphaned_events.saturating_add(1);
            log::error!(target: "terra_app::logical_frame", "orphaned frame trace event: {kind:?}");
            return false;
        };
        push_bounded(
            &mut self.events,
            EVENT_CAPACITY,
            FrameTraceEvent {
                at: now,
                kind,
                frame: identity.id,
                generation: identity.generation,
                evaluation,
                phase,
                quality,
                intent,
                duration: if self.verbose { duration } else { None },
                gpu_stats: None,
                refinement_job: None,
                completed_units: 0,
                total_units: 0,
                refinement_submission_depth: 0,
                output_identity: None,
                presentation: None,
                diagnostic: None,
                candidate_decision: None,
                presentation_mode: None,
            },
        );
        true
    }

    pub(crate) fn record_presentation(
        &mut self,
        now: Instant,
        identity: Option<FrameIdentity>,
        phase: Option<FramePhase>,
        evaluation: EvaluationTraceId,
        record: terra_render::TerrainPresentationRecord,
    ) {
        if !self.record(
            now,
            FrameTraceEventKind::TerrainPresentationTransition,
            identity,
            phase,
            Some(evaluation),
            Some(record.candidate.actual_quality),
            Some(record.candidate.intent),
            None,
        ) {
            return;
        }
        if let Some(event) = self.events.back_mut() {
            event.output_identity = Some(record.candidate);
            event.presentation = Some(record);
            event.diagnostic = record.shadow_diagnostic;
            event.candidate_decision = Some(record.decision);
            event.presentation_mode = Some(record.actual_mode);
        }
        if let Some(code) = record.shadow_diagnostic {
            self.report_first_violation(code, record);
        }
    }

    pub(crate) fn record_cpu_presentation(
        &mut self,
        now: Instant,
        identity: Option<FrameIdentity>,
        phase: Option<FramePhase>,
        evaluation: EvaluationTraceId,
        quality: PreviewQuality,
    ) {
        if !self.record(
            now,
            FrameTraceEventKind::TerrainPresentationTransition,
            identity,
            phase,
            Some(evaluation),
            Some(quality),
            None,
            None,
        ) {
            return;
        }
        if let Some(event) = self.events.back_mut() {
            event.presentation_mode = Some(terra_render::TerrainPresentationMode::CpuUpload);
            event.candidate_decision =
                Some(terra_render::TerrainPresentationDecisionCode::Accepted);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_candidate_refusal(
        &mut self,
        now: Instant,
        identity: Option<FrameIdentity>,
        phase: Option<FramePhase>,
        evaluation: EvaluationTraceId,
        quality: PreviewQuality,
        intent: GpuEvaluationIntent,
        output: Option<terra_gpu::output_identity::GpuTerrainOutputIdentity>,
        decision: terra_render::TerrainPresentationDecisionCode,
    ) {
        if !self.record(
            now,
            FrameTraceEventKind::CandidateRefused,
            identity,
            phase,
            Some(evaluation),
            Some(quality),
            Some(intent),
            None,
        ) {
            return;
        }
        if let Some(event) = self.events.back_mut() {
            event.output_identity = output;
            event.candidate_decision = Some(decision);
        }
    }

    pub(crate) fn record_evaluation_output(
        &mut self,
        now: Instant,
        identity: Option<FrameIdentity>,
        phase: Option<FramePhase>,
        evaluation: EvaluationTraceId,
        output: terra_gpu::output_identity::GpuTerrainOutputIdentity,
    ) {
        if !self.record(
            now,
            FrameTraceEventKind::EvaluationOutputSelected,
            identity,
            phase,
            Some(evaluation),
            Some(output.actual_quality),
            Some(output.intent),
            None,
        ) {
            return;
        }
        if let Some(event) = self.events.back_mut() {
            event.output_identity = Some(output);
        }
        if matches!(
            output.completeness,
            terra_gpu::output_identity::GpuOutputCompleteness::Complete
        ) && output.selected_field.selected != output.selected_field.expected_final
            && self.first_violation.is_none()
        {
            let code = terra_render::TerrainTransitionDiagnosticCode::NonFinalCurrentOutput;
            self.first_violation = Some(code);
            if let Some(event) = self.events.back_mut() {
                event.diagnostic = Some(code);
            }
            log::error!(
                target: "terra_app::terrain_transition",
                "terrain_transition_violation code={} stage=evaluation_output_selection output={} generation={} evaluation={} expected_field={:?} actual_field={:?} source_incarnation={} physical_allocation={}",
                code.as_str(), output.output.0, output.generation, output.evaluation_id,
                output.selected_field.expected_final, output.selected_field.selected,
                output.selected_field.resource_incarnation.0,
                output.selected_field.physical_allocation,
            );
            self.report_violation_trace_tail();
        }
    }

    fn report_first_violation(
        &mut self,
        code: terra_render::TerrainTransitionDiagnosticCode,
        record: terra_render::TerrainPresentationRecord,
    ) {
        if self.first_violation.is_some() {
            return;
        }
        self.first_violation = Some(code);
        log::error!(
            target: "terra_app::terrain_transition",
            "terrain_transition_violation code={} output={} generation={}/{} evaluation={} plan_revision={}/{} extent={:?}/{:?} expected_field={:?} actual_field={:?} baseline_before={:?} mode={:?}",
            code.as_str(),
            record.candidate.output.0,
            record.candidate.generation,
            record.expectations.generation,
            record.candidate.evaluation_id,
            record.candidate.plan_revision,
            record.expectations.plan_revision,
            record.candidate.extent,
            record.expectations.extent,
            record.candidate.selected_field.expected_final,
            record.candidate.selected_field.selected,
            record.baseline_before.map(|baseline| baseline.identity.output.0),
            record.actual_mode,
        );
        self.report_violation_trace_tail();
    }

    pub(crate) fn record_probe_result(
        &mut self,
        now: Instant,
        result: terra_render::TerrainIntegrityProbeResult,
    ) {
        let output = result.candidate;
        let identity = FrameIdentity {
            id: LogicalFrameId::new(output.frame_id),
            generation_at_start: EditGeneration::new(output.generation),
            generation: EditGeneration::new(output.generation),
        };
        if !self.record(
            now,
            FrameTraceEventKind::OutsideRegionProbeResolved,
            Some(identity),
            None,
            Some(EvaluationTraceId::new(output.evaluation_id)),
            Some(output.actual_quality),
            Some(output.intent),
            None,
        ) {
            return;
        }
        if let Some(event) = self.events.back_mut() {
            event.output_identity = Some(output);
            event.diagnostic = (!result.passed).then_some(
                terra_render::TerrainTransitionDiagnosticCode::OutsideDirtyRegionChanged,
            );
        }
        if !result.passed && self.first_violation.is_none() {
            let code = terra_render::TerrainTransitionDiagnosticCode::OutsideDirtyRegionChanged;
            self.first_violation = Some(code);
            log::error!(
                target: "terra_app::terrain_transition",
                "terrain_transition_violation code={} output={} generation={} evaluation={} expected_base={:?} rect={:?} max_delta={} first_probe={:?} compared={}",
                code.as_str(), output.output.0, output.generation, output.evaluation_id,
                result.expected_base.map(|id| id.0), result.rect, result.max_delta,
                result.first_failing_probe, result.probes_compared,
            );
            self.report_violation_trace_tail();
        }
    }

    fn report_violation_trace_tail(&self) {
        let omitted = violation_trace_tail_start(self.events.len());
        if omitted > 0 {
            log::error!(
                target: "terra_app::terrain_transition_trace",
                "trace_tail omitted_events={} retained_events={}",
                omitted,
                self.events.len() - omitted,
            );
        }
        for event in self.events.iter().skip(omitted) {
            log::error!(
                target: "terra_app::terrain_transition_trace",
                "trace kind={:?} frame={} generation={} evaluation={:?} phase={:?} output={:?} diagnostic={:?}",
                event.kind,
                event.frame.get(),
                event.generation.get(),
                event.evaluation.map(EvaluationTraceId::get),
                event.phase,
                event.output_identity.map(|output| output.output.0),
                event.diagnostic.map(terra_render::TerrainTransitionDiagnosticCode::as_str),
            );
        }
    }

    #[cfg(test)]
    pub(crate) const fn first_violation(
        &self,
    ) -> Option<terra_render::TerrainTransitionDiagnosticCode> {
        self.first_violation
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_refinement(
        &mut self,
        now: Instant,
        kind: FrameTraceEventKind,
        identity: Option<FrameIdentity>,
        phase: Option<FramePhase>,
        evaluation: EvaluationTraceId,
        quality: PreviewQuality,
        job: u64,
        completed_units: usize,
        total_units: usize,
        submission_depth: u8,
        duration: Option<Duration>,
    ) {
        if !self.record(
            now,
            kind,
            identity,
            phase,
            Some(evaluation),
            Some(quality),
            Some(GpuEvaluationIntent::Complete),
            duration,
        ) {
            return;
        }
        if let Some(event) = self.events.back_mut() {
            event.refinement_job = Some(job);
            event.completed_units = completed_units;
            event.total_units = total_units;
            event.refinement_submission_depth = submission_depth;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_evaluation_submission(
        &mut self,
        now: Instant,
        identity: Option<FrameIdentity>,
        phase: Option<FramePhase>,
        evaluation: EvaluationTraceId,
        quality: PreviewQuality,
        intent: GpuEvaluationIntent,
        duration: Duration,
        stats: terra_gpu_eval::GpuEvalStats,
    ) {
        if !self.record(
            now,
            FrameTraceEventKind::QueueSubmitted,
            identity,
            phase,
            Some(evaluation),
            Some(quality),
            Some(intent),
            Some(duration),
        ) {
            return;
        }
        if let Some(event) = self.events.back_mut() {
            event.gpu_stats = Some(stats);
        }
    }

    pub(crate) fn note_input_receipt(&mut self, now: Instant, generation: EditGeneration) {
        self.pending_input_receipt = Some((now, generation));
    }

    pub(crate) fn note_release(&mut self, now: Instant, generation: EditGeneration) {
        self.pending_release = Some((now, generation));
    }

    pub(crate) fn note_follow_up_press(&mut self, now: Instant) -> bool {
        let Some((released, _)) = self.pending_release.take() else {
            return false;
        };
        push_bounded(
            &mut self.release_to_follow_up_press_us,
            SAMPLE_CAPACITY,
            micros(now.saturating_duration_since(released)),
        );
        true
    }

    pub(crate) fn note_surface_presented(&mut self, now: Instant, generation: EditGeneration) {
        if let Some((received, expected)) = self.pending_input_receipt {
            if expected == generation {
                push_bounded(
                    &mut self.input_to_visible_us,
                    SAMPLE_CAPACITY,
                    micros(now.saturating_duration_since(received)),
                );
                self.pending_input_receipt = None;
            }
        }
        if let Some((released, expected)) = self.pending_release {
            if expected == generation {
                push_bounded(
                    &mut self.refinement_us,
                    SAMPLE_CAPACITY,
                    micros(now.saturating_duration_since(released)),
                );
                self.pending_release = None;
            }
        }
    }

    pub(crate) fn input_to_visible_summary(&self) -> LatencySummary {
        summarize(&self.input_to_visible_us)
    }

    pub(crate) fn refinement_summary(&self) -> LatencySummary {
        summarize(&self.refinement_us)
    }

    pub(crate) fn follow_up_press_summary(&self) -> LatencySummary {
        summarize(&self.release_to_follow_up_press_us)
    }

    #[cfg(test)]
    pub(crate) fn events(&self) -> &VecDeque<FrameTraceEvent> {
        &self.events
    }
}

fn micros(duration: Duration) -> u64 {
    duration.as_micros().min(u128::from(u64::MAX)) as u64
}

fn push_bounded<T>(queue: &mut VecDeque<T>, capacity: usize, value: T) {
    if queue.len() == capacity {
        queue.pop_front();
    }
    queue.push_back(value);
}

fn violation_trace_tail_start(event_count: usize) -> usize {
    event_count.saturating_sub(VIOLATION_TRACE_EVENT_LIMIT)
}

fn summarize(samples: &VecDeque<u64>) -> LatencySummary {
    if samples.is_empty() {
        return LatencySummary::default();
    }
    let mut sorted: Vec<_> = samples.iter().copied().collect();
    sorted.sort_unstable();
    let percentile = |numerator: usize| {
        let rank = (numerator * sorted.len()).div_ceil(100).max(1);
        sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
    };
    LatencySummary {
        count: sorted.len(),
        p50_us: percentile(50),
        p95_us: percentile(95),
        max_us: *sorted.last().expect("non-empty samples"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use terra_core::terrain_plan::FieldSlot;
    use terra_gpu::output_identity::*;

    #[test]
    fn percentile_summary_uses_nearest_rank() {
        let samples = VecDeque::from([1, 2, 3, 4, 100]);
        assert_eq!(
            summarize(&samples),
            LatencySummary {
                count: 5,
                p50_us: 3,
                p95_us: 100,
                max_us: 100,
            }
        );
    }

    #[test]
    fn records_are_bounded() {
        let mut trace = FrameTraceRecorder::default();
        let identity = FrameIdentity {
            id: LogicalFrameId::new(1),
            generation_at_start: EditGeneration::new(1),
            generation: EditGeneration::new(1),
        };
        for _ in 0..EVENT_CAPACITY + 7 {
            trace.record(
                Instant::now(),
                FrameTraceEventKind::OsInputReceipt,
                Some(identity),
                None,
                None,
                None,
                None,
                None,
            );
        }
        assert_eq!(trace.events().len(), EVENT_CAPACITY);
    }

    #[test]
    fn violation_trace_tail_is_bounded() {
        assert_eq!(violation_trace_tail_start(12), 0);
        assert_eq!(violation_trace_tail_start(VIOLATION_TRACE_EVENT_LIMIT), 0);
        assert_eq!(violation_trace_tail_start(EVENT_CAPACITY), 1_984);
        assert_eq!(
            EVENT_CAPACITY - violation_trace_tail_start(EVENT_CAPACITY),
            64
        );
    }

    /// Manual release-mode measurement used by the issue-168 diagnostic note.
    #[test]
    #[ignore = "run explicitly when measuring compact trace overhead"]
    fn compact_trace_overhead_probe() {
        const ITERATIONS: u32 = 1_000_000;
        let identity = FrameIdentity {
            id: LogicalFrameId::new(1),
            generation_at_start: EditGeneration::new(1),
            generation: EditGeneration::new(1),
        };
        let baseline_started = Instant::now();
        for _ in 0..ITERATIONS {
            std::hint::black_box(Instant::now());
        }
        let baseline = baseline_started.elapsed();

        let mut trace = FrameTraceRecorder::default();
        trace.verbose = false;
        let traced_started = Instant::now();
        for _ in 0..ITERATIONS {
            trace.record(
                Instant::now(),
                FrameTraceEventKind::OsInputReceipt,
                Some(identity),
                None,
                None,
                None,
                None,
                None,
            );
        }
        let traced = traced_started.elapsed();
        let incremental = traced.saturating_sub(baseline);
        eprintln!(
            "compact_trace event_bytes={} capacity={} baseline_ns_per_event={:.1} total_ns_per_event={:.1} incremental_ns_per_event={:.1}",
            std::mem::size_of::<FrameTraceEvent>(),
            EVENT_CAPACITY,
            baseline.as_nanos() as f64 / f64::from(ITERATIONS),
            traced.as_nanos() as f64 / f64::from(ITERATIONS),
            incremental.as_nanos() as f64 / f64::from(ITERATIONS),
        );
        assert_eq!(trace.events().len(), EVENT_CAPACITY);
    }

    #[test]
    fn records_without_a_logical_frame_are_rejected_and_counted() {
        let mut trace = FrameTraceRecorder::default();
        trace.record(
            Instant::now(),
            FrameTraceEventKind::OsInputReceipt,
            None,
            None,
            None,
            None,
            None,
            None,
        );

        assert!(trace.events().is_empty());
        assert_eq!(trace.orphaned_events(), 1);
    }

    #[test]
    fn compact_records_survive_when_verbose_details_are_disabled() {
        let mut trace = FrameTraceRecorder::default();
        let identity = FrameIdentity {
            id: LogicalFrameId::new(2),
            generation_at_start: EditGeneration::new(3),
            generation: EditGeneration::new(3),
        };
        trace.record(
            Instant::now(),
            FrameTraceEventKind::OsInputReceipt,
            Some(identity),
            None,
            None,
            None,
            None,
            None,
        );
        trace.verbose = false;
        trace.record_refinement(
            Instant::now(),
            FrameTraceEventKind::RefinementUnitProgress,
            Some(identity),
            None,
            EvaluationTraceId::new(4),
            PreviewQuality::Medium,
            99,
            1,
            2,
            1,
            None,
        );

        assert_eq!(trace.events().len(), 2);
        assert_eq!(trace.events().back().unwrap().refinement_job, Some(99));
        assert_eq!(trace.first_violation(), None);
    }

    #[test]
    fn refinement_events_preserve_frame_generation_progress_and_depth() {
        let mut trace = FrameTraceRecorder::default();
        let identity = FrameIdentity {
            id: LogicalFrameId::new(12),
            generation_at_start: EditGeneration::new(8),
            generation: EditGeneration::new(9),
        };
        let evaluation = trace.next_evaluation_id();
        trace.record_refinement(
            Instant::now(),
            FrameTraceEventKind::RefinementSubmissionQueued,
            Some(identity),
            Some(FramePhase::OptionalRefinement),
            evaluation,
            PreviewQuality::Full,
            41,
            7,
            19,
            1,
            None,
        );

        let event = trace.events().back().expect("refinement event");
        assert_eq!(event.frame, identity.id);
        assert_eq!(event.generation, identity.generation);
        assert_eq!(event.evaluation, Some(evaluation));
        assert_eq!(event.quality, Some(PreviewQuality::Full));
        assert_eq!(event.refinement_job, Some(41));
        assert_eq!(event.completed_units, 7);
        assert_eq!(event.total_units, 19);
        assert_eq!(event.refinement_submission_depth, 1);
    }

    #[test]
    fn visible_latency_only_closes_on_the_matching_generation() {
        let started = Instant::now();
        let mut trace = FrameTraceRecorder::default();
        trace.note_input_receipt(started, EditGeneration::new(7));
        trace.note_release(started, EditGeneration::new(7));

        trace.note_surface_presented(started + Duration::from_millis(2), EditGeneration::new(6));
        assert_eq!(trace.input_to_visible_summary().count, 0);
        assert_eq!(trace.refinement_summary().count, 0);

        trace.note_surface_presented(started + Duration::from_millis(5), EditGeneration::new(7));
        assert_eq!(trace.input_to_visible_summary().p50_us, 5_000);
        assert_eq!(trace.refinement_summary().p50_us, 5_000);
    }

    #[test]
    fn follow_up_press_preempts_the_pending_refinement_sample() {
        let released = Instant::now();
        let mut trace = FrameTraceRecorder::default();
        trace.note_release(released, EditGeneration::new(3));
        assert!(trace.note_follow_up_press(released + Duration::from_millis(7)));
        assert_eq!(trace.follow_up_press_summary().p95_us, 7_000);

        trace.note_surface_presented(released + Duration::from_millis(20), EditGeneration::new(3));
        assert_eq!(trace.refinement_summary().count, 0);
        assert!(!trace.note_follow_up_press(released + Duration::from_millis(30)));
    }

    #[test]
    fn first_transition_violation_is_latched_before_later_pixel_failure() {
        let identity = FrameIdentity {
            id: LogicalFrameId::new(9),
            generation_at_start: EditGeneration::new(3),
            generation: EditGeneration::new(3),
        };
        let output = GpuTerrainOutputIdentity {
            output: GpuOutputId(4),
            frame_id: 9,
            generation: 3,
            evaluation_id: 7,
            plan_revision: 5,
            requested_quality: PreviewQuality::Full,
            actual_quality: PreviewQuality::Full,
            intent: GpuEvaluationIntent::InteractiveLocal,
            selected_field: GpuSelectedFieldIdentity {
                selected: FieldSlot::from_index(2),
                expected_final: FieldSlot::from_index(1),
                resource_incarnation: GpuResourceIncarnation(2),
                physical_allocation: 0,
            },
            output_resource: GpuOutputResourceIdentity {
                device_generation: 1,
                incarnation: GpuResourceIncarnation(3),
                slot: GpuOutputSlot::Ping,
            },
            extent: (64, 64),
            coverage: GpuOutputCoverage::WholeField,
            completeness: GpuOutputCompleteness::Complete,
            invalidation: GpuInvalidationKind::Cold,
            last_write: GpuLastWriteIdentity {
                serial: GpuSubmissionSerial(8),
                completion: GpuSubmissionCompletion::Submitted,
            },
        };
        let baseline = terra_render::PresentedTerrainBaseline {
            identity: output,
            mode: terra_render::TerrainPresentationMode::Shared,
            complete: false,
            local_slots_coherent: false,
            local_slot_epoch: 1,
            last_full_generation: Some(3),
            height_lineage: output.output,
            normal_lineage: output.output,
        };
        let record = terra_render::TerrainPresentationRecord {
            candidate: output,
            expectations: terra_render::TerrainPresentationExpectations {
                plan_revision: 5,
                generation: 3,
                extent: (64, 64),
            },
            baseline_before: None,
            baseline_after: baseline,
            requested_mode: terra_render::TerrainPresentationMode::Shared,
            actual_mode: terra_render::TerrainPresentationMode::Shared,
            requested_rect: None,
            actual_rect: None,
            local_slots_coherent_before: false,
            local_slots_coherent_after: false,
            decision: terra_render::TerrainPresentationDecisionCode::Accepted,
            shadow_diagnostic: Some(
                terra_render::TerrainTransitionDiagnosticCode::NonFinalCurrentOutput,
            ),
        };
        let mut trace = FrameTraceRecorder::default();
        trace.record_presentation(
            Instant::now(),
            Some(identity),
            Some(FramePhase::RequiredInteractiveWork),
            EvaluationTraceId::new(7),
            record,
        );
        trace.record_probe_result(
            Instant::now(),
            terra_render::TerrainIntegrityProbeResult {
                candidate: output,
                expected_base: None,
                rect: terra_core::tiling::SampleRect {
                    x: 2,
                    y: 2,
                    w: 3,
                    h: 3,
                },
                passed: false,
                max_delta: 4.0,
                first_failing_probe: Some(0),
                probes_compared: 64,
            },
        );
        assert_eq!(
            trace.first_violation(),
            Some(terra_render::TerrainTransitionDiagnosticCode::NonFinalCurrentOutput)
        );
    }
}
