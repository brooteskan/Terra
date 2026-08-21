//! Bounded logical-frame tracing for the Base-brush/refinement workload.
//!
//! Recording is deliberately allocation-light and never performs I/O. The
//! benchmark/reporting path may snapshot and serialize records after a run.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use terra_core::eval::PreviewQuality;
use terra_gpu_eval::GpuEvaluationIntent;

use super::logical_frame::{EditGeneration, FrameIdentity, FramePhase, LogicalFrameId};

const EVENT_CAPACITY: usize = 2_048;
const SAMPLE_CAPACITY: usize = 256;

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
    CandidateRejectedStale,
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
        if !self.verbose {
            return false;
        }
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
                duration,
                gpu_stats: None,
                refinement_job: None,
                completed_units: 0,
                total_units: 0,
                refinement_submission_depth: 0,
            },
        );
        true
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
    fn disabled_verbose_wrappers_do_not_annotate_the_previous_event() {
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

        assert_eq!(trace.events().len(), 1);
        assert_eq!(trace.events().back().unwrap().refinement_job, None);
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
}
