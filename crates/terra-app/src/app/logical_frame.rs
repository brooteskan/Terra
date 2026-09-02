//! Event-driven logical-frame identities, phases, and optional-work budgets.

use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub(crate) struct LogicalFrameId(u64);

impl LogicalFrameId {
    pub(crate) const fn new(value: u64) -> Self {
        Self(value)
    }

    pub(crate) const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub(crate) struct EditGeneration(u64);

impl EditGeneration {
    pub(crate) const fn new(value: u64) -> Self {
        Self(value)
    }

    pub(crate) const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FramePhase {
    CollectingInput,
    SealingInput,
    ApplicationUpdate,
    RequiredInteractiveWork,
    PresentationRequest,
    OptionalRefinement,
    Complete,
}

impl FramePhase {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::CollectingInput => "collecting input",
            Self::SealingInput => "sealing input",
            Self::ApplicationUpdate => "application update",
            Self::RequiredInteractiveWork => "interactive work",
            Self::PresentationRequest => "presentation request",
            Self::OptionalRefinement => "optional refinement",
            Self::Complete => "complete",
        }
    }

    const fn ordinal(self) -> u8 {
        match self {
            Self::CollectingInput => 0,
            Self::SealingInput => 1,
            Self::ApplicationUpdate => 2,
            Self::RequiredInteractiveWork => 3,
            Self::PresentationRequest => 4,
            Self::OptionalRefinement => 5,
            Self::Complete => 6,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FrameRequestReason {
    Input,
    UiActions,
    Resize,
    RequiredEvaluation,
    Completion,
    Animation,
    SurfaceRecovery,
    OptionalRefinement,
    Shutdown,
}

impl FrameRequestReason {
    const fn priority(self) -> u8 {
        match self {
            Self::Shutdown => 9,
            Self::Input => 8,
            Self::Resize => 7,
            Self::UiActions => 6,
            Self::RequiredEvaluation => 5,
            Self::Completion | Self::SurfaceRecovery => 4,
            Self::Animation => 3,
            Self::OptionalRefinement => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FrameDeadlineKind {
    InteractiveEvaluation,
    DeferredFullField,
    OptionalRefinement,
    FullFieldRefinement,
}

#[derive(Debug, Clone, Copy)]
struct GenerationDeadline {
    generation: EditGeneration,
    at: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FrameWake {
    Wait,
    WaitUntil(Instant),
    Poll,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct FrameIdentity {
    pub(crate) id: LogicalFrameId,
    pub(crate) generation_at_start: EditGeneration,
    pub(crate) generation: EditGeneration,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FrameWorkBudget {
    started_at: Instant,
    optional_deadline: Option<Instant>,
}

impl FrameWorkBudget {
    #[cfg(test)]
    pub(crate) fn unbounded(started_at: Instant) -> Self {
        Self {
            started_at,
            optional_deadline: None,
        }
    }

    fn bounded(started_at: Instant, duration: Duration) -> Self {
        Self {
            started_at,
            optional_deadline: started_at.checked_add(duration),
        }
    }

    pub(crate) fn can_start_optional(self, now: Instant) -> bool {
        self.optional_deadline.is_none_or(|deadline| now < deadline)
    }

    pub(crate) fn elapsed(self, now: Instant) -> Duration {
        now.saturating_duration_since(self.started_at)
    }
}

#[derive(Debug, Clone, Copy)]
struct FrameState {
    identity: FrameIdentity,
    phase: FramePhase,
    reason: FrameRequestReason,
    input_events: usize,
    pointer_samples: usize,
    budget: FrameWorkBudget,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FrameDiagnostics {
    pub(crate) identity: FrameIdentity,
    pub(crate) phase: FramePhase,
    pub(crate) reason: FrameRequestReason,
    pub(crate) input_events: usize,
    pub(crate) pointer_samples: usize,
    pub(crate) elapsed: Duration,
}

#[derive(Debug, Default)]
pub(crate) struct LogicalFrameCoordinator {
    next_id: u64,
    pending: Option<(LogicalFrameId, EditGeneration, FrameRequestReason)>,
    active: Option<FrameState>,
    last_complete: Option<FrameDiagnostics>,
    presentation_pending_for: Option<FrameIdentity>,
    interactive_evaluation_deadline: Option<GenerationDeadline>,
    deferred_full_field_deadline: Option<GenerationDeadline>,
    optional_refinement_deadline: Option<GenerationDeadline>,
    full_field_refinement_deadline: Option<GenerationDeadline>,
    shutdown_requested: bool,
}

impl LogicalFrameCoordinator {
    pub(crate) fn request(
        &mut self,
        generation: EditGeneration,
        reason: FrameRequestReason,
    ) -> LogicalFrameId {
        if let Some((id, pending_generation, pending_reason)) = self.pending {
            let generation = generation.max(pending_generation);
            let reason = if reason.priority() > pending_reason.priority() {
                reason
            } else {
                pending_reason
            };
            self.pending = Some((id, generation, reason));
            return id;
        }
        self.next_id = self
            .next_id
            .checked_add(1)
            .expect("logical frame id exhausted");
        let id = LogicalFrameId(self.next_id);
        self.pending = Some((id, generation, reason));
        log::debug!(
            target: "terra_app::logical_frame",
            "frame={} generation={} phase={} reason={reason:?}",
            id.get(),
            generation.get(),
            FramePhase::CollectingInput.label()
        );
        id
    }

    pub(crate) fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    pub(crate) fn begin(
        &mut self,
        now: Instant,
        input_events: usize,
        pointer_samples: usize,
    ) -> Option<FrameIdentity> {
        let (id, generation, reason) = self.pending.take()?;
        let identity = FrameIdentity {
            id,
            generation_at_start: generation,
            generation,
        };
        self.active = Some(FrameState {
            identity,
            phase: FramePhase::CollectingInput,
            reason,
            input_events,
            pointer_samples,
            budget: FrameWorkBudget::bounded(
                now,
                Duration::from_millis(super::LOGICAL_FRAME_HOST_BUDGET_MS),
            ),
        });
        Some(identity)
    }

    pub(crate) fn transition(&mut self, phase: FramePhase) {
        let Some(active) = self.active.as_mut() else {
            return;
        };
        debug_assert!(
            phase.ordinal() >= active.phase.ordinal(),
            "logical frame phase moved backwards"
        );
        if phase != active.phase {
            log::debug!(
                target: "terra_app::logical_frame",
                "frame={} generation={} phase={} -> {}",
                active.identity.id.get(),
                active.identity.generation.get(),
                active.phase.label(),
                phase.label()
            );
        }
        active.phase = phase;
    }

    pub(crate) fn update_generation(&mut self, generation: EditGeneration) {
        if let Some(active) = self.active.as_mut() {
            active.identity.generation = generation;
        }
    }

    pub(crate) fn active_identity(&self) -> Option<FrameIdentity> {
        self.active.map(|active| active.identity)
    }

    pub(crate) fn active_phase(&self) -> Option<FramePhase> {
        self.active.map(|active| active.phase)
    }

    pub(crate) fn pending_identity(&self) -> Option<FrameIdentity> {
        self.pending.map(|(id, generation, _)| FrameIdentity {
            id,
            generation_at_start: generation,
            generation,
        })
    }

    pub(crate) fn can_start_optional(&self, now: Instant) -> bool {
        self.active
            .is_none_or(|active| active.budget.can_start_optional(now))
            && !self.has_pending()
    }

    pub(crate) fn schedule_deadline(
        &mut self,
        kind: FrameDeadlineKind,
        generation: EditGeneration,
        at: Instant,
    ) {
        let slot = self.deadline_slot_mut(kind);
        let at = slot
            .filter(|deadline| deadline.generation == generation)
            .map_or(at, |deadline| deadline.at.max(at));
        *slot = Some(GenerationDeadline { generation, at });
    }

    pub(crate) fn clear_deadline(&mut self, kind: FrameDeadlineKind) {
        *self.deadline_slot_mut(kind) = None;
    }

    pub(crate) fn deadline_ready(
        &self,
        kind: FrameDeadlineKind,
        generation: EditGeneration,
        now: Instant,
    ) -> bool {
        self.deadline(kind)
            .is_none_or(|deadline| deadline.generation == generation && now >= deadline.at)
    }

    pub(crate) fn discard_stale_deadlines(&mut self, generation: EditGeneration) {
        for kind in [
            FrameDeadlineKind::InteractiveEvaluation,
            FrameDeadlineKind::DeferredFullField,
            FrameDeadlineKind::OptionalRefinement,
            FrameDeadlineKind::FullFieldRefinement,
        ] {
            if self
                .deadline(kind)
                .is_some_and(|deadline| deadline.generation != generation)
            {
                self.clear_deadline(kind);
            }
        }
    }

    pub(crate) fn next_deadline(&self, generation: EditGeneration) -> Option<Instant> {
        [
            self.interactive_evaluation_deadline,
            self.deferred_full_field_deadline,
            self.optional_refinement_deadline,
            self.full_field_refinement_deadline,
        ]
        .into_iter()
        .flatten()
        .filter(|deadline| deadline.generation == generation)
        .map(|deadline| deadline.at)
        .min()
    }

    pub(crate) fn wake_decision(
        &self,
        generation: EditGeneration,
        now: Instant,
        continuous: bool,
        fallback_deadline: Option<Instant>,
    ) -> FrameWake {
        if continuous {
            return FrameWake::Poll;
        }
        let deadline = self
            .next_deadline(generation)
            .into_iter()
            .chain(fallback_deadline)
            .filter(|deadline| *deadline > now)
            .min();
        deadline.map_or(FrameWake::Wait, FrameWake::WaitUntil)
    }

    pub(crate) fn request_shutdown(&mut self, generation: EditGeneration) {
        self.shutdown_requested = true;
        self.request(generation, FrameRequestReason::Shutdown);
    }

    pub(crate) fn shutdown_requested(&self) -> bool {
        self.shutdown_requested
    }

    pub(crate) fn clear_presentation(&mut self) {
        self.presentation_pending_for = None;
    }

    pub(crate) fn abort(&mut self, now: Instant) -> Option<FrameDiagnostics> {
        self.pending = None;
        self.clear_presentation();
        self.interactive_evaluation_deadline = None;
        self.deferred_full_field_deadline = None;
        self.optional_refinement_deadline = None;
        self.full_field_refinement_deadline = None;
        self.complete(now)
    }

    fn deadline(&self, kind: FrameDeadlineKind) -> Option<GenerationDeadline> {
        match kind {
            FrameDeadlineKind::InteractiveEvaluation => self.interactive_evaluation_deadline,
            FrameDeadlineKind::DeferredFullField => self.deferred_full_field_deadline,
            FrameDeadlineKind::OptionalRefinement => self.optional_refinement_deadline,
            FrameDeadlineKind::FullFieldRefinement => self.full_field_refinement_deadline,
        }
    }

    fn deadline_slot_mut(&mut self, kind: FrameDeadlineKind) -> &mut Option<GenerationDeadline> {
        match kind {
            FrameDeadlineKind::InteractiveEvaluation => &mut self.interactive_evaluation_deadline,
            FrameDeadlineKind::DeferredFullField => &mut self.deferred_full_field_deadline,
            FrameDeadlineKind::OptionalRefinement => &mut self.optional_refinement_deadline,
            FrameDeadlineKind::FullFieldRefinement => &mut self.full_field_refinement_deadline,
        }
    }

    pub(crate) fn mark_presentation_requested(&mut self) {
        if let Some(active) = self.active {
            self.presentation_pending_for = Some(active.identity);
        }
    }

    pub(crate) fn take_presentation_identity(&mut self) -> Option<FrameIdentity> {
        self.presentation_pending_for.take()
    }

    pub(crate) fn complete(&mut self, now: Instant) -> Option<FrameDiagnostics> {
        let mut active = self.active.take()?;
        active.phase = FramePhase::Complete;
        let diagnostics = FrameDiagnostics {
            identity: active.identity,
            phase: active.phase,
            reason: active.reason,
            input_events: active.input_events,
            pointer_samples: active.pointer_samples,
            elapsed: active.budget.elapsed(now),
        };
        self.last_complete = Some(diagnostics);
        Some(diagnostics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_ids_increase_while_generations_are_independent() {
        let now = Instant::now();
        let mut frames = LogicalFrameCoordinator::default();
        assert_eq!(
            frames.request(EditGeneration::new(7), FrameRequestReason::Input),
            LogicalFrameId(1)
        );
        let first = frames.begin(now, 1, 0).unwrap();
        frames.complete(now);
        frames.request(
            EditGeneration::new(7),
            FrameRequestReason::RequiredEvaluation,
        );
        let second = frames.begin(now, 0, 0).unwrap();
        assert_eq!(first.generation, second.generation);
        assert!(second.id > first.id);
        frames.update_generation(EditGeneration::new(8));
        assert_eq!(frames.active_identity().unwrap().generation.get(), 8);
    }

    #[test]
    fn input_requested_after_seal_is_a_follow_up_frame() {
        let now = Instant::now();
        let mut frames = LogicalFrameCoordinator::default();
        frames.request(EditGeneration::new(2), FrameRequestReason::Input);
        let current = frames.begin(now, 1, 1).unwrap();
        let next = frames.request(EditGeneration::new(2), FrameRequestReason::Input);
        assert!(next > current.id);
        assert!(frames.has_pending());
    }

    #[test]
    fn pending_input_denies_optional_work() {
        let now = Instant::now();
        let mut frames = LogicalFrameCoordinator::default();
        frames.request(
            EditGeneration::new(1),
            FrameRequestReason::OptionalRefinement,
        );
        frames.begin(now, 0, 0);
        assert!(frames.can_start_optional(now));
        frames.request(EditGeneration::new(1), FrameRequestReason::Input);
        assert!(!frames.can_start_optional(now));
    }

    #[test]
    fn bounded_budget_is_a_start_gate() {
        let now = Instant::now();
        let budget = FrameWorkBudget::bounded(now, Duration::from_millis(5));
        assert!(budget.can_start_optional(now));
        assert!(!budget.can_start_optional(now + Duration::from_millis(5)));
        assert!(FrameWorkBudget::unbounded(now).can_start_optional(now + Duration::from_secs(60)));
    }

    #[test]
    fn input_promotes_an_existing_optional_request() {
        let now = Instant::now();
        let mut frames = LogicalFrameCoordinator::default();
        let id = frames.request(
            EditGeneration::new(3),
            FrameRequestReason::OptionalRefinement,
        );
        assert_eq!(
            frames.request(EditGeneration::new(4), FrameRequestReason::Input),
            id
        );
        frames.begin(now, 1, 1);
        let diagnostics = frames.complete(now).unwrap();
        assert_eq!(diagnostics.reason, FrameRequestReason::Input);
        assert_eq!(diagnostics.identity.generation.get(), 4);
    }

    #[test]
    fn deadlines_are_generation_aware_and_choose_the_earliest_wake() {
        let now = Instant::now();
        let mut frames = LogicalFrameCoordinator::default();
        let generation = EditGeneration::new(8);
        frames.schedule_deadline(
            FrameDeadlineKind::OptionalRefinement,
            generation,
            now + Duration::from_millis(80),
        );
        frames.schedule_deadline(
            FrameDeadlineKind::InteractiveEvaluation,
            generation,
            now + Duration::from_millis(40),
        );
        assert_eq!(
            frames.wake_decision(generation, now, false, None),
            FrameWake::WaitUntil(now + Duration::from_millis(40))
        );
        assert!(!frames.deadline_ready(FrameDeadlineKind::InteractiveEvaluation, generation, now));
        assert!(frames.deadline_ready(
            FrameDeadlineKind::InteractiveEvaluation,
            generation,
            now + Duration::from_millis(40)
        ));
        frames.discard_stale_deadlines(EditGeneration::new(9));
        assert_eq!(frames.next_deadline(EditGeneration::new(9)), None);
    }

    #[test]
    fn shutdown_abort_clears_pending_frame_presentation_and_deadlines() {
        let now = Instant::now();
        let mut frames = LogicalFrameCoordinator::default();
        frames.request(EditGeneration::new(1), FrameRequestReason::Input);
        frames.begin(now, 1, 1);
        frames.mark_presentation_requested();
        frames.schedule_deadline(
            FrameDeadlineKind::OptionalRefinement,
            EditGeneration::new(1),
            now + Duration::from_secs(1),
        );
        frames.request_shutdown(EditGeneration::new(2));
        assert!(frames.shutdown_requested());
        frames.abort(now);
        assert_eq!(frames.take_presentation_identity(), None);
        assert!(!frames.has_pending());
        assert_eq!(frames.next_deadline(EditGeneration::new(2)), None);
    }
}
