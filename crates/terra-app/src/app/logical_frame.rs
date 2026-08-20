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
    ScheduledWork,
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
}

impl LogicalFrameCoordinator {
    pub(crate) fn request(
        &mut self,
        generation: EditGeneration,
        reason: FrameRequestReason,
    ) -> LogicalFrameId {
        if let Some((id, _, _)) = self.pending {
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
        frames.request(EditGeneration::new(7), FrameRequestReason::ScheduledWork);
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
        frames.request(EditGeneration::new(1), FrameRequestReason::ScheduledWork);
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
}
