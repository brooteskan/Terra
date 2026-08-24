//! Explicit lifecycle for GPU resources compiled on first presentation demand.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptionalResourceState {
    Absent,
    Compiling { request_id: u64 },
    Ready,
    Failed { request_id: u64, message: String },
}

pub struct OptionalResource<T> {
    state: OptionalResourceState,
    value: Option<T>,
}

impl<T> Default for OptionalResource<T> {
    fn default() -> Self {
        Self {
            state: OptionalResourceState::Absent,
            value: None,
        }
    }
}

impl<T> OptionalResource<T> {
    pub fn state(&self) -> &OptionalResourceState {
        &self.state
    }

    pub fn ready(&self) -> Option<&T> {
        self.value.as_ref()
    }

    pub fn ready_mut(&mut self) -> Option<&mut T> {
        self.value.as_mut()
    }

    pub fn begin(&mut self, request_id: u64) -> bool {
        if matches!(self.state, OptionalResourceState::Ready)
            || matches!(self.state, OptionalResourceState::Compiling { request_id: id } if id == request_id)
        {
            return false;
        }
        self.value = None;
        self.state = OptionalResourceState::Compiling { request_id };
        true
    }

    pub fn install(&mut self, request_id: u64, value: T) -> Result<(), T> {
        if !matches!(self.state, OptionalResourceState::Compiling { request_id: id } if id == request_id)
        {
            return Err(value);
        }
        self.value = Some(value);
        self.state = OptionalResourceState::Ready;
        Ok(())
    }

    pub fn set_ready(&mut self, value: T) {
        self.value = Some(value);
        self.state = OptionalResourceState::Ready;
    }

    pub fn fail(&mut self, request_id: u64, message: String) -> bool {
        if !matches!(self.state, OptionalResourceState::Compiling { request_id: id } if id == request_id)
        {
            return false;
        }
        self.value = None;
        self.state = OptionalResourceState::Failed {
            request_id,
            message,
        };
        true
    }

    pub fn reset(&mut self) {
        self.value = None;
        self.state = OptionalResourceState::Absent;
    }

    pub fn reset_unready(&mut self) {
        if !self.is_ready() {
            self.reset();
        }
    }

    pub fn is_ready(&self) -> bool {
        self.value.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transitions_coalesce_retry_and_reject_stale_results() {
        let mut slot = OptionalResource::default();
        assert_eq!(slot.state(), &OptionalResourceState::Absent);
        assert!(slot.begin(1));
        assert!(!slot.begin(1));
        assert!(slot.fail(1, "failed".into()));
        assert!(matches!(slot.state(), OptionalResourceState::Failed { .. }));
        assert!(slot.begin(2));
        assert_eq!(slot.install(1, 10), Err(10));
        assert!(slot.install(2, 20).is_ok());
        assert_eq!(slot.ready(), Some(&20));
        assert!(!slot.begin(3));
        slot.reset();
        assert_eq!(slot.state(), &OptionalResourceState::Absent);
    }

    #[test]
    fn absent_and_wrong_request_results_do_not_mutate_state() {
        let mut slot = OptionalResource::<u32>::default();
        assert_eq!(slot.install(4, 9), Err(9));
        assert!(!slot.fail(4, "late".into()));
        assert_eq!(slot.state(), &OptionalResourceState::Absent);
        assert!(slot.begin(5));
        assert_eq!(slot.install(6, 10), Err(10));
        assert!(!slot.fail(6, "stale".into()));
        assert_eq!(
            slot.state(),
            &OptionalResourceState::Compiling { request_id: 5 }
        );
    }

    #[test]
    fn project_reset_preserves_ready_but_clears_unready_resources() {
        let mut ready = OptionalResource::default();
        ready.set_ready(3u32);
        ready.reset_unready();
        assert_eq!(ready.ready(), Some(&3));

        let mut compiling = OptionalResource::<u32>::default();
        compiling.begin(8);
        compiling.reset_unready();
        assert_eq!(compiling.state(), &OptionalResourceState::Absent);

        let mut failed = OptionalResource::<u32>::default();
        failed.begin(9);
        failed.fail(9, "bad shader".into());
        failed.reset_unready();
        assert_eq!(failed.state(), &OptionalResourceState::Absent);
    }
}
