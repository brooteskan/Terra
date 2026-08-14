use serde::{Deserialize, Serialize};

/// Viewport / terrain interaction phases for progressive refinement.
///
/// Driven by meaningful scene changes (camera, terrain, lighting, ...), not raw
/// mouse-button or UI-hover state. Timings are owned by [`RefinementTimings`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EditorRefinementState {
    Interactive,
    Settling,
    Refining,
    Converged,
    Export,
}

impl EditorRefinementState {
    pub fn label(self) -> &'static str {
        match self {
            Self::Interactive => "Interactive",
            Self::Settling => "Settling",
            Self::Refining => "Refining",
            Self::Converged => "Converged",
            Self::Export => "Export",
        }
    }

    pub fn terrain_budget_us(self) -> u64 {
        match self {
            Self::Interactive => 2_000,
            Self::Settling => 5_000,
            Self::Refining => 10_000,
            Self::Converged => 1_000,
            Self::Export => u64::MAX,
        }
    }

    pub fn simulation_iteration_cap(self) -> Option<u32> {
        match self {
            Self::Interactive => Some(4),
            Self::Settling => Some(24),
            Self::Refining => Some(256),
            Self::Converged => Some(256),
            Self::Export => None,
        }
    }
}

/// Configurable hysteresis for [`RefinementController`] state transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RefinementTimings {
    /// Milliseconds without meaningful interaction before Interactive -> Settling.
    pub settle_after_ms: u64,
    /// Milliseconds without meaningful interaction before Settling -> Refining.
    pub refine_after_ms: u64,
}

impl Default for RefinementTimings {
    fn default() -> Self {
        Self {
            settle_after_ms: 75,
            refine_after_ms: 225,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RefinementController {
    state: EditorRefinementState,
    last_interaction_ms: u64,
    timings: RefinementTimings,
    render_converged: bool,
}

impl Default for RefinementController {
    fn default() -> Self {
        Self {
            state: EditorRefinementState::Converged,
            last_interaction_ms: 0,
            timings: RefinementTimings::default(),
            render_converged: false,
        }
    }
}

impl RefinementController {
    pub fn state(&self) -> EditorRefinementState {
        self.state
    }

    pub fn timings(&self) -> RefinementTimings {
        self.timings
    }

    pub fn set_timings(&mut self, timings: RefinementTimings) {
        self.timings = timings;
    }

    pub fn begin_interaction(&mut self, now_ms: u64) {
        self.last_interaction_ms = now_ms;
        self.render_converged = false;
        self.state = EditorRefinementState::Interactive;
    }

    pub fn begin_export(&mut self) {
        self.state = EditorRefinementState::Export;
    }

    pub fn finish_export(&mut self, now_ms: u64) {
        self.last_interaction_ms = now_ms;
        self.render_converged = false;
        self.state = EditorRefinementState::Settling;
    }

    pub fn set_render_converged(&mut self, converged: bool) {
        self.render_converged = converged;
    }

    pub fn render_converged(&self) -> bool {
        self.render_converged
    }

    pub fn update(&mut self, now_ms: u64, interaction_active: bool) -> EditorRefinementState {
        if self.state == EditorRefinementState::Export {
            return self.state;
        }
        if interaction_active {
            self.begin_interaction(now_ms);
        } else {
            let elapsed = now_ms.saturating_sub(self.last_interaction_ms);
            self.state = if elapsed < self.timings.settle_after_ms {
                EditorRefinementState::Interactive
            } else if elapsed < self.timings.refine_after_ms {
                EditorRefinementState::Settling
            } else if self.render_converged {
                EditorRefinementState::Converged
            } else {
                EditorRefinementState::Refining
            };
        }
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refinement_state_has_hysteresis() {
        let mut controller = RefinementController::default();
        controller.begin_interaction(100);
        assert_eq!(
            controller.update(120, false),
            EditorRefinementState::Interactive
        );
        assert_eq!(
            controller.update(200, false),
            EditorRefinementState::Settling
        );
        assert_eq!(
            controller.update(400, false),
            EditorRefinementState::Refining
        );
        controller.set_render_converged(true);
        assert_eq!(
            controller.update(401, false),
            EditorRefinementState::Converged
        );
    }

    #[test]
    fn ui_without_scene_change_does_not_restart() {
        let mut controller = RefinementController::default();
        controller.begin_interaction(0);
        assert_eq!(
            controller.update(300, false),
            EditorRefinementState::Refining
        );
        assert_eq!(
            controller.update(350, false),
            EditorRefinementState::Refining
        );
    }
}
