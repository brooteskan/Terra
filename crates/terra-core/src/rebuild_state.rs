//! Session-owned rebuild state and artist-facing DTOs (data only).
//!
//! The leaf half of the rebuild-feedback split (issue #84): plain data types and
//! their local state-machine methods, with no dependency on `document` or
//! `rebuild_feedback`. [`crate::document::EditorSession`] stores
//! [`RebuildFeedbackState`] directly from here; the document-aware behavior that
//! reads and mutates the session lives in [`crate::rebuild_feedback`], which also
//! re-exports these types so `rebuild_feedback::…` stays a stable public path.

use crate::deps::NodeRef;
use crate::domain::SoftDiagnostic;
use crate::layer::{BuildStatus, LayerId};
use crate::simulation_scenario::{ScenarioResultState, SimulationScenarioId};
use crate::world_rules::WorldRuleId;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Artist build states (unified presentation over layer / scenario / cache)
// ---------------------------------------------------------------------------

/// Compact build state shown to artists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ArtistBuildState {
    /// Definition and cached result match.
    #[default]
    Current,
    /// Project definition changed; needs attention (not yet a result badge).
    Dirty,
    /// Cached result invalidated by upstream change.
    Invalidated,
    /// Rebuild scheduled (debounce / queue).
    Queued,
    /// Currently evaluating.
    Building,
    /// Result kept but no longer matches upstream (sims, frozen-stale).
    Outdated,
    /// Artist froze this result — never auto-discard.
    Frozen,
    /// Last build failed.
    Failed,
    /// Layer / scenario disabled.
    Disabled,
}

impl ArtistBuildState {
    pub fn label(self) -> &'static str {
        match self {
            Self::Current => "Current",
            Self::Dirty => "Dirty",
            Self::Invalidated => "Invalidated",
            Self::Queued => "Queued",
            Self::Building => "Building",
            Self::Outdated => "Outdated",
            Self::Frozen => "Frozen",
            Self::Failed => "Failed",
            Self::Disabled => "Disabled",
        }
    }

    /// Distinguish definition dirtiness vs outdated cache vs in-flight vs frozen/failed.
    pub fn category(self) -> BuildStateCategory {
        match self {
            Self::Dirty | Self::Invalidated => BuildStateCategory::DirtyDefinition,
            Self::Outdated => BuildStateCategory::OutdatedResult,
            Self::Queued | Self::Building => BuildStateCategory::Rebuilding,
            Self::Frozen => BuildStateCategory::FrozenResult,
            Self::Failed => BuildStateCategory::FailedResult,
            Self::Current | Self::Disabled => BuildStateCategory::Stable,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuildStateCategory {
    DirtyDefinition,
    OutdatedResult,
    Rebuilding,
    FrozenResult,
    FailedResult,
    Stable,
}

impl From<BuildStatus> for ArtistBuildState {
    fn from(s: BuildStatus) -> Self {
        match s {
            BuildStatus::Idle => Self::Current,
            BuildStatus::Pending => Self::Dirty,
            BuildStatus::Computing => Self::Building,
            BuildStatus::Ready => Self::Current,
            BuildStatus::Outdated => Self::Outdated,
            BuildStatus::Error => Self::Failed,
        }
    }
}

impl From<ScenarioResultState> for ArtistBuildState {
    fn from(s: ScenarioResultState) -> Self {
        match s {
            ScenarioResultState::Ready | ScenarioResultState::Current => Self::Current,
            ScenarioResultState::Running => Self::Building,
            ScenarioResultState::Outdated => Self::Outdated,
            ScenarioResultState::Frozen => Self::Frozen,
            ScenarioResultState::Failed => Self::Failed,
            ScenarioResultState::Cancelled => Self::Disabled,
        }
    }
}

// ---------------------------------------------------------------------------
// Affected content
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AffectedId {
    Layer(LayerId),
    Group(LayerId),
    Scenario(SimulationScenarioId),
    WorldRule(WorldRuleId),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AffectedItem {
    pub id: AffectedId,
    pub name: String,
    pub state: ArtistBuildState,
}

impl AffectedItem {
    pub fn layer(id: LayerId, name: impl Into<String>, state: ArtistBuildState) -> Self {
        Self {
            id: AffectedId::Layer(id),
            name: name.into(),
            state,
        }
    }
}

/// Compact “Updating:” feedback after an upstream change (non-blocking).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AffectedContentFeedback {
    pub source_name: String,
    pub items: Vec<AffectedItem>,
    pub why: String,
}

impl AffectedContentFeedback {
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Multi-line artist summary (no dialogs).
    pub fn format_updating(&self) -> String {
        if self.items.is_empty() {
            return String::new();
        }
        let mut lines = vec!["Updating:".to_string()];
        for item in &self.items {
            lines.push(format!("- {}", item.name));
        }
        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// Why diagnostics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhyRebuild {
    pub title: String,
    pub diagnostics: Vec<SoftDiagnostic>,
    pub depends_on: Vec<(String, NodeRef)>,
    pub used_by: Vec<(String, NodeRef)>,
}

impl WhyRebuild {
    pub fn summary(&self) -> String {
        self.diagnostics
            .first()
            .map(|d| d.message.clone())
            .unwrap_or_else(|| self.title.clone())
    }
}

// ---------------------------------------------------------------------------
// Preferences & debounce
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebuildPrefs {
    /// When false (default), expensive physics sims do not auto-rebuild on upstream edits.
    pub automatic_rebuild_expensive: bool,
    /// Draft-quality live preview while editing (shape / sculpt).
    pub live_preview: bool,
    /// Debounce for interactive terrain rebuilds (ms).
    pub edit_debounce_ms: u64,
    /// Extra debounce before queuing expensive physics (ms).
    pub physics_debounce_ms: u64,
}

impl Default for RebuildPrefs {
    fn default() -> Self {
        Self {
            automatic_rebuild_expensive: false,
            live_preview: true,
            edit_debounce_ms: 40,
            physics_debounce_ms: 600,
        }
    }
}

/// Session-owned rebuild feedback (not project JSON).
#[derive(Debug, Clone, Default)]
pub struct RebuildFeedbackState {
    pub prefs: RebuildPrefs,
    pub last_feedback: Option<AffectedContentFeedback>,
    pub queued: Vec<AffectedId>,
    pub building: Vec<AffectedId>,
    /// Monotonic clock ms of last interactive edit (sculpt / shape).
    pub last_edit_ms: u64,
    /// When Some, physics rebuild may fire after this time if auto-rebuild is on.
    pub physics_due_ms: Option<u64>,
    /// Why-outdated cache for the last queried node.
    pub last_why: Option<WhyRebuild>,
}

impl RebuildFeedbackState {
    pub fn record_edit(&mut self, now_ms: u64) {
        self.last_edit_ms = now_ms;
        if self.prefs.automatic_rebuild_expensive {
            self.physics_due_ms = Some(now_ms.saturating_add(self.prefs.physics_debounce_ms));
        } else {
            self.physics_due_ms = None;
        }
    }

    /// Whether expensive physics should run now (never while actively sculpting).
    pub fn should_rebuild_physics(&self, now_ms: u64, sculpting: bool) -> bool {
        if sculpting {
            return false;
        }
        if !self.prefs.automatic_rebuild_expensive {
            return false;
        }
        match self.physics_due_ms {
            Some(due) => now_ms >= due,
            None => false,
        }
    }

    /// Draft interactive rebuild allowed (live preview + debounce elapsed).
    pub fn should_draft_rebuild(&self, now_ms: u64, sculpting: bool) -> bool {
        if !self.prefs.live_preview {
            return false;
        }
        // While sculpting, paint path owns draft frames — don't double-schedule.
        if sculpting {
            return false;
        }
        now_ms.saturating_sub(self.last_edit_ms) >= self.prefs.edit_debounce_ms
    }

    pub fn clear_physics_due(&mut self) {
        self.physics_due_ms = None;
    }
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RebuildFeedbackAction {
    Rebuild {
        id: AffectedId,
    },
    RebuildAffected {
        source: NodeRef,
    },
    KeepFrozen {
        scenario: SimulationScenarioId,
    },
    PreviewOldResult {
        scenario: SimulationScenarioId,
        snapshot: uuid::Uuid,
    },
    DisableAutomaticRebuild,
    EnableAutomaticRebuild,
    EnableLivePreview,
    DisableLivePreview,
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn format_updating_compact() {
        let fb = AffectedContentFeedback {
            source_name: "Shape".into(),
            items: vec![
                AffectedItem::layer(
                    LayerId::new(),
                    "Alpine placement",
                    ArtistBuildState::Outdated,
                ),
                AffectedItem::layer(LayerId::new(), "Hydrology", ArtistBuildState::Outdated),
            ],
            why: "shape edited".into(),
        };
        let s = fb.format_updating();
        assert!(s.contains("Updating:"));
        assert!(s.contains("Alpine placement"));
        assert!(s.contains("Hydrology"));
    }

    #[test]
    fn states_distinguish_categories() {
        assert_eq!(
            ArtistBuildState::Dirty.category(),
            BuildStateCategory::DirtyDefinition
        );
        assert_eq!(
            ArtistBuildState::Outdated.category(),
            BuildStateCategory::OutdatedResult
        );
        assert_eq!(
            ArtistBuildState::Building.category(),
            BuildStateCategory::Rebuilding
        );
        assert_eq!(
            ArtistBuildState::Frozen.category(),
            BuildStateCategory::FrozenResult
        );
        assert_eq!(
            ArtistBuildState::Failed.category(),
            BuildStateCategory::FailedResult
        );
    }
}
