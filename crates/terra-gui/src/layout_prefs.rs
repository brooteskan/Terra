//! Runtime dock layout (resizable panels + collapse flags).
//!
//! Defaults are product-neutral; apps may override via persisted prefs.

use serde::{Deserialize, Serialize};

use crate::style::RIGHT_PANEL_W;

/// Default left rail width (mode/navigation column) for IDE-style shells.
const DEFAULT_LEFT_RAIL_W: f32 = 72.0;
/// Default contextual tool panel width.
const DEFAULT_TOOL_PANEL_W: f32 = 228.0;
/// Default full-height left dock width.
const DEFAULT_LEFT_DOCK_W: f32 = 320.0;

/// Persisted / live dock layout for an IDE-style shell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutPrefs {
    /// Contextual tool panel width (logical px). Ignored when collapsed.
    pub tool_panel_w: f32,
    /// Right rail (layers + inspector) width.
    pub right_panel_w: f32,
    /// Fraction of the right rail height for the layers stack (0–1).
    pub layers_frac: f32,
    /// When true, the contextual tool panel is hidden (mode rail remains when visible).
    pub tool_panel_collapsed: bool,
    /// When true, the inspector is collapsed and layers take the full right rail.
    pub inspector_collapsed: bool,
    /// Runtime: hide the mode rail. Not persisted meaningfully.
    #[serde(default)]
    pub hide_mode_rail: bool,
    /// Runtime: hide the layers tree. Not persisted meaningfully.
    #[serde(default)]
    pub hide_layers_panel: bool,
    /// Width of the full-height left dock. Persisted across sessions.
    #[serde(default = "default_mask_editor_panel_w")]
    pub mask_editor_panel_w: f32,
    /// Runtime: when set, left chrome is a single full-height dock of this width
    /// (mode rail + tool panel hidden).
    #[serde(default, skip_serializing)]
    pub left_dock_w: Option<f32>,
}

fn default_mask_editor_panel_w() -> f32 {
    DEFAULT_LEFT_DOCK_W
}

impl Default for LayoutPrefs {
    fn default() -> Self {
        Self {
            tool_panel_w: DEFAULT_TOOL_PANEL_W,
            right_panel_w: RIGHT_PANEL_W,
            layers_frac: 0.52,
            tool_panel_collapsed: false,
            inspector_collapsed: false,
            hide_mode_rail: false,
            hide_layers_panel: false,
            mask_editor_panel_w: DEFAULT_LEFT_DOCK_W,
            left_dock_w: None,
        }
    }
}

impl LayoutPrefs {
    pub const TOOL_PANEL_MIN: f32 = 140.0;
    pub const TOOL_PANEL_MAX: f32 = 280.0;
    pub const MASK_EDITOR_MIN: f32 = 260.0;
    pub const MASK_EDITOR_MAX: f32 = 420.0;
    pub const RIGHT_PANEL_MIN: f32 = 280.0;
    pub const RIGHT_PANEL_MAX: f32 = 520.0;
    pub const LAYERS_FRAC_MIN: f32 = 0.25;
    pub const LAYERS_FRAC_MAX: f32 = 0.80;

    pub fn clamp_mut(&mut self) {
        self.tool_panel_w = self
            .tool_panel_w
            .clamp(Self::TOOL_PANEL_MIN, Self::TOOL_PANEL_MAX);
        self.mask_editor_panel_w = self
            .mask_editor_panel_w
            .clamp(Self::MASK_EDITOR_MIN, Self::MASK_EDITOR_MAX);
        if let Some(w) = self.left_dock_w.as_mut() {
            *w = w.clamp(Self::MASK_EDITOR_MIN, Self::MASK_EDITOR_MAX);
        }
        self.right_panel_w = self
            .right_panel_w
            .clamp(Self::RIGHT_PANEL_MIN, Self::RIGHT_PANEL_MAX);
        self.layers_frac = self
            .layers_frac
            .clamp(Self::LAYERS_FRAC_MIN, Self::LAYERS_FRAC_MAX);
    }

    pub fn effective_tool_panel_w(&self) -> f32 {
        if self.left_dock_w.is_some() || self.tool_panel_collapsed {
            0.0
        } else {
            self.tool_panel_w
                .clamp(Self::TOOL_PANEL_MIN, Self::TOOL_PANEL_MAX)
        }
    }

    pub fn mode_rail_w(&self) -> f32 {
        if self.left_dock_w.is_some() || self.hide_mode_rail {
            0.0
        } else {
            DEFAULT_LEFT_RAIL_W
        }
    }

    pub fn left_chrome_w(&self) -> f32 {
        if let Some(w) = self.left_dock_w {
            return w.clamp(Self::MASK_EDITOR_MIN, Self::MASK_EDITOR_MAX);
        }
        self.mode_rail_w() + self.effective_tool_panel_w()
    }

    pub fn effective_right_w(&self) -> f32 {
        self.right_panel_w
            .clamp(Self::RIGHT_PANEL_MIN, Self::RIGHT_PANEL_MAX)
    }

    pub fn effective_layers_frac(&self) -> f32 {
        if self.hide_layers_panel {
            0.0
        } else if self.inspector_collapsed {
            1.0
        } else {
            self.layers_frac
                .clamp(Self::LAYERS_FRAC_MIN, Self::LAYERS_FRAC_MAX)
        }
    }

    /// Reset to design-token defaults.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Splitter hit thickness in logical px.
pub const SPLITTER_HIT: f32 = 5.0;
