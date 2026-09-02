//! Persistent interaction state across frames.

use std::collections::HashMap;

use crate::id::Id;
use crate::layout_prefs::LayoutPrefs;

#[derive(Debug, Clone, Copy)]
pub struct WindowDrag {
    pub id: Id,
    /// Pointer offset from window top-left when drag started.
    pub grab_x: f32,
    pub grab_y: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct ScrollDrag {
    pub id: Id,
    pub vertical: bool,
    pub start_scroll: f32,
    pub start_pointer: f32,
    pub scroll_per_pixel: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitterKind {
    ToolPanel,
    RightPanel,
    LayersInspector,
}

#[derive(Debug, Clone, Copy)]
pub struct SplitterDrag {
    pub kind: SplitterKind,
    pub start_pointer: f32,
    pub start_value: f32,
}

#[derive(Debug, Default)]
pub struct GuiState {
    pub hot: Option<Id>,
    pub active: Option<Id>,
    pub open_combo: Option<Id>,
    /// Combo selection applied on the next `combo()` call (set while drawing the menu).
    pub combo_pick: Option<(Id, usize)>,
    /// Vertical offsets for long combo menus, keyed by combo id.
    pub combo_scroll: HashMap<u64, f32>,
    /// Previous frame's primary button (for edge detection).
    pub was_primary_down: bool,
    /// Previous frame's secondary (right) button (for edge detection).
    pub was_secondary_down: bool,
    /// Persisted floating-window top-left positions (logical px), keyed by window id.
    pub window_pos: HashMap<u64, (f32, f32)>,
    pub window_drag: Option<WindowDrag>,
    pub scroll_drag: Option<ScrollDrag>,
    /// Last frame's max vertical scroll per region id (for gating wheel input).
    pub scroll_max: HashMap<u64, f32>,
    /// Live dock layout (mirrored from the app / UiState each frame).
    pub layout: LayoutPrefs,
    /// Active panel splitter drag.
    pub splitter_drag: Option<SplitterDrag>,
    /// Focused text field (numeric edit, rename, search).
    pub text_focus: Option<Id>,
    /// Buffer for the focused text field.
    pub text_buffer: String,
    /// True when Enter was pressed while a text field is focused (consumed by widgets).
    pub text_enter: bool,
}

impl GuiState {
    pub fn wants_pointer(&self) -> bool {
        self.hot.is_some()
            || self.active.is_some()
            || self.open_combo.is_some()
            || self.scroll_drag.is_some()
            || self.splitter_drag.is_some()
            || self.text_focus.is_some()
    }

    /// True while the user is actively capturing the pointer (drag, scroll,
    /// combo, text) — excludes mere hover so background work can continue.
    pub fn is_interacting(&self) -> bool {
        self.active.is_some()
            || self.open_combo.is_some()
            || self.scroll_drag.is_some()
            || self.splitter_drag.is_some()
            || self.text_focus.is_some()
    }

    pub fn wants_text_input(&self) -> bool {
        self.text_focus.is_some()
    }

    pub fn is_hot(&self, id: Id) -> bool {
        self.hot == Some(id)
    }

    pub fn is_active(&self, id: Id) -> bool {
        self.active == Some(id)
    }

    pub fn set_hot(&mut self, id: Id) {
        self.hot = Some(id);
    }

    pub fn clear_text_focus(&mut self) {
        self.text_focus = None;
        self.text_buffer.clear();
        self.text_enter = false;
    }
}
