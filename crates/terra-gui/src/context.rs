//! Per-frame GUI context (immediate-mode builder).

use std::collections::HashMap;

use crate::draw::{DrawCmd, DrawList};
use crate::font;
use crate::icons::Icon;
use crate::id::Id;
use crate::layout::Layout;
use crate::layout_prefs::{LayoutPrefs, SPLITTER_HIT};
use crate::scroll;
use crate::state::{GuiState, SplitterDrag, SplitterKind, WindowDrag};
use crate::style;
use crate::types::{Align, Color, Rect};
use crate::widgets::{self, WidgetLabState};

/// Default dock insets (logical px) — left tool chrome, right inspector, title + status.
pub const INSET_LEFT: f32 = 300.0; // 72 + 228 default left chrome
pub const INSET_RIGHT: f32 = style::RIGHT_PANEL_W;
pub const INSET_TOP: f32 = style::APP_BAR_H;
/// Default bottom inset (status only). Prefer [`GuiContext::bottom_inset`].
pub const INSET_BOTTOM: f32 = style::STATUS_STRIP_H;
/// Fraction of the right rail height reserved for the layers stack (rest = inspector).
pub const RIGHT_LAYERS_FRAC: f32 = 0.52;
pub const WINDOW_TITLE_H: f32 = 28.0;

#[derive(Debug, Clone)]
pub struct GuiInput {
    pub pointer: Option<(f32, f32)>,
    pub primary_down: bool,
    /// Set by `GuiContext::begin` from `GuiState::was_primary_down`.
    pub primary_pressed: bool,
    pub primary_released: bool,
    pub secondary_down: bool,
    /// Set by `GuiContext::begin` from `GuiState::was_secondary_down`.
    pub secondary_pressed: bool,
    pub secondary_released: bool,
    /// Vertical wheel delta (positive = scroll content up / finger up).
    pub scroll_delta: f32,
    /// Text typed since the previous frame, for lightweight editor search fields.
    pub text: String,
    /// Backspace pressed since the previous frame.
    pub backspace_pressed: bool,
    /// Escape pressed since the previous frame.
    pub escape_pressed: bool,
    /// Enter / Return pressed since the previous frame.
    pub enter_pressed: bool,
}

impl Default for GuiInput {
    fn default() -> Self {
        Self {
            pointer: None,
            primary_down: false,
            primary_pressed: false,
            primary_released: false,
            secondary_down: false,
            secondary_pressed: false,
            secondary_released: false,
            scroll_delta: 0.0,
            text: String::new(),
            backspace_pressed: false,
            escape_pressed: false,
            enter_pressed: false,
        }
    }
}

pub struct GuiContext<'a> {
    pub screen_w: f32,
    pub screen_h: f32,
    pub pixels_per_point: f32,
    pub input: GuiInput,
    pub draw: DrawList,
    /// Popups / menus — always composited after `draw`.
    pub overlay: DrawList,
    pub state: &'a mut GuiState,
    layout: Option<Layout>,
    clip: Option<Rect>,
    drawing_overlay: bool,
    pending_combo_menu: Option<PendingComboMenu>,
    pending_tooltips: Vec<PendingTooltip>,
    active_scroll: Option<ActiveScroll>,
    scroll_consumed: bool,
    overlay_stack: Vec<OverlayFrame>,
    /// RGBA images queued this frame; packed into one atlas at render time.
    pub images: Vec<(u32, u32, Vec<u8>)>,
    /// Frame-local content keys → `images` index (dedupe identical thumbs).
    image_keys: HashMap<u64, u32>,
    /// When set, background chrome has pointer edges cleared so open menus
    /// do not click-through. Restored by [`Self::with_menu_input`].
    pointer_edge_stash: Option<(bool, bool)>,
}

struct PendingComboMenu {
    id: Id,
    rect: Rect,
    /// The combo's own field. The press that opens a combo lands here, so it must
    /// be excluded from the click-outside dismiss (mirrors a menu's anchor).
    owner: Rect,
    items: Vec<String>,
    selected: usize,
}

struct PendingTooltip {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    title: String,
    body: String,
    shortcut: Option<String>,
}

struct ActiveScroll {
    id: Id,
    viewport: Rect,
    scroll_at_begin: f32,
    /// Matches the layout pad used when the panel was opened (bottom scroll extent).
    content_pad: f32,
}

/// Saved panel layout/clip so overlays can nest without destroying `allocate()`.
struct OverlayFrame {
    layout: Option<Layout>,
    clip: Option<Rect>,
    was_drawing_overlay: bool,
}

impl<'a> GuiContext<'a> {
    pub fn begin(
        screen_w: f32,
        screen_h: f32,
        pixels_per_point: f32,
        mut input: GuiInput,
        state: &'a mut GuiState,
    ) -> Self {
        input.primary_pressed = input.primary_down && !state.was_primary_down;
        input.primary_released = !input.primary_down && state.was_primary_down;
        input.secondary_pressed = input.secondary_down && !state.was_secondary_down;
        input.secondary_released = !input.secondary_down && state.was_secondary_down;
        // Fresh hover each frame (last-wins as widgets run).
        state.hot = None;
        let pixels_per_point = pixels_per_point.max(0.5);
        // Bake glyphs at the framebuffer pixel size for this DPI.
        font::prepare(pixels_per_point);
        if input.primary_released {
            // Keep open_combo; clear active after widgets process click this frame.
        }
        Self {
            screen_w: screen_w.max(1.0),
            screen_h: screen_h.max(1.0),
            pixels_per_point,
            input,
            draw: DrawList::default(),
            overlay: DrawList::default(),
            state,
            layout: None,
            clip: None,
            drawing_overlay: false,
            pending_combo_menu: None,
            pending_tooltips: Vec::new(),
            active_scroll: None,
            scroll_consumed: false,
            overlay_stack: Vec::new(),
            images: Vec::new(),
            image_keys: HashMap::new(),
            pointer_edge_stash: None,
        }
    }

    /// Content width of the active panel layout (0 if none).
    pub fn content_width(&self) -> f32 {
        self.layout
            .as_ref()
            .map(|l| l.content.width())
            .unwrap_or(0.0)
    }

    /// Suppress `primary_pressed` / `primary_released` for widgets under an open menu.
    /// Call after drawing the menu bar (which still needs those edges to open/switch menus).
    pub fn suspend_pointer_edges(&mut self) {
        if self.pointer_edge_stash.is_some() {
            return;
        }
        self.pointer_edge_stash = Some((self.input.primary_pressed, self.input.primary_released));
        self.input.primary_pressed = false;
        self.input.primary_released = false;
    }

    /// Run `f` with the real pointer edges restored (for the open menu / popup itself).
    pub fn with_menu_input<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        let stashed = self.pointer_edge_stash;
        if let Some((pressed, released)) = stashed {
            self.input.primary_pressed = pressed;
            self.input.primary_released = released;
        }
        let out = f(self);
        if stashed.is_some() {
            self.input.primary_pressed = false;
            self.input.primary_released = false;
            self.pointer_edge_stash = stashed;
        }
        out
    }

    /// Route subsequent draw/hit widgets into the overlay layer (no clip).
    /// Preserves the active panel layout so callers can resume `allocate` after `end_overlay`.
    pub fn begin_overlay(&mut self) {
        self.overlay_stack.push(OverlayFrame {
            layout: self.layout.take(),
            clip: self.clip,
            was_drawing_overlay: self.drawing_overlay,
        });
        // Overlay content is unclipped; clear scissor on both lists while drawing it.
        self.draw.set_clip(None);
        self.overlay.set_clip(None);
        self.layout = None;
        self.clip = None;
        self.drawing_overlay = true;
    }

    pub fn end_overlay(&mut self) {
        let Some(frame) = self.overlay_stack.pop() else {
            self.layout = None;
            self.clip = None;
            self.drawing_overlay = false;
            self.draw.set_clip(None);
            self.overlay.set_clip(None);
            return;
        };
        self.layout = frame.layout;
        self.clip = frame.clip;
        self.drawing_overlay = frame.was_drawing_overlay;
        let list = if self.drawing_overlay {
            &mut self.overlay
        } else {
            &mut self.draw
        };
        list.set_clip(self.clip);
    }

    /// Call once after building UI for the frame.
    pub fn end(&mut self) {
        let had_combo_menu = self.pending_combo_menu.is_some();
        if let Some(menu) = self.pending_combo_menu.take() {
            self.with_menu_input(|ui| {
                ui.begin_overlay();
                widgets::draw_combo_menu(ui, menu.id, menu.rect, &menu.items, menu.selected);
                // Match context menus: dismiss on press outside the popup.
                // Must run inside `with_menu_input` — `suspend_pointer_edges`
                // clears `primary_pressed` for background chrome while open.
                // Exclude the owner field too: the very press that opens the
                // combo lands on it, and would otherwise close it the same frame.
                if (ui.input.primary_pressed || ui.input.secondary_pressed)
                    && !ui.pointer_in(menu.rect)
                    && !ui.pointer_in(menu.owner)
                {
                    ui.state.open_combo = None;
                }
                ui.end_overlay();
            });
        }
        let tips = std::mem::take(&mut self.pending_tooltips);
        if !tips.is_empty() {
            self.begin_overlay();
            for tip in tips {
                flush_tooltip(self, tip);
            }
            self.end_overlay();
        }
        if self.input.enter_pressed {
            self.state.text_enter = true;
        }
        // Prefer button-up over the released edge: `suspend_pointer_edges` can
        // swallow `primary_released` while menus/popups are open, which would
        // leave scroll/splitter drags sticky.
        if !self.input.primary_down {
            self.state.active = None;
            self.state.window_drag = None;
            self.state.scroll_drag = None;
            self.state.splitter_drag = None;
        }
        if self.input.escape_pressed {
            self.state.open_combo = None;
        }
        // Owner scrolled away / not drawn: no popup to hit-test — any press closes.
        if !had_combo_menu {
            let primary = self
                .pointer_edge_stash
                .map(|(p, _)| p)
                .unwrap_or(self.input.primary_pressed);
            if (primary || self.input.secondary_pressed) && self.state.open_combo.is_some() {
                self.state.open_combo = None;
            }
        }
        self.state.was_primary_down = self.input.primary_down;
        self.state.was_secondary_down = self.input.secondary_down;
    }

    /// Queue a tooltip to draw on the overlay after all chrome (safe from scroll clips).
    pub fn queue_tooltip(&mut self, anchor: Rect, title: &str, body: &str, shortcut: Option<&str>) {
        let max_w = 280.0;
        let title_w = DrawList::text_width(title, style::FONT_SCALE);
        let body_empty = body.is_empty();
        let body_w = if body_empty {
            0.0
        } else {
            DrawList::text_width(body, style::FONT_SCALE * 0.85).min(max_w - 16.0)
        };
        let sc_w = shortcut
            .map(|s| DrawList::text_width(s, style::FONT_SCALE * 0.8) + 24.0)
            .unwrap_or(0.0);
        let w = (title_w.max(body_w) + sc_w + 20.0).clamp(80.0, max_w);
        let h = if shortcut.is_some() {
            54.0
        } else if body_empty {
            28.0
        } else {
            44.0
        };
        // Prefer to the right of the anchor; flip left if we would leave the screen.
        let mut x = anchor.max_x + 8.0;
        if x + w > self.screen_w - 8.0 {
            x = (anchor.min_x - w - 8.0).max(8.0);
        }
        let y = anchor.min_y.max(style::APP_BAR_H + 4.0);
        self.pending_tooltips.push(PendingTooltip {
            x,
            y,
            w,
            h,
            title: title.to_string(),
            body: body.to_string(),
            shortcut: shortcut.map(|s| s.to_string()),
        });
    }

    pub fn wants_pointer(&self) -> bool {
        self.state.wants_pointer() || self.state.window_drag.is_some()
    }

    /// Active pointer capture (not hover). Safe to refine preview underneath.
    pub fn is_interacting(&self) -> bool {
        self.state.is_interacting() || self.state.window_drag.is_some()
    }

    pub fn layout(&self) -> &LayoutPrefs {
        &self.state.layout
    }

    pub fn layout_mut(&mut self) -> &mut LayoutPrefs {
        &mut self.state.layout
    }

    /// Central 3D viewport in logical pixels (between side panels / menu).
    pub fn bottom_inset(&self) -> f32 {
        style::STATUS_STRIP_H
    }

    pub fn viewport_rect(&self) -> Rect {
        Self::viewport_rect_for(self.screen_w, self.screen_h, &self.state.layout)
    }

    pub fn viewport_rect_for(screen_w: f32, screen_h: f32, layout: &LayoutPrefs) -> Rect {
        let left = layout.left_chrome_w();
        let right = (screen_w - layout.effective_right_w()).max(left + 32.0);
        let top = INSET_TOP;
        let bottom = (screen_h - style::STATUS_STRIP_H).max(top + 32.0);
        Rect::from_min_max(left, top, right, bottom)
    }

    /// Full left chrome (mode rail + contextual tool panel).
    pub fn left_panel_rect(&self) -> Rect {
        Rect::from_min_max(
            0.0,
            INSET_TOP,
            self.state.layout.left_chrome_w(),
            self.screen_h - self.bottom_inset(),
        )
    }

    /// Far-left workspace mode selector (zero-width when hidden).
    pub fn mode_rail_rect(&self) -> Rect {
        let w = self.state.layout.mode_rail_w();
        Rect::from_min_max(0.0, INSET_TOP, w, self.screen_h - self.bottom_inset())
    }

    /// Contextual tools for the active intent / Advanced mode.
    pub fn tool_panel_rect(&self) -> Rect {
        let rail = self.state.layout.mode_rail_w();
        let w = self.state.layout.effective_tool_panel_w();
        Rect::from_min_max(
            rail,
            INSET_TOP,
            rail + w,
            self.screen_h - self.bottom_inset(),
        )
    }

    pub fn right_panel_rect(&self) -> Rect {
        let rw = self.state.layout.effective_right_w();
        Rect::from_min_max(
            self.screen_w - rw,
            INSET_TOP,
            self.screen_w,
            self.screen_h - self.bottom_inset(),
        )
    }

    /// Top half of the right rail — Layers list (empty when demoted for artist intents).
    pub fn right_layers_rect(&self) -> Rect {
        let full = self.right_panel_rect();
        if self.state.layout.hide_layers_panel {
            return Rect::from_min_max(full.min_x, full.min_y, full.max_x, full.min_y);
        }
        if self.state.layout.inspector_collapsed {
            // Leave the collapsed inspector header strip so the Layers footer
            // (add / duplicate / delete) stays visible above it.
            return Rect::from_min_max(
                full.min_x,
                full.min_y,
                full.max_x,
                full.max_y - style::HEADER_H,
            );
        }
        let frac = self.state.layout.effective_layers_frac();
        let mid = full.min_y + (full.height() * frac).max(120.0);
        let mid = mid.min(full.max_y - 160.0);
        Rect::from_min_max(full.min_x, full.min_y, full.max_x, mid)
    }

    /// Bottom half of the right rail — Inspector (full height when layers demoted).
    pub fn right_inspector_rect(&self) -> Rect {
        let full = self.right_panel_rect();
        if self.state.layout.hide_layers_panel {
            return full;
        }
        if self.state.layout.inspector_collapsed {
            // Keep a thin strip so the expand control remains reachable.
            return Rect::from_min_max(
                full.min_x,
                full.max_y - style::HEADER_H,
                full.max_x,
                full.max_y,
            );
        }
        let layers = self.right_layers_rect();
        Rect::from_min_max(full.min_x, layers.max_y, full.max_x, full.max_y)
    }

    /// Draw and interact with dock splitters. Call once per frame after panels.
    /// Returns `true` when layout changed this frame.
    pub fn draw_layout_splitters(&mut self) -> bool {
        let mut changed = false;
        let screen_w = self.screen_w;
        let top = INSET_TOP;
        let bottom = self.screen_h - self.bottom_inset();

        // Continue active drag.
        if let Some(drag) = self.state.splitter_drag {
            if let Some((px, py)) = self.input.pointer {
                match drag.kind {
                    SplitterKind::ToolPanel => {
                        let delta = px - drag.start_pointer;
                        if self.state.layout.left_dock_w.is_some() {
                            let w = (drag.start_value + delta)
                                .clamp(LayoutPrefs::MASK_EDITOR_MIN, LayoutPrefs::MASK_EDITOR_MAX);
                            if (w - self.state.layout.mask_editor_panel_w).abs() > 0.5 {
                                self.state.layout.mask_editor_panel_w = w;
                                self.state.layout.left_dock_w = Some(w);
                                changed = true;
                            }
                        } else {
                            let w = (drag.start_value + delta)
                                .clamp(LayoutPrefs::TOOL_PANEL_MIN, LayoutPrefs::TOOL_PANEL_MAX);
                            if self.state.layout.tool_panel_collapsed {
                                self.state.layout.tool_panel_collapsed = false;
                            }
                            if (w - self.state.layout.tool_panel_w).abs() > 0.5 {
                                self.state.layout.tool_panel_w = w;
                                changed = true;
                            }
                        }
                    }
                    SplitterKind::RightPanel => {
                        let delta = drag.start_pointer - px;
                        let w = (drag.start_value + delta)
                            .clamp(LayoutPrefs::RIGHT_PANEL_MIN, LayoutPrefs::RIGHT_PANEL_MAX);
                        if (w - self.state.layout.right_panel_w).abs() > 0.5 {
                            self.state.layout.right_panel_w = w;
                            changed = true;
                        }
                    }
                    SplitterKind::LayersInspector => {
                        if self.state.layout.hide_layers_panel {
                            // Ignore drag while layers are demoted.
                        } else {
                            let full = self.right_panel_rect();
                            let local = (py - full.min_y) / full.height().max(1.0);
                            let frac = local
                                .clamp(LayoutPrefs::LAYERS_FRAC_MIN, LayoutPrefs::LAYERS_FRAC_MAX);
                            if self.state.layout.inspector_collapsed {
                                self.state.layout.inspector_collapsed = false;
                            }
                            if (frac - self.state.layout.layers_frac).abs() > 0.002 {
                                self.state.layout.layers_frac = frac;
                                changed = true;
                            }
                        }
                    }
                }
            }
            self.state.set_hot(Id::new("__splitter_drag"));
            return changed;
        }

        // Left chrome | viewport splitter (tools panel, or full-height left dock).
        if self.state.layout.left_dock_w.is_some() || !self.state.layout.tool_panel_collapsed {
            let x = self.state.layout.left_chrome_w();
            let hit = Rect::from_pos_size(x - SPLITTER_HIT * 0.5, top, SPLITTER_HIT, bottom - top);
            let id = Id::new("split_tools");
            if self.pointer_in(hit) {
                self.state.set_hot(id);
                self.panel(hit, style::ACCENT_SOFT);
                if self.input.primary_pressed {
                    let start_value = self
                        .state
                        .layout
                        .left_dock_w
                        .unwrap_or(self.state.layout.tool_panel_w);
                    self.state.splitter_drag = Some(SplitterDrag {
                        kind: SplitterKind::ToolPanel,
                        start_pointer: self.input.pointer.map(|(x, _)| x).unwrap_or(x),
                        start_value,
                    });
                }
            }
        }

        // Viewport | right panel splitter
        {
            let x = screen_w - self.state.layout.effective_right_w();
            let hit = Rect::from_pos_size(x - SPLITTER_HIT * 0.5, top, SPLITTER_HIT, bottom - top);
            let id = Id::new("split_right");
            if self.pointer_in(hit) {
                self.state.set_hot(id);
                self.panel(hit, style::ACCENT_SOFT);
                if self.input.primary_pressed {
                    self.state.splitter_drag = Some(SplitterDrag {
                        kind: SplitterKind::RightPanel,
                        start_pointer: self.input.pointer.map(|(x, _)| x).unwrap_or(x),
                        start_value: self.state.layout.right_panel_w,
                    });
                }
            }
        }

        // Layers | inspector splitter (skipped when layers demoted for artist intents)
        if !self.state.layout.inspector_collapsed && !self.state.layout.hide_layers_panel {
            let layers = self.right_layers_rect();
            let y = layers.max_y;
            let hit = Rect::from_pos_size(
                layers.min_x,
                y - SPLITTER_HIT * 0.5,
                layers.width(),
                SPLITTER_HIT,
            );
            let id = Id::new("split_layers");
            if self.pointer_in(hit) {
                self.state.set_hot(id);
                self.panel(hit, style::ACCENT_SOFT);
                if self.input.primary_pressed {
                    self.state.splitter_drag = Some(SplitterDrag {
                        kind: SplitterKind::LayersInspector,
                        start_pointer: self.input.pointer.map(|(_, y)| y).unwrap_or(y),
                        start_value: self.state.layout.layers_frac,
                    });
                }
            }
        }

        changed
    }

    pub fn bottom_dock_rect(&self) -> Rect {
        Rect::from_min_max(
            0.0,
            self.screen_h - style::STATUS_STRIP_H,
            self.screen_w,
            self.screen_h,
        )
    }

    pub fn pointer_in(&self, rect: Rect) -> bool {
        let Some((x, y)) = self.input.pointer else {
            return false;
        };
        if !self.drawing_overlay {
            if let Some(clip) = self.clip {
                if !clip.contains(x, y) {
                    return false;
                }
            }
        }
        rect.contains(x, y)
    }

    /// Whether a rectangle can contribute pixels to the current clipped panel.
    /// Callers can use this before requesting expensive image or preview resources.
    pub fn rect_visible(&self, rect: Rect) -> bool {
        self.drawing_overlay
            || match self.clip {
                Some(clip) => rect.intersects(clip),
                None => true,
            }
    }

    fn push_panel(&mut self, rect: Rect, color: Color) {
        if self.drawing_overlay {
            self.overlay.panel(rect, color);
        } else {
            self.draw.panel(rect, color);
        }
    }

    pub fn panel(&mut self, rect: Rect, color: Color) {
        if !self.drawing_overlay {
            if let Some(clip) = self.clip {
                if !rect.intersects(clip) {
                    return;
                }
            }
        }
        self.push_panel(rect, color);
    }

    pub fn panel_rounded(&mut self, rect: Rect, color: Color, radius: f32) {
        if !self.drawing_overlay {
            if let Some(clip) = self.clip {
                if !rect.intersects(clip) {
                    return;
                }
            }
        }
        if self.drawing_overlay {
            self.overlay.panel_rounded(rect, color, radius);
        } else {
            self.draw.panel_rounded(rect, color, radius);
        }
    }

    pub fn label_at(&mut self, x: f32, y: f32, text: &str, color: Color, scale: f32) {
        if self.drawing_overlay {
            self.label_at_raw(x, y, text, color, scale);
            return;
        }
        // Keep text inside the active panel content / clip so long labels
        // cannot paint past the inspector / dock edge into the viewport.
        let right = self
            .layout
            .as_ref()
            .map(|l| l.content.max_x)
            .or_else(|| self.clip.map(|c| c.max_x));
        if let Some(max_x) = right {
            let max_w = (max_x - x).max(0.0);
            if DrawList::text_width(text, scale) > max_w {
                let drawn = DrawList::truncate_to_width(text, scale, max_w);
                let hit = Rect::from_pos_size(x, y, max_w, font::line_height(scale));
                if self.pointer_in(hit) {
                    self.queue_tooltip(hit, text, "", None);
                }
                self.label_at_raw(x, y, &drawn, color, scale);
                return;
            }
        }
        self.label_at_raw(x, y, text, color, scale);
    }

    pub(crate) fn label_at_raw(&mut self, x: f32, y: f32, text: &str, color: Color, scale: f32) {
        if !self.drawing_overlay {
            if let Some(clip) = self.clip {
                let h = font::line_height(scale);
                let w = DrawList::text_width(text, scale);
                // Cheap CPU cull; GPU scissor catches partial overflow.
                if y >= clip.max_y || y + h <= clip.min_y || x >= clip.max_x || x + w <= clip.min_x
                {
                    return;
                }
            }
        }
        if self.drawing_overlay {
            self.overlay.label(x, y, text, color, scale);
        } else {
            self.draw.label(x, y, text, color, scale);
        }
    }

    /// Draw left-aligned text vertically centred in `rect` (uses font line metrics).
    /// Text wider than `rect` is ellipsized so it cannot bleed into neighbouring columns.
    /// Returns `true` when the drawn text was truncated (hover shows a full-text tooltip).
    pub fn label_in_rect(&mut self, rect: Rect, text: &str, color: Color, scale: f32) -> bool {
        let y = font::text_top_in_row(rect.min_y, rect.height(), scale);
        let max_w = rect.width().max(0.0);
        let truncated = DrawList::text_width(text, scale) > max_w;
        let drawn = if truncated {
            DrawList::truncate_to_width(text, scale, max_w)
        } else {
            text.to_string()
        };
        self.label_at_raw(rect.min_x, y, &drawn, color, scale);
        if truncated && self.pointer_in(rect) {
            self.queue_tooltip(rect, text, "", None);
        }
        truncated
    }

    /// Draw centred text vertically centred in `rect`.
    pub fn label_centered_in_rect(&mut self, rect: Rect, text: &str, color: Color, scale: f32) {
        let y = font::text_top_in_row(rect.min_y, rect.height(), scale);
        self.label_centered(rect.center_x(), y, text, color, scale);
    }

    pub fn label_centered(&mut self, x: f32, y: f32, text: &str, color: Color, scale: f32) {
        if self.drawing_overlay {
            self.overlay
                .label_aligned(x, y, text, color, scale, Align::Center);
        } else {
            self.draw
                .label_aligned(x, y, text, color, scale, Align::Center);
        }
    }

    /// Centre an icon of `size` inside `rect`.
    pub fn icon_centered(&mut self, rect: Rect, icon: Icon, color: Color, size: f32) {
        let size = size.max(8.0);
        let x = rect.min_x + (rect.width() - size) * 0.5;
        let y = rect.min_y + (rect.height() - size) * 0.5;
        self.icon_at(x, y, icon, color, size);
    }

    /// Draw a Lucide icon at `(x, y)` with logical square size `size`.
    pub fn icon_at(&mut self, x: f32, y: f32, icon: Icon, color: Color, size: f32) {
        if !self.drawing_overlay {
            if let Some(clip) = self.clip {
                if y >= clip.max_y
                    || y + size <= clip.min_y
                    || x >= clip.max_x
                    || x + size <= clip.min_x
                {
                    return;
                }
            }
        }
        if self.drawing_overlay {
            self.overlay.icon(x, y, icon, color, size);
        } else {
            self.draw.icon(x, y, icon, color, size);
        }
    }

    pub fn begin_panel(&mut self, rect: Rect, color: Color) {
        self.open_panel(rect, color, 0.0, None);
    }

    /// Scrollable panel. Call [`end_panel_scrolled`] when finished so a scrollbar is drawn.
    pub fn begin_panel_scrolled(&mut self, id: Id, rect: Rect, color: Color, scroll_y: &mut f32) {
        self.begin_panel_scrolled_with_pad(id, rect, color, scroll_y, style::PAD);
    }

    /// Scrolled panel with explicit padding (inspector uses [`style::INSP_PAD`]).
    pub fn begin_panel_scrolled_padded(
        &mut self,
        id: Id,
        rect: Rect,
        color: Color,
        scroll_y: &mut f32,
        pad: f32,
    ) {
        self.begin_panel_scrolled_with_pad(id, rect, color, scroll_y, pad);
    }

    /// Scrollable panel with no content inset — rows sit flush against the shell edges.
    pub fn begin_panel_scrolled_flush(
        &mut self,
        id: Id,
        rect: Rect,
        color: Color,
        scroll_y: &mut f32,
    ) {
        self.begin_panel_scrolled_with_pad(id, rect, color, scroll_y, 0.0);
    }

    fn begin_panel_scrolled_with_pad(
        &mut self,
        id: Id,
        rect: Rect,
        color: Color,
        scroll_y: &mut f32,
        pad: f32,
    ) {
        // Apply in-progress thumb drag before laying out content (same-frame feedback).
        if let Some(drag) = self.state.scroll_drag {
            if !self.input.primary_down {
                self.state.scroll_drag = None;
            } else if drag.id == id && drag.vertical {
                if let Some((_, py)) = self.input.pointer {
                    let max = self
                        .state
                        .scroll_max
                        .get(&id.0)
                        .copied()
                        .unwrap_or(f32::MAX);
                    *scroll_y = (drag.start_scroll
                        + (py - drag.start_pointer) * drag.scroll_per_pixel)
                        .clamp(0.0, max);
                }
            }
        }
        self.apply_wheel_scroll_y(id, rect, scroll_y);
        self.open_panel_with_pad(rect, color, *scroll_y, Some(id), pad);
    }

    /// Apply mouse-wheel scrolling only when the region overflowed last frame.
    fn apply_wheel_scroll_y(&mut self, id: Id, viewport: Rect, scroll_y: &mut f32) {
        if self.scroll_consumed {
            return;
        }
        let hovering = self
            .input
            .pointer
            .map(|(x, y)| viewport.contains(x, y))
            .unwrap_or(false);
        if !hovering {
            return;
        }
        if self.input.scroll_delta.abs() < 1e-4 {
            return;
        }

        let max = self.state.scroll_max.get(&id.0).copied().unwrap_or(0.0);
        self.scroll_consumed = true;
        if max < 1.0 {
            // Content fits — keep offset at zero so layout doesn't jitter.
            *scroll_y = 0.0;
            return;
        }
        *scroll_y = (*scroll_y - self.input.scroll_delta * 24.0).clamp(0.0, max);
    }

    fn open_panel(&mut self, rect: Rect, color: Color, scroll_y: f32, scroll_id: Option<Id>) {
        self.open_panel_with_pad(rect, color, scroll_y, scroll_id, style::PAD);
    }

    fn open_panel_with_pad(
        &mut self,
        rect: Rect,
        color: Color,
        scroll_y: f32,
        scroll_id: Option<Id>,
        pad: f32,
    ) {
        // Background ignores content clip; content is GPU-scissored to `rect`.
        let list = if self.drawing_overlay {
            &mut self.overlay
        } else {
            &mut self.draw
        };
        list.panel(rect, color);
        list.set_clip(Some(rect));
        self.clip = Some(rect);
        let mut layout = Layout::with_pad(rect, pad);
        if scroll_id.is_some() {
            // Reserve a scrollbar gutter only when pad is too tight to host the thumb
            // inside the right inset — otherwise left/right content padding stay equal
            // and the bar paints in the right pad strip.
            let gutter = style::SCROLLBAR_W + style::SCROLLBAR_PAD;
            if pad + 0.5 < gutter {
                layout.content.max_x =
                    (layout.content.max_x - gutter).max(layout.content.min_x + 32.0);
            }
        }
        layout.cursor_y -= scroll_y.max(0.0);
        self.layout = Some(layout);
        self.active_scroll = scroll_id.map(|id| ActiveScroll {
            id,
            viewport: rect,
            scroll_at_begin: scroll_y.max(0.0),
            content_pad: pad,
        });
    }

    pub fn end_panel(&mut self) {
        let list = if self.drawing_overlay {
            &mut self.overlay
        } else {
            &mut self.draw
        };
        list.set_clip(None);
        self.clip = None;
        self.layout = None;
        self.active_scroll = None;
    }

    /// Finish a scrolled panel: clamp scroll and draw a vertical scrollbar when needed.
    pub fn end_panel_scrolled(&mut self, scroll_y: &mut f32) {
        let region = self.active_scroll.take();
        let cursor_y = self.layout_cursor_y();
        // Clear clip so the scrollbar is not scissored away at the edge.
        let list = if self.drawing_overlay {
            &mut self.overlay
        } else {
            &mut self.draw
        };
        list.set_clip(None);
        self.clip = None;
        self.layout = None;

        let Some(region) = region else {
            return;
        };
        let content_bottom = cursor_y.unwrap_or(region.viewport.max_y) + region.scroll_at_begin;
        let content_h = (content_bottom + region.content_pad - region.viewport.min_y)
            .max(region.viewport.height());
        let max_scroll = (content_h - region.viewport.height()).max(0.0);
        self.state.scroll_max.insert(region.id.0, max_scroll);
        if max_scroll < 1.0 {
            *scroll_y = 0.0;
        }
        scroll::scrollbar_y(self, region.id, region.viewport, content_h, scroll_y);
    }

    pub fn separator(&mut self) {
        let rect = self.allocate(style::SEPARATOR_H + style::GAP * 1.5);
        let line = Rect::from_pos_size(
            rect.min_x,
            rect.min_y + style::GAP * 0.75,
            rect.width(),
            style::SEPARATOR_H,
        );
        self.panel(line, style::BORDER);
    }

    pub fn gap(&mut self, h: f32) {
        let _ = self.allocate(h);
    }

    pub fn allocate(&mut self, height: f32) -> Rect {
        self.layout
            .as_mut()
            .expect("allocate without begin_panel")
            .allocate(height)
    }

    pub fn layout_cursor_y(&self) -> Option<f32> {
        self.layout.as_ref().map(|l| l.cursor_y)
    }

    pub fn widget_lab(&mut self, lab: &mut WidgetLabState) {
        widgets::widget_lab(self, lab);
    }

    /// Queue an RGBA image for the per-frame atlas (supports many thumbs + logo).
    /// Identical payloads within a frame reuse one atlas slot (no extra upload clone).
    pub fn image(&mut self, rect: Rect, width: u32, height: u32, rgba: &[u8]) {
        if !self.drawing_overlay {
            if let Some(clip) = self.clip {
                if !rect.intersects(clip) {
                    return;
                }
            }
        }
        let key = image_content_key(width, height, rgba);
        let image_id = if let Some(&id) = self.image_keys.get(&key) {
            id
        } else {
            let id = self.images.len() as u32;
            self.images.push((width, height, rgba.to_vec()));
            self.image_keys.insert(key, id);
            id
        };
        let list = if self.drawing_overlay {
            &mut self.overlay
        } else {
            &mut self.draw
        };
        list.cmds.push(DrawCmd::Image { rect, image_id });
    }

    /// Floating titled window that can be dragged by its title bar.
    /// `default_rect` is used the first time the window is opened; position is then persisted.
    pub fn begin_window(
        &mut self,
        id: crate::id::Id,
        title: &str,
        default_rect: Rect,
        open: &mut bool,
        scroll_y: &mut f32,
    ) -> bool {
        if !*open {
            return false;
        }

        let (mut x, mut y) = *self
            .state
            .window_pos
            .entry(id.0)
            .or_insert((default_rect.min_x, default_rect.min_y));
        let w = default_rect.width().max(160.0);
        let h = default_rect.height().max(120.0);

        // Title-bar drag (continues while primary is held, even off the bar).
        let drag_id = id.child("drag");
        if let Some(drag) = self.state.window_drag {
            if drag.id == id {
                if let Some((px, py)) = self.input.pointer {
                    x = px - drag.grab_x;
                    y = py - drag.grab_y;
                }
            }
        }

        // Keep the title bar reachable inside the app window.
        let max_x = (self.screen_w - w).max(0.0);
        let max_y = (self.screen_h - WINDOW_TITLE_H).max(0.0);
        x = x.clamp(0.0, max_x);
        y = y.clamp(0.0, max_y);
        self.state.window_pos.insert(id.0, (x, y));

        let rect = Rect::from_pos_size(x, y, w, h);
        let title_bar = Rect::from_pos_size(rect.min_x, rect.min_y, rect.width(), WINDOW_TITLE_H);
        let close = Rect::from_pos_size(rect.max_x - 26.0, rect.min_y + 4.0, 20.0, 20.0);

        if self
            .input
            .pointer
            .map(|(px, py)| rect.contains(px, py))
            .unwrap_or(false)
        {
            self.state.set_hot(id.child("bg"));
        }

        // Drag handle = title bar minus close button.
        let drag_zone = Rect::from_min_max(
            title_bar.min_x,
            title_bar.min_y,
            close.min_x,
            title_bar.max_y,
        );
        let drag_hov = self
            .input
            .pointer
            .map(|(px, py)| drag_zone.contains(px, py))
            .unwrap_or(false);
        if drag_hov {
            self.state.set_hot(drag_id);
        }
        if drag_hov && self.input.primary_pressed {
            self.state.active = Some(drag_id);
            if let Some((px, py)) = self.input.pointer {
                self.state.window_drag = Some(WindowDrag {
                    id,
                    grab_x: px - x,
                    grab_y: py - y,
                });
            }
        }

        self.push_panel(rect, style::PANEL_BG);
        self.push_panel(title_bar, Color::rgba(0.10, 0.12, 0.15, 0.98));
        // Accent underline under title.
        self.push_panel(
            Rect::from_pos_size(rect.min_x, title_bar.max_y - 1.0, rect.width(), 1.0),
            style::SEPARATOR,
        );
        self.label_at(
            title_bar.min_x + style::PAD,
            title_bar.min_y + 6.0,
            title,
            style::TEXT,
            style::FONT_SCALE,
        );

        let close_id = id.child("close");
        let close_hov = self
            .input
            .pointer
            .map(|(px, py)| close.contains(px, py))
            .unwrap_or(false);
        if close_hov {
            self.state.set_hot(close_id);
        }
        if close_hov && self.input.primary_pressed {
            self.state.active = Some(close_id);
        }
        if self.input.primary_released && self.state.is_active(close_id) && close_hov {
            *open = false;
            return false;
        }
        self.push_panel(
            close,
            if close_hov {
                style::BUTTON_HOVER
            } else {
                style::BUTTON_BG
            },
        );
        self.icon_at(
            close.min_x + 2.0,
            close.min_y + 2.0,
            Icon::X,
            style::TEXT,
            16.0,
        );

        let content = Rect::from_min_max(
            rect.min_x,
            rect.min_y + WINDOW_TITLE_H,
            rect.max_x,
            rect.max_y,
        );
        let scroll_id = id.child("scroll");
        if let Some(drag) = self.state.scroll_drag {
            if !self.input.primary_down {
                self.state.scroll_drag = None;
            } else if drag.id == scroll_id && drag.vertical {
                if let Some((_, py)) = self.input.pointer {
                    let max = self
                        .state
                        .scroll_max
                        .get(&scroll_id.0)
                        .copied()
                        .unwrap_or(f32::MAX);
                    *scroll_y = (drag.start_scroll
                        + (py - drag.start_pointer) * drag.scroll_per_pixel)
                        .clamp(0.0, max);
                }
            }
        }
        self.apply_wheel_scroll_y(scroll_id, content, scroll_y);

        let list = if self.drawing_overlay {
            &mut self.overlay
        } else {
            &mut self.draw
        };
        list.set_clip(Some(content));
        self.clip = Some(content);
        let mut layout = Layout::new(content);
        layout.content.max_x = (layout.content.max_x - style::SCROLLBAR_W - style::SCROLLBAR_PAD)
            .max(layout.content.min_x + 32.0);
        layout.cursor_y -= scroll_y.max(0.0);
        self.layout = Some(layout);
        self.active_scroll = Some(ActiveScroll {
            id: scroll_id,
            viewport: content,
            scroll_at_begin: scroll_y.max(0.0),
            content_pad: style::PAD,
        });
        true
    }

    pub fn end_window(&mut self, scroll_y: &mut f32) {
        self.end_panel_scrolled(scroll_y);
    }

    /// Draw a horizontal scrollbar for a free-form scrollable region (e.g. dock presets).
    pub fn scrollbar_x(&mut self, id: Id, viewport: Rect, content_w: f32, scroll_x: &mut f32) {
        scroll::scrollbar_x(self, id, viewport, content_w, scroll_x);
    }

    pub(crate) fn queue_combo_menu(
        &mut self,
        id: Id,
        rect: Rect,
        owner: Rect,
        items: &[&str],
        selected: usize,
    ) {
        self.pending_combo_menu = Some(PendingComboMenu {
            id,
            rect,
            owner,
            items: items.iter().map(|s| (*s).to_string()).collect(),
            selected,
        });
    }
}

fn flush_tooltip(ui: &mut GuiContext<'_>, tip: PendingTooltip) {
    let tip_rect = Rect::from_pos_size(tip.x, tip.y, tip.w, tip.h);
    ui.panel_rounded(tip_rect, style::POPUP_BG, style::RADIUS_SM);
    let title_y = font::text_top_in_row(
        tip_rect.min_y,
        if tip.body.is_empty() && tip.shortcut.is_none() {
            tip_rect.height()
        } else {
            22.0
        },
        style::FONT_SCALE,
    );
    ui.label_at_raw(
        tip_rect.min_x + 8.0,
        title_y,
        &tip.title,
        style::TEXT,
        style::FONT_SCALE,
    );
    if let Some(sc) = &tip.shortcut {
        let sw = DrawList::text_width(sc, style::FONT_SCALE * 0.8);
        ui.label_at_raw(
            tip_rect.max_x - sw - 10.0,
            tip_rect.min_y + 6.0,
            sc,
            style::ACCENT,
            style::FONT_SCALE * 0.8,
        );
    }
    if !tip.body.is_empty() {
        ui.label_at_raw(
            tip_rect.min_x + 8.0,
            tip_rect.min_y + 24.0,
            &tip.body,
            style::TEXT_DIM,
            style::FONT_SCALE * 0.8,
        );
    }
}

/// Cheap FNV-1a fingerprint for frame-local image atlas dedupe.
fn image_content_key(width: u32, height: u32, rgba: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let mix = |h: &mut u64, b: u8| {
        *h ^= u64::from(b);
        *h = h.wrapping_mul(0x100000001b3);
    };
    for b in width.to_le_bytes() {
        mix(&mut h, b);
    }
    for b in height.to_le_bytes() {
        mix(&mut h, b);
    }
    for b in (rgba.len() as u64).to_le_bytes() {
        mix(&mut h, b);
    }
    if rgba.is_empty() {
        return h;
    }
    // Sample head / mid / tail plus a sparse stride — enough for thumb identity.
    let n = rgba.len();
    let head = n.min(32);
    for &b in &rgba[..head] {
        mix(&mut h, b);
    }
    if n > 64 {
        let mid = n / 2;
        for &b in &rgba[mid..mid + 16.min(n - mid)] {
            mix(&mut h, b);
        }
        for &b in &rgba[n - 32..] {
            mix(&mut h, b);
        }
        let stride = (n / 64).max(1);
        let mut i = 0;
        while i < n {
            mix(&mut h, rgba[i]);
            i += stride;
        }
    }
    h
}
