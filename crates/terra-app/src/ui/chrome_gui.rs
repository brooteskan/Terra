//! Top application bar — brand, menus, project chip, export, window caption.

use crate::ui::style::{self, FONT_SCALE, PAD, TOOLBAR_BTN_H, TYPE_BODY, TYPE_LABEL};
use crate::ui::{FrameUiOutput, UiState, WindowResizeEdge};
use terra_core::document::TerrainDocument;
use terra_gui::{
    checkbox, chip_button, icon_button, menu_button as menu_row, Color, DrawList, GuiContext, Icon,
    Id, Rect, INSET_TOP,
};

/// Logical width reserved for minimize / maximize / close on the right.
pub fn caption_controls_width() -> f32 {
    CAPTION_BTN_W * 3.0
}

const CAPTION_BTN_W: f32 = 40.0;
/// Hit-test thickness for borderless resize edges.
const RESIZE_BORDER: f32 = 6.0;

/// Draw minimize / maximize-restore / close on the far right. Returns the cluster rect.
pub fn draw_caption_controls(
    ui: &mut GuiContext<'_>,
    out: &mut FrameUiOutput,
    maximized: bool,
) -> Rect {
    let y = (INSET_TOP - TOOLBAR_BTN_H) * 0.5;
    let cluster_w = caption_controls_width();
    let cluster = Rect::from_pos_size(ui.screen_w - cluster_w, 0.0, cluster_w, INSET_TOP);
    let btn_y = y;
    let close_r = Rect::from_pos_size(
        cluster.max_x - CAPTION_BTN_W,
        btn_y,
        CAPTION_BTN_W,
        TOOLBAR_BTN_H,
    );
    let max_r = Rect::from_pos_size(
        close_r.min_x - CAPTION_BTN_W,
        btn_y,
        CAPTION_BTN_W,
        TOOLBAR_BTN_H,
    );
    let min_r = Rect::from_pos_size(
        max_r.min_x - CAPTION_BTN_W,
        btn_y,
        CAPTION_BTN_W,
        TOOLBAR_BTN_H,
    );

    caption_icon_button(
        ui,
        Id::new("win_min"),
        min_r,
        Icon::Minus,
        false,
        &mut out.request_window_minimize,
    );
    caption_icon_button(
        ui,
        Id::new("win_max"),
        max_r,
        if maximized {
            Icon::Minimize2
        } else {
            Icon::Maximize2
        },
        false,
        &mut out.request_window_toggle_maximize,
    );
    caption_icon_button(
        ui,
        Id::new("win_close"),
        close_r,
        Icon::X,
        true,
        &mut out.request_window_close,
    );
    cluster
}

fn caption_icon_button(
    ui: &mut GuiContext<'_>,
    id: Id,
    rect: Rect,
    icon: Icon,
    is_close: bool,
    fired: &mut bool,
) {
    let hovered = ui.pointer_in(rect);
    if hovered {
        ui.state.set_hot(id);
    }
    if hovered && ui.input.primary_pressed {
        ui.state.active = Some(id);
    }
    if ui.input.primary_released && ui.state.is_active(id) && hovered {
        *fired = true;
    }
    if hovered {
        ui.panel(
            rect,
            if is_close {
                Color::rgba(0.85, 0.22, 0.25, 1.0)
            } else {
                Color::rgba(1.0, 1.0, 1.0, 0.10)
            },
        );
    }
    ui.icon_centered(
        rect,
        icon,
        if hovered && is_close {
            style::TEXT
        } else if hovered {
            style::TEXT
        } else {
            style::TEXT_DIM
        },
        14.0,
    );
}

/// Borderless frame: edge resize cursors/drags + title-bar empty-space window move.
///
/// `interactive` is true when the pointer is over a caption control / menu / button
/// that should not start a window drag.
pub fn apply_borderless_window_frame(
    ui: &mut GuiContext<'_>,
    out: &mut FrameUiOutput,
    maximized: bool,
    title_bar: Rect,
    interactive: bool,
) {
    if !maximized {
        if let Some((px, py)) = ui.input.pointer {
            if let Some(edge) = hit_resize_edge(ui.screen_w, ui.screen_h, px, py) {
                out.cursor = edge.cursor();
                if ui.input.primary_pressed {
                    out.request_window_drag_resize = Some(edge);
                }
                return;
            }
        }
    }

    let on_title = ui
        .input
        .pointer
        .map(|(x, y)| title_bar.contains(x, y))
        .unwrap_or(false);
    if on_title && !interactive && ui.input.primary_pressed {
        out.request_window_drag = true;
    }
}

fn hit_resize_edge(w: f32, h: f32, x: f32, y: f32) -> Option<WindowResizeEdge> {
    let b = RESIZE_BORDER;
    if x < 0.0 || y < 0.0 || x > w || y > h {
        return None;
    }
    let left = x <= b;
    let right = x >= w - b;
    let top = y <= b;
    let bottom = y >= h - b;
    match (left, right, top, bottom) {
        (true, false, true, false) => Some(WindowResizeEdge::NorthWest),
        (false, true, true, false) => Some(WindowResizeEdge::NorthEast),
        (true, false, false, true) => Some(WindowResizeEdge::SouthWest),
        (false, true, false, true) => Some(WindowResizeEdge::SouthEast),
        (true, false, false, false) => Some(WindowResizeEdge::West),
        (false, true, false, false) => Some(WindowResizeEdge::East),
        (false, false, true, false) => Some(WindowResizeEdge::North),
        (false, false, false, true) => Some(WindowResizeEdge::South),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MenuId {
    File,
    Edit,
    View,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RightMenu {
    Project,
}

#[derive(Debug, Default)]
pub struct ChromeGuiState {
    open: Option<MenuId>,
    /// Vertical scroll for the open File/Edit/View dropdown.
    menu_scroll_y: f32,
    right_open: Option<RightMenu>,
}

impl ChromeGuiState {
    /// True when a top-bar dropdown should eat clicks meant for chrome underneath.
    pub fn blocks_background_input(&self) -> bool {
        self.open.is_some() || self.right_open.is_some()
    }
}

pub fn draw_menu_bar(
    ui: &mut GuiContext<'_>,
    doc: &TerrainDocument,
    ui_state: &mut UiState,
    state: &mut ChromeGuiState,
    out: &mut FrameUiOutput,
) {
    let bar = Rect::from_min_max(0.0, 0.0, ui.screen_w, INSET_TOP);
    ui.panel(bar, style::TOOLBAR_BG);
    ui.panel(
        Rect::from_pos_size(0.0, INSET_TOP - 1.0, ui.screen_w, 1.0),
        style::SEPARATOR,
    );
    if ui
        .input
        .pointer
        .map(|(x, y)| bar.contains(x, y))
        .unwrap_or(false)
    {
        ui.state.set_hot(Id::new("__menu_bar"));
    }

    let y = (INSET_TOP - TOOLBAR_BTN_H) * 0.5;
    let mut x = PAD;
    let mut interactive = false;

    // Brand logo (wide wordmark — height-fit, preserve aspect).
    let (lw, lh, rgba) = crate::ui::brand::brand_logo();
    let logo_h = (TOOLBAR_BTN_H - 6.0).clamp(20.0, 26.0);
    let logo_w = logo_h * (*lw as f32 / (*lh as f32).max(1.0));
    let logo_r = Rect::from_pos_size(x, y + (TOOLBAR_BTN_H - logo_h) * 0.5, logo_w, logo_h);
    ui.image(logo_r, *lw, *lh, rgba);
    x += logo_w + 16.0;

    x = menu_button(ui, state, MenuId::File, "File", x, y, TOOLBAR_BTN_H);
    x = menu_button(ui, state, MenuId::Edit, "Edit", x, y, TOOLBAR_BTN_H);
    let left_end = menu_button(ui, state, MenuId::View, "View", x, y, TOOLBAR_BTN_H);
    if state.open.is_some() {
        interactive = true;
    }

    let caption = draw_caption_controls(ui, out, ui_state.window_maximized);
    if ui.pointer_in(caption) {
        interactive = true;
    }

    // Right cluster: project · world size · undo · redo · EXPORT · [caption].
    let icon_w = 32.0;
    let gap = 8.0;

    let build_label = "EXPORT";
    let build_w = DrawList::text_width(build_label, FONT_SCALE * TYPE_BODY) + 28.0;
    let build_r = Rect::from_pos_size(caption.min_x - gap - build_w, y, build_w, TOOLBAR_BTN_H);

    let redo_r = Rect::from_pos_size(build_r.min_x - gap - icon_w, y, icon_w, TOOLBAR_BTN_H);
    let undo_r = Rect::from_pos_size(redo_r.min_x - gap - icon_w, y, icon_w, TOOLBAR_BTN_H);

    let size_label = world_size_label(doc);
    let size_w = DrawList::text_width(&size_label, FONT_SCALE * TYPE_LABEL);
    let mut size_r = Rect::from_pos_size(undo_r.min_x - gap - size_w, y, size_w, TOOLBAR_BTN_H);

    let project_name = if doc.name.trim().is_empty() {
        "Untitled"
    } else {
        doc.name.as_str()
    };
    let mut project_w = DrawList::text_width(project_name, FONT_SCALE)
        .min(160.0)
        .max(72.0)
        + 36.0;
    let mut project_r =
        Rect::from_pos_size(size_r.min_x - 14.0 - project_w, y, project_w, TOOLBAR_BTN_H);

    // Narrow windows: collapse size, then shrink/hide project before overlapping menus.
    let show_size = size_r.min_x > left_end + gap * 2.0;
    if !show_size {
        size_r = Rect::from_pos_size(undo_r.min_x, y, 0.0, TOOLBAR_BTN_H);
        project_r =
            Rect::from_pos_size(undo_r.min_x - 14.0 - project_w, y, project_w, TOOLBAR_BTN_H);
    }
    let show_project = project_r.min_x > left_end + gap;
    if !show_project {
        // Still reserve a compact chip if there's any room.
        project_w = (undo_r.min_x - 14.0 - left_end - gap).max(0.0);
        project_r =
            Rect::from_pos_size(undo_r.min_x - 14.0 - project_w, y, project_w, TOOLBAR_BTN_H);
    }
    let show_project = project_r.width() >= 72.0;
    let project_open = state.right_open == Some(RightMenu::Project);
    if show_project {
        let project_id = Id::new("tb_project");
        let project_hovered = ui.pointer_in(project_r);
        if project_hovered {
            ui.state.set_hot(project_id);
            interactive = true;
        }
        if project_hovered && ui.input.primary_pressed {
            ui.state.active = Some(project_id);
            state.open = None;
            state.right_open = if project_open {
                None
            } else {
                Some(RightMenu::Project)
            };
        }
        ui.panel_rounded(
            project_r,
            if project_open || project_hovered {
                style::BUTTON_HOVER
            } else {
                style::SURFACE
            },
            style::RADIUS_SM,
        );
        ui.label_in_rect(
            Rect::from_pos_size(
                project_r.min_x + 10.0,
                project_r.min_y,
                project_r.width() - 28.0,
                project_r.height(),
            ),
            project_name,
            style::TEXT,
            FONT_SCALE,
        );
        ui.icon_centered(
            Rect::from_pos_size(
                project_r.max_x - 22.0,
                project_r.min_y,
                18.0,
                project_r.height(),
            ),
            Icon::ChevronDown,
            style::TEXT_MUTED,
            14.0,
        );
    } else if project_open {
        state.right_open = None;
    }

    // World extent (muted, non-interactive).
    if show_size {
        ui.label_centered_in_rect(
            size_r,
            &size_label,
            style::TEXT_MUTED,
            FONT_SCALE * TYPE_LABEL,
        );
    }

    // Undo / Redo.
    if ui.pointer_in(undo_r) || ui.pointer_in(redo_r) || ui.pointer_in(build_r) {
        interactive = true;
    }
    if icon_button(ui, Id::new("tb_undo"), Icon::Undo2, undo_r) {
        out.request_undo = true;
        state.right_open = None;
    }
    if icon_button(ui, Id::new("tb_redo"), Icon::Redo2, redo_r) {
        out.request_redo = true;
        state.right_open = None;
    }

    // Primary EXPORT CTA — file export is not ready yet.
    let build_id = Id::new("tb_build");
    let build_hovered = ui.pointer_in(build_r);
    if build_hovered {
        ui.state.set_hot(build_id);
    }
    if build_hovered && ui.input.primary_pressed {
        ui.state.active = Some(build_id);
    }
    if ui.input.primary_released && ui.state.is_active(build_id) && build_hovered {
        ui_state.show_export_unsupported = true;
        state.right_open = None;
    }
    ui.panel_rounded(
        build_r,
        if build_hovered {
            style::ACCENT
        } else {
            style::ACCENT_DIM
        },
        style::RADIUS_SM,
    );
    ui.label_centered_in_rect(build_r, build_label, style::TEXT, FONT_SCALE * TYPE_BODY);

    if let Some(menu) = state.right_open {
        interactive = true;
        draw_right_menu(ui, state, out, menu, project_r);
    }

    // Menu buttons also count as interactive for drag exclusion.
    if ui.pointer_in(Rect::from_min_max(PAD, 0.0, left_end, INSET_TOP)) {
        interactive = true;
    }

    apply_borderless_window_frame(ui, out, ui_state.window_maximized, bar, interactive);
}

fn world_size_label(doc: &TerrainDocument) -> String {
    // ASCII `x` only — the baked UI font is printable ASCII (× → "?").
    let wx = doc.metrics.world_size_x.max(0.0);
    let wz = doc.metrics.world_size_z.max(0.0);
    if wx >= 1000.0 || wz >= 1000.0 {
        format!("{:.0}x{:.0} km", wx / 1000.0, wz / 1000.0)
    } else if wx > 0.0 && wz > 0.0 {
        format!("{:.0}x{:.0} m", wx, wz)
    } else {
        let res = doc.preview_resolution.max(1);
        format!("{res}x{res}")
    }
}

fn draw_right_menu(
    ui: &mut GuiContext<'_>,
    state: &mut ChromeGuiState,
    out: &mut FrameUiOutput,
    menu: RightMenu,
    project_r: Rect,
) {
    ui.begin_overlay();
    let (anchor, width, rows) = match menu {
        RightMenu::Project => (project_r, project_r.width().max(180.0), 5.0),
    };
    let item_h = style::ROW_H;
    let menu_h = item_h * rows + PAD * 2.0;
    let popup = Rect::from_pos_size((anchor.max_x - width).max(PAD), INSET_TOP, width, menu_h);
    if ui.pointer_in(popup) {
        ui.state.set_hot(Id::new("right_menu"));
    }
    ui.begin_panel(popup, style::COMBO_MENU_BG);
    let close = match menu {
        RightMenu::Project => draw_project_menu(ui, out),
    };
    ui.end_panel();
    ui.end_overlay();

    if close {
        state.right_open = None;
    }

    if ui.input.primary_pressed {
        if let Some((px, py)) = ui.input.pointer {
            if !popup.contains(px, py) && !anchor.contains(px, py) {
                state.right_open = None;
            }
        }
    }
}

fn draw_project_menu(ui: &mut GuiContext<'_>, out: &mut FrameUiOutput) -> bool {
    if menu_item(ui, "Project Home") {
        out.request_close_project = true;
        return true;
    }
    if menu_item(ui, "New Project") {
        out.request_new_project = true;
        return true;
    }
    if menu_item(ui, "Open Project...") {
        out.request_load_path = true;
        return true;
    }
    if menu_item(ui, "Save") {
        out.request_save = true;
        return true;
    }
    if menu_item(ui, "Save Project As...") {
        out.request_save_as = true;
        return true;
    }
    false
}

/// Draw open File/Edit/View menus on the overlay layer (after all chrome).
pub fn draw_menu_overlays(
    ui: &mut GuiContext<'_>,
    ui_state: &mut UiState,
    state: &mut ChromeGuiState,
    out: &mut FrameUiOutput,
) {
    let Some(menu) = state.open else {
        return;
    };

    ui.with_menu_input(|ui| {
        ui.begin_overlay();
        let popup = draw_dropdown(ui, ui_state, state, out, menu);
        ui.end_overlay();

        let bar = Rect::from_min_max(0.0, 0.0, ui.screen_w, INSET_TOP);
        if ui.input.primary_pressed {
            if let Some((px, py)) = ui.input.pointer {
                let in_bar = bar.contains(px, py);
                let in_popup = popup.contains(px, py);
                if !in_bar && !in_popup {
                    state.open = None;
                    state.menu_scroll_y = 0.0;
                }
            }
        }
    });
}

fn menu_button(
    ui: &mut GuiContext<'_>,
    state: &mut ChromeGuiState,
    id: MenuId,
    label: &str,
    x: f32,
    y: f32,
    h: f32,
) -> f32 {
    let w = DrawList::text_width(label, FONT_SCALE) + 18.0;
    let rect = Rect::from_pos_size(x, y, w, h);
    let wid = match id {
        MenuId::File => Id::new("menu_file"),
        MenuId::Edit => Id::new("menu_edit"),
        MenuId::View => Id::new("menu_view"),
    };
    let hovered = ui.pointer_in(rect);
    if hovered {
        ui.state.set_hot(wid);
    }
    let open = state.open == Some(id);
    if hovered && ui.input.primary_pressed {
        let next = if open { None } else { Some(id) };
        if next != state.open {
            state.menu_scroll_y = 0.0;
        }
        state.open = next;
        state.right_open = None;
    } else if hovered && state.open.is_some() && state.open != Some(id) {
        state.menu_scroll_y = 0.0;
        state.open = Some(id);
        state.right_open = None;
    }
    if open || hovered {
        ui.panel_rounded(rect, style::BUTTON_HOVER, style::RADIUS_SM);
    }
    ui.label_at(
        rect.min_x + 9.0,
        rect.min_y + 7.0,
        label,
        style::TEXT_DIM,
        FONT_SCALE,
    );
    x + w + 2.0
}

fn draw_dropdown(
    ui: &mut GuiContext<'_>,
    ui_state: &mut UiState,
    state: &mut ChromeGuiState,
    out: &mut FrameUiOutput,
    menu: MenuId,
) -> Rect {
    let x = match menu {
        MenuId::File => PAD + 70.0,
        MenuId::Edit => PAD + 126.0,
        MenuId::View => PAD + 182.0,
    };
    // Natural height if the screen is tall enough; otherwise clamp and scroll.
    let natural_h = match menu {
        MenuId::File => style::ROW_H * 6.0 + PAD * 2.0,
        MenuId::Edit => style::ROW_H * 2.0 + PAD * 2.0,
        MenuId::View => style::ROW_H * 10.0 + (style::SEPARATOR_H + style::GAP) * 2.0 + PAD * 2.0,
    };
    let max_h = (ui.screen_h - INSET_TOP - PAD).max(style::ROW_H * 4.0);
    let popup = Rect::from_pos_size(x, INSET_TOP, 228.0, natural_h.min(max_h));
    if ui
        .input
        .pointer
        .map(|(px, py)| popup.contains(px, py))
        .unwrap_or(false)
    {
        ui.state.set_hot(Id::new("menu_popup"));
    }

    let scroll_id = match menu {
        MenuId::File => Id::new("menu_scroll_file"),
        MenuId::Edit => Id::new("menu_scroll_edit"),
        MenuId::View => Id::new("menu_scroll_view"),
    };
    ui.begin_panel_scrolled(
        scroll_id,
        popup,
        style::COMBO_MENU_BG,
        &mut state.menu_scroll_y,
    );
    match menu {
        MenuId::File => {
            if menu_item(ui, "New Project") {
                out.request_new_project = true;
                state.open = None;
                state.menu_scroll_y = 0.0;
            }
            if menu_item(ui, "Open Project...") {
                out.request_load_path = true;
                state.open = None;
                state.menu_scroll_y = 0.0;
            }
            if menu_item(ui, "Save") {
                out.request_save = true;
                state.open = None;
                state.menu_scroll_y = 0.0;
            }
            if menu_item(ui, "Save Project As...") {
                out.request_save_as = true;
                state.open = None;
                state.menu_scroll_y = 0.0;
            }
            if menu_item(ui, "Close Project") {
                out.request_close_project = true;
                state.open = None;
                state.menu_scroll_y = 0.0;
            }
            if menu_item(ui, "Export...") {
                ui_state.show_export = true;
                state.open = None;
                state.menu_scroll_y = 0.0;
            }
        }
        MenuId::Edit => {
            if menu_item(ui, "Undo") {
                out.request_undo = true;
                state.open = None;
                state.menu_scroll_y = 0.0;
            }
            if menu_item(ui, "Redo") {
                out.request_redo = true;
                state.open = None;
                state.menu_scroll_y = 0.0;
            }
        }
        MenuId::View => {
            {
                let mut mask_view = ui_state.is_mask_view();
                if checkbox(ui, "Mask View", &mut mask_view) {
                    if mask_view {
                        ui_state.enter_mask_view();
                    } else {
                        ui_state.leave_mask_view();
                    }
                }
            }
            checkbox(ui, "2D Preview", &mut ui_state.show_2d_preview);
            checkbox(ui, "Profiler", &mut ui_state.show_profiler);
            checkbox(ui, "History", &mut ui_state.show_history);
            ui.separator();
            if checkbox(
                ui,
                "Collapse Tool Panel",
                &mut ui_state.layout.tool_panel_collapsed,
            ) {
                ui_state.layout_dirty = true;
            }
            if checkbox(
                ui,
                "Collapse Inspector",
                &mut ui_state.layout.inspector_collapsed,
            ) {
                ui_state.layout_dirty = true;
            }
            if menu_item(ui, "Reset Layout") {
                ui_state.layout.reset();
                ui_state.layout_dirty = true;
                state.open = None;
                state.menu_scroll_y = 0.0;
            }
            ui.separator();
            if menu_item(ui, "Reset Camera") {
                out.camera_reset = true;
                state.open = None;
                state.menu_scroll_y = 0.0;
            }
            if menu_item(ui, "Top View") {
                out.camera_top_view = true;
                state.open = None;
                state.menu_scroll_y = 0.0;
            }
            if menu_item(ui, "Frame Selection") {
                out.camera_frame_selection = true;
                state.open = None;
                state.menu_scroll_y = 0.0;
            }
        }
    }
    ui.end_panel_scrolled(&mut state.menu_scroll_y);
    popup
}

fn menu_item(ui: &mut GuiContext<'_>, text: &str) -> bool {
    menu_row(ui, Id::new("menu_item").child(text), text)
}

/// Simple stub when the toolbar EXPORT button is pressed.
pub fn draw_export_unsupported_modal(ui: &mut GuiContext<'_>, open: &mut bool) {
    if !*open {
        return;
    }

    ui.begin_overlay();
    ui.panel(
        Rect::from_pos_size(0.0, 0.0, ui.screen_w, ui.screen_h),
        Color::rgba(0.0, 0.0, 0.0, 0.55),
    );

    let w = 420.0_f32.min(ui.screen_w - 40.0);
    let h = 160.0;
    let dialog = Rect::from_pos_size((ui.screen_w - w) * 0.5, (ui.screen_h - h) * 0.5, w, h);
    ui.panel_rounded(dialog, style::POPUP_BG, style::RADIUS_MD);

    ui.label_at(
        dialog.min_x + PAD * 1.5,
        dialog.min_y + PAD * 1.5,
        "Export",
        style::TEXT,
        FONT_SCALE * 1.2,
    );
    ui.label_at(
        dialog.min_x + PAD * 1.5,
        dialog.min_y + 48.0,
        "Not supported yet.",
        style::TEXT_DIM,
        FONT_SCALE,
    );

    let btn_w = 100.0;
    let btn_h = 34.0;
    let ok_r = Rect::from_pos_size(
        dialog.max_x - PAD - btn_w,
        dialog.max_y - PAD - btn_h,
        btn_w,
        btn_h,
    );
    if chip_button(ui, Id::new("export_unsupported_ok"), "OK", ok_r, true)
        || ui.input.escape_pressed
        || (ui.input.primary_pressed && !ui.pointer_in(dialog))
    {
        *open = false;
    }

    ui.end_overlay();
}
