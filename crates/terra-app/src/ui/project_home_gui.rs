//! Pre-editor project manager â€” New / Open / Recent before terrain loads.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use crate::ui::style::{self, FONT_SCALE, PAD, TYPE_LABEL};
use serde::{Deserialize, Serialize};
use terra_gui::{
    chip_button, chip_icon_button, icon_button, icon_toggle, segmented_button, Color, DrawList,
    GuiContext, Icon, Id, Rect, INSET_TOP,
};

use crate::ui::chrome_gui::{apply_borderless_window_frame, draw_caption_controls};
use crate::ui::presets::world_design_templates;
use crate::ui::FrameUiOutput;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RecentViewMode {
    #[default]
    List,
    Grid,
}

#[derive(Debug, Default)]
pub struct ProjectHomeGuiState {
    pub recent_scroll_y: f32,
    pub search: String,
    pub view_mode: RecentViewMode,
    /// Index into the filtered recent list for the open â‹® menu.
    pub row_menu: Option<usize>,
    /// Transient status line (e.g. open/browse feedback).
    pub notice: Option<String>,
}

#[derive(Debug, Clone)]
pub enum ProjectHomeAction {
    New,
    Open,
    /// Folder picker â†’ open a project found inside.
    Browse,
    OpenPath(PathBuf),
    RemoveRecent(PathBuf),
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ProjectPrefs {
    pub recent: Vec<RecentProject>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecentProject {
    pub path: String,
    pub name: String,
}

impl ProjectPrefs {
    pub const MAX_RECENT: usize = 10;

    pub fn push_recent(&mut self, path: &Path, name: &str) {
        let path_str = path.display().to_string();
        self.recent.retain(|r| r.path != path_str);
        self.recent.insert(
            0,
            RecentProject {
                path: path_str,
                name: name.to_string(),
            },
        );
        if self.recent.len() > Self::MAX_RECENT {
            self.recent.truncate(Self::MAX_RECENT);
        }
    }

    pub fn remove_recent(&mut self, path: &Path) {
        let path_str = path.display().to_string();
        self.recent.retain(|r| r.path != path_str);
    }
}

/// Full-window project home. Returns actions for the app to handle.
pub fn draw_project_home(
    ui: &mut GuiContext<'_>,
    state: &mut ProjectHomeGuiState,
    prefs: &ProjectPrefs,
    out: &mut FrameUiOutput,
    window_maximized: bool,
) -> Vec<ProjectHomeAction> {
    let mut actions = Vec::new();

    ui.panel(
        Rect::from_pos_size(0.0, 0.0, ui.screen_w, ui.screen_h),
        style::APP_BG,
    );

    let title_bar = Rect::from_min_max(0.0, 0.0, ui.screen_w, INSET_TOP);
    let caption = draw_caption_controls(ui, out, window_maximized);
    let interactive = ui.pointer_in(caption);
    apply_borderless_window_frame(ui, out, window_maximized, title_bar, interactive);

    let content_w = (ui.screen_w - 80.0).clamp(560.0, 920.0);
    let content_x = (ui.screen_w - content_w) * 0.5;
    let footer_h = 88.0;
    let top_pad = (ui.screen_h * 0.08).clamp(36.0, 72.0);

    // —— Brand header (assets/logo.png already includes TERRA wordmark) ——
    let mut y = top_pad;
    let (lw, lh, rgba) = crate::ui::brand::brand_logo();
    let logo_h = 56.0_f32;
    let logo_w = (logo_h * (*lw as f32 / (*lh as f32).max(1.0))).min(content_w * 0.72);
    let logo_h = logo_w * (*lh as f32 / (*lw as f32).max(1.0));
    let logo_r = Rect::from_pos_size(content_x + (content_w - logo_w) * 0.5, y, logo_w, logo_h);
    ui.image(logo_r, *lw, *lh, rgba);
    y += logo_h + 12.0;

    let tagline = "Create infinite landscapes.";
    let tw = DrawList::text_width(tagline, FONT_SCALE * 1.05);
    ui.label_at(
        content_x + (content_w - tw) * 0.5,
        y,
        tagline,
        style::TEXT_MUTED,
        FONT_SCALE * 1.05,
    );
    y += 42.0;

    // â€”â€” Action cards â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”
    let card_gap = 16.0;
    let card_w = (content_w - card_gap) * 0.5;
    let card_h = 108.0;
    let new_r = Rect::from_pos_size(content_x, y, card_w, card_h);
    let open_r = Rect::from_pos_size(content_x + card_w + card_gap, y, card_w, card_h);

    if draw_action_card(
        ui,
        Id::new("home_new_card"),
        new_r,
        true,
        Icon::Plus,
        "NEW PROJECT",
        "Start from scratch with a blank new world.",
    ) {
        actions.push(ProjectHomeAction::New);
    }
    if draw_action_card(
        ui,
        Id::new("home_open_card"),
        open_r,
        false,
        Icon::FolderOpen,
        "OPEN PROJECT",
        "Open an existing project from your computer.",
    ) {
        actions.push(ProjectHomeAction::Open);
    }
    y += card_h + 36.0;

    // â€”â€” Recent header + search / view toggles â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”
    let header_h = 32.0;
    ui.label_at(
        content_x,
        y + 6.0,
        "RECENT PROJECTS",
        style::TEXT_MUTED,
        FONT_SCALE * 0.88,
    );

    let toggle_w = 30.0;
    let toggle_gap = 4.0;
    let search_w = 220.0_f32.min(content_w * 0.38);
    let toggles_x = content_x + content_w - toggle_w * 2.0 - toggle_gap;
    let search_x = toggles_x - 10.0 - search_w;
    let search_r = Rect::from_pos_size(search_x, y, search_w, header_h);
    draw_search_field(ui, Id::new("home_search"), search_r, &mut state.search);

    let list_r = Rect::from_pos_size(toggles_x, y, toggle_w, header_h);
    let grid_r = Rect::from_pos_size(toggles_x + toggle_w + toggle_gap, y, toggle_w, header_h);
    if icon_toggle(
        ui,
        Id::new("home_view_list"),
        Icon::List,
        list_r,
        state.view_mode == RecentViewMode::List,
    ) {
        state.view_mode = RecentViewMode::List;
    }
    if icon_toggle(
        ui,
        Id::new("home_view_grid"),
        Icon::Grid3x3,
        grid_r,
        state.view_mode == RecentViewMode::Grid,
    ) {
        state.view_mode = RecentViewMode::Grid;
    }
    y += header_h + 14.0;

    let list_bottom = ui.screen_h - footer_h - 24.0;
    let list = Rect::from_min_max(
        content_x,
        y,
        content_x + content_w,
        list_bottom.max(y + 80.0),
    );
    // Single outer rounded shell for the recent list (rows are flat / full-bleed inside).
    ui.panel_rounded(list, style::SURFACE, style::RADIUS_MD);
    ui.begin_panel_scrolled_flush(
        Id::new("home_recent_scroll"),
        list,
        Color::rgba(0.0, 0.0, 0.0, 0.0),
        &mut state.recent_scroll_y,
    );

    let query = state.search.trim().to_ascii_lowercase();
    let filtered: Vec<(usize, &RecentProject)> = prefs
        .recent
        .iter()
        .enumerate()
        .filter(|(_, e)| {
            if query.is_empty() {
                return true;
            }
            e.name.to_ascii_lowercase().contains(&query)
                || e.path.to_ascii_lowercase().contains(&query)
        })
        .collect();

    if filtered.is_empty() {
        let empty = if prefs.recent.is_empty() {
            "No recent projects yet."
        } else {
            "No projects match your search."
        };
        ui.label_at(
            list.min_x + 16.0,
            list.min_y + 16.0,
            empty,
            style::TEXT_DIM,
            FONT_SCALE,
        );
    } else if state.view_mode == RecentViewMode::List {
        // Internal inset so icons/text aren't flush against full-bleed row edges.
        const ROW_PAD_X: f32 = 14.0;
        const ROW_PAD_Y: f32 = 12.0;
        let n = filtered.len();
        for (fi, (orig_i, entry)) in filtered.iter().enumerate() {
            let path = PathBuf::from(&entry.path);
            let meta = project_row_meta(&path);
            let row = ui.allocate(64.0);
            // Hit + hover fill use the full list shell width (allocate() reserves a
            // scrollbar gutter that must not leave a dead strip or clip hover chrome).
            let row_full = Rect::from_min_max(list.min_x, row.min_y, list.max_x, row.max_y);
            let id = Id::new("home_recent").with(*orig_i as u64);
            let hovered = ui.pointer_in(row_full);
            if hovered {
                ui.state.set_hot(id);
            }
            if hovered && ui.input.primary_pressed {
                ui.state.active = Some(id);
            }
            let clicked = ui.input.primary_released && ui.state.is_active(id) && hovered;

            // Continuous list chrome: square row fills; only ends use outer radius.
            // Hover is driven by pointer_in (mouse-over), not press/active.
            if hovered {
                let is_first = fi == 0;
                let is_last = fi + 1 == n;
                if is_first && is_last {
                    ui.panel_rounded(row_full, style::HOVER_BG, style::RADIUS_MD);
                } else if is_first {
                    paint_row_top_rounded(ui, row_full, style::HOVER_BG, style::RADIUS_MD);
                } else if is_last {
                    paint_row_bottom_rounded(ui, row_full, style::HOVER_BG, style::RADIUS_MD);
                } else {
                    ui.panel(row_full, style::HOVER_BG);
                }
            }
            if fi + 1 < n {
                ui.panel(
                    Rect::from_pos_size(
                        row.min_x + ROW_PAD_X,
                        row.max_y - 1.0,
                        row.width() - ROW_PAD_X * 2.0,
                        1.0,
                    ),
                    style::SEPARATOR,
                );
            }

            // Thumbnail
            let thumb =
                Rect::from_pos_size(row.min_x + ROW_PAD_X, row.min_y + ROW_PAD_Y, 44.0, 44.0);
            ui.panel_rounded(thumb, style::RAISED_BG, style::RADIUS_SM);
            ui.icon_centered(
                thumb,
                Icon::Mountain,
                if meta.exists {
                    style::ACCENT
                } else {
                    style::TEXT_DISABLED
                },
                22.0,
            );

            let name_color = if meta.exists {
                style::TEXT
            } else {
                style::TEXT_DISABLED
            };
            let path_color = if meta.exists {
                style::TEXT_MUTED
            } else {
                style::TEXT_DISABLED
            };
            let name_label = if meta.exists {
                entry.name.clone()
            } else {
                format!("{} (missing)", entry.name)
            };
            let name_x = thumb.max_x + 14.0;
            ui.label_at(
                name_x,
                row.min_y + ROW_PAD_Y + 2.0,
                &name_label,
                name_color,
                FONT_SCALE,
            );
            let path_label = truncate_middle(&entry.path, 52);
            ui.label_at(
                name_x,
                row.min_y + ROW_PAD_Y + 24.0,
                &path_label,
                path_color,
                FONT_SCALE * 0.85,
            );

            // Meta columns (right side)
            let menu_r = Rect::from_pos_size(
                row.max_x - ROW_PAD_X - 28.0,
                row.min_y + (row.height() - 28.0) * 0.5,
                28.0,
                28.0,
            );
            let size_x = menu_r.min_x - 150.0;
            let mod_x = size_x - 170.0;

            if let Some(ref modified) = meta.modified_label {
                ui.icon_at(
                    mod_x,
                    row.min_y + (row.height() - 14.0) * 0.5,
                    Icon::Calendar,
                    style::TEXT_MUTED,
                    14.0,
                );
                ui.label_at(
                    mod_x + 20.0,
                    row.min_y + (row.height() - 14.0) * 0.5,
                    modified,
                    style::TEXT_MUTED,
                    FONT_SCALE * 0.85,
                );
            }
            if let Some(ref size) = meta.size_label {
                ui.icon_at(
                    size_x,
                    row.min_y + (row.height() - 14.0) * 0.5,
                    Icon::Mountain,
                    style::TEXT_MUTED,
                    14.0,
                );
                ui.label_at(
                    size_x + 20.0,
                    row.min_y + (row.height() - 14.0) * 0.5,
                    size,
                    style::TEXT_MUTED,
                    FONT_SCALE * 0.85,
                );
            }

            if icon_button(ui, id.child("menu"), Icon::Ellipsis, menu_r) {
                state.row_menu = Some(fi);
            } else if clicked && meta.exists && state.row_menu != Some(fi) {
                actions.push(ProjectHomeAction::OpenPath(path.clone()));
            }

            // Row menu popup
            if state.row_menu == Some(fi) {
                let menu =
                    Rect::from_pos_size(menu_r.min_x - 140.0, menu_r.max_y + 4.0, 168.0, 68.0);
                ui.begin_overlay();
                ui.panel_rounded(menu, style::POPUP_BG, style::RADIUS_SM);
                let open_item =
                    Rect::from_pos_size(menu.min_x + 4.0, menu.min_y + 4.0, 160.0, 28.0);
                let rm_item = Rect::from_pos_size(menu.min_x + 4.0, menu.min_y + 34.0, 160.0, 28.0);
                if menu_item(ui, id.child("m_open"), open_item, "Open") && meta.exists {
                    actions.push(ProjectHomeAction::OpenPath(path.clone()));
                    state.row_menu = None;
                }
                if menu_item(ui, id.child("m_rm"), rm_item, "Remove from recent") {
                    actions.push(ProjectHomeAction::RemoveRecent(path.clone()));
                    state.row_menu = None;
                }
                if ui.input.primary_pressed && !ui.pointer_in(menu) && !ui.pointer_in(menu_r) {
                    state.row_menu = None;
                }
                if ui.input.escape_pressed {
                    state.row_menu = None;
                }
                ui.end_overlay();
            }
        }
    } else {
        // Grid cards
        let cell_w = 200.0;
        let cell_h = 148.0;
        let gap = 12.0;
        let cols = ((list.width() + gap) / (cell_w + gap)).floor().max(1.0) as usize;
        for (fi, (orig_i, entry)) in filtered.iter().enumerate() {
            let col = fi % cols;
            if col == 0 {
                let _ = ui.allocate(cell_h + gap);
            }
            let cursor_y = ui.layout_cursor_y().unwrap_or(list.min_y) - (cell_h + gap);
            let cell = Rect::from_pos_size(
                list.min_x + col as f32 * (cell_w + gap),
                cursor_y,
                cell_w,
                cell_h,
            );
            let path = PathBuf::from(&entry.path);
            let meta = project_row_meta(&path);
            let id = Id::new("home_grid").with(*orig_i as u64);
            let hovered = ui.pointer_in(cell);
            if hovered {
                ui.state.set_hot(id);
            }
            if hovered && ui.input.primary_pressed {
                ui.state.active = Some(id);
            }
            let clicked = ui.input.primary_released && ui.state.is_active(id) && hovered;
            ui.panel_rounded(
                cell,
                if hovered {
                    style::HOVER_BG
                } else {
                    style::SURFACE
                },
                style::RADIUS_MD,
            );
            let thumb =
                Rect::from_pos_size(cell.min_x + 12.0, cell.min_y + 12.0, cell_w - 24.0, 72.0);
            ui.panel_rounded(thumb, style::RAISED_BG, style::RADIUS_SM);
            ui.icon_centered(thumb, Icon::Mountain, style::ACCENT, 28.0);
            ui.label_at(
                cell.min_x + 12.0,
                thumb.max_y + 10.0,
                &entry.name,
                if meta.exists {
                    style::TEXT
                } else {
                    style::TEXT_DISABLED
                },
                FONT_SCALE,
            );
            let sub = meta
                .size_label
                .clone()
                .or(meta.modified_label.clone())
                .unwrap_or_else(|| truncate_middle(&entry.path, 28));
            ui.label_at(
                cell.min_x + 12.0,
                thumb.max_y + 30.0,
                &sub,
                style::TEXT_MUTED,
                FONT_SCALE * 0.82,
            );
            if clicked && meta.exists {
                actions.push(ProjectHomeAction::OpenPath(path));
            }
        }
    }

    ui.end_panel_scrolled(&mut state.recent_scroll_y);

    // â€”â€” Footer â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”
    let footer_y = ui.screen_h - footer_h;
    ui.panel(
        Rect::from_pos_size(0.0, footer_y - 1.0, ui.screen_w, 1.0),
        style::SEPARATOR,
    );
    let foot_w = content_w.min(780.0);
    let foot_x = (ui.screen_w - foot_w) * 0.5;
    // Only honest, wired actions â€” Settings / Recover stubs removed from the footer.
    let browse = Rect::from_pos_size(foot_x, footer_y + 14.0, foot_w, 60.0);
    if draw_footer_item(
        ui,
        Id::new("home_foot_browse"),
        browse,
        Icon::FolderSearch,
        "Browse Projects",
        "Find projects on disk",
    ) {
        actions.push(ProjectHomeAction::Browse);
    }

    if let Some(notice) = &state.notice {
        let nw = DrawList::text_width(notice, FONT_SCALE * TYPE_LABEL);
        ui.label_at(
            (ui.screen_w - nw) * 0.5,
            footer_y - 22.0,
            notice,
            style::TEXT_DIM,
            FONT_SCALE * TYPE_LABEL,
        );
    }

    actions
}

/// Hover fill with only the top corners rounded (square bottom).
fn paint_row_top_rounded(ui: &mut GuiContext<'_>, row: Rect, color: Color, radius: f32) {
    ui.panel_rounded(row, color, radius);
    ui.panel(
        Rect::from_pos_size(row.min_x, row.max_y - radius, row.width(), radius),
        color,
    );
}

/// Hover fill with only the bottom corners rounded (square top).
fn paint_row_bottom_rounded(ui: &mut GuiContext<'_>, row: Rect, color: Color, radius: f32) {
    ui.panel_rounded(row, color, radius);
    ui.panel(
        Rect::from_pos_size(row.min_x, row.min_y, row.width(), radius),
        color,
    );
}

fn draw_action_card(
    ui: &mut GuiContext<'_>,
    id: Id,
    rect: Rect,
    accent: bool,
    icon: Icon,
    title: &str,
    subtitle: &str,
) -> bool {
    let hovered = ui.pointer_in(rect);
    if hovered {
        ui.state.set_hot(id);
    }
    if hovered && ui.input.primary_pressed {
        ui.state.active = Some(id);
    }
    let clicked = ui.input.primary_released && ui.state.is_active(id) && hovered;

    let border = if accent {
        if hovered {
            style::ACCENT_HOVER
        } else {
            style::ACCENT
        }
    } else if hovered {
        style::BORDER
    } else {
        Color::rgba(1.0, 1.0, 1.0, 0.10)
    };
    let fill = if hovered {
        style::HOVER_BG
    } else {
        style::SURFACE
    };
    // Border via outer + inset fill.
    ui.panel_rounded(rect, border, style::RADIUS_MD);
    let inset = 1.5;
    ui.panel_rounded(
        Rect::from_min_max(
            rect.min_x + inset,
            rect.min_y + inset,
            rect.max_x - inset,
            rect.max_y - inset,
        ),
        fill,
        (style::RADIUS_MD - 1.0).max(0.0),
    );

    let icon_box = Rect::from_pos_size(rect.min_x + 18.0, rect.min_y + 28.0, 44.0, 44.0);
    if accent {
        ui.panel_rounded(icon_box, style::ACCENT_SOFT, style::RADIUS_SM);
        ui.icon_centered(icon_box, icon, style::ACCENT, 22.0);
    } else {
        ui.panel_rounded(icon_box, style::RAISED_BG, style::RADIUS_SM);
        ui.icon_centered(icon_box, icon, style::TEXT_DIM, 22.0);
    }

    let text_x = icon_box.max_x + 16.0;
    ui.label_at(
        text_x,
        rect.min_y + 28.0,
        title,
        if accent { style::ACCENT } else { style::TEXT },
        FONT_SCALE * 1.05,
    );
    // Wrap subtitle lightly.
    let (l1, l2) = split_desc(subtitle, 36);
    ui.label_at(
        text_x,
        rect.min_y + 52.0,
        l1,
        style::TEXT_MUTED,
        FONT_SCALE * 0.88,
    );
    if !l2.is_empty() {
        ui.label_at(
            text_x,
            rect.min_y + 68.0,
            l2,
            style::TEXT_MUTED,
            FONT_SCALE * 0.88,
        );
    }

    ui.icon_at(
        rect.max_x - 36.0,
        rect.min_y + (rect.height() - 18.0) * 0.5,
        Icon::ArrowRight,
        if accent {
            style::ACCENT
        } else {
            style::TEXT_DIM
        },
        18.0,
    );

    clicked
}

fn draw_footer_item(
    ui: &mut GuiContext<'_>,
    id: Id,
    rect: Rect,
    icon: Icon,
    title: &str,
    subtitle: &str,
) -> bool {
    let hovered = ui.pointer_in(rect);
    if hovered {
        ui.state.set_hot(id);
    }
    if hovered && ui.input.primary_pressed {
        ui.state.active = Some(id);
    }
    let clicked = ui.input.primary_released && ui.state.is_active(id) && hovered;
    if hovered {
        ui.panel_rounded(rect, style::HOVER_BG, style::RADIUS_SM);
    }
    let icon_r = Rect::from_pos_size(rect.min_x + 8.0, rect.min_y + 12.0, 28.0, 28.0);
    ui.icon_centered(
        icon_r,
        icon,
        if hovered {
            style::TEXT
        } else {
            style::TEXT_MUTED
        },
        18.0,
    );
    ui.label_at(
        icon_r.max_x + 10.0,
        rect.min_y + 10.0,
        title,
        style::TEXT,
        FONT_SCALE,
    );
    ui.label_at(
        icon_r.max_x + 10.0,
        rect.min_y + 30.0,
        subtitle,
        style::TEXT_MUTED,
        FONT_SCALE * 0.82,
    );
    clicked
}

fn draw_search_field(ui: &mut GuiContext<'_>, id: Id, rect: Rect, query: &mut String) {
    let hovered = ui.pointer_in(rect);
    let focused = ui.state.text_focus == Some(id);
    if hovered {
        ui.state.set_hot(id);
    }
    if hovered && ui.input.primary_pressed {
        ui.state.text_focus = Some(id);
        ui.state.text_buffer = query.clone();
        ui.state.active = Some(id);
    }
    if focused {
        if !ui.input.text.is_empty() {
            ui.state.text_buffer.push_str(&ui.input.text);
            *query = ui.state.text_buffer.clone();
        }
        if ui.input.backspace_pressed {
            ui.state.text_buffer.pop();
            *query = ui.state.text_buffer.clone();
        }
        if ui.input.escape_pressed {
            query.clear();
            ui.state.clear_text_focus();
        }
        if ui.input.primary_pressed && !hovered {
            ui.state.clear_text_focus();
        }
    }

    ui.panel_rounded(
        rect,
        if focused || hovered {
            style::BUTTON_HOVER
        } else {
            style::INPUT_BG
        },
        style::RADIUS_SM,
    );
    ui.icon_at(
        rect.min_x + 8.0,
        rect.min_y + (rect.height() - 14.0) * 0.5,
        Icon::Search,
        style::TEXT_MUTED,
        14.0,
    );
    let label = if query.is_empty() && !focused {
        "Search projects..."
    } else {
        query.as_str()
    };
    ui.label_at(
        rect.min_x + 28.0,
        rect.min_y + (rect.height() - 14.0) * 0.5,
        label,
        if query.is_empty() {
            style::TEXT_MUTED
        } else {
            style::TEXT
        },
        FONT_SCALE * 0.92,
    );
}

fn menu_item(ui: &mut GuiContext<'_>, id: Id, rect: Rect, label: &str) -> bool {
    let hovered = ui.pointer_in(rect);
    if hovered {
        ui.state.set_hot(id);
    }
    if hovered && ui.input.primary_pressed {
        ui.state.active = Some(id);
    }
    let clicked = ui.input.primary_released && ui.state.is_active(id) && hovered;
    if hovered {
        ui.panel_rounded(rect, style::HOVER_BG, style::RADIUS_SM);
    }
    ui.label_at(
        rect.min_x + 10.0,
        rect.min_y + 6.0,
        label,
        style::TEXT,
        FONT_SCALE,
    );
    clicked
}

struct ProjectRowMeta {
    exists: bool,
    modified_label: Option<String>,
    size_label: Option<String>,
}

fn project_row_meta(path: &Path) -> ProjectRowMeta {
    let exists = path.exists();
    if !exists {
        return ProjectRowMeta {
            exists: false,
            modified_label: None,
            size_label: None,
        };
    }
    let modified_label = std::fs::metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
        .map(|t| format!("Modified {}", format_modified(t)));
    let size_label = peek_world_size_label(path);
    ProjectRowMeta {
        exists: true,
        modified_label,
        size_label,
    }
}

fn format_modified(t: SystemTime) -> String {
    let Ok(elapsed) = SystemTime::now().duration_since(t) else {
        return "just now".into();
    };
    relative_time(elapsed)
}

fn relative_time(d: Duration) -> String {
    let secs = d.as_secs();
    if secs < 60 {
        "just now".into()
    } else if secs < 3600 {
        let m = secs / 60;
        format!("{m}m ago")
    } else if secs < 86_400 {
        let h = secs / 3600;
        format!("{h}h ago")
    } else if secs < 86_400 * 30 {
        let days = secs / 86_400;
        format!("{days}d ago")
    } else {
        let months = secs / (86_400 * 30);
        format!("{months}mo ago")
    }
}

/// Read only the head of a project JSON for metrics.world_size_* (files can be multi-MB).
fn peek_world_size_label(path: &Path) -> Option<String> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut buf = vec![0u8; 8192];
    let n = file.read(&mut buf).ok()?;
    let head = String::from_utf8_lossy(&buf[..n]);
    let wx = find_json_f32(&head, "\"world_size_x\"")?;
    let wz = find_json_f32(&head, "\"world_size_z\"").unwrap_or(wx);
    let km_x = (wx / 1000.0).round().max(1.0) as i32;
    let km_z = (wz / 1000.0).round().max(1.0) as i32;
    // ASCII `x` â€” baked UI font has no Ã— glyph (would render as "?").
    Some(format!("{km_x}x{km_z} km"))
}

fn find_json_f32(hay: &str, key: &str) -> Option<f32> {
    let i = hay.find(key)?;
    let rest = &hay[i + key.len()..];
    let after_colon = rest.find(':')?;
    let num_start = rest[after_colon + 1..]
        .char_indices()
        .find(|(_, c)| c.is_ascii_digit() || *c == '-' || *c == '.')
        .map(|(idx, _)| after_colon + 1 + idx)?;
    let slice = &rest[num_start..];
    let end = slice
        .char_indices()
        .find(|(_, c)| {
            !(c.is_ascii_digit() || *c == '.' || *c == 'e' || *c == 'E' || *c == '-' || *c == '+')
        })
        .map(|(i, _)| i)
        .unwrap_or(slice.len());
    slice[..end].parse().ok()
}

fn truncate_middle(s: &str, max_chars: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max_chars {
        return s.to_string();
    }
    let keep = max_chars.saturating_sub(1) / 2;
    let left: String = chars.iter().take(keep).collect();
    let right: String = chars.iter().skip(chars.len() - keep).collect();
    format!("{left}â€¦{right}")
}

#[derive(Debug, Clone, PartialEq)]
pub enum NewProjectTemplateChoice {
    Cancel,
    Create {
        template_id: String,
        /// World Creator–style Resolution: world extent in metres (samples derived).
        world_size_m: f32,
        sea_level: f32,
    },
}

/// World settings edited in the New Project modal.
#[derive(Debug, Clone)]
pub struct NewWorldSettings {
    /// World Creator–style Resolution — physical extent in metres.
    pub world_size_m: f32,
    pub sea_level: f32,
    /// UI-only: display resolution / sea level in kilometers.
    pub units_kilometers: bool,
    /// UI-only: horizontal scroll for World Design cards.
    pub design_scroll_x: f32,
}

impl Default for NewWorldSettings {
    fn default() -> Self {
        Self {
            world_size_m: 4096.0,
            sea_level: 0.0,
            units_kilometers: false,
            design_scroll_x: 0.0,
        }
    }
}

/// Modal template picker for New Project with World Design settings.
pub fn draw_new_project_templates(
    ui: &mut GuiContext<'_>,
    selected_id: &mut String,
    settings: &mut NewWorldSettings,
) -> Option<NewProjectTemplateChoice> {
    ui.begin_overlay();
    ui.panel(
        Rect::from_pos_size(0.0, 0.0, ui.screen_w, ui.screen_h),
        Color::rgba(0.0, 0.0, 0.0, 0.62),
    );

    let panel_w = (ui.screen_w - 64.0).clamp(720.0, 1080.0);
    let panel_h = (ui.screen_h - 40.0).clamp(640.0, 780.0);
    let panel = Rect::from_pos_size(
        (ui.screen_w - panel_w) * 0.5,
        ((ui.screen_h - panel_h) * 0.5).max(20.0),
        panel_w,
        panel_h,
    );
    ui.panel_rounded(panel, style::POPUP_BG, style::RADIUS_LG);

    let inset = style::SPACE_5;
    let content_x = panel.min_x + inset;
    let content_w = panel.width() - inset * 2.0;
    let mut y = panel.min_y + style::SPACE_4;

    // â€”â€” Header â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”
    let icon_box = Rect::from_pos_size(content_x, y + 2.0, 40.0, 40.0);
    ui.panel_rounded(icon_box, style::ACCENT_SOFT, style::RADIUS_MD);
    ui.icon_centered(icon_box, Icon::Package, style::ACCENT, 22.0);

    ui.label_at(
        icon_box.max_x + style::SPACE_3,
        y + 2.0,
        "New World",
        style::TEXT,
        FONT_SCALE * 1.35,
    );
    ui.label_at(
        icon_box.max_x + style::SPACE_3,
        y + 26.0,
        "Start from an archetype or create a blank world.",
        style::TEXT_MUTED,
        FONT_SCALE * 0.92,
    );

    let close_r = Rect::from_pos_size(panel.max_x - inset - 32.0, y + 4.0, 32.0, 32.0);
    let mut choice = None;
    let had_text_focus = ui.state.text_focus.is_some();
    if icon_button(ui, Id::new("new_tmpl_close"), Icon::X, close_r) {
        choice = Some(NewProjectTemplateChoice::Cancel);
    }
    y += 56.0;

    if selected_id.is_empty() {
        *selected_id = "blank".into();
        apply_template_defaults(selected_id, settings);
    }

    // â€”â€” WORLD DESIGN â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”
    ui.label_at(
        content_x,
        y,
        "WORLD DESIGN",
        style::TEXT_MUTED,
        FONT_SCALE * 0.78,
    );
    y += 20.0;

    let design = world_design_templates();
    let design_card_w = 280.0;
    let design_card_h = 220.0;
    let design_gap = style::SPACE_3;
    let design_strip = Rect::from_pos_size(content_x, y, content_w, design_card_h);
    let mut design_scroll = settings.design_scroll_x;
    draw_template_strip(
        ui,
        Id::new("nw_design_strip"),
        design_strip,
        &design,
        selected_id,
        settings,
        &mut design_scroll,
        design_card_w,
        design_card_h,
        design_gap,
    );
    settings.design_scroll_x = design_scroll;
    y += design_card_h + style::SPACE_5;

    // â€”â€” WORLD SETTINGS â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”
    ui.label_at(
        content_x,
        y + 4.0,
        "WORLD SETTINGS",
        style::TEXT_MUTED,
        FONT_SCALE * 0.78,
    );

    let units_w = 188.0;
    let units_h = 28.0;
    let units_r = Rect::from_pos_size(panel.max_x - inset - units_w, y, units_w, units_h);
    ui.panel_rounded(units_r, style::INPUT_BG, style::RADIUS_SM);
    let half = units_w * 0.5;
    let m_r = Rect::from_pos_size(
        units_r.min_x + 2.0,
        units_r.min_y + 2.0,
        half - 3.0,
        units_h - 4.0,
    );
    let k_r = Rect::from_pos_size(
        units_r.min_x + half + 1.0,
        units_r.min_y + 2.0,
        half - 3.0,
        units_h - 4.0,
    );
    if segmented_button(
        ui,
        Id::new("nw_units_m"),
        m_r,
        "Meters",
        !settings.units_kilometers,
    ) {
        settings.units_kilometers = false;
    }
    if segmented_button(
        ui,
        Id::new("nw_units_km"),
        k_r,
        "Kilometers",
        settings.units_kilometers,
    ) {
        settings.units_kilometers = true;
    }
    y += units_h + style::SPACE_3;

    let footer_h = 52.0;
    let settings_bottom = panel.max_y - inset - footer_h;
    let row_h = style::CONTROL_ROW_H + 4.0;
    let row_gap = style::SPACE_2;
    let settings_x = content_x;
    let settings_w = content_w;

    let km = settings.units_kilometers;
    let mut size_display = if km {
        settings.world_size_m / 1000.0
    } else {
        settings.world_size_m
    };
    let (size_min, size_max) = if km {
        (1.024, 100.0)
    } else {
        (1024.0, 100_000.0)
    };
    if draw_setting_row(
        ui,
        Id::new("nw_size"),
        Rect::from_pos_size(settings_x, y, settings_w, row_h),
        Icon::Maximize2,
        "Resolution",
        &mut size_display,
        size_min,
        size_max,
        !km,
    ) {
        settings.world_size_m = if km {
            (size_display * 1000.0).clamp(1024.0, 100_000.0)
        } else {
            size_display.clamp(1024.0, 100_000.0)
        };
    }
    y += row_h + row_gap;

    let mut sea_display = if km {
        settings.sea_level / 1000.0
    } else {
        settings.sea_level
    };
    let (sea_min, sea_max) = if km { (-0.05, 0.05) } else { (-50.0, 50.0) };
    if draw_setting_row(
        ui,
        Id::new("nw_sea"),
        Rect::from_pos_size(settings_x, y, settings_w, row_h),
        Icon::Waves,
        "Sea level",
        &mut sea_display,
        sea_min,
        sea_max,
        false,
    ) {
        settings.sea_level = if km {
            (sea_display * 1000.0).clamp(-50.0, 50.0)
        } else {
            sea_display.clamp(-50.0, 50.0)
        };
    }
    let _ = (y, row_h, row_gap);
    let _ = settings_bottom;

    // Footer
    let btn_h = 36.0;
    let btn_y = panel.max_y - inset - btn_h;
    let create_w = 148.0;
    let cancel_w = 100.0;
    let create_r = Rect::from_pos_size(panel.max_x - inset - create_w, btn_y, create_w, btn_h);
    let cancel_r = Rect::from_pos_size(
        create_r.min_x - style::SPACE_2 - cancel_w,
        btn_y,
        cancel_w,
        btn_h,
    );

    if chip_button(ui, Id::new("new_tmpl_cancel"), "Cancel", cancel_r, false) {
        choice = Some(NewProjectTemplateChoice::Cancel);
    }
    if chip_icon_button(
        ui,
        Id::new("new_tmpl_create"),
        Icon::Sparkles,
        "Create World",
        create_r,
        true,
    ) && !selected_id.is_empty()
    {
        choice = Some(NewProjectTemplateChoice::Create {
            template_id: selected_id.clone(),
            world_size_m: settings.world_size_m,
            sea_level: settings.sea_level,
        });
    }
    if ui.input.escape_pressed && !had_text_focus {
        choice = Some(NewProjectTemplateChoice::Cancel);
    }

    ui.end_overlay();
    choice
}

fn draw_template_strip(
    ui: &mut GuiContext<'_>,
    strip_id: Id,
    strip: Rect,
    templates: &[crate::ui::presets::ProjectTemplate],
    selected_id: &mut String,
    settings: &mut NewWorldSettings,
    scroll_x: &mut f32,
    card_w: f32,
    card_h: f32,
    gap: f32,
) {
    let chevron_w = 36.0;
    let n = templates.len();
    let content_w = if n == 0 {
        0.0
    } else {
        n as f32 * card_w + (n.saturating_sub(1) as f32) * gap
    };
    let view_w = (strip.width()
        - if content_w > strip.width() + 1.0 {
            chevron_w + style::SPACE_2
        } else {
            0.0
        })
    .max(80.0);
    let viewport = Rect::from_pos_size(strip.min_x, strip.min_y, view_w, card_h);
    let max_scroll = (content_w - view_w).max(0.0);
    *scroll_x = scroll_x.clamp(0.0, max_scroll);

    if ui.pointer_in(viewport) && ui.input.scroll_delta.abs() > 1e-4 {
        *scroll_x = (*scroll_x - ui.input.scroll_delta * 28.0).clamp(0.0, max_scroll);
    }

    // Clip card drawing to the strip viewport.
    ui.begin_panel(viewport, Color::rgba(0.0, 0.0, 0.0, 0.0));
    for (i, template) in templates.iter().enumerate() {
        let x = viewport.min_x + i as f32 * (card_w + gap) - *scroll_x;
        let card = Rect::from_pos_size(x, viewport.min_y, card_w, card_h);
        if card.max_x < viewport.min_x - 4.0 || card.min_x > viewport.max_x + 4.0 {
            continue;
        }
        draw_design_card(ui, template, selected_id, settings, card, viewport);
    }
    ui.end_panel();

    if max_scroll > 1.0 {
        let chevron = Rect::from_pos_size(
            strip.max_x - chevron_w,
            strip.min_y + (card_h - chevron_w) * 0.5,
            chevron_w,
            chevron_w,
        );
        if icon_button(ui, strip_id.child("chevron"), Icon::ChevronRight, chevron) {
            *scroll_x = (*scroll_x + card_w + gap).clamp(0.0, max_scroll);
        }
    }
}

fn apply_template_defaults(template_id: &str, settings: &mut NewWorldSettings) {
    match template_id {
        "river_valley" => {
            settings.sea_level = 20.0;
        }
        "blank" | "tropical_island" | "alpine" | "desert" => {
            settings.sea_level = 0.0;
        }
        _ => {}
    }
}

fn select_template(
    template: &crate::ui::presets::ProjectTemplate,
    selected_id: &mut String,
    settings: &mut NewWorldSettings,
) {
    *selected_id = template.id.to_string();
    apply_template_defaults(template.id, settings);
}

fn draw_design_card(
    ui: &mut GuiContext<'_>,
    template: &crate::ui::presets::ProjectTemplate,
    selected_id: &mut String,
    settings: &mut NewWorldSettings,
    card: Rect,
    hit_clip: Rect,
) {
    let id = Id::new("new_tmpl").child(template.id);
    let selected = selected_id.as_str() == template.id;
    let hovered = ui.pointer_in(card) && ui.pointer_in(hit_clip);
    if hovered {
        ui.state.set_hot(id);
    }
    if hovered && ui.input.primary_pressed {
        ui.state.active = Some(id);
    }
    if ui.input.primary_released && ui.state.is_active(id) && hovered {
        select_template(template, selected_id, settings);
    }

    let border = if selected {
        style::ACCENT
    } else if hovered {
        style::BORDER
    } else {
        Color::rgba(1.0, 1.0, 1.0, 0.08)
    };
    let fill = if selected {
        style::SELECTED_BG
    } else if hovered {
        style::HOVER_BG
    } else {
        style::SURFACE
    };
    ui.panel_rounded(card, border, style::RADIUS_MD);
    let inset = if selected { 2.0 } else { 1.0 };
    let inner = Rect::from_min_max(
        card.min_x + inset,
        card.min_y + inset,
        card.max_x - inset,
        card.max_y - inset,
    );
    ui.panel_rounded(inner, fill, (style::RADIUS_MD - 1.0).max(0.0));

    let pad = 10.0;
    let text_block_h = 58.0;
    let thumb_h = (inner.height() - pad * 3.0 - text_block_h).max(96.0);
    let thumb = Rect::from_pos_size(
        inner.min_x + pad,
        inner.min_y + pad,
        inner.width() - pad * 2.0,
        thumb_h,
    );
    paint_design_thumb(ui, template.id, thumb);

    if selected {
        let badge = Rect::from_pos_size(thumb.max_x - 22.0, thumb.min_y + 8.0, 20.0, 20.0);
        ui.panel_rounded(badge, style::ACCENT, 10.0);
        ui.icon_centered(badge, Icon::Check, style::TEXT, 12.0);
    }

    let text_x = inner.min_x + pad;
    let text_w = (inner.width() - pad * 2.0).max(8.0);
    let name_y = thumb.max_y + 8.0;
    let name_scale = FONT_SCALE * 1.02;
    ui.label_in_rect(
        Rect::from_pos_size(text_x, name_y, text_w, 18.0),
        template.name,
        style::TEXT,
        name_scale,
    );

    let desc_scale = FONT_SCALE * 0.82;
    let line_h = 16.0;
    let (l1, l2) = split_desc_to_width(template.description, desc_scale, text_w);
    ui.label_in_rect(
        Rect::from_pos_size(text_x, name_y + 20.0, text_w, line_h),
        &l1,
        style::TEXT_MUTED,
        desc_scale,
    );
    if !l2.is_empty() {
        ui.label_in_rect(
            Rect::from_pos_size(text_x, name_y + 20.0 + line_h, text_w, line_h),
            &l2,
            style::TEXT_MUTED,
            desc_scale,
        );
    }
}

fn paint_design_thumb(ui: &mut GuiContext<'_>, template_id: &str, thumb: Rect) {
    let (c0, c1, c2, icon) = match template_id {
        "tropical_island" => (
            Color::rgb(0.08, 0.28, 0.42),
            Color::rgb(0.12, 0.55, 0.48),
            Color::rgb(0.72, 0.82, 0.45),
            Icon::Waves,
        ),
        "blank" => (
            Color::rgb(0.14, 0.16, 0.22),
            Color::rgb(0.18, 0.24, 0.36),
            Color::rgb(0.28, 0.40, 0.62),
            Icon::Package,
        ),
        "alpine" => (
            Color::rgb(0.18, 0.24, 0.36),
            Color::rgb(0.32, 0.42, 0.55),
            Color::rgb(0.78, 0.84, 0.90),
            Icon::Mountain,
        ),
        "desert" => (
            Color::rgb(0.42, 0.26, 0.12),
            Color::rgb(0.72, 0.48, 0.22),
            Color::rgb(0.90, 0.72, 0.40),
            Icon::Sun,
        ),
        "river_valley" => (
            Color::rgb(0.12, 0.28, 0.18),
            Color::rgb(0.22, 0.48, 0.32),
            Color::rgb(0.35, 0.58, 0.72),
            Icon::Droplets,
        ),
        _ => (
            Color::rgb(0.16, 0.20, 0.28),
            Color::rgb(0.22, 0.30, 0.42),
            Color::rgb(0.35, 0.48, 0.68),
            Icon::Image,
        ),
    };

    ui.panel_rounded(thumb, c0, style::RADIUS_SM);
    // Soft layered bands to suggest a landscape without asset files.
    let band_h = thumb.height() * 0.38;
    ui.panel_rounded(
        Rect::from_pos_size(
            thumb.min_x + 6.0,
            thumb.max_y - band_h - 8.0,
            thumb.width() - 12.0,
            band_h,
        ),
        c1,
        style::RADIUS_SM,
    );
    ui.panel_rounded(
        Rect::from_pos_size(
            thumb.min_x + thumb.width() * 0.18,
            thumb.min_y + 10.0,
            thumb.width() * 0.55,
            thumb.height() * 0.42,
        ),
        c2,
        style::RADIUS_SM,
    );
    ui.icon_centered(
        Rect::from_pos_size(thumb.center_x() - 14.0, thumb.center_y() - 14.0, 28.0, 28.0),
        icon,
        Color::rgba(1.0, 1.0, 1.0, 0.55),
        20.0,
    );
}

fn draw_setting_row(
    ui: &mut GuiContext<'_>,
    id: Id,
    row: Rect,
    icon: Icon,
    label: &str,
    value: &mut f32,
    min: f32,
    max: f32,
    integer: bool,
) -> bool {
    let icon_r = Rect::from_pos_size(
        row.min_x,
        row.min_y + (row.height() - 22.0) * 0.5,
        22.0,
        22.0,
    );
    ui.icon_centered(icon_r, icon, style::TEXT_MUTED, 16.0);

    let label_x = icon_r.max_x + style::SPACE_2;
    let label_w = 140.0;
    ui.label_at(
        label_x,
        row.min_y + (row.height() - 14.0) * 0.5,
        label,
        style::TEXT_DIM,
        FONT_SCALE * TYPE_LABEL,
    );

    let value_w = style::CONTROL_VALUE_W + 8.0;
    let value_box = Rect::from_pos_size(
        row.max_x - value_w,
        row.min_y + (row.height() - 26.0) * 0.5,
        value_w,
        26.0,
    );
    let track = Rect::from_min_max(
        label_x + label_w + style::SPACE_2,
        row.min_y + (row.height() - 4.0) * 0.5,
        value_box.min_x - style::SPACE_3,
        row.min_y + (row.height() + 4.0) * 0.5,
    );

    let edit_id = id.child("edit");
    let editing = ui.state.text_focus == Some(edit_id);
    let value_hovered = ui.pointer_in(value_box);
    if value_hovered {
        ui.state.set_hot(edit_id);
    }
    if value_hovered && ui.input.primary_pressed {
        ui.state.text_focus = Some(edit_id);
        ui.state.text_buffer = if integer {
            format!("{:.0}", *value)
        } else {
            format!("{:.2}", *value)
        };
        ui.state.text_enter = false;
        ui.state.active = Some(edit_id);
    }

    let mut changed = false;
    let span = (max - min).max(1e-6);

    if editing {
        if !ui.input.text.is_empty() {
            for ch in ui.input.text.chars() {
                if ch.is_ascii_digit() || ch == '.' || ch == '-' {
                    ui.state.text_buffer.push(ch);
                }
            }
        }
        if ui.input.backspace_pressed {
            ui.state.text_buffer.pop();
        }
        let commit = ui.state.text_enter || ui.input.enter_pressed;
        let cancel = ui.input.escape_pressed;
        let clicked_away = ui.input.primary_pressed && !value_hovered;
        if commit || clicked_away {
            if let Ok(parsed) = ui.state.text_buffer.parse::<f32>() {
                let mut v = parsed.clamp(min, max);
                if integer {
                    v = v.round();
                }
                if (v - *value).abs() > 1e-6 {
                    *value = v;
                    changed = true;
                }
            }
            ui.state.clear_text_focus();
        } else if cancel {
            ui.state.clear_text_focus();
        }
    } else {
        let track_hit =
            Rect::from_min_max(track.min_x - 4.0, row.min_y, track.max_x + 4.0, row.max_y);
        let hovered = ui.pointer_in(track_hit);
        if hovered {
            ui.state.set_hot(id);
        }
        if hovered && ui.input.primary_pressed {
            ui.state.active = Some(id);
        }
        if ui.state.is_active(id) {
            if let Some((px, _)) = ui.input.pointer {
                let t = ((px - track.min_x) / track.width().max(1.0)).clamp(0.0, 1.0);
                let mut v = min + t * span;
                if integer {
                    v = v.round();
                }
                if (v - *value).abs() > 1e-6 {
                    *value = v.clamp(min, max);
                    changed = true;
                }
            }
        }
    }

    let t = ((*value - min) / span).clamp(0.0, 1.0);
    ui.panel_rounded(track, style::TRACK_BG, 2.0);
    let fill_w = track.width() * t;
    if fill_w > 0.5 {
        ui.panel_rounded(
            Rect::from_pos_size(track.min_x, track.min_y, fill_w, track.height()),
            style::ACCENT,
            2.0,
        );
    }
    let thumb_s = style::SLIDER_THUMB;
    let thumb_x = track.min_x + fill_w - thumb_s * 0.5;
    let thumb = Rect::from_pos_size(
        thumb_x.clamp(track.min_x - 2.0, track.max_x - thumb_s + 2.0),
        track.center_y() - thumb_s * 0.5,
        thumb_s,
        thumb_s,
    );
    ui.panel_rounded(thumb, style::THUMB_BG, thumb_s * 0.5);

    ui.panel_rounded(
        value_box,
        if editing || value_hovered {
            style::BUTTON_HOVER
        } else {
            style::INPUT_BG
        },
        style::RADIUS_SM,
    );
    let shown = if editing {
        ui.state.text_buffer.clone()
    } else if integer {
        format!("{:.0}", *value)
    } else {
        format!("{:.2}", *value)
    };
    let tw = DrawList::text_width(&shown, FONT_SCALE * 0.88);
    ui.label_at(
        value_box.min_x + (value_box.width() - tw) * 0.5,
        value_box.min_y + (value_box.height() - 12.0) * 0.5,
        &shown,
        style::TEXT,
        FONT_SCALE * 0.88,
    );

    changed
}

fn split_desc(text: &str, approx_chars: usize) -> (&str, &str) {
    if text.len() <= approx_chars {
        return (text, "");
    }
    let bytes = text.as_bytes();
    let mut split = approx_chars.min(bytes.len());
    while split > 0 && bytes[split] != b' ' {
        split -= 1;
    }
    if split == 0 {
        return (text, "");
    }
    (text[..split].trim_end(), text[split..].trim_start())
}

/// Split description into up to two lines that fit `max_w` (second line may still ellipsize).
fn split_desc_to_width(text: &str, scale: f32, max_w: f32) -> (String, String) {
    if DrawList::text_width(text, scale) <= max_w {
        return (text.to_string(), String::new());
    }
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return (String::new(), String::new());
    }
    let mut line1 = String::new();
    let mut i = 0;
    while i < words.len() {
        let candidate = if line1.is_empty() {
            words[i].to_string()
        } else {
            format!("{} {}", line1, words[i])
        };
        if DrawList::text_width(&candidate, scale) > max_w && !line1.is_empty() {
            break;
        }
        line1 = candidate;
        i += 1;
        if DrawList::text_width(&line1, scale) > max_w {
            break;
        }
    }
    let rest = words[i..].join(" ");
    (line1, rest)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscardConfirmChoice {
    Discard,
    Cancel,
}

/// Modal confirm when leaving a dirty project.
pub fn draw_discard_confirm(ui: &mut GuiContext<'_>) -> Option<DiscardConfirmChoice> {
    ui.begin_overlay();
    // Dim the screen.
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
        "Unsaved changes",
        style::TEXT,
        FONT_SCALE * 1.2,
    );
    ui.label_at(
        dialog.min_x + PAD * 1.5,
        dialog.min_y + 48.0,
        "Discard changes to the current project?",
        style::TEXT_DIM,
        FONT_SCALE,
    );

    let btn_w = 120.0;
    let btn_h = 34.0;
    let btn_y = dialog.max_y - PAD - btn_h;
    let cancel_r =
        Rect::from_pos_size(dialog.max_x - PAD - btn_w * 2.0 - 10.0, btn_y, btn_w, btn_h);
    let discard_r = Rect::from_pos_size(dialog.max_x - PAD - btn_w, btn_y, btn_w, btn_h);

    let mut choice = None;
    if chip_button(ui, Id::new("discard_cancel"), "Cancel", cancel_r, false) {
        choice = Some(DiscardConfirmChoice::Cancel);
    }
    if chip_button(ui, Id::new("discard_ok"), "Discard", discard_r, true) {
        choice = Some(DiscardConfirmChoice::Discard);
    }
    if ui.input.escape_pressed {
        choice = Some(DiscardConfirmChoice::Cancel);
    }

    ui.end_overlay();
    choice
}
