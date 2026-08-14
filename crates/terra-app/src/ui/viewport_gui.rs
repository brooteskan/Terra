//! Contextual viewport toolbar, brush bar, and view gizmo.

use crate::ui::style::{self, FONT_SCALE, GAP, PAD, TYPE_BODY, TYPE_CAPTION, TYPE_LABEL};
use crate::ui::{EditorTool, LightingPreset, Preview2dMode, UiState};
use terra_core::document::TerrainDocument;
use terra_gui::{
    checkbox, combo, section_header, slider_f32, slider_i32, Color, DrawList, GuiContext, Icon, Id,
    Rect,
};
use terra_render::{QualityPreset, ViewportRendererMode};

/// Bottom viewport tool-mode buttons (legacy authoring modes; kept for tools rail).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewportToolMode {
    Move,
    Measure,
    Sculpt,
    Mask,
    Biome,
}

impl ViewportToolMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Move => "Move",
            Self::Measure => "Measure",
            Self::Sculpt => "Brush",
            Self::Mask => "Mask",
            Self::Biome => "Biome",
        }
    }

    pub fn icon(self) -> Icon {
        match self {
            Self::Move => Icon::Move,
            Self::Measure => Icon::Activity,
            Self::Sculpt => Icon::Paintbrush,
            Self::Mask => Icon::CircleDot,
            Self::Biome => Icon::Sparkles,
        }
    }

    pub fn matches(self, tool: EditorTool) -> bool {
        match self {
            Self::Move => tool.is_move(),
            Self::Measure => matches!(tool, EditorTool::Measure),
            Self::Sculpt => {
                tool.is_sculpt() || matches!(tool, EditorTool::PaintMask | EditorTool::PaintBiome)
            }
            Self::Mask => matches!(tool, EditorTool::PaintMask),
            Self::Biome => matches!(tool, EditorTool::PaintBiome),
        }
    }
}

/// Camera speed multipliers shown in the bottom viewport bar.
pub const CAMERA_SPEED_STEPS: [f32; 5] = [0.25, 0.5, 1.0, 2.0, 4.0];

pub fn format_camera_speed(mult: f32) -> String {
    if (mult - mult.round()).abs() < 0.01 {
        format!("{}x", mult.round() as i32)
    } else if (mult - 0.25).abs() < 0.01 {
        "0.25x".into()
    } else if (mult - 0.5).abs() < 0.01 {
        "0.5x".into()
    } else {
        format!("{:.1}x", mult)
    }
}

/// Shared viewport overlay slots so chips / gizmo / scale don't collide.
pub struct ViewportOverlayLayout {
    pub vp: Rect,
    pub gizmo: Rect,
    /// Top Y for progressive status (below gizmo).
    pub progressive_top: f32,
    /// Top Y for biome/region viz chips (below progressive column).
    pub chip_top: f32,
    pub scale_label_y: f32,
    pub scale_bar: Rect,
    /// Bottom-right dirty-tiles panel origin.
    pub dirty_origin: (f32, f32),
}

pub fn compute_overlay_layout(
    vp: Rect,
    brush_active: bool,
    progressive: bool,
) -> ViewportOverlayLayout {
    // Always reserve the tool-mode bar; brush settings sit above it when active.
    let mut bottom_clear = style::VIEWPORT_TOOL_MODE_BAR_H + GAP;
    if brush_active {
        bottom_clear += style::BRUSH_BAR_H + GAP;
    }
    let scale_bottom = vp.max_y - PAD - bottom_clear;
    let scale_label_y = scale_bottom - style::SCALE_BAR_H;
    let scale_bar = Rect::from_pos_size(
        vp.min_x + PAD,
        scale_label_y + 14.0,
        style::SCALE_BAR_W,
        2.0,
    );

    let gizmo = Rect::from_pos_size(
        vp.max_x - PAD - style::VIEW_GIZMO_SIZE,
        vp.min_y + PAD,
        style::VIEW_GIZMO_SIZE,
        style::VIEW_GIZMO_SIZE,
    );
    let progressive_top = gizmo.max_y + GAP;
    let chip_top = if progressive {
        progressive_top + 24.0 + GAP
    } else {
        progressive_top
    };

    let dirty_w = 132.0f32;
    let dirty_h = 132.0f32;
    let dirty_y = {
        let above_brush = vp.max_y - PAD - bottom_clear - dirty_h;
        above_brush.min(scale_label_y - GAP - dirty_h)
    };
    let dirty_origin = (vp.max_x - PAD - dirty_w, dirty_y.max(vp.min_y + PAD));

    ViewportOverlayLayout {
        vp,
        gizmo,
        progressive_top,
        chip_top,
        scale_label_y,
        scale_bar,
        dirty_origin,
    }
}

fn format_scale_label(world_size_m: f32) -> Option<String> {
    if world_size_m <= 0.0 {
        return None;
    }
    // Decorative bar â‰ˆ 10% of world extent, snapped to a readable distance.
    let raw = (world_size_m * 0.1).max(1.0);
    let nice = nice_distance(raw);
    Some(format_distance(nice))
}

fn nice_distance(m: f32) -> f32 {
    let exp = 10f32.powf(m.log10().floor());
    let n = m / exp;
    let nice_n = if n < 1.5 {
        1.0
    } else if n < 3.5 {
        2.0
    } else if n < 7.5 {
        5.0
    } else {
        10.0
    };
    nice_n * exp
}

fn format_distance(m: f32) -> String {
    if m >= 1000.0 {
        let km = m / 1000.0;
        if (km - km.round()).abs() < 0.05 {
            format!("{:.0} km", km)
        } else {
            format!("{:.1} km", km)
        }
    } else {
        format!("{:.0} m", m)
    }
}

pub fn draw_viewport_overlays(
    ui: &mut GuiContext<'_>,
    ui_state: &mut UiState,
    doc: &TerrainDocument,
    world_size_m: f32,
) -> Vec<crate::ui::actions::PanelAction> {
    let mut actions = Vec::new();
    let vp = ui.viewport_rect();
    if vp.width() < 120.0 || vp.height() < 80.0 {
        return actions;
    }

    let brush_active = ui_state.editor_tool.is_brush();
    let progressive = ui_state.progressive_renderer_active;
    let layout = compute_overlay_layout(vp, brush_active, progressive);

    let mode_bar_w = estimate_viewport_mode_bar_width(ui_state, vp);
    let display_bar_w = estimate_viewport_display_bar_width();
    let cluster_gap = GAP;
    let cluster_w = mode_bar_w + cluster_gap + display_bar_w;
    let gizmo_left = vp.max_x - PAD - style::VIEW_GIZMO_SIZE - GAP;
    let mut cluster_x = vp.min_x + (vp.width() - cluster_w.min(vp.width() - PAD * 2.0)) * 0.5;
    if cluster_x + cluster_w > gizmo_left {
        cluster_x = (gizmo_left - cluster_w).max(vp.min_x + PAD);
    }
    let mode_bar = draw_viewport_mode_bar(ui, ui_state, doc, vp, cluster_x, mode_bar_w);
    draw_viewport_display_bar(
        ui,
        ui_state,
        mode_bar.min_x + mode_bar.width() + cluster_gap,
        mode_bar.min_y,
        display_bar_w,
    );
    draw_view_gizmo(ui, ui_state, layout.gizmo);
    if progressive {
        draw_progressive_status(ui, ui_state, &layout);
    }

    draw_viewport_tool_mode_bar(ui, ui_state, doc, vp, &mut actions);

    if brush_active {
        draw_brush_bar(ui, ui_state, vp);
    }

    if let Some(label) = format_scale_label(world_size_m) {
        ui.label_at(
            vp.min_x + PAD,
            layout.scale_label_y,
            &label,
            style::TEXT_DIM,
            FONT_SCALE * TYPE_LABEL,
        );
        ui.panel(layout.scale_bar, style::TEXT_DIM);
    }

    if ui_state.viewport_overlays.dirty_tiles {
        draw_dirty_tiles_overlay(ui, ui_state, &layout);
    }

    // Top-right chip column (below gizmo / progressive) â€” avoids gizmo overlap.
    let chip_top = layout.chip_top;
    if ui_state.preview_mode == Preview2dMode::Biome {
        if let Some((w, h, rgba)) = ui_state.preview_rgba.as_ref() {
            if *w > 1 && *h > 1 {
                draw_biome_map_overlay(ui, vp, *w, *h, rgba, chip_top);
            }
        }
    }
    actions
}

fn draw_biome_map_overlay(
    ui: &mut GuiContext<'_>,
    vp: Rect,
    width: u32,
    height: u32,
    rgba: &[u8],
    top_y: f32,
) -> f32 {
    let size = 148.0f32.min(vp.width() * 0.22).min(vp.height() * 0.28);
    let panel = Rect::from_pos_size(vp.max_x - PAD - size, top_y.max(vp.min_y + PAD), size, size);
    ui.begin_overlay();
    ui.panel_rounded(panel, style::OVERLAY_CHIP_BG, style::RADIUS_SM);
    ui.label_at(
        panel.min_x + 8.0,
        panel.min_y + 6.0,
        "BIOME MAP",
        style::TEXT_DIM,
        FONT_SCALE * TYPE_CAPTION,
    );
    let img = Rect::from_pos_size(
        panel.min_x + 6.0,
        panel.min_y + 24.0,
        panel.width() - 12.0,
        panel.height() - 30.0,
    );
    ui.panel(img, style::SURFACE);
    // Prefer a single atlas image (pack downscales >512); panel grid is fallback only.
    if width > 1 && height > 1 && rgba.len() >= (width as usize) * (height as usize) * 4 {
        ui.image(img, width, height, rgba);
    } else {
        draw_rgba_grid(ui, img, width, height, rgba);
    }
    ui.end_overlay();
    size
}

fn draw_rgba_grid(ui: &mut GuiContext<'_>, img: Rect, width: u32, height: u32, rgba: &[u8]) {
    let cells = 28u32;
    let cell_w = img.width() / cells as f32;
    let cell_h = img.height() / cells as f32;
    for cy in 0..cells {
        for cx in 0..cells {
            let u = (cx as f32 + 0.5) / cells as f32;
            let v = (cy as f32 + 0.5) / cells as f32;
            let ix = ((u * (width - 1) as f32).round() as u32).min(width - 1);
            let iy = ((v * (height - 1) as f32).round() as u32).min(height - 1);
            let idx = ((iy * width + ix) * 4) as usize;
            if idx + 3 >= rgba.len() {
                continue;
            }
            let a = rgba[idx + 3];
            if a < 8 {
                continue;
            }
            let cell = Rect::from_pos_size(
                img.min_x + cx as f32 * cell_w,
                img.min_y + cy as f32 * cell_h,
                cell_w.max(1.0),
                cell_h.max(1.0),
            );
            ui.panel(
                cell,
                Color::rgba(
                    rgba[idx] as f32 / 255.0,
                    rgba[idx + 1] as f32 / 255.0,
                    rgba[idx + 2] as f32 / 255.0,
                    (a as f32 / 255.0) * 0.95,
                ),
            );
        }
    }
}

fn draw_glass_bar(ui: &mut GuiContext<'_>, bar: Rect, radius: f32) {
    ui.panel_rounded(bar, Color::rgba(1.0, 1.0, 1.0, 0.10), radius);
    let fill = Rect::from_pos_size(
        bar.min_x + 1.0,
        bar.min_y + 1.0,
        (bar.width() - 2.0).max(0.0),
        (bar.height() - 2.0).max(0.0),
    );
    ui.panel_rounded(fill, style::VIEWPORT_TOOLBAR_BG, (radius - 1.0).max(0.0));
}

fn draw_bar_separator(ui: &mut GuiContext<'_>, x: f32, bar: Rect) {
    let h = (bar.height() - 14.0).max(12.0);
    ui.panel(
        Rect::from_pos_size(x, bar.min_y + (bar.height() - h) * 0.5, 1.0, h),
        Color::rgba(1.0, 1.0, 1.0, 0.14),
    );
}

fn estimate_viewport_mode_bar_width(state: &UiState, vp: Rect) -> f32 {
    let font_scale = FONT_SCALE * TYPE_LABEL;
    let pad_x = 8.0;
    let gap = 2.0;
    let label_pad_x = 10.0;
    let sep_gap = 10.0;
    let labels = ["Terrain", "Height", "Slope", "Flow", "Mask"];
    let modes_w: f32 = labels
        .iter()
        .map(|label| (DrawList::text_width(label, font_scale) + label_pad_x * 2.0).max(44.0))
        .sum::<f32>()
        + gap * (labels.len().saturating_sub(1) as f32);
    let lighting_w = (18.0
        + 4.0
        + DrawList::text_width(
            if state.lighting_customized {
                ""
            } else {
                state.lighting_preset.label()
            },
            font_scale,
        )
        + 14.0
        + label_pad_x)
        .max(72.0)
        .min(148.0);
    let render_w = (DrawList::text_width("Render", font_scale) + label_pad_x * 2.0).max(64.0);
    let content_w =
        modes_w + sep_gap + 1.0 + sep_gap + lighting_w + sep_gap + 1.0 + sep_gap + render_w;
    (content_w + pad_x * 2.0).min(vp.width() - PAD * 2.0)
}

fn estimate_viewport_display_bar_width() -> f32 {
    let bar_h = style::VIEWPORT_TOOLBAR_H;
    let pad_x = 6.0;
    let pad_y = 5.0;
    let btn = bar_h - pad_y * 2.0;
    let gap = 2.0;
    let n = 5.0;
    pad_x * 2.0 + btn * n + gap * (n - 1.0)
}

/// Top viewport mode bar at an explicit origin (clustered with the display aids bar).
fn draw_viewport_mode_bar(
    ui: &mut GuiContext<'_>,
    state: &mut UiState,
    _doc: &TerrainDocument,
    vp: Rect,
    bar_x: f32,
    bar_w: f32,
) -> Rect {
    #[derive(Clone, Copy)]
    enum ModeItem {
        Terrain,
        Height,
        Slope,
        Flow,
        Mask,
    }

    let items: [(ModeItem, &str); 5] = [
        (ModeItem::Terrain, "Terrain"),
        (ModeItem::Height, "Height"),
        (ModeItem::Slope, "Slope"),
        (ModeItem::Flow, "Flow"),
        (ModeItem::Mask, "Mask"),
    ];

    let bar_h = style::VIEWPORT_TOOLBAR_H;
    let pad_x = 8.0;
    let pad_y = 5.0;
    let btn_h = bar_h - pad_y * 2.0;
    let font_scale = FONT_SCALE * TYPE_LABEL;
    let gap = 2.0;
    let label_pad_x = 10.0;
    let bar_radius = style::RADIUS_MD;
    let item_radius = style::RADIUS_SM;
    let sep_gap = 10.0;

    let preset_label = if state.lighting_customized {
        ""
    } else {
        state.lighting_preset.label()
    };
    let lighting_w =
        (18.0 + 4.0 + DrawList::text_width(preset_label, font_scale) + 14.0 + label_pad_x)
            .max(72.0)
            .min(148.0);

    let widths: Vec<f32> = items
        .iter()
        .map(|(_, label)| (DrawList::text_width(label, font_scale) + label_pad_x * 2.0).max(44.0))
        .collect();
    let modes_w: f32 = widths.iter().sum::<f32>() + gap * (items.len().saturating_sub(1) as f32);
    let bar_w = bar_w.min(vp.width() - PAD * 2.0);
    let bar_y = vp.min_y + PAD;
    let bar = Rect::from_pos_size(bar_x, bar_y, bar_w, bar_h);

    ui.begin_overlay();
    draw_glass_bar(ui, bar, bar_radius);

    let btn_y = bar.min_y + pad_y;
    let mut x = bar.min_x + pad_x;
    let modes_end = bar.min_x + pad_x + modes_w;

    for (index, (item, label)) in items.iter().enumerate() {
        let width = widths[index];
        if x + width > modes_end + 1.0 {
            break;
        }
        let rect = Rect::from_pos_size(x, btn_y, width, btn_h);
        let active = match item {
            ModeItem::Terrain => state.preview_mode == Preview2dMode::Lit && !state.is_mask_view(),
            ModeItem::Height => {
                state.preview_mode == Preview2dMode::Height && !state.is_mask_view()
            }
            ModeItem::Slope => state.preview_mode == Preview2dMode::Slope && !state.is_mask_view(),
            ModeItem::Flow => state.preview_mode == Preview2dMode::Flow && !state.is_mask_view(),
            ModeItem::Mask => state.is_mask_view(),
        };
        if mode_button(
            ui,
            Id::new("viewport_mode_bar").with(index as u64),
            rect,
            label,
            active,
            font_scale,
            item_radius,
            false,
        ) {
            state.viewport_lighting_selected = false;
            state.lighting_menu_open = false;
            match item {
                ModeItem::Terrain => {
                    state.leave_mask_view();
                    state.preview_mode = Preview2dMode::Lit;
                }
                ModeItem::Height => {
                    state.leave_mask_view();
                    state.preview_mode = Preview2dMode::Height;
                }
                ModeItem::Slope => {
                    state.leave_mask_view();
                    state.preview_mode = Preview2dMode::Slope;
                }
                ModeItem::Flow => {
                    state.leave_mask_view();
                    state.preview_mode = Preview2dMode::Flow;
                }
                ModeItem::Mask => {
                    state.enter_mask_view();
                }
            }
        }
        x += width + gap;
    }

    x = modes_end + sep_gap;
    draw_bar_separator(ui, x, bar);
    x += 1.0 + sep_gap;

    let lighting_rect = Rect::from_pos_size(x, btn_y, lighting_w, btn_h);
    if lighting_combo_button(
        ui,
        Id::new("viewport_lighting"),
        lighting_rect,
        preset_label,
        state.lighting_menu_open,
        font_scale,
        item_radius,
    ) {
        state.lighting_menu_open = !state.lighting_menu_open;
        state.camera_speed_menu_open = false;
        state.viewport_lighting_selected = false;
    }

    if state.lighting_menu_open {
        ui.with_menu_input(|ui| draw_lighting_menu(ui, state, lighting_rect));
    }

    x = lighting_rect.max_x + sep_gap;
    draw_bar_separator(ui, x, bar);
    x += 1.0 + sep_gap;

    let render_w = (DrawList::text_width("Render", font_scale) + label_pad_x * 2.0).max(64.0);
    let render_rect = Rect::from_pos_size(x, btn_y, render_w, btn_h);
    if lighting_combo_button(
        ui,
        Id::new("viewport_render"),
        render_rect,
        "Render",
        state.viewport_render.menu_open,
        font_scale,
        item_radius,
    ) {
        state.viewport_render.menu_open = !state.viewport_render.menu_open;
        state.lighting_menu_open = false;
        state.camera_speed_menu_open = false;
    }
    if state.viewport_render.menu_open {
        ui.with_menu_input(|ui| draw_viewport_render_menu(ui, state, render_rect));
    }
    ui.end_overlay();

    bar
}

/// Display aids bar (wireframe, grid, bounds, …) — sits beside the mode bar.
fn draw_viewport_display_bar(
    ui: &mut GuiContext<'_>,
    state: &mut UiState,
    bar_x: f32,
    bar_y: f32,
    bar_w: f32,
) {
    #[derive(Clone, Copy)]
    enum DisplayAid {
        Wireframe,
        Grid,
        Bounds,
        Water,
        Contours,
    }

    let aids: [(DisplayAid, Icon, &str, &str); 5] = [
        (
            DisplayAid::Wireframe,
            Icon::Box,
            "Wireframe",
            "Toggle terrain wireframe",
        ),
        (DisplayAid::Grid, Icon::Grid3x3, "Grid", "Toggle world grid"),
        (
            DisplayAid::Bounds,
            Icon::Maximize2,
            "Bounds",
            "Toggle world bounds",
        ),
        (
            DisplayAid::Water,
            Icon::Waves,
            "Water level",
            "Toggle sea-level plane",
        ),
        (
            DisplayAid::Contours,
            Icon::Activity,
            "Contours",
            "Toggle height contours",
        ),
    ];

    let bar_h = style::VIEWPORT_TOOLBAR_H;
    let pad_x = 6.0;
    let pad_y = 5.0;
    let btn = bar_h - pad_y * 2.0;
    let gap = 2.0;
    let bar = Rect::from_pos_size(bar_x, bar_y, bar_w, bar_h);

    ui.begin_overlay();
    draw_glass_bar(ui, bar, style::RADIUS_MD);

    let mut x = bar.min_x + pad_x;
    let btn_y = bar.min_y + pad_y;
    for (i, (aid, icon, title, tip)) in aids.iter().enumerate() {
        let rect = Rect::from_pos_size(x, btn_y, btn, btn);
        let active = match aid {
            DisplayAid::Wireframe => state.viewport_overlays.wireframe,
            DisplayAid::Grid => state.viewport_overlays.grid,
            DisplayAid::Bounds => state.viewport_overlays.world_bounds,
            DisplayAid::Water => state.viewport_overlays.water_level,
            DisplayAid::Contours => state.viewport_overlays.contours,
        };
        if display_aid_toggle(
            ui,
            Id::new("viewport_display_aid").with(i as u64),
            rect,
            *icon,
            title,
            tip,
            active,
            style::RADIUS_SM,
        ) {
            match aid {
                DisplayAid::Wireframe => {
                    state.viewport_overlays.wireframe = !state.viewport_overlays.wireframe;
                }
                DisplayAid::Grid => {
                    state.viewport_overlays.grid = !state.viewport_overlays.grid;
                }
                DisplayAid::Bounds => {
                    state.viewport_overlays.world_bounds = !state.viewport_overlays.world_bounds;
                }
                DisplayAid::Water => {
                    state.viewport_overlays.water_level = !state.viewport_overlays.water_level;
                }
                DisplayAid::Contours => {
                    state.viewport_overlays.contours = !state.viewport_overlays.contours;
                }
            }
            state.lighting_menu_open = false;
            state.camera_speed_menu_open = false;
        }
        x += btn + gap;
    }
    ui.end_overlay();
}

fn display_aid_toggle(
    ui: &mut GuiContext<'_>,
    id: Id,
    rect: Rect,
    icon: Icon,
    title: &str,
    tip: &str,
    active: bool,
    radius: f32,
) -> bool {
    let hovered = ui.pointer_in(rect);
    if hovered {
        ui.state.set_hot(id);
        ui.queue_tooltip(rect, title, tip, None);
    }
    if hovered && ui.input.primary_pressed {
        ui.state.active = Some(id);
    }
    let clicked = ui.input.primary_released && ui.state.is_active(id) && hovered;
    if active {
        ui.panel_rounded(rect, style::ACCENT, radius);
    } else if hovered {
        ui.panel_rounded(rect, Color::rgba(1.0, 1.0, 1.0, 0.08), radius);
    }
    ui.icon_centered(
        rect,
        icon,
        if active || hovered {
            style::TEXT
        } else {
            style::TEXT_MUTED
        },
        14.0,
    );
    clicked
}

fn lighting_combo_button(
    ui: &mut GuiContext<'_>,
    id: Id,
    rect: Rect,
    preset_label: &str,
    open: bool,
    font_scale: f32,
    radius: f32,
) -> bool {
    let hovered = ui.pointer_in(rect);
    if hovered {
        ui.state.set_hot(id);
        ui.queue_tooltip(
            rect,
            "Lighting",
            "Environment lighting preset for the 3D viewport",
            None,
        );
    }
    if hovered && ui.input.primary_pressed {
        ui.state.active = Some(id);
    }
    let clicked = ui.input.primary_released && ui.state.is_active(id) && hovered;
    if open {
        ui.panel_rounded(rect, style::ACCENT_SOFT, radius);
    } else if hovered {
        ui.panel_rounded(rect, Color::rgba(1.0, 1.0, 1.0, 0.08), radius);
    }
    let fg = if open || hovered {
        style::TEXT
    } else {
        style::TEXT_MUTED
    };
    let icon_sz = 14.0;
    let chevron_slot = 16.0;
    let icon_x = rect.min_x + 8.0;
    ui.icon_at(
        icon_x,
        rect.min_y + (rect.height() - icon_sz) * 0.5,
        Icon::Sun,
        fg,
        icon_sz,
    );
    let label_rect = Rect::from_min_max(
        icon_x + icon_sz + 4.0,
        rect.min_y,
        rect.max_x - chevron_slot,
        rect.max_y,
    );
    ui.label_in_rect(label_rect, preset_label, fg, font_scale);
    ui.icon_centered(
        Rect::from_pos_size(
            rect.max_x - chevron_slot - 2.0,
            rect.min_y,
            chevron_slot,
            rect.height(),
        ),
        Icon::ChevronDown,
        fg,
        12.0,
    );
    clicked
}

fn mode_button(
    ui: &mut GuiContext<'_>,
    id: Id,
    rect: Rect,
    label: &str,
    active: bool,
    font_scale: f32,
    radius: f32,
    dropdown: bool,
) -> bool {
    let hovered = ui.pointer_in(rect);
    if hovered {
        ui.state.set_hot(id);
    }
    if hovered && ui.input.primary_pressed {
        ui.state.active = Some(id);
    }
    let clicked = ui.input.primary_released && ui.state.is_active(id) && hovered;
    if active {
        ui.panel_rounded(rect, style::ACCENT, radius);
    } else if hovered {
        ui.panel_rounded(rect, Color::rgba(1.0, 1.0, 1.0, 0.08), radius);
    }
    let fg = if active || hovered {
        style::TEXT
    } else {
        style::TEXT_MUTED
    };
    if dropdown {
        let chevron_slot = 16.0;
        let label_rect = Rect::from_min_max(
            rect.min_x,
            rect.min_y,
            rect.max_x - chevron_slot,
            rect.max_y,
        );
        ui.label_centered_in_rect(label_rect, label, fg, font_scale);
        ui.icon_centered(
            Rect::from_pos_size(
                rect.max_x - chevron_slot - 2.0,
                rect.min_y,
                chevron_slot,
                rect.height(),
            ),
            Icon::ChevronDown,
            fg,
            12.0,
        );
    } else {
        ui.label_centered_in_rect(rect, label, fg, font_scale);
    }
    clicked
}

fn draw_dirty_tiles_overlay(
    ui: &mut GuiContext<'_>,
    state: &UiState,
    layout: &ViewportOverlayLayout,
) {
    let (tiles_x, tiles_z) = state.dirty_tile_grid;
    if tiles_x == 0 || tiles_z == 0 {
        return;
    }
    let panel_w = 132.0f32;
    let panel_h = 132.0f32;
    let (origin_x, origin_y) = layout.dirty_origin;
    let panel = Rect::from_pos_size(origin_x, origin_y, panel_w, panel_h);
    ui.begin_overlay();
    ui.panel_rounded(panel, style::OVERLAY_BG, style::RADIUS_SM);
    let title = format!("Dirty tiles ({})", state.dirty_tile_ids.len());
    ui.label_at(
        panel.min_x + 8.0,
        panel.min_y + 6.0,
        &title,
        style::TEXT_DIM,
        FONT_SCALE * TYPE_LABEL,
    );
    let grid = Rect::from_pos_size(
        panel.min_x + 8.0,
        panel.min_y + 24.0,
        panel_w - 16.0,
        panel_h - 32.0,
    );
    ui.panel(grid, Color::rgba(0.16, 0.18, 0.22, 0.9));
    let cell_w = grid.width() / tiles_x as f32;
    let cell_h = grid.height() / tiles_z as f32;
    for &(tx, tz) in &state.dirty_tile_ids {
        if tx >= tiles_x || tz >= tiles_z {
            continue;
        }
        let cell = Rect::from_pos_size(
            grid.min_x + tx as f32 * cell_w,
            grid.min_y + tz as f32 * cell_h,
            cell_w.max(1.0),
            cell_h.max(1.0),
        );
        ui.panel(cell, Color::rgba(0.95, 0.55, 0.15, 0.75));
    }
    // Light grid lines for readability when few tiles.
    if tiles_x * tiles_z <= 64 {
        for tx in 0..=tiles_x {
            let x = grid.min_x + tx as f32 * cell_w;
            ui.panel(
                Rect::from_pos_size(x, grid.min_y, 1.0, grid.height()),
                Color::rgba(1.0, 1.0, 1.0, 0.12),
            );
        }
        for tz in 0..=tiles_z {
            let y = grid.min_y + tz as f32 * cell_h;
            ui.panel(
                Rect::from_pos_size(grid.min_x, y, grid.width(), 1.0),
                Color::rgba(1.0, 1.0, 1.0, 0.12),
            );
        }
    }
    ui.end_overlay();
}

fn draw_viewport_render_menu(ui: &mut GuiContext<'_>, state: &mut UiState, anchor: Rect) {
    let menu_w = 300.0;
    let menu_h = if state.viewport_render.advanced_open {
        790.0
    } else {
        560.0
    };
    let rect = Rect::from_pos_size(
        (anchor.max_x - menu_w).max(8.0),
        anchor.max_y + 4.0,
        menu_w,
        menu_h,
    );
    // Scroll offset must persist across frames — a per-frame `0.0` local would
    // snap the panel back to the top every frame, defeating wheel and thumb drag.
    let mut scroll = state.viewport_render.menu_scroll;
    let was_open = state.viewport_render.menu_open;
    let mut open = was_open;
    if ui.begin_window(
        Id::new("viewport_render_panel"),
        "Viewport Rendering",
        rect,
        &mut open,
        &mut scroll,
    ) {
        let vr = &mut state.viewport_render;
        let mode_labels: Vec<&str> = ViewportRendererMode::ALL
            .iter()
            .map(|m| m.label())
            .collect();
        let mut mode_idx = ViewportRendererMode::ALL
            .iter()
            .position(|m| *m == vr.mode)
            .unwrap_or(2);
        if combo(ui, "Mode", &mut mode_idx, &mode_labels) {
            vr.mode = ViewportRendererMode::ALL[mode_idx];
            if vr.mode == ViewportRendererMode::ProgressiveRayTraced {
                state.lighting_preset = LightingPreset::Progressive;
                state.lighting_customized = false;
            }
        }

        let preset_labels: Vec<&str> = QualityPreset::ALL.iter().map(|p| p.label()).collect();
        let mut preset_idx = QualityPreset::ALL
            .iter()
            .position(|p| *p == vr.preset)
            .unwrap_or(1);
        if combo(ui, "Quality preset", &mut preset_idx, &preset_labels) {
            vr.preset = QualityPreset::ALL[preset_idx];
        }

        let mut target_fps = vr.target_fps as i32;
        if slider_i32(ui, "Target FPS", &mut target_fps, 24, 120) {
            vr.target_fps = target_fps as f32;
        }
        let mut max_spp = vr.max_spp as i32;
        if slider_i32(ui, "Max spp", &mut max_spp, 8, 512) {
            vr.max_spp = max_spp as u32;
        }
        checkbox(ui, "Dynamic resolution", &mut vr.dynamic_resolution);
        checkbox(ui, "Denoise", &mut vr.denoise);

        ui.separator();
        section_header(ui, "Lighting");
        let mut lighting_edited = false;
        lighting_edited |= slider_f32(ui, "Sun azimuth °", &mut vr.sun_azimuth_deg, 0.0, 360.0);
        lighting_edited |= slider_f32(ui, "Sun elevation °", &mut vr.sun_elevation_deg, 0.0, 90.0);
        lighting_edited |= slider_f32(ui, "Sun intensity", &mut vr.sun_intensity, 0.0, 3.0);
        lighting_edited |= slider_f32(ui, "Exposure", &mut vr.exposure, 0.1, 4.0);
        lighting_edited |= slider_f32(ui, "Sky R", &mut vr.sky_color[0], 0.0, 1.0);
        lighting_edited |= slider_f32(ui, "Sky G", &mut vr.sky_color[1], 0.0, 1.0);
        lighting_edited |= slider_f32(ui, "Sky B", &mut vr.sky_color[2], 0.0, 1.0);
        section_header(ui, "Raster shading");
        lighting_edited |= slider_f32(ui, "Ambient", &mut vr.ambient_strength, 0.0, 2.0);
        lighting_edited |= slider_f32(ui, "Shadow", &mut vr.shadow_strength, 0.0, 1.0);
        lighting_edited |= slider_f32(ui, "Fog", &mut vr.fog_strength, 0.0, 2.0);
        if lighting_edited {
            // Any lighting edit detaches from the named preset (blank preset field).
            state.lighting_customized = true;
        }

        ui.separator();
        if button_toggle_advanced(ui, &mut vr.advanced_open) {
            // toggled in helper
        }
        if vr.advanced_open {
            section_header(ui, "Advanced");
            let mut interactive_spp = vr.interactive_spp as i32;
            if slider_i32(ui, "Interactive spp", &mut interactive_spp, 1, 4) {
                vr.interactive_spp = interactive_spp as u32;
            }
            let mut refining_spp = vr.refining_spp as i32;
            if slider_i32(ui, "Refining spp", &mut refining_spp, 1, 8) {
                vr.refining_spp = refining_spp as u32;
            }
            let mut bounces = vr.max_bounces_refining as i32;
            if slider_i32(ui, "Max bounces", &mut bounces, 1, 8) {
                vr.max_bounces_refining = bounces as u32;
            }
            slider_f32(
                ui,
                "Min internal scale",
                &mut vr.min_internal_scale,
                0.25,
                1.0,
            );
            slider_f32(
                ui,
                "Max internal scale",
                &mut vr.max_internal_scale,
                0.25,
                1.0,
            );
            slider_f32(ui, "History clamp", &mut vr.history_clamp_k, 0.5, 3.0);
            slider_f32(ui, "Converge fraction", &mut vr.converge_fraction, 0.5, 1.0);
            let mut settling_spp = vr.settling_spp as i32;
            if slider_i32(ui, "Settling spp", &mut settling_spp, 1, 8) {
                vr.settling_spp = settling_spp as u32;
            }
            section_header(ui, "Developer");
            let debug_labels = ["Final", "Noisy", "Variance", "Samples", "History"];
            let mut debug_idx = vr.debug_viz_mode.min(4) as usize;
            if combo(ui, "Debug viz", &mut debug_idx, &debug_labels) {
                vr.debug_viz_mode = debug_idx as u32;
            }
        }
        ui.end_window(&mut scroll);
    }
    state.viewport_render.menu_scroll = scroll;
    state.viewport_render.menu_open = open;
    // Persist render prefs when the panel closes after edits.
    if !open && was_open {
        state.layout_dirty = true;
    }
}

fn button_toggle_advanced(ui: &mut GuiContext<'_>, open: &mut bool) -> bool {
    let row = ui.allocate(style::ROW_H);
    let id = Id::new("render_advanced_toggle");
    let hovered = ui.pointer_in(row);
    if hovered {
        ui.state.set_hot(id);
    }
    if hovered && ui.input.primary_pressed {
        ui.state.active = Some(id);
    }
    let toggled = ui.input.primary_released && ui.state.is_active(id) && hovered;
    if toggled {
        *open = !*open;
    }
    ui.panel(
        row,
        if hovered {
            style::BUTTON_HOVER
        } else {
            style::BUTTON_BG
        },
    );
    ui.label_at(
        row.min_x + 8.0,
        row.min_y + 4.0,
        if *open {
            "Advanced ▲"
        } else {
            "Advanced ▼"
        },
        style::TEXT_DIM,
        FONT_SCALE * TYPE_LABEL,
    );
    toggled
}

fn draw_lighting_menu(ui: &mut GuiContext<'_>, state: &mut UiState, anchor: Rect) {
    ui.begin_overlay();
    let item_h = 26.0;
    let menu_w = 140.0;
    let menu = Rect::from_pos_size(
        (anchor.max_x - menu_w).max(8.0),
        anchor.max_y + 4.0,
        menu_w,
        item_h * LightingPreset::ALL.len() as f32 + 8.0,
    );
    ui.panel_rounded(menu, style::POPUP_BG, style::RADIUS_SM);
    if ui.pointer_in(menu) {
        ui.state.set_hot(Id::new("lighting_menu"));
    }
    for (i, preset) in LightingPreset::ALL.iter().enumerate() {
        let row = Rect::from_pos_size(
            menu.min_x + 4.0,
            menu.min_y + 4.0 + i as f32 * item_h,
            menu.width() - 8.0,
            item_h - 2.0,
        );
        let id = Id::new("lighting_item").with(i as u64);
        let hovered = ui.pointer_in(row);
        let selected = state.lighting_preset == *preset;
        if hovered {
            ui.state.set_hot(id);
        }
        if hovered && ui.input.primary_pressed {
            ui.state.active = Some(id);
        }
        if ui.input.primary_released && ui.state.is_active(id) && hovered {
            state.lighting_preset = *preset;
            state.lighting_customized = false;
            if preset.is_progressive() {
                state.viewport_render.mode =
                    terra_render::ViewportRendererMode::ProgressiveRayTraced;
            }
            state.lighting_menu_open = false;
            state.viewport_lighting_selected = false;
            // Lighting is presentation for the 3D terrain view â€” leave analysis
            // modes alone, but exit mask chrome so the viewport stays visible.
            if state.is_mask_view() {
                state.leave_mask_view();
                state.preview_mode = Preview2dMode::Lit;
            }
        }
        if selected || hovered {
            ui.panel_rounded(
                row,
                if selected {
                    style::SELECTED_BG
                } else {
                    style::HOVER_BG
                },
                style::RADIUS_SM,
            );
        }
        ui.label_at(
            row.min_x + 8.0,
            row.min_y + 5.0,
            preset.label(),
            style::TEXT,
            FONT_SCALE * TYPE_BODY,
        );
    }
    ui.end_overlay();

    if ui.input.primary_pressed {
        if let Some((px, py)) = ui.input.pointer {
            if !menu.contains(px, py) && !anchor.contains(px, py) {
                state.lighting_menu_open = false;
            }
        }
    }
}

fn draw_progressive_status(
    ui: &mut GuiContext<'_>,
    state: &UiState,
    layout: &ViewportOverlayLayout,
) {
    let profile = &state.profile;
    let label = match profile.interaction_state {
        "Interactive" => "Interactive".to_string(),
        "Settling" => "Settling".to_string(),
        "Converged" => format!(
            "Converged  {} spp",
            state.progressive_samples.min(profile.max_spp)
        ),
        "Refining" => {
            let pct = (profile.convergence_fraction * 100.0).clamp(0.0, 100.0) as u32;
            format!(
                "Refining {}%  {} / {} spp",
                pct,
                state.progressive_samples.min(profile.max_spp),
                profile.max_spp
            )
        }
        other => format!("{other}  {} spp", state.progressive_samples),
    };
    let width = DrawList::text_width(&label, FONT_SCALE * TYPE_LABEL) + 18.0;
    let panel = Rect::from_pos_size(
        layout.vp.max_x - PAD - width,
        layout.progressive_top,
        width,
        24.0,
    );
    ui.panel_rounded(panel, style::OVERLAY_BG, style::RADIUS_SM);
    ui.label_at(
        panel.min_x + 9.0,
        panel.min_y + 5.0,
        &label,
        style::STATUS_ACCENT,
        FONT_SCALE * TYPE_LABEL,
    );
}

fn draw_view_gizmo(ui: &mut GuiContext<'_>, state: &UiState, gizmo: Rect) {
    // Thin RGB axes â€” floating viewport chrome (no cube / ring).
    let cx = gizmo.center_x();
    let cy = gizmo.center_y() + 1.0;
    let scale = 22.0; // pixels per unit
    let axis_len = 1.0;

    let yaw = state.camera_yaw;
    let pitch = state.camera_pitch;
    let cp = pitch.cos();
    let sp = pitch.sin();
    let cyaw = yaw.cos();
    let syaw = yaw.sin();

    let fx = -cyaw * cp;
    let fy = -sp;
    let fz = -syaw * cp;

    let rx = -fz;
    let rz = fx;
    let rlen = (rx * rx + rz * rz).sqrt();
    let (rx, ry, rz) = if rlen > 1e-5 {
        (rx / rlen, 0.0, rz / rlen)
    } else {
        (-syaw, 0.0, cyaw)
    };

    let ux = ry * fz - rz * fy;
    let uy = rz * fx - rx * fz;
    let uz = rx * fy - ry * fx;

    let project = |ax: f32, ay: f32, az: f32| -> (f32, f32, f32) {
        let sx = ax * rx + ay * ry + az * rz;
        let sy = -(ax * ux + ay * uy + az * uz);
        let depth = ax * fx + ay * fy + az * fz;
        (cx + sx * scale, cy + sy * scale, depth)
    };

    // Soft centre disc so axes read against busy terrain.
    ui.panel_rounded(
        Rect::from_pos_size(cx - 3.0, cy - 3.0, 6.0, 6.0),
        Color::rgba(0.08, 0.09, 0.11, 0.55),
        3.0,
    );

    let mut axes = [
        (
            "X",
            Color::rgb(0.94, 0.32, 0.32),
            project(axis_len, 0.0, 0.0),
        ),
        (
            "Y",
            Color::rgb(0.38, 0.88, 0.42),
            project(0.0, axis_len, 0.0),
        ),
        (
            "Z",
            Color::rgb(0.36, 0.56, 1.0),
            project(0.0, 0.0, axis_len),
        ),
    ];
    axes.sort_by(|a, b| {
        a.2 .2
            .partial_cmp(&b.2 .2)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (label, color, (tx, ty, _depth)) in axes {
        draw_segment(ui, cx, cy, tx, ty, 1.35, color);
        // Axis tip dot.
        ui.panel_rounded(
            Rect::from_pos_size(tx - 2.0, ty - 2.0, 4.0, 4.0),
            color,
            2.0,
        );
        let dx = tx - cx;
        let dy = ty - cy;
        let len = (dx * dx + dy * dy).sqrt().max(1.0);
        let ox = dx / len * 8.0;
        let oy = dy / len * 8.0;
        ui.label_at(
            tx + ox - 3.5,
            ty + oy - 5.5,
            label,
            color,
            FONT_SCALE * TYPE_CAPTION,
        );
    }
}

fn draw_segment(
    ui: &mut GuiContext<'_>,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    thickness: f32,
    color: Color,
) {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 0.5 {
        return;
    }
    let steps = ((len / 1.1).ceil() as i32).max(2);
    let half = thickness * 0.5;
    let inv = 1.0 / steps as f32;
    for i in 0..=steps {
        let t = i as f32 * inv;
        let x = x0 + dx * t;
        let y = y0 + dy * t;
        ui.panel_rounded(
            Rect::from_pos_size(x - half, y - half, thickness, thickness),
            color,
            half,
        );
    }
}

/// Bottom-centered Move / Brush + camera speed bar.
fn draw_viewport_tool_mode_bar(
    ui: &mut GuiContext<'_>,
    ui_state: &mut UiState,
    doc: &TerrainDocument,
    vp: Rect,
    actions: &mut Vec<crate::ui::actions::PanelAction>,
) {
    const MODES: [ViewportToolMode; 3] = [
        ViewportToolMode::Move,
        ViewportToolMode::Measure,
        ViewportToolMode::Sculpt,
    ];

    let pad = 8.0;
    let gap = 4.0;
    let sep_gap = 10.0;
    let btn_w = 52.0;
    let speed_w = 64.0;
    let bar_h = style::VIEWPORT_TOOL_MODE_BAR_H;
    let bar_w = pad * 2.0
        + btn_w * MODES.len() as f32
        + gap * (MODES.len() as f32 - 1.0)
        + sep_gap * 2.0
        + 1.0
        + speed_w;
    let bar = Rect::from_pos_size(
        vp.min_x + (vp.width() - bar_w) * 0.5,
        vp.max_y - PAD - bar_h,
        bar_w,
        bar_h,
    );

    ui.begin_overlay();
    draw_glass_bar(ui, bar, style::RADIUS_LG);
    // Claim the whole bar so brush presses / camera drags don't punch through chrome.
    if ui.pointer_in(bar) {
        ui.state.set_hot(Id::new("vp_tool_mode_bar"));
    }

    let mut x = bar.min_x + pad;
    let caption = FONT_SCALE * TYPE_CAPTION;
    for (i, mode) in MODES.iter().enumerate() {
        let rect = Rect::from_pos_size(x, bar.min_y + 4.0, btn_w, bar_h - 8.0);
        let active = mode.matches(ui_state.editor_tool);
        let id = Id::new("vp_tool_mode").with(i as u64);
        let hovered = ui.pointer_in(rect);
        if hovered {
            ui.state.set_hot(id);
        }
        if hovered && ui.input.primary_pressed {
            ui.state.active = Some(id);
        }
        let clicked = ui.input.primary_released && ui.state.is_active(id) && hovered;

        if active {
            ui.panel_rounded(rect, style::ACCENT, style::RADIUS_SM);
        } else if hovered {
            ui.panel_rounded(rect, Color::rgba(1.0, 1.0, 1.0, 0.08), style::RADIUS_SM);
        }

        let fg = if active || hovered {
            style::TEXT
        } else {
            style::TEXT_MUTED
        };
        let icon_sz = 16.0;
        ui.icon_at(
            rect.center_x() - icon_sz * 0.5,
            rect.min_y + 6.0,
            mode.icon(),
            fg,
            icon_sz,
        );
        ui.label_centered_in_rect(
            Rect::from_pos_size(rect.min_x, rect.min_y + 24.0, rect.width(), 16.0),
            mode.label(),
            fg,
            caption,
        );

        if clicked {
            crate::ui::tools_gui::select_viewport_tool_mode(ui_state, doc, actions, *mode);
            ui_state.camera_speed_menu_open = false;
            ui_state.lighting_menu_open = false;
        }
        x += btn_w + gap;
    }

    x += sep_gap - gap;
    draw_bar_separator(ui, x, bar);
    x += sep_gap + 1.0;

    let speed_rect = Rect::from_pos_size(x, bar.min_y + 10.0, speed_w, bar_h - 20.0);
    let speed_active = ui_state.camera_speed_menu_open;
    if mode_button(
        ui,
        Id::new("vp_camera_speed"),
        speed_rect,
        "Speed",
        speed_active,
        FONT_SCALE * TYPE_LABEL,
        style::RADIUS_SM,
        true,
    ) {
        ui_state.camera_speed_menu_open = !ui_state.camera_speed_menu_open;
        ui_state.lighting_menu_open = false;
    }
    if ui_state.camera_speed_menu_open {
        ui.with_menu_input(|ui| draw_camera_speed_menu(ui, ui_state, speed_rect));
    }
    ui.end_overlay();
}

fn draw_camera_speed_menu(ui: &mut GuiContext<'_>, state: &mut UiState, anchor: Rect) {
    ui.begin_overlay();
    let item_h = 26.0;
    let menu_w = 72.0;
    let menu = Rect::from_pos_size(
        (anchor.center_x() - menu_w * 0.5).max(8.0),
        anchor.min_y - 4.0 - item_h * CAMERA_SPEED_STEPS.len() as f32 - 8.0,
        menu_w,
        item_h * CAMERA_SPEED_STEPS.len() as f32 + 8.0,
    );
    ui.panel_rounded(menu, style::POPUP_BG, style::RADIUS_SM);
    if ui.pointer_in(menu) {
        ui.state.set_hot(Id::new("camera_speed_menu"));
    }
    for (i, &mult) in CAMERA_SPEED_STEPS.iter().enumerate() {
        let row = Rect::from_pos_size(
            menu.min_x + 4.0,
            menu.min_y + 4.0 + i as f32 * item_h,
            menu.width() - 8.0,
            item_h - 2.0,
        );
        let id = Id::new("camera_speed_item").with(i as u64);
        let active = (state.camera_speed - mult).abs() < 0.01;
        let hovered = ui.pointer_in(row);
        if hovered {
            ui.state.set_hot(id);
            ui.panel_rounded(row, style::HOVER_BG, style::RADIUS_SM);
        } else if active {
            ui.panel_rounded(row, style::SELECTED_BG, style::RADIUS_SM);
        }
        let label = format_camera_speed(mult);
        ui.label_centered_in_rect(
            row,
            &label,
            if active { style::TEXT } else { style::TEXT_DIM },
            FONT_SCALE * TYPE_LABEL,
        );
        if hovered && ui.input.primary_released {
            state.camera_speed = mult;
            state.camera_speed_menu_open = false;
        }
    }
    if ui.input.primary_pressed {
        if let Some((px, py)) = ui.input.pointer {
            if !menu.contains(px, py) && !anchor.contains(px, py) {
                state.camera_speed_menu_open = false;
            }
        }
    }
    ui.end_overlay();
}

fn draw_brush_bar(ui: &mut GuiContext<'_>, state: &mut UiState, vp: Rect) {
    let bar_w = (vp.width() - PAD * 2.0).min(560.0).max(280.0);
    let bar = Rect::from_pos_size(
        vp.min_x + (vp.width() - bar_w) * 0.5,
        vp.max_y - PAD - style::VIEWPORT_TOOL_MODE_BAR_H - GAP - style::BRUSH_BAR_H,
        bar_w,
        style::BRUSH_BAR_H,
    );
    ui.begin_overlay();
    ui.panel_rounded(bar, style::VIEWPORT_TOOLBAR_BG, style::RADIUS_MD);
    if ui.pointer_in(bar) {
        ui.state.set_hot(Id::new("vp_brush_bar"));
    }

    // Brush preview circle.
    let preview = Rect::from_pos_size(bar.min_x + 10.0, bar.min_y + 8.0, 32.0, 32.0);
    ui.panel_rounded(preview, style::SURFACE, 16.0);
    let inner_r = 8.0 + state.brush_falloff * 6.0;
    ui.panel_rounded(
        Rect::from_pos_size(
            preview.center_x() - inner_r * 0.5,
            preview.center_y() - inner_r * 0.5,
            inner_r,
            inner_r,
        ),
        style::ACCENT_SOFT,
        inner_r * 0.5,
    );

    let mut x = bar.min_x + 50.0;
    let stop = bar.max_x - 160.0;
    if x + 100.0 <= stop {
        x = compact_slider(
            ui,
            Id::new("brush_radius"),
            x,
            bar,
            "Radius",
            &mut state.sculpt_radius,
            0.005,
            0.25,
        );
    }
    if x + 100.0 <= stop {
        x = compact_slider(
            ui,
            Id::new("brush_strength"),
            x,
            bar,
            "Strength",
            &mut state.sculpt_strength,
            0.1,
            40.0,
        );
    }
    if x + 100.0 <= stop {
        x = compact_slider(
            ui,
            Id::new("brush_falloff"),
            x,
            bar,
            "Falloff",
            &mut state.brush_falloff,
            0.0,
            1.0,
        );
    }
    if x + 100.0 <= stop {
        x = compact_slider(
            ui,
            Id::new("brush_flow"),
            x,
            bar,
            "Flow",
            &mut state.brush_flow,
            0.05,
            1.0,
        );
    }

    let mut rx = bar.max_x - 8.0;
    let toggle_w = 52.0;
    for (label, flag_idx) in [("Sym", 2u64), ("Inv", 1), ("Shape", 0)] {
        rx -= toggle_w + 4.0;
        let rect = Rect::from_pos_size(rx, bar.min_y + 8.0, toggle_w, bar.height() - 16.0);
        let active = match flag_idx {
            0 => false,
            1 => state.invert_brush,
            2 => state.brush_symmetry,
            _ => false,
        };
        if toolbar_button(
            ui,
            Id::new("brush_toggle").with(flag_idx),
            rect,
            label,
            active,
        ) {
            match flag_idx {
                1 => state.invert_brush = !state.invert_brush,
                2 => state.brush_symmetry = !state.brush_symmetry,
                _ => {}
            }
        }
    }
    let _ = x;
    ui.end_overlay();
}

fn compact_slider(
    ui: &mut GuiContext<'_>,
    id: Id,
    x: f32,
    bar: Rect,
    label: &str,
    value: &mut f32,
    min: f32,
    max: f32,
) -> f32 {
    let width = 100.0;
    let rect = Rect::from_pos_size(x, bar.min_y + 4.0, width, bar.height() - 8.0);
    let track = Rect::from_pos_size(rect.min_x, rect.max_y - 8.0, width - 4.0, 4.0);
    let hovered = ui.pointer_in(rect);
    if hovered {
        ui.state.set_hot(id);
    }
    if hovered && ui.input.primary_pressed {
        ui.state.active = Some(id);
    }
    if ui.state.is_active(id) {
        if let Some((pointer_x, _)) = ui.input.pointer {
            *value =
                (min + (pointer_x - track.min_x) / track.width() * (max - min)).clamp(min, max);
        }
    }
    let fill = ((*value - min) / (max - min)).clamp(0.0, 1.0) * track.width();
    ui.label_at(
        rect.min_x,
        rect.min_y,
        &format!("{label} {:.2}", *value),
        style::TEXT_DIM,
        FONT_SCALE * TYPE_LABEL,
    );
    ui.panel_rounded(track, style::TRACK_BG, 2.0);
    ui.panel_rounded(
        Rect::from_pos_size(track.min_x, track.min_y, fill.max(4.0), track.height()),
        style::ACCENT,
        2.0,
    );
    x + width + 8.0
}

fn toolbar_button(ui: &mut GuiContext<'_>, id: Id, rect: Rect, label: &str, active: bool) -> bool {
    let hovered = ui.pointer_in(rect);
    if hovered {
        ui.state.set_hot(id);
    }
    if hovered && ui.input.primary_pressed {
        ui.state.active = Some(id);
    }
    let clicked = ui.input.primary_released && ui.state.is_active(id) && hovered;
    ui.panel_rounded(
        rect,
        if active {
            style::ACCENT
        } else if hovered {
            style::BUTTON_HOVER
        } else {
            style::BUTTON_BG
        },
        style::RADIUS_SM,
    );
    ui.label_centered_in_rect(rect, label, style::TEXT, FONT_SCALE * TYPE_LABEL);
    clicked
}

#[cfg(test)]
mod tests {
    use super::*;
    use terra_gui::{GuiContext, GuiInput, GuiState};

    /// Drive one frame of the render panel exactly as the mode bar does:
    /// inside an overlay layer, through `with_menu_input`.
    fn render_menu_frame(gs: &mut GuiState, ui_state: &mut UiState, input: GuiInput, anchor: Rect) {
        let mut ctx = GuiContext::begin(1200.0, 800.0, 1.0, input, gs);
        ctx.begin_overlay();
        ctx.with_menu_input(|ctx| draw_viewport_render_menu(ctx, ui_state, anchor));
        ctx.end_overlay();
        ctx.end();
    }

    /// Regression guard: the advanced render panel's scroll offset must survive
    /// across frames. A frame-local `let mut scroll = 0.0` would snap it back to
    /// the top every frame, so the wheel (and thumb drag) would never move.
    #[test]
    fn render_panel_scroll_persists_across_frames() {
        let mut gs = GuiState::default();
        let mut ui_state = UiState::default();
        ui_state.viewport_render.menu_open = true;
        ui_state.viewport_render.advanced_open = true;

        // Anchor (the Render button): max_x = 900, max_y = 60 → window at (600, 64) 300x520.
        let anchor = Rect::from_pos_size(836.0, 40.0, 64.0, 20.0);
        let content_pt = (750.0, 338.0); // inside the panel's scrollable body

        // Frame 1: settle so the overflow (scroll_max) is measured for the wheel gate.
        render_menu_frame(
            &mut gs,
            &mut ui_state,
            GuiInput {
                pointer: Some(content_pt),
                ..Default::default()
            },
            anchor,
        );

        // Frame 2: wheel down over the panel body.
        render_menu_frame(
            &mut gs,
            &mut ui_state,
            GuiInput {
                pointer: Some(content_pt),
                scroll_delta: -3.0,
                ..Default::default()
            },
            anchor,
        );

        assert!(
            ui_state.viewport_render.menu_scroll > 0.0,
            "advanced render panel must retain scroll offset across frames, got {}",
            ui_state.viewport_render.menu_scroll
        );
    }
}
