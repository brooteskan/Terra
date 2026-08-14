//! Thin bottom status bar with mesh stats, backend, and processing feedback.

use crate::ui::style::{self, FONT_SCALE, PAD, STATUS_STRIP_H, TYPE_LABEL};
use crate::ui::UiState;
use terra_core::document::TerrainDocument;
use terra_core::eval::PreviewQuality;
use terra_gui::{DrawList, GuiContext, Id, Rect};

#[derive(Debug, Default)]
pub struct DockGuiState;

pub fn draw_bottom_dock(
    ui: &mut GuiContext<'_>,
    doc: &TerrainDocument,
    ui_state: &mut UiState,
    _state: &mut DockGuiState,
    out: &mut crate::ui::FrameUiOutput,
) {
    let dock = ui.bottom_dock_rect();
    if ui
        .input
        .pointer
        .map(|(x, y)| dock.contains(x, y))
        .unwrap_or(false)
    {
        ui.state.set_hot(Id::new("__dock_bg"));
    }

    ui.panel(dock, style::DOCK_BG);
    ui.panel(
        Rect::from_pos_size(dock.min_x, dock.min_y, dock.width(), 1.0),
        style::SEPARATOR,
    );

    let dock_id = Id::new("__dock_status");
    if ui.pointer_in(dock) {
        ui.state.set_hot(dock_id);
        if ui.input.primary_pressed {
            ui.state.active = Some(dock_id);
        }
        if ui.input.primary_released && ui.state.is_active(dock_id) {
            ui_state.show_profiler = true;
        }
    }

    let y = dock.min_y + (STATUS_STRIP_H - 14.0) * 0.5;
    let verts = if ui_state.profile.tex_w > 0 && ui_state.profile.tex_h > 0 {
        ui_state
            .profile
            .tex_w
            .saturating_mul(ui_state.profile.tex_h) as f32
            / 1_000_000.0
    } else {
        0.0
    };
    let res = if ui_state.profile.tex_w > 0 {
        format!("{}x{}", ui_state.profile.tex_w, ui_state.profile.tex_h)
    } else {
        format!("{}x{}", doc.preview_resolution, doc.preview_resolution)
    };
    let quality = quality_name(ui_state.quality);
    let backend = if ui_state.profile.path.is_empty() {
        "CPU"
    } else {
        ui_state.profile.path
    };
    let build_ms = ui_state.profile.eval_us as f32 / 1000.0;

    // Right: processing / idle status + cancel (layout first for left truncation).
    let (status_text, show_progress, progress) = if ui_state.refining {
        let pct = ui_state.build_progress.unwrap_or(0.0).clamp(0.0, 1.0);
        let name = ui_state.refining_layer_name.as_deref().unwrap_or("Terrain");
        (format!("{name} {:.0}%", pct * 100.0), true, pct)
    } else if ui_state.draft_displayed {
        (
            "Interactive preview active - full refinement pending".into(),
            false,
            0.0,
        )
    } else {
        (format!("Preview ready - {quality}"), false, 0.0)
    };

    let cancel_w = if show_progress { 64.0 } else { 0.0 };
    let bar_w = if show_progress { 100.0 } else { 0.0 };
    let status_w = DrawList::text_width(&status_text, FONT_SCALE * TYPE_LABEL);
    let mut rx = dock.max_x - PAD;

    if show_progress {
        let chip = Rect::from_pos_size(rx - cancel_w, dock.min_y + 6.0, cancel_w, 24.0);
        let cid = Id::new("dock_cancel_chip");
        let hovered = ui.pointer_in(chip);
        if hovered {
            ui.state.set_hot(cid);
        }
        if hovered && ui.input.primary_pressed {
            ui.state.active = Some(cid);
        }
        if ui.input.primary_released && ui.state.is_active(cid) && hovered {
            out.request_cancel_build = true;
        }
        ui.panel_rounded(chip, style::BUTTON_BG, style::RADIUS_SM);
        ui.label_centered_in_rect(chip, "Cancel", style::TEXT, FONT_SCALE * TYPE_LABEL);
        rx -= cancel_w + 8.0;

        let bar = Rect::from_pos_size(rx - bar_w, dock.min_y + 14.0, bar_w, 6.0);
        ui.panel_rounded(bar, style::TRACK_BG, 3.0);
        ui.panel_rounded(
            Rect::from_pos_size(bar.min_x, bar.min_y, (bar_w * progress).max(4.0), 6.0),
            style::ACCENT,
            3.0,
        );
        rx -= bar_w + 10.0;
    }

    let status_x = (rx - status_w).max(dock.min_x + PAD);
    ui.label_at(
        status_x,
        y,
        &status_text,
        if show_progress {
            style::TEXT
        } else {
            style::TEXT_MUTED
        },
        FONT_SCALE * TYPE_LABEL,
    );

    let left_max_w = (status_x - 16.0 - (dock.min_x + PAD)).max(40.0);
    // "Samples" = heightfield pixel grid (not WC Terrain "Resolution", which is world size in m).
    let left = format!(
        "Vertices {:.2}M   Samples {res}   Preview Quality: {quality}   {backend}   Build Time: {build_ms:.0} ms",
        verts
    );
    let left = DrawList::truncate_to_width(&left, FONT_SCALE * TYPE_LABEL, left_max_w);
    ui.label_at(
        dock.min_x + PAD,
        y,
        &left,
        style::TEXT_DIM,
        FONT_SCALE * TYPE_LABEL,
    );
}

fn quality_name(quality: PreviewQuality) -> &'static str {
    match quality {
        PreviewQuality::Draft => "Draft",
        PreviewQuality::Medium => "Medium",
        PreviewQuality::Full => "Full",
        PreviewQuality::Export => "Export",
    }
}
