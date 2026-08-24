//! Thin bottom status bar with mesh stats, backend, and processing feedback.

use crate::ui::style::{self, FONT_SCALE, PAD, STATUS_STRIP_H, TYPE_LABEL};
use crate::ui::{TerrainPipelineStatus, TerrainPreviewFreshness, UiState};
use terra_core::document::TerrainDocument;
use terra_core::quality::PreviewQuality;
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
    } else if let Some(bounded) = doc.bounded_settings() {
        format!(
            "{}x{}",
            bounded.preview_resolution, bounded.preview_resolution
        )
    } else {
        "Sparse tiles pending".into()
    };
    let quality = quality_name(ui_state.quality);
    let backend = if ui_state.profile.path.is_empty() {
        "CPU"
    } else {
        ui_state.profile.path
    };
    let build_ms = ui_state.profile.eval_us as f32 / 1000.0;

    // Right: processing / failure / idle status + action (layout first for truncation).
    let (status_text, show_progress, progress, show_retry) =
        if let TerrainPipelineStatus::Pending { label } = &ui_state.terrain_pipeline_status {
            (
                format!("Compiling presentation feature - {label}"),
                true,
                0.5,
                false,
            )
        } else if let TerrainPipelineStatus::Failed { message } = &ui_state.terrain_pipeline_status
        {
            (
                format!("Presentation compile failed - {message}"),
                false,
                0.0,
                true,
            )
        } else if let Some(progress) = ui_state.export_progress {
            (
                format!("Exporting height pyramid {:.0}%", progress * 100.0),
                true,
                progress.clamp(0.0, 1.0),
                false,
            )
        } else if let Some(failure) = ui_state.evaluation_failure.as_ref() {
            let layer = failure.layer_name.as_deref().unwrap_or("Terrain");
            let recovery = if failure.worker_restarted {
                "worker restarted; last good preview shown"
            } else {
                "last good preview shown"
            };
            (
                format!(
                    "{layer} failed at {} - {recovery}",
                    quality_name(failure.quality)
                ),
                false,
                0.0,
                true,
            )
        } else if let TerrainPreviewFreshness::Deferred {
            layer_name,
            deferred_layers,
            settling,
        } = &ui_state.terrain_preview_freshness
        {
            let later = deferred_layers.saturating_sub(1);
            let pending = if later == 0 {
                format!("{layer_name} pending")
            } else if later == 1 {
                format!("{layer_name} + 1 later layer pending")
            } else {
                format!("{layer_name} + {later} later layers pending")
            };
            let phase = if *settling { "settling" } else { "editing" };
            (
                format!("Local preview current - {pending} ({phase})"),
                false,
                0.0,
                false,
            )
        } else if let TerrainPreviewFreshness::RefiningSuffix {
            layer_name,
            quality,
        } = &ui_state.terrain_preview_freshness
        {
            let pct = ui_state.build_progress.unwrap_or(0.0).clamp(0.0, 1.0);
            (
                format!("Refining {layer_name} suffix - {}", quality_name(*quality)),
                true,
                pct,
                false,
            )
        } else if matches!(
            ui_state.terrain_preview_freshness,
            TerrainPreviewFreshness::LastCompleteStale
        ) {
            (
                "Edit queued - showing last complete preview".into(),
                false,
                0.0,
                false,
            )
        } else if ui_state.refining {
            let pct = ui_state.build_progress.unwrap_or(0.0).clamp(0.0, 1.0);
            let name = ui_state.refining_layer_name.as_deref().unwrap_or("Terrain");
            (format!("{name} {:.0}%", pct * 100.0), true, pct, false)
        } else if ui_state.draft_displayed {
            (
                "Interactive preview active - full refinement pending".into(),
                false,
                0.0,
                false,
            )
        } else {
            (format!("Preview ready - {quality}"), false, 0.0, false)
        };

    let action_w = if show_progress || show_retry {
        64.0
    } else {
        0.0
    };
    let bar_w = if show_progress { 100.0 } else { 0.0 };
    let status_w = DrawList::text_width(&status_text, FONT_SCALE * TYPE_LABEL);
    let mut rx = dock.max_x - PAD;

    if show_progress || show_retry {
        let chip = Rect::from_pos_size(rx - action_w, dock.min_y + 6.0, action_w, 24.0);
        let cid = Id::new(if show_retry {
            "dock_retry_chip"
        } else {
            "dock_cancel_chip"
        });
        let hovered = ui.pointer_in(chip);
        if hovered {
            ui.state.set_hot(cid);
        }
        if hovered && ui.input.primary_pressed {
            ui.state.active = Some(cid);
        }
        if ui.input.primary_released && ui.state.is_active(cid) && hovered {
            if show_retry {
                if matches!(
                    ui_state.terrain_pipeline_status,
                    TerrainPipelineStatus::Failed { .. }
                ) {
                    out.request_retry_terrain_pipeline = true;
                } else {
                    out.request_retry_evaluation = true;
                }
            } else {
                out.request_cancel_build = true;
            }
        }
        ui.panel_rounded(chip, style::BUTTON_BG, style::RADIUS_SM);
        ui.label_centered_in_rect(
            chip,
            if show_retry { "Retry" } else { "Cancel" },
            style::TEXT,
            FONT_SCALE * TYPE_LABEL,
        );
        rx -= action_w + 8.0;

        if show_progress {
            let bar = Rect::from_pos_size(rx - bar_w, dock.min_y + 14.0, bar_w, 6.0);
            ui.panel_rounded(bar, style::TRACK_BG, 3.0);
            ui.panel_rounded(
                Rect::from_pos_size(bar.min_x, bar.min_y, (bar_w * progress).max(4.0), 6.0),
                style::ACCENT,
                3.0,
            );
            rx -= bar_w + 10.0;
        }
    }

    let status_x = (rx - status_w).max(dock.min_x + PAD);
    ui.label_at(
        status_x,
        y,
        &status_text,
        if show_progress {
            style::TEXT
        } else if show_retry {
            style::ERROR
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
