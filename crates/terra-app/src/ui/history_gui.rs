//! Read-only artist history window with Undo and Redo controls.

use crate::ui::panels::viewport_float_rect;
use crate::ui::style::{self, FONT_SCALE, TYPE_CAPTION, TYPE_LABEL};
use crate::ui::{FrameUiOutput, UiState};
use terra_gui::{button, label_dim, GuiContext, Id};

pub fn draw_history_gui(
    ui: &mut GuiContext<'_>,
    ui_state: &mut UiState,
    scroll_y: &mut f32,
    out: &mut FrameUiOutput,
) {
    if !ui_state.show_history {
        return;
    }

    let rect = viewport_float_rect(ui, 300.0, 360.0, 0.35);
    if ui.begin_window(
        Id::new("win_history"),
        "History",
        rect,
        &mut ui_state.show_history,
        scroll_y,
    ) {
        if button(ui, "Undo") {
            out.request_undo = true;
        }
        if button(ui, "Redo") {
            out.request_redo = true;
        }
        ui.separator();
        if ui_state.history_descriptions.is_empty() {
            label_dim(ui, "No recorded edits yet.");
        } else {
            label_dim(ui, "Newest first");
            for (i, description) in ui_state.history_descriptions.iter().rev().enumerate() {
                let row = ui.allocate(26.0);
                if i == 0 {
                    ui.panel_rounded(row, style::SELECTED_BG, style::RADIUS_SM);
                }
                ui.label_at(
                    row.min_x + 8.0,
                    row.min_y + 6.0,
                    description,
                    if i == 0 { style::TEXT } else { style::TEXT_DIM },
                    FONT_SCALE * if i == 0 { TYPE_LABEL } else { TYPE_CAPTION },
                );
            }
        }
        ui.end_window(scroll_y);
    }
}
