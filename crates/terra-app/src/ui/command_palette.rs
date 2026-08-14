//! Global command palette and its editor-facing action bridge.

use crate::ui::actions::PanelAction;
use crate::ui::command_registry::{
    commands, format_shortcuts, fuzzy_match, resolve_workspace_command, CommandId,
};
use crate::ui::style::{self, FONT_SCALE, PAD, ROW_H};
use crate::ui::tool_catalog::{all_tools_cached, instantiate_layer_preset, ToolAction};
use crate::ui::{AppWorkspace, Preview2dMode, UiState};
use terra_gui::{button_id, Color, GuiContext, Id, Rect};

#[derive(Debug, Default)]
pub struct CommandPaletteState {
    pub query: String,
    pub scroll_y: f32,
}

#[derive(Debug)]
pub enum PaletteAction {
    Panel(PanelAction),
    Undo,
    Redo,
    Export,
    Save,
    SaveAs,
    NewProject,
    OpenProject,
    CloseProject,
    CameraReset,
    CameraTopView,
    CameraFrameSelection,
}

pub fn draw_command_palette(
    ui: &mut GuiContext<'_>,
    doc: &terra_core::document::TerrainDocument,
    ui_state: &mut UiState,
    state: &mut CommandPaletteState,
) -> Vec<PaletteAction> {
    if !ui_state.show_command_palette {
        return Vec::new();
    }
    if ui.input.backspace_pressed {
        state.query.pop();
    }
    state.query.push_str(&ui.input.text);
    if ui.input.escape_pressed {
        ui_state.show_command_palette = false;
        state.query.clear();
        return Vec::new();
    }

    let popup = Rect::from_pos_size(
        (ui.screen_w - 540.0).max(16.0) * 0.5,
        72.0,
        540.0_f32.min(ui.screen_w - 32.0),
        520.0_f32.min(ui.screen_h - 90.0),
    );
    let mut actions = Vec::new();
    ui.begin_overlay();
    ui.panel_rounded(popup, style::POPUP_BG, style::RADIUS_SM);
    ui.label_at(
        popup.min_x + PAD,
        popup.min_y + PAD,
        "Command Palette",
        style::TEXT,
        FONT_SCALE * 1.15,
    );
    let query_label = if state.query.is_empty() {
        "Search commands..."
    } else {
        &state.query
    };
    let search = Rect::from_pos_size(
        popup.min_x + PAD,
        popup.min_y + 38.0,
        popup.width() - PAD * 2.0,
        ROW_H,
    );
    ui.panel_rounded(search, style::SURFACE, style::RADIUS_SM);
    ui.label_at(
        search.min_x + 10.0,
        search.min_y + 5.0,
        query_label,
        style::TEXT_DIM,
        FONT_SCALE,
    );

    let list = Rect::from_min_max(
        popup.min_x + PAD,
        search.max_y + 8.0,
        popup.max_x - PAD,
        popup.max_y - PAD,
    );
    ui.begin_panel_scrolled(
        Id::new("command_palette_list"),
        list,
        Color::rgba(0.0, 0.0, 0.0, 0.0),
        &mut state.scroll_y,
    );
    for command in commands()
        .into_iter()
        .filter(|command| fuzzy_match(&state.query, command.name))
    {
        let shortcut = format_shortcuts(command.bindings);
        let label = format!(
            "{}  {}    {}",
            command.category.label(),
            command.name,
            shortcut
        );
        if button_id(ui, Id::new("command_palette").child(command.id), &label) {
            execute_command(command.id, doc, ui_state, &mut actions);
            ui_state.show_command_palette = false;
            state.query.clear();
            break;
        }
    }
    ui.end_panel_scrolled(&mut state.scroll_y);
    ui.end_overlay();

    if ui.input.primary_pressed && !ui.pointer_in(popup) {
        ui_state.show_command_palette = false;
        state.query.clear();
    }
    actions
}

fn execute_command(
    id: &str,
    doc: &terra_core::document::TerrainDocument,
    ui_state: &mut UiState,
    actions: &mut Vec<PaletteAction>,
) {
    if let Some(workspace) = resolve_workspace_command(id) {
        ui_state.switch_workspace(workspace);
        return;
    }

    match id {
        CommandId::ADD_MOUNTAIN => add_catalog_layer("shape.procedural", actions),
        CommandId::ADD_HYDRAULIC_EROSION => add_catalog_layer("sim.hydraulic", actions),
        CommandId::FRAME_TERRAIN => actions.push(PaletteAction::CameraReset),
        CommandId::TOP_VIEW => actions.push(PaletteAction::CameraTopView),
        CommandId::FRAME_SELECTION => actions.push(PaletteAction::CameraFrameSelection),
        CommandId::TOGGLE_HEIGHT_VIEW => {
            ui_state.preview_mode = if ui_state.preview_mode == Preview2dMode::Height {
                Preview2dMode::Lit
            } else {
                Preview2dMode::Height
            };
        }
        CommandId::TOGGLE_WIREFRAME => {
            ui_state.viewport_overlays.wireframe = !ui_state.viewport_overlays.wireframe
        }
        CommandId::OPEN_QUICK_ADD => {
            ui_state.app_workspace = AppWorkspace::Advanced;
            ui_state.show_quick_add = true;
        }
        CommandId::OPEN_COMMAND_PALETTE => ui_state.show_command_palette = true,
        CommandId::UNDO => actions.push(PaletteAction::Undo),
        CommandId::REDO => actions.push(PaletteAction::Redo),
        CommandId::EXPORT => actions.push(PaletteAction::Export),
        CommandId::SAVE => actions.push(PaletteAction::Save),
        CommandId::SAVE_AS => actions.push(PaletteAction::SaveAs),
        CommandId::NEW_PROJECT => actions.push(PaletteAction::NewProject),
        CommandId::OPEN_PROJECT => actions.push(PaletteAction::OpenProject),
        CommandId::CLOSE_PROJECT => actions.push(PaletteAction::CloseProject),
        CommandId::BAKE_SELECTED => {
            if let Some(layer_id) = doc.selected {
                actions.push(PaletteAction::Panel(PanelAction::SetCached {
                    id: layer_id,
                    cached: true,
                }));
                ui_state.status = "Baked selected layer.".into();
            } else {
                ui_state.status = "Select a layer to bake.".into();
            }
        }
        CommandId::OPEN_PROFILER => ui_state.show_profiler = true,
        CommandId::TOGGLE_HISTORY => ui_state.show_history = !ui_state.show_history,
        CommandId::TOGGLE_PIPELINE => ui_state.show_pipeline = !ui_state.show_pipeline,
        CommandId::CLEAR_GEOMORPH_DEBUG => {
            ui_state.geomorph_debug_field = None;
            ui_state.force_preview_refresh = true;
            ui_state.status = "Cleared geomorph debug overlay".into();
        }
        other if other.starts_with("debug.geomorph.") => {
            if let Some(field) = terra_core::GeomorphDebugField::ALL
                .iter()
                .copied()
                .find(|f| f.command_id() == other)
            {
                ui_state.geomorph_debug_field = Some(field);
                ui_state.show_2d_preview = true;
                ui_state.preview_mode = Preview2dMode::Height;
                ui_state.force_preview_refresh = true;
                ui_state.status = format!("Geomorph debug: {}", field.label());
            }
        }
        _ => {}
    }
}

fn add_catalog_layer(id: &str, actions: &mut Vec<PaletteAction>) {
    if let Some(tool) = all_tools_cached()
        .iter()
        .find(|tool| tool.id == id)
        .cloned()
    {
        match tool.action {
            ToolAction::AddLayer { name, kind } => {
                let layer = instantiate_layer_preset(name, &kind);
                if id.starts_with("shape.") {
                    actions.push(PaletteAction::Panel(PanelAction::AddLayerToCategory {
                        category: terra_core::layer::StackCategory::Shape,
                        layer,
                    }));
                } else {
                    actions.push(PaletteAction::Panel(PanelAction::AddLayer(layer)));
                }
            }
            ToolAction::CreateBiome { name } => {
                actions.push(PaletteAction::Panel(PanelAction::AddBiome {
                    name: name.into(),
                }));
            }
            ToolAction::AddMask { name, source } => {
                let mask_id = terra_core::mask::MaskId::new();
                let mut asset = terra_core::mask::MaskAsset {
                    id: mask_id,
                    name: name.into(),
                    source,
                    ops: Vec::new(),
                    paint: None,
                    display_color: terra_core::mask::display_color_for_mask_id(mask_id),
                };
                asset.prepare_for_document();
                actions.push(PaletteAction::Panel(PanelAction::AddMask(asset)));
            }
            _ => {}
        }
    }
}
