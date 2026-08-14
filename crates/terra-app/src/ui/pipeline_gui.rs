//! Terrain Recipe view — linear execution-order visualization of the terrain stack.
//!
//! Not a node graph. Rows come from [`terra_core::terrain_recipe::build_terrain_recipe_from_stack`]
//! so the view stays synchronized with the document stack.
//! Selecting a row selects the layer; drag-and-drop reorders within the stack.

use crate::ui::actions::PanelAction;
use crate::ui::panels::viewport_float_rect;
use crate::ui::style::{self, FONT_SCALE, PAD, ROW_H};
use crate::ui::{LayerDragSource, UiState};
use terra_core::document::TerrainDocument;
use terra_core::terrain_recipe::{
    build_terrain_recipe_from_stack, RecipeItem, RecipeItemKind, RecipeRebuildStatus,
};
use terra_gui::{button_id, label, DrawList, GuiContext, Icon, Id, Rect};

const INDENT: f32 = 14.0;

#[derive(Debug, Clone, Default)]
pub struct RecipeViewState {
    pub scroll_y: f32,
    pub drag_from: Option<LayerDragSource>,
}

/// Draw the Terrain Recipe floating window (View → Terrain Recipe).
pub fn draw_recipe_view(
    ui: &mut GuiContext<'_>,
    doc: &TerrainDocument,
    ui_state: &mut UiState,
    state: &mut RecipeViewState,
    actions: &mut Vec<PanelAction>,
) {
    if !ui_state.show_pipeline {
        return;
    }

    let rect = viewport_float_rect(ui, 340.0, 480.0, 0.0);
    if !ui.begin_window(
        Id::new("win_terrain_recipe"),
        "Terrain Recipe",
        rect,
        &mut ui_state.show_pipeline,
        &mut state.scroll_y,
    ) {
        return;
    }

    label(ui, "Execution order · not a node graph");
    ui.separator();

    let world_building = ui_state.build_progress.is_some();
    if let Some(p) = ui_state.build_progress {
        label(ui, &format!("Rebuild status: Building… {:.0}%", p * 100.0));
    } else if ui_state.draft_displayed {
        label(ui, "Rebuild status: Draft displayed");
    } else {
        label(ui, "Rebuild status: Ready");
    }
    if button_id(ui, Id::new("recipe_rebuild_all"), "Rebuild World") {
        actions.push(PanelAction::RebuildWorld);
    }
    ui.separator();

    let recipe = build_terrain_recipe_from_stack(&doc.stack);
    let mut drop_target: Option<usize> = None;

    for (i, item) in recipe.iter().enumerate() {
        let row_id = Id::new("recipe_row").with(i as u64);
        match item.kind {
            RecipeItemKind::FlowArrow => {
                draw_flow_arrow(ui);
                continue;
            }
            RecipeItemKind::GlobalSection => {
                draw_section_header(ui, &item.name);
                continue;
            }
            _ => {}
        }

        let selected = is_selected(doc, item);
        let status = item.rebuild_status(world_building);
        let (clicked, drag_hit, drop_hit) =
            draw_recipe_row(ui, row_id, item, selected, status, state);

        if clicked {
            push_select(actions, item);
        }
        if let Some(src) = drag_hit {
            state.drag_from = Some(src);
        }
        if let Some(dt) = drop_hit {
            drop_target = Some(dt);
        }
    }

    if ui.input.primary_released {
        if let (Some(from), Some(insert_index)) = (state.drag_from.take(), drop_target) {
            if from.root_idx != insert_index {
                actions.push(PanelAction::Reorder {
                    from: from.root_idx,
                    to: insert_index,
                });
            }
        }
    }

    let only_chrome = recipe.iter().all(|r| {
        matches!(
            r.kind,
            RecipeItemKind::GlobalSection | RecipeItemKind::FlowArrow
        )
    });
    if only_chrome {
        label(ui, "Empty recipe — add layers to the terrain stack.");
    }

    ui.end_window(&mut state.scroll_y);
}

/// Legacy entry point name used by `draw_editor_gui`.
pub fn draw_pipeline_gui(
    ui: &mut GuiContext<'_>,
    doc: &TerrainDocument,
    ui_state: &mut UiState,
    state: &mut RecipeViewState,
    actions: &mut Vec<PanelAction>,
) {
    draw_recipe_view(ui, doc, ui_state, state, actions);
}

fn is_selected(doc: &TerrainDocument, item: &RecipeItem) -> bool {
    matches!(item.kind, RecipeItemKind::Layer | RecipeItemKind::Group)
        && item.layer_id.is_some()
        && doc.selected == item.layer_id
}

fn push_select(actions: &mut Vec<PanelAction>, item: &RecipeItem) {
    if matches!(item.kind, RecipeItemKind::Layer | RecipeItemKind::Group) {
        if let Some(id) = item.layer_id {
            actions.push(PanelAction::Select(id));
        }
    }
}

fn draw_flow_arrow(ui: &mut GuiContext<'_>) {
    let rect = ui.allocate(ROW_H * 0.65);
    let cx = rect.min_x + rect.width() * 0.5 - 4.0;
    ui.label_at(
        cx,
        label_y(rect.min_y, rect.height(), FONT_SCALE),
        "↓",
        style::TEXT_MUTED,
        FONT_SCALE * 1.1,
    );
}

fn draw_section_header(ui: &mut GuiContext<'_>, name: &str) {
    let rect = ui.allocate(ROW_H);
    ui.panel_rounded(rect, style::SURFACE, style::RADIUS_SM);
    ui.label_at(
        rect.min_x + PAD,
        label_y(rect.min_y, ROW_H, FONT_SCALE * 0.85),
        name,
        style::ACCENT,
        FONT_SCALE * 0.85,
    );
}

fn draw_recipe_row(
    ui: &mut GuiContext<'_>,
    id: Id,
    item: &RecipeItem,
    selected: bool,
    status: RecipeRebuildStatus,
    state: &RecipeViewState,
) -> (bool, Option<LayerDragSource>, Option<usize>) {
    let _ = id;
    let rect = ui.allocate(ROW_H);

    let hovered = ui.pointer_in(rect);
    if selected {
        ui.panel_rounded(rect, style::SELECTED_BG, style::RADIUS_SM);
    } else if hovered {
        ui.panel_rounded(rect, style::HOVER_BG, style::RADIUS_SM);
    }

    let indent = PAD + item.depth as f32 * INDENT;
    let mut x = rect.min_x + indent;

    let icon = match item.kind {
        RecipeItemKind::Group => Icon::Folder,
        _ => Icon::Layers,
    };
    ui.icon_at(
        x,
        rect.min_y + (ROW_H - 14.0) * 0.5,
        icon,
        style::TEXT_MUTED,
        14.0,
    );
    x += 18.0;

    let status_tag = match status {
        RecipeRebuildStatus::Building => " · …",
        RecipeRebuildStatus::Ready => " · ok",
        RecipeRebuildStatus::Pending => " · ◯",
        RecipeRebuildStatus::Disabled => " · off",
        RecipeRebuildStatus::Idle => "",
    };
    let mut label_text = item.name.clone();
    if let Some(p) = item.priority {
        label_text = format!("{label_text}  P{p}");
    }
    if item.solo {
        label_text.push_str("  solo");
    }
    label_text.push_str(status_tag);

    let name_max = (rect.max_x - 4.0 - x).max(24.0);
    let clipped = DrawList::truncate_to_width(&label_text, FONT_SCALE * 0.9, name_max);
    let color = if item.enabled {
        style::TEXT
    } else {
        style::TEXT_DISABLED
    };
    ui.label_at(
        x,
        label_y(rect.min_y, ROW_H, FONT_SCALE * 0.9),
        &clipped,
        color,
        FONT_SCALE * 0.9,
    );

    let clicked = hovered && ui.input.primary_pressed;

    let mut drag_hit = None;
    if hovered
        && ui.input.primary_pressed
        && matches!(item.kind, RecipeItemKind::Layer | RecipeItemKind::Group)
    {
        if let (Some(lid), Some(root_idx)) = (item.layer_id, item.root_idx) {
            drag_hit = Some(LayerDragSource {
                id: lid,
                root_idx,
                name: item.name.clone(),
                icon,
            });
        }
    }

    let mut drop_hit = None;
    if state.drag_from.is_some() && hovered {
        let insert_index = match item.kind {
            RecipeItemKind::Layer | RecipeItemKind::Group => {
                item.root_idx.map(|i| i.saturating_add(1)).unwrap_or(0)
            }
            RecipeItemKind::GlobalSection => 0,
            _ => 0,
        };
        drop_hit = Some(insert_index);
        ui.panel_rounded(
            Rect::from_pos_size(
                rect.min_x + indent,
                rect.max_y - 2.0,
                (rect.width() - indent).max(8.0),
                2.0,
            ),
            style::ACCENT,
            0.0,
        );
    }

    (clicked, drag_hit, drop_hit)
}

fn label_y(min_y: f32, h: f32, scale: f32) -> f32 {
    min_y + (h - 14.0 * scale) * 0.5
}
