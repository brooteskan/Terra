//! TOOLS rail + icon tool shelf for the active workspace.
//!
//! Management chrome (World Rules lists, scenario Run buttons, Develop Add* rows,
//! biome palette, shape-history mode toggles) does **not** live here â€” those belong
//! in the Layers hierarchy / Inspector. Tools = pick / drag layer kinds and brushes.

use crate::ui::actions::PanelAction;
use crate::ui::style::{
    self, FONT_SCALE, MODE_ROW_H, PAD, TOOL_CARD_GAP, TOOL_CARD_H, TOOL_THUMB_SIZE, TYPE_CAPTION,
};
use crate::ui::tool_catalog::{
    instantiate_layer_preset, tools_for_workspace, ToolAction, ToolDef, ToolGroup,
};
use crate::ui::workspace::{workspace_definition, WorkspaceId};
use crate::ui::{EditorTool, UiState};
use terra_core::document::TerrainDocument;
use terra_core::mask::{MaskAsset, MaskId};
use terra_gui::{Color, DrawList, GuiContext, Icon, Id, Rect};

#[derive(Debug, Default)]
pub struct ToolsGuiState {
    pub scroll_y: f32,
    /// Tool categories listed here are collapsed (header visible, cards hidden).
    pub collapsed_groups: Vec<ToolGroup>,
}

impl ToolsGuiState {
    fn is_group_collapsed(&self, group: ToolGroup) -> bool {
        self.collapsed_groups.contains(&group)
    }

    fn toggle_group_collapsed(&mut self, group: ToolGroup) {
        if let Some(pos) = self.collapsed_groups.iter().position(|g| *g == group) {
            self.collapsed_groups.remove(pos);
        } else {
            self.collapsed_groups.push(group);
        }
    }

    /// Collapse every tool category (shape + filter shelves).
    pub fn collapse_all_categories(&mut self) {
        const ALL: &[ToolGroup] = &[
            ToolGroup::Landforms,
            ToolGroup::Hydrology,
            ToolGroup::Noise,
            ToolGroup::Simulation,
            ToolGroup::Masks,
            ToolGroup::Brushes,
            ToolGroup::Other,
            ToolGroup::FilterGeneral,
            ToolGroup::FilterDesign,
            ToolGroup::FilterEffect,
            ToolGroup::FilterNoise,
            ToolGroup::FilterArid,
            ToolGroup::FilterTerrace,
            ToolGroup::FilterDrift,
            ToolGroup::FilterBasicErosion,
            ToolGroup::FilterAdvancedErosion,
            ToolGroup::FilterSediment,
        ];
        self.collapsed_groups.clear();
        self.collapsed_groups.extend_from_slice(ALL);
    }
}

/// In-flight drag of a layer tool toward the viewport.
#[derive(Debug, Clone)]
pub struct LayerToolDrag {
    pub name: &'static str,
    pub kind: terra_core::layer::LayerKind,
    /// Terrain shape-rail tools must land in the Shape folder even when the
    /// kind is dual-role (e.g. ImportHeightmap as WC Design Height Map filter).
    pub force_shape_folder: bool,
}

pub fn draw_tools_gui(
    ui: &mut GuiContext<'_>,
    doc: &TerrainDocument,
    ui_state: &mut UiState,
    state: &mut ToolsGuiState,
) -> Vec<PanelAction> {
    let mut actions = Vec::new();
    if ui_state.app_workspace.shows_mode_rail() {
        draw_mode_rail(ui, ui_state);
    }
    draw_tool_panel(ui, doc, ui_state, state, &mut actions);
    finish_tool_drag(ui, ui_state, &mut actions);
    actions
}

fn draw_mode_rail(ui: &mut GuiContext<'_>, ui_state: &mut UiState) {
    let rail = ui.mode_rail_rect();
    if ui
        .input
        .pointer
        .map(|(x, y)| rail.contains(x, y))
        .unwrap_or(false)
    {
        ui.state.set_hot(Id::new("__mode_rail_bg"));
    }

    ui.panel(rail, style::MODE_RAIL_BG);
    ui.panel(
        Rect::from_pos_size(rail.max_x - 1.0, rail.min_y, 1.0, rail.height()),
        style::SEPARATOR,
    );

    // Section label Ã¢â‚¬â€ task focus, not a numbered progression.
    let header_tw = DrawList::text_width("TOOLS", FONT_SCALE * TYPE_CAPTION);
    ui.label_at(
        rail.min_x + (rail.width() - header_tw) * 0.5,
        rail.min_y + 10.0,
        "TOOLS",
        style::TEXT_MUTED,
        FONT_SCALE * TYPE_CAPTION,
    );

    let mut y = rail.min_y + 28.0;
    for id in WorkspaceId::ALL {
        let def = workspace_definition(id);
        let row = Rect::from_pos_size(rail.min_x + 4.0, y, rail.width() - 8.0, MODE_ROW_H);
        let row_id = Id::new("workspace").child(def.name);
        let selected = ui_state.active_workspace == id;
        let hovered = ui.pointer_in(row);
        if hovered {
            ui.state.set_hot(row_id);
        }
        if hovered && ui.input.primary_pressed {
            ui.state.active = Some(row_id);
        }
        if ui.input.primary_released && ui.state.is_active(row_id) && hovered {
            ui_state.switch_workspace(id);
        }

        if selected {
            ui.panel_rounded(row, style::ACCENT_SOFT, style::RADIUS_MD);
        } else if hovered {
            ui.panel_rounded(row, style::HOVER_BG, style::RADIUS_MD);
        }

        let color = if selected {
            style::ACCENT
        } else {
            def.accent()
        };
        ui.icon_at(
            row.min_x + (row.width() - style::ICON_LG) * 0.5,
            row.min_y + 8.0,
            def.icon,
            color,
            style::ICON_LG,
        );
        let label = def.name;
        let tw = DrawList::text_width(label, FONT_SCALE * TYPE_CAPTION);
        ui.label_at(
            row.min_x + (row.width() - tw) * 0.5,
            row.min_y + 30.0,
            label,
            if selected {
                style::TEXT
            } else {
                style::TEXT_MUTED
            },
            FONT_SCALE * TYPE_CAPTION,
        );

        if hovered {
            let shortcut: Option<&'static str> = match id.digit_shortcut() {
                Some(1) => Some("1"),
                Some(2) => Some("2"),
                Some(3) => Some("3"),
                Some(4) => Some("4"),
                Some(5) => Some("5"),
                Some(6) => Some("6"),
                Some(7) => Some("7"),
                Some(8) => Some("8"),
                Some(9) => Some("9"),
                _ => None,
            };
            ui.queue_tooltip(row, def.name, def.description, shortcut);
        }

        y += MODE_ROW_H + 2.0;
    }
}

fn draw_tool_panel(
    ui: &mut GuiContext<'_>,
    doc: &TerrainDocument,
    ui_state: &mut UiState,
    state: &mut ToolsGuiState,
    actions: &mut Vec<PanelAction>,
) {
    if ui_state.layout.tool_panel_collapsed {
        // Expand affordance on the left chrome edge.
        let left = if ui_state.app_workspace.shows_mode_rail() {
            ui.mode_rail_rect().max_x
        } else {
            0.0
        };
        let btn = Rect::from_pos_size(left.max(0.0), ui.mode_rail_rect().min_y + 8.0, 18.0, 24.0);
        let id = Id::new("tools_expand");
        let hovered = ui.pointer_in(btn);
        if hovered {
            ui.state.set_hot(id);
        }
        if hovered && ui.input.primary_pressed {
            ui.state.active = Some(id);
        }
        if ui.input.primary_released && ui.state.is_active(id) && hovered {
            ui_state.layout.tool_panel_collapsed = false;
            ui_state.layout_dirty = true;
        }
        ui.panel_rounded(btn, style::BUTTON_BG, style::RADIUS_SM);
        ui.icon_at(
            btn.min_x + 2.0,
            btn.min_y + 4.0,
            Icon::ChevronRight,
            style::TEXT_DIM,
            14.0,
        );
        return;
    }

    let panel = ui.tool_panel_rect();
    if panel.width() < 40.0 {
        return;
    }

    if ui
        .input
        .pointer
        .map(|(x, y)| panel.contains(x, y))
        .unwrap_or(false)
    {
        ui.state.set_hot(Id::new("__tools_bg"));
    }

    ui.panel(panel, style::PANEL_BG);
    ui.panel(
        Rect::from_pos_size(panel.max_x - 1.0, panel.min_y, 1.0, panel.height()),
        style::SEPARATOR,
    );

    let heading = ui_state.workspace_def().tools_heading();

    let header = Rect::from_pos_size(panel.min_x, panel.min_y, panel.width(), style::HEADER_H);
    ui.label_at(
        header.min_x + PAD,
        header.min_y + 11.0,
        heading,
        style::TEXT_MUTED,
        FONT_SCALE * TYPE_CAPTION,
    );
    // Collapse tool panel.
    let collapse = Rect::from_pos_size(header.max_x - 28.0, header.min_y + 6.0, 22.0, 22.0);
    let cid = Id::new("tools_collapse");
    if ui.pointer_in(collapse) {
        ui.state.set_hot(cid);
        if ui.input.primary_pressed {
            ui.state.active = Some(cid);
        }
        if ui.input.primary_released && ui.state.is_active(cid) {
            ui_state.layout.tool_panel_collapsed = true;
            ui_state.layout_dirty = true;
        }
    }
    ui.icon_at(
        collapse.min_x + 4.0,
        collapse.min_y + 4.0,
        Icon::ChevronDown,
        style::TEXT_MUTED,
        14.0,
    );

    // Icon shelf only Ã¢â‚¬â€ Shape / Filters / Masks / Sims live in the Layers hierarchy.
    let list = Rect::from_min_max(panel.min_x, header.max_y, panel.max_x, panel.max_y - 6.0);
    ui.begin_panel_scrolled(
        Id::new("tools_scroll"),
        list,
        Color::rgba(0.0, 0.0, 0.0, 0.0),
        &mut state.scroll_y,
    );

    if ui_state.app_workspace.shows_tool_catalog() {
        draw_tool_catalog_cards(ui, ui_state, state, doc, actions, ui_state.active_workspace);
    }

    ui.end_panel_scrolled(&mut state.scroll_y);
}

fn draw_tool_catalog_cards(
    ui: &mut GuiContext<'_>,
    ui_state: &mut UiState,
    state: &mut ToolsGuiState,
    doc: &TerrainDocument,
    actions: &mut Vec<PanelAction>,
    workspace: WorkspaceId,
) {
    let tools = tools_for_workspace(workspace);
    let mut visible: Vec<&ToolDef> = Vec::new();
    for &tool in &tools {
        if matches!(tool.action, ToolAction::Sculpt(EditorTool::PaintMask)) {
            let upper = doc
                .selected
                .and_then(|id| doc.stack.find(id))
                .is_some_and(|l| !l.kind.is_sculpt_base());
            if !upper && workspace != WorkspaceId::Rules {
                continue;
            }
        }
        visible.push(tool);
    }

    // Stable group order matching WC tool sections (no Generator/Modifier Shape taxonomy).
    const GROUP_ORDER: [ToolGroup; 7] = [
        ToolGroup::Landforms,
        ToolGroup::Hydrology,
        ToolGroup::Noise,
        ToolGroup::Simulation,
        ToolGroup::Masks,
        ToolGroup::Brushes,
        ToolGroup::Other,
    ];
    const FILTER_GROUP_ORDER: [ToolGroup; 10] = [
        ToolGroup::FilterGeneral,
        ToolGroup::FilterDesign,
        ToolGroup::FilterEffect,
        ToolGroup::FilterNoise,
        ToolGroup::FilterArid,
        ToolGroup::FilterTerrace,
        ToolGroup::FilterDrift,
        ToolGroup::FilterBasicErosion,
        ToolGroup::FilterAdvancedErosion,
        ToolGroup::FilterSediment,
    ];
    let group_order: &[ToolGroup] = if workspace == WorkspaceId::Develop {
        &FILTER_GROUP_ORDER
    } else {
        &GROUP_ORDER
    };
    // Within each group, suggested tools float to the top (no banner chrome).
    let suggested_ids = crate::ui::quick_add::contextual_suggestion_type_ids(doc, ui_state);
    for &group in group_order {
        let mut batch: Vec<&ToolDef> = visible
            .iter()
            .copied()
            .filter(|t| t.group() == group)
            .collect();
        if batch.is_empty() {
            continue;
        }
        batch.sort_by_key(|tool| match &tool.action {
            ToolAction::AddLayer { kind, .. } => {
                if suggested_ids.iter().any(|id| id == kind.type_id()) {
                    0usize
                } else {
                    usize::MAX
                }
            }
            _ => usize::MAX,
        });
        let expanded = draw_group_header(ui, state, group);
        if expanded {
            draw_card_grid_prioritized(ui, ui_state, doc, actions, &batch, &suggested_ids);
        }
    }
}

fn draw_card_grid_prioritized(
    ui: &mut GuiContext<'_>,
    ui_state: &mut UiState,
    doc: &TerrainDocument,
    actions: &mut Vec<PanelAction>,
    tools: &[&ToolDef],
    suggested_ids: &[String],
) {
    let content_w = ui.content_width();
    let cols: usize = if content_w >= style::TOOL_CARD_2COL_MIN_W {
        2
    } else {
        1
    };
    let gap = TOOL_CARD_GAP;

    for chunk in tools.chunks(cols) {
        let row = ui.allocate(TOOL_CARD_H);
        let card_w = ((row.width() - gap * (cols as f32 - 1.0).max(0.0)) / cols as f32).max(40.0);
        for (c, tool) in chunk.iter().enumerate() {
            let card = Rect::from_pos_size(
                row.min_x + c as f32 * (card_w + gap),
                row.min_y + 2.0,
                card_w,
                TOOL_CARD_H - 4.0,
            );
            let id = Id::new("tool").child(tool.id);
            let suggested = matches!(
                &tool.action,
                ToolAction::AddLayer { kind, .. }
                    if suggested_ids.iter().any(|s| s == kind.type_id())
            );
            tool_card_suggested(ui, ui_state, doc, actions, id, tool, card, suggested);
        }
    }
}

fn tool_card_suggested(
    ui: &mut GuiContext<'_>,
    ui_state: &mut UiState,
    doc: &TerrainDocument,
    actions: &mut Vec<PanelAction>,
    id: Id,
    tool: &ToolDef,
    card: Rect,
    suggested: bool,
) {
    if suggested {
        ui.panel_rounded(card, style::ACCENT_SOFT, style::RADIUS_SM);
    }
    tool_card(ui, ui_state, doc, actions, id, tool, card);
}

fn draw_group_header(ui: &mut GuiContext<'_>, state: &mut ToolsGuiState, group: ToolGroup) -> bool {
    ui.gap(style::SPACE_1);
    let row = ui.allocate(22.0);
    let id = Id::new("tool_group").child(group.label());
    let hovered = ui.pointer_in(row);
    if hovered {
        ui.state.set_hot(id);
        ui.panel_rounded(row, style::HOVER_BG, style::RADIUS_SM);
    }
    if hovered && ui.input.primary_pressed {
        ui.state.active = Some(id);
    }
    if ui.input.primary_released && ui.state.is_active(id) && hovered {
        state.toggle_group_collapsed(group);
    }
    let collapsed = state.is_group_collapsed(group);
    ui.label_in_rect(
        Rect::from_pos_size(
            row.min_x + 2.0,
            row.min_y,
            (row.width() - 24.0).max(8.0),
            row.height(),
        ),
        group.label(),
        style::TEXT_MUTED,
        FONT_SCALE * TYPE_CAPTION,
    );
    ui.icon_centered(
        Rect::from_pos_size(row.max_x - 20.0, row.min_y, 18.0, row.height()),
        if collapsed {
            Icon::ChevronRight
        } else {
            Icon::ChevronDown
        },
        style::TEXT_MUTED,
        12.0,
    );
    !collapsed
}

fn tool_card(
    ui: &mut GuiContext<'_>,
    ui_state: &mut UiState,
    doc: &TerrainDocument,
    actions: &mut Vec<PanelAction>,
    id: Id,
    tool: &ToolDef,
    card: Rect,
) {
    let hovered = ui.pointer_in(card);
    let selected = match &tool.action {
        ToolAction::Sculpt(t) => ui_state.editor_tool == *t,
        ToolAction::BiomeBrush(brush) => {
            ui_state.editor_tool == EditorTool::PaintBiome && ui_state.biome_paint_tool == *brush
        }
        ToolAction::AddLayer { name, .. } => {
            ui_state.tool_drag.as_ref().is_some_and(|d| d.name == *name)
        }
        ToolAction::CreateBiome { .. } | ToolAction::AddMask { .. } | ToolAction::BakeSelected => {
            false
        }
    };

    if hovered {
        ui.state.set_hot(id);
    }

    if hovered && ui.input.primary_pressed {
        ui.state.active = Some(id);
        match &tool.action {
            ToolAction::AddLayer { name, kind } => {
                ui_state.tool_drag = Some(LayerToolDrag {
                    name,
                    kind: kind.clone(),
                    force_shape_folder: tool.id.starts_with("shape."),
                });
            }
            ToolAction::Sculpt(_)
            | ToolAction::BiomeBrush(_)
            | ToolAction::CreateBiome { .. }
            | ToolAction::AddMask { .. }
            | ToolAction::BakeSelected => {}
        }
    }

    if ui.input.primary_released && ui.state.is_active(id) && hovered {
        match &tool.action {
            ToolAction::Sculpt(editor_tool) => {
                apply_sculpt_tool(ui_state, doc, actions, *editor_tool);
                push_recent(ui_state, tool.id);
            }
            ToolAction::BiomeBrush(brush) => {
                ui_state.set_editor_tool(EditorTool::PaintBiome);
                // Do not force a workspace hop — matches create auto-switch default (off).
                ui_state.tool_drag = None;
                actions.push(PanelAction::SetBiomePaintTool(*brush));
                actions.push(PanelAction::EnsureBiomePaintLayer);
                push_recent(ui_state, tool.id);
            }
            ToolAction::CreateBiome { name } => {
                actions.push(PanelAction::AddBiome {
                    name: (*name).into(),
                });
                push_recent(ui_state, tool.id);
            }
            ToolAction::AddMask { name, source } => {
                let id = MaskId::new();
                let mut asset = MaskAsset {
                    id,
                    name: (*name).into(),
                    source: source.clone(),
                    ops: Vec::new(),
                    paint: None,
                    display_color: terra_core::mask::display_color_for_mask_id(id),
                };
                asset.prepare_for_document();
                actions.push(PanelAction::AddMask(asset));
                // Selection / paint-mode arming happen in the AddMask handler.
                if let Some(layer_id) = doc.selected {
                    actions.push(PanelAction::BindMaskToLayer {
                        layer: layer_id,
                        mask: id,
                    });
                }
                push_recent(ui_state, tool.id);
            }
            ToolAction::BakeSelected => {
                if let Some(layer_id) = doc.selected {
                    actions.push(PanelAction::SetCached {
                        id: layer_id,
                        cached: true,
                    });
                    ui_state.status = "Baked selected layer.".into();
                    push_recent(ui_state, tool.id);
                } else {
                    ui_state.status = "Select a layer to bake.".into();
                }
            }
            ToolAction::AddLayer { .. } => {}
        }
    }

    let bg = if selected {
        style::SELECTED_BG
    } else if hovered {
        style::HOVER_BG
    } else {
        style::SURFACE
    };
    ui.panel_rounded(card, bg, style::RADIUS_MD);

    let thumb_s = TOOL_THUMB_SIZE.min(card.width() - 12.0).max(28.0);
    let thumb_x = card.min_x + (card.width() - thumb_s) * 0.5;
    let thumb_y = card.min_y + 8.0;
    let thumb_r = Rect::from_pos_size(thumb_x, thumb_y, thumb_s, thumb_s);

    let icon_color = if selected {
        style::ACCENT
    } else {
        style::TEXT_DIM
    };

    if ui.rect_visible(thumb_r) {
        ui.panel_rounded(thumb_r, style::RAISED_BG, style::RADIUS_SM);
        if let Some(thumb) = crate::ui::tool_thumbs::thumb_for_tool(tool.id) {
            ui.image(thumb_r, thumb.width, thumb.height, &thumb.rgba);
        } else {
            // Immediate placeholder while the single background decoder loads visible art.
            ui.icon_centered(thumb_r, tool.icon, icon_color, style::ICON_LG);
        }
    }

    let label_y = thumb_r.max_y + 4.0;
    let label_scale = FONT_SCALE * TYPE_CAPTION;
    let label_color = if selected {
        style::TEXT
    } else {
        style::TEXT_DIM
    };
    // Truncate long labels to the card width.
    let max_w = card.width() - 6.0;
    let display = DrawList::truncate_to_width(tool.label, label_scale, max_w);
    let tw = DrawList::text_width(&display, label_scale);
    ui.label_at(
        card.min_x + (card.width() - tw) * 0.5,
        label_y,
        &display,
        label_color,
        label_scale,
    );

    if hovered {
        ui.queue_tooltip(card, tool.label, tool.description, tool.shortcut);
    }
}

fn push_recent(ui_state: &mut UiState, id: &str) {
    ui_state.recent_tools.retain(|t| t != id);
    ui_state.recent_tools.insert(0, id.to_string());
    ui_state.recent_tools.truncate(8);
}

/// Arm Move / Sculpt (last brush) / Mask / Biome from the viewport tool bar.
pub fn select_viewport_tool_mode(
    ui_state: &mut UiState,
    doc: &TerrainDocument,
    actions: &mut Vec<PanelAction>,
    mode: crate::ui::viewport_gui::ViewportToolMode,
) {
    use crate::ui::viewport_gui::ViewportToolMode;
    match mode {
        ViewportToolMode::Move => {
            apply_sculpt_tool(ui_state, doc, actions, EditorTool::Move);
        }
        ViewportToolMode::Measure => {
            ui_state.set_editor_tool(EditorTool::Measure);
            ui_state.tool_drag = None;
            ui_state.paint_mask = None;
            ui_state.shape_session_layer = None;
            actions.push(PanelAction::SetEditorTool(EditorTool::Measure));
        }
        ViewportToolMode::Sculpt => {
            apply_sculpt_tool(ui_state, doc, actions, ui_state.remembered_sculpt_tool());
        }
        ViewportToolMode::Mask => {
            apply_sculpt_tool(ui_state, doc, actions, EditorTool::PaintMask);
        }
        ViewportToolMode::Biome => {
            ui_state.set_editor_tool(EditorTool::PaintBiome);
            ui_state.tool_drag = None;
            actions.push(PanelAction::EnsureBiomePaintLayer);
        }
    }
}

fn apply_sculpt_tool(
    ui_state: &mut UiState,
    doc: &TerrainDocument,
    actions: &mut Vec<PanelAction>,
    tool: EditorTool,
) {
    ui_state.set_editor_tool(tool);
    ui_state.tool_drag = None;
    if tool.is_move() {
        // Disarm paint/shape sessions so LMB look + WASD work immediately.
        ui_state.paint_mask = None;
        ui_state.shape_session_layer = None;
        actions.push(PanelAction::SetEditorTool(EditorTool::Move));
        return;
    }
    if tool.is_sculpt() {
        ui_state.paint_mask = None;
        // Shape history auto-creates SculptStrokes â€” never force SculptBase / Mask setup.
        if ui_state.shape_edit_mode == terra_core::shape_history::ShapeEditMode::NewLayerPerSession
        {
            ui_state.shape_session_layer = None;
        }
        actions.push(PanelAction::SetEditorTool(tool));
        return;
    } else if tool == EditorTool::PaintMask {
        let is_painted = |id: MaskId| {
            doc.masks
                .iter()
                .find(|asset| asset.id == id)
                .is_some_and(|asset| {
                    matches!(asset.source, terra_core::mask::MaskSource::Painted { .. })
                })
        };
        let selected_painted = ui_state.selected_mask.filter(|id| is_painted(*id));
        let bound_painted = doc.selected.and_then(|target| {
            doc.stack
                .find(target)
                .and_then(|layer| {
                    layer
                        .common
                        .masks
                        .iter()
                        .find(|entry| is_painted(entry.mask.id))
                })
                .or_else(|| {
                    doc.stack.find_group(target).and_then(|group| {
                        group.masks.iter().find(|entry| is_painted(entry.mask.id))
                    })
                })
                .map(|entry| entry.mask.id)
        });
        let mask_id = selected_painted.or(bound_painted).unwrap_or_else(|| {
            let asset =
                MaskAsset::new_painted(MaskId::new(), format!("Mask {}", doc.masks.len() + 1), 512);
            let id = asset.id;
            actions.push(PanelAction::AddMask(asset));
            if let Some(target) = doc.selected {
                actions.push(PanelAction::BindMaskToLayer {
                    layer: target,
                    mask: id,
                });
            }
            id
        });
        ui_state.paint_mask = Some(mask_id);
        ui_state.selected_mask = Some(mask_id);
        ui_state.arm_mask_paint();
    }
}

fn finish_tool_drag(
    ui: &mut GuiContext<'_>,
    ui_state: &mut UiState,
    actions: &mut Vec<PanelAction>,
) {
    let over_layers = ui
        .input
        .pointer
        .is_some_and(|(x, y)| ui.right_layers_rect().contains(x, y));

    if ui.input.primary_released {
        if let Some(drag) = ui_state.tool_drag.take() {
            let over_vp = ui
                .input
                .pointer
                .is_some_and(|(x, y)| ui.viewport_rect().contains(x, y));
            let over_tools = ui
                .input
                .pointer
                .is_some_and(|(x, y)| ui.tool_panel_rect().contains(x, y));
            if over_vp || over_tools {
                let layer = instantiate_layer_preset(drag.name, &drag.kind);
                if drag.force_shape_folder {
                    actions.push(PanelAction::AddLayerToCategory {
                        category: terra_core::layer::StackCategory::Shape,
                        layer,
                    });
                } else {
                    actions.push(PanelAction::AddLayer(layer));
                }
            } else if over_layers {
                // Layers panel consumes the drag this frame (drop onto biome / section).
                ui_state.tool_drag = Some(drag);
            }
            // else: cancel drop
        }
    }

    if let Some(drag) = &ui_state.tool_drag {
        if let Some((px, py)) = ui.input.pointer {
            ui.state.set_hot(Id::new("__tool_drag"));
            let tw = DrawList::text_width(drag.name, FONT_SCALE) + 16.0;
            let ghost = Rect::from_pos_size(px + 12.0, py + 12.0, tw, 22.0);
            ui.begin_overlay();
            ui.panel_rounded(ghost, style::ACCENT_DIM, style::RADIUS_SM);
            ui.label_at(
                ghost.min_x + 8.0,
                ghost.min_y + 4.0,
                drag.name,
                style::TEXT,
                FONT_SCALE,
            );
            ui.end_overlay();
        }
        // Keep the drag alive while over the layers panel so hierarchy can accept it.
        if !ui.input.primary_down && !over_layers {
            ui_state.tool_drag = None;
        }
    }
}
