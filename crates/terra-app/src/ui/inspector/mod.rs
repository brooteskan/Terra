//! Inspector panel drawn with `terra-gui` (replaces egui right panel content).

mod edit_kind;

use self::edit_kind::{edit_kind, kind_display_name, KindEditPane};
use crate::ui::actions::PanelAction;
use crate::ui::dist_kinds::{dist_base_kinds, dist_effect_kinds};
use crate::ui::presets::contextual_presets;
use crate::ui::style::{self, FONT_SCALE, INSP_PAD, PAD, ROW_H, TYPE_CAPTION, TYPE_LABEL};
use crate::ui::{hierarchy, EditorTool, InspectorSection, UiState};
use terra_core::document::TerrainDocument;
use terra_core::layer::*;
use terra_gui::{
    button_id, checkbox, collapsible_section, combo, icon_button, icon_toggle, inspector_tab_bar,
    label, label_dim, radio_toggle, section_header, slider_f32, Color, GuiContext, Icon, Id, Rect,
};

/// Lay out a compact wrapping grid of small buttons; returns the clicked index (if any).
fn button_grid(ui: &mut GuiContext<'_>, id_base: Id, items: &[&str], cols: usize) -> Option<usize> {
    let cols = cols.max(1);
    let mut clicked = None;
    let mut i = 0;
    while i < items.len() {
        let row = ui.allocate(ROW_H);
        let n = cols.min(items.len() - i);
        let cell_w = row.width() / cols as f32;
        let cell_gap = style::SPACE_1;
        for c in 0..n {
            let idx = i + c;
            let cell = Rect::from_pos_size(
                row.min_x + cell_w * c as f32,
                row.min_y,
                (cell_w - cell_gap).max(8.0),
                row.height(),
            );
            let id = id_base.with(idx as u64);
            let hovered = ui.pointer_in(cell);
            if hovered {
                ui.state.set_hot(id);
            }
            let active = ui.state.is_active(id);
            if hovered && ui.input.primary_pressed {
                ui.state.active = Some(id);
            }
            let is_clicked = ui.input.primary_released && active && hovered;
            let bg = if active && hovered {
                style::BUTTON_ACTIVE
            } else if hovered {
                style::BUTTON_HOVER
            } else {
                style::BUTTON_BG
            };
            ui.panel_rounded(cell, bg, style::RADIUS_SM);
            ui.label_centered_in_rect(cell, items[idx], style::TEXT, FONT_SCALE * 0.85);
            if is_clicked {
                clicked = Some(idx);
            }
        }
        i += n;
        ui.gap(style::SPACE_1);
    }
    clicked
}

/// Bind project masks onto a layer/group distribution target.
fn draw_project_mask_binds(
    ui: &mut GuiContext<'_>,
    doc: &TerrainDocument,
    actions: &mut Vec<PanelAction>,
    target: LayerId,
) {
    let target_key = target.0.to_string();
    section_header(ui, "PROJECT MASKS");
    if doc.masks.is_empty() {
        label(ui, "Create or paint a mask in the Masks workspace first.");
    } else {
        for (index, mask) in doc.masks.iter().enumerate() {
            if button_id(
                ui,
                Id::new("insp_project_mask")
                    .with(index as u64)
                    .child(&target_key),
                &format!("Use {}", mask.name),
            ) {
                actions.push(PanelAction::BindMaskToLayer {
                    layer: target,
                    mask: mask.id,
                });
            }
        }
    }
}

/// Generator / modifier grids for the distribution stack.
fn draw_distribution_stack_buttons(
    ui: &mut GuiContext<'_>,
    actions: &mut Vec<PanelAction>,
    target: LayerId,
) {
    let target_key = target.0.to_string();
    section_header(ui, "GENERATORS");
    let base = dist_base_kinds();
    let base_labels: Vec<&str> = base.iter().map(|(l, _)| *l).collect();
    if let Some(idx) = button_grid(
        ui,
        Id::new("insp_dist_base").child(&target_key),
        &base_labels,
        3,
    ) {
        actions.push(PanelAction::AddDistNode {
            target,
            kind: base[idx].1.clone(),
        });
    }
    section_header(ui, "MODIFIERS");
    let effects = dist_effect_kinds();
    let effect_labels: Vec<&str> = effects.iter().map(|(l, _)| *l).collect();
    if let Some(idx) = button_grid(
        ui,
        Id::new("insp_dist_effect").child(&target_key),
        &effect_labels,
        3,
    ) {
        actions.push(PanelAction::AddDistEffect {
            target,
            kind: effects[idx].1.clone(),
        });
    }
}

/// Full add UI (project masks + generators + modifiers) for biome/group paths.
fn draw_distribution_add_buttons(
    ui: &mut GuiContext<'_>,
    doc: &TerrainDocument,
    actions: &mut Vec<PanelAction>,
    target: LayerId,
) {
    draw_project_mask_binds(ui, doc, actions, target);
    draw_distribution_stack_buttons(ui, actions, target);
}

/// Short Layer → Masks chrome: counts, project-mask binds, open advanced stack.
fn draw_layer_masks_chrome(
    ui: &mut GuiContext<'_>,
    doc: &TerrainDocument,
    layer: &Layer,
    id: LayerId,
    actions: &mut Vec<PanelAction>,
) {
    label(
        ui,
        &format!(
            "Local Mask: {} refs, {} nodes",
            layer.common.masks.entries.len(),
            layer.common.masks.nodes.len()
        ),
    );
    label_dim(
        ui,
        "Combine with biome Distribution when this layer lives under a biome.",
    );
    draw_project_mask_binds(ui, doc, actions, id);
    if button_id(
        ui,
        Id::new("insp_open_advanced_mask"),
        "Open Advanced Mask Stack",
    ) {
        actions.push(PanelAction::OpenLayerAdvancedMask(id));
    }
}

/// Expand/collapse flags for Unreal Details–style inspector sections.
#[derive(Debug, Clone)]
pub struct DetailsExpandState {
    pub layer_apply_where: bool,
    pub layer_masks: bool,
    pub layer_parameters: bool,
    pub layer_advanced: bool,
    /// Noise nested under the Shape tab.
    pub layer_noise: bool,
}

impl Default for DetailsExpandState {
    fn default() -> Self {
        // Start collapsed so inspector chrome is minimized on project entry.
        Self {
            layer_apply_where: false,
            layer_masks: false,
            layer_parameters: false,
            layer_advanced: false,
            layer_noise: false,
        }
    }
}

/// Persistent scroll for the inspector panel.
#[derive(Debug)]
pub struct InspectorGuiState {
    pub scroll_y: f32,
    /// Active horizontal section tab for the selected layer.
    pub active_tab: InspectorSection,
    /// The layer whose default tab was initialized.
    pub tabs_for: Option<LayerId>,
    /// More-menu open for the selected layer.
    pub more_menu_open: bool,
    /// Inline rename buffer when renaming from the More menu.
    pub rename_buffer: Option<String>,
    /// Collapsible Details sections.
    pub details: DetailsExpandState,
}

impl Default for InspectorGuiState {
    fn default() -> Self {
        Self {
            scroll_y: 0.0,
            active_tab: InspectorSection::General,
            tabs_for: None,
            more_menu_open: false,
            rename_buffer: None,
            details: DetailsExpandState::default(),
        }
    }
}

impl InspectorGuiState {
    /// Reset Details sections to the project-entry default (all collapsed).
    pub fn reset_expand_for_project(&mut self) {
        self.details = DetailsExpandState::default();
    }
}

pub fn draw_inspector_gui(
    ui: &mut GuiContext<'_>,
    doc: &TerrainDocument,
    ui_state: &mut UiState,
    state: &mut InspectorGuiState,
) -> Vec<PanelAction> {
    let mut actions = Vec::new();
    let panel = ui.right_inspector_rect();

    if ui_state.layout.inspector_collapsed {
        if ui.pointer_in(panel) {
            ui.state.set_hot(Id::new("__inspector_collapsed"));
        }
        ui.panel(panel, style::PANEL_BG);
        let header = Rect::from_pos_size(panel.min_x, panel.min_y, panel.width(), style::HEADER_H);
        ui.panel(header, style::PANEL_BG);
        ui.label_at(
            header.min_x + PAD,
            header.min_y + 11.0,
            "INSPECTOR",
            style::TEXT_MUTED,
            FONT_SCALE * TYPE_LABEL,
        );
        let expand = Rect::from_pos_size(header.max_x - 28.0, header.min_y + 6.0, 22.0, 22.0);
        let eid = Id::new("inspector_expand");
        if ui.pointer_in(expand) {
            ui.state.set_hot(eid);
            if ui.input.primary_pressed {
                ui.state.active = Some(eid);
            }
            if ui.input.primary_released && ui.state.is_active(eid) {
                ui_state.layout.inspector_collapsed = false;
                ui_state.layout_dirty = true;
            }
        }
        ui.icon_at(
            expand.min_x + 4.0,
            expand.min_y + 4.0,
            Icon::ChevronUp,
            style::TEXT_MUTED,
            14.0,
        );
        return actions;
    }

    // Absorb camera while hovering the inspector chrome.
    if ui
        .input
        .pointer
        .map(|(x, y)| panel.contains(x, y))
        .unwrap_or(false)
    {
        ui.state.set_hot(Id::new("__inspector_bg"));
    }

    ui.panel(panel, style::PANEL_BG);
    ui.panel(
        Rect::from_pos_size(panel.min_x, panel.min_y, 1.0, panel.height()),
        style::SEPARATOR,
    );

    let header = Rect::from_pos_size(panel.min_x, panel.min_y, panel.width(), style::HEADER_H);
    ui.panel(header, style::PANEL_BG);
    ui.label_at(
        header.min_x + PAD,
        header.min_y + 11.0,
        "INSPECTOR",
        style::TEXT_MUTED,
        FONT_SCALE * TYPE_CAPTION,
    );

    // Collapse inspector (mirrors the tools panel header control).
    let collapse = Rect::from_pos_size(header.max_x - 28.0, header.min_y + 6.0, 22.0, 22.0);
    let cid = Id::new("inspector_collapse");
    if ui.pointer_in(collapse) {
        ui.state.set_hot(cid);
        if ui.input.primary_pressed {
            ui.state.active = Some(cid);
        }
        if ui.input.primary_released && ui.state.is_active(cid) {
            ui_state.layout.inspector_collapsed = true;
            ui_state.layout_dirty = true;
        }
    }
    ui.icon_at(
        collapse.min_x + 4.0,
        collapse.min_y + 4.0,
        Icon::ChevronDown,
        style::TEXT_DIM,
        14.0,
    );

    // Rule under the chrome header so body content starts cleanly.
    ui.panel(
        Rect::from_pos_size(
            panel.min_x + PAD,
            header.max_y - 1.0,
            panel.width() - PAD * 2.0,
            1.0,
        ),
        style::BORDER,
    );

    let body = Rect::from_min_max(panel.min_x, header.max_y, panel.max_x, panel.max_y);
    ui.begin_panel_scrolled_padded(
        Id::new("inspector_scroll"),
        body,
        Color::rgba(0.0, 0.0, 0.0, 0.0),
        &mut state.scroll_y,
        INSP_PAD,
    );

    // Sculpt inspector when Base is the focus; mask paint when that tool is armed.
    let selected_is_base = doc
        .selected
        .and_then(|id| doc.stack.find(id))
        .is_some_and(|l| l.kind.is_sculpt_base());
    if ui_state.editor_tool == EditorTool::PaintMask {
        draw_mask_tool_inspector(ui, doc, ui_state, &mut actions);
        ui.end_panel_scrolled(&mut state.scroll_y);
        return actions;
    }
    if ui_state.editor_tool.is_sculpt() && (selected_is_base || doc.selected.is_none()) {
        draw_tool_inspector(ui, doc, ui_state);
        ui.end_panel_scrolled(&mut state.scroll_y);
        return actions;
    }

    let Some(id) = doc.selected else {
        if ui_state.app_workspace == crate::ui::AppWorkspace::Layout {
            draw_blueprint_inspector(ui, doc, &mut actions);
        } else if ui_state.app_workspace == crate::ui::AppWorkspace::Landforms {
            draw_shape_inspector(ui, doc, &mut actions);
        } else if ui_state.app_workspace == crate::ui::AppWorkspace::Biomes {
            draw_biome_focus_inspector(ui, doc, ui_state, &mut actions);
        } else if ui_state.app_workspace == crate::ui::AppWorkspace::Hydrology {
            label(ui, "Toggle Nature processes in the left panel.");
        } else if ui_state.app_workspace == crate::ui::AppWorkspace::Surface {
            label(ui, "Toggle Look layers in the left panel.");
        } else if ui_state.app_workspace == crate::ui::AppWorkspace::Review {
            label(ui, "Use Review checklist, then Export.");
        } else {
            label(ui, "Select a layer or mask");
        }
        ui.end_panel_scrolled(&mut state.scroll_y);
        return actions;
    };

    // In Biomes intent, prefer artist focus even if a group is somehow selected.
    if ui_state.app_workspace == crate::ui::AppWorkspace::Biomes
        && (ui_state.biome_focus.is_some() || doc.biome_library.selected.is_some())
    {
        draw_biome_focus_inspector(ui, doc, ui_state, &mut actions);
        ui.end_panel_scrolled(&mut state.scroll_y);
        return actions;
    }

    // Group inspector (mode, input, cache).
    if let Some(group) = doc.stack.find_group(id) {
        let is_biome = group.is_biome();
        label(
            ui,
            &format!(
                "{}: {}",
                if is_biome { "Biome" } else { "Group" },
                group.name
            ),
        );
        ui.separator();
        section_header(ui, "GENERAL");
        label(ui, &format!("Kind: {}", group.group_kind.label()));
        label(ui, &format!("Eval mode: {}", group.eval_mode.label()));
        label(ui, &format!("Input mode: {}", group.input_mode.label()));
        let mut opacity = group.opacity;
        if slider_f32(ui, "Opacity", &mut opacity, 0.0, 1.0) {
            actions.push(PanelAction::SetOpacity { id, opacity });
        }
        let mut blend_idx = hierarchy::blend_mode_index(group.blend);
        if combo(ui, "Blend mode", &mut blend_idx, &hierarchy::BLEND_LABELS) {
            actions.push(PanelAction::SetBlend {
                id,
                blend: hierarchy::blend_mode_at(blend_idx),
            });
        }
        label(ui, &format!("Cache: {}", group.cache_policy.label()));
        if is_biome {
            label(ui, &format!("Color tag: {}", group.color_tag));
            if button_id(ui, Id::new("insp_set_active_biome"), "Set as Active Biome") {
                actions.push(PanelAction::SetActiveBiome(id));
            }
            if doc.active_biome == Some(id) {
                label(ui, "â— Active biome context");
            }
            if button_id(ui, Id::new("insp_biome_paint"), "Paint This Biome") {
                actions.push(PanelAction::SetActiveBiome(id));
                actions.push(PanelAction::EnsureBiomePaintLayer);
                actions.push(PanelAction::SetEditorTool(
                    crate::ui::EditorTool::PaintBiome,
                ));
            }
            let show_colors = ui_state.biome_color_preview;
            label(
                ui,
                &format!(
                    "Biome color overlay: {}",
                    if show_colors { "on" } else { "off" }
                ),
            );
            ui.separator();
            section_header(ui, "BIOME BLENDING");
            label(
                ui,
                &format!(
                    "Overwrite Filters: {}",
                    if group.overwrite_filters { "on" } else { "off" }
                ),
            );
            if button_id(
                ui,
                Id::new("insp_biome_toggle_ov_filters"),
                if group.overwrite_filters {
                    "Blend Filters (was Overwrite)"
                } else {
                    "Overwrite Filters (was Blend)"
                },
            ) {
                actions.push(PanelAction::SetBiomeOverwrite {
                    target: id,
                    overwrite_filters: !group.overwrite_filters,
                    overwrite_objects: group.overwrite_objects,
                });
            }
            label(
                ui,
                &format!(
                    "Overwrite Objects: {}",
                    if group.overwrite_objects { "on" } else { "off" }
                ),
            );
            if button_id(
                ui,
                Id::new("insp_biome_toggle_ov_objects"),
                if group.overwrite_objects {
                    "Blend Objects (was Overwrite)"
                } else {
                    "Overwrite Objects (was Blend)"
                },
            ) {
                actions.push(PanelAction::SetBiomeOverwrite {
                    target: id,
                    overwrite_filters: group.overwrite_filters,
                    overwrite_objects: !group.overwrite_objects,
                });
            }
            let mut blending = group.filter_blending;
            if slider_f32(ui, "Filter Blending", &mut blending, 0.0, 1.0) {
                actions.push(PanelAction::SetBiomeFilterBlending {
                    target: id,
                    blending,
                });
            }
            if button_id(
                ui,
                Id::new("insp_biome_cycle_tool"),
                &format!("Tool: {}", ui_state.biome_paint_tool.label()),
            ) {
                actions.push(PanelAction::CycleBiomePaintTool);
            }
        }
        ui.separator();
        label(ui, "DISTRIBUTION");
        label(
            ui,
            &format!(
                "{} mask refs, {} nodes",
                group.masks.entries.len(),
                group.masks.nodes.len()
            ),
        );
        draw_distribution_add_buttons(ui, doc, &mut actions, id);
        ui.separator();
        label(ui, "OUTPUTS");
        if group.outputs.is_empty() {
            label(ui, "(none)");
        } else {
            for out in &group.outputs {
                label(
                    ui,
                    &format!("â€¢ {} [{}]", out.name, out.field.display_name()),
                );
            }
        }
        ui.separator();
        label(ui, "CHILDREN");
        label(ui, &format!("{} nodes", group.children.len()));
        ui.end_panel_scrolled(&mut state.scroll_y);
        return actions;
    }

    let Some(layer) = doc.stack.find(id).cloned() else {
        label(ui, "Select a layer to edit its properties.");
        label(ui, "Or begin with a terrain building block.");
        if button_id(ui, Id::new("insp_quick_add"), "Browse layer tools") {
            actions.push(PanelAction::OpenQuickAdd);
        }
        if button_id(ui, Id::new("insp_add_mountain"), "Add Mountains") {
            actions.push(PanelAction::AddLayer(Layer::new(
                "Mountains",
                LayerKind::Mountains(MountainParams::default()),
            )));
        }
        if button_id(ui, Id::new("insp_add_uplift"), "Add Uplift Range") {
            actions.push(PanelAction::AddLayer(Layer::new(
                "Uplift",
                LayerKind::Uplift(UpliftParams::default()),
            )));
        }
        if button_id(ui, Id::new("insp_add_erosion"), "Add Hydraulic Erosion") {
            actions.push(PanelAction::AddLayer(Layer::new(
                "Hydraulic Erosion",
                LayerKind::HydraulicErosion(HydraulicErosionParams::default()),
            )));
        }
        if button_id(ui, Id::new("insp_add_material"), "Add Materials") {
            actions.push(PanelAction::AddLayer(Layer::new(
                "Materials",
                LayerKind::Materials(MaterialsParams::default()),
            )));
        }
        ui.end_panel_scrolled(&mut state.scroll_y);
        return actions;
    };

    let kind_name = kind_display_name(&layer.kind);
    let primary = primary_section(&layer.kind);
    let presets = contextual_presets(&layer.kind);
    let tabs = available_tabs(&layer.kind, !presets.is_empty());
    initialize_tabs(state, id, &tabs, primary);

    // Layer identity card: name-first (reference style) + action toolbar.
    let card_pad = style::SPACE_3;
    let icon_s = 28.0;
    let btn_s = 24.0;
    let btn_step = style::SPACE_2 + btn_s;
    let identity_h = style::INSP_SELECTION_H - 4.0;
    let mid_gap = style::SPACE_2;
    let card_h = card_pad + identity_h + mid_gap + btn_s + card_pad;
    let card = ui.allocate(card_h);
    ui.panel_rounded(card, style::SURFACE, style::RADIUS_MD);

    // Row 1 — identity: icon + layer name (primary) / kind (secondary).
    let icon_box = Rect::from_pos_size(
        card.min_x + card_pad,
        card.min_y + card_pad + (identity_h - icon_s) * 0.5,
        icon_s,
        icon_s,
    );
    ui.panel_rounded(icon_box, style::RAISED_BG, style::RADIUS_SM);
    ui.icon_centered(
        icon_box,
        hierarchy::layer_type_icon(&layer.kind),
        style::TEXT_DIM,
        16.0,
    );
    let title_x = icon_box.max_x + style::SPACE_2;
    let title_w = (card.max_x - title_x - card_pad).max(8.0);
    ui.label_in_rect(
        Rect::from_pos_size(title_x, card.min_y + card_pad, title_w, 20.0),
        &layer.common.name,
        style::TEXT,
        FONT_SCALE * style::TYPE_BODY,
    );
    ui.label_in_rect(
        Rect::from_pos_size(title_x, card.min_y + card_pad + 18.0, title_w, 16.0),
        kind_name,
        style::TEXT_MUTED,
        FONT_SCALE * TYPE_CAPTION,
    );

    // Row 2 — action toolbar under the identity block (left-aligned, no overlap).
    let btn_y = card.min_y + card_pad + identity_h + mid_gap;
    let mut cx = card.min_x + card_pad;
    let en_r = Rect::from_pos_size(cx, btn_y, btn_s, btn_s);
    cx += btn_step;
    let solo_r = Rect::from_pos_size(cx, btn_y, btn_s, btn_s);
    cx += btn_step;
    let lock_r = Rect::from_pos_size(cx, btn_y, btn_s, btn_s);
    cx += btn_step;
    let reset_r = Rect::from_pos_size(cx, btn_y, btn_s, btn_s);
    cx += btn_step;
    let is_base = layer.kind.is_sculpt_base();
    let del_r = if is_base {
        None
    } else {
        let r = Rect::from_pos_size(cx, btn_y, btn_s, btn_s);
        cx += btn_step;
        Some(r)
    };
    let more_r = Rect::from_pos_size(cx, btn_y, btn_s, btn_s);

    // Enabled toggle (checkbox look).
    let en_id = Id::new("insp_en_toggle");
    let en_hovered = ui.pointer_in(en_r);
    if en_hovered {
        ui.state.set_hot(en_id);
    }
    if en_hovered && ui.input.primary_pressed {
        ui.state.active = Some(en_id);
    }
    if ui.input.primary_released && ui.state.is_active(en_id) && en_hovered {
        actions.push(PanelAction::SetEnabled {
            id,
            enabled: !layer.common.enabled,
        });
    }
    ui.panel_rounded(
        en_r,
        if layer.common.enabled {
            style::CHECK_ON
        } else {
            style::CHECK_BG
        },
        style::RADIUS_SM,
    );
    if layer.common.enabled {
        ui.icon_centered(en_r, Icon::Check, style::TEXT, 14.0);
    }

    if icon_toggle(
        ui,
        Id::new("insp_solo_toggle"),
        Icon::Maximize2,
        solo_r,
        layer.common.solo,
    ) {
        actions.push(PanelAction::SetSolo {
            id,
            solo: !layer.common.solo,
        });
    }
    if icon_toggle(
        ui,
        Id::new("insp_lock_toggle"),
        Icon::Lock,
        lock_r,
        layer.common.locked,
    ) {
        actions.push(PanelAction::SetLocked {
            id,
            locked: !layer.common.locked,
        });
    }
    if icon_button(ui, Id::new("insp_reset_params"), Icon::Undo2, reset_r) {
        if is_base {
            actions.push(PanelAction::ResetSculptBase { id });
        } else {
            actions.push(PanelAction::SetKind {
                id,
                kind: default_kind_like(&layer.kind),
            });
        }
    }
    if let Some(del_r) = del_r {
        if icon_button(ui, Id::new("insp_delete_layer"), Icon::Trash2, del_r) {
            actions.push(PanelAction::RemoveSelected);
        }
    }
    if icon_toggle(
        ui,
        Id::new("insp_more_toggle"),
        Icon::Ellipsis,
        more_r,
        state.more_menu_open,
    ) {
        state.more_menu_open = !state.more_menu_open;
        if state.more_menu_open {
            state.rename_buffer = None;
        }
    }

    if state.more_menu_open {
        draw_inspector_more_menu(ui, ui_state, state, &layer, id, more_r, &mut actions);
    }

    // Inline rename field when active.
    if let Some(buf) = state.rename_buffer.as_mut() {
        let rename_id = Id::new("insp_rename_field");
        let row = ui.allocate(30.0);
        if ui.state.text_focus != Some(rename_id) {
            ui.state.text_focus = Some(rename_id);
            ui.state.text_buffer = buf.clone();
        }
        ui.panel_rounded(row, style::INPUT_BG, style::RADIUS_SM);
        let display = ui.state.text_buffer.clone();
        ui.label_at(
            row.min_x + 10.0,
            row.min_y + (row.height() - 14.0) * 0.5,
            &display,
            style::TEXT,
            FONT_SCALE,
        );
        if ui.state.text_focus == Some(rename_id) {
            if !ui.input.text.is_empty() {
                ui.state.text_buffer.push_str(&ui.input.text);
            }
            if ui.input.backspace_pressed {
                ui.state.text_buffer.pop();
            }
            *buf = ui.state.text_buffer.clone();
            if ui.input.enter_pressed || ui.state.text_enter {
                let name = ui.state.text_buffer.trim().to_string();
                if !name.is_empty() {
                    actions.push(PanelAction::Rename { id, name });
                }
                state.rename_buffer = None;
                ui.state.clear_text_focus();
            }
            if ui.input.escape_pressed {
                state.rename_buffer = None;
                ui.state.clear_text_focus();
            }
        }
    }

    // Clear split between identity chip and top section tabs.
    ui.gap(4.0);
    ui.separator();

    let tab_icons: Vec<Icon> = tabs.iter().map(|t| t.icon()).collect();
    let tab_labels_upper: Vec<String> = tabs.iter().map(|t| t.label().to_uppercase()).collect();
    let tab_refs: Vec<&str> = tab_labels_upper.iter().map(|s| s.as_str()).collect();
    let active_idx = tabs
        .iter()
        .position(|t| *t == state.active_tab)
        .unwrap_or(0);
    if let Some(next) =
        inspector_tab_bar(ui, Id::new("insp_tabs"), &tab_icons, &tab_refs, active_idx)
    {
        if let Some(tab) = tabs.get(next).copied() {
            state.active_tab = tab;
        }
    }
    let active = if tabs.contains(&state.active_tab) {
        state.active_tab
    } else {
        tabs.first().copied().unwrap_or(InspectorSection::General)
    };
    state.active_tab = active;

    // Primary noise split: kinds that expose a separate Noise pane under Shape.
    let split_noise =
        has_separate_noise_tab(&layer.kind) && !matches!(primary, InspectorSection::Noise);

    match active {
        InspectorSection::General => {
            // -- Apply Where --
            if collapsible_section(
                ui,
                Id::new("insp_sec_apply_where"),
                "APPLY WHERE",
                &mut state.details.layer_apply_where,
            ) {
                draw_operation_apply_where(ui, doc, &layer, id, &mut actions);
            }

            // -- Masks (short assignment chrome) --
            if collapsible_section(
                ui,
                Id::new("insp_sec_masks"),
                "MASKS",
                &mut state.details.layer_masks,
            ) {
                draw_layer_masks_chrome(ui, doc, &layer, id, &mut actions);
            }

            // -- Parameters (opacity/blend + presets) --
            if collapsible_section(
                ui,
                Id::new("insp_sec_parameters"),
                "PARAMETERS",
                &mut state.details.layer_parameters,
            ) {
                if layer.kind.is_sculpt_base() {
                    label(ui, "Use Raise / Lower / Smooth in the left palette.");
                    if button_id(ui, Id::new("reset_sculpt"), "Reset Base heights") {
                        actions.push(PanelAction::ResetSculptBase { id });
                    }
                } else {
                    let mut opacity = layer.common.opacity;
                    if slider_f32(ui, "Opacity", &mut opacity, 0.0, 1.0) {
                        actions.push(PanelAction::SetOpacity { id, opacity });
                    }
                    let mut blend_idx = hierarchy::blend_mode_index(layer.common.blend);
                    if combo(ui, "Blend mode", &mut blend_idx, &hierarchy::BLEND_LABELS) {
                        actions.push(PanelAction::SetBlend {
                            id,
                            blend: hierarchy::blend_mode_at(blend_idx),
                        });
                    }
                    let mut enabled = layer.common.enabled;
                    if radio_toggle(ui, "Enabled", &mut enabled) {
                        actions.push(PanelAction::SetEnabled { id, enabled });
                    }
                    let mut solo = layer.common.solo;
                    if radio_toggle(ui, "Solo", &mut solo) {
                        actions.push(PanelAction::SetSolo { id, solo });
                    }
                    let mut cached = layer.common.cached;
                    if radio_toggle(ui, "Cache Layer", &mut cached) {
                        actions.push(PanelAction::SetCached { id, cached });
                    }
                }
                if !presets.is_empty() {
                    ui.gap(4.0);
                    section_header(ui, "PRESETS");
                    for (index, preset) in presets.into_iter().enumerate() {
                        if button_id(
                            ui,
                            Id::new("insp_context_preset").with(index as u64),
                            preset.name,
                        ) {
                            actions.push(PanelAction::SetKind {
                                id,
                                kind: preset.kind,
                            });
                        }
                    }
                }
            }

            // -- Advanced (trailing collapsible under Layer) --
            if collapsible_section(
                ui,
                Id::new("insp_sec_advanced"),
                "ADVANCED",
                &mut state.details.layer_advanced,
            ) {
                ui_state.inspector_advanced = true;
                let mut kind = layer.kind.clone();
                if edit_kind(ui, &mut kind, KindEditPane::Advanced, id, &mut actions) {
                    actions.push(PanelAction::SetKind { id, kind });
                }
            } else {
                ui_state.inspector_advanced = false;
            }
        }
        InspectorSection::Shape => {
            let mut kind = layer.kind.clone();
            let mut changed = false;
            if edit_kind(
                ui,
                &mut kind,
                KindEditPane::Primary { split_noise },
                id,
                &mut actions,
            ) {
                changed = true;
            }
            if has_separate_noise_tab(&layer.kind) {
                if collapsible_section(
                    ui,
                    Id::new("insp_sec_noise"),
                    "NOISE",
                    &mut state.details.layer_noise,
                ) {
                    if edit_kind(ui, &mut kind, KindEditPane::Noise, id, &mut actions) {
                        changed = true;
                    }
                }
            }
            if changed {
                actions.push(PanelAction::SetKind { id, kind });
            }
            ui_state.inspector_advanced = false;
        }
        InspectorSection::Distribution => {
            label(
                ui,
                &format!(
                    "{} mask refs, {} nodes",
                    layer.common.masks.entries.len(),
                    layer.common.masks.nodes.len()
                ),
            );
            draw_distribution_stack_buttons(ui, &mut actions, id);
            ui_state.inspector_advanced = false;
        }
        InspectorSection::Performance => {
            label(
                ui,
                &format!("Preview quality: {}", quality_label(ui_state.quality)),
            );
            label(
                ui,
                &format!("Eval: {:.1} ms", ui_state.profile.eval_us as f32 / 1000.0),
            );
            let backend = if ui_state.profile.path.is_empty() {
                "CPU"
            } else {
                ui_state.profile.path
            };
            label(ui, &format!("Backend: {backend}"));
            if ui_state.refining {
                label(ui, "Status: refining...");
            } else if ui_state.draft_displayed {
                label(ui, "Status: interactive draft");
            } else {
                label(ui, "Status: cached / ready");
            }
            ui_state.inspector_advanced = false;
        }
        // Deep-links to retired peer tabs are remapped by `initialize_tabs`.
        _ => {
            ui_state.inspector_advanced = false;
        }
    }

    ui.end_panel_scrolled(&mut state.scroll_y);
    actions
}

fn draw_inspector_more_menu(
    ui: &mut GuiContext<'_>,
    ui_state: &mut UiState,
    state: &mut InspectorGuiState,
    layer: &Layer,
    id: LayerId,
    anchor: Rect,
    actions: &mut Vec<PanelAction>,
) {
    ui.with_menu_input(|ui| {
        draw_inspector_more_menu_inner(ui, ui_state, state, layer, id, anchor, actions);
    });
}

fn draw_inspector_more_menu_inner(
    ui: &mut GuiContext<'_>,
    ui_state: &mut UiState,
    state: &mut InspectorGuiState,
    layer: &Layer,
    id: LayerId,
    anchor: Rect,
    actions: &mut Vec<PanelAction>,
) {
    let items: &[(&str, &str)] = &[
        ("rename", "Rename"),
        ("dup", "Duplicate"),
        ("reset", "Reset Parameters"),
        ("copy", "Copy Settings"),
        ("paste", "Paste Settings"),
        ("cache", "Toggle Cache"),
        ("del", "Delete"),
        ("close", "Close"),
    ];
    let w = 170.0;
    let h = items.len() as f32 * 26.0 + 8.0;
    let menu = Rect::from_pos_size((anchor.max_x - w).max(8.0), anchor.max_y + 4.0, w, h);
    ui.begin_overlay();
    ui.panel_rounded(menu, style::POPUP_BG, style::RADIUS_SM);
    if ui.pointer_in(menu) {
        ui.state.set_hot(Id::new("insp_more_menu"));
    }
    let is_base = layer.kind.is_sculpt_base();
    let mut iy = menu.min_y + 4.0;
    for (i, (key, label_text)) in items.iter().enumerate() {
        let row = Rect::from_pos_size(menu.min_x + 4.0, iy, w - 8.0, 24.0);
        let disabled = match *key {
            "dup" | "del" => is_base,
            "paste" => ui_state.settings_clipboard.is_none() || is_base,
            "reset" => false,
            _ => false,
        };
        let hovered = ui.pointer_in(row) && !disabled;
        let id_row = Id::new("insp_more_item").with(i as u64);
        if hovered {
            ui.state.set_hot(id_row);
            ui.panel_rounded(row, style::HOVER_BG, style::RADIUS_SM);
        }
        if hovered && ui.input.primary_pressed {
            ui.state.active = Some(id_row);
        }
        ui.label_at(
            row.min_x + 8.0,
            row.min_y + 5.0,
            label_text,
            if disabled {
                style::TEXT_DISABLED
            } else {
                style::TEXT
            },
            FONT_SCALE * 0.9,
        );
        if ui.input.primary_released && ui.state.is_active(id_row) && hovered {
            match *key {
                "rename" => {
                    state.rename_buffer = Some(layer.common.name.clone());
                }
                "dup" if !is_base => actions.push(PanelAction::DuplicateSelected),
                "reset" => {
                    if is_base {
                        actions.push(PanelAction::ResetSculptBase { id });
                    } else {
                        actions.push(PanelAction::SetKind {
                            id,
                            kind: default_kind_like(&layer.kind),
                        });
                    }
                }
                "copy" => {
                    ui_state.settings_clipboard = Some(layer.kind.clone());
                }
                "paste" => {
                    if let Some(kind) = ui_state.settings_clipboard.clone() {
                        actions.push(PanelAction::SetKind { id, kind });
                    }
                }
                "cache" => actions.push(PanelAction::SetCached {
                    id,
                    cached: !layer.common.cached,
                }),
                "del" if !is_base => actions.push(PanelAction::RemoveSelected),
                _ => {}
            }
            if *key != "rename" {
                state.more_menu_open = false;
            } else {
                state.more_menu_open = false;
            }
        }
        iy += 26.0;
    }
    if ui.input.primary_pressed && !ui.pointer_in(menu) && !ui.pointer_in(anchor) {
        state.more_menu_open = false;
    }
    ui.end_overlay();
}

fn default_kind_like(kind: &LayerKind) -> LayerKind {
    use LayerKind::*;
    match kind {
        SculptBase(_) => SculptBase(Default::default()),
        SculptStrokes(_) => SculptStrokes(Default::default()),
        TerrainConstraints(_) => TerrainConstraints(Default::default()),
        GradientReconstruct(_) => GradientReconstruct(Default::default()),
        LandscapeEvolution(_) => LandscapeEvolution(Default::default()),
        HydrologyRepair(_) => HydrologyRepair(Default::default()),
        GeomorphicDetail(_) => GeomorphicDetail(Default::default()),
        EcosystemFeedback(_) => EcosystemFeedback(Default::default()),
        Flat(_) => Flat(Default::default()),
        Ramp(_) => Ramp(Default::default()),
        NoiseValue(_) => NoiseValue(Default::default()),
        NoisePerlin(_) => NoisePerlin(Default::default()),
        NoiseOpenSimplex(_) => NoiseOpenSimplex(Default::default()),
        NoiseWorley(_) => NoiseWorley(Default::default()),
        Fbm(_) => Fbm(Default::default()),
        Ridged(_) => Ridged(Default::default()),
        DomainWarp(_) => DomainWarp(Default::default()),
        Terrace(_) => Terrace(Default::default()),
        Plateau(_) => Plateau(Default::default()),
        Island(_) => Island(Default::default()),
        Mesa(_) => Mesa(Default::default()),
        Mountains(_) => Mountains(Default::default()),
        Volcano(_) => Volcano(Default::default()),
        Uplift(_) => Uplift(Default::default()),
        Canyons(_) => Canyons(Default::default()),
        Dunes(_) => Dunes(Default::default()),
        VoronoiRegions(_) => VoronoiRegions(Default::default()),
        ImportHeightmap(_) => ImportHeightmap(Default::default()),
        Blur(_) => Blur(Default::default()),
        ThermalErosion(_) => ThermalErosion(Default::default()),
        DebrisFlow(_) => DebrisFlow(Default::default()),
        HydraulicErosion(_) => HydraulicErosion(Default::default()),
        StreamPowerErosion(_) => StreamPowerErosion(Default::default()),
        MultiScaleAmplify(_) => MultiScaleAmplify(Default::default()),
        RiverCarve(_) => RiverCarve(Default::default()),
        Coastal(_) => Coastal(Default::default()),
        Materials(_) => Materials(Default::default()),
        Biomes(_) => Biomes(Default::default()),
        Vegetation(_) => Vegetation(Default::default()),
        OverhangStamp(_) => OverhangStamp(Default::default()),
        LocalSdf(_) => LocalSdf(Default::default()),
        EffectFilter(_) => EffectFilter(Default::default()),
        Path(_) => Path(Default::default()),
        RiverNetwork(_) => RiverNetwork(Default::default()),
        SandSimulation(_) => SandSimulation(Default::default()),
        FluidSimulation(_) => FluidSimulation(Default::default()),
        ProceduralShape(_) => ProceduralShape(Default::default()),
        Stamp2d(_) => Stamp2d(Default::default()),
        Stamp3d(_) => Stamp3d(Default::default()),
        PolygonHeight(_) => PolygonHeight(Default::default()),
    }
}

fn quality_label(quality: terra_core::eval::PreviewQuality) -> &'static str {
    match quality {
        terra_core::eval::PreviewQuality::Draft => "Draft",
        terra_core::eval::PreviewQuality::Medium => "Medium",
        terra_core::eval::PreviewQuality::Full => "Full",
        terra_core::eval::PreviewQuality::Export => "Export",
    }
}

fn initialize_tabs(
    state: &mut InspectorGuiState,
    id: LayerId,
    tabs: &[InspectorSection],
    _primary: InspectorSection,
) {
    if state.tabs_for == Some(id) {
        // Keep selection if still valid for this layer.
        if tabs.contains(&state.active_tab) {
            return;
        }
    }
    // Prefer Shape (hosts primary kind editing) when present; else Layer.
    state.active_tab = if tabs.contains(&InspectorSection::Shape) {
        InspectorSection::Shape
    } else {
        tabs.first().copied().unwrap_or(InspectorSection::General)
    };
    state.tabs_for = Some(id);
}

fn available_tabs(kind: &LayerKind, _has_presets: bool) -> Vec<InspectorSection> {
    let mut tabs = vec![InspectorSection::General, InspectorSection::Shape];
    if !kind.is_sculpt_base() {
        tabs.push(InspectorSection::Distribution);
    }
    tabs.push(InspectorSection::Performance);
    tabs
}

fn has_separate_noise_tab(kind: &LayerKind) -> bool {
    matches!(
        kind,
        LayerKind::Mountains(_)
            | LayerKind::Dunes(_)
            | LayerKind::Uplift(_)
            | LayerKind::Mesa(_)
            | LayerKind::Volcano(_)
            | LayerKind::Island(_)
            | LayerKind::DomainWarp(_)
    )
}

fn primary_section(kind: &LayerKind) -> InspectorSection {
    match kind {
        // WC Filters (incl. erosion-style filters)
        LayerKind::EffectFilter(_)
        | LayerKind::Blur(_)
        | LayerKind::Terrace(_)
        | LayerKind::Coastal(_)
        | LayerKind::ThermalErosion(_)
        | LayerKind::DebrisFlow(_)
        | LayerKind::HydraulicErosion(_)
        | LayerKind::StreamPowerErosion(_)
        | LayerKind::MultiScaleAmplify(_)
        | LayerKind::RiverCarve(_) => InspectorSection::Erosion,
        LayerKind::Materials(_) => InspectorSection::Materials,
        LayerKind::Biomes(_) => InspectorSection::Biome,
        LayerKind::Vegetation(_) => InspectorSection::Objects,
        LayerKind::SandSimulation(_)
        | LayerKind::FluidSimulation(_)
        | LayerKind::RiverNetwork(_)
        | LayerKind::LandscapeEvolution(_)
        | LayerKind::HydrologyRepair(_)
        | LayerKind::GeomorphicDetail(_)
        | LayerKind::EcosystemFeedback(_) => InspectorSection::Simulation,
        LayerKind::NoiseValue(_)
        | LayerKind::NoisePerlin(_)
        | LayerKind::NoiseOpenSimplex(_)
        | LayerKind::NoiseWorley(_)
        | LayerKind::Fbm(_)
        | LayerKind::Ridged(_)
        | LayerKind::VoronoiRegions(_) => InspectorSection::Noise,
        LayerKind::OverhangStamp(_) | LayerKind::LocalSdf(_) => InspectorSection::Shape,
        _ => InspectorSection::Shape,
    }
}

fn draw_tool_inspector(ui: &mut GuiContext<'_>, doc: &TerrainDocument, ui_state: &mut UiState) {
    let tool_name = match ui_state.editor_tool {
        EditorTool::Raise => "Raise",
        EditorTool::Lower => "Lower",
        EditorTool::Smooth => "Smooth",
        EditorTool::Ridge => "Ridge",
        EditorTool::Valley => "Valley",
        EditorTool::Roughness => "Roughness",
        EditorTool::UpliftBrush => "Uplift Field",
        EditorTool::Protect => "Protect Shape",
        EditorTool::Hardness => "Hardness",
        EditorTool::Sediment => "Sediment",
        EditorTool::RiverConstraint => "River Guide",
        _ => "Sculpt",
    };
    label(ui, &format!("Sculpt Â· {tool_name}"));
    label(
        ui,
        "Drag to author Base heights or semantic world-space strokes.",
    );
    slider_f32(ui, "Radius", &mut ui_state.sculpt_radius, 0.01, 0.2);
    if ui_state.editor_tool == EditorTool::Smooth {
        let mut s = (ui_state.sculpt_strength / 10.0).clamp(0.05, 1.0);
        if slider_f32(ui, "Strength", &mut s, 0.05, 1.0) {
            ui_state.sculpt_strength = s * 10.0;
        }
    } else {
        slider_f32(ui, "Strength (m)", &mut ui_state.sculpt_strength, 0.5, 40.0);
    }
    if let Some(base) = doc
        .stack
        .flatten_layers()
        .iter()
        .find(|l| l.kind.is_sculpt_base())
    {
        label(ui, &format!("Target: {}", base.common.name));
    }
}

/// Inspector chrome while the Mask tool is armed (TOOLS rail stays visible).
fn draw_mask_tool_inspector(
    ui: &mut GuiContext<'_>,
    doc: &TerrainDocument,
    ui_state: &mut UiState,
    actions: &mut Vec<PanelAction>,
) {
    use terra_core::mask::MaskPaintTool;

    section_header(ui, "MASK");
    label(
        ui,
        "Paint coverage for biomes, filters, materials, or masks.",
    );
    ui.separator();

    section_header(ui, "BRUSH");
    for (i, tool) in [
        MaskPaintTool::Paint,
        MaskPaintTool::Erase,
        MaskPaintTool::Smooth,
        MaskPaintTool::FloodFill,
    ]
    .iter()
    .enumerate()
    {
        let selected = ui_state.mask_paint_tool == *tool;
        let text = if selected {
            format!("[{}]", tool.label())
        } else {
            tool.label().to_string()
        };
        if button_id(ui, Id::new("insp_mask_paint_tool").with(i as u64), &text) {
            ui_state.mask_paint_tool = *tool;
        }
    }
    slider_f32(ui, "Radius", &mut ui_state.sculpt_radius, 0.01, 0.2);
    slider_f32(ui, "Strength", &mut ui_state.sculpt_strength, 0.05, 1.0);
    slider_f32(ui, "Falloff", &mut ui_state.brush_falloff, 0.0, 1.0);

    ui.separator();
    section_header(ui, "TARGET");
    if doc.masks.is_empty() {
        label_dim(
            ui,
            "No project masks yet — use + on Mask Layers (or the mask button) to pick a type.",
        );
    } else {
        for (i, mask) in doc.masks.iter().enumerate() {
            let sel =
                ui_state.paint_mask == Some(mask.id) || ui_state.selected_mask == Some(mask.id);
            let text = if sel {
                format!("â€¢ {}", mask.name)
            } else {
                format!("  {}", mask.name)
            };
            if button_id(ui, Id::new("insp_mask_target").with(i as u64), &text) {
                actions.push(PanelAction::SelectMask(mask.id));
            }
        }
    }

    let mut overlay = ui_state.viewport_overlays.mask_overlay;
    if checkbox(ui, "Show mask overlay", &mut overlay) {
        ui_state.viewport_overlays.mask_overlay = overlay;
    }

    ui.separator();
    label_dim(ui, "Views → Mask opens the full mask editor.");
    if button_id(ui, Id::new("insp_open_mask_view"), "Open full Mask editor") {
        ui_state.enter_mask_view();
    }
}

fn draw_operation_apply_where(
    ui: &mut GuiContext<'_>,
    doc: &TerrainDocument,
    layer: &terra_core::layer::Layer,
    id: terra_core::layer::LayerId,
    actions: &mut Vec<PanelAction>,
) {
    use terra_core::operation_placement::ApplyWhere;

    let biome_name = doc
        .stack
        .enclosing_biome(id)
        .map(|g| g.name.as_str())
        .unwrap_or("World");
    label(ui, &format!("Biome scope: {biome_name}"));
    label_dim(ui, "Inherited automatically â€” no Biome Mask to assign.");

    let op = &layer.common.operation_placement;
    for line in op.summary_lines(biome_name) {
        label(ui, &line);
    }
    ui.gap(4.0);

    let modes = ApplyWhere::all();
    let labels: Vec<&str> = modes.iter().map(|m| m.label()).collect();
    let mut idx = modes.iter().position(|m| *m == op.apply_where).unwrap_or(0);
    if combo(ui, "Apply Where", &mut idx, &labels) {
        actions.push(PanelAction::SetOperationApplyWhere {
            id,
            apply: modes[idx],
        });
    }

    match op.apply_where {
        ApplyWhere::HeightRange => {
            let mut lo = op.height_min;
            let mut hi = op.height_max;
            if slider_f32(ui, "Height min (m)", &mut lo, -200.0, 4000.0)
                | slider_f32(ui, "Height max (m)", &mut hi, -200.0, 4000.0)
            {
                actions.push(PanelAction::SetOperationPlacementParams {
                    id,
                    height_min: Some(lo),
                    height_max: Some(hi),
                    slope_min: None,
                    slope_max: None,
                    flow_min: None,
                    near_distance_m: None,
                });
            }
        }
        ApplyWhere::SlopeRange => {
            let mut lo = op.slope_min;
            let mut hi = op.slope_max;
            if slider_f32(ui, "Slope min (Â°)", &mut lo, 0.0, 90.0)
                | slider_f32(ui, "Slope max (Â°)", &mut hi, 0.0, 90.0)
            {
                actions.push(PanelAction::SetOperationPlacementParams {
                    id,
                    height_min: None,
                    height_max: None,
                    slope_min: Some(lo),
                    slope_max: Some(hi),
                    flow_min: None,
                    near_distance_m: None,
                });
            }
        }
        ApplyWhere::NearWater | ApplyWhere::NearRivers | ApplyWhere::FlowRange => {
            let mut near = op.near_distance_m;
            let mut flow = op.flow_min;
            let mut changed = false;
            if matches!(op.apply_where, ApplyWhere::NearWater) {
                changed |= slider_f32(ui, "Distance (m)", &mut near, 1.0, 500.0);
            } else {
                changed |= slider_f32(ui, "Flow min", &mut flow, 0.0, 1.0);
            }
            if changed {
                actions.push(PanelAction::SetOperationPlacementParams {
                    id,
                    height_min: None,
                    height_max: None,
                    slope_min: None,
                    slope_max: None,
                    flow_min: Some(flow),
                    near_distance_m: Some(near),
                });
            }
        }
        ApplyWhere::AdvancedMask => {
            label_dim(ui, "Edit the Mask Stack below for full DistNode control.");
            if button_id(
                ui,
                Id::new("insp_apply_adv_mask"),
                "Open Advanced Mask Stack",
            ) {
                actions.push(PanelAction::OpenLayerAdvancedMask(id));
            }
        }
        _ => {}
    }
}

fn draw_biome_focus_inspector(
    ui: &mut GuiContext<'_>,
    doc: &TerrainDocument,
    ui_state: &mut UiState,
    actions: &mut Vec<PanelAction>,
) {
    let focus_id = ui_state.biome_focus.or(doc.biome_library.selected);
    let Some(def_id) = focus_id else {
        label(ui, "Select a biome from the palette to paint.");
        return;
    };
    let Some(def) = doc.biome_library.get(def_id) else {
        label(ui, "Select a biome from the palette to paint.");
        return;
    };
    label(ui, "BIOME FOCUS");
    label(ui, &def.name);
    label(ui, "Ownership â€” where this biome applies.");
    ui.separator();
    label(
        ui,
        &format!("Mode: {}", def.placement.combine.artist_label()),
    );
    if button_id(
        ui,
        Id::new("biome_focus_combine"),
        "Toggle Paint owns / Guided",
    ) {
        actions.push(PanelAction::SetBiomePlacementCombine {
            definition: def_id,
            combine: def.placement.combine.cycle_artist(),
        });
    }
    if matches!(
        def.placement.combine,
        terra_core::biome_definition::PlacementCombineMode::PaintMulRules
    ) {
        label(ui, "Guided mode can zero strokes where rules are empty.");
    }
    ui.separator();
    section_header(ui, "PLACEMENT");
    let source = def
        .placement
        .definition
        .as_ref()
        .map(|d| d.source)
        .unwrap_or(terra_core::mask::PlacementSource::Rules);
    let space = def
        .placement
        .definition
        .as_ref()
        .map(|d| d.space)
        .unwrap_or(terra_core::mask::PlacementCoordinateSpace::RuleBased);
    match source {
        terra_core::mask::PlacementSource::Rules => {
            label(ui, "Source: Rules â†’ Mask Stack");
            let conds = def
                .placement
                .definition
                .as_ref()
                .map(|d| d.root.children.len())
                .unwrap_or_else(|| def.placement.rules.is_some() as usize);
            label(ui, &format!("Conditions: {conds} Â· Space: {space:?}"));
        }
        terra_core::mask::PlacementSource::Custom => {
            label(ui, "Source: Custom Mask Stack");
            label(ui, "Rules are frozen until Reset to Rules.");
            if button_id(ui, Id::new("biome_focus_reset_rules"), "Reset to Rules") {
                actions.push(PanelAction::ResetBiomePlacementToRules { definition: def_id });
            }
        }
    }
    if button_id(ui, Id::new("biome_focus_edit_stack"), "Edit Mask Stack") {
        actions.push(PanelAction::MarkBiomePlacementCustom { definition: def_id });
        if let Some(gid) = def.group_id {
            actions.push(PanelAction::Select(gid));
        }
        ui_state.inspector_advanced = true;
    }
    ui.separator();
    label(ui, &format!("Brush: {}", ui_state.biome_paint_tool.label()));
    if button_id(ui, Id::new("biome_focus_paint"), "Arm Paint Brush") {
        actions.push(PanelAction::SetBiomePaintTool(
            terra_core::biome_paint::BiomePaintTool::Paint,
        ));
        actions.push(PanelAction::EnsureBiomePaintLayer);
        actions.push(PanelAction::SetBiomeColorPreview(true));
        actions.push(PanelAction::SetEditorTool(
            crate::ui::EditorTool::PaintBiome,
        ));
    }
    let filters = def.terrain_layers.len();
    let mats = def.material_layers.len();
    label(
        ui,
        &format!("Local WHAT â€” {filters} filters Â· {mats} materials"),
    );
    if let Some(gid) = def.group_id {
        label(ui, "Linked to implementation group.");
        section_header(ui, "MASK STACK");
        draw_distribution_add_buttons(ui, doc, actions, gid);
        if button_id(ui, Id::new("biome_focus_advanced"), "Open in Advanced") {
            actions.push(PanelAction::MarkBiomePlacementCustom { definition: def_id });
            actions.push(PanelAction::Select(gid));
            ui_state.app_workspace = crate::ui::AppWorkspace::Advanced;
            ui_state.inspector_advanced = true;
        }
    } else {
        label(ui, "Not linked â€” recreate from a World Design template.");
    }
    if button_id(ui, Id::new("biome_focus_exit"), "Clear Focus") {
        ui_state.biome_focus = None;
    }
}

fn draw_blueprint_inspector(
    ui: &mut GuiContext<'_>,
    doc: &TerrainDocument,
    actions: &mut Vec<PanelAction>,
) {
    label(ui, "WORLD DESIGN");
    label(ui, &format!("Archetype: {}", doc.blueprint.archetype.0));
    ui.separator();
    label(ui, "ARTIST CONTROLS");
    let mut ridge = doc.blueprint.ridge_sharpness;
    let mut sea = doc.blueprint.sea_level;
    let mut age = doc.blueprint.geological_age;
    let mut rain = doc.blueprint.rainfall;
    let mut drainage = doc.blueprint.drainage_density;
    let mut changed = false;
    changed |= slider_f32(ui, "Ridge Sharpness", &mut ridge, 0.0, 1.0);
    changed |= slider_f32(ui, "Sea Level (m)", &mut sea, -50.0, 50.0);
    changed |= slider_f32(ui, "Geological Age", &mut age, 0.0, 1.0);
    changed |= slider_f32(ui, "Rainfall", &mut rain, 0.0, 3.0);
    changed |= slider_f32(ui, "Drainage Density", &mut drainage, 0.0, 1.0);
    if changed || button_id(ui, Id::new("insp_apply_blueprint"), "Apply to Terrain") {
        actions.push(PanelAction::ApplyBlueprintSemantics {
            ridge_sharpness: ridge,
            sea_level: sea,
            geological_age: age,
            rainfall: rain,
            drainage_density: drainage,
        });
    }
    label(ui, "Age → evolution time; Ridge → spine width");
}

fn draw_shape_inspector(
    ui: &mut GuiContext<'_>,
    doc: &TerrainDocument,
    actions: &mut Vec<PanelAction>,
) {
    label(ui, "LANDFORM SHAPE");
    let Some(id) = doc.shapes.selected else {
        label(ui, "Select a shape in the Landforms tools panel");
        return;
    };
    let Some(shape) = doc.shapes.get(id) else {
        label(ui, "Shape missing");
        return;
    };
    label(ui, &format!("Name: {}", shape.name));
    label(ui, &format!("Kind: {}", shape.kind.label()));
    label(ui, &format!("Width: {:.0} m", shape.width_m));
    label(ui, &format!("Points: {}", shape.points.len()));
    label(ui, &format!("Enabled: {}", shape.enabled));
    ui.separator();
    label(ui, "TRANSLATE");
    let step = 0.015;
    if button_id(ui, Id::new("insp_shape_w"), "â† West") {
        actions.push(PanelAction::TranslateShape {
            id,
            du: -step,
            dv: 0.0,
        });
    }
    if button_id(ui, Id::new("insp_shape_e"), "â†’ East") {
        actions.push(PanelAction::TranslateShape {
            id,
            du: step,
            dv: 0.0,
        });
    }
    if button_id(ui, Id::new("insp_shape_n"), "â†‘ North") {
        actions.push(PanelAction::TranslateShape {
            id,
            du: 0.0,
            dv: -step,
        });
    }
    if button_id(ui, Id::new("insp_shape_s"), "â†“ South") {
        actions.push(PanelAction::TranslateShape {
            id,
            du: 0.0,
            dv: step,
        });
    }
    if button_id(ui, Id::new("insp_shape_compile"), "Compile Shapes") {
        actions.push(PanelAction::CompileShapes);
    }
}
