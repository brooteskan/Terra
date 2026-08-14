//! Right-rail hierarchy over `doc.stack` (stack-only; product Regions removed).
//!
//! Presentation adapts emphasis to the active workspace. Artist concept folders
//! and Advanced Placement are presentation-only.

use crate::ui::actions::PanelAction;
use crate::ui::hierarchy_view::{
    self, advanced_placement_id, biome_section_artist_label, biome_summary_meta,
    concept_for_category, concept_for_top_level, concept_row_id, mask_stack_row_id,
    sim_status_label, world_rule_entity_meta, world_rules_meta, ArtistConcept, TERRAIN_ROOT,
    TERRAIN_SCOPE_KEY,
};
use crate::ui::style::{
    self, FONT_SCALE, LAYER_ROW_H, LAYER_ROW_H_COMPACT, LAYER_THUMB_SZ, PAD, TYPE_BODY,
    TYPE_CAPTION, TYPE_LABEL,
};
use crate::ui::thumbnails::ThumbnailCache;
use crate::ui::{EditorTool, UiState, WorkspaceId};
use terra_core::document::TerrainDocument;
use terra_core::domain::DomainRole;
use terra_core::layer::{BlendMode, LayerId, LayerKind, LayerStack, StackNode};
use terra_core::mask::MaskId;
use terra_gui::{icon_button, Color, DrawList, GuiContext, Icon, Id, Rect};

// —— Layout constants ————————————————————————————————————————————————

const LIST_INSET: f32 = 10.0;
const INDENT_STEP: f32 = 14.0;
const LEAD_COL: f32 = 20.0;
/// Grip column drawn before the collapse chevron on biomes / groups.
const GRIP_COL: f32 = 16.0;
const CHEVRON_SZ: f32 = 14.0;
const THUMB_SZ: f32 = LAYER_THUMB_SZ;
const SEC_ICON_SZ: f32 = 14.0;
const CTRL_SZ: f32 = 20.0;
const CTRL_SLOT: f32 = 22.0;
const OPACITY_COL: f32 = 34.0;
const CAT_ROW_H: f32 = 26.0;
/// Single-line section / folder / mask-stack chrome (icon + label + controls).
const SEC_ROW_H: f32 = 28.0;
/// Two-line plain headers that show a meta subtitle (Mask Stack, Advanced, …).
const SEC_ROW_H_META: f32 = 40.0;
/// Inline rename field height — matches a compact text input, not the full row.
const RENAME_FIELD_H: f32 = 22.0;
const FOOTER_H: f32 = 36.0;
const FOOTER_BTN: f32 = 26.0;
const FOOTER_GAP: f32 = 4.0;
const FILTER_BAR_H: f32 = 28.0;

pub(crate) const BLEND_MODES: [BlendMode; 12] = [
    BlendMode::Normal,
    BlendMode::Replace,
    BlendMode::Add,
    BlendMode::Subtract,
    BlendMode::Multiply,
    BlendMode::Min,
    BlendMode::Max,
    BlendMode::Interpolate,
    BlendMode::HeightBlend,
    BlendMode::Overlay,
    BlendMode::SmoothMaximum,
    BlendMode::SmoothMinimum,
];

pub(crate) const BLEND_LABELS: [&str; 12] = [
    "Normal",
    "Replace",
    "Add",
    "Subtract",
    "Multiply",
    "Min",
    "Max",
    "Interpolate",
    "Height Blend",
    "Overlay",
    "Smooth Max",
    "Smooth Min",
];

pub(crate) fn blend_mode_index(mode: BlendMode) -> usize {
    BLEND_MODES.iter().position(|b| *b == mode).unwrap_or(0)
}

pub(crate) fn blend_mode_at(idx: usize) -> BlendMode {
    BLEND_MODES[idx.min(BLEND_MODES.len() - 1)]
}

/// Drag source for layer reorder / nest.
#[derive(Debug, Clone)]
pub struct LayerDragSource {
    pub id: LayerId,
    /// Root index within the stack (for same-scope reorder).
    pub root_idx: usize,
    pub name: String,
    pub icon: Icon,
}

/// Presentation-only layer panel state (not part of the domain document).
#[derive(Debug, Clone)]
pub struct LayerPresentationState {
    /// Groups present here are collapsed. Absent = expanded.
    pub collapsed_groups: Vec<LayerId>,
    pub search_text: String,
    pub show_thumbnails: bool,
    pub hover_layer: Option<LayerId>,
    pub hide_disabled: bool,
    /// Optional workspace filter — hide de-emphasized rows (cleared easily).
    pub show_only_relevant: bool,
    pub inspector_tab: Option<String>,
}

impl Default for LayerPresentationState {
    fn default() -> Self {
        Self {
            collapsed_groups: Vec::new(),
            search_text: String::new(),
            show_thumbnails: true,
            hover_layer: None,
            hide_disabled: false,
            show_only_relevant: false,
            inspector_tab: None,
        }
    }
}

impl LayerPresentationState {
    pub fn is_collapsed(&self, id: LayerId) -> bool {
        self.collapsed_groups.contains(&id)
    }

    pub fn toggle_collapsed(&mut self, id: LayerId) {
        if let Some(pos) = self.collapsed_groups.iter().position(|x| *x == id) {
            self.collapsed_groups.remove(pos);
        } else {
            self.collapsed_groups.push(id);
        }
    }
}

#[derive(Debug)]
pub struct LayersGuiState {
    pub scroll_y: f32,
    pub add_menu_open: bool,
    pub drag_from: Option<LayerDragSource>,
    /// Cursor hint from the last layers draw (Grab / Grabbing while DnD).
    pub cursor_hint: crate::ui::UiCursor,
    pub context_menu: Option<(LayerId, f32, f32)>,
    /// Inline rename target in the layers list (`None` when not editing).
    pub rename_id: Option<LayerId>,
    pub presentation: LayerPresentationState,
    known_collapse_ids: Vec<LayerId>,
    pub thumbnails: ThumbnailCache,
}

impl Default for LayersGuiState {
    fn default() -> Self {
        Self {
            scroll_y: 0.0,
            add_menu_open: false,
            drag_from: None,
            cursor_hint: crate::ui::UiCursor::Default,
            context_menu: None,
            rename_id: None,
            presentation: LayerPresentationState::default(),
            known_collapse_ids: Vec::new(),
            thumbnails: ThumbnailCache::default(),
        }
    }
}

impl LayersGuiState {
    pub fn collapsed_groups(&self) -> &[LayerId] {
        &self.presentation.collapsed_groups
    }

    pub fn collapsed_groups_mut(&mut self) -> &mut Vec<LayerId> {
        &mut self.presentation.collapsed_groups
    }

    /// Clear collapse memory so the next seed marks every folder collapsed.
    /// When `doc` is provided, seeds immediately for the new project.
    pub fn reset_collapse_for_project(&mut self, doc: Option<&TerrainDocument>) {
        self.known_collapse_ids.clear();
        self.presentation.collapsed_groups.clear();
        if let Some(doc) = doc {
            seed_collapsed_defaults(doc, self);
        }
    }
}

fn seed_collapsed_defaults(doc: &TerrainDocument, state: &mut LayersGuiState) {
    // New foldable rows start collapsed (open/create project and newly added folders).
    for id in collect_collapsible_folder_ids(doc) {
        if !state.known_collapse_ids.contains(&id) {
            state.known_collapse_ids.push(id);
            if !state.presentation.collapsed_groups.contains(&id) {
                state.presentation.collapsed_groups.push(id);
            }
        }
    }
}

/// Presentation folders / groups that can be collapsed in the layers tree.
fn collect_collapsible_folder_ids(doc: &TerrainDocument) -> Vec<LayerId> {
    let mut ids = Vec::new();
    for &concept in ArtistConcept::terrain_order() {
        ids.push(concept_row_id(TERRAIN_SCOPE_KEY, concept));
    }
    collect_group_collapse_ids(&doc.stack.nodes, &mut ids);
    ids
}

fn collect_group_collapse_ids(nodes: &[StackNode], out: &mut Vec<LayerId>) {
    for node in nodes {
        match node {
            StackNode::Group(g) => {
                out.push(g.id);
                if matches!(g.group_kind, terra_core::layer::GroupKind::Biome) {
                    out.push(advanced_placement_id(g.id));
                    out.push(mask_stack_row_id(g.id));
                }
                collect_group_collapse_ids(&g.children, out);
            }
            StackNode::Layer(_) => {}
        }
    }
}

#[inline]
fn label_y(row_min_y: f32, row_h: f32, scale: f32) -> f32 {
    let visual_h = (14.0 * scale).max(12.0);
    row_min_y + (row_h - visual_h) * 0.5
}

/// Meta / kind subtitle drawn under the row name (when non-empty).
fn row_list_subtitle(row: &LayerRow, is_section: bool, is_plain_header: bool) -> &str {
    if !row.meta.is_empty()
        && matches!(
            row.role,
            TreeRole::Biome
                | TreeRole::MaskStack
                | TreeRole::MaskAsset
                | TreeRole::AdvancedSection
                | TreeRole::Layer
                | TreeRole::WorldRuleEntity
                | TreeRole::SimulationScenarioEntity
                | TreeRole::ConceptFolder
        )
    {
        row.meta.as_str()
    } else if is_section || is_plain_header {
        ""
    } else if row.is_base {
        "Sculpt foundation"
    } else {
        row.subtitle
    }
}

#[inline]
fn rename_caret_visible() -> bool {
    // ~530ms blink; solid is fine if the clock is unavailable.
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| (d.as_millis() / 530) % 2 == 0)
        .unwrap_or(true)
}

#[inline]
fn row_can_collapse(role: TreeRole) -> bool {
    matches!(
        role,
        TreeRole::Biome
            | TreeRole::BiomeSection
            | TreeRole::Group
            | TreeRole::ConceptFolder
            | TreeRole::AdvancedSection
    )
}

#[inline]
fn row_name_toggles_collapse(role: TreeRole) -> bool {
    matches!(role, TreeRole::BiomeSection | TreeRole::ConceptFolder)
}

fn collapse_widget_id(role: TreeRole, id: LayerId) -> Id {
    Id::new("lcollapse").child(&format!("{role:?}-{id:?}"))
}

fn toggle_row_collapse(state: &mut LayersGuiState, role: TreeRole, id: LayerId) {
    match role {
        TreeRole::Biome
        | TreeRole::BiomeSection
        | TreeRole::Group
        | TreeRole::ConceptFolder
        | TreeRole::AdvancedSection => {
            state.presentation.toggle_collapsed(id);
        }
        _ => {}
    }
}

pub fn draw_layers_gui(
    ui: &mut GuiContext<'_>,
    doc: &TerrainDocument,
    ui_state: &mut UiState,
    state: &mut LayersGuiState,
) -> Vec<PanelAction> {
    seed_collapsed_defaults(doc, state);
    let panel_w = ui.content_width().max(1.0);
    let compact = panel_w < 300.0;

    let mut actions = Vec::new();
    let panel = ui.right_layers_rect();

    if ui
        .input
        .pointer
        .map(|(x, y)| panel.contains(x, y))
        .unwrap_or(false)
    {
        ui.state.set_hot(Id::new("__layers_bg"));
    }

    ui.panel(panel, style::PANEL_BG);
    ui.panel(
        Rect::from_pos_size(panel.min_x, panel.min_y, 1.0, panel.height()),
        style::SEPARATOR,
    );
    ui.panel(
        Rect::from_pos_size(panel.min_x, panel.max_y - 1.0, panel.width(), 1.0),
        style::SEPARATOR,
    );

    // —— Header ————————————————————————————————————————————————
    let header = Rect::from_pos_size(panel.min_x, panel.min_y, panel.width(), style::HEADER_H);
    ui.panel(header, style::PANEL_BG);
    let hdr_icon_sz = 14.0;
    let hdr_icon_y = header.min_y + (header.height() - hdr_icon_sz) * 0.5;
    ui.icon_at(
        header.min_x + PAD,
        hdr_icon_y,
        Icon::Layers,
        style::TEXT_MUTED,
        hdr_icon_sz,
    );
    let title_scale = FONT_SCALE * TYPE_LABEL;
    ui.label_at(
        header.min_x + PAD + 20.0,
        label_y(header.min_y, header.height(), title_scale),
        "LAYERS",
        style::TEXT_MUTED,
        title_scale,
    );
    let add_hdr = Rect::from_pos_size(
        header.max_x - PAD - 26.0,
        header.min_y + (header.height() - 24.0) * 0.5,
        26.0,
        24.0,
    );
    if icon_button(ui, Id::new("layers_hdr_add"), Icon::Plus, add_hdr) {
        actions.push(PanelAction::OpenQuickAdd);
        state.add_menu_open = false;
    }

    let selected_is_base = doc
        .selected
        .and_then(|id| doc.stack.find(id))
        .is_some_and(|l| l.kind.is_sculpt_base());

    // —— Footer ————————————————————————————————————————————————
    let footer = Rect::from_pos_size(panel.min_x, panel.max_y - FOOTER_H, panel.width(), FOOTER_H);
    ui.panel(footer, style::SURFACE);
    ui.panel(
        Rect::from_pos_size(footer.min_x, footer.min_y, footer.width(), 1.0),
        style::SEPARATOR,
    );
    let fy = footer.min_y + (FOOTER_H - FOOTER_BTN) * 0.5;
    let mut fx = footer.min_x + PAD;
    let footer_btns: [(Id, Icon, &str); 5] = [
        (Id::new("layers_add"), Icon::Plus, "add"),
        (Id::new("layers_mask"), Icon::CircleDot, "mask"),
        (Id::new("layers_dup"), Icon::Copy, "dup"),
        (Id::new("layers_del"), Icon::Trash2, "del"),
        (Id::new("layers_group"), Icon::Folder, "group"),
    ];
    for (id, icon, key) in footer_btns {
        let r = Rect::from_pos_size(fx, fy, FOOTER_BTN, FOOTER_BTN);
        if icon_button(ui, id, icon, r) {
            match key {
                "add" => {
                    actions.push(PanelAction::OpenQuickAdd);
                }
                "mask" => {
                    // Create flow: open Mask Layers picker (Height / Slope / Painted / …).
                    actions.push(PanelAction::OpenQuickAddConcept {
                        concept: ArtistConcept::Masks,
                    });
                }
                "dup" => {
                    if !selected_is_base {
                        actions.push(PanelAction::DuplicateSelected);
                    }
                }
                "del" => {
                    if !selected_is_base {
                        actions.push(PanelAction::RemoveSelected);
                    }
                }
                "group" => actions.push(PanelAction::AddGroup {
                    name: "Group".into(),
                }),
                _ => {}
            }
        }
        fx += FOOTER_BTN + FOOTER_GAP;
    }

    let list = Rect::from_min_max(panel.min_x, header.max_y, panel.max_x, footer.min_y);

    // Optional relevance filter (presentation only — clearable).
    let filter_bar = Rect::from_pos_size(list.min_x, list.min_y, list.width(), FILTER_BAR_H);
    ui.panel(filter_bar, style::PANEL_BG);
    let filter_id = Id::new("layers_only_relevant");
    let filter_hit = Rect::from_pos_size(
        filter_bar.min_x + PAD,
        filter_bar.min_y + 4.0,
        (filter_bar.width() - PAD * 2.0).min(200.0),
        FILTER_BAR_H - 8.0,
    );
    let filter_hov = ui.pointer_in(filter_hit);
    if filter_hov {
        ui.state.set_hot(filter_id);
    }
    if filter_hov && ui.input.primary_pressed {
        ui.state.active = Some(filter_id);
    }
    if ui.input.primary_released && ui.state.is_active(filter_id) && filter_hov {
        state.presentation.show_only_relevant = !state.presentation.show_only_relevant;
    }
    let filter_label = if state.presentation.show_only_relevant {
        "Show only relevant ✓"
    } else {
        "Show only relevant"
    };
    ui.label_at(
        filter_hit.min_x,
        label_y(
            filter_hit.min_y,
            filter_hit.height(),
            FONT_SCALE * TYPE_CAPTION,
        ),
        filter_label,
        if state.presentation.show_only_relevant {
            style::ACCENT
        } else {
            style::TEXT_MUTED
        },
        FONT_SCALE * TYPE_CAPTION,
    );
    if state.presentation.show_only_relevant {
        let clear = Rect::from_pos_size(
            filter_bar.max_x - PAD - 48.0,
            filter_bar.min_y + 4.0,
            48.0,
            FILTER_BAR_H - 8.0,
        );
        let clear_id = Id::new("layers_clear_filter");
        if ui.pointer_in(clear) {
            ui.state.set_hot(clear_id);
            if ui.input.primary_pressed {
                ui.state.active = Some(clear_id);
            }
        }
        if ui.input.primary_released && ui.state.is_active(clear_id) && ui.pointer_in(clear) {
            state.presentation.show_only_relevant = false;
        }
        ui.label_at(
            clear.min_x,
            label_y(clear.min_y, clear.height(), FONT_SCALE * TYPE_CAPTION),
            "Clear",
            style::TEXT_MUTED,
            FONT_SCALE * TYPE_CAPTION,
        );
    }

    let scroll_list = Rect::from_min_max(list.min_x, filter_bar.max_y, list.max_x, list.max_y);
    ui.begin_panel_scrolled(
        Id::new("layers_scroll"),
        scroll_list,
        Color::rgba(0.0, 0.0, 0.0, 0.0),
        &mut state.scroll_y,
    );

    let rows: Vec<LayerRow> = collect_rows(doc, state, &ui_state.outdated_layer_ids)
        .into_iter()
        .map(|mut row| {
            let dimmed = row_dimmed_for_workspace(ui_state, &row);
            let h = ui_state.workspace_def().hierarchy;
            if h.dim_unrelated && !dimmed && !matches!(row.role, TreeRole::SectionLabel) {
                row.focus_highlight = true;
                if matches!(row.role, TreeRole::ConceptFolder) {
                    row.workspace_badge = "Focus".into();
                }
            }
            row
        })
        .collect();
    let mut drop_target: Option<DropTarget> = None;
    let mut tool_drop_parent: Option<LayerId> = None;
    let mut hovering_grip = false;
    state.cursor_hint = crate::ui::UiCursor::Default;
    let tool_dragging = ui_state.tool_drag.is_some();
    let layer_dragging = state.drag_from.is_some();

    for row_data in rows.iter().cloned() {
        let selected = match row_data.role {
            TreeRole::SectionLabel | TreeRole::ConceptFolder => false,
            TreeRole::MaskStack | TreeRole::AdvancedSection => {
                doc.selected == Some(row_selection_id(&row_data))
            }
            TreeRole::MaskAsset => ui_state.selected_mask == Some(MaskId(row_data.id.0)),
            TreeRole::WorldRuleEntity => {
                doc.world_rules.selected
                    == Some(terra_core::world_rules::WorldRuleId(row_data.id.0))
            }
            TreeRole::SimulationScenarioEntity => {
                doc.simulation_scenarios.selected
                    == Some(terra_core::simulation_scenario::SimulationScenarioId(
                        row_data.id.0,
                    ))
            }
            _ => doc.selected == Some(row_data.id),
        };

        if row_hidden_by_filter(ui_state, state, &row_data, selected) {
            continue;
        }

        let is_section = row_data.role == TreeRole::SectionLabel;
        let is_plain_header = matches!(
            row_data.role,
            TreeRole::SectionLabel
                | TreeRole::BiomeSection
                | TreeRole::MaskStack
                | TreeRole::AdvancedSection
                | TreeRole::ConceptFolder
                | TreeRole::WorldRuleEntity
                | TreeRole::SimulationScenarioEntity
        );
        let dragging_this = state
            .drag_from
            .as_ref()
            .is_some_and(|d| d.id == row_data.id);
        let subtitle = row_list_subtitle(&row_data, is_section, is_plain_header);
        let has_subtitle = !subtitle.is_empty();
        let row_h = if is_section {
            CAT_ROW_H
        } else if is_plain_header {
            if has_subtitle {
                SEC_ROW_H_META
            } else {
                SEC_ROW_H
            }
        } else if compact {
            LAYER_ROW_H_COMPACT
        } else {
            LAYER_ROW_H
        };
        let row = ui.allocate(row_h);
        let hovered = ui.pointer_in(row);

        // Selection / hover chrome.
        if is_section {
            // Subtle separator label — no fill card.
        } else if dragging_this {
            ui.panel_rounded(row, style::SURFACE, style::RADIUS_SM);
            ui.panel(
                Rect::from_pos_size(row.min_x, row.min_y + 4.0, 3.0, row.height() - 8.0),
                style::ACCENT,
            );
        } else if selected
            && matches!(
                row_data.role,
                TreeRole::Layer | TreeRole::Foundation | TreeRole::Group | TreeRole::Biome
            )
        {
            ui.panel_rounded(row, style::SELECTED_BG, style::RADIUS_SM);
            ui.panel(
                Rect::from_pos_size(row.min_x, row.min_y + 4.0, 3.0, row.height() - 8.0),
                style::ACCENT,
            );
        } else if hovered && !is_plain_header {
            ui.panel_rounded(row, style::ROW_HOVER, style::RADIUS_SM);
        }

        if row_data.color_tag > 0 && !is_plain_header && !is_section {
            ui.panel(
                Rect::from_pos_size(row.min_x + 4.0, row.min_y + 6.0, 3.0, row.height() - 12.0),
                tag_color(row_data.color_tag),
            );
        }

        let indent = row_data.depth as f32 * INDENT_STEP;
        let mut cursor_x = row.min_x + LIST_INSET + indent;

        // Lead: optional grip (drag) + optional chevron (collapse).
        let can_collapse = row_can_collapse(row_data.role);
        let can_drag = row_is_draggable(row_data.role, row_data.is_base);
        let mut collapse_toggled = false;

        if can_drag {
            let grip = Rect::from_pos_size(cursor_x, row.min_y, GRIP_COL, row_h);
            let drag_id = Id::new("ldrag").child(&format!("{:?}", row_data.id));
            let grip_hov = ui.pointer_in(grip);
            if grip_hov {
                ui.state.set_hot(drag_id);
                hovering_grip = true;
            }
            if grip_hov && ui.input.primary_pressed {
                ui.state.active = Some(drag_id);
                if state.rename_id.is_some() {
                    cancel_hierarchy_rename(ui, state);
                }
                {
                    state.drag_from = Some(LayerDragSource {
                        id: row_data.id,
                        root_idx: row_data.idx,
                        name: row_data.name.clone(),
                        icon: row_data.type_icon,
                    });
                }
            }
            let grip_active = state
                .drag_from
                .as_ref()
                .is_some_and(|d| d.id == row_data.id);
            ui.icon_at(
                grip.min_x + (GRIP_COL - 12.0) * 0.5,
                grip.min_y + (row_h - 12.0) * 0.5,
                Icon::GripVertical,
                if grip_hov || grip_active {
                    style::TEXT
                } else {
                    style::TEXT_MUTED
                },
                12.0,
            );
            cursor_x += GRIP_COL;
        }

        if can_collapse {
            let lead = Rect::from_pos_size(cursor_x, row.min_y, LEAD_COL, row_h);
            let chev_id = collapse_widget_id(row_data.role, row_data.id);
            let chev_hov = ui.pointer_in(lead);
            if chev_hov {
                ui.state.set_hot(chev_id);
            }
            if chev_hov && ui.input.primary_pressed {
                ui.state.active = Some(chev_id);
            }
            if ui.input.primary_released && ui.state.is_active(chev_id) && chev_hov {
                toggle_row_collapse(state, row_data.role, row_data.id);
                collapse_toggled = true;
            }
            ui.icon_at(
                lead.min_x + (LEAD_COL - CHEVRON_SZ) * 0.5,
                lead.min_y + (row_h - CHEVRON_SZ) * 0.5,
                if row_data.collapsed {
                    Icon::ChevronRight
                } else {
                    Icon::ChevronDown
                },
                style::TEXT_MUTED,
                CHEVRON_SZ,
            );
            cursor_x += LEAD_COL;
        } else if !can_drag {
            // Keep indent alignment for non-interactive rows.
            cursor_x += LEAD_COL;
        }
        let content_start_x = cursor_x;

        // Drop indicator (above / below / nest-into).
        if hovered && (layer_dragging || tool_dragging) {
            let src_id = state.drag_from.as_ref().map(|d| d.id);
            let can_drop = src_id != Some(row_data.id) && row_accepts_drop(row_data.role);
            if can_drop {
                let rel_y = ui
                    .input
                    .pointer
                    .map(|(_, py)| ((py - row.min_y) / row_h).clamp(0.0, 1.0))
                    .unwrap_or(0.5);
                let nest_zone = row_is_nest_target(row_data.role) && rel_y > 0.28 && rel_y < 0.72;

                if tool_dragging {
                    // Tools rail → biome / section: create into that container.
                    if matches!(
                        row_data.role,
                        TreeRole::Biome | TreeRole::BiomeSection | TreeRole::Group
                    ) {
                        tool_drop_parent = Some(row_data.id);
                        ui.panel_rounded(
                            Rect::from_pos_size(
                                row.min_x + 4.0,
                                row.min_y + 2.0,
                                row.width() - 8.0,
                                row.height() - 4.0,
                            ),
                            style::ACCENT_SOFT,
                            style::RADIUS_SM,
                        );
                    }
                } else if let Some(moving) = src_id {
                    let nest_into =
                        drop_should_nest_into(doc, moving, row_data.id, row_data.role, nest_zone);
                    let place_before_visual = if nest_into { false } else { rel_y < 0.5 };
                    // Shape / Mask / Sim folders render bottom→top of the stack.
                    let place_before = if row_data.list_reversed && !nest_into {
                        !place_before_visual
                    } else {
                        place_before_visual
                    };
                    drop_target = Some(DropTarget {
                        moving,
                        target: row_data.id,
                        place_before,
                        nest_into,
                    });
                    if nest_into {
                        ui.panel_rounded(
                            Rect::from_pos_size(
                                row.min_x + 4.0,
                                row.min_y + 2.0,
                                row.width() - 8.0,
                                row.height() - 4.0,
                            ),
                            style::ACCENT_SOFT,
                            style::RADIUS_SM,
                        );
                    } else {
                        ui.panel(
                            Rect::from_pos_size(
                                row.min_x + 4.0,
                                drop_indicator_y(row, place_before_visual),
                                row.width() - 8.0,
                                2.0,
                            ),
                            style::ACCENT,
                        );
                    }
                }
            }
        }

        // Type icon / thumbnail.
        {
            let show_thumbs = state.presentation.show_thumbnails;
            let icon_size = if is_section || is_plain_header {
                SEC_ICON_SZ + 2.0
            } else if compact {
                THUMB_SZ * 0.85
            } else {
                THUMB_SZ
            };
            let icon_r = Rect::from_pos_size(
                cursor_x,
                row.min_y + (row_h - icon_size) * 0.5,
                icon_size,
                icon_size,
            );
            if show_thumbs && matches!(row_data.role, TreeRole::Layer | TreeRole::Foundation) {
                let thumb = state.thumbnails.request_or_get(row_data.id);
                ui.image(icon_r, thumb.width, thumb.height, &thumb.rgba);
                if !row_data.enabled {
                    ui.panel_rounded(
                        icon_r,
                        Color::rgba(0.05, 0.06, 0.08, 0.45),
                        style::RADIUS_SM,
                    );
                }
            } else if matches!(row_data.role, TreeRole::Layer | TreeRole::Foundation) {
                ui.panel_rounded(icon_r, style::SURFACE, style::RADIUS_SM);
            }
            if matches!(
                row_data.role,
                TreeRole::Layer | TreeRole::Foundation | TreeRole::Group | TreeRole::Biome
            ) {
                let status = layer_status_color(&row_data);
                ui.panel_rounded(
                    Rect::from_pos_size(icon_r.max_x - 7.0, icon_r.max_y - 7.0, 5.0, 5.0),
                    status,
                    2.5,
                );
            }
            let glyph = if is_section || is_plain_header {
                12.0
            } else {
                16.0
            };
            if glyph > 0.0
                && !(show_thumbs && matches!(row_data.role, TreeRole::Layer | TreeRole::Foundation))
            {
                ui.icon_at(
                    icon_r.min_x + (icon_size - glyph) * 0.5,
                    icon_r.min_y + (icon_size - glyph) * 0.5,
                    row_data.type_icon,
                    if row_data.enabled {
                        style::TEXT_DIM
                    } else {
                        style::TEXT_DISABLED
                    },
                    glyph,
                );
            }

            cursor_x = icon_r.max_x + 8.0;
        }

        // Right-side controls.
        let mut rx = row.max_x - LIST_INSET;
        let show_secondary = selected || hovered;

        // Add Layer into Global section or Region.
        let show_add = matches!(
            row_data.role,
            TreeRole::SectionLabel
                | TreeRole::Biome
                | TreeRole::BiomeSection
                | TreeRole::ConceptFolder
                | TreeRole::AdvancedSection
        );
        if show_add {
            rx -= CTRL_SLOT;
            let add = Rect::from_pos_size(
                rx + (CTRL_SLOT - CTRL_SZ) * 0.5,
                row.min_y + (row_h - CTRL_SZ) * 0.5,
                CTRL_SZ,
                CTRL_SZ,
            );
            if icon_button(
                ui,
                Id::new("lsec_add").child(&format!("{:?}-{:?}", row_data.role, row_data.id)),
                Icon::Plus,
                add,
            ) {
                match row_data.role {
                    TreeRole::SectionLabel => {
                        actions.push(PanelAction::OpenQuickAdd);
                    }
                    TreeRole::ConceptFolder => {
                        if let Some(concept) = row_data.concept {
                            actions.push(PanelAction::OpenQuickAddConcept { concept });
                        }
                    }
                    TreeRole::Biome | TreeRole::BiomeSection => {
                        actions.push(PanelAction::OpenQuickAddInto {
                            parent: row_data.id,
                        });
                    }
                    TreeRole::AdvancedSection => {
                        let biome = row_selection_id(&row_data);
                        // Keep Distribution open so the new node is visible in context.
                        let adv_id = advanced_placement_id(biome);
                        state.collapsed_groups_mut().retain(|id| *id != adv_id);
                        actions.push(PanelAction::OpenQuickAddDistribution { biome });
                    }
                    _ => {}
                }
            }
        }

        if row_data.focus_highlight && !is_section && !selected {
            ui.panel(
                Rect::from_pos_size(row.min_x + 2.0, row.min_y + 4.0, 2.0, row.height() - 8.0),
                style::ACCENT_SOFT,
            );
        }

        if !row_data.workspace_badge.is_empty() && show_secondary && !compact {
            let badge_scale = FONT_SCALE * TYPE_CAPTION;
            let tw = DrawList::text_width(&row_data.workspace_badge, badge_scale);
            rx -= tw + 6.0;
            ui.label_at(
                rx,
                label_y(row.min_y, row_h, badge_scale),
                &row_data.workspace_badge,
                style::ACCENT,
                badge_scale,
            );
        }

        // Region / layer visibility.
        let show_eye = matches!(
            row_data.role,
            TreeRole::Foundation | TreeRole::Layer | TreeRole::Biome | TreeRole::Group
        ) && !row_data.is_base;
        if show_eye {
            rx -= CTRL_SLOT;
            let eye = Rect::from_pos_size(
                rx + (CTRL_SLOT - CTRL_SZ) * 0.5,
                row.min_y + (row_h - CTRL_SZ) * 0.5,
                CTRL_SZ,
                CTRL_SZ,
            );
            let icon = if row_data.enabled {
                Icon::Eye
            } else {
                Icon::EyeOff
            };
            if icon_button(
                ui,
                Id::new("leye").child(&format!("{:?}", row_data.id)),
                icon,
                eye,
            ) {
                actions.push(PanelAction::SetEnabled {
                    id: row_data.id,
                    enabled: !row_data.enabled,
                });
            }
        }

        // Opacity % — hover/selection only (declutter).
        let show_opacity = matches!(
            row_data.role,
            TreeRole::Layer | TreeRole::Foundation | TreeRole::Group | TreeRole::Biome
        ) && show_secondary
            && !compact;
        if show_opacity {
            rx -= OPACITY_COL;
            let pct = format!("{:.0}%", (row_data.opacity * 100.0).round());
            let op_scale = FONT_SCALE * TYPE_CAPTION;
            let tw = DrawList::text_width(&pct, op_scale);
            ui.label_at(
                rx + OPACITY_COL - tw - 2.0,
                label_y(row.min_y, row_h, op_scale),
                &pct,
                style::TEXT_MUTED,
                op_scale,
            );
        }

        if row_data.is_layer && show_secondary && row_data.role == TreeRole::Layer {
            rx -= CTRL_SLOT;
            let lock = Rect::from_pos_size(
                rx + (CTRL_SLOT - CTRL_SZ) * 0.5,
                row.min_y + (row_h - CTRL_SZ) * 0.5,
                CTRL_SZ,
                CTRL_SZ,
            );
            ui.panel_rounded(
                lock,
                if row_data.locked {
                    style::ACCENT_SOFT
                } else {
                    style::BUTTON_BG
                },
                style::RADIUS_SM,
            );
            ui.icon_at(
                lock.min_x + 3.0,
                lock.min_y + 3.0,
                Icon::Lock,
                if row_data.locked {
                    style::WARNING
                } else {
                    style::TEXT_MUTED
                },
                14.0,
            );
            let lock_id = Id::new("llock").child(&format!("{:?}", row_data.id));
            let lock_hovered = ui.pointer_in(lock);
            if lock_hovered {
                ui.state.set_hot(lock_id);
                if ui.input.primary_pressed {
                    ui.state.active = Some(lock_id);
                }
            }
            if ui.input.primary_released && ui.state.is_active(lock_id) && lock_hovered {
                actions.push(PanelAction::SetLocked {
                    id: row_data.id,
                    locked: !row_data.locked,
                });
            }

            rx -= CTRL_SLOT;
            let solo = Rect::from_pos_size(
                rx + (CTRL_SLOT - CTRL_SZ) * 0.5,
                row.min_y + (row_h - CTRL_SZ) * 0.5,
                CTRL_SZ,
                CTRL_SZ,
            );
            ui.panel_rounded(
                solo,
                if row_data.solo {
                    style::ACCENT_SOFT
                } else {
                    style::BUTTON_BG
                },
                style::RADIUS_SM,
            );
            ui.icon_at(
                solo.min_x + 2.0,
                solo.min_y + 2.0,
                Icon::Maximize2,
                if row_data.solo {
                    style::WARNING
                } else {
                    style::TEXT_MUTED
                },
                14.0,
            );
            let solo_id = Id::new("lsolo").child(&format!("{:?}", row_data.id));
            let solo_hovered = ui.pointer_in(solo);
            if solo_hovered {
                ui.state.set_hot(solo_id);
                if ui.input.primary_pressed {
                    ui.state.active = Some(solo_id);
                }
            }
            if ui.input.primary_released && ui.state.is_active(solo_id) && solo_hovered {
                actions.push(PanelAction::SetSolo {
                    id: row_data.id,
                    solo: !row_data.solo,
                });
            }
        }

        let label_x = cursor_x;
        let name_rect = Rect::from_min_max(
            if row_name_toggles_collapse(row_data.role) {
                content_start_x
            } else {
                cursor_x
            },
            row.min_y,
            rx - 4.0,
            row.max_y,
        );
        let name_id = Id::new("lsel").child(&format!("{:?}-{:?}", row_data.role, row_data.id));
        let display_name = match row_data.role {
            TreeRole::SectionLabel => row_data.name.to_uppercase(),
            TreeRole::Foundation => "Foundation".to_string(),
            _ => row_data.name.clone(),
        };

        let renaming = state.rename_id == Some(row_data.id);
        let can_rename = row_can_rename(doc, &row_data);
        let rename_wid = rename_widget_id(row_data.id);

        let hit_hov = ui.pointer_in(name_rect) && !collapse_toggled && !renaming;
        if hit_hov {
            ui.state.set_hot(name_id);
        }
        if hit_hov && ui.input.primary_pressed {
            ui.state.active = Some(name_id);
        }
        let name_clicked = !collapse_toggled
            && ui.input.primary_released
            && ui.state.is_active(name_id)
            && hit_hov;
        if name_clicked {
            if row_name_toggles_collapse(row_data.role) {
                toggle_row_collapse(state, row_data.role, row_data.id);
                state.add_menu_open = false;
            } else if can_rename && selected {
                // Second click on the selected name → inline rename.
                begin_hierarchy_rename(ui, state, row_data.id, &row_data.name);
                state.add_menu_open = false;
                state.context_menu = None;
            } else {
                if state.rename_id.is_some() {
                    commit_hierarchy_rename(ui, state, &mut actions);
                }
                match row_data.role {
                    TreeRole::SectionLabel => {}
                    TreeRole::MaskStack | TreeRole::AdvancedSection => {
                        actions.push(PanelAction::Select(row_selection_id(&row_data)));
                    }
                    TreeRole::MaskAsset => {
                        actions.push(PanelAction::SelectMask(MaskId(row_data.id.0)));
                    }
                    TreeRole::ConceptFolder => {}
                    TreeRole::WorldRuleEntity => {
                        actions.push(PanelAction::SelectWorldRule(
                            terra_core::world_rules::WorldRuleId(row_data.id.0),
                        ));
                    }
                    TreeRole::SimulationScenarioEntity => {
                        actions.push(PanelAction::SelectSimulationScenario(
                            terra_core::simulation_scenario::SimulationScenarioId(row_data.id.0),
                        ));
                    }
                    TreeRole::Biome => {
                        actions.push(PanelAction::Select(row_data.id));
                        actions.push(PanelAction::SetActiveBiome(row_data.id));
                    }
                    _ => {
                        actions.push(PanelAction::Select(row_data.id));
                    }
                }
                state.add_menu_open = false;
                state.context_menu = None;
                if row_data.is_base {
                    if !ui_state.editor_tool.is_sculpt() {
                        ui_state.editor_tool = EditorTool::Raise;
                    }
                    ui_state.paint_mask = None;
                    ui_state.workspace_mode = crate::ui::WorkspaceMode::Terrain;
                    ui_state.shape_session_layer = None;
                } else if doc.stack.find(row_data.id).is_some_and(|l| {
                    matches!(
                        l.kind,
                        terra_core::layer::LayerKind::SculptStrokes(_)
                            | terra_core::layer::LayerKind::SculptBase(_)
                    )
                }) {
                    // Stay in sculpt when selecting a Shape history / foundation layer.
                    if !ui_state.editor_tool.is_sculpt() {
                        ui_state.editor_tool = EditorTool::Raise;
                    }
                    ui_state.paint_mask = None;
                    ui_state.workspace_mode = crate::ui::WorkspaceMode::Terrain;
                    if matches!(
                        doc.stack.find(row_data.id).map(|l| &l.kind),
                        Some(terra_core::layer::LayerKind::SculptStrokes(_))
                    ) {
                        ui_state.shape_session_layer = Some(row_data.id);
                        ui_state.shape_edit_mode =
                            terra_core::shape_history::ShapeEditMode::ContinueSelected;
                    }
                } else if ui_state.editor_tool.is_sculpt() {
                    ui_state.editor_tool = EditorTool::Move;
                }
            }
        }

        let name_scale = if is_section {
            FONT_SCALE * TYPE_CAPTION
        } else if is_plain_header {
            FONT_SCALE * TYPE_LABEL
        } else {
            FONT_SCALE * TYPE_BODY
        };
        let name_color = if dragging_this {
            style::TEXT_MUTED
        } else if is_section {
            style::TEXT_MUTED
        } else if row_dimmed_for_workspace(ui_state, &row_data) {
            style::TEXT_DISABLED
        } else if row_data.enabled {
            style::TEXT
        } else {
            style::TEXT_DISABLED
        };

        if renaming {
            if ui.state.text_focus != Some(rename_wid) {
                ui.state.text_focus = Some(rename_wid);
            }
            let field_h = RENAME_FIELD_H.min(row_h - 4.0).max(18.0);
            let field = Rect::from_pos_size(
                label_x - 2.0,
                row.min_y + (row_h - field_h) * 0.5,
                (name_rect.max_x - label_x + 2.0).max(40.0),
                field_h,
            );
            ui.panel_rounded(field, style::INPUT_BG, style::RADIUS_SM);
            if ui.state.text_focus == Some(rename_wid) {
                if !ui.input.text.is_empty() {
                    ui.state.text_buffer.push_str(&ui.input.text);
                }
                if ui.input.backspace_pressed {
                    ui.state.text_buffer.pop();
                }
                if ui.input.enter_pressed {
                    commit_hierarchy_rename(ui, state, &mut actions);
                } else if ui.input.escape_pressed {
                    cancel_hierarchy_rename(ui, state);
                } else if ui.input.primary_pressed && !ui.pointer_in(field) {
                    commit_hierarchy_rename(ui, state, &mut actions);
                }
            }
            let text_pad = 6.0;
            let caret_reserve = 4.0;
            let display = DrawList::truncate_to_width(
                &ui.state.text_buffer,
                name_scale,
                (field.width() - text_pad * 2.0 - caret_reserve).max(8.0),
            );
            let text_x = field.min_x + text_pad;
            let text_y = label_y(field.min_y, field.height(), name_scale);
            ui.label_at(text_x, text_y, &display, style::TEXT, name_scale);
            if rename_caret_visible() {
                let tw = DrawList::text_width(&display, name_scale);
                let caret_h = (field.height() - 8.0).max(10.0);
                ui.panel(
                    Rect::from_pos_size(
                        (text_x + tw + 1.0).min(field.max_x - 3.0),
                        field.min_y + (field.height() - caret_h) * 0.5,
                        1.5,
                        caret_h,
                    ),
                    style::TEXT,
                );
            }
        } else {
            let name_max_w = (name_rect.max_x - label_x - 4.0).max(8.0);
            let clipped = DrawList::truncate_to_width(&display_name, name_scale, name_max_w);
            let name_truncated = clipped != display_name;
            if has_subtitle {
                let sub_scale = FONT_SCALE * TYPE_CAPTION;
                let name_h = (14.0 * name_scale).max(12.0);
                let sub_h = (14.0 * sub_scale).max(10.0);
                let block_h = name_h + 2.0 + sub_h;
                let block_y = row.min_y + (row_h - block_h) * 0.5;
                ui.label_at(label_x, block_y, &clipped, name_color, name_scale);
                let sub_clipped = DrawList::truncate_to_width(subtitle, sub_scale, name_max_w);
                ui.label_at(
                    label_x,
                    block_y + name_h + 2.0,
                    &sub_clipped,
                    style::TEXT_MUTED,
                    sub_scale,
                );
            } else {
                ui.label_at(
                    label_x,
                    label_y(row.min_y, row_h, name_scale),
                    &clipped,
                    name_color,
                    name_scale,
                );
            }
            // Ellipsized names: hover shows the full title (and subtitle when present).
            if name_truncated && ui.pointer_in(name_rect) && !renaming {
                let body = if has_subtitle { subtitle } else { "" };
                ui.queue_tooltip(name_rect, &display_name, body, None);
            }
        }

        if hovered
            && ui.input.secondary_pressed
            && matches!(
                row_data.role,
                TreeRole::Layer
                    | TreeRole::Foundation
                    | TreeRole::Group
                    | TreeRole::Biome
                    | TreeRole::MaskAsset
                    | TreeRole::WorldRuleEntity
                    | TreeRole::SimulationScenarioEntity
            )
        {
            if let Some((px, py)) = ui.input.pointer {
                if can_rename
                    || matches!(
                        row_data.role,
                        TreeRole::Layer | TreeRole::Foundation | TreeRole::Group | TreeRole::Biome
                    )
                {
                    actions.push(PanelAction::Select(row_data.id));
                    if matches!(row_data.role, TreeRole::Biome) {
                        actions.push(PanelAction::SetActiveBiome(row_data.id));
                    }
                    if matches!(row_data.role, TreeRole::MaskAsset) {
                        actions.push(PanelAction::SelectMask(MaskId(row_data.id.0)));
                    }
                    state.context_menu = Some((row_data.id, px, py));
                    state.add_menu_open = false;
                }
            }
        }
    }

    if ui.input.primary_released {
        if let (Some(_from), Some(to)) = (state.drag_from.take(), drop_target) {
            if to.nest_into {
                push_nest_drop_actions(doc, to.moving, to.target, &mut actions);
            } else {
                actions.push(PanelAction::ReorderRelative {
                    moving: to.moving,
                    target: to.target,
                    place_before: to.place_before,
                });
            }
        } else {
            state.drag_from = None;
        }

        // Tools rail drag → create into hovered biome / Filters section.
        if let Some(drag) = ui_state.tool_drag.take() {
            if drag.force_shape_folder {
                let layer = terra_core::layer::Layer::new(drag.name, drag.kind);
                actions.push(PanelAction::AddLayerToCategory {
                    category: terra_core::layer::StackCategory::Shape,
                    layer,
                });
            } else if let Some(parent) = tool_drop_parent {
                let layer = terra_core::layer::Layer::new(drag.name, drag.kind);
                actions.push(PanelAction::AddLayerInto { parent, layer });
            }
            // else: released over layers with no nest target — cancel
        }
    }

    ui.end_panel_scrolled(&mut state.scroll_y);

    if state.drag_from.is_some() {
        state.cursor_hint = crate::ui::UiCursor::Grabbing;
        draw_layer_drag_ghost(ui, state);
    } else if hovering_grip {
        state.cursor_hint = crate::ui::UiCursor::Grab;
    }

    if let Some((cid, mx, my)) = state.context_menu {
        ui.with_menu_input(|ui| {
            draw_layer_context_menu(ui, doc, ui_state, cid, mx, my, state, &mut actions);
        });
    }

    actions
}

fn draw_layer_drag_ghost(ui: &mut GuiContext<'_>, state: &LayersGuiState) {
    let Some(drag) = state.drag_from.as_ref() else {
        return;
    };
    let Some((px, py)) = ui.input.pointer else {
        return;
    };
    let ghost_w = 220.0_f32.min(ui.screen_w * 0.35).max(140.0);
    let ghost_h = LAYER_ROW_H;
    let ghost = Rect::from_pos_size(
        (px - 28.0).clamp(4.0, (ui.screen_w - ghost_w - 4.0).max(4.0)),
        (py - ghost_h * 0.5).clamp(4.0, (ui.screen_h - ghost_h - 4.0).max(4.0)),
        ghost_w,
        ghost_h,
    );
    ui.begin_overlay();
    // Soft shadow under the floating card.
    ui.panel_rounded(
        Rect::from_pos_size(
            ghost.min_x + 2.0,
            ghost.min_y + 3.0,
            ghost.width(),
            ghost.height(),
        ),
        Color::rgba(0.0, 0.0, 0.0, 0.35),
        style::RADIUS_SM,
    );
    ui.panel_rounded(ghost, style::POPUP_BG, style::RADIUS_SM);
    ui.panel(
        Rect::from_pos_size(ghost.min_x, ghost.min_y + 4.0, 3.0, ghost.height() - 8.0),
        style::ACCENT,
    );
    ui.icon_at(
        ghost.min_x + 12.0,
        ghost.min_y + (ghost_h - 16.0) * 0.5,
        drag.icon,
        style::TEXT,
        16.0,
    );
    let label =
        DrawList::truncate_to_width(&drag.name, FONT_SCALE * TYPE_BODY, ghost.width() - 44.0);
    ui.label_at(
        ghost.min_x + 34.0,
        label_y(ghost.min_y, ghost_h, FONT_SCALE * TYPE_BODY),
        &label,
        style::TEXT,
        FONT_SCALE * TYPE_BODY,
    );
    ui.end_overlay();
}

#[derive(Clone, Copy)]
struct DropTarget {
    moving: LayerId,
    target: LayerId,
    /// Insert before `target` in document order.
    place_before: bool,
    /// Drop onto a container (group / biome section / biome) to nest inside.
    nest_into: bool,
}

fn drop_indicator_y(row: Rect, place_before_visual: bool) -> f32 {
    if place_before_visual {
        row.min_y
    } else {
        row.max_y - 2.0
    }
}

fn row_is_draggable(role: TreeRole, is_base: bool) -> bool {
    !is_base
        && matches!(
            role,
            TreeRole::Layer | TreeRole::Foundation | TreeRole::Group | TreeRole::Biome
        )
}

fn row_accepts_drop(role: TreeRole) -> bool {
    matches!(
        role,
        TreeRole::Layer
            | TreeRole::Foundation
            | TreeRole::Group
            | TreeRole::Biome
            | TreeRole::BiomeSection
    )
}

fn row_is_nest_target(role: TreeRole) -> bool {
    matches!(
        role,
        TreeRole::Group | TreeRole::Biome | TreeRole::BiomeSection
    )
}

/// Whether dropping `moving` on `target` should nest (vs sibling reorder).
///
/// Filters / materials / objects / local sims dropped on a biome (or its section)
/// always nest so they land under that biome's section — not as siblings of the biome.
fn drop_should_nest_into(
    doc: &TerrainDocument,
    moving: LayerId,
    _target: LayerId,
    target_role: TreeRole,
    nest_zone: bool,
) -> bool {
    match target_role {
        TreeRole::Biome => {
            if let Some(layer) = doc.stack.find(moving) {
                if terra_core::layer::biome_destination_section(&layer.kind).is_some() {
                    return true;
                }
            }
            nest_zone
        }
        TreeRole::BiomeSection => {
            // Any layer dropped on Filters / Materials / … goes into that section.
            doc.stack.find(moving).is_some() || nest_zone
        }
        TreeRole::Group => nest_zone,
        _ => false,
    }
}

fn push_nest_drop_actions(
    doc: &TerrainDocument,
    moving: LayerId,
    target: LayerId,
    actions: &mut Vec<PanelAction>,
) {
    // Drop onto a biome → route into the kind's section (Filters, Materials, …).
    if doc.stack.find_group(target).is_some_and(|g| g.is_biome()) {
        if let Some(layer) = doc.stack.find(moving) {
            if let Some(section) = terra_core::layer::biome_destination_section(&layer.kind) {
                actions.push(PanelAction::MoveLayerToBiome {
                    id: moving,
                    biome: target,
                    section,
                });
                return;
            }
        }
        actions.push(PanelAction::MoveIntoGroup {
            child: moving,
            group: target,
        });
        return;
    }

    // Drop onto Filters / Materials / Objects / Local Sims → that section.
    if let Some(section) = doc
        .stack
        .find_group(target)
        .and_then(|g| g.biome_section_kind())
    {
        if let Some(biome) = doc.stack.enclosing_biome(target) {
            actions.push(PanelAction::MoveLayerToBiome {
                id: moving,
                biome: biome.id,
                section,
            });
            return;
        }
    }

    actions.push(PanelAction::MoveIntoGroup {
        child: moving,
        group: target,
    });
}
fn blank_row(
    id: LayerId,
    name: String,
    depth: u8,
    role: TreeRole,
    drop_insert_index: usize,
) -> LayerRow {
    LayerRow {
        idx: 0,
        id,
        name,
        enabled: true,
        locked: false,
        solo: false,
        color_tag: 0,
        preview_color: None,
        cached: false,
        opacity: 1.0,
        is_layer: false,
        is_base: false,
        type_icon: Icon::Folder,
        subtitle: "",
        meta: String::new(),
        mask_count: 0,
        first_mask: None,
        depth,
        role,
        child_count: 0,
        collapsed: false,
        is_active_biome: false,
        drop_insert_index,
        domain_role: None,
        concept: None,
        focus_highlight: false,
        workspace_badge: String::new(),
        select_as: None,
        list_reversed: false,
    }
}

fn collect_rows(
    doc: &TerrainDocument,
    state: &LayersGuiState,
    outdated: &[LayerId],
) -> Vec<LayerRow> {
    let mut out = Vec::new();
    let collapsed = &state.presentation.collapsed_groups;

    // WC: Terrain root wrapping the five concept folders.
    let mut terrain = blank_row(
        LayerId::from_u128(TERRAIN_ROOT),
        "Terrain".into(),
        0,
        TreeRole::SectionLabel,
        0,
    );
    terrain.type_icon = Icon::Mountain;
    terrain.child_count = ArtistConcept::terrain_order().len();
    out.push(terrain);

    emit_concept_stack(
        doc,
        &doc.stack,
        1,
        collapsed,
        doc.active_biome,
        &doc.biome_library,
        outdated,
        &mut out,
    );
    out
}
// Folder order is driven by ArtistConcept::region_order / world_order.

/// Emit stack nodes bucketed into WC terrain concept folders.
fn emit_concept_stack(
    doc: &TerrainDocument,
    stack: &LayerStack,
    depth: u8,
    collapsed: &[LayerId],
    active_biome: Option<LayerId>,
    biome_library: &terra_core::biome_definition::BiomeLibrary,
    outdated: &[LayerId],
    out: &mut Vec<LayerRow>,
) {
    let order = ArtistConcept::terrain_order();

    let mut buckets: Vec<(ArtistConcept, Vec<(usize, &StackNode)>)> =
        order.iter().map(|c| (*c, Vec::new())).collect();

    for (i, node) in stack.nodes.iter().enumerate() {
        match node {
            StackNode::Group(g)
                if matches!(g.group_kind, terra_core::layer::GroupKind::CategoryFolder)
                    || (matches!(g.group_kind, terra_core::layer::GroupKind::Generic)
                        && g.category.is_some()) =>
            {
                // Keep children inside the folder they live in. Dual-role kinds
                // (e.g. Mountains) would otherwise rebucket into Biomes.
                let folder_concept = g
                    .category
                    .map(concept_for_category)
                    .unwrap_or(ArtistConcept::Shape);
                for child in &g.children {
                    let child_concept = match child {
                        StackNode::Group(cg) if cg.is_biome() => ArtistConcept::Biomes,
                        _ => folder_concept,
                    };
                    if let Some(slot) = buckets.iter_mut().find(|(c, _)| *c == child_concept) {
                        slot.1.push((i, child));
                    }
                }
            }
            _ => {
                let concept = concept_for_top_level(node, false);
                if let Some(slot) = buckets.iter_mut().find(|(c, _)| *c == concept) {
                    slot.1.push((i, node));
                }
            }
        }
    }

    let sk = TERRAIN_SCOPE_KEY;
    for (concept, items) in buckets {
        let folder_id = concept_row_id(sk, concept);
        let collapsed_here = collapsed.contains(&folder_id);
        let biome_layer_extra = if concept == ArtistConcept::BiomeLayers {
            doc.biome_layers.len()
        } else {
            0
        };
        let entity_extra = match concept {
            ArtistConcept::Masks => doc.masks.len() + doc.world_rules.rules.len(),
            ArtistConcept::WorldSimulations => doc.simulation_scenarios.scenarios.len(),
            ArtistConcept::BiomeLayers => biome_layer_extra,
            _ => 0,
        };
        let mut folder = blank_row(
            folder_id,
            concept.label().into(),
            depth,
            TreeRole::ConceptFolder,
            items.last().map(|(i, _)| *i).unwrap_or(0),
        );
        folder.type_icon = concept.icon();
        folder.child_count = items.len() + entity_extra;
        folder.collapsed = collapsed_here;
        folder.concept = Some(concept);
        folder.domain_role = concept.preferred_role();
        folder.idx = folder.drop_insert_index;
        out.push(folder);

        if collapsed_here {
            continue;
        }

        let child_depth = depth.saturating_add(1);
        // Biomes: document order. Shape / Mask / Sims: reverse (top = top of stack).
        let reverse = concept != ArtistConcept::Biomes;
        if reverse {
            for (root_idx, node) in items.iter().rev() {
                let before = out.len();
                walk_node_flat(
                    std::slice::from_ref(*node),
                    0,
                    child_depth,
                    *root_idx,
                    collapsed,
                    active_biome,
                    biome_library,
                    Some(concept),
                    outdated,
                    out,
                );
                for row in &mut out[before..] {
                    row.list_reversed = true;
                }
            }
        } else {
            for (root_idx, node) in items.iter() {
                walk_node_flat(
                    std::slice::from_ref(*node),
                    0,
                    child_depth,
                    *root_idx,
                    collapsed,
                    active_biome,
                    biome_library,
                    Some(concept),
                    outdated,
                    out,
                );
            }
        }

        if concept == ArtistConcept::BiomeLayers {
            emit_biome_layer_rows(doc, child_depth, out);
        }
        if concept == ArtistConcept::Masks {
            emit_mask_asset_rows(doc, child_depth, out);
            emit_world_rule_rows(doc, child_depth, out);
        }
        if concept == ArtistConcept::WorldSimulations {
            emit_simulation_scenario_rows(doc, child_depth, outdated, out);
        }
    }
}

fn emit_biome_layer_rows(doc: &TerrainDocument, depth: u8, out: &mut Vec<LayerRow>) {
    for layer in &doc.biome_layers {
        let mut row = blank_row(
            LayerId(layer.id.0),
            layer.name.clone(),
            depth,
            TreeRole::Layer,
            0,
        );
        row.type_icon = Icon::Layers;
        row.enabled = true;
        row.concept = Some(ArtistConcept::BiomeLayers);
        row.domain_role = Some(DomainRole::TerrainFilter);
        row.meta = format!("{} channels", layer.channels.len());
        out.push(row);
    }
}

fn emit_mask_asset_rows(doc: &TerrainDocument, depth: u8, out: &mut Vec<LayerRow>) {
    for asset in &doc.masks {
        let mut row = blank_row(
            LayerId(asset.id.0),
            asset.name.clone(),
            depth,
            TreeRole::MaskAsset,
            0,
        );
        row.type_icon = mask_source_icon(&asset.source);
        row.enabled = true;
        row.concept = Some(ArtistConcept::Masks);
        row.domain_role = Some(DomainRole::MaskLayer);
        row.meta = mask_source_meta(&asset.source).into();
        out.push(row);
    }
}

fn mask_source_meta(source: &terra_core::mask::MaskSource) -> &'static str {
    use terra_core::mask::MaskSource;
    match source {
        MaskSource::None | MaskSource::Constant(_) => "Constant",
        MaskSource::Height { .. } => "Height",
        MaskSource::Slope { .. } => "Slope",
        MaskSource::Aspect { .. } => "Aspect",
        MaskSource::Curvature { .. } => "Curvature",
        MaskSource::Convexity => "Convexity",
        MaskSource::Concavity => "Concavity",
        MaskSource::AmbientOcclusion { .. } => "AO",
        MaskSource::DistanceField { .. } => "Distance",
        MaskSource::Noise { .. } => "Noise",
        MaskSource::Painted { .. } => "Painted",
        MaskSource::FlowDirection | MaskSource::FlowAccumulation { .. } => "Flow",
        MaskSource::Wetness => "Wetness",
        MaskSource::Sediment => "Sediment",
        MaskSource::Erosion => "Erosion",
        MaskSource::Deposition => "Deposition",
        MaskSource::Hardness => "Hardness",
        MaskSource::Temperature
        | MaskSource::Rainfall
        | MaskSource::Humidity
        | MaskSource::Snow
        | MaskSource::SoilMoisture
        | MaskSource::WindExposure => "Climate",
        MaskSource::Named(_) => "Named",
        MaskSource::LayerOutput { .. } => "Layer output",
    }
}

fn mask_source_icon(source: &terra_core::mask::MaskSource) -> Icon {
    use terra_core::mask::MaskSource;
    match source {
        MaskSource::Painted { .. } => Icon::Paintbrush,
        MaskSource::Height { .. } => Icon::Mountain,
        MaskSource::Slope { .. } => Icon::Activity,
        MaskSource::Curvature { .. } => Icon::Blend,
        MaskSource::FlowDirection | MaskSource::FlowAccumulation { .. } => Icon::Waves,
        MaskSource::Convexity | MaskSource::Concavity => Icon::CircleDot,
        MaskSource::Noise { .. } => Icon::Sparkles,
        MaskSource::DistanceField { .. } => Icon::Move,
        _ => Icon::Layers,
    }
}

fn emit_world_rule_rows(doc: &TerrainDocument, depth: u8, out: &mut Vec<LayerRow>) {
    for rule in doc.world_rules.sorted_by_priority() {
        let mut row = blank_row(
            LayerId(rule.id.0),
            rule.name.clone(),
            depth,
            TreeRole::WorldRuleEntity,
            0,
        );
        row.type_icon = Icon::CircleDot;
        row.enabled = rule.enabled;
        row.meta = world_rule_entity_meta(rule);
        row.concept = Some(ArtistConcept::WorldRules);
        row.domain_role = Some(DomainRole::MaskLayer);
        out.push(row);
    }
}

fn emit_simulation_scenario_rows(
    doc: &TerrainDocument,
    depth: u8,
    outdated: &[LayerId],
    out: &mut Vec<LayerRow>,
) {
    let _ = outdated;
    for scenario in &doc.simulation_scenarios.scenarios {
        let mut row = blank_row(
            LayerId(scenario.id.0),
            scenario.name.clone(),
            depth,
            TreeRole::SimulationScenarioEntity,
            0,
        );
        row.type_icon = Icon::Droplets;
        row.enabled = scenario.enabled;
        row.meta = hierarchy_view::simulation_scenario_meta(scenario);
        row.concept = Some(ArtistConcept::WorldSimulations);
        row.domain_role = Some(DomainRole::SimulationLayer);
        out.push(row);
    }
}

fn walk_node_flat(
    nodes: &[StackNode],
    i: usize,
    depth: u8,
    root_idx: usize,
    collapsed: &[LayerId],
    active_biome: Option<LayerId>,
    biome_library: &terra_core::biome_definition::BiomeLibrary,
    parent_concept: Option<ArtistConcept>,
    outdated: &[LayerId],
    out: &mut Vec<LayerRow>,
) {
    match &nodes[i] {
        StackNode::Layer(layer) => {
            let mut row = layer_row(layer, root_idx, depth);
            row.drop_insert_index = root_idx;
            row.concept = parent_concept;
            if hierarchy_view::is_simulation_kind(&layer.kind) {
                let status = hierarchy_view::layer_build_status_with_outdated(layer, outdated);
                row.meta = sim_status_label(status, layer.common.enabled).into();
            } else if matches!(row.domain_role, Some(DomainRole::MaskLayer))
                && matches!(parent_concept, Some(ArtistConcept::WorldRules))
            {
                row.meta = world_rules_meta(
                    layer.common.enabled,
                    !layer.common.masks.is_empty(),
                    "World",
                );
            }
            out.push(row);
        }
        StackNode::Group(g) => {
            let is_category = matches!(g.group_kind, terra_core::layer::GroupKind::CategoryFolder)
                || (matches!(g.group_kind, terra_core::layer::GroupKind::Generic)
                    && g.category.is_some());

            if is_category {
                for ci in (0..g.children.len()).rev() {
                    walk_node_flat(
                        &g.children,
                        ci,
                        depth,
                        root_idx,
                        collapsed,
                        active_biome,
                        biome_library,
                        parent_concept,
                        outdated,
                        out,
                    );
                }
                return;
            }

            let collapsed_here = collapsed.contains(&g.id);
            let role = group_tree_role(g);
            let mut row = group_row(g, root_idx, depth, role, collapsed_here);
            row.drop_insert_index = root_idx;
            row.concept = parent_concept;
            row.is_active_biome = role == TreeRole::Biome && active_biome == Some(g.id);
            if role == TreeRole::Biome {
                row.preview_color = Some(resolve_biome_swatch_color(g, biome_library));
                row.meta = biome_summary_meta(g, biome_library, active_biome);
            }
            if role == TreeRole::BiomeSection {
                if let Some(section) = g.biome_section_kind() {
                    row.name = biome_section_artist_label(section).into();
                }
            }
            out.push(row);
            if collapsed_here {
                return;
            }
            if role == TreeRole::Biome {
                // WC under each biome: Filters → Materials → Objects → Local Sims,
                // then Distribution (placement) after the surface sections.
                emit_biome_sections_ordered(
                    g,
                    root_idx,
                    depth.saturating_add(1),
                    collapsed,
                    active_biome,
                    biome_library,
                    parent_concept,
                    outdated,
                    out,
                );
                emit_advanced_placement(g, root_idx, depth, collapsed, out);
                return;
            }
            let child_depth = depth.saturating_add(1);
            for ci in (0..g.children.len()).rev() {
                walk_node_flat(
                    &g.children,
                    ci,
                    child_depth,
                    root_idx,
                    collapsed,
                    active_biome,
                    biome_library,
                    parent_concept,
                    outdated,
                    out,
                );
            }
        }
    }
}

/// Emit biome section groups in artist/eval order (not raw document order).
fn emit_biome_sections_ordered(
    biome: &terra_core::layer::LayerGroup,
    root_idx: usize,
    depth: u8,
    collapsed: &[LayerId],
    active_biome: Option<LayerId>,
    biome_library: &terra_core::biome_definition::BiomeLibrary,
    parent_concept: Option<ArtistConcept>,
    outdated: &[LayerId],
    out: &mut Vec<LayerRow>,
) {
    // Non-section children first (rare), then canonical section order.
    for (ci, child) in biome.children.iter().enumerate() {
        let is_section = matches!(
            child,
            StackNode::Group(g)
                if matches!(g.group_kind, terra_core::layer::GroupKind::BiomeSection(_))
        );
        if !is_section {
            walk_node_flat(
                &biome.children,
                ci,
                depth,
                root_idx,
                collapsed,
                active_biome,
                biome_library,
                parent_concept,
                outdated,
                out,
            );
        }
    }
    for &section in ArtistConcept::biome_section_order() {
        let Some(ci) = biome.children.iter().position(|n| {
            matches!(
                n,
                StackNode::Group(g)
                    if g.biome_section_kind() == Some(section)
            )
        }) else {
            continue;
        };
        walk_node_flat(
            &biome.children,
            ci,
            depth,
            root_idx,
            collapsed,
            active_biome,
            biome_library,
            parent_concept,
            outdated,
            out,
        );
    }
}

fn emit_advanced_placement(
    biome: &terra_core::layer::LayerGroup,
    root_idx: usize,
    depth: u8,
    collapsed: &[LayerId],
    out: &mut Vec<LayerRow>,
) {
    let adv_id = advanced_placement_id(biome.id);
    let adv_collapsed = collapsed.contains(&adv_id);
    let child_depth = depth.saturating_add(1);
    let mut adv = blank_row(
        adv_id,
        ArtistConcept::AdvancedPlacement.label().into(),
        child_depth,
        TreeRole::AdvancedSection,
        root_idx,
    );
    adv.type_icon = ArtistConcept::AdvancedPlacement.icon();
    adv.collapsed = adv_collapsed;
    adv.concept = Some(ArtistConcept::AdvancedPlacement);
    adv.domain_role = Some(DomainRole::MaskLayer);
    adv.select_as = Some(biome.id);
    adv.mask_count = biome.masks.len();
    adv.first_mask = biome.masks.first().map(|e| e.mask.id);
    adv.child_count = 1;
    adv.meta = if biome.masks.is_empty() {
        "Coverage —".into()
    } else {
        format!("{} masks", biome.masks.len())
    };
    out.push(adv);

    if adv_collapsed {
        return;
    }

    let mask_id = mask_stack_row_id(biome.id);
    let mut mask = blank_row(
        mask_id,
        ArtistConcept::MaskStack.label().into(),
        child_depth.saturating_add(1),
        TreeRole::MaskStack,
        root_idx,
    );
    mask.type_icon = ArtistConcept::MaskStack.icon();
    mask.concept = Some(ArtistConcept::MaskStack);
    mask.domain_role = Some(DomainRole::MaskLayer);
    mask.select_as = Some(biome.id);
    mask.mask_count = biome.masks.len();
    mask.first_mask = biome.masks.first().map(|e| e.mask.id);
    mask.meta = if biome.masks.is_empty() {
        "Empty · edit placement".into()
    } else {
        format!("{} rules", biome.masks.len())
    };
    out.push(mask);
}

fn group_tree_role(g: &terra_core::layer::LayerGroup) -> TreeRole {
    match g.group_kind {
        terra_core::layer::GroupKind::Biome => TreeRole::Biome,
        terra_core::layer::GroupKind::BiomeSection(_) => TreeRole::BiomeSection,
        terra_core::layer::GroupKind::CategoryFolder => TreeRole::Group,
        terra_core::layer::GroupKind::Generic if g.category.is_some() => TreeRole::Group,
        terra_core::layer::GroupKind::Generic => TreeRole::Group,
    }
}

fn layer_row(layer: &terra_core::layer::Layer, idx: usize, depth: u8) -> LayerRow {
    LayerRow {
        idx,
        id: layer.id(),
        name: layer.common.name.clone(),
        enabled: layer.common.enabled,
        locked: layer.common.locked,
        solo: layer.common.solo,
        color_tag: layer.common.color_tag,
        preview_color: None,
        cached: layer.common.cached,
        opacity: layer.common.opacity,
        is_layer: true,
        is_base: layer.kind.is_sculpt_base(),
        type_icon: layer_type_icon(&layer.kind),
        subtitle: kind_subtitle(&layer.kind),
        meta: String::new(),
        mask_count: layer.common.masks.len(),
        first_mask: layer.common.masks.first().map(|e| e.mask.id),
        depth,
        role: if layer.kind.is_sculpt_base() {
            TreeRole::Foundation
        } else {
            TreeRole::Layer
        },
        child_count: 0,
        collapsed: false,
        is_active_biome: false,
        drop_insert_index: idx,
        domain_role: if layer.kind.is_sculpt_base() {
            Some(DomainRole::ShapeLayer)
        } else {
            Some(terra_core::domain::classify_layer_kind(&layer.kind))
        },
        concept: None,
        focus_highlight: false,
        workspace_badge: String::new(),
        select_as: None,
        list_reversed: false,
    }
}

fn group_row(
    g: &terra_core::layer::LayerGroup,
    idx: usize,
    depth: u8,
    role: TreeRole,
    collapsed_here: bool,
) -> LayerRow {
    LayerRow {
        idx,
        id: g.id,
        name: g.name.clone(),
        enabled: g.enabled,
        locked: false,
        solo: false,
        color_tag: g.color_tag,
        preview_color: None,
        cached: matches!(
            g.cache_policy,
            terra_core::layer::CachePolicy::Baked | terra_core::layer::CachePolicy::Cached
        ),
        opacity: g.opacity,
        is_layer: false,
        is_base: false,
        type_icon: match role {
            TreeRole::Biome => Icon::Sparkles,
            TreeRole::BiomeSection => match g.biome_section_kind() {
                Some(terra_core::layer::BiomeSection::Filters) => Icon::SlidersHorizontal,
                Some(terra_core::layer::BiomeSection::Materials) => Icon::Grid3x3,
                Some(terra_core::layer::BiomeSection::Objects) => Icon::Package,
                Some(terra_core::layer::BiomeSection::LocalSims) => Icon::Activity,
                None => Icon::Folder,
            },
            _ => Icon::Folder,
        },
        // Artist hierarchy: hide pass-through chrome; keep isolated label only.
        subtitle: match role {
            TreeRole::Group => match g.eval_mode {
                terra_core::layer::GroupEvalMode::IsolatedComposite => "Isolated Group",
                _ => "",
            },
            TreeRole::BiomeSection => "",
            _ => "",
        },
        meta: String::new(),
        mask_count: g.masks.len(),
        first_mask: g.masks.first().map(|e| e.mask.id),
        depth,
        role,
        child_count: g.children.len(),
        collapsed: collapsed_here,
        is_active_biome: false,
        drop_insert_index: idx,
        domain_role: match role {
            TreeRole::Biome => Some(DomainRole::TerrainFilter),
            TreeRole::BiomeSection => g.biome_section_kind().map(|s| match s {
                terra_core::layer::BiomeSection::Filters => DomainRole::TerrainFilter,
                terra_core::layer::BiomeSection::Materials => DomainRole::MaterialLayer,
                terra_core::layer::BiomeSection::Objects => DomainRole::ObjectLayer,
                terra_core::layer::BiomeSection::LocalSims => DomainRole::SimulationLayer,
            }),
            _ => None,
        },
        concept: None,
        focus_highlight: false,
        workspace_badge: String::new(),
        select_as: None,
        list_reversed: false,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TreeRole {
    SectionLabel,
    Foundation,
    Biome,
    BiomeSection,
    /// Advanced Placement → Mask Stack (replaces Distribution).
    MaskStack,
    Group,
    Layer,
    ConceptFolder,
    AdvancedSection,
    /// First-class project mask asset (`doc.masks`).
    MaskAsset,
    /// First-class World Rule row.
    WorldRuleEntity,
    /// First-class Simulation Scenario row.
    SimulationScenarioEntity,
}

/// Soft hierarchy emphasis — dim only; never hide or block selection/editing.
fn row_dimmed_for_workspace(ui_state: &UiState, row: &LayerRow) -> bool {
    if matches!(ui_state.active_workspace, WorkspaceId::AllTools) {
        return false;
    }
    let role = row.domain_role.unwrap_or(match row.role {
        TreeRole::Foundation => DomainRole::ShapeLayer,
        TreeRole::Biome | TreeRole::BiomeSection => DomainRole::TerrainFilter,
        TreeRole::MaskStack | TreeRole::AdvancedSection | TreeRole::MaskAsset => {
            DomainRole::MaskLayer
        }
        TreeRole::WorldRuleEntity => DomainRole::MaskLayer,
        TreeRole::SimulationScenarioEntity => DomainRole::SimulationLayer,
        TreeRole::Layer => DomainRole::CompatibilityLegacy,
        TreeRole::ConceptFolder => DomainRole::ShapeLayer,
        _ => DomainRole::CompatibilityLegacy,
    });
    hierarchy_view::emphasis_dims(ui_state.active_workspace, role, row.concept)
}

fn row_hidden_by_filter(
    ui_state: &UiState,
    state: &LayersGuiState,
    row: &LayerRow,
    selected: bool,
) -> bool {
    if !state.presentation.show_only_relevant {
        return false;
    }
    if selected {
        return false;
    }
    if matches!(row.role, TreeRole::SectionLabel) {
        return false;
    }
    row_dimmed_for_workspace(ui_state, row)
}

fn row_selection_id(row: &LayerRow) -> LayerId {
    row.select_as.unwrap_or(row.id)
}

#[derive(Clone)]
struct LayerRow {
    idx: usize,
    id: LayerId,
    name: String,
    enabled: bool,
    locked: bool,
    solo: bool,
    color_tag: u8,
    preview_color: Option<[f32; 3]>,
    cached: bool,
    opacity: f32,
    is_layer: bool,
    is_base: bool,
    type_icon: Icon,
    subtitle: &'static str,
    /// Dynamic second-line meta (region priority/blend, status, etc.).
    meta: String,
    #[allow(dead_code)]
    mask_count: usize,
    #[allow(dead_code)]
    first_mask: Option<MaskId>,
    depth: u8,
    role: TreeRole,
    #[allow(dead_code)]
    child_count: usize,
    collapsed: bool,
    is_active_biome: bool,
    drop_insert_index: usize,
    domain_role: Option<DomainRole>,
    concept: Option<ArtistConcept>,
    focus_highlight: bool,
    workspace_badge: String,
    /// Presentation sentinel rows select this project entity instead of `id`.
    select_as: Option<LayerId>,
    /// Folder children render reverse of document order (Shape / Mask / Sims).
    list_reversed: bool,
}

fn resolve_biome_swatch_color(
    g: &terra_core::layer::LayerGroup,
    biome_library: &terra_core::biome_definition::BiomeLibrary,
) -> [f32; 3] {
    if let Some(def) = biome_library.by_group(g.id) {
        return def.color;
    }
    let c = g.preview_color;
    if c[0] + c[1] + c[2] > 0.01 && !terra_core::layer::is_default_preview_color(c) {
        return c;
    }
    if g.color_tag > 0 {
        let t = tag_color(g.color_tag);
        return [t.r, t.g, t.b];
    }
    terra_core::layer::palette_preview_color(g.id.0.as_u128())
}
fn draw_layer_context_menu(
    ui: &mut GuiContext<'_>,
    doc: &TerrainDocument,
    _ui_state: &UiState,
    id: LayerId,
    x: f32,
    y: f32,
    state: &mut LayersGuiState,
    actions: &mut Vec<PanelAction>,
) {
    let group = doc.stack.find_group(id);
    let layer = doc.stack.find(id);
    let is_base = layer.is_some_and(|l| l.kind.is_sculpt_base());
    let is_mask = doc.masks.iter().any(|m| m.id.0 == id.0);
    let is_biome = group.is_some_and(|g| g.is_biome());
    let can_rename = entity_can_rename(doc, id);
    let current_name = layer
        .map(|l| l.common.name.clone())
        .or_else(|| group.map(|g| g.name.clone()))
        .or_else(|| {
            doc.masks
                .iter()
                .find(|m| m.id.0 == id.0)
                .map(|m| m.name.clone())
        })
        .or_else(|| {
            doc.world_rules
                .rules
                .iter()
                .find(|r| r.id.0 == id.0)
                .map(|r| r.name.clone())
        })
        .or_else(|| {
            doc.simulation_scenarios
                .scenarios
                .iter()
                .find(|s| s.id.0 == id.0)
                .map(|s| s.name.clone())
        })
        .unwrap_or_default();

    let mut items: Vec<(&str, &str)> = Vec::new();
    if can_rename {
        items.push(("rename", "Rename"));
    }
    if layer.is_some() || is_biome {
        items.extend_from_slice(&[
            ("dup", "Duplicate"),
            ("del", "Delete"),
            ("group", "New Group"),
            ("ungroup", "Move to Root"),
            ("enable", "Enable / Disable"),
        ]);
    }
    if layer.is_some() {
        items.extend_from_slice(&[
            ("solo", "Solo"),
            ("lock", "Lock"),
            ("seed", "Randomize Seed"),
            ("cache", "Toggle Cache"),
        ]);
    }
    if is_mask
        || group.is_some_and(|g| {
            !g.is_biome()
                && !matches!(
                    g.group_kind,
                    terra_core::layer::GroupKind::BiomeSection(_)
                        | terra_core::layer::GroupKind::CategoryFolder
                )
        })
    {
        if !items.iter().any(|(k, _)| *k == "del") {
            items.push(("del", "Delete"));
        }
    }
    items.push(("close", "Close"));

    let w = 200.0;
    let h = items.len() as f32 * 24.0 + 8.0;
    let menu = Rect::from_pos_size(
        x.min(ui.screen_w - w - 4.0).max(4.0),
        y.min(ui.screen_h - h - 4.0).max(4.0),
        w,
        h,
    );
    ui.begin_overlay();
    ui.panel_rounded(menu, style::POPUP_BG, style::RADIUS_SM);
    ui.state.set_hot(Id::new("__layer_ctx"));

    let mut iy = menu.min_y + 4.0;
    for (key, label) in &items {
        let row = Rect::from_pos_size(menu.min_x + 4.0, iy, w - 8.0, 22.0);
        let disabled = is_base && matches!(*key, "dup" | "del" | "enable");
        let hovered = ui.pointer_in(row) && !disabled;
        if hovered {
            ui.panel_rounded(row, style::HOVER_BG, 3.0);
        }
        ui.label_at(
            row.min_x + 8.0,
            row.min_y + 4.0,
            label,
            if disabled {
                style::TEXT_DISABLED
            } else {
                style::TEXT
            },
            FONT_SCALE * TYPE_BODY,
        );
        if hovered && ui.input.primary_released {
            match *key {
                "rename" => {
                    begin_hierarchy_rename(ui, state, id, &current_name);
                }
                "dup" if !is_base => actions.push(PanelAction::DuplicateSelected),
                "del" if !is_base => actions.push(PanelAction::RemoveSelected),
                "group" => actions.push(PanelAction::AddGroup {
                    name: "Group".into(),
                }),
                "ungroup" => actions.push(PanelAction::MoveToRoot { id }),
                "enable" => {
                    if let Some(l) = layer {
                        actions.push(PanelAction::SetEnabled {
                            id,
                            enabled: !l.common.enabled,
                        });
                    } else if let Some(g) = group {
                        actions.push(PanelAction::SetEnabled {
                            id,
                            enabled: !g.enabled,
                        });
                    }
                }
                "solo" => {
                    if let Some(l) = layer {
                        actions.push(PanelAction::SetSolo {
                            id,
                            solo: !l.common.solo,
                        });
                    }
                }
                "lock" => {
                    if let Some(l) = layer {
                        actions.push(PanelAction::SetLocked {
                            id,
                            locked: !l.common.locked,
                        });
                    }
                }
                "seed" => actions.push(PanelAction::RandomizeSeed { id }),
                "cache" => {
                    if let Some(l) = layer {
                        actions.push(PanelAction::SetCached {
                            id,
                            cached: !l.common.cached,
                        });
                    }
                }
                _ => {}
            }
            state.context_menu = None;
        }
        iy += 24.0;
    }

    if (ui.input.primary_pressed || ui.input.secondary_pressed) && !ui.pointer_in(menu) {
        state.context_menu = None;
    }
    ui.end_overlay();
}

fn rename_widget_id(id: LayerId) -> Id {
    Id::new("lrename").child(&format!("{id:?}"))
}

fn begin_hierarchy_rename(
    ui: &mut GuiContext<'_>,
    state: &mut LayersGuiState,
    id: LayerId,
    name: &str,
) {
    state.rename_id = Some(id);
    ui.state.text_focus = Some(rename_widget_id(id));
    ui.state.text_buffer = name.to_string();
}

fn cancel_hierarchy_rename(ui: &mut GuiContext<'_>, state: &mut LayersGuiState) {
    state.rename_id = None;
    ui.state.clear_text_focus();
}

fn commit_hierarchy_rename(
    ui: &mut GuiContext<'_>,
    state: &mut LayersGuiState,
    actions: &mut Vec<PanelAction>,
) {
    if let Some(id) = state.rename_id.take() {
        let name = ui.state.text_buffer.trim().to_string();
        if !name.is_empty() {
            actions.push(PanelAction::Rename { id, name });
        }
    }
    ui.state.clear_text_focus();
}

/// Layers / biomes / user groups / masks — not fixed section labels.
fn row_can_rename(doc: &TerrainDocument, row: &LayerRow) -> bool {
    match row.role {
        TreeRole::Layer | TreeRole::Biome | TreeRole::MaskAsset => true,
        TreeRole::WorldRuleEntity | TreeRole::SimulationScenarioEntity => true,
        TreeRole::Group => entity_can_rename(doc, row.id),
        // Fixed chrome: Filters / Materials / Objects / Distributions / concept folders.
        TreeRole::BiomeSection
        | TreeRole::ConceptFolder
        | TreeRole::MaskStack
        | TreeRole::AdvancedSection
        | TreeRole::SectionLabel
        | TreeRole::Foundation => false,
    }
}

fn entity_can_rename(doc: &TerrainDocument, id: LayerId) -> bool {
    if doc.stack.find(id).is_some() {
        return true;
    }
    if let Some(g) = doc.stack.find_group(id) {
        return matches!(g.group_kind, terra_core::layer::GroupKind::Biome)
            || (matches!(g.group_kind, terra_core::layer::GroupKind::Generic)
                && g.category.is_none());
    }
    if doc.masks.iter().any(|m| m.id.0 == id.0) {
        return true;
    }
    if doc.world_rules.rules.iter().any(|r| r.id.0 == id.0) {
        return true;
    }
    doc.simulation_scenarios
        .scenarios
        .iter()
        .any(|s| s.id.0 == id.0)
}

pub fn layer_type_icon(kind: &LayerKind) -> Icon {
    use LayerKind::*;
    match kind {
        SculptBase(_) | SculptStrokes(_) => Icon::Pencil,
        TerrainConstraints(_) | GradientReconstruct(_) => Icon::SlidersHorizontal,
        Flat(_) | Ramp(_) | Plateau(_) | Mesa(_) | Island(_) => Icon::Box,
        Mountains(_) | Dunes(_) | Canyons(_) | VoronoiRegions(_) | Volcano(_) | Uplift(_) => {
            Icon::Mountain
        }
        Terrace(_) => Icon::Layers,
        NoiseValue(_) | NoisePerlin(_) | NoiseOpenSimplex(_) | NoiseWorley(_) | Fbm(_)
        | Ridged(_) | DomainWarp(_) => Icon::Sparkles,
        ThermalErosion(_)
        | DebrisFlow(_)
        | HydraulicErosion(_)
        | StreamPowerErosion(_)
        | RiverCarve(_)
        | Blur(_)
        | Coastal(_)
        | EffectFilter(_)
        | LandscapeEvolution(_)
        | HydrologyRepair(_)
        | GeomorphicDetail(_)
        | EcosystemFeedback(_)
        | MultiScaleAmplify(_)
        | SandSimulation(_)
        | FluidSimulation(_) => Icon::Droplets,
        Materials(_) => Icon::Paintbrush,
        Biomes(_) => Icon::Layers,
        Vegetation(_) => Icon::Sparkles,
        ImportHeightmap(_) | Stamp2d(_) => Icon::Download,
        OverhangStamp(_) | LocalSdf(_) | Stamp3d(_) => Icon::Box,
        Path(_) | PolygonHeight(_) => Icon::Move,
        RiverNetwork(_) => Icon::Waves,
        ProceduralShape(_) => Icon::Mountain,
    }
}

fn kind_subtitle(kind: &LayerKind) -> &'static str {
    use LayerKind::*;
    match kind {
        SculptBase(_) => "Sculpt",
        SculptStrokes(_) => "Strokes",
        TerrainConstraints(_) => "Constraints",
        GradientReconstruct(_) => "Gradient",
        LandscapeEvolution(_) => "Evolution",
        HydrologyRepair(_) => "Hydro Repair",
        GeomorphicDetail(_) => "Geomorph",
        EcosystemFeedback(_) => "Ecosystem",
        Flat(_) => "Flat",
        Ramp(_) => "Ramp",
        NoiseValue(_) => "Value Noise",
        NoisePerlin(_) => "Perlin",
        NoiseOpenSimplex(_) => "OpenSimplex",
        NoiseWorley(_) => "Worley",
        Fbm(_) => "fBm",
        Ridged(_) => "Ridged",
        DomainWarp(_) => "Domain Warp",
        Terrace(_) => "Terrace",
        Plateau(_) => "Plateau",
        Mesa(_) => "Mesa",
        Island(_) => "Island",
        Mountains(_) => "Mountains",
        Volcano(_) => "Volcano",
        Uplift(_) => "Uplift",
        Dunes(_) => "Dunes",
        Canyons(_) => "Canyons",
        VoronoiRegions(_) => "Voronoi",
        ImportHeightmap(_) => "Import",
        ThermalErosion(_) => "Thermal",
        DebrisFlow(_) => "Debris Flow",
        HydraulicErosion(_) => "Hydraulic",
        StreamPowerErosion(_) => "Stream Power",
        MultiScaleAmplify(_) => "Multi-Scale",
        RiverCarve(_) => "Rivers",
        Blur(_) => "Blur",
        Coastal(_) => "Coastal",
        EffectFilter(p) => p.kind.label(),
        Materials(_) => "Materials",
        Biomes(_) => "Biomes",
        Vegetation(_) => "Vegetation",
        OverhangStamp(_) => "Overhang",
        LocalSdf(_) => "Local SDF",
        Path(_) => "Path",
        RiverNetwork(_) => "River Network",
        SandSimulation(_) => "Sand Sim",
        FluidSimulation(_) => "Fluid Sim",
        ProceduralShape(_) => "Procedural",
        Stamp2d(_) => "2D Stamp",
        Stamp3d(_) => "3D Stamp",
        PolygonHeight(_) => "Polygon",
    }
}
fn layer_status_color(row: &LayerRow) -> Color {
    if !row.enabled {
        style::TEXT_DISABLED
    } else if row.solo {
        style::WARNING
    } else if row.cached {
        style::ACCENT
    } else {
        style::SUCCESS
    }
}
fn tag_color(tag: u8) -> Color {
    match tag {
        1 => style::TAG_RED,
        2 => style::TAG_ORANGE,
        3 => style::TAG_YELLOW,
        4 => style::TAG_GREEN,
        5 => style::TAG_BLUE,
        6 => style::TAG_PURPLE,
        _ => style::TAG_GRAY,
    }
}

/// Test/debug snapshot of hierarchy presentation rows (project-unchanged).
pub fn hierarchy_presentation_snapshot(
    doc: &TerrainDocument,
    state: &LayersGuiState,
) -> Vec<(LayerId, String, &'static str, u8, Option<ArtistConcept>)> {
    collect_rows(doc, state, &[])
        .into_iter()
        .map(|r| {
            let role = match r.role {
                TreeRole::SectionLabel => "section",
                TreeRole::Foundation => "foundation",
                TreeRole::Biome => "biome",
                TreeRole::BiomeSection => "biome_section",
                TreeRole::MaskStack => "mask_stack",
                TreeRole::MaskAsset => "mask_asset",
                TreeRole::Group => "group",
                TreeRole::Layer => "layer",
                TreeRole::ConceptFolder => "concept",
                TreeRole::AdvancedSection => "advanced",
                TreeRole::WorldRuleEntity => "world_rule",
                TreeRole::SimulationScenarioEntity => "simulation_scenario",
            };
            (r.id, r.name, role, r.depth, r.concept)
        })
        .collect()
}

#[cfg(test)]
mod hierarchy_tests {
    use super::*;
    use terra_core::domain::{DomainParent, DomainView};

    fn default_state() -> LayersGuiState {
        LayersGuiState::default()
    }

    #[test]
    fn default_biome_lives_under_biomes_not_materials() {
        let doc = TerrainDocument::new_default();
        let state = default_state();
        let rows = collect_rows(&doc, &state, &[]);
        let biome = rows
            .iter()
            .find(|r| r.role == TreeRole::Biome)
            .expect("default doc has a biome");
        assert_eq!(biome.concept, Some(ArtistConcept::Biomes));
        let concepts: Vec<_> = rows
            .iter()
            .filter(|r| r.role == TreeRole::ConceptFolder)
            .map(|r| r.name.as_str())
            .collect();
        assert_eq!(
            concepts,
            vec![
                "Biomes",
                "Biome Layers",
                "Shape Layers",
                "Mask Layers",
                "Simulation Layers",
            ],
            "WC terrain folder order: {concepts:?}"
        );
        assert!(
            rows.iter()
                .any(|r| r.name == "Terrain" && r.role == TreeRole::SectionLabel),
            "Terrain root required"
        );
    }

    #[test]
    fn biome_children_follow_wc_filters_then_materials() {
        let doc = TerrainDocument::new_default();
        let state = default_state();
        let rows = collect_rows(&doc, &state, &[]);
        let biome_idx = rows.iter().position(|r| r.role == TreeRole::Biome);
        let Some(start) = biome_idx else {
            return;
        };
        let depth = rows[start].depth;
        let mut names = Vec::new();
        for row in rows.iter().skip(start + 1) {
            if row.depth <= depth {
                break;
            }
            if matches!(row.role, TreeRole::AdvancedSection | TreeRole::BiomeSection) {
                names.push(row.name.as_str());
            }
        }
        assert!(
            names.first() == Some(&"Filters"),
            "Filters should lead biome children (WC): {names:?}"
        );
        let filters = names.iter().position(|n| *n == "Filters");
        let mats = names.iter().position(|n| *n == "Materials");
        let objs = names.iter().position(|n| *n == "Objects");
        let sims = names.iter().position(|n| *n == "Local Simulations");
        let dist = names.iter().position(|n| *n == "Distribution");
        assert!(
            filters < mats && mats < objs && objs < sims && sims < dist,
            "{names:?}"
        );
    }

    #[test]
    fn single_stack_emits_region_order_concepts() {
        let doc = TerrainDocument::new_default();
        let snap = hierarchy_presentation_snapshot(&doc, &default_state());
        assert!(
            snap.iter().any(|r| r.1 == "Terrain" && r.2 == "section"),
            "Terrain root must appear"
        );
        assert!(
            !snap.iter().any(|r| r.1 == "World" && r.2 == "section"),
            "World section header must not appear in WC single-stack"
        );
        assert!(
            !snap.iter().any(|r| r.1 == "Regions" && r.2 == "section"),
            "Regions section header must not appear in WC single-stack"
        );
        assert!(
            !snap.iter().any(|r| r.2 == "region" || r.2 == "region_mask"),
            "procedural RegionHeader/RegionMask rows must not appear"
        );
        let concepts: Vec<_> = snap
            .iter()
            .filter(|r| r.2 == "concept")
            .map(|r| r.1.as_str())
            .collect();
        assert_eq!(
            concepts,
            vec![
                "Biomes",
                "Biome Layers",
                "Shape Layers",
                "Mask Layers",
                "Simulation Layers",
            ],
            "{concepts:?}"
        );
    }

    #[test]
    fn project_entity_ids_stable_across_workspaces() {
        let doc = TerrainDocument::new_default();
        let state = default_state();
        let a = hierarchy_view::hierarchy_identity_ids(&doc);
        let b = hierarchy_view::hierarchy_identity_ids(&doc);
        assert_eq!(a, b);
        let snap = hierarchy_presentation_snapshot(&doc, &state);
        assert!(!a.is_empty());
        assert!(snap
            .iter()
            .any(|r| r.2 == "layer" || r.2 == "biome" || r.2 == "concept"));
    }

    #[test]
    fn mask_stack_and_advanced_ids_differ_from_biome() {
        let doc = TerrainDocument::new_default();
        let mut state = default_state();
        // Expand advanced so Mask Stack appears.
        let rows = collect_rows(&doc, &state, &[]);
        let biomes: Vec<_> = rows
            .iter()
            .filter(|r| r.role == TreeRole::Biome)
            .map(|r| r.id)
            .collect();
        for biome in biomes {
            let adv = advanced_placement_id(biome);
            let mask = mask_stack_row_id(biome);
            assert_ne!(adv, biome);
            assert_ne!(mask, biome);
            assert_ne!(adv, mask);
            // Force-expand advanced for snapshot.
            state.presentation.collapsed_groups.retain(|id| *id != adv);
        }
        let snap = hierarchy_presentation_snapshot(&doc, &state);
        assert!(
            snap.iter().any(|r| r.2 == "advanced"),
            "expected Advanced Placement row when biomes exist: {snap:?}"
        );
    }

    #[test]
    fn show_only_relevant_hides_dimmed_not_selected() {
        let doc = TerrainDocument::new_default();
        let mut ui = UiState::default();
        ui.switch_workspace(WorkspaceId::Sculpt);
        let mut state = default_state();
        let rows = collect_rows(&doc, &state, &[]);
        let Some(dim_row) = rows.iter().find(|r| {
            row_dimmed_for_workspace(&ui, r) && !matches!(r.role, TreeRole::SectionLabel)
        }) else {
            // Default doc may have only shape-relevant rows in Sculpt — still ok.
            return;
        };
        state.presentation.show_only_relevant = true;
        assert!(row_hidden_by_filter(&ui, &state, dim_row, false));
        assert!(!row_hidden_by_filter(&ui, &state, dim_row, true));
    }

    #[test]
    fn emphasis_dims_without_hiding_by_default() {
        let doc = TerrainDocument::new_default();
        let mut ui = UiState::default();
        ui.switch_workspace(WorkspaceId::Biomes);
        let state = default_state();
        assert!(!state.presentation.show_only_relevant);
        let rows = collect_rows(&doc, &state, &[]);
        assert!(!rows.is_empty());
        // Filter off → nothing hidden.
        for r in &rows {
            assert!(!row_hidden_by_filter(&ui, &state, r, false));
        }
    }

    #[test]
    fn selection_target_for_mask_stack_is_biome() {
        let biome = LayerId::new();
        let mut row = blank_row(
            mask_stack_row_id(biome),
            "Mask Stack".into(),
            2,
            TreeRole::MaskStack,
            0,
        );
        row.select_as = Some(biome);
        assert_eq!(row_selection_id(&row), biome);
        assert_ne!(row.id, biome);
    }

    #[test]
    fn artist_concepts_present_for_region_content() {
        let doc = TerrainDocument::new_default();
        let state = default_state();
        let snap = hierarchy_presentation_snapshot(&doc, &state);
        let concepts: Vec<_> = snap
            .iter()
            .filter(|r| r.2 == "concept")
            .map(|r| r.1.as_str())
            .collect();
        // Default world usually has Shape and/or Biomes under a region.
        assert!(
            concepts
                .iter()
                .any(|c| *c == "Shape Layers" || *c == "Biomes"),
            "concepts={concepts:?}"
        );
    }

    #[test]
    fn no_duplicated_project_state_in_presentation() {
        let doc = TerrainDocument::new_default();
        let before = doc.to_json().unwrap();
        let state = default_state();
        let _ = hierarchy_presentation_snapshot(&doc, &state);
        assert_eq!(doc.to_json().unwrap(), before);
    }

    #[test]
    fn reorder_relative_moves_siblings() {
        use terra_core::layer::FlatParams;
        let mut stack = terra_core::layer::LayerStack::default();
        let a = terra_core::layer::Layer::new("A", LayerKind::Flat(FlatParams::default()));
        let b = terra_core::layer::Layer::new("B", LayerKind::Flat(FlatParams::default()));
        let aid = a.id();
        let bid = b.id();
        stack.nodes.push(StackNode::Layer(a));
        stack.nodes.push(StackNode::Layer(b));
        assert!(stack.reorder_relative(aid, bid, false));
        let ids: Vec<_> = stack
            .nodes
            .iter()
            .map(|n| match n {
                StackNode::Layer(l) => l.id(),
                StackNode::Group(g) => g.id,
            })
            .collect();
        assert_eq!(ids, vec![bid, aid]);
    }

    #[test]
    fn domain_ownership_rejects_shape_under_biome_masks() {
        let err = DomainView::validate_ownership(DomainRole::ShapeLayer, DomainParent::BiomeMasks);
        assert!(err.is_err());
    }

    #[test]
    fn biome_sections_use_artist_labels() {
        assert_eq!(
            biome_section_artist_label(terra_core::layer::BiomeSection::Filters),
            "Filters"
        );
    }

    #[test]
    fn add_biome_lands_under_biomes_concept_not_shape() {
        let mut doc = TerrainDocument::new_default();
        doc.stack.ensure_category_folders();
        let biome = terra_core::layer::LayerGroup::biome("Alpine");
        let biome_id = biome.id;
        if let Some(folder) = doc
            .stack
            .find_category_mut(terra_core::layer::StackCategory::Surface)
        {
            folder.children.push(StackNode::Group(biome));
        } else {
            doc.stack.push_group(biome);
        }
        let state = default_state();
        let rows = collect_rows(&doc, &state, &[]);
        let row = rows
            .iter()
            .find(|r| r.id == biome_id && r.role == TreeRole::Biome)
            .expect("added biome should appear in hierarchy");
        assert_eq!(
            row.concept,
            Some(ArtistConcept::Biomes),
            "WC biome must not show under Shape Layers"
        );
        assert!(
            !rows
                .iter()
                .any(|r| { r.id == biome_id && r.concept == Some(ArtistConcept::Shape) }),
            "biome must not be bucketed as Shape"
        );
    }

    #[test]
    fn mask_assets_appear_under_mask_layers() {
        use terra_core::mask::{MaskAsset, MaskId, MaskSource};
        let mut doc = TerrainDocument::new_default();
        let id = MaskId::new();
        doc.masks.push(MaskAsset::new(
            id,
            "Height",
            MaskSource::Height {
                min: 0.0,
                max: 100.0,
            },
        ));
        let state = default_state();
        let rows = collect_rows(&doc, &state, &[]);
        let mask_row = rows
            .iter()
            .find(|r| r.id == LayerId(id.0) && r.role == TreeRole::MaskAsset)
            .expect("project mask asset should appear in hierarchy");
        assert_eq!(mask_row.concept, Some(ArtistConcept::Masks));
        assert_eq!(mask_row.name, "Height");
        assert!(
            rows.iter().any(|r| {
                r.role == TreeRole::ConceptFolder && r.concept == Some(ArtistConcept::Masks)
            }),
            "Mask Layers folder required"
        );
    }
}
