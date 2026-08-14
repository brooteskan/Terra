//! Searchable Quick Add modal for the Layers `+` button.
//!
//! Layout mirrors a browser-style picker: title + search, tile grid, detail
//! sidebar, and Confirm/Cancel footer. Tool art comes from the tools panel
//! (`tool_thumbs` / `ToolDef.icon`) — not bespoke reference illustrations.

use crate::ui::actions::PanelAction;
use crate::ui::command_registry::fuzzy_match;
use crate::ui::dist_kinds::{
    dist_base_kinds, dist_effect_kinds, dist_kind_description, dist_kind_icon,
};
use crate::ui::hierarchy_view::ArtistConcept;
use crate::ui::style::{
    self, FONT_SCALE, GAP, PAD, PAD_SM, ROW_H, TYPE_BODY, TYPE_CAPTION, TYPE_LABEL, TYPE_TITLE,
};
use crate::ui::tool_catalog::{
    all_tools_cached, instantiate_layer_preset, quick_add_entries, ToolAction, ToolDef,
};
use crate::ui::UiState;
use terra_core::document::TerrainDocument;
use terra_core::layer::{biome_destination_section, is_shape_kind, BiomeSection, StackCategory};
use terra_core::mask::{DistNodeKind, MaskAsset, MaskId};
use terra_gui::{Color, DrawList, GuiContext, Icon, Id, Rect};

const GRID_COLS: usize = 3;
const TILE_H: f32 = 72.0;
const TILE_GAP: f32 = 8.0;
const SIDEBAR_W: f32 = 260.0;
const FOOTER_H: f32 = 52.0;
const HEADER_BLOCK: f32 = 96.0;
const THUMB_SZ: f32 = 48.0;
const BORDER: f32 = 2.0;

#[derive(Debug, Default)]
pub struct QuickAddState {
    pub query: String,
    pub scroll_y: f32,
    /// Selected catalog / org entry id (confirm adds this).
    pub selected_id: Option<String>,
    /// When true, only show tools that appear in recent history.
    pub recent_only: bool,
    pub filter_menu_open: bool,
}

fn clear_quick_add(ui_state: &mut UiState, state: &mut QuickAddState) {
    ui_state.show_quick_add = false;
    ui_state.quick_add_category = None;
    ui_state.quick_add_concept = None;
    ui_state.quick_add_into = None;
    ui_state.quick_add_biome_section = None;
    ui_state.quick_add_distribution = None;

    state.query.clear();
    state.scroll_y = 0.0;
    state.selected_id = None;
    state.recent_only = false;
    state.filter_menu_open = false;
}

fn modal_title(ui_state: &UiState) -> &'static str {
    if ui_state.quick_add_distribution.is_some() {
        return "Add Distribution";
    }
    if let Some(section) = ui_state.quick_add_biome_section {
        return match section {
            BiomeSection::Filters => "Add Filter",
            BiomeSection::Materials => "Add Material",
            BiomeSection::Objects => "Add Object",
            BiomeSection::LocalSims => "Add Local Simulation",
        };
    }
    if ui_state.quick_add_into.is_some() {
        return "Add to Biome";
    }
    if let Some(concept) = ui_state.quick_add_concept {
        return concept.quick_add_title();
    }
    match ui_state.quick_add_category {
        Some(StackCategory::Shape) => "Add Shape Layer",
        Some(StackCategory::Surface) => "Add Biome",
        Some(StackCategory::Mask) => "Add Mask",
        Some(StackCategory::Simulation) => "Add Simulation",
        Some(StackCategory::Foundation) => "Add Foundation",
        None => "Quick Add",
    }
}

fn modal_subtitle(ui_state: &UiState) -> &'static str {
    if ui_state.quick_add_distribution.is_some() {
        return "Choose a generator or modifier for this biome’s distribution stack.";
    }
    if let Some(section) = ui_state.quick_add_biome_section {
        return match section {
            BiomeSection::Filters => "Choose a filter or landform effect for this biome.",
            BiomeSection::Materials => "Choose a material or surface dressing for this biome.",
            BiomeSection::Objects => "Choose vegetation or object scatter for this biome.",
            BiomeSection::LocalSims => "Choose a local sand / fluid simulation for this biome.",
        };
    }
    if ui_state.quick_add_into.is_some() {
        return "Choose content to place inside this biome’s sections.";
    }
    if let Some(concept) = ui_state.quick_add_concept {
        return match concept {
            ArtistConcept::Shape => "Choose a shape or generator for this terrain stack.",
            ArtistConcept::Biomes => "Create a biome container or add surface content.",
            ArtistConcept::BiomeLayers => "Choose a biome paint brush or weight layer.",
            ArtistConcept::Masks | ArtistConcept::MaskStack | ArtistConcept::WorldRules => {
                "Choose a mask to define where this layer will be applied."
            }
            ArtistConcept::WorldSimulations | ArtistConcept::LocalSimulations => {
                "Choose a simulation or erosion process."
            }
            ArtistConcept::GlobalMaterials => "Choose a material or surface dressing layer.",
            ArtistConcept::Objects | ArtistConcept::GlobalScatter => {
                "Choose an object or scatter layer."
            }
            ArtistConcept::AdvancedPlacement => "Choose a distribution or placement mask.",
        };
    }
    match ui_state.quick_add_category {
        Some(StackCategory::Mask) => "Choose a mask to define where this layer will be applied.",
        Some(StackCategory::Shape) | Some(StackCategory::Foundation) => {
            "Choose a shape layer for this terrain stack."
        }
        Some(StackCategory::Surface) => "Create a biome container or add surface content.",
        Some(StackCategory::Simulation) => "Choose a simulation or erosion process.",
        None => "Search and add layers, masks, biomes, and groups.",
    }
}

fn modal_tip(ui_state: &UiState) -> &'static str {
    if let Some(section) = ui_state.quick_add_biome_section {
        return match section {
            BiomeSection::Filters => {
                "Tip: Filters shape biome character; they stay scoped to this biome."
            }
            BiomeSection::Materials => {
                "Tip: Materials dress this biome’s surface and can drive hardness."
            }
            BiomeSection::Objects => {
                "Tip: Objects and vegetation scatter only where this biome owns."
            }
            BiomeSection::LocalSims => "Tip: Local sims (sand / fluid) run inside this biome only.",
        };
    }
    if ui_state.quick_add_into.is_some() {
        return "Tip: New layers land in the matching Filters / Materials / Objects / Local Sims section.";
    }
    if matches!(
        ui_state.quick_add_concept,
        Some(ArtistConcept::Masks | ArtistConcept::MaskStack | ArtistConcept::WorldRules)
    ) || ui_state.quick_add_category == Some(StackCategory::Mask)
    {
        "Tip: Masks can be combined and refined in the mask stack after adding."
    } else if matches!(ui_state.quick_add_concept, Some(ArtistConcept::Biomes))
        || ui_state.quick_add_category == Some(StackCategory::Surface)
    {
        "Tip: Biomes hold Filters, Materials, and Objects for local surface character."
    } else if matches!(ui_state.quick_add_concept, Some(ArtistConcept::Shape))
        || matches!(
            ui_state.quick_add_category,
            Some(StackCategory::Shape | StackCategory::Foundation)
        )
    {
        "Tip: Shape layers build the heightfield; bind a mask to limit their reach."
    } else {
        "Tip: Use search or Recent to jump to tools you use often."
    }
}

fn confirm_label(ui_state: &UiState) -> &'static str {
    modal_title(ui_state)
}

fn search_placeholder(ui_state: &UiState) -> &'static str {
    if ui_state.quick_add_distribution.is_some() {
        return "Search distributions (e.g. height, slope, perlin...)";
    }
    if let Some(section) = ui_state.quick_add_biome_section {
        return match section {
            BiomeSection::Filters => "Search filters (e.g. rocky, terrace, warp...)",
            BiomeSection::Materials => "Search materials (e.g. colour, wetness, rock...)",
            BiomeSection::Objects => "Search objects (e.g. trees, grass, rocks...)",
            BiomeSection::LocalSims => "Search local sims (e.g. sand, wind, fluid...)",
        };
    }
    if matches!(
        ui_state.quick_add_concept,
        Some(ArtistConcept::Masks | ArtistConcept::MaskStack)
    ) || ui_state.quick_add_category == Some(StackCategory::Mask)
    {
        "Search masks (e.g. height, slope, noise...)"
    } else {
        "Search tools..."
    }
}

/// WC Shape Layers: `shape.*` catalog entries, or kinds that route to the Shape folder.
///
/// Do **not** use `OperationCategory::Generator/Modifier` / `StackCategory::from_operation`
/// alone — Modifier-class filters (Terrace, EffectFilter, …) also map to Shape there.
fn tool_is_shape_layer(tool: &ToolDef) -> bool {
    match &tool.action {
        ToolAction::AddLayer { kind, .. } => tool.id.starts_with("shape.") || is_shape_kind(kind),
        _ => false,
    }
}

fn tool_matches_concept(tool: &ToolDef, concept: ArtistConcept) -> bool {
    match concept {
        ArtistConcept::Shape => tool_is_shape_layer(tool),
        ArtistConcept::Biomes => {
            matches!(&tool.action, ToolAction::CreateBiome { .. })
                || tool.id == "biome.create"
                || matches!(
                    &tool.action,
                    ToolAction::AddLayer { kind, .. }
                        if StackCategory::from_operation(kind.category()) == StackCategory::Surface
                )
        }
        ArtistConcept::BiomeLayers => {
            matches!(&tool.action, ToolAction::BiomeBrush(_))
                || (tool.id.starts_with("biome.")
                    && !matches!(
                        &tool.action,
                        ToolAction::AddLayer { .. } | ToolAction::CreateBiome { .. }
                    ))
        }
        ArtistConcept::Masks | ArtistConcept::MaskStack | ArtistConcept::WorldRules => {
            matches!(&tool.action, ToolAction::AddMask { .. }) || tool.id.starts_with("mask.")
        }
        ArtistConcept::WorldSimulations | ArtistConcept::LocalSimulations => matches!(
            &tool.action,
            ToolAction::AddLayer { kind, .. }
                if StackCategory::from_operation(kind.category()) == StackCategory::Simulation
        ),
        ArtistConcept::GlobalMaterials => {
            tool.id.starts_with("mat.")
                || matches!(
                    &tool.action,
                    ToolAction::AddLayer { kind, .. }
                        if kind.type_id().starts_with("mat.")
                            || kind.type_id().contains("material")
                )
        }
        ArtistConcept::Objects | ArtistConcept::GlobalScatter => {
            tool.id.starts_with("obj.")
                || matches!(
                    &tool.action,
                    ToolAction::AddLayer { kind, .. } if kind.type_id().starts_with("obj.")
                )
        }
        ArtistConcept::AdvancedPlacement => {
            tool.id.starts_with("mask.") || matches!(&tool.action, ToolAction::AddMask { .. })
        }
    }
}

fn tool_matches_category(tool: &ToolDef, cat: StackCategory) -> bool {
    match cat {
        StackCategory::Mask => {
            matches!(&tool.action, ToolAction::AddMask { .. }) || tool.id.starts_with("mask.")
        }
        StackCategory::Surface => {
            matches!(&tool.action, ToolAction::CreateBiome { .. })
                || tool.id == "biome.create"
                || matches!(
                    &tool.action,
                    ToolAction::AddLayer { kind, .. }
                        if StackCategory::from_operation(kind.category()) == StackCategory::Surface
                )
        }
        StackCategory::Shape | StackCategory::Foundation => tool_is_shape_layer(tool),
        StackCategory::Simulation => matches!(
            &tool.action,
            ToolAction::AddLayer { kind, .. }
                if StackCategory::from_operation(kind.category()) == StackCategory::Simulation
        ),
    }
}

fn tool_matches_biome_section(tool: &ToolDef, section: BiomeSection) -> bool {
    match &tool.action {
        ToolAction::AddLayer { kind, .. } => biome_destination_section(kind) == Some(section),
        _ => false,
    }
}

fn tool_matches_biome_root(tool: &ToolDef) -> bool {
    match &tool.action {
        ToolAction::AddLayer { kind, .. } => biome_destination_section(kind).is_some(),
        _ => false,
    }
}

fn catalog_for_modal(ui_state: &UiState) -> Vec<ToolDef> {
    // Explicit concept (e.g. Mask Layers → Height/Slope/Painted) wins over into-target
    // filtering. `quick_add_into` alone is for nesting into a biome / section.
    if let Some(concept) = ui_state.quick_add_concept {
        return all_tools_cached()
            .iter()
            .filter(|t| tool_matches_concept(t, concept))
            .cloned()
            .collect();
    }
    if ui_state.quick_add_into.is_some() {
        return all_tools_cached()
            .iter()
            .filter(|t| {
                if let Some(section) = ui_state.quick_add_biome_section {
                    tool_matches_biome_section(t, section)
                } else {
                    tool_matches_biome_root(t)
                }
            })
            .cloned()
            .collect();
    }
    if let Some(cat) = ui_state.quick_add_category {
        return all_tools_cached()
            .iter()
            .filter(|t| tool_matches_category(t, cat))
            .cloned()
            .collect();
    }
    quick_add_entries()
}

#[derive(Clone)]
enum PickerItem {
    Tool(ToolDef),
    Org {
        id: &'static str,
        label: &'static str,
        description: &'static str,
        icon: Icon,
    },
    /// DistNode generator or modifier for biome Distribution Quick Add.
    Dist {
        id: String,
        label: &'static str,
        description: &'static str,
        icon: Icon,
        kind: DistNodeKind,
        is_effect: bool,
    },
}

impl PickerItem {
    fn id(&self) -> &str {
        match self {
            Self::Tool(t) => t.id,
            Self::Org { id, .. } => id,
            Self::Dist { id, .. } => id.as_str(),
        }
    }

    fn label(&self) -> &str {
        match self {
            Self::Tool(t) => t.label,
            Self::Org { label, .. } => label,
            Self::Dist { label, .. } => label,
        }
    }

    fn description(&self) -> &str {
        match self {
            Self::Tool(t) => t.description,
            Self::Org { description, .. } => description,
            Self::Dist { description, .. } => description,
        }
    }

    fn icon(&self) -> Icon {
        match self {
            Self::Tool(t) => t.icon,
            Self::Org { icon, .. } => *icon,
            Self::Dist { icon, .. } => *icon,
        }
    }

    fn short_description(&self) -> String {
        let d = self.description();
        // One short line for the tile.
        let first = d.split('.').next().unwrap_or(d).trim();
        if first.is_empty() {
            d.to_string()
        } else if first.len() > 48 {
            format!("{}…", &first[..45])
        } else if d.contains('.') {
            format!("{first}.")
        } else {
            first.to_string()
        }
    }
}

fn org_items(ui_state: &UiState) -> Vec<PickerItem> {
    // Organisation shortcuts only for the unfiltered / generic Quick Add —
    // not when targeting a biome, biome section, distribution, or concept folder.
    let show_org = ui_state.quick_add_concept.is_none()
        && ui_state.quick_add_category.is_none()
        && ui_state.quick_add_into.is_none()
        && ui_state.quick_add_biome_section.is_none()
        && ui_state.quick_add_distribution.is_none();
    let mut items = Vec::new();
    if show_org {
        items.push(PickerItem::Org {
            id: "org.pass",
            label: "Pass-through Group",
            description: "Organise layers; children modify the live context.",
            icon: Icon::Folder,
        });
        items.push(PickerItem::Org {
            id: "org.isolated",
            label: "Isolated Terrain Group",
            description: "Private recipe composited back with a group mask.",
            icon: Icon::FolderOpen,
        });
        items.push(PickerItem::Org {
            id: "org.biome",
            label: "Biome",
            description: "Biome container with Filters, Materials, and Objects.",
            icon: Icon::Sparkles,
        });
        items.push(PickerItem::Org {
            id: "org.hole",
            label: "Hole Layer",
            description: "Cut a hole through the heightfield.",
            icon: Icon::CircleDot,
        });
    } else if matches!(ui_state.quick_add_concept, Some(ArtistConcept::Biomes))
        && ui_state.quick_add_into.is_none()
    {
        items.push(PickerItem::Org {
            id: "org.biome",
            label: "Biome",
            description: "New biome container under Biomes.",
            icon: Icon::Sparkles,
        });
    } else if matches!(ui_state.quick_add_concept, Some(ArtistConcept::BiomeLayers)) {
        items.push(PickerItem::Org {
            id: "org.biome_paint",
            label: "Biome Paint Layer",
            description: "Weight / ownership splat for biome placement.",
            icon: Icon::Paintbrush,
        });
    }
    items
}

fn distribution_items() -> Vec<PickerItem> {
    let mut items = Vec::new();
    for (i, (label, kind)) in dist_base_kinds().into_iter().enumerate() {
        items.push(PickerItem::Dist {
            id: format!("dist.gen.{i}"),
            label,
            description: dist_kind_description(label, false),
            icon: dist_kind_icon(&kind),
            kind,
            is_effect: false,
        });
    }
    for (i, (label, kind)) in dist_effect_kinds().into_iter().enumerate() {
        items.push(PickerItem::Dist {
            id: format!("dist.fx.{i}"),
            label,
            description: dist_kind_description(label, true),
            icon: dist_kind_icon(&kind),
            kind,
            is_effect: true,
        });
    }
    items
}

fn collect_items(
    doc: &TerrainDocument,
    ui_state: &UiState,
    state: &QuickAddState,
) -> Vec<PickerItem> {
    if ui_state.quick_add_distribution.is_some() {
        let mut items: Vec<PickerItem> = distribution_items()
            .into_iter()
            .filter(|item| {
                fuzzy_match(&state.query, item.label())
                    || fuzzy_match(&state.query, item.description())
            })
            .collect();
        if state.recent_only {
            items.retain(|item| ui_state.recent_tools.iter().any(|id| id == item.id()));
        }
        items.sort_by_key(|item| {
            let recent_rank = ui_state
                .recent_tools
                .iter()
                .position(|id| id == item.id())
                .unwrap_or(usize::MAX);
            let effect_rank = match item {
                PickerItem::Dist {
                    is_effect: true, ..
                } => 1u8,
                _ => 0u8,
            };
            (effect_rank, recent_rank, item.label().to_ascii_lowercase())
        });
        return items;
    }

    let suggested_ids = contextual_suggestion_type_ids(doc, ui_state);
    let tools = catalog_for_modal(ui_state);
    let mut items: Vec<PickerItem> = org_items(ui_state)
        .into_iter()
        .chain(tools.into_iter().map(PickerItem::Tool))
        .filter(|item| {
            fuzzy_match(&state.query, item.label()) || fuzzy_match(&state.query, item.description())
        })
        .collect();

    if state.recent_only {
        items.retain(|item| ui_state.recent_tools.iter().any(|id| id == item.id()));
    }

    items.sort_by_key(|item| {
        let suggest_rank = match item {
            PickerItem::Tool(tool) => match &tool.action {
                ToolAction::AddLayer { kind, .. } => {
                    if suggested_ids.iter().any(|id| id == kind.type_id()) {
                        0usize
                    } else {
                        usize::MAX
                    }
                }
                _ => usize::MAX,
            },
            PickerItem::Org { .. } => usize::MAX / 2,
            PickerItem::Dist { .. } => usize::MAX,
        };
        let recent_rank = ui_state
            .recent_tools
            .iter()
            .position(|id| id == item.id())
            .unwrap_or(usize::MAX);
        (suggest_rank, recent_rank, item.label().to_ascii_lowercase())
    });
    items
}

fn ensure_selection(state: &mut QuickAddState, items: &[PickerItem]) {
    if items.is_empty() {
        state.selected_id = None;
        return;
    }
    let still_valid = state
        .selected_id
        .as_ref()
        .is_some_and(|id| items.iter().any(|i| i.id() == id));
    if !still_valid {
        state.selected_id = Some(items[0].id().to_string());
    }
}

fn inputs_label(item: &PickerItem) -> &'static str {
    match item {
        PickerItem::Tool(t) => match &t.action {
            ToolAction::AddMask { source, .. } => match source {
                terra_core::mask::MaskSource::None | terra_core::mask::MaskSource::Constant(_) => {
                    "None"
                }
                terra_core::mask::MaskSource::Height { .. } => "Terrain height",
                terra_core::mask::MaskSource::Slope { .. } => "Terrain slope",
                terra_core::mask::MaskSource::Curvature { .. }
                | terra_core::mask::MaskSource::Convexity
                | terra_core::mask::MaskSource::Concavity => "Terrain curvature",
                terra_core::mask::MaskSource::FlowAccumulation { .. }
                | terra_core::mask::MaskSource::FlowDirection
                | terra_core::mask::MaskSource::Wetness => "Flow / drainage",
                terra_core::mask::MaskSource::Noise { .. } => "Procedural noise",
                terra_core::mask::MaskSource::Painted { .. } => "Painted strokes",
                terra_core::mask::MaskSource::DistanceField { .. } => "Distance field",
                _ => "Terrain fields",
            },
            ToolAction::AddLayer { .. } | ToolAction::CreateBiome { .. } => "Layer stack",
            ToolAction::BiomeBrush(_) => "Active biome",
            _ => "None",
        },
        PickerItem::Org { id, .. } if *id == "org.biome_paint" => "Active biome",
        PickerItem::Org { .. } => "None",
        PickerItem::Dist {
            is_effect: true, ..
        } => "Parent distribution",
        PickerItem::Dist { .. } => "Terrain fields",
    }
}

fn output_label(item: &PickerItem) -> &'static str {
    match item {
        PickerItem::Tool(t) => match &t.action {
            ToolAction::AddMask { .. } => "Grayscale mask",
            ToolAction::CreateBiome { .. } => "Biome container",
            ToolAction::BiomeBrush(_) => "Biome ownership",
            ToolAction::AddLayer { kind, .. } => {
                if kind.type_id().starts_with("mat.") || kind.type_id().contains("material") {
                    "Surface materials"
                } else if kind.type_id().starts_with("obj.") || kind.type_id().contains("veg") {
                    "Scatter / objects"
                } else if matches!(
                    StackCategory::from_operation(kind.category()),
                    StackCategory::Simulation
                ) {
                    "Simulation result"
                } else if biome_destination_section(kind) == Some(BiomeSection::Filters) {
                    "Biome filter"
                } else {
                    "Heightfield"
                }
            }
            _ => "Layer",
        },
        PickerItem::Org { id, .. } => match *id {
            "org.biome" => "Biome container",
            "org.biome_paint" => "Biome weight layer",
            "org.hole" => "Hole mask",
            _ => "Group",
        },
        PickerItem::Dist {
            is_effect: true, ..
        } => "Refined distribution",
        PickerItem::Dist { .. } => "Distribution mask",
    }
}

fn common_uses(item: &PickerItem) -> &'static [&'static str] {
    match item.id() {
        "mask.height" => &[
            "Limit snow or sand by altitude",
            "Carve plateaus and benches",
            "Hide lowlands or peaks",
        ],
        "mask.slope" => &[
            "Keep scree on steep faces",
            "Place paths on gentle grades",
            "Mask cliffs vs flats",
        ],
        "mask.curvature" | "mask.convexity" | "mask.concavity" => &[
            "Accent ridges and hollows",
            "Drive weathering detail",
            "Guide material breakup",
        ],
        "mask.flow" => &[
            "Follow drainage corridors",
            "Seed rivers and wetness",
            "Erode along flow paths",
        ],
        "mask.noise" | "mask.distance" | "mask.painted" | "mask.combined" => &[
            "Break up hard edges",
            "Author custom influence",
            "Blend multiple constraints",
        ],
        "biome.create" | "org.biome" => &[
            "Local Filters / Materials / Objects",
            "Paint ownership later",
            "Scope surface character",
        ],
        id if id.starts_with("mat.") => &[
            "Dress biome surfaces",
            "Drive hardness for erosion",
            "Colour and wetness cues",
        ],
        id if id.starts_with("sim.") => &[
            "Evolve large landforms",
            "Carve drainage naturally",
            "Refresh after sculpt edits",
        ],
        _ => &[
            "Apply where this effect belongs",
            "Combine with masks for control",
            "Iterate from the inspector",
        ],
    }
}

fn draw_selection_border(ui: &mut GuiContext<'_>, rect: Rect) {
    ui.panel_rounded(rect, style::ACCENT, style::RADIUS_MD);
}

fn draw_tool_thumb(ui: &mut GuiContext<'_>, rect: Rect, item: &PickerItem, selected: bool) {
    ui.panel_rounded(rect, style::RAISED_BG, style::RADIUS_SM);
    let icon_color = if selected {
        style::ACCENT
    } else {
        style::TEXT_DIM
    };
    if let Some(thumb) = crate::ui::tool_thumbs::thumb_for_tool(item.id()) {
        ui.image(rect, thumb.width, thumb.height, &thumb.rgba);
        return;
    }
    ui.icon_centered(rect, item.icon(), icon_color, style::ICON_LG);
}

fn commit_item(
    item: &PickerItem,
    ui_state: &mut UiState,
    state: &mut QuickAddState,
    actions: &mut Vec<PanelAction>,
) {
    match item {
        PickerItem::Tool(tool) => match &tool.action {
            ToolAction::CreateBiome { name } => {
                actions.push(PanelAction::AddBiome {
                    name: (*name).into(),
                });
            }
            ToolAction::AddLayer { name, kind } => {
                let layer = instantiate_layer_preset(name, kind);
                let concept = ui_state.quick_add_concept;
                if let Some(parent) = ui_state.quick_add_into {
                    actions.push(PanelAction::AddLayerInto { parent, layer });
                } else if let Some(cat) = ui_state
                    .quick_add_category
                    .or_else(|| concept.and_then(ArtistConcept::stack_category))
                {
                    if matches!(cat, StackCategory::Surface) {
                        actions.push(PanelAction::AddLayer(layer));
                    } else {
                        actions.push(PanelAction::AddLayerToCategory {
                            category: cat,
                            layer,
                        });
                    }
                } else {
                    actions.push(PanelAction::AddLayer(layer));
                }
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
            }
            ToolAction::BiomeBrush(brush) => {
                actions.push(PanelAction::SetBiomePaintTool(*brush));
                actions.push(PanelAction::EnsureBiomePaintLayer);
            }
            ToolAction::Sculpt(_) | ToolAction::BakeSelected => {}
        },
        PickerItem::Org { id, .. } => match *id {
            "org.pass" => actions.push(PanelAction::AddGroup {
                name: "Group".into(),
            }),
            "org.isolated" => actions.push(PanelAction::AddIsolatedGroup {
                name: "Terrain Group".into(),
            }),
            "org.biome" => actions.push(PanelAction::AddBiome {
                name: "Biome".into(),
            }),
            "org.hole" => actions.push(PanelAction::AddHoleLayer {
                name: "Hole".into(),
            }),
            "org.biome_paint" => actions.push(PanelAction::AddBiomePaintLayer {
                name: "Biome Paint".into(),
            }),
            _ => {}
        },
        PickerItem::Dist {
            kind, is_effect, ..
        } => {
            if let Some(biome) = ui_state.quick_add_distribution {
                if *is_effect {
                    actions.push(PanelAction::AddDistEffect {
                        target: biome,
                        kind: kind.clone(),
                    });
                } else {
                    actions.push(PanelAction::AddDistNode {
                        target: biome,
                        kind: kind.clone(),
                    });
                }
            }
        }
    }
    remember_tool(&mut ui_state.recent_tools, item.id());
    clear_quick_add(ui_state, state);
}

pub fn draw_quick_add(
    ui: &mut GuiContext<'_>,
    doc: &TerrainDocument,
    ui_state: &mut UiState,
    state: &mut QuickAddState,
) -> Vec<PanelAction> {
    if !ui_state.show_quick_add {
        return Vec::new();
    }
    apply_search_input(ui, &mut state.query);
    if ui.input.escape_pressed {
        clear_quick_add(ui_state, state);
        return Vec::new();
    }

    let items = collect_items(doc, ui_state, state);
    ensure_selection(state, &items);

    let vp = ui.viewport_rect();
    let popup_w = 920.0_f32.min(vp.width() - 48.0).max(640.0);
    let popup_h = 620.0_f32.min(vp.height() - 64.0).max(420.0);
    let popup = Rect::from_pos_size(
        vp.min_x + (vp.width() - popup_w) * 0.5,
        vp.min_y + (vp.height() - popup_h) * 0.5,
        popup_w,
        popup_h,
    );

    let mut actions = Vec::new();
    ui.begin_overlay();

    // Dim the editor behind the modal.
    ui.panel(
        Rect::from_min_max(0.0, 0.0, ui.screen_w, ui.screen_h),
        Color::rgba(0.0, 0.0, 0.0, 0.45),
    );
    ui.panel_rounded(popup, style::POPUP_BG, style::RADIUS_LG);
    ui.state.set_hot(Id::new("__quick_add_modal"));

    // —— Header ——————————————————————————————————————————————————————
    let title = modal_title(ui_state);
    ui.label_at(
        popup.min_x + PAD,
        popup.min_y + PAD,
        title,
        style::TEXT,
        FONT_SCALE * 1.35,
    );
    ui.label_at(
        popup.min_x + PAD,
        popup.min_y + PAD + 26.0,
        modal_subtitle(ui_state),
        style::TEXT_MUTED,
        FONT_SCALE * TYPE_LABEL,
    );

    let close_r = Rect::from_pos_size(popup.max_x - PAD - 28.0, popup.min_y + PAD, 28.0, 28.0);
    if icon_hit(ui, Id::new("qa_close"), Icon::X, close_r) {
        clear_quick_add(ui_state, state);
        ui.end_overlay();
        return actions;
    }

    // Search + Recent filter.
    let search_y = popup.min_y + 58.0;
    let filter_w = 110.0;
    let search = Rect::from_pos_size(
        popup.min_x + PAD,
        search_y,
        popup.width() - PAD * 2.0 - filter_w - GAP,
        ROW_H + 4.0,
    );
    let filter_r = Rect::from_pos_size(search.max_x + GAP, search_y, filter_w, search.height());
    ui.panel_rounded(search, style::INPUT_BG, style::RADIUS_MD);
    ui.icon_at(
        search.min_x + 10.0,
        search.min_y + (search.height() - 14.0) * 0.5,
        Icon::Search,
        style::TEXT_MUTED,
        14.0,
    );
    let query_label = if state.query.is_empty() {
        search_placeholder(ui_state)
    } else {
        state.query.as_str()
    };
    ui.label_at(
        search.min_x + 32.0,
        search.min_y + (search.height() - 14.0) * 0.5,
        query_label,
        if state.query.is_empty() {
            style::TEXT_DIM
        } else {
            style::TEXT
        },
        FONT_SCALE * TYPE_BODY,
    );

    let filter_hov = ui.pointer_in(filter_r);
    if filter_hov {
        ui.state.set_hot(Id::new("qa_filter"));
    }
    if filter_hov && ui.input.primary_pressed {
        ui.state.active = Some(Id::new("qa_filter"));
    }
    if ui.input.primary_released && ui.state.is_active(Id::new("qa_filter")) && filter_hov {
        state.filter_menu_open = !state.filter_menu_open;
    }
    ui.panel_rounded(
        filter_r,
        if filter_hov || state.filter_menu_open {
            style::BUTTON_HOVER
        } else {
            style::SURFACE
        },
        style::RADIUS_MD,
    );
    ui.icon_at(
        filter_r.min_x + 10.0,
        filter_r.min_y + (filter_r.height() - 14.0) * 0.5,
        Icon::History,
        style::TEXT_MUTED,
        14.0,
    );
    let filter_label = if state.recent_only { "Recent" } else { "All" };
    ui.label_at(
        filter_r.min_x + 30.0,
        filter_r.min_y + (filter_r.height() - 14.0) * 0.5,
        filter_label,
        style::TEXT,
        FONT_SCALE * TYPE_LABEL,
    );
    ui.icon_at(
        filter_r.max_x - 20.0,
        filter_r.min_y + (filter_r.height() - 12.0) * 0.5,
        Icon::ChevronDown,
        style::TEXT_MUTED,
        12.0,
    );

    if state.filter_menu_open {
        let menu = Rect::from_pos_size(filter_r.min_x, filter_r.max_y + 4.0, filter_w, 64.0);
        ui.panel_rounded(menu, style::COMBO_MENU_BG, style::RADIUS_SM);
        for (i, (label, recent)) in [("All tools", false), ("Recent", true)].iter().enumerate() {
            let row = Rect::from_pos_size(
                menu.min_x + 4.0,
                menu.min_y + 4.0 + i as f32 * 28.0,
                filter_w - 8.0,
                26.0,
            );
            let hov = ui.pointer_in(row);
            if hov {
                ui.panel_rounded(row, style::HOVER_BG, 4.0);
            }
            let mark = if state.recent_only == *recent {
                "● "
            } else {
                "   "
            };
            ui.label_at(
                row.min_x + 8.0,
                row.min_y + 5.0,
                &format!("{mark}{label}"),
                style::TEXT,
                FONT_SCALE * TYPE_LABEL,
            );
            if hov && ui.input.primary_released {
                state.recent_only = *recent;
                state.filter_menu_open = false;
            }
        }
        if ui.input.primary_pressed && !ui.pointer_in(menu) && !ui.pointer_in(filter_r) {
            state.filter_menu_open = false;
        }
    }

    // —— Body: grid + sidebar ————————————————————————————————————————
    let body_top = popup.min_y + HEADER_BLOCK;
    let body_bot = popup.max_y - FOOTER_H;
    let grid_area = Rect::from_min_max(
        popup.min_x + PAD,
        body_top,
        popup.max_x - PAD - SIDEBAR_W - GAP,
        body_bot,
    );
    let sidebar = Rect::from_min_max(grid_area.max_x + GAP, body_top, popup.max_x - PAD, body_bot);

    ui.begin_panel_scrolled(
        Id::new("quick_add_grid"),
        grid_area,
        Color::rgba(0.0, 0.0, 0.0, 0.0),
        &mut state.scroll_y,
    );

    let cols = if grid_area.width() < 420.0 {
        2
    } else {
        GRID_COLS
    };
    let tile_w = ((grid_area.width() - TILE_GAP * (cols as f32 - 1.0)) / cols as f32).max(140.0);

    if items.is_empty() {
        ui.label_at(
            grid_area.min_x + 8.0,
            grid_area.min_y + 16.0,
            "No matching tools.",
            style::TEXT_MUTED,
            FONT_SCALE * TYPE_BODY,
        );
        ui.gap(40.0);
    } else {
        let rows = (items.len() + cols - 1) / cols;
        let content_h = rows as f32 * (TILE_H + TILE_GAP) + PAD;
        ui.gap(content_h.max(1.0));

        for (i, item) in items.iter().enumerate() {
            let col = i % cols;
            let row = i / cols;
            let x = grid_area.min_x + col as f32 * (tile_w + TILE_GAP);
            let y = grid_area.min_y - state.scroll_y + row as f32 * (TILE_H + TILE_GAP);
            let tile = Rect::from_pos_size(x, y, tile_w, TILE_H);
            if tile.max_y < grid_area.min_y - 4.0 || tile.min_y > grid_area.max_y + 4.0 {
                continue;
            }

            let id = Id::new("qa_tile").child(item.id());
            let selected = state.selected_id.as_deref() == Some(item.id());
            let hovered = ui.pointer_in(tile) && ui.pointer_in(grid_area);
            if hovered {
                ui.state.set_hot(id);
            }
            if hovered && ui.input.primary_pressed {
                ui.state.active = Some(id);
            }
            if ui.input.primary_released && ui.state.is_active(id) && hovered {
                if selected {
                    commit_item(item, ui_state, state, &mut actions);
                    ui.end_panel_scrolled(&mut state.scroll_y);
                    ui.end_overlay();
                    return actions;
                }
                state.selected_id = Some(item.id().to_string());
            }

            if selected {
                draw_selection_border(ui, tile);
                let inner = Rect::from_pos_size(
                    tile.min_x + BORDER,
                    tile.min_y + BORDER,
                    tile.width() - BORDER * 2.0,
                    tile.height() - BORDER * 2.0,
                );
                ui.panel_rounded(inner, style::SELECTED_BG, style::RADIUS_MD - 1.0);
            } else {
                ui.panel_rounded(
                    tile,
                    if hovered {
                        style::HOVER_BG
                    } else {
                        style::SURFACE
                    },
                    style::RADIUS_MD,
                );
            }

            let thumb = Rect::from_pos_size(
                tile.min_x + 10.0,
                tile.min_y + (TILE_H - THUMB_SZ) * 0.5,
                THUMB_SZ,
                THUMB_SZ,
            );
            draw_tool_thumb(ui, thumb, item, selected);

            let text_x = thumb.max_x + 10.0;
            let text_w = (tile.max_x - 10.0 - text_x).max(40.0);
            let name = DrawList::truncate_to_width(item.label(), FONT_SCALE * TYPE_BODY, text_w);
            ui.label_at(
                text_x,
                tile.min_y + 16.0,
                &name,
                style::TEXT,
                FONT_SCALE * TYPE_BODY,
            );
            let desc = DrawList::truncate_to_width(
                &item.short_description(),
                FONT_SCALE * TYPE_CAPTION,
                text_w,
            );
            ui.label_at(
                text_x,
                tile.min_y + 36.0,
                &desc,
                style::TEXT_MUTED,
                FONT_SCALE * TYPE_CAPTION,
            );
        }
    }
    ui.end_panel_scrolled(&mut state.scroll_y);

    // Detail sidebar.
    ui.panel_rounded(sidebar, style::SURFACE, style::RADIUS_MD);
    if let Some(sel) = items
        .iter()
        .find(|i| Some(i.id()) == state.selected_id.as_deref())
    {
        draw_sidebar(ui, sidebar, sel);
    } else {
        ui.label_at(
            sidebar.min_x + PAD_SM + 4.0,
            sidebar.min_y + PAD,
            "Select a tool",
            style::TEXT_MUTED,
            FONT_SCALE * TYPE_BODY,
        );
    }

    // —— Footer ——————————————————————————————————————————————————————
    let footer = Rect::from_min_max(popup.min_x, body_bot, popup.max_x, popup.max_y);
    ui.panel(
        Rect::from_pos_size(
            footer.min_x + PAD,
            footer.min_y,
            footer.width() - PAD * 2.0,
            1.0,
        ),
        style::SEPARATOR,
    );
    ui.label_at(
        footer.min_x + PAD,
        footer.min_y + 18.0,
        modal_tip(ui_state),
        style::TEXT_DIM,
        FONT_SCALE * TYPE_CAPTION,
    );

    let btn_h = 32.0;
    let add_label = confirm_label(ui_state);
    let add_w = DrawList::text_width(add_label, FONT_SCALE * TYPE_BODY) + 28.0;
    let cancel_w = 88.0;
    let add_r = Rect::from_pos_size(
        footer.max_x - PAD - add_w,
        footer.min_y + (FOOTER_H - btn_h) * 0.5,
        add_w,
        btn_h,
    );
    let cancel_r = Rect::from_pos_size(add_r.min_x - GAP - cancel_w, add_r.min_y, cancel_w, btn_h);

    if text_button(ui, Id::new("qa_cancel"), "Cancel", cancel_r, false) {
        clear_quick_add(ui_state, state);
        ui.end_overlay();
        return actions;
    }
    let can_add = state.selected_id.is_some() && !items.is_empty();
    if can_add && text_button(ui, Id::new("qa_add"), add_label, add_r, true) {
        if let Some(sel) = items
            .iter()
            .find(|i| Some(i.id()) == state.selected_id.as_deref())
        {
            commit_item(sel, ui_state, state, &mut actions);
            ui.end_overlay();
            return actions;
        }
    } else if !can_add {
        // Disabled primary.
        ui.panel_rounded(add_r, style::ACCENT_DIM, style::RADIUS_SM);
        let tw = DrawList::text_width(add_label, FONT_SCALE * TYPE_BODY);
        ui.label_at(
            add_r.min_x + (add_r.width() - tw) * 0.5,
            add_r.min_y + (add_r.height() - 14.0) * 0.5,
            add_label,
            style::TEXT_DISABLED,
            FONT_SCALE * TYPE_BODY,
        );
    }

    if ui.input.enter_pressed && can_add {
        if let Some(sel) = items
            .iter()
            .find(|i| Some(i.id()) == state.selected_id.as_deref())
        {
            commit_item(sel, ui_state, state, &mut actions);
            ui.end_overlay();
            return actions;
        }
    }

    ui.end_overlay();

    if ui.input.primary_pressed && !ui.pointer_in(popup) {
        clear_quick_add(ui_state, state);
    }
    actions
}

#[cfg(test)]
mod tests {
    use super::*;
    use terra_core::layer::{EffectFilterKind, Layer, LayerKind};

    fn commit_catalog_layer(tool_id: &str) -> (LayerKind, Layer) {
        let tool = quick_add_entries()
            .into_iter()
            .find(|tool| tool.id == tool_id)
            .unwrap_or_else(|| panic!("Quick Add should expose {tool_id}"));
        let ToolAction::AddLayer { name, kind } = &tool.action else {
            panic!("{tool_id} should create a layer");
        };
        let expected_name = *name;
        let expected_kind = kind.clone();
        let item = PickerItem::Tool(tool);
        let mut ui_state = UiState::default();
        let mut state = QuickAddState::default();
        let mut actions = Vec::new();

        commit_item(&item, &mut ui_state, &mut state, &mut actions);

        assert_eq!(actions.len(), 1);
        let PanelAction::AddLayer(layer) = actions.remove(0) else {
            panic!("generic Quick Add should emit AddLayer");
        };
        assert_eq!(layer.common.name, expected_name);
        (expected_kind, layer)
    }

    fn assert_same_kind(expected: &LayerKind, actual: &LayerKind) {
        assert_eq!(
            serde_json::to_value(actual).expect("serialize created layer kind"),
            serde_json::to_value(expected).expect("serialize catalog layer kind")
        );
    }

    fn commit_org(id: &'static str) -> PanelAction {
        let item = PickerItem::Org {
            id,
            label: "Test organisation item",
            description: "Test organisation item.",
            icon: Icon::Folder,
        };
        let mut ui_state = UiState::default();
        let mut state = QuickAddState::default();
        let mut actions = Vec::new();

        commit_item(&item, &mut ui_state, &mut state, &mut actions);

        assert_eq!(actions.len(), 1, "{id} should emit exactly one action");
        actions.remove(0)
    }

    #[test]
    fn scoped_quick_add_includes_registry_derived_routes() {
        let mut shape_state = UiState::default();
        shape_state.quick_add_concept = Some(ArtistConcept::Shape);
        assert!(catalog_for_modal(&shape_state)
            .iter()
            .any(|tool| tool.registry_type_id == Some("island")));

        let mut simulation_state = UiState::default();
        simulation_state.quick_add_category = Some(StackCategory::Simulation);
        assert!(catalog_for_modal(&simulation_state)
            .iter()
            .any(|tool| tool.registry_type_id == Some("river_network")));

        let mut filter_state = UiState::default();
        filter_state.quick_add_into = Some(terra_core::layer::LayerId::new());
        filter_state.quick_add_biome_section = Some(BiomeSection::Filters);
        assert!(catalog_for_modal(&filter_state)
            .iter()
            .any(|tool| tool.registry_type_id == Some("blur")));
    }

    #[test]
    fn quick_add_commits_direct_island_layer() {
        let (_, layer) = commit_catalog_layer("gen.island");
        assert!(matches!(layer.kind, LayerKind::Island(_)));
    }

    #[test]
    fn quick_add_preserves_non_default_effect_filter_preset() {
        let (expected, layer) = commit_catalog_layer("filter.arid.rocky_plateaus");

        let LayerKind::EffectFilter(params) = &layer.kind else {
            panic!("Rocky Plateaus should remain an effect filter");
        };
        assert_eq!(params.kind, EffectFilterKind::RockyPlateaus);
        assert_ne!(params.kind, EffectFilterKind::Smooth);
        assert_same_kind(&expected, &layer.kind);
    }

    #[test]
    fn quick_add_preserves_non_default_vegetation_preset() {
        let (expected, layer) = commit_catalog_layer("obj.rocks");

        let LayerKind::Vegetation(params) = &layer.kind else {
            panic!("Rocks should remain a vegetation preset");
        };
        assert_eq!(params.density, 0.22);
        assert_eq!(params.min_distance, 5.0);
        assert_eq!(params.min_slope_deg, 22.0);
        assert_eq!(params.max_slope_deg, 90.0);
        assert!(!params.coverage.nodes.is_empty());
        assert_same_kind(&expected, &layer.kind);
    }

    #[test]
    fn quick_add_exposes_organisation_items_for_the_active_scope() {
        let generic = UiState::default();
        let generic_items = org_items(&generic);
        let generic_ids: Vec<_> = generic_items.iter().map(PickerItem::id).collect();
        assert_eq!(
            generic_ids,
            ["org.pass", "org.isolated", "org.biome", "org.hole"]
        );

        let mut biomes = UiState::default();
        biomes.quick_add_concept = Some(ArtistConcept::Biomes);
        let biome_items = org_items(&biomes);
        let biome_ids: Vec<_> = biome_items.iter().map(PickerItem::id).collect();
        assert_eq!(biome_ids, ["org.biome"]);

        let mut biome_layers = UiState::default();
        biome_layers.quick_add_concept = Some(ArtistConcept::BiomeLayers);
        let biome_layer_items = org_items(&biome_layers);
        let biome_layer_ids: Vec<_> = biome_layer_items.iter().map(PickerItem::id).collect();
        assert_eq!(biome_layer_ids, ["org.biome_paint"]);
    }

    #[test]
    fn quick_add_commits_organisation_items_through_picker_item_org() {
        assert!(matches!(
            commit_org("org.pass"),
            PanelAction::AddGroup { .. }
        ));
        assert!(matches!(
            commit_org("org.isolated"),
            PanelAction::AddIsolatedGroup { .. }
        ));
        assert!(matches!(
            commit_org("org.biome"),
            PanelAction::AddBiome { .. }
        ));
        assert!(matches!(
            commit_org("org.hole"),
            PanelAction::AddHoleLayer { .. }
        ));
        assert!(matches!(
            commit_org("org.biome_paint"),
            PanelAction::AddBiomePaintLayer { .. }
        ));
    }
}

fn draw_sidebar(ui: &mut GuiContext<'_>, sidebar: Rect, item: &PickerItem) {
    let mut y = sidebar.min_y + PAD;
    ui.label_at(
        sidebar.min_x + PAD,
        y,
        item.label(),
        style::TEXT,
        FONT_SCALE * TYPE_TITLE * 1.1,
    );
    y += 24.0;

    // Wrap description roughly by truncating to sidebar width twice.
    let desc_w = sidebar.width() - PAD * 2.0;
    let desc = item.description();
    let line1 = DrawList::truncate_to_width(desc, FONT_SCALE * TYPE_LABEL, desc_w);
    ui.label_at(
        sidebar.min_x + PAD,
        y,
        &line1,
        style::TEXT_MUTED,
        FONT_SCALE * TYPE_LABEL,
    );
    y += 18.0;
    if line1.len() < desc.len() {
        let rest = desc.get(line1.len().saturating_sub(1)..).unwrap_or("");
        let line2 = DrawList::truncate_to_width(
            rest.trim_start_matches(['…', ' ']),
            FONT_SCALE * TYPE_LABEL,
            desc_w,
        );
        if !line2.is_empty() {
            ui.label_at(
                sidebar.min_x + PAD,
                y,
                &line2,
                style::TEXT_MUTED,
                FONT_SCALE * TYPE_LABEL,
            );
            y += 18.0;
        }
    }
    y += 8.0;

    let preview_h = 140.0_f32.min(sidebar.height() * 0.35);
    let preview = Rect::from_pos_size(
        sidebar.min_x + PAD,
        y,
        sidebar.width() - PAD * 2.0,
        preview_h,
    );
    ui.panel_rounded(preview, style::RAISED_BG, style::RADIUS_MD);
    let thumb_s = (preview_h - 24.0).min(preview.width() - 24.0).max(48.0);
    let thumb = Rect::from_pos_size(
        preview.min_x + (preview.width() - thumb_s) * 0.5,
        preview.min_y + (preview.height() - thumb_s) * 0.5,
        thumb_s,
        thumb_s,
    );
    draw_tool_thumb(ui, thumb, item, true);
    y = preview.max_y + 14.0;

    ui.label_at(
        sidebar.min_x + PAD,
        y,
        "Common uses",
        style::TEXT,
        FONT_SCALE * TYPE_LABEL,
    );
    y += 18.0;
    for use_line in common_uses(item) {
        ui.label_at(
            sidebar.min_x + PAD,
            y,
            &format!("• {use_line}"),
            style::TEXT_MUTED,
            FONT_SCALE * TYPE_CAPTION,
        );
        y += 16.0;
    }
    y += 10.0;

    ui.label_at(
        sidebar.min_x + PAD,
        y,
        "Inputs required",
        style::TEXT,
        FONT_SCALE * TYPE_LABEL,
    );
    y += 18.0;
    ui.icon_at(
        sidebar.min_x + PAD,
        y,
        Icon::ArrowDown,
        style::TEXT_MUTED,
        12.0,
    );
    ui.label_at(
        sidebar.min_x + PAD + 18.0,
        y,
        inputs_label(item),
        style::TEXT_MUTED,
        FONT_SCALE * TYPE_CAPTION,
    );
    y += 22.0;

    ui.label_at(
        sidebar.min_x + PAD,
        y,
        "Output",
        style::TEXT,
        FONT_SCALE * TYPE_LABEL,
    );
    y += 18.0;
    ui.icon_at(
        sidebar.min_x + PAD,
        y,
        Icon::ArrowUp,
        style::TEXT_MUTED,
        12.0,
    );
    ui.label_at(
        sidebar.min_x + PAD + 18.0,
        y,
        output_label(item),
        style::TEXT_MUTED,
        FONT_SCALE * TYPE_CAPTION,
    );
}

fn icon_hit(ui: &mut GuiContext<'_>, id: Id, icon: Icon, rect: Rect) -> bool {
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
    ui.icon_centered(rect, icon, style::TEXT_MUTED, 14.0);
    clicked
}

fn text_button(ui: &mut GuiContext<'_>, id: Id, label: &str, rect: Rect, primary: bool) -> bool {
    let hovered = ui.pointer_in(rect);
    if hovered {
        ui.state.set_hot(id);
    }
    if hovered && ui.input.primary_pressed {
        ui.state.active = Some(id);
    }
    let clicked = ui.input.primary_released && ui.state.is_active(id) && hovered;
    let bg = if primary {
        if hovered {
            style::ACCENT_HOVER
        } else {
            style::ACCENT
        }
    } else if hovered {
        style::BUTTON_HOVER
    } else {
        style::BUTTON_BG
    };
    ui.panel_rounded(rect, bg, style::RADIUS_SM);
    let tw = DrawList::text_width(label, FONT_SCALE * TYPE_BODY);
    ui.label_at(
        rect.min_x + (rect.width() - tw) * 0.5,
        rect.min_y + (rect.height() - 14.0) * 0.5,
        label,
        style::TEXT,
        FONT_SCALE * TYPE_BODY,
    );
    clicked
}

fn remember_tool(recent: &mut Vec<String>, id: &str) {
    recent.retain(|existing| existing != id);
    recent.insert(0, id.to_string());
    recent.truncate(12);
}

fn apply_search_input(ui: &GuiContext<'_>, query: &mut String) {
    if ui.input.backspace_pressed {
        query.pop();
    }
    query.push_str(&ui.input.text);
}

pub fn contextual_suggestion_type_ids(_doc: &TerrainDocument, _ui_state: &UiState) -> Vec<String> {
    Vec::new()
}
