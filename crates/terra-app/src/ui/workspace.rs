//! Non-linear task workspaces Ã¢â‚¬â€ presentation focus, never workflow steps.
//!
//! Workspaces configure tools, overlays, and hierarchy *emphasis*. They must not
//! lock content, reorder evaluation, create entities, or act like a wizard.
//!
//! Copy uses: Workspace, Tools, Suggested actions, Current task.
//! Avoid: Step, Stage complete, Continue, Required next action, All Tools (removed).

use crate::ui::style;
use serde::{Deserialize, Serialize};
use terra_core::biome_definition::BiomeDefinitionId;
use terra_core::biome_paint::BiomePaintTool;
use terra_core::domain::DomainRole;
use terra_core::eval::PreviewQuality;
use terra_core::layer::{LayerId, StackCategory};
use terra_core::mask::{MaskId, MaskPaintTool};
use terra_gui::{Color, Icon};

use crate::ui::{EditorTool, LightingPreset, Preview2dMode, ViewportOverlayFlags};

// Workspace identity (persisted as editor preference, not project data)

/// Task-focused interface configuration. Order in [`WorkspaceId::ALL`] is for
/// UI listing only Ã¢â‚¬â€ it does **not** imply progression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkspaceId {
    World,
    #[default]
    Sculpt,
    Biomes,
    Develop,
    Rules,
    Simulation,
    Surface,
    Objects,
    AllTools,
}

impl WorkspaceId {
    /// Workspaces shown in the TOOLS rail (World / All Tools are command-only).
    pub const ALL: [WorkspaceId; 7] = [
        WorkspaceId::Sculpt,
        WorkspaceId::Biomes,
        WorkspaceId::Develop,
        WorkspaceId::Rules,
        WorkspaceId::Simulation,
        WorkspaceId::Surface,
        WorkspaceId::Objects,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::World => "world",
            Self::Sculpt => "sculpt",
            Self::Biomes => "biomes",
            Self::Develop => "filters",
            Self::Rules => "mask",
            Self::Simulation => "simulation",
            Self::Surface => "surface",
            Self::Objects => "objects",
            Self::AllTools => "all_tools",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "world" | "layout" => Some(Self::World),
            "sculpt" | "terrain" | "landforms" => Some(Self::Sculpt),
            "biomes" => Some(Self::Biomes),
            "develop" | "filters" | "filter" => Some(Self::Develop),
            "rules" | "masks" | "mask" => Some(Self::Rules),
            "simulation" | "hydrology" | "erosion" => Some(Self::Simulation),
            "surface" | "materials" => Some(Self::Surface),
            "objects" | "scatter" | "advanced" => Some(Self::Objects),
            // Legacy Ã¢â‚¬â€ no longer in the rail; still parse so old prefs load.
            "all_tools" | "all" | "utilities" => Some(Self::AllTools),
            _ => None,
        }
    }

    pub fn definition(self) -> &'static WorkspaceDefinition {
        workspace_definition(self)
    }

    /// Optional digit shortcut (listing aid only Ã¢â‚¬â€ not a step number).
    pub fn from_digit(digit: u8) -> Option<Self> {
        match digit {
            1 => Some(Self::World),
            2 => Some(Self::Sculpt),
            3 => Some(Self::Biomes),
            4 => Some(Self::Develop),
            5 => Some(Self::Rules),
            6 => Some(Self::Simulation),
            7 => Some(Self::Surface),
            8 => Some(Self::Objects),
            _ => None,
        }
    }

    pub fn digit_shortcut(self) -> Option<u8> {
        match self {
            Self::World => Some(1),
            Self::Sculpt => Some(2),
            Self::Biomes => Some(3),
            Self::Develop => Some(4),
            Self::Rules => Some(5),
            Self::Simulation => Some(6),
            Self::Surface => Some(7),
            Self::Objects => Some(8),
            Self::AllTools => None,
        }
    }

    /// Stable command id for workspaces exposed by the command palette.
    ///
    /// `AllTools` is compatibility-only and deliberately has no visible command.
    pub const fn command_id(self) -> Option<&'static str> {
        match self {
            Self::World => Some("workspace.world"),
            Self::Sculpt => Some("workspace.sculpt"),
            Self::Biomes => Some("workspace.biomes"),
            Self::Develop => Some("workspace.develop"),
            Self::Rules => Some("workspace.rules"),
            Self::Simulation => Some("workspace.simulation"),
            Self::Surface => Some("workspace.surface"),
            Self::Objects => Some("workspace.objects"),
            Self::AllTools => None,
        }
    }
}

// Metadata-driven definition

/// Soft hierarchy emphasis Ã¢â‚¬â€ never hides or locks nodes.
#[derive(Debug, Clone, Copy)]
pub struct HierarchyEmphasis {
    pub prefer_roles: &'static [DomainRole],
    pub prefer_categories: &'static [StackCategory],
    /// Visual dim only; unrelated items remain selectable and editable.
    pub dim_unrelated: bool,
}

impl HierarchyEmphasis {
    pub const NONE: Self = Self {
        prefer_roles: &[],
        prefer_categories: &[],
        dim_unrelated: false,
    };

    pub fn emphasizes_role(self, role: DomainRole) -> bool {
        self.prefer_roles.is_empty() || self.prefer_roles.contains(&role)
    }

    /// Visual dim only Ã¢â‚¬â€ unrelated rows stay selectable and editable.
    pub fn should_dim_role(self, role: DomainRole) -> bool {
        self.dim_unrelated && !self.emphasizes_role(role)
    }
}

/// Whether a layer row should be visually de-emphasized for the active workspace.
pub fn hierarchy_dim_for_kind(workspace: WorkspaceId, kind: &terra_core::layer::LayerKind) -> bool {
    let def = workspace_definition(workspace);
    let role = terra_core::domain::classify_layer_kind(kind);
    def.hierarchy.should_dim_role(role)
}

/// Which left-palette tool catalog modes to show.
#[derive(Debug, Clone, Copy)]
pub enum WorkspaceToolFilter {
    /// All Tools â€” minimal filtering.
    All,
    /// Include tools tagged with any of these legacy catalog modes.
    CatalogModes(&'static [WorkspaceMode]),
    /// World Creatorâ€“style biome Filters catalog (grouped by WC category).
    WcFilters,
}

/// Suggested quick action ids (presentation hints).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextualActionId {
    ArmSculptBrush,
    ArmBiomePaint,
    OpenMaskEditor,
    AddHydraulic,
    AddMaterials,
    AddScatter,
    FrameTerrain,
}

impl ContextualActionId {
    pub fn label(self) -> &'static str {
        match self {
            Self::ArmSculptBrush => "Arm sculpt brush",
            Self::ArmBiomePaint => "Arm biome paint",
            Self::OpenMaskEditor => "Open mask tools",
            Self::AddHydraulic => "Add hydraulic erosion",
            Self::AddMaterials => "Add materials",
            Self::AddScatter => "Add scatter",
            Self::FrameTerrain => "Frame terrain",
        }
    }
}

/// Static workspace metadata Ã¢â‚¬â€ drives presentation without hard-coded UI branches.
#[derive(Debug, Clone, Copy)]
pub struct WorkspaceDefinition {
    pub id: WorkspaceId,
    pub name: &'static str,
    pub description: &'static str,
    pub icon: Icon,
    /// Artist-facing command name, or `None` for compatibility-only workspaces.
    pub command_name: Option<&'static str>,
    pub tools: WorkspaceToolFilter,
    pub hierarchy: HierarchyEmphasis,
    pub preferred_overlays: ViewportOverlayFlags,
    pub preferred_viewport_mode: Option<Preview2dMode>,
    pub preferred_editor_tool: Option<EditorTool>,
    pub preferred_inspector_section: Option<&'static str>,
    pub contextual_actions: &'static [ContextualActionId],
    pub biome_color_preview: bool,
}

impl WorkspaceDefinition {
    pub fn accent(self) -> Color {
        match self.id {
            WorkspaceId::World => style::MODE_UTILITIES,
            WorkspaceId::Sculpt | WorkspaceId::Develop => style::MODE_TERRAIN,
            WorkspaceId::Biomes => style::MODE_BIOMES,
            WorkspaceId::Rules => style::MODE_MASKS,
            WorkspaceId::Simulation => style::MODE_SIMULATION,
            WorkspaceId::Surface => style::MODE_MATERIALS,
            WorkspaceId::Objects => style::MODE_OBJECTS,
            WorkspaceId::AllTools => style::ACCENT,
        }
    }

    pub fn tools_heading(self) -> &'static str {
        match self.id {
            WorkspaceId::World => "WORLD",
            WorkspaceId::Sculpt => "SCULPT",
            WorkspaceId::Biomes => "BIOMES",
            WorkspaceId::Develop => "FILTERS",
            WorkspaceId::Rules => "MASK",
            WorkspaceId::Simulation => "SIMULATION",
            WorkspaceId::Surface => "SURFACE",
            WorkspaceId::Objects => "OBJECTS",
            WorkspaceId::AllTools => "TOOLS",
        }
    }

    pub fn includes_catalog_mode(self, mode: WorkspaceMode) -> bool {
        match self.tools {
            WorkspaceToolFilter::All => true,
            WorkspaceToolFilter::CatalogModes(modes) => modes.contains(&mode),
            WorkspaceToolFilter::WcFilters => mode == WorkspaceMode::Filters,
        }
    }
}

pub fn workspace_definition(id: WorkspaceId) -> &'static WorkspaceDefinition {
    for def in &WORKSPACE_DEFINITIONS {
        if def.id == id {
            return def;
        }
    }
    &WORKSPACE_DEFINITIONS[1] // Sculpt fallback
}

pub fn all_workspace_definitions() -> &'static [WorkspaceDefinition] {
    &WORKSPACE_DEFINITIONS
}

static WORKSPACE_DEFINITIONS: [WorkspaceDefinition; 9] = [
    WorkspaceDefinition {
        id: WorkspaceId::World,
        name: "World",
        description: "Seed, size, sea level, blueprint, and regions.",
        icon: Icon::Package,
        command_name: Some("Switch Workspace: World"),
        tools: WorkspaceToolFilter::CatalogModes(&[WorkspaceMode::Utilities]),
        hierarchy: HierarchyEmphasis {
            prefer_roles: &[],
            prefer_categories: &[StackCategory::Foundation],
            dim_unrelated: false,
        },
        preferred_overlays: ViewportOverlayFlags {
            world_bounds: true,
            ..ViewportOverlayFlags::EMPTY
        },
        preferred_viewport_mode: Some(Preview2dMode::Lit),
        preferred_editor_tool: Some(EditorTool::Move),
        preferred_inspector_section: Some("world"),
        contextual_actions: &[ContextualActionId::FrameTerrain],
        biome_color_preview: false,
    },
    WorkspaceDefinition {
        id: WorkspaceId::Sculpt,
        name: "Sculpt",
        description: "Height brushes, landforms, and shapes.",
        icon: Icon::Pencil,
        command_name: Some("Switch Workspace: Sculpt"),
        tools: WorkspaceToolFilter::CatalogModes(&[WorkspaceMode::Terrain]),
        hierarchy: HierarchyEmphasis {
            prefer_roles: &[DomainRole::ShapeLayer],
            prefer_categories: &[StackCategory::Shape, StackCategory::Foundation],
            dim_unrelated: false,
        },
        preferred_overlays: ViewportOverlayFlags {
            brush_preview: true,
            ..ViewportOverlayFlags::EMPTY
        },
        preferred_viewport_mode: Some(Preview2dMode::Lit),
        preferred_editor_tool: Some(EditorTool::Move),
        preferred_inspector_section: Some("sculpt"),
        contextual_actions: &[ContextualActionId::ArmSculptBrush],
        biome_color_preview: false,
    },
    WorkspaceDefinition {
        id: WorkspaceId::Biomes,
        name: "Biomes",
        description: "Biome placement, paint, and definitions.",
        icon: Icon::Layers,
        command_name: Some("Switch Workspace: Biomes"),
        tools: WorkspaceToolFilter::CatalogModes(&[WorkspaceMode::Biomes]),
        hierarchy: HierarchyEmphasis {
            prefer_roles: &[DomainRole::MaskLayer, DomainRole::TerrainFilter],
            prefer_categories: &[StackCategory::Surface],
            dim_unrelated: false,
        },
        preferred_overlays: ViewportOverlayFlags::EMPTY,
        preferred_viewport_mode: Some(Preview2dMode::Biome),
        preferred_editor_tool: Some(EditorTool::PaintBiome),
        preferred_inspector_section: Some("biomes"),
        contextual_actions: &[ContextualActionId::ArmBiomePaint],
        biome_color_preview: true,
    },
    WorkspaceDefinition {
        id: WorkspaceId::Develop,
        name: "Filters",
        description:
            "World Creatorâ€“style terrain filters â€” general, effect, arid, erosion, sediment.",
        icon: Icon::SlidersHorizontal,
        command_name: Some("Switch Workspace: Filters"),
        tools: WorkspaceToolFilter::WcFilters,
        hierarchy: HierarchyEmphasis {
            prefer_roles: &[
                DomainRole::MaskLayer,
                DomainRole::TerrainFilter,
                DomainRole::SimulationLayer,
                DomainRole::MaterialLayer,
                DomainRole::ScatterLayer,
                DomainRole::ObjectLayer,
            ],
            prefer_categories: &[StackCategory::Surface, StackCategory::Shape],
            dim_unrelated: false,
        },
        preferred_overlays: ViewportOverlayFlags::EMPTY,
        preferred_viewport_mode: Some(Preview2dMode::Slope),
        preferred_editor_tool: Some(EditorTool::Move),
        preferred_inspector_section: Some("develop"),
        contextual_actions: &[
            ContextualActionId::AddMaterials,
            ContextualActionId::AddScatter,
        ],
        biome_color_preview: true,
    },
    WorkspaceDefinition {
        id: WorkspaceId::Rules,
        name: "Mask",
        description: "Paint and edit masks Ã¢â‚¬â€ coverage, placement, and advanced mask stacks.",
        icon: Icon::CircleDot,
        command_name: Some("Switch Workspace: Mask"),
        tools: WorkspaceToolFilter::CatalogModes(&[WorkspaceMode::Masks]),
        hierarchy: HierarchyEmphasis {
            prefer_roles: &[DomainRole::MaskLayer],
            prefer_categories: &[StackCategory::Mask],
            dim_unrelated: false,
        },
        preferred_overlays: ViewportOverlayFlags {
            mask_overlay: true,
            ..ViewportOverlayFlags::EMPTY
        },
        preferred_viewport_mode: None,
        preferred_editor_tool: Some(EditorTool::PaintMask),
        preferred_inspector_section: Some("rules"),
        contextual_actions: &[ContextualActionId::OpenMaskEditor],
        biome_color_preview: false,
    },
    WorkspaceDefinition {
        id: WorkspaceId::Simulation,
        name: "Simulation",
        description: "Simulation Scenarios Ã¢â‚¬â€ coherent physical setups (optional).",
        icon: Icon::Droplets,
        command_name: Some("Switch Workspace: Simulation"),
        tools: WorkspaceToolFilter::CatalogModes(&[WorkspaceMode::Simulation]),
        hierarchy: HierarchyEmphasis {
            prefer_roles: &[DomainRole::SimulationLayer],
            prefer_categories: &[StackCategory::Simulation],
            dim_unrelated: false,
        },
        preferred_overlays: ViewportOverlayFlags::EMPTY,
        preferred_viewport_mode: Some(Preview2dMode::Flow),
        preferred_editor_tool: Some(EditorTool::Move),
        preferred_inspector_section: Some("simulation"),
        contextual_actions: &[ContextualActionId::AddHydraulic],
        biome_color_preview: false,
    },
    WorkspaceDefinition {
        id: WorkspaceId::Surface,
        name: "Surface",
        description: "Materials and surface attributes.",
        icon: Icon::Paintbrush,
        command_name: Some("Switch Workspace: Surface"),
        tools: WorkspaceToolFilter::CatalogModes(&[WorkspaceMode::Materials]),
        hierarchy: HierarchyEmphasis {
            prefer_roles: &[DomainRole::MaterialLayer],
            prefer_categories: &[StackCategory::Surface],
            dim_unrelated: false,
        },
        preferred_overlays: ViewportOverlayFlags::EMPTY,
        preferred_viewport_mode: Some(Preview2dMode::Material),
        preferred_editor_tool: Some(EditorTool::Move),
        preferred_inspector_section: Some("materials"),
        contextual_actions: &[ContextualActionId::AddMaterials],
        biome_color_preview: false,
    },
    WorkspaceDefinition {
        id: WorkspaceId::Objects,
        name: "Objects",
        description: "Scatter, vegetation, and props.",
        icon: Icon::Sparkles,
        command_name: Some("Switch Workspace: Objects"),
        tools: WorkspaceToolFilter::CatalogModes(&[WorkspaceMode::Objects]),
        hierarchy: HierarchyEmphasis {
            prefer_roles: &[DomainRole::ScatterLayer, DomainRole::ObjectLayer],
            prefer_categories: &[],
            dim_unrelated: false,
        },
        preferred_overlays: ViewportOverlayFlags::EMPTY,
        preferred_viewport_mode: Some(Preview2dMode::VegetationDensity),
        preferred_editor_tool: Some(EditorTool::Move),
        preferred_inspector_section: Some("objects"),
        contextual_actions: &[ContextualActionId::AddScatter],
        biome_color_preview: false,
    },
    WorkspaceDefinition {
        id: WorkspaceId::AllTools,
        name: "All Tools",
        description: "Legacy Ã¢â‚¬â€ removed from the TOOLS rail; remaps to Objects.",
        icon: Icon::Grid3x3,
        command_name: None,
        tools: WorkspaceToolFilter::All,
        hierarchy: HierarchyEmphasis::NONE,
        preferred_overlays: ViewportOverlayFlags::EMPTY,
        preferred_viewport_mode: None,
        preferred_editor_tool: None,
        preferred_inspector_section: None,
        contextual_actions: &[],
        biome_color_preview: false,
    },
];

// Legacy catalog mode tags (tool_catalog still keys off these)

/// Tool-catalog category tag Ã¢â‚¬â€ not the artist-facing workspace selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum WorkspaceMode {
    #[default]
    Terrain,
    Biomes,
    Simulation,
    Materials,
    Objects,
    Masks,
    Utilities,
    /// WC-style biome Filters catalog (Filters workspace).
    Filters,
}

impl WorkspaceMode {
    pub const ALL: [WorkspaceMode; 8] = [
        WorkspaceMode::Terrain,
        WorkspaceMode::Biomes,
        WorkspaceMode::Simulation,
        WorkspaceMode::Materials,
        WorkspaceMode::Objects,
        WorkspaceMode::Masks,
        WorkspaceMode::Utilities,
        WorkspaceMode::Filters,
    ];

    pub fn label(self) -> &'static str {
        match self {
            WorkspaceMode::Terrain => "Terrain",
            WorkspaceMode::Biomes => "Biomes",
            WorkspaceMode::Simulation => "Simulation",
            WorkspaceMode::Materials => "Materials",
            WorkspaceMode::Objects => "Objects",
            WorkspaceMode::Masks => "Masks",
            WorkspaceMode::Utilities => "Utilities",
            WorkspaceMode::Filters => "Filters",
        }
    }

    pub fn short_label(self) -> &'static str {
        self.label()
    }

    pub fn tools_heading(self) -> &'static str {
        match self {
            WorkspaceMode::Terrain => "TERRAIN",
            WorkspaceMode::Biomes => "BIOMES",
            WorkspaceMode::Simulation => "SIMULATION",
            WorkspaceMode::Materials => "MATERIALS",
            WorkspaceMode::Objects => "OBJECTS",
            WorkspaceMode::Masks => "MASKS",
            WorkspaceMode::Utilities => "UTILITIES",
            WorkspaceMode::Filters => "FILTERS",
        }
    }

    pub fn icon(self) -> Icon {
        match self {
            WorkspaceMode::Terrain => Icon::Mountain,
            WorkspaceMode::Biomes => Icon::Layers,
            WorkspaceMode::Simulation => Icon::Droplets,
            WorkspaceMode::Materials => Icon::Paintbrush,
            WorkspaceMode::Objects => Icon::Sparkles,
            WorkspaceMode::Masks => Icon::CircleDot,
            WorkspaceMode::Utilities => Icon::SlidersHorizontal,
            WorkspaceMode::Filters => Icon::SlidersHorizontal,
        }
    }

    pub fn accent(self) -> Color {
        match self {
            WorkspaceMode::Terrain => style::MODE_TERRAIN,
            WorkspaceMode::Biomes => style::MODE_BIOMES,
            WorkspaceMode::Simulation => style::MODE_SIMULATION,
            WorkspaceMode::Materials => style::MODE_MATERIALS,
            WorkspaceMode::Objects => style::MODE_OBJECTS,
            WorkspaceMode::Masks => style::MODE_MASKS,
            WorkspaceMode::Utilities => style::MODE_UTILITIES,
            WorkspaceMode::Filters => style::MODE_TERRAIN,
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            WorkspaceMode::Terrain => "Landforms, noise, and height brushes.",
            WorkspaceMode::Biomes => "Paint biome ownership and placement.",
            WorkspaceMode::Simulation => "Erosion, rivers, coasts, and weathering.",
            WorkspaceMode::Materials => "Surface materials and attributes.",
            WorkspaceMode::Objects => "Trees, rocks, props, and scatter.",
            WorkspaceMode::Masks => "Painted and procedural masks.",
            WorkspaceMode::Utilities => "Camera, measure, and bake tools.",
            WorkspaceMode::Filters => "Terrain filters by category (WC-style).",
        }
    }

    pub fn to_workspace_id(self) -> WorkspaceId {
        match self {
            WorkspaceMode::Terrain => WorkspaceId::Sculpt,
            WorkspaceMode::Biomes => WorkspaceId::Biomes,
            WorkspaceMode::Simulation => WorkspaceId::Simulation,
            WorkspaceMode::Materials => WorkspaceId::Surface,
            WorkspaceMode::Objects => WorkspaceId::Objects,
            WorkspaceMode::Masks => WorkspaceId::Rules,
            WorkspaceMode::Utilities => WorkspaceId::World,
            WorkspaceMode::Filters => WorkspaceId::Develop,
        }
    }

    pub fn from_digit(digit: u8) -> Option<Self> {
        WorkspaceId::from_digit(digit).map(|id| match id {
            WorkspaceId::World => WorkspaceMode::Utilities,
            WorkspaceId::Sculpt => WorkspaceMode::Terrain,
            WorkspaceId::Develop => WorkspaceMode::Filters,
            WorkspaceId::Biomes => WorkspaceMode::Biomes,
            WorkspaceId::Rules => WorkspaceMode::Masks,
            WorkspaceId::Simulation => WorkspaceMode::Simulation,
            WorkspaceId::Surface => WorkspaceMode::Materials,
            WorkspaceId::Objects => WorkspaceMode::Objects,
            WorkspaceId::AllTools => WorkspaceMode::Utilities,
        })
    }

    pub fn digit_shortcut(self) -> u8 {
        self.to_workspace_id().digit_shortcut().unwrap_or(2)
    }

    pub fn shortcut_label(self) -> &'static str {
        match self.digit_shortcut() {
            1 => "1",
            2 => "2",
            3 => "3",
            4 => "4",
            5 => "5",
            6 => "6",
            7 => "7",
            8 => "8",
            _ => "",
        }
    }
}

/// Compatibility intent tabs (maps into [`WorkspaceId`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum AppWorkspace {
    #[default]
    Layout,
    Landforms,
    Biomes,
    Hydrology,
    Surface,
    Advanced,
    Review,
}

impl AppWorkspace {
    pub const ALL: [AppWorkspace; 7] = [
        AppWorkspace::Layout,
        AppWorkspace::Landforms,
        AppWorkspace::Biomes,
        AppWorkspace::Hydrology,
        AppWorkspace::Surface,
        AppWorkspace::Advanced,
        AppWorkspace::Review,
    ];

    pub fn label(self) -> &'static str {
        match self {
            AppWorkspace::Layout => "World",
            AppWorkspace::Landforms => "Terrain",
            AppWorkspace::Biomes => "Biomes",
            AppWorkspace::Hydrology => "Simulation",
            AppWorkspace::Surface => "Materials",
            AppWorkspace::Advanced => "Objects",
            AppWorkspace::Review => "Export",
        }
    }

    pub fn tools_heading(self) -> &'static str {
        match self {
            AppWorkspace::Layout => "WORLD",
            AppWorkspace::Landforms => "TERRAIN",
            AppWorkspace::Biomes => "BIOMES",
            AppWorkspace::Hydrology => "SIMULATION",
            AppWorkspace::Surface => "MATERIALS",
            AppWorkspace::Advanced => "OBJECTS",
            AppWorkspace::Review => "EXPORT",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            AppWorkspace::Layout => "Seed size, sea level, climate, and blueprint.",
            AppWorkspace::Landforms => "Block massing with shapes and landform bones.",
            AppWorkspace::Biomes => "Paint biome ownership and polish focus.",
            AppWorkspace::Hydrology => "Shared erosion, rivers, and climate processes.",
            AppWorkspace::Surface => "Materials and surface dressing.",
            AppWorkspace::Advanced => "Scatter objects, vegetation, and props.",
            AppWorkspace::Review => "Review at scale and export products.",
        }
    }

    pub fn is_advanced(self) -> bool {
        false
    }

    pub fn shows_mode_rail(self) -> bool {
        true
    }

    pub fn shows_layers_tree(self) -> bool {
        true
    }

    pub fn shows_tool_catalog(self) -> bool {
        true
    }

    pub fn to_workspace_id(self) -> WorkspaceId {
        match self {
            AppWorkspace::Layout | AppWorkspace::Review => WorkspaceId::World,
            AppWorkspace::Landforms => WorkspaceId::Sculpt,
            AppWorkspace::Biomes => WorkspaceId::Biomes,
            AppWorkspace::Hydrology => WorkspaceId::Simulation,
            AppWorkspace::Surface => WorkspaceId::Surface,
            AppWorkspace::Advanced => WorkspaceId::Objects,
        }
    }

    pub fn from_workspace_id(id: WorkspaceId) -> Self {
        match id {
            WorkspaceId::World => AppWorkspace::Layout,
            WorkspaceId::Sculpt | WorkspaceId::Develop => AppWorkspace::Landforms,
            WorkspaceId::Biomes => AppWorkspace::Biomes,
            WorkspaceId::Rules => AppWorkspace::Advanced,
            WorkspaceId::Simulation => AppWorkspace::Hydrology,
            WorkspaceId::Surface => AppWorkspace::Surface,
            WorkspaceId::Objects | WorkspaceId::AllTools => AppWorkspace::Advanced,
        }
    }

    pub fn from_digit(digit: u8) -> Option<Self> {
        WorkspaceId::from_digit(digit).map(Self::from_workspace_id)
    }

    pub fn digit_shortcut(self) -> u8 {
        self.to_workspace_id().digit_shortcut().unwrap_or(1)
    }

    pub fn shortcut_label(self) -> &'static str {
        match self.digit_shortcut() {
            1 => "1",
            2 => "2",
            3 => "3",
            4 => "4",
            5 => "5",
            6 => "6",
            7 => "7",
            _ => "",
        }
    }

    pub fn default_workspace_mode(self) -> WorkspaceMode {
        match self {
            AppWorkspace::Layout | AppWorkspace::Landforms | AppWorkspace::Review => {
                WorkspaceMode::Terrain
            }
            AppWorkspace::Biomes => WorkspaceMode::Biomes,
            AppWorkspace::Hydrology => WorkspaceMode::Simulation,
            AppWorkspace::Surface => WorkspaceMode::Materials,
            AppWorkspace::Advanced => WorkspaceMode::Objects,
        }
    }
}

// Session presentation state

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BrushWorkspaceState {
    pub radius: f32,
    pub strength: f32,
    pub falloff: f32,
    pub spacing: f32,
    pub flow: f32,
    pub invert: bool,
    pub symmetry: bool,
}

impl Default for BrushWorkspaceState {
    fn default() -> Self {
        Self {
            radius: 0.04,
            strength: 4.0,
            falloff: 0.5,
            spacing: 0.1,
            flow: 1.0,
            invert: false,
            symmetry: false,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TempSoloState {
    pub layer: Option<LayerId>,
    pub biome_group: Option<LayerId>,
}

impl TempSoloState {
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn is_empty(&self) -> bool {
        self.layer.is_none() && self.biome_group.is_none()
    }
}

/// Editor-only presentation. Must not contain terrain-generation data.
#[derive(Debug, Clone)]
pub struct WorkspaceState {
    pub active: WorkspaceId,
    pub mode: WorkspaceMode,
    pub app_workspace: AppWorkspace,
    pub editor_tool: EditorTool,
    pub viewport_mode: Preview2dMode,
    pub viewport_overlays: ViewportOverlayFlags,
    pub lighting_preset: LightingPreset,
    /// Whether the live lighting has been edited away from `lighting_preset`.
    /// Carried alongside the preset so switching workspaces preserves a custom
    /// look instead of snapping back to the preset on the next redraw.
    pub lighting_customized: bool,
    pub inspector_advanced: bool,
    pub brush: BrushWorkspaceState,
    pub biome_paint_tool: BiomePaintTool,
    pub mask_paint_tool: MaskPaintTool,
    pub selected_mask: Option<MaskId>,
    pub paint_mask: Option<MaskId>,
    pub biome_color_preview: bool,
    pub biome_focus: Option<BiomeDefinitionId>,
    pub temp_solo: TempSoloState,
    pub show_pipeline: bool,
    pub show_history: bool,
    pub build_progress: Option<f32>,
    pub quality: PreviewQuality,
    pub draft_displayed: bool,
    pub refining: bool,
    pub refining_layer_name: Option<String>,
    /// Camera Ã¢â‚¬â€ preserved across workspace switches (copied for snapshots).
    pub camera_xz: (f32, f32),
    pub camera_yaw: f32,
    pub camera_pitch: f32,
}

impl Default for WorkspaceState {
    fn default() -> Self {
        Self {
            active: WorkspaceId::Sculpt,
            mode: WorkspaceMode::Terrain,
            app_workspace: AppWorkspace::Landforms,
            editor_tool: EditorTool::Move,
            viewport_mode: Preview2dMode::Lit,
            viewport_overlays: ViewportOverlayFlags::default(),
            lighting_preset: LightingPreset::Studio,
            lighting_customized: false,
            inspector_advanced: false,
            brush: BrushWorkspaceState::default(),
            biome_paint_tool: BiomePaintTool::default(),
            mask_paint_tool: MaskPaintTool::default(),
            selected_mask: None,
            paint_mask: None,
            biome_color_preview: false,
            biome_focus: None,
            temp_solo: TempSoloState::default(),
            show_pipeline: false,
            show_history: false,
            build_progress: None,
            quality: PreviewQuality::Draft,
            draft_displayed: false,
            refining: false,
            refining_layer_name: None,
            camera_xz: (0.5, 0.5),
            camera_yaw: 0.0,
            camera_pitch: 0.0,
        }
    }
}

impl WorkspaceState {
    /// Switch task workspace. Presentation only Ã¢â‚¬â€ preserves camera / temp solo.
    pub fn switch_workspace(&mut self, id: WorkspaceId) {
        // All Tools is no longer in the rail; remap legacy prefs / keybinds.
        let id = if matches!(id, WorkspaceId::AllTools) {
            WorkspaceId::Objects
        } else {
            id
        };
        let def = workspace_definition(id);
        let camera_xz = self.camera_xz;
        let camera_yaw = self.camera_yaw;
        let camera_pitch = self.camera_pitch;
        let brush = self.brush;
        let temp_solo = self.temp_solo.clone();
        let biome_focus = self.biome_focus;
        let selected_mask = self.selected_mask;
        let paint_mask = self.paint_mask;
        let lighting = self.lighting_preset;
        let lighting_customized = self.lighting_customized;
        let inspector_advanced = self.inspector_advanced;

        self.active = id;
        self.app_workspace = AppWorkspace::from_workspace_id(id);
        self.mode = match id {
            WorkspaceId::World => WorkspaceMode::Utilities,
            WorkspaceId::Sculpt => WorkspaceMode::Terrain,
            WorkspaceId::Develop => WorkspaceMode::Filters,
            WorkspaceId::Biomes => WorkspaceMode::Biomes,
            WorkspaceId::Rules => WorkspaceMode::Masks,
            WorkspaceId::Simulation => WorkspaceMode::Simulation,
            WorkspaceId::Surface => WorkspaceMode::Materials,
            WorkspaceId::Objects => WorkspaceMode::Objects,
            WorkspaceId::AllTools => self.mode, // keep last catalog tab for compat
        };

        if let Some(mode) = def.preferred_viewport_mode {
            // Rules prefers mask tools but entering full Mask dock is opt-in;
            // use Masks preview suggestion without forcing mask editor session
            // except when already in mask view.
            if !matches!(mode, Preview2dMode::Masks) || matches!(id, WorkspaceId::Rules) {
                if !matches!(mode, Preview2dMode::Masks) {
                    self.viewport_mode = mode;
                } else {
                    self.viewport_mode = Preview2dMode::Masks;
                }
            }
        }
        if let Some(tool) = def.preferred_editor_tool {
            self.editor_tool = tool;
        }
        self.biome_color_preview = def.biome_color_preview;
        // Merge preferred overlays as suggestions (set flags that def enables).
        merge_overlay_suggestions(&mut self.viewport_overlays, def.preferred_overlays);

        // Restore preserved presentation.
        self.camera_xz = camera_xz;
        self.camera_yaw = camera_yaw;
        self.camera_pitch = camera_pitch;
        self.brush = brush;
        self.temp_solo = temp_solo;
        self.biome_focus = biome_focus;
        self.selected_mask = selected_mask;
        self.paint_mask = paint_mask;
        self.lighting_preset = lighting;
        self.lighting_customized = lighting_customized;
        self.inspector_advanced = inspector_advanced;
    }

    pub fn switch_mode(&mut self, mode: WorkspaceMode) {
        self.switch_workspace(mode.to_workspace_id());
    }

    pub fn switch_app_workspace(&mut self, workspace: AppWorkspace) {
        self.switch_workspace(workspace.to_workspace_id());
    }

    pub fn set_biome_color_preview(&mut self, on: bool) {
        self.biome_color_preview = on;
    }
}

fn merge_overlay_suggestions(dst: &mut ViewportOverlayFlags, src: ViewportOverlayFlags) {
    if src.grid {
        dst.grid = true;
    }
    if src.world_bounds {
        dst.world_bounds = true;
    }
    if src.water_level {
        dst.water_level = true;
    }
    if src.contours {
        dst.contours = true;
    }
    if src.brush_preview {
        dst.brush_preview = true;
    }
    if src.mask_overlay {
        dst.mask_overlay = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn definitions_cover_all_ids() {
        for id in WorkspaceId::ALL {
            assert_eq!(workspace_definition(id).id, id);
            assert!(!workspace_definition(id).name.is_empty());
        }
    }

    #[test]
    fn all_tools_legacy_still_defined() {
        let def = workspace_definition(WorkspaceId::AllTools);
        assert!(matches!(def.tools, WorkspaceToolFilter::All));
        assert!(!WorkspaceId::ALL.contains(&WorkspaceId::AllTools));
    }

    #[test]
    fn switch_remaps_all_tools_to_objects() {
        let mut ws = WorkspaceState::default();
        ws.switch_workspace(WorkspaceId::AllTools);
        assert_eq!(ws.active, WorkspaceId::Objects);
    }

    #[test]
    fn switch_preserves_camera_and_brush() {
        let mut ws = WorkspaceState::default();
        ws.camera_xz = (0.25, 0.75);
        ws.camera_yaw = 1.2;
        ws.brush.radius = 0.09;
        ws.switch_workspace(WorkspaceId::Biomes);
        assert_eq!(ws.active, WorkspaceId::Biomes);
        assert_eq!(ws.camera_xz, (0.25, 0.75));
        assert!((ws.camera_yaw - 1.2).abs() < 1e-6);
        assert!((ws.brush.radius - 0.09).abs() < 1e-6);
        assert!(ws.biome_color_preview);
    }

    #[test]
    fn no_wizard_wording_in_names() {
        for def in all_workspace_definitions() {
            let n = def.name.to_ascii_lowercase();
            assert!(!n.contains("step"));
            assert!(!n.contains("stage"));
            assert!(!n.contains("continue"));
        }
    }

    #[test]
    fn parse_persisted_ids() {
        assert_eq!(WorkspaceId::parse("sculpt"), Some(WorkspaceId::Sculpt));
        assert_eq!(WorkspaceId::parse("filters"), Some(WorkspaceId::Develop));
        assert_eq!(WorkspaceId::parse("mask"), Some(WorkspaceId::Rules));
        assert_eq!(WorkspaceId::parse("all_tools"), Some(WorkspaceId::AllTools));
        assert_eq!(WorkspaceId::parse("materials"), Some(WorkspaceId::Surface));
    }
}
