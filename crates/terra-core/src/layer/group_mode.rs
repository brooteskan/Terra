//! Explicit group evaluation and input modes.

use super::operation::OperationCategory;
use crate::fields::FieldId;
use crate::ids::OutputId;
use serde::{Deserialize, Serialize};

/// Default biome swatch tint (muted olive green) shown in lists / overlays.
pub fn default_preview_color() -> [f32; 3] {
    [0.4, 0.55, 0.35]
}

/// Deterministic biome swatch from a stable seed (layer id / name hash).
///
/// Used when a biome has no authored preview colour so every biome still
/// gets a distinct, stable tint in lists and overlays.
pub fn palette_preview_color(seed: u128) -> [f32; 3] {
    const PALETTE: [[f32; 3]; 8] = [
        [0.45, 0.72, 0.28], // grassland
        [0.18, 0.42, 0.72], // water
        [0.86, 0.78, 0.55], // beach
        [0.58, 0.52, 0.48], // rock
        [0.78, 0.48, 0.28], // desert
        [0.55, 0.38, 0.68], // heath
        [0.28, 0.62, 0.55], // wetland
        [0.72, 0.38, 0.42], // clay
    ];
    PALETTE[(seed as usize) % PALETTE.len()]
}

/// True when `c` matches the serde/new-group default olive (untinted).
pub fn is_default_preview_color(c: [f32; 3]) -> bool {
    let d = default_preview_color();
    (c[0] - d[0]).abs() < 0.02 && (c[1] - d[1]).abs() < 0.02 && (c[2] - d[2]).abs() < 0.02
}

/// World Creator–style terrain category folders under the root stack.
///
/// Evaluation remains one ordered tree (bottom → top). Folder roles organise
/// the UI and route Add Layer:
/// Foundation → Shape → Biomes → Mask → Simulation (global).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StackCategory {
    /// Sculpt / import foundation (usually lives as a root layer, not a folder).
    Foundation,
    /// Shape Layers folder (WC Terrain / Shape authoring).
    Shape,
    /// Optional global erosion / physical sims (WC "Simulation Layers").
    Simulation,
    /// Mask-producing / mask-bound authoring layers (WC "Mask Layers").
    Mask,
    /// Biome container root (WC "Biomes") — not climate classification.
    Surface,
}

impl StackCategory {
    pub fn label(self) -> &'static str {
        match self {
            StackCategory::Foundation => "Foundation",
            StackCategory::Shape => "Terrain",
            StackCategory::Simulation => "Simulation",
            StackCategory::Mask => "Masks",
            StackCategory::Surface => "Biomes",
        }
    }

    pub fn short_badge(self) -> &'static str {
        match self {
            StackCategory::Foundation => "BASE",
            StackCategory::Shape => "TERRAIN",
            StackCategory::Simulation => "SIM",
            StackCategory::Mask => "MASKS",
            StackCategory::Surface => "BIOMES",
        }
    }

    /// Preferred evaluation order among category folders (lower = earlier / bottom).
    pub fn stack_order(self) -> u8 {
        match self {
            StackCategory::Foundation => 0,
            StackCategory::Shape => 1,
            StackCategory::Surface => 2,
            StackCategory::Mask => 3,
            StackCategory::Simulation => 4,
        }
    }

    pub fn from_operation(cat: OperationCategory) -> Self {
        match cat {
            OperationCategory::Generator
            | OperationCategory::ImportedData
            | OperationCategory::Modifier => StackCategory::Shape,
            OperationCategory::Simulation | OperationCategory::Analysis => {
                StackCategory::Simulation
            }
            OperationCategory::Surface => StackCategory::Surface,
            OperationCategory::Group | OperationCategory::Organisation => StackCategory::Shape,
        }
    }

    /// Root category folders created for new / migrated documents (v4+).
    pub fn all_folders() -> &'static [StackCategory] {
        &[
            StackCategory::Shape,
            StackCategory::Surface,
            StackCategory::Mask,
            StackCategory::Simulation,
        ]
    }
}

/// Structural role of a [`super::LayerGroup`] in the WC biome workflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum GroupKind {
    /// Generic / legacy group (no special role).
    #[default]
    Generic,
    /// Root category folder (Shape / Biomes / Mask / Simulation).
    CategoryFolder,
    /// World Creator biome container (isolated composite).
    Biome,
    /// Pass-through section inside a biome.
    BiomeSection(BiomeSection),
}

impl GroupKind {
    pub fn label(self) -> &'static str {
        match self {
            GroupKind::Generic => "Group",
            GroupKind::CategoryFolder => "Folder",
            GroupKind::Biome => "Biome",
            GroupKind::BiomeSection(s) => s.label(),
        }
    }

    pub fn short_badge(self) -> &'static str {
        match self {
            GroupKind::Generic => "GRP",
            GroupKind::CategoryFolder => "FOLDER",
            GroupKind::Biome => "BIOME",
            GroupKind::BiomeSection(s) => s.short_badge(),
        }
    }
}

/// Typed child sections inside a biome container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BiomeSection {
    Filters,
    Materials,
    Objects,
    LocalSims,
}

impl BiomeSection {
    pub fn label(self) -> &'static str {
        match self {
            BiomeSection::Filters => "Filters",
            BiomeSection::Materials => "Materials",
            BiomeSection::Objects => "Objects",
            BiomeSection::LocalSims => "Local Sims",
        }
    }

    pub fn short_badge(self) -> &'static str {
        match self {
            BiomeSection::Filters => "FILT",
            BiomeSection::Materials => "MAT",
            BiomeSection::Objects => "OBJ",
            BiomeSection::LocalSims => "LSIM",
        }
    }

    pub fn all() -> &'static [BiomeSection] {
        &[
            BiomeSection::Filters,
            BiomeSection::Materials,
            BiomeSection::Objects,
            BiomeSection::LocalSims,
        ]
    }

    /// Preferred section for an operation when adding into a biome.
    pub fn for_operation(cat: OperationCategory) -> Self {
        match cat {
            OperationCategory::Surface => BiomeSection::Materials,
            // Erosion filters live under Filters in WC; LocalSims is sand/debris/snow/fluid.
            OperationCategory::Simulation | OperationCategory::Analysis => BiomeSection::Filters,
            _ => BiomeSection::Filters,
        }
    }

    /// Prefer Objects for vegetation-like surface ops (callers can override).
    pub fn for_layer_kind_hint(kind_name: &str) -> Option<Self> {
        let n = kind_name.to_ascii_lowercase();
        if n.contains("vegetation") || n.contains("scatter") || n.contains("object") {
            Some(BiomeSection::Objects)
        } else if n.contains("sand")
            || n.contains("debris")
            || n.contains("snow")
            || n.contains("fluid")
        {
            Some(BiomeSection::LocalSims)
        } else if n.contains("material") {
            Some(BiomeSection::Materials)
        } else {
            None
        }
    }
}

/// How a group's children participate in the live terrain context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum GroupEvalMode {
    /// Children modify the live context sequentially (organisational folder).
    #[default]
    PassThrough,
    /// Private working context; composite result back onto the parent.
    IsolatedComposite,
}

impl GroupEvalMode {
    pub fn label(self) -> &'static str {
        match self {
            GroupEvalMode::PassThrough => "Pass-through",
            GroupEvalMode::IsolatedComposite => "Isolated",
        }
    }
}

/// How an isolated group initialises its private context.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum GroupInputMode {
    /// Private copy (or COW view) of the terrain beneath the group.
    #[default]
    CopyInput,
    /// Empty height field; generate independently before compositing.
    EmptyHeight,
    /// Initialise from a chosen field or named layer/group output.
    SelectedField(SelectedGroupInput),
}

impl GroupInputMode {
    pub fn label(&self) -> &'static str {
        match self {
            GroupInputMode::CopyInput => "Copy input",
            GroupInputMode::EmptyHeight => "Empty height",
            GroupInputMode::SelectedField(_) => "Selected field",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SelectedGroupInput {
    Field(FieldId),
    Output(OutputId),
}

/// Soft transition settings for region-scoped groups.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionSettings {
    #[serde(default)]
    pub width: f32,
    #[serde(default)]
    pub falloff: FalloffCurve,
    #[serde(default)]
    pub edge_breakup: f32,
    #[serde(default)]
    pub expand: f32,
    #[serde(default)]
    pub contract: f32,
    #[serde(default = "default_true")]
    pub world_units: bool,
}

fn default_true() -> bool {
    true
}

impl Default for TransitionSettings {
    fn default() -> Self {
        Self {
            width: 0.0,
            falloff: FalloffCurve::Smoothstep,
            edge_breakup: 0.0,
            expand: 0.0,
            contract: 0.0,
            world_units: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FalloffCurve {
    #[default]
    Linear,
    Smoothstep,
    Smooth,
}

impl FalloffCurve {
    pub fn apply(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            FalloffCurve::Linear => t,
            FalloffCurve::Smoothstep | FalloffCurve::Smooth => t * t * (3.0 - 2.0 * t),
        }
    }
}
