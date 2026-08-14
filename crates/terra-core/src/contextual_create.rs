//! Contextual creation — owner inference and execute paths for viewport / hierarchy Add.
//!
//! Artists create entities from any workspace without a fixed sequence. Owners are
//! inferred from selection, cursor, and active Biome — never silently attached
//! when the context is ambiguous.

use crate::biome_paint::BiomeLayer;
use crate::command::EditorCommand;
use crate::document::EditorSession;
use crate::layer::{Layer, LayerGroup, LayerId, LayerKind, StackCategory, StackNode};
use crate::matter_sim::{MatterSimConfig, MatterType};
use crate::operation_placement::{create_develop_operation, DevelopCategory};
use crate::shape_object::{ShapeKind, ShapeObject};
use crate::simulation_scenario::{SimulationScenarioCommand, SimulationScenarioId};
use crate::world_rules::{WorldRule, WorldRuleCommand, WorldRuleId};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Kinds & owners
// ---------------------------------------------------------------------------

/// Create-here choices offered in viewport / hierarchy context menus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CreateKind {
    SculptLayer,
    Biome,
    BiomePlacement,
    TerrainEffect,
    MaterialRule,
    WorldRule,
    SimulationSource,
    SimulationDomain,
    ScatterArea,
    Object,
    RiverSource,
    SnowSource,
    SandSource,
    DebrisSource,
}

impl CreateKind {
    pub fn all() -> &'static [CreateKind] {
        &[
            Self::SculptLayer,
            Self::Biome,
            Self::BiomePlacement,
            Self::TerrainEffect,
            Self::MaterialRule,
            Self::WorldRule,
            Self::SimulationSource,
            Self::SimulationDomain,
            Self::ScatterArea,
            Self::Object,
            Self::RiverSource,
            Self::SnowSource,
            Self::SandSource,
            Self::DebrisSource,
        ]
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::SculptLayer => "Noise Layer",
            Self::Biome => "Biome",
            Self::BiomePlacement => "Biome Placement",
            Self::TerrainEffect => "Terrain Effect",
            Self::MaterialRule => "Material Rule",
            Self::WorldRule => "World Rule",
            Self::SimulationSource => "Simulation Source",
            Self::SimulationDomain => "Simulation Scenario",
            Self::ScatterArea => "Scatter Area",
            Self::Object => "Object",
            Self::RiverSource => "River Source",
            Self::SnowSource => "Snow Source",
            Self::SandSource => "Sand Source",
            Self::DebrisSource => "Debris Source",
        }
    }

    /// Workspace that emphasises this entity (presentation only).
    pub fn home_workspace(self) -> CreateWorkspace {
        match self {
            Self::SculptLayer => CreateWorkspace::Sculpt,
            Self::Biome | Self::BiomePlacement => CreateWorkspace::Biomes,
            Self::TerrainEffect => CreateWorkspace::Develop,
            Self::MaterialRule => CreateWorkspace::Surface,
            Self::WorldRule => CreateWorkspace::Rules,
            Self::SimulationSource
            | Self::SimulationDomain
            | Self::RiverSource
            | Self::SnowSource
            | Self::SandSource
            | Self::DebrisSource => CreateWorkspace::Simulation,
            Self::ScatterArea | Self::Object => CreateWorkspace::Objects,
        }
    }

    pub fn default_name(self) -> &'static str {
        match self {
            Self::SculptLayer => "Noise Layer",
            Self::Biome => "Biome",
            Self::BiomePlacement => "Biome Placement",
            Self::TerrainEffect => "Terrain Effect",
            Self::MaterialRule => "Material",
            Self::WorldRule => "World Rule",
            Self::SimulationSource => "Simulation Source",
            Self::SimulationDomain => "Simulation Scenario",
            Self::ScatterArea => "Scatter",
            Self::Object => "Object",
            Self::RiverSource => "River Source",
            Self::SnowSource => "Snow Source",
            Self::SandSource => "Sand Source",
            Self::DebrisSource => "Debris Source",
        }
    }
}

/// Structural owner for newly created content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CreateOwner {
    /// World-level (rules, scenarios).
    World,
    /// Document layer stack.
    Global,
    Biome(LayerId),
    /// Ambiguous — must be chosen explicitly before create.
    Unspecified,
}

impl CreateOwner {
    pub fn is_resolved(self) -> bool {
        !matches!(self, Self::Unspecified)
    }
}

/// Task workspace id for create filtering (mirrors UI `WorkspaceId`, kept in core for tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CreateWorkspace {
    World,
    Sculpt,
    Biomes,
    Develop,
    Rules,
    Simulation,
    Surface,
    Objects,
    AllTools,
}

impl CreateWorkspace {
    pub fn all() -> &'static [CreateWorkspace] {
        &[
            Self::World,
            Self::Sculpt,
            Self::Biomes,
            Self::Develop,
            Self::Rules,
            Self::Simulation,
            Self::Surface,
            Self::Objects,
            Self::AllTools,
        ]
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "world" | "layout" => Some(Self::World),
            "sculpt" | "terrain" | "landforms" => Some(Self::Sculpt),
            "biomes" => Some(Self::Biomes),
            "develop" | "filters" => Some(Self::Develop),
            "rules" | "masks" => Some(Self::Rules),
            "simulation" | "hydrology" | "erosion" => Some(Self::Simulation),
            "surface" | "materials" => Some(Self::Surface),
            "objects" | "scatter" | "advanced" => Some(Self::Objects),
            "all_tools" | "all" | "utilities" => Some(Self::AllTools),
            _ => None,
        }
    }
}

/// Viewport / hierarchy context used for inference and availability.
#[derive(Debug, Clone)]
pub struct CreateContext {
    pub workspace: CreateWorkspace,
    pub selected_layer: Option<LayerId>,
    pub active_biome: Option<LayerId>,
    pub cursor_uv: Option<(f32, f32)>,
    pub has_terrain: bool,
    pub empty_project: bool,
    /// When true, UI may switch to the entity's home workspace after create.
    pub auto_switch_workspace: bool,
}

impl CreateContext {
    pub fn from_document(
        doc: &crate::document::TerrainDocument,
        workspace: CreateWorkspace,
        auto_switch_workspace: bool,
    ) -> Self {
        let empty_project = doc.stack.nodes.is_empty();
        let has_sculpt = |stack: &crate::layer::LayerStack| {
            stack
                .layer_ids()
                .iter()
                .any(|id| stack.find(*id).is_some_and(|l| l.kind.is_sculpt_base()))
        };
        let has_terrain = has_sculpt(&doc.stack);
        Self {
            workspace,
            selected_layer: doc.selected,
            active_biome: doc.active_biome,
            cursor_uv: None,
            has_terrain,
            empty_project,
            auto_switch_workspace,
        }
    }

    pub fn with_cursor(mut self, uv: Option<(f32, f32)>) -> Self {
        self.cursor_uv = uv;
        self
    }
}

/// Proposed owner plus display label and alternatives for override.
#[derive(Debug, Clone)]
pub struct OwnerProposal {
    pub owner: CreateOwner,
    pub label: String,
    pub ambiguous: bool,
    pub alternatives: Vec<(CreateOwner, String)>,
}

impl OwnerProposal {
    pub fn summary_line(&self, kind: CreateKind) -> String {
        format!("Create {} in:\n{}", kind.label(), self.label)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CreatedEntity {
    Layer(LayerId),
    BiomeGroup(LayerId),
    BiomePaint(crate::biome_paint::BiomeLayerId),
    WorldRule(WorldRuleId),
    Scenario(SimulationScenarioId),
    Shape(crate::shape_object::ShapeObjectId),
}

/// Preferred viewport tool after create (string keys mapped by UI).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreateToolHint {
    PaintBiome,
    Move,
    Raise,
    PaintMask,
    None,
}

#[derive(Debug, Clone)]
pub struct CreateOutcome {
    pub kind: CreateKind,
    pub owner: CreateOwner,
    pub entity: CreatedEntity,
    pub name: String,
    pub preferred_tool: CreateToolHint,
    pub preferred_inspector_section: Option<&'static str>,
    pub home_workspace: CreateWorkspace,
    pub should_switch_workspace: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CreateError {
    /// Kind requires an explicit owner (ambiguous Region / missing Biome).
    OwnerRequired {
        kind: CreateKind,
        reason: String,
    },
    /// Context cannot support this kind (e.g. Terrain Effect with no Biome).
    InvalidContext {
        kind: CreateKind,
        reason: String,
    },
    Failed {
        kind: CreateKind,
        reason: String,
    },
}

impl CreateError {
    pub fn message(&self) -> String {
        match self {
            Self::OwnerRequired { kind, reason } => {
                format!("Choose an owner for {}: {}", kind.label(), reason)
            }
            Self::InvalidContext { kind, reason } => {
                format!("Cannot create {}: {}", kind.label(), reason)
            }
            Self::Failed { kind, reason } => {
                format!("Failed to create {}: {}", kind.label(), reason)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Availability
// ---------------------------------------------------------------------------

/// Whether `kind` can be offered given context (before owner resolution).
pub fn kind_available(kind: CreateKind, ctx: &CreateContext) -> bool {
    match kind {
        // Procedural Regions removed (WC single-stack).
        CreateKind::WorldRule => true,
        CreateKind::SimulationSource
        | CreateKind::SimulationDomain
        | CreateKind::RiverSource
        | CreateKind::SnowSource
        | CreateKind::SandSource
        | CreateKind::DebrisSource => true,
        CreateKind::Biome => true,
        CreateKind::BiomePlacement => !ctx.empty_project || ctx.active_biome.is_some(),
        CreateKind::SculptLayer => ctx.has_terrain || !ctx.empty_project,
        CreateKind::TerrainEffect | CreateKind::MaterialRule => {
            ctx.active_biome.is_some() || !ctx.empty_project
        }
        CreateKind::ScatterArea | CreateKind::Object => {
            ctx.active_biome.is_some() || !ctx.empty_project
        }
    }
}

/// Soft emphasis: kinds emphasised by the current workspace appear first.
pub fn available_kinds(ctx: &CreateContext) -> Vec<CreateKind> {
    let mut kinds: Vec<CreateKind> = CreateKind::all()
        .iter()
        .copied()
        .filter(|k| kind_available(*k, ctx))
        .collect();
    let home = |k: CreateKind| k.home_workspace() == ctx.workspace;
    kinds.sort_by_key(|k| if home(*k) { 0 } else { 1 });
    // Empty project: keep Region / World Rule / matter / Biome; drop develop ops that need biome.
    if ctx.empty_project {
        kinds.retain(|k| {
            matches!(
                k,
                CreateKind::WorldRule
                    | CreateKind::Biome
                    | CreateKind::SculptLayer
                    | CreateKind::SimulationSource
                    | CreateKind::SimulationDomain
                    | CreateKind::RiverSource
                    | CreateKind::SnowSource
                    | CreateKind::SandSource
                    | CreateKind::DebrisSource
            )
        });
    }
    kinds
}

/// Emphasised (not exclusive) kinds for a workspace — used for hierarchy Add shortcuts.
pub fn emphasised_kinds(workspace: CreateWorkspace) -> &'static [CreateKind] {
    match workspace {
        CreateWorkspace::World => &[CreateKind::Biome, CreateKind::WorldRule],
        CreateWorkspace::Sculpt => &[CreateKind::SculptLayer, CreateKind::Biome],
        CreateWorkspace::Biomes => &[CreateKind::Biome, CreateKind::BiomePlacement],
        CreateWorkspace::Develop => &[
            CreateKind::TerrainEffect,
            CreateKind::MaterialRule,
            CreateKind::ScatterArea,
            CreateKind::Object,
        ],
        CreateWorkspace::Rules => &[CreateKind::WorldRule, CreateKind::MaterialRule],
        CreateWorkspace::Simulation => &[
            CreateKind::SimulationSource,
            CreateKind::SimulationDomain,
            CreateKind::RiverSource,
            CreateKind::SnowSource,
            CreateKind::SandSource,
            CreateKind::DebrisSource,
        ],
        CreateWorkspace::Surface => &[CreateKind::MaterialRule, CreateKind::Biome],
        CreateWorkspace::Objects => &[CreateKind::ScatterArea, CreateKind::Object],
        CreateWorkspace::AllTools => CreateKind::all(),
    }
}

// ---------------------------------------------------------------------------
// Owner inference
// ---------------------------------------------------------------------------

fn biome_label(doc: &crate::document::TerrainDocument, id: LayerId) -> String {
    doc.stack
        .find_group(id)
        .map(|g| g.name.clone())
        .unwrap_or_else(|| "Biome".into())
}

fn biome_alternatives(doc: &crate::document::TerrainDocument) -> Vec<(CreateOwner, String)> {
    let mut alts = Vec::new();
    collect_biome_groups(&doc.stack.nodes, &mut alts);
    if alts.is_empty() {
        alts.push((CreateOwner::Unspecified, "(no biomes)".into()));
    }
    alts
}

fn collect_biome_groups(nodes: &[StackNode], out: &mut Vec<(CreateOwner, String)>) {
    for n in nodes {
        if let StackNode::Group(g) = n {
            if g.is_biome() {
                out.push((CreateOwner::Biome(g.id), g.name.clone()));
            }
            collect_biome_groups(&g.children, out);
        }
    }
}

/// Infer a sensible owner for `kind`. Never picks an arbitrary Region when several exist
/// and there is no selection / cursor / active signal.
pub fn propose_owner(
    doc: &crate::document::TerrainDocument,
    kind: CreateKind,
    ctx: &CreateContext,
) -> OwnerProposal {
    match kind {
        CreateKind::WorldRule => OwnerProposal {
            owner: CreateOwner::World,
            label: "World".into(),
            ambiguous: false,
            alternatives: vec![
                (CreateOwner::World, "World".into()),
                (CreateOwner::Global, "Global".into()),
            ],
        },
        CreateKind::SimulationSource
        | CreateKind::SimulationDomain
        | CreateKind::RiverSource
        | CreateKind::SnowSource
        | CreateKind::SandSource
        | CreateKind::DebrisSource => OwnerProposal {
            owner: CreateOwner::World,
            label: "World".into(),
            ambiguous: false,
            alternatives: vec![(CreateOwner::World, "World".into())],
        },
        CreateKind::TerrainEffect
        | CreateKind::MaterialRule
        | CreateKind::ScatterArea
        | CreateKind::Object
        | CreateKind::BiomePlacement => propose_biome_owner(doc, ctx),
        CreateKind::Biome => propose_biome_host(doc, ctx),
        CreateKind::SculptLayer => propose_sculpt_owner(doc, ctx),
    }
}

fn propose_biome_owner(
    doc: &crate::document::TerrainDocument,
    ctx: &CreateContext,
) -> OwnerProposal {
    let alts = biome_alternatives(doc);
    if let Some(id) = ctx.active_biome {
        if doc.stack.find_group(id).is_some_and(|g| g.is_biome()) {
            return OwnerProposal {
                owner: CreateOwner::Biome(id),
                label: biome_label(doc, id),
                ambiguous: false,
                alternatives: alts,
            };
        }
    }
    // Single biome in project → clear.
    if alts.len() == 1 {
        if let (CreateOwner::Biome(id), label) = &alts[0] {
            return OwnerProposal {
                owner: CreateOwner::Biome(*id),
                label: label.clone(),
                ambiguous: false,
                alternatives: alts,
            };
        }
    }
    OwnerProposal {
        owner: CreateOwner::Unspecified,
        label: "Choose a Biome…".into(),
        ambiguous: true,
        alternatives: alts,
    }
}

fn propose_biome_host(
    doc: &crate::document::TerrainDocument,
    _ctx: &CreateContext,
) -> OwnerProposal {
    // WC single-stack: biomes always host on the terrain stack.
    let _ = doc;
    OwnerProposal {
        owner: CreateOwner::World,
        label: "Terrain".into(),
        ambiguous: false,
        alternatives: vec![(CreateOwner::World, "Terrain".into())],
    }
}

fn propose_sculpt_owner(
    doc: &crate::document::TerrainDocument,
    _ctx: &CreateContext,
) -> OwnerProposal {
    let _ = doc;
    OwnerProposal {
        owner: CreateOwner::World,
        label: "Terrain".into(),
        ambiguous: false,
        alternatives: vec![(CreateOwner::World, "Terrain".into())],
    }
}

/// Whether `owner` is a valid structural host for `kind`.
pub fn owner_compatible(kind: CreateKind, owner: CreateOwner) -> bool {
    match kind {
        CreateKind::WorldRule => {
            matches!(owner, CreateOwner::World | CreateOwner::Global)
        }
        CreateKind::SimulationSource
        | CreateKind::SimulationDomain
        | CreateKind::RiverSource
        | CreateKind::SnowSource
        | CreateKind::SandSource
        | CreateKind::DebrisSource => {
            matches!(owner, CreateOwner::World | CreateOwner::Global)
        }
        CreateKind::TerrainEffect
        | CreateKind::MaterialRule
        | CreateKind::ScatterArea
        | CreateKind::Object
        | CreateKind::BiomePlacement => {
            matches!(owner, CreateOwner::Biome(_))
        }
        CreateKind::Biome => {
            matches!(owner, CreateOwner::Global | CreateOwner::World)
        }
        CreateKind::SculptLayer => {
            matches!(owner, CreateOwner::Global | CreateOwner::World)
        }
    }
}

/// Apply an explicit owner override; validates that the owner exists and fits the kind.
pub fn override_owner(
    doc: &crate::document::TerrainDocument,
    kind: CreateKind,
    owner: CreateOwner,
) -> Result<OwnerProposal, CreateError> {
    if !owner_compatible(kind, owner) {
        return Err(CreateError::InvalidContext {
            kind,
            reason: "owner type is not valid for this create action".into(),
        });
    }
    match owner {
        CreateOwner::Unspecified => Err(CreateError::OwnerRequired {
            kind,
            reason: "owner is unspecified".into(),
        }),
        CreateOwner::Biome(id) => {
            if !doc.stack.find_group(id).is_some_and(|g| g.is_biome()) {
                return Err(CreateError::InvalidContext {
                    kind,
                    reason: "biome not found".into(),
                });
            }
            Ok(OwnerProposal {
                owner,
                label: biome_label(doc, id),
                ambiguous: false,
                alternatives: biome_alternatives(doc),
            })
        }
        CreateOwner::Global | CreateOwner::World => Ok(OwnerProposal {
            owner,
            label: "Terrain".into(),
            ambiguous: false,
            alternatives: vec![(CreateOwner::World, "Terrain".into())],
        }),
    }
}

/// Resolve the owner used for create: override if set, else proposal (must be resolved).
pub fn resolve_owner(
    doc: &crate::document::TerrainDocument,
    kind: CreateKind,
    ctx: &CreateContext,
    owner_override: Option<CreateOwner>,
) -> Result<OwnerProposal, CreateError> {
    if let Some(o) = owner_override {
        return override_owner(doc, kind, o);
    }
    let proposal = propose_owner(doc, kind, ctx);
    if !proposal.owner.is_resolved() {
        return Err(CreateError::OwnerRequired {
            kind,
            reason: proposal.label.clone(),
        });
    }
    Ok(proposal)
}

// ---------------------------------------------------------------------------
// Execute
// ---------------------------------------------------------------------------

/// Create an entity under the resolved owner. Integrates with undo stacks.
pub fn execute_create(
    session: &mut EditorSession,
    kind: CreateKind,
    ctx: &CreateContext,
    owner_override: Option<CreateOwner>,
    name: Option<String>,
) -> Result<CreateOutcome, CreateError> {
    let proposal = resolve_owner(&session.document, kind, ctx, owner_override)?;
    let owner = proposal.owner;
    let name = name.unwrap_or_else(|| kind.default_name().to_string());
    let home = kind.home_workspace();
    let should_switch = ctx.auto_switch_workspace && home != ctx.workspace;

    let outcome = match kind {
        CreateKind::SculptLayer => create_sculpt_layer(session, owner, &name)?,
        CreateKind::Biome => create_biome(session, owner, &name)?,
        CreateKind::BiomePlacement => create_biome_placement(session, owner, &name)?,
        CreateKind::TerrainEffect => {
            create_develop(session, owner, DevelopCategory::Terrain, &name)?
        }
        CreateKind::MaterialRule => {
            create_develop(session, owner, DevelopCategory::Materials, &name)?
        }
        CreateKind::WorldRule => create_world_rule(session, &name)?,
        CreateKind::SimulationSource | CreateKind::SimulationDomain => {
            create_matter(session, MatterType::WaterRivers, &name)?
        }
        CreateKind::ScatterArea => {
            create_develop(session, owner, DevelopCategory::Vegetation, &name)?
        }
        CreateKind::Object => create_develop(session, owner, DevelopCategory::Objects, &name)?,
        CreateKind::RiverSource => create_matter(session, MatterType::WaterRivers, &name)?,
        CreateKind::SnowSource => create_matter(session, MatterType::Snow, &name)?,
        CreateKind::SandSource => create_matter(session, MatterType::Sand, &name)?,
        CreateKind::DebrisSource => create_matter(session, MatterType::Debris, &name)?,
    };

    Ok(CreateOutcome {
        kind,
        owner,
        entity: outcome.0,
        name: outcome.1,
        preferred_tool: outcome.2,
        preferred_inspector_section: outcome.3,
        home_workspace: home,
        should_switch_workspace: should_switch,
    })
}

fn create_sculpt_layer(
    session: &mut EditorSession,
    owner: CreateOwner,
    name: &str,
) -> Result<(CreatedEntity, String, CreateToolHint, Option<&'static str>), CreateError> {
    let layer = Layer::new(
        name,
        LayerKind::NoiseValue(crate::layer::NoiseParams::default()),
    );
    let id = layer.id();
    match owner {
        CreateOwner::Global | CreateOwner::World => {
            session.document.stack.ensure_category_folders();
            let index = session.document.stack.nodes.len();
            if let Some(folder) = session
                .document
                .stack
                .find_category_mut(StackCategory::Shape)
            {
                folder.children.push(StackNode::Layer(layer.clone()));
            } else {
                session.document.add_shape_layer(layer.clone());
            }
            session
                .history
                .push_executed(EditorCommand::AddLayer { layer, index });
            session.document.selected = Some(id);
        }
        CreateOwner::Biome(_) | CreateOwner::Unspecified => {
            return Err(CreateError::InvalidContext {
                kind: CreateKind::SculptLayer,
                reason: "sculpt layers require terrain ownership".into(),
            });
        }
    }
    Ok((
        CreatedEntity::Layer(id),
        name.into(),
        CreateToolHint::Raise,
        Some("noise"),
    ))
}

fn create_biome(
    session: &mut EditorSession,
    _owner: CreateOwner,
    name: &str,
) -> Result<(CreatedEntity, String, CreateToolHint, Option<&'static str>), CreateError> {
    let biome = LayerGroup::biome(name);
    let id = biome.id;
    session.document.stack.ensure_category_folders();
    if let Some(folder) = session
        .document
        .stack
        .find_category_mut(StackCategory::Surface)
    {
        folder.children.push(StackNode::Group(biome));
    } else {
        session.document.stack.push_group(biome);
    }
    session.history.push_executed(EditorCommand::AddGroup {
        name: name.into(),
        id,
        index: 0,
    });
    session.document.active_biome = Some(id);
    session.document.selected = Some(id);
    Ok((
        CreatedEntity::BiomeGroup(id),
        name.into(),
        CreateToolHint::PaintBiome,
        Some("general"),
    ))
}

fn create_biome_placement(
    session: &mut EditorSession,
    owner: CreateOwner,
    name: &str,
) -> Result<(CreatedEntity, String, CreateToolHint, Option<&'static str>), CreateError> {
    let CreateOwner::Biome(biome_id) = owner else {
        return Err(CreateError::OwnerRequired {
            kind: CreateKind::BiomePlacement,
            reason: "biome placement requires a Biome owner".into(),
        });
    };
    let mut bl = BiomeLayer::new(name);
    bl.show_biome_colors = true;
    let id = bl.id;
    session.document.biome_layers.push(bl);
    session.document.selected_biome_layer = Some(id);
    session.document.active_biome = Some(biome_id);
    session.document.selected = Some(biome_id);
    session.history.push_executed(EditorCommand::Annotate {
        label: format!("Added biome placement {name}"),
    });
    Ok((
        CreatedEntity::BiomePaint(id),
        name.into(),
        CreateToolHint::PaintBiome,
        Some("general"),
    ))
}

fn create_develop(
    session: &mut EditorSession,
    owner: CreateOwner,
    category: DevelopCategory,
    name: &str,
) -> Result<(CreatedEntity, String, CreateToolHint, Option<&'static str>), CreateError> {
    let CreateOwner::Biome(biome_id) = owner else {
        return Err(CreateError::OwnerRequired {
            kind: match category {
                DevelopCategory::Terrain => CreateKind::TerrainEffect,
                DevelopCategory::Materials => CreateKind::MaterialRule,
                DevelopCategory::Vegetation => CreateKind::ScatterArea,
                DevelopCategory::Objects => CreateKind::Object,
                _ => CreateKind::TerrainEffect,
            },
            reason: "requires a Biome owner".into(),
        });
    };
    let Some(section) = category.biome_section() else {
        session.document.selected = Some(biome_id);
        return Err(CreateError::InvalidContext {
            kind: CreateKind::BiomePlacement,
            reason: "placement opens biome inspector".into(),
        });
    };
    if !session
        .document
        .stack
        .find_group(biome_id)
        .is_some_and(|g| g.is_biome())
    {
        return Err(CreateError::InvalidContext {
            kind: CreateKind::TerrainEffect,
            reason: "biome group missing".into(),
        });
    }
    let layer = create_develop_operation(category, name);
    let id = layer.id();
    if let Some(biome) = session.document.stack.find_group_mut(biome_id) {
        biome.ensure_biome_sections();
        if let Some(sec) = biome.find_section_mut(section) {
            sec.children.push(StackNode::Layer(layer.clone()));
        } else {
            biome.children.push(StackNode::Layer(layer.clone()));
        }
    }
    session
        .history
        .push_executed(EditorCommand::AddLayer { layer, index: 0 });
    session.document.selected = Some(id);
    session.document.active_biome = Some(biome_id);
    Ok((
        CreatedEntity::Layer(id),
        name.into(),
        CreateToolHint::None,
        Some("general"),
    ))
}

fn create_world_rule(
    session: &mut EditorSession,
    name: &str,
) -> Result<(CreatedEntity, String, CreateToolHint, Option<&'static str>), CreateError> {
    let rule = WorldRule::new(name);
    let id = rule.id;
    let index = session.document.world_rules.rules.len();
    session.push_world_rule_command(WorldRuleCommand::Add { rule, index });
    session.document.world_rules.selected = Some(id);
    Ok((
        CreatedEntity::WorldRule(id),
        name.into(),
        CreateToolHint::None,
        Some("general"),
    ))
}

fn create_matter(
    session: &mut EditorSession,
    matter: MatterType,
    name: &str,
) -> Result<(CreatedEntity, String, CreateToolHint, Option<&'static str>), CreateError> {
    let cfg = MatterSimConfig::new(matter);
    let mut scenario = cfg.build_scenario();
    scenario.name = name.into();
    scenario.matter.push(cfg);
    let id = scenario.id;
    let index = session.document.simulation_scenarios.scenarios.len();
    session.push_scenario_command(SimulationScenarioCommand::Add { scenario, index });
    session.document.simulation_scenarios.selected = Some(id);
    session.document.simulation_scenarios.active = Some(id);
    Ok((
        CreatedEntity::Scenario(id),
        name.into(),
        CreateToolHint::PaintMask,
        Some("general"),
    ))
}

/// Optional shape create helper (sculpt landform under cursor).
pub fn execute_create_shape(
    session: &mut EditorSession,
    kind: ShapeKind,
    name: impl Into<String>,
    uv: Option<(f32, f32)>,
) -> CreatedEntity {
    let mut shape = ShapeObject::new(name, kind);
    if let Some((u, v)) = uv {
        use crate::authoring::SculptPoint;
        shape.points = vec![SculptPoint {
            u,
            v,
            pressure: 1.0,
        }];
    }
    let id = shape.id;
    session.document.shapes.push(shape);
    session.history.push_executed(EditorCommand::Annotate {
        label: "Added shape".into(),
    });
    CreatedEntity::Shape(id)
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::document::TerrainDocument;

    #[test]
    fn empty_project_limits_kinds() {
        let mut doc = TerrainDocument::new_default();
        doc.stack.nodes.clear();
        doc.active_biome = None;
        let ctx = CreateContext::from_document(&doc, CreateWorkspace::World, false);
        assert!(ctx.empty_project);
        let kinds = available_kinds(&CreateContext {
            empty_project: true,
            ..ctx
        });
        assert!(kinds.contains(&CreateKind::WorldRule));
        assert!(kinds.contains(&CreateKind::Biome));
        assert!(!kinds.contains(&CreateKind::TerrainEffect));
    }
}
