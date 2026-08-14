//! Panel and mask-edit actions emitted by the editor UI.

use terra_core::layer::{Layer, LayerId, LayerKind};
use terra_core::mask::{MaskAsset, MaskId, MaskPaintTool};

#[derive(Debug, Default)]
pub struct TerrainSettingsUpdate {
    pub export_resolution: Option<u32>,
    pub preview_resolution: Option<u32>,
    pub precision: Option<f32>,
    pub world_scale: Option<f32>,
    pub max_level: Option<u32>,
    pub high_detail: Option<terra_core::analyze::HighDetailMode>,
    pub show_hd_outline: Option<bool>,
}

impl TerrainSettingsUpdate {
    pub fn is_empty(&self) -> bool {
        self.export_resolution.is_none()
            && self.preview_resolution.is_none()
            && self.precision.is_none()
            && self.world_scale.is_none()
            && self.max_level.is_none()
            && self.high_detail.is_none()
            && self.show_hd_outline.is_none()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum MaskEditAction {
    Clear,
    Fill,
    FlipX,
    FlipY,
    RotateLeft,
    RotateRight,
}

#[derive(Debug)]
pub enum PanelAction {
    UpdateTerrainSettings(TerrainSettingsUpdate),
    AddLayer(Layer),
    /// Add a layer into a specific World Creator—style category folder.
    AddLayerToCategory {
        category: terra_core::layer::StackCategory,
        layer: Layer,
    },
    RemoveSelected,
    DuplicateSelected,
    Reorder {
        from: usize,
        to: usize,
    },
    Select(LayerId),
    SetEnabled {
        id: LayerId,
        enabled: bool,
    },
    SetOpacity {
        id: LayerId,
        opacity: f32,
    },
    SetBlend {
        id: LayerId,
        blend: terra_core::layer::BlendMode,
    },
    SetKind {
        id: LayerId,
        kind: LayerKind,
    },
    Rename {
        id: LayerId,
        name: String,
    },
    ApplyPreset(String),
    AddMask(MaskAsset),
    UpdateMaskAsset(MaskAsset),
    SelectMask(MaskId),
    BindMaskToLayer {
        layer: LayerId,
        mask: MaskId,
    },
    UnbindMask {
        layer: LayerId,
        mask: MaskId,
    },
    UpdateMaskBinding {
        layer: LayerId,
        mask: MaskId,
        strength: f32,
        invert: bool,
    },
    CycleMaskCombine {
        layer: LayerId,
        mask: MaskId,
    },
    PaintMaskStamp {
        mask_id: MaskId,
        u: f32,
        v: f32,
        radius: f32,
        strength: f32,
        hardness: f32,
        tool: MaskPaintTool,
    },
    EditMask {
        mask_id: MaskId,
        action: MaskEditAction,
    },
    /// Stamp onto the sculptable Base height buffer.
    PaintSculptStamp {
        layer: LayerId,
        u: f32,
        v: f32,
        radius: f32,
        strength: f32,
        /// Non-destructive stroke kind stored on the Shape Layer.
        stroke_kind: terra_core::authoring::SculptStrokeKind,
        /// Absolute height for Flatten / HeightStamp (metres).
        target_height: f32,
    },
    /// Merge selected Shape history layers (SculptStrokes) into the first.
    MergeShapeLayers {
        keep: LayerId,
        others: Vec<LayerId>,
    },
    /// Set operation Apply Where (Develop biome-scoped placement).
    SetOperationApplyWhere {
        id: LayerId,
        apply: terra_core::operation_placement::ApplyWhere,
    },
    /// Update height/slope/flow parameters on operation placement.
    SetOperationPlacementParams {
        id: LayerId,
        height_min: Option<f32>,
        height_max: Option<f32>,
        slope_min: Option<f32>,
        slope_max: Option<f32>,
        flow_min: Option<f32>,
        near_distance_m: Option<f32>,
    },
    /// Move a layer into another biome section (Develop).
    MoveLayerToBiome {
        id: LayerId,
        biome: LayerId,
        section: terra_core::layer::BiomeSection,
    },
    /// Contextual Develop create under active biome.
    AddDevelopOperation {
        category: terra_core::operation_placement::DevelopCategory,
        name: String,
    },
    /// Open Advanced Mask Stack for a layer (preserves access).
    OpenLayerAdvancedMask(LayerId),
    /// Add a blank World Rule.
    AddWorldRule {
        name: String,
    },
    /// Instantiate a built-in World Rule preset.
    AddWorldRulePreset {
        name: String,
    },
    SelectWorldRule(terra_core::world_rules::WorldRuleId),
    SetWorldRuleEnabled {
        id: terra_core::world_rules::WorldRuleId,
        enabled: bool,
    },
    RemoveWorldRule(terra_core::world_rules::WorldRuleId),
    ReorderWorldRule {
        id: terra_core::world_rules::WorldRuleId,
        before: Option<terra_core::world_rules::WorldRuleId>,
    },
    SetWorldRuleScope {
        id: terra_core::world_rules::WorldRuleId,
        scope: terra_core::world_rules::WorldRuleScope,
    },
    SetWorldRulePhase {
        id: terra_core::world_rules::WorldRuleId,
        phase: Option<terra_core::world_rules::WorldRulePhase>,
    },
    ToggleWorldRuleEffect {
        rule: terra_core::world_rules::WorldRuleId,
        effect: String,
        enabled: bool,
    },
    /// Simulation Scenario actions.
    AddSimulationScenario {
        name: String,
    },
    AddContinentalHydrologyPreset,
    SelectSimulationScenario(terra_core::simulation_scenario::SimulationScenarioId),
    SetSimulationScenarioEnabled {
        id: terra_core::simulation_scenario::SimulationScenarioId,
        enabled: bool,
    },
    RunSimulationScenario(terra_core::simulation_scenario::SimulationScenarioId),
    CancelSimulationScenario(terra_core::simulation_scenario::SimulationScenarioId),
    ResetSimulationScenario(terra_core::simulation_scenario::SimulationScenarioId),
    RebuildSimulationScenario(terra_core::simulation_scenario::SimulationScenarioId),
    FreezeScenarioResult {
        id: terra_core::simulation_scenario::SimulationScenarioId,
        snapshot: Option<String>,
    },
    PreviewScenarioSnapshot {
        id: terra_core::simulation_scenario::SimulationScenarioId,
        snapshot: String,
    },
    CompareScenarioSnapshot {
        id: terra_core::simulation_scenario::SimulationScenarioId,
        snapshot: Option<String>,
    },
    ApplyScenarioOutputs {
        id: terra_core::simulation_scenario::SimulationScenarioId,
        snapshot: Option<String>,
    },
    AddScenarioPass {
        id: terra_core::simulation_scenario::SimulationScenarioId,
        kind: terra_core::simulation_scenario::ScenarioPassKind,
    },
    ReorderScenarioPass {
        id: terra_core::simulation_scenario::SimulationScenarioId,
        pass: String,
        to_index: usize,
    },
    /// Add a matter-sim preset scenario (Water / Snow / Sand / Debris).
    AddMatterSimScenario {
        matter: terra_core::matter_sim::MatterType,
    },
    /// Toggle source painting for the selected matter config.
    SetMatterSourcePaint {
        scenario: terra_core::simulation_scenario::SimulationScenarioId,
        matter_index: usize,
        enabled: bool,
    },
    /// Set selective apply fields for a matter config.
    SetMatterApplySelected {
        scenario: terra_core::simulation_scenario::SimulationScenarioId,
        matter_index: usize,
        fields: Vec<terra_core::fields::FieldId>,
    },
    /// Set Shape history edit mode (new layer vs continue selected).
    SetShapeEditMode(terra_core::shape_history::ShapeEditMode),
    /// Explicitly request full-quality commit after draft sculpt preview.
    CommitShapePreview,
    ResetSculptBase {
        id: LayerId,
    },
    MarkDirty(Option<LayerId>),
    SetLocked {
        id: LayerId,
        locked: bool,
    },
    SetSolo {
        id: LayerId,
        solo: bool,
    },
    SetColorTag {
        id: LayerId,
        tag: u8,
    },
    SetCached {
        id: LayerId,
        cached: bool,
    },
    AddGroup {
        name: String,
    },
    /// Add an isolated composite terrain group.
    AddIsolatedGroup {
        name: String,
    },
    /// Add a World Creatorâ€“style biome container under Biomes.
    AddBiome {
        name: String,
    },
    /// Set the active biome for paint / add-layer context.
    SetActiveBiome(LayerId),
    /// Append a DistNode to a layer or group's distribution.
    AddDistNode {
        target: LayerId,
        kind: terra_core::mask::DistNodeKind,
    },
    /// Append an effect as a child of the last DistNode (or as a new node).
    AddDistEffect {
        target: LayerId,
        kind: terra_core::mask::DistNodeKind,
    },
    /// Stamp into the selected biome paint / placement layer for `active_biome`.
    PaintBiomeStamp {
        biome: LayerId,
        u: f32,
        v: f32,
        radius: f32,
        strength: f32,
        erase: bool,
        /// Optional mode override (smooth / replace / add). None = paint/erase.
        mode: Option<terra_core::biome_paint::BiomePaintTool>,
    },
    /// Normalize weights on the selected placement layer.
    NormalizeBiomePlacement,
    /// Begin a biome paint stroke (capture undo snapshot).
    BeginBiomePaintStroke {
        biome: LayerId,
    },
    /// End / commit the current biome paint stroke.
    EndBiomePaintStroke,
    /// Ensure a default biome paint layer exists and select it.
    EnsureBiomePaintLayer,
    /// Add an empty hole pierce layer.
    AddHoleLayer {
        name: String,
    },
    /// Switch the editor brush tool.
    SetEditorTool(crate::ui::EditorTool),
    /// Parent the selected layer/group into the named group id.
    MoveIntoGroup {
        child: LayerId,
        group: LayerId,
    },
    /// Ungroup: move a nested node to the root stack.
    MoveToRoot {
        id: LayerId,
    },
    /// Request opening Quick Add popup.
    OpenQuickAdd,
    /// Open Quick Add filtered to a WC category.
    OpenQuickAddCategory {
        category: terra_core::layer::StackCategory,
    },
    /// Open Quick Add filtered to tools for one artist concept folder.
    OpenQuickAddConcept {
        concept: crate::ui::hierarchy_view::ArtistConcept,
    },
    /// Open Quick Add targeting a specific group (biome / section / folder).
    OpenQuickAddInto {
        parent: LayerId,
    },
    /// Open Quick Add for DistNode generators on a biome's Distribution stack.
    OpenQuickAddDistribution {
        biome: LayerId,
    },
    /// Add a layer as a direct child of a group (biome section preferred).
    AddLayerInto {
        parent: LayerId,
        layer: Layer,
    },
    /// Randomize seed on selected procedural layer.
    RandomizeSeed {
        id: LayerId,
    },
    /// Set a biome's Filters / Objects overwrite-vs-blend behaviour.
    SetBiomeOverwrite {
        target: LayerId,
        overwrite_filters: bool,
        overwrite_objects: bool,
    },
    /// Set how strongly a biome's Filters blend with lower biomes (0..1).
    SetBiomeFilterBlending {
        target: LayerId,
        blending: f32,
    },
    /// Cycle the active biome paint brush tool.
    CycleBiomePaintTool,
    /// Set the active biome paint brush tool directly.
    SetBiomePaintTool(terra_core::biome_paint::BiomePaintTool),
    /// Add a WC Biome Layers paint/splat layer.
    AddBiomePaintLayer {
        name: String,
    },
    /// Select a biome paint layer and arm the paint tool.
    SelectBiomePaintLayer(terra_core::biome_paint::BiomeLayerId),
    /// Select a biome definition from the artist palette (makes it paintable).
    SelectBiomeDefinition(terra_core::biome_definition::BiomeDefinitionId),
    /// Select a shape object for editing.
    SelectShape(terra_core::shape_object::ShapeObjectId),
    /// Translate selected shape in UV.
    TranslateShape {
        id: terra_core::shape_object::ShapeObjectId,
        du: f32,
        dv: f32,
    },
    /// Recompile shape objects into the managed constraints layer.
    CompileShapes,
    /// Move a shape control point to a new UV (local transform inverted).
    SetShapePoint {
        id: terra_core::shape_object::ShapeObjectId,
        index: usize,
        u: f32,
        v: f32,
    },
    /// Apply artist blueprint semantic controls to global processes.
    ApplyBlueprintSemantics {
        ridge_sharpness: f32,
        sea_level: f32,
        geological_age: f32,
        rainfall: f32,
        drainage_density: f32,
    },
    /// Instantiate a built-in GroupRecipe as a biome / isolated group.
    InstantiateRecipe {
        recipe_name: String,
    },
    /// Reorder a layer/group relative to a sibling (document order).
    ReorderRelative {
        moving: LayerId,
        target: LayerId,
        place_before: bool,
    },
    /// Ensure shared hydrology / evolution process layers exist.
    EnsureHydrologyProcesses,
    /// Ensure materials + vegetation surface layers exist.
    EnsureSurfaceProcesses,
    /// Open the Export floating panel (Review workspace).
    ShowExportPanel,
    /// Turn biome colour ownership preview on/off for the selected placement layer.
    SetBiomeColorPreview(bool),
    /// Create a new shape object and select it (Terrain intent).
    CreateShape {
        kind: terra_core::shape_object::ShapeKind,
        name: String,
    },
    /// Set how paint combines with procedural rules for a biome definition.
    SetBiomePlacementCombine {
        definition: terra_core::biome_definition::BiomeDefinitionId,
        combine: terra_core::biome_definition::PlacementCombineMode,
    },
    /// Mark biome placement Mask Stack as Custom (stop auto-recompile from rules).
    MarkBiomePlacementCustom {
        definition: terra_core::biome_definition::BiomeDefinitionId,
    },
    /// Recompile PlacementDefinition rules into the biome group's Mask Stack.
    ResetBiomePlacementToRules {
        definition: terra_core::biome_definition::BiomeDefinitionId,
    },
    /// Full terrain rebuild.
    RebuildWorld,
    /// Add a registry layer suggested for the current stack context.
    AddSuggestedLayer {
        type_id: String,
        label: String,
    },
    /// Contextual Create-here (viewport RMB / hierarchy Add).
    ContextualCreate {
        kind: terra_core::contextual_create::CreateKind,
        /// Explicit owner override; `None` uses inference (must resolve).
        owner: Option<terra_core::contextual_create::CreateOwner>,
        uv: Option<(f32, f32)>,
    },
    /// Open the viewport Create-here menu at a screen position.
    OpenViewportContextMenu {
        x: f32,
        y: f32,
        uv: Option<(f32, f32)>,
        locked_owner: Option<terra_core::contextual_create::CreateOwner>,
    },
    /// Focus an inspector section after create (string key: general/noise/â€¦).
    SetInspectorSection(String),
    /// Rebuild outdated / affected content (selective).
    RebuildAffected,
    /// Toggle automatic expensive physics rebuild.
    SetAutomaticRebuildExpensive(bool),
    /// Toggle draft live preview while editing.
    SetLivePreview(bool),
    /// Ask why the selected / given layer is outdated.
    WhyRebuild {
        layer: Option<terra_core::layer::LayerId>,
    },
    /// Open a file dialog to pick a heightmap image for Import / Stamp2d.
    BrowseHeightmapPath {
        layer: LayerId,
    },
    /// Open a file dialog to pick a mesh (.obj) or height image for Stamp3d.
    BrowseMeshPath {
        layer: LayerId,
    },
}
