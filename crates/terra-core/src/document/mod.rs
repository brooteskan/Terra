//! Serializable project document.

mod session;

#[cfg(test)]
mod tests;

pub use session::{EditorSession, MaskPaintStrokeUndo, PaintStrokeUndo};

use crate::deps::DependencyGraph;
use crate::heightfield::HeightfieldMetrics;
use crate::layer::{
    BiomeSection, GroupEvalMode, GroupKind, Layer, LayerGroup, LayerId, LayerKind, LayerStack,
    NoiseParams, SculptParams, StackCategory, StackNode,
};
use crate::mask::{MaskAsset, MaskId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Highest persisted document version ever emitted. This value is monotonic and
/// must never be decremented. Readers accept supported older versions and reject
/// only genuinely newer versions; normalization may add defaults but must retain
/// authored identity and semantics. Persisted enum tags are additive (renames
/// require aliases or migration), and every new persisted field requires a Serde
/// default or an explicit migration. Writers always stamp this current version.
pub const DOCUMENT_VERSION: u32 = 2;

/// Presentation lighting for the 3D viewport, saved with the project so a custom
/// look is restored on load. Angles are degrees; strengths are renderer multipliers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewportLighting {
    pub sun_azimuth_deg: f32,
    pub sun_elevation_deg: f32,
    pub sun_intensity: f32,
    pub exposure: f32,
    pub sky_color: [f32; 3],
    pub ambient_strength: f32,
    pub shadow_strength: f32,
    pub fog_strength: f32,
    /// UI preset chip label; empty when the values were customized (blank chip).
    #[serde(default)]
    pub preset: String,
}

impl Default for ViewportLighting {
    fn default() -> Self {
        // Matches the app's default Studio-seeded lighting.
        Self {
            sun_azimuth_deg: 30.98,
            sun_elevation_deg: 72.91,
            sun_intensity: 1.05,
            exposure: 1.05,
            sky_color: [0.18, 0.20, 0.24],
            ambient_strength: 1.0,
            shadow_strength: 0.0,
            fog_strength: 1.0,
            preset: String::from("Studio"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainDocument {
    pub version: u32,
    pub name: String,
    pub metrics: HeightfieldMetrics,
    pub preview_resolution: u32,
    pub export_resolution: u32,
    pub stack: LayerStack,
    pub masks: Vec<MaskAsset>,
    pub selected: Option<LayerId>,
    pub presets_used: Vec<String>,
    /// Active biome for paint / add-layer context.
    #[serde(default)]
    pub active_biome: Option<LayerId>,
    /// Global level-step / precision settings (WC Terrain panel).
    #[serde(default)]
    pub level_steps: crate::analyze::LevelStepSettings,
    /// Biome paint / placement layers (WHERE).
    #[serde(default)]
    pub biome_layers: Vec<crate::biome_paint::BiomeLayer>,
    /// Hole pierce masks (data model present; pierce eval not fully wired).
    #[serde(default)]
    pub hole_layers: Vec<crate::biome_paint::HoleLayer>,
    /// Selected biome paint / placement layer.
    #[serde(default)]
    pub selected_biome_layer: Option<crate::biome_paint::BiomeLayerId>,
    /// World-scale landscape blueprint.
    #[serde(default)]
    pub blueprint: crate::landscape_blueprint::LandscapeBlueprint,
    /// Direct-manipulation shape objects.
    #[serde(default)]
    pub shapes: crate::shape_object::ShapeObjectStore,
    /// Reusable biome definitions.
    #[serde(default)]
    pub biome_library: crate::biome_definition::BiomeLibrary,
    /// Sparse world-space paint pages.
    #[serde(default)]
    pub sparse_paint: crate::sparse_paint::SparsePaintStore,
    /// First-class World Rules (condition-driven ops).
    #[serde(default)]
    pub world_rules: crate::world_rules::WorldRuleLibrary,
    /// Simulation Scenarios — authoring containers over Simulation Layers.
    #[serde(default)]
    pub simulation_scenarios: crate::simulation_scenario::SimulationScenarioLibrary,
    /// Presentation lighting for the 3D viewport (saved with the project).
    #[serde(default)]
    pub viewport_lighting: ViewportLighting,
}

impl Default for TerrainDocument {
    fn default() -> Self {
        Self::new_default()
    }
}

fn ensure_all_biome_sections(nodes: &mut [StackNode]) {
    for n in nodes {
        if let StackNode::Group(g) = n {
            if g.is_biome() {
                g.ensure_biome_sections();
            }
            ensure_all_biome_sections(&mut g.children);
        }
    }
}

impl TerrainDocument {
    pub fn new_default() -> Self {
        let metrics = HeightfieldMetrics::preview_default();
        let mut stack = LayerStack::new();
        // Sculptable foundation — selected by default for raise/lower.
        let base = Layer::new(
            "Base",
            LayerKind::SculptBase(SculptParams::filled(512, 20.0)),
        );
        let base_id = base.id();
        stack.push(base);
        // World Creator–style category folders (pass-through).
        stack.ensure_category_folders();
        // Light detail under Shape Layers — re-evaluates when Base is sculpted.
        stack.push_into_category(Layer::new(
            "Hills",
            LayerKind::NoiseValue(NoiseParams {
                seed: 1,
                frequency: 0.0015,
                amplitude: 80.0,
                octaves: 4,
                lacunarity: 2.0,
                persistence: 0.5,
                ..NoiseParams::default()
            }),
        ));
        let biome_id = stack.ensure_default_biome();
        let biome_layers = {
            let mut bl = crate::biome_paint::BiomeLayer::new("Primary Biome Placement");
            bl.show_biome_colors = true;
            vec![bl]
        };
        let selected_biome_layer = biome_layers.first().map(|b| b.id);
        let mut biome_library = crate::biome_definition::BiomeLibrary::default_world_palette();
        // Link Default definition to the default biome group when possible.
        if let Some(def) = biome_library
            .definitions
            .iter_mut()
            .find(|d| d.name == "Default")
        {
            def.group_id = Some(biome_id);
        }
        let world_extent = metrics.world_size_x.max(metrics.world_size_z);
        let mut blueprint = crate::landscape_blueprint::LandscapeBlueprint::default();
        blueprint.world_size_m = world_extent;
        blueprint.metres_per_sample = (world_extent / metrics.width.max(1) as f32).max(0.5);
        let mut sparse_paint =
            crate::sparse_paint::SparsePaintStore::new(blueprint.metres_per_sample, 256);
        sparse_paint.metres_per_sample = blueprint.metres_per_sample;
        Self {
            version: DOCUMENT_VERSION,
            name: "Untitled".into(),
            metrics,
            preview_resolution: metrics.width,
            export_resolution: 2048,
            stack,
            masks: Vec::new(),
            selected: Some(base_id),
            presets_used: Vec::new(),
            active_biome: Some(biome_id),
            level_steps: crate::analyze::LevelStepSettings::default(),
            biome_layers,
            hole_layers: Vec::new(),
            selected_biome_layer,
            blueprint,
            shapes: crate::shape_object::ShapeObjectStore::default(),
            biome_library,
            sparse_paint,
            world_rules: crate::world_rules::WorldRuleLibrary::default(),
            simulation_scenarios: crate::simulation_scenario::SimulationScenarioLibrary::default(),
            viewport_lighting: ViewportLighting::default(),
        }
    }

    /// Demonstration stack: Base + Alpine biome under Surface.
    pub fn alpine_demo() -> Self {
        use crate::layer::{HydraulicErosionParams, MountainParams, ThermalErosionParams};
        use crate::mask::{MaskAsset, MaskSource, PaintBuffer};

        let mut doc = Self::new_default();
        doc.name = "Alpine Demo".into();
        // Clear shape children (Hills) but keep category folders + Base.
        if let Some(shape) = doc.stack.find_category_mut(StackCategory::Shape) {
            shape.children.clear();
        }
        // Replace Default Biome with Alpine.
        if let Some(biomes) = doc.stack.find_category_mut(StackCategory::Surface) {
            biomes.children.clear();
        }

        let mask = MaskAsset {
            id: MaskId::new(),
            name: "Alpine Area".into(),
            source: MaskSource::Height {
                min: 0.0,
                max: 2000.0,
            },
            ops: Vec::new(),
            paint: Some({
                let mut p = PaintBuffer::new(256, 256);
                for s in &mut p.samples {
                    *s = 1.0;
                }
                p
            }),
            display_color: crate::mask::default_mask_display_color(),
        };
        let mask_id = mask.id;
        doc.masks.push(mask);

        let mut alpine = LayerGroup::biome("Alpine Mountains");
        alpine.color_tag = 2;
        alpine.masks.push(crate::mask::MaskRef::new(mask_id));
        alpine.push_into_section(Layer::new(
            "Mountain Range",
            LayerKind::Mountains(MountainParams::default()),
        ));
        alpine.push_into_section(Layer::new(
            "Hydraulic Erosion",
            LayerKind::HydraulicErosion(HydraulicErosionParams::default()),
        ));
        alpine.push_into_section(Layer::new(
            "Thermal Erosion",
            LayerKind::ThermalErosion(ThermalErosionParams::default()),
        ));
        let alpine_id = alpine.id;
        if let Some(biomes) = doc.stack.find_category_mut(StackCategory::Surface) {
            biomes.children.push(StackNode::Group(alpine));
        } else {
            doc.stack.push_group(alpine);
        }
        doc.active_biome = Some(alpine_id);
        doc.presets_used.push("Alpine Demo".into());
        doc
    }

    /// Ensure WC category folders + default biome exist, and relocate any loose
    /// root layers into Shape / Default Biome (idempotent).
    pub fn normalize_wc_tree(&mut self) {
        self.stack.ensure_category_folders();
        // Mark existing category folders.
        for n in &mut self.stack.nodes {
            if let StackNode::Group(g) = n {
                if g.category.is_some() {
                    g.group_kind = GroupKind::CategoryFolder;
                }
            }
        }

        // Pull loose root layers / generic groups out of the root.
        let mut loose: Vec<StackNode> = Vec::new();
        let mut kept: Vec<StackNode> = Vec::new();
        for n in self.stack.nodes.drain(..) {
            match n {
                StackNode::Layer(l) if l.kind.is_sculpt_base() => {
                    kept.push(StackNode::Layer(l));
                }
                StackNode::Group(g) if g.category.is_some() || g.is_biome() => {
                    kept.push(StackNode::Group(g));
                }
                other => loose.push(other),
            }
        }
        self.stack.nodes = kept;
        self.stack.ensure_category_folders();

        if !loose.is_empty() {
            let biome_id = self.stack.ensure_default_biome();
            for node in loose {
                match node {
                    StackNode::Layer(l) => {
                        use crate::layer::OperationCategory;
                        let is_shape = matches!(
                            l.category(),
                            OperationCategory::Generator
                                | OperationCategory::ImportedData
                                | OperationCategory::Modifier
                        );
                        if is_shape {
                            self.stack.push_into_category(l);
                        } else if let Some(biome) = self.stack.find_group_mut(biome_id) {
                            biome.push_into_section(l);
                        } else {
                            self.stack.push(l);
                        }
                    }
                    StackNode::Group(mut g) => {
                        if matches!(g.eval_mode, GroupEvalMode::IsolatedComposite) {
                            g.group_kind = GroupKind::Biome;
                            g.ensure_biome_sections();
                            if let Some(biomes) =
                                self.stack.find_category_mut(StackCategory::Surface)
                            {
                                biomes.children.push(StackNode::Group(g));
                            }
                        } else if let Some(biome) = self.stack.find_group_mut(biome_id) {
                            if let Some(filters) = biome.find_section_mut(BiomeSection::Filters) {
                                filters.children.push(StackNode::Group(g));
                            }
                        }
                    }
                }
            }
        }

        let biome_id = self.stack.ensure_default_biome();
        // Promote isolated recipe groups under Biomes that lack Biome kind.
        if let Some(surface) = self.stack.find_category_mut(StackCategory::Surface) {
            for child in &mut surface.children {
                if let StackNode::Group(g) = child {
                    if !g.is_biome() && matches!(g.eval_mode, GroupEvalMode::IsolatedComposite) {
                        g.group_kind = GroupKind::Biome;
                    }
                    if g.is_biome() {
                        g.ensure_biome_sections();
                    }
                }
            }
        }
        ensure_all_biome_sections(&mut self.stack.nodes);
        if self.active_biome.is_none()
            || !self
                .active_biome
                .and_then(|id| self.stack.find_group(id))
                .is_some_and(|g| g.is_biome())
        {
            self.active_biome = Some(biome_id);
        }
        if self.biome_layers.is_empty() {
            let mut bl = crate::biome_paint::BiomeLayer::new("Biome Paint");
            bl.show_biome_colors = true;
            self.selected_biome_layer = Some(bl.id);
            self.biome_layers.push(bl);
        }
        self.version = DOCUMENT_VERSION;
    }

    /// Build a WC-structured document from a flat preset/template layer list.
    pub fn from_flat_layers(layers: Vec<Layer>) -> Self {
        let mut doc = Self::new_default();
        // Clear the seeded Hills under Shape; keep Base + folders + Default Biome.
        if let Some(shape) = doc.stack.find_category_mut(StackCategory::Shape) {
            shape.children.clear();
        }
        if let Some(biomes) = doc.stack.find_category_mut(StackCategory::Surface) {
            biomes.children.clear();
        }
        // Replace foundation if template provides a sculpt base.
        let mut foundation: Option<Layer> = None;
        let mut rest = Vec::new();
        for l in layers {
            if l.kind.is_sculpt_base() && foundation.is_none() {
                foundation = Some(l);
            } else {
                rest.push(l);
            }
        }
        if let Some(base) = foundation {
            let base_id = base.id();
            // Remove existing foundation layer(s) at root.
            doc.stack.nodes.retain(|n| match n {
                StackNode::Layer(l) => !l.kind.is_sculpt_base(),
                _ => true,
            });
            doc.stack.nodes.insert(0, StackNode::Layer(base));
            doc.selected = Some(base_id);
        }
        for l in rest {
            use crate::layer::OperationCategory;
            let is_shape = matches!(
                l.category(),
                OperationCategory::Generator
                    | OperationCategory::ImportedData
                    | OperationCategory::Modifier
            );
            if is_shape {
                doc.stack.push_into_category(l);
            } else {
                let biome_id = doc.stack.ensure_default_biome();
                if let Some(biome) = doc.stack.find_group_mut(biome_id) {
                    biome.push_into_section(l);
                }
            }
        }
        doc.normalize_wc_tree();
        doc
    }

    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        let mut doc: Self = serde_json::from_str(s)?;
        if doc.version > DOCUMENT_VERSION {
            return Err(serde::de::Error::custom(format!(
                "unsupported document version {} (latest supported {})",
                doc.version, DOCUMENT_VERSION
            )));
        }
        doc.normalize_wc_tree();
        Ok(doc)
    }

    /// Interactive viewport preview — the single terrain stack.
    pub fn preview_eval_stack(&self) -> LayerStack {
        self.stack.clone()
    }

    /// Domain hierarchy view from the single terrain stack.
    pub fn domain_view(&self) -> crate::domain::DomainView {
        crate::domain::DomainView::from_stack(&self.stack)
    }

    /// Evaluate final height from the single terrain stack.
    pub fn evaluate_final_height(
        &mut self,
        ctx: &mut crate::eval::EvalContext,
    ) -> Result<crate::Heightfield, crate::eval::EvalError> {
        let mut evaluator = crate::eval::StackEvaluator::new();
        evaluator.rebuild_all(&self.stack, ctx)
    }

    /// Add a layer via context-aware stack routing.
    pub fn add_layer(&mut self, layer: Layer) -> LayerId {
        let id = layer.id();
        self.stack.ensure_category_folders();
        self.stack.push_routed(layer, self.active_biome, false);
        self.selected = Some(id);
        id
    }

    /// Add a layer into the Shape category folder.
    pub fn add_shape_layer(&mut self, layer: Layer) -> LayerId {
        let id = layer.id();
        self.stack.ensure_category_folders();
        self.stack.push_into_category(layer);
        self.selected = Some(id);
        id
    }

    /// JSON for disk (compact).
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        self.clone().into_json()
    }

    /// Consume a document into JSON.
    pub fn into_json(mut self) -> Result<String, serde_json::Error> {
        self.version = DOCUMENT_VERSION;
        serde_json::to_string(&self)
    }

    /// Resolve the selected biome placement layer, falling back to the first.
    pub fn selected_placement_layer(&self) -> Option<&crate::biome_paint::BiomeLayer> {
        if let Some(id) = self.selected_biome_layer {
            if let Some(layer) = self.biome_layers.iter().find(|l| l.id == id) {
                return Some(layer);
            }
        }
        self.biome_layers.first()
    }

    pub fn selected_placement_layer_mut(&mut self) -> Option<&mut crate::biome_paint::BiomeLayer> {
        let id = self
            .selected_biome_layer
            .or_else(|| self.biome_layers.first().map(|l| l.id));
        let id = id?;
        self.selected_biome_layer = Some(id);
        self.biome_layers.iter_mut().find(|l| l.id == id)
    }

    /// Ensure a primary placement layer exists and is selected.
    pub fn ensure_placement_layer(&mut self) -> crate::biome_paint::BiomeLayerId {
        if let Some(id) = self.selected_biome_layer {
            if self.biome_layers.iter().any(|l| l.id == id) {
                return id;
            }
        }
        if let Some(first) = self.biome_layers.first() {
            let id = first.id;
            self.selected_biome_layer = Some(id);
            return id;
        }
        let mut bl = crate::biome_paint::BiomeLayer::new("Primary Biome Placement");
        bl.show_biome_colors = true;
        let id = bl.id;
        self.biome_layers.push(bl);
        self.selected_biome_layer = Some(id);
        id
    }

    /// Bake every biome's sparse/dense placement paint into MaskAssets and attach
    /// them to the corresponding biome groups. Safe to call before export.
    pub fn sync_all_biome_paint_masks(&mut self) {
        let mut biome_ids: Vec<LayerId> = self
            .biome_library
            .definitions
            .iter()
            .filter_map(|d| d.group_id)
            .collect();
        fn collect_biomes(nodes: &[StackNode], out: &mut Vec<LayerId>) {
            for n in nodes {
                match n {
                    StackNode::Group(g) => {
                        if g.is_biome() {
                            out.push(g.id);
                        }
                        collect_biomes(&g.children, out);
                    }
                    StackNode::Layer(_) => {}
                }
            }
        }
        collect_biomes(&self.stack.nodes, &mut biome_ids);
        biome_ids.sort_by_key(|id| id.0);
        biome_ids.dedup();
        for biome_id in biome_ids {
            self.sync_biome_paint_to_mask(biome_id);
        }
    }

    /// Bake one biome's placement paint into a MaskAsset bound on that biome group.
    pub fn sync_biome_paint_to_mask(&mut self, biome_id: LayerId) {
        let placement_id = self.ensure_placement_layer();
        let world_x = self.metrics.world_size_x;
        let world_z = self.metrics.world_size_z;
        let key = crate::sparse_paint::SparsePaintChannelKey {
            placement_id,
            biome_id,
        };
        let res = self.preview_resolution.min(1024).max(64);
        let paint = if self.sparse_paint.has_channel(key) {
            let samples = self.sparse_paint.bake_uv(key, res, res, world_x, world_z);
            crate::mask::PaintBuffer {
                width: res,
                height: res,
                samples,
            }
        } else {
            let Some(splat) = self.selected_placement_layer() else {
                return;
            };
            let Some(ch) = splat.channels.iter().find(|c| c.biome_id == biome_id) else {
                return;
            };
            ch.paint.clone()
        };
        let mask_name = format!("BiomePaint_{}", &biome_id.0.to_string()[..8]);
        let existing = self
            .masks
            .iter()
            .find(|m| m.name == mask_name)
            .map(|m| m.id);
        let mask_id = if let Some(id) = existing {
            if let Some(asset) = self.masks.iter_mut().find(|m| m.id == id) {
                asset.paint = Some(paint);
                asset.source = crate::mask::MaskSource::Painted { mask_id: id };
            }
            id
        } else {
            let id = MaskId::new();
            self.masks.push(MaskAsset {
                id,
                name: mask_name,
                source: crate::mask::MaskSource::Painted { mask_id: id },
                ops: Vec::new(),
                paint: Some(paint),
                display_color: crate::mask::default_mask_display_color(),
            });
            id
        };
        let mode = self
            .biome_library
            .by_group(biome_id)
            .map(|d| d.placement.combine)
            .unwrap_or_default();
        if let Some(biome) = self.stack.find_group_mut(biome_id) {
            if !mode.uses_manual_paint() {
                biome.masks.entries.retain(|e| e.mask.id != mask_id);
                return;
            }
            let combine = mode.mask_combine();
            let already = biome.masks.entries.iter().any(|e| e.mask.id == mask_id);
            if !already {
                let mut entry =
                    crate::mask::DistributionEntry::new(crate::mask::MaskRef::new(mask_id));
                entry.combine = combine;
                biome.masks.entries.push(entry);
            } else if let Some(entry) = biome
                .masks
                .entries
                .iter_mut()
                .find(|e| e.mask.id == mask_id)
            {
                entry.combine = combine;
            }
        }
    }

    /// Compile shape objects into the managed TerrainConstraints layer.
    pub fn compile_shapes_into_stack(&mut self) {
        use crate::layer::LayerKind;
        let params = self.shapes.compile_constraints();
        if let Some(id) = self.shapes.managed_constraints_layer {
            if let Some(layer) = self.stack.find_mut(id) {
                layer.kind = LayerKind::TerrainConstraints(params);
                return;
            }
        }
        // Create managed layer under Shape category.
        let layer = Layer::new(
            "Shape Objects (compiled)",
            LayerKind::TerrainConstraints(params),
        );
        let id = layer.id();
        self.stack.push_into_category(layer);
        self.shapes.managed_constraints_layer = Some(id);
    }

    pub fn mask_map(&self) -> HashMap<MaskId, &MaskAsset> {
        self.masks.iter().map(|m| (m.id, m)).collect()
    }

    pub fn dependency_graph(&self) -> DependencyGraph {
        let mask_ids: Vec<_> = self.masks.iter().map(|m| m.id).collect();
        DependencyGraph::build_from_stack(&self.stack, &mask_ids)
    }

    pub fn validate_dependencies(&self) -> Result<(), String> {
        let g = self.dependency_graph();
        if let Err(e) = g.detect_cycle() {
            return Err(e.to_string());
        }
        Ok(())
    }
}
