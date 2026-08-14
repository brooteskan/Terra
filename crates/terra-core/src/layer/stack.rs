use super::{
    BiomeSection, BlendMode, CachePolicy, GroupEvalMode, GroupInputMode, GroupKind, Layer, LayerId,
    NamedOutputDecl, StackCategory, TransitionSettings,
};
use crate::mask::Distribution;
use serde::{Deserialize, Serialize};

/// Nested group or single layer in the stack (bottom → top order in Vec).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StackNode {
    Layer(Layer),
    Group(LayerGroup),
}

/// Composition unit: children evaluate as a subtree, then (for isolated groups)
/// the result is mixed onto the parent stack with opacity × group masks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerGroup {
    pub id: LayerId,
    pub name: String,
    pub enabled: bool,
    /// Opacity in \[0, 1\] applied when mixing the group result onto the parent.
    #[serde(default = "default_opacity")]
    pub opacity: f32,
    /// Blend mode used when mixing the group result (usually [`BlendMode::Normal`]).
    #[serde(default)]
    pub blend: BlendMode,
    /// Masks that scope where the group result replaces the parent.
    #[serde(default)]
    pub masks: Distribution,
    pub children: Vec<StackNode>,
    /// Pass-through vs isolated composite (v3+).
    #[serde(default)]
    pub eval_mode: GroupEvalMode,
    /// How an isolated group initialises its private context.
    #[serde(default)]
    pub input_mode: GroupInputMode,
    #[serde(default)]
    pub transition: TransitionSettings,
    #[serde(default)]
    pub outputs: Vec<NamedOutputDecl>,
    #[serde(default)]
    pub cache_policy: CachePolicy,
    #[serde(default)]
    pub collapsed: bool,
    #[serde(default)]
    pub frozen: bool,
    /// World Creator–style category folder role (Shape / Simulation / Mask / Biomes).
    #[serde(default)]
    pub category: Option<StackCategory>,
    /// Structural role (biome container, section, category folder).
    #[serde(default)]
    pub group_kind: GroupKind,
    /// Optional colour tag for biome paint / viewport overlay (0 = none).
    #[serde(default)]
    pub color_tag: u8,
    /// Swatch colour shown in biome lists / viewport overlays (WC-style preview tint).
    #[serde(default = "super::group_mode::default_preview_color")]
    pub preview_color: [f32; 3],
    /// When true, this biome's Filters override (replace) lower biomes instead of blending.
    #[serde(default)]
    pub overwrite_filters: bool,
    /// When true, this biome's Objects override lower biomes instead of blending.
    #[serde(default)]
    pub overwrite_objects: bool,
    /// Blend factor \[0,1\] applied to Filters mixing with lower biomes (1 = full mix).
    #[serde(default = "default_filter_blending")]
    pub filter_blending: f32,
}

fn default_opacity() -> f32 {
    1.0
}

fn default_filter_blending() -> f32 {
    1.0
}

impl LayerGroup {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: LayerId::new(),
            name: name.into(),
            enabled: true,
            opacity: 1.0,
            blend: BlendMode::Normal,
            masks: Distribution::new(),
            children: Vec::new(),
            eval_mode: GroupEvalMode::PassThrough,
            input_mode: GroupInputMode::CopyInput,
            transition: TransitionSettings::default(),
            outputs: Vec::new(),
            cache_policy: CachePolicy::Live,
            collapsed: false,
            frozen: false,
            category: None,
            group_kind: GroupKind::Generic,
            color_tag: 0,
            preview_color: super::group_mode::default_preview_color(),
            overwrite_filters: false,
            overwrite_objects: false,
            filter_blending: 1.0,
        }
    }

    /// Isolated terrain recipe group (typical for alpine / desert regions).
    pub fn isolated(name: impl Into<String>) -> Self {
        let mut g = Self::new(name);
        g.eval_mode = GroupEvalMode::IsolatedComposite;
        g.input_mode = GroupInputMode::CopyInput;
        g
    }

    /// Pass-through World Creator category folder.
    pub fn category_folder(category: StackCategory) -> Self {
        let mut g = Self::new(category.label());
        g.eval_mode = GroupEvalMode::PassThrough;
        g.category = Some(category);
        g.group_kind = GroupKind::CategoryFolder;
        g
    }

    /// World Creator biome container with empty Filters / Materials / Objects / Local Sims.
    pub fn biome(name: impl Into<String>) -> Self {
        let mut g = Self::isolated(name);
        g.group_kind = GroupKind::Biome;
        g.preview_color = super::group_mode::palette_preview_color(g.id.0.as_u128());
        g.ensure_biome_sections();
        g
    }

    /// Pass-through biome section folder.
    pub fn biome_section(section: BiomeSection) -> Self {
        let mut g = Self::new(section.label());
        g.eval_mode = GroupEvalMode::PassThrough;
        g.group_kind = GroupKind::BiomeSection(section);
        g
    }

    pub fn is_category_folder(&self) -> bool {
        self.category.is_some() || matches!(self.group_kind, GroupKind::CategoryFolder)
    }

    pub fn is_biome(&self) -> bool {
        matches!(self.group_kind, GroupKind::Biome)
    }

    pub fn biome_section_kind(&self) -> Option<BiomeSection> {
        match self.group_kind {
            GroupKind::BiomeSection(s) => Some(s),
            _ => None,
        }
    }

    /// Ensure Filters / Materials / Objects / Local Sims children exist (idempotent).
    pub fn ensure_biome_sections(&mut self) {
        for &section in BiomeSection::all() {
            let exists = self.children.iter().any(|n| match n {
                StackNode::Group(g) => g.biome_section_kind() == Some(section),
                _ => false,
            });
            if !exists {
                self.children
                    .push(StackNode::Group(Self::biome_section(section)));
            }
        }
        // Keep sections in canonical order at the front; leave other children after.
        let mut sections = Vec::new();
        let mut rest = Vec::new();
        for n in self.children.drain(..) {
            match &n {
                StackNode::Group(g) if g.biome_section_kind().is_some() => sections.push(n),
                _ => rest.push(n),
            }
        }
        sections.sort_by_key(|n| match n {
            StackNode::Group(g) => match g.biome_section_kind() {
                Some(BiomeSection::Filters) => 0u8,
                Some(BiomeSection::Materials) => 1,
                Some(BiomeSection::Objects) => 2,
                Some(BiomeSection::LocalSims) => 3,
                None => 4,
            },
            _ => 4,
        });
        // Orphan layers that aren't under a section go into Filters.
        for n in rest {
            match n {
                StackNode::Layer(l) => {
                    let section = BiomeSection::for_operation(l.category());
                    if let Some(sec) = sections.iter_mut().find_map(|sn| match sn {
                        StackNode::Group(g) if g.biome_section_kind() == Some(section) => Some(g),
                        _ => None,
                    }) {
                        sec.children.push(StackNode::Layer(l));
                    } else {
                        sections.push(StackNode::Layer(l));
                    }
                }
                other => sections.push(other),
            }
        }
        self.children = sections;
    }

    pub fn find_section(&self, section: BiomeSection) -> Option<&LayerGroup> {
        self.children.iter().find_map(|n| match n {
            StackNode::Group(g) if g.biome_section_kind() == Some(section) => Some(g),
            _ => None,
        })
    }

    pub fn find_section_mut(&mut self, section: BiomeSection) -> Option<&mut LayerGroup> {
        self.children.iter_mut().find_map(|n| match n {
            StackNode::Group(g) if g.biome_section_kind() == Some(section) => Some(g),
            _ => None,
        })
    }

    /// Push a layer into the matching biome section (creates sections if needed).
    pub fn push_into_section(&mut self, layer: Layer) -> LayerId {
        self.ensure_biome_sections();
        let section = match &layer.kind {
            crate::layer::LayerKind::Vegetation(_) => BiomeSection::Objects,
            crate::layer::LayerKind::Materials(_) | crate::layer::LayerKind::Biomes(_) => {
                BiomeSection::Materials
            }
            crate::layer::LayerKind::SandSimulation(_)
            | crate::layer::LayerKind::FluidSimulation(_) => BiomeSection::LocalSims,
            _ => BiomeSection::for_operation(layer.category()),
        };
        let id = layer.id();
        if let Some(sec) = self.find_section_mut(section) {
            sec.children.push(StackNode::Layer(layer));
        } else {
            self.children.push(StackNode::Layer(layer));
        }
        id
    }

    /// True when this group changes composition vs a pure organizational flatten.
    pub fn is_scoped(&self) -> bool {
        matches!(self.eval_mode, GroupEvalMode::IsolatedComposite)
            || self.opacity < 0.999
            || !self.masks.is_empty()
            || !matches!(self.blend, BlendMode::Normal | BlendMode::Replace)
            || self.is_biome()
    }

    pub fn is_pass_through(&self) -> bool {
        matches!(self.eval_mode, GroupEvalMode::PassThrough) && !self.is_scoped()
    }
}

/// Root ordered layer stack (index 0 = bottom).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LayerStack {
    pub nodes: Vec<StackNode>,
}

impl LayerStack {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, layer: Layer) {
        self.nodes.push(StackNode::Layer(layer));
    }

    pub fn push_group(&mut self, group: LayerGroup) {
        self.nodes.push(StackNode::Group(group));
    }

    /// Find the root category folder for a World Creator–style section.
    pub fn find_category(&self, category: StackCategory) -> Option<&LayerGroup> {
        self.nodes.iter().find_map(|n| match n {
            StackNode::Group(g) if g.category == Some(category) => Some(g),
            _ => None,
        })
    }

    pub fn find_category_mut(&mut self, category: StackCategory) -> Option<&mut LayerGroup> {
        self.nodes.iter_mut().find_map(|n| match n {
            StackNode::Group(g) if g.category == Some(category) => Some(g),
            _ => None,
        })
    }

    pub fn category_id(&self, category: StackCategory) -> Option<LayerId> {
        self.find_category(category).map(|g| g.id)
    }

    /// Stable-sort layer children within each category/group by [`LayerKind::eval_stage`].
    /// Groups keep relative order; only Layer nodes are reordered among themselves
    /// while preserving group boundaries (groups stay where they are).
    pub fn sort_by_eval_stage(&mut self) {
        fn sort_nodes(nodes: &mut [StackNode]) {
            // Partition into runs of consecutive Layer nodes and sort each run.
            let mut i = 0;
            while i < nodes.len() {
                if matches!(&nodes[i], StackNode::Group(_)) {
                    if let StackNode::Group(g) = &mut nodes[i] {
                        sort_nodes(&mut g.children);
                    }
                    i += 1;
                    continue;
                }
                let start = i;
                while i < nodes.len() && matches!(&nodes[i], StackNode::Layer(_)) {
                    i += 1;
                }
                nodes[start..i].sort_by_key(|n| match n {
                    StackNode::Layer(l) => l.kind.eval_stage() as u8,
                    StackNode::Group(_) => 255,
                });
            }
        }
        sort_nodes(&mut self.nodes);
    }

    /// Ensure WC category folders exist at the root in evaluation order.
    /// Existing non-category root layers are left in place (typically Foundation).
    pub fn ensure_category_folders(&mut self) {
        for &cat in StackCategory::all_folders() {
            if self.find_category(cat).is_some() {
                continue;
            }
            let folder = LayerGroup::category_folder(cat);
            // Insert after any earlier categories / foundation layers.
            let insert_at = self
                .nodes
                .iter()
                .enumerate()
                .find_map(|(i, n)| match n {
                    StackNode::Group(g) => g.category.and_then(|c| {
                        if c.stack_order() > cat.stack_order() {
                            Some(i)
                        } else {
                            None
                        }
                    }),
                    _ => None,
                })
                .unwrap_or(self.nodes.len());
            self.nodes.insert(insert_at, StackNode::Group(folder));
        }
    }

    /// Push a layer into the matching WC category folder (creates folders if needed).
    pub fn push_into_category(&mut self, layer: Layer) -> LayerId {
        let cat = StackCategory::from_operation(layer.category());
        self.ensure_category_folders();
        let id = layer.id();
        if let Some(folder) = self.find_category_mut(cat) {
            folder.children.push(StackNode::Layer(layer));
        } else {
            self.nodes.push(StackNode::Layer(layer));
        }
        id
    }

    /// Ensure at least one Default biome exists under the Biomes folder.
    pub fn ensure_default_biome(&mut self) -> LayerId {
        self.ensure_category_folders();
        if let Some(biomes) = self.find_category(StackCategory::Surface) {
            for child in &biomes.children {
                if let StackNode::Group(g) = child {
                    if g.is_biome() {
                        return g.id;
                    }
                }
            }
        }
        let biome = LayerGroup::biome("Global");
        let id = biome.id;
        if let Some(folder) = self.find_category_mut(StackCategory::Surface) {
            folder.children.push(StackNode::Group(biome));
        } else {
            self.nodes.push(StackNode::Group(biome));
        }
        id
    }

    /// Find the first biome under the Biomes root (if any).
    pub fn first_biome(&self) -> Option<&LayerGroup> {
        let biomes = self.find_category(StackCategory::Surface)?;
        biomes.children.iter().find_map(|n| match n {
            StackNode::Group(g) if g.is_biome() => Some(g),
            _ => None,
        })
    }

    pub fn first_biome_mut(&mut self) -> Option<&mut LayerGroup> {
        let biomes = self.find_category_mut(StackCategory::Surface)?;
        biomes.children.iter_mut().find_map(|n| match n {
            StackNode::Group(g) if g.is_biome() => Some(g),
            _ => None,
        })
    }

    /// Walk ancestors of `id` and return the enclosing biome group (if any).
    pub fn enclosing_biome(&self, id: LayerId) -> Option<&LayerGroup> {
        let mut current = id;
        loop {
            let parent = find_parent_id(&self.nodes, current)?;
            let g = self.find_group(parent)?;
            if g.is_biome() {
                return Some(g);
            }
            current = parent;
        }
    }

    /// Route a new layer into the best WC destination given selection context.
    ///
    /// - **Shape layers** → Region Shape folder (always)
    /// - **Terrain filters** → Biome Filters section (active/enclosing/default biome)
    /// - Sims / materials / objects / scatter → matching biome section
    /// - `prefer_global_shape` forces Shape folder even for dual-role kinds
    pub fn push_routed(
        &mut self,
        layer: Layer,
        context: Option<LayerId>,
        prefer_global_shape: bool,
    ) -> LayerId {
        // Prefer biome context when selection is inside a biome.
        let biome_id = context.and_then(|cid| {
            if self.find_group(cid).is_some_and(|g| g.is_biome()) {
                Some(cid)
            } else {
                self.enclosing_biome(cid).map(|g| g.id)
            }
        });

        let is_shape = prefer_global_shape || is_shape_kind(&layer.kind);
        let is_terrain_filter = is_biome_filter_kind(&layer.kind);

        // WC: shape authoring lives on the stack Shape folder, never under a Biome section.
        if is_shape {
            return self.push_into_category(layer);
        }

        // Filters / sims / materials / objects → biome section.
        let biome_id = biome_id.unwrap_or_else(|| self.ensure_default_biome());
        let id = layer.id();
        if let Some(biome) = self.find_group_mut(biome_id) {
            // Terrain filters always land in Filters (not Local Sims / Materials).
            if is_terrain_filter {
                biome.ensure_biome_sections();
                if let Some(sec) = biome.find_section_mut(BiomeSection::Filters) {
                    sec.children.push(StackNode::Layer(layer));
                    return id;
                }
            }
            biome.push_into_section(layer);
            id
        } else {
            self.push_into_category(layer)
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Flatten to layers in bottom→top evaluation order, skipping disabled groups.
    ///
    /// Prefer tree evaluation ([`crate::eval::StackEvaluator`]) when groups are scoped;
    /// flatten remains useful for leaf iteration, dirty ids, and GPU approximate paths.
    pub fn flatten_layers(&self) -> Vec<&Layer> {
        let mut out = Vec::new();
        fn walk<'a>(nodes: &'a [StackNode], out: &mut Vec<&'a Layer>) {
            for n in nodes {
                match n {
                    StackNode::Layer(l) => out.push(l),
                    StackNode::Group(g) if g.enabled => walk(&g.children, out),
                    StackNode::Group(_) => {}
                }
            }
        }
        walk(&self.nodes, &mut out);
        out
    }

    pub fn flatten_layers_mut(&mut self) -> Vec<&mut Layer> {
        let mut out = Vec::new();
        fn walk<'a>(nodes: &'a mut [StackNode], out: &mut Vec<&'a mut Layer>) {
            for n in nodes {
                match n {
                    StackNode::Layer(l) => out.push(l),
                    StackNode::Group(g) if g.enabled => walk(&mut g.children, out),
                    StackNode::Group(_) => {}
                }
            }
        }
        walk(&mut self.nodes, &mut out);
        out
    }

    pub fn find(&self, id: LayerId) -> Option<&Layer> {
        self.flatten_layers().into_iter().find(|l| l.id() == id)
    }

    pub fn find_mut(&mut self, id: LayerId) -> Option<&mut Layer> {
        self.flatten_layers_mut().into_iter().find(|l| l.id() == id)
    }

    pub fn find_group(&self, id: LayerId) -> Option<&LayerGroup> {
        fn walk(nodes: &[StackNode], id: LayerId) -> Option<&LayerGroup> {
            for n in nodes {
                match n {
                    StackNode::Group(g) if g.id == id => return Some(g),
                    StackNode::Group(g) => {
                        if let Some(found) = walk(&g.children, id) {
                            return Some(found);
                        }
                    }
                    StackNode::Layer(_) => {}
                }
            }
            None
        }
        walk(&self.nodes, id)
    }

    pub fn find_group_mut(&mut self, id: LayerId) -> Option<&mut LayerGroup> {
        fn walk(nodes: &mut [StackNode], id: LayerId) -> Option<&mut LayerGroup> {
            for node in nodes {
                if let StackNode::Group(g) = node {
                    if g.id == id {
                        return Some(g);
                    }
                    if let Some(found) = walk(&mut g.children, id) {
                        return Some(found);
                    }
                }
            }
            None
        }
        walk(&mut self.nodes, id)
    }

    /// Whether any enabled group uses opacity / masks / non-Normal blend.
    pub fn has_scoped_groups(&self) -> bool {
        fn has_enabled_layer(nodes: &[StackNode]) -> bool {
            nodes.iter().any(|node| match node {
                StackNode::Layer(layer) => layer.common.enabled,
                StackNode::Group(group) => group.enabled && has_enabled_layer(&group.children),
            })
        }
        fn walk(nodes: &[StackNode]) -> bool {
            for n in nodes {
                if let StackNode::Group(g) = n {
                    if g.enabled
                        && ((g.is_scoped() && has_enabled_layer(&g.children)) || walk(&g.children))
                    {
                        return true;
                    }
                }
            }
            false
        }
        walk(&self.nodes)
    }

    /// Whether evaluation must preserve the stack tree instead of flattening leaves.
    ///
    /// Scoped groups compose their children as a unit, while solo mode filters sibling
    /// nodes at each tree level. Neither contract can be represented by a flattened suffix.
    pub fn requires_tree_evaluation(&self) -> bool {
        self.has_scoped_groups() || self.flatten_layers().iter().any(|layer| layer.common.solo)
    }

    pub fn index_of(&self, id: LayerId) -> Option<usize> {
        self.nodes.iter().position(|n| match n {
            StackNode::Layer(l) => l.id() == id,
            StackNode::Group(g) => g.id == id,
        })
    }

    /// Parent group (if any) and sibling index of `id` anywhere in the tree.
    pub fn sibling_location(&self, id: LayerId) -> Option<(Option<LayerId>, usize)> {
        if let Some(idx) = self.index_of(id) {
            return Some((None, idx));
        }
        fn walk(
            nodes: &[StackNode],
            parent: LayerId,
            id: LayerId,
        ) -> Option<(Option<LayerId>, usize)> {
            if let Some(idx) = nodes.iter().position(|n| node_id(n) == id) {
                return Some((Some(parent), idx));
            }
            for n in nodes {
                if let StackNode::Group(g) = n {
                    if let Some(found) = walk(&g.children, g.id, id) {
                        return Some(found);
                    }
                }
            }
            None
        }
        for n in &self.nodes {
            if let StackNode::Group(g) = n {
                if let Some(found) = walk(&g.children, g.id, id) {
                    return Some(found);
                }
            }
        }
        None
    }

    /// Remove a node (layer or group) from anywhere in the tree.
    pub fn remove(&mut self, id: LayerId) -> Option<StackNode> {
        if let Some(idx) = self.index_of(id) {
            return Some(self.nodes.remove(idx));
        }
        fn walk(nodes: &mut Vec<StackNode>, id: LayerId) -> Option<StackNode> {
            if let Some(idx) = nodes.iter().position(|n| match n {
                StackNode::Layer(l) => l.id() == id,
                StackNode::Group(g) => g.id == id,
            }) {
                return Some(nodes.remove(idx));
            }
            for n in nodes.iter_mut() {
                if let StackNode::Group(g) = n {
                    if let Some(found) = walk(&mut g.children, id) {
                        return Some(found);
                    }
                }
            }
            None
        }
        walk(&mut self.nodes, id)
    }

    pub fn reorder(&mut self, from: usize, to: usize) {
        if from >= self.nodes.len() || to >= self.nodes.len() || from == to {
            return;
        }
        let node = self.nodes.remove(from);
        self.nodes.insert(to, node);
    }

    /// Reorder `moving` relative to `target`.
    ///
    /// Same-parent → sibling reorder. Different parent → relocate into the
    /// target's parent at the relative index (hierarchy change).
    pub fn reorder_relative(
        &mut self,
        moving: LayerId,
        target: LayerId,
        place_before: bool,
    ) -> bool {
        if moving == target {
            return false;
        }
        let Some((to_parent, _)) = self.sibling_location(target) else {
            return false;
        };
        if Some(moving) == to_parent {
            return false;
        }
        // Prevent parenting a group into its own descendant.
        if let Some(g) = self.find_group(moving) {
            if let Some(pid) = to_parent {
                if contains_id(&g.children, pid) || contains_id(&g.children, target) {
                    return false;
                }
            }
        }
        let Some(node) = self.remove(moving) else {
            return false;
        };
        let Some((_, target_idx)) = self.sibling_location(target) else {
            self.insert_at_parent(to_parent, usize::MAX, node);
            return false;
        };
        let insert_at = if place_before {
            target_idx
        } else {
            target_idx + 1
        };
        self.insert_at_parent(to_parent, insert_at, node);
        true
    }

    fn insert_at_parent(&mut self, parent: Option<LayerId>, index: usize, node: StackNode) {
        let nodes = match parent {
            None => &mut self.nodes,
            Some(pid) => {
                if let Some(g) = self.find_group_mut(pid) {
                    &mut g.children
                } else {
                    &mut self.nodes
                }
            }
        };
        let idx = index.min(nodes.len());
        nodes.insert(idx, node);
    }

    /// Move a layer into a biome's section. Returns true on success.
    /// Preserves layer data; resets nothing about placement (caller may keep Apply Where).
    pub fn move_layer_to_biome_section(
        &mut self,
        layer_id: LayerId,
        biome_id: LayerId,
        section: BiomeSection,
    ) -> bool {
        // Region shape layers stay on the Shape stack — never under a biome section.
        if self.find(layer_id).is_some_and(|l| is_shape_kind(&l.kind)) {
            return false;
        }
        let Some(node) = self.remove(layer_id) else {
            return false;
        };
        let StackNode::Layer(layer) = node else {
            // Groups are not biome-section children; restore at root.
            self.nodes.push(node);
            return false;
        };
        let Some(biome) = self.find_group_mut(biome_id) else {
            self.nodes.push(StackNode::Layer(layer));
            return false;
        };
        if !biome.is_biome() {
            self.nodes.push(StackNode::Layer(layer));
            return false;
        }
        biome.ensure_biome_sections();
        if let Some(sec) = biome.find_section_mut(section) {
            sec.children.push(StackNode::Layer(layer));
        } else {
            biome.children.push(StackNode::Layer(layer));
        }
        true
    }

    pub fn move_into_group(&mut self, child_id: LayerId, group_id: LayerId) -> bool {
        if child_id == group_id {
            return false;
        }
        // Prevent parenting a group into its own descendant.
        if let Some(g) = self.find_group(child_id) {
            if contains_id(&g.children, group_id) {
                return false;
            }
        }
        let Some(node) = self.remove(child_id) else {
            return false;
        };
        if let Some(group) = self.find_group_mut(group_id) {
            group.children.push(node);
            true
        } else {
            // Restore at root if group vanished.
            self.nodes.push(node);
            false
        }
    }

    /// Move `child_id` out of its parent group onto the root stack (after the parent).
    pub fn move_to_root(&mut self, child_id: LayerId) -> bool {
        // Already at root?
        if self.index_of(child_id).is_some() {
            return true;
        }
        let parent_id = find_parent_id(&self.nodes, child_id);
        let Some(node) = self.remove(child_id) else {
            return false;
        };
        if let Some(pid) = parent_id {
            if let Some(idx) = self.index_of(pid) {
                self.nodes.insert(idx + 1, node);
                return true;
            }
        }
        self.nodes.push(node);
        true
    }

    pub fn duplicate(&mut self, id: LayerId) -> Option<LayerId> {
        // Root-level duplicate (existing behaviour) or nested.
        if let Some(idx) = self.index_of(id) {
            let dup = duplicate_node(&self.nodes[idx]);
            let new_id = node_id(&dup);
            self.nodes.insert(idx + 1, dup);
            return Some(new_id);
        }
        fn walk(nodes: &mut Vec<StackNode>, id: LayerId) -> Option<LayerId> {
            if let Some(idx) = nodes.iter().position(|n| node_id(n) == id) {
                let dup = duplicate_node(&nodes[idx]);
                let new_id = node_id(&dup);
                nodes.insert(idx + 1, dup);
                return Some(new_id);
            }
            for n in nodes.iter_mut() {
                if let StackNode::Group(g) = n {
                    if let Some(found) = walk(&mut g.children, id) {
                        return Some(found);
                    }
                }
            }
            None
        }
        walk(&mut self.nodes, id)
    }

    /// Layer ids in flatten order from bottom to top.
    pub fn layer_ids(&self) -> Vec<LayerId> {
        self.flatten_layers().iter().map(|l| l.id()).collect()
    }

    /// Index in flatten order.
    pub fn flatten_index(&self, id: LayerId) -> Option<usize> {
        self.layer_ids().iter().position(|&x| x == id)
    }
}

fn node_id(n: &StackNode) -> LayerId {
    match n {
        StackNode::Layer(l) => l.id(),
        StackNode::Group(g) => g.id,
    }
}

fn duplicate_node(n: &StackNode) -> StackNode {
    match n {
        StackNode::Layer(l) => StackNode::Layer(l.duplicate()),
        StackNode::Group(g) => {
            let mut ng = g.clone();
            ng.id = LayerId::new();
            ng.name = format!("{} Copy", g.name);
            ng.children = g.children.iter().map(duplicate_node).collect();
            // Refresh child layer ids so caches don't collide.
            rename_subtree_ids(&mut ng.children);
            StackNode::Group(ng)
        }
    }
}

fn rename_subtree_ids(nodes: &mut [StackNode]) {
    for n in nodes {
        match n {
            StackNode::Layer(l) => {
                l.common.id = LayerId::new();
            }
            StackNode::Group(g) => {
                g.id = LayerId::new();
                rename_subtree_ids(&mut g.children);
            }
        }
    }
}

fn contains_id(nodes: &[StackNode], id: LayerId) -> bool {
    for n in nodes {
        match n {
            StackNode::Layer(l) if l.id() == id => return true,
            StackNode::Group(g) if g.id == id => return true,
            StackNode::Group(g) => {
                if contains_id(&g.children, id) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

fn find_parent_id(nodes: &[StackNode], child: LayerId) -> Option<LayerId> {
    for n in nodes {
        if let StackNode::Group(g) = n {
            for c in &g.children {
                if node_id(c) == child {
                    return Some(g.id);
                }
            }
            if let Some(found) = find_parent_id(&g.children, child) {
                return Some(found);
            }
        }
    }
    None
}

/// Kinds that belong under the Shape category folder (WC shape layers).
///
/// Dual-role kinds (e.g. [`ImportHeightmap`](crate::layer::LayerKind::ImportHeightmap))
/// are *not* included — callers that mean the Shape tool entry must force Shape via
/// `prefer_global_shape` / `shape.*` catalog ids.
pub fn is_shape_kind(kind: &crate::layer::LayerKind) -> bool {
    use crate::layer::LayerKind::*;
    matches!(
        kind,
        SculptBase(_)
            | SculptStrokes(_)
            | TerrainConstraints(_)
            | Flat(_)
            | Ramp(_)
            | NoiseValue(_)
            | NoisePerlin(_)
            | NoiseOpenSimplex(_)
            | NoiseWorley(_)
            | Fbm(_)
            | Ridged(_)
            | DomainWarp(_)
            | Island(_)
            | VoronoiRegions(_)
            | GradientReconstruct(_)
            | Path(_)
            | ProceduralShape(_)
            | Stamp2d(_)
            | Stamp3d(_)
            | PolygonHeight(_)
    )
}

/// Kinds that belong under a Biome's Filters section (WC biome filters).
fn is_biome_filter_kind(kind: &crate::layer::LayerKind) -> bool {
    use crate::layer::LayerKind::*;
    matches!(
        kind,
        Mountains(_)
            | Mesa(_)
            | Volcano(_)
            | Dunes(_)
            | Canyons(_)
            | Terrace(_)
            | Plateau(_)
            | Blur(_)
            | Coastal(_)
            | EffectFilter(_)
            | GeomorphicDetail(_)
            | OverhangStamp(_)
            | LocalSdf(_)
            // WC Design "Height Map" filter — Terrain `shape.heightmap` forces Shape via
            // `prefer_global_shape` / AddLayerToCategory instead.
            | ImportHeightmap(_)
    )
}

/// Which biome section a layer kind lands in when added into a biome.
///
/// Returns `None` for shape / foundation kinds that must stay on the Region
/// Shape stack (see [`LayerStack::push_routed`]).
pub fn biome_destination_section(kind: &crate::layer::LayerKind) -> Option<BiomeSection> {
    if is_shape_kind(kind) {
        return None;
    }
    Some(match kind {
        crate::layer::LayerKind::Vegetation(_) => BiomeSection::Objects,
        crate::layer::LayerKind::Materials(_) | crate::layer::LayerKind::Biomes(_) => {
            BiomeSection::Materials
        }
        crate::layer::LayerKind::SandSimulation(_)
        | crate::layer::LayerKind::FluidSimulation(_) => BiomeSection::LocalSims,
        k if is_biome_filter_kind(k) => BiomeSection::Filters,
        k => BiomeSection::for_operation(k.category()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::{
        EffectFilterParams, FlatParams, LayerKind, ProceduralShapeParams, Stamp2dParams,
    };

    #[test]
    fn move_layer_to_biome_section_puts_filter_under_filters() {
        let mut stack = LayerStack::new();
        stack.ensure_category_folders();
        let biome_id = stack.ensure_default_biome();
        let filter = Layer::new(
            "Rocky",
            LayerKind::EffectFilter(EffectFilterParams::default()),
        );
        let fid = filter.id();
        stack.push_into_category(filter);
        assert!(stack.enclosing_biome(fid).is_none());
        assert!(stack.move_layer_to_biome_section(fid, biome_id, BiomeSection::Filters));
        let biome = stack.find_group(biome_id).unwrap();
        let filters = biome.find_section(BiomeSection::Filters).unwrap();
        assert!(
            filters.children.iter().any(|n| match n {
                StackNode::Layer(l) => l.id() == fid,
                _ => false,
            }),
            "filter should appear under biome Filters"
        );
    }

    #[test]
    fn push_routed_shape_stays_under_shape_even_inside_biome() {
        let mut stack = LayerStack::new();
        stack.ensure_category_folders();
        let biome_id = stack.ensure_default_biome();
        let shape = Layer::new(
            "Proc",
            LayerKind::ProceduralShape(ProceduralShapeParams::default()),
        );
        let sid = shape.id();
        stack.push_routed(shape, Some(biome_id), false);
        assert!(
            stack.enclosing_biome(sid).is_none(),
            "shape layers must not parent under a biome"
        );
        let folder = stack.find_category(StackCategory::Shape).unwrap();
        assert!(
            folder
                .children
                .iter()
                .any(|n| matches!(n, StackNode::Layer(l) if l.id() == sid)),
            "procedural shape should land in Shape folder"
        );
    }

    #[test]
    fn push_routed_filter_lands_in_biome_filters() {
        let mut stack = LayerStack::new();
        stack.ensure_category_folders();
        let filter = Layer::new(
            "Soft Flows",
            LayerKind::EffectFilter(EffectFilterParams::soft_flows()),
        );
        let fid = filter.id();
        stack.push_routed(filter, None, false);
        let biome = stack.enclosing_biome(fid).expect("filter under biome");
        let filters = biome
            .find_section(BiomeSection::Filters)
            .expect("Filters section");
        assert!(filters
            .children
            .iter()
            .any(|n| matches!(n, StackNode::Layer(l) if l.id() == fid)));
    }

    #[test]
    fn biome_destination_section_routes_expected_kinds() {
        use crate::layer::{MaterialsParams, SandSimParams, TerraceParams, VegetationParams};
        assert_eq!(
            biome_destination_section(&LayerKind::EffectFilter(EffectFilterParams::default())),
            Some(BiomeSection::Filters)
        );
        assert_eq!(
            biome_destination_section(&LayerKind::Terrace(TerraceParams::default())),
            Some(BiomeSection::Filters)
        );
        assert!(is_shape_kind(&LayerKind::Path(
            crate::layer::PathParams::default()
        )));
        assert!(is_shape_kind(&LayerKind::PolygonHeight(
            crate::layer::PolygonHeightParams::default()
        )));
        assert!(!is_shape_kind(
            &LayerKind::Terrace(TerraceParams::default())
        ));
        assert_eq!(
            biome_destination_section(&LayerKind::Materials(MaterialsParams::default())),
            Some(BiomeSection::Materials)
        );
        assert_eq!(
            biome_destination_section(&LayerKind::Vegetation(VegetationParams::default())),
            Some(BiomeSection::Objects)
        );
        assert_eq!(
            biome_destination_section(&LayerKind::SandSimulation(SandSimParams::default())),
            Some(BiomeSection::LocalSims)
        );
        assert_eq!(
            biome_destination_section(
                &LayerKind::ProceduralShape(ProceduralShapeParams::default())
            ),
            None
        );
        assert_eq!(
            biome_destination_section(&LayerKind::ImportHeightmap(
                crate::layer::ImportHeightmapParams::default()
            )),
            Some(BiomeSection::Filters)
        );
        assert_eq!(
            biome_destination_section(&LayerKind::Dunes(crate::layer::DuneParams::default())),
            Some(BiomeSection::Filters)
        );
    }

    #[test]
    fn push_routed_stamp2d_is_shape() {
        let mut stack = LayerStack::new();
        stack.ensure_category_folders();
        let stamp = Layer::new("Stamp", LayerKind::Stamp2d(Stamp2dParams::default()));
        let id = stamp.id();
        stack.push_routed(stamp, None, false);
        assert!(stack.enclosing_biome(id).is_none());
        assert!(stack
            .find_category(StackCategory::Shape)
            .unwrap()
            .children
            .iter()
            .any(|n| matches!(n, StackNode::Layer(l) if l.id() == id)));
    }

    #[test]
    fn push_routed_height_map_filter_lands_in_biome_filters() {
        let mut stack = LayerStack::new();
        stack.ensure_category_folders();
        let layer = Layer::new(
            "Height Map",
            LayerKind::ImportHeightmap(crate::layer::ImportHeightmapParams::default()),
        );
        let id = layer.id();
        stack.push_routed(layer, None, false);
        let biome = stack.enclosing_biome(id).expect("height map under biome");
        let filters = biome
            .find_section(BiomeSection::Filters)
            .expect("Filters section");
        assert!(filters
            .children
            .iter()
            .any(|n| matches!(n, StackNode::Layer(l) if l.id() == id)));
    }

    #[test]
    fn push_routed_height_map_prefer_shape_stays_shape() {
        let mut stack = LayerStack::new();
        stack.ensure_category_folders();
        let layer = Layer::new(
            "Heightmap",
            LayerKind::ImportHeightmap(crate::layer::ImportHeightmapParams::default()),
        );
        let id = layer.id();
        stack.push_routed(layer, None, true);
        assert!(stack.enclosing_biome(id).is_none());
        assert!(stack
            .find_category(StackCategory::Shape)
            .unwrap()
            .children
            .iter()
            .any(|n| matches!(n, StackNode::Layer(l) if l.id() == id)));
    }

    #[test]
    fn reorder_and_duplicate() {
        let mut stack = LayerStack::new();
        let a = Layer::new("A", LayerKind::Flat(FlatParams { height: 1.0 }));
        let b = Layer::new("B", LayerKind::Flat(FlatParams { height: 2.0 }));
        let id_a = a.id();
        let id_b = b.id();
        stack.push(a);
        stack.push(b);
        stack.reorder(0, 1);
        assert_eq!(stack.layer_ids()[0], id_b);
        assert_eq!(stack.layer_ids()[1], id_a);
        let dup = stack.duplicate(id_b).unwrap();
        assert_ne!(dup, id_b);
        assert_eq!(stack.len(), 3);
    }

    #[test]
    fn move_into_group_and_scoped_flag() {
        use crate::mask::{MaskId, MaskRef};
        let mut stack = LayerStack::new();
        let layer = Layer::new("L", LayerKind::Flat(FlatParams { height: 5.0 }));
        let lid = layer.id();
        stack.push(layer);
        let mut group = LayerGroup::new("G");
        let gid = group.id;
        group.masks.push(MaskRef::new(MaskId::new()));
        stack.push_group(group);
        assert!(stack.move_into_group(lid, gid));
        assert!(stack.find(lid).is_some());
        assert!(stack.find_group(gid).unwrap().children.len() == 1);
        assert!(stack.has_scoped_groups());
        assert!(stack.move_to_root(lid));
        assert_eq!(stack.nodes.len(), 2);
    }

    #[test]
    fn reorder_relative_relocates_across_parents() {
        let mut stack = LayerStack::new();
        let a = Layer::new("A", LayerKind::Flat(FlatParams { height: 1.0 }));
        let b = Layer::new("B", LayerKind::Flat(FlatParams { height: 2.0 }));
        let aid = a.id();
        let bid = b.id();
        stack.push(a);
        let g = LayerGroup::new("Folder");
        let gid = g.id;
        stack.push_group(g);
        stack.push(b);
        assert!(stack.move_into_group(bid, gid));
        // Place root layer A before nested B → A joins the folder.
        assert!(stack.reorder_relative(aid, bid, true));
        let folder = stack.find_group(gid).unwrap();
        let ids: Vec<_> = folder
            .children
            .iter()
            .filter_map(|n| match n {
                StackNode::Layer(l) => Some(l.id()),
                _ => None,
            })
            .collect();
        assert_eq!(ids, vec![aid, bid]);
        assert!(stack.index_of(aid).is_none());
    }

    #[test]
    fn empty_default_biome_does_not_disable_flat_gpu_evaluation() {
        let mut stack = LayerStack::new();
        stack.ensure_category_folders();
        stack.ensure_default_biome();
        assert!(!stack.has_scoped_groups());
        assert!(!stack.requires_tree_evaluation());
    }

    #[test]
    fn tree_evaluation_requirement_tracks_scoped_solo_and_pass_through_stacks() {
        let mut pass_through = LayerStack::new();
        let mut folder = LayerGroup::new("Folder");
        folder.children.push(StackNode::Layer(Layer::new(
            "Child",
            LayerKind::Flat(FlatParams { height: 1.0 }),
        )));
        pass_through.push_group(folder);
        assert!(!pass_through.requires_tree_evaluation());

        let mut scoped = LayerStack::new();
        let mut group = LayerGroup::isolated("Scoped");
        group.children.push(StackNode::Layer(Layer::new(
            "Child",
            LayerKind::Flat(FlatParams { height: 2.0 }),
        )));
        scoped.push_group(group);
        assert!(scoped.requires_tree_evaluation());

        let mut disabled_scoped = scoped.clone();
        if let StackNode::Group(group) = &mut disabled_scoped.nodes[0] {
            group.enabled = false;
        }
        assert!(!disabled_scoped.requires_tree_evaluation());

        let mut solo = LayerStack::new();
        let mut layer = Layer::new("Solo", LayerKind::Flat(FlatParams { height: 3.0 }));
        layer.common.solo = true;
        solo.push(layer);
        assert!(solo.requires_tree_evaluation());
    }

    #[test]
    fn remove_nested_layer_via_sibling_location() {
        let mut stack = LayerStack::new();
        let layer = Layer::new("Nested", LayerKind::Flat(FlatParams { height: 3.0 }));
        let lid = layer.id();
        stack.push(layer);
        let group = LayerGroup::new("Folder");
        let gid = group.id;
        stack.push_group(group);
        assert!(stack.move_into_group(lid, gid));
        assert!(
            stack.index_of(lid).is_none(),
            "nested layer is not root-indexed"
        );
        let (parent, index) = stack.sibling_location(lid).expect("sibling location");
        assert_eq!(parent, Some(gid));
        assert_eq!(index, 0);
        let node = stack.remove(lid).expect("remove nested");
        assert!(matches!(node, StackNode::Layer(_)));
        assert!(stack.find(lid).is_none());
        // Restore into parent like undo.
        stack
            .find_group_mut(gid)
            .unwrap()
            .children
            .insert(index, node);
        assert!(stack.find(lid).is_some());
    }
}
