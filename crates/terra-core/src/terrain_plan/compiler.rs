//! Recursive lowering from the authored layer tree into the terrain-plan IR.

use std::collections::{HashMap, HashSet};

use crate::deps::NodeRef;
use crate::field_data::FieldId;
use crate::ids::{LayerId, OutputId};
use crate::invalidation::{AuxReach, Reach};
use crate::layer::{
    BindingSource, GroupInputMode, GroupKind, Layer, LayerGroup, LayerStack, NamedOutputDecl,
    SelectedGroupInput, StackNode,
};
use crate::mask::{
    ClimateMaskChannel, DistNode, DistNodeKind, Distribution, MaskAsset, MaskId, MaskSource,
};

use super::{
    CompiledTerrainPlan, FieldSlot, GroupAuxComposite, GroupCompositeMode, LogicalFieldKind,
    PlanBuildError, PlanNodeSelection, PlanOrigin, SeedSource, TerrainOp, TerrainOpKind,
    TerrainPlanBuilder, TerrainPlanStamp,
};

/// A source-owned problem that prevents deterministic plan lowering.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TerrainPlanDiagnostic {
    #[error("authored node id {0:?} occurs more than once")]
    DuplicateNodeId(LayerId),
    #[error("published output id {0:?} occurs more than once")]
    DuplicateOutputId(OutputId),
    #[error("{owner:?} references missing mask {mask:?}")]
    MissingMask { owner: NodeRef, mask: MaskId },
    #[error("{owner:?} references missing output {output:?}")]
    MissingOutput { owner: NodeRef, output: OutputId },
    #[error("{owner:?} references disabled or excluded output {output:?}")]
    DisabledOutput { owner: NodeRef, output: OutputId },
    #[error("{owner:?} cannot publish unavailable field {field:?}")]
    UnavailableField { owner: NodeRef, field: FieldId },
    #[error("{owner:?} references output {output:?} before it is available")]
    UnavailableOutput { owner: NodeRef, output: OutputId },
    #[error("authored dependency cycle involves {nodes:?}")]
    DependencyCycle { nodes: Vec<NodeRef> },
    #[error(transparent)]
    Build(#[from] PlanBuildError),
}

/// Compile one authored stack for a particular structural revision.
///
/// The compiler reads payload metadata but stores no CPU fields or backend
/// resources. Disabled and solo-excluded nodes emit no operations.
pub fn compile_terrain_plan(
    stack: &LayerStack,
    mask_assets: &[MaskAsset],
    stamp: TerrainPlanStamp,
) -> Result<CompiledTerrainPlan, Vec<TerrainPlanDiagnostic>> {
    let diagnostics = preflight(stack, mask_assets);
    if !diagnostics.is_empty() {
        return Err(diagnostics);
    }

    let selection = collect_plan_node_selection(&stack.nodes);
    let mut builder = TerrainPlanBuilder::new(stamp);
    for (owner, node_selection) in &selection {
        builder.record_node_selection(*owner, *node_selection);
    }
    let mut compiler = Compiler {
        builder,
        mask_assets,
        selection: selection.into_iter().collect(),
        published_outputs: HashMap::new(),
    };
    let root = compiler
        .builder
        .add_field(LogicalFieldKind::Height, PlanOrigin::Root);
    compiler.builder.add_operation(TerrainOp {
        origin: PlanOrigin::Root,
        reach: Reach::LOCAL,
        aux_reach: AuxReach::HeightOnly,
        kind: TerrainOpKind::Seed {
            source: SeedSource::Zero,
            output: root,
        },
    });

    let mut state = PlanState::new(root);
    if let Err(diagnostic) = compiler.compile_nodes(&stack.nodes, &mut state) {
        return Err(vec![diagnostic]);
    }
    compiler
        .builder
        .finish(state.height)
        .map_err(|error| vec![TerrainPlanDiagnostic::Build(error)])
}

#[derive(Debug, Clone)]
struct PlanState {
    height: FieldSlot,
    /// Stable first-production order; do not replace with iteration over a hash map.
    aux: Vec<(FieldId, FieldSlot)>,
}

impl PlanState {
    fn new(height: FieldSlot) -> Self {
        Self {
            height,
            aux: Vec::new(),
        }
    }

    fn field(&self, field: &FieldId) -> Option<FieldSlot> {
        if *field == FieldId::Height {
            return Some(self.height);
        }
        self.aux
            .iter()
            .find_map(|(candidate, slot)| (candidate == field).then_some(*slot))
    }

    fn set_aux(&mut self, field: FieldId, slot: FieldSlot) {
        if let Some((_, current)) = self
            .aux
            .iter_mut()
            .find(|(candidate, _)| *candidate == field)
        {
            *current = slot;
        } else {
            self.aux.push((field, slot));
        }
    }

    fn output_slots(&self) -> impl Iterator<Item = FieldSlot> + '_ {
        std::iter::once(self.height).chain(self.aux.iter().map(|(_, slot)| *slot))
    }
}

struct Compiler<'a> {
    builder: TerrainPlanBuilder,
    mask_assets: &'a [MaskAsset],
    selection: HashMap<NodeRef, PlanNodeSelection>,
    /// Resolved publications in authored execution order. References are stable
    /// by `OutputId`; the display name and sibling position are irrelevant.
    published_outputs: HashMap<OutputId, FieldSlot>,
}

impl Compiler<'_> {
    fn compile_nodes(
        &mut self,
        nodes: &[StackNode],
        state: &mut PlanState,
    ) -> Result<(), TerrainPlanDiagnostic> {
        for node in nodes {
            let owner = node_ref(node);
            if !self
                .selection
                .get(&owner)
                .copied()
                .unwrap_or(PlanNodeSelection::Unfiltered)
                .participates()
            {
                continue;
            }
            match node {
                StackNode::Layer(layer) if layer.common.enabled => {
                    self.compile_layer(layer, state)?
                }
                StackNode::Layer(_) => {}
                StackNode::Group(group) if group.enabled => self.compile_group(group, state)?,
                StackNode::Group(_) => {}
            }
        }
        Ok(())
    }

    fn compile_layer(
        &mut self,
        layer: &Layer,
        state: &mut PlanState,
    ) -> Result<(), TerrainPlanDiagnostic> {
        let owner_ref = NodeRef::Layer(layer.id());
        let span_start = self.builder.operation_count();
        let origin = PlanOrigin::Authored(owner_ref);
        let base = state.height;
        let input_fields = self.resolve_layer_inputs(layer, state)?;
        let candidate = self.builder.add_field(LogicalFieldKind::Height, origin);

        let mut produced = Vec::new();
        for field in layer.kind.produced_fields() {
            if field == FieldId::Height || produced.iter().any(|(known, _)| *known == field) {
                continue;
            }
            let slot = self
                .builder
                .add_field(LogicalFieldKind::Auxiliary(field.clone()), origin);
            produced.push((field, slot));
        }
        let output_fields = produced.iter().map(|(_, slot)| *slot).collect();
        let reach = configured_layer_reach(layer);
        let kernel = self.builder.add_operation(TerrainOp {
            origin,
            reach,
            aux_reach: layer.kind.aux_reach(),
            kind: TerrainOpKind::RunLayerKernel {
                layer: layer.id(),
                type_id: layer.kind.type_id().into(),
                input_height: base,
                input_fields,
                output_candidate: candidate,
                output_fields,
            },
        });
        for binding in &layer.common.param_bindings {
            match binding.source {
                BindingSource::Mask(mask) => self.builder.record_authored_dependency(
                    kernel,
                    NodeRef::Mask(mask),
                    crate::deps::DepKind::ParamBinding,
                ),
                BindingSource::LayerOutput(output) | BindingSource::GroupOutput(output) => {
                    self.builder.record_authored_dependency(
                        kernel,
                        NodeRef::Output(output),
                        crate::deps::DepKind::ParamBinding,
                    );
                }
                _ => {}
            }
        }
        let mask = self.builder.add_field(LogicalFieldKind::Mask, origin);
        let mask_inputs =
            self.resolve_distribution_inputs(owner_ref, &layer.common.masks, state)?;
        let mask_op = self.builder.add_operation(TerrainOp {
            origin,
            reach: crate::mask::distribution_reach(&layer.common.masks, self.mask_assets),
            aux_reach: AuxReach::HeightOnly,
            kind: TerrainOpKind::EvaluateMask {
                input_height: base,
                input_fields: mask_inputs,
                output_mask: mask,
            },
        });
        self.record_distribution_dependencies(mask_op, &layer.common.masks);
        let output = self.builder.add_field(LogicalFieldKind::Height, origin);
        self.builder.add_operation(TerrainOp {
            origin,
            reach: Reach::LOCAL,
            aux_reach: AuxReach::HeightOnly,
            kind: TerrainOpKind::CompositeLayer {
                layer: layer.id(),
                base,
                candidate,
                mask,
                output,
            },
        });
        state.height = output;
        for (field, child) in produced {
            let parent = state.field(&field);
            let output = self
                .builder
                .add_field(LogicalFieldKind::Auxiliary(field.clone()), origin);
            self.builder.add_operation(TerrainOp {
                origin,
                reach: Reach::LOCAL,
                aux_reach: AuxReach::PerTexel,
                kind: TerrainOpKind::CompositeAuxField {
                    owner: owner_ref,
                    mask,
                    composite: GroupAuxComposite {
                        field: field.clone(),
                        parent,
                        child,
                        output,
                    },
                },
            });
            state.set_aux(field, output);
        }
        self.publish_outputs(owner_ref, &layer.common.outputs, state)?;
        for field in state.output_slots() {
            self.builder.record_owner_field(owner_ref, field);
        }
        self.builder
            .record_owner_span(owner_ref, span_start, self.builder.operation_count());
        Ok(())
    }

    fn compile_group(
        &mut self,
        group: &LayerGroup,
        state: &mut PlanState,
    ) -> Result<(), TerrainPlanDiagnostic> {
        let owner_ref = NodeRef::Group(group.id);
        let span_start = self.builder.operation_count();
        if group.is_pass_through() {
            self.compile_nodes(&group.children, state)?;
            self.publish_outputs(owner_ref, &group.outputs, state)?;
            for field in state.output_slots() {
                self.builder.record_owner_field(owner_ref, field);
            }
            self.builder
                .record_owner_span(owner_ref, span_start, self.builder.operation_count());
            return Ok(());
        }

        let origin = PlanOrigin::Authored(owner_ref);
        let parent = state.clone();
        // Group distributions observe the parent context at the group boundary,
        // before private children mutate their isolated state.
        let mask = self.builder.add_field(LogicalFieldKind::Mask, origin);
        let mask_inputs = self.resolve_distribution_inputs(owner_ref, &group.masks, &parent)?;
        let mask_op = self.builder.add_operation(TerrainOp {
            origin,
            reach: crate::mask::distribution_reach(&group.masks, self.mask_assets),
            aux_reach: AuxReach::HeightOnly,
            kind: TerrainOpKind::EvaluateMask {
                input_height: parent.height,
                input_fields: mask_inputs,
                output_mask: mask,
            },
        });
        self.record_distribution_dependencies(mask_op, &group.masks);

        let private_seed = self.builder.add_field(LogicalFieldKind::Height, origin);
        let source = match &group.input_mode {
            GroupInputMode::CopyInput => SeedSource::Copy(parent.height),
            GroupInputMode::EmptyHeight => SeedSource::Zero,
            GroupInputMode::SelectedField(SelectedGroupInput::Field(field)) => {
                let Some(slot) = parent.field(field) else {
                    return Err(TerrainPlanDiagnostic::UnavailableField {
                        owner: owner_ref,
                        field: field.clone(),
                    });
                };
                SeedSource::Selected(slot)
            }
            GroupInputMode::SelectedField(SelectedGroupInput::Output(output)) => {
                let Some(slot) = self.published_outputs.get(output).copied() else {
                    return Err(TerrainPlanDiagnostic::UnavailableOutput {
                        owner: owner_ref,
                        output: *output,
                    });
                };
                SeedSource::Selected(slot)
            }
        };
        let seed = self.builder.add_operation(TerrainOp {
            origin,
            reach: Reach::LOCAL,
            aux_reach: AuxReach::HeightOnly,
            kind: TerrainOpKind::Seed {
                source,
                output: private_seed,
            },
        });
        if let GroupInputMode::SelectedField(SelectedGroupInput::Output(output)) = &group.input_mode
        {
            self.builder.record_authored_dependency(
                seed,
                NodeRef::Output(*output),
                crate::deps::DepKind::GroupInput,
            );
        }

        let mut private = parent.clone();
        private.height = private_seed;
        self.compile_nodes(&group.children, &mut private)?;

        let mut aux = Vec::new();
        for (field, child) in &private.aux {
            let parent_slot = parent.field(field);
            if parent_slot == Some(*child) {
                continue;
            }
            let output = self
                .builder
                .add_field(LogicalFieldKind::Auxiliary(field.clone()), origin);
            aux.push(GroupAuxComposite {
                field: field.clone(),
                parent: parent_slot,
                child: *child,
                output,
            });
        }

        let output = self.builder.add_field(LogicalFieldKind::Height, origin);
        let mode = if matches!(group.group_kind, GroupKind::Biome)
            && matches!(group.input_mode, GroupInputMode::CopyInput)
        {
            GroupCompositeMode::BiomeHeightDelta
        } else {
            GroupCompositeMode::Standard
        };
        self.builder.add_operation(TerrainOp {
            origin,
            reach: Reach::LOCAL,
            aux_reach: if aux.is_empty() {
                AuxReach::HeightOnly
            } else {
                AuxReach::PerTexel
            },
            kind: TerrainOpKind::CompositeGroup {
                group: group.id,
                parent: parent.height,
                private_seed,
                child_output: private.height,
                mask,
                output,
                mode,
            },
        });

        for composite in &aux {
            self.builder.add_operation(TerrainOp {
                origin,
                reach: Reach::LOCAL,
                aux_reach: AuxReach::PerTexel,
                kind: TerrainOpKind::CompositeAuxField {
                    owner: owner_ref,
                    mask,
                    composite: composite.clone(),
                },
            });
        }

        state.height = output;
        state.aux = parent.aux;
        for merged in aux {
            state.set_aux(merged.field, merged.output);
        }
        self.publish_outputs(owner_ref, &group.outputs, state)?;
        for field in state.output_slots() {
            self.builder.record_owner_field(owner_ref, field);
        }
        self.builder
            .record_owner_span(owner_ref, span_start, self.builder.operation_count());
        Ok(())
    }

    fn publish_outputs(
        &mut self,
        owner: NodeRef,
        outputs: &[NamedOutputDecl],
        state: &PlanState,
    ) -> Result<(), TerrainPlanDiagnostic> {
        for output in outputs.iter().filter(|output| output.enabled) {
            let Some(source) = state.field(&output.field) else {
                return Err(TerrainPlanDiagnostic::UnavailableField {
                    owner,
                    field: output.field.clone(),
                });
            };
            self.builder.record_output_owner(output.id, owner);
            self.builder.add_operation(TerrainOp {
                origin: PlanOrigin::Authored(NodeRef::Output(output.id)),
                reach: Reach::LOCAL,
                aux_reach: AuxReach::HeightOnly,
                kind: TerrainOpKind::PublishOutput {
                    output: output.id,
                    source,
                },
            });
            self.published_outputs.insert(output.id, source);
        }
        Ok(())
    }

    fn resolve_layer_inputs(
        &self,
        layer: &Layer,
        state: &PlanState,
    ) -> Result<Vec<FieldSlot>, TerrainPlanDiagnostic> {
        let owner = NodeRef::Layer(layer.id());
        let mut slots = Vec::new();
        let mut fields = layer.kind.required_fields();
        fields.extend(layer.kind.optional_fields());
        for field in fields {
            if field != FieldId::Height {
                if let Some(slot) = state.field(&field) {
                    push_slot(&mut slots, slot);
                }
            }
        }
        for binding in &layer.common.param_bindings {
            match &binding.source {
                BindingSource::Field(field) => {
                    let Some(slot) = state.field(field) else {
                        return Err(TerrainPlanDiagnostic::UnavailableField {
                            owner,
                            field: field.clone(),
                        });
                    };
                    push_slot(&mut slots, slot);
                }
                BindingSource::LayerOutput(output) | BindingSource::GroupOutput(output) => {
                    let Some(slot) = self.published_outputs.get(output).copied() else {
                        return Err(TerrainPlanDiagnostic::UnavailableOutput {
                            owner,
                            output: *output,
                        });
                    };
                    push_slot(&mut slots, slot);
                }
                BindingSource::Mask(_) | BindingSource::Constant(_) => {}
            }
        }
        Ok(slots)
    }

    fn resolve_distribution_inputs(
        &self,
        owner: NodeRef,
        distribution: &Distribution,
        state: &PlanState,
    ) -> Result<Vec<FieldSlot>, TerrainPlanDiagnostic> {
        let mut fields = Vec::new();
        let mut outputs = Vec::new();
        for node in &distribution.nodes {
            collect_node_sources(node, self.mask_assets, &mut fields, &mut outputs);
        }
        for entry in distribution.iter() {
            if let Some(asset) = self
                .mask_assets
                .iter()
                .find(|asset| asset.id == entry.mask.id)
            {
                collect_mask_source(&asset.source, &mut fields, &mut outputs);
            }
        }
        let mut slots = Vec::new();
        for field in fields {
            if let Some(slot) = state.field(&field) {
                push_slot(&mut slots, slot);
            }
        }
        for output in outputs {
            // Missing identifiers are reported in preflight; an existing but not-yet
            // published output is a deterministic availability error at the consumer.
            let Some(slot) = self.published_outputs.get(&output).copied() else {
                return Err(TerrainPlanDiagnostic::UnavailableOutput { owner, output });
            };
            push_slot(&mut slots, slot);
        }
        Ok(slots)
    }

    fn record_distribution_dependencies(
        &mut self,
        operation: super::PlanOpId,
        distribution: &Distribution,
    ) {
        for mask in distribution_mask_ids(distribution) {
            self.builder.record_authored_dependency(
                operation,
                NodeRef::Mask(mask),
                crate::deps::DepKind::MaskRef,
            );
            if let Some(asset) = self.mask_assets.iter().find(|asset| asset.id == mask) {
                if let MaskSource::LayerOutput { output_id } = asset.source {
                    self.builder.record_authored_dependency(
                        operation,
                        NodeRef::Output(output_id),
                        crate::deps::DepKind::NamedOutput,
                    );
                }
            }
        }
    }
}

fn configured_layer_reach(layer: &Layer) -> Reach {
    if !layer.common.param_bindings.is_empty() {
        Reach::Full
    } else {
        layer.kind.intrinsic_reach()
    }
}

fn collect_node_sources(
    node: &DistNode,
    mask_assets: &[MaskAsset],
    fields: &mut Vec<FieldId>,
    outputs: &mut Vec<OutputId>,
) {
    match &node.kind {
        DistNodeKind::Flow { .. } => push_field(fields, FieldId::FlowAccumulation),
        DistNodeKind::Climate { channel } => push_field(
            fields,
            match channel {
                ClimateMaskChannel::Temperature => FieldId::Temperature,
                ClimateMaskChannel::Rainfall => FieldId::Rainfall,
                ClimateMaskChannel::Humidity => FieldId::Humidity,
                ClimateMaskChannel::Snow => FieldId::Snow,
                ClimateMaskChannel::SoilMoisture => FieldId::SoilMoisture,
                ClimateMaskChannel::WindExposure => FieldId::WindExposure,
            },
        ),
        DistNodeKind::MaskAsset { mask }
        | DistNodeKind::Paint { mask }
        | DistNodeKind::ImportedMask { mask }
        | DistNodeKind::Distance { mask, .. } => {
            if let Some(asset) = mask_assets.iter().find(|asset| asset.id == mask.id) {
                collect_mask_source(&asset.source, fields, outputs);
            }
        }
        _ => {}
    }
    for child in &node.children {
        collect_node_sources(child, mask_assets, fields, outputs);
    }
}

fn collect_mask_source(
    source: &MaskSource,
    fields: &mut Vec<FieldId>,
    outputs: &mut Vec<OutputId>,
) {
    let field = match source {
        MaskSource::FlowDirection => Some(FieldId::FlowDirection),
        MaskSource::FlowAccumulation { .. } => Some(FieldId::FlowAccumulation),
        MaskSource::Wetness => Some(FieldId::Wetness),
        MaskSource::Sediment => Some(FieldId::Sediment),
        MaskSource::Erosion => Some(FieldId::Erosion),
        MaskSource::Deposition => Some(FieldId::Deposition),
        MaskSource::Hardness => Some(FieldId::Hardness),
        MaskSource::Temperature => Some(FieldId::Temperature),
        MaskSource::Rainfall => Some(FieldId::Rainfall),
        MaskSource::Humidity => Some(FieldId::Humidity),
        MaskSource::Snow => Some(FieldId::Snow),
        MaskSource::SoilMoisture => Some(FieldId::SoilMoisture),
        MaskSource::WindExposure => Some(FieldId::WindExposure),
        MaskSource::Named(name) => Some(FieldId::Named(name.clone())),
        MaskSource::LayerOutput { output_id } => {
            if !outputs.contains(output_id) {
                outputs.push(*output_id);
            }
            None
        }
        _ => None,
    };
    if let Some(field) = field {
        push_field(fields, field);
    }
}

fn push_slot(slots: &mut Vec<FieldSlot>, slot: FieldSlot) {
    if !slots.contains(&slot) {
        slots.push(slot);
    }
}

fn push_field(fields: &mut Vec<FieldId>, field: FieldId) {
    if !fields.contains(&field) {
        fields.push(field);
    }
}

fn distribution_mask_ids(distribution: &Distribution) -> Vec<MaskId> {
    let mut masks = Vec::new();
    for entry in distribution.iter() {
        if !masks.contains(&entry.mask.id) {
            masks.push(entry.mask.id);
        }
    }
    for node in &distribution.nodes {
        collect_node_masks(node, &mut masks);
    }
    masks
}

fn collect_node_masks(node: &DistNode, masks: &mut Vec<MaskId>) {
    let referenced = match &node.kind {
        DistNodeKind::MaskAsset { mask }
        | DistNodeKind::Paint { mask }
        | DistNodeKind::ImportedMask { mask }
        | DistNodeKind::Distance { mask, .. } => Some(mask.id),
        _ => None,
    };
    if let Some(mask) = referenced {
        if !masks.contains(&mask) {
            masks.push(mask);
        }
    }
    for child in &node.children {
        collect_node_masks(child, masks);
    }
}

fn collect_plan_node_selection(nodes: &[StackNode]) -> Vec<(NodeRef, PlanNodeSelection)> {
    fn walk(
        nodes: &[StackNode],
        ancestor_participates: bool,
        selection: &mut Vec<(NodeRef, PlanNodeSelection)>,
    ) {
        let soloing = ancestor_participates && nodes.iter().any(StackNode::contains_solo);
        for node in nodes {
            let node_selection = if !ancestor_participates {
                PlanNodeSelection::ExcludedBySolo
            } else if !soloing {
                PlanNodeSelection::Unfiltered
            } else if node.contains_solo() {
                PlanNodeSelection::IncludedBySolo
            } else {
                PlanNodeSelection::ExcludedBySolo
            };
            selection.push((node_ref(node), node_selection));
            if let StackNode::Group(group) = node {
                walk(&group.children, node_selection.participates(), selection);
            }
        }
    }

    let mut selection = Vec::new();
    walk(nodes, true, &mut selection);
    selection
}

fn node_ref(node: &StackNode) -> NodeRef {
    match node {
        StackNode::Layer(layer) => NodeRef::Layer(layer.id()),
        StackNode::Group(group) => NodeRef::Group(group.id),
    }
}

fn preflight(stack: &LayerStack, mask_assets: &[MaskAsset]) -> Vec<TerrainPlanDiagnostic> {
    let known_masks: HashSet<_> = mask_assets.iter().map(|asset| asset.id).collect();
    let mut node_ids = HashSet::new();
    let mut output_ids = HashSet::new();
    let mut active_output_ids = HashSet::new();
    let mut diagnostics = Vec::new();
    collect_id_diagnostics(
        &stack.nodes,
        &mut node_ids,
        &mut output_ids,
        &mut diagnostics,
    );
    collect_active_outputs(&stack.nodes, true, &mut active_output_ids);
    collect_reference_diagnostics(
        &stack.nodes,
        mask_assets,
        &known_masks,
        &output_ids,
        &active_output_ids,
        &mut diagnostics,
    );
    let dependencies = crate::deps::DependencyGraph::build_from_document(stack, mask_assets);
    if let Err(crate::deps::DepError::Cycle(nodes)) = dependencies.detect_cycle() {
        diagnostics.push(TerrainPlanDiagnostic::DependencyCycle { nodes });
    }
    diagnostics
}

fn collect_active_outputs(
    nodes: &[StackNode],
    ancestor_enabled: bool,
    active: &mut HashSet<OutputId>,
) {
    let soloing = ancestor_enabled && nodes.iter().any(StackNode::contains_solo);
    for node in nodes {
        let selected = ancestor_enabled && (!soloing || node.contains_solo());
        match node {
            StackNode::Layer(layer) => {
                if selected && layer.common.enabled {
                    active.extend(
                        layer
                            .common
                            .outputs
                            .iter()
                            .filter(|output| output.enabled)
                            .map(|output| output.id),
                    );
                }
            }
            StackNode::Group(group) => {
                let enabled = selected && group.enabled;
                if enabled {
                    active.extend(
                        group
                            .outputs
                            .iter()
                            .filter(|output| output.enabled)
                            .map(|output| output.id),
                    );
                }
                collect_active_outputs(&group.children, enabled, active);
            }
        }
    }
}

fn collect_id_diagnostics(
    nodes: &[StackNode],
    nodes_seen: &mut HashSet<LayerId>,
    outputs_seen: &mut HashSet<OutputId>,
    diagnostics: &mut Vec<TerrainPlanDiagnostic>,
) {
    for node in nodes {
        let (id, outputs, children) = match node {
            StackNode::Layer(layer) => (layer.id(), layer.common.outputs.as_slice(), None),
            StackNode::Group(group) => (
                group.id,
                group.outputs.as_slice(),
                Some(group.children.as_slice()),
            ),
        };
        if !nodes_seen.insert(id) {
            diagnostics.push(TerrainPlanDiagnostic::DuplicateNodeId(id));
        }
        for output in outputs {
            if !outputs_seen.insert(output.id) {
                diagnostics.push(TerrainPlanDiagnostic::DuplicateOutputId(output.id));
            }
        }
        if let Some(children) = children {
            collect_id_diagnostics(children, nodes_seen, outputs_seen, diagnostics);
        }
    }
}

fn collect_reference_diagnostics(
    nodes: &[StackNode],
    mask_assets: &[MaskAsset],
    known_masks: &HashSet<MaskId>,
    known_outputs: &HashSet<OutputId>,
    active_outputs: &HashSet<OutputId>,
    diagnostics: &mut Vec<TerrainPlanDiagnostic>,
) {
    for node in nodes {
        match node {
            StackNode::Layer(layer) if layer.common.enabled => {
                let owner = NodeRef::Layer(layer.id());
                validate_distribution(
                    owner,
                    &layer.common.masks,
                    mask_assets,
                    known_masks,
                    known_outputs,
                    active_outputs,
                    diagnostics,
                );
                for binding in &layer.common.param_bindings {
                    match binding.source {
                        BindingSource::Mask(mask) if !known_masks.contains(&mask) => {
                            diagnostics.push(TerrainPlanDiagnostic::MissingMask { owner, mask })
                        }
                        BindingSource::LayerOutput(output) | BindingSource::GroupOutput(output) => {
                            if !known_outputs.contains(&output) {
                                diagnostics
                                    .push(TerrainPlanDiagnostic::MissingOutput { owner, output });
                            } else if !active_outputs.contains(&output) {
                                diagnostics
                                    .push(TerrainPlanDiagnostic::DisabledOutput { owner, output });
                            }
                        }
                        _ => {}
                    }
                }
            }
            StackNode::Layer(_) => {}
            StackNode::Group(group) if group.enabled => {
                let owner = NodeRef::Group(group.id);
                validate_distribution(
                    owner,
                    &group.masks,
                    mask_assets,
                    known_masks,
                    known_outputs,
                    active_outputs,
                    diagnostics,
                );
                if let GroupInputMode::SelectedField(SelectedGroupInput::Output(output)) =
                    group.input_mode
                {
                    if !known_outputs.contains(&output) {
                        diagnostics.push(TerrainPlanDiagnostic::MissingOutput { owner, output });
                    } else if !active_outputs.contains(&output) {
                        diagnostics.push(TerrainPlanDiagnostic::DisabledOutput { owner, output });
                    }
                }
                collect_reference_diagnostics(
                    &group.children,
                    mask_assets,
                    known_masks,
                    known_outputs,
                    active_outputs,
                    diagnostics,
                );
            }
            StackNode::Group(_) => {}
        }
    }
}

fn validate_distribution(
    owner: NodeRef,
    distribution: &Distribution,
    mask_assets: &[MaskAsset],
    known_masks: &HashSet<MaskId>,
    known_outputs: &HashSet<OutputId>,
    active_outputs: &HashSet<OutputId>,
    diagnostics: &mut Vec<TerrainPlanDiagnostic>,
) {
    for entry in distribution.iter() {
        if !known_masks.contains(&entry.mask.id) {
            diagnostics.push(TerrainPlanDiagnostic::MissingMask {
                owner,
                mask: entry.mask.id,
            });
        }
    }
    for node in &distribution.nodes {
        validate_dist_node(owner, node, known_masks, diagnostics);
    }
    for mask in distribution_mask_ids(distribution) {
        let Some(asset) = mask_assets.iter().find(|asset| asset.id == mask) else {
            continue;
        };
        if let MaskSource::LayerOutput { output_id } = asset.source {
            if !known_outputs.contains(&output_id) {
                diagnostics.push(TerrainPlanDiagnostic::MissingOutput {
                    owner,
                    output: output_id,
                });
            } else if !active_outputs.contains(&output_id) {
                diagnostics.push(TerrainPlanDiagnostic::DisabledOutput {
                    owner,
                    output: output_id,
                });
            }
        }
    }
}

fn validate_dist_node(
    owner: NodeRef,
    node: &DistNode,
    known_masks: &HashSet<MaskId>,
    diagnostics: &mut Vec<TerrainPlanDiagnostic>,
) {
    if let DistNodeKind::MaskAsset { mask } = &node.kind {
        if !known_masks.contains(&mask.id) {
            diagnostics.push(TerrainPlanDiagnostic::MissingMask {
                owner,
                mask: mask.id,
            });
        }
    }
    for child in &node.children {
        validate_dist_node(owner, child, known_masks, diagnostics);
    }
}
