//! Recursive lowering from the authored layer tree into the terrain-plan IR.

use std::collections::HashSet;

use crate::deps::NodeRef;
use crate::field_data::FieldId;
use crate::ids::{LayerId, OutputId};
use crate::invalidation::{AuxReach, Reach};
use crate::layer::{
    BindingSource, GroupInputMode, GroupKind, Layer, LayerGroup, LayerStack, NamedOutputDecl,
    StackNode,
};
use crate::mask::{DistNode, DistNodeKind, Distribution, MaskAsset, MaskId};

use super::{
    CompiledTerrainPlan, FieldSlot, GroupAuxComposite, GroupCompositeMode, LogicalFieldKind,
    PlanBuildError, PlanOrigin, SeedSource, TerrainOp, TerrainOpKind, TerrainPlanBuilder,
    TerrainPlanStamp,
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
    #[error("{owner:?} cannot publish unavailable field {field:?}")]
    UnavailableField { owner: NodeRef, field: FieldId },
    #[error("solo layer {layer:?} is deferred to the solo-plan compiler")]
    UnsupportedSolo { layer: LayerId },
    #[error("selected-field input for group {group:?} is deferred to cross-edge compilation")]
    UnsupportedSelectedField { group: LayerId },
    #[error("output binding on {owner:?} is deferred to cross-edge compilation")]
    UnsupportedOutputBinding { owner: NodeRef },
    #[error(transparent)]
    Build(#[from] PlanBuildError),
}

/// Compile one authored stack for a particular structural revision.
///
/// The compiler reads payload metadata but stores no CPU fields or backend
/// resources. Disabled nodes emit no operations. Solo and selected-field
/// semantics fail explicitly until their dedicated compiler phases land.
pub fn compile_terrain_plan(
    stack: &LayerStack,
    mask_assets: &[MaskAsset],
    stamp: TerrainPlanStamp,
) -> Result<CompiledTerrainPlan, Vec<TerrainPlanDiagnostic>> {
    let diagnostics = preflight(stack, mask_assets);
    if !diagnostics.is_empty() {
        return Err(diagnostics);
    }

    let mut compiler = Compiler {
        builder: TerrainPlanBuilder::new(stamp),
    };
    let root = compiler
        .builder
        .add_field(LogicalFieldKind::Height, PlanOrigin::Root);
    compiler.builder.add_operation(TerrainOp {
        origin: PlanOrigin::Root,
        reach: Reach::LOCAL,
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

    fn all_aux_slots(&self) -> Vec<FieldSlot> {
        self.aux.iter().map(|(_, slot)| *slot).collect()
    }
}

struct Compiler {
    builder: TerrainPlanBuilder,
}

impl Compiler {
    fn compile_nodes(
        &mut self,
        nodes: &[StackNode],
        state: &mut PlanState,
    ) -> Result<(), TerrainPlanDiagnostic> {
        for node in nodes {
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
        let origin = PlanOrigin::Authored(owner_ref);
        let base = state.height;
        let input_fields = resolve_layer_inputs(layer, state);
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
        self.builder.add_operation(TerrainOp {
            origin,
            reach,
            kind: TerrainOpKind::RunLayerKernel {
                layer: layer.id(),
                type_id: layer.kind.type_id().into(),
                input_height: base,
                input_fields,
                output_candidate: candidate,
                output_fields,
            },
        });
        for (field, slot) in produced {
            state.set_aux(field, slot);
        }

        let mask = self.builder.add_field(LogicalFieldKind::Mask, origin);
        self.builder.add_operation(TerrainOp {
            origin,
            reach: distribution_reach(&layer.common.masks),
            kind: TerrainOpKind::EvaluateMask {
                input_height: base,
                input_fields: state.all_aux_slots(),
                output_mask: mask,
            },
        });
        let output = self.builder.add_field(LogicalFieldKind::Height, origin);
        self.builder.add_operation(TerrainOp {
            origin,
            reach: Reach::LOCAL,
            kind: TerrainOpKind::CompositeLayer {
                layer: layer.id(),
                base,
                candidate,
                mask,
                output,
            },
        });
        state.height = output;
        self.publish_outputs(owner_ref, &layer.common.outputs, state)
    }

    fn compile_group(
        &mut self,
        group: &LayerGroup,
        state: &mut PlanState,
    ) -> Result<(), TerrainPlanDiagnostic> {
        let owner_ref = NodeRef::Group(group.id);
        if group.is_pass_through() {
            self.compile_nodes(&group.children, state)?;
            return self.publish_outputs(owner_ref, &group.outputs, state);
        }

        let origin = PlanOrigin::Authored(owner_ref);
        let parent = state.clone();
        // Group distributions observe the parent context at the group boundary,
        // before private children mutate their isolated state.
        let mask = self.builder.add_field(LogicalFieldKind::Mask, origin);
        self.builder.add_operation(TerrainOp {
            origin,
            reach: distribution_reach(&group.masks),
            kind: TerrainOpKind::EvaluateMask {
                input_height: parent.height,
                input_fields: parent.all_aux_slots(),
                output_mask: mask,
            },
        });

        let private_seed = self.builder.add_field(LogicalFieldKind::Height, origin);
        let source = match group.input_mode {
            GroupInputMode::CopyInput => SeedSource::Copy(parent.height),
            GroupInputMode::EmptyHeight => SeedSource::Zero,
            GroupInputMode::SelectedField(_) => {
                return Err(TerrainPlanDiagnostic::UnsupportedSelectedField { group: group.id });
            }
        };
        self.builder.add_operation(TerrainOp {
            origin,
            reach: Reach::LOCAL,
            kind: TerrainOpKind::Seed {
                source,
                output: private_seed,
            },
        });

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
            kind: TerrainOpKind::CompositeGroup {
                group: group.id,
                parent: parent.height,
                private_seed,
                child_output: private.height,
                mask,
                output,
                mode,
                aux: aux.clone(),
            },
        });

        state.height = output;
        state.aux = parent.aux;
        for merged in aux {
            state.set_aux(merged.field, merged.output);
        }
        self.publish_outputs(owner_ref, &group.outputs, state)
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
            self.builder.add_operation(TerrainOp {
                origin: PlanOrigin::Authored(NodeRef::Output(output.id)),
                reach: Reach::LOCAL,
                kind: TerrainOpKind::PublishOutput {
                    output: output.id,
                    source,
                },
            });
        }
        Ok(())
    }
}

fn resolve_layer_inputs(layer: &Layer, state: &PlanState) -> Vec<FieldSlot> {
    let mut fields = layer.kind.required_fields();
    fields.extend(layer.kind.optional_fields());
    fields.extend(layer.common.param_bindings.iter().filter_map(|binding| {
        if let BindingSource::Field(field) = &binding.source {
            Some(field.clone())
        } else {
            None
        }
    }));
    let mut slots = Vec::new();
    for field in fields {
        if field == FieldId::Height {
            continue;
        }
        if let Some(slot) = state.field(&field) {
            if !slots.contains(&slot) {
                slots.push(slot);
            }
        }
    }
    slots
}

fn configured_layer_reach(layer: &Layer) -> Reach {
    if !layer.common.param_bindings.is_empty() || layer.kind.aux_reach() == AuxReach::Global {
        Reach::Full
    } else {
        layer.kind.intrinsic_reach()
    }
}

fn distribution_reach(distribution: &Distribution) -> Reach {
    if distribution.is_empty() {
        Reach::LOCAL
    } else {
        // The full mask dependency/reach compiler lands with dependency-aware
        // validation. Full is conservative and cannot under-invalidate here.
        Reach::Full
    }
}

fn preflight(stack: &LayerStack, mask_assets: &[MaskAsset]) -> Vec<TerrainPlanDiagnostic> {
    let known_masks: HashSet<_> = mask_assets.iter().map(|asset| asset.id).collect();
    let mut node_ids = HashSet::new();
    let mut output_ids = HashSet::new();
    let mut diagnostics = Vec::new();
    collect_id_diagnostics(
        &stack.nodes,
        &mut node_ids,
        &mut output_ids,
        &mut diagnostics,
    );
    collect_reference_diagnostics(&stack.nodes, &known_masks, &output_ids, &mut diagnostics);
    diagnostics
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
    known_masks: &HashSet<MaskId>,
    known_outputs: &HashSet<OutputId>,
    diagnostics: &mut Vec<TerrainPlanDiagnostic>,
) {
    for node in nodes {
        match node {
            StackNode::Layer(layer) if layer.common.enabled => {
                let owner = NodeRef::Layer(layer.id());
                if layer.common.solo {
                    diagnostics.push(TerrainPlanDiagnostic::UnsupportedSolo { layer: layer.id() });
                }
                validate_distribution(owner, &layer.common.masks, known_masks, diagnostics);
                for binding in &layer.common.param_bindings {
                    match binding.source {
                        BindingSource::Mask(mask) if !known_masks.contains(&mask) => {
                            diagnostics.push(TerrainPlanDiagnostic::MissingMask { owner, mask })
                        }
                        BindingSource::LayerOutput(output) | BindingSource::GroupOutput(output) => {
                            if !known_outputs.contains(&output) {
                                diagnostics
                                    .push(TerrainPlanDiagnostic::MissingOutput { owner, output });
                            } else {
                                diagnostics.push(TerrainPlanDiagnostic::UnsupportedOutputBinding {
                                    owner,
                                });
                            }
                        }
                        _ => {}
                    }
                }
            }
            StackNode::Layer(_) => {}
            StackNode::Group(group) if group.enabled => {
                let owner = NodeRef::Group(group.id);
                validate_distribution(owner, &group.masks, known_masks, diagnostics);
                if matches!(group.input_mode, GroupInputMode::SelectedField(_)) {
                    diagnostics
                        .push(TerrainPlanDiagnostic::UnsupportedSelectedField { group: group.id });
                }
                collect_reference_diagnostics(
                    &group.children,
                    known_masks,
                    known_outputs,
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
    known_masks: &HashSet<MaskId>,
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
