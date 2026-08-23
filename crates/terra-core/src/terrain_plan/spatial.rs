//! Project-mode spatial contracts for compiled terrain operations.

use std::collections::HashSet;

use crate::deps::NodeRef;
use crate::invalidation::{
    InfiniteOperationCapability, OperationSpatialContract, Reach, SpatialRejectReason,
};
use crate::layer::LayerStack;
use crate::mask::{DistNode, DistNodeKind, Distribution, MaskAsset, MaskId, MaskSource};

use super::{CompiledTerrainPlan, PlanOpId, TerrainOpKind};

/// Inspect the complete spatial contract of one compiled operation.
///
/// The compiled `Reach` remains authoritative for halo size. This function adds
/// the project-mode capability without recalculating reach or relying on a GPU
/// backend allow-list.
pub fn operation_spatial_contract(
    stack: &LayerStack,
    mask_assets: &[MaskAsset],
    plan: &CompiledTerrainPlan,
    operation_id: PlanOpId,
) -> Option<OperationSpatialContract> {
    let operation = plan.operation(operation_id)?;
    let infinite = match &operation.kind {
        TerrainOpKind::Seed { .. }
        | TerrainOpKind::CompositeLayer { .. }
        | TerrainOpKind::CompositeGroup { .. }
        | TerrainOpKind::CompositeAuxField { .. }
        | TerrainOpKind::PublishOutput { .. } => InfiniteOperationCapability::Direct,
        TerrainOpKind::RunLayerKernel { layer, .. } => {
            let Some(layer) = stack.find(*layer) else {
                return Some(OperationSpatialContract {
                    reach: operation.reach,
                    aux_reach: operation.aux_reach,
                    infinite: InfiniteOperationCapability::Unsupported(
                        SpatialRejectReason::UnclassifiedOperation,
                    ),
                });
            };
            if !layer.common.param_bindings.is_empty() {
                InfiniteOperationCapability::Unsupported(
                    SpatialRejectReason::GlobalParameterReduction,
                )
            } else {
                layer.kind.infinite_capability()
            }
        }
        TerrainOpKind::EvaluateMask { .. } => {
            let owner = plan.provenance().owner_of(operation_id);
            owner_distribution(stack, owner).map_or_else(
                || {
                    InfiniteOperationCapability::Unsupported(
                        SpatialRejectReason::UnclassifiedOperation,
                    )
                },
                |distribution| distribution_infinite_capability(distribution, mask_assets),
            )
        }
    };

    let infinite = if operation.reach == Reach::Full && infinite.rejection().is_none() {
        InfiniteOperationCapability::Unsupported(SpatialRejectReason::RequiresCompleteField)
    } else {
        infinite
    };
    Some(OperationSpatialContract {
        reach: operation.reach,
        aux_reach: operation.aux_reach,
        infinite,
    })
}

fn owner_distribution(stack: &LayerStack, owner: Option<NodeRef>) -> Option<&Distribution> {
    match owner? {
        NodeRef::Layer(id) => stack.find(id).map(|layer| &layer.common.masks),
        NodeRef::Group(id) => stack.find_group(id).map(|group| &group.masks),
        NodeRef::Mask(_) | NodeRef::Output(_) => None,
    }
}

/// Sparse compatibility of a configured mask distribution. Reach is computed
/// separately by `distribution_reach`; this only rejects bounded authored data
/// and algorithms that need a complete field.
pub fn distribution_infinite_capability(
    distribution: &Distribution,
    mask_assets: &[MaskAsset],
) -> InfiniteOperationCapability {
    let mut visiting = HashSet::new();
    for entry in &distribution.entries {
        let capability = mask_ref_capability(entry.mask.id, mask_assets, &mut visiting);
        if capability.rejection().is_some() {
            return capability;
        }
    }
    for node in &distribution.nodes {
        let capability = node_capability(node, mask_assets, &mut visiting);
        if capability.rejection().is_some() {
            return capability;
        }
    }
    InfiniteOperationCapability::Direct
}

fn node_capability(
    node: &DistNode,
    mask_assets: &[MaskAsset],
    visiting: &mut HashSet<MaskId>,
) -> InfiniteOperationCapability {
    if !node.enabled {
        return InfiniteOperationCapability::Direct;
    }
    use DistNodeKind::*;
    let capability = match &node.kind {
        Paint { mask } | ImportedMask { mask } => {
            let _ = mask;
            InfiniteOperationCapability::Unsupported(SpatialRejectReason::BoundedAuthoredData)
        }
        Polygon { .. } | Spline { .. } => {
            InfiniteOperationCapability::Unsupported(SpatialRejectReason::BoundedAuthoredData)
        }
        Distance { .. } | EffectDistortion { .. } | EffectDilate { .. } | EffectErode { .. } => {
            InfiniteOperationCapability::Unsupported(SpatialRejectReason::RequiresCompleteField)
        }
        MaskAsset { mask } => mask_ref_capability(mask.id, mask_assets, visiting),
        _ => InfiniteOperationCapability::Direct,
    };
    if capability.rejection().is_some() {
        return capability;
    }
    for child in &node.children {
        let capability = node_capability(child, mask_assets, visiting);
        if capability.rejection().is_some() {
            return capability;
        }
    }
    InfiniteOperationCapability::Direct
}

fn mask_ref_capability(
    id: MaskId,
    mask_assets: &[MaskAsset],
    visiting: &mut HashSet<MaskId>,
) -> InfiniteOperationCapability {
    if !visiting.insert(id) {
        return InfiniteOperationCapability::Unsupported(
            SpatialRejectReason::UnclassifiedOperation,
        );
    }
    let result = mask_assets.iter().find(|asset| asset.id == id).map_or(
        InfiniteOperationCapability::Unsupported(SpatialRejectReason::UnclassifiedOperation),
        |asset| mask_source_infinite_capability(&asset.source),
    );
    visiting.remove(&id);
    result
}

/// Infinite capability of a standalone mask source. Dynamic field sources are
/// direct here; the compiled dependency walk separately verifies their complete
/// producer chain and global-auxiliary reach.
pub fn mask_source_infinite_capability(source: &MaskSource) -> InfiniteOperationCapability {
    match source {
        MaskSource::Painted { .. } => {
            InfiniteOperationCapability::Unsupported(SpatialRejectReason::BoundedAuthoredData)
        }
        MaskSource::DistanceField { .. } => {
            InfiniteOperationCapability::Unsupported(SpatialRejectReason::RequiresCompleteField)
        }
        MaskSource::LayerOutput { .. }
        | MaskSource::None
        | MaskSource::Constant(_)
        | MaskSource::Height { .. }
        | MaskSource::Slope { .. }
        | MaskSource::Aspect { .. }
        | MaskSource::Curvature { .. }
        | MaskSource::Convexity
        | MaskSource::Concavity
        | MaskSource::AmbientOcclusion { .. }
        | MaskSource::FlowDirection
        | MaskSource::FlowAccumulation { .. }
        | MaskSource::Wetness
        | MaskSource::Sediment
        | MaskSource::Erosion
        | MaskSource::Deposition
        | MaskSource::Hardness
        | MaskSource::Temperature
        | MaskSource::Rainfall
        | MaskSource::Humidity
        | MaskSource::Snow
        | MaskSource::SoilMoisture
        | MaskSource::WindExposure
        | MaskSource::Noise { .. }
        | MaskSource::Named(_) => InfiniteOperationCapability::Direct,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mask::{MaskAsset, MaskRef};

    #[test]
    fn painted_mask_is_bounded_authored_data() {
        let mut asset = MaskAsset::new_painted(MaskId::new(), "paint", 8);
        asset.prepare_for_document();
        let distribution = Distribution::from_refs(vec![MaskRef::new(asset.id)]);
        assert_eq!(
            distribution_infinite_capability(&distribution, &[asset]),
            InfiniteOperationCapability::Unsupported(SpatialRejectReason::BoundedAuthoredData)
        );
    }
}
