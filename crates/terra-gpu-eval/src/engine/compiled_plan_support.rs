//! Shared support for compiled-plan execution and refinement.

use super::compiled_plan::CompiledDispatchError;
use super::*;

pub(super) fn owner_layer_id(owner: Option<NodeRef>) -> Option<LayerId> {
    match owner {
        Some(NodeRef::Layer(layer) | NodeRef::Group(layer)) => Some(layer),
        _ => None,
    }
}

pub(super) fn plan_fallback_diagnostic(
    plan: &CompiledTerrainPlan,
    stack: &LayerStack,
    operation: PlanOpId,
    reason: GpuFallbackReason,
) -> GpuFallbackDiagnostic {
    let owner = plan.provenance().owner_of(operation);
    let layer_id = owner_layer_id(owner).unwrap_or_default();
    let layer_name = match owner {
        Some(NodeRef::Layer(layer)) => stack
            .find(layer)
            .map(|layer| layer.common.name.clone())
            .unwrap_or_else(|| format!("layer {layer:?}")),
        Some(NodeRef::Group(group)) => stack
            .find_group(group)
            .map(|group| group.name.clone())
            .unwrap_or_else(|| format!("group {group:?}")),
        Some(other) => format!("{other:?}"),
        None => "terrain plan root".to_string(),
    };
    GpuFallbackDiagnostic {
        operation: Some(operation),
        owner,
        layer_index: operation.index(),
        layer_id,
        layer_name,
        reason,
    }
}

pub(super) fn plan_fallback_result(
    metrics: HeightfieldMetrics,
    height_range: (f32, f32),
    diagnostic: GpuFallbackDiagnostic,
) -> GpuEvalResult {
    GpuEvalResult {
        width: metrics.width,
        height: metrics.height,
        world_size: (metrics.world_size_x, metrics.world_size_z),
        height_range,
        fully_gpu: false,
        freshness: GpuPreviewFreshness::Current,
        cpu: None,
        resume_cpu_from: Some(0),
        cpu_fallback: Some(diagnostic),
        did_eval: false,
        output_identity: None,
    }
}

pub(super) fn plan_operation_fallback(error: CompiledDispatchError) -> GpuFallbackReason {
    match error {
        CompiledDispatchError::Gpu(GpuError::RequiresCpu(reason)) => reason,
        CompiledDispatchError::Gpu(error) => GpuFallbackReason::new(
            GpuFallbackCode::RuntimeResourceLimit,
            "GPU execution",
            error.to_string(),
        ),
        CompiledDispatchError::Plan(GpuPlanOperationError::UnsupportedBlend(blend)) => {
            GpuFallbackReason::new(
                GpuFallbackCode::BlendMode,
                "blend",
                format!("{blend:?} is not supported by the GPU"),
            )
        }
        CompiledDispatchError::Plan(GpuPlanOperationError::UnsupportedMaskNodes) => {
            GpuFallbackReason::new(
                GpuFallbackCode::MaskNodes,
                "mask",
                "distribution nodes or auxiliary mask inputs are not GPU-resident",
            )
        }
        CompiledDispatchError::Plan(GpuPlanOperationError::MissingMaskAsset(mask)) => {
            GpuFallbackReason::new(
                GpuFallbackCode::MissingMaskAsset,
                "mask",
                format!("mask asset {mask:?} is missing"),
            )
        }
        CompiledDispatchError::Plan(GpuPlanOperationError::UnsupportedMaskSource(source)) => {
            GpuFallbackReason::new(GpuFallbackCode::MaskSource, "mask", source)
        }
        CompiledDispatchError::Plan(GpuPlanOperationError::MaskBlurRadius(radius)) => {
            GpuFallbackReason::new(
                GpuFallbackCode::MaskOperations,
                "mask",
                format!("blur radius {radius} exceeds the GPU limit"),
            )
        }
        CompiledDispatchError::Plan(error) => GpuFallbackReason::new(
            GpuFallbackCode::InvalidConfiguration,
            "compiled operation",
            error.to_string(),
        ),
    }
}

/// Generators whose height field does not depend on the composed input below them.
pub(super) fn layer_input_independent(kind: &LayerKind) -> bool {
    matches!(
        kind,
        LayerKind::Flat(_)
            | LayerKind::Ramp(_)
            | LayerKind::NoiseValue(_)
            | LayerKind::NoisePerlin(_)
            | LayerKind::Fbm(_)
            | LayerKind::Ridged(_)
            | LayerKind::Mountains(_)
            | LayerKind::Dunes(_)
            | LayerKind::Canyons(_)
            | LayerKind::Mesa(_)
            | LayerKind::Volcano(_)
            | LayerKind::Uplift(_)
            | LayerKind::Island(_)
            | LayerKind::VoronoiRegions(_)
            | LayerKind::ProceduralShape(_)
            | LayerKind::ImportHeightmap(_)
            | LayerKind::Stamp2d(_)
    )
}

/// Whether height alone completely represents the enabled prefix before `resume_index`.
///
/// CPU suffix evaluation also observes auxiliary fields and named layer outputs. Until
/// `GpuEvalResult` carries those checkpoints, any prefix that publishes them must restart
/// on the CPU from layer zero rather than borrowing stale state from another generation.
pub(super) fn cpu_resume_prefix_is_height_only(layers: &[&Layer], resume_index: usize) -> bool {
    layers.iter().take(resume_index).all(|layer| {
        !layer.common.enabled
            || (layer.common.outputs.is_empty()
                && layer
                    .kind
                    .produced_fields()
                    .into_iter()
                    .all(|field| field == FieldId::Height))
    })
}

pub(super) struct BridgePrefix<'a> {
    pub(super) height: &'a Heightfield,
    pub(super) first_dirty_layer: LayerId,
}

pub(super) fn bridge_plan_boundary(
    plan: &CompiledTerrainPlan,
    bridge: &BridgePrefix<'_>,
) -> Option<(PlanOpId, terra_core::terrain_plan::FieldSlot)> {
    plan.operations()
        .iter()
        .enumerate()
        .find_map(|(index, operation)| match operation.kind {
            TerrainOpKind::RunLayerKernel {
                layer,
                input_height,
                ..
            } if layer == bridge.first_dirty_layer => {
                Some((PlanOpId::from_index(index), input_height))
            }
            _ => None,
        })
}

pub(super) fn resample_bridge_prefix(src: &Heightfield, dst: HeightfieldMetrics) -> Vec<f32> {
    let width = dst.width as usize;
    let height = dst.height as usize;
    let mut output = vec![0.0; width.saturating_mul(height)];
    if src.metrics.width == 0 || src.metrics.height == 0 || width == 0 || height == 0 {
        return output;
    }
    let dense = src.to_dense();
    let source_width = src.metrics.width as usize;
    let source_height = src.metrics.height as usize;
    for y in 0..height {
        for x in 0..width {
            let source_x = (((x as f32 + 0.5) / width as f32) * source_width as f32) as usize;
            let source_y = (((y as f32 + 0.5) / height as f32) * source_height as f32) as usize;
            output[y * width + x] = dense
                [source_y.min(source_height - 1) * source_width + source_x.min(source_width - 1)];
        }
    }
    output
}
