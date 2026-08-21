//! GPU evaluator compiled plan implementation.

use super::*;

#[derive(Debug)]
pub(super) enum CompiledDispatchError {
    Plan(GpuPlanOperationError),
    Gpu(GpuError),
}

impl From<GpuPlanOperationError> for CompiledDispatchError {
    fn from(error: GpuPlanOperationError) -> Self {
        Self::Plan(error)
    }
}

impl From<GpuError> for CompiledDispatchError {
    fn from(error: GpuError) -> Self {
        Self::Gpu(error)
    }
}

fn kernel_runs_in_place(kernel: GpuKernel) -> bool {
    matches!(
        kernel,
        GpuKernel::Blur
            | GpuKernel::EffectFilter
            | GpuKernel::Terrace
            | GpuKernel::Thermal
            | GpuKernel::Hydraulic
            | GpuKernel::RiverCarve
            | GpuKernel::StreamPower
            | GpuKernel::MultiScaleAmplify
    )
}

fn plan_scope_region(
    scope: PropagatedDirtyScope,
    metrics: HeightfieldMetrics,
) -> (u32, u32, u32, u32) {
    match scope.scope {
        PlanDirtyScope::FullField => (0, 0, metrics.width, metrics.height),
        PlanDirtyScope::Region(region) => {
            if metrics.width == 0 || metrics.height == 0 {
                return (0, 0, 0, 0);
            }
            let x0 = ((region.min_u.clamp(0.0, 1.0) * metrics.width as f32).floor() as u32)
                .min(metrics.width - 1)
                .saturating_sub(scope.halo_samples);
            let y0 = ((region.min_v.clamp(0.0, 1.0) * metrics.height as f32).floor() as u32)
                .min(metrics.height - 1)
                .saturating_sub(scope.halo_samples);
            let x1 = ((region.max_u.clamp(0.0, 1.0) * metrics.width as f32).ceil() as u32)
                .max(x0 + 1)
                .saturating_add(scope.halo_samples)
                .min(metrics.width);
            let y1 = ((region.max_v.clamp(0.0, 1.0) * metrics.height as f32).ceil() as u32)
                .max(y0 + 1)
                .saturating_add(scope.halo_samples)
                .min(metrics.height);
            (x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0))
        }
    }
}

fn plan_distribution(
    stack: &LayerStack,
    origin: PlanOrigin,
) -> Option<&terra_core::mask::Distribution> {
    match origin {
        PlanOrigin::Authored(NodeRef::Layer(layer)) => {
            stack.find(layer).map(|layer| &layer.common.masks)
        }
        PlanOrigin::Authored(NodeRef::Group(group)) => {
            stack.find_group(group).map(|group| &group.masks)
        }
        _ => None,
    }
}

fn owner_layer_id(owner: Option<NodeRef>) -> Option<LayerId> {
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

fn plan_fallback_result(
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
fn layer_input_independent(kind: &LayerKind) -> bool {
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
                && (matches!(&layer.kind, LayerKind::Path(params) if !params.carve)
                    || layer
                        .kind
                        .produced_fields()
                        .into_iter()
                        .all(|field| field == FieldId::Height)))
    })
}

pub(super) struct BridgePrefix<'a> {
    pub(super) height: &'a Heightfield,
    pub(super) first_dirty_layer: LayerId,
}

fn bridge_plan_boundary(
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

fn resample_bridge_prefix(src: &Heightfield, dst: HeightfieldMetrics) -> Vec<f32> {
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

impl GpuTerrainEngine {
    fn cache_compiled_stamp_mask(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        layer: &Layer,
        region: (u32, u32, u32, u32),
    ) {
        if !matches!(layer.kind, LayerKind::Stamp2d(_)) {
            return;
        }
        let id = layer.id();
        let needs_texture = self.stamp_mask_cache.get(&id).is_none_or(|texture| {
            texture.width != self.metrics.width || texture.height != self.metrics.height
        });
        if needs_texture {
            self.stamp_mask_cache.insert(
                id,
                HeightTex::new(
                    device,
                    "compiled-stamp-mask-cache",
                    self.metrics.width,
                    self.metrics.height,
                ),
            );
        }
        let cached = self
            .stamp_mask_cache
            .get(&id)
            .expect("stamp mask cache just ensured");
        record_copy_views_region(
            device,
            encoder,
            &self.copy,
            &self.stamp_mask.view,
            &cached.view,
            self.metrics.width,
            self.metrics.height,
            region,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn record_compiled_layer_composite(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        resources: &GpuPlanResources,
        authored: &Layer,
        base: terra_core::terrain_plan::FieldSlot,
        layer_candidate: terra_core::terrain_plan::FieldSlot,
        mask: terra_core::terrain_plan::FieldSlot,
        output: terra_core::terrain_plan::FieldSlot,
        region: (u32, u32, u32, u32),
    ) -> Result<(), CompiledDispatchError> {
        if matches!(authored.kind, LayerKind::Stamp2d(_)) {
            // Stamp2d's transform footprint is produced beside the sampled raster
            // and cached with that reusable candidate. Restore it as a second
            // outer-composite mask; raw `layer_tex` alone clamps the raster across
            // the whole field, while shared scratch can belong to another stamp.
            let cached_mask = self.stamp_mask_cache.get(&authored.id()).ok_or_else(|| {
                CompiledDispatchError::Gpu(cpu_required(
                    GpuFallbackCode::RuntimeResourceLimit,
                    "Stamp2d transform mask",
                    "compiled Stamp2d candidate has no matching transform-mask cache",
                ))
            })?;
            record_copy_views_region(
                device,
                encoder,
                &self.copy,
                &cached_mask.view,
                &self.stamp_mask.view,
                self.metrics.width,
                self.metrics.height,
                region,
            );
            for (source, destination) in [
                (
                    resources.view(base).map_err(GpuPlanOperationError::from)?,
                    &self.ping.view,
                ),
                (
                    resources
                        .view(layer_candidate)
                        .map_err(GpuPlanOperationError::from)?,
                    &self.layer_tex.view,
                ),
                (
                    resources.view(mask).map_err(GpuPlanOperationError::from)?,
                    &self.unit_mask.view,
                ),
            ] {
                record_copy_views_region(
                    device,
                    encoder,
                    &self.copy,
                    source,
                    destination,
                    self.metrics.width,
                    self.metrics.height,
                    region,
                );
            }
            self.current = 0;
            self.blend_into_current_with_mask_region(
                device,
                queue,
                encoder,
                authored.common.opacity,
                authored.common.blend,
                [TexSlot::UnitMask, TexSlot::StampMask],
                Some(region),
            )?;
            let source = if self.current == 0 {
                &self.ping.view
            } else {
                &self.pong.view
            };
            record_copy_views_region(
                device,
                encoder,
                &self.copy,
                source,
                resources
                    .view(output)
                    .map_err(GpuPlanOperationError::from)?,
                self.metrics.width,
                self.metrics.height,
                region,
            );
            return Ok(());
        }

        self.plan_operations.composite_group_region(
            device,
            encoder,
            resources,
            base,
            base,
            layer_candidate,
            mask,
            output,
            GpuGroupCompositeParams {
                blend: authored.common.blend,
                opacity: authored.common.opacity,
                mode: GroupCompositeMode::Standard,
            },
            Some(region),
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_compiled_operation(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        stack: &LayerStack,
        mask_assets: &[MaskAsset],
        plan: &CompiledTerrainPlan,
        candidate: &GpuPlanResources,
        operation_id: PlanOpId,
        kernel: Option<GpuLayerPlan>,
        quality: PreviewQuality,
        scope: PropagatedDirtyScope,
        cold: bool,
        invalidation: &PlanInvalidation,
        published_output_slots: &HashMap<
            terra_core::ids::OutputId,
            terra_core::terrain_plan::FieldSlot,
        >,
    ) -> Result<(), CompiledDispatchError> {
        let operation = plan
            .operation(operation_id)
            .expect("selected operation belongs to plan");
        let region = plan_scope_region(scope, self.metrics);
        match &operation.kind {
            TerrainOpKind::Seed { source, output } => {
                self.plan_operations.seed_field_region(
                    device,
                    encoder,
                    candidate,
                    *source,
                    *output,
                    Some(region),
                )?;
            }
            TerrainOpKind::EvaluateMask {
                input_height,
                input_fields: _,
                output_mask,
            } => {
                let result = if let Some(distribution) = plan_distribution(stack, operation.origin)
                {
                    self.plan_operations.evaluate_distribution_resolved_region(
                        device,
                        encoder,
                        candidate,
                        *input_height,
                        *output_mask,
                        distribution,
                        mask_assets,
                        published_output_slots,
                        self.metrics.dx(),
                        self.metrics.dz(),
                        Some(region),
                    )
                } else {
                    Err(GpuPlanOperationError::UnsupportedMaskNodes)
                };
                result?;
            }
            TerrainOpKind::RunLayerKernel {
                layer,
                input_height,
                output_candidate,
                ..
            } => {
                let authored = stack.find(*layer).ok_or_else(|| {
                    CompiledDispatchError::Gpu(cpu_required(
                        GpuFallbackCode::UnsupportedOptions,
                        "terrain plan",
                        "compiled layer owner is missing from the authored document",
                    ))
                })?;
                let kernel = kernel.ok_or_else(|| {
                    CompiledDispatchError::Gpu(cpu_required(
                        GpuFallbackCode::UnsupportedOptions,
                        "GPU capability graph",
                        "layer has no executable GPU kernel choice",
                    ))
                })?;
                record_copy_views_region(
                    device,
                    encoder,
                    &self.copy,
                    candidate
                        .view(*input_height)
                        .map_err(GpuPlanOperationError::from)?,
                    &self.ping.view,
                    self.metrics.width,
                    self.metrics.height,
                    region,
                );
                self.current = 0;
                self.last_dirty_rect = (!scope.is_full()).then_some(region);
                let runs_in_place = kernel_runs_in_place(kernel.kernel);
                if runs_in_place {
                    record_copy_views_region(
                        device,
                        encoder,
                        &self.copy,
                        &self.ping.view,
                        &self.layer_tex.view,
                        self.metrics.width,
                        self.metrics.height,
                        region,
                    );
                }
                if let LayerKind::SculptBase(params) = &authored.kind {
                    let patch_region = (!cold
                        && invalidation.patched_operations.contains(&operation_id)
                        && !scope.is_full())
                    .then_some(region);
                    self.record_sculpt_to_layer(device, encoder, params, patch_region);
                }
                let mut candidate_layer = authored.clone();
                candidate_layer.common.opacity = 1.0;
                candidate_layer.common.blend = BlendMode::Replace;
                candidate_layer.common.masks = Distribution::default();
                self.eval_layer(
                    device,
                    queue,
                    encoder,
                    &candidate_layer,
                    kernel.kernel,
                    quality,
                )?;
                self.cache_compiled_stamp_mask(device, encoder, authored, region);
                let source = if runs_in_place {
                    if self.current == 0 {
                        &self.ping.view
                    } else {
                        &self.pong.view
                    }
                } else {
                    &self.layer_tex.view
                };
                record_copy_views_region(
                    device,
                    encoder,
                    &self.copy,
                    source,
                    candidate
                        .view(*output_candidate)
                        .map_err(GpuPlanOperationError::from)?,
                    self.metrics.width,
                    self.metrics.height,
                    region,
                );
            }
            TerrainOpKind::CompositeLayer {
                layer,
                base,
                candidate: layer_candidate,
                mask,
                output,
            } => {
                let authored = stack.find(*layer).expect("compiled layer owner");
                self.record_compiled_layer_composite(
                    device,
                    queue,
                    encoder,
                    candidate,
                    authored,
                    *base,
                    *layer_candidate,
                    *mask,
                    *output,
                    region,
                )?;
            }
            TerrainOpKind::CompositeGroup {
                group,
                parent,
                private_seed,
                child_output,
                mask,
                output,
                mode,
            } => {
                let authored = stack.find_group(*group).expect("compiled group owner");
                let opacity = if authored.group_kind == terra_core::layer::GroupKind::Biome {
                    authored.opacity * authored.filter_blending
                } else {
                    authored.opacity
                };
                self.plan_operations.composite_group_region(
                    device,
                    encoder,
                    candidate,
                    *parent,
                    *private_seed,
                    *child_output,
                    *mask,
                    *output,
                    GpuGroupCompositeParams {
                        blend: authored.blend,
                        opacity,
                        mode: *mode,
                    },
                    Some(region),
                )?;
            }
            TerrainOpKind::CompositeAuxField {
                group,
                mask,
                composite,
            } => {
                let authored = stack.find_group(*group).expect("compiled group owner");
                let opacity = if authored.group_kind == terra_core::layer::GroupKind::Biome {
                    authored.opacity * authored.filter_blending
                } else {
                    authored.opacity
                };
                self.plan_operations.composite_aux_region(
                    device,
                    encoder,
                    candidate,
                    composite.parent,
                    composite.child,
                    *mask,
                    composite.output,
                    opacity,
                    Some(region),
                )?;
            }
            TerrainOpKind::PublishOutput { .. } => {}
        }
        Ok(())
    }

    /// Execute a validated terrain plan. Operation order, field wiring, dirty
    /// propagation, and provenance come exclusively from `plan`; `stack` is
    /// consulted only to resolve mutable authored payloads by stable id.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_compiled_with_intent(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        stack: &LayerStack,
        mask_assets: &[MaskAsset],
        plan: &CompiledTerrainPlan,
        expected_revision: PlanStructureRevision,
        invalidation: &PlanInvalidation,
        metrics: HeightfieldMetrics,
        quality: PreviewQuality,
        want_cpu: bool,
        intent: GpuEvaluationIntent,
    ) -> Result<GpuEvalResult, GpuError> {
        self.evaluate_compiled_with_bridge(
            device,
            queue,
            stack,
            mask_assets,
            plan,
            expected_revision,
            invalidation,
            metrics,
            quality,
            want_cpu,
            intent,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn evaluate_compiled_with_bridge(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        stack: &LayerStack,
        mask_assets: &[MaskAsset],
        plan: &CompiledTerrainPlan,
        expected_revision: PlanStructureRevision,
        invalidation: &PlanInvalidation,
        metrics: HeightfieldMetrics,
        quality: PreviewQuality,
        want_cpu: bool,
        intent: GpuEvaluationIntent,
        bridge_prefix: Option<BridgePrefix<'_>>,
    ) -> Result<GpuEvalResult, GpuError> {
        profiling::scope!("gpu_compiled_plan_eval");
        if !plan.matches_structure_revision(expected_revision) {
            return Err(GpuError::StalePlan {
                plan_revision: plan.stamp().structure_revision.get(),
                expected_revision: expected_revision.get(),
            });
        }

        self.last_eval_stats = GpuEvalStats::default();
        self.last_eval_stats.resolution = metrics.width.max(metrics.height);
        let resource_prepare_started = std::time::Instant::now();
        self.ensure_size(device, metrics);
        self.uniform_pool.reset();
        self.plan_operations.begin_evaluation();
        self.last_plan_operation_trace.clear();
        let quality_changed = self.last_quality.replace(quality) != Some(quality);
        #[cfg(test)]
        {
            self.executed_plan_operations.clear();
            self.executed_kernels.clear();
        }

        let key = GpuPlanResourceKey::new(metrics.width, metrics.height, self.device_generation);
        let compatible_active = self.plan_resources.current().is_some_and(|resources| {
            resources.key() == key
                && resources.layout().structure_signature() == plan.structure_signature()
        });
        let cold = quality_changed
            || !compatible_active
            || self.active_plan_revision != Some(expected_revision);
        // Cold/structural/resource executions remain transactional candidates.
        // A compatible warm content edit executes against the retained realization
        // after preflight, avoiding per-dab allocation and whole-texture copies.
        let mut staged_candidate = if cold {
            Some(
                self.plan_resources
                    .stage_candidate(device, plan, key)
                    .map_err(|error| GpuError::Wgpu(error.to_string()))?,
            )
        } else {
            None
        };
        self.last_eval_stats.cold_execution = cold;
        self.last_eval_stats.resource_prepare_us =
            resource_prepare_started.elapsed().as_micros() as u64;
        let capability_preflight_started = std::time::Instant::now();

        let bridge_boundary = bridge_prefix
            .as_ref()
            .and_then(|bridge| bridge_plan_boundary(plan, bridge));
        if bridge_prefix.is_some() && bridge_boundary.is_none() {
            return Err(cpu_required(
                GpuFallbackCode::UnsupportedOptions,
                "bridge prefix",
                "the first dirty layer has no executable operation in the compiled terrain plan",
            ));
        }
        let mut requested: Vec<PlanOpId> = if let Some((boundary, _)) = bridge_boundary {
            plan.operations()
                .iter()
                .enumerate()
                .filter_map(|(index, _)| {
                    let id = PlanOpId::from_index(index);
                    (index >= boundary.index() && plan.analysis().operation_is_live(id))
                        .then_some(id)
                })
                .collect()
        } else if cold {
            plan.operations()
                .iter()
                .enumerate()
                .filter_map(|(index, _)| {
                    let id = PlanOpId::from_index(index);
                    plan.analysis().operation_is_live(id).then_some(id)
                })
                .collect()
        } else {
            invalidation
                .operations
                .iter()
                .map(|dirty| dirty.operation)
                .collect()
        };
        let mut reused_plan_candidates = 0u32;
        let mut reused_plan_operations = Vec::new();
        if !cold {
            requested.retain(|operation_id| {
                let Some(operation) = plan.operation(*operation_id) else {
                    return false;
                };
                let TerrainOpKind::RunLayerKernel { layer, .. } = operation.kind else {
                    return true;
                };
                let patched = invalidation.patched_operations.contains(operation_id);
                let input_independent = stack
                    .find(layer)
                    .is_some_and(|layer| layer_input_independent(&layer.kind));
                if !patched && input_independent {
                    reused_plan_candidates += 1;
                    reused_plan_operations.push(*operation_id);
                    false
                } else {
                    true
                }
            });
        }
        self.last_eval_stats.reused_contributions = reused_plan_candidates;
        self.last_eval_stats.operations_reused = reused_plan_candidates;
        let mut selected = staged_candidate
            .as_ref()
            .map(|candidate| candidate.layout())
            .or_else(|| self.plan_resources.current().map(|active| active.layout()))
            .expect("cold candidate or compatible active plan resources")
            .materialization_operations(plan, &requested);
        if let Some((boundary, _)) = bridge_boundary {
            selected.retain(|operation| operation.index() >= boundary.index());
        }
        self.last_eval_stats.operations_materialized =
            selected.len().saturating_sub(requested.len()) as u32;
        let merged_scope = if cold {
            PropagatedDirtyScope::new(PlanDirtyScope::FullField)
        } else {
            invalidation
                .operations
                .iter()
                .map(|dirty| dirty.scope)
                .reduce(PropagatedDirtyScope::merge)
                .unwrap_or_else(|| PropagatedDirtyScope::new(PlanDirtyScope::FullField))
        };
        let scope_for = |operation: PlanOpId| {
            if cold {
                return PropagatedDirtyScope::new(PlanDirtyScope::FullField);
            }
            invalidation
                .operations
                .iter()
                .find_map(|dirty| (dirty.operation == operation).then_some(dirty.scope))
                .unwrap_or(merged_scope)
        };

        let deferred_at = if !cold && intent == GpuEvaluationIntent::InteractiveLocal {
            requested
                .iter()
                .copied()
                .find(|operation| scope_for(*operation).is_full())
        } else {
            None
        };
        if !cold
            && requested.iter().any(|operation| {
                let Some(TerrainOpKind::RunLayerKernel { layer, .. }) =
                    plan.operation(*operation).map(|operation| &operation.kind)
                else {
                    return false;
                };
                stack
                    .flatten_layers()
                    .first()
                    .is_some_and(|first| first.id() == *layer)
                    && !scope_for(*operation).is_full()
            })
        {
            self.last_eval_stats.used_layer_zero_region = true;
        }
        let execution_end = deferred_at.map_or(usize::MAX, PlanOpId::index);
        self.last_eval_stats.operations_deferred = requested
            .iter()
            .filter(|operation| operation.index() >= execution_end)
            .count() as u32;
        let mut selected: Vec<PlanOpId> = selected
            .into_iter()
            .filter(|operation| operation.index() < execution_end)
            .collect();
        if intent == GpuEvaluationIntent::Complete {
            if let Some((revision, resume)) = self.deferred_plan_resume {
                if revision == expected_revision {
                    selected.retain(|operation| operation.index() >= resume.index());
                }
            }
        }

        // Consume the one flat capability compilation as the backend adapter for
        // authored payloads. The terrain plan remains the scheduling authority;
        // this graph supplies only executable kernel choices and rejection detail.
        self.last_graph = compile_gpu_graph(stack, mask_assets);
        let flat_layers = stack.flatten_layers();
        let layer_gpu_decisions: HashMap<
            LayerId,
            (Option<GpuLayerPlan>, Option<GpuFallbackReason>),
        > = flat_layers
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                (
                    layer.id(),
                    (
                        self.last_graph.plans.get(index).copied().flatten(),
                        self.last_graph
                            .fallback_reasons
                            .get(index)
                            .cloned()
                            .flatten(),
                    ),
                )
            })
            .collect();
        let mut kernels = HashMap::<PlanOpId, GpuLayerPlan>::new();
        let mut planned_fallback = None;
        for operation_id in &selected {
            let operation = plan
                .operation(*operation_id)
                .expect("selected operation belongs to plan");
            if let TerrainOpKind::RunLayerKernel {
                layer,
                output_fields,
                ..
            } = &operation.kind
            {
                let Some(_authored) = stack.find(*layer) else {
                    let diagnostic = plan_fallback_diagnostic(
                        plan,
                        stack,
                        *operation_id,
                        GpuFallbackReason::new(
                            GpuFallbackCode::UnsupportedOptions,
                            "terrain plan",
                            "compiled layer owner is missing from the authored document",
                        ),
                    );
                    planned_fallback = Some(diagnostic);
                    break;
                };
                if output_fields.iter().any(|field| {
                    plan.analysis().field_is_live(*field)
                        && plan.analysis().consumers(*field).iter().any(|consumer| {
                            !plan.operation(*consumer).is_some_and(|operation| {
                                matches!(operation.kind, TerrainOpKind::PublishOutput { .. })
                            })
                        })
                }) {
                    let diagnostic = plan_fallback_diagnostic(
                        plan,
                        stack,
                        *operation_id,
                        GpuFallbackReason::new(
                            GpuFallbackCode::AuxiliaryDependency,
                            "auxiliary field",
                            "the GPU kernel does not yet publish a live auxiliary output",
                        ),
                    );
                    planned_fallback = Some(diagnostic);
                    break;
                }
                match layer_gpu_decisions.get(layer) {
                    Some((Some(kernel), _)) => {
                        kernels.insert(*operation_id, *kernel);
                    }
                    decision => {
                        let reason = decision
                            .and_then(|(_, reason)| reason.clone())
                            .unwrap_or_else(|| {
                                GpuFallbackReason::new(
                                    GpuFallbackCode::UnsupportedOptions,
                                    "GPU capability graph",
                                    "layer has no executable GPU kernel choice",
                                )
                            });
                        let diagnostic =
                            plan_fallback_diagnostic(plan, stack, *operation_id, reason);
                        planned_fallback = Some(diagnostic);
                        break;
                    }
                }
            }
        }
        if let Some(diagnostic) = &planned_fallback {
            if stack.requires_tree_evaluation() {
                return Ok(plan_fallback_result(
                    metrics,
                    self.approx_range,
                    diagnostic.clone(),
                ));
            }
            let boundary = diagnostic.operation.map_or(0, PlanOpId::index);
            selected.retain(|operation| operation.index() < boundary);
        }

        let warm_execution = staged_candidate.is_none();
        let candidate = staged_candidate.take().unwrap_or_else(|| {
            self.last_eval_stats.warm_plan_resource_reuses = self
                .last_eval_stats
                .warm_plan_resource_reuses
                .saturating_add(1);
            self.plan_resources
                .take_current()
                .expect("compatible warm realization checked above")
        });
        if let (Some(bridge), Some((_, input_height))) = (bridge_prefix.as_ref(), bridge_boundary) {
            let dense = resample_bridge_prefix(bridge.height, metrics);
            let texture = match candidate.texture(input_height) {
                Ok(texture) => texture,
                Err(error) => {
                    if warm_execution {
                        self.plan_resources.restore_current(candidate);
                    }
                    return Err(GpuError::Wgpu(error.to_string()));
                }
            };
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&dense),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(metrics.width * 4),
                    rows_per_image: Some(metrics.height),
                },
                wgpu::Extent3d {
                    width: metrics.width,
                    height: metrics.height,
                    depth_or_array_layers: 1,
                },
            );
            self.approx_range = bridge.height.min_max();
        }
        self.last_eval_stats.selected_operations = selected.len() as u32;
        self.last_eval_stats.capability_preflight_us =
            capability_preflight_started.elapsed().as_micros() as u64;
        let command_encode_started = std::time::Instant::now();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compiled-terrain-plan"),
        });
        let evaluation_timing_slot = self
            .evaluation_timer
            .as_mut()
            .and_then(|timer| timer.begin(&mut encoder));

        let mut last_height = bridge_boundary.map(|(_, input_height)| input_height);
        let published_output_slots: HashMap<_, _> = plan
            .operations()
            .iter()
            .filter_map(|operation| match operation.kind {
                TerrainOpKind::PublishOutput { output, source } => Some((output, source)),
                _ => None,
            })
            .collect();
        for operation_id in &selected {
            let operation = plan
                .operation(*operation_id)
                .expect("selected operation belongs to plan");
            let region = plan_scope_region(scope_for(*operation_id), metrics);
            let execution = (|| -> Result<(), CompiledDispatchError> {
                match &operation.kind {
                    TerrainOpKind::Seed { source, output } => {
                        self.plan_operations.seed_field_region(
                            device,
                            &mut encoder,
                            &candidate,
                            *source,
                            *output,
                            Some(region),
                        )?;
                        Ok(())
                    }
                    TerrainOpKind::EvaluateMask {
                        input_height,
                        input_fields: _,
                        output_mask,
                    } => {
                        let result = if let Some(distribution) =
                            plan_distribution(stack, operation.origin)
                        {
                            self.plan_operations.evaluate_distribution_resolved_region(
                                device,
                                &mut encoder,
                                &candidate,
                                *input_height,
                                *output_mask,
                                distribution,
                                mask_assets,
                                &published_output_slots,
                                metrics.dx(),
                                metrics.dz(),
                                Some(region),
                            )
                        } else {
                            Err(GpuPlanOperationError::UnsupportedMaskNodes)
                        };
                        result?;
                        Ok(())
                    }
                    TerrainOpKind::RunLayerKernel {
                        layer,
                        input_height,
                        output_candidate,
                        ..
                    } => {
                        let authored = stack.find(*layer).expect("preflight resolved layer");
                        let kernel = kernels
                            .get(operation_id)
                            .expect("preflight compiled layer kernel");
                        record_copy_views_region(
                            device,
                            &mut encoder,
                            &self.copy,
                            candidate
                                .view(*input_height)
                                .map_err(GpuPlanOperationError::from)?,
                            &self.ping.view,
                            metrics.width,
                            metrics.height,
                            // `ping` is also the last-presented height texture. A
                            // warm local execution must preserve its pixels outside
                            // the propagated scope; the kernel only reads the
                            // region (including its planned halo) below.
                            region,
                        );
                        self.current = 0;
                        self.last_dirty_rect =
                            (!scope_for(*operation_id).is_full()).then_some(region);
                        let runs_in_place = kernel_runs_in_place(kernel.kernel);
                        if runs_in_place {
                            record_copy_views_region(
                                device,
                                &mut encoder,
                                &self.copy,
                                &self.ping.view,
                                &self.layer_tex.view,
                                metrics.width,
                                metrics.height,
                                region,
                            );
                        }
                        if let LayerKind::SculptBase(params) = &authored.kind {
                            let patch_region = (!cold
                                && invalidation.patched_operations.contains(operation_id)
                                && !scope_for(*operation_id).is_full())
                            .then_some(region);
                            self.record_sculpt_to_layer(device, &mut encoder, params, patch_region);
                        }
                        // Legacy kernels historically performed the authored outer
                        // composite themselves. A compiled plan has an explicit
                        // `CompositeLayer` operation, so run the adapter in candidate
                        // mode and leave authored opacity/blend/mask to that operation.
                        let mut candidate_layer = authored.clone();
                        candidate_layer.common.opacity = 1.0;
                        candidate_layer.common.blend = BlendMode::Replace;
                        candidate_layer.common.masks = Distribution::default();
                        self.eval_layer(
                            device,
                            queue,
                            &mut encoder,
                            &candidate_layer,
                            kernel.kernel,
                            quality,
                        )?;
                        self.cache_compiled_stamp_mask(device, &mut encoder, authored, region);
                        let source = if runs_in_place {
                            if self.current == 0 {
                                &self.ping.view
                            } else {
                                &self.pong.view
                            }
                        } else {
                            &self.layer_tex.view
                        };
                        record_copy_views_region(
                            device,
                            &mut encoder,
                            &self.copy,
                            source,
                            candidate
                                .view(*output_candidate)
                                .map_err(GpuPlanOperationError::from)?,
                            metrics.width,
                            metrics.height,
                            region,
                        );
                        Ok(())
                    }
                    TerrainOpKind::CompositeLayer {
                        layer,
                        base,
                        candidate: layer_candidate,
                        mask,
                        output,
                    } => {
                        let authored = stack.find(*layer).expect("compiled layer owner");
                        self.record_compiled_layer_composite(
                            device,
                            queue,
                            &mut encoder,
                            &candidate,
                            authored,
                            *base,
                            *layer_candidate,
                            *mask,
                            *output,
                            region,
                        )?;
                        Ok(())
                    }
                    TerrainOpKind::CompositeGroup {
                        group,
                        parent,
                        private_seed,
                        child_output,
                        mask,
                        output,
                        mode,
                    } => {
                        let authored = stack.find_group(*group).expect("compiled group owner");
                        let opacity = if authored.group_kind == terra_core::layer::GroupKind::Biome
                        {
                            authored.opacity * authored.filter_blending
                        } else {
                            authored.opacity
                        };
                        self.plan_operations.composite_group_region(
                            device,
                            &mut encoder,
                            &candidate,
                            *parent,
                            *private_seed,
                            *child_output,
                            *mask,
                            *output,
                            GpuGroupCompositeParams {
                                blend: authored.blend,
                                opacity,
                                mode: *mode,
                            },
                            Some(region),
                        )?;
                        Ok(())
                    }
                    TerrainOpKind::CompositeAuxField {
                        group,
                        mask,
                        composite,
                    } => {
                        let authored = stack.find_group(*group).expect("compiled group owner");
                        let opacity = if authored.group_kind == terra_core::layer::GroupKind::Biome
                        {
                            authored.opacity * authored.filter_blending
                        } else {
                            authored.opacity
                        };
                        self.plan_operations.composite_aux_region(
                            device,
                            &mut encoder,
                            &candidate,
                            composite.parent,
                            composite.child,
                            *mask,
                            composite.output,
                            opacity,
                            Some(region),
                        )?;
                        Ok(())
                    }
                    TerrainOpKind::PublishOutput { .. } => Ok(()),
                }
            })();
            if let Err(error) = execution {
                let diagnostic = plan_fallback_diagnostic(
                    plan,
                    stack,
                    *operation_id,
                    plan_operation_fallback(error),
                );
                if warm_execution {
                    self.plan_resources.restore_current(candidate);
                }
                return Ok(plan_fallback_result(metrics, self.approx_range, diagnostic));
            }
            if matches!(operation.kind, TerrainOpKind::PublishOutput { .. }) {
                self.last_eval_stats.operations_published =
                    self.last_eval_stats.operations_published.saturating_add(1);
            } else {
                self.last_eval_stats.operations_dispatched =
                    self.last_eval_stats.operations_dispatched.saturating_add(1);
                self.last_eval_stats.plan_workgroups =
                    self.last_eval_stats.plan_workgroups.saturating_add(
                        u64::from(region.2.div_ceil(8)) * u64::from(region.3.div_ceil(8)),
                    );
            }
            for field in plan.analysis().outputs(*operation_id) {
                if plan.field(*field).is_some_and(|field| {
                    matches!(
                        field.kind,
                        terra_core::terrain_plan::LogicalFieldKind::Height
                    )
                }) {
                    last_height = Some(*field);
                }
            }
            #[cfg(test)]
            self.executed_plan_operations.push(*operation_id);
        }

        let freshness = deferred_at.map_or(GpuPreviewFreshness::Current, |operation| {
            let owner = plan.provenance().owner_of(operation);
            let from_layer = owner_layer_id(owner).unwrap_or_default();
            let from_index = flat_layers
                .iter()
                .position(|layer| layer.id() == from_layer)
                .unwrap_or(0);
            GpuPreviewFreshness::Deferred {
                from_index,
                from_layer,
                deferred_layers: flat_layers.len().saturating_sub(from_index),
            }
        });
        let presentation_field = if deferred_at.is_some() || planned_fallback.is_some() {
            last_height.unwrap_or(plan.final_height())
        } else {
            plan.final_height()
        };
        let present_scope = if cold {
            PropagatedDirtyScope::new(PlanDirtyScope::FullField)
        } else {
            merged_scope
        };
        let present_region = plan_scope_region(present_scope, metrics);
        let presentation_view = match candidate.view(presentation_field) {
            Ok(view) => view,
            Err(error) => {
                if warm_execution {
                    self.plan_resources.restore_current(candidate);
                }
                return Err(GpuError::Wgpu(error.to_string()));
            }
        };
        let presentation_binding = candidate
            .layout()
            .binding(presentation_field)
            .expect("presented field has a physical binding");
        let source_resource_incarnation = candidate.incarnation();
        let trace_context = self.pending_evaluation_trace.unwrap_or_default();
        let expected_base = self.last_output_identity.map(|identity| identity.output);
        record_copy_views_region(
            device,
            &mut encoder,
            &self.copy,
            presentation_view,
            &self.ping.view,
            metrics.width,
            metrics.height,
            present_region,
        );
        let (mask_scratch_texture_allocations, mask_scratch_reuses) =
            self.plan_operations.mask_scratch_stats();
        self.last_eval_stats.mask_scratch_texture_allocations = mask_scratch_texture_allocations;
        self.last_eval_stats.mask_scratch_reuses = mask_scratch_reuses;
        self.last_eval_stats.dirty_texels =
            u64::from(present_region.2).saturating_mul(u64::from(present_region.3));
        if let (Some(timer), Some(slot), Some(context)) = (
            self.evaluation_timer.as_mut(),
            evaluation_timing_slot,
            self.pending_evaluation_trace.take(),
        ) {
            timer.finish(&mut encoder, slot, context);
        }
        self.last_eval_stats.command_encode_us =
            command_encode_started.elapsed().as_micros() as u64;
        let queue_submit_started = std::time::Instant::now();
        queue.submit(Some(encoder.finish()));
        let submission_serial = self.allocate_submission_serial();
        self.last_eval_stats.queue_submit_us = queue_submit_started.elapsed().as_micros() as u64;
        self.current = 0;
        self.last_dirty_rect = None;
        if warm_execution {
            self.plan_resources.restore_current(candidate);
        } else {
            self.plan_resources.commit_candidate(candidate);
        }
        self.active_plan_revision = Some(expected_revision);
        self.deferred_plan_resume = deferred_at.map(|operation| (expected_revision, operation));
        if present_scope.is_full() {
            self.mark_all_tiles_dirty();
        } else {
            self.mark_tiles_overlapping_rect(present_region);
        }

        let fallback_resume = planned_fallback.as_ref().map(|diagnostic| {
            flat_layers
                .iter()
                .position(|layer| layer.id() == diagnostic.layer_id)
                .unwrap_or(0)
        });
        let fallback_resume = match fallback_resume {
            Some(index) if !cpu_resume_prefix_is_height_only(&flat_layers, index) => Some(0),
            other => other,
        };
        let cpu = if want_cpu && fallback_resume == Some(0) {
            Some(Heightfield::zeros(metrics))
        } else if want_cpu {
            self.last_eval_stats.readback_bytes = u64::from(metrics.width)
                .saturating_mul(u64::from(metrics.height))
                .saturating_mul(4);
            Some(self.readback_current(device, queue)?)
        } else {
            None
        };
        let live_operations = plan
            .operations()
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                plan.analysis()
                    .operation_is_live(PlanOpId::from_index(*index))
            })
            .count() as u32;
        self.last_eval_stats.operations_skipped = live_operations.saturating_sub(
            self.last_eval_stats
                .operations_dispatched
                .saturating_add(self.last_eval_stats.operations_published)
                .saturating_add(self.last_eval_stats.operations_deferred)
                .saturating_add(self.last_eval_stats.operations_reused),
        );
        let full_scope = PropagatedDirtyScope::new(PlanDirtyScope::FullField);
        self.last_plan_operation_trace = plan
            .operations()
            .iter()
            .enumerate()
            .filter_map(|(index, _)| {
                let operation = PlanOpId::from_index(index);
                if !plan.analysis().operation_is_live(operation) {
                    return None;
                }
                let dirty = invalidation
                    .operations
                    .iter()
                    .find(|dirty| dirty.operation == operation);
                let incoming_scope = if cold {
                    full_scope
                } else {
                    dirty.map_or(merged_scope, |dirty| dirty.incoming_scope)
                };
                let output_scope = if cold {
                    full_scope
                } else {
                    dirty.map_or(merged_scope, |dirty| dirty.scope)
                };
                let disposition = if selected.contains(&operation)
                    && matches!(
                        plan.operations()[index].kind,
                        TerrainOpKind::PublishOutput { .. }
                    ) {
                    GpuPlanOperationDisposition::Published
                } else if selected.contains(&operation) {
                    GpuPlanOperationDisposition::Dispatched
                } else if operation.index() >= execution_end && requested.contains(&operation) {
                    GpuPlanOperationDisposition::Deferred
                } else if reused_plan_operations.contains(&operation) {
                    GpuPlanOperationDisposition::Reused
                } else {
                    GpuPlanOperationDisposition::Skipped
                };
                Some(GpuPlanOperationTrace {
                    operation,
                    owner: plan.provenance().owner_of(operation),
                    incoming_scope,
                    output_scope,
                    incoming_region: plan_scope_region(incoming_scope, metrics),
                    output_region: plan_scope_region(output_scope, metrics),
                    disposition,
                })
            })
            .collect();
        let has_deferred_suffix = deferred_at.is_some();
        let has_hybrid_prefix = planned_fallback.is_some();
        let output_identity = GpuTerrainOutputIdentity {
            output: self.allocate_output_id(),
            frame_id: trace_context.frame_id,
            generation: trace_context.generation,
            evaluation_id: trace_context.evaluation_id,
            plan_revision: expected_revision.get(),
            requested_quality: quality,
            actual_quality: quality,
            intent,
            selected_field: terra_gpu::output_identity::GpuSelectedFieldIdentity {
                selected: presentation_field,
                expected_final: plan.final_height(),
                resource_incarnation: source_resource_incarnation,
                physical_allocation: presentation_binding.physical.index(),
            },
            output_resource: terra_gpu::output_identity::GpuOutputResourceIdentity {
                device_generation: self.device_generation,
                incarnation: self.output_resource_incarnation,
                slot: self.output_slot(),
            },
            extent: (metrics.width, metrics.height),
            coverage: if present_scope.is_full() {
                terra_gpu::output_identity::GpuOutputCoverage::WholeField
            } else {
                terra_gpu::output_identity::GpuOutputCoverage::Patch {
                    rect: terra_core::tiling::SampleRect {
                        x: present_region.0,
                        y: present_region.1,
                        w: present_region.2,
                        h: present_region.3,
                    },
                    expected_base,
                }
            },
            completeness: if has_deferred_suffix {
                terra_gpu::output_identity::GpuOutputCompleteness::DeferredSuffix
            } else if has_hybrid_prefix {
                terra_gpu::output_identity::GpuOutputCompleteness::HybridPrefix
            } else {
                terra_gpu::output_identity::GpuOutputCompleteness::Complete
            },
            invalidation: if cold {
                terra_gpu::output_identity::GpuInvalidationKind::Cold
            } else if present_scope.is_full() {
                terra_gpu::output_identity::GpuInvalidationKind::FullField
            } else {
                terra_gpu::output_identity::GpuInvalidationKind::Regional
            },
            last_write: terra_gpu::output_identity::GpuLastWriteIdentity {
                serial: submission_serial,
                completion: terra_gpu::output_identity::GpuSubmissionCompletion::Submitted,
            },
        };
        self.last_output_identity = Some(output_identity);
        Ok(GpuEvalResult {
            width: metrics.width,
            height: metrics.height,
            world_size: (metrics.world_size_x, metrics.world_size_z),
            height_range: self.approx_range,
            fully_gpu: deferred_at.is_none() && planned_fallback.is_none(),
            freshness,
            cpu,
            resume_cpu_from: fallback_resume,
            cpu_fallback: planned_fallback,
            did_eval: !selected.is_empty(),
            output_identity: Some(output_identity),
        })
    }
}
