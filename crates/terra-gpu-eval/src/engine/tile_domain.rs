use super::*;
use std::sync::mpsc;
use std::sync::Arc;

use terra_core::terrain_plan::{
    resolve_plan_execution_strategy, FieldSlot, TerrainPlanCheckpoint, TerrainPlanDomainRejection,
    TerrainPlanDomainSlice, TerrainPlanExecutionStrategy,
};
use terra_core::{TerrainContentStamp, TerrainEvaluationDomain};

pub(super) struct GpuCheckpointField {
    pub field: FieldSlot,
    pub texture: Arc<wgpu::Texture>,
}

/// Immutable GPU fields produced by one complete-field plan prefix.
pub(super) struct GpuTerrainCheckpoint {
    content: TerrainContentStamp,
    width: u32,
    height: u32,
    boundary: usize,
    quality: PreviewQuality,
    pub fields: Vec<GpuCheckpointField>,
}

#[derive(Clone, Copy)]
pub(super) struct GpuCheckpointSeed<'a> {
    pub checkpoint: &'a GpuTerrainCheckpoint,
    pub origin_x: u32,
    pub origin_z: u32,
}

#[derive(Clone, Copy)]
pub(super) struct CompiledPlanExecutionOverride<'a> {
    pub operations: &'a [PlanOpId],
    pub checkpoint_seed: Option<GpuCheckpointSeed<'a>>,
}

#[derive(Debug)]
pub enum GpuTileEvaluationError {
    Domain(TerrainPlanDomainRejection),
    StalePlan {
        plan: u64,
        requested: u64,
    },
    HaloMismatch {
        resolved: u32,
        supplied: u32,
    },
    UnsupportedOperation {
        operation: PlanOpId,
        owner: Option<NodeRef>,
        detail: String,
    },
    Engine(GpuError),
}

impl From<GpuError> for GpuTileEvaluationError {
    fn from(error: GpuError) -> Self {
        Self::Engine(error)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuTileProducerStats {
    pub engine_allocations: u64,
    pub engine_reuses: u64,
    pub submitted: u64,
    pub completed: u64,
    pub cancelled: u64,
    pub evaluated_texels: u64,
    pub checkpoint_builds: u64,
    pub checkpoint_reuses: u64,
}

/// One isolated tile evaluation. Its texture is never visible through the
/// complete-field engine and cannot be published until `is_complete` is true.
pub struct GpuTileEvaluationJob {
    engine: Option<GpuTerrainEngine>,
    domain: TerrainEvaluationDomain,
    slice: TerrainPlanDomainSlice,
    completion: mpsc::Receiver<()>,
    complete: bool,
}

impl GpuTileEvaluationJob {
    pub fn domain(&self) -> &TerrainEvaluationDomain {
        &self.domain
    }

    pub fn slice(&self) -> &TerrainPlanDomainSlice {
        &self.slice
    }

    pub fn is_complete(&self) -> bool {
        self.complete
    }

    pub fn output_texture_view(&self) -> Option<&wgpu::TextureView> {
        self.complete.then(|| {
            self.engine
                .as_ref()
                .expect("job owns engine")
                .output_texture_view()
        })
    }

    #[doc(hidden)]
    pub fn readback_height(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Heightfield, GpuError> {
        if !self.complete {
            return Err(GpuError::Wgpu(
                "tile evaluation readback requested before completion".into(),
            ));
        }
        self.engine
            .as_mut()
            .expect("job owns engine")
            .readback_current(device, queue)
    }
}

/// Recycles tile-sized evaluators independently of the complete-field engine.
#[derive(Default)]
pub struct GpuCompiledTileProducer {
    idle: Vec<GpuTerrainEngine>,
    retired: Vec<GpuTileEvaluationJob>,
    stats: GpuTileProducerStats,
    checkpoint_engine: Option<GpuTerrainEngine>,
    checkpoint: Option<GpuTerrainCheckpoint>,
}

impl GpuCompiledTileProducer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stats(&self) -> GpuTileProducerStats {
        self.stats
    }

    pub fn analyze(
        stack: &LayerStack,
        plan: &CompiledTerrainPlan,
    ) -> Result<TerrainPlanDomainSlice, GpuTileEvaluationError> {
        match resolve_plan_execution_strategy(plan, plan.final_height()) {
            TerrainPlanExecutionStrategy::Local(slice) => {
                preflight_tile_operations(stack, plan, &slice, &slice.operations)?;
                Ok(slice)
            }
            TerrainPlanExecutionStrategy::Checkpointed { checkpoint, suffix } => {
                let mut complete = checkpoint.prefix_operations.clone();
                complete.extend(suffix.operations.iter().copied());
                preflight_tile_operations(stack, plan, &suffix, &complete)?;
                Ok(suffix)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn begin(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        stack: &LayerStack,
        mask_assets: &[MaskAsset],
        plan: &CompiledTerrainPlan,
        invalidation: &PlanInvalidation,
        quality: PreviewQuality,
        domain: TerrainEvaluationDomain,
    ) -> Result<GpuTileEvaluationJob, GpuTileEvaluationError> {
        self.reclaim_retired(device);
        let revision = plan.stamp().structure_revision;
        if revision.get() != domain.content.plan_revision {
            return Err(GpuTileEvaluationError::StalePlan {
                plan: revision.get(),
                requested: domain.content.plan_revision,
            });
        }
        let strategy = resolve_plan_execution_strategy(plan, plan.final_height());
        let slice = Self::analyze(stack, plan)?;
        if slice.operation_halo != domain.operation_halo {
            return Err(GpuTileEvaluationError::HaloMismatch {
                resolved: slice.operation_halo,
                supplied: domain.operation_halo,
            });
        }
        let checkpoint_seed = match &strategy {
            TerrainPlanExecutionStrategy::Local(_) => None,
            TerrainPlanExecutionStrategy::Checkpointed { checkpoint, .. } => {
                self.ensure_checkpoint(
                    device,
                    queue,
                    stack,
                    mask_assets,
                    plan,
                    invalidation,
                    quality,
                    &domain,
                    checkpoint,
                )?;
                let ready = self.checkpoint.as_ref().expect("checkpoint just ensured");
                Some(GpuCheckpointSeed {
                    checkpoint: ready,
                    origin_x: domain.evaluation.origin_x,
                    origin_z: domain.evaluation.origin_z,
                })
            }
        };

        let mut engine = if let Some(engine) = self.idle.pop() {
            self.stats.engine_reuses = self.stats.engine_reuses.saturating_add(1);
            engine
        } else {
            self.stats.engine_allocations = self.stats.engine_allocations.saturating_add(1);
            GpuTerrainEngine::new(
                device,
                domain.evaluation.width.max(domain.evaluation.height),
            )
        };
        engine.tile_sample_window = Some(TileSampleWindow {
            origin_x: domain.evaluation.origin_x,
            origin_z: domain.evaluation.origin_z,
            level_width: domain.world.level_width,
            level_height: domain.world.level_height,
        });
        engine.mark_all_dirty(stack);
        // A recycled evaluator retains tile-sized plan textures. Domain origin
        // is part of generator/sculpt content even when authored parameters did
        // not change, so candidate-producing kernels must be patched while the
        // allocation itself remains reusable.
        let mut tile_invalidation = invalidation.clone();
        tile_invalidation
            .patched_operations
            .extend(slice.operations.iter().copied().filter(|operation| {
                plan.operation(*operation).is_some_and(|operation| {
                    matches!(operation.kind, TerrainOpKind::RunLayerKernel { .. })
                })
            }));
        tile_invalidation
            .patched_operations
            .sort_by_key(|operation| operation.index());
        tile_invalidation.patched_operations.dedup();
        let result = engine.evaluate_compiled_with_bridge(
            device,
            queue,
            stack,
            mask_assets,
            plan,
            revision,
            &tile_invalidation,
            domain.local_metrics(),
            quality,
            false,
            GpuEvaluationIntent::Complete,
            None,
            Some(CompiledPlanExecutionOverride {
                operations: &slice.operations,
                checkpoint_seed,
            }),
        )?;
        if !result.fully_gpu || (!result.did_eval && checkpoint_seed.is_none()) {
            return Err(GpuTileEvaluationError::UnsupportedOperation {
                operation: slice
                    .operations
                    .last()
                    .copied()
                    .unwrap_or(PlanOpId::from_index(0)),
                owner: slice
                    .operations
                    .last()
                    .and_then(|operation| plan.provenance().owner_of(*operation)),
                detail: "tile plan did not produce a complete GPU result".into(),
            });
        }
        let (sender, receiver) = mpsc::channel();
        queue.on_submitted_work_done(move || {
            let _ = sender.send(());
        });
        self.stats.submitted = self.stats.submitted.saturating_add(1);
        self.stats.evaluated_texels = self.stats.evaluated_texels.saturating_add(
            u64::from(domain.evaluation.width) * u64::from(domain.evaluation.height),
        );
        Ok(GpuTileEvaluationJob {
            engine: Some(engine),
            domain,
            slice,
            completion: receiver,
            complete: false,
        })
    }

    pub fn poll(&mut self, device: &wgpu::Device, job: &mut GpuTileEvaluationJob) -> bool {
        self.reclaim_retired(device);
        if job.complete {
            return true;
        }
        let _ = device.poll(wgpu::Maintain::Poll);
        match job.completion.try_recv() {
            Ok(()) | Err(mpsc::TryRecvError::Disconnected) => {
                job.complete = true;
                self.stats.completed = self.stats.completed.saturating_add(1);
                true
            }
            Err(mpsc::TryRecvError::Empty) => false,
        }
    }

    pub fn recycle(&mut self, mut job: GpuTileEvaluationJob) {
        if let Some(mut engine) = job.engine.take() {
            engine.tile_sample_window = None;
            self.idle.push(engine);
        }
    }

    pub fn cancel(&mut self, job: GpuTileEvaluationJob) {
        self.stats.cancelled = self.stats.cancelled.saturating_add(1);
        if job.complete {
            self.recycle(job);
        } else {
            self.retired.push(job);
        }
    }

    fn reclaim_retired(&mut self, device: &wgpu::Device) {
        if self.retired.is_empty() {
            return;
        }
        let _ = device.poll(wgpu::Maintain::Poll);
        let mut waiting = Vec::new();
        for mut job in std::mem::take(&mut self.retired) {
            match job.completion.try_recv() {
                Ok(()) | Err(mpsc::TryRecvError::Disconnected) => {
                    job.complete = true;
                    self.recycle(job);
                }
                Err(mpsc::TryRecvError::Empty) => waiting.push(job),
            }
        }
        self.retired = waiting;
    }

    #[allow(clippy::too_many_arguments)]
    fn ensure_checkpoint(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        stack: &LayerStack,
        mask_assets: &[MaskAsset],
        plan: &CompiledTerrainPlan,
        invalidation: &PlanInvalidation,
        quality: PreviewQuality,
        domain: &TerrainEvaluationDomain,
        descriptor: &TerrainPlanCheckpoint,
    ) -> Result<(), GpuTileEvaluationError> {
        let reusable = self.checkpoint.as_ref().is_some_and(|checkpoint| {
            checkpoint.content == domain.content
                && checkpoint.width == domain.world.level_width
                && checkpoint.height == domain.world.level_height
                && checkpoint.boundary == descriptor.boundary
                && checkpoint.quality == quality
        });
        if reusable {
            self.stats.checkpoint_reuses = self.stats.checkpoint_reuses.saturating_add(1);
            return Ok(());
        }

        // Drop the old immutable publication before building a new revision.
        // Queue submission order keeps its textures alive for already-submitted
        // tile jobs, so cancellation can never splice two revisions together.
        self.checkpoint = None;
        let mut engine = self.checkpoint_engine.take().unwrap_or_else(|| {
            GpuTerrainEngine::new(
                device,
                domain.world.level_width.max(domain.world.level_height),
            )
        });
        engine.tile_sample_window = None;
        engine.mark_all_dirty(stack);
        let metrics = HeightfieldMetrics {
            width: domain.world.level_width,
            height: domain.world.level_height,
            world_size_x: domain.world.world_size_x,
            world_size_z: domain.world.world_size_z,
            tile_size: domain
                .world
                .level_width
                .max(domain.world.level_height)
                .max(1),
            halo: 0,
        };
        let revision = plan.stamp().structure_revision;
        let result = engine.evaluate_compiled_with_bridge(
            device,
            queue,
            stack,
            mask_assets,
            plan,
            revision,
            invalidation,
            metrics,
            quality,
            false,
            GpuEvaluationIntent::Complete,
            None,
            Some(CompiledPlanExecutionOverride {
                operations: &descriptor.prefix_operations,
                checkpoint_seed: None,
            }),
        )?;
        if !result.fully_gpu || !result.did_eval {
            self.checkpoint_engine = Some(engine);
            return Err(GpuTileEvaluationError::UnsupportedOperation {
                operation: descriptor
                    .blockers
                    .last()
                    .map(|blocker| blocker.operation)
                    .unwrap_or(PlanOpId::from_index(0)),
                owner: descriptor.blockers.last().and_then(|blocker| blocker.owner),
                detail: result
                    .cpu_fallback
                    .map(|fallback| fallback.reason.detail)
                    .unwrap_or_else(|| "complete-field checkpoint prefix was not produced".into()),
            });
        }

        let resources = engine
            .plan_resources
            .current()
            .expect("successful compiled execution retains plan resources");
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("terrain-plan-checkpoint-snapshot"),
        });
        let mut fields = Vec::with_capacity(descriptor.frontier_fields.len());
        for field in &descriptor.frontier_fields {
            let texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("terrain-plan-checkpoint-field"),
                size: wgpu::Extent3d {
                    width: metrics.width,
                    height: metrics.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            }));
            encoder.copy_texture_to_texture(
                resources
                    .texture(*field)
                    .map_err(|error| {
                        GpuTileEvaluationError::Engine(GpuError::Wgpu(error.to_string()))
                    })?
                    .as_image_copy(),
                texture.as_image_copy(),
                wgpu::Extent3d {
                    width: metrics.width,
                    height: metrics.height,
                    depth_or_array_layers: 1,
                },
            );
            fields.push(GpuCheckpointField {
                field: *field,
                texture,
            });
        }
        queue.submit(Some(encoder.finish()));
        self.checkpoint = Some(GpuTerrainCheckpoint {
            content: domain.content,
            width: metrics.width,
            height: metrics.height,
            boundary: descriptor.boundary,
            quality,
            fields,
        });
        self.checkpoint_engine = Some(engine);
        self.stats.checkpoint_builds = self.stats.checkpoint_builds.saturating_add(1);
        Ok(())
    }
}

fn preflight_tile_operations(
    stack: &LayerStack,
    plan: &CompiledTerrainPlan,
    slice: &TerrainPlanDomainSlice,
    complete_strategy_operations: &[PlanOpId],
) -> Result<(), GpuTileEvaluationError> {
    let selected: std::collections::HashSet<_> = slice.operations.iter().copied().collect();
    let complete: std::collections::HashSet<_> =
        complete_strategy_operations.iter().copied().collect();
    for (index, operation) in plan.operations().iter().enumerate() {
        let id = PlanOpId::from_index(index);
        if !plan.analysis().operation_is_live(id) {
            continue;
        }
        if !complete.contains(&id) && !matches!(operation.kind, TerrainOpKind::PublishOutput { .. })
        {
            return Err(GpuTileEvaluationError::UnsupportedOperation {
                operation: id,
                owner: plan.provenance().owner_of(id),
                detail: "live named-output work is outside the requested height slice".into(),
            });
        }
        if !selected.contains(&id) {
            continue;
        }
        let TerrainOpKind::RunLayerKernel { layer, .. } = operation.kind else {
            continue;
        };
        let authored =
            stack
                .find(layer)
                .ok_or_else(|| GpuTileEvaluationError::UnsupportedOperation {
                    operation: id,
                    owner: plan.provenance().owner_of(id),
                    detail: "authored layer is missing".into(),
                })?;
        if !matches!(
            authored.kind,
            LayerKind::Flat(_)
                | LayerKind::Blur(_)
                | LayerKind::SculptBase(_)
                | LayerKind::NoiseValue(_)
                | LayerKind::NoisePerlin(_)
        ) {
            return Err(GpuTileEvaluationError::UnsupportedOperation {
                operation: id,
                owner: plan.provenance().owner_of(id),
                detail: format!(
                    "{} is not yet domain-coordinate-safe for compiled tile execution",
                    authored.kind.type_id()
                ),
            });
        }
    }
    Ok(())
}
