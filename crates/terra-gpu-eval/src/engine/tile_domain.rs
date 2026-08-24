use super::*;
use std::sync::mpsc;
use std::sync::Arc;

use terra_core::terrain_plan::{
    resolve_infinite_plan_domain, resolve_plan_execution_strategy, FieldSlot,
    TerrainPlanCheckpoint, TerrainPlanDomainRejection, TerrainPlanDomainSlice,
    TerrainPlanExecutionStrategy,
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

/// CPU-owned packed height page produced asynchronously from one completed tile
/// evaluation. The layout matches the atlas page contract: fixed square extent,
/// interior at `(halo, halo)`, clamped world-edge halo, deterministic zero fill.
#[derive(Debug)]
pub struct GpuPackedHeightTile {
    pub key: terra_core::TerrainTileKey,
    pub content: TerrainContentStamp,
    pub page_extent: u32,
    pub interior_width: u32,
    pub interior_height: u32,
    pub halo: u32,
    pub samples: Vec<f32>,
}

/// Non-blocking mapped transfer which retains its evaluator until the producer
/// observes completion and recycles it.
pub struct GpuPackedTileReadback {
    evaluation: Option<GpuTileEvaluationJob>,
    buffer: wgpu::Buffer,
    padded_bytes_per_row: u32,
    page_extent: u32,
    receiver: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
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

    /// Copy this completed tile evaluator into an asynchronously mapped buffer.
    /// Packing is finalized when [`GpuCompiledTileProducer::poll_packed_readback`]
    /// observes the map callback.
    pub fn begin_packed_readback(
        self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        page_extent: u32,
    ) -> Result<GpuPackedTileReadback, GpuError> {
        if !self.complete {
            return Err(GpuError::Wgpu(
                "packed tile readback requested before evaluation completion".into(),
            ));
        }
        let width = self.domain.evaluation.width;
        let height = self.domain.evaluation.height;
        let unpadded = width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = unpadded.div_ceil(align) * align;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-packed-terrain-tile-readback"),
            size: u64::from(padded_bytes_per_row) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu-packed-terrain-tile-readback-copy"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: self
                    .engine
                    .as_ref()
                    .expect("job owns engine")
                    .output_texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));
        let (sender, receiver) = mpsc::channel();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        Ok(GpuPackedTileReadback {
            evaluation: Some(self),
            buffer,
            padded_bytes_per_row,
            page_extent,
            receiver: Some(receiver),
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

    pub fn analyze_infinite(
        stack: &LayerStack,
        mask_assets: &[MaskAsset],
        plan: &CompiledTerrainPlan,
    ) -> Result<TerrainPlanDomainSlice, GpuTileEvaluationError> {
        let slice = resolve_infinite_plan_domain(stack, mask_assets, plan, plan.final_height())
            .map_err(GpuTileEvaluationError::Domain)?;
        preflight_tile_operations(stack, plan, &slice, &slice.operations)?;
        Ok(slice)
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
        let (slice, checkpoint_descriptor) = if domain.is_infinite() {
            (Self::analyze_infinite(stack, mask_assets, plan)?, None)
        } else {
            match resolve_plan_execution_strategy(plan, plan.final_height()) {
                TerrainPlanExecutionStrategy::Local(slice) => {
                    preflight_tile_operations(stack, plan, &slice, &slice.operations)?;
                    (slice, None)
                }
                TerrainPlanExecutionStrategy::Checkpointed { checkpoint, suffix } => {
                    let mut complete = checkpoint.prefix_operations.clone();
                    complete.extend(suffix.operations.iter().copied());
                    preflight_tile_operations(stack, plan, &suffix, &complete)?;
                    (suffix, Some(checkpoint))
                }
            }
        };
        if slice.operation_halo != domain.operation_halo {
            return Err(GpuTileEvaluationError::HaloMismatch {
                resolved: slice.operation_halo,
                supplied: domain.operation_halo,
            });
        }
        let checkpoint_seed = match checkpoint_descriptor.as_ref() {
            None => None,
            Some(checkpoint) => {
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
        engine.tile_sample_window = Some(if domain.is_infinite() {
            let transform = domain.spatial.interior.transform;
            let spacing = transform.spacing();
            let origin = transform.origin();
            let samples = domain.spatial.evaluation.origin;
            TileSampleWindow::Infinite {
                world_origin_x: origin.x_m() + samples.x as f64 * spacing.x_m(),
                world_origin_z: origin.z_m() + samples.z as f64 * spacing.z_m(),
            }
        } else {
            TileSampleWindow::Bounded {
                origin_x: domain.evaluation.origin_x,
                origin_z: domain.evaluation.origin_z,
                level_width: domain.world.level_width,
                level_height: domain.world.level_height,
            }
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

    /// Poll a mapped tile transfer and finalize the atlas-compatible page on the
    /// CPU. The mapping is compact and export-only; viewport publication remains
    /// GPU-to-GPU through `GpuTileAtlas`.
    pub fn poll_packed_readback(
        &mut self,
        device: &wgpu::Device,
        readback: &mut GpuPackedTileReadback,
    ) -> Result<Option<GpuPackedHeightTile>, GpuError> {
        let Some(receiver) = readback.receiver.as_ref() else {
            return Ok(None);
        };
        let _ = device.poll(wgpu::Maintain::Poll);
        match receiver.try_recv() {
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => {
                readback.receiver = None;
                if let Some(job) = readback.evaluation.take() {
                    self.recycle(job);
                }
                Err(GpuError::Wgpu(
                    "packed tile readback callback disconnected".into(),
                ))
            }
            Ok(Err(error)) => {
                readback.receiver = None;
                if let Some(job) = readback.evaluation.take() {
                    self.recycle(job);
                }
                Err(GpuError::Wgpu(error.to_string()))
            }
            Ok(Ok(())) => {
                readback.receiver = None;
                let job = readback
                    .evaluation
                    .take()
                    .expect("mapped readback retains evaluation");
                let domain = job.domain.clone();
                let mapped = readback.buffer.slice(..).get_mapped_range();
                let row_floats = (readback.padded_bytes_per_row / 4) as usize;
                let mapped_floats: &[f32] = bytemuck::cast_slice(&mapped);
                let mut local = Vec::with_capacity(
                    domain.evaluation.width as usize * domain.evaluation.height as usize,
                );
                for z in 0..domain.evaluation.height as usize {
                    let start = z * row_floats;
                    local.extend_from_slice(
                        &mapped_floats[start..start + domain.evaluation.width as usize],
                    );
                }
                drop(mapped);
                readback.buffer.unmap();
                let minimum_extent =
                    domain.interior.width.max(domain.interior.height) + domain.publication_halo * 2;
                if readback.page_extent < minimum_extent {
                    self.recycle(job);
                    return Err(GpuError::Wgpu(format!(
                        "packed page extent {} is smaller than required {}",
                        readback.page_extent, minimum_extent
                    )));
                }
                let mut samples =
                    vec![0.0; readback.page_extent as usize * readback.page_extent as usize];
                let valid_width = domain.interior.width + domain.publication_halo * 2;
                let valid_height = domain.interior.height + domain.publication_halo * 2;
                for pz in 0..valid_height {
                    for px in 0..valid_width {
                        let (local_x, local_z) = if domain.is_infinite() {
                            (domain.operation_halo + px, domain.operation_halo + pz)
                        } else {
                            let global_x = (i64::from(domain.interior.origin_x) + i64::from(px)
                                - i64::from(domain.publication_halo))
                            .clamp(0, i64::from(domain.world.level_width) - 1)
                                as u32;
                            let global_z = (i64::from(domain.interior.origin_z) + i64::from(pz)
                                - i64::from(domain.publication_halo))
                            .clamp(0, i64::from(domain.world.level_height) - 1)
                                as u32;
                            (
                                global_x - domain.evaluation.origin_x,
                                global_z - domain.evaluation.origin_z,
                            )
                        };
                        samples[(pz * readback.page_extent + px) as usize] =
                            local[(local_z * domain.evaluation.width + local_x) as usize];
                    }
                }
                let result = GpuPackedHeightTile {
                    key: domain.key.clone(),
                    content: domain.content,
                    page_extent: readback.page_extent,
                    interior_width: domain.interior.width,
                    interior_height: domain.interior.height,
                    halo: domain.publication_halo,
                    samples,
                };
                self.recycle(job);
                Ok(Some(result))
            }
        }
    }

    pub fn cancel_packed_readback(&mut self, mut readback: GpuPackedTileReadback) {
        self.stats.cancelled = self.stats.cancelled.saturating_add(1);
        if let Some(job) = readback.evaluation.take() {
            // The copy precedes all later submissions on the same queue, so the
            // evaluator can be reused without changing the bytes being copied.
            self.recycle(job);
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
        let domain_coordinate_safe = match &authored.kind {
            LayerKind::Flat(_)
            | LayerKind::Blur(_)
            | LayerKind::SculptBase(_)
            | LayerKind::NoiseValue(_)
            | LayerKind::NoisePerlin(_)
            | LayerKind::Fbm(_)
            | LayerKind::Ridged(_) => true,
            // Empty shape history is emitted by Infinite project templates. With
            // no enabled stroke the shader only forwards height and clears its
            // auxiliary outputs, so it has no bounded-coordinate dependency.
            LayerKind::SculptStrokes(params) => params.strokes.iter().all(|stroke| !stroke.enabled),
            _ => false,
        };
        if !domain_coordinate_safe {
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
