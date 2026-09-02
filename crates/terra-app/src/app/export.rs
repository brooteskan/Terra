//! Programmatic height-pyramid package exporter. The editor dialog uses scalar field exports.

use std::path::PathBuf;

use terra_core::document::TerrainDocument;
use terra_core::layer::LayerStack;
use terra_core::mask::MaskAsset;
use terra_core::quality::PreviewQuality;
use terra_core::terrain_plan::{
    compile_terrain_plan, propagate_plan_edits, CompiledTerrainPlan, PlanInvalidation,
    PlanStructureRevision, TerrainEditClass, TerrainPlanStamp,
};
use terra_core::{
    PyramidConfig, TerrainContentStamp, TerrainDemandClass, TerrainEvaluationDomain,
    TerrainPyramid, TerrainTileWorkBudget, TerrainTileWorkKey, TerrainTileWorkLease,
    TerrainTileWorkRequest, TerrainTileWorkScheduler, TerrainTileWorkSource,
};
use terra_gpu_eval::{GpuCompiledTileProducer, GpuPackedTileReadback, GpuTileEvaluationJob};
use terra_io::{HeightPyramidPackageBuilder, HeightPyramidPackageResult};

struct EvaluationWork {
    lease: TerrainTileWorkLease,
    job: GpuTileEvaluationJob,
}

struct ReadbackWork {
    lease: TerrainTileWorkLease,
    readback: GpuPackedTileReadback,
}

/// Shared preflight for the output list and the actual export action. This only
/// inspects authored data: it neither evaluates terrain nor writes any files.
fn prepare_height_pyramid_export(
    document: &TerrainDocument,
    generation: u64,
) -> Result<(CompiledTerrainPlan, TerrainPyramid, u32), String> {
    document
        .metrics
        .at_resolution(document.export_resolution)
        .map_err(|error| error.to_string())?;
    let revision = PlanStructureRevision::new(generation);
    let plan = compile_terrain_plan(
        &document.stack,
        &document.masks,
        TerrainPlanStamp::new(revision),
    )
    .map_err(|error| format!("compiled export plan failed: {error:?}"))?;
    let slice = GpuCompiledTileProducer::analyze(&document.stack, &plan)
        .map_err(|error| format!("streaming export is unsupported: {error:?}"))?;
    let mut config = PyramidConfig::new(
        document.export_resolution,
        document.metrics.world_size_x,
        document.metrics.world_size_z,
    );
    config.tile_size = document.metrics.tile_size;
    config.halo = document.metrics.halo;
    Ok((plan, TerrainPyramid::new(config), slice.operation_halo))
}

#[cfg(test)]
fn height_pyramid_export_preview(document: &TerrainDocument) -> Result<TerrainPyramid, String> {
    prepare_height_pyramid_export(document, 1).map(|(_, pyramid, _)| pyramid)
}

/// App-owned export state. It changes only the demand source and publication
/// consumer; plan execution is the same `GpuCompiledTileProducer` path used by
/// camera-driven terrain work.
#[derive(Default)]
pub struct HeightPyramidExportController {
    stack: Option<LayerStack>,
    masks: Vec<MaskAsset>,
    plan: Option<CompiledTerrainPlan>,
    invalidation: PlanInvalidation,
    pyramid: Option<TerrainPyramid>,
    stamp: TerrainContentStamp,
    scheduler: TerrainTileWorkScheduler,
    producer: GpuCompiledTileProducer,
    evaluation: Option<EvaluationWork>,
    readback: Option<ReadbackWork>,
    builder: Option<HeightPyramidPackageBuilder>,
    current_level: u8,
    total_tiles: usize,
    completed_tiles: usize,
    cancel_requested: bool,
    busy: bool,
    result: Option<Result<HeightPyramidPackageResult, String>>,
}

impl HeightPyramidExportController {
    pub fn start(
        &mut self,
        mut document: TerrainDocument,
        root: PathBuf,
        generation: u64,
    ) -> Result<(), String> {
        if self.busy {
            return Err("height-pyramid export is already running".into());
        }
        document.sync_all_biome_paint_masks();
        let (plan, pyramid, operation_halo) = prepare_height_pyramid_export(&document, generation)?;
        let stack = document.preview_eval_stack();
        let revision = PlanStructureRevision::new(generation);
        let invalidation = propagate_plan_edits(&plan, &[TerrainEditClass::Structure]);
        let total_tiles = pyramid.metadata_len() as usize;
        let finest_tiles = pyramid
            .level_metrics(pyramid.max_level())
            .map_or(1, |metrics| metrics.tile_count() as usize);
        let builder = HeightPyramidPackageBuilder::new(root, pyramid.clone())
            .map_err(|error| error.to_string())?;
        let stamp = TerrainContentStamp {
            document_revision: generation,
            plan_revision: revision.get(),
            output_revision: generation,
            content_revision: generation,
        };
        self.stack = Some(stack);
        self.masks = document.masks;
        self.plan = Some(plan);
        self.invalidation = invalidation;
        self.pyramid = Some(pyramid);
        self.stamp = stamp;
        self.scheduler = TerrainTileWorkScheduler::new(finest_tiles.max(1));
        self.producer = GpuCompiledTileProducer::new();
        self.evaluation = None;
        self.readback = None;
        self.builder = Some(builder);
        self.current_level = 0;
        self.total_tiles = total_tiles;
        self.completed_tiles = 0;
        self.cancel_requested = false;
        self.busy = true;
        self.result = None;
        self.queue_level(0, operation_halo);
        Ok(())
    }

    pub fn is_busy(&self) -> bool {
        self.busy
    }

    pub fn progress(&self) -> f32 {
        self.completed_tiles as f32 / self.total_tiles.max(1) as f32
    }

    pub fn cancel(&mut self) {
        if self.busy {
            self.cancel_requested = true;
        }
    }

    pub fn take_result(&mut self) -> Option<Result<HeightPyramidPackageResult, String>> {
        self.result.take()
    }

    /// Advance at most one externally visible boundary per frame.
    pub fn pump(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
        if !self.busy {
            return false;
        }
        if self.cancel_requested {
            self.cancel_now();
            return false;
        }

        if let Some(mut work) = self.readback.take() {
            match self
                .producer
                .poll_packed_readback(device, &mut work.readback)
            {
                Ok(Some(tile)) => {
                    let write = self
                        .builder
                        .as_mut()
                        .expect("busy export owns package builder")
                        .write_tile(&tile.key, &tile.samples);
                    if let Err(error) = write {
                        self.fail(error.to_string());
                        return false;
                    }
                    self.scheduler.complete(work.lease, self.stamp);
                    self.completed_tiles += 1;
                    self.advance_level_or_finish();
                }
                Ok(None) => self.readback = Some(work),
                Err(error) => {
                    self.scheduler.fail(work.lease);
                    self.fail(error.to_string());
                }
            }
            return self.busy;
        }

        if let Some(mut work) = self.evaluation.take() {
            if !self.producer.poll(device, &mut work.job) {
                self.evaluation = Some(work);
                return true;
            }
            let page_extent = self
                .pyramid
                .as_ref()
                .expect("busy export owns pyramid")
                .config
                .tile_size
                + self
                    .pyramid
                    .as_ref()
                    .expect("busy export owns pyramid")
                    .config
                    .halo
                    * 2;
            match work.job.begin_packed_readback(device, queue, page_extent) {
                Ok(readback) => {
                    self.readback = Some(ReadbackWork {
                        lease: work.lease,
                        readback,
                    });
                }
                Err(error) => {
                    self.scheduler.fail(work.lease);
                    self.fail(error.to_string());
                }
            }
            return self.busy;
        }

        let Some(lease) = self
            .scheduler
            .dequeue_budgeted(TerrainTileWorkBudget::new(u64::MAX, 1, 1))
            .into_iter()
            .next()
        else {
            self.advance_level_or_finish();
            return self.busy;
        };
        let key = lease.request.key.tile.clone();
        let slice = match GpuCompiledTileProducer::analyze(
            self.stack.as_ref().expect("busy export owns stack"),
            self.plan.as_ref().expect("busy export owns plan"),
        ) {
            Ok(slice) => slice,
            Err(error) => {
                self.scheduler.fail(lease);
                self.fail(format!("streaming export preflight changed: {error:?}"));
                return false;
            }
        };
        let domain = match TerrainEvaluationDomain::for_tile(
            self.pyramid.as_ref().expect("busy export owns pyramid"),
            key,
            self.pyramid
                .as_ref()
                .expect("busy export owns pyramid")
                .config
                .halo,
            slice.operation_halo,
            self.stamp,
        ) {
            Ok(domain) => domain,
            Err(error) => {
                self.scheduler.fail(lease);
                self.fail(format!("invalid export tile domain: {error:?}"));
                return false;
            }
        };
        match self.producer.begin(
            device,
            queue,
            self.stack.as_ref().expect("busy export owns stack"),
            &self.masks,
            self.plan.as_ref().expect("busy export owns plan"),
            &self.invalidation,
            PreviewQuality::Export,
            domain,
        ) {
            Ok(job) => self.evaluation = Some(EvaluationWork { lease, job }),
            Err(error) => {
                self.scheduler.fail(lease);
                self.fail(format!("GPU height tile production failed: {error:?}"));
            }
        }
        self.busy
    }

    fn queue_level(&mut self, level: u8, operation_halo: u32) {
        let requests = self
            .pyramid
            .as_ref()
            .expect("busy export owns pyramid")
            .height_tiles_at_level(level)
            .expect("valid export level")
            .map(|tile| TerrainTileWorkRequest {
                key: TerrainTileWorkKey {
                    tile,
                    plan_revision: self.stamp.plan_revision,
                    output_revision: self.stamp.output_revision,
                },
                content: self.stamp,
                source: TerrainTileWorkSource::GpuCompiledPlan,
                class: TerrainDemandClass::CoarseCoverage,
                visible: true,
                projected_error_px: f32::MAX,
                distance_m: 0.0,
                estimated_us: u64::from(operation_halo.max(1)) * 250,
            })
            .collect::<Vec<_>>();
        self.scheduler.reconcile(self.stamp, requests);
    }

    fn advance_level_or_finish(&mut self) {
        if !self.busy
            || !self.scheduler.is_empty()
            || self.evaluation.is_some()
            || self.readback.is_some()
        {
            return;
        }
        let max_level = self
            .pyramid
            .as_ref()
            .expect("busy export owns pyramid")
            .max_level();
        if self.current_level < max_level {
            self.current_level += 1;
            let operation_halo = GpuCompiledTileProducer::analyze(
                self.stack.as_ref().expect("busy export owns stack"),
                self.plan.as_ref().expect("busy export owns plan"),
            )
            .map(|slice| slice.operation_halo)
            .unwrap_or(0);
            self.queue_level(self.current_level, operation_halo);
            return;
        }
        let builder = self.builder.take().expect("busy export owns builder");
        match builder.finish() {
            Ok(result) => {
                self.busy = false;
                self.result = Some(Ok(result));
            }
            Err(error) => self.fail(error.to_string()),
        }
    }

    fn cancel_now(&mut self) {
        if let Some(work) = self.evaluation.take() {
            self.producer.cancel(work.job);
        }
        if let Some(work) = self.readback.take() {
            self.producer.cancel_packed_readback(work.readback);
        }
        self.scheduler.clear();
        self.builder = None;
        self.busy = false;
        self.cancel_requested = false;
        self.result = None;
    }

    fn fail(&mut self, message: String) {
        if let Some(work) = self.evaluation.take() {
            self.producer.cancel(work.job);
        }
        if let Some(work) = self.readback.take() {
            self.producer.cancel_packed_readback(work.readback);
        }
        self.scheduler.clear();
        self.builder = None;
        self.busy = false;
        self.result = Some(Err(message));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use terra_core::layer::{FlatParams, Layer, LayerKind, LayerStack};
    use terra_io::HeightPyramidPackage;

    fn flat_document() -> TerrainDocument {
        let mut document = TerrainDocument::new_default();
        document.export_resolution = 17;
        document.stack = LayerStack::new();
        document.stack.push(Layer::new(
            "Export flat",
            LayerKind::Flat(FlatParams { height: 37.5 }),
        ));
        document
    }

    #[test]
    fn export_preview_tracks_resolution_and_tile_size_not_preview_resolution() {
        let mut document = flat_document();
        document.metrics.tile_size = 8;
        let preview = height_pyramid_export_preview(&document).unwrap();
        let files = terra_io::height_pyramid_output_paths(&preview).collect::<Vec<_>>();
        // Levels 2, 3, 5, 9, 17 have 1 + 1 + 1 + 4 + 9 tiles, plus two metadata files.
        assert_eq!(files.len(), 18);
        assert_eq!(
            files.last().unwrap(),
            "packages/<content-id>/height/l04/000002_000002.<hash>.r32"
        );

        document.preview_resolution = 8192;
        assert_eq!(
            terra_io::height_pyramid_output_paths(
                &height_pyramid_export_preview(&document).unwrap()
            )
            .collect::<Vec<_>>(),
            files
        );
        document.export_resolution = 8;
        assert_eq!(
            height_pyramid_export_preview(&document)
                .unwrap()
                .metadata_len(),
            3
        );
        document.export_resolution = 17;
        document.metrics.tile_size = 16;
        assert_eq!(
            height_pyramid_export_preview(&document)
                .unwrap()
                .metadata_len(),
            8
        );
    }

    #[test]
    fn export_preview_rejects_invalid_metrics_and_broken_active_mask_references() {
        let mut document = flat_document();
        document.metrics.world_size_x = 0.0;
        assert!(height_pyramid_export_preview(&document).is_err());

        let mut document = flat_document();
        let mut layer = Layer::new("Masked height", LayerKind::Flat(Default::default()));
        layer.common.masks.push(terra_core::mask::MaskRef::new(
            terra_core::mask::MaskId::new(),
        ));
        let id = layer.id();
        document.stack.push(layer);
        assert!(height_pyramid_export_preview(&document).is_err());
        document.stack.find_mut(id).unwrap().common.enabled = false;
        assert!(height_pyramid_export_preview(&document).is_ok());
    }

    #[test]
    fn gpu_controller_exports_complete_deterministic_package() {
        let gpu = terra_test_gpu::headless_required();
        let root =
            std::env::temp_dir().join(format!("terra-app-height-pyramid-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let mut controller = HeightPyramidExportController::default();
        controller.start(flat_document(), root.clone(), 71).unwrap();
        let start = std::time::Instant::now();
        while controller.is_busy() {
            controller.pump(&gpu.device, &gpu.queue);
            let _ = gpu.device.poll(wgpu::Maintain::Wait);
            assert!(
                start.elapsed() < std::time::Duration::from_secs(120),
                "GPU export did not complete"
            );
        }
        let result = controller.take_result().expect("finished result").unwrap();
        let package = HeightPyramidPackage::open(&root).unwrap();
        assert_eq!(result.tile_count, package.manifest.tiles.len());
        let descriptor = package.descriptor().unwrap();
        assert_eq!(result.tile_count, descriptor.metadata_len() as usize);
        assert!(package
            .reconstruct_region(descriptor.max_level(), 5, 5, 4, 4)
            .unwrap()
            .iter()
            .all(|height| (*height - 37.5).abs() <= 1.0e-5));
        assert_eq!(
            controller.producer.stats().submitted,
            result.tile_count as u64
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn cancellation_drops_staging_without_publication() {
        let gpu = terra_test_gpu::headless_required();
        let root = std::env::temp_dir().join(format!(
            "terra-app-height-pyramid-cancel-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let mut controller = HeightPyramidExportController::default();
        controller.start(flat_document(), root.clone(), 73).unwrap();
        controller.pump(&gpu.device, &gpu.queue);
        controller.cancel();
        controller.pump(&gpu.device, &gpu.queue);
        assert!(!controller.is_busy());
        assert!(controller.take_result().is_none());
        assert!(!root.join("height-pyramid.current").exists());
        let _ = std::fs::remove_dir_all(root);
    }
}
