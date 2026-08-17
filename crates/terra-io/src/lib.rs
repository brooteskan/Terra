//! Import/export and background build jobs.

mod export;
mod geotiff;
mod import;

pub use export::{
    build_tile_manifest, changed_tiles, export_package, ExportRequest, ExportResult, TileManifest,
    TileManifestEntry,
};
pub use geotiff::{read_geotiff_heights, GeoTiffInfo};
pub use import::{import_heightmap_png, import_heightmap_raw};

use std::path::PathBuf;
use terra_core::document::TerrainDocument;
use terra_core::eval::{EvalContext, PreviewQuality, StackEvaluator};
use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
use terra_jobs::{spawn_one_shot, CancelToken, JobCtx, JobError, JobHandle, Pending, Pollable};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum IoError {
    #[error("{0}")]
    Msg(String),
    #[error(transparent)]
    Image(#[from] image::ImageError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub struct BuildJob {
    pub progress: f32,
    pub done: bool,
    pub result: Option<Result<ExportResult, String>>,
}

/// Non-blocking export worker.
pub struct BackgroundExporter {
    handle: Option<JobHandle<Result<ExportResult, String>>>,
    pub job: BuildJob,
}

impl BackgroundExporter {
    pub fn new() -> Self {
        Self {
            handle: None,
            job: BuildJob {
                progress: 0.0,
                done: true,
                result: None,
            },
        }
    }

    pub fn start(&mut self, doc: TerrainDocument, out_dir: PathBuf) {
        self.start_job(move |ctx| {
            ctx.set_progress(0.1);
            let mut doc = doc;
            // Ensure sparse biome paint is baked into mask assets for this export.
            doc.sync_all_biome_paint_masks();
            ctx.set_progress(0.3);
            // Thread the job's cancel token into eval so a cancelled export stops
            // between layers and inside fill-based generators (#101).
            match evaluate_document_for_export(&doc, ctx.token().clone()) {
                Ok((hf, eval_ctx)) => {
                    ctx.set_progress(0.7);
                    let req = ExportRequest {
                        out_dir,
                        ..ExportRequest::default()
                    };
                    export_package(&hf, &eval_ctx, &req).map_err(|e| e.to_string())
                }
                Err(e) => Err(e.to_string()),
            }
        });
    }

    /// Spawn `f` as the export job, superseding any job already in flight.
    ///
    /// Private seam: `start` is the only public entry, but tests drive this with
    /// arbitrary bodies (a panicking one, a spin-until-cancelled one).
    fn start_job<F>(&mut self, f: F)
    where
        F: FnOnce(&JobCtx) -> Result<ExportResult, String> + Send + 'static,
    {
        // A superseded job should stop burning CPU rather than run on detached.
        if let Some(handle) = self.handle.take() {
            handle.cancel();
        }
        self.job = BuildJob {
            progress: 0.0,
            done: false,
            result: None,
        };
        self.handle = Some(spawn_one_shot("terra-export", f));
    }

    pub fn poll(&mut self) {
        let (progress, outcome) = match self.handle.as_ref() {
            Some(handle) => (handle.progress(), handle.try_take()),
            None => return,
        };
        self.job.progress = progress;
        let Some(outcome) = outcome else {
            return;
        };
        self.handle = None;
        match outcome {
            Ok(result) => {
                self.job.progress = 1.0;
                self.job.done = true;
                self.job.result = Some(result);
            }
            Err(JobError::Panicked(message)) => {
                // A panicking export used to leave `done == false` forever (silent
                // stuck progress bar); surface it as a failed job instead.
                self.job.progress = 1.0;
                self.job.done = true;
                self.job.result = Some(Err(format!("panicked: {message}")));
            }
            Err(JobError::Cancelled) => {
                // A cancelled export returns to idle: no Done result, no stuck
                // busy/done state.
                self.job.progress = 0.0;
                self.job.done = true;
                self.job.result = None;
            }
        }
    }

    /// Request cancellation of the in-flight export, if any. The next `poll`
    /// observes the job resolve to cancelled and returns the exporter to idle.
    pub fn cancel(&self) {
        if let Some(handle) = self.handle.as_ref() {
            handle.cancel();
        }
    }
}

/// Evaluate through the same v8 single-stack authority used by the viewport.
///
/// Keeping this as a small, testable boundary prevents compatibility containers
/// (`TerrainDocument::world`) from silently becoming an export source again.
fn evaluate_document_for_export(
    doc: &TerrainDocument,
    cancel: CancelToken,
) -> Result<(Heightfield, EvalContext), terra_core::eval::EvalError> {
    let metrics = HeightfieldMetrics {
        width: doc.export_resolution,
        height: doc.export_resolution,
        world_size_x: doc.metrics.world_size_x,
        world_size_z: doc.metrics.world_size_z,
        tile_size: doc.metrics.tile_size.min(doc.export_resolution),
        halo: doc.metrics.halo,
    };
    let mut evaluator = StackEvaluator::new();
    // Export runs a full rebuild from scratch and never reloads its own baked
    // checkpoints; spilling export-resolution bakes into the shared cache dir (and
    // stomping preview bakes) was pure waste. Run memory-only (B1-D8).
    evaluator.cache.disable_disk();
    let mut ctx = EvalContext::new(metrics);
    // Cancellation from the export job (or `never` on the synchronous test path).
    ctx.set_cancel_token(cancel);
    ctx.quality = PreviewQuality::Export;
    ctx.level_steps = doc.level_steps.clone();
    ctx.mask_assets = doc.masks.clone();
    let reference = Heightfield::zeros(metrics);
    ctx.masks = terra_core::mask::bake_mask_assets(
        &doc.masks,
        &reference,
        metrics,
        &std::collections::HashMap::new(),
    );
    let height = evaluator.rebuild_all(&doc.stack, &mut ctx)?;
    Ok((height, ctx))
}

impl Default for BackgroundExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl Pollable for BackgroundExporter {
    /// Poll the in-flight export, then report whether one is still running. An
    /// export streams to disk with a progress bar the surrounding frame already
    /// repaints at loop cadence, so it wants `redraw` while busy but not the
    /// ~16 ms `animate` wakes. The completion frame (the "one more frame to show
    /// status" case) is driven app-side off `job.done`, not from here.
    fn pump(&mut self) -> Pending {
        self.poll();
        let busy = !self.job.done;
        Pending {
            busy,
            animate: false,
            redraw: busy,
        }
    }
}

pub fn save_project(doc: &TerrainDocument, path: &std::path::Path) -> Result<(), IoError> {
    let json = doc.to_json()?;
    std::fs::write(path, json)?;
    Ok(())
}

pub fn load_project(path: &std::path::Path) -> Result<TerrainDocument, IoError> {
    let s = std::fs::read_to_string(path)?;
    Ok(TerrainDocument::from_json(&s)?)
}

/// Result of a finished background project I/O job.
#[derive(Debug)]
pub enum ProjectIoResult {
    Saved {
        path: PathBuf,
    },
    Loaded {
        path: PathBuf,
        // Boxed so lighter variants aren't sized to the large `TerrainDocument`
        // payload (clippy::large_enum_variant).
        doc: Box<TerrainDocument>,
    },
    Failed {
        path: PathBuf,
        error: String,
    },
}

/// Non-blocking project save/load worker (serialize/parse/fs off the UI thread).
pub struct BackgroundProjectIo {
    handle: Option<JobHandle<ProjectIoResult>>,
    // Remembered at spawn so a panicked/cancelled job (which yields no
    // `ProjectIoResult`) can still report which path failed.
    pending_path: Option<PathBuf>,
    busy: bool,
    status: Option<&'static str>,
    pub result: Option<ProjectIoResult>,
}

impl BackgroundProjectIo {
    pub fn new() -> Self {
        Self {
            handle: None,
            pending_path: None,
            busy: false,
            status: None,
            result: None,
        }
    }

    pub fn is_busy(&self) -> bool {
        self.busy
    }

    pub fn status(&self) -> Option<&'static str> {
        self.status
    }

    /// Clone-owned document is moved to the worker for sync + compact JSON + write.
    pub fn start_save(&mut self, doc: TerrainDocument, path: PathBuf) {
        self.pending_path = Some(path.clone());
        self.busy = true;
        self.status = Some("Saving…");
        self.result = None;
        self.handle = Some(spawn_one_shot("terra-project-save", move |_ctx| {
            let json = match doc.into_json() {
                Ok(json) => json,
                Err(e) => {
                    return ProjectIoResult::Failed {
                        path,
                        error: e.to_string(),
                    }
                }
            };
            match std::fs::write(&path, json) {
                Ok(()) => ProjectIoResult::Saved { path },
                Err(e) => ProjectIoResult::Failed {
                    path,
                    error: e.to_string(),
                },
            }
        }));
    }

    pub fn start_load(&mut self, path: PathBuf) {
        self.pending_path = Some(path.clone());
        self.busy = true;
        self.status = Some("Loading…");
        self.result = None;
        self.handle = Some(spawn_one_shot("terra-project-load", move |_ctx| {
            let contents = match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(e) => {
                    return ProjectIoResult::Failed {
                        path,
                        error: e.to_string(),
                    }
                }
            };
            match TerrainDocument::from_json(&contents) {
                Ok(doc) => ProjectIoResult::Loaded {
                    path,
                    doc: Box::new(doc),
                },
                Err(e) => ProjectIoResult::Failed {
                    path,
                    error: e.to_string(),
                },
            }
        }));
    }

    pub fn poll(&mut self) {
        let outcome = match self.handle.as_ref() {
            Some(handle) => handle.try_take(),
            None => return,
        };
        let Some(outcome) = outcome else {
            return;
        };
        self.handle = None;
        self.busy = false;
        self.status = None;
        let path = self.pending_path.take().unwrap_or_default();
        self.result = Some(match outcome {
            Ok(result) => result,
            Err(JobError::Panicked(message)) => ProjectIoResult::Failed {
                path,
                error: format!("job panicked: {message}"),
            },
            // No cancel surface is exposed for project IO; map defensively so an
            // impossible state surfaces loudly rather than vanishing.
            Err(JobError::Cancelled) => ProjectIoResult::Failed {
                path,
                error: "job cancelled".to_string(),
            },
        });
    }

    /// Spawn `f` as the project-IO job. Test-only seam so a panicking body can be
    /// exercised; production uses `start_save` / `start_load`.
    #[cfg(test)]
    fn start_job_for_test<F>(&mut self, path: PathBuf, f: F)
    where
        F: FnOnce(&JobCtx) -> ProjectIoResult + Send + 'static,
    {
        self.pending_path = Some(path);
        self.busy = true;
        self.status = Some("Working…");
        self.result = None;
        self.handle = Some(spawn_one_shot("terra-project-test", f));
    }
}

impl Default for BackgroundProjectIo {
    fn default() -> Self {
        Self::new()
    }
}

impl Pollable for BackgroundProjectIo {
    /// Poll the in-flight save/load, then report whether one is still running.
    /// The typed result (a `ProjectIoResult`) and the transient status string are
    /// drained by the app after the tick — this only reports busy/repaint facts.
    /// Save/load shows a static "Saving…"/"Loading…" status, so it wants `redraw`
    /// while busy but not `animate` wakes.
    fn pump(&mut self) -> Pending {
        self.poll();
        let busy = self.is_busy();
        Pending {
            busy,
            animate: false,
            redraw: busy,
        }
    }
}

#[cfg(test)]
mod worker_tests {
    use super::*;
    use terra_core::layer::{FlatParams, Layer, LayerKind, LayerStack};

    /// Spin (yielding) until `cond` holds, with a generous safety timeout so a
    /// regression fails loudly instead of hanging the suite.
    fn wait_until(mut cond: impl FnMut() -> bool) {
        let start = std::time::Instant::now();
        while !cond() {
            assert!(
                start.elapsed() < std::time::Duration::from_secs(5),
                "background job did not reach the expected state within the timeout"
            );
            std::thread::yield_now();
        }
    }

    fn flat_doc() -> TerrainDocument {
        let mut doc = TerrainDocument::new_default();
        doc.export_resolution = 32;
        doc.stack = LayerStack::new();
        doc.stack.push(Layer::new(
            "Export Height",
            LayerKind::Flat(FlatParams { height: 73.0 }),
        ));
        doc
    }

    #[test]
    fn v8_export_evaluates_the_authoritative_stack() {
        let doc = flat_doc();
        assert!(!doc.stack.flatten_layers().is_empty());

        let (height, _) = evaluate_document_for_export(&doc, CancelToken::never())
            .expect("the stack should evaluate for export");
        assert!((height.get(16, 16) - 73.0).abs() < 1.0e-4);
    }

    #[test]
    fn export_eval_honors_a_pre_cancelled_token() {
        // Proves the job's token is actually threaded into EvalContext: a
        // pre-cancelled token trips the between-layer check immediately. If the
        // `set_cancel_token` wiring is ever dropped, this fails without relying on
        // any timing.
        let doc = flat_doc();
        let (token, flag) = CancelToken::flag();
        flag.cancel();
        let result = evaluate_document_for_export(&doc, token);
        assert!(
            matches!(result, Err(terra_core::eval::EvalError::Cancelled)),
            "a cancelled token must abort export eval"
        );
    }

    #[test]
    fn panicking_export_surfaces_as_failed_not_hung() {
        // Regression: a panicking export thread used to leave job.done == false
        // forever (silent stuck progress bar).
        let mut exporter = BackgroundExporter::new();
        exporter.start_job(|_| panic!("boom in export"));
        wait_until(|| {
            exporter.poll();
            exporter.job.done && exporter.job.result.is_some()
        });
        match exporter.job.result.take() {
            Some(Err(message)) => assert!(
                message.contains("boom in export"),
                "the panic message should surface, got: {message}"
            ),
            Some(Ok(_)) => panic!("expected a failed export, got a successful result"),
            None => panic!("expected a failed export result, got none"),
        }
    }

    #[test]
    fn cancelled_export_finishes_with_no_result() {
        let mut exporter = BackgroundExporter::new();
        // Body spins until cancelled, then returns a value the cancel must discard.
        exporter.start_job(|ctx| {
            while !ctx.token().is_cancelled() {
                std::thread::yield_now();
            }
            Err("value produced after cancel — must be discarded".to_string())
        });
        exporter.cancel();
        wait_until(|| {
            exporter.poll();
            exporter.job.done
        });
        assert!(
            exporter.job.result.is_none(),
            "a cancelled export must leave no Done result"
        );
    }

    #[test]
    fn panicking_project_io_surfaces_as_failed() {
        let mut io = BackgroundProjectIo::new();
        io.start_job_for_test(PathBuf::from("project.terra"), |_| panic!("io boom"));
        wait_until(|| {
            io.poll();
            !io.is_busy() && io.result.is_some()
        });
        match io.result.take() {
            Some(ProjectIoResult::Failed { path, error }) => {
                assert_eq!(path, PathBuf::from("project.terra"));
                assert!(error.contains("io boom"), "got: {error}");
            }
            other => panic!("expected a Failed result, got {other:?}"),
        }
    }

    #[test]
    fn save_then_load_round_trips_through_the_worker() {
        let path = std::env::temp_dir().join(format!("terra-io-test-{}.terra", std::process::id()));

        let doc = TerrainDocument::new_default();
        let mut io = BackgroundProjectIo::new();
        io.start_save(doc, path.clone());
        wait_until(|| {
            io.poll();
            !io.is_busy() && io.result.is_some()
        });
        assert!(
            matches!(io.result.take(), Some(ProjectIoResult::Saved { .. })),
            "save should complete"
        );

        io.start_load(path.clone());
        wait_until(|| {
            io.poll();
            !io.is_busy() && io.result.is_some()
        });
        assert!(
            matches!(io.result.take(), Some(ProjectIoResult::Loaded { .. })),
            "load should complete"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn exporter_pump_reports_busy_while_running_then_idle() {
        use std::sync::mpsc;
        use terra_jobs::Pollable;

        let mut exporter = BackgroundExporter::new();
        // Idle before any job: no busy, no repaint, never animation-cadence.
        let idle = exporter.pump();
        assert!(!idle.busy && !idle.redraw && !idle.animate);

        // Gate the body so the export is provably in flight when we pump.
        let (gate_tx, gate_rx) = mpsc::channel::<()>();
        exporter.start_job(move |_ctx| {
            gate_rx.recv().expect("await release");
            Err("done".to_string())
        });
        let busy = exporter.pump();
        assert!(busy.busy, "export in flight");
        assert!(busy.redraw, "a busy export repaints its progress");
        assert!(!busy.animate, "export is not animation-cadence work");

        gate_tx.send(()).expect("release");
        // Pump reports idle once the job resolves; the "one more frame to show
        // status" is an app-side concern keyed off job.done, not a pump fact.
        wait_until(|| {
            let p = exporter.pump();
            !p.busy && !p.redraw
        });
        assert!(exporter.job.done);
    }

    #[test]
    fn exporter_pump_returns_to_idle_after_cancel() {
        use terra_jobs::Pollable;

        let mut exporter = BackgroundExporter::new();
        exporter.start_job(|ctx| {
            while !ctx.token().is_cancelled() {
                std::thread::yield_now();
            }
            Err("produced after cancel — must be discarded".to_string())
        });
        assert!(exporter.pump().busy, "busy until cancelled");

        exporter.cancel();
        wait_until(|| !exporter.pump().busy);
        assert!(
            exporter.job.result.is_none(),
            "a cancelled export leaves no Done result"
        );
    }

    #[test]
    fn project_io_pump_reports_busy_while_running_then_idle() {
        use std::sync::mpsc;
        use terra_jobs::Pollable;

        let mut io = BackgroundProjectIo::new();
        assert!(!io.pump().busy, "idle before any job");

        let (gate_tx, gate_rx) = mpsc::channel::<()>();
        io.start_job_for_test(PathBuf::from("gated.terra"), move |_| {
            gate_rx.recv().expect("await release");
            ProjectIoResult::Saved {
                path: PathBuf::from("gated.terra"),
            }
        });
        let busy = io.pump();
        assert!(busy.busy, "save in flight");
        assert!(busy.redraw, "a busy save repaints its status");
        assert!(!busy.animate, "project IO is not animation-cadence work");

        gate_tx.send(()).expect("release");
        wait_until(|| !io.pump().busy);
        // The typed result survives the pump for the app-side drain.
        assert!(matches!(
            io.result.take(),
            Some(ProjectIoResult::Saved { .. })
        ));
    }
}
