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
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use terra_core::document::TerrainDocument;
use terra_core::eval::{EvalContext, PreviewQuality, StackEvaluator};
use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
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

enum JobMsg {
    Progress(f32),
    Done(Result<ExportResult, String>),
}

/// Non-blocking export worker.
pub struct BackgroundExporter {
    tx: Option<Sender<JobMsg>>,
    rx: Receiver<JobMsg>,
    pub job: BuildJob,
}

impl BackgroundExporter {
    pub fn new() -> Self {
        let (_tx, rx) = mpsc::channel();
        Self {
            tx: None,
            rx,
            job: BuildJob {
                progress: 0.0,
                done: true,
                result: None,
            },
        }
    }

    pub fn start(&mut self, doc: TerrainDocument, out_dir: PathBuf) {
        let (tx, rx) = mpsc::channel();
        self.rx = rx;
        self.tx = Some(tx.clone());
        self.job = BuildJob {
            progress: 0.0,
            done: false,
            result: None,
        };
        thread::spawn(move || {
            let _ = tx.send(JobMsg::Progress(0.1));
            let mut doc = doc;
            // Ensure sparse biome paint is baked into mask assets for this export.
            doc.sync_all_biome_paint_masks();
            let _ = tx.send(JobMsg::Progress(0.3));
            match evaluate_document_for_export(&doc) {
                Ok((hf, ctx)) => {
                    let _ = tx.send(JobMsg::Progress(0.7));
                    let req = ExportRequest {
                        out_dir,
                        ..ExportRequest::default()
                    };
                    let result = export_package(&hf, &ctx, &req).map_err(|e| e.to_string());
                    let _ = tx.send(JobMsg::Done(result));
                }
                Err(e) => {
                    let _ = tx.send(JobMsg::Done(Err(e.to_string())));
                }
            }
        });
    }

    pub fn poll(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                JobMsg::Progress(p) => self.job.progress = p,
                JobMsg::Done(r) => {
                    self.job.progress = 1.0;
                    self.job.done = true;
                    self.job.result = Some(r);
                }
            }
        }
    }
}

/// Evaluate through the same v8 single-stack authority used by the viewport.
///
/// Keeping this as a small, testable boundary prevents compatibility containers
/// (`TerrainDocument::world`) from silently becoming an export source again.
fn evaluate_document_for_export(
    doc: &TerrainDocument,
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
    let mut ctx = EvalContext::new(metrics);
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

pub fn save_project(doc: &TerrainDocument, path: &std::path::Path) -> Result<(), IoError> {
    let json = doc.to_json()?;
    std::fs::write(path, json)?;
    Ok(())
}

pub fn load_project(path: &std::path::Path) -> Result<TerrainDocument, IoError> {
    let s = std::fs::read_to_string(path)?;
    Ok(TerrainDocument::from_json(&s)?)
}

enum ProjectIoMsg {
    Progress(&'static str),
    Saved { path: PathBuf },
    Loaded { path: PathBuf, doc: TerrainDocument },
    Failed { path: PathBuf, error: String },
}

/// Result of a finished background project I/O job.
#[derive(Debug)]
pub enum ProjectIoResult {
    Saved { path: PathBuf },
    Loaded { path: PathBuf, doc: TerrainDocument },
    Failed { path: PathBuf, error: String },
}

/// Non-blocking project save/load worker (serialize/parse/fs off the UI thread).
pub struct BackgroundProjectIo {
    rx: Receiver<ProjectIoMsg>,
    busy: bool,
    status: Option<&'static str>,
    pub result: Option<ProjectIoResult>,
}

impl BackgroundProjectIo {
    pub fn new() -> Self {
        let (_tx, rx) = mpsc::channel();
        Self {
            rx,
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
        let (tx, rx) = mpsc::channel();
        self.rx = rx;
        self.busy = true;
        self.status = Some("Saving…");
        self.result = None;
        thread::spawn(move || {
            let _ = tx.send(ProjectIoMsg::Progress("Saving…"));
            match doc.into_json() {
                Ok(json) => match std::fs::write(&path, json) {
                    Ok(()) => {
                        let _ = tx.send(ProjectIoMsg::Saved { path });
                    }
                    Err(e) => {
                        let _ = tx.send(ProjectIoMsg::Failed {
                            path,
                            error: e.to_string(),
                        });
                    }
                },
                Err(e) => {
                    let _ = tx.send(ProjectIoMsg::Failed {
                        path,
                        error: e.to_string(),
                    });
                }
            }
        });
    }

    pub fn start_load(&mut self, path: PathBuf) {
        let (tx, rx) = mpsc::channel();
        self.rx = rx;
        self.busy = true;
        self.status = Some("Loading…");
        self.result = None;
        thread::spawn(move || {
            let _ = tx.send(ProjectIoMsg::Progress("Loading…"));
            match std::fs::read_to_string(&path) {
                Ok(s) => match TerrainDocument::from_json(&s) {
                    Ok(doc) => {
                        let _ = tx.send(ProjectIoMsg::Loaded { path, doc });
                    }
                    Err(e) => {
                        let _ = tx.send(ProjectIoMsg::Failed {
                            path,
                            error: e.to_string(),
                        });
                    }
                },
                Err(e) => {
                    let _ = tx.send(ProjectIoMsg::Failed {
                        path,
                        error: e.to_string(),
                    });
                }
            }
        });
    }

    pub fn poll(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                ProjectIoMsg::Progress(s) => self.status = Some(s),
                ProjectIoMsg::Saved { path } => {
                    self.busy = false;
                    self.status = None;
                    self.result = Some(ProjectIoResult::Saved { path });
                }
                ProjectIoMsg::Loaded { path, doc } => {
                    self.busy = false;
                    self.status = None;
                    self.result = Some(ProjectIoResult::Loaded { path, doc });
                }
                ProjectIoMsg::Failed { path, error } => {
                    self.busy = false;
                    self.status = None;
                    self.result = Some(ProjectIoResult::Failed { path, error });
                }
            }
        }
    }
}

impl Default for BackgroundProjectIo {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod export_worker_tests {
    use super::*;
    use terra_core::layer::{FlatParams, Layer, LayerKind, LayerStack};

    #[test]
    fn v8_export_evaluates_the_authoritative_stack() {
        let mut doc = TerrainDocument::new_default();
        doc.export_resolution = 32;
        doc.stack = LayerStack::new();
        doc.stack.push(Layer::new(
            "Export Height",
            LayerKind::Flat(FlatParams { height: 73.0 }),
        ));

        assert!(!doc.stack.flatten_layers().is_empty());

        let (height, _) =
            evaluate_document_for_export(&doc).expect("the stack should evaluate for export");
        assert!((height.get(16, 16) - 73.0).abs() < 1.0e-4);
    }
}
