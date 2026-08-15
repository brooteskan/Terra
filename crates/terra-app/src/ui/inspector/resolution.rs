//! Read-only source and evaluation-resolution presentation for the Inspector.

use super::quality_label;
use crate::ui::UiState;
use std::path::{Path, PathBuf};
use terra_core::document::TerrainDocument;
use terra_core::layer::{
    CachePolicy, EvaluationResolutionBehavior, GridDimensions, Layer, LayerId,
    LayerResolutionSource,
};
use terra_gui::{label, section_header, GuiContext};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ImportedDimensions {
    NotSelected,
    Available(GridDimensions),
    Unavailable,
}

#[derive(Debug)]
struct ImportedSourceEntry {
    layer: LayerId,
    path: PathBuf,
    dimensions: ImportedDimensions,
}

/// Single-entry cache: only the currently inspected imported source matters.
#[derive(Debug, Default)]
pub(super) struct ImportedSourceCache {
    entry: Option<ImportedSourceEntry>,
}

impl ImportedSourceCache {
    fn resolve(&mut self, layer: LayerId, path: &str) -> ImportedDimensions {
        self.resolve_with(layer, path, |path| {
            image::image_dimensions(path)
                .ok()
                .map(|(width, height)| GridDimensions::new(width, height))
        })
    }

    fn resolve_with<F>(&mut self, layer: LayerId, path: &str, probe: F) -> ImportedDimensions
    where
        F: FnOnce(&Path) -> Option<GridDimensions>,
    {
        if path.is_empty() {
            self.entry = None;
            return ImportedDimensions::NotSelected;
        }

        let path_ref = Path::new(path);
        if let Some(entry) = &self.entry {
            if entry.layer == layer && entry.path == path_ref {
                return entry.dimensions;
            }
        }

        let dimensions = probe(path_ref)
            .map(ImportedDimensions::Available)
            .unwrap_or(ImportedDimensions::Unavailable);
        self.entry = Some(ImportedSourceEntry {
            layer,
            path: path_ref.to_path_buf(),
            dimensions,
        });
        dimensions
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolutionRows {
    source: String,
    evaluation: String,
    behavior: &'static str,
    cache: Option<String>,
}

pub(super) fn draw_layer_resolution(
    ui: &mut GuiContext<'_>,
    doc: &TerrainDocument,
    ui_state: &UiState,
    state: &mut super::InspectorGuiState,
    layer: &Layer,
) {
    let rows = layer_resolution_rows(doc, ui_state, &mut state.source_metadata, layer);
    section_header(ui, "RESOLUTION");
    label(ui, &format!("Source: {}", rows.source));
    label(ui, &format!("Evaluation: {}", rows.evaluation));
    label(ui, &format!("Behavior: {}", rows.behavior));
    if let Some(cache) = rows.cache {
        label(ui, &format!("Cache: {cache}"));
    }
}

pub(super) fn draw_painted_mask_resolution(
    ui: &mut GuiContext<'_>,
    doc: &TerrainDocument,
    ui_state: &UiState,
) {
    let selected = ui_state.paint_mask.or(ui_state.selected_mask);
    let Some(paint) = selected
        .and_then(|id| doc.masks.iter().find(|mask| mask.id == id))
        .filter(|mask| mask.is_painted())
        .and_then(|mask| mask.paint.as_ref())
    else {
        return;
    };

    section_header(ui, "RESOLUTION");
    label(
        ui,
        &format!(
            "Source: {}",
            painted_mask_source_text(paint.width, paint.height)
        ),
    );
    label(
        ui,
        &format!("Evaluation: {}", evaluation_text(doc, ui_state)),
    );
    label(ui, "Behavior: Resampled to evaluation resolution");
}

fn painted_mask_source_text(width: u32, height: u32) -> String {
    format!("{width} x {height} (painted mask)")
}

fn layer_resolution_rows(
    doc: &TerrainDocument,
    ui_state: &UiState,
    source_cache: &mut ImportedSourceCache,
    layer: &Layer,
) -> ResolutionRows {
    let semantics = layer.kind.resolution_semantics();
    let source = match semantics.source {
        LayerResolutionSource::FixedRaster(dimensions) => {
            format!(
                "{} x {} (fixed raster)",
                dimensions.width, dimensions.height
            )
        }
        LayerResolutionSource::ImportedRaster { path } => {
            match source_cache.resolve(layer.id(), path) {
                ImportedDimensions::NotSelected => "Not selected (imported raster)".into(),
                ImportedDimensions::Available(dimensions) => {
                    format!("{} x {} (imported)", dimensions.width, dimensions.height)
                }
                ImportedDimensions::Unavailable => "Unavailable (imported raster)".into(),
            }
        }
        LayerResolutionSource::ResolutionIndependent => "Resolution-independent".into(),
        LayerResolutionSource::EvaluationInput => "Evaluation input".into(),
        LayerResolutionSource::MeshGeometry { .. } => {
            "Mesh geometry (resolution-independent)".into()
        }
    };
    let behavior = match semantics.behavior {
        EvaluationResolutionBehavior::Resampled => "Resampled to evaluation resolution",
        EvaluationResolutionBehavior::Rasterized => "Rasterized at evaluation resolution",
        EvaluationResolutionBehavior::Generated => "Generated at evaluation resolution",
        EvaluationResolutionBehavior::Processed => "Processed at evaluation resolution",
        EvaluationResolutionBehavior::Simulated => "Simulated at evaluation resolution",
    };
    let policy = layer.common.resolved_cache_policy();
    let cache = (policy != CachePolicy::Live)
        .then(|| format!("{} (per evaluation resolution)", policy.label()));

    ResolutionRows {
        source,
        evaluation: evaluation_text(doc, ui_state),
        behavior,
        cache,
    }
}

fn evaluation_text(doc: &TerrainDocument, ui_state: &UiState) -> String {
    let dimensions = if ui_state.profile.tex_w > 0 && ui_state.profile.tex_h > 0 {
        GridDimensions::new(ui_state.profile.tex_w, ui_state.profile.tex_h)
    } else {
        GridDimensions::square(
            ui_state
                .quality
                .resolution(doc.preview_resolution, doc.export_resolution),
        )
    };
    format!(
        "{} x {} ({})",
        dimensions.width,
        dimensions.height,
        quality_label(ui_state.quality)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use terra_core::eval::PreviewQuality;
    use terra_core::layer::{
        EffectFilterParams, ImportHeightmapParams, LayerKind, SculptParams, SculptStrokeParams,
        ThermalErosionParams,
    };

    #[test]
    fn imported_source_probe_is_cached_until_layer_or_path_changes() {
        let first = LayerId::new();
        let second = LayerId::new();
        let calls = Cell::new(0);
        let mut cache = ImportedSourceCache::default();
        let probe = |_: &Path| {
            calls.set(calls.get() + 1);
            Some(GridDimensions::new(2048, 1024))
        };

        assert_eq!(
            cache.resolve_with(first, "height.png", probe),
            ImportedDimensions::Available(GridDimensions::new(2048, 1024))
        );
        assert_eq!(
            cache.resolve_with(first, "height.png", probe),
            ImportedDimensions::Available(GridDimensions::new(2048, 1024))
        );
        assert_eq!(calls.get(), 1);

        cache.resolve_with(first, "other.png", probe);
        cache.resolve_with(second, "other.png", probe);
        assert_eq!(calls.get(), 3);
    }

    #[test]
    fn missing_and_empty_imports_never_fall_back_to_evaluation_dimensions() {
        let id = LayerId::new();
        let mut cache = ImportedSourceCache::default();
        assert_eq!(
            cache.resolve_with(id, "", |_| unreachable!()),
            ImportedDimensions::NotSelected
        );
        assert_eq!(
            cache.resolve_with(id, "missing.raw", |_| None),
            ImportedDimensions::Unavailable
        );
    }

    #[test]
    fn evaluation_text_tracks_quality_and_presented_dimensions() {
        let doc = TerrainDocument {
            preview_resolution: 4096,
            export_resolution: 8192,
            ..TerrainDocument::default()
        };
        let mut ui_state = UiState {
            quality: PreviewQuality::Draft,
            ..UiState::default()
        };

        assert_eq!(evaluation_text(&doc, &ui_state), "512 x 512 (Draft)");
        ui_state.quality = PreviewQuality::Full;
        assert_eq!(evaluation_text(&doc, &ui_state), "4096 x 4096 (Full)");
        ui_state.quality = PreviewQuality::Export;
        assert_eq!(evaluation_text(&doc, &ui_state), "8192 x 8192 (Export)");

        ui_state.profile.tex_w = 1536;
        ui_state.profile.tex_h = 1024;
        assert_eq!(evaluation_text(&doc, &ui_state), "1536 x 1024 (Export)");
    }

    #[test]
    fn rows_distinguish_fixed_semantic_filter_simulation_and_imported_sources() {
        let doc = TerrainDocument::default();
        let ui_state = UiState::default();
        let mut cache = ImportedSourceCache::default();

        let fixed = Layer::new(
            "Base",
            LayerKind::SculptBase(SculptParams::filled(320, 0.0)),
        );
        let fixed_rows = layer_resolution_rows(&doc, &ui_state, &mut cache, &fixed);
        assert_eq!(fixed_rows.source, "320 x 320 (fixed raster)");
        assert_eq!(fixed_rows.behavior, "Resampled to evaluation resolution");

        let semantic = Layer::new(
            "Strokes",
            LayerKind::SculptStrokes(SculptStrokeParams::default()),
        );
        let semantic_rows = layer_resolution_rows(&doc, &ui_state, &mut cache, &semantic);
        assert_eq!(semantic_rows.source, "Resolution-independent");
        assert_eq!(
            semantic_rows.behavior,
            "Rasterized at evaluation resolution"
        );

        let filter = Layer::new(
            "Filter",
            LayerKind::EffectFilter(EffectFilterParams::default()),
        );
        let filter_rows = layer_resolution_rows(&doc, &ui_state, &mut cache, &filter);
        assert_eq!(filter_rows.source, "Evaluation input");
        assert_eq!(filter_rows.behavior, "Processed at evaluation resolution");

        let simulation = Layer::new(
            "Thermal",
            LayerKind::ThermalErosion(ThermalErosionParams::default()),
        );
        let simulation_rows = layer_resolution_rows(&doc, &ui_state, &mut cache, &simulation);
        assert_eq!(
            simulation_rows.behavior,
            "Simulated at evaluation resolution"
        );

        let imported = Layer::new(
            "Import",
            LayerKind::ImportHeightmap(ImportHeightmapParams {
                path: "missing.png".into(),
                ..ImportHeightmapParams::default()
            }),
        );
        let imported_rows = layer_resolution_rows(&doc, &ui_state, &mut cache, &imported);
        assert_eq!(imported_rows.source, "Unavailable (imported raster)");
    }

    #[test]
    fn non_live_cache_policy_is_reported_without_changing_source_semantics() {
        let doc = TerrainDocument::default();
        let ui_state = UiState::default();
        let mut cache = ImportedSourceCache::default();
        let mut layer = Layer::new("Cached", LayerKind::SculptBase(SculptParams::default()));
        layer.common.set_cache_policy(CachePolicy::Baked);

        let rows = layer_resolution_rows(&doc, &ui_state, &mut cache, &layer);
        assert_eq!(rows.source, "512 x 512 (fixed raster)");
        assert_eq!(
            rows.cache.as_deref(),
            Some("Baked (per evaluation resolution)")
        );
    }

    #[test]
    fn painted_masks_report_their_stored_rectangular_grid() {
        assert_eq!(
            painted_mask_source_text(640, 384),
            "640 x 384 (painted mask)"
        );
    }
}
