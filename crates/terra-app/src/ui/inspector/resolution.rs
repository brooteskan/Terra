//! Source/evaluation resolution presentation and explicit owned-raster resizing.

use super::quality_label;
use crate::ui::actions::PanelAction;
use crate::ui::style::{self, FONT_SCALE, PAD};
use crate::ui::UiState;
use std::path::{Path, PathBuf};
use terra_core::command::OwnedRasterTarget;
use terra_core::document::TerrainDocument;
use terra_core::layer::{
    effective_detail, CachePolicy, EvaluationResolutionBehavior, GridDimensions, Layer, LayerId,
    LayerResolutionSource,
};
use terra_gui::{chip_button, combo, label, section_header, Color, GuiContext, Id, Rect};

const RESOLUTION_PRESETS: &[u32] = &[128, 256, 512, 1024, 2048, 4096, 8192];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PendingRasterResize {
    target: OwnedRasterTarget,
    from: GridDimensions,
    to: GridDimensions,
}

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
    effective_detail: Option<String>,
}

pub(super) fn draw_layer_resolution(
    ui: &mut GuiContext<'_>,
    doc: &TerrainDocument,
    ui_state: &UiState,
    state: &mut super::InspectorGuiState,
    layer: &Layer,
    actions: &mut Vec<PanelAction>,
) {
    let rows = layer_resolution_rows(doc, ui_state, &mut state.source_metadata, layer);
    section_header(ui, "RESOLUTION");
    if let terra_core::layer::LayerKind::SculptBase(params) = &layer.kind {
        draw_source_dropdown(
            ui,
            state,
            OwnedRasterTarget::SculptBase(layer.id()),
            params.dimensions(),
            actions,
        );
    } else {
        label(ui, &format!("Source: {}", rows.source));
    }
    label(ui, &format!("Evaluation: {}", rows.evaluation));
    label(ui, &format!("Behavior: {}", rows.behavior));
    if let Some(detail) = rows.effective_detail {
        label(ui, &format!("Effective detail: {detail}"));
    }
    if let Some(cache) = rows.cache {
        label(ui, &format!("Cache: {cache}"));
    }
}

pub(super) fn draw_painted_mask_resolution(
    ui: &mut GuiContext<'_>,
    doc: &TerrainDocument,
    ui_state: &UiState,
    state: &mut super::InspectorGuiState,
    actions: &mut Vec<PanelAction>,
) {
    let selected = ui_state.paint_mask.or(ui_state.selected_mask);
    let Some((mask_id, paint)) = selected
        .and_then(|id| doc.masks.iter().find(|mask| mask.id == id))
        .filter(|mask| mask.is_painted())
        .and_then(|mask| mask.paint.as_ref().map(|paint| (mask.id, paint)))
    else {
        return;
    };

    section_header(ui, "RESOLUTION");
    draw_source_dropdown(
        ui,
        state,
        OwnedRasterTarget::PaintedMask(mask_id),
        paint.dimensions(),
        actions,
    );
    label(
        ui,
        &format!("Evaluation: {}", evaluation_text(doc, ui_state)),
    );
    label(ui, "Behavior: Resampled to evaluation resolution");
    let evaluation = evaluation_dimensions(doc, ui_state);
    let detail = effective_detail(paint.dimensions(), evaluation);
    label(
        ui,
        &format!(
            "Effective detail: {} x {} ({})",
            detail.dimensions.width,
            detail.dimensions.height,
            detail.limit.label()
        ),
    );
}

fn draw_source_dropdown(
    ui: &mut GuiContext<'_>,
    state: &mut super::InspectorGuiState,
    target: OwnedRasterTarget,
    source: GridDimensions,
    actions: &mut Vec<PanelAction>,
) {
    let mut dimensions: Vec<GridDimensions> = RESOLUTION_PRESETS
        .iter()
        .filter_map(|preset| dimensions_for_preset(source, *preset))
        .collect();
    let mut labels: Vec<String> = dimensions
        .iter()
        .map(|dimensions| format!("{} x {}", dimensions.width, dimensions.height))
        .collect();
    let current_preset = dimensions
        .iter()
        .position(|dimensions| *dimensions == source);
    let mut selected = current_preset.unwrap_or(0);
    if current_preset.is_none() {
        labels.insert(0, format!("Current ({} x {})", source.width, source.height));
        dimensions.insert(0, source);
        selected = 0;
    }
    let items: Vec<&str> = labels.iter().map(String::as_str).collect();
    if combo(ui, "Source", &mut selected, &items) {
        queue_or_confirm_resize(state, target, source, dimensions[selected], actions);
    }
}

fn dimensions_for_preset(source: GridDimensions, preset: u32) -> Option<GridDimensions> {
    if source.width == 0 || source.height == 0 {
        return None;
    }
    let dimensions = if source.width >= source.height {
        GridDimensions::new(
            preset,
            (f64::from(preset) * f64::from(source.height) / f64::from(source.width)).round() as u32,
        )
    } else {
        GridDimensions::new(
            (f64::from(preset) * f64::from(source.width) / f64::from(source.height)).round() as u32,
            preset,
        )
    };
    (dimensions.width >= 128 && dimensions.height >= 128).then_some(dimensions)
}

pub(crate) fn draw_resize_confirmation_modal(
    ui: &mut GuiContext<'_>,
    state: &mut super::InspectorGuiState,
) -> Option<PanelAction> {
    let pending = state.pending_source_downsize?;

    ui.begin_overlay();
    ui.panel(
        Rect::from_pos_size(0.0, 0.0, ui.screen_w, ui.screen_h),
        Color::rgba(0.0, 0.0, 0.0, 0.55),
    );
    let width = 440.0_f32.min(ui.screen_w - 40.0);
    let height = 184.0;
    let dialog = Rect::from_pos_size(
        (ui.screen_w - width) * 0.5,
        (ui.screen_h - height) * 0.5,
        width,
        height,
    );
    ui.panel_rounded(dialog, style::POPUP_BG, style::RADIUS_MD);
    ui.label_at(
        dialog.min_x + PAD * 1.5,
        dialog.min_y + PAD * 1.5,
        "Reduce source resolution?",
        style::TEXT,
        FONT_SCALE * 1.2,
    );
    ui.label_at(
        dialog.min_x + PAD * 1.5,
        dialog.min_y + 50.0,
        &format!(
            "{} x {} to {} x {}",
            pending.from.width, pending.from.height, pending.to.width, pending.to.height
        ),
        style::TEXT,
        FONT_SCALE,
    );
    ui.label_at(
        dialog.min_x + PAD * 1.5,
        dialog.min_y + 76.0,
        "This permanently discards source detail.",
        style::TEXT_DIM,
        FONT_SCALE,
    );

    let button_width = 100.0;
    let button_height = 34.0;
    let button_y = dialog.max_y - PAD - button_height;
    let cancel_rect = Rect::from_pos_size(
        dialog.max_x - PAD - button_width * 2.0 - 10.0,
        button_y,
        button_width,
        button_height,
    );
    let ok_rect = Rect::from_pos_size(
        dialog.max_x - PAD - button_width,
        button_y,
        button_width,
        button_height,
    );
    let cancel = chip_button(
        ui,
        Id::new("source_resize_cancel"),
        "Cancel",
        cancel_rect,
        false,
    ) || ui.input.escape_pressed;
    let confirm = chip_button(ui, Id::new("source_resize_ok"), "OK", ok_rect, true);
    ui.end_overlay();

    if cancel {
        state.pending_source_downsize = None;
        None
    } else if confirm {
        state.pending_source_downsize = None;
        Some(PanelAction::ResizeRasterSource {
            target: pending.target,
            dimensions: pending.to,
        })
    } else {
        None
    }
}

fn queue_or_confirm_resize(
    state: &mut super::InspectorGuiState,
    target: OwnedRasterTarget,
    source: GridDimensions,
    dimensions: GridDimensions,
    actions: &mut Vec<PanelAction>,
) {
    if dimensions.is_smaller_than(source) {
        state.pending_source_downsize = Some(PendingRasterResize {
            target,
            from: source,
            to: dimensions,
        });
    } else {
        state.pending_source_downsize = None;
        actions.push(PanelAction::ResizeRasterSource { target, dimensions });
    }
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

    let evaluation_dimensions = evaluation_dimensions(doc, ui_state);
    let effective_detail = match semantics.source {
        LayerResolutionSource::FixedRaster(dimensions) => {
            Some(effective_detail(dimensions, evaluation_dimensions))
        }
        LayerResolutionSource::ImportedRaster { path } => {
            match source_cache.resolve(layer.id(), path) {
                ImportedDimensions::Available(dimensions) => {
                    Some(effective_detail(dimensions, evaluation_dimensions))
                }
                _ => None,
            }
        }
        _ => None,
    }
    .map(|detail| {
        format!(
            "{} x {} ({})",
            detail.dimensions.width,
            detail.dimensions.height,
            detail.limit.label()
        )
    });

    ResolutionRows {
        source,
        evaluation: evaluation_text(doc, ui_state),
        behavior,
        cache,
        effective_detail,
    }
}

fn evaluation_dimensions(doc: &TerrainDocument, ui_state: &UiState) -> GridDimensions {
    if ui_state.profile.tex_w > 0 && ui_state.profile.tex_h > 0 {
        GridDimensions::new(ui_state.profile.tex_w, ui_state.profile.tex_h)
    } else {
        GridDimensions::square(doc.bounded_settings().map_or(0, |bounded| {
            ui_state
                .quality
                .resolution(bounded.preview_resolution, bounded.export_resolution)
        }))
    }
}

fn evaluation_text(doc: &TerrainDocument, ui_state: &UiState) -> String {
    let dimensions = evaluation_dimensions(doc, ui_state);
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
    use terra_core::layer::{
        EffectFilterParams, ImportHeightmapParams, LayerKind, SculptParams, SculptStrokeParams,
        ThermalErosionParams,
    };
    use terra_core::quality::PreviewQuality;
    use terra_gui::{GuiInput, GuiState};

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
        let mut doc = TerrainDocument::default();
        let bounded = doc.bounded_settings_mut().unwrap();
        bounded.preview_resolution = 4096;
        bounded.export_resolution = 8192;
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
    fn downsizing_requires_confirmation_but_upsizing_queues_immediately() {
        let target = OwnedRasterTarget::SculptBase(LayerId::new());
        let mut state = super::super::InspectorGuiState::default();
        let mut actions = Vec::new();

        queue_or_confirm_resize(
            &mut state,
            target,
            GridDimensions::square(512),
            GridDimensions::square(256),
            &mut actions,
        );
        assert!(actions.is_empty());
        assert_eq!(
            state.pending_source_downsize.unwrap().to,
            GridDimensions::square(256)
        );

        queue_or_confirm_resize(
            &mut state,
            target,
            GridDimensions::square(512),
            GridDimensions::square(1024),
            &mut actions,
        );
        assert!(state.pending_source_downsize.is_none());
        assert!(matches!(
            actions.as_slice(),
            [PanelAction::ResizeRasterSource { dimensions, .. }]
                if *dimensions == GridDimensions::square(1024)
        ));
    }

    #[test]
    fn presets_preserve_rectangular_source_aspect_ratio() {
        let source = GridDimensions::new(640, 384);
        assert_eq!(
            dimensions_for_preset(source, 1280),
            Some(GridDimensions::new(1280, 768))
        );
        assert_eq!(640_u64 * 768, 384_u64 * 1280);
        assert_eq!(dimensions_for_preset(source, 128), None);
    }

    #[test]
    fn confirmation_buttons_receive_clicks_while_background_input_is_suspended() {
        let target = OwnedRasterTarget::SculptBase(LayerId::new());
        let mut inspector = super::super::InspectorGuiState::default();
        inspector.pending_source_downsize = Some(PendingRasterResize {
            target,
            from: GridDimensions::square(512),
            to: GridDimensions::square(256),
        });
        let mut gui_state = GuiState::default();
        let pointer = Some((558.0, 363.0));

        {
            let mut ui = GuiContext::begin(
                800.0,
                600.0,
                1.0,
                GuiInput {
                    pointer,
                    primary_down: true,
                    ..GuiInput::default()
                },
                &mut gui_state,
            );
            ui.suspend_pointer_edges();
            assert!(ui
                .with_menu_input(|ui| draw_resize_confirmation_modal(ui, &mut inspector))
                .is_none());
            ui.end();
        }
        let action = {
            let mut ui = GuiContext::begin(
                800.0,
                600.0,
                1.0,
                GuiInput {
                    pointer,
                    primary_down: false,
                    ..GuiInput::default()
                },
                &mut gui_state,
            );
            ui.suspend_pointer_edges();
            let action =
                ui.with_menu_input(|ui| draw_resize_confirmation_modal(ui, &mut inspector));
            ui.end();
            action
        };

        assert!(matches!(
            action,
            Some(PanelAction::ResizeRasterSource {
                target: action_target,
                dimensions,
            }) if action_target == target && dimensions == GridDimensions::square(256)
        ));
        assert!(inspector.pending_source_downsize.is_none());
    }
}
