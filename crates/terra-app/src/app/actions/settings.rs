use crate::ui::{PanelAction, TerrainSettingsUpdate};

use super::super::TerraApp;
use super::ApplyCtx;

pub(crate) fn try_apply(
    app: &mut TerraApp,
    action: PanelAction,
    ctx: &mut ApplyCtx,
) -> Result<(), PanelAction> {
    let PanelAction::UpdateTerrainSettings(update) = action else {
        return Err(action);
    };

    let preview_changed = apply_update(&mut app.session.document, update);
    if preview_changed.is_none() {
        return Ok(());
    }

    ctx.doc_mutated = true;
    if preview_changed == Some(true) {
        app.mark_all_layers_dirty();
        app.request_rebuild();
    }
    Ok(())
}

/// Returns `None` for a no-op, or whether the applied change affects preview evaluation.
fn apply_update(
    document: &mut terra_core::document::TerrainDocument,
    update: TerrainSettingsUpdate,
) -> Option<bool> {
    let mut changed = false;
    let mut preview_changed = false;

    if let Some(value) = update.export_resolution {
        assign_if_changed(
            &mut document.export_resolution,
            value.clamp(512, 8192),
            &mut changed,
        );
    }
    if let Some(value) = update.preview_resolution {
        let value = value.clamp(256, 8192);
        if document.preview_resolution != value {
            document.preview_resolution = value;
            changed = true;
            preview_changed = true;
        }
    }
    if let Some(value) = finite_clamped(update.precision, 0.25, 4.0) {
        if document.level_steps.precision != value {
            document.level_steps.precision = value;
            changed = true;
            preview_changed = true;
        }
    }
    if let Some(value) = finite_clamped(update.world_scale, 0.1, 10.0) {
        if document.level_steps.world_scale != value {
            document.level_steps.world_scale = value;
            changed = true;
            preview_changed = true;
        }
    }
    if let Some(value) = update.max_level {
        let value = value.clamp(0, 12);
        if document.level_steps.max_level != value {
            document.level_steps.max_level = value;
            changed = true;
            preview_changed = true;
        }
    }
    if let Some(value) = update.high_detail {
        if document.level_steps.high_detail != value {
            document.level_steps.high_detail = value;
            changed = true;
            preview_changed = true;
        }
    }
    if let Some(value) = update.show_hd_outline {
        if document.level_steps.show_hd_outline != value {
            document.level_steps.show_hd_outline = value;
            changed = true;
            preview_changed = true;
        }
    }

    changed.then_some(preview_changed)
}

fn assign_if_changed<T: PartialEq>(current: &mut T, value: T, changed: &mut bool) {
    if *current != value {
        *current = value;
        *changed = true;
    }
}

fn finite_clamped(value: Option<f32>, min: f32, max: f32) -> Option<f32> {
    value
        .filter(|value| value.is_finite())
        .map(|value| value.clamp(min, max))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use terra_core::analyze::HighDetailMode;
    use terra_core::eval::CachedOutput;
    use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
    use terra_core::layer::LayerId;

    use super::*;

    fn seed_clean_evaluator_cache(app: &mut TerraApp) -> LayerId {
        let id = app.session.document.preview_eval_stack().layer_ids()[0];
        app.scheduler.evaluator.cache.insert(
            id,
            CachedOutput {
                height: Heightfield::zeros(HeightfieldMetrics::new(4, 4, 4.0, 4.0)),
                generation: 0,
                dirty: false,
                aux: HashMap::new(),
                strata: None,
            },
        );
        assert!(!app.scheduler.evaluator.cache.is_dirty(id));
        id
    }

    #[test]
    fn export_only_change_marks_document_without_rebuilding_preview() {
        let mut app = TerraApp::default();
        app.worker_mark_all_dirty = false;
        let cached_layer = seed_clean_evaluator_cache(&mut app);
        let token_before = app.eval_token;
        let runtime_before = app.terrain_runtime.output_revision();

        app.apply_actions(vec![PanelAction::UpdateTerrainSettings(
            TerrainSettingsUpdate {
                export_resolution: Some(4096),
                ..Default::default()
            },
        )]);

        assert_eq!(app.session.document.export_resolution, 4096);
        assert!(app.document_dirty);
        assert_eq!(app.eval_token, token_before);
        assert_eq!(app.terrain_runtime.output_revision(), runtime_before);
        assert!(!app.pending_eval);
        assert!(!app.worker_mark_all_dirty);
        assert!(!app.scheduler.evaluator.cache.is_dirty(cached_layer));
    }

    #[test]
    fn normalized_no_op_and_non_finite_values_do_nothing() {
        let mut app = TerraApp::default();
        app.session.document.export_resolution = 512;
        app.session.document.preview_resolution = 256;
        app.worker_mark_all_dirty = false;
        let cached_layer = seed_clean_evaluator_cache(&mut app);
        let token_before = app.eval_token;
        let runtime_before = app.terrain_runtime.output_revision();

        app.apply_actions(vec![PanelAction::UpdateTerrainSettings(
            TerrainSettingsUpdate {
                export_resolution: Some(1),
                preview_resolution: Some(1),
                precision: Some(f32::NAN),
                world_scale: Some(f32::INFINITY),
                ..Default::default()
            },
        )]);

        assert!(!app.document_dirty);
        assert_eq!(app.eval_token, token_before);
        assert_eq!(app.terrain_runtime.output_revision(), runtime_before);
        assert!(!app.pending_eval);
        assert!(!app.worker_mark_all_dirty);
        assert!(!app.scheduler.evaluator.cache.is_dirty(cached_layer));
    }

    #[test]
    fn preview_resolution_change_invalidates_all_preview_backends_once() {
        let mut app = TerraApp::default();
        app.worker_mark_all_dirty = false;
        let cached_layer = seed_clean_evaluator_cache(&mut app);
        let token_before = app.eval_token;
        let runtime_before = app.terrain_runtime.output_revision();

        app.apply_actions(vec![PanelAction::UpdateTerrainSettings(
            TerrainSettingsUpdate {
                preview_resolution: Some(u32::MAX),
                ..Default::default()
            },
        )]);

        assert_eq!(app.session.document.preview_resolution, 8192);
        assert!(app.document_dirty);
        assert_eq!(app.eval_token, token_before.wrapping_add(1));
        assert_eq!(app.terrain_runtime.output_revision(), runtime_before + 1);
        assert_eq!(app.terrain_runtime.pyramid.config.target_resolution, 8192);
        assert!(app.pending_eval);
        assert!(app.worker_mark_all_dirty);
        assert!(app.scheduler.evaluator.cache.is_dirty(cached_layer));
    }

    #[test]
    fn combined_level_step_update_clamps_and_requests_one_rebuild() {
        let mut app = TerraApp::default();
        app.worker_mark_all_dirty = false;
        app.session.document.level_steps.hd_zone = [0.1, 0.2, 0.8, 0.9];
        app.session.document.level_steps.min_preview_wavelength_m = 3.5;
        let token_before = app.eval_token;
        let runtime_before = app.terrain_runtime.output_revision();

        app.apply_actions(vec![PanelAction::UpdateTerrainSettings(
            TerrainSettingsUpdate {
                precision: Some(99.0),
                world_scale: Some(-50.0),
                max_level: Some(u32::MAX),
                high_detail: Some(HighDetailMode::Camera),
                show_hd_outline: Some(true),
                ..Default::default()
            },
        )]);

        let level_steps = &app.session.document.level_steps;
        assert_eq!(level_steps.precision, 4.0);
        assert_eq!(level_steps.world_scale, 0.1);
        assert_eq!(level_steps.max_level, 12);
        assert_eq!(level_steps.high_detail, HighDetailMode::Camera);
        assert!(level_steps.show_hd_outline);
        assert_eq!(level_steps.hd_zone, [0.1, 0.2, 0.8, 0.9]);
        assert_eq!(level_steps.min_preview_wavelength_m, 3.5);
        assert!(app.document_dirty);
        assert_eq!(app.eval_token, token_before.wrapping_add(1));
        assert_eq!(app.terrain_runtime.output_revision(), runtime_before + 1);
        assert!(app.pending_eval);
        assert!(app.worker_mark_all_dirty);
    }
}
