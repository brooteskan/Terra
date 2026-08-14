use crate::ui::PanelAction;
use terra_core::command::{apply, EditorCommand};
use terra_core::layer::LayerKind;

use super::super::helpers::coalesce_layer_id;
use super::super::TerraApp;
use super::ApplyCtx;

pub(crate) fn try_apply(
    app: &mut TerraApp,
    action: PanelAction,
    ctx: &mut ApplyCtx,
) -> Result<(), PanelAction> {
    let result = match action {
        PanelAction::SetOperationApplyWhere { id, apply: mode } => {
            let (previous, previous_masks) = app
                .session
                .document
                .stack
                .find(id)
                .map(|l| (l.common.operation_placement.clone(), l.common.masks.clone()))
                .unwrap_or_default();
            let mut placement = previous.clone();
            placement.apply_where = mode;
            placement.sync_definition_from_apply_where();
            let cmd = EditorCommand::SetOperationPlacement {
                id,
                placement,
                previous,
                previous_masks,
            };
            apply(&cmd, &mut app.session.document.stack);
            app.session.history.push_executed(cmd);
            app.mark_dirty_from_stage(id);
            ctx.dirty_from = Some(id);
            ctx.doc_mutated = true;
        }
        PanelAction::SetOperationPlacementParams {
            id,
            height_min,
            height_max,
            slope_min,
            slope_max,
            flow_min,
            near_distance_m,
        } => {
            let (previous, previous_masks) = app
                .session
                .document
                .stack
                .find(id)
                .map(|l| (l.common.operation_placement.clone(), l.common.masks.clone()))
                .unwrap_or_default();
            let mut placement = previous.clone();
            if let Some(v) = height_min {
                placement.height_min = v;
            }
            if let Some(v) = height_max {
                placement.height_max = v;
            }
            if let Some(v) = slope_min {
                placement.slope_min = v;
            }
            if let Some(v) = slope_max {
                placement.slope_max = v;
            }
            if let Some(v) = flow_min {
                placement.flow_min = v;
            }
            if let Some(v) = near_distance_m {
                placement.near_distance_m = v;
            }
            placement.sync_definition_from_apply_where();
            let cmd = EditorCommand::SetOperationPlacement {
                id,
                placement,
                previous,
                previous_masks,
            };
            apply(&cmd, &mut app.session.document.stack);
            app.session
                .history
                .push_coalesced(cmd, Some((coalesce_layer_id(id), "apply_where")));
            app.mark_dirty_from_stage(id);
            ctx.dirty_from = Some(id);
            ctx.doc_mutated = true;
        }
        PanelAction::BrowseHeightmapPath { layer } => {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Heightmap", &["png", "jpg", "jpeg", "tif", "tiff", "raw"])
                .pick_file()
            {
                let path_str = path.display().to_string();
                let Some(previous) = app
                    .session
                    .document
                    .stack
                    .find(layer)
                    .map(|l| l.kind.clone())
                else {
                    {
                        ctx.continue_loop = true;
                        return Ok(());
                    }
                };
                let mut kind = previous.clone();
                let updated = match &mut kind {
                    LayerKind::ImportHeightmap(p) => {
                        p.path = path_str;
                        true
                    }
                    LayerKind::Stamp2d(p) => {
                        p.heightmap.path = path_str;
                        true
                    }
                    _ => false,
                };
                if updated {
                    let cmd = EditorCommand::SetKind {
                        id: layer,
                        kind,
                        previous,
                    };
                    apply(&cmd, &mut app.session.document.stack);
                    app.session.history.push_executed(cmd);
                    ctx.dirty_from = Some(layer);
                    ctx.doc_mutated = true;
                    app.ui_state.status = "Heightmap path updated".into();
                }
            }
        }
        PanelAction::BrowseMeshPath { layer } => {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter(
                    "Mesh / Height",
                    &["obj", "png", "jpg", "jpeg", "tif", "tiff"],
                )
                .pick_file()
            {
                let path_str = path.display().to_string();
                let Some(previous) = app
                    .session
                    .document
                    .stack
                    .find(layer)
                    .map(|l| l.kind.clone())
                else {
                    {
                        ctx.continue_loop = true;
                        return Ok(());
                    }
                };
                let mut kind = previous.clone();
                let updated = match &mut kind {
                    LayerKind::Stamp3d(p) => {
                        p.path = path_str;
                        true
                    }
                    _ => false,
                };
                if updated {
                    let cmd = EditorCommand::SetKind {
                        id: layer,
                        kind,
                        previous,
                    };
                    apply(&cmd, &mut app.session.document.stack);
                    app.session.history.push_executed(cmd);
                    ctx.dirty_from = Some(layer);
                    ctx.doc_mutated = true;
                    app.ui_state.status = "Mesh path updated".into();
                }
            }
        }
        other => return Err(other),
    };
    let _ = result;
    Ok(())
}
