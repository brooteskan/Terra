use crate::ui::PanelAction;
use terra_core::command::{resize_raster_source, OwnedRasterTarget};
use terra_core::raster::RasterResizeLimits;

use super::super::TerraApp;
use super::ApplyCtx;

pub(crate) fn try_apply(
    app: &mut TerraApp,
    action: PanelAction,
    ctx: &mut ApplyCtx,
) -> Result<(), PanelAction> {
    let PanelAction::ResizeRasterSource { target, dimensions } = action else {
        return Err(action);
    };

    let source = match target {
        OwnedRasterTarget::SculptBase(id) => {
            app.session
                .document
                .stack
                .find(id)
                .and_then(|layer| match &layer.kind {
                    terra_core::layer::LayerKind::SculptBase(params) => Some(params.dimensions()),
                    _ => None,
                })
        }
        OwnedRasterTarget::PaintedMask(id) => app
            .session
            .document
            .masks
            .iter()
            .find(|asset| asset.id == id)
            .and_then(|asset| asset.paint.as_ref())
            .map(|paint| paint.dimensions()),
    };
    match resize_raster_source(
        &mut app.session.document.stack,
        &mut app.session.document.masks,
        target,
        dimensions,
        RasterResizeLimits::default(),
    ) {
        Ok(command) => {
            app.session.history.push_executed(command);
            match target {
                OwnedRasterTarget::SculptBase(id) => ctx.dirty_from = Some(id),
                OwnedRasterTarget::PaintedMask(_) => {
                    ctx.mask_assets_mutated = true;
                    app.mask_overlay_dirty = true;
                    app.preview_dirty = true;
                }
            }
            ctx.doc_mutated = true;
            ctx.deferred_rebuild = true;
            app.ui_state.status = if source.is_some_and(|source| {
                dimensions.width >= source.width
                    && dimensions.height >= source.height
                    && dimensions != source
            }) {
                format!(
                    "Upsampled stored source to {} x {}; this does not add source detail",
                    dimensions.width, dimensions.height
                )
            } else {
                format!(
                    "Resized stored source to {} x {}",
                    dimensions.width, dimensions.height
                )
            };
        }
        Err(error) => app.ui_state.status = format!("Source resize failed: {error}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::PanelAction;
    use terra_core::command::OwnedRasterTarget;
    use terra_core::layer::{GridDimensions, LayerKind};
    use terra_core::mask::{MaskAsset, MaskId};

    #[test]
    fn sculpt_resize_invalidates_once_and_is_undoable() {
        let mut app = TerraApp::default();
        let id = app
            .session
            .document
            .stack
            .flatten_layers()
            .into_iter()
            .find(|layer| layer.kind.is_sculpt_base())
            .unwrap()
            .id();
        let token_before = app.eval_token;

        app.apply_actions(vec![PanelAction::ResizeRasterSource {
            target: OwnedRasterTarget::SculptBase(id),
            dimensions: GridDimensions::new(256, 128),
        }]);

        let LayerKind::SculptBase(params) = &app.session.document.stack.find(id).unwrap().kind
        else {
            unreachable!()
        };
        assert_eq!(params.dimensions(), GridDimensions::new(256, 128));
        assert_eq!(app.eval_token, token_before.wrapping_add(1));
        assert!(app.session.history.can_undo());

        app.undo();
        let LayerKind::SculptBase(params) = &app.session.document.stack.find(id).unwrap().kind
        else {
            unreachable!()
        };
        assert_eq!(params.dimensions(), GridDimensions::square(512));
    }

    #[test]
    fn painted_mask_resize_invalidates_once() {
        let mut app = TerraApp::default();
        let id = MaskId::new();
        app.session
            .document
            .masks
            .push(MaskAsset::new_painted(id, "Paint", 256));
        let token_before = app.eval_token;

        app.apply_actions(vec![PanelAction::ResizeRasterSource {
            target: OwnedRasterTarget::PaintedMask(id),
            dimensions: GridDimensions::new(128, 512),
        }]);

        assert_eq!(
            app.session.document.masks[0]
                .paint
                .as_ref()
                .unwrap()
                .dimensions(),
            GridDimensions::new(128, 512)
        );
        assert_eq!(app.eval_token, token_before.wrapping_add(1));
        assert!(app.mask_overlay_dirty);
    }
}
