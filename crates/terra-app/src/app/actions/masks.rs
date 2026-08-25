use crate::ui::{MaskEditAction, PanelAction};

use super::super::TerraApp;
use super::ApplyCtx;

fn reject_infinite_mask_source(app: &mut TerraApp, source: &terra_core::mask::MaskSource) -> bool {
    if app.session.document.infinite_settings().is_none() {
        return false;
    }
    let Some(reason) =
        terra_core::terrain_plan::mask_source_infinite_capability(source).rejection()
    else {
        return false;
    };
    app.ui_state.status = format!("Mask is unavailable in Infinite projects: {reason}");
    true
}

// Returns the unhandled action on `Err` so the next handler in the chain can try it
// (see actions/mod.rs); that payload is the intrinsic-size `PanelAction` (`LayerKind`).
#[allow(clippy::result_large_err)]
pub(crate) fn try_apply(
    app: &mut TerraApp,
    action: PanelAction,
    ctx: &mut ApplyCtx,
) -> Result<(), PanelAction> {
    match action {
        PanelAction::AddMask(mut asset) => {
            if reject_infinite_mask_source(app, &asset.source) {
                ctx.continue_loop = true;
                return Ok(());
            }
            asset.prepare_for_document();
            let id = asset.id;
            let painted = asset.is_painted();
            app.session.document.masks.push(asset);
            app.ui_state.selected_mask = Some(id);
            if painted {
                app.ui_state.paint_mask = Some(id);
            } else {
                app.ui_state.paint_mask = None;
            }
            app.ui_state.focus_created_mask(painted);
            ctx.mask_assets_mutated = true;
            ctx.doc_mutated = true;
            app.preview_dirty = true;
            app.mask_overlay_dirty = true;
        }
        PanelAction::SelectMask(mask_id) => {
            app.ui_state.selected_mask = Some(mask_id);
            let painted = app
                .session
                .document
                .masks
                .iter()
                .find(|a| a.id == mask_id)
                .is_some_and(|a| a.is_painted());
            if painted {
                app.ui_state.paint_mask = Some(mask_id);
                app.ui_state.focus_created_mask(true);
            } else {
                app.ui_state.paint_mask = None;
                if app.ui_state.editor_tool == crate::ui::EditorTool::PaintMask {
                    app.ui_state.set_editor_tool(crate::ui::EditorTool::Move);
                }
            }
            app.preview_dirty = true;
            app.mask_overlay_dirty = true;
        }
        PanelAction::UpdateMaskAsset(asset) => {
            if reject_infinite_mask_source(app, &asset.source) {
                ctx.continue_loop = true;
                return Ok(());
            }
            if let Some(existing) = app
                .session
                .document
                .masks
                .iter_mut()
                .find(|existing| existing.id == asset.id)
            {
                let color_only = existing.display_color != asset.display_color
                    && existing.name == asset.name
                    && existing.ops.len() == asset.ops.len()
                    && existing
                        .paint
                        .as_ref()
                        .map(|p| (p.width, p.height, p.samples.len()))
                        == asset
                            .paint
                            .as_ref()
                            .map(|p| (p.width, p.height, p.samples.len()))
                    && existing
                        .paint
                        .as_ref()
                        .zip(asset.paint.as_ref())
                        .map(|(a, b)| a.samples == b.samples)
                        .unwrap_or(true);
                *existing = asset;
                if color_only {
                    // Display colour is visual-only â€” refresh overlay, skip rebuild.
                    app.mask_overlay_dirty = true;
                } else {
                    ctx.mask_assets_mutated = true;
                    app.preview_dirty = true;
                    app.mask_overlay_dirty = true;
                }
                ctx.doc_mutated = true;
            }
        }
        PanelAction::BindMaskToLayer { layer, mask } => {
            let rejected = app
                .session
                .document
                .masks
                .iter()
                .find(|asset| asset.id == mask)
                .map(|asset| asset.source.clone())
                .is_some_and(|source| reject_infinite_mask_source(app, &source));
            if rejected {
                ctx.continue_loop = true;
                return Ok(());
            }
            if let Some(target) = app.session.document.stack.find_mut(layer) {
                if !target
                    .common
                    .masks
                    .iter()
                    .any(|binding| binding.mask.id == mask)
                {
                    target
                        .common
                        .masks
                        .push(terra_core::mask::MaskRef::new(mask));
                    ctx.dirty_from = Some(layer);
                }
            } else if let Some(group) = app.session.document.stack.find_group_mut(layer) {
                if !group.masks.iter().any(|binding| binding.mask.id == mask) {
                    group.masks.push(terra_core::mask::MaskRef::new(mask));
                    app.mark_all_layers_dirty();
                    app.request_rebuild();
                    ctx.doc_mutated = true;
                }
            }
        }
        PanelAction::UnbindMask { layer, mask } => {
            if let Some(target) = app.session.document.stack.find_mut(layer) {
                target
                    .common
                    .masks
                    .retain(|binding| binding.mask.id != mask);
                ctx.dirty_from = Some(layer);
            } else if let Some(group) = app.session.document.stack.find_group_mut(layer) {
                group.masks.retain(|binding| binding.mask.id != mask);
                app.mark_all_layers_dirty();
                app.request_rebuild();
                ctx.doc_mutated = true;
            }
        }
        PanelAction::UpdateMaskBinding {
            layer,
            mask,
            strength,
            invert,
        } => {
            if let Some(binding) = app
                .session
                .document
                .stack
                .find_mut(layer)
                .and_then(|target| {
                    target
                        .common
                        .masks
                        .iter_mut()
                        .find(|binding| binding.mask.id == mask)
                })
            {
                binding.mask.strength = strength;
                binding.mask.invert = invert;
                ctx.dirty_from = Some(layer);
            } else if let Some(binding) = app
                .session
                .document
                .stack
                .find_group_mut(layer)
                .and_then(|group| group.masks.iter_mut().find(|entry| entry.mask.id == mask))
            {
                binding.mask.strength = strength;
                binding.mask.invert = invert;
                app.mark_all_layers_dirty();
                app.request_rebuild();
                ctx.doc_mutated = true;
            }
        }
        PanelAction::CycleMaskCombine { layer, mask } => {
            use terra_core::mask::MaskCombine;
            let cycle = |c: MaskCombine| c.cycle();
            let mut cycled = false;
            if let Some(binding) = app
                .session
                .document
                .stack
                .find_mut(layer)
                .and_then(|target| {
                    target
                        .common
                        .masks
                        .iter_mut()
                        .find(|binding| binding.mask.id == mask)
                })
            {
                binding.combine = cycle(binding.combine);
                ctx.dirty_from = Some(layer);
                cycled = true;
            } else if let Some(group) = app.session.document.stack.find_group_mut(layer) {
                if let Some(binding) = group.masks.iter_mut().find(|b| b.mask.id == mask) {
                    binding.combine = cycle(binding.combine);
                    cycled = true;
                }
            }
            if cycled && ctx.dirty_from.is_none() {
                app.mark_all_layers_dirty();
                app.request_rebuild();
                ctx.doc_mutated = true;
            }
        }
        PanelAction::PaintMaskStamp {
            mask_id,
            stamp,
            strength,
            hardness,
            tool,
        } => {
            let terra_core::AuthoringBrushStamp::Bounded {
                uv,
                radius_uv: radius,
                ..
            } = stamp
            else {
                app.ui_state.status =
                    "Sparse mask storage is not available until issue #209.".into();
                return Ok(());
            };
            let (u, v) = uv.tuple();
            // Snapshot paint buffer once per stroke for undo.
            if app.mask_paint_stroke_before.is_none() {
                if let Some(asset) = app.session.document.masks.iter().find(|a| a.id == mask_id) {
                    if let Some(paint) = asset.paint.as_ref() {
                        app.mask_paint_stroke_before =
                            Some((mask_id, paint.samples.clone(), paint.width, paint.height));
                    } else {
                        app.mask_paint_stroke_before = Some((mask_id, Vec::new(), 512, 512));
                    }
                }
            }
            if let Some(asset) = app
                .session
                .document
                .masks
                .iter_mut()
                .find(|asset| asset.id == mask_id)
            {
                let paint = asset
                    .paint
                    .get_or_insert_with(|| terra_core::mask::PaintBuffer::new(512, 512));
                paint.edit_circle(u, v, radius, strength, hardness, tool);
                // Critical: update paint buffer + overlay only — no full terrain
                // rebuild per brush dab (was the main mask-paint lag cause).
                app.mask_overlay_dirty = true;
                // Defer document-dirty marking to stroke commit.
            }
        }
        PanelAction::EditMask { mask_id, action } => {
            if let Some(asset) = app
                .session
                .document
                .masks
                .iter_mut()
                .find(|asset| asset.id == mask_id)
            {
                let paint = asset
                    .paint
                    .get_or_insert_with(|| terra_core::mask::PaintBuffer::new(512, 512));
                match action {
                    MaskEditAction::Clear => paint.clear(),
                    MaskEditAction::Fill => paint.fill(),
                    MaskEditAction::FlipX => paint.flip_x(),
                    MaskEditAction::FlipY => paint.flip_y(),
                    MaskEditAction::RotateLeft => paint.rotate_left(),
                    MaskEditAction::RotateRight => paint.rotate_right(),
                }
                ctx.mask_assets_mutated = true;
                ctx.doc_mutated = true;
                app.preview_dirty = true;
                app.mask_overlay_dirty = true;
            }
        }
        PanelAction::PaintSculptStamp {
            layer,
            stamp,
            strength,
            stroke_kind,
            target_height,
        } => {
            use terra_core::layer::{BrushDab, BrushEditable, EditSupport};
            let falloff = app.ui_state.sculpt_falloff_exponent();
            match stamp {
                terra_core::AuthoringBrushStamp::Bounded {
                    uv,
                    radius_uv: radius,
                    ..
                } => {
                    let (u, v) = uv.tuple();
                    let Some(metrics) = app
                        .session
                        .document
                        .bounded_settings()
                        .map(|settings| settings.metrics)
                    else {
                        app.ui_state.status =
                            "Bounded sculpt stamp used in an Infinite project".into();
                        return Ok(());
                    };
                    let continuing = matches!(
                        app.last_paint_point,
                        Some(terra_core::AuthoringPoint::Bounded { .. })
                    );
                    let world_radius = radius * 0.5 * (metrics.world_size_x + metrics.world_size_z);
                    if let Some(target) = app.session.document.stack.find_mut(layer) {
                        let support = target.brush_support(stroke_kind);
                        if support == EditSupport::Unsupported {
                            app.ui_state.status = if target.kind.is_sculpt_base() {
                                format!(
                                    "{} isn't supported on the Foundation layer — use a Shape Layer",
                                    stroke_kind.label()
                                )
                            } else {
                                format!(
                                    "{} isn't supported on layer \"{}\"",
                                    stroke_kind.label(),
                                    target.common.name
                                )
                            };
                        } else {
                            let is_shape_history =
                                terra_core::shape_history::is_shape_history_layer(&target.kind);
                            target.apply_brush(
                                stroke_kind,
                                BrushDab {
                                    u,
                                    v,
                                    radius_uv: radius,
                                    radius_m: world_radius,
                                    strength,
                                    target_height,
                                    falloff,
                                    continuing,
                                },
                            );
                            ctx.dirty_from = Some(layer);
                            app.session.document.selected = Some(layer);
                            if is_shape_history {
                                app.ui_state.shape_session_layer = Some(layer);
                            }
                        }
                    }
                    if ctx.dirty_from == Some(layer) {
                        let mut stamp_uv =
                            terra_core::tiling::UvRect::from_center_radius(u, v, radius);
                        if let Some(terra_core::AuthoringPoint::Bounded { uv, .. }) =
                            app.last_paint_point
                        {
                            let (pu, pv) = uv.tuple();
                            stamp_uv = stamp_uv.union(
                                terra_core::tiling::UvRect::from_center_radius(pu, pv, radius),
                            );
                        }
                        ctx.sculpt_dirty_region_uv = Some(match ctx.sculpt_dirty_region_uv {
                            Some(existing) => existing.union(stamp_uv),
                            None => stamp_uv,
                        });
                    }
                }
                terra_core::AuthoringBrushStamp::Infinite {
                    world, radius_m, ..
                } => {
                    if app.session.document.infinite_settings().is_none() {
                        app.ui_state.status = "World sculpt stamp used in a bounded project".into();
                        return Ok(());
                    }
                    let continuing = matches!(
                        app.last_paint_point,
                        Some(terra_core::AuthoringPoint::Infinite { .. })
                    );
                    let change = app
                        .session
                        .document
                        .stack
                        .find_mut(layer)
                        .and_then(|target| match &mut target.kind {
                            terra_core::layer::LayerKind::SculptStrokes(params)
                                if params.strokes.is_empty() =>
                            {
                                Some(params.stamp_world_stroke(
                                    stroke_kind,
                                    world,
                                    radius_m,
                                    strength,
                                    target_height,
                                    falloff,
                                    continuing,
                                ))
                            }
                            _ => None,
                        });
                    let Some(change) = change else {
                        app.ui_state.status =
                            "Infinite sculpting requires a world-space Shape Layer".into();
                        return Ok(());
                    };
                    let change = match change {
                        Ok(change) => change,
                        Err(error) => {
                            app.ui_state.status = format!("Invalid world sculpt stroke: {error}");
                            return Ok(());
                        }
                    };
                    if let Err(error) = app.apply_sculpt_bounds_change(change) {
                        app.ui_state.status =
                            format!("Could not index world sculpt stroke: {error}");
                        let _ = app.rebuild_authored_feature_index();
                        return Ok(());
                    }
                    ctx.dirty_from = Some(layer);
                    ctx.doc_mutated = true;
                    app.session.document.selected = Some(layer);
                    app.ui_state.shape_session_layer = Some(layer);
                }
            }
        }
        PanelAction::AddDistNode { target, kind } => {
            let node = terra_core::mask::DistNode::new(kind);
            if let Some(g) = app.session.document.stack.find_group_mut(target) {
                g.masks.push_node(node);
                TerraApp::mark_biome_placement_custom_for_group(&mut app.session.document, target);
                app.mark_all_layers_dirty();
                app.request_rebuild();
                ctx.doc_mutated = true;
            } else if let Some(l) = app.session.document.stack.find_mut(target) {
                l.common.masks.push_node(node);
                ctx.dirty_from = Some(target);
                ctx.doc_mutated = true;
            }
        }
        PanelAction::AddDistEffect { target, kind } => {
            let mut applied = false;
            if let Some(g) = app.session.document.stack.find_group_mut(target) {
                let effect = terra_core::mask::DistNode::new(kind.clone());
                if let Some(last) = g.masks.nodes.last_mut() {
                    last.children.push(effect);
                } else {
                    g.masks.push_node(effect);
                }
                TerraApp::mark_biome_placement_custom_for_group(&mut app.session.document, target);
                applied = true;
                app.mark_all_layers_dirty();
                app.request_rebuild();
                ctx.doc_mutated = true;
            } else if let Some(l) = app.session.document.stack.find_mut(target) {
                let effect = terra_core::mask::DistNode::new(kind);
                if let Some(last) = l.common.masks.nodes.last_mut() {
                    last.children.push(effect);
                } else {
                    l.common.masks.push_node(effect);
                }
                applied = true;
                ctx.dirty_from = Some(target);
                ctx.doc_mutated = true;
            }
            let _ = applied;
        }
        other => return Err(other),
    };
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::super::TerraApp;
    use crate::ui::PanelAction;
    use terra_core::authoring::SculptStrokeKind;
    use terra_core::layer::LayerStack;
    use terra_core::shape_history::create_shape_layer;

    fn bounded_stamp(u: f32, v: f32, radius: f32) -> terra_core::AuthoringBrushStamp {
        terra_core::AuthoringBrushStamp::bounded(
            terra_core::BoundedUv::try_new(u, v).unwrap(),
            radius,
            0.0,
        )
        .unwrap()
    }

    fn bounded_point(u: f32, v: f32) -> terra_core::AuthoringPoint {
        terra_core::AuthoringPoint::bounded(terra_core::BoundedUv::try_new(u, v).unwrap(), 0.0)
            .unwrap()
    }

    fn infinite_stamp(x_m: f64, z_m: f64, radius_m: f64) -> terra_core::AuthoringBrushStamp {
        terra_core::AuthoringBrushStamp::infinite(
            terra_world::WorldPosition::try_new(x_m, z_m).unwrap(),
            radius_m,
            0.0,
        )
        .unwrap()
    }

    fn infinite_point(x_m: f64, z_m: f64) -> terra_core::AuthoringPoint {
        terra_core::AuthoringPoint::infinite(
            terra_world::WorldPosition::try_new(x_m, z_m).unwrap(),
            0.0,
        )
        .unwrap()
    }

    fn infinite_app_with_shape() -> (TerraApp, terra_core::layer::LayerId) {
        let mut app = TerraApp::default();
        let mut document = terra_core::document::TerrainDocument::new_infinite(
            terra_core::document::InfiniteProceduralWorldSettings::default(),
        )
        .unwrap();
        let layer = create_shape_layer("World Shape");
        let id = layer.id();
        document.stack.push_into_category(layer);
        document.selected = Some(id);
        let world = document.world.clone();
        app.session.document = document;
        app.reset_runtime_for_document(&world, None);
        app.rebuild_authored_feature_index().unwrap();
        (app, id)
    }

    #[test]
    fn infinite_sculpt_stamps_store_world_records_and_update_sparse_index() {
        let (mut app, id) = infinite_app_with_shape();
        app.apply_actions(vec![PanelAction::PaintSculptStamp {
            layer: id,
            stamp: infinite_stamp(-300.25, -5.0, 20.0),
            strength: 7.0,
            stroke_kind: SculptStrokeKind::Raise,
            target_height: 0.0,
        }]);
        let first_id = match &app.session.document.stack.find(id).unwrap().kind {
            terra_core::layer::LayerKind::SculptStrokes(params) => {
                assert!(params.strokes.is_empty());
                assert_eq!(params.world_strokes.len(), 1);
                params.world_strokes[0].id
            }
            other => panic!("expected SculptStrokes, got {other:?}"),
        };
        app.last_paint_point = Some(infinite_point(-300.25, -5.0));
        app.apply_actions(vec![PanelAction::PaintSculptStamp {
            layer: id,
            stamp: infinite_stamp(300.75, 8.0, 20.0),
            strength: 9.0,
            stroke_kind: SculptStrokeKind::Raise,
            target_height: 0.0,
        }]);
        let params = match &app.session.document.stack.find(id).unwrap().kind {
            terra_core::layer::LayerKind::SculptStrokes(params) => params,
            other => panic!("expected SculptStrokes, got {other:?}"),
        };
        assert_eq!(params.world_strokes.len(), 1);
        assert_eq!(params.world_strokes[0].id, first_id);
        assert_eq!(params.world_strokes[0].points.len(), 2);
        let index = app.authored_feature_index.as_ref().unwrap();
        assert_eq!(index.len(), 1);
        let bounds = index.bounds(first_id).unwrap();
        assert!(bounds.min().x_m() < 0.0 && bounds.max().x_m() > 0.0);
        let negative = terra_world::TileAddress::new(
            terra_world::Lod::FINEST,
            terra_world::TileCoord { x: -2, z: -1 },
        );
        let positive = terra_world::TileAddress::new(
            terra_world::Lod::FINEST,
            terra_world::TileCoord { x: 1, z: 0 },
        );
        assert_eq!(index.query_tile(negative).unwrap(), vec![first_id]);
        assert_eq!(index.query_tile(positive).unwrap(), vec![first_id]);
    }

    #[test]
    fn infinite_sculpt_edit_actions_leave_no_stale_index_entry() {
        let (mut app, layer) = infinite_app_with_shape();
        app.apply_actions(vec![PanelAction::PaintSculptStamp {
            layer,
            stamp: infinite_stamp(-40.0, 10.0, 5.0),
            strength: 4.0,
            stroke_kind: SculptStrokeKind::Flatten,
            target_height: 125.0,
        }]);
        let stroke = match &app.session.document.stack.find(layer).unwrap().kind {
            terra_core::layer::LayerKind::SculptStrokes(params) => params.world_strokes[0].id,
            _ => unreachable!(),
        };
        let old = app
            .authored_feature_index
            .as_ref()
            .unwrap()
            .bounds(stroke)
            .unwrap();
        app.apply_actions(vec![PanelAction::MoveWorldStroke {
            layer,
            stroke,
            dx_m: 100.0,
            dz_m: -25.0,
        }]);
        let moved = app
            .authored_feature_index
            .as_ref()
            .unwrap()
            .bounds(stroke)
            .unwrap();
        assert_eq!(moved.min().x_m(), old.min().x_m() + 100.0);
        assert_eq!(moved.min().z_m(), old.min().z_m() - 25.0);

        app.apply_actions(vec![PanelAction::SetWorldStrokeEnabled {
            layer,
            stroke,
            enabled: false,
        }]);
        assert!(app
            .authored_feature_index
            .as_ref()
            .unwrap()
            .bounds(stroke)
            .is_none());
        app.apply_actions(vec![PanelAction::SetWorldStrokeEnabled {
            layer,
            stroke,
            enabled: true,
        }]);
        assert!(app
            .authored_feature_index
            .as_ref()
            .unwrap()
            .bounds(stroke)
            .is_some());
        app.apply_actions(vec![PanelAction::DeleteWorldStroke { layer, stroke }]);
        assert!(app
            .authored_feature_index
            .as_ref()
            .unwrap()
            .bounds(stroke)
            .is_none());
        app.undo();
        assert_eq!(
            app.authored_feature_index.as_ref().unwrap().bounds(stroke),
            Some(moved),
            "undo restores the same stable feature identity and bounds"
        );
        app.redo();
        assert!(app
            .authored_feature_index
            .as_ref()
            .unwrap()
            .bounds(stroke)
            .is_none());
    }

    #[test]
    fn infinite_sculpt_does_not_mix_legacy_bounded_points_into_world_history() {
        let (mut app, legacy_id) = infinite_app_with_shape();
        let terra_core::layer::LayerKind::SculptStrokes(params) =
            &mut app.session.document.stack.find_mut(legacy_id).unwrap().kind
        else {
            unreachable!()
        };
        params
            .strokes
            .push(terra_core::authoring::SculptStroke::default());
        app.session.document.selected = Some(legacy_id);

        let world_id = app
            .ensure_shape_history_target(terra_core::shape_history::ShapeTool::Raise)
            .expect("coordinate mismatch creates a world history");
        assert_ne!(world_id, legacy_id);
        let terra_core::layer::LayerKind::SculptStrokes(world) =
            &app.session.document.stack.find(world_id).unwrap().kind
        else {
            unreachable!()
        };
        assert!(world.strokes.is_empty());
        assert!(world.world_strokes.is_empty());
    }

    #[test]
    fn infinite_action_gate_rejects_bounded_painted_masks() {
        let mut app = TerraApp::default();
        app.session.document = terra_core::document::TerrainDocument::new_infinite(
            terra_core::document::InfiniteProceduralWorldSettings::default(),
        )
        .unwrap();
        let mask = terra_core::mask::MaskAsset::new_painted(
            terra_core::mask::MaskId::new(),
            "Painted",
            32,
        );

        app.apply_actions(vec![PanelAction::AddMask(mask)]);

        assert!(app.session.document.masks.is_empty());
        assert!(app
            .ui_state
            .status
            .contains("unavailable in Infinite projects"));
    }

    /// A sculpt stamp retains its UV footprint through the full apply path
    /// (`masks::try_apply` → `actions::mod` dispatch → `track_worker_dirty_from`),
    /// so the CPU worker receives a bounded `worker_dirty_region` covering the stamp
    /// rather than a whole-field escalation.
    #[test]
    fn sculpt_stamp_populates_worker_dirty_region() {
        let mut app = TerraApp::default();
        let layer = create_shape_layer("Shape");
        let id = layer.id();
        app.session.document.stack = LayerStack::new();
        app.session.document.stack.push(layer);
        app.session.document.selected = Some(id);

        // Steady state right after a completed job: nothing pending.
        app.worker_mark_all_dirty = false;
        app.worker_dirty_from = None;
        app.worker_dirty_region = None;
        app.last_paint_point = None; // fresh stroke, no previous point

        let (u, v, radius) = (0.5f32, 0.5f32, 0.05f32);
        app.apply_actions(vec![PanelAction::PaintSculptStamp {
            layer: id,
            stamp: bounded_stamp(u, v, radius),
            strength: 1.0,
            stroke_kind: SculptStrokeKind::Raise,
            target_height: 0.0,
        }]);

        let region = app
            .worker_dirty_region
            .expect("a sculpt stamp seeds a bounded worker scope");
        // A fresh single stamp's footprint is exactly the stamp's clamped UV box.
        let expected = terra_core::tiling::UvRect::from_center_radius(u, v, radius);
        assert_eq!(region, expected);
        assert_eq!(app.worker_dirty_from, Some(id));
        assert!(
            !app.worker_mark_all_dirty,
            "a bounded sculpt edit must not escalate to whole-field"
        );
    }

    /// The brush strength and edge-falloff must land on the painted stroke: falloff
    /// has no other UI path onto the stroke IR, and strength must stay live across a
    /// continuing drag (not freeze at the first dab). Regression for the "falloff /
    /// strength don't affect raise strokes" report.
    #[test]
    fn brush_strength_and_falloff_reach_the_painted_stroke() {
        use terra_core::layer::LayerKind;

        let mut app = TerraApp::default();
        let layer = create_shape_layer("Shape");
        let id = layer.id();
        app.session.document.stack = LayerStack::new();
        app.session.document.stack.push(layer);
        app.session.document.selected = Some(id);
        app.last_paint_point = None;

        // Soft brush, first dab: creates the stroke.
        app.ui_state.brush_falloff = 0.2;
        let soft_falloff = app.ui_state.sculpt_falloff_exponent();
        app.apply_actions(vec![PanelAction::PaintSculptStamp {
            layer: id,
            stamp: bounded_stamp(0.5, 0.5, 0.05),
            strength: 7.0,
            stroke_kind: SculptStrokeKind::Raise,
            target_height: 0.0,
        }]);
        let stroke = |app: &TerraApp| match &app.session.document.stack.find(id).unwrap().kind {
            LayerKind::SculptStrokes(p) => p.strokes.last().unwrap().clone(),
            other => panic!("expected SculptStrokes, got {other:?}"),
        };
        let first = stroke(&app);
        assert_eq!(first.strength, 7.0);
        assert!((first.falloff - soft_falloff).abs() < 1e-6);

        // Continue the same drag with a harder brush and higher strength: the active
        // stroke must pick up both, not stay frozen at the first dab's values.
        app.last_paint_point = Some(bounded_point(0.5, 0.5));
        app.ui_state.brush_falloff = 0.9;
        let hard_falloff = app.ui_state.sculpt_falloff_exponent();
        assert!(
            hard_falloff > soft_falloff,
            "harder brush => larger exponent"
        );
        app.apply_actions(vec![PanelAction::PaintSculptStamp {
            layer: id,
            stamp: bounded_stamp(0.52, 0.5, 0.05),
            strength: 30.0,
            stroke_kind: SculptStrokeKind::Raise,
            target_height: 0.0,
        }]);
        let updated = stroke(&app);
        assert_eq!(updated.points.len(), 2, "same stroke, appended point");
        assert_eq!(updated.strength, 30.0, "strength stays live mid-drag");
        assert!(
            (updated.falloff - hard_falloff).abs() < 1e-6,
            "falloff stays live"
        );
    }

    /// #97 regression: a brush the legacy foundation raster can't represent
    /// (Terrace here) must never silently raise the terrain. Before the fix it
    /// fell through the mode `match` to `_ => 0` (Raise) and edited the heights.
    #[test]
    fn unsupported_foundation_brush_does_not_silently_raise() {
        use terra_core::layer::{Layer, LayerKind, SculptParams};

        let mut app = TerraApp::default();
        let base = Layer::new(
            "Base",
            LayerKind::SculptBase(SculptParams::filled(64, 20.0)),
        );
        let id = base.id();
        app.session.document.stack = LayerStack::new();
        app.session.document.stack.push(base);
        app.session.document.selected = Some(id);

        let samples = |app: &TerraApp| match &app.session.document.stack.find(id).unwrap().kind {
            LayerKind::SculptBase(p) => p.samples.clone(),
            other => panic!("expected SculptBase, got {other:?}"),
        };
        let before = samples(&app);

        // Reset dirty tracking so the refusal can be shown to invalidate nothing.
        app.worker_mark_all_dirty = false;
        app.worker_dirty_from = None;
        app.worker_dirty_region = None;
        app.last_paint_point = None;

        // Terrace has no foundation mode — it must be refused, not raised.
        app.apply_actions(vec![PanelAction::PaintSculptStamp {
            layer: id,
            stamp: bounded_stamp(0.5, 0.5, 0.2),
            strength: 30.0,
            stroke_kind: SculptStrokeKind::Terrace,
            target_height: 0.0,
        }]);

        assert_eq!(
            samples(&app),
            before,
            "an unsupported foundation brush must not modify heights"
        );
        assert!(
            app.worker_dirty_region.is_none() && app.worker_dirty_from.is_none(),
            "a refused stroke invalidates nothing"
        );
        assert!(!app.worker_mark_all_dirty);
        assert!(
            app.ui_state.status.contains("Terrace") && app.ui_state.status.contains("Foundation"),
            "the artist must be told why the brush did nothing: {:?}",
            app.ui_state.status
        );

        // Positive control: a supported brush (Lower) still edits the foundation.
        app.apply_actions(vec![PanelAction::PaintSculptStamp {
            layer: id,
            stamp: bounded_stamp(0.5, 0.5, 0.2),
            strength: 30.0,
            stroke_kind: SculptStrokeKind::Lower,
            target_height: 0.0,
        }]);
        assert!(
            samples(&app).iter().any(|&s| s < 20.0),
            "Lower is supported on the foundation and must lower heights"
        );
        assert_eq!(app.session.document.selected, Some(id));
    }

    #[test]
    fn approximate_constraint_brush_uses_explicit_roughness_mapping() {
        use terra_core::layer::{Layer, LayerKind, TerrainConstraintKind, TerrainConstraintParams};

        let mut app = TerraApp::default();
        let constraints = Layer::new(
            "Constraints",
            LayerKind::TerrainConstraints(TerrainConstraintParams::default()),
        );
        let id = constraints.id();
        app.session.document.stack = LayerStack::new();
        app.session.document.stack.push(constraints);
        app.session.document.selected = Some(id);
        app.worker_mark_all_dirty = false;
        app.worker_dirty_from = None;
        app.worker_dirty_region = None;

        app.apply_actions(vec![PanelAction::PaintSculptStamp {
            layer: id,
            stamp: bounded_stamp(0.5, 0.5, 0.05),
            strength: 0.7,
            stroke_kind: SculptStrokeKind::Hardness,
            target_height: 0.0,
        }]);

        let params = match &app.session.document.stack.find(id).unwrap().kind {
            LayerKind::TerrainConstraints(params) => params,
            other => panic!("expected TerrainConstraints, got {other:?}"),
        };
        assert_eq!(params.constraints.len(), 1);
        assert_eq!(params.constraints[0].kind, TerrainConstraintKind::Roughness);
        assert_eq!(app.worker_dirty_from, Some(id));
    }

    #[test]
    fn unsupported_constraint_brush_creates_nothing() {
        use terra_core::layer::{Layer, LayerKind, TerrainConstraintParams};

        let mut app = TerraApp::default();
        let constraints = Layer::new(
            "Constraints",
            LayerKind::TerrainConstraints(TerrainConstraintParams::default()),
        );
        let id = constraints.id();
        app.session.document.stack = LayerStack::new();
        app.session.document.stack.push(constraints);
        app.session.document.selected = Some(id);
        app.worker_mark_all_dirty = false;
        app.worker_dirty_from = None;
        app.worker_dirty_region = None;

        app.apply_actions(vec![PanelAction::PaintSculptStamp {
            layer: id,
            stamp: bounded_stamp(0.5, 0.5, 0.05),
            strength: 1.0,
            stroke_kind: SculptStrokeKind::Raise,
            target_height: 0.0,
        }]);

        let params = match &app.session.document.stack.find(id).unwrap().kind {
            LayerKind::TerrainConstraints(params) => params,
            other => panic!("expected TerrainConstraints, got {other:?}"),
        };
        assert!(params.constraints.is_empty());
        assert!(app.worker_dirty_region.is_none() && app.worker_dirty_from.is_none());
        assert!(!app.worker_mark_all_dirty);
        assert!(
            app.ui_state.status.contains("Raise") && app.ui_state.status.contains("Constraints"),
            "unsupported constraint brush should explain its refusal: {:?}",
            app.ui_state.status
        );
    }
}
