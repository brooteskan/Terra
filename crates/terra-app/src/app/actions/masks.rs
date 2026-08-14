use crate::ui::{MaskEditAction, PanelAction};

use super::super::TerraApp;
use super::ApplyCtx;

pub(crate) fn try_apply(
    app: &mut TerraApp,
    action: PanelAction,
    ctx: &mut ApplyCtx,
) -> Result<(), PanelAction> {
    let result = match action {
        PanelAction::AddMask(mut asset) => {
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
            u,
            v,
            radius,
            strength,
            hardness,
            tool,
        } => {
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
            u,
            v,
            radius,
            strength,
            stroke_kind,
            target_height,
        } => {
            if let Some(target) = app.session.document.stack.find_mut(layer) {
                let continuing = app.last_paint_uv.is_some();
                let world_radius = radius
                    * 0.5
                    * (app.session.document.metrics.world_size_x
                        + app.session.document.metrics.world_size_z);
                if let terra_core::layer::LayerKind::SculptStrokes(params) = &mut target.kind {
                    terra_core::shape_history::stamp_stroke(
                        params,
                        stroke_kind,
                        u,
                        v,
                        world_radius.max(1.0),
                        strength,
                        target_height,
                        continuing,
                    );
                    ctx.dirty_from = Some(layer);
                    app.session.document.selected = Some(layer);
                    app.ui_state.shape_session_layer = Some(layer);
                } else if let terra_core::layer::LayerKind::TerrainConstraints(params) =
                    &mut target.kind
                {
                    use terra_core::layer::{
                        SculptPoint, TerrainConstraint, TerrainConstraintKind,
                    };
                    let constraint_kind = match stroke_kind {
                        terra_core::authoring::SculptStrokeKind::Ridge
                        | terra_core::authoring::SculptStrokeKind::MountainStamp => {
                            TerrainConstraintKind::Ridge
                        }
                        terra_core::authoring::SculptStrokeKind::Valley
                        | terra_core::authoring::SculptStrokeKind::ValleyStamp => {
                            TerrainConstraintKind::Valley
                        }
                        terra_core::authoring::SculptStrokeKind::RiverPath => {
                            TerrainConstraintKind::River
                        }
                        terra_core::authoring::SculptStrokeKind::Coastline => {
                            TerrainConstraintKind::Coastline
                        }
                        terra_core::authoring::SculptStrokeKind::Protect => {
                            TerrainConstraintKind::Protect
                        }
                        terra_core::authoring::SculptStrokeKind::PlateauStamp => {
                            TerrainConstraintKind::Plateau
                        }
                        _ => TerrainConstraintKind::Roughness,
                    };
                    let point = SculptPoint {
                        u,
                        v,
                        pressure: 1.0,
                    };
                    let append = continuing
                        && params.constraints.last().is_some_and(|last| {
                            last.kind == constraint_kind
                                && (last.width_m - world_radius).abs()
                                    <= world_radius.max(1.0) * 0.05
                        });
                    if append {
                        params.constraints.last_mut().unwrap().points.push(point);
                    } else {
                        params.constraints.push(TerrainConstraint {
                            kind: constraint_kind,
                            points: vec![point],
                            width_m: world_radius.max(1.0),
                            value: strength,
                            strength: if matches!(constraint_kind, TerrainConstraintKind::Protect) {
                                strength.clamp(0.0, 1.0)
                            } else {
                                1.0
                            },
                        });
                    }
                    ctx.dirty_from = Some(layer);
                    app.session.document.selected = Some(layer);
                } else if let terra_core::layer::LayerKind::SculptBase(params) = &mut target.kind {
                    // Legacy foundation path â€” prefer Shape history for new strokes.
                    let mode = match stroke_kind {
                        terra_core::authoring::SculptStrokeKind::Lower
                        | terra_core::authoring::SculptStrokeKind::Erode => 1u8,
                        terra_core::authoring::SculptStrokeKind::Smooth
                        | terra_core::authoring::SculptStrokeKind::Pinch => 2u8,
                        _ => 0u8,
                    };
                    params.stamp_circle(u, v, radius, strength, mode);
                    ctx.dirty_from = Some(layer);
                    app.session.document.selected = Some(layer);
                }
            }
            if ctx.dirty_from == Some(layer) {
                let resolution = app
                    .scheduler
                    .quality
                    .resolution(
                        app.session.document.preview_resolution.min(8192),
                        app.session.document.export_resolution,
                    )
                    .max(1);
                let x0 = ((u - radius).clamp(0.0, 1.0) * resolution as f32).floor() as u32;
                let y0 = ((v - radius).clamp(0.0, 1.0) * resolution as f32).floor() as u32;
                let x1 = ((u + radius).clamp(0.0, 1.0) * resolution as f32).ceil() as u32;
                let y1 = ((v + radius).clamp(0.0, 1.0) * resolution as f32).ceil() as u32;
                let next = (
                    x0,
                    y0,
                    x1.saturating_sub(x0).max(1),
                    y1.saturating_sub(y0).max(1),
                );
                ctx.sculpt_dirty_rect = Some(match ctx.sculpt_dirty_rect {
                    Some((ox, oy, ow, oh)) => {
                        let ex = (ox + ow).max(next.0 + next.2);
                        let ey = (oy + oh).max(next.1 + next.3);
                        let nx = ox.min(next.0);
                        let ny = oy.min(next.1);
                        (nx, ny, ex - nx, ey - ny)
                    }
                    None => next,
                });
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
    let _ = result;
    Ok(())
}
