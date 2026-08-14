use crate::ui::{layers_from_preset, PanelAction};
use terra_core::command::{apply, EditorCommand};
use terra_core::layer::LayerKind;

use super::super::helpers::{coalesce_layer_id, randomize_layer_seed};
use super::super::{
    apply_blueprint_to_stack, ensure_hydrology_processes, ensure_surface_processes, TerraApp,
};
use super::ApplyCtx;

pub(crate) fn try_apply(
    app: &mut TerraApp,
    action: PanelAction,
    ctx: &mut ApplyCtx,
) -> Result<(), PanelAction> {
    let result = match action {
        PanelAction::AddLayer(layer) => {
            let kind = layer.kind.clone();
            if layer.kind.is_sculpt_base() {
                // Only one sculpt Base allowed.
                {
                    ctx.continue_loop = true;
                    return Ok(());
                }
            }
            let id = layer.id();
            // Prefer explicit Quick Add target (biome / Filters / Materials…).
            if let Some(parent) = app.ui_state.quick_add_into {
                if let Some(group) = app.session.document.stack.find_group_mut(parent) {
                    if group.is_biome() {
                        group.push_into_section(layer);
                    } else {
                        group
                            .children
                            .push(terra_core::layer::StackNode::Layer(layer));
                    }
                } else {
                    app.session
                        .document
                        .stack
                        .push_routed(layer, Some(parent), false);
                }
            } else {
                let ctx = app
                    .session
                    .document
                    .selected
                    .or(app.session.document.active_biome);
                app.session.document.stack.ensure_category_folders();
                app.session.document.stack.push_routed(layer, ctx, false);
            }
            // Track active biome when adding into one.
            if let Some(b) = app.session.document.stack.enclosing_biome(id) {
                app.session.document.active_biome = Some(b.id);
            }
            app.session.document.selected = Some(id);
            app.ui_state.focus_view_for_new_layer(&kind);
            ctx.dirty_from = Some(id);
            ctx.doc_mutated = true;
        }
        PanelAction::AddLayerToCategory { category, layer } => {
            if layer.kind.is_sculpt_base() {
                {
                    ctx.continue_loop = true;
                    return Ok(());
                }
            }
            let kind = layer.kind.clone();
            let id = layer.id();
            app.session.document.stack.ensure_category_folders();
            if let Some(folder) = app.session.document.stack.find_category_mut(category) {
                folder
                    .children
                    .push(terra_core::layer::StackNode::Layer(layer));
            } else {
                app.session.document.stack.push_into_category(layer);
            }
            app.session.document.selected = Some(id);
            app.ui_state.focus_view_for_new_layer(&kind);
            ctx.dirty_from = Some(id);
            ctx.doc_mutated = true;
        }
        PanelAction::RemoveSelected => {
            if let Some(id) = app.session.document.selected {
                if app
                    .session
                    .document
                    .stack
                    .find(id)
                    .is_some_and(|l| l.kind.is_sculpt_base())
                {
                    {
                        ctx.continue_loop = true;
                        return Ok(());
                    }
                }
                let loc = app.session.document.stack.sibling_location(id);
                if let Some(node) = app.session.document.stack.remove(id) {
                    let (parent, index) = loc.unwrap_or((None, 0));
                    let cmd = EditorCommand::RemoveLayer {
                        id,
                        node,
                        index,
                        parent,
                    };
                    app.session.history.push_executed(cmd);
                    app.session.document.selected =
                        app.session.document.stack.layer_ids().last().copied();
                    app.mark_all_layers_dirty();
                    app.request_rebuild();
                    ctx.doc_mutated = true;
                }
            }
        }
        PanelAction::DuplicateSelected => {
            if let Some(id) = app.session.document.selected {
                if app
                    .session
                    .document
                    .stack
                    .find(id)
                    .is_some_and(|l| l.kind.is_sculpt_base())
                {
                    {
                        ctx.continue_loop = true;
                        return Ok(());
                    }
                }
                if let Some(new_id) = app.session.document.stack.duplicate(id) {
                    let cmd = EditorCommand::Duplicate { source: id, new_id };
                    app.session.history.push_executed(cmd);
                    app.session.document.selected = Some(new_id);
                    ctx.dirty_from = Some(new_id);
                    ctx.doc_mutated = true;
                }
            }
        }
        PanelAction::Reorder { from, to } => {
            let cmd = EditorCommand::Reorder { from, to };
            apply(&cmd, &mut app.session.document.stack);
            app.session.history.push_executed(cmd);
            app.mark_all_layers_dirty();
            app.request_rebuild();
            ctx.doc_mutated = true;
        }
        PanelAction::Select(id) => {
            app.session.document.selected = Some(id);
        }
        PanelAction::SetEnabled { id, enabled } => {
            let previous = app
                .session
                .document
                .stack
                .find(id)
                .map(|l| l.common.enabled)
                .or_else(|| app.session.document.stack.find_group(id).map(|g| g.enabled))
                .unwrap_or(true);
            let cmd = EditorCommand::SetEnabled {
                id,
                enabled,
                previous,
            };
            apply(&cmd, &mut app.session.document.stack);
            app.session.history.push_executed(cmd);
            ctx.doc_mutated = true;
            app.mark_all_layers_dirty();
            app.request_rebuild();
        }
        PanelAction::SetOpacity { id, opacity } => {
            let previous = app
                .session
                .document
                .stack
                .find(id)
                .map(|l| l.common.opacity)
                .or_else(|| app.session.document.stack.find_group(id).map(|g| g.opacity))
                .unwrap_or(1.0);
            let cmd = EditorCommand::SetOpacity {
                id,
                opacity,
                previous,
            };
            apply(&cmd, &mut app.session.document.stack);
            app.session
                .history
                .push_coalesced(cmd, Some((coalesce_layer_id(id), "opacity")));
            ctx.dirty_from = Some(id);
        }
        PanelAction::SetBlend { id, blend } => {
            let previous = app
                .session
                .document
                .stack
                .find(id)
                .map(|l| l.common.blend)
                .or_else(|| app.session.document.stack.find_group(id).map(|g| g.blend))
                .unwrap_or_default();
            let cmd = EditorCommand::SetBlend {
                id,
                blend,
                previous,
            };
            apply(&cmd, &mut app.session.document.stack);
            app.session.history.push_executed(cmd);
            ctx.dirty_from = Some(id);
        }
        PanelAction::SetKind { id, kind } => {
            let previous = app
                .session
                .document
                .stack
                .find(id)
                .map(|l| l.kind.clone())
                .unwrap_or(LayerKind::Flat(Default::default()));
            let cmd = EditorCommand::SetKind { id, kind, previous };
            apply(&cmd, &mut app.session.document.stack);
            app.session
                .history
                .push_coalesced(cmd, Some((coalesce_layer_id(id), "kind")));
            ctx.dirty_from = Some(id);
        }
        PanelAction::Rename { id, name } => {
            let name = name.trim().to_string();
            if name.is_empty() {
                {
                    ctx.continue_loop = true;
                    return Ok(());
                }
            }
            if let Some(layer) = app.session.document.stack.find(id) {
                if layer.common.locked {
                    {
                        ctx.continue_loop = true;
                        return Ok(());
                    }
                }
                let previous = layer.common.name.clone();
                if previous == name {
                    {
                        ctx.continue_loop = true;
                        return Ok(());
                    }
                }
                let cmd = EditorCommand::Rename { id, name, previous };
                apply(&cmd, &mut app.session.document.stack);
                app.session.history.push_executed(cmd);
                ctx.doc_mutated = true;
            } else if let Some(group) = app.session.document.stack.find_group(id) {
                // Fixed structure: Filters / Materials / Objects / category folders.
                if matches!(
                    group.group_kind,
                    terra_core::layer::GroupKind::BiomeSection(_)
                        | terra_core::layer::GroupKind::CategoryFolder
                ) {
                    {
                        ctx.continue_loop = true;
                        return Ok(());
                    }
                }
                let previous = group.name.clone();
                if previous == name {
                    {
                        ctx.continue_loop = true;
                        return Ok(());
                    }
                }
                let cmd = EditorCommand::Rename { id, name, previous };
                apply(&cmd, &mut app.session.document.stack);
                app.session.history.push_executed(cmd);
                ctx.doc_mutated = true;
            } else if let Some(mask) = app
                .session
                .document
                .masks
                .iter_mut()
                .find(|m| m.id.0 == id.0)
            {
                if mask.name != name {
                    mask.name = name;
                    ctx.doc_mutated = true;
                    app.preview_dirty = true;
                }
            } else if let Some(rule) = app
                .session
                .document
                .world_rules
                .rules
                .iter_mut()
                .find(|r| r.id.0 == id.0)
            {
                if rule.name != name {
                    rule.name = name;
                    ctx.doc_mutated = true;
                }
            } else if let Some(scenario) = app
                .session
                .document
                .simulation_scenarios
                .scenarios
                .iter_mut()
                .find(|s| s.id.0 == id.0)
            {
                if scenario.name != name {
                    scenario.name = name;
                    ctx.doc_mutated = true;
                }
            }
        }
        PanelAction::ApplyPreset(name) => {
            if let Some(layers) = layers_from_preset(&name) {
                let kept_name = app.session.document.name.clone();
                let kept_masks = app.session.document.masks.clone();
                let mut doc = terra_core::document::TerrainDocument::from_flat_layers(layers);
                doc.name = kept_name;
                doc.masks = kept_masks;
                doc.presets_used = app.session.document.presets_used.clone();
                doc.presets_used.push(name);
                app.session.document = doc;
                app.ui_state.editor_tool = crate::ui::EditorTool::Raise;
                app.ui_state.paint_mask = None;
                app.mark_all_layers_dirty();
                app.request_rebuild();
                ctx.doc_mutated = true;
            }
        }
        PanelAction::MergeShapeLayers { keep, others } => {
            let sources: Vec<terra_core::authoring::SculptStrokeParams> = others
                .iter()
                .filter_map(|id| {
                    app.session.document.stack.find(*id).and_then(|l| {
                        if let terra_core::layer::LayerKind::SculptStrokes(p) = &l.kind {
                            Some(p.clone())
                        } else {
                            None
                        }
                    })
                })
                .collect();
            if let Some(target) = app.session.document.stack.find_mut(keep) {
                if let terra_core::layer::LayerKind::SculptStrokes(params) = &mut target.kind {
                    let refs: Vec<&_> = sources.iter().collect();
                    terra_core::shape_history::merge_sculpt_stroke_layers(params, &refs);
                }
            }
            for id in &others {
                let _ = app.session.document.stack.remove(*id);
            }
            app.session.document.selected = Some(keep);
            ctx.dirty_from = Some(keep);
            ctx.doc_mutated = true;
        }
        PanelAction::MoveLayerToBiome { id, biome, section } => {
            if app
                .session
                .document
                .stack
                .move_layer_to_biome_section(id, biome, section)
            {
                app.session.document.active_biome = Some(biome);
                app.session.document.selected = Some(id);
                // Placement stays Entire Biome / local â€” inheritance is by enclosure.
                app.mark_all_layers_dirty();
                app.request_rebuild();
                ctx.doc_mutated = true;
            }
        }
        PanelAction::AddDevelopOperation { category, name } => {
            let biome_id = app.session.document.active_biome.or_else(|| {
                app.ui_state.biome_focus.and_then(|def_id| {
                    app.session
                        .document
                        .biome_library
                        .get(def_id)
                        .and_then(|d| d.group_id)
                })
            });
            let Some(biome_id) = biome_id else {
                app.ui_state.status = "Select a Biome before adding Develop operations".into();
                {
                    ctx.continue_loop = true;
                    return Ok(());
                }
            };
            let Some(section) = category.biome_section() else {
                // Placement category â†’ focus biome inspector.
                app.session.document.selected = Some(biome_id);
                {
                    ctx.continue_loop = true;
                    return Ok(());
                }
            };
            let layer = terra_core::operation_placement::create_develop_operation(category, name);
            let id = layer.id();
            if !app
                .session
                .document
                .stack
                .find_group(biome_id)
                .is_some_and(|g| g.is_biome())
            {
                app.ui_state.status = "Active Biome group missing".into();
                {
                    ctx.continue_loop = true;
                    return Ok(());
                }
            }
            // Ensure sections, then push into the right section.
            if let Some(biome) = app.session.document.stack.find_group_mut(biome_id) {
                biome.ensure_biome_sections();
                if let Some(sec) = biome.find_section_mut(section) {
                    sec.children
                        .push(terra_core::layer::StackNode::Layer(layer));
                } else {
                    biome
                        .children
                        .push(terra_core::layer::StackNode::Layer(layer));
                }
            }
            app.session.document.selected = Some(id);
            app.session.document.active_biome = Some(biome_id);
            app.mark_dirty_from_stage(id);
            ctx.dirty_from = Some(id);
            ctx.doc_mutated = true;
            app.ui_state.status = format!("Added {} under biome (Entire Biome)", category.label());
        }
        PanelAction::OpenLayerAdvancedMask(id) => {
            if let Some(l) = app.session.document.stack.find_mut(id) {
                let (previous, previous_masks) =
                    (l.common.operation_placement.clone(), l.common.masks.clone());
                l.common.operation_placement.apply_where =
                    terra_core::operation_placement::ApplyWhere::AdvancedMask;
                l.common.operation_placement.definition.source =
                    terra_core::mask::PlacementSource::Custom;
                if l.common
                    .operation_placement
                    .definition
                    .custom_stack
                    .is_none()
                {
                    l.common.operation_placement.definition.custom_stack =
                        Some(l.common.masks.clone());
                }
                let placement = l.common.operation_placement.clone();
                let cmd = EditorCommand::SetOperationPlacement {
                    id,
                    placement,
                    previous,
                    previous_masks,
                };
                // Already applied on layer; record for undo.
                app.session.history.push_executed(cmd);
                app.inspector_gui.details.layer_masks = true;
                app.ui_state.inspector_advanced = true;
                app.session.document.selected = Some(id);
                ctx.doc_mutated = true;
            }
        }
        PanelAction::SetShapeEditMode(mode) => {
            app.ui_state.shape_edit_mode = mode;
            app.ui_state.status = mode.label().into();
        }
        PanelAction::CommitShapePreview => {
            app.ui_state.shape_commit_full = true;
            app.request_rebuild();
        }
        PanelAction::ResetSculptBase { id } => {
            if let Some(target) = app.session.document.stack.find_mut(id) {
                if let terra_core::layer::LayerKind::SculptBase(params) = &mut target.kind {
                    params.reset();
                    ctx.dirty_from = Some(id);
                }
            }
        }
        PanelAction::MarkDirty(id) => {
            ctx.dirty_from = id.or(app.session.document.selected);
        }
        PanelAction::SetLocked { id, locked } => {
            let previous = app
                .session
                .document
                .stack
                .find(id)
                .map(|l| l.common.locked)
                .unwrap_or(false);
            let cmd = EditorCommand::SetLocked {
                id,
                locked,
                previous,
            };
            apply(&cmd, &mut app.session.document.stack);
            app.session.history.push_executed(cmd);
            ctx.doc_mutated = true;
        }
        PanelAction::SetSolo { id, solo } => {
            let previous = app
                .session
                .document
                .stack
                .find(id)
                .map(|l| l.common.solo)
                .unwrap_or(false);
            let cmd = EditorCommand::SetSolo { id, solo, previous };
            apply(&cmd, &mut app.session.document.stack);
            app.session.history.push_executed(cmd);
            ctx.dirty_from = Some(id);
        }
        PanelAction::SetColorTag { id, tag } => {
            let previous = app
                .session
                .document
                .stack
                .find(id)
                .map(|l| l.common.color_tag)
                .unwrap_or(0);
            let cmd = EditorCommand::SetColorTag { id, tag, previous };
            apply(&cmd, &mut app.session.document.stack);
            app.session.history.push_executed(cmd);
            ctx.doc_mutated = true;
        }
        PanelAction::SetCached { id, cached } => {
            let previous = app
                .session
                .document
                .stack
                .find(id)
                .map(|l| l.common.cached)
                .unwrap_or(false);
            let cmd = EditorCommand::SetCached {
                id,
                cached,
                previous,
            };
            apply(&cmd, &mut app.session.document.stack);
            app.session.history.push_executed(cmd);
            ctx.dirty_from = Some(id);
        }
        PanelAction::AddGroup { name } => {
            let id = terra_core::layer::LayerId::new();
            let index = app.session.document.stack.nodes.len();
            let cmd = EditorCommand::AddGroup { name, id, index };
            apply(&cmd, &mut app.session.document.stack);
            app.session.history.push_executed(cmd);
            app.session.document.selected = Some(id);
            ctx.doc_mutated = true;
        }
        PanelAction::AddIsolatedGroup { name } => {
            let group = terra_core::layer::LayerGroup::isolated(name);
            let id = group.id;
            app.session.document.stack.push_group(group);
            app.session.document.selected = Some(id);
            app.mark_all_layers_dirty();
            app.request_rebuild();
            ctx.doc_mutated = true;
        }
        PanelAction::SetEditorTool(tool) => {
            let prev = app.ui_state.editor_tool;
            app.ui_state.set_editor_tool(tool);
            if tool.is_move() {
                app.sculpt_stroke_active = false;
                app.last_paint_uv = None;
                app.dragging_shape_point = None;
                app.dragging_layer_point = None;
            }
            if tool != prev {
                // New tool â†’ new stroke session (unless ContinueSelected).
                if app.ui_state.shape_edit_mode
                    == terra_core::shape_history::ShapeEditMode::NewLayerPerSession
                {
                    app.ui_state.shape_session_layer = None;
                }
            }
            if tool == crate::ui::EditorTool::PaintBiome {
                app.ui_state.workspace_mode = crate::ui::WorkspaceMode::Biomes;
            }
            if tool == crate::ui::EditorTool::Measure {
                app.measure_anchor = None;
                app.ui_state.status = "Measure: click start point".into();
            } else if prev == crate::ui::EditorTool::Measure {
                app.measure_anchor = None;
            }
        }
        PanelAction::MoveIntoGroup { child, group } => {
            if app.session.document.stack.move_into_group(child, group) {
                app.mark_all_layers_dirty();
                app.request_rebuild();
                ctx.doc_mutated = true;
            }
        }
        PanelAction::MoveToRoot { id } => {
            if app.session.document.stack.move_to_root(id) {
                app.mark_all_layers_dirty();
                app.request_rebuild();
                ctx.doc_mutated = true;
            }
        }
        PanelAction::OpenQuickAdd => {
            app.ui_state.show_quick_add = true;
            app.ui_state.quick_add_category = None;
            app.ui_state.quick_add_concept = None;
            app.ui_state.quick_add_into = None;
            app.ui_state.quick_add_biome_section = None;
            app.ui_state.quick_add_distribution = None;
        }
        PanelAction::OpenQuickAddCategory { category } => {
            app.ui_state.show_quick_add = true;
            app.ui_state.quick_add_category = Some(category);
            app.ui_state.quick_add_concept = None;
            app.ui_state.quick_add_biome_section = None;
            app.ui_state.quick_add_distribution = None;
            // Biomes folder + → create biome; otherwise add into that folder.
            app.ui_state.quick_add_into = app.session.document.stack.category_id(category);
            if matches!(category, terra_core::layer::StackCategory::Surface) {
                // Prefer adding a new biome when targeting Biomes root.
                app.ui_state.quick_add_into = None;
            }
        }
        PanelAction::OpenQuickAddConcept { concept } => {
            app.ui_state.show_quick_add = true;
            app.ui_state.quick_add_concept = Some(concept);
            app.ui_state.quick_add_category = concept.stack_category();
            app.ui_state.quick_add_biome_section = None;
            app.ui_state.quick_add_distribution = None;
            // Masks / world rules are project assets — not nested into a stack folder.
            app.ui_state.quick_add_into = match concept {
                crate::ui::ArtistConcept::Masks
                | crate::ui::ArtistConcept::MaskStack
                | crate::ui::ArtistConcept::WorldRules
                | crate::ui::ArtistConcept::Biomes => None,
                _ => concept
                    .stack_category()
                    .and_then(|cat| app.session.document.stack.category_id(cat)),
            };
        }
        PanelAction::OpenQuickAddInto { parent } => {
            app.ui_state.show_quick_add = true;
            app.ui_state.quick_add_into = Some(parent);
            app.ui_state.quick_add_category = None;
            app.ui_state.quick_add_concept = None;
            app.ui_state.quick_add_distribution = None;
            app.ui_state.quick_add_biome_section = app
                .session
                .document
                .stack
                .find_group(parent)
                .and_then(|g| g.biome_section_kind());
            // Select the parent so routing context is consistent.
            app.session.document.selected = Some(parent);
            if app
                .session
                .document
                .stack
                .find_group(parent)
                .is_some_and(|g| g.is_biome())
            {
                app.session.document.active_biome = Some(parent);
            } else if let Some(b) = app.session.document.stack.enclosing_biome(parent) {
                app.session.document.active_biome = Some(b.id);
            }
        }
        PanelAction::OpenQuickAddDistribution { biome } => {
            app.ui_state.show_quick_add = true;
            app.ui_state.quick_add_distribution = Some(biome);
            app.ui_state.quick_add_category = None;
            app.ui_state.quick_add_concept = None;
            app.ui_state.quick_add_into = None;
            app.ui_state.quick_add_biome_section = None;
            app.session.document.selected = Some(biome);
            if app
                .session
                .document
                .stack
                .find_group(biome)
                .is_some_and(|g| g.is_biome())
            {
                app.session.document.active_biome = Some(biome);
            }
        }
        PanelAction::AddLayerInto { parent, layer } => {
            if layer.kind.is_sculpt_base() {
                {
                    ctx.continue_loop = true;
                    return Ok(());
                }
            }
            let kind = layer.kind.clone();
            let id = layer.id();
            if let Some(group) = app.session.document.stack.find_group(parent) {
                if group.is_biome() {
                    // Route filters→Filters, materials→Materials, etc.; shapes stay on Shape.
                    app.session
                        .document
                        .stack
                        .push_routed(layer, Some(parent), false);
                } else if group.biome_section_kind().is_some() {
                    // Dropped directly on Filters / Materials / Objects / Local Sims.
                    if let Some(g) = app.session.document.stack.find_group_mut(parent) {
                        g.children.push(terra_core::layer::StackNode::Layer(layer));
                    }
                } else if let Some(group) = app.session.document.stack.find_group_mut(parent) {
                    group
                        .children
                        .push(terra_core::layer::StackNode::Layer(layer));
                }
            } else {
                app.session
                    .document
                    .stack
                    .push_routed(layer, Some(parent), false);
            }
            app.session.document.selected = Some(id);
            if let Some(b) = app.session.document.stack.enclosing_biome(id) {
                app.session.document.active_biome = Some(b.id);
            } else if app
                .session
                .document
                .stack
                .find_group(parent)
                .is_some_and(|g| g.is_biome())
            {
                app.session.document.active_biome = Some(parent);
            }
            app.ui_state.focus_view_for_new_layer(&kind);
            ctx.dirty_from = Some(id);
            ctx.doc_mutated = true;
        }
        PanelAction::RandomizeSeed { id } => {
            if let Some(layer) = app.session.document.stack.find_mut(id) {
                randomize_layer_seed(&mut layer.kind);
                ctx.dirty_from = Some(id);
            }
        }
        PanelAction::SelectShape(id) => {
            if app.session.document.shapes.get(id).is_some() {
                app.session.document.shapes.selected = Some(id);
                app.session.document.selected = None;
                app.ui_state.app_workspace = crate::ui::AppWorkspace::Landforms;
                app.ui_state.workspace_mode =
                    crate::ui::AppWorkspace::Landforms.default_workspace_mode();
            }
        }
        PanelAction::CreateShape { kind, name } => {
            use terra_core::authoring::SculptPoint;
            use terra_core::shape_object::ShapeObject;
            let mut shape = ShapeObject::new(name, kind);
            shape.points = match kind {
                terra_core::shape_object::ShapeKind::CoastlinePolygon
                | terra_core::shape_object::ShapeKind::LandmassPolygon
                | terra_core::shape_object::ShapeKind::PlateauPolygon
                | terra_core::shape_object::ShapeKind::LakeBasin => vec![
                    SculptPoint {
                        u: 0.35,
                        v: 0.35,
                        pressure: 1.0,
                    },
                    SculptPoint {
                        u: 0.65,
                        v: 0.35,
                        pressure: 1.0,
                    },
                    SculptPoint {
                        u: 0.65,
                        v: 0.65,
                        pressure: 1.0,
                    },
                    SculptPoint {
                        u: 0.35,
                        v: 0.65,
                        pressure: 1.0,
                    },
                ],
                terra_core::shape_object::ShapeKind::MountainSpine
                | terra_core::shape_object::ShapeKind::RidgeSpline
                | terra_core::shape_object::ShapeKind::ValleySpline
                | terra_core::shape_object::ShapeKind::RiverPath
                | terra_core::shape_object::ShapeKind::CanyonPath => vec![
                    SculptPoint {
                        u: 0.3,
                        v: 0.5,
                        pressure: 1.0,
                    },
                    SculptPoint {
                        u: 0.5,
                        v: 0.48,
                        pressure: 1.0,
                    },
                    SculptPoint {
                        u: 0.7,
                        v: 0.5,
                        pressure: 1.0,
                    },
                ],
                terra_core::shape_object::ShapeKind::UpliftCentre
                | terra_core::shape_object::ShapeKind::Volcano
                | terra_core::shape_object::ShapeKind::HeightStamp => {
                    vec![SculptPoint {
                        u: 0.5,
                        v: 0.5,
                        pressure: 1.0,
                    }]
                }
            };
            if matches!(
                kind,
                terra_core::shape_object::ShapeKind::MountainSpine
                    | terra_core::shape_object::ShapeKind::RidgeSpline
            ) {
                shape.width_m = app.session.document.blueprint.ridge_width_m();
            }
            let id = app.session.document.shapes.push(shape);
            app.session.document.shapes.selected = Some(id);
            app.session.document.selected = None;
            app.session.document.compile_shapes_into_stack();
            if let Some(lid) = app.session.document.shapes.managed_constraints_layer {
                ctx.dirty_from = Some(lid);
            }
            app.ui_state.app_workspace = crate::ui::AppWorkspace::Landforms;
            app.ui_state.editor_tool = crate::ui::EditorTool::Move;
            app.ui_state.status = "Shape created â€” drag control points on terrain".into();
            ctx.doc_mutated = true;
        }
        PanelAction::TranslateShape { id, du, dv } => {
            if let Some(shape) = app.session.document.shapes.get_mut(id) {
                shape.translate_uv(du, dv);
                app.session.document.compile_shapes_into_stack();
                if let Some(lid) = app.session.document.shapes.managed_constraints_layer {
                    ctx.dirty_from = Some(lid);
                }
                ctx.doc_mutated = true;
            }
        }
        PanelAction::CompileShapes => {
            app.session.document.compile_shapes_into_stack();
            if let Some(lid) = app.session.document.shapes.managed_constraints_layer {
                ctx.dirty_from = Some(lid);
            }
            ctx.doc_mutated = true;
        }
        PanelAction::SetShapePoint { id, index, u, v } => {
            if let Some(shape) = app.session.document.shapes.get_mut(id) {
                if shape.set_point_world_uv(index, u, v) {
                    ctx.doc_mutated = true;
                }
            }
        }
        PanelAction::ApplyBlueprintSemantics {
            ridge_sharpness,
            sea_level,
            geological_age,
            rainfall,
            drainage_density,
        } => {
            app.session.document.blueprint.ridge_sharpness = ridge_sharpness.clamp(0.0, 1.0);
            app.session.document.blueprint.sea_level = sea_level;
            app.session.document.blueprint.geological_age = geological_age.clamp(0.0, 1.0);
            app.session.document.blueprint.rainfall = rainfall.max(0.0);
            app.session.document.blueprint.drainage_density = drainage_density.clamp(0.0, 1.0);
            apply_blueprint_to_stack(&mut app.session.document);
            app.mark_all_layers_dirty();
            ctx.doc_mutated = true;
        }
        PanelAction::EnsureHydrologyProcesses => {
            ensure_hydrology_processes(&mut app.session.document);
            app.mark_all_layers_dirty();
            app.request_rebuild();
            ctx.doc_mutated = true;
        }
        PanelAction::EnsureSurfaceProcesses => {
            ensure_surface_processes(&mut app.session.document);
            app.mark_all_layers_dirty();
            app.request_rebuild();
            ctx.doc_mutated = true;
        }
        PanelAction::ReorderRelative {
            moving,
            target,
            place_before,
        } => {
            if app
                .session
                .document
                .stack
                .reorder_relative(moving, target, place_before)
            {
                app.mark_all_layers_dirty();
                app.request_rebuild();
                ctx.doc_mutated = true;
            }
        }
        PanelAction::ShowExportPanel => {
            app.ui_state.show_export = true;
        }
        PanelAction::RebuildWorld => {
            app.mark_all_layers_dirty();
            app.request_rebuild();
            app.ui_state.status = "Rebuilding world".into();
            ctx.doc_mutated = true;
        }
        PanelAction::AddSuggestedLayer { type_id, label } => {
            let reg = terra_core::layer::LayerTypeRegistry::builtin();
            if let Some(layer) = reg.create_named(&type_id, label.clone()) {
                let id = layer.id();
                let kind = layer.kind.clone();
                app.session.document.stack.push(layer);
                app.session.document.selected = Some(id);
                app.ui_state.focus_view_for_new_layer(&kind);
                app.mark_all_layers_dirty();
                app.request_rebuild();
                ctx.dirty_from = Some(id);
                ctx.doc_mutated = true;
                app.ui_state.status = format!("Added suggested {label}");
            } else {
                app.ui_state.status = format!("Unknown suggestion type: {type_id}");
            }
        }
        PanelAction::OpenViewportContextMenu {
            x,
            y,
            uv,
            locked_owner,
        } => {
            app.ui_state.viewport_context_menu = Some(crate::ui::ViewportContextMenu {
                x,
                y,
                uv,
                locked_owner,
                picking_owner_for: None,
                owner_override: locked_owner,
            });
        }
        PanelAction::SetInspectorSection(key) => {
            use crate::ui::InspectorSection;
            let section = match key.to_ascii_lowercase().as_str() {
                "noise" => InspectorSection::Shape,
                "shape" => InspectorSection::Shape,
                "details" => InspectorSection::Shape,
                "erosion" | "filters" | "filter" => InspectorSection::Shape,
                "output" | "materials" | "material" => InspectorSection::Shape,
                "objects" | "object" | "vegetation" => InspectorSection::Shape,
                "distribution" | "distributions" => InspectorSection::Distribution,
                "performance" => InspectorSection::Performance,
                "layer" | "general" => InspectorSection::General,
                _ => InspectorSection::General,
            };
            app.inspector_gui.active_tab = section;
            if let Some(id) = app.session.document.selected {
                app.inspector_gui.tabs_for = Some(id);
            }
        }
        PanelAction::ContextualCreate { kind, owner, uv } => {
            use crate::ui::{create_to_workspace, workspace_to_create};
            use terra_core::contextual_create::{
                execute_create, CreateContext, CreateToolHint, CreatedEntity,
            };
            let mut create_ctx = CreateContext::from_document(
                &app.session.document,
                workspace_to_create(app.ui_state.active_workspace),
                app.ui_state.auto_switch_workspace_on_create,
            )
            .with_cursor(uv);
            if let Some(o) = owner {
                if let terra_core::contextual_create::CreateOwner::Biome(id) = o {
                    create_ctx.active_biome = Some(id);
                }
            }
            match execute_create(&mut app.session, kind, &create_ctx, owner, None) {
                Ok(out) => {
                    match out.entity {
                        CreatedEntity::Layer(id) | CreatedEntity::BiomeGroup(id) => {
                            app.session.document.selected = Some(id);
                            ctx.dirty_from = Some(id);
                        }
                        CreatedEntity::WorldRule(id) => {
                            app.session.document.world_rules.selected = Some(id);
                        }
                        CreatedEntity::Scenario(id) => {
                            app.session.document.simulation_scenarios.selected = Some(id);
                        }
                        CreatedEntity::BiomePaint(id) => {
                            app.session.document.selected_biome_layer = Some(id);
                        }
                        CreatedEntity::Shape(id) => {
                            app.session.document.shapes.selected = Some(id);
                        }
                    }
                    if let Some(section) = out.preferred_inspector_section {
                        use crate::ui::InspectorSection;
                        let tab = match section {
                            "noise" | "shape" => InspectorSection::Shape,
                            "distribution" | "distributions" => InspectorSection::Distribution,
                            "performance" => InspectorSection::Performance,
                            _ => InspectorSection::General,
                        };
                        app.inspector_gui.active_tab = tab;
                        if let Some(id) = app.session.document.selected {
                            app.inspector_gui.tabs_for = Some(id);
                        }
                    }
                    app.ui_state
                        .ensure_workspace(create_to_workspace(out.home_workspace));
                    match out.preferred_tool {
                        CreateToolHint::PaintBiome => {
                            app.ui_state.focus_created_biome();
                        }
                        CreateToolHint::Move => {
                            app.ui_state.set_editor_tool(crate::ui::EditorTool::Move);
                        }
                        CreateToolHint::Raise => {
                            app.ui_state
                                .focus_created_shape(Some(crate::ui::EditorTool::Raise));
                        }
                        CreateToolHint::PaintMask => {
                            app.ui_state.focus_created_mask(true);
                        }
                        CreateToolHint::None => {}
                    }
                    app.ui_state.status = format!("Created {}", out.name);
                    app.mark_all_layers_dirty();
                    app.request_rebuild();
                    ctx.doc_mutated = true;
                    app.ui_state.viewport_context_menu = None;
                }
                Err(e) => {
                    app.ui_state.status = e.message();
                }
            }
        }
        PanelAction::RebuildAffected => {
            let pending = app.session.outdated_sim_layers.clone();
            if pending.is_empty() && app.session.rebuild_feedback.queued.is_empty() {
                app.ui_state.status = "Nothing outdated to rebuild".into();
            } else if terra_core::rebuild_feedback::is_redundant_rebuild(&app.session, &pending) {
                app.ui_state.status = "Already rebuilding affected content".into();
            } else {
                let ids = terra_core::rebuild_feedback::rebuild_affected(&mut app.session);
                for id in &ids {
                    app.mark_dirty_from(*id);
                }
                app.request_rebuild();
                app.ui_state.affected_feedback = None;
                app.ui_state.status = format!("Rebuilding {} affected item(s)", ids.len());
                ctx.doc_mutated = true;
            }
        }
        PanelAction::SetAutomaticRebuildExpensive(on) => {
            let action = if on {
                terra_core::rebuild_feedback::RebuildFeedbackAction::EnableAutomaticRebuild
            } else {
                terra_core::rebuild_feedback::RebuildFeedbackAction::DisableAutomaticRebuild
            };
            app.ui_state.status =
                terra_core::rebuild_feedback::apply_feedback_action(&mut app.session, action);
        }
        PanelAction::SetLivePreview(on) => {
            let action = if on {
                terra_core::rebuild_feedback::RebuildFeedbackAction::EnableLivePreview
            } else {
                terra_core::rebuild_feedback::RebuildFeedbackAction::DisableLivePreview
            };
            app.ui_state.status =
                terra_core::rebuild_feedback::apply_feedback_action(&mut app.session, action);
        }
        PanelAction::WhyRebuild { layer } => {
            let graph = app.session.document.dependency_graph();
            let target = layer
                .or(app.session.document.selected)
                .map(terra_core::deps::NodeRef::Layer);
            if let Some(target) = target {
                let why = terra_core::rebuild_feedback::why_outdated(
                    &app.session.document,
                    &graph,
                    target,
                    &app.session.outdated_sim_layers,
                    app.session
                        .rebuild_feedback
                        .last_feedback
                        .as_ref()
                        .map(|f| f.why.as_str()),
                );
                app.ui_state.why_rebuild_text = Some(format!(
                    "{}\n{}",
                    why.title,
                    why.diagnostics
                        .iter()
                        .map(|d| d.message.as_str())
                        .collect::<Vec<_>>()
                        .join("\n")
                ));
                app.session.rebuild_feedback.last_why = Some(why);
                app.ui_state.status = "See inspector â€” Why outdated?".into();
            } else {
                app.ui_state.status = "Select a layer or region first".into();
            }
        }
        other => return Err(other),
    };
    let _ = result;
    Ok(())
}
