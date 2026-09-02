//! Floating editor windows (mask / content / export / preview / profiler).

use crate::ui::presets::builtin_presets;
use crate::ui::style::{self, GAP, PAD};
use crate::ui::{FrameUiOutput, Preview2dMode, TerrainSettingsUpdate, UiState};
use terra_core::document::TerrainDocument;
use terra_core::mask::{MaskAsset, MaskId, MaskOp, MaskSource, PaintBuffer};
use terra_gui::{
    button, button_id, checkbox, combo, label, label_dim, section_header, selectable, slider_f32,
    slider_f32_id, GuiContext, Id, Rect,
};

pub use crate::ui::actions::{MaskEditAction, PanelAction};

/// Default rect for a floating window, anchored inside the 3D viewport.
pub fn viewport_float_rect(ui: &GuiContext<'_>, width: f32, height: f32, x_bias: f32) -> Rect {
    let vp = ui.viewport_rect();
    let w = width.min((vp.width() - GAP * 2.0).max(160.0));
    let h = height.min((vp.height() - GAP * 2.0).max(120.0));
    let x = (vp.min_x + GAP + (vp.width() - w - GAP * 2.0).max(0.0) * x_bias.clamp(0.0, 1.0))
        .clamp(vp.min_x + GAP, (vp.max_x - w - GAP).max(vp.min_x + GAP));
    let y = (vp.min_y + GAP).min((vp.max_y - h - GAP).max(vp.min_y + GAP));
    Rect::from_pos_size(x, y, w, h)
}

#[derive(Debug, Default)]
pub struct WindowsGuiState {
    pub mask_scroll: f32,
    pub content_scroll: f32,
    pub export_scroll: f32,
    pub preview_scroll: f32,
    pub profiler_scroll: f32,
    pub recipe: crate::ui::pipeline_gui::RecipeViewState,
    pub history_scroll: f32,
    pub bookmarks_scroll: f32,
}

pub fn draw_windows(
    ui: &mut GuiContext<'_>,
    doc: &TerrainDocument,
    ui_state: &mut UiState,
    win: &mut WindowsGuiState,
    out: &mut FrameUiOutput,
) {
    // Keep a floating fallback only when the legacy flag is on outside Mask view.
    if ui_state.show_mask_editor && !ui_state.is_mask_view() {
        let rect = viewport_float_rect(ui, 360.0, 420.0, 0.05);
        if ui.begin_window(
            Id::new("win_mask"),
            "Mask Editor",
            rect,
            &mut ui_state.show_mask_editor,
            &mut win.mask_scroll,
        ) {
            draw_legacy_mask_editor_contents(ui, doc, ui_state, &mut out.actions);
            ui.end_window(&mut win.mask_scroll);
        }
    }

    if ui_state.show_content_browser {
        let rect = viewport_float_rect(ui, 360.0, 320.0, 0.08);
        if ui.begin_window(
            Id::new("win_content"),
            "Recipes",
            rect,
            &mut ui_state.show_content_browser,
            &mut win.content_scroll,
        ) {
            content_browser(ui, &mut out.actions);
            ui.end_window(&mut win.content_scroll);
        }
    }

    if ui_state.show_export {
        let rect = viewport_float_rect(ui, 520.0, 560.0, 0.2);
        if ui.begin_window(
            Id::new("win_export"),
            "Export",
            rect,
            &mut ui_state.show_export,
            &mut win.export_scroll,
        ) {
            export_panel(ui, doc, ui_state, out);
            ui.end_window(&mut win.export_scroll);
        }
    }

    if ui_state.show_2d_preview {
        let rect = viewport_float_rect(ui, 420.0, 400.0, 0.15);
        if ui.begin_window(
            Id::new("win_preview"),
            "2D Preview",
            rect,
            &mut ui_state.show_2d_preview,
            &mut win.preview_scroll,
        ) {
            preview_panel(ui, ui_state);
            ui.end_window(&mut win.preview_scroll);
        }
    }

    if ui_state.show_profiler {
        let rect = viewport_float_rect(ui, 280.0, 280.0, 1.0);
        if ui.begin_window(
            Id::new("win_profiler"),
            "Profiler",
            rect,
            &mut ui_state.show_profiler,
            &mut win.profiler_scroll,
        ) {
            profiler_panel(ui, ui_state);
            ui.end_window(&mut win.profiler_scroll);
        }
    }
}

/// Legacy project mask list / paint target (used in Mask-view dock when no region mask is selected).
pub fn draw_legacy_mask_editor_contents(
    ui: &mut GuiContext<'_>,
    doc: &TerrainDocument,
    ui_state: &mut UiState,
    actions: &mut Vec<PanelAction>,
) {
    label(ui, "Project mask layers: paint once, reuse anywhere.");
    if button(ui, "Add Painted Mask") {
        actions.push(PanelAction::AddMask(MaskAsset::new_painted(
            MaskId::new(),
            format!("Mask {}", doc.masks.len() + 1),
            512,
        )));
    }
    ui.separator();

    for asset in &doc.masks {
        let selected = ui_state.selected_mask == Some(asset.id);
        if selectable(ui, &asset.name, selected) {
            actions.push(PanelAction::SelectMask(asset.id));
        }
    }

    let selected = ui_state
        .selected_mask
        .or_else(|| doc.masks.first().map(|asset| asset.id));
    if let Some(mask_id) = selected {
        ui_state.selected_mask = Some(mask_id);
        if let Some(asset) = doc.masks.iter().find(|asset| asset.id == mask_id) {
            let mut updated = asset.clone();
            let mut changed = false;
            ui.separator();
            label(ui, &format!("MASK LAYER: {}", updated.name));

            section_header(ui, "DISPLAY");
            let mut cr = updated.display_color[0];
            let mut cg = updated.display_color[1];
            let mut cb = updated.display_color[2];
            if slider_f32(ui, "Colour R", &mut cr, 0.0, 1.0)
                | slider_f32(ui, "Colour G", &mut cg, 0.0, 1.0)
                | slider_f32(ui, "Colour B", &mut cb, 0.0, 1.0)
            {
                updated.display_color = [cr, cg, cb];
                changed = true;
            }
            label_dim(ui, "Viewport overlay only — does not affect evaluation.");

            let kind: usize = match updated.source {
                MaskSource::Constant(_) => 0,
                MaskSource::Height { .. } => 1,
                MaskSource::Slope { .. } => 2,
                MaskSource::Painted { .. } => 3,
                _ => 0,
            };
            let mut new_kind = kind;
            let kinds = ["Constant", "Height", "Slope", "Painted"];
            let _ = combo(ui, "Source", &mut new_kind, &kinds);
            if new_kind != kind {
                updated.source = match new_kind {
                    1 => MaskSource::Height {
                        min: 0.0,
                        max: 200.0,
                    },
                    2 => MaskSource::Slope {
                        min_deg: 20.0,
                        max_deg: 60.0,
                    },
                    3 => {
                        updated
                            .paint
                            .get_or_insert_with(|| PaintBuffer::new(512, 512));
                        MaskSource::Painted { mask_id }
                    }
                    _ => MaskSource::Constant(1.0),
                };
                changed = true;
            }

            match &mut updated.source {
                MaskSource::Constant(value) => {
                    changed |= slider_f32(ui, "Value", value, 0.0, 1.0);
                }
                MaskSource::Height { min, max } => {
                    changed |= slider_f32(ui, "Min", min, -500.0, 2000.0);
                    changed |= slider_f32(ui, "Max", max, -500.0, 2000.0);
                }
                MaskSource::Slope { min_deg, max_deg } => {
                    changed |= slider_f32(ui, "Min deg", min_deg, 0.0, 90.0);
                    changed |= slider_f32(ui, "Max deg", max_deg, 0.0, 90.0);
                }
                MaskSource::Painted { .. } => {
                    section_header(ui, "EDIT");
                    let painting = ui_state.paint_mask == Some(mask_id);
                    // Mask view already hosts painting — toggle paint arming, not a
                    // separate "Edit in Viewport" entry that competed with Viewsâ†’Mask.
                    let paint_label = if ui_state.is_mask_view() {
                        if painting {
                            "Stop Painting"
                        } else {
                            "Start Painting"
                        }
                    } else if painting {
                        "Stop Editing"
                    } else {
                        "Edit in Viewport"
                    };
                    if button(ui, paint_label) {
                        ui_state.paint_mask = if painting { None } else { Some(mask_id) };
                        if !painting {
                            ui_state.arm_mask_paint();
                        } else if !ui_state.is_mask_view() {
                            ui_state.set_editor_tool(crate::ui::EditorTool::Move);
                        }
                    }
                    if !ui_state.is_mask_view()
                        && button_id(ui, Id::new("mask_show"), "Show full Mask editor")
                    {
                        ui_state.enter_mask_view();
                        ui_state.show_2d_preview = true;
                    }
                    if button_id(
                        ui,
                        Id::new("mask_tool"),
                        &format!("Tool: {}", ui_state.mask_paint_tool.label()),
                    ) {
                        ui_state.mask_paint_tool = ui_state.mask_paint_tool.cycle();
                    }
                    slider_f32(ui, "Brush Radius", &mut ui_state.sculpt_radius, 0.01, 0.2);
                    slider_f32(ui, "Brush Strength", &mut ui_state.brush_flow, 0.01, 1.0);
                    slider_f32(ui, "Brush Hardness", &mut ui_state.brush_falloff, 0.0, 1.0);
                    label(
                        ui,
                        "Left-drag to edit. Shift temporarily reverses Paint/Erase.",
                    );

                    section_header(ui, "ACTIONS");
                    for (index, (label_text, action)) in [
                        ("Clear", MaskEditAction::Clear),
                        ("Fill", MaskEditAction::Fill),
                        ("Flip X", MaskEditAction::FlipX),
                        ("Flip Y", MaskEditAction::FlipY),
                        ("Rotate Left", MaskEditAction::RotateLeft),
                        ("Rotate Right", MaskEditAction::RotateRight),
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        if button_id(
                            ui,
                            Id::new("mask_edit_action").with(index as u64),
                            label_text,
                        ) {
                            actions.push(PanelAction::EditMask { mask_id, action });
                        }
                    }
                }
                _ => {}
            }

            section_header(ui, "MODIFIERS");
            if button_id(ui, Id::new("mask_add_inv"), "Add Invert") {
                updated.ops.push(MaskOp::Invert);
                changed = true;
            }
            if button_id(ui, Id::new("mask_add_blur"), "Add Blur") {
                updated.ops.push(MaskOp::Blur { radius: 2 });
                changed = true;
            }
            if button_id(ui, Id::new("mask_add_levels"), "Add Levels") {
                updated.ops.push(MaskOp::Levels {
                    in_black: 0.1,
                    in_white: 0.9,
                    gamma: 1.0,
                });
                changed = true;
            }
            if button_id(ui, Id::new("mask_add_clamp"), "Add Clamp") {
                updated.ops.push(MaskOp::Clamp { min: 0.0, max: 1.0 });
                changed = true;
            }
            if changed {
                actions.push(PanelAction::UpdateMaskAsset(updated));
            }
        }
    }

    let target = doc.selected.and_then(|target_id| {
        if let Some(layer) = doc.stack.find(target_id) {
            Some((target_id, layer.common.name.as_str(), &layer.common.masks))
        } else {
            doc.stack
                .find_group(target_id)
                .map(|group| (target_id, group.name.as_str(), &group.masks))
        }
    });
    if let Some((target_id, target_name, distribution)) = target {
        ui.separator();
        label(ui, &format!("USE MASK ON: {target_name}"));
        if let Some(mask_id) = ui_state.selected_mask {
            if distribution
                .iter()
                .any(|binding| binding.mask.id == mask_id)
            {
                if button(ui, "Remove selected mask") {
                    actions.push(PanelAction::UnbindMask {
                        layer: target_id,
                        mask: mask_id,
                    });
                }
            } else if button(ui, "Use selected mask here") {
                actions.push(PanelAction::BindMaskToLayer {
                    layer: target_id,
                    mask: mask_id,
                });
            }
        }
        for (i, binding) in distribution.iter().enumerate() {
            let mut strength = binding.mask.strength;
            let mut invert = binding.mask.invert;
            let binding_name = doc
                .masks
                .iter()
                .find(|asset| asset.id == binding.mask.id)
                .map(|asset| asset.name.as_str())
                .unwrap_or("Missing Mask");
            label(ui, &format!("{}: {}", i + 1, binding_name));
            let strength_changed = slider_f32_id(
                ui,
                Id::new("mask_str").with(i as u64),
                "Strength",
                &mut strength,
                0.0,
                1.0,
            );
            let invert_changed = checkbox(ui, "Invert", &mut invert);
            if strength_changed || invert_changed {
                actions.push(PanelAction::UpdateMaskBinding {
                    layer: target_id,
                    mask: binding.mask.id,
                    strength,
                    invert,
                });
            }
            if button_id(
                ui,
                Id::new("mask_combine").with(i as u64),
                &format!("Combine: {}", binding.combine.label()),
            ) {
                actions.push(PanelAction::CycleMaskCombine {
                    layer: target_id,
                    mask: binding.mask.id,
                });
            }
        }
    } else {
        ui.separator();
        label(
            ui,
            "Select a biome, filter, material, object, or group to use this mask.",
        );
    }
}

fn content_browser(ui: &mut GuiContext<'_>, actions: &mut Vec<PanelAction>) {
    label(ui, "Recipes");
    label(ui, "Reusable biome templates — add into the current world.");
    ui.separator();
    for (i, recipe) in crate::ui::recipe::builtin_recipes().into_iter().enumerate() {
        if button_id(ui, Id::new("recipe").with(i as u64), &recipe.name) {
            actions.push(PanelAction::InstantiateRecipe {
                recipe_name: recipe.name.clone(),
            });
        }
        label(ui, &recipe.description);
        ui.gap(style::GAP);
    }
    ui.separator();
    label(ui, "Legacy stack presets");
    label(
        ui,
        "Replace the whole stack (showcase demos — use Advanced).",
    );
    ui.separator();
    for (i, preset) in builtin_presets().into_iter().enumerate() {
        if button_id(ui, Id::new("preset").with(i as u64), &preset.name) {
            actions.push(PanelAction::ApplyPreset(preset.name.clone()));
        }
        label(ui, &preset.description);
        ui.gap(style::GAP);
    }
}

fn export_panel(
    ui: &mut GuiContext<'_>,
    doc: &TerrainDocument,
    ui_state: &mut UiState,
    out: &mut FrameUiOutput,
) {
    use terra_io::FieldExportFormat;
    let mut format_index = FieldExportFormat::ALL
        .iter()
        .position(|format| *format == ui_state.export_options.format)
        .unwrap_or(0);
    let formats = FieldExportFormat::ALL.map(FieldExportFormat::label);
    if combo(ui, "Format", &mut format_index, &formats) {
        ui_state.export_options.format = FieldExportFormat::ALL[format_index];
    }
    let available = terra_io::exportable_fields(doc);
    match &available {
        Ok(fields) => {
            ui_state.export_options.retain_available(fields);
            for field in fields {
                let mut selected = ui_state.export_options.fields.contains(field);
                if terra_gui::checkbox_id(
                    ui,
                    Id::new("export_field").child(&field.cache_key()),
                    &field.display_name(),
                    &mut selected,
                ) {
                    if selected {
                        ui_state.export_options.fields.push(field.clone());
                    } else {
                        ui_state.export_options.fields.retain(|id| id != field);
                    }
                }
            }
        }
        Err(error) => label_dim(ui, error),
    }
    ui.separator();
    const RESOLUTIONS: [u32; 6] = [256, 512, 1024, 2048, 4096, 8192];
    const RESOLUTION_LABELS: [&str; 6] = ["256", "512", "1024", "2048", "4096", "8192"];
    // Older projects may contain arbitrary slider values. Round those up to
    // the next supported size so the displayed choice matches the export grid.
    let mut resolution_index = RESOLUTIONS
        .partition_point(|resolution| *resolution < doc.export_resolution)
        .min(RESOLUTIONS.len() - 1);
    if combo(ui, "Resolution", &mut resolution_index, &RESOLUTION_LABELS)
        || RESOLUTIONS[resolution_index] != doc.export_resolution
    {
        out.actions
            .push(PanelAction::UpdateTerrainSettings(TerrainSettingsUpdate {
                export_resolution: Some(RESOLUTIONS[resolution_index]),
                ..Default::default()
            }));
    }
    ui.separator();
    label(
        ui,
        &format!(
            "Directory: {}",
            ui_state.export_path.as_deref().unwrap_or("Not selected")
        ),
    );
    if button(ui, "Choose Export Directory...") {
        out.request_export_path = true;
    }
    terra_gui::checkbox_id(
        ui,
        Id::new("export_open_folder"),
        "Open after export",
        &mut ui_state.open_export_folder_when_finished,
    );
    ui.gap(3.0);
    let enabled = available.is_ok()
        && !ui_state.export_options.fields.is_empty()
        && ui_state.export_progress.is_none();
    if export_button(ui, enabled) {
        out.request_start_export = true;
    }
    if let Some(progress) = ui_state.export_progress {
        label(ui, &format!("Exporting... {:.0}%", progress * 100.0));
    }
}

fn export_button(ui: &mut GuiContext<'_>, enabled: bool) -> bool {
    let id = Id::new("export_start");
    if enabled {
        return button_id(ui, id, "Export");
    }
    // Keep the button visible, but remove both its hit target and any held press.
    if ui.state.is_active(id) {
        ui.state.active = None;
    }
    let rect = ui.allocate(style::ROW_H);
    ui.panel_rounded(rect, style::BUTTON_BG, style::RADIUS_SM);
    ui.label_centered_in_rect(rect, "Export", style::TEXT_DIM, style::FONT_SCALE);
    ui.gap(3.0);
    false
}
fn preview_panel(ui: &mut GuiContext<'_>, ui_state: &mut UiState) {
    let modes = [
        (Preview2dMode::Height, "Height"),
        (Preview2dMode::Slope, "Slope"),
        (Preview2dMode::Flow, "Flow"),
        (Preview2dMode::Masks, "Mask"),
    ];
    let row = ui.allocate(crate::ui::style::ROW_H);
    let cell_w = row.width() / modes.len() as f32;
    for (i, (mode, name)) in modes.iter().enumerate() {
        let cell = Rect::from_pos_size(
            row.min_x + cell_w * i as f32,
            row.min_y,
            cell_w - 2.0,
            row.height(),
        );
        let selected = ui_state.preview_mode == *mode;
        let id = Id::new("preview_mode").with(i as u64);
        let hovered = ui.pointer_in(cell);
        if hovered {
            ui.state.set_hot(id);
        }
        if hovered && ui.input.primary_pressed {
            ui.state.active = Some(id);
        }
        if ui.input.primary_released && ui.state.is_active(id) && hovered {
            ui_state.preview_mode = *mode;
        }
        ui.panel(
            cell,
            if selected {
                style::SELECTED_BG
            } else if hovered {
                style::BUTTON_HOVER
            } else {
                style::BUTTON_BG
            },
        );
        ui.label_centered(cell.center_x(), cell.min_y + 4.0, name, style::TEXT, 1.0);
    }

    ui.gap(PAD);
    if let Some((width, height, rgba)) = &ui_state.preview_rgba {
        let avail = ui.allocate(220.0);
        let aspect = *width as f32 / (*height as f32).max(1.0);
        let (w, h) = if avail.width() / avail.height() > aspect {
            (avail.height() * aspect, avail.height())
        } else {
            (avail.width(), avail.width() / aspect)
        };
        let img = Rect::from_pos_size(avail.min_x, avail.min_y, w, h);
        ui.image(img, *width, *height, rgba);
    } else {
        label(ui, "Waiting for a completed terrain evaluation.");
    }
}

fn profiler_panel(ui: &mut GuiContext<'_>, ui_state: &UiState) {
    let p = &ui_state.profile;
    label(
        ui,
        &format!(
            "Logical frame: {}  |  Phase: {}",
            p.logical_frame_id,
            if p.logical_phase.is_empty() {
                "idle"
            } else {
                p.logical_phase
            }
        ),
    );
    label(
        ui,
        &format!(
            "Edit/present generation: {}/{}  |  Input events/samples: {}/{}{}",
            p.edit_generation,
            p.presented_generation,
            p.input_event_count,
            p.pointer_sample_count,
            if p.input_frame_pending {
                "  (next pending)"
            } else {
                ""
            }
        ),
    );
    label(ui, &format!("Generation ID: {}", p.gen_id));
    label(
        ui,
        &format!("Trace orphaned events: {}", p.trace_orphaned_events),
    );
    label(
        ui,
        &format!(
            "Base trace (n={}): input-visible p50/p95/max {} / {} / {} us",
            p.brush_trace_samples,
            p.input_visible_p50_us,
            p.input_visible_p95_us,
            p.input_visible_max_us
        ),
    );
    label(
        ui,
        &format!(
            "Release-refined p50/p95/max {} / {} / {} us",
            p.refinement_p50_us, p.refinement_p95_us, p.refinement_max_us
        ),
    );
    label(
        ui,
        &format!(
            "Release-next press p50/p95/max {} / {} / {} us",
            p.follow_up_press_p50_us, p.follow_up_press_p95_us, p.follow_up_press_max_us
        ),
    );
    label(
        ui,
        &format!("Quality: {}  |  Tex {}x{}", p.quality, p.tex_w, p.tex_h),
    );
    label(
        ui,
        &format!(
            "Eval path: {}",
            if p.path.is_empty() { "-" } else { p.path }
        ),
    );
    if let Some(fallback) = &p.gpu_fallback {
        label(
            ui,
            &format!(
                "CPU boundary: #{} {} [{:?}]",
                fallback.layer_index, fallback.layer_name, fallback.reason.code
            ),
        );
        label(ui, &format!("Reason: {}", fallback.reason.user_message()));
    }
    if let Some(reason) = &p.terrain_tile_fallback {
        label(ui, "Terrain tiles: complete-field fallback");
        label(ui, &format!("Reason: {reason}"));
    }
    label(
        ui,
        &format!(
            "Terrain grid: {}²  |  Tiles {}x{}",
            p.terrain_grid_size, p.tiles_x, p.tiles_z
        ),
    );
    ui.separator();
    label(ui, &format!("Layer eval:  {:>6} us", p.eval_us));
    label(
        ui,
        &format!("GPU eval:    {:>6} us (delayed)", p.gpu_evaluation_us),
    );
    label(
        ui,
        &format!(
            "Visible/settled: {:>6} / {:>6} us",
            p.first_visible_preview_us, p.settled_authoritative_us
        ),
    );
    label(
        ui,
        &format!(
            "Plan compile: {} ({} us)  |  walks/deps {}/{}",
            p.plan.plan_compiles,
            p.plan.plan_compile_us,
            p.plan.authored_tree_walks,
            p.plan.dependency_builds
        ),
    );
    label(
        ui,
        &format!(
            "GPU ops run/publish/skip/reuse/defer: {}/{}/{}/{}/{}  |  groups {}",
            p.gpu.operations_dispatched,
            p.gpu.operations_published,
            p.gpu.operations_skipped,
            p.gpu.operations_reused,
            p.gpu.operations_deferred,
            p.gpu.plan_workgroups
        ),
    );
    label(
        ui,
        &format!(
            "Eval prep/preflight/encode/submit: {}/{}/{}/{} us  |  {} {}² ops {} dirty {} texels",
            p.gpu.resource_prepare_us,
            p.gpu.capability_preflight_us,
            p.gpu.command_encode_us,
            p.gpu.queue_submit_us,
            if p.gpu.cold_execution { "cold" } else { "warm" },
            p.gpu.resolution,
            p.gpu.selected_operations,
            p.gpu.dirty_texels
        ),
    );
    label(
        ui,
        &format!(
            "Mask scratch alloc/reuse: {}/{}",
            p.gpu.mask_scratch_texture_allocations, p.gpu.mask_scratch_reuses
        ),
    );
    label(
        ui,
        &format!(
            "GPU bytes upload/readback: {}/{}  |  CPU jobs submit/cancel/done/publish: {}/{}/{}/{}",
            p.gpu.total_upload_bytes(),
            p.gpu.readback_bytes,
            p.cpu_worker.submitted,
            p.cpu_worker.cancelled + p.cpu_worker.stale_skipped,
            p.cpu_worker.completed,
            p.cpu_published
        ),
    );
    label(ui, &format!("GPU upload:  {:>6} us", p.upload_us));
    label(ui, &format!("Terrain draw:{:>6} us", p.render_us));
    if p.gpu_timestamps_supported {
        label(ui, &format!("GPU terrain: {:>6} us", p.gpu_terrain_us));
        label(ui, &format!("GPU shadow:  {:>6} us", p.gpu_shadow_us));
    } else {
        label(ui, "GPU timestamps: unsupported");
    }
    label(ui, &format!("UI:          {:>6} us", p.ui_us));
    label(ui, &format!("Frame total: {:>6} us", p.frame_us));
    ui.separator();
    label(ui, "Viewport never waits on eval;");
    label(ui, "last-good textures stay on screen.");
    if !p.renderer_mode.is_empty() {
        ui.separator();
        label(ui, &format!("Renderer: {}", p.renderer_mode));
        label(
            ui,
            &format!(
                "Interaction: {}  |  accum {}/{} spp  (frame {})",
                p.interaction_state, p.spp_this_frame, p.max_spp, p.accum_frame
            ),
        );
        label(
            ui,
            &format!(
                "Convergence {:.0}%  |  tiles active {} / reduced {} / converged {}",
                p.convergence_fraction * 100.0,
                p.active_tiles,
                p.reduced_tiles,
                p.converged_tiles
            ),
        );
        label(
            ui,
            &format!(
                "Internal scale {:.2}  |  GPU {:.2} ms (smooth {:.2})",
                p.internal_scale, p.last_gpu_ms, p.smoothed_gpu_ms
            ),
        );
        label(
            ui,
            &format!(
                "Versions cam {} ter {} lit {}  |  last {}",
                p.camera_version, p.terrain_version, p.lighting_version, p.last_invalidation
            ),
        );
        if p.gpu_timestamps_supported {
            label(
                ui,
                &format!(
                    "PT {} us  temporal {} us  denoise {} us",
                    p.path_trace_us, p.temporal_us, p.denoise_us
                ),
            );
        }
        label(
            ui,
            &format!(
                "Global frame {}  |  bounces {}  |  samples {}",
                p.global_frame, p.bounce_count, p.spp_this_frame
            ),
        );
    }
}

#[cfg(test)]
mod export_tests {
    use super::*;
    use terra_core::fields::FieldId;
    use terra_gui::{GuiInput, GuiState};
    use terra_io::FieldExportFormat;

    fn draw_export(
        ui_state: &mut UiState,
        state: &mut GuiState,
        input: GuiInput,
    ) -> (FrameUiOutput, f32) {
        let doc = TerrainDocument::new_default();
        draw_export_document(&doc, ui_state, state, input)
    }

    fn draw_export_document(
        doc: &TerrainDocument,
        ui_state: &mut UiState,
        state: &mut GuiState,
        input: GuiInput,
    ) -> (FrameUiOutput, f32) {
        let mut ui = GuiContext::begin(800.0, 900.0, 1.0, input, state);
        ui.begin_panel(Rect::from_pos_size(0.0, 0.0, 520.0, 900.0), style::PANEL_BG);
        let mut out = FrameUiOutput::default();
        export_panel(&mut ui, doc, ui_state, &mut out);
        let button_y = ui.layout_cursor_y().unwrap() - 3.0 - style::ROW_H / 2.0;
        ui.end_panel();
        ui.end();
        (out, button_y)
    }

    #[test]
    fn export_dialog_updates_format_and_allows_deselecting_height() {
        let mut state = GuiState::default();
        let mut ui_state = UiState::default();
        state.combo_pick = Some((Id::new("Format").child("combo"), 1));
        draw_export(&mut ui_state, &mut state, GuiInput::default());
        assert_eq!(ui_state.export_options.format, FieldExportFormat::Tiff16);
        let pointer = Some((
            100.0,
            style::PAD + style::CONTROL_ROW_H + style::ROW_H / 2.0,
        ));
        for primary_down in [true, false] {
            draw_export(
                &mut ui_state,
                &mut state,
                GuiInput {
                    pointer,
                    primary_down,
                    ..Default::default()
                },
            );
        }
        assert!(ui_state.export_options.fields.is_empty());
    }

    #[test]
    fn export_resolution_dropdown_selects_each_supported_size() {
        for (index, resolution) in [256, 512, 1024, 2048, 4096, 8192].into_iter().enumerate() {
            let mut state = GuiState::default();
            let mut ui_state = UiState::default();
            state.combo_pick = Some((Id::new("Resolution").child("combo"), index));
            let (out, _) = draw_export(&mut ui_state, &mut state, GuiInput::default());
            let update = out.actions.iter().find_map(|action| match action {
                PanelAction::UpdateTerrainSettings(update) => update.export_resolution,
                _ => None,
            });
            if resolution == TerrainDocument::new_default().export_resolution {
                assert_eq!(
                    update, None,
                    "the current size should not dirty the project"
                );
            } else {
                assert_eq!(update, Some(resolution));
            }
        }
    }

    #[test]
    fn export_resolution_dropdown_normalizes_legacy_sizes() {
        for (stored, expected) in [(128, 256), (1000, 1024), (8193, 8192)] {
            let mut doc = TerrainDocument::new_default();
            doc.export_resolution = stored;
            let (out, _) = draw_export_document(
                &doc,
                &mut UiState::default(),
                &mut GuiState::default(),
                GuiInput::default(),
            );
            assert!(out.actions.iter().any(|action| matches!(
                action,
                PanelAction::UpdateTerrainSettings(update)
                    if update.export_resolution == Some(expected)
            )));
        }
    }

    #[test]
    fn export_button_is_inert_without_fields_or_while_busy() {
        for (fields, progress, expected) in [
            (vec![], None, false),
            (vec![FieldId::Height], Some(0.5), false),
            (vec![FieldId::Height], None, true),
        ] {
            let mut ui_state = UiState::default();
            ui_state.export_options.fields = fields;
            let mut state = GuiState::default();
            let (_, button_y) = draw_export(&mut ui_state, &mut state, GuiInput::default());
            ui_state.export_progress = progress;
            let pointer = Some((180.0, button_y));
            draw_export(
                &mut ui_state,
                &mut state,
                GuiInput {
                    pointer,
                    primary_down: true,
                    ..Default::default()
                },
            );
            let (out, _) = draw_export(
                &mut ui_state,
                &mut state,
                GuiInput {
                    pointer,
                    ..Default::default()
                },
            );
            assert_eq!(out.request_start_export, expected);
        }
    }

    #[test]
    fn export_open_folder_checkbox_toggles_without_starting_export() {
        let mut state = GuiState::default();
        let mut ui_state = UiState::default();
        assert!(!ui_state.open_export_folder_when_finished);
        let (_, export_button_y) = draw_export(&mut ui_state, &mut state, GuiInput::default());
        let pointer = Some((180.0, export_button_y - style::ROW_H - 3.0));
        for expected in [true, false] {
            for primary_down in [true, false] {
                let (out, _) = draw_export(
                    &mut ui_state,
                    &mut state,
                    GuiInput {
                        pointer,
                        primary_down,
                        ..Default::default()
                    },
                );
                assert!(!out.request_start_export);
            }
            assert_eq!(ui_state.open_export_folder_when_finished, expected);
        }
    }
}
