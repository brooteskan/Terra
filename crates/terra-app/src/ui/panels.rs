//! Floating editor windows (mask / content / export / preview / profiler).

use crate::ui::presets::builtin_presets;
use crate::ui::style::{self, GAP, PAD};
use crate::ui::{FrameUiOutput, Preview2dMode, TerrainSettingsUpdate, UiState};
use terra_core::document::TerrainDocument;
use terra_core::mask::{MaskAsset, MaskId, MaskOp, MaskSource, PaintBuffer};
use terra_gui::{
    button, button_id, checkbox, combo, label, label_dim, section_header, selectable, slider_f32,
    slider_f32_id, slider_i32, GuiContext, Id, Rect,
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
        let rect = viewport_float_rect(ui, 380.0, 420.0, 0.2);
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
                    if !ui_state.is_mask_view() {
                        if button_id(ui, Id::new("mask_show"), "Show full Mask editor") {
                            ui_state.enter_mask_view();
                            ui_state.show_2d_preview = true;
                        }
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
    label(ui, "What you'll get");
    label(ui, "- height.png / height.r32 - heightmap");
    label(ui, "- height_meta.json - size / world extents");
    label(ui, "- mask_*.png - baked aux masks");
    label(ui, "- splat.png + splat_ids.json - material IDs");
    label(ui, "- vegetation_instances.json - foliage points");
    label(ui, "- terrain_collision.obj - coarse mesh");
    label(ui, "- tile_manifest.json - streaming tiles");
    ui.separator();

    label(ui, "Resolution");
    let mut export_res = doc.export_resolution as i32;
    let mut update = TerrainSettingsUpdate::default();
    if slider_i32(ui, "Export res", &mut export_res, 512, 8192) {
        update.export_resolution = Some(export_res as u32);
    }
    let mut preview_res = doc.preview_resolution as i32;
    if slider_i32(ui, "Preview res", &mut preview_res, 256, 8192) {
        update.preview_resolution = Some(preview_res as u32);
    }
    ui.separator();
    label(ui, "LEVEL STEPS (Terrain)");
    let levels =
        terra_core::analyze::LevelStepSettings::level_count_for_resolution(preview_res as u32);
    label(ui, &format!("Upsample levels at preview: {levels}"));
    changed_slider_terrain(doc, ui, &mut update);
    label(
        ui,
        &format!(
            "Export uses {}x{} at export quality (not Draft).",
            export_res, export_res
        ),
    );
    if !update.is_empty() {
        out.actions.push(PanelAction::UpdateTerrainSettings(update));
    }
    ui.separator();

    label(ui, "Outputs (per Start Export)");
    label(ui, "- height.png (16-bit) + height.r32");
    label(ui, "- splat.png (RGBA material weights)");
    label(ui, "- color.png (baked albedo tint)");
    label(ui, "- normal.png (from height derivatives)");
    label(ui, "- mask_*.png aux fields");
    label(ui, "Meta JSON carries world size for DCC handoff.");
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
    if button_id(ui, Id::new("export_start"), "Start Export") {
        out.request_start_export = true;
    }

    if let Some(progress) = ui_state.export_progress {
        label(ui, &format!("Exporting... {:.0}%", progress * 100.0));
    } else if ui_state.export_path.is_some() {
        label(ui, "Ready - Start Export writes files in the background.");
    } else {
        label(ui, "Pick a directory, then Start Export.");
    }
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
    label(ui, &format!("Generation ID: {}", p.gen_id));
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
    label(
        ui,
        &format!(
            "Terrain grid: {}²  |  Tiles {}x{}",
            p.terrain_grid_size, p.tiles_x, p.tiles_z
        ),
    );
    ui.separator();
    label(ui, &format!("Layer eval:  {:>6} us", p.eval_us));
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

fn changed_slider_terrain(
    doc: &TerrainDocument,
    ui: &mut GuiContext<'_>,
    update: &mut TerrainSettingsUpdate,
) {
    let mut precision = doc.level_steps.precision;
    if slider_f32(ui, "Precision", &mut precision, 0.25, 4.0) {
        update.precision = Some(precision);
    }
    let mut world = doc.level_steps.world_scale;
    if slider_f32(ui, "World Scale", &mut world, 0.1, 10.0) {
        update.world_scale = Some(world);
    }
    let mut max_level = doc.level_steps.max_level as i32;
    if slider_i32(ui, "Max Level (0=auto)", &mut max_level, 0, 12) {
        update.max_level = Some(max_level as u32);
    }
    if button_id(
        ui,
        Id::new("hd_mode_cycle"),
        &format!("HD Preview: {}", doc.level_steps.high_detail.label()),
    ) {
        update.high_detail = Some(doc.level_steps.high_detail.cycle());
    }
    let mut outline = doc.level_steps.show_hd_outline;
    if checkbox(ui, "Show HD Outline", &mut outline) {
        update.show_hd_outline = Some(outline);
    }
}
