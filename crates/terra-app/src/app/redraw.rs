use std::time::Instant;

use crate::ui::{
    draw_discard_confirm, draw_editor_gui, draw_new_project_templates, draw_project_home,
    DiscardConfirmChoice, NewProjectTemplateChoice,
};
use terra_core::eval::PreviewQuality;
use terra_core::layer::LayerKind;
use terra_gui::{GuiContext, GuiInput};
use winit::event::MouseButton;

use super::helpers::{aux_maps_fingerprint, mask_field_fingerprint, vegetation_instance_params};
use super::prefs::{save_editor_prefs, EditorPrefs};
use super::{AppScreen, PendingProjectAction, TerraApp};
impl TerraApp {
    pub(crate) fn redraw(&mut self) {
        let frame_t0 = Instant::now();
        let Some(window) = self.window.clone() else {
            return;
        };

        // Coalesce fast brush motion: many stamps per frame, one Draft present.
        // Mask paint updates overlay only â€” never force a height eval mid-stroke.
        if self.pending_eval
            && (self.sculpt_stroke_active
                || (self.mouse_pressed == Some(MouseButton::Left)
                    && self.viewport_paint_active()
                    && self.ui_state.editor_tool != crate::ui::EditorTool::PaintMask))
        {
            self.flush_live_paint_preview();
        }

        self.refresh_viewport_rect();
        self.refresh_2d_preview();
        if self.should_show_mask_overlay() {
            let target = self.ui_state.paint_mask.or(self.ui_state.selected_mask);
            if self.mask_overlay_dirty || self.last_mask_overlay_id != target {
                self.sync_mask_overlay_to_renderer();
                self.last_mask_overlay_id = target;
            }
        } else {
            if self.last_mask_overlay_id.take().is_some() {
                self.placement_tint_dirty = true;
            }
            if self.placement_tint_dirty {
                self.sync_placement_tint_to_renderer();
            }
        }

        if self.needs_height_upload {
            if let Some(hf) = self.last_height.clone() {
                self.ensure_climate_for_lit();
                let material_palette = self
                    .session
                    .document
                    .stack
                    .flatten_layers()
                    .into_iter()
                    .rev()
                    .find_map(|layer| match &layer.kind {
                        LayerKind::Materials(p) if layer.common.enabled => Some(p.clone()),
                        _ => None,
                    });
                if let Some(r) = self.renderer.as_mut() {
                    r.upload_heightfield(&hf);
                    let ocean_level = if self.ui_state.viewport_overlays.water_level {
                        self.session
                            .document
                            .stack
                            .flatten_layers()
                            .into_iter()
                            .rev()
                            .find_map(|layer| {
                                if !layer.common.enabled {
                                    None
                                } else {
                                    match &layer.kind {
                                        LayerKind::Island(p) => Some(p.sea_level),
                                        LayerKind::Coastal(p) => Some(p.sea_level),
                                        _ => None,
                                    }
                                }
                            })
                    } else {
                        None
                    };
                    let aux_fp = aux_maps_fingerprint(&self.scheduler.last_aux);
                    if aux_fp != self.aux_upload_fp {
                        r.upload_aux_maps_ex(
                            self.scheduler.last_aux.get("materials"),
                            self.scheduler.last_aux.get("wetness"),
                            self.scheduler.last_aux.get("vegetation"),
                            self.scheduler.last_aux.get("temperature"),
                            self.scheduler.last_aux.get("rainfall"),
                            self.scheduler.last_aux.get("snow"),
                            self.scheduler.last_aux.get("soil_moisture"),
                            self.scheduler.last_aux.get("biomes"),
                            self.scheduler.last_aux.get("flow_accumulation"),
                        );
                        self.aux_upload_fp = aux_fp;
                    }
                    r.upload_material_palette(material_palette.as_ref());
                    r.set_ocean_level(ocean_level);
                    self.placement_tint_dirty = true;
                    let (veg_scale_min, veg_scale_max, veg_yaw) =
                        vegetation_instance_params(&self.session.document);
                    let veg_fp = self
                        .scheduler
                        .last_aux
                        .get("vegetation")
                        .map(mask_field_fingerprint)
                        .unwrap_or(0)
                        ^ ((hf.metrics.width as u64) << 32)
                        ^ hf.metrics.height as u64
                        ^ (veg_scale_min.to_bits() as u64)
                        ^ ((veg_scale_max.to_bits() as u64) << 17)
                        ^ (veg_yaw.to_bits() as u64).rotate_left(7);
                    if veg_fp != self.veg_upload_fp {
                        r.sync_vegetation_instances(
                            &hf,
                            self.scheduler.last_aux.get("vegetation"),
                            veg_scale_min,
                            veg_scale_max,
                            veg_yaw,
                        );
                        self.veg_upload_fp = veg_fp;
                    }
                    // Phase J: dual-height overhang / cave roof proxy from aux maps.
                    let overhang_fp = self
                        .scheduler
                        .last_aux
                        .get("overhang_ceiling")
                        .map(mask_field_fingerprint)
                        .unwrap_or(0)
                        ^ self
                            .scheduler
                            .last_aux
                            .get("overhang_mask")
                            .map(mask_field_fingerprint)
                            .unwrap_or(0)
                            .rotate_left(13);
                    if overhang_fp != self.overhang_upload_fp {
                        let overhang_mesh = match (
                            self.scheduler.last_aux.get("overhang_ceiling"),
                            self.scheduler.last_aux.get("overhang_mask"),
                        ) {
                            (Some(ceiling), Some(mask)) => {
                                let mesh = terra_core::volumetric::build_overhang_mesh(
                                    &hf, ceiling, mask, 0.12,
                                );
                                if mesh.is_empty() {
                                    None
                                } else {
                                    Some(mesh)
                                }
                            }
                            _ => None,
                        };
                        r.sync_overhang_mesh(overhang_mesh.as_ref());
                        self.overhang_upload_fp = overhang_fp;
                    }
                    self.ui_state.profile.upload_us = r.last_upload_us;
                }
                self.needs_height_upload = false;
            }
        }

        let pointer = self.cursor_logical();
        let primary_down = self.mouse_pressed == Some(MouseButton::Left);
        let secondary_down = self.mouse_pressed == Some(MouseButton::Right);
        let scroll_delta = std::mem::take(&mut self.gui_scroll_delta);
        let text = std::mem::take(&mut self.gui_text);
        let backspace_pressed = std::mem::take(&mut self.gui_backspace);
        let escape_pressed = std::mem::take(&mut self.gui_escape);
        let enter_pressed = std::mem::take(&mut self.gui_enter);
        let pixels_per_point = window.scale_factor() as f32;

        // Brush ring tracks the cursor on the height surface while sculpt/mask tools are armed.
        self.update_brush_gizmo();

        let ui_out = {
            let Some(renderer) = self.renderer.as_mut() else {
                return;
            };
            let Some(gui_renderer) = self.gui_renderer.as_mut() else {
                return;
            };
            let Some(gpu) = self.gpu.as_ref() else {
                return;
            };

            // Terrain pass — always draws last-good GPU textures (never waits on eval).
            let render_t0 = Instant::now();
            {
                // (Re)seed the editable lighting when a preset is selected: either a
                // different preset, or the same one re-picked to clear a customized
                // (blank) state. Editing any control sets `lighting_customized`, which
                // blanks the preset field and suppresses re-seeding so the edit sticks.
                let preset = self.ui_state.lighting_preset;
                let customized = self.ui_state.lighting_customized;
                if !customized
                    && (preset != self.last_lighting_preset || self.last_lighting_customized)
                {
                    renderer.notify_invalidation(terra_render::InvalidationReason::LightingChanged);
                    // Keep the render mode consistent with the chosen preset in both
                    // directions: Progressive RT selects the path tracer, every other
                    // preset returns to Raster.
                    self.ui_state.viewport_render.mode = preset.suggested_renderer_mode();
                    let (preset_dir, preset_exposure, preset_clear) = preset.params();
                    let (az, el) = crate::ui::sun_az_el_from_dir([
                        preset_dir[0],
                        preset_dir[1],
                        preset_dir[2],
                    ]);
                    let vr = &mut self.ui_state.viewport_render;
                    vr.sun_azimuth_deg = az;
                    vr.sun_elevation_deg = el;
                    vr.sun_intensity = preset_dir[3];
                    vr.exposure = preset_exposure;
                    vr.sky_color = preset_clear;
                    // Raster shading is not yet preset-defined; reset to neutral so a
                    // preset restores a known, full lighting state.
                    vr.ambient_strength = 1.0;
                    vr.shadow_strength = 0.0;
                    vr.fog_strength = 1.0;
                }
                self.last_lighting_preset = preset;
                self.last_lighting_customized = customized;

                // All viewport lighting is driven by the editable values (seeded from the
                // preset above); the Home splash keeps its fixed look.
                let (light_dir, exposure, clear) = if self.screen == AppScreen::Home {
                    ([-0.35, -0.90, -0.20, 1.00], 1.0, [0.071, 0.082, 0.102])
                } else {
                    let vr = &self.ui_state.viewport_render;
                    let dir =
                        crate::ui::sun_dir_from_az_el(vr.sun_azimuth_deg, vr.sun_elevation_deg);
                    (
                        [dir[0], dir[1], dir[2], vr.sun_intensity],
                        vr.exposure,
                        vr.sky_color,
                    )
                };
                renderer.lighting.light_dir = light_dir;
                renderer.lighting.exposure = exposure;
                renderer.lighting.clear = clear;
                // Raster shading controls are independent of the preset (not seeded),
                // so push them straight from the editable state every frame.
                renderer.lighting.ambient_strength = self.ui_state.viewport_render.ambient_strength;
                renderer.lighting.shadow_strength = self.ui_state.viewport_render.shadow_strength;
                renderer.lighting.fog_strength = self.ui_state.viewport_render.fog_strength;

                let vr = &mut self.ui_state.viewport_render;
                renderer.set_renderer_mode(vr.mode);
                let quality = renderer.quality_mut();
                if quality.config.preset != vr.preset {
                    quality.set_preset(vr.preset);
                }
                quality.config.target_fps = vr.target_fps.clamp(15.0, 240.0);
                quality.config.max_accumulated_spp = vr.max_spp.max(1);
                quality.config.dynamic_resolution_enabled = vr.dynamic_resolution;
                quality.config.denoise_enabled = vr.denoise;
                quality.config.interactive_spp = vr.interactive_spp.max(1);
                quality.config.settling_spp = vr.settling_spp.max(1);
                quality.config.refining_spp = vr.refining_spp.max(1);
                quality.config.max_bounces_interactive = vr.max_bounces_interactive.max(1);
                quality.config.max_bounces_refining = vr.max_bounces_refining.max(1);
                quality.config.min_internal_scale =
                    vr.min_internal_scale.clamp(0.25, vr.max_internal_scale);
                quality.config.max_internal_scale = vr.max_internal_scale.clamp(0.25, 1.0);
                quality.config.history_clamp_k = vr.history_clamp_k.max(0.1);
                quality.config.converge_fraction = vr.converge_fraction.clamp(0.0, 1.0);
                let debug_viz = vr.debug_viz_mode;
                renderer.set_debug_viz_mode(debug_viz);
                // Mode alone selects the presentation backend (no dual progressive flag).
                let progressive_active =
                    self.screen == AppScreen::Editor && vr.mode.uses_progressive_path_tracer();
                self.ui_state.progressive_renderer_active = progressive_active;

                let shading = if self.ui_state.is_mask_view() {
                    terra_render::ViewportShadingMode::Lit
                } else {
                    match self.ui_state.preview_mode {
                        crate::ui::Preview2dMode::Height => {
                            terra_render::ViewportShadingMode::Height
                        }
                        crate::ui::Preview2dMode::Slope => terra_render::ViewportShadingMode::Slope,
                        crate::ui::Preview2dMode::Flow => terra_render::ViewportShadingMode::Flow,
                        _ => terra_render::ViewportShadingMode::Lit,
                    }
                };
                let ov = &self.ui_state.viewport_overlays;
                renderer.set_display_aids(terra_render::ViewportDisplayAids {
                    wireframe: ov.wireframe,
                    grid: ov.grid,
                    world_bounds: ov.world_bounds,
                    contours: ov.contours,
                    shading,
                });
                renderer.update_visible_tile_plan(&self.terrain_runtime.pyramid);
            }
            // Frame seam (see TerrainRenderer::render_terrain's contract): this
            // acquires + submits terrain and returns the un-presented frame. The
            // block below must keep that order — build the GUI view from *this*
            // frame, submit exactly one GUI pass (LoadOp::Load) over it, and
            // present only after that submit. Reordering present before the GUI
            // submit shows a stale frame; the compositing tests guard the rest.
            let frame = match renderer.render_terrain() {
                Ok(f) => f,
                // Exhaustive by design (A1-G2): no `_` arm, so a future
                // wgpu::SurfaceError variant — or collapsing RenderError — fails
                // to compile here instead of silently dropping frames forever.
                Err(terra_render::RenderError::Surface(e)) => {
                    match e {
                        wgpu::SurfaceError::Timeout => {
                            // Expected transient stall; present nothing this frame.
                            log::debug!("render: surface acquire timeout; frame skipped");
                        }
                        wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost => {
                            // Swap chain invalid (display re-plug, driver reset,
                            // DPI/fullscreen change without a Resized event).
                            // Reconfigure and schedule the recovery frame — the
                            // on-demand loop won't repaint on its own.
                            log::warn!("render: {e}; reconfiguring surface");
                            renderer.reconfigure();
                            window.request_redraw();
                        }
                        wgpu::SurfaceError::OutOfMemory => {
                            // Unrecoverable; exit cleanly via about_to_wait.
                            log::error!("render: {e}; exiting");
                            self.pending_exit = true;
                        }
                        wgpu::SurfaceError::Other => {
                            log::warn!("render: {e}; frame skipped");
                        }
                    }
                    return;
                }
                Err(e @ terra_render::RenderError::Msg(_)) => {
                    log::warn!("render: {e}");
                    return;
                }
            };
            self.ui_state.profile.render_us = render_t0.elapsed().as_micros() as u64;
            self.ui_state.profile.terrain_grid_size = renderer.last_grid_resolution;
            self.ui_state.profile.gpu_terrain_us = renderer.last_gpu_timings.terrain_us;
            self.ui_state.profile.gpu_shadow_us = renderer.last_gpu_timings.shadow_us;
            self.ui_state.profile.gpu_timestamps_supported = renderer.last_gpu_timings.supported;
            self.ui_state.profile.update_visible_tiles(
                renderer.last_tile_plan_exact,
                renderer.last_tile_plan_fallback,
                renderer.last_tile_plan_missing,
            );
            let progressive_samples = renderer.progressive_samples();
            self.ui_state.progressive_samples = progressive_samples;
            let render_converged =
                progressive_samples >= renderer.quality().config.max_accumulated_spp;
            self.terrain_runtime
                .refinement
                .set_render_converged(render_converged);
            self.ui_state.profile.update_progressive(
                renderer.scene_versions_snapshot(),
                renderer.progressive_last_invalidation(),
                renderer.progressive_accumulation_frame(),
                renderer.global_frame_index(),
                renderer.quality(),
                &renderer.last_gpu_timings,
                renderer.interaction_state(),
                renderer.quality().config.mode,
                progressive_samples,
            );
            self.ui_state.camera_xz = (
                (renderer.camera.target.x / renderer.heights.world_size.0.max(1.0)).clamp(0.0, 1.0),
                (renderer.camera.target.z / renderer.heights.world_size.1.max(1.0)).clamp(0.0, 1.0),
            );
            self.ui_state.camera_yaw = renderer.camera.yaw;
            self.ui_state.camera_pitch = renderer.camera.pitch;
            let view = frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            let (surface_w, surface_h) = renderer.size();
            let screen_w = surface_w as f32 / pixels_per_point;
            let screen_h = surface_h as f32 / pixels_per_point;

            let ui_t0 = Instant::now();
            let hist_fp = self.session.history.ui_fingerprint();
            if hist_fp != self.ui_history_fp {
                self.ui_history_fp = hist_fp;
                self.ui_state.history_descriptions = self.session.history.undo_descriptions();
            }
            let outdated_fp = self.session.outdated_sim_layers.len();
            if outdated_fp != self.ui_outdated_fp
                || self.ui_state.outdated_layer_ids.len() != outdated_fp
            {
                self.ui_outdated_fp = outdated_fp;
                self.ui_state.outdated_layer_ids = self.session.outdated_sim_layers.clone();
            }
            // Soft incomplete-project diagnostics (empty is valid â€” explain, don't block).
            let diag_fp = (self.session.document.stack.nodes.len() as u64)
                ^ ((self.session.document.biome_library.definitions.len() as u64) << 16);
            if diag_fp != self.ui_soft_diag_fp {
                self.ui_soft_diag_fp = diag_fp;
                let diags = terra_core::domain::incomplete_project_diagnostics(
                    &self.session.document.stack,
                    &self.session.document.biome_library,
                    &self.session.document.biome_layers,
                );
                self.ui_state.soft_project_diag = diags.first().map(|d| d.message.clone());
            }
            self.gui_state.layout = self.ui_state.layout.clone();
            if let Some(window) = self.window.as_ref() {
                self.ui_state.window_maximized = window.is_maximized();
            }
            let mut gui = GuiContext::begin(
                screen_w,
                screen_h,
                pixels_per_point,
                GuiInput {
                    pointer,
                    primary_down,
                    secondary_down,
                    scroll_delta,
                    text,
                    backspace_pressed,
                    escape_pressed,
                    enter_pressed,
                    ..Default::default()
                },
                &mut self.gui_state,
            );
            self.viewport_rect = gui.viewport_rect();

            let mut home_actions = Vec::new();
            let ui_out = if self.screen == AppScreen::Home {
                let mut out = crate::ui::FrameUiOutput::default();
                home_actions = draw_project_home(
                    &mut gui,
                    &mut self.project_home,
                    &self.project_prefs,
                    &mut out,
                    self.ui_state.window_maximized,
                );
                out
            } else {
                draw_editor_gui(
                    &mut gui,
                    &self.session.document,
                    &mut self.ui_state,
                    &mut self.chrome_gui,
                    &mut self.tools_gui,
                    &mut self.layers_gui,
                    &mut self.inspector_gui,
                    &mut self.windows_gui,
                    &mut self.dock_gui,
                )
            };

            let discard_choice = if self.pending_project_action.is_some() {
                draw_discard_confirm(&mut gui)
            } else {
                None
            };

            let template_choice =
                if self.show_new_template_picker && self.pending_project_action.is_none() {
                    draw_new_project_templates(
                        &mut gui,
                        &mut self.new_template_selected,
                        &mut self.new_world_settings,
                    )
                } else {
                    None
                };

            if self.ui_state.show_widget_lab && self.screen == AppScreen::Editor {
                gui.widget_lab(&mut self.widget_lab);
            }
            gui.end();
            self.gui_wants_pointer = gui.wants_pointer()
                || (self.screen == AppScreen::Editor && self.ui_state.tool_drag.is_some())
                || self.layers_gui.drag_from.is_some();
            self.gui_interacting = gui.is_interacting()
                || (self.screen == AppScreen::Editor && self.ui_state.tool_drag.is_some())
                || self.layers_gui.drag_from.is_some();
            self.ui_state.profile.ui_us = ui_t0.elapsed().as_micros() as u64;

            let (surface_w, surface_h) = renderer.size();
            gui_renderer.render(
                &gpu.device,
                &gpu.queue,
                &view,
                &mut gui,
                surface_w,
                surface_h,
            );
            frame.present();
            (ui_out, home_actions, discard_choice, template_choice)
        };

        let (ui_out, home_actions, discard_choice, template_choice) = ui_out;

        if let Some(window) = self.window.clone() {
            self.ui_state.window_maximized = window.is_maximized();
            use crate::ui::{UiCursor, WindowResizeEdge};
            use winit::window::{CursorIcon, ResizeDirection};
            let icon = match ui_out.cursor {
                UiCursor::Default => CursorIcon::Default,
                UiCursor::Grab => CursorIcon::Grab,
                UiCursor::Grabbing => CursorIcon::Grabbing,
                UiCursor::NResize => CursorIcon::NResize,
                UiCursor::SResize => CursorIcon::SResize,
                UiCursor::EResize => CursorIcon::EResize,
                UiCursor::WResize => CursorIcon::WResize,
                UiCursor::NeResize => CursorIcon::NeResize,
                UiCursor::NwResize => CursorIcon::NwResize,
                UiCursor::SeResize => CursorIcon::SeResize,
                UiCursor::SwResize => CursorIcon::SwResize,
            };
            window.set_cursor(icon);

            if ui_out.request_window_minimize {
                window.set_minimized(true);
            }
            if ui_out.request_window_toggle_maximize {
                window.set_maximized(!window.is_maximized());
            }
            if ui_out.request_window_close {
                self.pending_exit = true;
            }
            if ui_out.request_window_drag {
                let _ = window.drag_window();
            }
            if let Some(edge) = ui_out.request_window_drag_resize {
                let dir = match edge {
                    WindowResizeEdge::East => ResizeDirection::East,
                    WindowResizeEdge::North => ResizeDirection::North,
                    WindowResizeEdge::NorthEast => ResizeDirection::NorthEast,
                    WindowResizeEdge::NorthWest => ResizeDirection::NorthWest,
                    WindowResizeEdge::South => ResizeDirection::South,
                    WindowResizeEdge::SouthEast => ResizeDirection::SouthEast,
                    WindowResizeEdge::SouthWest => ResizeDirection::SouthWest,
                    WindowResizeEdge::West => ResizeDirection::West,
                };
                let _ = window.drag_resize_window(dir);
            }
        }

        self.ui_state.profile.frame_us = frame_t0.elapsed().as_micros() as u64;

        if let Some(choice) = discard_choice {
            match choice {
                DiscardConfirmChoice::Discard => {
                    if let Some(action) = self.pending_project_action.take() {
                        self.perform_project_action(action);
                    }
                }
                DiscardConfirmChoice::Cancel => {
                    self.pending_project_action = None;
                }
            }
        }

        if let Some(choice) = template_choice {
            match choice {
                NewProjectTemplateChoice::Cancel => {
                    self.show_new_template_picker = false;
                }
                NewProjectTemplateChoice::Create {
                    template_id,
                    world_size_m,
                    sea_level,
                } => {
                    self.new_project_with_template(&template_id, world_size_m, sea_level);
                }
            }
        }

        self.handle_home_actions(home_actions);

        if self.screen == AppScreen::Editor {
            self.apply_actions(ui_out.actions);
            if ui_out.request_undo {
                self.undo();
            }
            if ui_out.request_redo {
                self.redo();
            }
            if ui_out.request_save {
                self.save_current_project();
            }
            if ui_out.request_save_as {
                self.save_project_as();
            }
            if ui_out.request_load_path {
                self.request_project_action(PendingProjectAction::Open);
            }
            if ui_out.request_new_project {
                self.request_project_action(PendingProjectAction::New);
            }
            if ui_out.request_close_project {
                self.request_project_action(PendingProjectAction::Close);
            }
            if ui_out.request_export_path {
                self.choose_export_directory();
            }
            if ui_out.request_start_export {
                self.start_export();
            }
            if ui_out.camera_reset {
                self.frame_terrain();
            }
            if ui_out.camera_top_view {
                if let Some(r) = self.renderer.as_mut() {
                    r.camera_top_view();
                }
            }
            if ui_out.camera_frame_selection {
                if let Some(r) = self.renderer.as_mut() {
                    r.frame_camera_to_selection();
                }
            }
            if ui_out.request_full_build {
                self.scheduler.quality = PreviewQuality::Full;
                self.ui_state.quality = PreviewQuality::Full;
                self.ui_state.profile.quality = "Full (EXPORT)";
                self.request_rebuild();
            }
            if ui_out.request_cancel_build {
                // Bump the eval token so the worker discards the in-flight job.
                self.eval_token = self.eval_token.wrapping_add(1);
                self.eval_worker.set_token(self.eval_token);
                self.ui_state.refining = false;
                self.ui_state.build_progress = None;
                self.ui_state.refining_layer_name = None;
                self.ui_state.status = "Build cancelled".into();
            }
            if ui_out.request_save_bookmark {
                let slot = self
                    .ui_state
                    .bookmarks
                    .iter()
                    .position(|b| b.is_none())
                    .unwrap_or(0);
                self.save_camera_bookmark(slot);
            }
            if let Some(slot) = ui_out.request_save_bookmark_slot {
                self.save_camera_bookmark(slot);
            }
            if let Some(slot) = ui_out.request_recall_bookmark {
                self.recall_camera_bookmark(slot);
            }
        }
        if self.ui_state.layout_dirty {
            self.ui_state.layout.clamp_mut();
            let prefs = EditorPrefs {
                layout: self.ui_state.layout.clone(),
                preferred_workspace: self.ui_state.preferred_workspace.clone(),
                auto_switch_workspace_on_create: self.ui_state.auto_switch_workspace_on_create,
                viewport_render: self.ui_state.viewport_render.to_prefs(),
            };
            save_editor_prefs(&prefs);
            self.ui_state.layout_dirty = false;
            self.refresh_viewport_rect();
        }

        if self.gui_wants_pointer
            || self.pending_project_action.is_some()
            || self.show_new_template_picker
        {
            window.request_redraw();
        }
    }
}
