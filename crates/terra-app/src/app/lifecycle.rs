use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::ui::{resolve_shortcut_for_input, PanelAction, ShortcutChord, ShortcutModifiers};
use terra_core::command::EditorCommand;
use terra_core::eval::PreviewQuality;
use terra_gpu::{GpuTerrainEngine, GpuTileAtlas};
use terra_gui::GuiRenderer;
use terra_render::TerrainRenderer;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use super::helpers::{search_character, ui_tool_search_focused};
use super::{
    quality_in_flight_progress, quality_stage_progress, AppScreen, TerraApp, EDIT_DEBOUNCE_MS,
    PAINT_DEBOUNCE_MS, REFINE_INTERVAL_MS,
};

impl ApplicationHandler for TerraApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Terra")
                        .with_decorations(false)
                        .with_resizable(true)
                        .with_inner_size(winit::dpi::LogicalSize::new(1600, 900)),
                )
                .expect("window"),
        );
        let (gpu, target) =
            pollster::block_on(terra_render::init_gpu(window.clone())).expect("gpu init");
        let renderer = TerrainRenderer::new(&gpu, target);
        let tile_config = self.terrain_runtime.pyramid.config;
        let tile_atlas =
            match GpuTileAtlas::new(&gpu.device, tile_config.tile_size, tile_config.halo, 128) {
                Ok(atlas) => Some(atlas),
                Err(error) => {
                    log::warn!("GPU tile atlas disabled: {error}");
                    None
                }
            };

        let gpu_engine = GpuTerrainEngine::new(&gpu.device, 256);
        let gui_renderer = GuiRenderer::new(&gpu.device, &gpu.queue, gpu.surface_format);
        self.window = Some(window);
        self.renderer = Some(renderer);
        self.gpu = Some(gpu);
        self.gpu_engine = Some(gpu_engine);
        self.tile_atlas = tile_atlas;
        self.gui_renderer = Some(gui_renderer);
        self.refresh_window_title();
        // Decode 1024² tool thumbs on a background pool before the user opens
        // Quick Add / Tools — avoids Lucide→3D icon flash on first dialog open.
        crate::ui::prefetch_tool_thumbnails();
        // Terrain eval waits until the user opens or creates a project.
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let mut want_redraw = false;
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(r) = self.renderer.as_mut() {
                    r.resize(size);
                }
                self.refresh_viewport_rect();
                want_redraw = true;
            }
            WindowEvent::RedrawRequested => self.redraw(),
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    let pressed = event.state == ElementState::Pressed;
                    // Always track fly keys so release isn't missed while UI steals focus.
                    self.camera_keys.set(code, pressed);
                }
                if event.state == ElementState::Pressed {
                    if let PhysicalKey::Code(code) = event.physical_key {
                        let wants_chars = self.gui_state.wants_text_input()
                            || self.ui_state.show_quick_add
                            || self.ui_state.show_command_palette
                            || self.inspector_gui.rename_buffer.is_some()
                            || ui_tool_search_focused(&self.ui_state, &self.gui_state);
                        let bookmark = match code {
                            KeyCode::Digit1 => Some(0usize),
                            KeyCode::Digit2 => Some(1),
                            KeyCode::Digit3 => Some(2),
                            KeyCode::Digit4 => Some(3),
                            KeyCode::Digit5 => Some(4),
                            KeyCode::Digit6 => Some(5),
                            KeyCode::Digit7 => Some(6),
                            KeyCode::Digit8 => Some(7),
                            KeyCode::Digit9 => Some(8),
                            _ => None,
                        };
                        if let Some(index) = bookmark {
                            if self.screen == AppScreen::Editor {
                                if self.modifiers_ctrl && !self.modifiers_alt {
                                    self.save_camera_bookmark(index);
                                } else if self.modifiers_alt {
                                    self.recall_camera_bookmark(index);
                                }
                            }
                        }
                        let chord = ShortcutChord::new(
                            code,
                            ShortcutModifiers {
                                ctrl: self.modifiers_ctrl,
                                shift: self.modifiers_shift,
                                alt: self.modifiers_alt,
                                super_key: self.modifiers_super,
                            },
                        );
                        if let Some(command) = resolve_shortcut_for_input(chord, wants_chars) {
                            self.dispatch_command(command);
                        }
                        match code {
                            KeyCode::Backspace
                                if self.gui_state.wants_text_input()
                                    || self.ui_state.show_quick_add
                                    || self.ui_state.show_command_palette
                                    || !self.ui_state.tool_search.is_empty()
                                    || self.inspector_gui.rename_buffer.is_some() =>
                            {
                                self.gui_backspace = true;
                            }
                            KeyCode::Escape
                                if self.gui_state.wants_text_input()
                                    || self.ui_state.show_quick_add
                                    || self.ui_state.show_command_palette
                                    || self.ui_state.viewport_context_menu.is_some()
                                    || self.inspector_gui.rename_buffer.is_some()
                                    || self.pending_project_action.is_some()
                                    || self.show_new_template_picker
                                    || self.ui_state.is_mask_view() =>
                            {
                                if self.ui_state.viewport_context_menu.is_some() {
                                    self.ui_state.viewport_context_menu = None;
                                } else {
                                    self.gui_escape = true;
                                }
                            }
                            KeyCode::Enter | KeyCode::NumpadEnter
                                if self.gui_state.wants_text_input()
                                    || self.inspector_gui.rename_buffer.is_some() =>
                            {
                                self.gui_enter = true;
                            }
                            KeyCode::Enter | KeyCode::NumpadEnter
                                if self.ui_state.biome_paint_tool
                                    == terra_core::biome_paint::BiomePaintTool::PolygonFill
                                    && self.biome_polygon_points.len() >= 3
                                    && self.session.document.active_biome.is_some() =>
                            {
                                self.commit_biome_polygon_fill();
                            }
                            _ => {}
                        }
                        if !self.modifiers_ctrl && wants_chars {
                            if let Some(ch) = search_character(code, self.modifiers_shift) {
                                self.gui_text.push(ch);
                            }
                        }
                        want_redraw = true;
                    }
                }
            }
            WindowEvent::ModifiersChanged(m) => {
                self.modifiers_shift = m.state().shift_key();
                self.modifiers_alt = m.state().alt_key();
                self.modifiers_ctrl = m.state().control_key();
                self.modifiers_super = m.state().super_key();
                self.ui_state.shift_context = self.modifiers_shift;
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let in_viewport = self.cursor_in_viewport();
                let painting = button == MouseButton::Left
                    && self.viewport_paint_active()
                    && in_viewport
                    && !self.modifiers_alt
                    && !self.gui_wants_pointer;
                if state == ElementState::Pressed {
                    self.mouse_pressed = Some(button);
                    self.mouse_press_cursor = self.last_cursor;
                    if button == MouseButton::Right {
                        self.right_drag_distance = 0.0;
                    }
                    if painting {
                        self.paint_at_cursor();
                    } else if button == MouseButton::Left
                        && self.ui_state.editor_tool.is_place_point()
                        && in_viewport
                        && !self.modifiers_alt
                        && !self.gui_wants_pointer
                    {
                        self.place_point_at_cursor();
                    } else if button == MouseButton::Left
                        && in_viewport
                        && !self.modifiers_alt
                        && !self.gui_wants_pointer
                    {
                        if self.ui_state.editor_tool.is_move()
                            && matches!(
                                self.ui_state.app_workspace,
                                crate::ui::AppWorkspace::Layout | crate::ui::AppWorkspace::Review
                            )
                        {
                        } else {
                            let _ = self.try_pick_shape_at_cursor();
                        }
                    }
                } else if state == ElementState::Released {
                    let press_pos = self.mouse_press_cursor.take();
                    let was_right = button == MouseButton::Right;
                    let right_drag = self.right_drag_distance;
                    self.mouse_pressed = None;
                    if was_right
                        && in_viewport
                        && !self.gui_wants_pointer
                        && right_drag < 6.0
                        && self.screen != AppScreen::Home
                    {
                        let (sx, sy) = self.cursor_logical().unwrap_or((0.0, 0.0));
                        let uv = self.pick_paint_uv();
                        self.ui_state.viewport_context_menu =
                            Some(crate::ui::ViewportContextMenu {
                                x: sx,
                                y: sy,
                                uv,
                                locked_owner: None,
                                picking_owner_for: None,
                                owner_override: None,
                            });
                        let _ = press_pos;
                    }
                    if button == MouseButton::Left {
                        self.end_shape_point_drag();
                        self.end_layer_point_drag();
                        let was_biome_paint = self.ui_state.editor_tool
                            == crate::ui::EditorTool::PaintBiome
                            && self.last_paint_uv.is_some();
                        self.last_paint_uv = None;
                        if was_biome_paint {
                            self.apply_actions(vec![PanelAction::EndBiomePaintStroke]);
                        }
                        // Mask paint: commit stroke undo; defer/coalesce terrain rebuild.
                        if self.ui_state.editor_tool == crate::ui::EditorTool::PaintMask {
                            self.commit_mask_paint_stroke();
                        }
                        if self.sculpt_stroke_active {
                            // Non-destructive Shape history â€” undoable via layer/command history later.
                            self.session.history.push_executed(EditorCommand::Annotate {
                                label: "Shape stroke (draft â†’ full on refine)".into(),
                            });
                            self.sculpt_stroke_active = false;
                            // Mark dependents outdated without forcing sim rebuilds now.
                            self.mark_shape_dependents_outdated();
                            self.ui_state.shape_commit_full = true;
                            // One final Draft; full quality follows when requested / idle refine.
                            self.flush_live_paint_preview();
                        }
                    }
                }
                want_redraw = true;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let previous = self.last_cursor;
                self.last_cursor = Some((position.x, position.y));
                if self.dragging_shape_point.is_some()
                    && self.mouse_pressed == Some(MouseButton::Left)
                {
                    self.update_shape_point_drag();
                }
                if self.dragging_layer_point.is_some()
                    && self.mouse_pressed == Some(MouseButton::Left)
                {
                    self.update_layer_point_drag();
                    want_redraw = true;
                }
                let painting = self.mouse_pressed == Some(MouseButton::Left)
                    && self.viewport_paint_active()
                    && self.cursor_in_viewport()
                    && !self.modifiers_alt
                    && (!self.gui_wants_pointer || self.sculpt_stroke_active);
                if painting {
                    self.paint_at_cursor();
                    want_redraw = true;
                } else if self.mouse_pressed.is_some() && self.viewport_camera_active() {
                    if let (Some(btn), Some((lx, ly)), Some(r)) =
                        (self.mouse_pressed, previous, self.renderer.as_mut())
                    {
                        let dx = (position.x - lx) as f32;
                        let dy = (position.y - ly) as f32;
                        match btn {
                            MouseButton::Left => {
                                let speed = self.ui_state.camera_speed.max(0.05);
                                let dx = dx * speed;
                                let dy = dy * speed;
                                // Game-engine look: rotate around the camera eye.
                                // Alt+LMB keeps classic orbit around the look-at target
                                // (Alt already unlocks camera while a brush is armed).
                                if self.modifiers_alt {
                                    r.camera.orbit(dx, dy);
                                } else {
                                    r.camera.look(dx, dy);
                                }
                                r.camera.clamp_to_world(r.heights.world_size);
                            }
                            MouseButton::Right | MouseButton::Middle => {
                                let speed = self.ui_state.camera_speed.max(0.05);
                                let dx = dx * speed;
                                let dy = dy * speed;
                                if btn == MouseButton::Right {
                                    self.right_drag_distance += dx.abs() + dy.abs();
                                }
                                r.camera.pan(dx, dy);
                                r.camera.clamp_to_world(r.heights.world_size);
                            }
                            _ => {}
                        }
                        want_redraw = true;
                    }
                } else if self.viewport_paint_tool_armed() && self.cursor_in_viewport() {
                    // Keep brush gizmo tracking the cursor.
                    want_redraw = true;
                } else if self.gui_wants_pointer || self.screen == AppScreen::Home {
                    // Home is all chrome under ControlFlow::Wait: without a move redraw,
                    // hover/`hot` never establishes (chicken-and-egg with gui_wants_pointer).
                    want_redraw = true;
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let d = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
                // Prefer terra-gui scroll over left/right chrome (or any hot widget).
                let over_chrome = self.cursor_logical().is_some_and(|(x, y)| {
                    let vp = &self.viewport_rect;
                    y < vp.min_y || y >= vp.max_y || x < vp.min_x || x >= vp.max_x
                });
                if over_chrome || self.gui_wants_pointer {
                    self.gui_scroll_delta += d;
                    want_redraw = true;
                } else if self.viewport_paint_tool_armed() && !self.modifiers_alt {
                    // Brush tools: wheel adjusts radius; Alt falls through to zoom.
                    self.ui_state.ensure_sculpt_defaults();
                    let step = if d.abs() >= 1.0 {
                        d.signum() * 0.008
                    } else {
                        d * 0.008
                    };
                    self.ui_state.sculpt_radius =
                        (self.ui_state.sculpt_radius + step).clamp(0.005, 0.25);
                    want_redraw = true;
                } else if self.viewport_camera_active() {
                    if let Some(r) = self.renderer.as_mut() {
                        let speed = self.ui_state.camera_speed.max(0.05);
                        r.camera.zoom(d * 40.0 * speed);
                        want_redraw = true;
                    }
                }
            }
            _ => {}
        }

        if want_redraw {
            if let Some(w) = &self.window {
                w.request_redraw();
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.pending_exit {
            event_loop.exit();
            return;
        }
        self.exporter.poll();
        self.poll_project_io();
        let camera_flying = self.apply_camera_fly();
        let tool_thumbnail_ready = crate::ui::take_tool_thumbnail_ready_signal();
        let tool_thumbnails_pending = crate::ui::tool_thumbnails_pending();
        let mut export_busy = !self.exporter.job.done;
        let project_io_busy = self.project_io.is_busy();
        if export_busy {
            self.ui_state.export_progress = Some(self.exporter.job.progress);
            self.ui_state.status = format!("Export {:.0}%", self.exporter.job.progress * 100.0);
        } else if let Some(result) = self.exporter.job.result.take() {
            self.ui_state.export_progress = None;
            self.terrain_runtime
                .refinement
                .finish_export(self.runtime_started.elapsed().as_millis() as u64);
            match result {
                Ok(res) => {
                    self.ui_state.status = format!("Exported {}", res.height_path.display());
                }
                Err(err) => {
                    self.ui_state.status = format!("Export failed: {err}");
                }
            }
            export_busy = true; // one more frame to show status
        } else {
            self.ui_state.export_progress = None;
        }

        // Stall heavy eval during camera orbit / UI drag â€” never during an active
        // sculpt/paint stroke (GUI overlays must not block live height updates).
        let mut did_eval = false;
        let mut live_paint = false;
        let mut work_pending = export_busy || project_io_busy;

        if self.screen == AppScreen::Editor {
            let mask_painting = self.ui_state.editor_tool == crate::ui::EditorTool::PaintMask
                && self.mouse_pressed == Some(MouseButton::Left)
                && self.ui_state.paint_mask.is_some()
                && !self.modifiers_alt;
            let live_geometry =
                self.dragging_shape_point.is_some() || self.dragging_layer_point.is_some();
            live_paint = self.sculpt_stroke_active
                || live_geometry
                || (self.mouse_pressed == Some(MouseButton::Left)
                    && self.viewport_paint_active()
                    && !self.modifiers_alt
                    && !mask_painting);
            // Stall Draft while the user is actively dragging UI / holding a button â€”
            // mere hover over panels must not block rebuilds or progressive refine.
            // Also stall during mask paint so overlay stays responsive (no height rebuild).
            let stall_draft = if mask_painting {
                true
            } else if live_paint {
                false
            } else {
                // Slider drags still get debounced Draft feedback. Camera gestures
                // remain stalled, while Full refinement waits for all interaction.
                self.mouse_pressed.is_some() && !self.gui_interacting
            };
            let stall_refine = self.mouse_pressed.is_some() || self.gui_interacting;
            let now_ms = self.runtime_started.elapsed().as_millis() as u64;
            let scene_meaningful = self
                .renderer
                .as_ref()
                .map(|r| r.scene_versions().meaningful_this_frame())
                .unwrap_or(false);
            let terrain_edits = self.pending_eval
                || self.needs_height_upload
                || self.placement_tint_dirty
                || self.mask_overlay_dirty;
            let meaningful_interaction = scene_meaningful || live_paint || terrain_edits;
            self.terrain_runtime
                .update_refinement(now_ms, meaningful_interaction);
            if let Some(renderer) = self.renderer.as_mut() {
                renderer.set_interaction_state(self.terrain_runtime.refinement.state());
            }
            if let Some(engine) = self.gpu_engine.as_mut() {
                engine.set_simulation_iteration_cap(
                    self.terrain_runtime
                        .refinement
                        .state()
                        .simulation_iteration_cap(),
                );
            }
            // The worker is never awaited: consume its newest matching result, if available.
            if let Some(result) = self.eval_worker.try_recv_matching(self.eval_token) {
                let quality = result.quality;
                let height = result.height;
                self.scheduler.last_aux = result.aux;
                self.scheduler.last_strata = result.strata;
                self.scheduler.last_layer_timings = result.layer_timings;
                self.ui_state
                    .profile
                    .update_layer_timings(&self.scheduler.last_layer_timings);
                let height = std::sync::Arc::new(height);
                self.scheduler.last_good = Some(std::sync::Arc::clone(&height));
                self.last_height = Some((*height).clone());
                // Ingest every CPU layer checkpoint so GPU can bake unsupported shapes
                // and keep EffectFilters live on the next edit.
                if let (Some(engine), Some(gpu)) = (self.gpu_engine.as_mut(), self.gpu.as_ref()) {
                    let preview = self.session.document.preview_eval_stack();
                    for layer in preview.flatten_layers() {
                        if let Some(cached) = self.scheduler.evaluator.cache.get(layer.id()) {
                            if cached.dirty {
                                continue;
                            }
                            if cached.height.metrics.width == 0 {
                                continue;
                            }
                            let (lo, hi) = cached.height.min_max();
                            engine.ingest_height(
                                &gpu.device,
                                &gpu.queue,
                                layer.id(),
                                &cached.height,
                                (lo, hi),
                            );
                        }
                    }
                }
                self.queue_final_tile_uploads();
                self.preview_dirty = true;
                self.needs_height_upload = true;
                self.worker_refine_pending = false;
                self.ui_state.profile.eval_us = result.eval_us;
                self.ui_state.profile.tex_w = self.last_height.as_ref().unwrap().metrics.width;
                self.ui_state.profile.tex_h = self.last_height.as_ref().unwrap().metrics.height;
                self.ui_state.profile.tiles_x =
                    self.last_height.as_ref().unwrap().metrics.tiles_x();
                self.ui_state.profile.tiles_z =
                    self.last_height.as_ref().unwrap().metrics.tiles_z();
                self.ui_state.profile.path = "CPU (async)";
                self.ui_state.profile.quality = match quality {
                    PreviewQuality::Draft => "Draft (fast)",
                    PreviewQuality::Medium => "Medium",
                    PreviewQuality::Full => "Final (viewport)",
                    PreviewQuality::Export => "Export quality",
                };
                self.ui_state.quality = quality;
                self.ui_state.build_progress = Some(quality_stage_progress(quality));
                self.ui_state.draft_displayed =
                    matches!(quality, PreviewQuality::Draft | PreviewQuality::Medium);
                self.ui_state.refining = quality.next_refine().is_some();
                if !self.ui_state.refining {
                    self.ui_state.build_progress = None;
                    self.ui_state.refining_layer_name = None;
                    self.ui_state.draft_displayed = false;
                }
                self.last_refine = Instant::now();
                did_eval = true;
            }

            // Debounced draft eval. During paint/sculpt, key off last eval time â€” stamps
            // keep resetting last_edit, which would otherwise starve live updates.
            if !stall_draft && self.pending_eval {
                let edit_ms = self.session.rebuild_feedback.prefs.edit_debounce_ms.max(1) as u128;
                let live_ok = self.session.rebuild_feedback.prefs.live_preview;
                let ready = if live_paint {
                    self.last_refine.elapsed().as_millis() >= PAINT_DEBOUNCE_MS
                } else if !live_ok {
                    false
                } else {
                    self.last_edit.elapsed().as_millis() >= edit_ms
                };
                if ready {
                    self.pending_eval = false;
                    self.force_draft = true;
                    self.run_eval_step();
                    self.last_refine = Instant::now();
                    did_eval = true;
                }
            }

            // Expensive physics: only when automatic rebuild is on, debounce elapsed,
            // and the artist is not actively sculpting.
            {
                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                if self
                    .session
                    .rebuild_feedback
                    .should_rebuild_physics(now_ms, self.sculpt_stroke_active)
                    && !self.session.outdated_sim_layers.is_empty()
                {
                    let ids = terra_core::rebuild_feedback::rebuild_affected(&mut self.session);
                    if !ids.is_empty()
                        && !terra_core::rebuild_feedback::is_redundant_rebuild(&self.session, &ids)
                    {
                        for id in &ids {
                            self.mark_dirty_from(*id);
                        }
                        self.request_rebuild();
                        self.ui_state.status =
                            format!("Auto-rebuilding {} physics layer(s)", ids.len());
                    }
                    self.session.rebuild_feedback.clear_physics_due();
                }
            }

            // Progressive refine — one quality step per interval, never while interacting.
            // WC path: stay GPU-resident when the last Draft/Medium was fully_gpu.
            // CPU worker is only for unsupported suffixes / export oracle.
            if !stall_refine
                && self.ui_state.refining
                && !self.pending_eval
                && !self.worker_refine_pending
                && self.last_refine.elapsed().as_millis() >= REFINE_INTERVAL_MS
            {
                if self.scheduler.advance_quality() {
                    // HD preview: skip Medium so Camera/Zone sees Full carve sooner.
                    if !matches!(
                        self.session.document.level_steps.high_detail,
                        terra_core::analyze::HighDetailMode::None
                    ) && matches!(self.scheduler.quality, PreviewQuality::Medium)
                    {
                        if let Some(next) = self.scheduler.quality.next_refine() {
                            self.scheduler.quality = next;
                        }
                    }
                    // Keep Zone rect synced under the camera for Camera HD mode.
                    if matches!(
                        self.session.document.level_steps.high_detail,
                        terra_core::analyze::HighDetailMode::Camera
                    ) {
                        if let Some(r) = self.renderer.as_ref() {
                            let u = (r.camera.target.x / r.heights.world_size.0.max(1e-3))
                                .clamp(0.05, 0.95);
                            let v = (r.camera.target.z / r.heights.world_size.1.max(1e-3))
                                .clamp(0.05, 0.95);
                            let half = 0.18;
                            self.session.document.level_steps.hd_zone = [
                                (u - half).max(0.0),
                                (v - half).max(0.0),
                                (u + half).min(1.0),
                                (v + half).min(1.0),
                            ];
                        }
                    }
                    self.ui_state.quality = self.scheduler.quality;
                    // Always prefer GPU-resident refine when an engine exists. Hybrid stacks
                    // still present Draft/Medium on GPU; run_eval_step enqueues CPU only for
                    // unsupported bake correction — never block the viewport on the worker.
                    if self.gpu_engine.is_some() {
                        self.run_eval_step();
                    } else {
                        self.enqueue_refine_job();
                    }
                    // Show in-flight progress toward the queued quality (not frozen at prior stage).
                    self.ui_state.build_progress =
                        Some(quality_in_flight_progress(self.scheduler.quality, 0.0));
                    did_eval = true;
                } else {
                    self.ui_state.refining = false;
                    self.ui_state.quality = PreviewQuality::Full;
                    self.ui_state.build_progress = None;
                    self.ui_state.refining_layer_name = None;
                }
                self.last_refine = Instant::now();
            }

            // Keep the dock bar moving while Medium/Full run on the worker.
            if self.worker_refine_pending {
                let t = self.last_refine.elapsed().as_secs_f32();
                self.ui_state.build_progress =
                    Some(quality_in_flight_progress(self.scheduler.quality, t));
                did_eval = true;
            }
            if self.upload_pending_terrain_tiles() > 0 {
                did_eval = true;
            }

            work_pending = self.pending_eval
                || self.worker_refine_pending
                || self.ui_state.refining
                || !self.pending_tile_uploads.is_empty()
                || export_busy
                || project_io_busy;
        }

        if live_paint && self.pending_eval {
            // Keep pumping the event loop so Draft can flush between mouse moves.
            event_loop.set_control_flow(ControlFlow::Poll);
        } else if camera_flying {
            // Smooth WASD fly while keys are held.
            event_loop.set_control_flow(ControlFlow::Poll);
        } else if self.worker_refine_pending || tool_thumbnails_pending {
            // Wake often enough to animate progress and pick up the worker result.
            event_loop.set_control_flow(ControlFlow::WaitUntil(
                Instant::now() + Duration::from_millis(16),
            ));
        } else if work_pending {
            let wait_ms = if self.pending_eval {
                if live_paint {
                    PAINT_DEBOUNCE_MS
                        .saturating_sub(self.last_refine.elapsed().as_millis())
                        .max(1)
                } else {
                    EDIT_DEBOUNCE_MS
                        .saturating_sub(self.last_edit.elapsed().as_millis())
                        .max(1)
                }
            } else {
                REFINE_INTERVAL_MS
                    .saturating_sub(self.last_refine.elapsed().as_millis())
                    .max(1)
            };
            event_loop.set_control_flow(ControlFlow::WaitUntil(
                Instant::now() + Duration::from_millis(wait_ms as u64),
            ));
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
        }

        if did_eval
            || export_busy
            || project_io_busy
            || self.needs_height_upload
            || tool_thumbnail_ready
            || camera_flying
        {
            if let Some(w) = &self.window {
                w.request_redraw();
            }
        }
    }
}

impl TerraApp {
    /// Apply WASD/QE fly when the viewport camera is active and UI isn't capturing text.
    /// Returns true while keys are held (even if movement was gated this frame).
    fn apply_camera_fly(&mut self) -> bool {
        if !self.camera_keys.any() {
            self.last_camera_move = Instant::now();
            return false;
        }
        let text_capture = self.gui_state.wants_text_input()
            || self.ui_state.show_quick_add
            || self.ui_state.show_command_palette
            || self.inspector_gui.rename_buffer.is_some()
            || ui_tool_search_focused(&self.ui_state, &self.gui_state);
        if self.screen != AppScreen::Editor
            || text_capture
            || self.modifiers_ctrl
            || !self.viewport_camera_fly_active()
        {
            // Keep dt fresh so the first unlocked frame doesn't jump.
            self.last_camera_move = Instant::now();
            return self.camera_keys.any();
        }
        let now = Instant::now();
        let dt = now
            .duration_since(self.last_camera_move)
            .as_secs_f32()
            .clamp(0.0, 0.05);
        self.last_camera_move = now;
        if dt <= 0.0 {
            return true;
        }
        let forward = (self.camera_keys.w as i8 - self.camera_keys.s as i8) as f32;
        let right = (self.camera_keys.d as i8 - self.camera_keys.a as i8) as f32;
        let up = (self.camera_keys.e as i8 - self.camera_keys.q as i8) as f32;
        if let Some(r) = self.renderer.as_mut() {
            let speed = self.ui_state.camera_speed.max(0.05);
            r.camera
                .fly(forward, right, up, dt * speed, self.modifiers_shift);
            r.camera.clamp_to_world(r.heights.world_size);
        }
        true
    }
}
