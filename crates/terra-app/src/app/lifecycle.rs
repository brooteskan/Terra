use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::startup::{self, StartupError};
use crate::ui::{
    resolve_shortcut_for_input, PanelAction, ShortcutChord, ShortcutModifiers,
    TerrainPreviewFreshness,
};
use terra_core::command::EditorCommand;
use terra_core::quality::PreviewQuality;
use terra_cpu_eval::EvalWorkerEvent;
use terra_gpu::GpuTileAtlas;
use terra_gpu_eval::{GpuEvaluationIntent, GpuTerrainEngine};
use terra_gui::{Color, GuiContext, GuiInput, GuiRenderer, GuiState, Rect};
use terra_render::{GpuContext, TerrainRenderer};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use super::frame_trace::FrameTraceEventKind;
use super::helpers::{search_character, ui_tool_search_focused};
use super::input::{InputEvent, InputModifiers, PointerCancelReason};
use super::logical_frame::{
    EditGeneration, FrameDeadlineKind, FrameIdentity, FramePhase, FrameRequestReason, FrameWake,
    LogicalFrameId,
};
use super::{
    quality_in_flight_progress, quality_stage_progress, AppScreen, RuntimeEvent, TerraApp,
    FULL_FIELD_REFINE_MS, POST_INPUT_REFINE_GRACE_MS, REFINE_INTERVAL_MS,
};

impl TerraApp {
    pub(crate) fn request_app_frame(&mut self, reason: FrameRequestReason) {
        self.logical_frames
            .request(EditGeneration::new(self.eval_token), reason);
    }

    pub(crate) fn record_frame_event(&mut self, kind: FrameTraceEventKind) {
        let identity = self
            .logical_frames
            .active_identity()
            .or_else(|| self.logical_frames.pending_identity());
        self.frame_trace.record(
            Instant::now(),
            kind,
            identity,
            self.logical_frames.active_phase(),
            None,
            None,
            None,
            None,
        );
    }

    fn capture_surface_resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.pending_surface_resize = Some(size);
        self.request_app_frame(FrameRequestReason::Resize);
        self.record_frame_event(FrameTraceEventKind::ResizeCaptured);
    }

    fn handle_runtime_event(&mut self, event: RuntimeEvent) {
        match event {
            RuntimeEvent::DeviceLost { reason, message }
                if reason != wgpu::DeviceLostReason::Destroyed =>
            {
                log::error!("GPU device lost ({reason:?}): {message}");
                self.ui_state.status = format!("GPU device lost: {message}");
                self.logical_frames
                    .request_shutdown(EditGeneration::new(self.eval_token));
                self.record_frame_event(FrameTraceEventKind::DeviceLost);
            }
            RuntimeEvent::DeviceLost { .. } => {}
        }
    }

    fn queue_input(&mut self, event: InputEvent) {
        let now = Instant::now();
        let generation = EditGeneration::new(self.eval_token);
        self.input.record(event, now);
        self.request_app_frame(FrameRequestReason::Input);
        let identity = self.logical_frames.pending_identity();
        self.frame_trace.record(
            now,
            FrameTraceEventKind::OsInputReceipt,
            identity,
            Some(FramePhase::CollectingInput),
            None,
            None,
            None,
            None,
        );
        if matches!(
            event,
            InputEvent::PointerButton {
                state: ElementState::Pressed,
                button: MouseButton::Left,
            }
        ) && self.frame_trace.note_follow_up_press(now)
        {
            self.frame_trace.record(
                now,
                FrameTraceEventKind::FollowUpPressReceipt,
                identity,
                Some(FramePhase::CollectingInput),
                None,
                None,
                None,
                None,
            );
        }
        if matches!(
            event,
            InputEvent::PointerButton {
                button: MouseButton::Left,
                ..
            }
        ) {
            self.logical_frames.schedule_deadline(
                FrameDeadlineKind::OptionalRefinement,
                generation,
                now + Duration::from_millis(POST_INPUT_REFINE_GRACE_MS),
            );
        }
    }

    /// Seal and replay one immutable input snapshot. Events queued after `seal`
    /// remain in the accumulator and already own a distinct follow-up frame ID.
    fn process_pending_input_frame(&mut self) -> bool {
        if !self.input.has_pending() {
            return false;
        }
        if !self.logical_frames.has_pending() {
            self.logical_frames.request(
                EditGeneration::new(self.eval_token),
                FrameRequestReason::Input,
            );
        }
        let snapshot = self.input.seal();
        let event_count = snapshot.len();
        let pointer_samples = snapshot.pointer_sample_count();
        let first_primary_press = snapshot.first_primary_press_receipt();
        self.logical_frames
            .begin(Instant::now(), event_count, pointer_samples);
        self.logical_frames.transition(FramePhase::SealingInput);
        self.logical_frames
            .transition(FramePhase::ApplicationUpdate);
        let first_receipt = snapshot.events().first().map(|event| event.received_at());
        let last_sequence = snapshot.events().last().map(|event| event.sequence());
        let mut want_redraw = false;
        for event in snapshot.into_events() {
            want_redraw |= self.apply_input_event(event.event());
        }
        self.logical_frames
            .update_generation(EditGeneration::new(self.eval_token));
        let identity = self.logical_frames.active_identity();
        self.frame_trace.record(
            Instant::now(),
            FrameTraceEventKind::SnapshotSealed,
            identity,
            Some(FramePhase::SealingInput),
            None,
            None,
            None,
            first_receipt.map(|receipt| receipt.elapsed()),
        );
        self.frame_trace.record(
            Instant::now(),
            FrameTraceEventKind::ToolUpdateComplete,
            identity,
            Some(FramePhase::ApplicationUpdate),
            None,
            None,
            None,
            None,
        );
        if let Some(received_at) = first_primary_press {
            let generation = EditGeneration::new(self.eval_token);
            self.frame_trace.note_input_receipt(received_at, generation);
            self.frame_trace.record(
                Instant::now(),
                FrameTraceEventKind::FollowUpPressSealed,
                identity,
                Some(FramePhase::ApplicationUpdate),
                None,
                None,
                None,
                Some(received_at.elapsed()),
            );
        }
        self.logical_frames
            .transition(FramePhase::RequiredInteractiveWork);
        if let (Some(identity), Some(received_at), Some(sequence)) = (
            self.logical_frames.active_identity(),
            first_receipt,
            last_sequence,
        ) {
            log::debug!(
                target: "terra_app::logical_frame",
                "frame={} generation={} sealed_events={} pointer_samples={} last_sequence={} receipt_to_seal_us={}",
                identity.id.get(),
                identity.generation.get(),
                event_count,
                pointer_samples,
                sequence,
                received_at.elapsed().as_micros()
            );
        }
        want_redraw || event_count > 0
    }

    fn begin_scheduled_frame_if_needed(&mut self) {
        if self.logical_frames.active_identity().is_some() {
            return;
        }
        let known_work = self.pending_eval
            || self.worker_refine_pending
            || self.refinement_job.is_some()
            || self.ui_state.refining
            || self.deferred_full_field.is_some()
            || !self.pending_tile_uploads.is_empty()
            || !self.pending_ui_effects.is_empty()
            || self.pending_surface_resize.is_some()
            || self.logical_frames.has_pending();
        if !known_work {
            return;
        }
        if !self.logical_frames.has_pending() {
            let reason = if self.pending_eval {
                FrameRequestReason::RequiredEvaluation
            } else if self.refinement_job.is_some() || self.ui_state.refining {
                FrameRequestReason::OptionalRefinement
            } else {
                FrameRequestReason::Animation
            };
            self.request_app_frame(reason);
        }
        self.logical_frames.begin(Instant::now(), 0, 0);
        self.logical_frames
            .transition(FramePhase::RequiredInteractiveWork);
    }

    fn apply_input_event(&mut self, event: InputEvent) -> bool {
        match event {
            InputEvent::Keyboard { code, state } => self.apply_keyboard_input(code, state),
            InputEvent::Modifiers(modifiers) => {
                self.modifiers_shift = modifiers.shift;
                self.modifiers_alt = modifiers.alt;
                self.modifiers_ctrl = modifiers.ctrl;
                self.modifiers_super = modifiers.super_key;
                self.ui_state.shift_context = modifiers.shift;
                true
            }
            InputEvent::PointerButton { state, button } => self.apply_pointer_button(state, button),
            InputEvent::PointerMoved { x, y } => self.apply_pointer_motion(x, y),
            InputEvent::Wheel { delta } => self.apply_wheel(delta),
            InputEvent::Focused(focused) => {
                if !focused {
                    self.camera_keys = super::CameraKeys::default();
                }
                true
            }
            InputEvent::CursorEntered | InputEvent::CursorLeft => true,
            InputEvent::PointerCancelled(
                PointerCancelReason::FocusLost | PointerCancelReason::CaptureLost,
            ) => self.cancel_pointer_gesture(),
        }
    }

    fn apply_keyboard_input(&mut self, code: Option<KeyCode>, state: ElementState) -> bool {
        let Some(code) = code else {
            return false;
        };
        let pressed = state == ElementState::Pressed;
        self.camera_keys.set(code, pressed);
        if !pressed {
            return false;
        }

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
        true
    }

    fn apply_pointer_button(&mut self, state: ElementState, button: MouseButton) -> bool {
        match (button, state) {
            (MouseButton::Left, ElementState::Pressed) => self.gui_primary_pressed = true,
            (MouseButton::Left, ElementState::Released) => self.gui_primary_released = true,
            (MouseButton::Right, ElementState::Pressed) => self.gui_secondary_pressed = true,
            (MouseButton::Right, ElementState::Released) => self.gui_secondary_released = true,
            _ => {}
        }
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
                && !(self.ui_state.editor_tool.is_move()
                    && matches!(
                        self.ui_state.app_workspace,
                        crate::ui::AppWorkspace::Layout | crate::ui::AppWorkspace::Review
                    ))
            {
                let _ = self.try_pick_shape_at_cursor();
            }
        } else {
            self.finish_pointer_release(button, in_viewport, false);
        }
        true
    }

    fn finish_pointer_release(&mut self, button: MouseButton, in_viewport: bool, cancelled: bool) {
        let press_pos = self.mouse_press_cursor.take();
        let was_right = button == MouseButton::Right;
        let right_drag = self.right_drag_distance;
        self.mouse_pressed = None;
        if !cancelled
            && was_right
            && in_viewport
            && !self.gui_wants_pointer
            && right_drag < 6.0
            && self.screen != AppScreen::Home
        {
            let (sx, sy) = self.cursor_logical().unwrap_or((0.0, 0.0));
            let uv = self.pick_paint_uv();
            self.ui_state.viewport_context_menu = Some(crate::ui::ViewportContextMenu {
                x: sx,
                y: sy,
                uv,
                locked_owner: None,
                picking_owner_for: None,
                owner_override: None,
            });
            let _ = press_pos;
        }
        if button != MouseButton::Left {
            return;
        }
        self.end_shape_point_drag();
        self.end_layer_point_drag();
        let was_biome_paint = self.ui_state.editor_tool == crate::ui::EditorTool::PaintBiome
            && self.last_paint_uv.is_some();
        self.last_paint_uv = None;
        if was_biome_paint {
            self.apply_actions(vec![PanelAction::EndBiomePaintStroke]);
        }
        if self.ui_state.editor_tool == crate::ui::EditorTool::PaintMask {
            self.commit_mask_paint_stroke();
        }
        if self.sculpt_stroke_active {
            self.session.history.push_executed(EditorCommand::Annotate {
                label: "Shape stroke (draft → full on refine)".into(),
            });
            self.sculpt_stroke_active = false;
            self.mark_shape_dependents_outdated();
            self.ui_state.shape_commit_full = true;
            // The required-work phase below performs the final Draft. Release is
            // intentionally bounded to state finalization and a work request.
            if self.pending_eval {
                self.force_draft = true;
            }
            let now = Instant::now();
            let generation = EditGeneration::new(self.eval_token);
            self.logical_frames.schedule_deadline(
                FrameDeadlineKind::OptionalRefinement,
                generation,
                now + Duration::from_millis(POST_INPUT_REFINE_GRACE_MS),
            );
            self.frame_trace.note_release(now, generation);
            self.frame_trace.record(
                now,
                FrameTraceEventKind::StrokeRelease,
                self.logical_frames.active_identity(),
                self.logical_frames.active_phase(),
                None,
                None,
                None,
                None,
            );
        }
    }

    fn cancel_pointer_gesture(&mut self) -> bool {
        if let Some(button) = self.mouse_pressed {
            self.finish_pointer_release(button, false, true);
        }
        self.camera_keys = super::CameraKeys::default();
        true
    }

    fn apply_pointer_motion(&mut self, x: f64, y: f64) -> bool {
        let previous = self.last_cursor;
        self.last_cursor = Some((x, y));
        let mut want_redraw = false;
        if self.dragging_shape_point.is_some() && self.mouse_pressed == Some(MouseButton::Left) {
            self.update_shape_point_drag();
        }
        if self.dragging_layer_point.is_some() && self.mouse_pressed == Some(MouseButton::Left) {
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
            if let (Some(button), Some((last_x, last_y)), Some(renderer)) =
                (self.mouse_pressed, previous, self.renderer.as_mut())
            {
                let dx = (x - last_x) as f32;
                let dy = (y - last_y) as f32;
                match button {
                    MouseButton::Left => {
                        let speed = self.ui_state.camera_speed.max(0.05);
                        if self.modifiers_alt {
                            renderer.camera.orbit(dx * speed, dy * speed);
                        } else {
                            renderer.camera.look(dx * speed, dy * speed);
                        }
                        renderer.camera.clamp_to_world(renderer.heights.world_size);
                    }
                    MouseButton::Right | MouseButton::Middle => {
                        let speed = self.ui_state.camera_speed.max(0.05);
                        let dx = dx * speed;
                        let dy = dy * speed;
                        if button == MouseButton::Right {
                            self.right_drag_distance += dx.abs() + dy.abs();
                        }
                        renderer.camera.pan(dx, dy);
                        renderer.camera.clamp_to_world(renderer.heights.world_size);
                    }
                    _ => {}
                }
                want_redraw = true;
            }
        } else if (self.viewport_paint_tool_armed() && self.cursor_in_viewport())
            || self.gui_wants_pointer
            || self.screen == AppScreen::Home
        {
            want_redraw = true;
        }
        want_redraw
    }

    fn apply_wheel(&mut self, delta: f32) -> bool {
        let over_chrome = self.cursor_logical().is_some_and(|(x, y)| {
            let viewport = &self.viewport_rect;
            y < viewport.min_y || y >= viewport.max_y || x < viewport.min_x || x >= viewport.max_x
        });
        if over_chrome || self.gui_wants_pointer {
            self.gui_scroll_delta += delta;
            return true;
        }
        if self.viewport_paint_tool_armed() && !self.modifiers_alt {
            self.ui_state.ensure_sculpt_defaults();
            let step = if delta.abs() >= 1.0 {
                delta.signum() * 0.008
            } else {
                delta * 0.008
            };
            self.ui_state.sculpt_radius = (self.ui_state.sculpt_radius + step).clamp(0.005, 0.25);
            return true;
        }
        if self.viewport_camera_active() {
            if let Some(renderer) = self.renderer.as_mut() {
                renderer
                    .camera
                    .zoom(delta * 40.0 * self.ui_state.camera_speed.max(0.05));
                return true;
            }
        }
        false
    }
}

impl ApplicationHandler<RuntimeEvent> for TerraApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window = match event_loop.create_window(
            Window::default_attributes()
                .with_title("Terra")
                .with_decorations(false)
                .with_resizable(true)
                .with_visible(false)
                .with_inner_size(winit::dpi::LogicalSize::new(1600, 900)),
        ) {
            Ok(w) => Arc::new(w),
            Err(error) => {
                self.startup_failure = Some(StartupError::Window(error));
                event_loop.exit();
                return;
            }
        };

        if startup::injected_fault("gpu-init") {
            self.startup_failure = Some(StartupError::Gpu(terra_render::RenderError::Msg(
                "injected gpu-init fault".into(),
            )));
            event_loop.exit();
            return;
        }

        let (gpu, target) = match pollster::block_on(terra_render::init_gpu(window.clone())) {
            Ok(result) => result,
            Err(error) => {
                self.startup_failure = Some(StartupError::Gpu(error));
                event_loop.exit();
                return;
            }
        };
        if let Some(proxy) = self.runtime_event_proxy.clone() {
            gpu.device.set_device_lost_callback(move |reason, message| {
                let _ = proxy.send_event(RuntimeEvent::DeviceLost { reason, message });
            });
        }

        // The GUI renderer only needs the device/queue/format, so it is ready
        // long before the terrain pipelines. Paint the first splash frame, reveal
        // the window, then build the heavy GPU pipelines on a worker thread while
        // the main loop keeps animating the splash — the window stays responsive
        // instead of freezing on a white void.
        let mut gui_renderer = GuiRenderer::new(&gpu.device, &gpu.queue, gpu.surface_format);
        // Count shader compiles from zero for this boot's splash status line.
        terra_core::shader_progress::reset();
        let pending = target.into_pending();
        Self::paint_splash_frame(&pending, &gpu, &mut gui_renderer, &window);
        window.set_visible(true);

        // Build TerrainRenderer / engine / tile atlas off-thread. These are the
        // shader/pipeline compiles; `wgpu` device & queue are Send + Sync, so we
        // build against a cloned GpuContext and hand the finished objects back.
        let config = pending.config().clone();
        let size = pending.size();
        let tile_config = self.terrain_runtime.pyramid.config;
        let (tile_size, tile_halo) = (tile_config.tile_size, tile_config.halo);
        let worker_gpu = gpu.clone();
        let inject_boot_fault = startup::injected_fault("boot-worker");
        let job = terra_jobs::spawn_one_shot("terra-gpu-init", move |_ctx| {
            if inject_boot_fault {
                panic!("injected boot-worker fault");
            }
            let renderer = TerrainRenderer::new_detached(&worker_gpu, config, size);
            let gpu_engine = GpuTerrainEngine::new(&worker_gpu.device, 256);
            let tile_atlas = match GpuTileAtlas::new(&worker_gpu.device, tile_size, tile_halo, 128)
            {
                Ok(atlas) => Some(atlas),
                Err(error) => {
                    log::warn!("GPU tile atlas disabled: {error}");
                    None
                }
            };
            super::BootResult {
                renderer,
                tile_atlas,
                gpu_engine,
            }
        });

        self.window = Some(window);
        self.gui_renderer = Some(gui_renderer);
        self.boot = Some(super::BootState {
            gpu,
            pending,
            job,
            started: Instant::now(),
            failure: None,
        });
        // Animate: keep repainting the splash until the worker result lands
        // (about_to_wait polls `boot.job` and finalizes).
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Guard: if a startup failure is already stored (surfaces 3–4) or
        // pending exit, swallow events so straggler redraws can't reach the
        // editor path with renderer: None.
        if self.startup_failure.is_some() {
            return;
        }

        // Boot-failure splash: the boot worker failed and its error is parked
        // in boot.failure. Any key press, mouse click, or close request
        // acknowledges it and exits.
        if let Some(boot) = &self.boot {
            if boot.failure.is_some() {
                let dismiss = matches!(
                    event,
                    WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput { .. }
                        | WindowEvent::MouseInput { .. }
                );
                if dismiss {
                    let failure = self.boot.as_mut().unwrap().failure.take().unwrap();
                    self.startup_failure = Some(failure);
                    self.failure_presented = true;
                    event_loop.exit();
                }
                return;
            }
        }

        // Input callbacks are deliberately capture-only. Application/tool mutation
        // happens from an immutable snapshot at `about_to_wait`, before required
        // interactive work and optional refinement are considered.
        let event = match event {
            WindowEvent::KeyboardInput { event, .. } => {
                let code = match event.physical_key {
                    PhysicalKey::Code(code) => Some(code),
                    _ => None,
                };
                self.queue_input(InputEvent::Keyboard {
                    code,
                    state: event.state,
                });
                return;
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                let state = modifiers.state();
                self.queue_input(InputEvent::Modifiers(InputModifiers {
                    shift: state.shift_key(),
                    alt: state.alt_key(),
                    ctrl: state.control_key(),
                    super_key: state.super_key(),
                }));
                return;
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.queue_input(InputEvent::PointerButton { state, button });
                return;
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.queue_input(InputEvent::PointerMoved {
                    x: position.x,
                    y: position.y,
                });
                return;
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let delta = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(position) => position.y as f32 / 40.0,
                };
                self.queue_input(InputEvent::Wheel { delta });
                return;
            }
            WindowEvent::Focused(focused) => {
                self.queue_input(InputEvent::Focused(focused));
                if !focused {
                    self.queue_input(InputEvent::PointerCancelled(PointerCancelReason::FocusLost));
                }
                return;
            }
            WindowEvent::CursorEntered { .. } => {
                self.queue_input(InputEvent::CursorEntered);
                return;
            }
            WindowEvent::CursorLeft { .. } => {
                self.queue_input(InputEvent::CursorLeft);
                self.queue_input(InputEvent::PointerCancelled(
                    PointerCancelReason::CaptureLost,
                ));
                return;
            }
            event => event,
        };

        match event {
            WindowEvent::CloseRequested => {
                self.logical_frames
                    .request_shutdown(EditGeneration::new(self.eval_token));
                self.record_frame_event(FrameTraceEventKind::ShutdownRequested);
            }
            WindowEvent::Resized(size) => {
                self.capture_surface_resize(size);
            }
            WindowEvent::RedrawRequested => {
                let presentation_identity = self.logical_frames.take_presentation_identity();
                if let Some(identity) = presentation_identity {
                    self.ui_state.profile.logical_frame_id = identity.id.get();
                    self.ui_state.profile.edit_generation = identity.generation.get();
                    self.ui_state.profile.presented_generation =
                        self.last_complete_generation.get();
                    if let Some(renderer) = self.renderer.as_mut() {
                        renderer.set_presentation_trace_context(
                            terra_render::GpuPresentationTraceContext {
                                frame_id: identity.id.get(),
                                generation: self.last_complete_generation.get(),
                                evaluation_id: self.last_accepted_evaluation_id,
                            },
                        );
                    }
                }
                self.redraw();
                let probe_results = self.renderer.as_mut().map_or_else(
                    Vec::new,
                    terra_render::TerrainRenderer::poll_integrity_probes,
                );
                for result in probe_results {
                    self.frame_trace.record_probe_result(Instant::now(), result);
                }
                if let Some(timings) = self
                    .renderer
                    .as_ref()
                    .map(|renderer| renderer.last_gpu_timings)
                {
                    let newly_resolved =
                        timings.source_frame > self.last_reported_presentation_timing_frame;
                    if newly_resolved {
                        self.last_reported_presentation_timing_frame = timings.source_frame;
                    }
                    if newly_resolved && timings.context.frame_id != 0 {
                        let gpu_us = timings
                            .terrain_us
                            .saturating_add(timings.shadow_us)
                            .saturating_add(timings.path_trace_us)
                            .saturating_add(timings.temporal_us)
                            .saturating_add(timings.denoise_us);
                        self.frame_trace.record(
                            Instant::now(),
                            FrameTraceEventKind::GpuPresentationResolved,
                            Some(FrameIdentity {
                                id: LogicalFrameId::new(timings.context.frame_id),
                                generation_at_start: EditGeneration::new(
                                    timings.context.generation,
                                ),
                                generation: EditGeneration::new(timings.context.generation),
                            }),
                            None,
                            Some(super::frame_trace::EvaluationTraceId::new(
                                timings.context.evaluation_id,
                            )),
                            None,
                            None,
                            Some(Duration::from_micros(gpu_us)),
                        );
                    }
                }
                if let Some(identity) = presentation_identity {
                    let now = Instant::now();
                    self.frame_trace
                        .note_surface_presented(now, self.last_complete_generation);
                    let input_visible = self.frame_trace.input_to_visible_summary();
                    let refinement = self.frame_trace.refinement_summary();
                    let follow_up_press = self.frame_trace.follow_up_press_summary();
                    self.ui_state.profile.brush_trace_samples = input_visible.count;
                    self.ui_state.profile.input_visible_p50_us = input_visible.p50_us;
                    self.ui_state.profile.input_visible_p95_us = input_visible.p95_us;
                    self.ui_state.profile.input_visible_max_us = input_visible.max_us;
                    self.ui_state.profile.refinement_p50_us = refinement.p50_us;
                    self.ui_state.profile.refinement_p95_us = refinement.p95_us;
                    self.ui_state.profile.refinement_max_us = refinement.max_us;
                    self.ui_state.profile.follow_up_press_p50_us = follow_up_press.p50_us;
                    self.ui_state.profile.follow_up_press_p95_us = follow_up_press.p95_us;
                    self.ui_state.profile.follow_up_press_max_us = follow_up_press.max_us;
                    self.frame_trace.record(
                        now,
                        FrameTraceEventKind::SurfacePresented,
                        Some(identity),
                        Some(FramePhase::PresentationRequest),
                        None,
                        Some(self.scheduler.quality),
                        None,
                        None,
                    );
                }
            }
            _ => {}
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: RuntimeEvent) {
        self.handle_runtime_event(event);
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.frame_trace.set_verbose(self.ui_state.show_profiler);
        self.ui_state.profile.trace_orphaned_events = self.frame_trace.orphaned_events();
        if self.pending_exit
            || self.startup_failure.is_some()
            || self.logical_frames.shutdown_requested()
        {
            self.prepare_shutdown();
            event_loop.exit();
            return;
        }

        // Startup: poll the GPU-init worker. Until it lands, keep repainting the
        // splash at ~30 fps so the sweep animates and the window stays responsive.
        // When the worker has failed, the failure splash is static — switch to
        // Wait (input-driven) and request one final redraw for the failure frame.
        if self.is_booting() {
            let finished = self.try_finish_boot();
            let boot_failed = self.boot.as_ref().is_some_and(|b| b.failure.is_some());
            if finished {
                // The handoff redraw is the final bootstrap exception. Prepare
                // the newly attached renderer before that bounded presentation.
                self.prepare_presentation();
            }
            if let Some(w) = &self.window {
                w.request_redraw();
            }
            if finished || boot_failed {
                event_loop.set_control_flow(ControlFlow::Wait);
            } else {
                event_loop.set_control_flow(ControlFlow::WaitUntil(
                    Instant::now() + Duration::from_millis(33),
                ));
            }
            return;
        }

        let completed_gpu_evaluations =
            if let (Some(engine), Some(gpu)) = (self.gpu_engine.as_mut(), self.gpu.as_ref()) {
                engine.poll_evaluation_timings(&gpu.device, &gpu.queue)
            } else {
                Vec::new()
            };
        for timing in completed_gpu_evaluations {
            self.ui_state.profile.gpu_evaluation_us = timing.gpu_us;
            self.frame_trace.record(
                Instant::now(),
                FrameTraceEventKind::GpuEvaluationResolved,
                Some(FrameIdentity {
                    id: LogicalFrameId::new(timing.context.frame_id),
                    generation_at_start: EditGeneration::new(timing.context.generation),
                    generation: EditGeneration::new(timing.context.generation),
                }),
                None,
                Some(super::frame_trace::EvaluationTraceId::new(
                    timing.context.evaluation_id,
                )),
                None,
                None,
                Some(Duration::from_micros(timing.gpu_us)),
            );
        }

        let input_frame_processed = self.process_pending_input_frame();
        self.begin_scheduled_frame_if_needed();
        let ui_effects_applied = self.apply_pending_ui_effects();
        if ui_effects_applied {
            self.record_frame_event(FrameTraceEventKind::UiEffectsApplied);
        }
        let mut surface_changed = false;
        if self.pending_surface_reconfigure {
            if let Some(renderer) = self.renderer.as_mut() {
                renderer.reconfigure();
            }
            self.pending_surface_reconfigure = false;
            surface_changed = true;
        }
        if let Some(size) = self.pending_surface_resize.take() {
            if let Some(renderer) = self.renderer.as_mut() {
                renderer.resize(size);
            }
            self.refresh_viewport_rect();
            self.record_frame_event(FrameTraceEventKind::ResizeApplied);
            surface_changed = true;
        }

        // One registry tick pumps the background subsystems (export, project IO,
        // tool thumbnails) and folds their pending/wake facts together. The Arc
        // clone lets `tick` borrow `self` mutably while the fixed entry list is
        // read through the shared handle. Typed results are drained just below,
        // where the concrete types are in hand.
        let jobs = Arc::clone(&self.jobs).tick(self);
        self.drain_project_io();
        let camera_flying = self.apply_camera_fly();
        let mut export_busy = !self.exporter.job.done;
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
                    log::error!("export failed: {err}");
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
        let mut work_pending = export_busy || jobs.any_pending;

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
                let refinement_state = self.terrain_runtime.refinement.state();
                engine.set_simulation_iteration_cap(refinement_state.simulation_iteration_cap());
            }
            // The worker is never awaited: drain available completion/failure events.
            while let Some(event) = self.eval_worker.try_recv_event() {
                match event {
                    EvalWorkerEvent::Completed(result) if result.token == self.eval_token => {
                        self.ui_state.profile.cpu_published =
                            self.ui_state.profile.cpu_published.saturating_add(1);
                        self.ui_state.evaluation_failure = None;
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
                        if let (Some(engine), Some(gpu)) =
                            (self.gpu_engine.as_mut(), self.gpu.as_ref())
                        {
                            let preview = self.session.document.preview_eval_stack();
                            for layer in preview.flatten_layers() {
                                if let Some(cached) = self.scheduler.evaluator.cache.get(layer.id())
                                {
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
                        self.pending_gpu_dirty_region = None;
                        self.deferred_full_field = None;
                        self.ui_state.terrain_preview_freshness = TerrainPreviewFreshness::Current;
                        // Loss-proof transport: a *fresh* result for this token proves
                        // no edit occurred after its submit (an edit bumps the token,
                        // making the result stale-discarded below), so the accumulators
                        // still hold exactly what this job carried — clear them now
                        // (the submit only copied them). Record the resolution the
                        // worker cache now holds so the straight-to-Full ladder gate
                        // knows when the Full-res checkpoints exist to reuse.
                        self.worker_mark_all_dirty = false;
                        self.worker_dirty_from = None;
                        self.worker_dirty_region = None;
                        self.worker_cache_res = Some(height.metrics.width);
                        self.last_complete_generation = EditGeneration::new(result.token);
                        self.ui_state.profile.eval_us = result.eval_us;
                        self.ui_state.profile.tex_w =
                            self.last_height.as_ref().unwrap().metrics.width;
                        self.ui_state.profile.tex_h =
                            self.last_height.as_ref().unwrap().metrics.height;
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
                        self.note_refinement_activity();
                        did_eval = true;
                    }
                    EvalWorkerEvent::Completed(result) => {
                        log::debug!(
                            target: "terra_app::evaluation",
                            "discarding stale evaluation result; {}",
                            self.evaluation_log_context(result.token, result.quality)
                        );
                    }
                    EvalWorkerEvent::Failed(failure) => {
                        let layer_name = failure.error.layer_name().map(str::to_owned);
                        self.handle_evaluation_failure_details(
                            failure.token,
                            failure.quality,
                            layer_name,
                            false,
                            format!("evaluation worker failed: {}", failure.error),
                        );
                    }
                    EvalWorkerEvent::Disconnected => {
                        self.eval_worker.restart();
                        self.eval_worker.set_token(self.eval_token);
                        self.worker_mark_all_dirty = true;
                        self.worker_dirty_from = None;
                        self.worker_dirty_region = None;
                        self.worker_cache_res = None;
                        self.handle_evaluation_failure_details(
                            self.eval_token,
                            self.scheduler.quality,
                            None,
                            true,
                            "evaluation worker disconnected or panicked",
                        );
                    }
                }
            }

            // Debounced draft eval. Live paint/sculpt has no time window — it rebuilds
            // Draft as soon as the previous preview finishes, with per-frame coalescing
            // handled in `redraw`. Manual edits debounce on last_edit.
            if !stall_draft && self.pending_eval {
                let live_ok = self.session.rebuild_feedback.prefs.live_preview;
                let generation = EditGeneration::new(self.eval_token);
                let deadline_ready = self.logical_frames.deadline_ready(
                    FrameDeadlineKind::InteractiveEvaluation,
                    generation,
                    Instant::now(),
                );
                let ready =
                    self.pending_eval_immediate || live_paint || (live_ok && deadline_ready);
                if ready {
                    let immediate = self.pending_eval_immediate;
                    self.pending_eval = false;
                    self.pending_eval_immediate = false;
                    self.logical_frames
                        .clear_deadline(FrameDeadlineKind::InteractiveEvaluation);
                    let intent = if immediate {
                        GpuEvaluationIntent::Complete
                    } else {
                        self.force_draft = true;
                        if self.pending_gpu_dirty_region.is_some() {
                            GpuEvaluationIntent::InteractiveLocal
                        } else {
                            GpuEvaluationIntent::Complete
                        }
                    };
                    self.run_eval_step_with_intent(intent);
                    self.note_refinement_activity();
                    did_eval = true;
                }
            }

            // A bounded edit keeps presenting its exact local prefix during the
            // gesture. Mouse-up starts one coalescing window for the entire globally
            // coupled suffix; mouse moves never push this deadline forward.
            if self
                .deferred_full_field
                .as_ref()
                .is_some_and(|pending| pending.generation != self.eval_token)
            {
                self.deferred_full_field = None;
                self.logical_frames
                    .clear_deadline(FrameDeadlineKind::FullFieldRefinement);
            }
            if let Some(pending) = self.deferred_full_field.as_mut() {
                if live_paint {
                    pending.hold_during_gesture();
                    self.ui_state.terrain_preview_freshness = TerrainPreviewFreshness::Deferred {
                        layer_name: pending.layer_name.clone(),
                        deferred_layers: pending.deferred_layers,
                        settling: false,
                    };
                } else {
                    let now = Instant::now();
                    if pending.arm_after_gesture(now) {
                        self.logical_frames.schedule_deadline(
                            FrameDeadlineKind::DeferredFullField,
                            EditGeneration::new(self.eval_token),
                            pending.settle_at.expect("deadline armed"),
                        );
                        self.logical_frames.schedule_deadline(
                            FrameDeadlineKind::FullFieldRefinement,
                            EditGeneration::new(self.eval_token),
                            now + Duration::from_millis(FULL_FIELD_REFINE_MS),
                        );
                        self.ui_state.terrain_preview_freshness =
                            TerrainPreviewFreshness::Deferred {
                                layer_name: pending.layer_name.clone(),
                                deferred_layers: pending.deferred_layers,
                                settling: true,
                            };
                    }
                }
            }
            let suffix_ready = self
                .deferred_full_field
                .as_ref()
                .is_some_and(|pending| pending.ready(self.eval_token, Instant::now()))
                && !self.pending_eval
                && !self.worker_refine_pending;
            if suffix_ready {
                self.logical_frames
                    .clear_deadline(FrameDeadlineKind::DeferredFullField);
                if let Some(pending) = self.deferred_full_field.as_ref() {
                    self.ui_state.terrain_preview_freshness =
                        TerrainPreviewFreshness::RefiningSuffix {
                            layer_name: pending.layer_name.clone(),
                            quality: PreviewQuality::Draft,
                        };
                }
                // Complete the suffix at the quality of the resident local prefix.
                // A forced Draft here replaces a Full texture immediately after
                // gesture end and permanently knocks rapid dabs onto the coarse path.
                self.complete_deferred_full_field();
                did_eval = true;
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

            // Required input/evaluation work gets a presentation request before
            // optional refinement is allowed to start. The actual surface present
            // occurs on the following RedrawRequested callback.
            if input_frame_processed
                || ui_effects_applied
                || surface_changed
                || did_eval
                || self.needs_height_upload
            {
                self.logical_frames
                    .transition(FramePhase::PresentationRequest);
                self.logical_frames.mark_presentation_requested();
                self.frame_trace.record(
                    Instant::now(),
                    FrameTraceEventKind::PresentationRequested,
                    self.logical_frames.active_identity(),
                    Some(FramePhase::PresentationRequest),
                    None,
                    Some(self.scheduler.quality),
                    None,
                    None,
                );
            }
            self.logical_frames
                .transition(FramePhase::OptionalRefinement);

            // Existing optional GPU jobs advance only after required work and at
            // safe allocation/submission boundaries. Input/generation checks are
            // repeated inside `advance_gpu_refinement` before publication.
            if self.refinement_job.is_some()
                && !stall_refine
                && self.logical_frames.can_start_optional(Instant::now())
                && !self.input.has_pending()
            {
                did_eval |= self.advance_gpu_refinement();
            }

            // Create the next quality job only after the established idle grace.
            // CPU fallback retains the latest-wins worker path.
            if self.refinement_job.is_none()
                && !stall_refine
                && self.logical_frames.can_start_optional(Instant::now())
                && !self.input.has_pending()
                && self.logical_frames.deadline_ready(
                    FrameDeadlineKind::OptionalRefinement,
                    EditGeneration::new(self.eval_token),
                    Instant::now(),
                )
                && self.ui_state.refining
                && !self.pending_eval
                && !self.worker_refine_pending
                && self.deferred_full_field.is_none()
                && self.logical_frames.deadline_ready(
                    FrameDeadlineKind::FullFieldRefinement,
                    EditGeneration::new(self.eval_token),
                    Instant::now(),
                )
            {
                self.logical_frames
                    .clear_deadline(FrameDeadlineKind::OptionalRefinement);
                self.logical_frames
                    .clear_deadline(FrameDeadlineKind::FullFieldRefinement);
                if let Some(mut target_quality) = self.scheduler.quality.next_refine() {
                    // HD preview: skip Medium so Camera/Zone sees Full carve sooner.
                    if !matches!(
                        self.session.document.level_steps.high_detail,
                        terra_core::analyze::HighDetailMode::None
                    ) && matches!(target_quality, PreviewQuality::Medium)
                    {
                        if let Some(next) = target_quality.next_refine() {
                            target_quality = next;
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
                    if self.gpu_engine.is_some() && self.begin_gpu_refinement(target_quality) {
                        did_eval = true;
                    } else {
                        self.scheduler.quality = target_quality;
                        self.ui_state.quality = target_quality;
                        self.enqueue_refine_job();
                        self.ui_state.build_progress =
                            Some(quality_in_flight_progress(target_quality, 0.0));
                        did_eval = true;
                    }
                } else {
                    self.ui_state.refining = false;
                    self.ui_state.quality = PreviewQuality::Full;
                    self.ui_state.build_progress = None;
                    self.ui_state.refining_layer_name = None;
                }
                self.note_refinement_activity();
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
                || self.refinement_job.is_some()
                || self.ui_state.refining
                || self.deferred_full_field.is_some()
                || !self.pending_tile_uploads.is_empty()
                || export_busy
                || jobs.any_pending;
        }

        let now = Instant::now();
        let continuous = (live_paint && self.pending_eval) || camera_flying;
        let fallback_deadline =
            if self.worker_refine_pending || self.refinement_job.is_some() || jobs.animate {
                Some(now + Duration::from_millis(16))
            } else if export_busy || jobs.any_pending || work_pending {
                Some(now + Duration::from_millis(REFINE_INTERVAL_MS as u64))
            } else {
                None
            };
        match self.logical_frames.wake_decision(
            EditGeneration::new(self.eval_token),
            now,
            continuous,
            fallback_deadline,
        ) {
            FrameWake::Poll => event_loop.set_control_flow(ControlFlow::Poll),
            FrameWake::WaitUntil(deadline) => {
                event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
            }
            FrameWake::Wait => event_loop.set_control_flow(ControlFlow::Wait),
        }

        if did_eval
            || input_frame_processed
            || ui_effects_applied
            || surface_changed
            || export_busy
            || self.needs_height_upload
            || jobs.redraw
            || camera_flying
        {
            self.prepare_presentation();
            self.request_window_presentation();
        }
        if let Some(diagnostics) = self.logical_frames.complete(Instant::now()) {
            self.ui_state.profile.logical_frame_id = diagnostics.identity.id.get();
            self.ui_state.profile.edit_generation = diagnostics.identity.generation.get();
            self.ui_state.profile.logical_phase = diagnostics.phase.label();
            self.ui_state.profile.input_event_count = diagnostics.input_events;
            self.ui_state.profile.pointer_sample_count = diagnostics.pointer_samples;
            self.ui_state.profile.input_frame_pending = self.logical_frames.has_pending();
            log::debug!(
                target: "terra_app::logical_frame",
                "frame={} generation={} phase={} reason={:?} events={} pointer_samples={} elapsed_us={}",
                diagnostics.identity.id.get(),
                diagnostics.identity.generation.get(),
                diagnostics.phase.label(),
                diagnostics.reason,
                diagnostics.input_events,
                diagnostics.pointer_samples,
                diagnostics.elapsed.as_micros()
            );
            if diagnostics.elapsed > Duration::from_millis(super::LOGICAL_FRAME_HOST_BUDGET_MS) {
                self.frame_trace.record(
                    Instant::now(),
                    FrameTraceEventKind::HeartbeatOverBudget,
                    Some(diagnostics.identity),
                    Some(diagnostics.phase),
                    None,
                    None,
                    None,
                    Some(diagnostics.elapsed),
                );
            }
        }
    }
}

impl TerraApp {
    /// The only post-bootstrap adapter from logical presentation demand to winit.
    fn request_window_presentation(&mut self) {
        self.logical_frames.mark_presentation_requested();
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn prepare_shutdown(&mut self) {
        if self.logical_frames.active_identity().is_some() {
            self.record_frame_event(FrameTraceEventKind::FrameAborted);
        }
        self.eval_token = self.eval_token.wrapping_add(1);
        self.scheduler.current_token = self.eval_token;
        self.eval_worker.set_token(self.eval_token);
        self.supersede_gpu_refinement();
        self.worker_refine_pending = false;
        self.pending_eval = false;
        self.pending_eval_immediate = false;
        self.deferred_full_field = None;
        self.pending_tile_uploads.clear();
        self.pending_ui_effects.clear();
        self.input.clear();
        self.pending_surface_resize = None;
        self.pending_surface_reconfigure = false;
        self.logical_frames.clear_presentation();
        self.logical_frames.abort(Instant::now());
    }
}

impl TerraApp {
    /// True while GPU pipelines are still compiling on the boot worker.
    pub(crate) fn is_booting(&self) -> bool {
        self.boot.is_some()
    }

    /// Present one animated splash frame from the main-thread-held surface.
    /// When the boot worker has failed, paints a static failure frame instead
    /// of the animated sweep. Called from `redraw` while `boot` is set.
    pub(crate) fn draw_boot_splash(&mut self) {
        let Some(window) = self.window.clone() else {
            return;
        };
        let boot = match self.boot.as_ref() {
            Some(boot) => boot,
            None => return,
        };
        let Some(gui_renderer) = self.gui_renderer.as_mut() else {
            return;
        };
        let ppp = (window.scale_factor() as f32).max(0.5);
        let phys = boot.pending.size();
        let screen_w = (phys.width as f32 / ppp).max(1.0);
        let screen_h = (phys.height as f32 / ppp).max(1.0);
        let gui_state = &mut self.gui_state;

        if let Some(error) = &boot.failure {
            let lines = startup::failure_splash_lines(error, None);
            boot.pending.present_splash(&boot.gpu, SPLASH_BG, |view| {
                let mut gui =
                    GuiContext::begin(screen_w, screen_h, ppp, GuiInput::default(), gui_state);
                paint_failure_splash(&mut gui, screen_w, screen_h, &lines);
                gui.end();
                gui_renderer.render(
                    &boot.gpu.device,
                    &boot.gpu.queue,
                    view,
                    &mut gui,
                    phys.width.max(1),
                    phys.height.max(1),
                );
            });
        } else {
            let elapsed = boot.started.elapsed().as_secs_f32();
            let shaders = terra_core::shader_progress::shaders_compiled();
            boot.pending.present_splash(&boot.gpu, SPLASH_BG, |view| {
                let mut gui =
                    GuiContext::begin(screen_w, screen_h, ppp, GuiInput::default(), gui_state);
                paint_splash(&mut gui, screen_w, screen_h, elapsed, shaders);
                gui.end();
                gui_renderer.render(
                    &boot.gpu.device,
                    &boot.gpu.queue,
                    view,
                    &mut gui,
                    phys.width.max(1),
                    phys.height.max(1),
                );
            });
        }
    }

    /// Present the first splash frame during startup, before the worker begins.
    /// Associated fn so it borrows no `self` fields.
    fn paint_splash_frame(
        pending: &terra_render::PendingSurface,
        gpu: &GpuContext,
        gui_renderer: &mut GuiRenderer,
        window: &Window,
    ) {
        let ppp = (window.scale_factor() as f32).max(0.5);
        let phys = pending.size();
        let screen_w = (phys.width as f32 / ppp).max(1.0);
        let screen_h = (phys.height as f32 / ppp).max(1.0);
        let mut gui_state = GuiState::default();
        let shaders = terra_core::shader_progress::shaders_compiled();
        pending.present_splash(gpu, SPLASH_BG, |view| {
            let mut gui =
                GuiContext::begin(screen_w, screen_h, ppp, GuiInput::default(), &mut gui_state);
            paint_splash(&mut gui, screen_w, screen_h, 0.0, shaders);
            gui.end();
            gui_renderer.render(
                &gpu.device,
                &gpu.queue,
                view,
                &mut gui,
                phys.width.max(1),
                phys.height.max(1),
            );
        });
    }

    /// Poll the boot worker; when it has produced the GPU objects, attach the
    /// surface and install them. Returns `true` if the app finished booting
    /// this call (caller should request a real redraw). Returns `false` when
    /// the worker is still running, has already been consumed, or has failed
    /// (in which case `boot.failure` is set and the splash becomes a failure
    /// frame until the user acknowledges it).
    pub(crate) fn try_finish_boot(&mut self) -> bool {
        let Some(boot) = self.boot.as_ref() else {
            return false;
        };
        if boot.failure.is_some() {
            return false;
        }
        let result = match startup::classify_boot_poll(boot.job.try_take()) {
            startup::BootPoll::Pending => return false,
            startup::BootPoll::Ready(value) => value,
            startup::BootPoll::Failed(error) => {
                startup::report_failure(&error, None, false);
                self.boot.as_mut().expect("boot present").failure = Some(error);
                return false;
            }
            startup::BootPoll::Shutdown => {
                self.boot.take();
                self.pending_exit = true;
                return false;
            }
        };
        let boot = self.boot.take().expect("boot present");
        log::info!(
            "terra: GPU init complete — {} shaders compiled in {} ms",
            terra_core::shader_progress::shaders_compiled(),
            boot.started.elapsed().as_millis()
        );
        let super::BootResult {
            mut renderer,
            tile_atlas,
            gpu_engine,
        } = result;
        boot.pending.attach(&mut renderer);
        // Reconcile against the live window size in case it changed during init.
        if let Some(window) = &self.window {
            renderer.resize(window.inner_size());
        }
        self.renderer = Some(renderer);
        self.tile_atlas = tile_atlas;
        self.gpu_engine = Some(gpu_engine);
        self.gpu = Some(boot.gpu);
        self.refresh_window_title();
        self.refresh_viewport_rect();
        // Decode 1024² tool thumbs on a background pool before the user opens
        // Quick Add / Tools — avoids Lucide→3D icon flash on first dialog open.
        crate::ui::prefetch_tool_thumbnails();
        true
    }

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

/// Startup splash background — matches the Home viewport clear tone
/// (see redraw.rs, AppScreen::Home lighting.clear).
const SPLASH_BG: [f32; 3] = [0.071, 0.082, 0.102];

/// Draw the Terra wordmark, an indeterminate progress sweep, and a status line
/// with the live shader-compile count. `t` is elapsed seconds; the sweep is purely
/// time-based so it keeps animating while the GPU-init worker is busy (the window
/// never looks frozen). `shaders` is the count reported by the render/GPU crates as
/// each shader module compiles. Uses ASCII "..." — the GUI font has no `…` glyph.
fn paint_splash(gui: &mut GuiContext, screen_w: f32, screen_h: f32, t: f32, shaders: u32) {
    // Opaque plate (the clear already matches; this also guards platforms where
    // the clear color space differs slightly).
    gui.panel(
        Rect::from_pos_size(0.0, 0.0, screen_w, screen_h),
        Color::rgb(SPLASH_BG[0], SPLASH_BG[1], SPLASH_BG[2]),
    );

    // Centered wordmark, width-fit to a comfortable fraction of the window.
    let (lw, lh, rgba) = crate::ui::brand_logo();
    let logo_w = (screen_w * 0.30).clamp(240.0, 520.0);
    let logo_h = logo_w * (*lh as f32 / (*lw as f32).max(1.0));
    let lx = (screen_w - logo_w) * 0.5;
    let ly = (screen_h - logo_h) * 0.5 - 24.0;
    gui.image(Rect::from_pos_size(lx, ly, logo_w, logo_h), *lw, *lh, rgba);

    // Indeterminate progress sweep under the mark.
    let bar_w = (screen_w * 0.24).clamp(200.0, 420.0);
    let bar_h = 3.0;
    let bar_x = (screen_w - bar_w) * 0.5;
    let bar_y = ly + logo_h + 22.0;
    gui.panel_rounded(
        Rect::from_pos_size(bar_x, bar_y, bar_w, bar_h),
        Color::rgba(1.0, 1.0, 1.0, 0.08),
        bar_h * 0.5,
    );
    // Ping-pong highlight (Cylon sweep): triangle wave 0..1..0 over ~1.8s.
    let seg_w = bar_w * 0.32;
    let ping = 1.0 - (2.0 * (t * 0.55).fract() - 1.0).abs();
    let seg_x = bar_x + (bar_w - seg_w) * ping;
    gui.panel_rounded(
        Rect::from_pos_size(seg_x, bar_y, seg_w, bar_h),
        Color::rgba(0.36, 0.80, 0.74, 0.90),
        bar_h * 0.5,
    );

    // Status line: "Compiling shaders..." with the live count once it starts.
    let status = if shaders > 0 {
        format!("Compiling shaders... {shaders}")
    } else {
        "Compiling shaders...".to_string()
    };
    gui.label_centered(
        screen_w * 0.5,
        bar_y + 16.0,
        &status,
        Color::rgba(0.72, 0.78, 0.85, 0.85),
        1.05,
    );
}

/// Paint the boot-failure splash: logo, a red status bar, and the error lines.
fn paint_failure_splash(gui: &mut GuiContext, screen_w: f32, screen_h: f32, lines: &[String]) {
    gui.panel(
        Rect::from_pos_size(0.0, 0.0, screen_w, screen_h),
        Color::rgb(SPLASH_BG[0], SPLASH_BG[1], SPLASH_BG[2]),
    );

    let (lw, lh, rgba) = crate::ui::brand_logo();
    let logo_w = (screen_w * 0.30).clamp(240.0, 520.0);
    let logo_h = logo_w * (*lh as f32 / (*lw as f32).max(1.0));
    let lx = (screen_w - logo_w) * 0.5;
    let ly = (screen_h * 0.15).max(16.0);
    gui.image(Rect::from_pos_size(lx, ly, logo_w, logo_h), *lw, *lh, rgba);

    // Static red bar replaces the animated sweep.
    let bar_w = (screen_w * 0.24).clamp(200.0, 420.0);
    let bar_h = 3.0;
    let bar_x = (screen_w - bar_w) * 0.5;
    let bar_y = ly + logo_h + 22.0;
    gui.panel_rounded(
        Rect::from_pos_size(bar_x, bar_y, bar_w, bar_h),
        Color::rgba(0.85, 0.25, 0.20, 0.90),
        bar_h * 0.5,
    );

    let mut y = bar_y + 28.0;
    let line_height = 18.0;
    let text_color = Color::rgba(0.78, 0.80, 0.85, 0.90);
    let dim_color = Color::rgba(0.55, 0.58, 0.65, 0.75);
    for (i, line) in lines.iter().enumerate() {
        if line.is_empty() {
            y += line_height * 0.5;
            continue;
        }
        let color = if i == lines.len() - 1 {
            dim_color
        } else {
            text_color
        };
        gui.label_centered(screen_w * 0.5, y, line, color, 1.0);
        y += line_height;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_focus_and_capture_loss_cancel_a_pointer_gesture_once() {
        let mut app = TerraApp::default();
        app.mouse_pressed = Some(MouseButton::Left);
        assert!(app.apply_input_event(InputEvent::PointerCancelled(PointerCancelReason::FocusLost)));
        assert_eq!(app.mouse_pressed, None);
        assert!(app.apply_input_event(InputEvent::PointerCancelled(
            PointerCancelReason::CaptureLost
        )));
        assert_eq!(app.mouse_pressed, None);
    }

    #[test]
    fn resize_capture_coalesces_to_the_latest_size() {
        let mut app = TerraApp::default();
        app.capture_surface_resize(winit::dpi::PhysicalSize::new(800, 600));
        app.capture_surface_resize(winit::dpi::PhysicalSize::new(1200, 700));
        assert_eq!(
            app.pending_surface_resize,
            Some(winit::dpi::PhysicalSize::new(1200, 700))
        );
        assert!(app.logical_frames.has_pending());
    }

    #[test]
    fn device_loss_requests_controlled_shutdown() {
        let mut app = TerraApp::default();
        app.handle_runtime_event(RuntimeEvent::DeviceLost {
            reason: wgpu::DeviceLostReason::Unknown,
            message: "injected".into(),
        });
        assert!(app.logical_frames.shutdown_requested());
        assert!(app.ui_state.status.contains("injected"));
    }

    #[test]
    fn shutdown_clears_pending_input_and_work() {
        let mut app = TerraApp::default();
        app.queue_input(InputEvent::CursorEntered);
        app.pending_eval = true;
        app.pending_eval_immediate = true;
        app.worker_refine_pending = true;
        app.logical_frames
            .request_shutdown(EditGeneration::new(app.eval_token));

        app.prepare_shutdown();

        assert!(!app.input.has_pending());
        assert!(!app.pending_eval);
        assert!(!app.pending_eval_immediate);
        assert!(!app.worker_refine_pending);
        assert!(!app.logical_frames.has_pending());
        assert_eq!(app.logical_frames.active_identity(), None);
        assert_eq!(app.logical_frames.take_presentation_identity(), None);
    }
}
