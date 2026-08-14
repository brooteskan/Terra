use std::path::PathBuf;

use crate::ui::{
    project_template_by_id, resolve_workspace_command, CommandId, NewWorldSettings,
    ProjectHomeAction,
};
use terra_core::document::EditorSession;
use terra_io::{save_project, ProjectIoResult};

use super::{
    default_terra_projects_dir, document_from_world_settings, prepare_project_path,
    project_name_from_path, save_project_prefs, AppScreen, PendingProjectAction, TerraApp,
};
use terra_core::eval::PreviewQuality;
impl TerraApp {
    pub(crate) fn poll_project_io(&mut self) {
        self.project_io.poll();
        if let Some(status) = self.project_io.status() {
            self.ui_state.status = status.to_string();
            if let Some(w) = &self.window {
                w.request_redraw();
            }
        }
        let Some(result) = self.project_io.result.take() else {
            return;
        };
        match result {
            ProjectIoResult::Saved { path } => {
                if let Some((doc, pending_path)) = self.pending_enter_after_save.take() {
                    if pending_path == path {
                        self.show_new_template_picker = false;
                        self.enter_editor(doc, Some(path.clone()), false);
                        self.remember_recent(&path);
                        self.ui_state.status = format!("Created {}", path.display());
                        return;
                    }
                    self.pending_enter_after_save = Some((doc, pending_path));
                }
                self.ui_state.status = format!("Saved {}", path.display());
                self.project_path = Some(path.clone());
                self.document_dirty = false;
                self.remember_recent(&path);
                self.refresh_window_title();
            }
            ProjectIoResult::Loaded { path, doc } => {
                let name = doc.name.clone();
                self.enter_editor(doc, Some(path.clone()), false);
                self.project_prefs.push_recent(&path, &name);
                save_project_prefs(&self.project_prefs);
                self.ui_state.status = format!("Loaded {}", path.display());
            }
            ProjectIoResult::Failed { path, error } => {
                self.pending_enter_after_save = None;
                self.ui_state.status = format!("{} failed: {error}", path.display());
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
        }
    }

    pub(crate) fn begin_background_save(&mut self, path: PathBuf) {
        if self.project_io.is_busy() {
            self.ui_state.status = "Save already in progressâ€¦".into();
            return;
        }
        self.ui_state.status = "Savingâ€¦".into();
        self.sync_lighting_to_document();
        self.project_io
            .start_save(self.session.document.clone(), path);
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    /// Copy the editable viewport lighting into the document so File > Save persists it.
    pub(crate) fn sync_lighting_to_document(&mut self) {
        let vr = &self.ui_state.viewport_render;
        let preset = if self.ui_state.lighting_customized {
            String::new()
        } else {
            self.ui_state.lighting_preset.label().to_string()
        };
        self.session.document.viewport_lighting = terra_core::document::ViewportLighting {
            sun_azimuth_deg: vr.sun_azimuth_deg,
            sun_elevation_deg: vr.sun_elevation_deg,
            sun_intensity: vr.sun_intensity,
            exposure: vr.exposure,
            sky_color: vr.sky_color,
            ambient_strength: vr.ambient_strength,
            shadow_strength: vr.shadow_strength,
            fog_strength: vr.fog_strength,
            preset,
        };
    }

    /// Restore editable viewport lighting from the current document (on open / new).
    pub(crate) fn apply_document_lighting(&mut self) {
        let l = self.session.document.viewport_lighting.clone();
        {
            let vr = &mut self.ui_state.viewport_render;
            vr.sun_azimuth_deg = l.sun_azimuth_deg;
            vr.sun_elevation_deg = l.sun_elevation_deg;
            vr.sun_intensity = l.sun_intensity;
            vr.exposure = l.exposure;
            vr.sky_color = l.sky_color;
            vr.ambient_strength = l.ambient_strength;
            vr.shadow_strength = l.shadow_strength;
            vr.fog_strength = l.fog_strength;
        }
        match crate::ui::LightingPreset::ALL
            .iter()
            .find(|p| p.label() == l.preset.as_str())
        {
            Some(preset) => {
                self.ui_state.lighting_preset = *preset;
                self.ui_state.lighting_customized = false;
            }
            None => {
                // Saved as a customized (blank) look; keep the loaded values as-is.
                self.ui_state.lighting_customized = true;
            }
        }
        // Match the change-tracker so the redraw seed does not overwrite the loaded
        // values on the next frame.
        self.last_lighting_preset = self.ui_state.lighting_preset;
        self.last_lighting_customized = self.ui_state.lighting_customized;
    }

    pub(crate) fn save_project_as(&mut self) {
        let default_name = self
            .project_path
            .as_ref()
            .and_then(|path| path.file_name())
            .and_then(|name| name.to_str())
            .unwrap_or("terra_project.json");
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Terra Project", &["json"])
            .set_file_name(default_name)
            .save_file()
        else {
            return;
        };
        self.begin_background_save(path);
    }

    pub(crate) fn save_current_project(&mut self) {
        if self.screen != AppScreen::Editor {
            return;
        }
        if let Some(path) = self.project_path.clone() {
            self.begin_background_save(path);
        } else {
            self.save_project_as();
        }
    }

    pub(crate) fn remember_recent(&mut self, path: &PathBuf) {
        self.project_prefs
            .push_recent(path, &self.session.document.name);
        save_project_prefs(&self.project_prefs);
    }

    pub(crate) fn refresh_window_title(&mut self) {
        let title = match self.screen {
            AppScreen::Home => "Terra".to_string(),
            AppScreen::Editor => {
                let dirty = if self.document_dirty { "*" } else { "" };
                let name = &self.session.document.name;
                match &self.project_path {
                    Some(path) => format!("Terra â€” {name}{dirty} â€” {}", path.display()),
                    None => format!("Terra â€” {name}{dirty}"),
                }
            }
        };
        if let Some(window) = &self.window {
            window.set_title(&title);
        }
    }

    pub(crate) fn mark_document_dirty(&mut self) {
        if !self.document_dirty {
            self.document_dirty = true;
            self.refresh_window_title();
        } else {
            self.document_dirty = true;
        }
    }

    pub(crate) fn request_project_action(&mut self, action: PendingProjectAction) {
        if self.screen == AppScreen::Editor && self.document_dirty {
            self.pending_project_action = Some(action);
            if let Some(w) = &self.window {
                w.request_redraw();
            }
            return;
        }
        self.perform_project_action(action);
    }

    pub(crate) fn perform_project_action(&mut self, action: PendingProjectAction) {
        match action {
            PendingProjectAction::New => self.begin_new_project(),
            PendingProjectAction::Open => self.open_project_dialog(),
            PendingProjectAction::Close => self.close_project(),
            PendingProjectAction::OpenPath(path) => self.open_project_at(path),
        }
    }

    pub(crate) fn begin_new_project(&mut self) {
        self.new_template_selected = "blank".into();
        self.new_world_settings = NewWorldSettings::default();
        self.show_new_template_picker = true;
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    pub(crate) fn new_project_with_template(
        &mut self,
        template_id: &str,
        world_size_m: f32,
        sea_level: f32,
    ) {
        let Some(template) = project_template_by_id(template_id) else {
            self.ui_state.status = format!("Unknown template: {template_id}");
            return;
        };

        let projects_root = default_terra_projects_dir();
        if let Err(error) = std::fs::create_dir_all(&projects_root) {
            self.ui_state.status = format!("Could not create projects folder: {error}");
            if let Some(w) = &self.window {
                w.request_redraw();
            }
            return;
        }

        let Some(picked) = rfd::FileDialog::new()
            .add_filter("Terra Project", &["json"])
            .set_directory(&projects_root)
            .set_file_name(template.default_file_name)
            .save_file()
        else {
            return;
        };

        let name = project_name_from_path(&picked);
        let path = match prepare_project_path(&projects_root, &name) {
            Ok(path) => path,
            Err(error) => {
                self.ui_state.status = format!("Could not create project folder: {error}");
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
                return;
            }
        };

        let mut doc = document_from_world_settings(template_id, world_size_m, sea_level);
        doc.name = name;
        doc.presets_used.push(template.name.to_string());

        match save_project(&doc, &path) {
            Ok(()) => {
                self.show_new_template_picker = false;
                self.enter_editor(doc, Some(path.clone()), false);
                self.remember_recent(&path);
                self.ui_state.status = format!("Created {}", path.display());
            }
            Err(error) => {
                self.ui_state.status = format!("Could not create project: {error}");
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
        }
    }

    pub(crate) fn open_project_dialog(&mut self) {
        let projects_root = default_terra_projects_dir();
        let mut dialog = rfd::FileDialog::new().add_filter("Terra Project", &["json"]);
        if projects_root.is_dir() {
            dialog = dialog.set_directory(&projects_root);
        }
        let Some(path) = dialog.pick_file() else {
            return;
        };
        self.open_project_at(path);
    }

    pub(crate) fn open_project_at(&mut self, path: PathBuf) {
        if self.project_io.is_busy() {
            self.ui_state.status = "Load already in progressâ€¦".into();
            return;
        }
        self.ui_state.status = "Loadingâ€¦".into();
        self.project_io.start_load(path);
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    pub(crate) fn enter_editor(
        &mut self,
        mut document: terra_core::document::TerrainDocument,
        path: Option<PathBuf>,
        dirty: bool,
    ) {
        use terra_core::command::CommandHistory;
        document.normalize_wc_tree();

        // Cancel any in-flight eval for the previous document before swapping state.
        self.eval_token = self.eval_token.wrapping_add(1);
        self.eval_worker.set_token(self.eval_token);
        self.worker_refine_pending = false;
        self.force_draft = false;
        self.pending_eval = false;

        // Fresh session (undo stacks, outdated sims, rebuild feedback) — same as a cold open.
        let world_size = (document.metrics.world_size_x, document.metrics.world_size_z);
        let ocean = Some(document.blueprint.sea_level).filter(|v| v.is_finite());
        let mut session = EditorSession::new();
        session.document = document;
        session.history = CommandHistory::default();
        session.dirty_eval = true;
        self.session = session;

        self.project_path = path;
        self.document_dirty = dirty;
        self.screen = AppScreen::Editor;
        self.pending_project_action = None;

        self.reset_runtime_for_document(world_size, ocean);
        self.apply_document_lighting();

        // Editor chrome starts minimized on create/open.
        self.layers_gui
            .reset_collapse_for_project(Some(&self.session.document));
        self.tools_gui.collapse_all_categories();
        self.inspector_gui.reset_expand_for_project();

        self.mark_all_layers_dirty();
        self.request_rebuild();
        self.pending_eval = false;
        self.run_eval_step();
        self.refresh_window_title();
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    /// Drop GPU/CPU preview state so the next document cannot inherit the previous one.
    /// Shared by new-project, open-project, and close-project paths.
    pub(crate) fn reset_runtime_for_document(
        &mut self,
        world_size: (f32, f32),
        ocean_level: Option<f32>,
    ) {
        self.last_height = None;
        self.scheduler.last_good = None;
        self.scheduler.last_aux.clear();
        self.scheduler.last_strata = None;
        self.scheduler.last_layer_timings.clear();
        self.scheduler.quality = PreviewQuality::Draft;
        self.scheduler.evaluator.clear_project_caches();

        self.worker_dirty_from = None;
        self.worker_mark_all_dirty = true;
        self.needs_height_upload = false;
        self.preview_dirty = true;
        self.ui_state.refining = false;
        self.ui_state.build_progress = None;
        self.ui_state.draft_displayed = false;
        self.ui_state.quality = PreviewQuality::Draft;
        self.ui_state.dirty_tile_ids.clear();
        self.pending_tile_uploads.clear();

        let metrics = self.session.document.metrics;
        self.terrain_runtime
            .reconfigure(terra_core::PyramidConfig::new(
                self.session.document.preview_resolution.max(256),
                metrics.world_size_x,
                metrics.world_size_z,
            ));

        if let Some(engine) = self.gpu_engine.as_mut() {
            if let Some(gpu) = self.gpu.as_ref() {
                engine.reset_project_state(&gpu.device, &gpu.queue);
            }
        }
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.reset_project_state(world_size, ocean_level);
        }
        // Preserve the GPU allocation, but make all previous-document pages unreachable.
        if let (Some(atlas), Some(gpu)) = (self.tile_atlas.as_mut(), self.gpu.as_ref()) {
            atlas.clear(&gpu.queue);
            self.ui_state
                .profile
                .update_tile_cache(atlas.residency().stats(), 0);
        }
    }

    pub(crate) fn close_project(&mut self) {
        use terra_core::command::CommandHistory;
        // Cancel in-flight eval work for the closed document.
        self.eval_token = self.eval_token.wrapping_add(1);
        self.eval_worker.set_token(self.eval_token);
        self.pending_eval = false;
        self.worker_refine_pending = false;
        self.force_draft = false;
        self.session = EditorSession::new();
        self.session.history = CommandHistory::default();
        self.project_path = None;
        self.document_dirty = false;
        self.reset_runtime_for_document((1000.0, 1000.0), None);
        self.pending_project_action = None;
        self.screen = AppScreen::Home;
        self.ui_state.status = String::new();
        self.refresh_window_title();
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    pub(crate) fn handle_home_actions(&mut self, actions: Vec<ProjectHomeAction>) {
        for action in actions {
            match action {
                ProjectHomeAction::New => {
                    self.project_home.notice = None;
                    self.request_project_action(PendingProjectAction::New);
                }
                ProjectHomeAction::Open => {
                    self.project_home.notice = None;
                    self.request_project_action(PendingProjectAction::Open);
                }
                ProjectHomeAction::Browse => {
                    self.project_home.notice = None;
                    self.browse_projects_folder();
                }
                ProjectHomeAction::OpenPath(path) => {
                    self.project_home.notice = None;
                    self.request_project_action(PendingProjectAction::OpenPath(path));
                }
                ProjectHomeAction::RemoveRecent(path) => {
                    self.project_prefs.remove_recent(&path);
                    save_project_prefs(&self.project_prefs);
                }
            }
        }
    }

    /// Folder picker: open a Terra project found inside the chosen directory.
    pub(crate) fn browse_projects_folder(&mut self) {
        let projects_root = default_terra_projects_dir();
        let mut dialog = rfd::FileDialog::new();
        if projects_root.is_dir() {
            dialog = dialog.set_directory(&projects_root);
        }
        let Some(dir) = dialog.pick_folder() else {
            return;
        };
        let candidate = dir
            .file_name()
            .map(|n| dir.join(format!("{}.json", n.to_string_lossy())));
        if let Some(path) = candidate.filter(|p| p.is_file()) {
            self.open_project_at(path);
            return;
        }
        if let Ok(entries) = std::fs::read_dir(&dir) {
            let mut jsons: Vec<PathBuf> = entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.extension()
                        .and_then(|e| e.to_str())
                        .is_some_and(|e| e.eq_ignore_ascii_case("json"))
                })
                .collect();
            jsons.sort();
            if let Some(path) = jsons.into_iter().next() {
                self.open_project_at(path);
                return;
            }
        }
        self.project_home.notice = Some(format!(
            "No Terra project (.json) found in {}",
            dir.display()
        ));
        self.ui_state.status = format!("No project found in {}", dir.display());
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    pub(crate) fn choose_export_directory(&mut self) {
        let Some(path) = rfd::FileDialog::new().pick_folder() else {
            return;
        };
        self.ui_state.export_path = Some(path.display().to_string());
        self.ui_state.status = format!("Export directory: {}", path.display());
    }

    pub(crate) fn start_export(&mut self) {
        let path = if let Some(existing) = self.ui_state.export_path.clone() {
            std::path::PathBuf::from(existing)
        } else {
            let Some(path) = rfd::FileDialog::new().pick_folder() else {
                return;
            };
            self.ui_state.export_path = Some(path.display().to_string());
            path
        };
        if !self.exporter.job.done {
            self.ui_state.status = "Export already running".into();
            return;
        }
        // Bake sparse biome paint into masks before the export worker clones the doc.
        self.session.document.sync_all_biome_paint_masks();
        self.terrain_runtime.refinement.begin_export();
        self.ui_state.export_progress = Some(0.0);
        self.ui_state.status = format!("Exporting to {}", path.display());
        self.exporter.start(self.session.document.clone(), path);
    }

    /// Ensure a Shape history layer for the active sculpt tool.
    pub(crate) fn ensure_shape_history_target(
        &mut self,
        tool: terra_core::shape_history::ShapeTool,
    ) -> Option<terra_core::layer::LayerId> {
        use terra_core::shape_history::{
            create_shape_layer, resolve_shape_target, ShapeTargetDecision,
        };
        let decision = resolve_shape_target(
            &self.session.document.stack,
            self.session.document.selected,
            self.ui_state.shape_edit_mode,
            self.ui_state.shape_session_layer,
            tool,
        );
        match decision {
            ShapeTargetDecision::UseExisting(id) => {
                if self.session.document.stack.find(id).is_some_and(|l| {
                    matches!(l.kind, terra_core::layer::LayerKind::SculptStrokes(_))
                }) {
                    self.ui_state.shape_session_layer = Some(id);
                }
                self.session.document.selected = Some(id);
                Some(id)
            }
            ShapeTargetDecision::CreateNew { name, .. } => {
                let layer = create_shape_layer(name);
                let id = layer.id();
                self.session.document.stack.ensure_category_folders();
                self.session.document.stack.push_routed(layer, None, false);
                self.session.document.selected = Some(id);
                self.ui_state.shape_session_layer = Some(id);
                Some(id)
            }
        }
    }

    pub(crate) fn ensure_shape_authoring_layer(&mut self) -> Option<terra_core::layer::LayerId> {
        if let Some(id) = self.session.document.shapes.managed_constraints_layer {
            if self.session.document.stack.find(id).is_some() {
                return Some(id);
            }
        }
        // Prefer an existing TerrainConstraints layer.
        if let Some(existing) = self
            .session
            .document
            .stack
            .flatten_layers()
            .iter()
            .find(|l| matches!(l.kind, terra_core::layer::LayerKind::TerrainConstraints(_)))
        {
            let id = existing.id();
            self.session.document.shapes.managed_constraints_layer = Some(id);
            return Some(id);
        }
        let layer = terra_core::layer::Layer::new(
            "Shape Objects (compiled)",
            terra_core::layer::LayerKind::TerrainConstraints(Default::default()),
        );
        let id = layer.id();
        self.session.document.stack.push_into_category(layer);
        self.session.document.shapes.managed_constraints_layer = Some(id);
        Some(id)
    }

    /// Copy painted biome splat weights into a mask asset bound to the biome group.
    /// Uses the **selected** placement layer (falls back to first). Prefer sparse bake when present.
    pub(crate) fn sync_biome_paint_to_mask(&mut self, biome_id: terra_core::layer::LayerId) {
        self.session.document.sync_biome_paint_to_mask(biome_id);
    }

    pub(crate) fn dispatch_command(&mut self, command: &str) {
        if self.screen == AppScreen::Editor {
            if let Some(workspace) = resolve_workspace_command(command) {
                let previous = self.ui_state.biome_color_preview;
                self.ui_state.switch_workspace(workspace);
                if self.ui_state.biome_color_preview != previous
                    || workspace == crate::ui::WorkspaceId::Biomes
                {
                    self.placement_tint_dirty = true;
                    self.preview_dirty = true;
                }
                return;
            }
        }
        match command {
            CommandId::OPEN_COMMAND_PALETTE if self.screen == AppScreen::Editor => {
                self.ui_state.show_command_palette = true
            }
            CommandId::OPEN_QUICK_ADD if self.screen == AppScreen::Editor => {
                self.ui_state.show_quick_add = true
            }
            CommandId::UNDO if self.screen == AppScreen::Editor => self.undo(),
            CommandId::REDO if self.screen == AppScreen::Editor => self.redo(),
            CommandId::SAVE if self.screen == AppScreen::Editor => self.save_current_project(),
            CommandId::SAVE_AS if self.screen == AppScreen::Editor => self.save_project_as(),
            CommandId::FRAME_TERRAIN if self.screen == AppScreen::Editor => self.frame_terrain(),
            CommandId::NEW_PROJECT => self.request_project_action(PendingProjectAction::New),
            CommandId::OPEN_PROJECT => self.request_project_action(PendingProjectAction::Open),
            CommandId::CLOSE_PROJECT if self.screen == AppScreen::Editor => {
                self.request_project_action(PendingProjectAction::Close)
            }
            _ => {}
        }
    }

    pub(crate) fn frame_terrain(&mut self) {
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.request_camera_reframe();
            renderer.frame_camera_to_terrain();
        }
    }

    pub(crate) fn save_camera_bookmark(&mut self, index: usize) {
        let Some(renderer) = self.renderer.as_ref() else {
            return;
        };
        let camera = &renderer.camera;
        self.ui_state.bookmarks[index] = Some(crate::ui::CameraBookmark {
            x: camera.target.x,
            y: camera.target.y,
            z: camera.target.z,
            yaw: camera.yaw,
            pitch: camera.pitch,
            distance: camera.distance,
        });
        self.ui_state.status = format!("Saved camera bookmark {}", index + 1);
    }

    pub(crate) fn recall_camera_bookmark(&mut self, index: usize) {
        let Some(bookmark) = self.ui_state.bookmarks[index] else {
            self.ui_state.status = format!("Camera bookmark {} is empty", index + 1);
            return;
        };
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.camera.target.x = bookmark.x;
            renderer.camera.target.y = bookmark.y;
            renderer.camera.target.z = bookmark.z;
            renderer.camera.yaw = bookmark.yaw;
            renderer.camera.pitch = bookmark.pitch;
            renderer.camera.distance = bookmark.distance;
            renderer.camera.clamp_to_world(renderer.heights.world_size);
            self.ui_state.status = format!("Recalled camera bookmark {}", index + 1);
        }
    }

    pub(crate) fn undo(&mut self) {
        if let Some(_stroke) = self.session.undo_mask_paint() {
            self.mark_all_layers_dirty();
            self.request_rebuild();
            self.mask_overlay_dirty = true;
            self.preview_dirty = true;
            self.mark_document_dirty();
            self.ui_state.status = "Undid mask paint stroke".into();
            return;
        }
        if self.session.undo_world_rule() {
            self.mark_all_layers_dirty();
            self.mark_document_dirty();
            self.request_rebuild();
            self.ui_state.status = "Undid World Rule edit".into();
            return;
        }
        if self.session.undo_scenario() {
            self.mark_document_dirty();
            self.ui_state.status = "Undid Scenario edit".into();
            return;
        }
        if self.session.undo_paint_stroke() {
            self.mark_all_layers_dirty();
            self.mark_document_dirty();
            self.request_rebuild();
            self.ui_state.status = "Undid biome paint stroke".into();
            return;
        }
        if let Some(id) = self.session.history.undo(&mut self.session.document.stack) {
            self.mark_dirty_from(id);
        } else {
            self.mark_all_layers_dirty();
        }
        self.mark_document_dirty();
        self.request_rebuild();
    }

    pub(crate) fn redo(&mut self) {
        if self.session.redo_world_rule() {
            self.mark_all_layers_dirty();
            self.mark_document_dirty();
            self.request_rebuild();
            self.ui_state.status = "Redid World Rule edit".into();
            return;
        }
        if self.session.redo_scenario() {
            self.mark_document_dirty();
            self.ui_state.status = "Redid Scenario edit".into();
            return;
        }
        if let Some(id) = self.session.history.redo(&mut self.session.document.stack) {
            self.mark_dirty_from(id);
        } else {
            self.mark_all_layers_dirty();
        }
        self.mark_document_dirty();
        self.request_rebuild();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use terra_core::{FieldId, Heightfield, HeightfieldMetrics, TerrainTileKey};
    use terra_gpu::GpuTileAtlas;
    use terra_render::{GpuContext, TerrainRenderer};

    #[test]
    fn new_world_settings_document_round_trips_material_bounds() {
        let doc = document_from_world_settings("alpine", 2_048.0, 0.0);
        let json = doc.to_json().expect("new-world document must serialize");
        assert!(!json.contains("\"min_height\":null"));
        assert!(!json.contains("\"max_height\":null"));
        let loaded = terra_core::document::TerrainDocument::from_json(&json)
            .expect("new-world document must reload");
        let normalized = loaded.to_json().expect("new-world document must resave");
        terra_core::document::TerrainDocument::from_json(&normalized)
            .expect("resaved new-world document must reload");
    }

    /// Revert check for #34: document reset must retain an empty atlas and the
    /// existing upload/sync path must make it streamable again.
    #[test]
    fn reset_then_upload_preserves_atlas_and_invalidates_old_page() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let context = GpuContext {
            device: gpu.device.clone(),
            queue: gpu.queue.clone(),
            surface_format: wgpu::TextureFormat::Rgba8Unorm,
        };
        let mut app = TerraApp::default();
        app.renderer = Some(TerrainRenderer::new_headless(&context, 64, 64));
        app.gpu = Some(context);
        let config = app.terrain_runtime.pyramid.config;
        app.tile_atlas = Some(
            GpuTileAtlas::new(&gpu.device, config.tile_size, config.halo, 4).expect("test atlas"),
        );

        let metrics = HeightfieldMetrics {
            width: config.tile_size,
            height: config.tile_size,
            world_size_x: 1000.0,
            world_size_z: 1000.0,
            tile_size: config.tile_size,
            halo: config.halo,
        };
        app.last_height = Some(Heightfield::filled(metrics, 1.0));
        app.queue_final_tile_uploads();
        let (_, old_level, old_tile) = *app
            .pending_tile_uploads
            .front()
            .expect("old document upload queued");
        assert_eq!(app.upload_pending_terrain_tiles(), 1);
        let old_key = TerrainTileKey {
            layer: None,
            field: FieldId::Height,
            level: old_level,
            tile: old_tile,
        };
        let old_handle = app
            .tile_atlas
            .as_mut()
            .expect("atlas before reset")
            .lookup(&old_key)
            .expect("old page resident");
        assert_eq!(
            app.terrain_runtime
                .pyramid
                .record(&old_key)
                .expect("pyramid mirrors uploaded page")
                .handle,
            old_handle
        );
        assert!(app
            .renderer
            .as_ref()
            .expect("renderer")
            .tile_stream_enabled());
        let atlas_config = {
            let atlas = app.tile_atlas.as_ref().unwrap();
            (atlas.tile_size(), atlas.halo(), atlas.max_pages())
        };

        app.reset_runtime_for_document((1000.0, 1000.0), None);

        let atlas = app.tile_atlas.as_ref().expect("reset must retain atlas");
        assert_eq!(
            (atlas.tile_size(), atlas.halo(), atlas.max_pages()),
            atlas_config
        );
        assert_eq!(atlas.residency().stats().resident_tiles, 0);
        assert_eq!(atlas.residency().resolve_handle(old_handle), None);
        assert!(app.terrain_runtime.pyramid.record(&old_key).is_none());
        assert!(app.pending_tile_uploads.is_empty());
        assert!(!app
            .renderer
            .as_ref()
            .expect("renderer")
            .tile_stream_enabled());

        app.last_height = Some(Heightfield::filled(metrics, 2.0));
        app.queue_final_tile_uploads();
        let (_, new_level, new_tile) = *app
            .pending_tile_uploads
            .front()
            .expect("new document upload queued");
        assert_eq!(app.upload_pending_terrain_tiles(), 1);
        let new_key = TerrainTileKey {
            layer: None,
            field: FieldId::Height,
            level: new_level,
            tile: new_tile,
        };
        let atlas = app.tile_atlas.as_mut().expect("atlas after upload");
        let new_handle = atlas.lookup(&new_key).expect("new page resident");
        assert_eq!(
            app.terrain_runtime
                .pyramid
                .record(&new_key)
                .expect("pyramid mirrors replacement page")
                .handle,
            new_handle
        );
        assert_eq!(new_handle.slot, old_handle.slot);
        assert_ne!(new_handle.generation, old_handle.generation);
        assert_eq!(atlas.residency().resolve_handle(old_handle), None);
        assert!(app
            .renderer
            .as_ref()
            .expect("renderer")
            .tile_stream_enabled());
    }
}
