use std::path::{Path, PathBuf};
#[cfg(any(target_os = "windows", target_os = "macos", unix))]
use std::process::Command;

use crate::ui::{
    project_template_by_id, resolve_workspace_command, CommandId, NewWorldSettings,
    ProjectHomeAction,
};
use terra_core::document::EditorSession;
use terra_io::{save_project, ProjectIoResult};

use super::logical_frame::FrameDeadlineKind;
use super::logical_frame::FrameRequestReason;
use super::{
    default_terra_projects_dir, document_from_world_settings, prepare_project_path,
    project_name_from_path, save_project_prefs, AppScreen, PendingProjectAction, TerraApp,
};
use terra_core::quality::PreviewQuality;

#[cfg(any(target_os = "windows", target_os = "macos", unix))]
fn open_directory(path: &Path) -> std::io::Result<()> {
    #[cfg(target_os = "windows")]
    let program = "explorer.exe";
    #[cfg(target_os = "macos")]
    let program = "open";
    #[cfg(all(unix, not(target_os = "macos")))]
    let program = "xdg-open";

    Command::new(program).arg(path).spawn().map(|_| ())
}

#[cfg(not(any(target_os = "windows", target_os = "macos", unix)))]
fn open_directory(_path: &Path) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "opening directories is unsupported on this platform",
    ))
}

impl TerraApp {
    /// Drain a finished background save/load into the session and status line.
    ///
    /// The poll itself now happens in the [`terra_jobs::JobRegistry`] tick at the
    /// top of `about_to_wait`; this only consumes what that pump surfaced — the
    /// transient status string and the typed [`ProjectIoResult`].
    pub(crate) fn drain_project_io(&mut self) {
        if let Some(status) = self.project_io.status() {
            self.ui_state.status = status.to_string();
            self.request_app_frame(FrameRequestReason::Completion);
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
                self.enter_editor(*doc, Some(path.clone()), false);
                self.project_prefs.push_recent(&path, &name);
                save_project_prefs(&self.project_prefs);
                self.ui_state.status = format!("Loaded {}", path.display());
            }
            ProjectIoResult::Failed { path, error } => {
                self.pending_enter_after_save = None;
                log::error!("{} failed: {error}", path.display());
                self.ui_state.status = format!("{} failed: {error}", path.display());
                self.request_app_frame(FrameRequestReason::Completion);
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
        self.request_app_frame(FrameRequestReason::Completion);
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

    pub(crate) fn remember_recent(&mut self, path: &Path) {
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
            self.request_app_frame(FrameRequestReason::UiActions);
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
        self.request_app_frame(FrameRequestReason::UiActions);
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
            self.request_app_frame(FrameRequestReason::UiActions);
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
                self.request_app_frame(FrameRequestReason::UiActions);
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
                self.request_app_frame(FrameRequestReason::UiActions);
            }
        }
    }

    pub(crate) fn new_infinite_project(
        &mut self,
        settings: terra_core::document::InfiniteProceduralWorldSettings,
    ) {
        if let Err(error) = settings.validate() {
            self.ui_state.status = format!("Invalid Infinite project settings: {error}");
            self.request_app_frame(FrameRequestReason::UiActions);
            return;
        }

        let projects_root = default_terra_projects_dir();
        if let Err(error) = std::fs::create_dir_all(&projects_root) {
            self.ui_state.status = format!("Could not create projects folder: {error}");
            self.request_app_frame(FrameRequestReason::UiActions);
            return;
        }
        let Some(picked) = rfd::FileDialog::new()
            .add_filter("Terra Project", &["json"])
            .set_directory(&projects_root)
            .set_file_name("infinite_world.json")
            .save_file()
        else {
            return;
        };
        let name = project_name_from_path(&picked);
        let path = match prepare_project_path(&projects_root, &name) {
            Ok(path) => path,
            Err(error) => {
                self.ui_state.status = format!("Could not create project folder: {error}");
                self.request_app_frame(FrameRequestReason::UiActions);
                return;
            }
        };
        let mut doc = match terra_core::document::TerrainDocument::new_infinite(settings) {
            Ok(doc) => doc,
            Err(error) => {
                self.ui_state.status = format!("Could not create Infinite project: {error}");
                return;
            }
        };
        doc.name = name;
        doc.presets_used.push("Infinite Procedural World".into());
        match save_project(&doc, &path) {
            Ok(()) => {
                self.show_new_template_picker = false;
                self.enter_editor(doc, Some(path.clone()), false);
                self.remember_recent(&path);
                self.ui_state.status = format!("Created {}", path.display());
            }
            Err(error) => {
                self.ui_state.status = format!("Could not create project: {error}");
                self.request_app_frame(FrameRequestReason::UiActions);
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
        self.request_app_frame(FrameRequestReason::Completion);
    }

    pub(crate) fn enter_editor(
        &mut self,
        mut document: terra_core::document::TerrainDocument,
        path: Option<PathBuf>,
        dirty: bool,
    ) {
        use terra_core::command::CommandHistory;
        document.normalize_wc_tree();

        self.project_generation = self.project_generation.wrapping_add(1);
        self.pipeline_compile.cancel_all();
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.cancel_optional_pipeline_compiles();
        }
        if let Some(overlays) = self.editor_overlays.as_mut() {
            overlays.cancel_compiles();
        }
        self.ui_state.terrain_pipeline_status = crate::ui::TerrainPipelineStatus::Idle;

        // Cancel any in-flight eval for the previous document before swapping state.
        self.eval_token = self.eval_token.wrapping_add(1);
        self.eval_worker.set_token(self.eval_token);
        self.worker_refine_pending = false;
        self.force_draft = false;
        self.pending_eval = false;
        self.pending_eval_immediate = false;

        // Fresh session (undo stacks, outdated sims, rebuild feedback) — same as a cold open.
        let project_world = document.world.clone();
        let ocean = Some(document.blueprint.sea_level).filter(|v| v.is_finite());
        let mut session = EditorSession::with_document(document);
        session.history = CommandHistory::default();
        session.dirty_eval = true;
        self.session = session;

        self.project_path = path;
        self.document_dirty = dirty;
        self.screen = AppScreen::Editor;
        self.pending_project_action = None;

        let presentation_ready = self.reset_runtime_for_document(&project_world, ocean);
        self.apply_document_lighting();

        // Editor chrome starts minimized on create/open.
        self.layers_gui
            .reset_collapse_for_project(Some(&self.session.document));
        self.layers_gui
            .reveal_populated_biome_sections(&self.session.document);
        self.layers_gui.reveal_selection(&self.session.document);
        self.tools_gui.collapse_all_categories();
        self.inspector_gui.reset_expand_for_project();

        self.mark_all_layers_dirty();
        if presentation_ready {
            self.request_rebuild_immediate();
        } else {
            self.pending_eval = false;
            self.pending_eval_immediate = false;
            self.ui_state.status = "Compiling Infinite terrain renderer…".into();
        }
        self.refresh_window_title();
        self.request_app_frame(FrameRequestReason::RequiredEvaluation);
    }

    /// Drop GPU/CPU preview state so the next document cannot inherit the previous one.
    /// Shared by new-project, open-project, and close-project paths.
    pub(crate) fn reset_runtime_for_document(
        &mut self,
        project_world: &terra_core::document::ProjectWorld,
        ocean_level: Option<f32>,
    ) -> bool {
        self.ui_state.profile.infinite_streaming = Default::default();
        self.supersede_gpu_refinement();
        self.last_height = None;
        self.scheduler.last_good = None;
        self.scheduler.last_aux.clear();
        self.scheduler.last_strata = None;
        self.scheduler.last_layer_timings.clear();
        self.scheduler.quality = PreviewQuality::Draft;
        self.scheduler.evaluator.clear_project_caches();
        self.aux_upload_fp = u64::MAX;
        self.veg_upload_fp = u64::MAX;
        self.overhang_upload_fp = u64::MAX;
        self.placement_tint_dirty = true;
        self.mask_overlay_dirty = true;
        self.terrain_plan_cache = terra_core::terrain_plan::TerrainPlanCache::new();
        self.pending_plan_edits = vec![terra_core::terrain_plan::TerrainEditClass::Structure];
        self.pending_plan_invalidation = None;

        self.worker_dirty_from = None;
        self.worker_dirty_region = None;
        self.worker_cache_res = None;
        self.worker_mark_all_dirty = true;
        self.pending_gpu_dirty_region = None;
        self.deferred_full_field = None;
        self.logical_frames
            .clear_deadline(FrameDeadlineKind::FullFieldRefinement);
        self.needs_height_upload = false;
        self.preview_dirty = true;
        self.ui_state.refining = false;
        self.ui_state.evaluation_failure = None;
        self.ui_state.terrain_preview_freshness = crate::ui::TerrainPreviewFreshness::Current;
        self.ui_state.build_progress = None;
        self.ui_state.draft_displayed = false;
        self.ui_state.quality = PreviewQuality::Draft;
        self.ui_state.dirty_tile_ids.clear();
        self.clear_terrain_tile_work();

        let (world_size, traversal_mode, infinite_frame) = match project_world {
            terra_core::document::ProjectWorld::BoundedHeightfield(settings) => {
                self.terrain_runtime
                    .reconfigure(terra_core::PyramidConfig::new(
                        settings.preview_resolution.max(256),
                        settings.metrics.world_size_x,
                        settings.metrics.world_size_z,
                    ));
                (
                    (settings.metrics.world_size_x, settings.metrics.world_size_z),
                    terra_render::TerrainTraversalMode::Bounded,
                    None,
                )
            }
            terra_core::document::ProjectWorld::InfiniteProceduralWorld(settings) => {
                match settings.topology() {
                    Ok(topology) => {
                        if let Err(error) = self.terrain_runtime.try_reconfigure(
                            terra_core::TerrainRuntimeConfig::Infinite(topology.config()),
                        ) {
                            self.ui_state.status = format!("Invalid Infinite topology: {error}");
                        }
                    }
                    Err(error) => {
                        self.ui_state.status = format!("Invalid Infinite topology: {error}");
                    }
                }
                let local_span = (settings.horizon_m * 2.0).clamp(1.0, f64::from(f32::MAX)) as f32;
                (
                    (local_span, local_span),
                    terra_render::TerrainTraversalMode::Infinite,
                    settings.topology().ok().map(|topology| {
                        (
                            topology.config(),
                            settings.horizon_m,
                            settings.preview_radius_m as f32,
                        )
                    }),
                )
            }
        };

        if let Some(gpu) = self.gpu.as_ref() {
            let device_limit = gpu.device.limits().max_texture_array_layers.max(1);
            let (tile_size, halo, max_pages, infinite_topology) = match project_world {
                terra_core::document::ProjectWorld::BoundedHeightfield(_) => {
                    let pyramid = self
                        .terrain_runtime
                        .bounded_pyramid()
                        .expect("bounded runtime configured above");
                    (
                        pyramid.config.tile_size,
                        pyramid.config.halo,
                        128u32.min(device_limit),
                        None,
                    )
                }
                terra_core::document::ProjectWorld::InfiniteProceduralWorld(settings) => (
                    settings.tile_size_samples,
                    settings.publication_halo_samples,
                    u32::try_from(super::eval::infinite_gpu_page_capacity(settings))
                        .unwrap_or(u32::MAX)
                        .min(device_limit)
                        .max(1),
                    settings.topology().ok(),
                ),
            };
            let replace = self.tile_atlas.as_ref().is_none_or(|atlas| {
                atlas.tile_size() != tile_size
                    || atlas.halo() != halo
                    || (infinite_topology.is_some() && atlas.max_pages() != max_pages)
            });
            if replace {
                self.tile_atlas =
                    match terra_gpu::GpuTileAtlas::new(&gpu.device, tile_size, halo, max_pages) {
                        Ok(atlas) => Some(atlas),
                        Err(error) => {
                            self.ui_state.status = format!("GPU tile atlas disabled: {error}");
                            None
                        }
                    };
            }
            if let Some(atlas) = self.tile_atlas.as_mut() {
                if let Some(topology) = infinite_topology {
                    atlas.configure_infinite(&gpu.device, &gpu.queue, topology);
                } else if let Some(pyramid) = self.terrain_runtime.bounded_pyramid() {
                    atlas.configure_hierarchy(&gpu.device, &gpu.queue, pyramid);
                }
            }
        }

        if let Some(engine) = self.gpu_engine.as_mut() {
            if let Some(gpu) = self.gpu.as_ref() {
                engine.reset_project_state(&gpu.device, &gpu.queue);
            }
        }
        let mut compile_infinite = false;
        let mut presentation_ready = true;
        if let Some(renderer) = self.renderer.as_mut() {
            let requested_variant = if infinite_frame.is_some() {
                terra_render::TerrainShaderVariant::Infinite
            } else {
                terra_render::TerrainShaderVariant::Bounded
            };
            if renderer.has_pipeline_variant(requested_variant) {
                if let Err(error) = renderer.activate_pipeline_variant(requested_variant) {
                    self.ui_state.status = error;
                }
                renderer.reset_project_state(world_size, ocean_level, traversal_mode);
                if let Some((topology, horizon_m, preview_radius)) = infinite_frame {
                    renderer.configure_infinite_presentation(
                        terra_render::InfinitePresentationConfig {
                            topology,
                            horizon_m,
                        },
                    );
                    renderer.frame_camera_to_infinite(
                        topology.origin.x_m(),
                        topology.origin.z_m(),
                        preview_radius,
                    );
                }
            } else {
                presentation_ready = false;
                let _ =
                    renderer.activate_pipeline_variant(terra_render::TerrainShaderVariant::Bounded);
                // Keep a neutral, bounded placeholder until the matching bundle
                // is installed. Infinite traversal must not become active early.
                renderer.reset_project_state(
                    world_size,
                    None,
                    terra_render::TerrainTraversalMode::Bounded,
                );
                compile_infinite = true;
            }
        }
        if compile_infinite {
            self.request_presentation_pipeline(
                terra_render::PresentationPipelineFeature::InfiniteTerrain,
                false,
            );
        }
        // Preserve the GPU allocation, but make all previous-document pages
        // unreachable and stop streaming until the new document re-syncs.
        self.retire_streamed_residency();
        presentation_ready
    }

    pub(crate) fn request_presentation_pipeline(
        &mut self,
        feature: terra_render::PresentationPipelineFeature,
        retry: bool,
    ) {
        let state = match feature {
            terra_render::PresentationPipelineFeature::Guides
            | terra_render::PresentationPipelineFeature::Brush => self
                .editor_overlays
                .as_ref()
                .map(|overlays| overlays.state(feature).clone()),
            _ => self
                .renderer
                .as_ref()
                .map(|renderer| renderer.optional_pipeline_state(feature).clone()),
        };
        let should_request = state.is_some_and(|state| {
            matches!(state, terra_render::OptionalResourceState::Absent)
                || (retry && matches!(state, terra_render::OptionalResourceState::Failed { .. }))
        });
        if !should_request {
            return;
        }
        let Some(compiler) = self
            .renderer
            .as_ref()
            .map(|renderer| renderer.presentation_pipeline_compiler())
        else {
            return;
        };
        let submission = self.pipeline_compile.request(
            feature,
            self.project_generation,
            self.device_generation,
            compiler,
        );
        if submission.new_request {
            match feature {
                terra_render::PresentationPipelineFeature::Guides
                | terra_render::PresentationPipelineFeature::Brush => {
                    if let Some(overlays) = self.editor_overlays.as_mut() {
                        overlays.begin_compile(feature, submission.request.id);
                    }
                }
                _ => {
                    if let Some(renderer) = self.renderer.as_mut() {
                        renderer.begin_optional_pipeline_compile(feature, submission.request.id);
                    }
                }
            }
        }
        self.ui_state.terrain_pipeline_status = crate::ui::TerrainPipelineStatus::Pending {
            label: feature.label(),
        };
    }

    pub(crate) fn drain_pipeline_compile(&mut self) {
        use super::pipeline_compile::PipelineCompileStatus;

        if let Some((_, message)) = self.pipeline_compile.retained_failure() {
            self.ui_state.terrain_pipeline_status =
                crate::ui::TerrainPipelineStatus::Failed { message };
        } else {
            match self.pipeline_compile.status().clone() {
                PipelineCompileStatus::Idle => {}
                PipelineCompileStatus::Pending(request) => {
                    self.ui_state.terrain_pipeline_status =
                        crate::ui::TerrainPipelineStatus::Pending {
                            label: request.feature.label(),
                        };
                }
                PipelineCompileStatus::Failed { message, .. } => {
                    self.ui_state.terrain_pipeline_status =
                        crate::ui::TerrainPipelineStatus::Failed { message };
                }
            }
        }

        while let Some(completion) = self.pipeline_compile.take_completion() {
            let request = completion.request;
            if !request.matches_live(self.project_generation, self.device_generation) {
                log::info!("discarding stale {:?} pipeline completion", request.feature);
                continue;
            }
            let bundle = match completion.result {
                Ok(bundle) => bundle,
                Err(message) => {
                    match request.feature {
                        terra_render::PresentationPipelineFeature::Guides
                        | terra_render::PresentationPipelineFeature::Brush => {
                            if let Some(overlays) = self.editor_overlays.as_mut() {
                                overlays.fail_compile(request.feature, request.id, message.clone());
                            }
                        }
                        _ => {
                            if let Some(renderer) = self.renderer.as_mut() {
                                renderer.fail_optional_pipeline_compile(
                                    request.feature,
                                    request.id,
                                    message.clone(),
                                );
                            }
                        }
                    }
                    log::error!("{} compilation failed: {message}", request.feature.label());
                    self.ui_state.terrain_pipeline_status =
                        crate::ui::TerrainPipelineStatus::Failed { message };
                    self.request_app_frame(FrameRequestReason::Completion);
                    continue;
                }
            };

            let install = match request.feature {
                terra_render::PresentationPipelineFeature::Guides
                | terra_render::PresentationPipelineFeature::Brush => {
                    match (
                        self.editor_overlays.as_mut(),
                        self.gpu.as_ref(),
                        self.renderer.as_ref(),
                    ) {
                        (Some(overlays), Some(gpu), Some(renderer)) => {
                            overlays.install(gpu, renderer, request.id, bundle)
                        }
                        _ => Err("editor overlay owner is unavailable".into()),
                    }
                }
                _ => self
                    .renderer
                    .as_mut()
                    .ok_or_else(|| "renderer is unavailable".to_string())
                    .and_then(|renderer| {
                        renderer.install_optional_pipeline_bundle(request.id, bundle)
                    }),
            };
            if let Err(message) = install {
                match request.feature {
                    terra_render::PresentationPipelineFeature::Guides
                    | terra_render::PresentationPipelineFeature::Brush => {
                        if let Some(overlays) = self.editor_overlays.as_mut() {
                            overlays.fail_compile(request.feature, request.id, message.clone());
                        }
                    }
                    _ => {
                        if let Some(renderer) = self.renderer.as_mut() {
                            renderer.fail_optional_pipeline_compile(
                                request.feature,
                                request.id,
                                message.clone(),
                            );
                        }
                    }
                }
                self.pipeline_compile
                    .record_install_failure(request, message.clone());
                self.ui_state.terrain_pipeline_status =
                    crate::ui::TerrainPipelineStatus::Failed { message };
                self.request_app_frame(FrameRequestReason::Completion);
                continue;
            }

            if request.feature == terra_render::PresentationPipelineFeature::InfiniteTerrain {
                self.finish_infinite_pipeline_install();
            } else {
                if matches!(
                    request.feature,
                    terra_render::PresentationPipelineFeature::Overhang
                        | terra_render::PresentationPipelineFeature::Vegetation
                ) {
                    self.needs_height_upload = true;
                }
                self.ui_state.terrain_pipeline_status = crate::ui::TerrainPipelineStatus::Idle;
                self.ui_state.status = format!("{} ready", request.feature.label());
                self.request_app_frame(FrameRequestReason::Completion);
            }
        }
    }

    fn finish_infinite_pipeline_install(&mut self) {
        let Some(settings) = self.session.document.infinite_settings().cloned() else {
            return;
        };
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        if let Err(message) =
            renderer.activate_pipeline_variant(terra_render::TerrainShaderVariant::Infinite)
        {
            self.ui_state.terrain_pipeline_status =
                crate::ui::TerrainPipelineStatus::Failed { message };
            return;
        }
        let Ok(topology) = settings.topology().map(|topology| topology.config()) else {
            return;
        };
        let local_span = (settings.horizon_m * 2.0).clamp(1.0, f64::from(f32::MAX)) as f32;
        let ocean =
            Some(self.session.document.blueprint.sea_level).filter(|value| value.is_finite());
        renderer.reset_project_state(
            (local_span, local_span),
            ocean,
            terra_render::TerrainTraversalMode::Infinite,
        );
        renderer.configure_infinite_presentation(terra_render::InfinitePresentationConfig {
            topology,
            horizon_m: settings.horizon_m,
        });
        renderer.frame_camera_to_infinite(
            topology.origin.x_m(),
            topology.origin.z_m(),
            settings.preview_radius_m as f32,
        );
        self.ui_state.terrain_pipeline_status = crate::ui::TerrainPipelineStatus::Idle;
        self.ui_state.status = "Infinite terrain renderer ready".into();
        self.mark_all_layers_dirty();
        self.request_rebuild_immediate();
        self.request_app_frame(FrameRequestReason::Completion);
    }

    pub(crate) fn retry_failed_pipeline_compile(&mut self) {
        let Some(feature) = self
            .pipeline_compile
            .failed_request()
            .map(|request| request.feature)
        else {
            return;
        };
        self.request_presentation_pipeline(feature, true);
        self.ui_state.status = format!("Retrying {}…", feature.label());
        self.request_app_frame(FrameRequestReason::UiActions);
    }

    pub(crate) fn close_project(&mut self) {
        use terra_core::command::CommandHistory;
        // Cancel in-flight eval work for the closed document.
        self.eval_token = self.eval_token.wrapping_add(1);
        self.eval_worker.set_token(self.eval_token);
        self.pending_eval = false;
        self.pending_eval_immediate = false;
        self.worker_refine_pending = false;
        self.force_draft = false;
        self.project_generation = self.project_generation.wrapping_add(1);
        self.pipeline_compile.cancel_all();
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.cancel_optional_pipeline_compiles();
        }
        if let Some(overlays) = self.editor_overlays.as_mut() {
            overlays.cancel_compiles();
        }
        self.ui_state.terrain_pipeline_status = crate::ui::TerrainPipelineStatus::Idle;
        self.session = EditorSession::new();
        self.session.history = CommandHistory::default();
        self.project_path = None;
        self.document_dirty = false;
        let project_world = self.session.document.world.clone();
        let _ = self.reset_runtime_for_document(&project_world, None);
        self.pending_project_action = None;
        self.screen = AppScreen::Home;
        self.ui_state.status = String::new();
        self.refresh_window_title();
        self.request_app_frame(FrameRequestReason::UiActions);
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
                ProjectHomeAction::OpenLogs => {
                    self.open_logs_folder();
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

    pub(crate) fn open_logs_folder(&mut self) {
        self.project_home.notice = None;
        let result = crate::logging::log_directory().and_then(|directory| {
            std::fs::create_dir_all(&directory).map_err(|error| {
                format!(
                    "could not create log directory {}: {error}",
                    directory.display()
                )
            })?;
            open_directory(&directory).map_err(|error| {
                format!(
                    "could not open log directory {}: {error}",
                    directory.display()
                )
            })?;
            log::info!("opened log directory: {}", directory.display());
            Ok(directory)
        });

        match result {
            Ok(directory) => {
                self.project_home.notice =
                    Some(format!("Opened log folder: {}", directory.display()));
            }
            Err(error) => {
                log::error!("{error}");
                self.project_home.notice = Some(format!("Could not open logs: {error}"));
                self.ui_state.status = error;
            }
        }
        self.request_app_frame(FrameRequestReason::UiActions);
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
        self.request_app_frame(FrameRequestReason::UiActions);
    }

    pub(crate) fn choose_export_directory(&mut self) {
        let Some(path) = rfd::FileDialog::new().pick_folder() else {
            return;
        };
        self.ui_state.export_path = Some(path.display().to_string());
        self.ui_state.status = format!("Export directory: {}", path.display());
    }

    pub(crate) fn start_export(&mut self) {
        if self.gpu.is_none() {
            self.ui_state.status = "Export requires an initialized GPU".into();
            return;
        }
        let path = if let Some(existing) = self.ui_state.export_path.clone() {
            std::path::PathBuf::from(existing)
        } else {
            let Some(path) = rfd::FileDialog::new().pick_folder() else {
                return;
            };
            self.ui_state.export_path = Some(path.display().to_string());
            path
        };
        if self.height_pyramid_export.is_busy() || !self.exporter.job.done {
            self.ui_state.status = "Export already running".into();
            return;
        }
        // Bake sparse biome paint into masks before the export worker clones the doc.
        self.session.document.sync_all_biome_paint_masks();
        self.terrain_runtime.refinement.begin_export();
        self.ui_state.export_progress = Some(0.0);
        self.ui_state.status = format!("Exporting to {}", path.display());
        let generation = self.eval_token.wrapping_add(1).max(1);
        if let Err(error) =
            self.height_pyramid_export
                .start(self.session.document.clone(), path, generation)
        {
            self.ui_state.export_progress = None;
            self.ui_state.status = format!("Export failed: {error}");
            self.terrain_runtime
                .refinement
                .finish_export(self.runtime_started.elapsed().as_millis() as u64);
        }
    }

    /// Ensure a Shape history layer for the active sculpt tool.
    pub(crate) fn ensure_shape_history_target(
        &mut self,
        tool: terra_core::shape_history::ShapeTool,
    ) -> Option<terra_core::layer::LayerId> {
        use terra_core::shape_history::{
            create_shape_layer, resolve_shape_target, ShapeTargetDecision,
        };
        // Track the layer selected before the brush resolves its target, so we
        // can tell the artist when a stroke silently retargets or creates a layer
        // (e.g. brushing with a generator like "Flat" selected redirects to a
        // Sculpt-Strokes layer). Fires once per switch, not per stamp.
        let prev_selected = self.session.document.selected;
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
                if prev_selected != Some(id) {
                    let name = self
                        .session
                        .document
                        .stack
                        .find(id)
                        .map(|l| l.common.name.clone())
                        .unwrap_or_else(|| "Shape Layer".into());
                    let msg = format!("Brush is painting on Shape Layer \"{name}\"");
                    log::info!("{msg}");
                    self.ui_state.status = msg;
                }
                Some(id)
            }
            ShapeTargetDecision::CreateNew { name, .. } => {
                let display_name = name.clone();
                let layer = create_shape_layer(name);
                let id = layer.id();
                self.session.document.stack.ensure_category_folders();
                self.session.document.stack.push_routed(layer, None, false);
                self.session.document.selected = Some(id);
                self.ui_state.shape_session_layer = Some(id);
                // The layer is created once at gesture start. Its first dab is a
                // content edit, but the compiled plan must first observe the new
                // topology; subsequent dabs reuse that single structural revision.
                self.pending_plan_edits
                    .push(terra_core::terrain_plan::TerrainEditClass::Structure);
                let msg = format!("Created new Shape Layer \"{display_name}\" to paint on");
                log::info!("{msg}");
                self.ui_state.status = msg;
                Some(id)
            }
            ShapeTargetDecision::UnavailableOnFoundation { tool, .. } => {
                self.ui_state.status = format!(
                    "{} isn't supported by the selected Foundation layer",
                    tool.label()
                );
                None
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
            renderer.constrain_camera();
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
        match self.session.history.undo_document(
            &mut self.session.document.stack,
            &mut self.session.document.masks,
        ) {
            Some(terra_core::command::CommandImpact::Layer(id)) => self.mark_dirty_from(id),
            Some(terra_core::command::CommandImpact::Masks) => {
                self.mark_all_layers_dirty();
                self.mask_overlay_dirty = true;
                self.preview_dirty = true;
            }
            _ => self.mark_all_layers_dirty(),
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
        match self.session.history.redo_document(
            &mut self.session.document.stack,
            &mut self.session.document.masks,
        ) {
            Some(terra_core::command::CommandImpact::Layer(id)) => self.mark_dirty_from(id),
            Some(terra_core::command::CommandImpact::Masks) => {
                self.mark_all_layers_dirty();
                self.mask_overlay_dirty = true;
                self.preview_dirty = true;
            }
            _ => self.mark_all_layers_dirty(),
        }
        self.mark_document_dirty();
        self.request_rebuild();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::PanelAction;
    use terra_core::{Heightfield, HeightfieldMetrics, TerrainTileKey};
    use terra_gpu::{GpuPageTableEntry, GpuTileAtlas};
    use terra_render::{GpuContext, TerrainRenderer};

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

    #[test]
    fn blank_new_world_has_no_shape_constraints_or_reconstruction() {
        let doc = document_from_world_settings("blank", 4_096.0, 0.0);
        assert!(doc.shapes.shapes.is_empty());
        assert!(doc.shapes.managed_constraints_layer.is_none());
        assert!(doc.stack.flatten_layers().iter().all(|layer| !matches!(
            layer.kind,
            terra_core::LayerKind::TerrainConstraints(_)
                | terra_core::LayerKind::GradientReconstruct(_)
        )));
        let base = doc
            .stack
            .flatten_layers()
            .into_iter()
            .find_map(|layer| match &layer.kind {
                terra_core::LayerKind::SculptBase(params) => Some(params),
                _ => None,
            })
            .expect("blank new world has a sculptable base");
        assert!(!base.samples.is_empty());
        assert!(base.samples.iter().all(|sample| *sample == 8.0));

        let alpine = document_from_world_settings("alpine", 4_096.0, 0.0);
        assert!(!alpine.shapes.shapes.is_empty());
        assert!(alpine.shapes.managed_constraints_layer.is_some());
        assert!(alpine
            .stack
            .flatten_layers()
            .iter()
            .any(|layer| matches!(&layer.kind, terra_core::LayerKind::TerrainConstraints(_))));
    }

    /// #145: a layer-creating brush changes topology once at gesture start; its
    /// first and later dabs are runtime content patches on that retained plan.
    #[test]
    fn shape_layer_creation_compiles_once_and_continuing_dabs_do_not_recompile() {
        use terra_core::authoring::SculptStrokeKind;
        use terra_core::layer::LayerStack;
        use terra_core::shape_history::ShapeTool;
        use terra_core::terrain_plan::{TerrainEditClass, TerrainPlanCache};

        let mut app = TerraApp::default();
        app.session.document.stack = LayerStack::new();
        app.session.document.selected = None;
        app.pending_plan_edits.clear();
        app.terrain_plan_cache = TerrainPlanCache::new();
        let empty = app.session.document.preview_eval_stack();
        app.terrain_plan_cache
            .acquire(&empty, &app.session.document.masks)
            .expect("prime empty authored topology");
        app.terrain_plan_cache.stats_mut().reset();

        let layer = app
            .ensure_shape_history_target(ShapeTool::Raise)
            .expect("Raise creates a Shape Layer");
        assert_eq!(
            app.pending_plan_edits
                .iter()
                .filter(|edit| matches!(edit, TerrainEditClass::Structure))
                .count(),
            1,
            "gesture start records exactly one topology change"
        );

        let dab = |u| PanelAction::PaintSculptStamp {
            layer,
            stamp: bounded_stamp(u, 0.5, 0.04),
            strength: 8.0,
            stroke_kind: SculptStrokeKind::Raise,
            target_height: 0.0,
        };
        app.apply_actions(vec![dab(0.48)]);
        let edits = std::mem::take(&mut app.pending_plan_edits);
        let stack = app.session.document.preview_eval_stack();
        app.terrain_plan_cache
            .update(&stack, &app.session.document.masks, &edits)
            .expect("compile the gesture-start topology");

        for (previous, u) in [(0.48, 0.50), (0.50, 0.52)] {
            app.last_paint_point = Some(bounded_point(previous, 0.5));
            app.apply_actions(vec![dab(u)]);
            let edits = std::mem::take(&mut app.pending_plan_edits);
            assert!(edits.iter().all(|edit| matches!(
                edit,
                TerrainEditClass::Content { owner, .. }
                    if *owner == terra_core::deps::NodeRef::Layer(layer)
            )));
            let stack = app.session.document.preview_eval_stack();
            app.terrain_plan_cache
                .update(&stack, &app.session.document.masks, &edits)
                .expect("patch continuing dab");
        }

        let stats = app.terrain_plan_cache.stats().snapshot();
        assert_eq!(stats.plan_compiles, 1);
        assert_eq!(stats.successful_compiles, 1);
        assert_eq!(stats.plan_cache_hits, 2);
        let terra_core::LayerKind::SculptStrokes(params) =
            &app.session.document.stack.find(layer).unwrap().kind
        else {
            panic!("created target is not a stroke layer");
        };
        assert_eq!(params.strokes.len(), 1, "one continuing authored stroke");
        assert_eq!(params.strokes[0].points.len(), 3, "three runtime dabs");
    }

    /// #145: Base painting is content-only even when the document contains an
    /// authored tree, so repeated dabs retain the structural plan revision.
    #[test]
    fn base_dabs_in_tree_are_content_only_plan_cache_hits() {
        use terra_core::authoring::SculptStrokeKind;
        use terra_core::layer::{Layer, LayerGroup, LayerKind, LayerStack, SculptParams};
        use terra_core::terrain_plan::{TerrainEditClass, TerrainPlanCache};

        let mut app = TerraApp::default();
        let base = Layer::new(
            "Base",
            LayerKind::SculptBase(SculptParams::filled(64, 12.0)),
        );
        let base_id = base.id();
        let mut tree = LayerGroup::new("Tree");
        tree.children
            .push(terra_core::layer::StackNode::Layer(base));
        app.session.document.stack = LayerStack::new();
        app.session.document.stack.push_group(tree);
        app.session.document.selected = Some(base_id);
        app.pending_plan_edits.clear();
        app.terrain_plan_cache = TerrainPlanCache::new();
        let stack = app.session.document.preview_eval_stack();
        app.terrain_plan_cache
            .acquire(&stack, &app.session.document.masks)
            .expect("prime tree plan");
        let revision = app.terrain_plan_cache.structure_revision();
        app.terrain_plan_cache.stats_mut().reset();

        for u in [0.48, 0.50, 0.52] {
            app.apply_actions(vec![PanelAction::PaintSculptStamp {
                layer: base_id,
                stamp: bounded_stamp(u, 0.5, 0.04),
                strength: 3.0,
                stroke_kind: SculptStrokeKind::Raise,
                target_height: 0.0,
            }]);
            let edits = std::mem::take(&mut app.pending_plan_edits);
            assert!(edits.iter().all(|edit| matches!(
                edit,
                TerrainEditClass::Content { owner, .. }
                    if *owner == terra_core::deps::NodeRef::Layer(base_id)
            )));
            let stack = app.session.document.preview_eval_stack();
            app.terrain_plan_cache
                .update(&stack, &app.session.document.masks, &edits)
                .expect("patch Base dab");
        }

        let stats = app.terrain_plan_cache.stats().snapshot();
        assert_eq!(app.terrain_plan_cache.structure_revision(), revision);
        assert_eq!(stats.plan_compiles, 0);
        assert_eq!(stats.plan_cache_hits, 3);
    }

    /// Revert check for #34: document reset must retain an empty atlas and the
    /// existing upload/sync path must make it streamable again.
    #[test]
    fn reset_then_upload_preserves_atlas_and_invalidates_old_page() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let context = GpuContext::new(
            gpu.device.clone(),
            gpu.queue.clone(),
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let mut app = TerraApp::default();
        app.renderer = Some(TerrainRenderer::new_headless(&context, 64, 64));
        app.gpu = Some(context);
        let config = app.terrain_runtime.bounded_pyramid().unwrap().config;
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
        let old_pending = app
            .terrain_tile_scheduler
            .queued_requests()
            .next()
            .expect("old document upload queued");
        let old_key = old_pending.key.tile.clone();
        assert_eq!(app.upload_pending_terrain_tiles(), 1);
        let old_handle = app
            .tile_atlas
            .as_mut()
            .expect("atlas before reset")
            .lookup(&old_key)
            .expect("old page resident");
        assert!(app
            .renderer
            .as_ref()
            .expect("renderer")
            .tile_stream_enabled());
        let atlas_config = {
            let atlas = app.tile_atlas.as_ref().unwrap();
            (atlas.tile_size(), atlas.halo(), atlas.max_pages())
        };

        let world = app.session.document.world.clone();
        app.reset_runtime_for_document(&world, None);

        let atlas = app.tile_atlas.as_ref().expect("reset must retain atlas");
        assert_eq!(
            (atlas.tile_size(), atlas.halo(), atlas.max_pages()),
            atlas_config
        );
        assert_eq!(atlas.residency().stats().resident_tiles, 0);
        assert_eq!(atlas.residency().resolve_handle(old_handle), None);
        assert!(app.terrain_tile_scheduler.is_empty());
        assert!(!app
            .renderer
            .as_ref()
            .expect("renderer")
            .tile_stream_enabled());
        assert!(
            atlas
                .read_page_table_blocking(&gpu.device, &gpu.queue)
                .iter()
                .all(|entry| entry.valid == 0),
            "reset must leave the GPU page table fully invalid"
        );

        app.last_height = Some(Heightfield::filled(metrics, 2.0));
        app.queue_final_tile_uploads();
        let new_pending = app
            .terrain_tile_scheduler
            .queued_requests()
            .next()
            .expect("new document upload queued");
        let new_key = new_pending.key.tile.clone();
        assert_eq!(app.upload_pending_terrain_tiles(), 1);
        let atlas = app.tile_atlas.as_mut().expect("atlas after upload");
        let new_handle = atlas.lookup(&new_key).expect("new page resident");
        assert_eq!(new_handle.slot, old_handle.slot);
        assert_ne!(new_handle.generation, old_handle.generation);
        assert_eq!(atlas.residency().resolve_handle(old_handle), None);
        assert_single_live_page_at_current_revision(&app, &new_key);
    }

    /// Prime an app with a single streamed page resident at the current output
    /// revision. Returns the seeded metrics, the resident page's key, and its
    /// handle so a caller can assert the page is retired after an edit.
    fn app_with_one_streamed_page(
        gpu: &terra_test_gpu::TestGpu,
    ) -> (
        TerraApp,
        HeightfieldMetrics,
        TerrainTileKey,
        terra_core::TilePageHandle,
    ) {
        let context = GpuContext::new(
            gpu.device.clone(),
            gpu.queue.clone(),
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let mut app = TerraApp::default();
        app.renderer = Some(TerrainRenderer::new_headless(&context, 64, 64));
        app.gpu = Some(context);
        let config = app.terrain_runtime.bounded_pyramid().unwrap().config;
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
        let pending = app
            .terrain_tile_scheduler
            .queued_requests()
            .next()
            .expect("page upload queued");
        let key = pending.key.tile.clone();
        assert_eq!(app.upload_pending_terrain_tiles(), 1);
        let handle = app
            .tile_atlas
            .as_mut()
            .expect("atlas")
            .lookup(&key)
            .expect("page resident");
        let entry = assert_single_live_page_at_current_revision(&app, &key);
        assert_eq!(
            entry.generation, handle.generation,
            "page table records the seeded handle's generation"
        );
        (app, metrics, key, handle)
    }

    /// Assert the revision boundary retired streamed residency: the atlas has no
    /// resident tiles and cannot resolve the old handle, no uploads remain queued,
    /// and the renderer has stopped streaming (so the shader falls back to the
    /// monolithic texture).
    fn assert_streamed_residency_retired(app: &TerraApp, old_handle: terra_core::TilePageHandle) {
        let atlas = app.tile_atlas.as_ref().expect("atlas retained");
        assert_eq!(atlas.residency().stats().resident_tiles, 0);
        assert_eq!(atlas.residency().resolve_handle(old_handle), None);
        assert!(app.terrain_tile_scheduler.is_empty());
        assert!(!app
            .renderer
            .as_ref()
            .expect("renderer")
            .tile_stream_enabled());

        // The CPU mirrors above can agree while the GPU page table the shader
        // actually reads still holds the retired revision's pages -- that is the
        // exact E1-C2 divergence. Read the buffer back and require no row survives.
        let gpu = app.gpu.as_ref().expect("gpu context");
        assert!(
            atlas
                .read_page_table_blocking(&gpu.device, &gpu.queue)
                .iter()
                .all(|entry| entry.valid == 0),
            "no page-table row may stay valid once residency is retired"
        );
    }

    /// Assert the atlas holds exactly one resident page, that it carries `key`'s
    /// identity stamped with the runtime's *current* output revision, and that the
    /// renderer streams at that same revision. This is the post-sync invariant the
    /// shader's stale-page gate relies on (uniform revision == page-stamp revision);
    /// returns the live entry so callers can assert more (e.g. handle generation).
    fn assert_single_live_page_at_current_revision(
        app: &TerraApp,
        key: &TerrainTileKey,
    ) -> GpuPageTableEntry {
        let revision = app.terrain_runtime.output_revision();
        let renderer = app.renderer.as_ref().expect("renderer");
        assert!(
            renderer.tile_stream_enabled(),
            "streaming stays enabled after a completed sync"
        );
        assert_eq!(
            renderer.tile_stream_revision(),
            revision,
            "renderer streams the current output revision"
        );
        let gpu = app.gpu.as_ref().expect("gpu context");
        let atlas = app.tile_atlas.as_ref().expect("atlas");
        let live: Vec<GpuPageTableEntry> = atlas
            .read_page_table_blocking(&gpu.device, &gpu.queue)
            .into_iter()
            .filter(|entry| entry.valid != 0)
            .collect();
        assert_eq!(live.len(), 1, "exactly one page resident after sync");
        let entry = live[0];
        let (level, tile) = app
            .terrain_runtime
            .bounded_pyramid()
            .unwrap()
            .level_and_tile(key.address)
            .expect("resident key belongs to bounded pyramid");
        assert_eq!(entry.level as u8, level, "page level matches key");
        assert_eq!(
            (entry.tile_x, entry.tile_z),
            (tile.tx, tile.tz),
            "page tile coords match key"
        );
        assert_eq!(
            (entry.output_revision_hi, entry.output_revision_lo),
            ((revision >> 32) as u32, revision as u32),
            "page stamped with the current output revision"
        );
        entry
    }

    /// Ratchet 3 (count honesty): every residency source must agree, so the HUD
    /// can never report "no residency" while stale pages still render (the E1-C2
    /// symptom).
    ///
    /// The sources cross-checked are the GPU page table (what the shader actually
    /// samples, authoritative), `atlas.residency().stats()` (the CPU mirror the HUD
    /// is fed from), and `profile.tile_cache_resident` (the count the artist reads).
    /// It also forbids stale rows and pins stream-enable honesty (streaming on only
    /// with live pages at the current revision). With the write-only CPU pyramid
    /// plan retired (#91) and its name blacklist replaced by behavioral authority
    /// guards (#171), residency has one shader-visible source (the GPU page table)
    /// and one CPU policy mirror (`TileResidencyCache`).
    fn assert_residency_sources_agree(app: &TerraApp) {
        let revision = app.terrain_runtime.output_revision();
        let rev_lo = revision as u32;
        let rev_hi = (revision >> 32) as u32;

        let entries = {
            let gpu = app.gpu.as_ref().expect("gpu context");
            app.tile_atlas
                .as_ref()
                .expect("atlas")
                .read_page_table_blocking(&gpu.device, &gpu.queue)
        };
        let live = entries.iter().filter(|e| e.valid != 0).count();
        let stale = entries
            .iter()
            .filter(|e| {
                e.valid != 0 && (e.output_revision_lo, e.output_revision_hi) != (rev_lo, rev_hi)
            })
            .count();
        assert_eq!(
            stale, 0,
            "no page-table row may carry a prior output revision"
        );

        // GPU truth vs the CPU mirror that feeds the HUD, and the HUD field itself.
        let resident = app
            .tile_atlas
            .as_ref()
            .expect("atlas")
            .residency()
            .stats()
            .resident_tiles;
        assert_eq!(
            resident, live,
            "atlas residency stats must match the page-table valid count"
        );
        assert_eq!(
            app.ui_state.profile.tile_cache_resident, live,
            "HUD tile_cache_resident must match the page-table valid count"
        );

        // Stream-enable honesty: streaming is on only with live pages at this revision.
        let (enabled, stream_rev) = {
            let renderer = app.renderer.as_ref().expect("renderer");
            (
                renderer.tile_stream_enabled(),
                renderer.tile_stream_revision(),
            )
        };
        if enabled {
            assert!(live >= 1, "streaming enabled while the page table is empty");
            assert_eq!(
                stream_rev, revision,
                "streaming enabled at a revision the pages are not stamped with"
            );
        }
    }

    /// Revert check for #86: advancing the output revision via `mark_dirty_from`
    /// must retire the prior revision's streamed pages, and the #34 upload/sync
    /// lifecycle must then re-populate the atlas and re-enable streaming.
    #[test]
    fn edit_via_mark_dirty_from_retires_streamed_pages_until_resync() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let (mut app, metrics, _old_key, old_handle) = app_with_one_streamed_page(gpu);
        let revision_before = app.terrain_runtime.output_revision();

        // An unknown layer id dirties the whole stack — the strongest edit shape.
        app.mark_dirty_from(terra_core::LayerId::new());

        assert_ne!(app.terrain_runtime.output_revision(), revision_before);
        assert_streamed_residency_retired(&app, old_handle);

        // #34 lifecycle: worker completion re-queues tiles and re-enables streaming.
        app.last_height = Some(Heightfield::filled(metrics, 2.0));
        app.queue_final_tile_uploads();
        let new_pending = app
            .terrain_tile_scheduler
            .queued_requests()
            .next()
            .expect("resync upload queued");
        let new_key = new_pending.key.tile.clone();
        assert_eq!(app.upload_pending_terrain_tiles(), 1);
        // Re-enable ratchet: the page repopulates and the renderer streams again,
        // now stamped with the post-edit revision (not the retired one).
        assert_ne!(app.terrain_runtime.output_revision(), revision_before);
        assert_single_live_page_at_current_revision(&app, &new_key);
    }

    /// The stage-aware edit path advances the revision too, so it must retire
    /// streamed pages identically.
    #[test]
    fn edit_via_mark_dirty_from_stage_retires_streamed_pages() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let (mut app, _metrics, _old_key, old_handle) = app_with_one_streamed_page(gpu);
        let revision_before = app.terrain_runtime.output_revision();

        app.mark_dirty_from_stage(terra_core::LayerId::new());

        assert_ne!(app.terrain_runtime.output_revision(), revision_before);
        assert_streamed_residency_retired(&app, old_handle);
    }

    /// Whole-stack edits advance the revision via `reconfigure`; that boundary
    /// must retire streamed pages as well.
    #[test]
    fn mark_all_layers_dirty_retires_streamed_pages() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let (mut app, _metrics, _old_key, old_handle) = app_with_one_streamed_page(gpu);

        app.mark_all_layers_dirty();

        assert_streamed_residency_retired(&app, old_handle);
    }

    /// Consistency ratchet (#87): the residency counts the HUD reads and the page
    /// table the shader samples must agree at every point in the edit lifecycle,
    /// so the E1-C2 divergence -- HUD reporting no residency while stale pages
    /// still render -- cannot silently reappear.
    #[test]
    fn residency_counts_agree_across_hud_sources_and_page_table() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let (mut app, metrics, _old_key, _old_handle) = app_with_one_streamed_page(gpu);

        // Stage 1: freshly streamed -- every source reports the one resident page.
        assert_residency_sources_agree(&app);

        // Stage 2: the E1-C2 moment. The edit retires residency; no source may
        // still count (or let the shader sample) the prior revision's page.
        app.mark_dirty_from(terra_core::LayerId::new());
        assert_residency_sources_agree(&app);

        // Stage 3: worker completion re-streams; every source agrees once more,
        // now at the post-edit revision.
        app.last_height = Some(Heightfield::filled(metrics, 2.0));
        app.queue_final_tile_uploads();
        assert_eq!(app.upload_pending_terrain_tiles(), 1);
        assert_residency_sources_agree(&app);
    }

    /// Revert check for the streamed-tile corner artifact: the streamed resolution
    /// the renderer hands the shader must track the resident pages (`last_height`),
    /// independent of the monolithic height-texture size. If the shader
    /// denormalized streamed UVs by `tex_size` again, a coarse result would render
    /// into a `page_res / tex_size` corner at the origin.
    #[test]
    fn streamed_resolution_tracks_pages_not_monolithic_tex_size() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let (mut app, metrics, _key, _handle) = app_with_one_streamed_page(gpu);
        let page_res = metrics.width;

        {
            let renderer = app.renderer.as_ref().expect("renderer");
            assert!(renderer.tile_stream_enabled());
            assert_eq!(renderer.tile_stream_res(), (page_res, page_res));
        }

        // Drive the monolithic texture to a larger, different resolution, then
        // re-sync. The streamed resolution must still equal the resident pages'
        // resolution, not the monolithic tex_size.
        let big = HeightfieldMetrics {
            width: page_res * 2,
            height: page_res * 2,
            world_size_x: metrics.world_size_x,
            world_size_z: metrics.world_size_z,
            tile_size: metrics.tile_size,
            halo: metrics.halo,
        };
        app.renderer
            .as_mut()
            .expect("renderer")
            .upload_heightfield(&Heightfield::filled(big, 3.0));
        app.sync_tile_stream_to_renderer();

        let renderer = app.renderer.as_ref().expect("renderer");
        assert!(renderer.tile_stream_enabled());
        assert_eq!(
            renderer.tile_stream_res(),
            (page_res, page_res),
            "streamed resolution must track resident pages, not the monolithic tex_size"
        );
    }

    /// A result whose resolution is not a pyramid level (e.g. an interactive
    /// Export-quality field larger than every level) must not stream: the upload
    /// queue stays empty and the renderer falls back to the monolithic texture,
    /// instead of stamping pages at a fabricated level the shader samples at a
    /// different one (the divergent `max_level()` vs `0` fallback this replaces).
    #[test]
    fn non_pyramid_resolution_disables_streaming() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let (mut app, metrics, _key, _handle) = app_with_one_streamed_page(gpu);
        assert!(app
            .renderer
            .as_ref()
            .expect("renderer")
            .tile_stream_enabled());

        // An odd resolution cannot be a power-of-two pyramid level.
        let odd = metrics.width * 2 + 1;
        let non_level = HeightfieldMetrics {
            width: odd,
            height: odd,
            world_size_x: metrics.world_size_x,
            world_size_z: metrics.world_size_z,
            tile_size: metrics.tile_size,
            halo: metrics.halo,
        };
        app.last_height = Some(Heightfield::filled(non_level, 1.0));
        assert_eq!(
            app.streamed_level_for_last_height(),
            None,
            "an odd resolution must not resolve to a pyramid level"
        );

        app.queue_final_tile_uploads();
        assert!(
            app.terrain_tile_scheduler.is_empty(),
            "a non-pyramid resolution must not queue tile uploads"
        );

        app.sync_tile_stream_to_renderer();
        assert!(
            !app.renderer
                .as_ref()
                .expect("renderer")
                .tile_stream_enabled(),
            "a non-pyramid resolution must fall back to the monolithic path"
        );
    }
}
