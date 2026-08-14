use std::time::{Duration, Instant};

use crate::ui::Preview2dMode;
use terra_core::eval::{EvalWorkRequest, PreviewQuality};
use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
use terra_core::layer::{LayerId, LayerKind};
use terra_core::mask::bake_mask_assets;

use super::{quality_stage_progress, TerraApp};

/// Interactive Full preview ceiling — same 1 m footing as WC (world metres ≈ samples),
/// capped at export-class 8192² so extreme worlds stay bounded.
const INTERACTIVE_PREVIEW_CAP: u32 = 8192;

impl TerraApp {
    pub(crate) fn request_rebuild(&mut self) {
        self.eval_token = self.scheduler.request_rebuild();
        self.eval_worker.set_token(self.eval_token);
        self.worker_refine_pending = false;
        self.ui_state.refining = true;
        self.ui_state.quality = PreviewQuality::Draft;
        self.ui_state.build_progress = Some(0.0);
        let preview_stack = self.session.document.preview_eval_stack();
        self.ui_state.refining_layer_name = self
            .session
            .document
            .selected
            .and_then(|id| preview_stack.find(id))
            .map(|layer| layer.common.name.clone())
            .or_else(|| Some("terrain".into()));
        self.scheduler.quality = PreviewQuality::Draft;
        self.last_edit = Instant::now();
        self.pending_eval = true;
        self.force_draft = true;
        self.ui_state.profile.gen_id = self.eval_token;
    }

    /// Interactive structural edit (add filter/layer): keep current viewport resolution
    /// and present supported work immediately through the GPU. Unsupported CPU work is
    /// submitted to the existing eval worker, leaving last-good content on screen until
    /// the worker publishes its result.
    pub(crate) fn request_rebuild_immediate(&mut self) {
        // Cancel any in-flight CPU job so a late Full result cannot pop the viewport.
        self.eval_token = self.eval_token.wrapping_add(1);
        self.scheduler.current_token = self.eval_token;
        self.eval_worker.set_token(self.eval_token);
        self.worker_refine_pending = false;
        self.force_draft = false;
        self.pending_eval = false;
        self.ui_state.refining = false;
        self.ui_state.build_progress = None;
        self.ui_state.profile.gen_id = self.eval_token;
        self.last_edit = Instant::now();

        // Match the resolution already on screen (renderer tex or last_height).
        if let Some(r) = self.renderer.as_ref() {
            let w = r.heights.tex_size.0.max(1);
            let preview = self
                .session
                .document
                .preview_resolution
                .min(INTERACTIVE_PREVIEW_CAP);
            let medium = PreviewQuality::Medium.resolution(preview, preview);
            self.scheduler.quality = if w >= preview {
                PreviewQuality::Full
            } else if w >= medium {
                PreviewQuality::Medium
            } else {
                PreviewQuality::Draft
            };
            self.ui_state.quality = self.scheduler.quality;
        }

        if self.gpu_engine.is_some() && self.renderer.is_some() {
            self.run_eval_step();
            self.last_refine = Instant::now();
            // Incomplete/failed GPU present → ensure Draft CPU is running.
            let needs_cpu = matches!(
                self.ui_state.profile.path,
                "GPU→async CPU" | "async CPU" | "GPU*" | ""
            );
            if needs_cpu && !self.worker_refine_pending {
                let q = PreviewQuality::Draft;
                self.scheduler.quality = q;
                self.ui_state.quality = q;
                self.enqueue_async_eval(q);
            }
        } else {
            self.request_rebuild();
            self.last_edit = Instant::now()
                .checked_sub(Duration::from_secs(1))
                .unwrap_or_else(Instant::now);
        }
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    /// Submit the current Medium/Full snapshot to the CPU worker without blocking the UI.
    pub(crate) fn enqueue_refine_job(&mut self) {
        self.enqueue_async_eval(self.scheduler.quality);
    }

    /// Offload CPU stack eval to the worker — never block the UI thread.
    pub(crate) fn enqueue_async_eval(&mut self, quality: PreviewQuality) {
        let preview_stack = self.session.document.preview_eval_stack();
        self.ui_state.refining_layer_name = self
            .session
            .document
            .selected
            .and_then(|id| preview_stack.find(id))
            .map(|layer| layer.common.name.clone())
            .or_else(|| Some("terrain".into()));
        let mark_all_dirty = std::mem::take(&mut self.worker_mark_all_dirty);
        let dirty_from = self.worker_dirty_from.take();

        self.eval_worker.submit(EvalWorkRequest {
            token: self.eval_token,
            quality,
            stack: preview_stack,
            masks: self.session.document.masks.clone(),
            base_metrics: self.session.document.metrics,
            level_steps: self.session.document.level_steps.clone(),
            preview_res: self
                .session
                .document
                .preview_resolution
                .min(INTERACTIVE_PREVIEW_CAP),
            export_res: self.session.document.export_resolution,
            aux: self.scheduler.last_aux.clone(),
            strata: self.scheduler.last_strata.clone(),
            mask_reference: self.scheduler.last_good.clone(),
            mark_all_dirty,
            dirty_from,
        });
        self.worker_refine_pending = true;
        self.ui_state.refining = true;
    }

    pub(crate) fn queue_final_tile_uploads(&mut self) {
        self.pending_tile_uploads.clear();
        if self.tile_atlas.is_none() {
            return;
        }
        let Some(height) = self.last_height.as_ref() else {
            return;
        };
        let level = self
            .terrain_runtime
            .pyramid
            .levels
            .iter()
            .position(|candidate| candidate.resolution == height.metrics.width)
            .unwrap_or_else(|| self.terrain_runtime.pyramid.max_level() as usize)
            as u8;
        let revision = self.terrain_runtime.output_revision();
        self.pending_tile_uploads
            .extend(height.tiles().iter().map(|tile| (revision, level, tile.id)));
    }

    pub(crate) fn upload_pending_terrain_tiles(&mut self) -> usize {
        if self.tile_atlas.is_none() || self.gpu.is_none() || self.last_height.is_none() {
            self.pending_tile_uploads.clear();
            return 0;
        }
        let budget_us = self.terrain_runtime.refinement.state().terrain_budget_us();
        let upload_limit = if budget_us == u64::MAX {
            128
        } else {
            (budget_us / 250).clamp(1, 32) as usize
        };
        let live_revision = self.terrain_runtime.output_revision();
        let mut uploaded = 0;
        while uploaded < upload_limit {
            let Some((revision, level, tile_id)) = self.pending_tile_uploads.pop_front() else {
                break;
            };
            if revision != live_revision {
                continue;
            }
            let Some(tile) = self
                .last_height
                .as_ref()
                .and_then(|height| height.tile(tile_id))
            else {
                continue;
            };
            let key = terra_core::TerrainTileKey {
                layer: None,
                field: terra_core::FieldId::Height,
                level,
                tile: tile_id,
            };
            let published_key = key.clone();
            let result = self.tile_atlas.as_mut().unwrap().upload_height_tile(
                &self.gpu.as_ref().unwrap().queue,
                key,
                tile,
                revision,
                revision,
            );
            match result {
                Ok(upload) => {
                    uploaded += 1;
                    let frame = self.runtime_started.elapsed().as_millis() as u64;
                    for evicted in upload.evicted {
                        self.terrain_runtime.pyramid.remove_resident(&evicted);
                    }
                    self.terrain_runtime.pyramid.publish_resident(
                        published_key,
                        upload.handle,
                        revision,
                        revision,
                        frame,
                    );
                }
                Err(error) => {
                    log::warn!("terrain tile upload failed: {error}");
                    self.pending_tile_uploads.clear();
                    break;
                }
            }
            if let Some(atlas) = self.tile_atlas.as_ref() {
                self.ui_state
                    .profile
                    .update_tile_cache(atlas.residency().stats(), self.pending_tile_uploads.len());
            }
        }
        if uploaded > 0 {
            self.sync_tile_stream_to_renderer();
        }
        uploaded
    }

    pub(crate) fn sync_tile_stream_to_renderer(&mut self) {
        let (atlas_view, page_table, tile_size, halo, max_pages, _resident, budget_util) = {
            let Some(atlas) = self.tile_atlas.as_ref() else {
                return;
            };
            (
                atlas.create_texture_view(),
                atlas.page_table_buffer_cloned(),
                atlas.tile_size(),
                atlas.halo(),
                atlas.max_pages(),
                atlas.residency().stats().resident_tiles,
                atlas.memory_budget().utilization(),
            )
        };
        let _ = budget_util;
        let level = self
            .last_height
            .as_ref()
            .and_then(|height| {
                self.terrain_runtime
                    .pyramid
                    .levels
                    .iter()
                    .position(|candidate| candidate.resolution == height.metrics.width)
            })
            .unwrap_or(0) as u8;
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        renderer.set_tile_stream_resources(
            atlas_view, page_table, tile_size, halo, max_pages, level,
            // Streamed height is primary; shader falls back to monolithic on miss.
            true,
        );
    }

    pub(crate) fn mark_dirty_from(&mut self, id: LayerId) {
        let preview = self.session.document.preview_eval_stack();
        self.scheduler.evaluator.mark_dirty_from(&preview, id);
        self.terrain_runtime.advance_output_revision();
        self.track_worker_dirty_from(&preview, id);
        if let Some(gpu) = self.gpu_engine.as_mut() {
            gpu.mark_dirty_from(&preview, id);
        }
    }

    pub(crate) fn mark_dirty_from_stage(&mut self, id: LayerId) {
        let preview = self.session.document.preview_eval_stack();
        self.scheduler.evaluator.mark_dirty_from_stage(&preview, id);
        self.terrain_runtime.advance_output_revision();
        self.track_worker_dirty_from(&preview, id);
        if let Some(gpu) = self.gpu_engine.as_mut() {
            // GPU path still uses suffix dirty; stage-aware CPU cache is the main win.
            gpu.mark_dirty_from(&preview, id);
        }
    }

    pub(crate) fn track_worker_dirty_from(
        &mut self,
        stack: &terra_core::layer::LayerStack,
        id: LayerId,
    ) {
        if self.worker_mark_all_dirty {
            return;
        }
        let ids = stack.layer_ids();
        let Some(next_index) = ids.iter().position(|candidate| *candidate == id) else {
            self.worker_mark_all_dirty = true;
            self.worker_dirty_from = None;
            return;
        };
        let replace = self
            .worker_dirty_from
            .and_then(|current| ids.iter().position(|candidate| *candidate == current))
            .is_none_or(|current_index| next_index < current_index);
        if replace {
            self.worker_dirty_from = Some(id);
        }
    }

    pub(crate) fn mark_all_layers_dirty(&mut self) {
        let preview = self.session.document.preview_eval_stack();
        self.scheduler.evaluator.mark_all_dirty(&preview);
        let metrics = self.session.document.metrics;
        self.terrain_runtime
            .reconfigure(terra_core::PyramidConfig::new(
                self.session.document.preview_resolution,
                metrics.world_size_x,
                metrics.world_size_z,
            ));
        self.worker_mark_all_dirty = true;
        self.worker_dirty_from = None;
        if let Some(gpu) = self.gpu_engine.as_mut() {
            gpu.mark_all_dirty(&preview);
        }
    }

    /// After Shape history edits: mark downstream simulation layers outdated
    /// without scheduling an immediate sim rebuild.
    pub(crate) fn mark_shape_dependents_outdated(&mut self) {
        let Some(shape_id) = self
            .ui_state
            .shape_session_layer
            .or(self.session.document.selected)
        else {
            return;
        };
        self.mark_dirty_from(shape_id);
        let now_ms = self.last_edit.elapsed().as_millis().saturating_add(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0),
        ) as u64;
        let feedback = terra_core::rebuild_feedback::apply_upstream_change(
            &mut self.session,
            terra_core::deps::NodeRef::Layer(shape_id),
            "Shape / region geometry edited",
            now_ms,
        );
        if !feedback.is_empty() {
            self.ui_state.status = feedback.format_updating().replace('\n', " Â· ");
            self.ui_state.affected_feedback = Some(feedback.format_updating());
        }
    }

    /// Bind / create Simulation Layers for unbound scenario passes (reuse existing solvers).
    pub(crate) fn ensure_scenario_layers(
        &mut self,
        id: terra_core::simulation_scenario::SimulationScenarioId,
    ) {
        use terra_core::layer::{Layer, StackCategory};
        use terra_core::simulation_scenario::layer_kind_is_scenario_compatible;

        let Some(scenario) = self.session.document.simulation_scenarios.get(id).cloned() else {
            return;
        };
        for pass in &scenario.passes {
            if !pass.enabled {
                continue;
            }
            if let Some(lid) = pass.layer_id {
                if self.session.document.stack.find(lid).is_some() {
                    continue;
                }
            }
            // Prefer an existing compatible unbound layer of the same kind.
            let existing = self
                .session
                .document
                .stack
                .flatten_layers()
                .into_iter()
                .find(|l| {
                    layer_kind_is_scenario_compatible(&l.kind)
                        && terra_core::simulation_scenario::ScenarioPassKind::from_layer_kind(
                            &l.kind,
                        ) == Some(pass.kind)
                })
                .map(|l| l.id());
            if let Some(lid) = existing {
                if let Some(s) = self.session.document.simulation_scenarios.get_mut(id) {
                    let _ = s.bind_pass_layer(pass.id, lid);
                }
                continue;
            }
            // Create a new Simulation Layer via the existing framework.
            let kind = pass.kind.default_layer_kind();
            let layer = Layer::new(pass.name.clone(), kind);
            let lid = layer.id();
            self.session.document.stack.push_into_category(layer);
            // Ensure it landed in Simulation category when possible.
            let _ = StackCategory::Simulation;
            if let Some(s) = self.session.document.simulation_scenarios.get_mut(id) {
                let _ = s.bind_pass_layer(pass.id, lid);
            }
        }
    }

    /// When a biome group's DistNode stack is edited by hand, freeze Placement rules.
    pub(crate) fn mark_biome_placement_custom_for_group(
        doc: &mut terra_core::document::TerrainDocument,
        group_id: terra_core::layer::LayerId,
    ) {
        let stack = doc
            .stack
            .find_group(group_id)
            .map(|g| g.masks.clone())
            .unwrap_or_default();
        if let Some(def) = doc.biome_library.by_group_mut(group_id) {
            def.placement.mark_mask_stack_custom(stack);
        }
    }

    pub(crate) fn run_eval_step(&mut self) {
        profiling::scope!("eval_step");
        let t0 = Instant::now();
        if self.force_draft {
            self.scheduler.quality = PreviewQuality::Draft;
            self.force_draft = false;
        }
        let preview = self
            .session
            .document
            .preview_resolution
            .min(INTERACTIVE_PREVIEW_CAP);
        let export = self.session.document.export_resolution;
        let base = self.session.document.metrics;
        let quality = self.scheduler.quality;
        // Global + active region â€” interactive preview must include world.global.
        let preview_stack = self.session.document.preview_eval_stack();
        self.ui_state.refining_layer_name = self
            .session
            .document
            .selected
            .and_then(|id| preview_stack.find(id))
            .map(|layer| layer.common.name.clone())
            .or_else(|| Some("terrain".into()));
        let res = quality.resolution(preview, export);
        let metrics = HeightfieldMetrics {
            width: res,
            height: res,
            world_size_x: base.world_size_x,
            world_size_z: base.world_size_z,
            tile_size: base.tile_size.min(res),
            halo: base.halo,
        };
        let token = self.eval_token;
        // Interactive evaluation must never force a GPU readback/Wait on the UI thread.
        let want_cpu = false;

        // Bridge for interactive suffix edits:
        // 1) CPU layer checkpoint for prev id (param tweaks)
        // 2) last_height / last_good when first dirty layer is *new* (add filter) —
        //    safe prefix: prior stack output does not include the new layer yet.
        let bridge_owned: Option<Heightfield> = {
            let layers = preview_stack.flatten_layers();
            let first_dirty = self
                .gpu_engine
                .as_ref()
                .and_then(|gpu| gpu.first_dirty_index(&preview_stack))
                .unwrap_or(0);
            if first_dirty == 0 || first_dirty >= layers.len() {
                None
            } else {
                let prev_id = layers[first_dirty - 1].id();
                let dirty_id = layers[first_dirty].id();
                let from_cache = self.scheduler.evaluator.cache.get(prev_id).and_then(|c| {
                    if c.dirty {
                        return None;
                    }
                    Some(c.height.clone())
                });
                from_cache.or_else(|| {
                    let gpu_has_new = self
                        .gpu_engine
                        .as_ref()
                        .is_some_and(|g| g.has_layer_cache(dirty_id, metrics));
                    let cpu_has_new = self
                        .scheduler
                        .evaluator
                        .cache
                        .get(dirty_id)
                        .is_some_and(|c| !c.dirty);
                    let is_new_layer = !gpu_has_new && !cpu_has_new;
                    if !is_new_layer {
                        return None;
                    }
                    self.last_height
                        .clone()
                        .or_else(|| self.scheduler.last_good.as_ref().map(|h| (**h).clone()))
                })
            }
        };
        let bridge_prefix = bridge_owned.as_ref();

        let mut used_gpu = false;
        let mut eval_completed = false;
        if token == self.eval_token {
            if let (Some(engine), Some(renderer), Some(gpu)) = (
                self.gpu_engine.as_mut(),
                self.renderer.as_mut(),
                self.gpu.as_ref(),
            ) {
                match engine.evaluate(
                    &gpu.device,
                    &gpu.queue,
                    &preview_stack,
                    &self.session.document.masks,
                    metrics,
                    quality,
                    want_cpu,
                    bridge_prefix,
                ) {
                    Ok(result) => {
                        if token != self.eval_token {
                            // Stale generation â€” discard.
                        } else {
                            let dx = metrics.dx();
                            let dz = metrics.dz();
                            // Snapshot dirty tiles before take clears the scheduler.
                            let dirty_ids: Vec<(u32, u32)> =
                                engine.dirty_tiles().iter().map(|t| (t.tx, t.tz)).collect();
                            self.ui_state.dirty_tile_ids = dirty_ids;
                            self.ui_state.dirty_tile_grid = (metrics.tiles_x(), metrics.tiles_z());
                            // Only skip present when evaluate failed to seed (no filter work ran).
                            let skip_present = !result.did_eval;
                            if !skip_present {
                                let region = engine.take_dirty_region(1);
                                // Full-field updates bind the engine texture directly (WC path).
                                // Partial rects still copy through the double-buffer.
                                let full_field = region.is_none_or(|r| {
                                    r.x == 0
                                        && r.y == 0
                                        && r.w == result.width
                                        && r.h == result.height
                                });
                                if full_field {
                                    renderer.present_gpu_height_shared(
                                        engine.output_texture(),
                                        engine.output_texture_view(),
                                        result.width,
                                        result.height,
                                        result.world_size,
                                        result.height_range,
                                        dx,
                                        dz,
                                        None,
                                    );
                                } else if let Some(region) = region {
                                    renderer.present_gpu_height_region(
                                        engine.output_texture(),
                                        result.width,
                                        result.height,
                                        result.world_size,
                                        result.height_range,
                                        dx,
                                        dz,
                                        Some(region),
                                    );
                                }
                                self.ui_state.profile.upload_us = renderer.last_upload_us;
                                self.ui_state.profile.terrain_grid_size =
                                    renderer.last_grid_resolution;
                                // GPU present owns the viewport — don't let a stale CPU
                                // upload from a prior job clobber it on the next redraw.
                                self.needs_height_upload = false;
                            } else {
                                let _ = engine.take_dirty_region(1);
                            }
                            self.ui_state.profile.tex_w = result.width;
                            self.ui_state.profile.tex_h = result.height;
                            self.ui_state.profile.tiles_x = metrics.tiles_x();
                            self.ui_state.profile.tiles_z = metrics.tiles_z();
                            self.ui_state.quality = quality;
                            self.ui_state.profile.quality = match quality {
                                PreviewQuality::Draft => "Draft (fast)",
                                PreviewQuality::Medium => "Medium",
                                PreviewQuality::Full => "Final (viewport)",
                                PreviewQuality::Export => "Export quality",
                            };

                            // Interactive path: GPU present is authoritative for the frame.
                            // Never sync-evaluate CPU on the UI thread — that hangs the app
                            // when stacks include unsupported layers (painted masks, SPE, …).
                            let needs_cpu_suffix = result.resume_cpu_from.is_some();
                            self.last_eval_fully_gpu = result.fully_gpu;
                            if result.resume_cpu_from == Some(0)
                                && !result.fully_gpu
                                && !result.did_eval
                            {
                                // Seed failed — keep last-good, refine async at current quality.
                                self.ui_state.profile.path = "GPU→async CPU";
                                used_gpu = true;
                                eval_completed = true;
                                if !self.worker_refine_pending {
                                    self.enqueue_async_eval(quality);
                                }
                            } else if let (Some(prefix), Some(start)) =
                                (result.cpu.clone(), result.resume_cpu_from)
                            {
                                // Full/Export hybrid only (Draft/Medium no longer read back).
                                // The GPU result carries height, not aux/published-output state,
                                // so never seed this generation from scheduler leftovers.
                                let masks = self.session.document.masks.clone();
                                let mut ctx = terra_core::eval::EvalContext::new(metrics);
                                ctx.quality = quality;
                                ctx.level_steps = self.session.document.level_steps.clone();
                                ctx.mask_assets = masks.clone();
                                ctx.masks = terra_core::mask::bake_mask_assets(
                                    &masks,
                                    &prefix,
                                    metrics,
                                    &std::collections::HashMap::new(),
                                );
                                let cpu_result = if start == 0 {
                                    self.scheduler
                                        .evaluator
                                        .rebuild_all(&preview_stack, &mut ctx)
                                } else {
                                    self.scheduler.evaluator.evaluate_suffix(
                                        &preview_stack,
                                        &mut ctx,
                                        start,
                                        prefix,
                                    )
                                };
                                match cpu_result {
                                    Ok(hf) => {
                                        ctx.sync_aux_hashmap();
                                        self.scheduler.last_strata = ctx.aux_maps.strata.clone();
                                        self.scheduler.last_layer_timings =
                                            ctx.layer_timings.clone();
                                        self.ui_state.profile.update_layer_timings(
                                            &self.scheduler.last_layer_timings,
                                        );
                                        self.scheduler.last_aux = ctx.aux;
                                        self.scheduler.last_good =
                                            Some(std::sync::Arc::new(hf.clone()));
                                        self.last_height = Some(hf);
                                        self.preview_dirty = true;
                                        self.needs_height_upload = true;
                                        self.ui_state.profile.path = "GPU+CPU";
                                        used_gpu = true;
                                        eval_completed = true;
                                    }
                                    Err(_) => {
                                        used_gpu = true;
                                        eval_completed = true;
                                        if !self.worker_refine_pending {
                                            self.enqueue_async_eval(quality);
                                        }
                                    }
                                }
                            } else {
                                self.ui_state.profile.path =
                                    if needs_cpu_suffix { "GPU*" } else { "GPU" };
                                if let Some(hf) = result.cpu {
                                    self.scheduler.last_good =
                                        Some(std::sync::Arc::new(hf.clone()));
                                    self.last_height = Some(hf);
                                    self.preview_dirty = true;
                                }
                                used_gpu = true;
                                eval_completed = true;
                                if needs_cpu_suffix {
                                    // Unsupported layers need CPU bake at *this* quality
                                    // (not Draft), then lifecycle advances Draft→Medium→Full.
                                    self.ui_state.profile.path = "GPU→async CPU";
                                    self.ui_state.refining = true;
                                    self.ui_state.build_progress =
                                        Some(quality_stage_progress(quality).max(0.15));
                                    if !self.worker_refine_pending {
                                        self.enqueue_async_eval(quality);
                                    }
                                } else {
                                    // GPU finished this quality — keep climbing toward Full.
                                    self.ui_state.refining = quality.next_refine().is_some();
                                    if !self.ui_state.refining {
                                        self.ui_state.build_progress = None;
                                    }
                                }
                            }
                            if used_gpu && !eval_completed {
                                eval_completed = true;
                            }
                        }
                    }
                    Err(_) => {
                        // GPU path failed — async CPU, keep last-good on screen.
                        self.last_eval_fully_gpu = false;
                        self.ui_state.profile.path = "async CPU";
                        if !self.worker_refine_pending {
                            self.enqueue_async_eval(quality);
                        }
                        used_gpu = true; // skip sync fallback below
                        eval_completed = true;
                    }
                }
            }
        }

        if !used_gpu {
            // Last resort when no GPU engine: still avoid blocking if a worker exists.
            if !self.worker_refine_pending {
                self.enqueue_async_eval(quality);
                eval_completed = true;
                self.ui_state.profile.path = "async CPU";
            }
        }

        let elapsed = t0.elapsed().as_micros() as u64;
        self.ui_state.profile.eval_us = elapsed;
        if eval_completed {
            self.ui_state.quality = quality;
            self.ui_state.build_progress = Some(quality_stage_progress(quality));
            self.ui_state.draft_displayed =
                matches!(quality, PreviewQuality::Draft | PreviewQuality::Medium);
            if matches!(quality, PreviewQuality::Full | PreviewQuality::Export) {
                self.ui_state.draft_displayed = false;
            }
        }
    }

    /// When a climate overlay is selected and aux is missing, bake from the
    /// current height + Biomes layer params (no full stack re-eval / GPU readback).
    pub(crate) fn ensure_climate_overlay_aux(&mut self) {
        self.ensure_climate_aux(false);
    }

    /// Bake climate aux for lit 3D (snow/temp/rain) whenever a Biomes layer is present.
    pub(crate) fn ensure_climate_for_lit(&mut self) {
        self.ensure_climate_aux(true);
    }

    pub(crate) fn ensure_climate_aux(&mut self, for_lit: bool) {
        if !for_lit {
            let key = match self.ui_state.preview_mode {
                Preview2dMode::Temperature => "temperature",
                Preview2dMode::Rainfall => "rainfall",
                Preview2dMode::Snow => "snow",
                Preview2dMode::SoilMoisture => "soil_moisture",
                Preview2dMode::Biome => "biomes",
                _ => return,
            };
            if self.scheduler.last_aux.contains_key(key) {
                return;
            }
        } else if self.scheduler.last_aux.contains_key("snow")
            && self.scheduler.last_aux.contains_key("temperature")
            && self.scheduler.last_aux.contains_key("rainfall")
        {
            return;
        }
        let Some(hf) = self.last_height.as_ref() else {
            return;
        };
        let biomes_params = self
            .session
            .document
            .stack
            .flatten_layers()
            .into_iter()
            .find_map(|layer| match &layer.kind {
                LayerKind::Biomes(p) if layer.common.enabled && p.use_climate => Some(p.clone()),
                _ => None,
            });
        let Some(params) = biomes_params else {
            return;
        };
        let wetness = self.scheduler.last_aux.get("wetness");
        let sediment = self.scheduler.last_aux.get("sediment");
        let maps = terra_core::surface::bake_biomes_climate(hf, &params, wetness, sediment);
        self.scheduler
            .last_aux
            .insert("temperature".into(), maps.temperature);
        self.scheduler
            .last_aux
            .insert("rainfall".into(), maps.rainfall);
        self.scheduler
            .last_aux
            .insert("humidity".into(), maps.humidity);
        self.scheduler
            .last_aux
            .insert("aridity".into(), maps.aridity);
        self.scheduler.last_aux.insert("snow".into(), maps.snow);
        self.scheduler
            .last_aux
            .insert("soil_moisture".into(), maps.soil_moisture);
        self.scheduler
            .last_aux
            .insert("wind_exposure".into(), maps.wind_exposure);
        self.scheduler.last_aux.insert("biomes".into(), maps.biomes);
        self.preview_dirty = true;
        if for_lit {
            // Lit path will upload on next height present; don't force re-upload loop.
        } else {
            self.needs_height_upload = true;
        }
    }

    pub(crate) fn refresh_2d_preview(&mut self) {
        if self.ui_state.preview_mode != self.last_preview_mode {
            self.last_preview_mode = self.ui_state.preview_mode;
            self.preview_dirty = true;
        }
        if self.ui_state.force_preview_refresh {
            self.ui_state.force_preview_refresh = false;
            self.preview_dirty = true;
        }
        self.ensure_climate_overlay_aux();
        if !self.preview_dirty {
            return;
        }
        let Some(hf) = self.last_height.as_ref() else {
            self.ui_state.preview_rgba = None;
            self.preview_dirty = false;
            return;
        };
        let metrics = hf.metrics;
        // Opt-in Phase 2 geomorph debug overlays (command palette only).
        if let Some(field) = self.ui_state.geomorph_debug_field {
            let mask = terra_core::bake_debug_field(hf, field, None);
            let mut rgba = Vec::with_capacity(mask.data().len() * 4);
            for &value in mask.data() {
                let value = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
                rgba.extend_from_slice(&[value, value, value, 255]);
            }
            self.ui_state.preview_rgba = Some((metrics.width, metrics.height, rgba));
            self.preview_dirty = false;
            return;
        }
        // Coloured biome placement overlay â€” bypass grayscale path.
        if self.ui_state.preview_mode == Preview2dMode::Biome {
            if let Some(layer) = self.session.document.selected_placement_layer() {
                if !layer.channels.is_empty() {
                    let doc = &self.session.document;
                    let isolate = if layer.isolate_active {
                        doc.active_biome
                    } else {
                        None
                    };
                    let rgba = layer.bake_color_rgba(
                        metrics.width,
                        metrics.height,
                        &|gid| {
                            doc.biome_library
                                .by_group(gid)
                                .map(|d| d.color)
                                .or_else(|| doc.stack.find_group(gid).map(|g| g.preview_color))
                                .unwrap_or([0.45, 0.55, 0.4])
                        },
                        isolate,
                    );
                    self.ui_state.preview_rgba = Some((metrics.width, metrics.height, rgba));
                    self.preview_dirty = false;
                    return;
                }
            }
        }
        let values = match self.ui_state.preview_mode {
            Preview2dMode::Height => {
                let (min, max) = hf.min_max();
                let span = (max - min).max(1e-6);
                hf.to_dense()
                    .into_iter()
                    .map(|value| (value - min) / span)
                    .collect()
            }
            Preview2dMode::Slope => terra_core::analyze::slope_degrees(hf).data().to_vec(),
            Preview2dMode::Flow => {
                let Some(flow) = self.scheduler.last_aux.get("flow_accumulation") else {
                    self.ui_state.preview_rgba = None;
                    return;
                };
                let max = flow
                    .data()
                    .iter()
                    .copied()
                    .into_iter()
                    .fold(1.0f32, f32::max)
                    .ln_1p();
                flow.data()
                    .iter()
                    .copied()
                    .map(|value| value.max(0.0).ln_1p() / max)
                    .collect()
            }
            Preview2dMode::Mask | Preview2dMode::Masks => {
                let baked = bake_mask_assets(
                    &self.session.document.masks,
                    hf,
                    metrics,
                    &self.scheduler.last_aux,
                );
                self.ui_state
                    .selected_mask
                    .and_then(|id| baked.get(&id))
                    .or_else(|| baked.values().next())
                    .map(|mask| mask.data().to_vec())
                    .or_else(|| {
                        self.scheduler
                            .last_aux
                            .get("materials")
                            .map(|mask| mask.data().to_vec())
                    })
                    .unwrap_or_else(|| vec![0.0; (metrics.width * metrics.height) as usize])
            }
            Preview2dMode::Material | Preview2dMode::Biome | Preview2dMode::VegetationDensity => {
                let key = match self.ui_state.preview_mode {
                    Preview2dMode::Material => "materials",
                    Preview2dMode::Biome => "biomes",
                    Preview2dMode::VegetationDensity => "vegetation",
                    _ => unreachable!(),
                };
                self.scheduler
                    .last_aux
                    .get(key)
                    .map(|field| field.data().to_vec())
                    .unwrap_or_else(|| vec![0.0; (metrics.width * metrics.height) as usize])
            }
            Preview2dMode::Water
            | Preview2dMode::Sediment
            | Preview2dMode::Hardness
            | Preview2dMode::Erosion
            | Preview2dMode::Deposition
            | Preview2dMode::StreamOrder
            | Preview2dMode::SpeIncision
            | Preview2dMode::Temperature
            | Preview2dMode::Rainfall
            | Preview2dMode::Snow
            | Preview2dMode::SoilMoisture
            | Preview2dMode::Overhang => {
                let key = match self.ui_state.preview_mode {
                    Preview2dMode::Water => "wetness",
                    Preview2dMode::Sediment => "sediment",
                    Preview2dMode::Hardness => "hardness",
                    Preview2dMode::Erosion => "erosion",
                    Preview2dMode::Deposition => "deposition",
                    Preview2dMode::StreamOrder => "stream_order",
                    Preview2dMode::SpeIncision => "spe_incision",
                    Preview2dMode::Temperature => "temperature",
                    Preview2dMode::Rainfall => "rainfall",
                    Preview2dMode::Snow => "snow",
                    Preview2dMode::SoilMoisture => "soil_moisture",
                    Preview2dMode::Overhang => "overhang_mask",
                    _ => unreachable!(),
                };
                self.scheduler
                    .last_aux
                    .get(key)
                    .map(|field| {
                        let data = field.data();
                        let max_v = data.iter().copied().fold(1e-6f32, f32::max);
                        data.iter().map(|&v| (v / max_v).clamp(0.0, 1.0)).collect()
                    })
                    .unwrap_or_else(|| vec![0.0; (metrics.width * metrics.height) as usize])
            }
            // TODO: render colorized 2D previews for these diagnostics. The 3D viewport
            // continues to own their shader presentation; retain a height preview here.
            Preview2dMode::Lit
            | Preview2dMode::Unlit
            | Preview2dMode::Curvature
            | Preview2dMode::Convexity
            | Preview2dMode::Concavity
            | Preview2dMode::Normals
            | Preview2dMode::Wireframe
            | Preview2dMode::AmbientOcclusion => {
                let (min, max) = hf.min_max();
                let span = (max - min).max(1e-6);
                hf.to_dense()
                    .into_iter()
                    .map(|value| (value - min) / span)
                    .collect()
            }
        };
        let mut rgba = Vec::with_capacity(values.len() * 4);
        for value in values {
            let value = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
            rgba.extend_from_slice(&[value, value, value, 255]);
        }
        self.ui_state.preview_rgba = Some((metrics.width, metrics.height, rgba));
        self.preview_dirty = false;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
    use terra_core::layer::{Layer, LayerKind, LayerStack, StreamPowerParams};

    use super::TerraApp;

    /// Revert check for #33: restoring a fresh-layer CPU shortcut in
    /// `request_rebuild_immediate` will populate the cache and replace last-good before
    /// this test reaches the worker assertions.
    #[test]
    fn add_layer_keeps_last_good_until_async_cpu_evaluation_publishes() {
        let mut app = TerraApp::default();
        app.session.document.stack = LayerStack::new();

        let layer = Layer::new(
            "Unsupported SPE",
            LayerKind::StreamPowerErosion(StreamPowerParams::default()),
        );
        let layer_id = layer.id();
        app.session.document.stack.push(layer);
        app.session.document.selected = Some(layer_id);

        let height = Heightfield::filled(HeightfieldMetrics::new(32, 32, 320.0, 320.0), 7.0);
        let last_good = Arc::new(height.clone());
        app.last_height = Some(height);
        app.scheduler.last_good = Some(Arc::clone(&last_good));

        app.request_rebuild_immediate();

        assert!(
            app.scheduler.evaluator.cache.get(layer_id).is_none(),
            "the add-layer event path must not evaluate or cache CPU layer output"
        );
        assert!(
            Arc::ptr_eq(
                app.scheduler
                    .last_good
                    .as_ref()
                    .expect("seeded last-good must remain available"),
                &last_good,
            ),
            "last-good viewport content must remain authoritative while CPU work is pending"
        );
        assert!(
            app.pending_eval,
            "the add-layer edit must queue Draft evaluation"
        );
        assert!(
            app.force_draft,
            "the queued evaluation must start at Draft quality"
        );

        // This is the same non-blocking step the lifecycle runs after the edit debounce.
        // With no GPU engine, it must submit the existing worker rather than evaluate here.
        app.run_eval_step();

        assert!(
            app.worker_refine_pending,
            "CPU fallback must be owned by EvalWorker"
        );
        assert_eq!(app.ui_state.profile.path, "async CPU");
        assert!(
            app.scheduler.evaluator.cache.get(layer_id).is_none(),
            "submitting the worker must not synchronously mutate the UI-thread cache"
        );
        assert!(
            Arc::ptr_eq(
                app.scheduler
                    .last_good
                    .as_ref()
                    .expect("last-good must remain while the worker runs"),
                &last_good,
            ),
            "the viewport must retain last-good until the worker result is consumed"
        );
    }
}
