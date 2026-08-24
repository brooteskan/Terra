use crate::ui::PanelAction;
use terra_core::mask::bake_mask_assets;
use terra_gui::GuiContext;
use terra_render::pick_terrain_uv_on_surface;

use super::{AppScreen, TerraApp};
impl TerraApp {
    pub(crate) fn cursor_logical(&self) -> Option<(f32, f32)> {
        let (cursor, window) = (self.last_cursor?, self.window.as_ref()?);
        let ppp = window.scale_factor() as f32;
        Some((cursor.0 as f32 / ppp, cursor.1 as f32 / ppp))
    }

    pub(crate) fn cursor_in_viewport(&self) -> bool {
        let Some((x, y)) = self.cursor_logical() else {
            return false;
        };
        self.viewport_rect.contains(x, y)
    }

    /// Camera owns the pointer over the 3D viewport (and not over terra-gui).
    /// Move tool (and non-brush tools) navigate; Alt temporarily restores camera while brushing.
    pub(crate) fn viewport_camera_active(&self) -> bool {
        if self.screen != AppScreen::Editor {
            return false;
        }
        if self.gui_wants_pointer || !self.cursor_in_viewport() {
            return false;
        }
        self.viewport_camera_fly_active()
    }

    /// WASD/QE fly — same tool rules as mouse camera, but does not require the cursor
    /// to sit inside the viewport (game-engine style continuous move).
    pub(crate) fn viewport_camera_fly_active(&self) -> bool {
        if self.screen != AppScreen::Editor {
            return false;
        }
        if self.ui_state.editor_tool.is_place_point() && !self.modifiers_alt {
            return false;
        }
        self.modifiers_alt || !self.viewport_paint_active()
    }

    /// Foundation is an explicit brush target; all other selections may begin a
    /// new Semantic Sculpt session. Keep this runtime check in addition to the
    /// disabled tool card so a brush armed before selecting Foundation cannot
    /// redirect a stroke behind the artist's back.
    fn active_sculpt_brush_available_for_selection(&self) -> bool {
        use terra_core::layer::{BrushEditable, EditSupport};

        let Some(brush) = self.ui_state.editor_tool.sculpt_stroke_kind() else {
            return true;
        };
        self.session
            .document
            .selected
            .and_then(|id| self.session.document.stack.find(id))
            .filter(|layer| layer.kind.is_sculpt_base())
            .is_none_or(|foundation| foundation.brush_support(brush) != EditSupport::Unsupported)
    }

    /// True when left-drag should stamp Base heights or a mask.
    pub(crate) fn viewport_paint_active(&self) -> bool {
        if self.screen != AppScreen::Editor {
            return false;
        }
        if self.ui_state.editor_tool.is_sculpt() {
            return self.active_sculpt_brush_available_for_selection();
        }
        if self.ui_state.editor_tool == crate::ui::EditorTool::PaintBiome {
            return self.session.document.active_biome.is_some();
        }
        self.ui_state.editor_tool == crate::ui::EditorTool::PaintMask
            && self.ui_state.paint_mask.is_some()
    }

    pub(crate) fn refresh_viewport_rect(&mut self) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        let ppp = window.scale_factor() as f32;
        let size = window.inner_size();
        let screen_w = size.width as f32 / ppp;
        let screen_h = size.height as f32 / ppp;
        self.viewport_rect =
            GuiContext::viewport_rect_for(screen_w, screen_h, &self.ui_state.layout);
    }

    fn authoring_stamp(
        &self,
        point: terra_core::AuthoringPoint,
        bounded_radius_uv: f32,
    ) -> terra_core::AuthoringBrushStamp {
        terra_core::AuthoringBrushStamp::from_point(
            point,
            bounded_radius_uv,
            f64::from(self.ui_state.infinite_brush_radius_m),
        )
        .expect("UI brush radii and picker points are validated")
    }

    fn interpolated_brush_points(
        &self,
        point: terra_core::AuthoringPoint,
        bounded_radius_uv: f32,
        spacing_fraction: f32,
    ) -> Vec<terra_core::AuthoringPoint> {
        let Some(previous) = self.last_paint_point else {
            return Vec::new();
        };
        let Some(distance) = previous.horizontal_distance(point) else {
            return Vec::new();
        };
        let radius = match point {
            terra_core::AuthoringPoint::Bounded { .. } => f64::from(bounded_radius_uv),
            terra_core::AuthoringPoint::Infinite { .. } => {
                f64::from(self.ui_state.infinite_brush_radius_m)
            }
        };
        let minimum = match point {
            terra_core::AuthoringPoint::Bounded { .. } => 0.002,
            terra_core::AuthoringPoint::Infinite { .. } => self
                .session
                .document
                .infinite_settings()
                .map_or(0.01, |settings| settings.finest_spacing_m.max(0.01)),
        };
        let spacing = (radius * f64::from(spacing_fraction)).max(minimum);
        if distance <= spacing {
            return Vec::new();
        }
        let steps = ((distance / spacing).ceil() as u32).clamp(1, 32);
        (1..steps)
            .filter_map(|index| previous.lerp(point, f64::from(index) / f64::from(steps)))
            .collect()
    }

    pub(crate) fn paint_at_cursor(&mut self) {
        if self.ui_state.editor_tool.is_sculpt()
            && !self.active_sculpt_brush_available_for_selection()
        {
            self.ui_state.status =
                "That brush isn't supported by the selected Foundation layer.".into();
            return;
        }
        let Some(hit) = self.pick_terrain_surface() else {
            return;
        };
        let point = hit.authoring_point(self.session.document.infinite_settings().is_some());
        let bounded_uv = point.bounded_uv().map(|uv| uv.tuple());

        if let Some(shape_tool) = self.ui_state.editor_tool.shape_tool() {
            let Some(layer_id) = self.ensure_shape_history_target(shape_tool) else {
                return;
            };
            self.ui_state.ensure_sculpt_defaults();
            self.sculpt_stroke_active = true;
            let strength = match shape_tool {
                terra_core::shape_history::ShapeTool::Smooth
                | terra_core::shape_history::ShapeTool::Pinch => {
                    (self.ui_state.sculpt_strength / 10.0).clamp(0.05, 1.0)
                }
                terra_core::shape_history::ShapeTool::MountainStamp
                | terra_core::shape_history::ShapeTool::CraterStamp => {
                    self.ui_state.sculpt_strength.max(8.0)
                }
                _ => self.ui_state.sculpt_strength,
            };
            let radius = self.ui_state.sculpt_radius;
            let target_height = match shape_tool {
                // HeightStamp / PlateauStamp stamp toward the height under the
                // cursor. Flatten needs no target here — the sculpt kernel derives
                // it from the mean of the terrain within the brush footprint.
                terra_core::shape_history::ShapeTool::HeightStamp
                | terra_core::shape_history::ShapeTool::PlateauStamp => point.height_m(),
                _ => 0.0,
            };
            let stroke_kind = shape_tool.stroke_kind();
            let mut actions = Vec::new();
            let stamp_once = shape_tool.is_stamp() && self.last_paint_point.is_some();
            if stamp_once {
                // Stamps are one-shot per click (drag doesn't spam new mountains).
                return;
            }
            if !shape_tool.is_stamp() {
                for intermediate in self.interpolated_brush_points(point, radius, 0.35) {
                    actions.push(PanelAction::PaintSculptStamp {
                        layer: layer_id,
                        stamp: self.authoring_stamp(intermediate, radius),
                        strength,
                        stroke_kind,
                        target_height,
                    });
                }
            }
            actions.push(PanelAction::PaintSculptStamp {
                layer: layer_id,
                stamp: self.authoring_stamp(point, radius),
                strength,
                stroke_kind,
                target_height,
            });
            self.apply_actions(actions);
            self.last_paint_point = Some(point);
            // Draft preview while painting â€” do not force simulation rebuilds.
            self.force_draft = true;
            return;
        }

        // Field brushes (Protect / Hardness / Sediment) â†’ constraints authoring layer.
        if matches!(
            self.ui_state.editor_tool,
            crate::ui::EditorTool::Protect
                | crate::ui::EditorTool::Hardness
                | crate::ui::EditorTool::Sediment
        ) {
            use terra_core::authoring::SculptStrokeKind;
            let stroke_kind = match self.ui_state.editor_tool {
                crate::ui::EditorTool::Protect => SculptStrokeKind::Protect,
                crate::ui::EditorTool::Hardness => SculptStrokeKind::Hardness,
                _ => SculptStrokeKind::Sediment,
            };
            let Some(layer_id) = self.ensure_shape_authoring_layer() else {
                return;
            };
            self.ui_state.ensure_sculpt_defaults();
            self.sculpt_stroke_active = true;
            let strength = self.ui_state.sculpt_strength.clamp(0.05, 1.0);
            let radius = self.ui_state.sculpt_radius;
            let actions = vec![PanelAction::PaintSculptStamp {
                layer: layer_id,
                stamp: self.authoring_stamp(point, radius),
                strength,
                stroke_kind,
                target_height: 0.0,
            }];
            self.apply_actions(actions);
            self.last_paint_point = Some(point);
            self.force_draft = true;
            return;
        }

        if self.ui_state.editor_tool == crate::ui::EditorTool::PaintBiome {
            let Some(biome) = self.session.document.active_biome else {
                self.ui_state.status = "Select a biome (or create one) before painting.".into();
                return;
            };
            self.session.document.ensure_placement_layer();
            // Ensure active biome has a channel / library link.
            if let Some(def) = self.session.document.biome_library.by_group(biome) {
                let _ = def;
            }
            let tool = self.ui_state.biome_paint_tool;
            let radius = self.ui_state.sculpt_radius.max(0.02);
            let strength = self.ui_state.sculpt_strength.clamp(0.05, 1.0);
            let erase =
                tool == terra_core::biome_paint::BiomePaintTool::Erase || self.modifiers_alt;
            let mut actions = Vec::new();

            if tool == terra_core::biome_paint::BiomePaintTool::Normalize {
                actions.push(PanelAction::NormalizeBiomePlacement);
                self.apply_actions(actions);
                return;
            }

            if tool == terra_core::biome_paint::BiomePaintTool::FloodFill {
                if self.last_paint_point.is_some() {
                    return; // one-shot per click
                }
                actions.push(PanelAction::BeginBiomePaintStroke { biome });
                actions.push(PanelAction::PaintBiomeStamp {
                    biome,
                    stamp: self.authoring_stamp(point, radius),
                    strength,
                    erase,
                    mode: Some(tool),
                });
                actions.push(PanelAction::EndBiomePaintStroke);
                self.apply_actions(actions);
                self.last_paint_point = Some(point);
                self.placement_tint_dirty = true;
                return;
            }

            if tool == terra_core::biome_paint::BiomePaintTool::PolygonFill {
                let Some((u, v)) = bounded_uv else {
                    self.ui_state.status =
                        "Biome polygon fill is unavailable for Infinite projects.".into();
                    return;
                };
                // Close polygon when clicking near the first vertex.
                if let Some(&(fu, fv)) = self.biome_polygon_points.first() {
                    if (fu - u).hypot(fv - v) < 0.025 && self.biome_polygon_points.len() >= 3 {
                        let pts = std::mem::take(&mut self.biome_polygon_points);
                        actions.push(PanelAction::BeginBiomePaintStroke { biome });
                        let Some(res) = self
                            .session
                            .document
                            .bounded_settings()
                            .map(|settings| settings.preview_resolution.clamp(64, 8192))
                        else {
                            self.ui_state.status =
                                "Biome polygon fill is unavailable for Infinite projects.".into();
                            return;
                        };
                        if let Some(layer) = self.session.document.selected_placement_layer_mut() {
                            layer.fill_polygon(
                                biome,
                                &pts,
                                if erase { 0.0 } else { strength },
                                res,
                            );
                        }
                        self.sync_biome_paint_to_mask(biome);
                        actions.push(PanelAction::EndBiomePaintStroke);
                        self.apply_actions(actions);
                        self.placement_tint_dirty = true;
                        self.preview_dirty = true;
                        self.ui_state.status = format!("Polygon fill ({} pts)", pts.len());
                        return;
                    }
                }
                self.biome_polygon_points.push((u, v));
                self.ui_state.status = format!(
                    "Polygon vertex {} â€” click near first point to close",
                    self.biome_polygon_points.len()
                );
                self.last_paint_point = Some(point);
                return;
            }

            // Capture undo snapshot at the start of a drag.
            if self.last_paint_point.is_none() {
                actions.push(PanelAction::BeginBiomePaintStroke { biome });
            }

            if tool.paints_mask() {
                for intermediate in self.interpolated_brush_points(point, radius, 0.35) {
                    actions.push(PanelAction::PaintBiomeStamp {
                        biome,
                        stamp: self.authoring_stamp(intermediate, radius),
                        strength,
                        erase,
                        mode: Some(tool),
                    });
                }
                actions.push(PanelAction::PaintBiomeStamp {
                    biome,
                    stamp: self.authoring_stamp(point, radius),
                    strength,
                    erase,
                    mode: Some(tool),
                });
            }

            if tool.sculpts() {
                // Raise / Lower / Flatten also stamp Shape history under the same brush.
                use terra_core::authoring::SculptStrokeKind;
                let stroke_kind = match tool {
                    terra_core::biome_paint::BiomePaintTool::Raise
                    | terra_core::biome_paint::BiomePaintTool::RaisePaint => {
                        SculptStrokeKind::Raise
                    }
                    terra_core::biome_paint::BiomePaintTool::Lower
                    | terra_core::biome_paint::BiomePaintTool::LowerPaint => {
                        SculptStrokeKind::Lower
                    }
                    terra_core::biome_paint::BiomePaintTool::Flatten
                    | terra_core::biome_paint::BiomePaintTool::FlattenPaint => {
                        SculptStrokeKind::Flatten
                    }
                    _ => SculptStrokeKind::Raise,
                };
                let shape_tool = match stroke_kind {
                    SculptStrokeKind::Lower => terra_core::shape_history::ShapeTool::Lower,
                    SculptStrokeKind::Flatten => terra_core::shape_history::ShapeTool::Flatten,
                    _ => terra_core::shape_history::ShapeTool::Raise,
                };
                let sculpt_layer_id = self.ensure_shape_history_target(shape_tool);
                if let Some(layer_id) = sculpt_layer_id {
                    self.ui_state.ensure_sculpt_defaults();
                    let sculpt_strength = if matches!(stroke_kind, SculptStrokeKind::Smooth) {
                        (self.ui_state.sculpt_strength / 10.0).clamp(0.05, 1.0)
                    } else {
                        self.ui_state.sculpt_strength
                    };
                    let target_height = point.height_m();
                    for intermediate in self.interpolated_brush_points(point, radius, 0.35) {
                        actions.push(PanelAction::PaintSculptStamp {
                            layer: layer_id,
                            stamp: self.authoring_stamp(intermediate, radius),
                            strength: sculpt_strength,
                            stroke_kind,
                            target_height,
                        });
                    }
                    actions.push(PanelAction::PaintSculptStamp {
                        layer: layer_id,
                        stamp: self.authoring_stamp(point, radius),
                        strength: sculpt_strength,
                        stroke_kind,
                        target_height,
                    });
                }
            }

            self.sculpt_stroke_active = true;
            self.apply_actions(actions);
            self.last_paint_point = Some(point);
            return;
        }

        let Some(mask_id) = self.ui_state.paint_mask else {
            return;
        };
        let radius = self.ui_state.sculpt_radius.max(0.02);
        let hardness = self.ui_state.brush_falloff;
        let mut tool = self.ui_state.mask_paint_tool;
        if self.modifiers_shift || self.ui_state.invert_brush {
            tool = match tool {
                terra_core::mask::MaskPaintTool::Paint => terra_core::mask::MaskPaintTool::Erase,
                terra_core::mask::MaskPaintTool::Erase => terra_core::mask::MaskPaintTool::Paint,
                terra_core::mask::MaskPaintTool::Smooth => terra_core::mask::MaskPaintTool::Smooth,
                terra_core::mask::MaskPaintTool::FloodFill => {
                    terra_core::mask::MaskPaintTool::FloodFill
                }
            };
        }
        let strength = (0.18 * self.ui_state.brush_flow).clamp(0.002, 0.18);
        let mut actions = Vec::new();
        for intermediate in self.interpolated_brush_points(
            point,
            radius,
            self.ui_state.brush_spacing.clamp(0.05, 1.0),
        ) {
            actions.push(PanelAction::PaintMaskStamp {
                mask_id,
                stamp: self.authoring_stamp(intermediate, radius),
                strength,
                hardness,
                tool,
            });
        }
        actions.push(PanelAction::PaintMaskStamp {
            mask_id,
            stamp: self.authoring_stamp(point, radius),
            strength,
            hardness,
            tool,
        });
        self.apply_actions(actions);
        self.last_paint_point = Some(point);
    }

    /// Push Draft heights to the GPU while a brush stroke is active.
    /// Call at most once per frame â€” stamps coalesce via `pending_eval`.
    pub(crate) fn commit_biome_polygon_fill(&mut self) {
        let Some(biome) = self.session.document.active_biome else {
            return;
        };
        if self.biome_polygon_points.len() < 3 {
            return;
        }
        let pts = std::mem::take(&mut self.biome_polygon_points);
        let strength = self.ui_state.sculpt_strength.clamp(0.05, 1.0);
        let erase = self.modifiers_alt;
        let Some(res) = self
            .session
            .document
            .bounded_settings()
            .map(|settings| settings.preview_resolution.clamp(64, 8192))
        else {
            self.ui_state.status =
                "Biome polygon fill is unavailable for Infinite projects.".into();
            return;
        };
        self.session.document.ensure_placement_layer();
        self.apply_actions(vec![PanelAction::BeginBiomePaintStroke { biome }]);
        if let Some(layer) = self.session.document.selected_placement_layer_mut() {
            layer.fill_polygon(biome, &pts, if erase { 0.0 } else { strength }, res);
        }
        self.sync_biome_paint_to_mask(biome);
        self.apply_actions(vec![PanelAction::EndBiomePaintStroke]);
        self.placement_tint_dirty = true;
        self.preview_dirty = true;
        self.ui_state.status = format!("Polygon fill ({} pts)", pts.len());
    }

    /// Raycast the cursor onto the visible terrain in an explicit project frame.
    pub(crate) fn pick_terrain_surface(&mut self) -> Option<terra_core::TerrainSurfaceHit> {
        let (x, y) = self.cursor_logical()?;
        if !self.viewport_rect.contains(x, y) {
            return None;
        }
        // Once a stroke has started, keep stamping even if the cursor grazes overlays
        // (brush bar / gizmo chrome) â€” otherwise live preview stalls mid-drag.
        if self.gui_wants_pointer && !self.sculpt_stroke_active {
            return None;
        }
        let window = self.window.as_ref()?;
        let ppp = window.scale_factor() as f32;
        let renderer = self.renderer.as_ref()?;
        let gpu = self.gpu.as_ref()?;
        let editor_overlays = self.editor_overlays.as_mut()?;
        let (surface_w, surface_h) = renderer.size();
        let screen_w = surface_w as f32 / ppp;
        let screen_h = surface_h as f32 / ppp;
        let aspect = surface_w as f32 / surface_h.max(1) as f32;
        editor_overlays.poll_brush(&gpu.device);
        if let Some(pick) =
            editor_overlays.latest_surface_pick(renderer, (x, y), (screen_w, screen_h))
        {
            return Some(pick.hit);
        }
        if renderer.traversal_mode() == terra_render::TerrainTraversalMode::Infinite {
            // Infinite has no monolithic CPU heightfield whose UV could be used
            // as a truthful fallback. Wait for the current GPU depth result.
            return None;
        }
        let uv = pick_terrain_uv_on_surface(
            &renderer.camera,
            aspect,
            (x, y),
            (screen_w, screen_h),
            renderer.heights.world_size,
            self.last_height.as_ref(),
        )?;
        let bounded_uv = terra_core::BoundedUv::try_new(uv.0, uv.1).ok()?;
        let height_m = self.last_height.as_ref().map_or(0.0, |heightfield| {
            let x = ((uv.0 * (heightfield.metrics.width.saturating_sub(1)) as f32).round() as u32)
                .min(heightfield.metrics.width.saturating_sub(1));
            let z = ((uv.1 * (heightfield.metrics.height.saturating_sub(1)) as f32).round() as u32)
                .min(heightfield.metrics.height.saturating_sub(1));
            heightfield.get(x, z)
        });
        let world = terra_core::WorldPosition::try_new(
            f64::from(uv.0 * renderer.heights.world_size.0),
            f64::from(uv.1 * renderer.heights.world_size.1),
        )
        .ok()?;
        terra_core::TerrainSurfaceHit::try_new(world, height_m, Some(bounded_uv)).ok()
    }

    /// Legacy bounded helper retained for bounded-only shape editing paths.
    pub(crate) fn pick_paint_uv(&mut self) -> Option<(f32, f32)> {
        self.pick_terrain_surface()?.bounded_uv.map(|uv| uv.tuple())
    }

    pub(crate) fn brush_gizmo_color(&self) -> [f32; 4] {
        use crate::ui::EditorTool::*;
        match self.ui_state.editor_tool {
            Raise => [0.25, 0.75, 1.0, 0.95],
            Lower => [1.0, 0.45, 0.2, 0.95],
            Smooth => [1.0, 0.9, 0.35, 0.95],
            PaintMask => [0.95, 0.95, 1.0, 0.9],
            Ridge => [0.8, 0.55, 1.0, 0.95],
            Valley => [0.25, 0.85, 0.85, 0.95],
            Roughness => [0.8, 0.8, 0.35, 0.95],
            UpliftBrush => [0.9, 0.4, 0.75, 0.95],
            Protect => [0.35, 1.0, 0.45, 0.95],
            Hardness => [0.75, 0.75, 0.8, 0.95],
            Sediment => [0.75, 0.55, 0.3, 0.95],
            RiverConstraint => [0.15, 0.55, 1.0, 0.95],
            PaintBiome => [0.45, 0.95, 0.55, 0.95],
            EditPath => [0.35, 0.85, 1.0, 0.95],
            EditPolygon => [0.95, 0.65, 0.2, 0.95],
            EditRiverSpring => [0.25, 0.65, 1.0, 0.95],
            _ => [0.5, 0.8, 1.0, 0.8],
        }
    }

    /// Click empty terrain to add a node. Existing Path/Polygon nodes can be
    /// dragged, Shift-dragged vertically, or Ctrl-clicked to delete.
    pub(crate) fn update_brush_gizmo(&mut self) {
        let can_pick = self.cursor_in_viewport() && !self.gui_wants_pointer && !self.modifiers_alt;
        if !can_pick {
            if let (Some(editor_overlays), Some(gpu)) =
                (self.editor_overlays.as_mut(), self.gpu.as_ref())
            {
                editor_overlays.hide_brush(&gpu.queue);
            }
            return;
        }
        // Keep an authoritative surface hit warm even for Move/context-menu
        // interactions. Infinite projects deliberately have no imprecise CPU
        // fallback, so right-click placement consumes this asynchronous result.
        let show = self.viewport_paint_tool_armed();
        if self.renderer.is_some() {
            self.request_presentation_pipeline(
                terra_render::PresentationPipelineFeature::Brush,
                false,
            );
        }
        self.ui_state.ensure_sculpt_defaults();
        let infinite = self.renderer.as_ref().is_some_and(|renderer| {
            renderer.traversal_mode() == terra_render::TerrainTraversalMode::Infinite
        });
        let radius = if !show {
            self.session
                .document
                .infinite_settings()
                .map_or(0.012, |settings| settings.finest_spacing_m as f32 * 3.0)
        } else if infinite {
            if self.ui_state.editor_tool.is_place_point() {
                self.session
                    .document
                    .infinite_settings()
                    .map_or(1.0, |settings| settings.finest_spacing_m as f32 * 3.0)
            } else {
                self.ui_state.infinite_brush_radius_m
            }
        } else if self.ui_state.editor_tool.is_place_point() {
            0.012
        } else if self.ui_state.editor_tool.is_sculpt() {
            self.ui_state.sculpt_radius
        } else {
            self.ui_state.sculpt_radius.max(0.02)
        };
        let Some((x, y)) = self.cursor_logical() else {
            return;
        };
        let Some(window) = self.window.as_ref() else {
            return;
        };
        let ppp = window.scale_factor() as f32;
        let color = self.brush_gizmo_color();
        if let (Some(renderer), Some(editor_overlays), Some(gpu)) = (
            self.renderer.as_ref(),
            self.editor_overlays.as_mut(),
            self.gpu.as_ref(),
        ) {
            let (surface_w, surface_h) = renderer.size();
            editor_overlays.request_surface_pick(
                gpu,
                renderer,
                (x, y),
                (surface_w as f32 / ppp, surface_h as f32 / ppp),
                radius,
                color,
                show,
            );
        }
    }

    /// True when a paint/sculpt tool is selected (gizmo should track the cursor).
    pub(crate) fn viewport_paint_tool_armed(&self) -> bool {
        self.ui_state.editor_tool.is_sculpt()
            || self.ui_state.editor_tool.is_place_point()
            || (self.ui_state.editor_tool == crate::ui::EditorTool::PaintMask
                && self.ui_state.paint_mask.is_some())
            || self.ui_state.editor_tool == crate::ui::EditorTool::PaintBiome
    }

    /// Execute keyboard bindings through the shared command IDs.
    pub(crate) fn commit_mask_paint_stroke(&mut self) {
        let Some((mask_id, before, w, h)) = self.mask_paint_stroke_before.take() else {
            return;
        };
        self.session
            .push_mask_paint_undo(terra_core::document::MaskPaintStrokeUndo {
                label: "Painted Mask".into(),
                mask_id,
                before_samples: before,
                width: w,
                height: h,
            });
        // Project/layer masks: one rebuild after the stroke (not per dab).
        self.mark_all_layers_dirty();
        self.request_rebuild();
        self.preview_dirty = true;
        self.mask_overlay_dirty = true;
        self.mark_document_dirty();
    }

    /// True when the active mask should tint the 3D terrain.
    pub(crate) fn should_show_mask_overlay(&self) -> bool {
        let has_target =
            self.ui_state.paint_mask.is_some() || self.ui_state.selected_mask.is_some();
        if !has_target {
            return false;
        }
        // Paint tool, or Mask view (coloured overlay + colour UI stay in sync).
        self.ui_state.editor_tool == crate::ui::EditorTool::PaintMask
            || self.ui_state.is_mask_view()
    }

    /// Upload the active painted mask as a coloured translucent overlay on the terrain.
    pub(crate) fn sync_mask_overlay_to_renderer(&mut self) {
        let Some(r) = self.renderer.as_mut() else {
            return;
        };
        let mask_id = self.ui_state.paint_mask.or(self.ui_state.selected_mask);
        let Some(mask_id) = mask_id else {
            r.upload_placement_tint(1, 1, &[0, 0, 0, 0]);
            r.set_biome_tint_strength(0.0);
            self.mask_overlay_dirty = false;
            return;
        };
        let Some(asset) = self.session.document.masks.iter().find(|a| a.id == mask_id) else {
            r.upload_placement_tint(1, 1, &[0, 0, 0, 0]);
            r.set_biome_tint_strength(0.0);
            self.mask_overlay_dirty = false;
            return;
        };
        let color = asset.display_color;
        if let Some(paint) = asset.paint.as_ref() {
            if paint.width > 0 && paint.height > 0 && !paint.samples.is_empty() {
                let rgba = paint.bake_overlay_rgba(color);
                r.upload_placement_tint(paint.width, paint.height, &rgba);
                r.set_biome_tint_strength(0.68);
                self.mask_overlay_dirty = false;
                return;
            }
        }
        // Procedural / baked masks: sample evaluated field when heights are available.
        if let Some(hf) = self.last_height.as_ref() {
            let baked = bake_mask_assets(
                &self.session.document.masks,
                hf,
                hf.metrics,
                &self.scheduler.last_aux,
            );
            if let Some(field) = baked.get(&mask_id) {
                let w = field.metrics.width;
                let h = field.metrics.height;
                let data = field.data();
                let cr = (color[0].clamp(0.0, 1.0) * 255.0).round() as u8;
                let cg = (color[1].clamp(0.0, 1.0) * 255.0).round() as u8;
                let cb = (color[2].clamp(0.0, 1.0) * 255.0).round() as u8;
                let mut rgba = Vec::with_capacity(data.len() * 4);
                for &v in data {
                    let a = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                    rgba.extend_from_slice(&[cr, cg, cb, a]);
                }
                r.upload_placement_tint(w, h, &rgba);
                r.set_biome_tint_strength(0.68);
                self.mask_overlay_dirty = false;
                return;
            }
        }
        r.upload_placement_tint(1, 1, &[0, 0, 0, 0]);
        r.set_biome_tint_strength(0.0);
        self.mask_overlay_dirty = false;
    }
}

#[cfg(test)]
mod tests {
    use super::TerraApp;
    use crate::ui::EditorTool;
    use terra_core::layer::{FlatParams, Layer, LayerKind, LayerStack, SculptParams};

    fn app_with_selected_layer(kind: LayerKind) -> TerraApp {
        let mut app = TerraApp::default();
        let layer = Layer::new("Selected", kind);
        let id = layer.id();
        app.session.document.stack = LayerStack::new();
        app.session.document.stack.push(layer);
        app.session.document.selected = Some(id);
        app
    }

    #[test]
    fn runtime_gate_applies_only_to_selected_foundation() {
        let mut foundation =
            app_with_selected_layer(LayerKind::SculptBase(SculptParams::filled(8, 0.0)));
        foundation.ui_state.editor_tool = EditorTool::Raise;
        assert!(foundation.active_sculpt_brush_available_for_selection());
        foundation.ui_state.editor_tool = EditorTool::Terrace;
        assert!(!foundation.active_sculpt_brush_available_for_selection());

        let mut flat = app_with_selected_layer(LayerKind::Flat(FlatParams::default()));
        flat.ui_state.editor_tool = EditorTool::Terrace;
        assert!(flat.active_sculpt_brush_available_for_selection());
    }
}
