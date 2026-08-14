use crate::ui::PanelAction;
use terra_core::mask::bake_mask_assets;
use terra_gui::GuiContext;
use terra_render::{pick_terrain_uv_on_surface, BrushGizmo};

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

    /// True when left-drag should stamp Base heights or a mask.
    pub(crate) fn viewport_paint_active(&self) -> bool {
        if self.screen != AppScreen::Editor {
            return false;
        }
        if self.ui_state.editor_tool.is_sculpt() {
            return self
                .session
                .document
                .stack
                .flatten_layers()
                .iter()
                .any(|l| l.kind.is_sculpt_base())
                || self.session.document.selected.is_some_and(|id| {
                    matches!(
                        self.session.document.stack.find(id).map(|l| &l.kind),
                        Some(
                            terra_core::layer::LayerKind::SculptStrokes(_)
                                | terra_core::layer::LayerKind::TerrainConstraints(_)
                        )
                    )
                });
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

    pub(crate) fn paint_at_cursor(&mut self) {
        let Some((u, v)) = self.pick_paint_uv() else {
            return;
        };

        if let Some(shape_tool) = self.ui_state.editor_tool.shape_tool() {
            let Some(layer_id) = self.ensure_shape_history_target(shape_tool) else {
                self.ui_state.status =
                    "Could not create a Shape Layer â€” select a Region first.".into();
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
                terra_core::shape_history::ShapeTool::Flatten
                | terra_core::shape_history::ShapeTool::HeightStamp
                | terra_core::shape_history::ShapeTool::PlateauStamp => {
                    // Sample approx from last height field centre if available.
                    self.last_height
                        .as_ref()
                        .map(|h| {
                            let i = (u.clamp(0.0, 1.0) * (h.metrics.width - 1) as f32) as u32;
                            let j = (v.clamp(0.0, 1.0) * (h.metrics.height - 1) as f32) as u32;
                            h.get(i.min(h.metrics.width - 1), j.min(h.metrics.height - 1))
                        })
                        .unwrap_or(50.0)
                }
                _ => 0.0,
            };
            let stroke_kind = shape_tool.stroke_kind();
            let mut actions = Vec::new();
            let stamp_once = shape_tool.is_stamp() && self.last_paint_uv.is_some();
            if stamp_once {
                // Stamps are one-shot per click (drag doesn't spam new mountains).
                return;
            }
            if !shape_tool.is_stamp() {
                if let Some((pu, pv)) = self.last_paint_uv {
                    let du = u - pu;
                    let dv = v - pv;
                    let dist = (du * du + dv * dv).sqrt();
                    let spacing = (radius * 0.35).max(0.002);
                    if dist > spacing {
                        let steps = ((dist / spacing).ceil() as u32).clamp(1, 32);
                        for i in 1..steps {
                            let t = i as f32 / steps as f32;
                            actions.push(PanelAction::PaintSculptStamp {
                                layer: layer_id,
                                u: pu + du * t,
                                v: pv + dv * t,
                                radius,
                                strength,
                                stroke_kind,
                                target_height,
                            });
                        }
                    }
                }
            }
            actions.push(PanelAction::PaintSculptStamp {
                layer: layer_id,
                u,
                v,
                radius,
                strength,
                stroke_kind,
                target_height,
            });
            self.apply_actions(actions);
            self.last_paint_uv = Some((u, v));
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
            let Some(layer_id) = self.ensure_shape_authoring_layer() else {
                return;
            };
            use terra_core::authoring::SculptStrokeKind;
            let stroke_kind = match self.ui_state.editor_tool {
                crate::ui::EditorTool::Protect => SculptStrokeKind::Protect,
                crate::ui::EditorTool::Hardness => SculptStrokeKind::Hardness,
                _ => SculptStrokeKind::Sediment,
            };
            self.ui_state.ensure_sculpt_defaults();
            self.sculpt_stroke_active = true;
            let strength = self.ui_state.sculpt_strength.clamp(0.05, 1.0);
            let radius = self.ui_state.sculpt_radius;
            let mut actions = Vec::new();
            actions.push(PanelAction::PaintSculptStamp {
                layer: layer_id,
                u,
                v,
                radius,
                strength,
                stroke_kind,
                target_height: 0.0,
            });
            self.apply_actions(actions);
            self.last_paint_uv = Some((u, v));
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
                if self.last_paint_uv.is_some() {
                    return; // one-shot per click
                }
                actions.push(PanelAction::BeginBiomePaintStroke { biome });
                actions.push(PanelAction::PaintBiomeStamp {
                    biome,
                    u,
                    v,
                    radius,
                    strength,
                    erase,
                    mode: Some(tool),
                });
                actions.push(PanelAction::EndBiomePaintStroke);
                self.apply_actions(actions);
                self.last_paint_uv = Some((u, v));
                self.placement_tint_dirty = true;
                return;
            }

            if tool == terra_core::biome_paint::BiomePaintTool::PolygonFill {
                // Close polygon when clicking near the first vertex.
                if let Some(&(fu, fv)) = self.biome_polygon_points.first() {
                    if (fu - u).hypot(fv - v) < 0.025 && self.biome_polygon_points.len() >= 3 {
                        let pts = std::mem::take(&mut self.biome_polygon_points);
                        actions.push(PanelAction::BeginBiomePaintStroke { biome });
                        let res = self.session.document.preview_resolution.min(8192).max(64);
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
                self.last_paint_uv = Some((u, v));
                return;
            }

            // Capture undo snapshot at the start of a drag.
            if self.last_paint_uv.is_none() {
                actions.push(PanelAction::BeginBiomePaintStroke { biome });
            }

            if tool.paints_mask() {
                if let Some((pu, pv)) = self.last_paint_uv {
                    let du = u - pu;
                    let dv = v - pv;
                    let dist = (du * du + dv * dv).sqrt();
                    let spacing = (radius * 0.35).max(0.002);
                    if dist > spacing {
                        let steps = ((dist / spacing).ceil() as u32).clamp(1, 32);
                        for i in 1..steps {
                            let t = i as f32 / steps as f32;
                            actions.push(PanelAction::PaintBiomeStamp {
                                biome,
                                u: pu + du * t,
                                v: pv + dv * t,
                                radius,
                                strength,
                                erase,
                                mode: Some(tool),
                            });
                        }
                    }
                }
                actions.push(PanelAction::PaintBiomeStamp {
                    biome,
                    u,
                    v,
                    radius,
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
                    let target_height = self
                        .last_height
                        .as_ref()
                        .map(|h| {
                            let i = (u.clamp(0.0, 1.0) * (h.metrics.width - 1) as f32) as u32;
                            let j = (v.clamp(0.0, 1.0) * (h.metrics.height - 1) as f32) as u32;
                            h.get(i.min(h.metrics.width - 1), j.min(h.metrics.height - 1))
                        })
                        .unwrap_or(0.0);
                    if let Some((pu, pv)) = self.last_paint_uv {
                        let du = u - pu;
                        let dv = v - pv;
                        let dist = (du * du + dv * dv).sqrt();
                        let spacing = (radius * 0.35).max(0.002);
                        if dist > spacing {
                            let steps = ((dist / spacing).ceil() as u32).clamp(1, 32);
                            for i in 1..steps {
                                let t = i as f32 / steps as f32;
                                actions.push(PanelAction::PaintSculptStamp {
                                    layer: layer_id,
                                    u: pu + du * t,
                                    v: pv + dv * t,
                                    radius,
                                    strength: sculpt_strength,
                                    stroke_kind,
                                    target_height,
                                });
                            }
                        }
                    }
                    actions.push(PanelAction::PaintSculptStamp {
                        layer: layer_id,
                        u,
                        v,
                        radius,
                        strength: sculpt_strength,
                        stroke_kind,
                        target_height,
                    });
                }
            }

            self.sculpt_stroke_active = true;
            self.apply_actions(actions);
            self.last_paint_uv = Some((u, v));
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
        if let Some((pu, pv)) = self.last_paint_uv {
            let du = u - pu;
            let dv = v - pv;
            let dist = (du * du + dv * dv).sqrt();
            let spacing = (radius * self.ui_state.brush_spacing.clamp(0.05, 1.0)).max(0.002);
            if dist > spacing {
                let steps = ((dist / spacing).ceil() as u32).clamp(1, 32);
                for i in 1..steps {
                    let t = i as f32 / steps as f32;
                    actions.push(PanelAction::PaintMaskStamp {
                        mask_id,
                        u: pu + du * t,
                        v: pv + dv * t,
                        radius,
                        strength,
                        hardness,
                        tool,
                    });
                }
            }
        }
        actions.push(PanelAction::PaintMaskStamp {
            mask_id,
            u,
            v,
            radius,
            strength,
            hardness,
            tool,
        });
        self.apply_actions(actions);
        self.last_paint_uv = Some((u, v));
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
        let res = self.session.document.preview_resolution.min(8192).max(64);
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

    /// Raycast cursor onto the height surface â†’ terrain UV (same mapping as the brush gizmo).
    pub(crate) fn pick_paint_uv(&self) -> Option<(f32, f32)> {
        let (x, y) = self.cursor_logical()?;
        if !self.viewport_rect.contains(x, y) {
            return None;
        }
        // Once a stroke has started, keep stamping even if the cursor grazes overlays
        // (brush bar / gizmo chrome) â€” otherwise live preview stalls mid-drag.
        if self.gui_wants_pointer && !self.sculpt_stroke_active {
            return None;
        }
        let renderer = self.renderer.as_ref()?;
        let window = self.window.as_ref()?;
        let ppp = window.scale_factor() as f32;
        let (surface_w, surface_h) = renderer.size();
        let screen_w = surface_w as f32 / ppp;
        let screen_h = surface_h as f32 / ppp;
        let aspect = surface_w as f32 / surface_h.max(1) as f32;
        pick_terrain_uv_on_surface(
            &renderer.camera,
            aspect,
            (x, y),
            (screen_w, screen_h),
            renderer.heights.world_size,
            self.last_height.as_ref(),
        )
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
        let show = self.viewport_paint_tool_armed()
            && self.cursor_in_viewport()
            && !self.gui_wants_pointer
            && !self.modifiers_alt;
        if !show {
            if let Some(renderer) = self.renderer.as_mut() {
                renderer.set_brush_gizmo(None);
                renderer.sync_brush_geometry(None);
            }
            return;
        }
        self.ui_state.ensure_sculpt_defaults();
        let radius = if self.ui_state.editor_tool.is_place_point() {
            0.012
        } else if self.ui_state.editor_tool.is_sculpt() {
            self.ui_state.sculpt_radius
        } else {
            self.ui_state.sculpt_radius.max(0.02)
        };
        let color = self.brush_gizmo_color();
        let Some((u, v)) = self.pick_paint_uv() else {
            if let Some(renderer) = self.renderer.as_mut() {
                renderer.set_brush_gizmo(None);
                renderer.sync_brush_geometry(None);
            }
            return;
        };
        let world = self
            .renderer
            .as_ref()
            .map(|r| r.heights.world_size)
            .unwrap_or((4096.0, 4096.0));
        let ring_y = terra_render::BrushOverlay::sample_ring_heights(
            self.last_height.as_ref(),
            world,
            u,
            v,
            radius,
        );
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.set_brush_gizmo(Some(BrushGizmo {
                u,
                v,
                radius_uv: radius,
                color,
            }));
            renderer.sync_brush_geometry(Some(&ring_y));
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
