use std::time::Instant;

use crate::ui::PanelAction;
use terra_core::layer::LayerKind;

use super::{LayerPointDrag, LayerPointKind, TerraApp, PAINT_DEBOUNCE_MS};
impl TerraApp {
    pub(crate) fn flush_live_paint_preview(&mut self) {
        if !self.pending_eval {
            return;
        }
        if self.last_refine.elapsed().as_millis() < PAINT_DEBOUNCE_MS {
            return;
        }
        self.pending_eval = false;
        self.force_draft = true;
        self.run_eval_step();
        self.last_refine = Instant::now();
    }

    /// Click-select a nearby landform shape, or start dragging a control point.
    pub(crate) fn try_pick_shape_at_cursor(&mut self) -> bool {
        if self.ui_state.app_workspace != crate::ui::AppWorkspace::Landforms {
            return false;
        }
        if self.ui_state.editor_tool != crate::ui::EditorTool::Move {
            return false;
        }
        let Some((u, v)) = self.pick_paint_uv() else {
            return false;
        };
        // Prefer dragging a control point on the already-selected shape.
        if let Some(sel) = self.session.document.shapes.selected {
            if let Some(shape) = self.session.document.shapes.get(sel) {
                let mut best_i = None;
                let mut best_d = 0.028_f32;
                for (i, p) in shape.transformed_points().iter().enumerate() {
                    let d = (p.u - u).hypot(p.v - v);
                    if d < best_d {
                        best_d = d;
                        best_i = Some(i);
                    }
                }
                if let Some(index) = best_i {
                    self.dragging_shape_point = Some((sel, index));
                    self.ui_state.status =
                        format!("Dragging point {} on {}", index + 1, shape.name);
                    return true;
                }
            }
        }
        let mut best = None;
        let mut best_d = 0.035_f32;
        for shape in &self.session.document.shapes.shapes {
            if !shape.enabled {
                continue;
            }
            for p in shape.transformed_points() {
                let d = (p.u - u).hypot(p.v - v);
                if d < best_d {
                    best_d = d;
                    best = Some(shape.id);
                }
            }
        }
        if let Some(id) = best {
            self.session.document.shapes.selected = Some(id);
            self.session.document.selected = None;
            self.ui_state.status = format!(
                "Selected {}",
                self.session
                    .document
                    .shapes
                    .get(id)
                    .map(|s| s.name.as_str())
                    .unwrap_or("shape")
            );
            true
        } else {
            false
        }
    }

    pub(crate) fn update_shape_point_drag(&mut self) {
        let Some((id, index)) = self.dragging_shape_point else {
            return;
        };
        let Some((u, v)) = self.pick_paint_uv() else {
            return;
        };
        self.apply_actions(vec![PanelAction::SetShapePoint { id, index, u, v }]);
    }

    pub(crate) fn end_shape_point_drag(&mut self) {
        if self.dragging_shape_point.take().is_some() {
            self.apply_actions(vec![PanelAction::CompileShapes]);
            self.ui_state.status = "Shape updated".into();
        }
    }

    pub(crate) fn sync_placement_tint_to_renderer(&mut self) {
        let Some(r) = self.renderer.as_mut() else {
            return;
        };
        let doc = &self.session.document;
        let Some(layer) = doc.selected_placement_layer() else {
            r.upload_placement_tint(1, 1, &[0, 0, 0, 0]);
            r.set_biome_tint_strength(0.0);
            self.placement_tint_dirty = false;
            return;
        };
        if layer.channels.is_empty() || !self.ui_state.biome_color_preview {
            r.upload_placement_tint(1, 1, &[0, 0, 0, 0]);
            r.set_biome_tint_strength(0.0);
            self.placement_tint_dirty = false;
            return;
        }
        let res = doc.preview_resolution.min(512).max(64);
        let isolate = if layer.isolate_active {
            doc.active_biome
        } else {
            None
        };
        let rgba = layer.bake_color_rgba(
            res,
            res,
            &|gid| {
                doc.biome_library
                    .by_group(gid)
                    .map(|d| d.color)
                    .or_else(|| doc.stack.find_group(gid).map(|g| g.preview_color))
                    .unwrap_or([0.45, 0.55, 0.4])
            },
            isolate,
        );
        r.upload_placement_tint(res, res, &rgba);
        r.set_biome_tint_strength(layer.mask_visibility.clamp(0.35, 0.85));
        self.placement_tint_dirty = false;
    }

    pub(crate) fn place_point_at_cursor(&mut self) {
        let Some((u, v)) = self.pick_paint_uv() else {
            return;
        };
        if self.ui_state.editor_tool == crate::ui::EditorTool::Measure {
            self.handle_measure_click(u, v);
            return;
        }
        let Some(id) = self.session.document.selected else {
            return;
        };
        let Some(layer) = self.session.document.stack.find(id) else {
            return;
        };
        let mut kind = layer.kind.clone();
        match (&mut kind, self.ui_state.editor_tool) {
            (LayerKind::Path(p), crate::ui::EditorTool::EditPath) => {
                let nearest = p
                    .nodes
                    .iter()
                    .enumerate()
                    .filter_map(|(index, node)| {
                        let distance = (node.u - u).hypot(node.v - v);
                        (distance < 0.028).then_some((index, distance))
                    })
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|(index, _)| index);
                if let Some(index) = nearest {
                    if self.modifiers_ctrl {
                        p.nodes.remove(index);
                        self.apply_actions(vec![PanelAction::SetKind { id, kind }]);
                        self.ui_state.status = "Path node deleted".into();
                        return;
                    }
                    let node = &p.nodes[index];
                    self.dragging_layer_point = Some(LayerPointDrag {
                        layer: id,
                        index,
                        kind: LayerPointKind::Path,
                        start_cursor: self.last_cursor.unwrap_or_default(),
                        start_u: node.u,
                        start_v: node.v,
                        start_height: node.height,
                    });
                    self.ui_state.status =
                        "Dragging path node Â· Shift: elevation Â· Ctrl-click: delete".into();
                    return;
                }
                p.nodes.push(terra_core::layer::PathNode {
                    u,
                    v,
                    height: 0.0,
                    width: 1.0,
                });
            }
            (LayerKind::PolygonHeight(p), crate::ui::EditorTool::EditPolygon) => {
                let nearest = p
                    .points
                    .iter()
                    .enumerate()
                    .filter_map(|(index, point)| {
                        let distance = (point[0] - u).hypot(point[1] - v);
                        (distance < 0.028).then_some((index, distance))
                    })
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|(index, _)| index);
                if let Some(index) = nearest {
                    if self.modifiers_ctrl {
                        p.points.remove(index);
                        self.apply_actions(vec![PanelAction::SetKind { id, kind }]);
                        self.ui_state.status = "Polygon vertex deleted".into();
                        return;
                    }
                    let point = p.points[index];
                    self.dragging_layer_point = Some(LayerPointDrag {
                        layer: id,
                        index,
                        kind: LayerPointKind::Polygon,
                        start_cursor: self.last_cursor.unwrap_or_default(),
                        start_u: point[0],
                        start_v: point[1],
                        start_height: p.height,
                    });
                    self.ui_state.status =
                        "Dragging polygon vertex Â· Shift: height Â· Ctrl-click: delete".into();
                    return;
                }
                p.points.push([u, v]);
            }
            (LayerKind::RiverNetwork(p), crate::ui::EditorTool::EditRiverSpring) => {
                p.springs.push(terra_core::layer::RiverNode {
                    u,
                    v,
                    flow: 1.0,
                    width: 1.0,
                });
            }
            _ => return,
        }
        self.apply_actions(vec![PanelAction::SetKind { id, kind }]);
    }

    pub(crate) fn update_layer_point_drag(&mut self) {
        let Some(drag) = self.dragging_layer_point else {
            return;
        };
        let Some(layer) = self.session.document.stack.find(drag.layer) else {
            self.dragging_layer_point = None;
            return;
        };
        let mut kind = layer.kind.clone();
        if self.modifiers_shift {
            let Some((_, cursor_y)) = self.last_cursor else {
                return;
            };
            let world_span = self
                .session
                .document
                .metrics
                .world_size_x
                .max(self.session.document.metrics.world_size_z);
            let meters_per_pixel = (world_span / 2048.0).clamp(0.1, 5.0);
            let height =
                drag.start_height - (cursor_y - drag.start_cursor.1) as f32 * meters_per_pixel;
            match (&mut kind, drag.kind) {
                (LayerKind::Path(params), LayerPointKind::Path) => {
                    let Some(node) = params.nodes.get_mut(drag.index) else {
                        return;
                    };
                    node.u = drag.start_u;
                    node.v = drag.start_v;
                    node.height = height;
                    self.ui_state.status = format!("Path node elevation {height:.1} m");
                }
                (LayerKind::PolygonHeight(params), LayerPointKind::Polygon) => {
                    params.height = height.max(0.0);
                    self.ui_state.status = format!("Polygon height {:.1} m", params.height);
                }
                _ => return,
            }
        } else {
            let Some((u, v)) = self.pick_paint_uv() else {
                return;
            };
            match (&mut kind, drag.kind) {
                (LayerKind::Path(params), LayerPointKind::Path) => {
                    let Some(node) = params.nodes.get_mut(drag.index) else {
                        return;
                    };
                    node.u = u;
                    node.v = v;
                }
                (LayerKind::PolygonHeight(params), LayerPointKind::Polygon) => {
                    let Some(point) = params.points.get_mut(drag.index) else {
                        return;
                    };
                    *point = [u, v];
                }
                _ => return,
            }
        }
        self.apply_actions(vec![PanelAction::SetKind {
            id: drag.layer,
            kind,
        }]);
    }

    pub(crate) fn end_layer_point_drag(&mut self) {
        if self.dragging_layer_point.take().is_some() {
            self.ui_state.status = "Shape layer updated".into();
        }
    }

    pub(crate) fn handle_measure_click(&mut self, u: f32, v: f32) {
        match self.measure_anchor.take() {
            None => {
                self.measure_anchor = Some((u, v));
                self.ui_state.status = "Measure: click end point".into();
            }
            Some((u0, v0)) => {
                let m = &self.session.document.metrics;
                let dx = (u - u0) * m.world_size_x;
                let dz = (v - v0) * m.world_size_z;
                let mut dy = 0.0;
                if let Some(hf) = self.last_height.as_ref() {
                    let sample = |uu: f32, vv: f32| {
                        let i = ((uu * (hf.metrics.width as f32 - 1.0)).round() as u32)
                            .min(hf.metrics.width.saturating_sub(1));
                        let j = ((vv * (hf.metrics.height as f32 - 1.0)).round() as u32)
                            .min(hf.metrics.height.saturating_sub(1));
                        hf.get(i, j)
                    };
                    dy = sample(u, v) - sample(u0, v0);
                }
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                self.ui_state.status = format!("{dist:.1} m");
            }
        }
    }
}
