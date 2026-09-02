//! UI PanelAction dispatch, split by domain.

mod biomes;
mod layers;
mod masks;
mod resolution;
mod scenarios;
mod settings;
mod tools;
mod world_rules;

use crate::ui::PanelAction;
use terra_core::deps::NodeRef;
use terra_core::field_data::FieldId;
use terra_core::layer::LayerId;
use terra_core::terrain_plan::{PlanDirtyScope, TerrainEditClass, TerrainOpKind};
use terra_core::tiling::UvRect;

use super::logical_frame::FrameDeadlineKind;
use super::TerraApp;

/// Shared mutation flags for a single apply_actions batch.
pub(crate) struct ApplyCtx {
    pub dirty_from: Option<LayerId>,
    /// Resolution-free bounded edit scope shared by GPU preview and CPU worker.
    /// It is converted to texels only after the evaluation quality is resolved.
    pub sculpt_dirty_region_uv: Option<UvRect>,
    pub doc_mutated: bool,
    pub mask_assets_mutated: bool,
    pub deferred_rebuild: bool,
    pub continue_loop: bool,
}

impl ApplyCtx {
    pub fn new() -> Self {
        Self {
            dirty_from: None,
            sculpt_dirty_region_uv: None,
            doc_mutated: false,
            mask_assets_mutated: false,
            deferred_rebuild: false,
            continue_loop: false,
        }
    }
}

/// Fold a sculpt-edit footprint (normalized UV) into the batch's dirty
/// accumulators, exactly as [`masks::try_apply`]'s paint-dab arm does for a stamp:
/// union the UV rect carried to the CPU worker and GPU preview paths. Used by
/// the per-stroke edit arms (#121) so an inspector slider / eye toggle / delete
/// rescopes to the edited stroke's footprint instead of escalating whole-field.
pub(crate) fn accumulate_sculpt_footprint(_app: &TerraApp, ctx: &mut ApplyCtx, uv: UvRect) {
    ctx.sculpt_dirty_region_uv = Some(match ctx.sculpt_dirty_region_uv {
        Some(existing) => existing.union(uv),
        None => uv,
    });
}

impl TerraApp {
    pub(crate) fn apply_actions(&mut self, actions: Vec<PanelAction>) {
        let selection_before = self.session.document.selected;
        let mut ctx = ApplyCtx::new();
        for action in actions {
            ctx.continue_loop = false;
            let action = match resolution::try_apply(self, action, &mut ctx) {
                Ok(()) => continue,
                Err(a) => a,
            };
            let action = match layers::try_apply(self, action, &mut ctx) {
                Ok(()) => continue,
                Err(a) => a,
            };
            if ctx.continue_loop {
                continue;
            }
            let action = match masks::try_apply(self, action, &mut ctx) {
                Ok(()) => continue,
                Err(a) => a,
            };
            if ctx.continue_loop {
                continue;
            }
            let action = match biomes::try_apply(self, action, &mut ctx) {
                Ok(()) => continue,
                Err(a) => a,
            };
            if ctx.continue_loop {
                continue;
            }
            let action = match world_rules::try_apply(self, action, &mut ctx) {
                Ok(()) => continue,
                Err(a) => a,
            };
            if ctx.continue_loop {
                continue;
            }
            let action = match scenarios::try_apply(self, action, &mut ctx) {
                Ok(()) => continue,
                Err(a) => a,
            };
            if ctx.continue_loop {
                continue;
            }
            let action = match settings::try_apply(self, action, &mut ctx) {
                Ok(()) => continue,
                Err(a) => a,
            };
            if ctx.continue_loop {
                continue;
            }
            match tools::try_apply(self, action, &mut ctx) {
                Ok(()) => {}
                Err(_a) => {
                    log::warn!("Unhandled PanelAction in apply_actions");
                }
            }
        }
        let dirty_from = ctx.dirty_from;
        let sculpt_dirty_region_uv = ctx.sculpt_dirty_region_uv;
        let doc_mutated = ctx.doc_mutated;
        let mask_assets_mutated = ctx.mask_assets_mutated;
        // Set by a paint dab (SculptBase/SculptStrokes stamp) or a per-stroke edit
        // (#121): both carry a bounded UV footprint, so both take the scoped worker
        // path and present just the dirty rect through the GPU.
        let has_sculpt_footprint = sculpt_dirty_region_uv.is_some();
        if let Some(id) = dirty_from {
            let edit = if has_sculpt_footprint {
                TerrainEditClass::Content {
                    owner: NodeRef::Layer(id),
                    fields: vec![FieldId::Height],
                    scope: PlanDirtyScope::Region(
                        sculpt_dirty_region_uv.expect("footprint checked above"),
                    ),
                }
            } else if self
                .terrain_plan_cache
                .last_good_plan()
                .is_some_and(|plan| {
                    plan_owner_shape_matches(plan, &self.session.document.stack, id)
                })
            {
                TerrainEditClass::Parameters {
                    owner: if self.session.document.stack.find_group(id).is_some() {
                        NodeRef::Group(id)
                    } else {
                        NodeRef::Layer(id)
                    },
                }
            } else {
                TerrainEditClass::Structure
            };
            self.pending_plan_edits.push(edit);
        }
        if let Some(region) = sculpt_dirty_region_uv {
            self.pending_gpu_dirty_region = Some(match self.pending_gpu_dirty_region {
                Some(existing) => existing.union(region),
                None => region,
            });
            self.deferred_full_field = None;
            self.logical_frames
                .clear_deadline(FrameDeadlineKind::FullFieldRefinement);
        }
        if let Some(id) = dirty_from {
            // Suffix-only dirty â€” do not mark_all_dirty (preserves layer cache).
            // A SculptBase stamp only changes the base paint buffer; keep GPU dependents
            // clean so Draft can reuse cached noise/shape contributions and just re-blend.
            // Use preview stack so Global layer ids resolve (they are not in doc.stack).
            let preview = self.session.document.preview_eval_stack();
            let is_global = false;
            if is_global {
                self.mark_all_layers_dirty();
                self.request_rebuild();
            } else {
                let sculpt_only = has_sculpt_footprint
                    && matches!(
                        preview.find(id).map(|layer| &layer.kind),
                        Some(terra_core::layer::LayerKind::SculptBase(_))
                    );
                self.scheduler.evaluator.mark_dirty_from(&preview, id);
                // Paint dabs and per-stroke edits carry a UV footprint; other suffix
                // edits are whole-field.
                let footprint = if has_sculpt_footprint {
                    sculpt_dirty_region_uv
                } else {
                    None
                };
                self.track_worker_dirty_from(&preview, id, footprint);
                self.advance_output_revision();
                if let Some(gpu) = self.gpu_engine.as_mut() {
                    if sculpt_only {
                        gpu.mark_dirty(id);
                    } else {
                        gpu.mark_dirty_from(&preview, id);
                    }
                }
                if has_sculpt_footprint || ctx.deferred_rebuild {
                    self.request_rebuild();
                } else {
                    // Add/reorder/param: present Draft on the next tick (WC realtime).
                    self.request_rebuild_immediate();
                }
            }
        }
        if mask_assets_mutated {
            self.mark_all_layers_dirty();
            self.request_rebuild();
        }
        if doc_mutated || dirty_from.is_some() || has_sculpt_footprint {
            self.mark_document_dirty();
        }
        if self.session.document.selected != selection_before {
            self.layers_gui.reveal_selection(&self.session.document);
        }
    }
}

fn plan_owner_shape_matches(
    plan: &terra_core::terrain_plan::CompiledTerrainPlan,
    stack: &terra_core::layer::LayerStack,
    id: LayerId,
) -> bool {
    if let Some(layer) = stack.find(id) {
        return plan.operations().iter().any(|operation| {
            matches!(
                &operation.kind,
                TerrainOpKind::RunLayerKernel { layer: owner, type_id, .. }
                    if *owner == id && type_id == layer.kind.type_id()
            )
        });
    }
    stack.find_group(id).is_some()
        && !plan
            .provenance()
            .operations_for(NodeRef::Group(id))
            .is_empty()
}
