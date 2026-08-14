//! UI PanelAction dispatch, split by domain.

mod biomes;
mod layers;
mod masks;
mod scenarios;
mod settings;
mod tools;
mod world_rules;

use crate::ui::PanelAction;
use terra_core::layer::LayerId;

use super::TerraApp;

/// Shared mutation flags for a single apply_actions batch.
pub(crate) struct ApplyCtx {
    pub dirty_from: Option<LayerId>,
    pub sculpt_dirty_rect: Option<(u32, u32, u32, u32)>,
    pub doc_mutated: bool,
    pub mask_assets_mutated: bool,
    pub continue_loop: bool,
}

impl ApplyCtx {
    pub fn new() -> Self {
        Self {
            dirty_from: None,
            sculpt_dirty_rect: None,
            doc_mutated: false,
            mask_assets_mutated: false,
            continue_loop: false,
        }
    }
}

impl TerraApp {
    pub(crate) fn apply_actions(&mut self, actions: Vec<PanelAction>) {
        let mut ctx = ApplyCtx::new();
        for action in actions {
            ctx.continue_loop = false;
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
        let sculpt_dirty_rect = ctx.sculpt_dirty_rect;
        let doc_mutated = ctx.doc_mutated;
        let mask_assets_mutated = ctx.mask_assets_mutated;
        let sculpt_stamp = sculpt_dirty_rect.is_some();
        if let Some(rect) = sculpt_dirty_rect {
            if let Some(gpu) = self.gpu_engine.as_mut() {
                gpu.set_dirty_rect(Some(rect));
            }
        }
        if let Some(id) = dirty_from {
            // Suffix-only dirty â€” do not mark_all_dirty (preserves layer cache).
            // Sculpt stamps only change the base paint buffer; keep GPU dependents clean so
            // Draft can reuse cached noise/shape contributions and just re-blend.
            // Use preview stack so Global layer ids resolve (they are not in doc.stack).
            let preview = self.session.document.preview_eval_stack();
            let is_global = false;
            if is_global {
                self.mark_all_layers_dirty();
                self.request_rebuild();
            } else {
                let sculpt_only = sculpt_stamp
                    && matches!(
                        preview.find(id).map(|layer| &layer.kind),
                        Some(terra_core::layer::LayerKind::SculptBase(_))
                    );
                self.scheduler.evaluator.mark_dirty_from(&preview, id);
                self.track_worker_dirty_from(&preview, id);
                self.terrain_runtime.advance_output_revision();
                if let Some(gpu) = self.gpu_engine.as_mut() {
                    if sculpt_only {
                        gpu.mark_dirty(id);
                    } else {
                        gpu.mark_dirty_from(&preview, id);
                    }
                }
                if sculpt_stamp {
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
        if doc_mutated || dirty_from.is_some() || sculpt_stamp {
            self.mark_document_dirty();
        }
    }
}
