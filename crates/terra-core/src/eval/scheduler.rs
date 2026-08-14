use super::{EvalContext, EvalError, StackEvaluator};
use crate::analyze::LevelStepSettings;
use crate::heightfield::{Heightfield, HeightfieldMetrics};
use crate::layer::LayerStack;
use crate::mask::{bake_mask_assets, MaskAsset, MaskField};
use crate::quality::PreviewQuality;
use std::collections::HashMap;
use std::sync::Arc;

pub struct EvalJob {
    pub quality: PreviewQuality,
    pub token: u64,
}

/// Progressive / cancellable evaluation helper.
pub struct EvalScheduler {
    pub evaluator: StackEvaluator,
    pub current_token: u64,
    pub last_good: Option<Arc<Heightfield>>,
    pub last_aux: HashMap<String, MaskField>,
    /// Materials strata preserved across HashMap aux round-trips.
    pub last_strata: Option<Vec<crate::layer::Stratum>>,
    /// Most recent per-layer CPU timing and cache provenance.
    pub last_layer_timings: Vec<super::LayerEvalTiming>,
    pub quality: PreviewQuality,
}

impl Default for EvalScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl EvalScheduler {
    pub fn new() -> Self {
        Self {
            evaluator: StackEvaluator::new(),
            current_token: 0,
            last_good: None,
            last_aux: HashMap::new(),
            last_strata: None,
            last_layer_timings: Vec::new(),
            quality: PreviewQuality::Draft,
        }
    }

    pub fn request_rebuild(&mut self) -> u64 {
        self.current_token = self.current_token.wrapping_add(1);
        self.quality = PreviewQuality::Draft;
        self.current_token
    }

    pub fn run_step(
        &mut self,
        stack: &LayerStack,
        base_metrics: HeightfieldMetrics,
        preview_res: u32,
        export_res: u32,
        token: u64,
        level_steps: &LevelStepSettings,
        mask_assets: &[MaskAsset],
        aux: &HashMap<String, MaskField>,
    ) -> Result<Option<Heightfield>, EvalError> {
        if token != self.current_token {
            return Err(EvalError::Cancelled);
        }
        let res = self.quality.resolution(preview_res, export_res);
        let metrics = HeightfieldMetrics {
            width: res,
            height: res,
            world_size_x: base_metrics.world_size_x,
            world_size_z: base_metrics.world_size_z,
            tile_size: base_metrics.tile_size.min(res),
            halo: base_metrics.halo,
        };
        let mut ctx = EvalContext::new(metrics);
        ctx.quality = self.quality;
        ctx.level_steps = level_steps.clone();
        ctx.mask_assets = mask_assets.to_vec();
        ctx.set_aux_hashmap(aux.clone());
        if let Some(strata) = &self.last_strata {
            ctx.aux_maps.strata = Some(strata.clone());
        }
        // Resolution change invalidates CPU layer cache dimensions.
        if self
            .last_good
            .as_ref()
            .map(|h| h.metrics.width != metrics.width || h.metrics.height != metrics.height)
            .unwrap_or(false)
        {
            self.evaluator.mark_all_dirty(stack);
        }
        // Bake masks from last-good (or flat zero) so layer MaskRefs affect compositing.
        let owned_zero;
        let reference: &Heightfield = match self.last_good.as_ref() {
            Some(h) => h.as_ref(),
            None => {
                owned_zero = Heightfield::zeros(metrics);
                &owned_zero
            }
        };
        ctx.masks = bake_mask_assets(mask_assets, reference, metrics, aux);
        let hf = self.evaluator.rebuild_incremental(stack, &mut ctx)?;
        if token != self.current_token {
            return Err(EvalError::Cancelled);
        }
        ctx.sync_aux_hashmap();
        self.last_strata = ctx.aux_maps.strata.clone();
        self.last_layer_timings = ctx.layer_timings.clone();
        self.last_aux = ctx.aux;
        self.last_good = Some(Arc::new(hf.clone()));
        Ok(Some(hf))
    }

    pub fn advance_quality(&mut self) -> bool {
        if let Some(next) = self.quality.next_refine() {
            self.quality = next;
            true
        } else {
            false
        }
    }
}
