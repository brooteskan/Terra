use terra_core::heightfield::Heightfield;
use terra_core::layer::LayerStack;
use terra_core::mask::{bake_mask_assets, MaskAsset};
use terra_core::terrain_plan::{
    resolve_infinite_plan_domain, CompiledTerrainPlan, TerrainPlanDomainRejection,
};
use terra_core::{TerrainContentStamp, TerrainEvaluationDomain, TerrainTileKey};

use crate::{EvalContext, EvalError, StackEvaluator};

#[derive(Debug)]
pub enum CpuTileEvaluationError {
    NotInfinite,
    StalePlan { plan: u64, requested: u64 },
    Domain(TerrainPlanDomainRejection),
    Evaluation(EvalError),
}

#[derive(Debug, Clone, PartialEq)]
pub struct CpuPackedHeightTile {
    pub key: TerrainTileKey,
    pub content: TerrainContentStamp,
    pub width: u32,
    pub height: u32,
    pub halo: u32,
    pub samples: Vec<f32>,
}

/// CPU tile orchestrator. Layer dispatch remains owned by `StackEvaluator`.
pub struct InfiniteTileEvaluator {
    evaluator: StackEvaluator,
}

impl InfiniteTileEvaluator {
    pub fn new() -> Self {
        Self {
            evaluator: StackEvaluator::new(),
        }
    }

    /// Deterministically evaluate one admitted Infinite sparse tile.
    pub fn evaluate(
        &mut self,
        stack: &LayerStack,
        mask_assets: &[MaskAsset],
        plan: &CompiledTerrainPlan,
        domain: TerrainEvaluationDomain,
    ) -> Result<CpuPackedHeightTile, CpuTileEvaluationError> {
        if !domain.is_infinite() {
            return Err(CpuTileEvaluationError::NotInfinite);
        }
        let revision = plan.stamp().structure_revision.get();
        if revision != domain.content.plan_revision {
            return Err(CpuTileEvaluationError::StalePlan {
                plan: revision,
                requested: domain.content.plan_revision,
            });
        }
        resolve_infinite_plan_domain(stack, mask_assets, plan, plan.final_height())
            .map_err(CpuTileEvaluationError::Domain)?;

        let metrics = domain.local_metrics();
        let seed = Heightfield::zeros(metrics);
        let mut context = EvalContext::new(metrics);
        context.mask_assets = mask_assets.to_vec();
        context.masks = bake_mask_assets(mask_assets, &seed, metrics, &context.aux);
        context.set_evaluation_domain(domain.clone());
        let evaluated = self
            .evaluator
            .evaluate_nodes(&stack.nodes, &mut context, &seed)
            .map_err(CpuTileEvaluationError::Evaluation)?;
        let dense = evaluated.to_dense();
        let width = domain.published_width();
        let height = domain.published_height();
        let mut samples = Vec::with_capacity((width * height) as usize);
        for z in 0..height {
            let source_z = domain.operation_halo + z;
            let start = (source_z * metrics.width + domain.operation_halo) as usize;
            samples.extend_from_slice(&dense[start..start + width as usize]);
        }
        Ok(CpuPackedHeightTile {
            key: domain.key,
            content: domain.content,
            width,
            height,
            halo: domain.publication_halo,
            samples,
        })
    }
}

impl Default for InfiniteTileEvaluator {
    fn default() -> Self {
        Self::new()
    }
}
