//! Structural plan caching, revision ownership, and instrumentation.

use crate::deps::NodeRef;
use crate::invalidation::{AuxReach, Reach};
use crate::layer::LayerStack;
use crate::mask::MaskAsset;

use super::{
    compile_terrain_plan, propagate_plan_edits, CompiledTerrainPlan, PlanInvalidation,
    PlanStructureRevision, TerrainEditClass, TerrainOpKind, TerrainPlanDiagnostic,
    TerrainPlanStamp, TerrainPlanWork,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PlanCacheStatsSnapshot {
    pub plan_compiles: u64,
    pub successful_compiles: u64,
    pub plan_cache_hits: u64,
    pub patched_operations: u64,
    pub operations_reached: u64,
    pub full_field_escalations: u64,
}

#[derive(Debug, Clone, Default)]
pub struct PlanCacheStats {
    snapshot: PlanCacheStatsSnapshot,
}

impl PlanCacheStats {
    pub const fn snapshot(&self) -> PlanCacheStatsSnapshot {
        self.snapshot
    }

    pub fn reset(&mut self) {
        self.snapshot = PlanCacheStatsSnapshot::default();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "terrain plan revision {plan_revision} is stale; authored structure is at revision {current_revision}"
)]
pub struct PlanRevisionError {
    pub plan_revision: u64,
    pub current_revision: u64,
}

/// Runtime owner of the current authored-structure generation and last-good
/// compiled plan. Candidate failures never discard the prior plan, but stale
/// plans cannot be acquired for execution against a newer structure.
#[derive(Debug, Default)]
pub struct TerrainPlanCache {
    structure_revision: PlanStructureRevision,
    last_good: Option<CompiledTerrainPlan>,
    pending_work: TerrainPlanWork,
    stats: PlanCacheStats,
}

impl TerrainPlanCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn structure_revision(&self) -> PlanStructureRevision {
        self.structure_revision
    }

    pub const fn pending_work(&self) -> TerrainPlanWork {
        self.pending_work
    }

    pub fn take_pending_work(&mut self) -> TerrainPlanWork {
        std::mem::take(&mut self.pending_work)
    }

    /// Record one action batch. Any number of structural edits in this batch
    /// advances the structural generation exactly once and therefore causes at
    /// most one compile on the next acquisition.
    pub fn note_edits<'a>(
        &mut self,
        edits: impl IntoIterator<Item = &'a TerrainEditClass>,
    ) -> TerrainPlanWork {
        let mut batch = TerrainPlanWork::NONE;
        for edit in edits {
            batch = batch.merge(edit.required_work());
        }
        if batch.compile_structure {
            self.structure_revision = self.structure_revision.next();
        }
        self.pending_work = self.pending_work.merge(batch);
        batch
    }

    pub fn note_edit(&mut self, edit: &TerrainEditClass) -> TerrainPlanWork {
        self.note_edits(std::iter::once(edit))
    }

    /// Return a plan compatible with the current authored revision, compiling a
    /// candidate only when the last-good plan is absent or stale.
    pub fn acquire(
        &mut self,
        stack: &LayerStack,
        mask_assets: &[MaskAsset],
    ) -> Result<&CompiledTerrainPlan, Vec<TerrainPlanDiagnostic>> {
        if self
            .last_good
            .as_ref()
            .is_some_and(|plan| plan.matches_structure_revision(self.structure_revision))
        {
            self.stats.snapshot.plan_cache_hits =
                self.stats.snapshot.plan_cache_hits.saturating_add(1);
            return Ok(self.last_good.as_ref().expect("checked above"));
        }

        self.stats.snapshot.plan_compiles = self.stats.snapshot.plan_compiles.saturating_add(1);
        let candidate = compile_terrain_plan(
            stack,
            mask_assets,
            TerrainPlanStamp::new(self.structure_revision),
        )?;
        self.stats.snapshot.successful_compiles =
            self.stats.snapshot.successful_compiles.saturating_add(1);
        self.last_good = Some(candidate);
        Ok(self.last_good.as_ref().expect("candidate installed"))
    }

    /// Acquire only if the retained plan is compatible with the current
    /// authored structure. Backends call this gate before mutating resources.
    pub fn current_plan(&self) -> Result<&CompiledTerrainPlan, PlanRevisionError> {
        let plan = self.last_good.as_ref().ok_or(PlanRevisionError {
            plan_revision: u64::MAX,
            current_revision: self.structure_revision.get(),
        })?;
        if plan.matches_structure_revision(self.structure_revision) {
            Ok(plan)
        } else {
            Err(PlanRevisionError {
                plan_revision: plan.stamp().structure_revision.get(),
                current_revision: self.structure_revision.get(),
            })
        }
    }

    /// Last successfully compiled plan, including a stale one retained solely
    /// so presentation can keep the previous backend output alive.
    pub const fn last_good_plan(&self) -> Option<&CompiledTerrainPlan> {
        self.last_good.as_ref()
    }

    pub const fn stats(&self) -> &PlanCacheStats {
        &self.stats
    }

    pub fn stats_mut(&mut self) -> &mut PlanCacheStats {
        &mut self.stats
    }

    pub fn record_invalidation(&mut self, invalidation: &PlanInvalidation) {
        self.stats.snapshot.patched_operations = self
            .stats
            .snapshot
            .patched_operations
            .saturating_add(invalidation.patched_operations.len() as u64);
        self.stats.snapshot.operations_reached = self
            .stats
            .snapshot
            .operations_reached
            .saturating_add(invalidation.operations.len() as u64);
        if invalidation.first_full_field_escalation.is_some() {
            self.stats.snapshot.full_field_escalations =
                self.stats.snapshot.full_field_escalations.saturating_add(1);
        }
    }

    /// Apply one authored edit batch, compile only when its topology changed,
    /// refresh dynamic reach metadata through provenance, and return the exact
    /// invalidation report for the current plan.
    pub fn update(
        &mut self,
        stack: &LayerStack,
        mask_assets: &[MaskAsset],
        edits: &[TerrainEditClass],
    ) -> Result<PlanInvalidation, Vec<TerrainPlanDiagnostic>> {
        let work = self.note_edits(edits);
        self.acquire(stack, mask_assets)?;
        if !work.compile_structure {
            self.refresh_dynamic_metadata(stack, mask_assets, edits);
        }
        let invalidation = propagate_plan_edits(
            self.current_plan()
                .expect("successful acquisition installs the current revision"),
            edits,
        );
        self.record_invalidation(&invalidation);
        self.pending_work = TerrainPlanWork::NONE;
        Ok(invalidation)
    }

    fn refresh_dynamic_metadata(
        &mut self,
        stack: &LayerStack,
        mask_assets: &[MaskAsset],
        edits: &[TerrainEditClass],
    ) {
        let Some(plan) = self.last_good.as_mut() else {
            return;
        };
        let mut operations = Vec::new();
        for edit in edits {
            let owner = match edit {
                TerrainEditClass::Content { owner, .. }
                | TerrainEditClass::Parameters { owner } => Some(*owner),
                _ => None,
            };
            let Some(owner) = owner else {
                continue;
            };
            for operation in plan
                .provenance()
                .operations_for(owner)
                .iter()
                .chain(plan.provenance().consumers_for(owner))
            {
                if !operations.contains(operation) {
                    operations.push(*operation);
                }
            }
        }

        for operation_id in operations {
            let origin = plan
                .operation(operation_id)
                .and_then(|operation| operation.origin.authored());
            let Some(operation) = plan.operation_mut(operation_id) else {
                continue;
            };
            match &operation.kind {
                TerrainOpKind::RunLayerKernel { layer, .. } => {
                    if let Some(authored) = stack.find(*layer) {
                        operation.reach = if authored.common.param_bindings.is_empty() {
                            authored.kind.intrinsic_reach()
                        } else {
                            Reach::Full
                        };
                        operation.aux_reach = authored.kind.aux_reach();
                    }
                }
                TerrainOpKind::EvaluateMask { .. } => {
                    let distribution = match origin {
                        Some(NodeRef::Layer(layer)) => {
                            stack.find(layer).map(|layer| &layer.common.masks)
                        }
                        Some(NodeRef::Group(group)) => {
                            stack.find_group(group).map(|group| &group.masks)
                        }
                        _ => None,
                    };
                    if let Some(distribution) = distribution {
                        operation.reach =
                            crate::mask::distribution_reach(distribution, mask_assets);
                    }
                }
                TerrainOpKind::CompositeGroup { aux, .. } => {
                    operation.aux_reach = if aux.is_empty() {
                        AuxReach::HeightOnly
                    } else {
                        AuxReach::PerTexel
                    };
                }
                _ => {}
            }
        }
    }
}
