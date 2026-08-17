use crate::ui::PanelAction;

use super::super::TerraApp;
use super::ApplyCtx;

// Returns the unhandled action on `Err` so the next handler in the chain can try it
// (see actions/mod.rs); that payload is the intrinsic-size `PanelAction` (`LayerKind`).
#[allow(clippy::result_large_err)]
pub(crate) fn try_apply(
    app: &mut TerraApp,
    action: PanelAction,
    ctx: &mut ApplyCtx,
) -> Result<(), PanelAction> {
    match action {
        PanelAction::AddWorldRule { name } => {
            let rule = terra_core::world_rules::WorldRule::new(name);
            let index = app.session.document.world_rules.rules.len();
            app.session
                .push_world_rule_command(terra_core::world_rules::WorldRuleCommand::Add {
                    rule,
                    index,
                });
            app.request_rebuild();
            ctx.doc_mutated = true;
        }
        PanelAction::AddWorldRulePreset { name } => {
            if let Some(rule) = terra_core::world_rules::world_rule_preset_by_name(&name) {
                let index = app.session.document.world_rules.rules.len();
                app.session.push_world_rule_command(
                    terra_core::world_rules::WorldRuleCommand::Add { rule, index },
                );
                app.ui_state.status = format!("Added World Rule preset {name}");
                app.request_rebuild();
                ctx.doc_mutated = true;
            } else {
                app.ui_state.status = format!("Unknown World Rule preset: {name}");
            }
        }
        PanelAction::SelectWorldRule(id) => {
            app.session.document.world_rules.selected = Some(id);
            ctx.doc_mutated = true;
        }
        PanelAction::SetWorldRuleEnabled { id, enabled } => {
            let previous = app
                .session
                .document
                .world_rules
                .get(id)
                .map(|r| r.enabled)
                .unwrap_or(true);
            app.session.push_world_rule_command(
                terra_core::world_rules::WorldRuleCommand::SetEnabled {
                    id,
                    enabled,
                    previous,
                },
            );
            if let Some(stage) = app
                .session
                .document
                .world_rules
                .invalidation_stage_for(&[id])
            {
                let preview = app.session.document.preview_eval_stack();
                app.scheduler
                    .evaluator
                    .mark_dirty_from_eval_stage(&preview, stage);
                app.track_worker_dirty_from_eval_stage(&preview, stage);
            }
            app.request_rebuild();
            ctx.doc_mutated = true;
        }
        PanelAction::RemoveWorldRule(id) => {
            if let Some((index, rule)) = app
                .session
                .document
                .world_rules
                .rules
                .iter()
                .enumerate()
                .find(|(_, r)| r.id == id)
                .map(|(i, r)| (i, r.clone()))
            {
                app.session.push_world_rule_command(
                    terra_core::world_rules::WorldRuleCommand::Remove { rule, index },
                );
                app.request_rebuild();
                ctx.doc_mutated = true;
            }
        }
        PanelAction::ReorderWorldRule { id, before } => {
            let previous_priorities: Vec<_> = app
                .session
                .document
                .world_rules
                .rules
                .iter()
                .map(|r| (r.id, r.priority))
                .collect();
            let from = app
                .session
                .document
                .world_rules
                .rules
                .iter()
                .position(|r| r.id == id);
            let to = before
                .and_then(|b| {
                    app.session
                        .document
                        .world_rules
                        .rules
                        .iter()
                        .position(|r| r.id == b)
                })
                .unwrap_or(app.session.document.world_rules.rules.len());
            if let Some(from) = from {
                app.session.push_world_rule_command(
                    terra_core::world_rules::WorldRuleCommand::Reorder {
                        id,
                        from,
                        to,
                        previous_priorities,
                    },
                );
                ctx.doc_mutated = true;
            }
        }
        PanelAction::SetWorldRuleScope { id, scope } => {
            if let Some(before) = app.session.document.world_rules.get(id).cloned() {
                let mut after = before.clone();
                after.scope = scope;
                app.session.push_world_rule_command(
                    terra_core::world_rules::WorldRuleCommand::Replace { id, before, after },
                );
                app.request_rebuild();
                ctx.doc_mutated = true;
            }
        }
        PanelAction::SetWorldRulePhase { id, phase } => {
            if let Some(before) = app.session.document.world_rules.get(id).cloned() {
                let mut after = before.clone();
                after.phase_override = phase;
                app.session.push_world_rule_command(
                    terra_core::world_rules::WorldRuleCommand::Replace { id, before, after },
                );
                if let Some(stage) = app
                    .session
                    .document
                    .world_rules
                    .invalidation_stage_for(&[id])
                {
                    let preview = app.session.document.preview_eval_stack();
                    app.scheduler
                        .evaluator
                        .mark_dirty_from_eval_stage(&preview, stage);
                    app.track_worker_dirty_from_eval_stage(&preview, stage);
                }
                app.request_rebuild();
                ctx.doc_mutated = true;
            }
        }
        PanelAction::ToggleWorldRuleEffect {
            rule,
            effect,
            enabled,
        } => {
            if let Some(before) = app.session.document.world_rules.get(rule).cloned() {
                let mut after = before.clone();
                if let Some(fx) = after
                    .effects
                    .iter_mut()
                    .find(|e| e.id.to_string() == effect)
                {
                    fx.enabled = enabled;
                }
                app.session.push_world_rule_command(
                    terra_core::world_rules::WorldRuleCommand::Replace {
                        id: rule,
                        before,
                        after,
                    },
                );
                app.request_rebuild();
                ctx.doc_mutated = true;
            }
        }
        other => return Err(other),
    };
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use terra_core::eval::CachedOutput;
    use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
    use terra_core::layer::{FlatParams, Layer, LayerId, LayerKind, LayerStack, MaterialsParams};
    use terra_core::world_rules::{WorldRule, WorldRuleId, WorldRulePhase};

    /// A headless app whose stack is `Base(Flat, Blueprint) → Mats(Materials)` with a
    /// fresh World Rule. A rule with no effects and no phase override resolves to the
    /// `Materials` phase → `EvalStage::Materials`, so a stage dirty from that rule
    /// invalidates the Materials layer (order 6) but not the Flat base (order 0).
    /// Both layers start clean in the UI-thread evaluator and the worker accumulators
    /// start drained (the steady state right after a completed run), so any dirtying
    /// we observe is caused solely by the action under test.
    fn app_with_world_rule() -> (super::super::TerraApp, LayerId, WorldRuleId) {
        let mut app = super::super::TerraApp::default();

        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "Base",
            LayerKind::Flat(FlatParams { height: 10.0 }),
        ));
        let mats = Layer::new("Mats", LayerKind::Materials(MaterialsParams::default()));
        let mats_id = mats.id();
        stack.push(mats);
        app.session.document.stack = stack;

        let rule = WorldRule::new("Test Rule");
        let rule_id = rule.id;
        app.session.document.world_rules.rules.push(rule);

        // Seed both layers clean in the UI-thread evaluator cache.
        let metrics = HeightfieldMetrics::new(4, 4, 4.0, 4.0);
        for id in app.session.document.stack.layer_ids() {
            app.scheduler.evaluator.cache.insert(
                id,
                CachedOutput {
                    height: Heightfield::zeros(metrics),
                    generation: 0,
                    dirty: false,
                    aux: HashMap::new(),
                    strata: None,
                },
            );
        }
        // Steady state: the previous run's dirt was drained into a completed job.
        app.worker_mark_all_dirty = false;
        app.worker_dirty_from = None;

        (app, mats_id, rule_id)
    }

    /// The worker recomputes `layer` only if it is told to: either the whole field is
    /// dirty, or the suffix-dirty layer is at or below it in the stack.
    fn worker_would_recompute(app: &super::super::TerraApp, layer: LayerId) -> bool {
        if app.worker_mark_all_dirty {
            return true;
        }
        let ids = app.session.document.stack.layer_ids();
        match app.worker_dirty_from {
            Some(from) => {
                let from_idx = ids.iter().position(|x| *x == from);
                let layer_idx = ids.iter().position(|x| *x == layer);
                matches!((from_idx, layer_idx), (Some(f), Some(h)) if f <= h)
            }
            None => false,
        }
    }

    /// Toggling a rule's enabled flag dirties the UI-thread evaluator at the rule's
    /// invalidation stage; it must also dirty the worker so the (worker-authoritative)
    /// affected layers actually recompute instead of reusing a stale checkpoint.
    #[test]
    fn toggle_world_rule_enabled_reaches_worker() {
        let (mut app, mats_id, rule_id) = app_with_world_rule();
        app.apply_actions(vec![PanelAction::SetWorldRuleEnabled {
            id: rule_id,
            enabled: false,
        }]);

        assert!(
            app.scheduler.evaluator.cache.is_dirty(mats_id),
            "precondition: toggling dirties the UI-thread evaluator at the Materials stage"
        );
        assert!(
            worker_would_recompute(&app, mats_id),
            "toggling a World Rule must dirty the worker for the affected layers"
        );
    }

    /// Changing a rule's phase override takes the same stage-dirty path.
    #[test]
    fn set_world_rule_phase_reaches_worker() {
        let (mut app, mats_id, rule_id) = app_with_world_rule();
        app.apply_actions(vec![PanelAction::SetWorldRulePhase {
            id: rule_id,
            phase: Some(WorldRulePhase::Materials),
        }]);

        assert!(
            app.scheduler.evaluator.cache.is_dirty(mats_id),
            "precondition: a phase change dirties the UI-thread evaluator at the Materials stage"
        );
        assert!(
            worker_would_recompute(&app, mats_id),
            "changing a World Rule's phase must dirty the worker for the affected layers"
        );
    }
}
