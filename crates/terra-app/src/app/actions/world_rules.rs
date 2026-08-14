use crate::ui::PanelAction;

use super::super::TerraApp;
use super::ApplyCtx;

pub(crate) fn try_apply(
    app: &mut TerraApp,
    action: PanelAction,
    ctx: &mut ApplyCtx,
) -> Result<(), PanelAction> {
    let result =
        match action {
            PanelAction::AddWorldRule { name } => {
                let rule = terra_core::world_rules::WorldRule::new(name);
                let index = app.session.document.world_rules.rules.len();
                app.session.push_world_rule_command(
                    terra_core::world_rules::WorldRuleCommand::Add { rule, index },
                );
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
                    .unwrap_or_else(|| app.session.document.world_rules.rules.len());
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
    let _ = result;
    Ok(())
}
