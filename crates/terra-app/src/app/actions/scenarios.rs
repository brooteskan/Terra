use crate::ui::PanelAction;

use super::super::TerraApp;
use super::ApplyCtx;

pub(crate) fn try_apply(
    app: &mut TerraApp,
    action: PanelAction,
    ctx: &mut ApplyCtx,
) -> Result<(), PanelAction> {
    let result = match action {
        PanelAction::AddSimulationScenario { name } => {
            let scenario = terra_core::simulation_scenario::SimulationScenario::new(name);
            let index = app.session.document.simulation_scenarios.scenarios.len();
            app.session.push_scenario_command(
                terra_core::simulation_scenario::SimulationScenarioCommand::Add { scenario, index },
            );
            ctx.doc_mutated = true;
        }
        PanelAction::AddContinentalHydrologyPreset => {
            let scenario =
                terra_core::simulation_scenario::SimulationScenario::continental_hydrology_preset();
            let index = app.session.document.simulation_scenarios.scenarios.len();
            app.session.push_scenario_command(
                terra_core::simulation_scenario::SimulationScenarioCommand::Add { scenario, index },
            );
            app.ui_state.status = "Added Continental Hydrology scenario".into();
            ctx.doc_mutated = true;
        }
        PanelAction::SelectSimulationScenario(id) => {
            app.session.document.simulation_scenarios.selected = Some(id);
            ctx.doc_mutated = true;
        }
        PanelAction::SetSimulationScenarioEnabled { id, enabled } => {
            let previous = app
                .session
                .document
                .simulation_scenarios
                .get(id)
                .map(|s| s.enabled)
                .unwrap_or(true);
            app.session.push_scenario_command(
                terra_core::simulation_scenario::SimulationScenarioCommand::SetEnabled {
                    id,
                    enabled,
                    previous,
                },
            );
            ctx.doc_mutated = true;
        }
        PanelAction::RunSimulationScenario(id) => {
            // Ensure bound Simulation Layers exist, then dirty SharedHydro.
            app.ensure_scenario_layers(id);
            if let Some(s) = app.session.document.simulation_scenarios.get_mut(id) {
                s.begin_run();
                let layer_ids = s.bound_layer_ids();
                // Synchronous MVP: complete immediately after marking dirty.
                // Real async Pause/Cancel hooks cancel_requested during long runs.
                if s.cancel_requested {
                    s.mark_cancelled();
                } else {
                    let gen = app.scheduler.evaluator.cache.generation;
                    let _ = s.complete_run(gen.wrapping_add(1));
                }
                for lid in layer_ids {
                    ctx.dirty_from = Some(lid);
                    // Clear outdated badge for this layer.
                    app.session.outdated_sim_layers.retain(|x| *x != lid);
                }
            }
            let preview = app.session.document.preview_eval_stack();
            app.scheduler
                .evaluator
                .mark_dirty_from_eval_stage(&preview, terra_core::EvalStage::SharedHydro);
            app.request_rebuild();
            app.ui_state.status = "Running simulation scenario".into();
            ctx.doc_mutated = true;
        }
        PanelAction::CancelSimulationScenario(id) => {
            if let Some(s) = app.session.document.simulation_scenarios.get_mut(id) {
                s.request_cancel();
                s.mark_cancelled();
                app.ui_state.status = "Cancelled simulation scenario".into();
                ctx.doc_mutated = true;
            }
        }
        PanelAction::ResetSimulationScenario(id) => {
            if let Some(before) = app.session.document.simulation_scenarios.get(id).cloned() {
                let mut after = before.clone();
                after.reset();
                app.session.push_scenario_command(
                    terra_core::simulation_scenario::SimulationScenarioCommand::Replace {
                        id,
                        before,
                        after,
                    },
                );
                ctx.doc_mutated = true;
            }
        }
        PanelAction::RebuildSimulationScenario(id) => {
            if let Some(s) = app.session.document.simulation_scenarios.get_mut(id) {
                s.rebuild();
            }
            app.ensure_scenario_layers(id);
            let preview = app.session.document.preview_eval_stack();
            app.scheduler
                .evaluator
                .mark_dirty_from_eval_stage(&preview, terra_core::EvalStage::SharedHydro);
            app.request_rebuild();
            ctx.doc_mutated = true;
        }
        PanelAction::FreezeScenarioResult { id, snapshot } => {
            if let Some(scen) = app.session.document.simulation_scenarios.get_mut(id) {
                let snap_id = snapshot.and_then(|s| {
                    scen.snapshots
                        .iter()
                        .find(|snap| snap.id.to_string() == s)
                        .map(|snap| snap.id)
                });
                if scen.freeze_result(snap_id) {
                    app.ui_state.status = "Froze scenario result".into();
                    ctx.doc_mutated = true;
                }
            }
        }
        PanelAction::PreviewScenarioSnapshot { id, snapshot } => {
            if let Some(scen) = app.session.document.simulation_scenarios.get_mut(id) {
                if let Some(sid) = scen
                    .snapshots
                    .iter()
                    .find(|s| s.id.to_string() == snapshot)
                    .map(|s| s.id)
                {
                    if scen.preview_old_result(sid) {
                        app.ui_state.status = "Previewing old scenario result".into();
                        ctx.doc_mutated = true;
                    }
                }
            }
        }
        PanelAction::CompareScenarioSnapshot { id, snapshot } => {
            if let Some(scen) = app.session.document.simulation_scenarios.get_mut(id) {
                let snap = snapshot.and_then(|s| {
                    scen.snapshots
                        .iter()
                        .find(|snap| snap.id.to_string() == s)
                        .map(|snap| snap.id)
                });
                scen.set_compare(snap);
                app.ui_state.status = "Compare snapshot set".into();
                ctx.doc_mutated = true;
            }
        }
        PanelAction::ApplyScenarioOutputs { id, snapshot } => {
            if let Some(scen) = app.session.document.simulation_scenarios.get_mut(id) {
                let snap = snapshot.and_then(|s| {
                    scen.snapshots
                        .iter()
                        .find(|snap| snap.id.to_string() == s)
                        .map(|snap| snap.id)
                });
                let fields = scen.apply_selected_outputs(snap);
                app.ui_state.status = format!("Applied {} scenario output field(s)", fields.len());
                let preview = app.session.document.preview_eval_stack();
                app.scheduler
                    .evaluator
                    .mark_dirty_from_eval_stage(&preview, terra_core::EvalStage::SharedHydro);
                app.request_rebuild();
                ctx.doc_mutated = true;
            }
        }
        PanelAction::AddScenarioPass { id, kind } => {
            if let Some(before) = app.session.document.simulation_scenarios.get(id).cloned() {
                let mut after = before.clone();
                after.add_pass(kind);
                app.session.push_scenario_command(
                    terra_core::simulation_scenario::SimulationScenarioCommand::Replace {
                        id,
                        before,
                        after,
                    },
                );
                ctx.doc_mutated = true;
            }
        }
        PanelAction::ReorderScenarioPass { id, pass, to_index } => {
            if let Some(before) = app.session.document.simulation_scenarios.get(id).cloned() {
                let mut after = before.clone();
                if let Some(pid) = after
                    .passes
                    .iter()
                    .find(|p| p.id.to_string() == pass)
                    .map(|p| p.id)
                {
                    if after.reorder_pass(pid, to_index) {
                        app.session.push_scenario_command(
                            terra_core::simulation_scenario::SimulationScenarioCommand::Replace {
                                id,
                                before,
                                after,
                            },
                        );
                        ctx.doc_mutated = true;
                    }
                }
            }
        }
        PanelAction::AddMatterSimScenario { matter } => {
            let cfg = terra_core::matter_sim::MatterSimConfig::new(matter);
            let mut scenario = cfg.build_scenario();
            scenario.matter.push(cfg);
            let index = app.session.document.simulation_scenarios.scenarios.len();
            app.session.push_scenario_command(
                terra_core::simulation_scenario::SimulationScenarioCommand::Add { scenario, index },
            );
            app.ui_state.status = format!("Added {} matter scenario", matter.label());
            ctx.doc_mutated = true;
        }
        PanelAction::SetMatterSourcePaint {
            scenario,
            matter_index,
            enabled,
        } => {
            if let Some(before) = app
                .session
                .document
                .simulation_scenarios
                .get(scenario)
                .cloned()
            {
                let mut after = before.clone();
                if let Some(m) = after.matter.get_mut(matter_index) {
                    m.artist.paint_sources = enabled;
                    if enabled {
                        app.ui_state.editor_tool = crate::ui::EditorTool::PaintMask;
                    }
                }
                app.session.push_scenario_command(
                    terra_core::simulation_scenario::SimulationScenarioCommand::Replace {
                        id: scenario,
                        before,
                        after,
                    },
                );
                ctx.doc_mutated = true;
            }
        }
        PanelAction::SetMatterApplySelected {
            scenario,
            matter_index,
            fields,
        } => {
            if let Some(before) = app
                .session
                .document
                .simulation_scenarios
                .get(scenario)
                .cloned()
            {
                let mut after = before.clone();
                if let Some(m) = after.matter.get_mut(matter_index) {
                    m.apply_selected = fields;
                }
                if let Some(m) = after.matter.get(matter_index).cloned() {
                    terra_core::matter_sim::sync_scenario_outputs_from_matter(&mut after, &m);
                }
                app.session.push_scenario_command(
                    terra_core::simulation_scenario::SimulationScenarioCommand::Replace {
                        id: scenario,
                        before,
                        after,
                    },
                );
                ctx.doc_mutated = true;
            }
        }
        other => return Err(other),
    };
    let _ = result;
    Ok(())
}
