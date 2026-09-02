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
            app.track_worker_dirty_from_eval_stage(&preview, terra_core::EvalStage::SharedHydro);
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
            app.track_worker_dirty_from_eval_stage(&preview, terra_core::EvalStage::SharedHydro);
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
                app.track_worker_dirty_from_eval_stage(
                    &preview,
                    terra_core::EvalStage::SharedHydro,
                );
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
    use terra_core::layer::{HydraulicErosionParams, Layer, LayerId, LayerKind, LayerStack};
    use terra_core::simulation_scenario::{
        ScenarioPassKind, SimulationScenario, SimulationScenarioId,
    };
    use terra_cpu_eval::CachedOutput;

    /// A headless app whose stack is `Base(Flat) → Hydro(HydraulicErosion)` with a
    /// scenario whose single pass is bound to the Hydro layer. Both layers start
    /// clean in the UI-thread evaluator cache and the worker accumulators start
    /// drained (the steady state right after a completed run), so any dirtying we
    /// observe is caused solely by the action under test.
    fn app_with_bound_scenario() -> (super::TerraApp, LayerId, SimulationScenarioId) {
        let mut app = super::TerraApp::default();

        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "Base",
            LayerKind::Flat(terra_core::layer::FlatParams { height: 10.0 }),
        ));
        let hydro = Layer::new(
            "Hydro",
            LayerKind::HydraulicErosion(HydraulicErosionParams::default()),
        );
        let hydro_id = hydro.id();
        stack.push(hydro);
        app.session.document.stack = stack;

        let mut scenario = SimulationScenario::new("Test");
        let pass_id = scenario.add_pass(ScenarioPassKind::HydraulicErosion);
        assert!(scenario.bind_pass_layer(pass_id, hydro_id));
        let scen_id = scenario.id;
        app.session
            .document
            .simulation_scenarios
            .scenarios
            .push(scenario);

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

        (app, hydro_id, scen_id)
    }

    /// The worker recomputes the Hydro layer only if it is told to: either the
    /// whole field is dirty, or the suffix-dirty layer is at or below Hydro.
    fn worker_would_recompute(app: &super::TerraApp, hydro_id: LayerId) -> bool {
        if app.worker_mark_all_dirty {
            return true;
        }
        let ids = app.session.document.stack.layer_ids();
        match app.worker_dirty_from {
            Some(from) => {
                let from_idx = ids.iter().position(|x| *x == from);
                let hydro_idx = ids.iter().position(|x| *x == hydro_id);
                matches!((from_idx, hydro_idx), (Some(f), Some(h)) if f <= h)
            }
            None => false,
        }
    }

    /// Baseline: Run sets `ctx.dirty_from` to the bound layer, so the worker is told
    /// to recompute it. (Present so a regression that drops this is caught here too.)
    #[test]
    fn run_scenario_reaches_worker_for_bound_layer() {
        let (mut app, hydro_id, scen_id) = app_with_bound_scenario();
        app.apply_actions(vec![PanelAction::RunSimulationScenario(scen_id)]);
        assert!(
            worker_would_recompute(&app, hydro_id),
            "Run must dirty the worker for the bound sim layer"
        );
    }

    /// Rebuild dirties the UI-thread evaluator (SharedHydro stage) but leaves the
    /// worker accumulators untouched, so the worker reuses its stale clean checkpoint
    /// for the sim layer — the gap under investigation.
    #[test]
    fn rebuild_scenario_reaches_worker_for_bound_layer() {
        let (mut app, hydro_id, scen_id) = app_with_bound_scenario();
        app.apply_actions(vec![PanelAction::RebuildSimulationScenario(scen_id)]);

        assert!(
            app.scheduler.evaluator.cache.is_dirty(hydro_id),
            "precondition: Rebuild dirties the UI-thread evaluator"
        );
        assert!(
            worker_would_recompute(&app, hydro_id),
            "Rebuild must dirty the worker for the bound sim layer, not just the UI evaluator"
        );
    }

    /// Apply Outputs takes the same UI-only path as Rebuild.
    #[test]
    fn apply_outputs_reaches_worker_for_bound_layer() {
        let (mut app, hydro_id, scen_id) = app_with_bound_scenario();
        // A current snapshot must exist for apply_selected_outputs to act on.
        if let Some(s) = app.session.document.simulation_scenarios.get_mut(scen_id) {
            s.begin_run();
            let _ = s.complete_run(1);
        }
        app.apply_actions(vec![PanelAction::ApplyScenarioOutputs {
            id: scen_id,
            snapshot: None,
        }]);

        assert!(
            app.scheduler.evaluator.cache.is_dirty(hydro_id),
            "precondition: Apply Outputs dirties the UI-thread evaluator"
        );
        assert!(
            worker_would_recompute(&app, hydro_id),
            "Apply Outputs must dirty the worker for the bound sim layer"
        );
    }
}
