//! Artist-facing rebuild / dependency feedback (document-aware behavior).
//!
//! Builds on [`crate::deps::DependencyGraph`] — never invents a parallel UI dependency model.
//! Simulation results become **Outdated** (snapshots kept) rather than being discarded.
//!
//! The data-only state and DTOs live in [`crate::rebuild_state`]; this module owns the
//! behavior that reads and mutates the [`EditorSession`] / [`TerrainDocument`]. Those types
//! are re-exported below so `rebuild_feedback::…` stays a stable public path.

use crate::deps::{DependencyGraph, NodeRef};
use crate::document::{EditorSession, TerrainDocument};
use crate::domain::SoftDiagnostic;
use crate::layer::{BuildStatus, CachePolicy, LayerId, LayerKind, StackNode};
use crate::simulation_scenario::ScenarioResultState;
use std::collections::HashSet;

pub use crate::rebuild_state::{
    AffectedContentFeedback, AffectedId, AffectedItem, ArtistBuildState, BuildStateCategory,
    RebuildFeedbackAction, RebuildFeedbackState, RebuildPrefs, WhyRebuild,
};

// ---------------------------------------------------------------------------
// Classification helpers
// ---------------------------------------------------------------------------

pub fn is_expensive_physics(kind: &LayerKind) -> bool {
    matches!(
        kind,
        LayerKind::HydraulicErosion(_)
            | LayerKind::ThermalErosion(_)
            | LayerKind::StreamPowerErosion(_)
            | LayerKind::LandscapeEvolution(_)
            | LayerKind::HydrologyRepair(_)
            | LayerKind::GeomorphicDetail(_)
            | LayerKind::EcosystemFeedback(_)
            | LayerKind::SandSimulation(_)
            | LayerKind::FluidSimulation(_)
            | LayerKind::MultiScaleAmplify(_)
    )
}

/// Artist-facing label for a layer (Biome placement, Hydrology, Materials, …).
pub fn artist_layer_label(doc: &TerrainDocument, id: LayerId) -> String {
    let layer = doc.stack.find(id);
    let Some(layer) = layer else {
        return "Layer".into();
    };
    let base = layer.common.name.clone();
    let kind_hint = match &layer.kind {
        LayerKind::HydraulicErosion(_)
        | LayerKind::HydrologyRepair(_)
        | LayerKind::RiverCarve(_)
        | LayerKind::RiverNetwork(_)
        | LayerKind::FluidSimulation(_) => Some("Hydrology"),
        LayerKind::Materials(_) => Some("Materials"),
        LayerKind::Vegetation(_) => Some("Vegetation"),
        LayerKind::ThermalErosion(_) | LayerKind::GeomorphicDetail(_) => Some("Terrain"),
        LayerKind::SandSimulation(_) => Some("Sand"),
        _ => None,
    };
    if let Some(biome) = doc.stack.enclosing_biome(id) {
        if let Some(hint) = kind_hint {
            return format!("{} {}", biome.name, hint.to_ascii_lowercase());
        }
        if biome.is_biome() {
            return format!("{} — {}", biome.name, base);
        }
    }
    kind_hint.map(|h| h.to_string()).unwrap_or(base)
}

pub fn resolve_node_name(doc: &TerrainDocument, node: NodeRef) -> String {
    match node {
        NodeRef::Layer(id) => artist_layer_label(doc, id),
        NodeRef::Group(id) => doc
            .stack
            .find_group(id)
            .map(|g| {
                if g.is_biome() {
                    format!("{} placement", g.name)
                } else {
                    g.name.clone()
                }
            })
            .unwrap_or_else(|| "Group".into()),
        NodeRef::Mask(id) => doc
            .masks
            .iter()
            .find(|m| m.id == id)
            .map(|m| m.name.clone())
            .unwrap_or_else(|| "Mask".into()),
        NodeRef::Output(_) => "Output".into(),
    }
}

pub fn build_status_for_layer(
    session: &EditorSession,
    id: LayerId,
    enabled: bool,
    computing: bool,
) -> BuildStatus {
    if !enabled {
        return BuildStatus::Idle;
    }
    if computing {
        return BuildStatus::Computing;
    }
    if session.outdated_sim_layers.contains(&id)
        || session
            .rebuild_feedback
            .queued
            .iter()
            .any(|a| matches!(a, AffectedId::Layer(x) if *x == id))
    {
        if session
            .rebuild_feedback
            .building
            .iter()
            .any(|a| matches!(a, AffectedId::Layer(x) if *x == id))
        {
            return BuildStatus::Computing;
        }
        return BuildStatus::Outdated;
    }
    BuildStatus::Ready
}

pub fn artist_state_for_layer(
    session: &EditorSession,
    id: LayerId,
    enabled: bool,
    policy: CachePolicy,
    computing: bool,
) -> ArtistBuildState {
    if !enabled {
        return ArtistBuildState::Disabled;
    }
    if matches!(policy, CachePolicy::Frozen) {
        return ArtistBuildState::Frozen;
    }
    if computing
        || session
            .rebuild_feedback
            .building
            .iter()
            .any(|a| matches!(a, AffectedId::Layer(x) if *x == id))
    {
        return ArtistBuildState::Building;
    }
    if session
        .rebuild_feedback
        .queued
        .iter()
        .any(|a| matches!(a, AffectedId::Layer(x) if *x == id))
    {
        return ArtistBuildState::Queued;
    }
    if session.outdated_sim_layers.contains(&id) {
        return ArtistBuildState::Outdated;
    }
    ArtistBuildState::Current
}

// ---------------------------------------------------------------------------
// Graph-driven invalidation
// ---------------------------------------------------------------------------

/// Collect compact affected items downstream of `source` using the real dependency graph.
pub fn collect_affected(
    doc: &TerrainDocument,
    graph: &DependencyGraph,
    source: NodeRef,
) -> Vec<AffectedItem> {
    let deps = graph.dependents_of(source);
    let mut items = Vec::new();
    let mut seen_names = HashSet::new();

    for node in &deps {
        let name = resolve_node_name(doc, *node);
        // Deduplicate by artist label for compact “Updating:” lists.
        if !seen_names.insert(name.clone()) {
            continue;
        }
        let (id, state) = match node {
            NodeRef::Layer(id) => (AffectedId::Layer(*id), ArtistBuildState::Outdated),
            NodeRef::Group(id) => (AffectedId::Group(*id), ArtistBuildState::Invalidated),
            _ => continue,
        };
        items.push(AffectedItem { id, name, state });
    }

    // Scenarios / world rules are not NodeRefs yet — append from libraries when
    // any expensive physics layer was affected.
    let physics_hit = deps.iter().any(|n| {
        if let NodeRef::Layer(id) = n {
            doc.stack
                .find(*id)
                .is_some_and(|l| is_expensive_physics(&l.kind))
        } else {
            false
        }
    });
    if physics_hit || matches!(source, NodeRef::Layer(_)) {
        for sc in &doc.simulation_scenarios.scenarios {
            if sc.result_state == ScenarioResultState::Frozen {
                items.push(AffectedItem {
                    id: AffectedId::Scenario(sc.id),
                    name: format!("{} (frozen)", sc.name),
                    state: ArtistBuildState::Frozen,
                });
            } else {
                items.push(AffectedItem {
                    id: AffectedId::Scenario(sc.id),
                    name: sc.name.clone(),
                    state: ArtistBuildState::Outdated,
                });
            }
        }
        for rule in &doc.world_rules.rules {
            if !rule.enabled {
                continue;
            }
            items.push(AffectedItem {
                id: AffectedId::WorldRule(rule.id),
                name: rule.name.clone(),
                state: ArtistBuildState::Invalidated,
            });
        }
    }

    // Cap compact list; keep order stable.
    if items.len() > 12 {
        items.truncate(12);
    }
    items
}

/// Explain why a node is outdated / needs rebuild — uses the dependency graph only.
pub fn why_outdated(
    doc: &TerrainDocument,
    graph: &DependencyGraph,
    target: NodeRef,
    outdated_layers: &[LayerId],
    source_hint: Option<&str>,
) -> WhyRebuild {
    let name = resolve_node_name(doc, target);
    let mut diagnostics = Vec::new();
    let depends: Vec<_> = graph
        .direct_dependencies(target)
        .into_iter()
        .map(|n| (resolve_node_name(doc, n), n))
        .collect();
    let used_by: Vec<_> = graph
        .direct_dependents(target)
        .into_iter()
        .map(|n| (resolve_node_name(doc, n), n))
        .collect();

    if let Some(hint) = source_hint {
        diagnostics.push(SoftDiagnostic::new(
            "outdated_due_to_upstream",
            format!("{name} is outdated because {hint} changed"),
        ));
    }

    if let NodeRef::Layer(id) = target {
        if outdated_layers.contains(&id) {
            diagnostics.push(SoftDiagnostic::new(
                "outdated_sim_result",
                format!(
                    "{name} cached result is outdated — upstream shape/geometry changed (snapshot kept)"
                ),
            ));
        }
        if let Some(layer) = doc.stack.find(id) {
            if is_expensive_physics(&layer.kind) {
                diagnostics.push(SoftDiagnostic::new(
                    "waiting_manual_rebuild",
                    format!(
                        "{name} is expensive physics — not auto-rebuilt while sculpting unless Live / automatic rebuild is enabled"
                    ),
                ));
            }
            if matches!(
                layer.common.cache_policy.unwrap_or_default(),
                CachePolicy::Frozen
            ) {
                diagnostics.push(SoftDiagnostic::new(
                    "frozen_result",
                    format!("{name} is frozen — showing previous result"),
                ));
            }
        }
    }

    if diagnostics.is_empty() {
        diagnostics.push(SoftDiagnostic::new(
            "why_rebuild",
            format!("Why did {name} update? Upstream dependency change in the project graph."),
        ));
    }

    WhyRebuild {
        title: format!("Why is {name} outdated?"),
        diagnostics,
        depends_on: depends,
        used_by,
    }
}

/// Apply an upstream edit: mark sims Outdated (keep snapshots / frozen), record feedback.
/// Does **not** schedule expensive physics rebuilds unless prefs allow.
pub fn apply_upstream_change(
    session: &mut EditorSession,
    source: NodeRef,
    reason: impl Into<String>,
    now_ms: u64,
) -> AffectedContentFeedback {
    let reason = reason.into();
    let graph = session.document.dependency_graph();
    let source_name = resolve_node_name(&session.document, source);
    let mut items = collect_affected(&session.document, &graph, source);

    // Mark simulation layers outdated (graph dependents + stack-order fallback).
    session.outdated_sim_layers.clear();
    for item in &items {
        if let AffectedId::Layer(id) = item.id {
            let is_sim =
                find_layer(&session.document, id).is_some_and(|l| is_expensive_physics(&l.kind));
            if is_sim {
                session.outdated_sim_layers.push(id);
            }
        }
    }
    // Fallback: stack order after a shape layer (when graph edges are sparse).
    if let NodeRef::Layer(shape_id) = source {
        let preview = session.document.preview_eval_stack();
        for id in collect_physics_after(&preview.nodes, shape_id) {
            if !session.outdated_sim_layers.contains(&id) {
                session.outdated_sim_layers.push(id);
                let name = artist_layer_label(&session.document, id);
                if !items.iter().any(|i| i.name == name) {
                    items.push(AffectedItem::layer(id, name, ArtistBuildState::Outdated));
                }
            }
        }
    }

    // Scenarios → Outdated (Frozen preserved inside mark_outdated).
    session.document.simulation_scenarios.mark_all_outdated();

    // Refresh scenario rows in feedback after mark.
    for item in &mut items {
        if let AffectedId::Scenario(sid) = item.id {
            if let Some(sc) = session.document.simulation_scenarios.get(sid) {
                item.state = ArtistBuildState::from(sc.result_state);
                if sc.result_state == ScenarioResultState::Frozen {
                    item.name = format!("{} (frozen)", sc.name);
                }
            }
        }
    }

    let feedback = AffectedContentFeedback {
        source_name: source_name.clone(),
        items,
        why: reason.clone(),
    };

    session.rebuild_feedback.last_feedback = Some(feedback.clone());
    session.rebuild_feedback.last_why = Some(why_outdated(
        &session.document,
        &graph,
        source,
        &session.outdated_sim_layers,
        Some(&reason),
    ));
    session.rebuild_feedback.record_edit(now_ms);

    // Queue expensive rebuilds only when automatic + Live policy.
    if session.rebuild_feedback.prefs.automatic_rebuild_expensive {
        for id in &session.outdated_sim_layers {
            let live = session.document.stack.find(*id).is_some_and(|l| {
                matches!(
                    l.common.cache_policy.unwrap_or(CachePolicy::Manual),
                    CachePolicy::Live
                )
            });
            if live {
                let aid = AffectedId::Layer(*id);
                if !session.rebuild_feedback.queued.contains(&aid) {
                    session.rebuild_feedback.queued.push(aid);
                }
            }
        }
    }

    feedback
}

/// Selective rebuild of outdated sims / queued items (artist-triggered).
pub fn rebuild_affected(session: &mut EditorSession) -> Vec<LayerId> {
    let mut ids = session.outdated_sim_layers.clone();
    for q in &session.rebuild_feedback.queued {
        if let AffectedId::Layer(id) = q {
            if !ids.contains(id) {
                ids.push(*id);
            }
        }
    }
    session.rebuild_feedback.building = ids.iter().map(|id| AffectedId::Layer(*id)).collect();
    session.rebuild_feedback.queued.clear();
    session.outdated_sim_layers.clear();
    session.rebuild_feedback.clear_physics_due();
    ids
}

/// Apply a feedback action; returns soft status message.
pub fn apply_feedback_action(session: &mut EditorSession, action: RebuildFeedbackAction) -> String {
    match action {
        RebuildFeedbackAction::DisableAutomaticRebuild => {
            session.rebuild_feedback.prefs.automatic_rebuild_expensive = false;
            session.rebuild_feedback.clear_physics_due();
            "Automatic physics rebuild disabled".into()
        }
        RebuildFeedbackAction::EnableAutomaticRebuild => {
            session.rebuild_feedback.prefs.automatic_rebuild_expensive = true;
            "Automatic physics rebuild enabled".into()
        }
        RebuildFeedbackAction::EnableLivePreview => {
            session.rebuild_feedback.prefs.live_preview = true;
            "Live preview enabled (draft quality while editing)".into()
        }
        RebuildFeedbackAction::DisableLivePreview => {
            session.rebuild_feedback.prefs.live_preview = false;
            "Live preview disabled".into()
        }
        RebuildFeedbackAction::KeepFrozen { scenario } => {
            if session
                .document
                .simulation_scenarios
                .get_mut(scenario)
                .is_some_and(|s| s.freeze_result(None))
            {
                "Result frozen — will not be discarded".into()
            } else {
                "No snapshot to freeze".into()
            }
        }
        RebuildFeedbackAction::PreviewOldResult { scenario, snapshot } => {
            if session
                .document
                .simulation_scenarios
                .get_mut(scenario)
                .is_some_and(|s| s.preview_old_result(snapshot))
            {
                "Previewing old simulation result".into()
            } else {
                "Snapshot not found".into()
            }
        }
        RebuildFeedbackAction::Rebuild { id } => match id {
            AffectedId::Layer(lid) => {
                session.outdated_sim_layers.retain(|x| *x != lid);
                session
                    .rebuild_feedback
                    .building
                    .push(AffectedId::Layer(lid));
                format!("Rebuilding {}", artist_layer_label(&session.document, lid))
            }
            AffectedId::Scenario(sid) => {
                if let Some(s) = session.document.simulation_scenarios.get_mut(sid) {
                    s.rebuild();
                    format!("Rebuilding scenario {}", s.name)
                } else {
                    "Scenario missing".into()
                }
            }
            _ => "Rebuild requested".into(),
        },
        RebuildFeedbackAction::RebuildAffected { source } => {
            let _ = source;
            let n = rebuild_affected(session).len();
            format!("Rebuilding {n} affected item(s)")
        }
    }
}

/// Detect redundant rebuild requests (same set already building).
pub fn is_redundant_rebuild(session: &EditorSession, ids: &[LayerId]) -> bool {
    if ids.is_empty() {
        return true;
    }
    ids.iter().all(|id| {
        session
            .rebuild_feedback
            .building
            .iter()
            .any(|a| matches!(a, AffectedId::Layer(x) if x == id))
    })
}

fn find_layer(doc: &TerrainDocument, id: LayerId) -> Option<&crate::layer::Layer> {
    doc.stack.find(id)
}

/// Walk stack nodes collecting expensive physics layer ids after `after_id` (legacy fallback).
pub fn collect_physics_after(stack_nodes: &[StackNode], after_id: LayerId) -> Vec<LayerId> {
    let mut out = Vec::new();
    let mut after = false;
    fn walk(nodes: &[StackNode], after: &mut bool, after_id: LayerId, out: &mut Vec<LayerId>) {
        for n in nodes {
            match n {
                StackNode::Layer(l) => {
                    if l.id() == after_id {
                        *after = true;
                        continue;
                    }
                    if *after && is_expensive_physics(&l.kind) {
                        out.push(l.id());
                    }
                }
                StackNode::Group(g) => walk(&g.children, after, after_id, out),
            }
        }
    }
    walk(stack_nodes, &mut after, after_id, &mut out);
    out
}
