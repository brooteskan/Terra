//! World Rules — first-class condition-driven operations.

use std::collections::HashMap;
use terra_core::document::EditorSession;
use terra_core::eval::{EvalContext, PreviewQuality, StackEvaluator};
use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
use terra_core::layer::{Layer, LayerKind, LayerStack, MaterialsParams, MountainParams, StackNode};
use terra_core::mask::{bake_mask_assets, CompareOp, ConditionChannel, DistBakeContext};
use terra_core::world_rules::{
    beach_preset, builtin_world_rule_presets, cliff_preset, diagnose_world_rule,
    high_altitude_rock_preset, placement_from_conditions, riverbank_preset, snowline_preset,
    underwater_sand_preset, world_rule_preset_by_name, WorldRule, WorldRuleCommand,
    WorldRuleEffect, WorldRuleEffectKind, WorldRuleLibrary, WorldRulePhase, WorldRuleScope,
};
use terra_core::{coastal_wetness_preset, EvalStage};

fn metrics() -> HeightfieldMetrics {
    HeightfieldMetrics::new(32, 32, 1024.0, 1024.0)
}

#[test]
fn scope_entire_world_and_selected_biomes() {
    let bid = terra_core::biome_definition::BiomeDefinitionId::new();
    let other = terra_core::biome_definition::BiomeDefinitionId::new();
    let scope = WorldRuleScope::SelectedBiomes(vec![bid]);
    assert!(scope.includes_biome(bid));
    assert!(!scope.includes_biome(other));
    assert!(WorldRuleScope::EntireWorld.includes_biome(bid));

    let excl = WorldRuleScope::Excluding { biomes: vec![bid] };
    assert!(!excl.includes_biome(bid));
    assert!(excl.includes_biome(other));
}

#[test]
fn placement_compiles_into_mask_infrastructure() {
    let rule = beach_preset();
    let dist = rule.compile_placement();
    assert!(
        !dist.is_empty() || !rule.placement.root.children.is_empty(),
        "beach placement must compile to DistNodes or have rules"
    );
    // Compile produces DistNode tree for bake.
    let compiled = rule.placement.compile();
    assert!(!compiled.is_empty());
}

#[test]
fn multiple_effects_share_one_placement() {
    let rule = beach_preset();
    assert_eq!(rule.effects.len(), 3);
    assert!(rule.effects.iter().any(|e| e.name.contains("Sand")));
    assert!(rule
        .effects
        .iter()
        .any(|e| matches!(e.kind, WorldRuleEffectKind::TerrainEffect)));
    assert!(rule
        .effects
        .iter()
        .any(|e| matches!(e.kind, WorldRuleEffectKind::Scatter)));
    // Single placement IR for all effects.
    assert_eq!(rule.placement.root.children.len(), 3);
}

#[test]
fn effects_can_be_independently_toggled() {
    let mut rule = beach_preset();
    rule.effects[1].enabled = false;
    let enabled: Vec<_> = rule.enabled_effects().map(|e| e.name.as_str()).collect();
    assert_eq!(enabled.len(), 2);
    assert!(!enabled.contains(&"Slight Terrain Smoothing"));
}

#[test]
fn priority_reordering() {
    let mut lib = WorldRuleLibrary::default();
    let a = snowline_preset();
    let b = beach_preset();
    let aid = a.id;
    let bid = b.id;
    lib.push(a);
    lib.push(b);
    lib.set_priority(aid, 10);
    lib.set_priority(bid, 90);
    let sorted = lib.sorted_by_priority();
    assert_eq!(sorted[0].id, bid);
    assert_eq!(sorted[1].id, aid);

    assert!(lib.reorder(aid, Some(bid))); // move snowline before beach → top
    let sorted = lib.sorted_by_priority();
    assert_eq!(sorted[0].id, aid);
}

#[test]
fn empty_coverage_is_not_an_error() {
    let mut rule = WorldRule::new("Impossible Heights");
    rule.placement = placement_from_conditions(vec![terra_core::mask::Condition {
        channel: ConditionChannel::Height,
        op: CompareOp::Above,
        a: 50_000.0,
        b: 0.0,
        falloff: 1.0,
        seed: 0,
        frequency: 0.02,
    }]);
    rule.effects
        .push(WorldRuleEffect::new("Snow", WorldRuleEffectKind::Material));

    let m = metrics();
    let height = Heightfield::zeros(m);
    let ctx = DistBakeContext {
        height: Some(&height),
        slope_deg: None,
        curvature: None,
        flow: None,
        masks: &HashMap::new(),
        aux: None,
    };
    let cov = rule.refresh_coverage(m, &ctx);
    assert!(cov < 0.05);
    // Allowed — diagnostics are soft, not hard errors.
    let diags = diagnose_world_rule(&rule);
    assert!(diags.iter().any(|d| d.code == "world_rule_zero_coverage"));
    assert!(rule.enabled); // still a valid project entity
}

#[test]
fn execution_phases_inferred_and_overridable() {
    let snow = snowline_preset();
    assert_eq!(snow.resolved_phase(), WorldRulePhase::Materials);
    assert_eq!(snow.invalidation_stage(), EvalStage::Materials);

    let mut beach = beach_preset();
    assert!(beach.phase_needs_user_input());
    beach.phase_override = Some(WorldRulePhase::Materials);
    assert_eq!(beach.resolved_phase(), WorldRulePhase::Materials);
    assert!(!beach.phase_needs_user_input() || beach.phase_override.is_some());

    let cliff = cliff_preset();
    // Material + ObjectExclusion → ambiguous or Objects/Materials
    let phase = cliff.resolved_phase();
    assert!(matches!(
        phase,
        WorldRulePhase::Materials | WorldRulePhase::Objects | WorldRulePhase::BeforePhysics
    ));
}

#[test]
fn selective_invalidation_by_phase() {
    let mut stack = LayerStack::new();
    let mountains = Layer::new("Mountains", LayerKind::Mountains(MountainParams::default()));
    let materials = Layer::new("Rock", LayerKind::Materials(MaterialsParams::default()));
    let mid = mountains.id();
    let mat_id = materials.id();
    stack.nodes.push(StackNode::Layer(mountains));
    stack.nodes.push(StackNode::Layer(materials));

    let m = metrics();
    let mut ctx = EvalContext::new(m);
    ctx.quality = PreviewQuality::Draft;
    let seed = Heightfield::zeros(m);
    ctx.masks = bake_mask_assets(&[], &seed, m, &HashMap::new());
    let mut eval = StackEvaluator::new();
    let _ = eval.rebuild_all(&stack, &mut ctx).expect("eval");
    if let Some(e) = eval.cache.get_mut(mid) {
        e.dirty = false;
    }
    if let Some(e) = eval.cache.get_mut(mat_id) {
        e.dirty = false;
    }

    // Materials-phase World Rule should not dirty height.
    eval.mark_dirty_from_eval_stage(&stack, EvalStage::Materials);
    assert!(!eval.cache.is_dirty(mid));
    assert!(eval.cache.is_dirty(mat_id));
}

#[test]
fn serialization_roundtrip() {
    let mut lib = WorldRuleLibrary::default();
    lib.push(beach_preset());
    lib.push(snowline_preset());
    let json = serde_json::to_string(&lib).expect("ser");
    assert!(json.contains("Beaches") || json.contains("Snowline"));
    let back: WorldRuleLibrary = serde_json::from_str(&json).expect("de");
    assert_eq!(back.rules.len(), 2);
    assert_eq!(back.rules[0].effects.len(), 3);
}

#[test]
fn document_serializes_world_rules_field() {
    let mut session = EditorSession::new();
    session
        .document
        .world_rules
        .push(world_rule_preset_by_name("Cliff").unwrap());
    let json = session.document.to_json().expect("doc ser");
    // Field present (serde may omit empty; we have a rule).
    assert!(json.contains("world_rules") || json.contains("Cliff"));
    let back = terra_core::document::TerrainDocument::from_json(&json).expect("doc de");
    assert_eq!(back.world_rules.rules.len(), 1);
    assert_eq!(back.world_rules.rules[0].name, "Cliff");
}

#[test]
fn presets_cover_required_set() {
    let names: Vec<_> = builtin_world_rule_presets()
        .into_iter()
        .map(|r| r.name)
        .collect();
    for expected in [
        "Snowline",
        "Beaches",
        "Cliff",
        "Coastal Wetness",
        "Riverbank",
        "Underwater Sand",
        "High Altitude Rock",
    ] {
        assert!(
            names.iter().any(|n| n == expected),
            "missing preset {expected}"
        );
    }
    assert!(coastal_wetness_preset().effects[0].kind == WorldRuleEffectKind::Material);
    assert!(riverbank_preset().placement.root.children.len() == 1);
    assert!(underwater_sand_preset().name.contains("Underwater"));
    assert!(high_altitude_rock_preset().placement.root.children.len() == 2);
}

#[test]
fn undo_redo_world_rules() {
    let mut session = EditorSession::new();
    let rule = snowline_preset();
    let id = rule.id;
    session.push_world_rule_command(WorldRuleCommand::Add { rule, index: 0 });
    assert_eq!(session.document.world_rules.rules.len(), 1);

    assert!(session.undo_world_rule());
    assert!(session.document.world_rules.rules.is_empty());

    assert!(session.redo_world_rule());
    assert_eq!(session.document.world_rules.rules.len(), 1);
    assert_eq!(session.document.world_rules.rules[0].id, id);

    session.push_world_rule_command(WorldRuleCommand::SetEnabled {
        id,
        enabled: false,
        previous: true,
    });
    assert!(!session.document.world_rules.get(id).unwrap().enabled);
    assert!(session.undo_world_rule());
    assert!(session.document.world_rules.get(id).unwrap().enabled);
}

#[test]
fn rules_for_phase_filters_enabled() {
    let mut lib = WorldRuleLibrary::default();
    let mut snow = snowline_preset();
    snow.phase_override = Some(WorldRulePhase::Materials);
    let mut beach = beach_preset();
    beach.phase_override = Some(WorldRulePhase::BeforePhysics);
    beach.enabled = false;
    lib.push(snow);
    lib.push(beach);
    assert_eq!(lib.rules_for_phase(WorldRulePhase::Materials).len(), 1);
    assert!(lib
        .rules_for_phase(WorldRulePhase::BeforePhysics)
        .is_empty());
}

#[test]
fn contradictory_range_diagnostic() {
    let mut rule = WorldRule::new("Bad Range");
    rule.placement = placement_from_conditions(vec![terra_core::mask::Condition {
        channel: ConditionChannel::Height,
        op: CompareOp::Between,
        a: 100.0,
        b: 10.0,
        falloff: 1.0,
        seed: 0,
        frequency: 0.02,
    }]);
    let diags = diagnose_world_rule(&rule);
    assert!(diags
        .iter()
        .any(|d| d.code == "world_rule_contradictory_range"));
}
