use std::collections::HashMap;

use terra_core::eval::{EvalContext, StackEvaluator};
use terra_core::fields::{keys, AuxMaps};
use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
use terra_core::layer::{
    DebrisFlowParams, FlatParams, HydraulicErosionParams, Layer, LayerId, LayerKind, LayerStack,
    ThermalErosionParams, TransportModel,
};
use terra_core::mask::MaskField;

const RESOLUTION: u32 = 16;
const INITIAL_SEDIMENT: f32 = 1.25;
const EXPECTED_SEDIMENT_SUM: f32 = 320.0;

struct StackRun {
    evaluator: StackEvaluator,
    context: EvalContext,
    height: Heightfield,
    hydraulic: LayerId,
    debris: LayerId,
    thermal: LayerId,
}

fn run_no_transport_stack(base_height: f32) -> StackRun {
    let metrics = HeightfieldMetrics::new(RESOLUTION, RESOLUTION, 160.0, 160.0);
    let hydraulic_params = HydraulicErosionParams {
        iterations: 1,
        rainfall: 0.0,
        evaporation: 0.0,
        capacity: 0.0,
        erosion: 0.0,
        deposition: 0.0,
        particle_density: 0.0,
        transport_model: TransportModel::HydraulicSediment,
        layered_materials: true,
        initial_sediment_thickness: INITIAL_SEDIMENT,
        level_count: 1,
        ..HydraulicErosionParams::default()
    };
    let debris_params = DebrisFlowParams {
        iterations: 1,
        thermal_k: 0.0,
        abrasion_k: 0.0,
        fluvial_k: 0.0,
        fluvial_deposition: 0.0,
        hillslope_k: 0.0,
        precipitation: 0.0,
        initial_debris_thickness: 0.0,
        initial_sediment_thickness: 0.0,
        ..DebrisFlowParams::default()
    };
    let thermal_params = ThermalErosionParams {
        iterations: 1,
        strength: 0.0,
        weathering_rate: 0.0,
        material_amount: 0.0,
        layered_materials: true,
        level_count: 1,
        ..ThermalErosionParams::default()
    };

    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "Base",
        LayerKind::Flat(FlatParams {
            height: base_height,
        }),
    ));
    let hydraulic_layer = Layer::new("Hydraulic", LayerKind::HydraulicErosion(hydraulic_params));
    let hydraulic = hydraulic_layer.id();
    stack.push(hydraulic_layer);
    let debris_layer = Layer::new("Debris", LayerKind::DebrisFlow(debris_params));
    let debris = debris_layer.id();
    stack.push(debris_layer);
    let thermal_layer = Layer::new("Thermal", LayerKind::ThermalErosion(thermal_params));
    let thermal = thermal_layer.id();
    stack.push(thermal_layer);

    let mut evaluator = StackEvaluator::new();
    evaluator.cache.disable_disk();
    let mut context = EvalContext::new(metrics);
    let height = evaluator
        .rebuild_all(&stack, &mut context)
        .expect("erosion stack evaluates");
    StackRun {
        evaluator,
        context,
        height,
        hydraulic,
        debris,
        thermal,
    }
}

fn field<'a>(aux: &'a HashMap<String, MaskField>, key: &str) -> &'a MaskField {
    aux.get(key).unwrap_or_else(|| panic!("missing {key}"))
}

fn sum(field: &MaskField) -> f32 {
    field.data().iter().sum()
}

fn assert_canonical_inventory(
    aux: &HashMap<String, MaskField>,
    expected_bedrock: f32,
    expected_debris: f32,
) {
    assert!(aux.contains_key(keys::BEDROCK_HEIGHT));
    assert!(aux.contains_key(keys::SEDIMENT_THICKNESS));
    assert!(!aux.contains_key(keys::SEDIMENT_DEPTH));
    assert!(!aux.contains_key(keys::LOOSE_SEDIMENT));
    for &value in field(aux, keys::BEDROCK_HEIGHT).data() {
        assert!((value - expected_bedrock).abs() < 1e-5);
    }
    if let Some(debris) = aux.get(keys::DEBRIS_DEPTH) {
        for &value in debris.data() {
            assert!((value - expected_debris).abs() < 1e-5);
        }
    } else {
        assert_eq!(expected_debris, 0.0, "missing non-zero debris inventory");
    }
    for &value in field(aux, keys::SEDIMENT_THICKNESS).data() {
        assert!((value - INITIAL_SEDIMENT).abs() < 1e-5);
    }
    assert!((sum(field(aux, keys::SEDIMENT_THICKNESS)) - EXPECTED_SEDIMENT_SUM).abs() < 1e-4);
}

#[test]
fn hydraulic_sediment_survives_debris_and_thermal_stack() {
    let run = run_no_transport_stack(10.0);
    let expected_bedrock = 10.0 - INITIAL_SEDIMENT;
    for id in [run.hydraulic, run.debris, run.thermal] {
        let checkpoint = run.evaluator.cache.get(id).expect("layer checkpoint");
        assert_canonical_inventory(&checkpoint.aux, expected_bedrock, 0.0);
    }
    assert_canonical_inventory(&run.context.aux, expected_bedrock, 0.0);
}

#[test]
fn sub_zero_handoff_reconstructs_the_same_surface() {
    let run = run_no_transport_stack(-10.0);
    for id in [run.hydraulic, run.debris, run.thermal] {
        let checkpoint = run.evaluator.cache.get(id).expect("layer checkpoint");
        assert_canonical_inventory(&checkpoint.aux, 0.0, 0.0);
        for &height in &checkpoint.height.to_dense() {
            assert!((height + 10.0).abs() < 1e-5);
        }
    }
    for &height in &run.height.to_dense() {
        assert!((height + 10.0).abs() < 1e-5);
    }

    let reconstructed = AuxMaps::from_hashmap(&run.context.aux);
    let state = terra_core::analyze::MassWastingState::with_optional_layers(
        &run.height,
        reconstructed.bedrock_height.as_ref(),
        reconstructed.get(keys::DEBRIS_DEPTH),
        reconstructed.sediment_thickness.as_ref(),
        0.0,
        0.0,
    );
    for &height in &state.sync_surface() {
        assert!((height + 10.0).abs() < 1e-5);
    }
}
