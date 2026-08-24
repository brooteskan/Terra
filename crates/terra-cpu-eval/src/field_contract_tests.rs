use super::*;
use std::collections::HashSet;
use std::path::PathBuf;
use terra_core::field_data::{keys, FieldId};
use terra_core::layer::{LayerKind, LayerTypeRegistry, MaterialsParams, Stratum};
use terra_core::mask::MaskSource;

fn canonical_key(key: &str) -> String {
    FieldId::from_cache_key(key).cache_key()
}

fn undeclared_writes(layer: &Layer, observed: impl IntoIterator<Item = String>) -> Vec<String> {
    let declared: HashSet<_> = layer
        .kind
        .produced_fields()
        .into_iter()
        .chain(layer.kind.modified_fields())
        .filter(|field| *field != FieldId::Height)
        .map(|field| canonical_key(&field.cache_key()))
        .collect();
    let mut undeclared: Vec<_> = observed
        .into_iter()
        .map(|key| canonical_key(&key))
        .filter(|key| !declared.contains(key))
        .collect();
    undeclared.sort();
    undeclared.dedup();
    undeclared
}

fn relief(metrics: HeightfieldMetrics) -> Heightfield {
    let mut input = Heightfield::zeros(metrics);
    let width = metrics.width.saturating_sub(1).max(1) as f32;
    let height = metrics.height.saturating_sub(1).max(1) as f32;
    for j in 0..metrics.height {
        for i in 0..metrics.width {
            let x = i as f32 / width;
            let z = j as f32 / height;
            input.set(i, j, 40.0 * (x * 6.0).sin() * (z * 5.0).cos() + 60.0 * x);
        }
    }
    input.refresh_halos();
    input
}

fn seed_common_aux(ctx: &mut EvalContext) {
    let metrics = ctx.metrics;
    for (key, value) in [
        (keys::HARDNESS, 0.45),
        (keys::WETNESS, 0.35),
        (keys::SEDIMENT, 0.1),
        (keys::SEDIMENT_THICKNESS, 0.2),
        (keys::DEBRIS_DEPTH, 0.1),
        (keys::RAINFALL, 0.6),
        (keys::SOIL_DEPTH, 0.4),
        (keys::BIOMES, 0.25),
        (keys::VEGETATION, 0.3),
    ] {
        ctx.aux_insert(key, MaskField::filled(metrics, value));
    }
}

fn seed_strata(ctx: &mut EvalContext, input: &Heightfield) {
    ctx.aux_insert(
        keys::STRATA_REFERENCE,
        MaskField::from_raw(input.metrics, &input.to_dense()),
    );
    ctx.aux_maps.strata = Some(vec![Stratum::soft_cap(20.0), Stratum::hard_base()]);
}

fn evaluate_write_set(
    layer: &Layer,
    configure: impl FnOnce(&mut EvalContext, &Heightfield),
) -> Result<Vec<String>, String> {
    let metrics = HeightfieldMetrics::new(32, 32, 256.0, 256.0);
    let input = relief(metrics);
    let mut ctx = EvalContext::new(metrics);
    ctx.quality = PreviewQuality::Draft;
    seed_common_aux(&mut ctx);
    configure(&mut ctx, &input);
    ctx.begin_aux_write_capture();
    let mut evaluator = StackEvaluator::new();
    evaluator
        .evaluate_layer(&mut ctx, &input, layer)
        .map_err(|error| format!("{} failed evaluation: {error}", layer.kind.type_id()))?;
    Ok(ctx.take_aux_write_capture().into_keys().collect())
}

fn repository_image() -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../assets/logo.png")
        .to_string_lossy()
        .into_owned()
}

fn configure_file_backed_default(layer: &mut Layer) {
    let path = repository_image();
    match &mut layer.kind {
        LayerKind::ImportHeightmap(params) => params.path = path,
        LayerKind::Stamp2d(params) => params.heightmap.path = path,
        LayerKind::Stamp3d(params) => params.path = path,
        _ => {}
    }
}

fn record_violations(
    label: &str,
    layer: &Layer,
    result: Result<Vec<String>, String>,
    violations: &mut Vec<String>,
) {
    match result {
        Ok(writes) => {
            for key in undeclared_writes(layer, writes) {
                violations.push(format!("{label} writes undeclared '{key}'"));
            }
        }
        Err(error) => violations.push(error),
    }
}

#[test]
fn processor_writes_stay_within_declared_contracts() {
    let registry = LayerTypeRegistry::builtin();
    let mut violations = Vec::new();

    for metadata in registry.all() {
        let mut layer = registry
            .create(metadata.type_id)
            .unwrap_or_else(|| panic!("registered kind {} must construct", metadata.type_id));
        configure_file_backed_default(&mut layer);
        record_violations(
            metadata.type_id,
            &layer,
            evaluate_write_set(&layer, |_, _| {}),
            &mut violations,
        );
    }

    let materials = Layer::new(
        "Materials with strata",
        LayerKind::Materials(MaterialsParams {
            strata: vec![Stratum::soft_cap(20.0), Stratum::hard_base()],
            ..MaterialsParams::default()
        }),
    );
    record_violations(
        "materials strata path",
        &materials,
        evaluate_write_set(&materials, |_, _| {}),
        &mut violations,
    );

    let mut vegetation = registry.create("vegetation").expect("vegetation");
    if let LayerKind::Vegetation(params) = &mut vegetation.kind {
        params.root_cohesion = 0.5;
    }
    record_violations(
        "vegetation root-cohesion path",
        &vegetation,
        evaluate_write_set(&vegetation, |_, _| {}),
        &mut violations,
    );

    for type_id in ["thermal_erosion", "hydraulic_erosion", "stream_power"] {
        let mut layer = registry
            .create(type_id)
            .expect("registered strata evaluator");
        match &mut layer.kind {
            LayerKind::ThermalErosion(params) => params.hardness_source = MaskSource::Hardness,
            LayerKind::HydraulicErosion(params) => params.hardness_source = MaskSource::Hardness,
            LayerKind::StreamPowerErosion(params) => params.hardness_source = MaskSource::Hardness,
            _ => unreachable!("type id selected a different kind"),
        }
        record_violations(
            &format!("{type_id} strata path"),
            &layer,
            evaluate_write_set(&layer, seed_strata),
            &mut violations,
        );
    }

    violations.sort();
    assert!(
        violations.is_empty(),
        "layer processors wrote fields outside their static contracts:\n  {}",
        violations.join("\n  ")
    );
}

#[test]
fn contract_ratchet_detects_an_undeclared_runtime_write() {
    let layer = Layer::new("Flat", LayerKind::Flat(Default::default()));
    assert_eq!(
        undeclared_writes(&layer, [keys::WETNESS.to_string()]),
        [keys::WETNESS]
    );
}

#[test]
fn contract_ratchet_canonicalizes_legacy_sediment_aliases() {
    let layer = Layer::new("Sculpt", LayerKind::SculptStrokes(Default::default()));
    assert!(undeclared_writes(
        &layer,
        [
            keys::SEDIMENT_THICKNESS.to_string(),
            keys::SEDIMENT_DEPTH.to_string(),
            keys::LOOSE_SEDIMENT.to_string(),
        ],
    )
    .is_empty());
}
