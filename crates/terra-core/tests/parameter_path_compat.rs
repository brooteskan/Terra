//! Legacy layer parameter paths remain aliases of their canonical owners.

#[test]
fn legacy_layer_parameter_paths_name_the_canonical_types() {
    let erosion = terra_core::analyze::HydraulicErosionParams::default();
    let _: terra_core::layer::HydraulicErosionParams = erosion;

    let stream_power = terra_core::analyze::StreamPowerParams::default();
    let _: terra_core::layer::StreamPowerParams = stream_power;

    let river = terra_core::hydro::RiverCarveParams::default();
    let _: terra_core::layer::RiverCarveParams = river;

    let filter = terra_core::generators::EffectFilterParams::default();
    let _: terra_core::layer::EffectFilterParams = filter;

    let materials = terra_core::material_schema::MaterialsParams::default();
    let _: terra_core::layer::MaterialsParams = materials;

    let vegetation = terra_core::scatter::VegetationParams::default();
    let _: terra_core::layer::VegetationParams = vegetation;

    let overhang = terra_core::generators::OverhangStampParams::default();
    let _: terra_core::layer::OverhangStampParams = overhang;
}
