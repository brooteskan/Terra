//! Explicit compatibility manifest for enum tags reachable from project JSON.
//!
//! This is deliberately a value manifest, not Rust source reflection. New variants
//! may be added without changing old fixtures, but a recorded tag cannot disappear
//! unless the reader keeps an alias or migration for it.

use std::collections::BTreeSet;

use serde::{de::DeserializeOwned, Serialize};
use serde_json::{json, Value};
use terra_core::analyze::HighDetailMode;
use terra_core::authoring::{SculptStrokeKind, TerrainConstraintKind};
use terra_core::biome_definition::{BiomeOverlapPolicy, PlacementCombineMode};
use terra_core::layer::{
    BedGeometry, BiomeSection, BlendMode, BoundaryMode, CachePolicy, EffectFilterKind,
    EvolutionSolverMode, FalloffCurve, GroupEvalMode, GroupInputMode, GroupKind, IslandArchetype,
    LayerGroup, LayerKind, LayerTypeRegistry, PolygonHeightMode, ProceduralGenerator,
    SelectedGroupInput, StackCategory, StackNode, StratumMaterial, TransportModel, UpliftMode,
};
use terra_core::mask::{
    ClimateMaskChannel, CompareOp, Condition, ConditionChannel, CoverageTerm, DistNodeKind,
    MaskCombine, MaskId, MaskOp, MaskRef, MaskSource, PlacementCoordinateSpace,
    PlacementRefinement, PlacementSource, RuleGroup, RuleGroupMode, RuleNode,
};
use terra_core::noise::{FractalNoiseType, WorleyFeature, WorleyMetric};
use terra_core::operation_placement::{ApplyWhere, DevelopCategory};
use terra_core::simulation_scenario::{
    MatterArtistSource, MatterSourceKind, MatterType, OutputInfluence, ScenarioPassKind,
    ScenarioQuality, ScenarioResultState, ScenarioScope, SimulationDomain,
};
use terra_core::world_rules::{WorldRuleEffectKind, WorldRulePhase, WorldRuleScope};
use terra_core::{FieldId, ShapeKind};

fn round_trip_tag<T>(value: T, expected_tag: &str)
where
    T: Serialize + DeserializeOwned,
{
    let encoded = serde_json::to_value(&value).expect("serialize manifested enum value");
    let actual_tag = match &encoded {
        Value::String(tag) => tag.as_str(),
        Value::Object(object) if object.len() == 1 => {
            object.keys().next().expect("external enum tag")
        }
        other => panic!("expected unit or externally tagged enum, got {other}"),
    };
    assert_eq!(actual_tag, expected_tag);

    let decoded: T = serde_json::from_value(encoded.clone()).expect("deserialize manifested tag");
    assert_eq!(
        serde_json::to_value(decoded).expect("reserialize manifested tag"),
        encoded
    );
}

fn round_trip_internal_tag<T>(value: T, expected_tag: &str)
where
    T: Serialize + DeserializeOwned,
{
    let encoded = serde_json::to_value(&value).expect("serialize internally tagged enum");
    assert_eq!(encoded.get("kind"), Some(&json!(expected_tag)));
    let decoded: T = serde_json::from_value(encoded.clone()).expect("deserialize internal tag");
    assert_eq!(serde_json::to_value(decoded).unwrap(), encoded);
}

macro_rules! unit_tags {
    ($($value:expr => $tag:literal),+ $(,)?) => {{
        $(round_trip_tag($value, $tag);)+
    }};
}

#[test]
fn layer_kind_and_stack_node_tags_are_frozen() {
    let registry = LayerTypeRegistry::builtin();
    let actual: BTreeSet<String> = registry
        .all()
        .iter()
        .map(|meta| {
            let layer = registry
                .create(meta.type_id)
                .unwrap_or_else(|| panic!("factory for {}", meta.type_id));
            let encoded = serde_json::to_value(&layer.kind).expect("serialize LayerKind");
            let Value::Object(object) = encoded else {
                panic!("LayerKind must remain externally tagged");
            };
            assert_eq!(object.len(), 1);
            object.keys().next().unwrap().clone()
        })
        .collect();
    let expected: BTreeSet<String> = [
        "SculptBase",
        "SculptStrokes",
        "TerrainConstraints",
        "GradientReconstruct",
        "LandscapeEvolution",
        "HydrologyRepair",
        "GeomorphicDetail",
        "EcosystemFeedback",
        "Flat",
        "Ramp",
        "NoiseValue",
        "NoisePerlin",
        "NoiseOpenSimplex",
        "NoiseWorley",
        "Fbm",
        "Ridged",
        "DomainWarp",
        "Terrace",
        "Plateau",
        "Mesa",
        "Island",
        "Mountains",
        "Volcano",
        "Uplift",
        "Dunes",
        "Canyons",
        "VoronoiRegions",
        "ImportHeightmap",
        "ThermalErosion",
        "HydraulicErosion",
        "DebrisFlow",
        "StreamPowerErosion",
        "MultiScaleAmplify",
        "RiverCarve",
        "Blur",
        "Coastal",
        "EffectFilter",
        "Materials",
        "Biomes",
        "Vegetation",
        "OverhangStamp",
        "LocalSdf",
        "Path",
        "RiverNetwork",
        "SandSimulation",
        "FluidSimulation",
        "ProceduralShape",
        "Stamp2d",
        "Stamp3d",
        "PolygonHeight",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    assert_eq!(actual, expected);

    round_trip_tag(
        StackNode::Layer(terra_core::layer::Layer::new(
            "tag",
            LayerKind::Flat(Default::default()),
        )),
        "Layer",
    );
    round_trip_tag(StackNode::Group(LayerGroup::new("tag")), "Group");
}

#[test]
fn structural_and_distribution_mode_tags_are_frozen() {
    unit_tags!(
        BlendMode::Normal => "Normal", BlendMode::Replace => "Replace",
        BlendMode::Add => "Add", BlendMode::Subtract => "Subtract",
        BlendMode::Multiply => "Multiply", BlendMode::Min => "Min", BlendMode::Max => "Max",
        BlendMode::Interpolate => "Interpolate", BlendMode::HeightBlend => "HeightBlend",
        BlendMode::Overlay => "Overlay", BlendMode::SmoothMaximum => "SmoothMaximum",
        BlendMode::SmoothMinimum => "SmoothMinimum", BlendMode::SmoothUnion => "SmoothUnion",
        BlendMode::SmoothSubtraction => "SmoothSubtraction"
    );
    unit_tags!(CachePolicy::Live => "Live", CachePolicy::Cached => "Cached", CachePolicy::Manual => "Manual", CachePolicy::Frozen => "Frozen", CachePolicy::Baked => "Baked");
    unit_tags!(StackCategory::Foundation => "Foundation", StackCategory::Shape => "Shape", StackCategory::Simulation => "Simulation", StackCategory::Mask => "Mask", StackCategory::Surface => "Surface");
    unit_tags!(BiomeSection::Filters => "Filters", BiomeSection::Materials => "Materials", BiomeSection::Objects => "Objects", BiomeSection::LocalSims => "LocalSims");
    unit_tags!(GroupKind::Generic => "Generic", GroupKind::CategoryFolder => "CategoryFolder", GroupKind::Biome => "Biome");
    round_trip_tag(
        GroupKind::BiomeSection(BiomeSection::Materials),
        "BiomeSection",
    );
    unit_tags!(GroupEvalMode::PassThrough => "PassThrough", GroupEvalMode::IsolatedComposite => "IsolatedComposite");
    unit_tags!(GroupInputMode::CopyInput => "CopyInput", GroupInputMode::EmptyHeight => "EmptyHeight");
    round_trip_tag(
        GroupInputMode::SelectedField(SelectedGroupInput::Field(FieldId::Height)),
        "SelectedField",
    );
    unit_tags!(FalloffCurve::Linear => "Linear", FalloffCurve::Smoothstep => "Smoothstep", FalloffCurve::Smooth => "Smooth");
    unit_tags!(MaskCombine::Multiply => "Multiply", MaskCombine::Add => "Add", MaskCombine::Subtract => "Subtract", MaskCombine::Min => "Min", MaskCombine::Max => "Max", MaskCombine::Replace => "Replace", MaskCombine::Invert => "Invert", MaskCombine::PaintOverride => "PaintOverride");
}

#[test]
fn mask_source_and_operation_tags_are_frozen() {
    let mask = MaskId::nil();
    round_trip_tag(MaskSource::None, "None");
    round_trip_tag(MaskSource::Constant(0.5), "Constant");
    round_trip_tag(MaskSource::Height { min: 0.0, max: 1.0 }, "Height");
    round_trip_tag(
        MaskSource::Slope {
            min_deg: 0.0,
            max_deg: 45.0,
        },
        "Slope",
    );
    round_trip_tag(
        MaskSource::Aspect {
            center_deg: 90.0,
            width_deg: 30.0,
        },
        "Aspect",
    );
    round_trip_tag(
        MaskSource::Curvature {
            min: -1.0,
            max: 1.0,
        },
        "Curvature",
    );
    round_trip_tag(MaskSource::Convexity, "Convexity");
    round_trip_tag(MaskSource::Concavity, "Concavity");
    round_trip_tag(
        MaskSource::AmbientOcclusion {
            radius: 2,
            strength: 1.0,
        },
        "AmbientOcclusion",
    );
    round_trip_tag(
        MaskSource::DistanceField { threshold: 0.5 },
        "DistanceField",
    );
    round_trip_tag(
        MaskSource::Noise {
            seed: 1,
            frequency: 0.1,
        },
        "Noise",
    );
    round_trip_tag(MaskSource::Painted { mask_id: mask }, "Painted");
    round_trip_tag(MaskSource::FlowDirection, "FlowDirection");
    round_trip_tag(
        MaskSource::FlowAccumulation { min: 0.0, max: 1.0 },
        "FlowAccumulation",
    );
    for (value, tag) in [
        (MaskSource::Wetness, "Wetness"),
        (MaskSource::Sediment, "Sediment"),
        (MaskSource::Erosion, "Erosion"),
        (MaskSource::Deposition, "Deposition"),
        (MaskSource::Hardness, "Hardness"),
        (MaskSource::Temperature, "Temperature"),
        (MaskSource::Rainfall, "Rainfall"),
        (MaskSource::Humidity, "Humidity"),
        (MaskSource::Snow, "Snow"),
        (MaskSource::SoilMoisture, "SoilMoisture"),
        (MaskSource::WindExposure, "WindExposure"),
    ] {
        round_trip_tag(value, tag);
    }
    round_trip_tag(MaskSource::Named("legacy".into()), "Named");
    round_trip_tag(
        MaskSource::LayerOutput {
            output_id: Default::default(),
        },
        "LayerOutput",
    );

    round_trip_tag(MaskOp::Add { amount: 1.0 }, "Add");
    round_trip_tag(MaskOp::Subtract { amount: 1.0 }, "Subtract");
    round_trip_tag(MaskOp::Multiply { amount: 1.0 }, "Multiply");
    round_trip_tag(MaskOp::Min { value: 0.0 }, "Min");
    round_trip_tag(MaskOp::Max { value: 1.0 }, "Max");
    round_trip_tag(MaskOp::Invert, "Invert");
    round_trip_tag(MaskOp::Clamp { min: 0.0, max: 1.0 }, "Clamp");
    round_trip_tag(
        MaskOp::Levels {
            in_black: 0.0,
            in_white: 1.0,
            gamma: 1.0,
        },
        "Levels",
    );
    round_trip_tag(
        MaskOp::Smoothstep {
            edge0: 0.0,
            edge1: 1.0,
        },
        "Smoothstep",
    );
    round_trip_tag(MaskOp::Blur { radius: 1 }, "Blur");
    round_trip_tag(
        MaskOp::Remap {
            out_min: 0.0,
            out_max: 1.0,
        },
        "Remap",
    );
}

#[test]
fn distribution_node_tags_are_frozen() {
    let mask = MaskRef::new(MaskId::nil());
    round_trip_tag(DistNodeKind::Fill { value: 1.0 }, "Fill");
    round_trip_tag(DistNodeKind::MaskAsset { mask: mask.clone() }, "MaskAsset");
    round_trip_tag(
        DistNodeKind::Noise {
            seed: 1,
            frequency: 0.1,
        },
        "Noise",
    );
    round_trip_tag(
        DistNodeKind::NoisePerlin {
            seed: 1,
            frequency: 0.1,
            octaves: 2,
        },
        "NoisePerlin",
    );
    round_trip_tag(
        DistNodeKind::NoiseRidged {
            seed: 1,
            frequency: 0.1,
            octaves: 2,
        },
        "NoiseRidged",
    );
    round_trip_tag(
        DistNodeKind::NoiseWorley {
            seed: 1,
            frequency: 0.1,
        },
        "NoiseWorley",
    );
    round_trip_tag(
        DistNodeKind::NoiseBillow {
            seed: 1,
            frequency: 0.1,
            octaves: 2,
        },
        "NoiseBillow",
    );
    round_trip_tag(DistNodeKind::Height { min: 0.0, max: 1.0 }, "Height");
    round_trip_tag(
        DistNodeKind::Slope {
            min_deg: 0.0,
            max_deg: 45.0,
        },
        "Slope",
    );
    round_trip_tag(
        DistNodeKind::Curvature {
            min: -1.0,
            max: 1.0,
        },
        "Curvature",
    );
    round_trip_tag(DistNodeKind::Cavity { strength: 1.0 }, "Cavity");
    round_trip_tag(DistNodeKind::Flow { min: 0.0, max: 1.0 }, "Flow");
    round_trip_tag(
        DistNodeKind::SeaLevel {
            level: 0.0,
            width: 1.0,
        },
        "SeaLevel",
    );
    round_trip_tag(
        DistNodeKind::Occlusion {
            radius: 2,
            strength: 1.0,
        },
        "Occlusion",
    );
    round_trip_tag(
        DistNodeKind::Steepness {
            min_deg: 0.0,
            max_deg: 45.0,
        },
        "Steepness",
    );
    round_trip_tag(
        DistNodeKind::Angle {
            degrees: 90.0,
            spread: 30.0,
        },
        "Angle",
    );
    round_trip_tag(
        DistNodeKind::Roughness {
            radius: 2,
            strength: 1.0,
        },
        "Roughness",
    );
    round_trip_tag(
        DistNodeKind::Rocks {
            density: 0.5,
            threshold: 0.5,
        },
        "Rocks",
    );
    round_trip_tag(
        DistNodeKind::RockyEdges {
            width: 1.0,
            strength: 1.0,
        },
        "RockyEdges",
    );
    round_trip_tag(DistNodeKind::EffectInvert, "EffectInvert");
    round_trip_tag(DistNodeKind::EffectBlur { radius: 1 }, "EffectBlur");
    round_trip_tag(
        DistNodeKind::EffectLevels {
            in_black: 0.0,
            in_white: 1.0,
            gamma: 1.0,
        },
        "EffectLevels",
    );
    round_trip_tag(
        DistNodeKind::EffectRemap {
            in_min: 0.0,
            in_max: 1.0,
        },
        "EffectRemap",
    );
    round_trip_tag(
        DistNodeKind::EffectContrast { amount: 1.0 },
        "EffectContrast",
    );
    round_trip_tag(
        DistNodeKind::EffectClamp { min: 0.0, max: 1.0 },
        "EffectClamp",
    );
    round_trip_tag(DistNodeKind::EffectCurve { contrast: 1.0 }, "EffectCurve");
    round_trip_tag(
        DistNodeKind::EffectDistortion {
            seed: 1,
            frequency: 0.1,
            amount: 1.0,
        },
        "EffectDistortion",
    );
    round_trip_tag(DistNodeKind::EffectEdge { strength: 1.0 }, "EffectEdge");
    round_trip_tag(
        DistNodeKind::EffectSmoothstep {
            edge0: 0.0,
            edge1: 1.0,
        },
        "EffectSmoothstep",
    );
    round_trip_tag(
        DistNodeKind::EffectSimpleFlow {
            iterations: 1,
            strength: 1.0,
        },
        "EffectSimpleFlow",
    );
    round_trip_tag(DistNodeKind::Paint { mask: mask.clone() }, "Paint");
    round_trip_tag(
        DistNodeKind::Polygon {
            points: vec![[0.0, 0.0]],
            soft: 0.0,
        },
        "Polygon",
    );
    round_trip_tag(
        DistNodeKind::Spline {
            points: vec![[0.0, 0.0]],
            width: 1.0,
        },
        "Spline",
    );
    round_trip_tag(
        DistNodeKind::Distance {
            mask: mask.clone(),
            max_distance: 1.0,
        },
        "Distance",
    );
    round_trip_tag(
        DistNodeKind::Climate {
            channel: ClimateMaskChannel::Temperature,
        },
        "Climate",
    );
    round_trip_tag(
        DistNodeKind::Voronoi {
            seed: 1,
            frequency: 0.1,
            edge_weight: 0.0,
        },
        "Voronoi",
    );
    round_trip_tag(DistNodeKind::ImportedMask { mask }, "ImportedMask");
    round_trip_tag(DistNodeKind::GroupAll, "GroupAll");
    round_trip_tag(DistNodeKind::GroupAny, "GroupAny");
    round_trip_tag(DistNodeKind::EffectDilate { radius_m: 1.0 }, "EffectDilate");
    round_trip_tag(DistNodeKind::EffectErode { radius_m: 1.0 }, "EffectErode");
    unit_tags!(ClimateMaskChannel::Temperature => "Temperature", ClimateMaskChannel::Rainfall => "Rainfall", ClimateMaskChannel::Humidity => "Humidity", ClimateMaskChannel::Snow => "Snow", ClimateMaskChannel::SoilMoisture => "SoilMoisture", ClimateMaskChannel::WindExposure => "WindExposure");
}

#[test]
fn nested_parameter_enum_tags_are_frozen() {
    unit_tags!(WorleyMetric::Euclidean => "Euclidean", WorleyMetric::Manhattan => "Manhattan", WorleyMetric::Chebyshev => "Chebyshev");
    unit_tags!(WorleyFeature::F1 => "F1", WorleyFeature::F2 => "F2", WorleyFeature::F2MinusF1 => "F2MinusF1");
    unit_tags!(FractalNoiseType::Value => "Value", FractalNoiseType::Perlin => "Perlin", FractalNoiseType::OpenSimplex => "OpenSimplex");
    unit_tags!(TransportModel::Hydraulic => "hydraulic", TransportModel::SoftFlows => "soft_flows", TransportModel::RidgedFlows => "ridged_flows", TransportModel::ThinFlows => "thin_flows", TransportModel::WideFlows => "wide_flows", TransportModel::SedimentFlows => "sediment_flows", TransportModel::HydraulicSediment => "hydraulic_sediment");
    unit_tags!(EvolutionSolverMode::Fast => "Fast", EvolutionSolverMode::Accurate => "Accurate");
    unit_tags!(UpliftMode::Uniform => "Uniform", UpliftMode::Radial => "Radial", UpliftMode::LinearBelt => "LinearBelt", UpliftMode::Painted => "Painted", UpliftMode::Procedural => "Procedural", UpliftMode::ShapeDerived => "ShapeDerived");
    unit_tags!(BoundaryMode::SeaLevel => "SeaLevel", BoundaryMode::Fixed => "Fixed", BoundaryMode::OpenDrainage => "OpenDrainage", BoundaryMode::OutletMask => "OutletMask");
    unit_tags!(HighDetailMode::None => "None", HighDetailMode::Zone => "Zone", HighDetailMode::Camera => "Camera");
    unit_tags!(StratumMaterial::Sedimentary => "sedimentary", StratumMaterial::Igneous => "igneous", StratumMaterial::Metamorphic => "metamorphic", StratumMaterial::Unconsolidated => "unconsolidated", StratumMaterial::Soil => "soil", StratumMaterial::Ice => "ice");
    round_trip_tag(BedGeometry::Horizontal, "horizontal");
    round_trip_tag(
        BedGeometry::Tilted {
            dip_deg: 10.0,
            azimuth_deg: 20.0,
        },
        "tilted",
    );
    round_trip_tag(
        BedGeometry::Folded {
            amplitude_m: 1.0,
            wavelength_m: 2.0,
            seed: 3,
        },
        "folded",
    );
    round_trip_tag(
        BedGeometry::Warped {
            frequency: 0.1,
            amplitude_m: 1.0,
            seed: 3,
        },
        "warped",
    );
    unit_tags!(IslandArchetype::VolcanicHighIsland => "VolcanicHighIsland", IslandArchetype::Archipelago => "Archipelago", IslandArchetype::Atoll => "Atoll");
    unit_tags!(ProceduralGenerator::Mountain => "mountain", ProceduralGenerator::Hills => "hills", ProceduralGenerator::Plateau => "plateau", ProceduralGenerator::Mesa => "mesa", ProceduralGenerator::Volcano => "volcano", ProceduralGenerator::Dunes => "dunes", ProceduralGenerator::Canyon => "canyon", ProceduralGenerator::Crater => "crater", ProceduralGenerator::Noise => "noise");
    unit_tags!(PolygonHeightMode::RaiseBy => "RaiseBy", PolygonHeightMode::SetElevation => "SetElevation");
}

#[test]
fn shape_and_placement_tags_are_frozen() {
    let sculpt_tags = [
        (SculptStrokeKind::Raise, "Raise"),
        (SculptStrokeKind::Lower, "Lower"),
        (SculptStrokeKind::Smooth, "Smooth"),
        (SculptStrokeKind::Flatten, "Flatten"),
        (SculptStrokeKind::Ridge, "Ridge"),
        (SculptStrokeKind::Valley, "Valley"),
        (SculptStrokeKind::Terrace, "Terrace"),
        (SculptStrokeKind::Roughness, "Roughness"),
        (SculptStrokeKind::Uplift, "Uplift"),
        (SculptStrokeKind::Hardness, "Hardness"),
        (SculptStrokeKind::Sediment, "Sediment"),
        (SculptStrokeKind::Protect, "Protect"),
        (SculptStrokeKind::EncourageErosion, "EncourageErosion"),
        (SculptStrokeKind::Pinch, "Pinch"),
        (SculptStrokeKind::Inflate, "Inflate"),
        (SculptStrokeKind::Erode, "Erode"),
        (SculptStrokeKind::Noise, "Noise"),
        (SculptStrokeKind::MountainStamp, "MountainStamp"),
        (SculptStrokeKind::ValleyStamp, "ValleyStamp"),
        (SculptStrokeKind::PlateauStamp, "PlateauStamp"),
        (SculptStrokeKind::CraterStamp, "CraterStamp"),
        (SculptStrokeKind::Coastline, "Coastline"),
        (SculptStrokeKind::RiverPath, "RiverPath"),
        (SculptStrokeKind::HeightStamp, "HeightStamp"),
    ];
    for (value, tag) in sculpt_tags {
        round_trip_tag(value, tag);
    }
    let constraint_tags = [
        (TerrainConstraintKind::Elevation, "Elevation"),
        (TerrainConstraintKind::MinElevation, "MinElevation"),
        (TerrainConstraintKind::MaxElevation, "MaxElevation"),
        (TerrainConstraintKind::Ridge, "Ridge"),
        (TerrainConstraintKind::Valley, "Valley"),
        (TerrainConstraintKind::River, "River"),
        (TerrainConstraintKind::Coastline, "Coastline"),
        (TerrainConstraintKind::Plateau, "Plateau"),
        (TerrainConstraintKind::Cliff, "Cliff"),
        (TerrainConstraintKind::PreferredSlope, "PreferredSlope"),
        (TerrainConstraintKind::Roughness, "Roughness"),
        (TerrainConstraintKind::Outlet, "Outlet"),
        (TerrainConstraintKind::Divide, "Divide"),
        (TerrainConstraintKind::Protect, "Protect"),
    ];
    for (value, tag) in constraint_tags {
        round_trip_tag(value, tag);
    }
    let shape_tags = [
        (ShapeKind::CoastlinePolygon, "CoastlinePolygon"),
        (ShapeKind::LandmassPolygon, "LandmassPolygon"),
        (ShapeKind::MountainSpine, "MountainSpine"),
        (ShapeKind::RidgeSpline, "RidgeSpline"),
        (ShapeKind::ValleySpline, "ValleySpline"),
        (ShapeKind::RiverPath, "RiverPath"),
        (ShapeKind::CanyonPath, "CanyonPath"),
        (ShapeKind::PlateauPolygon, "PlateauPolygon"),
        (ShapeKind::LakeBasin, "LakeBasin"),
        (ShapeKind::Volcano, "Volcano"),
        (ShapeKind::UpliftCentre, "UpliftCentre"),
        (ShapeKind::HeightStamp, "HeightStamp"),
    ];
    for (value, tag) in shape_tags {
        round_trip_tag(value, tag);
    }

    unit_tags!(ApplyWhere::EntireBiome => "entire_biome", ApplyWhere::PaintedRestriction => "painted_restriction", ApplyWhere::HeightRange => "height_range", ApplyWhere::SlopeRange => "slope_range", ApplyWhere::NearWater => "near_water", ApplyWhere::NearRivers => "near_rivers", ApplyWhere::FlowRange => "flow_range", ApplyWhere::Curvature => "curvature", ApplyWhere::CustomConditions => "custom_conditions", ApplyWhere::AdvancedMask => "advanced_mask");
    unit_tags!(DevelopCategory::Terrain => "terrain", DevelopCategory::Materials => "materials", DevelopCategory::Simulation => "simulation", DevelopCategory::Vegetation => "vegetation", DevelopCategory::Objects => "objects", DevelopCategory::Placement => "placement");
    unit_tags!(PlacementCoordinateSpace::WorldSpace => "world_space", PlacementCoordinateSpace::SurfaceSpace => "surface_space", PlacementCoordinateSpace::RuleBased => "rule_based");
    unit_tags!(PlacementSource::Rules => "rules", PlacementSource::Custom => "custom");
    unit_tags!(RuleGroupMode::All => "all", RuleGroupMode::Any => "any");
    unit_tags!(CompareOp::Above => "above", CompareOp::Below => "below", CompareOp::Between => "between");
    unit_tags!(ConditionChannel::Height => "height", ConditionChannel::Slope => "slope", ConditionChannel::Curvature => "curvature", ConditionChannel::Flow => "flow", ConditionChannel::WaterDistance => "water_distance", ConditionChannel::CoastDistance => "coast_distance", ConditionChannel::Temperature => "temperature", ConditionChannel::Rainfall => "rainfall", ConditionChannel::WindExposure => "wind_exposure", ConditionChannel::Noise => "noise");

    let mask = MaskId::nil();
    round_trip_internal_tag(CoverageTerm::PaintedWorld { mask }, "painted_world");
    round_trip_internal_tag(CoverageTerm::PaintedSurface { mask }, "painted_surface");
    round_trip_internal_tag(
        CoverageTerm::Polygon {
            points: vec![],
            soft: 0.0,
        },
        "polygon",
    );
    round_trip_internal_tag(
        CoverageTerm::Spline {
            points: vec![],
            width: 1.0,
        },
        "spline",
    );
    round_trip_internal_tag(CoverageTerm::ImportedMask { mask }, "imported_mask");
    round_trip_internal_tag(RuleNode::Group(RuleGroup::default()), "group");
    round_trip_internal_tag(
        RuleNode::Condition(Condition {
            channel: ConditionChannel::Height,
            op: CompareOp::Above,
            a: 1.0,
            b: 0.0,
            falloff: 0.0,
            seed: 0,
            frequency: 0.1,
        }),
        "condition",
    );
    round_trip_internal_tag(RuleNode::CoverageRef { index: 0 }, "coverage_ref");
    round_trip_internal_tag(
        PlacementRefinement::ExcludeBiome {
            biome_group: Default::default(),
            mask: None,
        },
        "exclude_biome",
    );
    round_trip_internal_tag(
        PlacementRefinement::ExcludeRiverCorridor { flow_min: 0.1 },
        "exclude_river_corridor",
    );
    round_trip_internal_tag(PlacementRefinement::ExcludeRoads { mask }, "exclude_roads");
    round_trip_internal_tag(PlacementRefinement::Expand { radius_m: 1.0 }, "expand");
    round_trip_internal_tag(PlacementRefinement::Contract { radius_m: 1.0 }, "contract");
    round_trip_internal_tag(PlacementRefinement::Smooth { radius_samples: 1 }, "smooth");
    round_trip_internal_tag(
        PlacementRefinement::BoundaryNoise {
            seed: 1,
            frequency: 0.1,
            amount: 1.0,
        },
        "boundary_noise",
    );
    round_trip_internal_tag(
        PlacementRefinement::Falloff {
            edge0: 0.0,
            edge1: 1.0,
        },
        "falloff",
    );
    unit_tags!(PlacementCombineMode::PaintOnly => "PaintOnly", PlacementCombineMode::RulesOnly => "RulesOnly", PlacementCombineMode::PaintMulRules => "PaintMulRules", PlacementCombineMode::PaintAddRules => "PaintAddRules", PlacementCombineMode::PaintOverridesRules => "PaintOverridesRules", PlacementCombineMode::RulesOutsidePaint => "RulesOutsidePaint");
    unit_tags!(BiomeOverlapPolicy::HeightDelta => "HeightDelta", BiomeOverlapPolicy::PriorityOverride => "PriorityOverride", BiomeOverlapPolicy::NormalizedBlend => "NormalizedBlend");
}

#[test]
fn world_rule_and_scenario_tags_are_frozen() {
    round_trip_tag(WorldRuleScope::EntireWorld, "entire_world");
    round_trip_tag(WorldRuleScope::SelectedBiomes(vec![]), "selected_biomes");
    round_trip_tag(WorldRuleScope::Excluding { biomes: vec![] }, "excluding");
    round_trip_tag(
        WorldRuleScope::PaintedRestriction { paint_mask: None },
        "painted_restriction",
    );
    unit_tags!(WorldRulePhase::BeforePhysics => "before_physics", WorldRulePhase::PhysicsInput => "physics_input", WorldRulePhase::AfterPhysics => "after_physics", WorldRulePhase::Materials => "materials", WorldRulePhase::Scatter => "scatter", WorldRulePhase::Objects => "objects");
    unit_tags!(WorldRuleEffectKind::TerrainEffect => "terrain_effect", WorldRuleEffectKind::Material => "material", WorldRuleEffectKind::SimulationInput => "simulation_input", WorldRuleEffectKind::Scatter => "scatter", WorldRuleEffectKind::ObjectExclusion => "object_exclusion", WorldRuleEffectKind::BiomeInfluence => "biome_influence", WorldRuleEffectKind::GeneratedOutputMap => "generated_output_map");

    round_trip_tag(ScenarioScope::World, "world");
    round_trip_tag(ScenarioScope::SelectedBiomes(vec![]), "selected_biomes");
    round_trip_tag(
        ScenarioScope::PaintedDomain { paint_mask: None },
        "painted_domain",
    );
    round_trip_tag(
        ScenarioScope::CustomConditions {
            placement: Default::default(),
        },
        "custom_conditions",
    );
    round_trip_tag(MatterSourceKind::Rainfall, "rainfall");
    round_trip_tag(MatterSourceKind::PaintedSprings, "painted_springs");
    round_trip_tag(MatterSourceKind::Snowmelt, "snowmelt");
    round_trip_tag(MatterSourceKind::RiverInflow, "river_inflow");
    round_trip_tag(MatterSourceKind::WindSediment, "wind_sediment");
    round_trip_tag(
        MatterSourceKind::Custom {
            label: "custom".into(),
        },
        "custom",
    );
    round_trip_tag(SimulationDomain::InheritScope, "inherit_scope");
    round_trip_tag(SimulationDomain::EntireWorld, "entire_world");
    round_trip_tag(
        SimulationDomain::NamedRegion {
            name: "region".into(),
        },
        "named_region",
    );
    round_trip_tag(SimulationDomain::Painted { paint_mask: None }, "painted");
    round_trip_tag(
        SimulationDomain::CustomConditions {
            placement: Default::default(),
        },
        "custom_conditions",
    );
    round_trip_tag(OutputInfluence::EntireDomain, "entire_domain");
    round_trip_tag(OutputInfluence::InheritScope, "inherit_scope");
    round_trip_tag(OutputInfluence::Painted { paint_mask: None }, "painted");
    round_trip_tag(
        OutputInfluence::CustomConditions {
            placement: Default::default(),
        },
        "custom_conditions",
    );
    let pass_tags = [
        (ScenarioPassKind::Flow, "flow"),
        (ScenarioPassKind::RiverExtraction, "river_extraction"),
        (ScenarioPassKind::HydraulicErosion, "hydraulic_erosion"),
        (ScenarioPassKind::ThermalErosion, "thermal_erosion"),
        (ScenarioPassKind::DebrisFlow, "debris_flow"),
        (ScenarioPassKind::StreamPower, "stream_power"),
        (ScenarioPassKind::Sediment, "sediment"),
        (ScenarioPassKind::LandscapeEvolution, "landscape_evolution"),
        (ScenarioPassKind::HydrologyRepair, "hydrology_repair"),
        (ScenarioPassKind::Sand, "sand"),
        (ScenarioPassKind::Fluid, "fluid"),
        (ScenarioPassKind::Coastal, "coastal"),
        (ScenarioPassKind::RiverCarve, "river_carve"),
        (ScenarioPassKind::EcosystemFeedback, "ecosystem_feedback"),
        (ScenarioPassKind::GeomorphicDetail, "geomorphic_detail"),
        (ScenarioPassKind::MultiScaleAmplify, "multi_scale_amplify"),
    ];
    for (kind, tag) in pass_tags {
        round_trip_tag(kind, tag);
    }
    unit_tags!(ScenarioQuality::Draft => "draft", ScenarioQuality::Medium => "medium", ScenarioQuality::High => "high", ScenarioQuality::Final => "final");
    unit_tags!(ScenarioResultState::Ready => "ready", ScenarioResultState::Running => "running", ScenarioResultState::Current => "current", ScenarioResultState::Outdated => "outdated", ScenarioResultState::Frozen => "frozen", ScenarioResultState::Failed => "failed", ScenarioResultState::Cancelled => "cancelled");
    unit_tags!(MatterType::WaterRivers => "water_rivers", MatterType::Snow => "snow", MatterType::Sand => "sand", MatterType::Debris => "debris");
    round_trip_tag(MatterArtistSource::Rainfall { strength: 1.0 }, "rainfall");
    round_trip_tag(
        MatterArtistSource::PaintedSprings { mask: None },
        "painted_springs",
    );
    round_trip_tag(MatterArtistSource::SourceSplines, "source_splines");
    round_trip_tag(MatterArtistSource::Lakes, "lakes");
    round_trip_tag(MatterArtistSource::Snowmelt { strength: 1.0 }, "snowmelt");
    round_trip_tag(
        MatterArtistSource::ImportedFlow { mask: None },
        "imported_flow",
    );
    round_trip_tag(MatterArtistSource::Snowfall { strength: 1.0 }, "snowfall");
    round_trip_tag(
        MatterArtistSource::PaintedSnow { mask: None },
        "painted_snow",
    );
    round_trip_tag(MatterArtistSource::ClimateRule, "climate_rule");
    round_trip_tag(
        MatterArtistSource::PaintedSand { mask: None },
        "painted_sand",
    );
    round_trip_tag(MatterArtistSource::CoastalSediment, "coastal_sediment");
    round_trip_tag(MatterArtistSource::DesertBiome, "desert_biome");
    round_trip_tag(
        MatterArtistSource::ImportedMap { mask: None },
        "imported_map",
    );
    round_trip_tag(
        MatterArtistSource::SteepSlopes { min_deg: 35.0 },
        "steep_slopes",
    );
    round_trip_tag(MatterArtistSource::ErodedCliffs, "eroded_cliffs");
    round_trip_tag(
        MatterArtistSource::PaintedUnstable { mask: None },
        "painted_unstable",
    );
    round_trip_tag(
        MatterArtistSource::HydraulicUndercutting,
        "hydraulic_undercutting",
    );
}

#[test]
fn effect_filter_canonical_and_pascal_case_aliases_are_frozen() {
    let canonical = [
        "smooth",
        "distortion",
        "spike_removal",
        "shore",
        "strata",
        "crater",
        "denoise",
        "rocky_sharp",
        "rocky_wide",
        "rocky_layers",
        "cliff_reinforce",
        "soft_flows",
        "thin_flows",
        "ridged_flows",
        "wide_flows",
        "talus_fill",
        "sediment_fill_soft",
        "mud_settle",
        "hydraulic_sediment",
        "rocky_plateaus",
        "rocky_cliffs",
        "rocky_hard",
        "canyon",
        "chipped",
        "cliffs",
        "rocky",
        "sediment_flows",
        "angle_break",
        "wind_carve",
        "inflate",
        "deflate",
        "balloon",
        "blocks",
        "ridged",
        "rugged",
        "smooth_ridges",
        "angle_blur",
        "directional_blur",
        "squeeze",
        "swirl",
        "washed_off",
        "hexagons",
        "scatter_detail",
        "flatten_filter",
        "zero_edge",
        "border_blend",
        "curve",
        "cutoff",
        "kuwahara",
        "terrace_simple",
        "terrace_irregular",
        "terrace_steep",
        "add_set",
        "design_voronoi",
        "noise_billow",
        "noise_gabor",
        "noise_perlin",
        "noise_phasor",
        "noise_ridged",
        "noise_simplex",
        "noise_value",
        "noise_voronoi",
        "noise_wave",
        "noise_white",
    ];
    let actual: Vec<_> = EffectFilterKind::ALL
        .iter()
        .map(|kind| {
            serde_json::to_value(kind)
                .unwrap()
                .as_str()
                .unwrap()
                .to_owned()
        })
        .collect();
    assert_eq!(actual, canonical);

    for alias in [
        "Smooth",
        "Distortion",
        "SpikeRemoval",
        "Shore",
        "Strata",
        "Crater",
        "Denoise",
        "RockySharp",
        "RockyWide",
        "RockyLayers",
        "CliffReinforce",
        "SoftFlows",
        "ThinFlows",
        "RidgedFlows",
        "WideFlows",
        "TalusFill",
        "SedimentFillSoft",
        "MudSettle",
        "HydraulicSediment",
    ] {
        let _: EffectFilterKind = serde_json::from_value(json!(alias))
            .unwrap_or_else(|error| panic!("historical EffectFilterKind tag {alias}: {error}"));
    }
}

#[test]
fn representative_field_id_tags_are_frozen() {
    unit_tags!(FieldId::Height => "Height", FieldId::BedrockHeight => "BedrockHeight", FieldId::FlowAccumulation => "FlowAccumulation", FieldId::Sediment => "Sediment", FieldId::Temperature => "Temperature", FieldId::Materials => "Materials", FieldId::Biomes => "Biomes", FieldId::Vegetation => "Vegetation", FieldId::DebrisDepth => "DebrisDepth", FieldId::FineErosion => "FineErosion");
}
