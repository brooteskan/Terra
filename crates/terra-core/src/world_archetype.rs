//! Editable New World templates — Phase 11 cause→effect stacks.
//!
//! Presets are parameter sets over real algorithms (see [`crate::landscape_style`]).
//! Do not bake unique hardcoded generators per look.

use crate::authoring::{GradientReconstructParams, HydrologyRepairParams, SculptPoint};
use crate::biome_definition::BiomeLibrary;
use crate::biome_paint::BiomeLayer;
use crate::document::TerrainDocument;
use crate::heightfield::HeightfieldMetrics;
use crate::landscape_blueprint::LandscapeBlueprint;
use crate::landscape_style::{LandscapeStyle, LandscapeStyleParams};
use crate::layer::{
    CanyonParams, CoastalParams, DuneParams, FlatParams, IslandArchetype, IslandParams, Layer,
    LayerKind, MaterialsParams, MesaParams, MountainParams, NoiseParams, RiverCarveParams,
    SandSimParams, SculptParams,
};
use crate::mask::MaskSource;
use crate::shape_object::{ShapeKind, ShapeObject, ShapeObjectStore};

/// Stable New World template identity (matches project template ids).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorldTemplate {
    Blank,
    TropicalIsland,
    Alpine,
    Desert,
    RiverValley,
    Badlands,
    YoungMountains,
    OldMountains,
    DuneField,
    Coastal,
}

impl WorldTemplate {
    pub fn all() -> &'static [WorldTemplate] {
        &[
            Self::Blank,
            Self::TropicalIsland,
            Self::Alpine,
            Self::Desert,
            Self::RiverValley,
            Self::Badlands,
            Self::YoungMountains,
            Self::OldMountains,
            Self::DuneField,
            Self::Coastal,
        ]
    }

    pub fn id(self) -> &'static str {
        match self {
            Self::Blank => "blank",
            Self::TropicalIsland => "tropical_island",
            Self::Alpine => "alpine",
            Self::Desert => "desert",
            Self::RiverValley => "river_valley",
            Self::Badlands => "badlands",
            Self::YoungMountains => "young_mountains",
            Self::OldMountains => "old_mountains",
            Self::DuneField => "dune_field",
            Self::Coastal => "coastal",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Blank => "Blank",
            Self::TropicalIsland => "Tropical Island",
            Self::Alpine => "Alpine Range",
            Self::Desert => "Desert Mesa",
            Self::RiverValley => "River Valley",
            Self::Badlands => "Badlands",
            Self::YoungMountains => "Young Mountains",
            Self::OldMountains => "Old Mountains",
            Self::DuneField => "Dune Field",
            Self::Coastal => "Coastal",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Blank => "Scaffold only: blueprint, shapes, biome library — no baked evolution.",
            Self::TropicalIsland => {
                "Island landmass → evolution → shore profile → geomorphic detail."
            }
            Self::Alpine => "Uplift → materials → landscape evolution → debris → talus → detail.",
            Self::Desert => "Mesa + canyon → layered strata → evolution → thermal talus → detail.",
            Self::RiverValley => {
                "Valley uplift → SPE → floodplain fill → river carve → coastal edge."
            }
            Self::Badlands => "Soft mesa → soft strata → dense SPE → thermal → detail.",
            Self::YoungMountains => {
                "Strong uplift → short evolution → immature SPE → debris → detail."
            }
            Self::OldMountains => "Broad uplift → long evolution → mild SPE → deposition → detail.",
            Self::DuneField => "Hard floor → substrate → dunes → sand transport → detail.",
            Self::Coastal => "Coastal landmass → uplift → SPE to sea → shore → detail.",
        }
    }

    pub fn style(self) -> Option<LandscapeStyle> {
        match self {
            Self::Blank => None,
            Self::TropicalIsland => Some(LandscapeStyle::TropicalIsland),
            Self::Alpine => Some(LandscapeStyle::Alpine),
            Self::Desert => Some(LandscapeStyle::Mesa),
            Self::RiverValley => Some(LandscapeStyle::RiverValley),
            Self::Badlands => Some(LandscapeStyle::Badlands),
            Self::YoungMountains => Some(LandscapeStyle::YoungMountains),
            Self::OldMountains => Some(LandscapeStyle::OldMountains),
            Self::DuneField => Some(LandscapeStyle::DuneField),
            Self::Coastal => Some(LandscapeStyle::Coastal),
        }
    }

    fn blueprint(self, world_size_m: f32) -> LandscapeBlueprint {
        match self {
            Self::Blank => {
                let mut b = LandscapeBlueprint::default();
                b.world_size_m = world_size_m;
                b.metres_per_sample = (world_size_m / 2048.0).max(0.5);
                b
            }
            Self::TropicalIsland => LandscapeBlueprint::tropical_island(world_size_m),
            Self::Alpine => LandscapeBlueprint::alpine(world_size_m),
            Self::Desert => LandscapeBlueprint::desert(world_size_m),
            Self::RiverValley => LandscapeBlueprint::river_valley(world_size_m),
            Self::Badlands => LandscapeBlueprint::badlands(world_size_m),
            Self::YoungMountains => LandscapeBlueprint::young_mountains(world_size_m),
            Self::OldMountains => LandscapeBlueprint::old_mountains(world_size_m),
            Self::DuneField => LandscapeBlueprint::dune_field(world_size_m),
            Self::Coastal => LandscapeBlueprint::coastal(world_size_m),
        }
    }

    /// Build an editable World Design document for this template.
    pub fn build(self, world_size_m: f32, preview_res: u32) -> TerrainDocument {
        build_world(self, world_size_m, preview_res)
    }
}

/// Build any New World template from a shared cause→effect recipe.
pub fn build_world(
    template: WorldTemplate,
    world_size_m: f32,
    preview_res: u32,
) -> TerrainDocument {
    let world_size_m = match template {
        WorldTemplate::Blank => world_size_m.clamp(1024.0, 200_000.0),
        _ => world_size_m.clamp(2048.0, 200_000.0),
    };
    let preview_res = preview_res.clamp(128, 8192);
    let mps = (world_size_m / preview_res as f32).max(1.0);

    let mut doc = scaffold_doc(template, world_size_m, preview_res, mps);
    clear_shape_category(&mut doc);

    match template {
        WorldTemplate::Blank => {
            push_blank_shapes(&mut doc);
            finish_biomes(&mut doc, BiomeLibrary::default_world_palette(), mps, false);
            for def in &mut doc.biome_library.definitions {
                def.placement.combine =
                    crate::biome_definition::PlacementCombineMode::PaintOverridesRules;
            }
            return doc;
        }
        WorldTemplate::TropicalIsland => push_macro_tropical_island(&mut doc),
        WorldTemplate::Alpine => push_macro_alpine(&mut doc),
        WorldTemplate::Desert => push_macro_desert(&mut doc),
        WorldTemplate::RiverValley => push_macro_river_valley(&mut doc),
        WorldTemplate::Badlands => push_macro_badlands(&mut doc),
        WorldTemplate::YoungMountains => push_macro_young_mountains(&mut doc),
        WorldTemplate::OldMountains => push_macro_old_mountains(&mut doc),
        WorldTemplate::DuneField => push_macro_dune_field(&mut doc),
        WorldTemplate::Coastal => push_macro_coastal(&mut doc),
    }

    let style = template
        .style()
        .expect("non-blank templates have a LandscapeStyle")
        .params();
    sync_blueprint_from_style(&mut doc, &style);
    push_process_chain(&mut doc, template, &style);

    let biomes = match template {
        WorldTemplate::TropicalIsland => BiomeLibrary::tropical_island_palette(),
        WorldTemplate::Alpine | WorldTemplate::YoungMountains | WorldTemplate::OldMountains => {
            BiomeLibrary::alpine_palette()
        }
        WorldTemplate::Desert | WorldTemplate::Badlands | WorldTemplate::DuneField => {
            BiomeLibrary::desert_palette()
        }
        WorldTemplate::RiverValley | WorldTemplate::Coastal => BiomeLibrary::river_valley_palette(),
        WorldTemplate::Blank => unreachable!(),
    };
    finish_biomes(
        &mut doc,
        biomes,
        mps,
        matches!(template, WorldTemplate::Alpine),
    );
    doc
}

// —— Public thin wrappers (stable API) ————————————————————————————————

pub fn blank_world_design(world_size_m: f32, preview_res: u32) -> TerrainDocument {
    WorldTemplate::Blank.build(world_size_m, preview_res)
}

pub fn tropical_island_world(world_size_m: f32, preview_res: u32) -> TerrainDocument {
    WorldTemplate::TropicalIsland.build(world_size_m, preview_res)
}

pub fn alpine_world(world_size_m: f32, preview_res: u32) -> TerrainDocument {
    WorldTemplate::Alpine.build(world_size_m, preview_res)
}

pub fn desert_world(world_size_m: f32, preview_res: u32) -> TerrainDocument {
    WorldTemplate::Desert.build(world_size_m, preview_res)
}

pub fn river_valley_world(world_size_m: f32, preview_res: u32) -> TerrainDocument {
    WorldTemplate::RiverValley.build(world_size_m, preview_res)
}

pub fn badlands_world(world_size_m: f32, preview_res: u32) -> TerrainDocument {
    WorldTemplate::Badlands.build(world_size_m, preview_res)
}

pub fn young_mountains_world(world_size_m: f32, preview_res: u32) -> TerrainDocument {
    WorldTemplate::YoungMountains.build(world_size_m, preview_res)
}

pub fn old_mountains_world(world_size_m: f32, preview_res: u32) -> TerrainDocument {
    WorldTemplate::OldMountains.build(world_size_m, preview_res)
}

pub fn dune_field_world(world_size_m: f32, preview_res: u32) -> TerrainDocument {
    WorldTemplate::DuneField.build(world_size_m, preview_res)
}

pub fn coastal_world(world_size_m: f32, preview_res: u32) -> TerrainDocument {
    WorldTemplate::Coastal.build(world_size_m, preview_res)
}

// —— Scaffold / shared helpers ————————————————————————————————————————

fn scaffold_doc(
    template: WorldTemplate,
    world_size_m: f32,
    preview_res: u32,
    mps: f32,
) -> TerrainDocument {
    let mut doc = TerrainDocument::new_default();
    doc.name = template.label().into();
    if matches!(template, WorldTemplate::Blank) {
        doc.name = "Untitled World".into();
    }
    doc.metrics = HeightfieldMetrics::new(preview_res, preview_res, world_size_m, world_size_m);
    doc.preview_resolution = preview_res;
    doc.export_resolution = (world_size_m / mps).round().clamp(1024.0, 8192.0) as u32;
    doc.blueprint = template.blueprint(world_size_m);
    doc.presets_used.push(template.id().into());
    doc
}

fn clear_shape_category(doc: &mut TerrainDocument) {
    if let Some(shape) = doc
        .stack
        .find_category_mut(crate::layer::StackCategory::Shape)
    {
        shape.children.clear();
    }
}

fn sync_blueprint_from_style(doc: &mut TerrainDocument, style: &LandscapeStyleParams) {
    doc.blueprint.geological_age = style.geological_age;
    doc.blueprint.rainfall = style.rainfall;
    doc.blueprint.drainage_density = style.drainage_density;
    doc.blueprint.ridge_sharpness = style.ridge_sharpness;
}

fn scale_ridge_shapes(doc: &mut TerrainDocument) {
    let ridge_w = doc.blueprint.ridge_width_m();
    for shape in &mut doc.shapes.shapes {
        if matches!(
            shape.kind,
            ShapeKind::MountainSpine | ShapeKind::RidgeSpline
        ) {
            shape.width_m = ridge_w;
        }
    }
}

fn push_gradient_reconstruct(doc: &mut TerrainDocument, iterations: u32, strength: f32) {
    doc.stack.push_into_category(Layer::new(
        "Seamless Constraint Reconstruction",
        LayerKind::GradientReconstruct(GradientReconstructParams {
            iterations,
            screening: 0.12,
            constraint_strength: strength,
            gradient_smoothing: 0.2,
        }),
    ));
}

/// Shared process chain: geology → evolution → hydro repair → meso extras → micro detail.
fn push_process_chain(
    doc: &mut TerrainDocument,
    template: WorldTemplate,
    style: &LandscapeStyleParams,
) {
    // Geology before erosion so hardness drives incision / talus.
    if !matches!(template, WorldTemplate::DuneField) {
        // Dune field pushes substrate in the macro stage.
        doc.stack.push_into_category(Layer::new(
            "Geology / Hardness",
            LayerKind::Materials(style.materials.clone()),
        ));
    }

    let mut evo = style.evolution.clone();
    evo.iterations = evo.iterations.max(doc.blueprint.evolution_iterations());
    evo.geological_age = doc.blueprint.geological_age_norm();
    evo.rainfall = doc.blueprint.rainfall_scale();
    evo.drainage_scale = doc.blueprint.drainage_density.clamp(0.0, 1.0);
    doc.stack.push_into_category(Layer::new(
        "Landscape Evolution",
        LayerKind::LandscapeEvolution(evo),
    ));
    doc.stack.push_into_category(Layer::new(
        "Hydrology Repair",
        LayerKind::HydrologyRepair(HydrologyRepairParams::default()),
    ));

    match template {
        WorldTemplate::Alpine | WorldTemplate::YoungMountains => {
            if matches!(template, WorldTemplate::YoungMountains) {
                doc.stack.push_into_category(Layer::new(
                    "Immature Incision",
                    LayerKind::StreamPowerErosion(style.stream_power.clone()),
                ));
            }
            doc.stack.push_into_category(Layer::new(
                "Debris Scars",
                LayerKind::DebrisFlow(style.debris.clone()),
            ));
            doc.stack.push_into_category(Layer::new(
                "Talus Weathering",
                LayerKind::ThermalErosion(style.thermal.clone()),
            ));
        }
        WorldTemplate::Desert | WorldTemplate::Badlands | WorldTemplate::OldMountains => {
            if matches!(template, WorldTemplate::OldMountains) {
                doc.stack.push_into_category(Layer::new(
                    "Mature Stream Power",
                    LayerKind::StreamPowerErosion(style.stream_power.clone()),
                ));
                doc.stack.push_into_category(Layer::new(
                    "Valley Fill",
                    LayerKind::HydraulicErosion(style.hydraulic.clone()),
                ));
            } else if matches!(template, WorldTemplate::Badlands) {
                doc.stack.push_into_category(Layer::new(
                    "Dense Drainage Incision",
                    LayerKind::StreamPowerErosion(style.stream_power.clone()),
                ));
            }
            doc.stack.push_into_category(Layer::new(
                "Differential / Talus Weathering",
                LayerKind::ThermalErosion(style.thermal.clone()),
            ));
        }
        WorldTemplate::RiverValley => {
            doc.stack.push_into_category(Layer::new(
                "Stream Power",
                LayerKind::StreamPowerErosion(style.stream_power.clone()),
            ));
            doc.stack.push_into_category(Layer::new(
                "Valley Fill",
                LayerKind::HydraulicErosion(style.hydraulic.clone()),
            ));
            doc.stack.push_into_category(Layer::new(
                "River Carve",
                LayerKind::RiverCarve(RiverCarveParams {
                    accumulation_threshold: 28.0,
                    depth: 36.0,
                    width: 7.0,
                    guide: MaskSource::Wetness,
                    guide_boost: 3.5,
                    ..RiverCarveParams::default()
                }),
            ));
            doc.stack.push_into_category(Layer::new(
                "Coastal Edge",
                LayerKind::Coastal(CoastalParams {
                    sea_level: doc.blueprint.sea_level,
                    ..CoastalParams::default()
                }),
            ));
        }
        WorldTemplate::TropicalIsland | WorldTemplate::Coastal => {
            if matches!(template, WorldTemplate::Coastal) {
                doc.stack.push_into_category(Layer::new(
                    "Drainage to Sea",
                    LayerKind::StreamPowerErosion(style.stream_power.clone()),
                ));
            }
            doc.stack.push_into_category(Layer::new(
                "Shore Profile",
                LayerKind::Coastal(CoastalParams {
                    sea_level: doc.blueprint.sea_level,
                    ..CoastalParams::default()
                }),
            ));
        }
        WorldTemplate::DuneField => {
            // Macro already placed dunes + sand sim; keep light thermal only.
            doc.stack.push_into_category(Layer::new(
                "Sand Avalanche",
                LayerKind::ThermalErosion(style.thermal.clone()),
            ));
        }
        WorldTemplate::Blank => {}
    }

    doc.stack.push_into_category(Layer::new(
        "Geomorphic Detail",
        LayerKind::GeomorphicDetail(style.geomorphic_detail.clone()),
    ));
}

fn install_biome_library(doc: &mut TerrainDocument, lib: BiomeLibrary) {
    doc.biome_library = lib;
    let mut placement = BiomeLayer::new("Primary Biome Placement");
    placement.show_biome_colors = true;
    let placement_id = placement.id;
    doc.biome_layers = vec![placement];
    doc.selected_biome_layer = Some(placement_id);

    if let Some(surface) = doc
        .stack
        .find_category_mut(crate::layer::StackCategory::Surface)
    {
        surface.children.retain(|n| {
            !matches!(
                n,
                crate::layer::StackNode::Group(g)
                    if g.is_biome() && (g.name == "Default Biome" || g.name == "Global")
            )
        });
    }
    for def in &mut doc.biome_library.definitions {
        let group = def.to_group();
        let gid = group.id;
        def.group_id = Some(gid);
        if let Some(surface) = doc
            .stack
            .find_category_mut(crate::layer::StackCategory::Surface)
        {
            surface.children.push(crate::layer::StackNode::Group(group));
        }
    }
    if let Some(first) = doc.biome_library.definitions.first() {
        doc.biome_library.selected = Some(first.id);
        doc.active_biome = first.group_id;
    }
}

fn finish_biomes(doc: &mut TerrainDocument, biomes: BiomeLibrary, mps: f32, hd_outline: bool) {
    install_biome_library(doc, biomes);
    doc.sparse_paint = crate::sparse_paint::SparsePaintStore::new(mps, 256);
    doc.level_steps.show_hd_outline = hd_outline;
    doc.normalize_wc_tree();
}

// —— Macro stages ————————————————————————————————————————————————————

fn push_blank_shapes(doc: &mut TerrainDocument) {
    let mut landmass = ShapeObject::new("Initial Landmass", ShapeKind::LandmassPolygon);
    landmass.points = vec![
        SculptPoint {
            u: 0.2,
            v: 0.2,
            pressure: 1.0,
        },
        SculptPoint {
            u: 0.8,
            v: 0.2,
            pressure: 1.0,
        },
        SculptPoint {
            u: 0.8,
            v: 0.8,
            pressure: 1.0,
        },
        SculptPoint {
            u: 0.2,
            v: 0.8,
            pressure: 1.0,
        },
    ];
    landmass.width_m = 80.0;
    let mut uplift = ShapeObject::new("Initial Uplift", ShapeKind::UpliftCentre);
    uplift.points = vec![SculptPoint {
        u: 0.5,
        v: 0.5,
        pressure: 1.0,
    }];
    uplift.width_m = 400.0;
    uplift.value = 40.0;
    doc.shapes.push(landmass);
    doc.shapes.push(uplift);
    doc.compile_shapes_into_stack();
    push_gradient_reconstruct(doc, 48, 6.0);

    if let Some(base) = doc
        .stack
        .flatten_layers_mut()
        .into_iter()
        .find(|l| l.kind.is_sculpt_base())
    {
        *base = Layer::new(
            "Base",
            LayerKind::SculptBase(SculptParams::filled(doc.preview_resolution.min(512), 8.0)),
        );
    }
}

fn push_macro_tropical_island(doc: &mut TerrainDocument) {
    doc.stack.push_into_category(Layer::new(
        "Island Landmass",
        LayerKind::Island(IslandParams {
            seed: 63,
            archetype: IslandArchetype::VolcanicHighIsland,
            center_u: 0.49,
            center_v: 0.51,
            rotation_deg: 24.0,
            radius: 0.66,
            aspect: 1.22,
            sea_level: doc.blueprint.sea_level,
            ocean_floor: -240.0,
            mountain_height: 540.0,
            shelf_width: 360.0,
            shelf_depth: 48.0,
            beach_width: 85.0,
            beach_height: 7.0,
            reef_width: 190.0,
            reef_depth: 4.0,
            coastline_warp: 0.22,
            coastline_frequency: 0.00115,
            mountain_power: 1.62,
            ridge_strength: 0.52,
            ridge_frequency: 0.0027,
            lagoon_radius: 0.42,
        }),
    ));
    doc.shapes = ShapeObjectStore::tropical_island_shapes();
    scale_ridge_shapes(doc);
    doc.compile_shapes_into_stack();
    push_gradient_reconstruct(doc, 64, 8.5);
}

fn push_macro_alpine(doc: &mut TerrainDocument) {
    let style = LandscapeStyle::Alpine.params();
    doc.stack.push_into_category(Layer::new(
        "Mountain Range",
        LayerKind::Mountains(MountainParams {
            base: NoiseParams {
                seed: 41,
                frequency: 0.0012,
                amplitude: 780.0,
                octaves: 5,
                ..NoiseParams::default()
            },
            ridge_sharpness: 2.1,
            range_angle: 0.55,
            range_width: 0.4,
            crest_detail: 55.0,
        }),
    ));
    doc.stack.push_into_category(Layer::new(
        "Uplift Corridors",
        LayerKind::Uplift(style.uplift.clone()),
    ));
    doc.shapes = ShapeObjectStore::alpine_shapes();
    scale_ridge_shapes(doc);
    doc.compile_shapes_into_stack();
    push_gradient_reconstruct(doc, 48, 7.5);
}

fn push_macro_desert(doc: &mut TerrainDocument) {
    doc.stack.push_into_category(Layer::new(
        "Mesa Landmass",
        LayerKind::Mesa(MesaParams::default()),
    ));
    doc.stack.push_into_category(Layer::new(
        "Canyons",
        LayerKind::Canyons(CanyonParams::default()),
    ));
    doc.shapes = ShapeObjectStore::desert_shapes();
    doc.compile_shapes_into_stack();
    push_gradient_reconstruct(doc, 48, 6.0);
}

fn push_macro_river_valley(doc: &mut TerrainDocument) {
    let style = LandscapeStyle::RiverValley.params();
    doc.stack.push_into_category(Layer::new(
        "Valley Uplift",
        LayerKind::Uplift(style.uplift.clone()),
    ));
    doc.shapes = ShapeObjectStore::river_valley_shapes();
    doc.compile_shapes_into_stack();
    push_gradient_reconstruct(doc, 48, 6.0);
}

fn push_macro_badlands(doc: &mut TerrainDocument) {
    doc.stack.push_into_category(Layer::new(
        "Soft Plateau",
        LayerKind::Mesa(MesaParams {
            height: 180.0,
            edge_steepness: 2.8,
            soft: 0.18,
            ..MesaParams::default()
        }),
    ));
    push_gradient_reconstruct(doc, 40, 5.5);
}

fn push_macro_young_mountains(doc: &mut TerrainDocument) {
    let style = LandscapeStyle::YoungMountains.params();
    doc.stack.push_into_category(Layer::new(
        "Sharp Range",
        LayerKind::Mountains(MountainParams {
            base: NoiseParams {
                seed: 7,
                frequency: 0.0016,
                amplitude: 920.0,
                octaves: 6,
                ..NoiseParams::default()
            },
            ridge_sharpness: 3.2,
            range_width: 0.22,
            crest_detail: 80.0,
            ..MountainParams::default()
        }),
    ));
    doc.stack.push_into_category(Layer::new(
        "Active Uplift",
        LayerKind::Uplift(style.uplift.clone()),
    ));
    push_gradient_reconstruct(doc, 40, 7.0);
}

fn push_macro_old_mountains(doc: &mut TerrainDocument) {
    let style = LandscapeStyle::OldMountains.params();
    doc.stack.push_into_category(Layer::new(
        "Worn Massif",
        LayerKind::Mountains(MountainParams {
            base: NoiseParams {
                seed: 19,
                frequency: 0.0009,
                amplitude: 320.0,
                octaves: 4,
                ..NoiseParams::default()
            },
            ridge_sharpness: 0.85,
            range_width: 0.55,
            crest_detail: 18.0,
            ..MountainParams::default()
        }),
    ));
    doc.stack.push_into_category(Layer::new(
        "Broad Uplift",
        LayerKind::Uplift(style.uplift.clone()),
    ));
    push_gradient_reconstruct(doc, 48, 6.0);
}

fn push_macro_dune_field(doc: &mut TerrainDocument) {
    doc.stack.push_into_category(Layer::new(
        "Desert Floor",
        LayerKind::Flat(FlatParams { height: 12.0 }),
    ));
    doc.stack.push_into_category(Layer::new(
        "Hard Substrate",
        LayerKind::Materials(MaterialsParams::soft_over_hard(4.0)),
    ));
    doc.stack.push_into_category(Layer::new(
        "Aeolian Dunes",
        LayerKind::Dunes(DuneParams {
            base: NoiseParams {
                seed: 101,
                frequency: 0.004,
                amplitude: 28.0,
                octaves: 3,
                ..NoiseParams::default()
            },
            wind_strength: 1.35,
            sand_supply: 1.2,
            ..DuneParams::default()
        }),
    ));
    doc.stack.push_into_category(Layer::new(
        "Sand Transport",
        LayerKind::SandSimulation(SandSimParams::default()),
    ));
    push_gradient_reconstruct(doc, 32, 4.0);
}

fn push_macro_coastal(doc: &mut TerrainDocument) {
    let style = LandscapeStyle::Coastal.params();
    doc.stack.push_into_category(Layer::new(
        "Coastal Landmass",
        LayerKind::Island(IslandParams {
            seed: 55,
            archetype: IslandArchetype::VolcanicHighIsland,
            sea_level: doc.blueprint.sea_level,
            mountain_height: 280.0,
            coastline_warp: 0.28,
            ..IslandParams::default()
        }),
    ));
    doc.stack.push_into_category(Layer::new(
        "Inland Uplift",
        LayerKind::Uplift(style.uplift.clone()),
    ));
    push_gradient_reconstruct(doc, 48, 6.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn process_kinds(doc: &TerrainDocument) -> Vec<LayerKind> {
        // Exclude biome-group contents — biome recipes may carry EffectFilters.
        let mut out = Vec::new();
        for cat in [
            crate::layer::StackCategory::Foundation,
            crate::layer::StackCategory::Shape,
            crate::layer::StackCategory::Simulation,
            crate::layer::StackCategory::Mask,
            crate::layer::StackCategory::Surface,
        ] {
            if let Some(group) = doc.stack.find_category(cat) {
                collect_non_biome_kinds(group, &mut out);
            }
        }
        // Root layers (e.g. Base) outside categories.
        for node in &doc.stack.nodes {
            if let crate::layer::StackNode::Layer(l) = node {
                out.push(l.kind.clone());
            }
        }
        out
    }

    fn collect_non_biome_kinds(group: &crate::layer::LayerGroup, out: &mut Vec<LayerKind>) {
        for child in &group.children {
            match child {
                crate::layer::StackNode::Layer(l) => out.push(l.kind.clone()),
                crate::layer::StackNode::Group(g) if g.is_biome() => {}
                crate::layer::StackNode::Group(g) => collect_non_biome_kinds(g, out),
            }
        }
    }

    fn has_process(doc: &TerrainDocument, pred: impl Fn(&LayerKind) -> bool) -> bool {
        process_kinds(doc).iter().any(pred)
    }

    fn kinds(doc: &TerrainDocument) -> Vec<&LayerKind> {
        doc.stack.flatten_layers().iter().map(|l| &l.kind).collect()
    }

    fn has_kind(doc: &TerrainDocument, pred: impl Fn(&LayerKind) -> bool) -> bool {
        kinds(doc).into_iter().any(pred)
    }

    fn index_of(doc: &TerrainDocument, pred: impl Fn(&LayerKind) -> bool) -> Option<usize> {
        kinds(doc).into_iter().position(pred)
    }

    #[test]
    fn blank_is_scaffold_without_evolution() {
        let doc = blank_world_design(8192.0, 256);
        assert!(!doc.biome_library.definitions.is_empty());
        assert!(doc.selected_biome_layer.is_some());
        assert_eq!(doc.shapes.shapes.len(), 2);
        assert!(has_kind(&doc, |k| matches!(
            k,
            LayerKind::GradientReconstruct(_)
        )));
        assert!(!has_kind(&doc, |k| matches!(
            k,
            LayerKind::LandscapeEvolution(_)
        )));
        assert!(!has_process(&doc, |k| matches!(
            k,
            LayerKind::EffectFilter(_)
        )));
    }

    #[test]
    fn tropical_island_has_shapes_and_hierarchy() {
        let doc = tropical_island_world(10_000.0, 256);
        assert_eq!(doc.version, crate::document::DOCUMENT_VERSION);
        assert_eq!(doc.shapes.shapes.len(), 3);
        assert!(!doc.biome_library.definitions.is_empty());
        assert!(has_kind(&doc, |k| matches!(k, LayerKind::Island(_))));
        assert!(has_kind(&doc, |k| matches!(k, LayerKind::Materials(_))));
        assert!(has_kind(&doc, |k| matches!(
            k,
            LayerKind::LandscapeEvolution(_)
        )));
        assert!(has_kind(&doc, |k| matches!(k, LayerKind::Coastal(_))));
        assert!(has_kind(&doc, |k| matches!(
            k,
            LayerKind::GeomorphicDetail(_)
        )));
        let mat = index_of(&doc, |k| matches!(k, LayerKind::Materials(_))).unwrap();
        let evo = index_of(&doc, |k| matches!(k, LayerKind::LandscapeEvolution(_))).unwrap();
        assert!(mat < evo, "materials must precede landscape evolution");
    }

    #[test]
    fn all_non_blank_templates_follow_cause_effect() {
        for t in WorldTemplate::all()
            .iter()
            .copied()
            .filter(|t| *t != WorldTemplate::Blank)
        {
            let doc = t.build(8_000.0, 128);
            assert_eq!(doc.presets_used.first().map(String::as_str), Some(t.id()));
            assert!(!doc.biome_library.definitions.is_empty(), "{}", t.id());
            assert!(
                has_kind(&doc, |k| matches!(k, LayerKind::LandscapeEvolution(_))),
                "{}: missing evolution",
                t.id()
            );
            assert!(
                has_kind(&doc, |k| matches!(k, LayerKind::HydrologyRepair(_))),
                "{}: missing hydro repair",
                t.id()
            );
            assert!(
                has_kind(&doc, |k| matches!(k, LayerKind::GeomorphicDetail(_))),
                "{}: missing geomorphic detail",
                t.id()
            );
            assert!(
                !has_process(&doc, |k| matches!(k, LayerKind::EffectFilter(_))),
                "{}: process stack must not use EffectFilter soup",
                t.id()
            );
            if !matches!(t, WorldTemplate::DuneField) {
                let mat = index_of(&doc, |k| matches!(k, LayerKind::Materials(_)))
                    .unwrap_or_else(|| panic!("{}: missing materials", t.id()));
                let evo =
                    index_of(&doc, |k| matches!(k, LayerKind::LandscapeEvolution(_))).unwrap();
                assert!(mat < evo, "{}: materials before evolution", t.id());
            }
        }
    }

    #[test]
    fn alpine_has_debris_and_talus_after_evolution() {
        let doc = alpine_world(8_000.0, 128);
        let evo = index_of(&doc, |k| matches!(k, LayerKind::LandscapeEvolution(_))).unwrap();
        let debris = index_of(&doc, |k| matches!(k, LayerKind::DebrisFlow(_))).unwrap();
        let talus = index_of(&doc, |k| matches!(k, LayerKind::ThermalErosion(_))).unwrap();
        assert!(evo < debris && debris < talus);
    }

    #[test]
    fn world_template_ids_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for t in WorldTemplate::all() {
            assert!(seen.insert(t.id()));
        }
        assert_eq!(seen.len(), 10);
    }
}
