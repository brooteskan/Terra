//! Baked tool thumbnail icons (mockup-style grayscale landform previews).
//!
//! Source assets are large (1024²) PNGs. Decoding + downscale is done on a
//! small background pool so the UI can show Lucide placeholders until ready.
//! Prefer [`prefetch_all`] at startup so Quick Add / Tools don't hitch on first open.

use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    OnceLock,
};
#[cfg(not(test))]
use std::sync::{mpsc, Arc, Mutex};

#[cfg(not(test))]
type ThumbJob = Box<dyn FnOnce() + Send + 'static>;
static THUMB_READY: AtomicBool = AtomicBool::new(false);
static THUMBS_PENDING: AtomicUsize = AtomicUsize::new(0);

/// Display size baked into the thumb cache (source assets are 1024²).
const THUMB_PX: u32 = 128;

pub struct ToolThumb {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

fn load(bytes: &'static [u8]) -> ToolThumb {
    let img = image::load_from_memory(bytes)
        .expect("tool thumb png")
        .resize_to_fill(THUMB_PX, THUMB_PX, image::imageops::FilterType::Triangle)
        .to_rgba8();
    ToolThumb {
        width: img.width(),
        height: img.height(),
        rgba: img.into_raw(),
    }
}

#[cfg(not(test))]
fn worker_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().clamp(2, 8))
        .unwrap_or(4)
}

#[cfg(not(test))]
fn enqueue(job: ThumbJob) {
    static TX: OnceLock<mpsc::Sender<ThumbJob>> = OnceLock::new();
    let tx = TX.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<ThumbJob>();
        let rx = Arc::new(Mutex::new(rx));
        let n = worker_count();
        for i in 0..n {
            let rx = Arc::clone(&rx);
            std::thread::Builder::new()
                .name(format!("terra-tool-thumbs-{i}"))
                .spawn(move || loop {
                    let job = {
                        let Ok(guard) = rx.lock() else {
                            break;
                        };
                        guard.recv()
                    };
                    match job {
                        Ok(job) => {
                            job();
                            THUMB_READY.store(true, Ordering::Release);
                            THUMBS_PENDING.fetch_sub(1, Ordering::AcqRel);
                        }
                        Err(_) => break,
                    }
                })
                .expect("spawn tool thumbnail worker");
        }
        tx
    });
    THUMBS_PENDING.fetch_add(1, Ordering::AcqRel);
    if tx.send(job).is_err() {
        THUMBS_PENDING.fetch_sub(1, Ordering::AcqRel);
    }
}

/// True once one or more background thumbnail decodes completed since the last poll.
pub fn take_ready_signal() -> bool {
    THUMB_READY.swap(false, Ordering::AcqRel)
}

/// True while the decoder has queued or executing work.
pub fn has_pending_work() -> bool {
    THUMBS_PENDING.load(Ordering::Acquire) > 0
}

/// Kick off background decode for every catalog tool thumb (idempotent).
///
/// Call once at app startup so Quick Add / Tools open with art already warm.
pub fn prefetch_all() {
    #[cfg(not(test))]
    {
        for tool in crate::ui::tool_catalog::all_tools_cached() {
            let _ = thumb_for_tool(tool.id);
        }
        // Org Quick Add entries share art with the aliases above.
        for id in [
            "org.biome",
            "org.biome_paint",
            "org.hole",
            "org.pass",
            "org.isolated",
        ] {
            let _ = thumb_for_tool(id);
        }
    }
}

macro_rules! thumb {
    ($fn_name:ident, $path:literal) => {
        fn $fn_name() -> Option<&'static ToolThumb> {
            static T: OnceLock<ToolThumb> = OnceLock::new();
            #[cfg(test)]
            {
                return Some(T.get_or_init(|| load(include_bytes!($path))));
            }
            #[cfg(not(test))]
            {
                static REQUESTED: AtomicBool = AtomicBool::new(false);
                if let Some(thumb) = T.get() {
                    return Some(thumb);
                }
                if REQUESTED
                    .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    enqueue(Box::new(|| {
                        T.get_or_init(|| load(include_bytes!($path)));
                    }));
                }
                None
            }
        }
    };
}

// —— Landforms ————————————————————————————————————————————————
thumb!(mountain, "../../../../assets/tools/tool_mountain.png");
thumb!(hill, "../../../../assets/tools/tool_hill.png");
thumb!(mesa, "../../../../assets/tools/tool_mesa.png");
thumb!(volcano, "../../../../assets/tools/tool_volcano.png");
thumb!(crater, "../../../../assets/tools/tool_crater.png");
thumb!(plateau, "../../../../assets/tools/tool_plateau.png");
thumb!(canyon, "../../../../assets/tools/tool_canyon.png");
thumb!(island, "../../../../assets/tools/tool_island.png");
thumb!(dunes, "../../../../assets/tools/tool_dunes.png");
thumb!(flat, "../../../../assets/tools/tool_flat.png");
thumb!(flatten, "../../../../assets/tools/tool_flatten.png");
thumb!(terrace, "../../../../assets/tools/tool_terrace.png");
thumb!(warp, "../../../../assets/tools/tool_warp.png");
thumb!(fbm, "../../../../assets/tools/tool_fbm.png");
thumb!(ridged, "../../../../assets/tools/tool_ridged.png");
thumb!(cellular, "../../../../assets/tools/tool_cellular.png");

// —— Hydrology / simulation ————————————————————————————————————
thumb!(river, "../../../../assets/tools/tool_river.png");
thumb!(lake, "../../../../assets/tools/tool_lake.png");
thumb!(waterfall, "../../../../assets/tools/tool_waterfall.png");
thumb!(coastal, "../../../../assets/tools/tool_coastal.png");
thumb!(hydraulic, "../../../../assets/tools/tool_hydraulic.png");
thumb!(thermal, "../../../../assets/tools/tool_thermal.png");
thumb!(wind, "../../../../assets/tools/tool_wind.png");
thumb!(sediment, "../../../../assets/tools/tool_sediment.png");
thumb!(
    filter_fills_soft,
    "../../../../assets/tools/tool_filter_fills_soft.png"
);
thumb!(talus, "../../../../assets/tools/tool_talus.png");
thumb!(weathering, "../../../../assets/tools/tool_weathering.png");

// —— Brushes ———————————————————————————————————————————————————
thumb!(raise, "../../../../assets/tools/tool_raise.png");
thumb!(lower, "../../../../assets/tools/tool_lower.png");
thumb!(smooth, "../../../../assets/tools/tool_smooth.png");
thumb!(pinch, "../../../../assets/tools/tool_pinch.png");
thumb!(inflate, "../../../../assets/tools/tool_inflate.png");

// —— Shape layers ——————————————————————————————————————————————
thumb!(
    shape_sculpt,
    "../../../../assets/tools/tool_shape_sculpt.png"
);
thumb!(
    shape_stamp_2d,
    "../../../../assets/tools/tool_shape_stamp_2d.png"
);
thumb!(
    shape_stamp_3d,
    "../../../../assets/tools/tool_shape_stamp_3d.png"
);
thumb!(
    shape_procedural,
    "../../../../assets/tools/tool_shape_procedural.png"
);
thumb!(
    shape_polygon,
    "../../../../assets/tools/tool_shape_polygon.png"
);
thumb!(shape_path, "../../../../assets/tools/tool_shape_path.png");
thumb!(
    shape_heightmap,
    "../../../../assets/tools/tool_shape_heightmap.png"
);

// —— Biomes ————————————————————————————————————————————————————
thumb!(biome_paint, "../../../../assets/tools/tool_biome_paint.png");
thumb!(biome_erase, "../../../../assets/tools/tool_biome_erase.png");
thumb!(
    biome_smooth,
    "../../../../assets/tools/tool_biome_smooth.png"
);
thumb!(
    biome_replace,
    "../../../../assets/tools/tool_biome_replace.png"
);
thumb!(
    biome_normalize,
    "../../../../assets/tools/tool_biome_normalize.png"
);
thumb!(biome_flood, "../../../../assets/tools/tool_biome_flood.png");
thumb!(
    biome_polygon,
    "../../../../assets/tools/tool_biome_polygon.png"
);
thumb!(
    biome_create,
    "../../../../assets/tools/tool_biome_create.png"
);

// —— Materials —————————————————————————————————————————————————
thumb!(
    mat_material,
    "../../../../assets/tools/tool_mat_material.png"
);
thumb!(mat_colour, "../../../../assets/tools/tool_mat_colour.png");
thumb!(mat_wetness, "../../../../assets/tools/tool_mat_wetness.png");
thumb!(mat_snow, "../../../../assets/tools/tool_mat_snow.png");
thumb!(mat_rock, "../../../../assets/tools/tool_mat_rock.png");
thumb!(mat_grass, "../../../../assets/tools/tool_mat_grass.png");

// —— Objects ———————————————————————————————————————————————————
thumb!(obj_trees, "../../../../assets/tools/tool_obj_trees.png");
thumb!(obj_rocks, "../../../../assets/tools/tool_obj_rocks.png");
thumb!(obj_grass, "../../../../assets/tools/tool_obj_grass.png");
thumb!(obj_debris, "../../../../assets/tools/tool_obj_debris.png");
thumb!(obj_custom, "../../../../assets/tools/tool_obj_custom.png");
thumb!(obj_density, "../../../../assets/tools/tool_obj_density.png");
thumb!(obj_scale, "../../../../assets/tools/tool_obj_scale.png");
thumb!(
    obj_rotation,
    "../../../../assets/tools/tool_obj_rotation.png"
);
thumb!(obj_slope, "../../../../assets/tools/tool_obj_slope.png");
thumb!(obj_height, "../../../../assets/tools/tool_obj_height.png");

// —— Masks —————————————————————————————————————————————————————
thumb!(mask_height, "../../../../assets/tools/tool_mask_height.png");
thumb!(mask_slope, "../../../../assets/tools/tool_mask_slope.png");
thumb!(
    mask_curvature,
    "../../../../assets/tools/tool_mask_curvature.png"
);
thumb!(
    mask_convexity,
    "../../../../assets/tools/tool_mask_convexity.png"
);
thumb!(
    mask_concavity,
    "../../../../assets/tools/tool_mask_concavity.png"
);
thumb!(mask_flow, "../../../../assets/tools/tool_mask_flow.png");
thumb!(mask_noise, "../../../../assets/tools/tool_mask_noise.png");
thumb!(
    mask_distance,
    "../../../../assets/tools/tool_mask_distance.png"
);
thumb!(
    mask_painted,
    "../../../../assets/tools/tool_mask_painted.png"
);
thumb!(
    mask_combined,
    "../../../../assets/tools/tool_mask_combined.png"
);

// —— Utilities —————————————————————————————————————————————————
thumb!(util_move, "../../../../assets/tools/tool_util_move.png");
thumb!(
    util_measure,
    "../../../../assets/tools/tool_util_measure.png"
);
thumb!(util_bake, "../../../../assets/tools/tool_util_bake.png");

// —— Filters (WC Develop) ——————————————————————————————————————
thumb!(
    filter_add_set,
    "../../../../assets/tools/tool_filter_add_set.png"
);
thumb!(
    filter_border_blend,
    "../../../../assets/tools/tool_filter_border_blend.png"
);
thumb!(
    filter_zero_edge,
    "../../../../assets/tools/tool_filter_zero_edge.png"
);
thumb!(
    filter_curve,
    "../../../../assets/tools/tool_filter_curve.png"
);
thumb!(
    filter_cutoff,
    "../../../../assets/tools/tool_filter_cutoff.png"
);
thumb!(
    filter_angle_blur,
    "../../../../assets/tools/tool_filter_angle_blur.png"
);
thumb!(
    filter_balloon,
    "../../../../assets/tools/tool_filter_balloon.png"
);
thumb!(
    filter_blocks,
    "../../../../assets/tools/tool_filter_blocks.png"
);
thumb!(
    filter_deflate,
    "../../../../assets/tools/tool_filter_deflate.png"
);
thumb!(
    filter_denoise,
    "../../../../assets/tools/tool_filter_denoise.png"
);
thumb!(
    filter_distortion,
    "../../../../assets/tools/tool_filter_distortion.png"
);
thumb!(
    filter_hexagons,
    "../../../../assets/tools/tool_filter_hexagons.png"
);
thumb!(
    filter_kuwahara,
    "../../../../assets/tools/tool_filter_kuwahara.png"
);
thumb!(
    filter_rugged,
    "../../../../assets/tools/tool_filter_rugged.png"
);
thumb!(
    filter_scatter,
    "../../../../assets/tools/tool_filter_scatter.png"
);
thumb!(
    filter_shore,
    "../../../../assets/tools/tool_filter_shore.png"
);
thumb!(
    filter_smooth_ridges,
    "../../../../assets/tools/tool_filter_smooth_ridges.png"
);
thumb!(
    filter_squeeze,
    "../../../../assets/tools/tool_filter_squeeze.png"
);
thumb!(
    filter_strata,
    "../../../../assets/tools/tool_filter_strata.png"
);
thumb!(
    filter_swirl,
    "../../../../assets/tools/tool_filter_swirl.png"
);
thumb!(
    filter_washed_off,
    "../../../../assets/tools/tool_filter_washed_off.png"
);
thumb!(
    filter_billow,
    "../../../../assets/tools/tool_filter_billow.png"
);
thumb!(
    filter_gabor,
    "../../../../assets/tools/tool_filter_gabor.png"
);
thumb!(
    filter_perlin,
    "../../../../assets/tools/tool_filter_perlin.png"
);
thumb!(
    filter_phasor,
    "../../../../assets/tools/tool_filter_phasor.png"
);
thumb!(
    filter_simplex,
    "../../../../assets/tools/tool_filter_simplex.png"
);
thumb!(
    filter_value_noise,
    "../../../../assets/tools/tool_filter_value_noise.png"
);
thumb!(filter_wave, "../../../../assets/tools/tool_filter_wave.png");
thumb!(
    filter_white,
    "../../../../assets/tools/tool_filter_white.png"
);
thumb!(
    filter_chipped,
    "../../../../assets/tools/tool_filter_chipped.png"
);
thumb!(
    filter_rocky_hard,
    "../../../../assets/tools/tool_filter_rocky_hard.png"
);
thumb!(
    filter_rocky_layers,
    "../../../../assets/tools/tool_filter_rocky_layers.png"
);
thumb!(
    filter_rocky_sharp,
    "../../../../assets/tools/tool_filter_rocky_sharp.png"
);
thumb!(
    filter_rocky_wide,
    "../../../../assets/tools/tool_filter_rocky_wide.png"
);
thumb!(
    filter_terrace_irregular,
    "../../../../assets/tools/tool_filter_terrace_irregular.png"
);
thumb!(
    filter_terrace_steep,
    "../../../../assets/tools/tool_filter_terrace_steep.png"
);
thumb!(
    filter_angle_break,
    "../../../../assets/tools/tool_filter_angle_break.png"
);
thumb!(
    filter_ridged_flows,
    "../../../../assets/tools/tool_filter_ridged_flows.png"
);
thumb!(
    filter_soft_flows,
    "../../../../assets/tools/tool_filter_soft_flows.png"
);
thumb!(
    filter_rocky,
    "../../../../assets/tools/tool_filter_rocky.png"
);
thumb!(
    filter_thin_flows,
    "../../../../assets/tools/tool_filter_thin_flows.png"
);
thumb!(
    filter_wide_flows,
    "../../../../assets/tools/tool_filter_wide_flows.png"
);
thumb!(
    sculpt_valley,
    "../../../../assets/tools/tool_sculpt_valley.png"
);
thumb!(
    sculpt_height_stamp,
    "../../../../assets/tools/tool_sculpt_height_stamp.png"
);

/// Lookup a baked thumbnail for a catalog tool id, if available.
///
/// Aliases are only for true synonyms of the same tool (e.g. `gen.hills` | `gen.hill`).
/// Distinct catalog tools never share art.
pub fn thumb_for_tool(id: &str) -> Option<&'static ToolThumb> {
    match id {
        // Landforms
        "gen.mountain" | "sculpt.mountain_stamp" => mountain(),
        "gen.hills" | "gen.hill" => hill(),
        "gen.mesa" => mesa(),
        "gen.volcano" => volcano(),
        "gen.crater" | "sculpt.crater_stamp" | "filter.effect.crater" => crater(),
        "gen.plateau"
        | "sculpt.plateau_stamp"
        | "filter.arid.rocky_plateaus"
        | "sim.rocky_plateaus" => plateau(),
        "gen.canyon" | "gen.canyons" | "filter.arid.canyon" | "sim.canyon_filter" => canyon(),
        "gen.island" => island(),
        "gen.dunes" | "filter.arid.dunes" => dunes(),
        "gen.flat" => flat(),
        "terrain.flatten" | "sculpt.flatten" | "filter.general.flatten" | "sim.flatten_filter" => {
            flatten()
        }
        "terrain.terrace" | "sculpt.terrace" | "filter.terrace.simple" | "sim.terrace_simple" => {
            terrace()
        }
        "gen.warp" => warp(),
        "gen.noise" | "gen.fbm" | "gen.hills_fbm" | "sculpt.noise_brush" => fbm(),
        "gen.ridge" | "gen.ridged" | "filter.effect.ridged" | "filter.noise.ridged" => ridged(),
        "gen.cellular" | "gen.voronoi" | "filter.design.voronoi" | "filter.noise.voronoi" => {
            cellular()
        }

        // Hydrology / simulation
        "sim.hydrology_repair" => river(),
        "sim.landscape_evolution" => hydraulic(),
        "sim.multi_scale_amplify" => filter_rugged(),
        "filter.general.geomorphic_detail" => filter_phasor(),
        "water.river" | "gen.river" | "erosion.river" | "sim.river" | "sculpt.river_path" => {
            river()
        }
        "water.lake" | "sim.lake" => lake(),
        "water.waterfall" | "sim.waterfall" => waterfall(),
        "sim.coastal" | "water.coastal" | "erosion.coastal" | "sculpt.coastline" => coastal(),
        "erosion.hydraulic"
        | "sim.hydraulic"
        | "filter.basic.hydraulic"
        | "filter.advanced.hydraulic_sediment" => hydraulic(),
        "erosion.thermal" | "sim.thermal" => thermal(),
        "sim.debris_flow" | "erosion.debris" => talus(),
        "erosion.wind" | "sim.wind" | "filter.drift.wind" | "sim.wind_carve" => wind(),
        "sim.sediment"
        | "filter.sediment.mud"
        | "sim.mud"
        | "filter.advanced.sediment_flows"
        | "sim.sediment_flows" => sediment(),
        "filter.sediment.fill_soft" => filter_fills_soft(),
        "erosion.talus" | "sim.talus" | "sculpt.erode_brush" | "filter.sediment.talus" => talus(),
        "sim.weathering" => weathering(),
        "filter.arid.rocky_cliffs" | "sim.rocky_cliffs" | "filter.effect.cliff_reinforce" => mesa(),

        // Brushes
        "sculpt.raise" => raise(),
        "sculpt.lower" => lower(),
        "sculpt.smooth" => smooth(),
        "sculpt.pinch" => pinch(),
        "sculpt.inflate" | "filter.effect.inflate" | "sim.inflate" => inflate(),
        "sculpt.valley_stamp" => sculpt_valley(),
        "sculpt.height_stamp" => sculpt_height_stamp(),

        // Shape layers
        "shape.sculpt" => shape_sculpt(),
        "shape.stamp_2d" => shape_stamp_2d(),
        "shape.stamp_3d" => shape_stamp_3d(),
        "shape.procedural" => shape_procedural(),
        "shape.polygon" => shape_polygon(),
        "shape.path" => shape_path(),
        "shape.heightmap" | "filter.design.height_map" => shape_heightmap(),

        // Biomes
        "biome.paint" => biome_paint(),
        "biome.erase" => biome_erase(),
        "biome.smooth" => biome_smooth(),
        "biome.replace" => biome_replace(),
        "biome.normalize" => biome_normalize(),
        "biome.flood" => biome_flood(),
        "biome.polygon" => biome_polygon(),
        "biome.create" => biome_create(),
        "biome.climate" => biome_create(),

        // Organisation shortcuts (Quick Add) — reuse closest catalog art.
        "org.biome" => biome_create(),
        "org.biome_paint" => biome_paint(),
        "org.hole" => filter_zero_edge(),
        "org.pass" => mask_combined(),
        "org.isolated" => filter_border_blend(),

        // Materials
        "mat.material" => mat_material(),
        "mat.colour" => mat_colour(),
        "mat.wetness" => mat_wetness(),
        "mat.snow" => mat_snow(),
        "mat.rock" => mat_rock(),
        "mat.grass" => mat_grass(),

        // Objects
        "obj.trees" => obj_trees(),
        "obj.rocks" => obj_rocks(),
        "obj.grass" => obj_grass(),
        "obj.debris" => obj_debris(),
        "obj.custom" => obj_custom(),
        "obj.density" => obj_density(),
        "obj.scale" => obj_scale(),
        "obj.rotation" => obj_rotation(),
        "obj.slope" => obj_slope(),
        "obj.height" => obj_height(),

        // Masks
        "mask.height" => mask_height(),
        "mask.slope" => mask_slope(),
        "mask.curvature" => mask_curvature(),
        "mask.convexity" => mask_convexity(),
        "mask.concavity" => mask_concavity(),
        "mask.flow" => mask_flow(),
        "mask.noise" => mask_noise(),
        "mask.distance" => mask_distance(),
        "mask.painted" => mask_painted(),
        "mask.combined" => mask_combined(),

        // Utilities
        "util.move" | "sculpt.move" => util_move(),
        "util.measure" => util_measure(),
        "util.bake" => util_bake(),

        // Filters — unique art
        "filter.general.add_set" => filter_add_set(),
        "filter.general.border_blend" => filter_border_blend(),
        "filter.general.zero_edge" => filter_zero_edge(),
        "filter.design.curve" | "sim.curve" => filter_curve(),
        "filter.design.cutoff" | "sim.cutoff" => filter_cutoff(),
        "filter.effect.angle_blur" | "filter.effect.directional_blur" => filter_angle_blur(),
        "filter.effect.balloon" => filter_balloon(),
        "filter.effect.blocks" | "sim.blocks" => filter_blocks(),
        "filter.effect.deflate" | "sim.deflate" => filter_deflate(),
        "filter.effect.denoise" => filter_denoise(),
        "filter.effect.smooth" => smooth(),
        "filter.effect.distortion" => filter_distortion(),
        "filter.effect.hexagons" | "sim.hexagons" => filter_hexagons(),
        "filter.effect.kuwahara" | "sim.kuwahara" => filter_kuwahara(),
        "filter.effect.rugged" => filter_rugged(),
        "filter.effect.scatter" => filter_scatter(),
        "filter.effect.shore" => filter_shore(),
        "filter.effect.smooth_ridges" => filter_smooth_ridges(),
        "filter.effect.squeeze" => filter_squeeze(),
        "filter.effect.strata" => filter_strata(),
        "filter.effect.swirl" | "sim.swirl" => filter_swirl(),
        "filter.effect.washed_off" => filter_washed_off(),
        "filter.noise.billow" => filter_billow(),
        "filter.noise.gabor" => filter_gabor(),
        "filter.noise.perlin" => filter_perlin(),
        "filter.noise.phasor" => filter_phasor(),
        "filter.noise.simplex" => filter_simplex(),
        "filter.noise.value" => filter_value_noise(),
        "filter.noise.wave" => filter_wave(),
        "filter.noise.white" => filter_white(),
        "filter.arid.chipped" => filter_chipped(),
        "filter.arid.rocky_hard" => filter_rocky_hard(),
        "filter.arid.rocky_layers" => filter_rocky_layers(),
        "filter.arid.rocky_sharp" => filter_rocky_sharp(),
        "filter.arid.rocky_wide" => filter_rocky_wide(),
        "filter.terrace.irregular" | "sim.terrace_irregular" => filter_terrace_irregular(),
        "filter.terrace.steep" | "sim.terrace_steep" => filter_terrace_steep(),
        "filter.drift.angle_break" => filter_angle_break(),
        "filter.basic.ridged_flows" => filter_ridged_flows(),
        "filter.basic.soft_flows" => filter_soft_flows(),
        "filter.basic.rocky" => filter_rocky(),
        "filter.advanced.thin_flows" => filter_thin_flows(),
        "filter.advanced.wide_flows" => filter_wide_flows(),

        _ => None,
    }
}
