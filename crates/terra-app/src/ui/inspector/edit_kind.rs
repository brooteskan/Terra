//! LayerKind parameter editors.

use crate::ui::actions::PanelAction;
use crate::ui::style::{self, FONT_SCALE, ROW_H};
use crate::ui::EditorTool;
use terra_core::layer::*;
use terra_gui::{
    button_id, checkbox, combo, label, label_dim, section_header, slider_f32, slider_f32_id,
    slider_i32, slider_i32_id, GuiContext, Icon, Id, Rect,
};

/// Which parameter pane `edit_kind` should draw.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum KindEditPane {
    /// Artist-facing primary controls. When `split_noise`, omit seed/noise fields.
    Primary { split_noise: bool },
    /// Noise / seed / fractal controls.
    Noise,
    /// Expert extras.
    Advanced,
}

impl KindEditPane {
    fn is_advanced(self) -> bool {
        matches!(self, KindEditPane::Advanced)
    }

    fn is_noise(self) -> bool {
        matches!(self, KindEditPane::Noise)
    }

    fn split_noise(self) -> bool {
        matches!(self, KindEditPane::Primary { split_noise: true })
    }
}

pub(crate) fn edit_kind(
    ui: &mut GuiContext<'_>,
    kind: &mut LayerKind,
    pane: KindEditPane,
    layer_id: LayerId,
    actions: &mut Vec<PanelAction>,
) -> bool {
    if pane.is_noise() {
        return edit_kind_noise(ui, kind);
    }
    let advanced = pane.is_advanced();
    let split_noise = pane.split_noise();
    let mut changed = false;
    match kind {
        LayerKind::SculptBase(p) => {
            changed |= slider_f32(ui, "Fill height", &mut p.fill_height, -100.0, 500.0);
        }
        LayerKind::SculptStrokes(p) => {
            label(
                ui,
                &format!("{} editable semantic stroke(s)", p.strokes.len()),
            );
            changed |= slider_f32(ui, "Reconcile", &mut p.reconcile, 0.0, 1.0);
            if advanced {
                label(
                    ui,
                    "Strokes preserve intent at every evaluation resolution.",
                );
            }
        }
        LayerKind::TerrainConstraints(p) => {
            label(
                ui,
                &format!("{} terrain constraint(s)", p.constraints.len()),
            );
            changed |= slider_f32(ui, "Preview Strength", &mut p.preview_strength, 0.0, 1.0);
            if advanced {
                label(
                    ui,
                    "Supports elevation, ridge, valley, river, coast, divide and protection.",
                );
            }
        }
        LayerKind::GradientReconstruct(p) => {
            let mut iterations = p.iterations as i32;
            if slider_i32(ui, "Solve Iterations", &mut iterations, 4, 400) {
                p.iterations = iterations as u32;
                changed = true;
            }
            changed |= slider_f32(ui, "Screening", &mut p.screening, 0.0, 2.0);
            changed |= slider_f32(
                ui,
                "Constraint Strength",
                &mut p.constraint_strength,
                0.0,
                24.0,
            );
            if advanced {
                changed |= slider_f32(
                    ui,
                    "Gradient Smoothing",
                    &mut p.gradient_smoothing,
                    0.0,
                    1.0,
                );
            }
        }
        LayerKind::LandscapeEvolution(p) => {
            changed |= slider_f32(ui, "Uplift", &mut p.uplift, 0.0, 2.0);
            changed |= slider_f32(ui, "Erosion", &mut p.erosion, 0.0, 2.0);
            changed |= slider_f32(ui, "Rainfall", &mut p.rainfall, 0.0, 3.0);
            changed |= slider_f32(ui, "River Incision", &mut p.river_incision, 0.0, 2.0);
            changed |= slider_f32(ui, "Geological Age", &mut p.geological_age, 0.0, 1.0);
            changed |= slider_f32(
                ui,
                "Terrain Resistance",
                &mut p.terrain_resistance,
                0.0,
                1.0,
            );
            changed |= slider_f32(ui, "Drainage Scale", &mut p.drainage_scale, 0.0, 1.0);
            // Solver mode toggle
            {
                use terra_core::layer::EvolutionSolverMode;
                let label = match p.solver {
                    EvolutionSolverMode::Fast => "Solver: Fast / Interactive",
                    EvolutionSolverMode::Accurate => "Solver: Accurate / Iterative",
                };
                if button_id(ui, Id::new("evo_solver"), label) {
                    p.solver = match p.solver {
                        EvolutionSolverMode::Fast => EvolutionSolverMode::Accurate,
                        EvolutionSolverMode::Accurate => EvolutionSolverMode::Fast,
                    };
                    changed = true;
                }
            }
            {
                use terra_core::layer::UpliftMode;
                let label = match p.uplift_mode {
                    UpliftMode::Uniform => "Uplift Mode: Uniform",
                    UpliftMode::Radial => "Uplift Mode: Radial",
                    UpliftMode::LinearBelt => "Uplift Mode: Linear Belt",
                    UpliftMode::Painted => "Uplift Mode: Painted",
                    UpliftMode::Procedural => "Uplift Mode: Procedural",
                    UpliftMode::ShapeDerived => "Uplift Mode: Shape-Derived",
                };
                if button_id(ui, Id::new("evo_uplift_mode"), label) {
                    p.uplift_mode = match p.uplift_mode {
                        UpliftMode::Uniform => UpliftMode::Radial,
                        UpliftMode::Radial => UpliftMode::LinearBelt,
                        UpliftMode::LinearBelt => UpliftMode::Painted,
                        UpliftMode::Painted => UpliftMode::Procedural,
                        UpliftMode::Procedural => UpliftMode::ShapeDerived,
                        UpliftMode::ShapeDerived => UpliftMode::Uniform,
                    };
                    changed = true;
                }
            }
            if advanced {
                changed |= slider_f32(ui, "K (erodibility)", &mut p.k, 1e-7, 1e-3);
                changed |= slider_f32(ui, "m (area exp)", &mut p.m, 0.1, 1.0);
                changed |= slider_f32(ui, "n (slope exp)", &mut p.n, 0.5, 2.0);
                changed |= slider_f32(ui, "Time Scale (yr)", &mut p.time_scale, 1e4, 5e6);
                changed |= slider_f32(ui, "dt (yr)", &mut p.dt, 100.0, 2e4);
                let mut fp = p.fixed_point_iters as i32;
                if slider_i32(ui, "Fixed-Point Passes", &mut fp, 1, 24) {
                    p.fixed_point_iters = fp as u32;
                    changed = true;
                }
                changed |= slider_f32(ui, "Deposition Coeff", &mut p.deposition_coeff, 0.0, 1.0);
                let mut iterations = p.iterations as i32;
                if slider_i32(ui, "Accurate Steps", &mut iterations, 1, 240) {
                    p.iterations = iterations as u32;
                    changed = true;
                }
                changed |= slider_f32(
                    ui,
                    "Hillslope Diffusion",
                    &mut p.hillslope_diffusion,
                    0.0,
                    1.0,
                );
                changed |= slider_f32(ui, "Talus Angle", &mut p.talus_angle_deg, 15.0, 55.0);
                changed |= slider_f32(
                    ui,
                    "Sediment Transport",
                    &mut p.sediment_transport,
                    0.0,
                    1.0,
                );
                changed |= slider_f32(
                    ui,
                    "Constraint Lock",
                    &mut p.constraint_preservation,
                    0.0,
                    1.0,
                );
                changed |= slider_f32(ui, "Base Level", &mut p.base_level, -500.0, 1000.0);
                changed |= slider_f32(
                    ui,
                    "Drainage Invalidate (m)",
                    &mut p.drainage_invalidation_m,
                    0.1,
                    20.0,
                );
                {
                    use terra_core::layer::BoundaryMode;
                    let label = match p.boundary {
                        BoundaryMode::SeaLevel => "Boundary: Sea Level",
                        BoundaryMode::Fixed => "Boundary: Fixed",
                        BoundaryMode::OpenDrainage => "Boundary: Open Drainage",
                        BoundaryMode::OutletMask => "Boundary: Outlet Mask",
                    };
                    if button_id(ui, Id::new("evo_boundary"), label) {
                        p.boundary = match p.boundary {
                            BoundaryMode::SeaLevel => BoundaryMode::Fixed,
                            BoundaryMode::Fixed => BoundaryMode::OpenDrainage,
                            BoundaryMode::OpenDrainage => BoundaryMode::OutletMask,
                            BoundaryMode::OutletMask => BoundaryMode::SeaLevel,
                        };
                        changed = true;
                    }
                }
                // Legacy mirrors
                changed |= slider_f32(ui, "Legacy Uplift/Step", &mut p.uplift_rate, 0.0, 1.0);
                changed |= slider_f32(ui, "Legacy Incision K", &mut p.incision_k, 0.0, 0.01);
            }
        }
        LayerKind::HydrologyRepair(p) => {
            let mut iterations = p.iterations as i32;
            if slider_i32(ui, "Repair Steps", &mut iterations, 1, 64) {
                p.iterations = iterations as u32;
                changed = true;
            }
            changed |= slider_f32(ui, "Incision", &mut p.incision, 0.0, 0.15);
            changed |= slider_f32(ui, "Repair Radius", &mut p.repair_radius_m, 1.0, 2000.0);
            changed |= slider_f32(
                ui,
                "Constraint Lock",
                &mut p.constraint_preservation,
                0.0,
                1.0,
            );
        }
        LayerKind::GeomorphicDetail(p) => {
            changed |= slider_f32(ui, "Meso Amplitude (m)", &mut p.amplitude, 0.0, 40.0);
            let mut micro = p
                .micro_amplitude_m
                .unwrap_or((p.amplitude * 0.22).max(0.35));
            if slider_f32(ui, "Micro Amplitude (m)", &mut micro, 0.0, 12.0) {
                p.micro_amplitude_m = Some(micro);
                changed = true;
            }
            let mut cascades = p.octaves as i32;
            if slider_i32(ui, "Cascades", &mut cascades, 2, 6) {
                p.octaves = cascades as u32;
                changed = true;
            }
            changed |= slider_f32(ui, "Flow Alignment", &mut p.flow_alignment, 0.0, 1.0);
            changed |= slider_f32(ui, "Slope Gate", &mut p.slope_gate, 0.02, 0.6);
            changed |= slider_f32(ui, "Drainage Lock", &mut p.preserve_drainage, 0.0, 1.0);
            changed |= slider_f32(ui, "Silhouette Lock", &mut p.silhouette_lock, 0.0, 1.0);
            changed |= slider_f32(ui, "Ridge Breakup", &mut p.ridge_breakup, 0.0, 1.0);
            changed |= slider_f32(ui, "Gully Strength", &mut p.gully_strength, 0.0, 1.0);
            changed |= slider_f32(ui, "Rock Roughness", &mut p.rock_roughness, 0.0, 1.0);
        }
        LayerKind::EcosystemFeedback(p) => {
            let mut passes = p.passes as i32;
            if slider_i32(ui, "Feedback Passes", &mut passes, 1, 12) {
                p.passes = passes as u32;
                changed = true;
            }
            changed |= slider_f32(ui, "Root Cohesion", &mut p.root_cohesion, 0.0, 1.0);
            changed |= slider_f32(
                ui,
                "Rain Interception",
                &mut p.rainfall_interception,
                0.0,
                1.0,
            );
            changed |= slider_f32(ui, "Weathering", &mut p.weathering, 0.0, 1.0);
            changed |= slider_f32(ui, "Sediment Capture", &mut p.sediment_capture, 0.0, 1.0);
            changed |= slider_f32(ui, "Feedback Strength", &mut p.strength, 0.0, 1.0);
        }
        LayerKind::Flat(p) => {
            changed |= slider_f32(ui, "Height", &mut p.height, -500.0, 2000.0);
        }
        LayerKind::Ramp(p) => {
            changed |= slider_f32(ui, "Min", &mut p.height_min, -200.0, 500.0);
            changed |= slider_f32(ui, "Max", &mut p.height_max, -200.0, 2000.0);
            if advanced {
                changed |= slider_f32(ui, "Direction", &mut p.direction, 0.0, 6.28);
            }
        }
        LayerKind::NoiseValue(p) | LayerKind::NoisePerlin(p) | LayerKind::NoiseOpenSimplex(p) => {
            changed |= edit_noise(ui, p, advanced);
        }
        LayerKind::NoiseWorley(p) => {
            changed |= edit_noise(ui, &mut p.base, advanced);
        }
        LayerKind::Fbm(p) | LayerKind::Ridged(p) => {
            changed |= edit_noise(ui, &mut p.base, advanced);
        }
        LayerKind::DomainWarp(p) => {
            if !split_noise {
                changed |= edit_noise(ui, &mut p.base, advanced);
            }
            changed |= slider_f32(ui, "Warp Strength", &mut p.warp_strength, 0.0, 300.0);
            if advanced {
                changed |= slider_f32(ui, "Warp Freq", &mut p.warp_frequency, 0.0001, 0.02);
            }
        }
        LayerKind::Terrace(p) => {
            let mut levels = p.levels as i32;
            if slider_i32(ui, "Levels", &mut levels, 2, 32) {
                p.levels = levels as u32;
                changed = true;
            }
            changed |= slider_f32(ui, "Sharpness", &mut p.sharpness, 0.0, 1.0);
        }
        LayerKind::Plateau(p) => {
            changed |= slider_f32(ui, "Low", &mut p.low, -100.0, 500.0);
            changed |= slider_f32(ui, "High", &mut p.high, 0.0, 1000.0);
            if advanced {
                changed |= slider_f32(ui, "Soft", &mut p.soft, 0.1, 50.0);
            }
        }
        LayerKind::Mesa(p) => {
            if !split_noise {
                changed |= seed_row(ui, "mesa_seed", &mut p.seed);
            }
            changed |= slider_f32(ui, "Height", &mut p.height, 40.0, 800.0);
            changed |= slider_f32(ui, "Radius", &mut p.radius, 0.04, 0.55);
            changed |= slider_f32(ui, "Edge Steepness", &mut p.edge_steepness, 1.0, 6.0);
            if advanced {
                changed |= slider_f32(ui, "Cap Noise", &mut p.cap_noise, 0.0, 40.0);
                changed |= slider_f32(ui, "Talus Soft", &mut p.soft, 0.0, 0.6);
                changed |= slider_f32(ui, "Center U", &mut p.center_u, 0.0, 1.0);
                changed |= slider_f32(ui, "Center V", &mut p.center_v, 0.0, 1.0);
            }
        }
        LayerKind::Island(p) => {
            use terra_core::layer::IslandArchetype;

            let archetype_label = match p.archetype {
                IslandArchetype::VolcanicHighIsland => "Type: Volcanic High Island",
                IslandArchetype::Archipelago => "Type: Archipelago",
                IslandArchetype::Atoll => "Type: Atoll",
            };
            if button_id(ui, Id::new("island_archetype"), archetype_label) {
                p.archetype = match p.archetype {
                    IslandArchetype::VolcanicHighIsland => IslandArchetype::Archipelago,
                    IslandArchetype::Archipelago => IslandArchetype::Atoll,
                    IslandArchetype::Atoll => IslandArchetype::VolcanicHighIsland,
                };
                changed = true;
            }
            if !split_noise {
                changed |= seed_row(ui, "island_seed", &mut p.seed);
            }
            changed |= slider_f32(ui, "Island Radius", &mut p.radius, 0.15, 0.95);
            changed |= slider_f32(ui, "Island Aspect", &mut p.aspect, 0.45, 2.2);
            changed |= slider_f32(ui, "Rotation", &mut p.rotation_deg, -180.0, 180.0);
            changed |= slider_f32(ui, "Sea Level", &mut p.sea_level, -100.0, 300.0);

            if p.archetype == IslandArchetype::Atoll {
                changed |= slider_f32(ui, "Lagoon Radius", &mut p.lagoon_radius, 0.15, 0.75);
                changed |= slider_f32(ui, "Ring Height", &mut p.mountain_height, 1.0, 40.0);
            } else {
                changed |= slider_f32(ui, "Mountain Height", &mut p.mountain_height, 40.0, 1600.0);
            }
            changed |= slider_f32(ui, "Coast Irregularity", &mut p.coastline_warp, 0.0, 0.42);

            if advanced {
                changed |= slider_f32(ui, "Center U", &mut p.center_u, 0.05, 0.95);
                changed |= slider_f32(ui, "Center V", &mut p.center_v, 0.05, 0.95);
                changed |= slider_f32(ui, "Ocean Floor", &mut p.ocean_floor, -1200.0, -10.0);
                changed |= slider_f32(ui, "Shelf Width (m)", &mut p.shelf_width, 20.0, 1200.0);
                changed |= slider_f32(ui, "Shelf Depth (m)", &mut p.shelf_depth, 2.0, 180.0);
                changed |= slider_f32(ui, "Beach Width (m)", &mut p.beach_width, 5.0, 300.0);
                changed |= slider_f32(ui, "Beach Height (m)", &mut p.beach_height, 0.5, 30.0);
                changed |= slider_f32(ui, "Reef Width (m)", &mut p.reef_width, 0.0, 700.0);
                changed |= slider_f32(ui, "Reef Depth (m)", &mut p.reef_depth, 0.5, 30.0);
                changed |= slider_f32(
                    ui,
                    "Coast Frequency",
                    &mut p.coastline_frequency,
                    0.0002,
                    0.006,
                );
                changed |= slider_f32(ui, "Upland Power", &mut p.mountain_power, 0.4, 4.0);
                changed |= slider_f32(ui, "Ridge Strength", &mut p.ridge_strength, 0.0, 1.0);
                changed |= slider_f32(ui, "Ridge Frequency", &mut p.ridge_frequency, 0.0002, 0.012);
            }
        }
        LayerKind::Mountains(p) => {
            if advanced {
                // Shape/Noise tabs own artist + fractal controls; Advanced is extras only.
                changed |= slider_f32(ui, "Range Angle", &mut p.range_angle, 0.0, 6.28);
                changed |= slider_f32(ui, "Offset X", &mut p.base.offset_x, -10000.0, 10000.0);
                changed |= slider_f32(ui, "Offset Z", &mut p.base.offset_z, -10000.0, 10000.0);
                changed |= slider_f32(ui, "Lacunarity", &mut p.base.lacunarity, 1.1, 4.0);
                changed |= slider_f32(ui, "Persistence", &mut p.base.persistence, 0.1, 0.95);
                changed |= slider_f32(ui, "In Min", &mut p.base.remap_min, -2.0, 0.0);
                changed |= slider_f32(ui, "In Max", &mut p.base.remap_max, 0.0, 2.0);
            } else if split_noise {
                // Shape tab â€” landform controls only; seed/noise live on Noise tab.
                changed |= slider_f32(ui, "Peak Height", &mut p.base.amplitude, 0.0, 1000.0);
                changed |= slider_f32(ui, "Ridge Sharpness", &mut p.ridge_sharpness, 0.5, 4.0);
                changed |= slider_f32(ui, "Crest Gouges", &mut p.crest_detail, 0.0, 120.0);
                changed |= slider_f32(ui, "Range Width", &mut p.range_width, 0.05, 0.8);
            } else {
                changed |= seed_row(ui, "mnt_seed", &mut p.base.seed);
                changed |= slider_f32(ui, "Peak Height", &mut p.base.amplitude, 0.0, 1000.0);
                let mut octaves = p.base.octaves as i32;
                if slider_i32(ui, "Fine Detail", &mut octaves, 1, 12) {
                    p.base.octaves = octaves as u32;
                    changed = true;
                }
                changed |= slider_f32(ui, "Ridge Sharpness", &mut p.ridge_sharpness, 0.5, 4.0);
                changed |= slider_f32(ui, "Crest Gouges", &mut p.crest_detail, 0.0, 120.0);
            }
        }
        LayerKind::Volcano(p) => {
            if !split_noise {
                changed |= seed_row(ui, "volcano_seed", &mut p.seed);
            }
            changed |= slider_f32(ui, "Height", &mut p.height, 50.0, 1200.0);
            changed |= slider_f32(ui, "Radius", &mut p.radius, 0.05, 0.6);
            changed |= slider_f32(ui, "Crater Radius", &mut p.crater_radius, 0.0, 0.6);
            changed |= slider_f32(ui, "Crater Depth", &mut p.crater_depth, 0.0, 300.0);
            if advanced {
                changed |= slider_f32(ui, "Flank Power", &mut p.flank_power, 0.4, 3.0);
                changed |= slider_f32(ui, "Roughness", &mut p.roughness, 0.0, 80.0);
                changed |= slider_f32(ui, "Center U", &mut p.center_u, 0.0, 1.0);
                changed |= slider_f32(ui, "Center V", &mut p.center_v, 0.0, 1.0);
            }
        }
        LayerKind::Uplift(p) => {
            if !split_noise && !advanced {
                changed |= seed_row(ui, "uplift_seed", &mut p.seed);
            }
            if advanced {
                changed |= slider_f32(ui, "Range Angle", &mut p.range_angle, 0.0, 6.28);
                changed |= slider_f32(ui, "Warp", &mut p.warp_strength, 0.0, 1.5);
            } else {
                changed |= slider_f32(ui, "Uplift Height", &mut p.amplitude, 0.0, 1200.0);
                changed |= slider_f32(ui, "Corridor Width", &mut p.corridor_width, 0.08, 0.9);
                changed |= slider_f32(ui, "Ridge Power", &mut p.ridge_power, 0.5, 4.0);
                if !split_noise {
                    changed |= slider_f32(ui, "Detail", &mut p.detail_amplitude, 0.0, 150.0);
                }
                changed |= slider_f32(ui, "Valley Clarity", &mut p.altitude_fade, 0.0, 1.0);
            }
        }
        LayerKind::Dunes(p) => {
            changed |= slider_f32(ui, "Direction", &mut p.direction_deg, 0.0, 360.0);
            changed |= slider_f32(ui, "Wind Strength", &mut p.wind_strength, 0.05, 2.0);
            changed |= slider_f32(ui, "Sand Supply", &mut p.sand_supply, 0.05, 2.0);
            let mut scale = if p.dune_scale > 1e-8 {
                p.dune_scale
            } else {
                p.wave_frequency
            };
            if slider_f32(ui, "Dune Scale", &mut scale, 0.001, 0.05) {
                p.dune_scale = scale;
                p.wave_frequency = scale;
                changed = true;
            }
            let mut height = if p.dune_height > 1e-5 {
                p.dune_height
            } else {
                p.base.amplitude
            };
            if slider_f32(ui, "Dune Height", &mut height, 1.0, 200.0) {
                p.dune_height = height;
                p.base.amplitude = height;
                changed = true;
            }
            let mut crest = if p.crest_sharpness > 1e-5 {
                p.crest_sharpness
            } else {
                p.asymmetry
            };
            if slider_f32(ui, "Crest Sharpness", &mut crest, 0.0, 1.0) {
                p.crest_sharpness = crest;
                p.asymmetry = crest;
                changed = true;
            }
            changed |= slider_f32(ui, "Linearity", &mut p.linearity, 0.0, 1.0);
            changed |= slider_f32(ui, "Transport Length", &mut p.transport_length, 0.5, 16.0);
            changed |= slider_f32(ui, "Avalanche Angle", &mut p.avalanche_angle, 15.0, 45.0);
            let mut iters = p.iterations as i32;
            if slider_i32(ui, "Iterations", &mut iters, 1, 48) {
                p.iterations = iters as u32;
                changed = true;
            }
            if advanced {
                changed |= slider_f32(ui, "Trough Depth", &mut p.trough_depth, 0.0, 40.0);
                changed |= slider_f32(ui, "Basin Floor", &mut p.basin_floor, 0.0, 0.5);
                changed |= slider_f32(ui, "Lacunarity", &mut p.base.lacunarity, 1.1, 4.0);
                changed |= slider_f32(ui, "Persistence", &mut p.base.persistence, 0.1, 0.95);
            }
        }
        LayerKind::Canyons(p) => {
            changed |= slider_f32(ui, "Depth", &mut p.depth, 10.0, 500.0);
            changed |= slider_f32(ui, "Width", &mut p.width, 10.0, 400.0);
            if advanced {
                changed |= slider_f32(ui, "Meander", &mut p.meander, 0.0, 1.0);
            }
        }
        LayerKind::VoronoiRegions(p) => {
            changed |= edit_noise(ui, &mut p.base, advanced);
            if advanced {
                changed |= slider_f32(ui, "Jitter", &mut p.cell_jitter, 0.0, 1.0);
            }
        }
        LayerKind::ImportHeightmap(p) => {
            label(ui, "Path:");
            label(
                ui,
                if p.path.is_empty() {
                    "(empty)"
                } else {
                    &p.path
                },
            );
            if button_id(ui, Id::new("insp_browse_heightmap"), "Browseâ€¦") {
                actions.push(PanelAction::BrowseHeightmapPath { layer: layer_id });
            }
            changed |= slider_f32(ui, "Scale", &mut p.height_scale, 1.0, 2000.0);
            if advanced {
                changed |= slider_f32(ui, "Offset", &mut p.height_offset, -500.0, 500.0);
            }
        }
        LayerKind::ThermalErosion(p) => {
            changed |= slider_f32(ui, "Strength", &mut p.strength, 0.0, 1.0);
            changed |= slider_f32(ui, "Talus Angle", &mut p.talus_angle_deg, 5.0, 60.0);
            changed |= slider_f32(ui, "Material Amount", &mut p.material_amount, 0.1, 40.0);
            changed |= slider_f32(ui, "Weathering Rate", &mut p.weathering_rate, 0.0, 4.0);
            changed |= slider_f32(
                ui,
                "Transport Distance",
                &mut p.transport_distance,
                1.0,
                8.0,
            );
            changed |= slider_f32(ui, "Bedrock Hardness", &mut p.hardness, 0.0, 1.0);
            changed |= materials_k_toggle(ui, &mut p.hardness_source);
            changed |= checkbox(ui, "Layered bedrock / debris", &mut p.layered_materials);
            changed |= edit_level_step_ex(
                ui,
                &mut p.level_count,
                &mut p.start_level,
                &mut p.level_step_strength,
                Some(&mut p.level_step_curve),
            );
            if advanced {
                let mut iters = p.iterations as i32;
                if slider_i32(ui, "Iterations", &mut iters, 1, 200) {
                    p.iterations = iters as u32;
                    changed = true;
                }
            }
        }
        LayerKind::DebrisFlow(p) => {
            let mut iters = p.iterations as i32;
            if slider_i32(ui, "Iterations", &mut iters, 1, 200) {
                p.iterations = iters as u32;
                changed = true;
            }
            changed |= slider_f32(ui, "Time Step", &mut p.dt, 0.1, 40.0);
            changed |= slider_f32(ui, "Talus Angle", &mut p.talus_angle_deg, 15.0, 45.0);
            changed |= slider_f32(ui, "Thermal K", &mut p.thermal_k, 0.0, 0.05);
            changed |= slider_f32(ui, "Abrasion K", &mut p.abrasion_k, 0.0, 0.02);
            changed |= slider_f32(ui, "Yield Stress", &mut p.yield_stress, 0.05, 1.5);
            changed |= slider_f32(ui, "Fluvial K", &mut p.fluvial_k, 0.0, 0.01);
            changed |= slider_f32(ui, "Fluvial Deposit", &mut p.fluvial_deposition, 0.0, 0.2);
            changed |= slider_f32(ui, "Hillslope K", &mut p.hillslope_k, 0.0, 0.2);
            changed |= slider_f32(ui, "Precipitation", &mut p.precipitation, 0.0, 4.0);
            changed |= slider_f32(ui, "Bedrock Hardness", &mut p.hardness, 0.0, 1.0);
            changed |= materials_k_toggle(ui, &mut p.hardness_source);
            changed |= checkbox(ui, "Reroute after deposition", &mut p.refill_depressions);
        }
        LayerKind::HydraulicErosion(p) => {
            if advanced {
                let mut iters = p.iterations as i32;
                if slider_i32(ui, "Iterations", &mut iters, 1, 200) {
                    p.iterations = iters as u32;
                    changed = true;
                }
                changed |= slider_f32(ui, "Rain", &mut p.rainfall, 0.0, 0.2);
                changed |= slider_f32(ui, "Evap", &mut p.evaporation, 0.0, 0.2);
                changed |= slider_f32(ui, "Capacity", &mut p.capacity, 0.0, 1.0);
                changed |= slider_f32(ui, "Erosion", &mut p.erosion, 0.0, 1.0);
                changed |= slider_f32(ui, "Deposit", &mut p.deposition, 0.0, 1.0);
                changed |= slider_f32(ui, "Fan Boost", &mut p.fan_boost, 0.0, 2.5);
                changed |= slider_f32(ui, "Floodplain", &mut p.floodplain_bias, 0.0, 2.0);
                changed |= slider_f32(ui, "Bank Slip", &mut p.bank_slip, 0.0, 1.0);
                changed |= slider_f32(ui, "Sed Soft", &mut p.sediment_softness, 0.0, 1.0);
                changed |= slider_f32(ui, "Timestep", &mut p.timestep, 0.01, 2.0);
                changed |= slider_f32(ui, "Hardness", &mut p.hardness, 0.0, 1.0);
                changed |= materials_k_toggle(ui, &mut p.hardness_source);
                changed |= slider_f32(ui, "Rill Detail", &mut p.particle_density, 0.0, 0.5);
                changed |= slider_f32(ui, "Rill Inertia", &mut p.particle_inertia, 0.0, 0.95);
                let mut lifetime = p.particle_lifetime as i32;
                if slider_i32(ui, "Rill Lifetime", &mut lifetime, 4, 96) {
                    p.particle_lifetime = lifetime as u32;
                    changed = true;
                }
                let mut radius = p.particle_radius as i32;
                if slider_i32(ui, "Rill Radius", &mut radius, 1, 6) {
                    p.particle_radius = radius as u32;
                    changed = true;
                }
                changed |= named_source_toggle(
                    ui,
                    "Use Climate Rainfall",
                    &mut p.rainfall_source,
                    terra_core::fields::keys::RAINFALL,
                );
                changed |= named_source_toggle(
                    ui,
                    "Protect Mountain Mass",
                    &mut p.protection_source,
                    terra_core::fields::keys::MOUNTAIN_MASK,
                );
                changed |= slider_f32(ui, "Protection", &mut p.protection_strength, 0.0, 1.0);
                changed |= edit_level_step_ex(
                    ui,
                    &mut p.level_count,
                    &mut p.start_level,
                    &mut p.level_step_strength,
                    Some(&mut p.level_step_curve),
                );
            } else {
                // Strength, Wetness, Iterations
                changed |= slider_f32(ui, "Weathering Strength", &mut p.erosion, 0.0, 1.0);
                changed |= slider_f32(ui, "Rainfall", &mut p.rainfall, 0.0, 0.2);
                changed |= slider_f32(ui, "Deposit", &mut p.deposition, 0.0, 1.0);
                changed |= slider_f32(ui, "Fan Boost", &mut p.fan_boost, 0.0, 2.5);
                changed |= slider_f32(ui, "Hardness", &mut p.hardness, 0.0, 1.0);
                changed |= materials_k_toggle(ui, &mut p.hardness_source);
                let mut iters = p.iterations as i32;
                if slider_i32(ui, "Iterations", &mut iters, 1, 200) {
                    p.iterations = iters as u32;
                    changed = true;
                }
                changed |= named_source_toggle(
                    ui,
                    "Use Climate Rainfall",
                    &mut p.rainfall_source,
                    terra_core::fields::keys::RAINFALL,
                );
                changed |= named_source_toggle(
                    ui,
                    "Protect Mountain Mass",
                    &mut p.protection_source,
                    terra_core::fields::keys::MOUNTAIN_MASK,
                );
                changed |= slider_f32(ui, "Protection", &mut p.protection_strength, 0.0, 1.0);
                changed |= edit_level_step_ex(
                    ui,
                    &mut p.level_count,
                    &mut p.start_level,
                    &mut p.level_step_strength,
                    Some(&mut p.level_step_curve),
                );
            }
        }
        LayerKind::RiverCarve(p) => {
            changed |= slider_f32(ui, "Depth", &mut p.depth, 1.0, 100.0);
            changed |= slider_f32(ui, "Width", &mut p.width, 1.0, 20.0);
            changed |= slider_f32(ui, "Guide Boost", &mut p.guide_boost, 0.0, 8.0);
            let guide_label = match &p.guide {
                terra_core::mask::MaskSource::None => "Guide: None",
                terra_core::mask::MaskSource::Wetness => "Guide: Wetness",
                terra_core::mask::MaskSource::FlowAccumulation { .. } => "Guide: Flow Acc",
                terra_core::mask::MaskSource::Painted { .. } => "Guide: Painted",
                _ => "Guide: Custom",
            };
            if button_id(ui, Id::new("insp_river_guide"), guide_label) {
                p.guide = match &p.guide {
                    terra_core::mask::MaskSource::None => terra_core::mask::MaskSource::Wetness,
                    terra_core::mask::MaskSource::Wetness => {
                        terra_core::mask::MaskSource::FlowAccumulation { min: 0.0, max: 1.0 }
                    }
                    _ => terra_core::mask::MaskSource::None,
                };
                changed = true;
            }
            if advanced {
                changed |= slider_f32(ui, "Threshold", &mut p.accumulation_threshold, 1.0, 500.0);
                changed |= slider_f32(ui, "Bank Smooth", &mut p.bank_smooth, 0.0, 10.0);
                changed |= checkbox(ui, "D-inf routing", &mut p.use_dinfinity);
            }
        }
        LayerKind::StreamPowerErosion(p) => {
            changed |= slider_f32(ui, "Erodibility K", &mut p.k, 0.0, 0.5);
            changed |= slider_f32(ui, "Hardness", &mut p.hardness, 0.0, 1.0);
            changed |= materials_k_toggle(ui, &mut p.hardness_source);
            let mut iters = p.iterations as i32;
            if slider_i32(ui, "Iterations", &mut iters, 1, 80) {
                p.iterations = iters as u32;
                changed = true;
            }
            changed |= edit_level_step_ex(
                ui,
                &mut p.level_count,
                &mut p.start_level,
                &mut p.level_step_strength,
                Some(&mut p.level_step_curve),
            );
            if advanced {
                changed |= slider_f32(ui, "Area exp m", &mut p.m, 0.1, 1.0);
                changed |= slider_f32(ui, "Slope exp n", &mut p.n, 0.1, 2.0);
                changed |= slider_f32(ui, "Timestep", &mut p.dt, 0.1, 4.0);
                changed |= slider_f32(ui, "Uplift/iter", &mut p.uplift_rate, 0.0, 5.0);
                changed |= slider_f32(ui, "Base Level", &mut p.base_level, -50.0, 200.0);
                changed |= slider_f32(ui, "Valley Seed", &mut p.dendritic_seed, 0.0, 2.0);
                changed |= slider_f32(ui, "Stream Thresh", &mut p.stream_threshold, 1.0, 200.0);
                changed |= checkbox(ui, "D-inf routing", &mut p.use_dinfinity);
                changed |= checkbox(ui, "Refill each iter", &mut p.refill_each_iter);
            } else {
                changed |= slider_f32(ui, "Valley Seed", &mut p.dendritic_seed, 0.0, 2.0);
            }
        }
        LayerKind::MultiScaleAmplify(p) => {
            changed |= slider_f32(ui, "Thermal", &mut p.thermal_strength, 0.0, 1.0);
            changed |= slider_f32(ui, "SPE Strength", &mut p.spe_strength, 0.0, 1.5);
            changed |= slider_f32(ui, "Detail Boost", &mut p.detail_boost, 0.5, 2.0);
            changed |= slider_f32(ui, "Hardness", &mut p.hardness, 0.0, 1.0);
            changed |= slider_f32(ui, "Lock Strength", &mut p.lock_strength, 0.0, 1.0);
            if advanced {
                let mut levels = p.level_count as i32;
                if slider_i32(ui, "Levels (0=auto)", &mut levels, 0, 4) {
                    p.level_count = levels as u32;
                    changed = true;
                }
                let mut th = p.thermal_iters as i32;
                if slider_i32(ui, "Thermal Iters", &mut th, 1, 40) {
                    p.thermal_iters = th as u32;
                    changed = true;
                }
                let mut spe = p.spe_iters as i32;
                if slider_i32(ui, "SPE Iters", &mut spe, 0, 24) {
                    p.spe_iters = spe as u32;
                    changed = true;
                }
                changed |= slider_f32(ui, "Talus deg", &mut p.talus_angle_deg, 5.0, 60.0);
                changed |= slider_f32(ui, "Deposition", &mut p.deposition_strength, 0.0, 1.5);
            }
        }
        LayerKind::Blur(p) => {
            let mut radius = p.radius as i32;
            if slider_i32(ui, "Radius", &mut radius, 1, 8) {
                p.radius = radius as u32;
                changed = true;
            }
            if advanced {
                let mut iters = p.iterations as i32;
                if slider_i32_id(ui, Id::new("blur_iters"), "Iterations", &mut iters, 1, 8) {
                    p.iterations = iters as u32;
                    changed = true;
                }
            }
        }
        LayerKind::Coastal(p) => {
            changed |= slider_f32(ui, "Sea Level", &mut p.sea_level, -50.0, 100.0);
            changed |= slider_f32(ui, "Beach", &mut p.beach_width, 1.0, 100.0);
            if advanced {
                changed |= checkbox(ui, "Flatten below sea", &mut p.flatten_below);
                changed |= slider_f32(ui, "Shelf Depth", &mut p.shelf_depth, 0.0, 200.0);
            }
        }
        LayerKind::Vegetation(p) => {
            let mut seed = p.seed.min(99999) as i32;
            if slider_i32(ui, "Seed", &mut seed, 0, 99999) {
                p.seed = seed as u64;
                changed = true;
            }
            changed |= slider_f32(ui, "Density", &mut p.density, 0.0, 1.0);
            changed |= slider_f32(ui, "Scale Min", &mut p.scale_min, 0.1, 4.0);
            changed |= slider_f32(ui, "Scale Max", &mut p.scale_max, 0.1, 4.0);
            changed |= slider_f32(ui, "Yaw Variation", &mut p.yaw_variation_deg, 0.0, 180.0);
            label(ui, &format!("Coverage masks: {}", p.coverage.len()));
            if advanced {
                changed |= slider_f32(ui, "Min Dist", &mut p.min_distance, 1.0, 20.0);
                changed |= slider_f32(ui, "Min Slope", &mut p.min_slope_deg, 0.0, 90.0);
                changed |= slider_f32(ui, "Max Slope", &mut p.max_slope_deg, 0.0, 90.0);
                changed |= slider_f32(ui, "Root Cohesion", &mut p.root_cohesion, 0.0, 0.5);
            }
        }
        LayerKind::Materials(p) => {
            label(
                ui,
                &format!(
                    "{} rules, {} strata, {} coverage",
                    p.rules.len(),
                    p.strata.len(),
                    p.coverage.len()
                ),
            );
            changed |= slider_f32(ui, "Default K", &mut p.default_hardness, 0.0, 1.0);
            if !advanced {
                label(ui, "Enable Advanced to edit rules / strata.");
            }
            if advanced {
                for (i, rule) in p.rules.iter_mut().enumerate() {
                    label(ui, &rule.name);
                    let mut rid = rule.id as i32;
                    if slider_i32_id(ui, Id::new("mat_id").with(i as u64), "ID", &mut rid, 0, 64) {
                        rule.id = rid as u32;
                        changed = true;
                    }
                    changed |= slider_f32_id(
                        ui,
                        Id::new("mat_hardness").with(i as u64),
                        "Hardness",
                        &mut rule.hardness,
                        0.0,
                        1.0,
                    );
                    changed |= slider_f32_id(
                        ui,
                        Id::new("mat_min_s").with(i as u64),
                        "Min Slope",
                        &mut rule.min_slope_deg,
                        0.0,
                        90.0,
                    );
                    changed |= slider_f32_id(
                        ui,
                        Id::new("mat_max_s").with(i as u64),
                        "Max Slope",
                        &mut rule.max_slope_deg,
                        0.0,
                        90.0,
                    );
                    label(ui, "Albedo PNG:");
                    label(
                        ui,
                        rule.albedo_path
                            .as_deref()
                            .filter(|p| !p.is_empty())
                            .unwrap_or("(tint only)"),
                    );
                    if rule.albedo_path.as_ref().is_some_and(|p| !p.is_empty())
                        && button_id(
                            ui,
                            Id::new("mat_albedo_clear").with(i as u64),
                            "Clear Albedo",
                        )
                    {
                        rule.albedo_path = None;
                        changed = true;
                    }
                }
                if button_id(ui, Id::new("add_mat_rule"), "Add Material Rule") {
                    let id = p.rules.iter().map(|rule| rule.id).max().unwrap_or(0) + 1;
                    p.rules.push(MaterialRule {
                        name: format!("Material {id}"),
                        id,
                        min_slope_deg: 0.0,
                        max_slope_deg: 90.0,
                        min_height: OPEN_HEIGHT_MIN,
                        max_height: OPEN_HEIGHT_MAX,
                        mask: terra_core::mask::MaskSource::None,
                        hardness: 0.5,
                        tint: [0.45, 0.42, 0.38],
                        roughness: 0.75,
                        metalness: 0.0,
                        albedo_path: None,
                    });
                    changed = true;
                }
                label(ui, "Strata (surface → bedrock)");
                if button_id(
                    ui,
                    Id::new("bed_geom"),
                    &format!("Beds: {}", p.bed_geometry.label()),
                ) {
                    p.bed_geometry = p.bed_geometry.cycle();
                    changed = true;
                }
                match &mut p.bed_geometry {
                    BedGeometry::Tilted {
                        dip_deg,
                        azimuth_deg,
                    } => {
                        changed |= slider_f32(ui, "Dip °", dip_deg, 0.0, 60.0);
                        changed |= slider_f32(ui, "Dip Azimuth °", azimuth_deg, 0.0, 360.0);
                    }
                    BedGeometry::Folded {
                        amplitude_m,
                        wavelength_m,
                        seed,
                    } => {
                        changed |= slider_f32(ui, "Fold Amp m", amplitude_m, 1.0, 120.0);
                        changed |= slider_f32(ui, "Fold Wavelength m", wavelength_m, 20.0, 800.0);
                        let mut s = (*seed).min(i32::MAX as u64) as i32;
                        if slider_i32(ui, "Fold Seed", &mut s, 0, 9999) {
                            *seed = s as u64;
                            changed = true;
                        }
                    }
                    BedGeometry::Warped {
                        frequency,
                        amplitude_m,
                        seed,
                    } => {
                        changed |= slider_f32(ui, "Warp Freq", frequency, 0.001, 0.08);
                        changed |= slider_f32(ui, "Warp Amp m", amplitude_m, 1.0, 120.0);
                        let mut s = (*seed).min(i32::MAX as u64) as i32;
                        if slider_i32(ui, "Warp Seed", &mut s, 0, 9999) {
                            *seed = s as u64;
                            changed = true;
                        }
                    }
                    BedGeometry::Horizontal => {}
                }
                for (i, s) in p.strata.iter_mut().enumerate() {
                    label(ui, &s.name);
                    if button_id(
                        ui,
                        Id::new("stratum_mat").with(i as u64),
                        &format!("Type: {}", s.material_type.label()),
                    ) {
                        s.material_type = s.material_type.cycle();
                        changed = true;
                    }
                    changed |= slider_f32_id(
                        ui,
                        Id::new("stratum_k").with(i as u64),
                        "Hardness",
                        &mut s.hardness,
                        0.0,
                        1.0,
                    );
                    let mut ero = if s.erodibility < 0.0 {
                        1.0 - s.hardness
                    } else {
                        s.erodibility
                    };
                    if slider_f32_id(
                        ui,
                        Id::new("stratum_e").with(i as u64),
                        "Erodibility",
                        &mut ero,
                        0.0,
                        1.0,
                    ) {
                        s.erodibility = ero;
                        changed = true;
                    }
                    changed |= slider_f32_id(
                        ui,
                        Id::new("stratum_t").with(i as u64),
                        "Thickness m",
                        &mut s.thickness,
                        0.5,
                        200.0,
                    );
                }
                if button_id(ui, Id::new("add_stratum"), "Add Soft-over-Hard Stack") {
                    *p = MaterialsParams::soft_over_hard(12.0);
                    changed = true;
                }
            }
        }
        LayerKind::Biomes(p) => {
            changed |= checkbox(ui, "Use Climate", &mut p.use_climate);
            if p.use_climate {
                section_header(ui, "CLIMATE");
                changed |= slider_f32(ui, "Sea Temp", &mut p.climate.sea_level_temp, 0.0, 1.0);
                changed |= slider_f32(ui, "Lapse Rate", &mut p.climate.lapse_rate, 0.0, 0.005);
                changed |= slider_f32(ui, "Latitude", &mut p.climate.latitude, 0.0, 1.0);
                changed |= slider_f32(ui, "Temp Gradient", &mut p.climate.temp_gradient, 0.0, 0.5);
                changed |= slider_f32(ui, "Wind Dir Â°", &mut p.climate.wind_dir_deg, 0.0, 360.0);
                changed |= slider_f32(ui, "Base Precip", &mut p.climate.base_precip, 0.0, 1.0);
                changed |= slider_f32(
                    ui,
                    "Orographic",
                    &mut p.climate.orographic_strength,
                    0.0,
                    2.0,
                );
                changed |= slider_f32(
                    ui,
                    "Rain Shadow",
                    &mut p.climate.rain_shadow_strength,
                    0.0,
                    1.0,
                );
                if advanced {
                    changed |= slider_f32(ui, "Humidity", &mut p.climate.base_humidity, 0.0, 1.0);
                    changed |= slider_f32(ui, "Sea Level", &mut p.climate.sea_level, -20.0, 80.0);
                    changed |= slider_f32(
                        ui,
                        "Water Infl. m",
                        &mut p.climate.water_influence,
                        10.0,
                        500.0,
                    );
                    changed |= slider_f32(ui, "Snow Temp", &mut p.climate.snow_temp, 0.0, 0.6);
                    changed |= slider_f32(
                        ui,
                        "Snow Line m",
                        &mut p.climate.snow_line_height,
                        20.0,
                        600.0,
                    );
                    changed |= slider_f32(
                        ui,
                        "Wetnessâ†’Moist",
                        &mut p.climate.moisture_from_wetness,
                        0.0,
                        1.0,
                    );
                }
            }
            label(ui, &format!("{} biome bands (LUT)", p.bands.len()));
        }
        LayerKind::OverhangStamp(p) => {
            let mut seed = p.seed.min(99999) as i32;
            if slider_i32(ui, "Seed", &mut seed, 0, 99999) {
                p.seed = seed as u64;
                changed = true;
            }
            changed |= slider_f32(ui, "Center U", &mut p.u, 0.0, 1.0);
            changed |= slider_f32(ui, "Center V", &mut p.v, 0.0, 1.0);
            changed |= slider_f32(ui, "Radius UV", &mut p.radius_uv, 0.01, 0.4);
            changed |= slider_f32(ui, "Depth m", &mut p.depth, 1.0, 80.0);
            changed |= slider_f32(ui, "Lip m", &mut p.lip_height, 0.0, 20.0);
            changed |= slider_f32(ui, "Entrance Â°", &mut p.entrance_angle_deg, 0.0, 360.0);
            if advanced {
                changed |= slider_f32(ui, "Falloff", &mut p.falloff, 0.05, 1.0);
                changed |= slider_f32(ui, "Noise", &mut p.noise_amplitude, 0.0, 1.0);
            }
            label(ui, "Opt-in dual-height undercut; DEM floor + ceiling aux.");
        }
        LayerKind::LocalSdf(p) => {
            let mut seed = p.seed.min(99999) as i32;
            if slider_i32(ui, "Seed", &mut seed, 0, 99999) {
                p.seed = seed as u64;
                changed = true;
            }
            changed |= slider_f32(ui, "Center U", &mut p.u, 0.0, 1.0);
            changed |= slider_f32(ui, "Center V", &mut p.v, 0.0, 1.0);
            changed |= slider_f32(ui, "Size X", &mut p.size_x, 4.0, 120.0);
            changed |= slider_f32(ui, "Size Y", &mut p.size_y, 2.0, 60.0);
            changed |= slider_f32(ui, "Size Z", &mut p.size_z, 4.0, 120.0);
            changed |= slider_f32(ui, "Depth m", &mut p.depth, 2.0, 80.0);
            changed |= slider_f32(ui, "Entrance R", &mut p.entrance_radius, 1.0, 30.0);
            changed |= slider_f32(ui, "Entrance Â°", &mut p.entrance_angle_deg, 0.0, 360.0);
            if advanced {
                changed |= slider_f32(ui, "Lip m", &mut p.lip_height, 0.0, 10.0);
                changed |= slider_f32(ui, "Noise", &mut p.noise_amplitude, 0.0, 1.0);
                let mut samples = p.vertical_samples as i32;
                if slider_i32(ui, "Y Samples", &mut samples, 8, 64) {
                    p.vertical_samples = samples as u32;
                    changed = true;
                }
            }
            label(ui, "Local SDF cave pocket â†’ dual-height (CPU).");
        }
        LayerKind::EffectFilter(p) => {
            let filter_labels: Vec<&'static str> = EffectFilterKind::ALL
                .iter()
                .map(|kind| kind.label())
                .collect();
            let mut kind_index = EffectFilterKind::ALL
                .iter()
                .position(|kind| *kind == p.kind)
                .unwrap_or(0);
            if combo(ui, "Filter", &mut kind_index, &filter_labels) {
                p.kind = EffectFilterKind::ALL[kind_index];
                changed = true;
            }
            changed |= slider_f32(ui, "Strength", &mut p.strength, 0.0, 1.0);
            changed |= checkbox(ui, "Invert", &mut p.invert);

            let is_terrace = matches!(
                p.kind,
                EffectFilterKind::TerraceSimple
                    | EffectFilterKind::TerraceIrregular
                    | EffectFilterKind::TerraceSteep
            );
            let uses_radius = matches!(
                p.kind,
                EffectFilterKind::Smooth
                    | EffectFilterKind::SpikeRemoval
                    | EffectFilterKind::Denoise
                    | EffectFilterKind::Kuwahara
                    | EffectFilterKind::AngleBlur
                    | EffectFilterKind::DirectionalBlur
                    | EffectFilterKind::SmoothRidges
                    | EffectFilterKind::SoftFlows
                    | EffectFilterKind::WideFlows
                    | EffectFilterKind::BorderBlend
                    | EffectFilterKind::Canyon
                    | EffectFilterKind::Cliffs
                    | EffectFilterKind::CliffReinforce
                    | EffectFilterKind::Chipped
                    | EffectFilterKind::RockyCliffs
                    | EffectFilterKind::RockyPlateaus
                    | EffectFilterKind::RockyLayers
                    | EffectFilterKind::WindCarve
                    | EffectFilterKind::FlattenFilter
                    | EffectFilterKind::Inflate
                    | EffectFilterKind::Deflate
                    | EffectFilterKind::Balloon
                    | EffectFilterKind::WashedOff
            );
            let uses_frequency = matches!(
                p.kind,
                EffectFilterKind::Distortion
                    | EffectFilterKind::Strata
                    | EffectFilterKind::TerraceIrregular
                    | EffectFilterKind::RockyLayers
                    | EffectFilterKind::RockySharp
                    | EffectFilterKind::RockyWide
                    | EffectFilterKind::RockyCliffs
                    | EffectFilterKind::RockyHard
                    | EffectFilterKind::RockyPlateaus
                    | EffectFilterKind::Chipped
                    | EffectFilterKind::Rocky
                    | EffectFilterKind::Rugged
                    | EffectFilterKind::ScatterDetail
                    | EffectFilterKind::DesignVoronoi
                    | EffectFilterKind::NoiseBillow
                    | EffectFilterKind::NoiseGabor
                    | EffectFilterKind::NoisePerlin
                    | EffectFilterKind::NoisePhasor
                    | EffectFilterKind::NoiseRidged
                    | EffectFilterKind::NoiseSimplex
                    | EffectFilterKind::NoiseValue
                    | EffectFilterKind::NoiseVoronoi
                    | EffectFilterKind::NoiseWave
                    | EffectFilterKind::NoiseWhite
                    | EffectFilterKind::Crater
                    | EffectFilterKind::AngleBreak
                    | EffectFilterKind::Hexagons
                    | EffectFilterKind::Blocks
                    | EffectFilterKind::Ridged
                    | EffectFilterKind::SmoothRidges
                    | EffectFilterKind::Swirl
            );
            let uses_noise_fractal = p.is_noise_family()
                || matches!(
                    p.kind,
                    EffectFilterKind::DesignVoronoi
                        | EffectFilterKind::Rugged
                        | EffectFilterKind::Ridged
                        | EffectFilterKind::ScatterDetail
                );
            let uses_slope_gate = matches!(
                p.kind,
                EffectFilterKind::TerraceSteep
                    | EffectFilterKind::RockySharp
                    | EffectFilterKind::RockyWide
                    | EffectFilterKind::CliffReinforce
                    | EffectFilterKind::TalusFill
                    | EffectFilterKind::RockyPlateaus
                    | EffectFilterKind::RockyCliffs
                    | EffectFilterKind::RockyHard
                    | EffectFilterKind::Cliffs
                    | EffectFilterKind::Chipped
                    | EffectFilterKind::RockyLayers
                    | EffectFilterKind::AngleBreak
                    | EffectFilterKind::WindCarve
                    | EffectFilterKind::SedimentFlows
            );
            let uses_flow_gate = matches!(
                p.kind,
                EffectFilterKind::SoftFlows
                    | EffectFilterKind::ThinFlows
                    | EffectFilterKind::RidgedFlows
                    | EffectFilterKind::WideFlows
                    | EffectFilterKind::SedimentFlows
                    | EffectFilterKind::HydraulicSediment
                    | EffectFilterKind::WashedOff
                    | EffectFilterKind::Canyon
                    | EffectFilterKind::Cliffs
                    | EffectFilterKind::Chipped
                    | EffectFilterKind::RockyCliffs
                    | EffectFilterKind::RockyPlateaus
            );
            let uses_arid_lithology = matches!(
                p.kind,
                EffectFilterKind::Canyon
                    | EffectFilterKind::Cliffs
                    | EffectFilterKind::CliffReinforce
                    | EffectFilterKind::Chipped
                    | EffectFilterKind::RockySharp
                    | EffectFilterKind::RockyWide
                    | EffectFilterKind::RockyLayers
                    | EffectFilterKind::RockyPlateaus
                    | EffectFilterKind::RockyCliffs
                    | EffectFilterKind::RockyHard
                    | EffectFilterKind::Rocky
            );

            let arid_canyon = matches!(p.kind, EffectFilterKind::Canyon);
            let arid_cliffs = matches!(
                p.kind,
                EffectFilterKind::Cliffs | EffectFilterKind::CliffReinforce
            );

            if uses_radius {
                let mut radius = p.radius as i32;
                let radius_label = if arid_canyon {
                    "Canyon Width"
                } else if arid_cliffs {
                    "Detection Radius"
                } else {
                    "Radius"
                };
                if slider_i32(ui, radius_label, &mut radius, 1, 16) {
                    p.radius = radius as u32;
                    changed = true;
                }
            }
            if is_terrace {
                changed |= slider_f32(ui, "Terrace Height m", &mut p.terrace_height, 0.0, 80.0);
                changed |= slider_f32(ui, "Terrace Interval", &mut p.amount, 2.0, 32.0);
                changed |= slider_f32(ui, "Offset", &mut p.terrace_offset, 0.0, 1.0);
                changed |= slider_f32(ui, "Top Smoothness", &mut p.top_smoothness, 0.0, 1.0);
                changed |= slider_f32(ui, "Riser Sharpness", &mut p.riser_sharpness, 0.0, 1.0);
            } else {
                let amount_label = if arid_canyon {
                    "Incision"
                } else if arid_cliffs {
                    "Cliff Strength"
                } else {
                    "Amount"
                };
                changed |= slider_f32(ui, amount_label, &mut p.amount, 0.0, 120.0);
                let mut iterations = p.iterations as i32;
                let iter_label = if arid_canyon { "Age" } else { "Iterations" };
                if slider_i32(ui, iter_label, &mut iterations, 1, 64) {
                    p.iterations = iterations as u32;
                    changed = true;
                }
            }

            if uses_frequency {
                changed |= slider_f32(ui, "Scale m", &mut p.scale_m, 0.0, 500.0);
                changed |= slider_f32(ui, "Frequency", &mut p.frequency, 0.001, 0.2);
                let mut seed = p.seed.min(i32::MAX as u64) as i32;
                if slider_i32(ui, "Seed", &mut seed, 0, 9999) {
                    p.seed = seed as u64;
                    changed = true;
                }
            }
            if uses_noise_fractal {
                let mut octaves = p.octaves as i32;
                if slider_i32(ui, "Octaves", &mut octaves, 1, 12) {
                    p.octaves = octaves as u32;
                    changed = true;
                }
                changed |= slider_f32(ui, "Lacunarity", &mut p.lacunarity, 1.1, 4.0);
                changed |= slider_f32(ui, "Persistence", &mut p.persistence, 0.05, 0.95);
                changed |= slider_f32(ui, "Rotation", &mut p.rotation_deg, 0.0, 360.0);
                changed |= slider_f32(ui, "Anisotropy", &mut p.anisotropy, 0.2, 4.0);
                changed |= slider_f32(ui, "Warp Strength", &mut p.warp_strength, 0.0, 120.0);
                changed |= slider_f32(ui, "Warp Frequency", &mut p.warp_frequency, 0.001, 0.1);
                changed |= checkbox(ui, "Tileable", &mut p.tileable);
            }
            if matches!(
                p.kind,
                EffectFilterKind::DesignVoronoi | EffectFilterKind::NoiseVoronoi
            ) {
                let labels = ["F1", "F2", "F2−F1"];
                let mut idx = match p.voronoi_feature {
                    terra_core::layer::WorleyFeature::F1 => 0,
                    terra_core::layer::WorleyFeature::F2 => 1,
                    terra_core::layer::WorleyFeature::F2MinusF1 => 2,
                };
                if combo(ui, "Voronoi Feature", &mut idx, &labels) {
                    p.voronoi_feature = match idx {
                        0 => terra_core::layer::WorleyFeature::F1,
                        1 => terra_core::layer::WorleyFeature::F2,
                        _ => terra_core::layer::WorleyFeature::F2MinusF1,
                    };
                    changed = true;
                }
            }
            if matches!(
                p.kind,
                EffectFilterKind::DirectionalBlur
                    | EffectFilterKind::WindCarve
                    | EffectFilterKind::AngleBreak
                    | EffectFilterKind::NoiseWave
                    | EffectFilterKind::NoiseGabor
                    | EffectFilterKind::NoisePhasor
            ) && !uses_noise_fractal
            {
                changed |= slider_f32(ui, "Rotation", &mut p.rotation_deg, 0.0, 360.0);
            }
            if matches!(p.kind, EffectFilterKind::Shore) {
                changed |= slider_f32(ui, "Sea Level", &mut p.sea_level, -50.0, 200.0);
                changed |= slider_f32(ui, "Beach Width", &mut p.beach_width, 1.0, 80.0);
            }
            if matches!(p.kind, EffectFilterKind::Crater) {
                changed |= slider_f32(ui, "Crater Radius", &mut p.crater_radius, 0.05, 0.45);
            }
            if uses_slope_gate {
                changed |= slider_f32(ui, "Slope Min", &mut p.slope_min, 0.0, 90.0);
                changed |= slider_f32(ui, "Slope Max", &mut p.slope_max, 0.0, 90.0);
            }
            if uses_flow_gate {
                let flow_label = if arid_canyon {
                    "Tributary Influence"
                } else if arid_cliffs {
                    "Relief Threshold"
                } else {
                    "Flow Thresh"
                };
                changed |= slider_f32(ui, flow_label, &mut p.flow_threshold, 0.0, 1.0);
            }
            if uses_arid_lithology {
                changed |= slider_f32(ui, "Rock Hardness", &mut p.rock_hardness, 0.0, 1.0);
                if arid_canyon || arid_cliffs {
                    changed |= slider_f32(ui, "Wall Steepness", &mut p.wall_steepness, 0.0, 1.0);
                }
                if arid_canyon || matches!(p.kind, EffectFilterKind::RockyPlateaus) {
                    changed |= slider_f32(ui, "Valley Floor", &mut p.valley_floor, 0.0, 1.0);
                }
                if matches!(
                    p.kind,
                    EffectFilterKind::Canyon
                        | EffectFilterKind::Cliffs
                        | EffectFilterKind::RockyCliffs
                        | EffectFilterKind::RockyHard
                        | EffectFilterKind::RockyPlateaus
                ) {
                    changed |= slider_f32(ui, "Talus", &mut p.talus_mix, 0.0, 1.0);
                }
            }
            if advanced && !uses_frequency {
                let mut seed = p.seed.min(i32::MAX as u64) as i32;
                if slider_i32(ui, "Seed", &mut seed, 0, 9999) {
                    p.seed = seed as u64;
                    changed = true;
                }
            }
        }
        LayerKind::Path(p) => {
            changed |= slider_f32(ui, "Width", &mut p.width, 1.0, 80.0);
            changed |= slider_f32(ui, "Falloff", &mut p.falloff, 1.0, 80.0);
            changed |= slider_f32(ui, "Base Height / Depth", &mut p.height_offset, 0.0, 200.0);
            changed |= checkbox(ui, "Carve", &mut p.carve);
            changed |= checkbox(ui, "Closed", &mut p.closed);
            changed |= checkbox(ui, "Smooth Spline", &mut p.spline);
            changed |= slider_f32(ui, "Bank Profile", &mut p.profile, 0.25, 4.0);
            changed |= slider_f32(ui, "Natural Variation", &mut p.noise_strength, 0.0, 30.0);
            changed |= slider_f32(ui, "Variation Scale", &mut p.noise_scale, 0.0001, 0.1);
            for (index, node) in p.nodes.iter_mut().enumerate().take(16) {
                let h_label = format!("Node {} Height", index + 1);
                let w_label = format!("Node {} Width", index + 1);
                changed |= slider_f32(ui, &h_label, &mut node.height, -200.0, 500.0);
                changed |= slider_f32(ui, &w_label, &mut node.width, 0.1, 4.0);
            }
            if p.nodes.len() > 16 {
                label_dim(ui, "Showing the first 16 node profiles.");
            }
            label(ui, &format!("Nodes: {}", p.nodes.len()));
            if button_id(ui, Id::new("insp_edit_path"), "Edit in Viewport") {
                actions.push(PanelAction::SetEditorTool(EditorTool::EditPath));
            }
            if button_id(ui, Id::new("insp_finish_path"), "Finish Drawing") {
                actions.push(PanelAction::SetEditorTool(EditorTool::Move));
            }
            if button_id(ui, Id::new("insp_clear_path"), "Clear Nodes") {
                p.nodes.clear();
                changed = true;
            }
            if button_id(ui, Id::new("insp_undo_path_node"), "Remove Last Node")
                && !p.nodes.is_empty()
            {
                p.nodes.pop();
                changed = true;
            }
        }
        LayerKind::RiverNetwork(p) => {
            changed |= checkbox(ui, "Auto Generate", &mut p.auto_generate);
            changed |= slider_f32(ui, "Carve Depth", &mut p.carve_depth, 0.5, 80.0);
            changed |= slider_f32(ui, "Valley Width", &mut p.valley_width, 2.0, 400.0);
            let mut max_len = p.max_length as i32;
            if slider_i32(ui, "Max Length", &mut max_len, 16, 1024) {
                p.max_length = max_len as u32;
                changed = true;
            }
            label(ui, &format!("Springs: {}", p.springs.len()));
            if button_id(
                ui,
                Id::new("insp_edit_springs"),
                "Place Springs in Viewport",
            ) {
                actions.push(PanelAction::SetEditorTool(EditorTool::EditRiverSpring));
            }
            if button_id(ui, Id::new("insp_clear_springs"), "Clear Springs") {
                p.springs.clear();
                changed = true;
            }
            if button_id(ui, Id::new("insp_undo_spring"), "Remove Last Spring")
                && !p.springs.is_empty()
            {
                p.springs.pop();
                changed = true;
            }
        }
        LayerKind::SandSimulation(p) => {
            changed |= slider_f32(ui, "Direction", &mut p.wind_direction_deg, 0.0, 360.0);
            changed |= slider_f32(ui, "Wind Speed", &mut p.wind_speed, 0.05, 2.0);
            changed |= slider_f32(ui, "Sand Supply", &mut p.spawn_amount, 0.1, 12.0);
            changed |= slider_f32(ui, "Transport Length", &mut p.transport_length, 0.5, 20.0);
            changed |= slider_f32(ui, "Avalanche Angle", &mut p.slope_angle_deg, 12.0, 45.0);
            changed |= slider_f32(ui, "Linearity", &mut p.linearity, 0.0, 1.0);
            let mut iters = p.iterations as i32;
            if slider_i32(ui, "Iterations", &mut iters, 1, 4096) {
                p.iterations = iters as u32;
                changed = true;
            }
            changed |= slider_f32(ui, "Strength", &mut p.strength, 0.0, 1.0);
            if advanced {
                let mut aval = p.avalanche_iters as i32;
                if slider_i32(ui, "Avalanche Iters", &mut aval, 1, 64) {
                    p.avalanche_iters = aval as u32;
                    changed = true;
                }
                changed |= slider_f32(ui, "Abrasion", &mut p.abrasion, 0.0, 1.0);
                changed |= slider_f32(ui, "Reptation", &mut p.reptation, 0.0, 1.0);
                changed |= slider_f32(ui, "Seed Amount", &mut p.seed_amount, 0.0, 1.5);
                changed |= slider_f32(ui, "Seed Scale", &mut p.seed_scale, 0.001, 0.05);
            }
        }
        LayerKind::FluidSimulation(p) => {
            let mut iters = p.iterations as i32;
            if slider_i32(ui, "Iterations", &mut iters, 1, 128) {
                p.iterations = iters as u32;
                changed = true;
            }
            changed |= slider_f32(ui, "Viscosity", &mut p.viscosity, 0.05, 0.95);
            changed |= slider_f32(ui, "Spawn Amount", &mut p.spawn_amount, 0.1, 8.0);
            changed |= slider_f32(ui, "Strength", &mut p.strength, 0.0, 1.0);
        }
        LayerKind::ProceduralShape(p) => {
            section_header(ui, "LANDFORM");
            label(ui, p.generator.label());
            // Cycle through WC landscape generators (Mountain â†’ â€¦ â†’ Noise).
            if button_id(ui, Id::new("proc_shape_cycle"), "Change Landform") {
                p.generator = p.generator.cycle();
                changed = true;
            }
            ui.gap(4.0);
            match p.generator {
                ProceduralGenerator::Mountain => {
                    changed |= slider_f32(
                        ui,
                        "Peak Height",
                        &mut p.mountain.base.amplitude,
                        0.0,
                        1000.0,
                    );
                    changed |= slider_f32(
                        ui,
                        "Ridge Sharpness",
                        &mut p.mountain.ridge_sharpness,
                        0.5,
                        4.0,
                    );
                    changed |=
                        slider_f32(ui, "Crest Gouges", &mut p.mountain.crest_detail, 0.0, 120.0);
                    changed |=
                        slider_f32(ui, "Range Width", &mut p.mountain.range_width, 0.05, 0.8);
                }
                ProceduralGenerator::Hills | ProceduralGenerator::Plateau => {
                    changed |=
                        slider_f32(ui, "Amplitude", &mut p.hills.base.amplitude, 0.0, 1000.0);
                    changed |=
                        slider_f32(ui, "Frequency", &mut p.hills.base.frequency, 0.0005, 0.05);
                    if matches!(p.generator, ProceduralGenerator::Plateau) {
                        changed |= slider_f32(ui, "Low", &mut p.plateau.low, -200.0, 2000.0);
                        changed |= slider_f32(ui, "High", &mut p.plateau.high, -200.0, 2000.0);
                        changed |= slider_f32(ui, "Soft", &mut p.plateau.soft, 0.5, 80.0);
                    }
                }
                ProceduralGenerator::Mesa => {
                    changed |= slider_f32(ui, "Height", &mut p.mesa.height, 10.0, 800.0);
                    changed |= slider_f32(ui, "Radius", &mut p.mesa.radius, 0.05, 0.6);
                    changed |=
                        slider_f32(ui, "Edge Steepness", &mut p.mesa.edge_steepness, 1.0, 8.0);
                }
                ProceduralGenerator::Volcano => {
                    changed |= slider_f32(ui, "Height", &mut p.volcano.height, 50.0, 1200.0);
                    changed |= slider_f32(ui, "Radius", &mut p.volcano.radius, 0.05, 0.6);
                    changed |=
                        slider_f32(ui, "Crater Radius", &mut p.volcano.crater_radius, 0.0, 0.6);
                    changed |=
                        slider_f32(ui, "Crater Depth", &mut p.volcano.crater_depth, 0.0, 300.0);
                }
                ProceduralGenerator::Dunes => {
                    changed |= slider_f32(ui, "Amplitude", &mut p.dunes.base.amplitude, 0.0, 400.0);
                    changed |=
                        slider_f32(ui, "Frequency", &mut p.dunes.base.frequency, 0.0005, 0.05);
                }
                ProceduralGenerator::Canyon => {
                    changed |= slider_f32(ui, "Depth", &mut p.canyon.depth, 10.0, 600.0);
                }
                ProceduralGenerator::Crater => {
                    changed |= slider_f32(ui, "Strength", &mut p.crater.strength, 0.0, 1.0);
                    changed |= slider_f32(ui, "Amount", &mut p.crater.amount, 1.0, 120.0);
                    changed |=
                        slider_f32(ui, "Crater Radius", &mut p.crater.crater_radius, 0.05, 0.45);
                }
                ProceduralGenerator::Noise => {
                    changed |= edit_noise(ui, &mut p.noise, advanced);
                }
            }
        }
        LayerKind::Stamp2d(p) => {
            label(ui, "Path:");
            label(
                ui,
                if p.heightmap.path.is_empty() {
                    "(empty)"
                } else {
                    &p.heightmap.path
                },
            );
            if button_id(ui, Id::new("insp_browse_stamp2d"), "Browseâ€¦") {
                actions.push(PanelAction::BrowseHeightmapPath { layer: layer_id });
            }
            changed |= slider_f32(ui, "Scale", &mut p.heightmap.height_scale, 1.0, 2000.0);
            if advanced {
                changed |= slider_f32(ui, "Offset", &mut p.heightmap.height_offset, -500.0, 500.0);
            }
        }
        LayerKind::Stamp3d(p) => {
            label(ui, "Mesh path:");
            label(
                ui,
                if p.path.is_empty() {
                    "(empty â€” procedural rock)"
                } else {
                    &p.path
                },
            );
            if button_id(ui, Id::new("insp_browse_stamp3d"), "Browseâ€¦") {
                actions.push(PanelAction::BrowseMeshPath { layer: layer_id });
            }
            changed |= slider_f32(ui, "Scale", &mut p.height_scale, 0.01, 100.0);
            changed |= slider_f32(ui, "Offset", &mut p.height_offset, -500.0, 500.0);
        }
        LayerKind::PolygonHeight(p) => {
            changed |= slider_f32(ui, "Height", &mut p.height, 0.0, 1000.0);
            changed |= slider_f32(ui, "Falloff", &mut p.falloff, 0.0, 0.5);
            changed |= checkbox(ui, "Carve", &mut p.carve);
            let mode_label = match p.mode {
                PolygonHeightMode::RaiseBy => "Mode: Raise by Height",
                PolygonHeightMode::SetElevation => "Mode: Set Elevation",
            };
            if button_id(ui, Id::new("insp_polygon_mode"), mode_label) {
                p.mode = match p.mode {
                    PolygonHeightMode::RaiseBy => PolygonHeightMode::SetElevation,
                    PolygonHeightMode::SetElevation => PolygonHeightMode::RaiseBy,
                };
                changed = true;
            }

            label(ui, &format!("Vertices: {}", p.points.len()));
            if button_id(ui, Id::new("insp_edit_polygon"), "Edit in Viewport") {
                actions.push(PanelAction::SetEditorTool(EditorTool::EditPolygon));
            }
            if button_id(ui, Id::new("insp_finish_polygon"), "Finish Drawing") {
                actions.push(PanelAction::SetEditorTool(EditorTool::Move));
            }
            if button_id(ui, Id::new("insp_clear_polygon"), "Clear Vertices") {
                p.points.clear();
                changed = true;
            }
            if button_id(
                ui,
                Id::new("insp_undo_polygon_vertex"),
                "Remove Last Vertex",
            ) && !p.points.is_empty()
            {
                p.points.pop();
                changed = true;
            }
        }
    }
    changed
}

pub(crate) fn edit_noise(ui: &mut GuiContext<'_>, p: &mut NoiseParams, advanced: bool) -> bool {
    let mut changed = false;
    if advanced {
        section_header(ui, "TRANSFORM");
        changed |= seed_row(ui, "noise_seed", &mut p.seed);
        changed |= slider_f32(ui, "Offset X", &mut p.offset_x, -10000.0, 10000.0);
        changed |= slider_f32(ui, "Offset Z", &mut p.offset_z, -10000.0, 10000.0);

        section_header(ui, "SHAPE");
        changed |= slider_f32(ui, "Terrain Scale", &mut p.frequency, 0.0001, 0.05);
        changed |= slider_f32(ui, "Height Range", &mut p.amplitude, 0.0, 1000.0);

        section_header(ui, "FRACTAL");
        let mut octaves = p.octaves as i32;
        if slider_i32(ui, "Fine Detail", &mut octaves, 1, 12) {
            p.octaves = octaves as u32;
            changed = true;
        }
        changed |= slider_f32(ui, "Lacunarity", &mut p.lacunarity, 1.1, 4.0);
        changed |= slider_f32(ui, "Persistence", &mut p.persistence, 0.1, 0.95);

        section_header(ui, "OUTPUT RANGE");
        changed |= slider_f32(ui, "In Min", &mut p.remap_min, -2.0, 0.0);
        changed |= slider_f32(ui, "In Max", &mut p.remap_max, 0.0, 2.0);
    } else {
        changed |= seed_row(ui, "noise_seed", &mut p.seed);
        changed |= slider_f32(ui, "Terrain Scale", &mut p.frequency, 0.0001, 0.05);
        changed |= slider_f32(ui, "Height Range", &mut p.amplitude, 0.0, 1000.0);
        let mut octaves = p.octaves as i32;
        if slider_i32(ui, "Fine Detail", &mut octaves, 1, 12) {
            p.octaves = octaves as u32;
            changed = true;
        }
    }
    changed
}

/// Seed field with a refresh button (mockup style).
pub(crate) fn seed_row(ui: &mut GuiContext<'_>, id_key: &str, seed: &mut u64) -> bool {
    let mut value = (*seed).min(99999) as i32;
    let changed = slider_i32_id(
        ui,
        Id::new(id_key).child("seed_slider"),
        "Seed",
        &mut value,
        0,
        99999,
    );
    if changed {
        *seed = value as u64;
    }
    // Full-width secondary action â€” clearer hit target than a lone icon.
    ui.gap(2.0);
    let row = ui.allocate(ROW_H);
    let id = Id::new(id_key).child("seed_refresh");
    let hovered = ui.pointer_in(row);
    if hovered {
        ui.state.set_hot(id);
    }
    if hovered && ui.input.primary_pressed {
        ui.state.active = Some(id);
    }
    let refreshed = ui.input.primary_released && ui.state.is_active(id) && hovered;
    let bg = if ui.state.is_active(id) && hovered {
        style::BUTTON_ACTIVE
    } else if hovered {
        style::BUTTON_HOVER
    } else {
        style::BUTTON_BG
    };
    ui.panel_rounded(row, bg, style::RADIUS_SM);
    let icon_sz = 14.0;
    ui.icon_at(
        row.min_x + 10.0,
        row.min_y + (row.height() - icon_sz) * 0.5,
        Icon::Redo2,
        style::TEXT,
        icon_sz,
    );
    let text_r = Rect::from_pos_size(
        row.min_x + 30.0,
        row.min_y,
        row.width() - 40.0,
        row.height(),
    );
    ui.label_in_rect(text_r, "Randomize seed", style::TEXT, FONT_SCALE);
    if refreshed {
        *seed = seed.wrapping_mul(1103515245).wrapping_add(12345) % 100_000;
        if *seed == 0 {
            *seed = 1;
        }
    }
    changed || refreshed
}

/// Noise-tab content for layers that expose a separate Noise section.
pub(crate) fn edit_kind_noise(ui: &mut GuiContext<'_>, kind: &mut LayerKind) -> bool {
    match kind {
        LayerKind::NoiseValue(p) | LayerKind::NoisePerlin(p) | LayerKind::NoiseOpenSimplex(p) => {
            edit_noise(ui, p, false)
        }
        LayerKind::NoiseWorley(p) => edit_noise(ui, &mut p.base, false),
        LayerKind::Fbm(p) | LayerKind::Ridged(p) => edit_noise(ui, &mut p.base, false),
        LayerKind::DomainWarp(p) => edit_noise(ui, &mut p.base, false),
        LayerKind::VoronoiRegions(p) => edit_noise(ui, &mut p.base, false),
        LayerKind::Mountains(p) => edit_noise(ui, &mut p.base, false),
        LayerKind::Dunes(p) => edit_noise(ui, &mut p.base, false),
        LayerKind::Mesa(p) => seed_row(ui, "mesa_noise_seed", &mut p.seed),
        LayerKind::Volcano(p) => {
            let mut changed = seed_row(ui, "volcano_noise_seed", &mut p.seed);
            changed |= slider_f32(ui, "Roughness", &mut p.roughness, 0.0, 80.0);
            changed
        }
        LayerKind::Island(p) => {
            let mut changed = seed_row(ui, "island_noise_seed", &mut p.seed);
            changed |= slider_f32(
                ui,
                "Coast Frequency",
                &mut p.coastline_frequency,
                0.0002,
                0.006,
            );
            changed |= slider_f32(ui, "Ridge Frequency", &mut p.ridge_frequency, 0.0002, 0.012);
            changed
        }
        LayerKind::Uplift(p) => {
            let mut changed = seed_row(ui, "uplift_noise_seed", &mut p.seed);
            changed |= slider_f32(ui, "Frequency", &mut p.frequency, 0.0001, 0.01);
            changed |= slider_f32(ui, "Detail", &mut p.detail_amplitude, 0.0, 150.0);
            changed |= slider_f32(ui, "Detail Freq", &mut p.detail_frequency, 0.0005, 0.02);
            let mut oct = p.detail_octaves as i32;
            if slider_i32(ui, "Detail Octaves", &mut oct, 1, 8) {
                p.detail_octaves = oct as u32;
                changed = true;
            }
            changed |= slider_f32(ui, "Warp", &mut p.warp_strength, 0.0, 1.5);
            changed
        }
        _ => {
            label(ui, "No noise parameters for this layer.");
            false
        }
    }
}

/// Toggle erosion `hardness_source` between Materials aux \(K\) and constant hardness.
pub(crate) fn materials_k_toggle(
    ui: &mut GuiContext<'_>,
    source: &mut terra_core::mask::MaskSource,
) -> bool {
    let mut use_mats = matches!(source, terra_core::mask::MaskSource::Hardness);
    if checkbox(ui, "Use Materials K", &mut use_mats) {
        *source = if use_mats {
            terra_core::mask::MaskSource::Hardness
        } else {
            terra_core::mask::MaskSource::None
        };
        true
    } else {
        false
    }
}

pub(crate) fn named_source_toggle(
    ui: &mut GuiContext<'_>,
    label_text: &str,
    source: &mut terra_core::mask::MaskSource,
    key: &str,
) -> bool {
    let mut enabled = matches!(source, terra_core::mask::MaskSource::Named(name) if name == key);
    if checkbox(ui, label_text, &mut enabled) {
        *source = if enabled {
            terra_core::mask::MaskSource::Named(key.to_string())
        } else {
            terra_core::mask::MaskSource::None
        };
        true
    } else {
        false
    }
}

/// WC-style multilevel schedule controls shared by Thermal / Hydraulic / SPE.
pub(crate) fn edit_level_step_ex(
    ui: &mut GuiContext<'_>,
    level_count: &mut u32,
    start_level: &mut u32,
    level_step_strength: &mut f32,
    curve: Option<&mut Vec<f32>>,
) -> bool {
    let mut changed = false;
    label(ui, "LEVEL STEPS");
    let mut levels = *level_count as i32;
    if slider_i32(ui, "Levels (0=auto)", &mut levels, 0, 8) {
        *level_count = levels as u32;
        changed = true;
    }
    let mut start = *start_level as i32;
    if slider_i32(ui, "Start Level", &mut start, 0, 7) {
        *start_level = start as u32;
        changed = true;
    }
    changed |= slider_f32(ui, "Level Step Strength", level_step_strength, 0.25, 2.0);
    if let Some(curve) = curve {
        let bars = if *level_count == 0 {
            4usize
        } else {
            (*level_count as usize).clamp(1, 8)
        };
        if curve.len() != bars {
            curve.resize(bars, 1.0);
            changed = true;
        }
        for i in 0..bars {
            let label_s = format!("L{} Strength", i);
            if slider_f32(ui, &label_s, &mut curve[i], 0.0, 2.0) {
                changed = true;
            }
        }
    }
    changed
}

pub(crate) fn kind_display_name(kind: &LayerKind) -> &'static str {
    use LayerKind::*;
    match kind {
        SculptBase(_) => "Base",
        Flat(_) => "Flat",
        Ramp(_) => "Ramp",
        NoiseValue(_) => "Value Noise",
        SculptStrokes(_) => "Sculpt",
        TerrainConstraints(_) => "Constraints",
        GradientReconstruct(_) => "Gradient",
        LandscapeEvolution(_) => "Landscape Evolution",
        HydrologyRepair(_) => "Hydrology",
        GeomorphicDetail(_) => "Geomorphic Detail",
        EcosystemFeedback(_) => "Ecosystem",
        NoisePerlin(_) => "Perlin",
        NoiseOpenSimplex(_) => "OpenSimplex",
        NoiseWorley(_) => "Worley",
        Fbm(_) => "fBm",
        Ridged(_) => "Ridged",
        DomainWarp(_) => "Domain Warp",
        Terrace(_) => "Terrace",
        Plateau(_) => "Plateau",
        Mesa(_) => "Mesa",
        Mountains(_) => "Mountains",
        Volcano(_) => "Volcano",
        Uplift(_) => "Uplift",
        Dunes(_) => "Dunes",
        Canyons(_) => "Canyons",
        VoronoiRegions(_) => "Voronoi",
        ImportHeightmap(_) => "Heightmap",
        ThermalErosion(_) => "Thermal",
        DebrisFlow(_) => "Debris Flow",
        HydraulicErosion(_) => "Hydraulic",
        StreamPowerErosion(_) => "Stream Power",
        MultiScaleAmplify(_) => "Multi-Scale",
        RiverCarve(_) => "Rivers",
        Island(_) => "Island",
        Blur(_) => "Blur",
        Coastal(_) => "Coastal",
        Materials(_) => "Materials",
        Biomes(_) => "Biome",
        Vegetation(_) => "Objects",
        OverhangStamp(_) => "Overhang",
        LocalSdf(_) => "Cave",
        EffectFilter(p) => p.kind.label(),
        Path(_) => "Path",
        RiverNetwork(_) => "River Network",
        SandSimulation(_) => "Sand Sim",
        FluidSimulation(_) => "Fluid Sim",
        // WC Procedural shape tool — show the active landform, not "Procedural".
        ProceduralShape(p) => p.generator.label(),
        Stamp2d(_) => "2D Stamp",
        Stamp3d(_) => "3D Stamp",
        PolygonHeight(_) => "Polygon",
    }
}
