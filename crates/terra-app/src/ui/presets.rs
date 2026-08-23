use terra_core::layer::*;
use terra_core::WorldTemplate;

#[derive(Debug, Clone)]
pub struct LayerPreset {
    pub name: String,
    pub description: String,
    pub layers: Vec<(String, LayerKind)>,
}

/// Starting recipe for a new project (home / File â†’ New).
#[derive(Debug, Clone)]
pub struct ProjectTemplate {
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    pub default_file_name: &'static str,
    pub layers: Vec<(String, LayerKind)>,
}

pub fn world_design_templates() -> Vec<ProjectTemplate> {
    // WorldTemplate::all() is already Blank-first cause→effect catalog order.
    project_templates()
}

/// A non-destructive preset for the currently selected layer.
#[derive(Debug, Clone)]
pub struct ContextualPreset {
    pub name: &'static str,
    pub description: &'static str,
    pub kind: LayerKind,
}

/// Returns artist-facing variations for a layer kind.
pub fn contextual_presets(kind: &LayerKind) -> Vec<ContextualPreset> {
    let preset = |name, description, modify: fn(&mut LayerKind)| {
        let mut variant = kind.clone();
        modify(&mut variant);
        ContextualPreset {
            name,
            description,
            kind: variant,
        }
    };

    match kind {
        LayerKind::Mountains(_) => vec![
            preset("Alpine", "Tall, crisp mountain ranges.", |k| {
                if let LayerKind::Mountains(p) = k {
                    p.base.amplitude = 650.0;
                    p.base.frequency = 0.0012;
                    p.base.octaves = 7;
                    p.ridge_sharpness = 2.8;
                    p.range_width = 0.28;
                    p.crest_detail = 55.0;
                }
            }),
            preset(
                "Volcanic",
                "Softer ridged massif (not a cone â€” use Volcano layer).",
                |k| {
                    if let LayerKind::Mountains(p) = k {
                        p.base.amplitude = 420.0;
                        p.base.frequency = 0.0022;
                        p.base.octaves = 4;
                        p.ridge_sharpness = 1.1;
                        p.range_width = 0.5;
                        p.crest_detail = 20.0;
                    }
                },
            ),
            preset("Rounded", "Weathered, rolling highlands.", |k| {
                if let LayerKind::Mountains(p) = k {
                    p.base.amplitude = 280.0;
                    p.base.frequency = 0.0018;
                    p.base.octaves = 4;
                    p.ridge_sharpness = 0.8;
                    p.range_width = 0.6;
                    p.crest_detail = 12.0;
                }
            }),
            preset("Jagged", "Knife-edge peaks with crest gouges.", |k| {
                if let LayerKind::Mountains(p) = k {
                    p.base.amplitude = 800.0;
                    p.base.frequency = 0.0025;
                    p.base.octaves = 9;
                    p.ridge_sharpness = 3.6;
                    p.range_width = 0.18;
                    p.crest_detail = 85.0;
                }
            }),
            preset(
                "Reference Peak",
                "Tall alpine spine for deep couloir erosion.",
                |k| {
                    if let LayerKind::Mountains(p) = k {
                        p.base.amplitude = 720.0;
                        p.base.frequency = 0.0014;
                        p.base.octaves = 7;
                        p.ridge_sharpness = 3.1;
                        p.range_width = 0.24;
                        p.crest_detail = 75.0;
                    }
                },
            ),
        ],
        LayerKind::Volcano(_) => vec![
            preset("Shield", "Broad, gentle shield volcano.", |k| {
                if let LayerKind::Volcano(p) = k {
                    p.height = 380.0;
                    p.radius = 0.38;
                    p.flank_power = 1.05;
                    p.crater_radius = 0.12;
                    p.crater_depth = 40.0;
                    p.roughness = 12.0;
                }
            }),
            preset("Stratovolcano", "Tall cone with deep crater.", |k| {
                if let LayerKind::Volcano(p) = k {
                    p.height = 620.0;
                    p.radius = 0.24;
                    p.flank_power = 1.55;
                    p.crater_radius = 0.2;
                    p.crater_depth = 110.0;
                    p.roughness = 22.0;
                }
            }),
            preset("Caldera", "Wide crater, collapsed rim.", |k| {
                if let LayerKind::Volcano(p) = k {
                    p.height = 480.0;
                    p.radius = 0.32;
                    p.flank_power = 1.2;
                    p.crater_radius = 0.42;
                    p.crater_depth = 160.0;
                    p.roughness = 16.0;
                }
            }),
        ],
        LayerKind::Mesa(_) => vec![
            preset("Mesa", "Broad hard-cap tableland.", |k| {
                if let LayerKind::Mesa(p) = k {
                    *p = MesaParams::default();
                }
            }),
            preset("Butte", "Small footprint, tall steep walls.", |k| {
                if let LayerKind::Mesa(p) = k {
                    *p = MesaParams::butte();
                }
            }),
            preset("Wide Cap", "Large radius, gentler walls.", |k| {
                if let LayerKind::Mesa(p) = k {
                    p.radius = 0.34;
                    p.height = 220.0;
                    p.edge_steepness = 2.4;
                    p.soft = 0.22;
                    p.cap_noise = 8.0;
                }
            }),
        ],
        LayerKind::Uplift(_) => vec![
            preset("Range Spine", "Narrow corridor, sharp crest.", |k| {
                if let LayerKind::Uplift(p) = k {
                    p.corridor_width = 0.28;
                    p.ridge_power = 2.2;
                    p.altitude_fade = 0.85;
                    p.detail_amplitude = 35.0;
                }
            }),
            preset("Broad Massif", "Wide uplift with soft flanks.", |k| {
                if let LayerKind::Uplift(p) = k {
                    p.corridor_width = 0.55;
                    p.ridge_power = 1.2;
                    p.amplitude = 320.0;
                    p.altitude_fade = 0.55;
                }
            }),
            preset("Meandered", "Warped corridors for drainage.", |k| {
                if let LayerKind::Uplift(p) = k {
                    p.warp_strength = 0.65;
                    p.corridor_width = 0.38;
                    p.detail_amplitude = 55.0;
                }
            }),
        ],
        LayerKind::HydraulicErosion(_) => vec![
            preset(
                "Light Weathering",
                "A subtle pass over existing terrain.",
                |k| {
                    if let LayerKind::HydraulicErosion(p) = k {
                        p.iterations = 20;
                        p.rainfall = 0.01;
                        p.erosion = 0.12;
                        p.deposition = 0.15;
                        p.level_step_strength = 0.85;
                    }
                },
            ),
            preset(
                "Young Terrain",
                "Sharp channels and limited sediment.",
                |k| {
                    if let LayerKind::HydraulicErosion(p) = k {
                        p.iterations = 45;
                        p.rainfall = 0.035;
                        p.erosion = 0.45;
                        p.deposition = 0.12;
                        p.level_count = 3;
                        p.level_step_strength = 1.15;
                    }
                },
            ),
            preset(
                "Mature Valleys",
                "Deep drainage with settled basins.",
                |k| {
                    if let LayerKind::HydraulicErosion(p) = k {
                        p.iterations = 120;
                        p.rainfall = 0.06;
                        p.erosion = 0.5;
                        p.deposition = 0.5;
                        p.capacity = 0.22;
                        p.level_count = 3;
                        p.level_step_strength = 1.25;
                    }
                },
            ),
            preset(
                "Alluvial Fan",
                "Deposition-forward fan / floodplain.",
                |k| {
                    if let LayerKind::HydraulicErosion(p) = k {
                        *p = HydraulicErosionParams::depositional();
                    }
                },
            ),
        ],
        LayerKind::ThermalErosion(_) => vec![
            preset("Light Polish", "Gentle talus cleanup.", |k| {
                if let LayerKind::ThermalErosion(p) = k {
                    p.iterations = 20;
                    p.strength = 0.35;
                    p.level_step_strength = 0.9;
                }
            }),
            preset("Strong Talus", "Heavy scree / bank-slip look.", |k| {
                if let LayerKind::ThermalErosion(p) = k {
                    p.iterations = 55;
                    p.strength = 0.7;
                    p.talus_angle_deg = 32.0;
                    p.level_count = 3;
                    p.level_step_strength = 1.2;
                }
            }),
        ],
        LayerKind::StreamPowerErosion(_) => vec![
            preset("Gentle Incision", "Light SPE valleys.", |k| {
                if let LayerKind::StreamPowerErosion(p) = k {
                    p.iterations = 16;
                    p.k = 0.05;
                    p.level_step_strength = 0.9;
                }
            }),
            preset("Deep Couloirs", "Aggressive alpine SPE.", |k| {
                if let LayerKind::StreamPowerErosion(p) = k {
                    p.iterations = 36;
                    p.k = 0.12;
                    p.dendritic_seed = 0.7;
                    p.level_count = 3;
                    p.level_step_strength = 1.3;
                }
            }),
        ],
        LayerKind::Dunes(_) => vec![
            preset("Linear", "Long parallel dune ridges.", |k| {
                if let LayerKind::Dunes(p) = k {
                    p.base.frequency = 0.002;
                    p.base.amplitude = 32.0;
                    p.wave_frequency = 0.018;
                    p.dune_scale = 0.018;
                    p.dune_height = 32.0;
                    p.asymmetry = 0.55;
                    p.crest_sharpness = 0.55;
                    p.linearity = 0.92;
                    p.sand_supply = 0.95;
                    p.wind_strength = 0.8;
                    p.trough_depth = 6.0;
                    p.iterations = 12;
                }
            }),
            preset("Crescent", "Compact, wind-shaped barchans.", |k| {
                if let LayerKind::Dunes(p) = k {
                    p.base.frequency = 0.005;
                    p.base.amplitude = 45.0;
                    p.wave_frequency = 0.012;
                    p.dune_scale = 0.012;
                    p.dune_height = 45.0;
                    p.asymmetry = 0.78;
                    p.crest_sharpness = 0.82;
                    p.linearity = 0.35;
                    p.sand_supply = 0.4;
                    p.wind_strength = 1.15;
                    p.transport_length = 6.5;
                    p.trough_depth = 10.0;
                    p.iterations = 14;
                }
            }),
            preset("Star", "High dunes from variable winds.", |k| {
                if let LayerKind::Dunes(p) = k {
                    p.base.frequency = 0.0035;
                    p.base.amplitude = 75.0;
                    p.base.octaves = 5;
                    p.wave_frequency = 0.008;
                    p.dune_scale = 0.008;
                    p.dune_height = 75.0;
                    p.asymmetry = 0.45;
                    p.crest_sharpness = 0.5;
                    p.linearity = 0.12;
                    p.sand_supply = 1.1;
                    p.wind_strength = 0.9;
                    p.trough_depth = 14.0;
                    p.iterations = 16;
                }
            }),
            preset(
                "Basin Dunes",
                "Deep interdune troughs and soft basin floor.",
                |k| {
                    if let LayerKind::Dunes(p) = k {
                        p.base.frequency = 0.0028;
                        p.base.amplitude = 55.0;
                        p.wave_frequency = 0.011;
                        p.dune_scale = 0.011;
                        p.dune_height = 55.0;
                        p.asymmetry = 0.7;
                        p.crest_sharpness = 0.7;
                        p.linearity = 0.7;
                        p.sand_supply = 0.85;
                        p.trough_depth = 22.0;
                        p.basin_floor = 0.08;
                        p.iterations = 12;
                    }
                },
            ),
            preset("Wind Swept", "Sparse ridges under strong wind.", |k| {
                if let LayerKind::Dunes(p) = k {
                    p.base.frequency = 0.0015;
                    p.base.amplitude = 28.0;
                    p.wave_frequency = 0.025;
                    p.dune_scale = 0.025;
                    p.dune_height = 28.0;
                    p.asymmetry = 0.9;
                    p.crest_sharpness = 0.9;
                    p.linearity = 0.8;
                    p.sand_supply = 0.35;
                    p.wind_strength = 1.35;
                    p.trough_depth = 10.0;
                    p.basin_floor = 0.02;
                    p.iterations = 10;
                }
            }),
            preset("Transverse Field", "Dense wind-aligned dune field.", |k| {
                if let LayerKind::Dunes(p) = k {
                    p.dune_scale = 0.014;
                    p.wave_frequency = 0.014;
                    p.dune_height = 38.0;
                    p.base.amplitude = 38.0;
                    p.linearity = 0.85;
                    p.sand_supply = 1.2;
                    p.crest_sharpness = 0.6;
                    p.asymmetry = 0.6;
                    p.wind_strength = 1.0;
                    p.iterations = 14;
                }
            }),
        ],
        LayerKind::Vegetation(_) => vec![
            preset(
                "Sparse Forest",
                "Widely spaced trees on gentle slopes.",
                |k| {
                    if let LayerKind::Vegetation(p) = k {
                        p.density = 0.12;
                        p.min_distance = 8.0;
                        p.max_slope_deg = 25.0;
                    }
                },
            ),
            preset(
                "Dense Forest",
                "Thick coverage across rolling terrain.",
                |k| {
                    if let LayerKind::Vegetation(p) = k {
                        p.density = 0.72;
                        p.min_distance = 2.5;
                        p.max_slope_deg = 35.0;
                    }
                },
            ),
            preset("Alpine Scrub", "Sparse growth above steep terrain.", |k| {
                if let LayerKind::Vegetation(p) = k {
                    p.density = 0.24;
                    p.min_distance = 5.0;
                    p.min_slope_deg = 8.0;
                    p.max_slope_deg = 48.0;
                }
            }),
        ],
        _ => Vec::new(),
    }
}

fn alpine_spe() -> StreamPowerParams {
    StreamPowerParams {
        iterations: 36,
        k: 0.0008,
        dt: 0.25,
        m: 0.5,
        n: 1.05,
        dendritic_seed: 0.65,
        refill_each_iter: true,
        stream_threshold: 28.0,
        hardness: 0.0,
        hardness_source: terra_core::mask::MaskSource::Hardness,
        level_count: 3,
        level_step_strength: 1.25,
        ..StreamPowerParams::default()
    }
}

fn alpine_thermal() -> ThermalErosionParams {
    ThermalErosionParams {
        talus_angle_deg: 36.0,
        iterations: 48,
        strength: 0.55,
        hardness: 0.0,
        hardness_source: terra_core::mask::MaskSource::Hardness,
        level_count: 3,
        level_step_strength: 1.2,
        ..ThermalErosionParams::default()
    }
}

fn fidelity_amplify(
    spe_strength: f32,
    detail_boost: f32,
    deposition: f32,
) -> MultiScaleAmplifyParams {
    MultiScaleAmplifyParams {
        level_count: 4,
        thermal_strength: 0.32,
        thermal_iters: 14,
        talus_angle_deg: 35.0,
        spe_strength,
        spe_iters: 8,
        deposition_strength: deposition,
        detail_boost,
        hardness: 0.0,
        hardness_source: terra_core::mask::MaskSource::Hardness,
        ridge_lock: terra_core::mask::MaskSource::Hardness,
        lock_strength: 0.88,
    }
}

fn mountain_uplift_stack() -> Vec<(String, LayerKind)> {
    vec![
        (
            "Base".into(),
            LayerKind::SculptBase(SculptParams::filled(512, 35.0)),
        ),
        (
            "Uplift".into(),
            LayerKind::Uplift(UpliftParams {
                seed: 17,
                amplitude: 520.0,
                corridor_width: 0.3,
                ridge_power: 2.15,
                warp_strength: 0.45,
                detail_amplitude: 55.0,
                detail_octaves: 4,
                altitude_fade: 0.88,
                ..UpliftParams::default()
            }),
        ),
        (
            "Crest Detail".into(),
            LayerKind::Mountains(MountainParams {
                base: NoiseParams {
                    seed: 91,
                    frequency: 0.0028,
                    amplitude: 95.0,
                    octaves: 5,
                    ..NoiseParams::default()
                },
                ridge_sharpness: 2.8,
                range_angle: 0.35,
                range_width: 0.32,
                crest_detail: 40.0,
            }),
        ),
        (
            "Rock & Snow".into(),
            LayerKind::Materials(MaterialsParams::alpine_peak()),
        ),
        (
            "Stream Power".into(),
            LayerKind::StreamPowerErosion(StreamPowerParams {
                iterations: 32,
                k: 0.0008,
                dt: 0.25,
                dendritic_seed: 0.55,
                refill_each_iter: true,
                hardness_source: terra_core::mask::MaskSource::Hardness,
                level_count: 3,
                level_step_strength: 1.2,
                ..StreamPowerParams::default()
            }),
        ),
        (
            "Rill Weathering".into(),
            LayerKind::HydraulicErosion(HydraulicErosionParams {
                iterations: 64,
                rainfall: 0.028,
                capacity: 0.16,
                erosion: 0.38,
                deposition: 0.24,
                bank_slip: 0.28,
                particle_density: 0.13,
                particle_lifetime: 40,
                particle_inertia: 0.24,
                particle_radius: 2,
                hardness_source: terra_core::mask::MaskSource::Hardness,
                level_count: 4,
                level_step_strength: 1.18,
                ..HydraulicErosionParams::default()
            }),
        ),
        (
            "Micro Geomorphology".into(),
            LayerKind::MultiScaleAmplify(fidelity_amplify(0.045, 1.42, 0.22)),
        ),
        (
            "Talus".into(),
            LayerKind::ThermalErosion(ThermalErosionParams {
                iterations: 36,
                strength: 0.5,
                talus_angle_deg: 35.0,
                hardness_source: terra_core::mask::MaskSource::Hardness,
                level_count: 3,
                level_step_strength: 1.15,
                ..ThermalErosionParams::default()
            }),
        ),
    ]
}

pub fn builtin_presets() -> Vec<LayerPreset> {
    vec![
        LayerPreset {
            name: "Rolling Hills".into(),
            description: "Soft fBm hills over a flat base".into(),
            layers: vec![
                (
                    "Base".into(),
                    LayerKind::SculptBase(SculptParams::filled(512, 30.0)),
                ),
                (
                    "Hills".into(),
                    LayerKind::Fbm(FbmParams {
                        base: NoiseParams {
                            seed: 11,
                            frequency: 0.0012,
                            amplitude: 90.0,
                            octaves: 5,
                            ..NoiseParams::default()
                        },
                        noise: FractalNoiseType::Perlin,
                    }),
                ),
            ],
        },
        LayerPreset {
            name: "Volcano".into(),
            description: "Radial cone with crater, materials hardness, thermal talus".into(),
            layers: vec![
                (
                    "Base".into(),
                    LayerKind::SculptBase(SculptParams::filled(512, 35.0)),
                ),
                (
                    "Massif".into(),
                    LayerKind::Fbm(FbmParams {
                        base: NoiseParams {
                            seed: 7,
                            frequency: 0.0009,
                            amplitude: 55.0,
                            octaves: 3,
                            ..NoiseParams::default()
                        },
                        noise: FractalNoiseType::Perlin,
                    }),
                ),
                ("Cone".into(), LayerKind::Volcano(VolcanoParams::default())),
                (
                    "Caprock".into(),
                    LayerKind::Materials(MaterialsParams {
                        default_hardness: 0.55,
                        ..MaterialsParams::soft_over_hard(14.0)
                    }),
                ),
                (
                    "Thermal".into(),
                    LayerKind::ThermalErosion(ThermalErosionParams {
                        talus_angle_deg: 32.0,
                        iterations: 40,
                        strength: 0.6,
                        hardness: 0.0,
                        hardness_source: terra_core::mask::MaskSource::Hardness,
                        level_step_strength: 1.1,
                        ..ThermalErosionParams::default()
                    }),
                ),
            ],
        },
        LayerPreset {
            name: "Alpine Range".into(),
            description: "WC arid-mountain craft: peaks â†’ Rocky Plateaus â†’ Soft/Thin Flows â†’ Talus Fill".into(),
            layers: vec![
                (
                    "Base".into(),
                    LayerKind::SculptBase(SculptParams::filled(512, 45.0)),
                ),
                (
                    "Peaks".into(),
                    LayerKind::Mountains(MountainParams {
                        base: NoiseParams {
                            seed: 31,
                            frequency: 0.00135,
                            amplitude: 680.0,
                            octaves: 8,
                            lacunarity: 2.2,
                            persistence: 0.48,
                            ..NoiseParams::default()
                        },
                        ridge_sharpness: 3.2,
                        range_angle: 0.62,
                        range_width: 0.26,
                        crest_detail: 72.0,
                    }),
                ),
                (
                    "Rock & Snow".into(),
                    LayerKind::Materials(MaterialsParams::alpine_peak()),
                ),
                ("Stream Power".into(), LayerKind::StreamPowerErosion(alpine_spe())),
                (
                    "Rocky Plateaus".into(),
                    LayerKind::EffectFilter(EffectFilterParams::rocky_plateaus()),
                ),
                (
                    "Soft Flows".into(),
                    LayerKind::EffectFilter(EffectFilterParams::soft_flows()),
                ),
                (
                    "Thin Flows".into(),
                    LayerKind::EffectFilter(EffectFilterParams::thin_flows()),
                ),
                (
                    "Talus Fill".into(),
                    LayerKind::EffectFilter(EffectFilterParams::talus_fill()),
                ),
                (
                    "Gullies".into(),
                    LayerKind::HydraulicErosion(HydraulicErosionParams {
                        iterations: 72,
                        rainfall: 0.03,
                        erosion: 0.4,
                        deposition: 0.26,
                        capacity: 0.16,
                        fan_boost: 0.35,
                        bank_slip: 0.3,
                        particle_density: 0.15,
                        particle_lifetime: 44,
                        particle_inertia: 0.22,
                        particle_radius: 2,
                        hardness: 0.0,
                        hardness_source: terra_core::mask::MaskSource::Hardness,
                        level_count: 3,
                        level_step_strength: 1.2,
                        ..HydraulicErosionParams::default()
                    }),
                ),
                (
                    "Micro Geomorphology".into(),
                    LayerKind::MultiScaleAmplify(fidelity_amplify(0.045, 1.48, 0.2)),
                ),
                ("Talus".into(), LayerKind::ThermalErosion(alpine_thermal())),
                (
                    "Climate".into(),
                    LayerKind::Biomes(BiomesParams {
                        use_climate: true,
                        climate: ClimateParams {
                            sea_level_temp: 0.42,
                            latitude: 0.62,
                            lapse_rate: 0.0011,
                            snow_line_height: 180.0,
                            snow_temp: 0.48,
                            orographic_strength: 1.2,
                            rain_shadow_strength: 0.55,
                            wind_dir_deg: 240.0,
                            sea_level: 5.0,
                            ..ClimateParams::default()
                        },
                        ..BiomesParams::default()
                    }),
                ),
            ],
        },
        LayerPreset {
            name: "Mountain Range (Uplift)".into(),
            description: "High-fidelity uplift spine with couloirs, particle rills, micro-relief and alpine PBR".into(),
            layers: mountain_uplift_stack(),
        },
        LayerPreset {
            name: "Dendritic Range".into(),
            description: "High-fidelity dendritic basins with facet drainage, particle tributaries and alluvial fans".into(),
            layers: vec![
                (
                    "Base".into(),
                    LayerKind::SculptBase(SculptParams::filled(512, 40.0)),
                ),
                (
                    "Peaks".into(),
                    LayerKind::Mountains(MountainParams {
                        base: NoiseParams {
                            seed: 44,
                            frequency: 0.0015,
                            amplitude: 560.0,
                            octaves: 6,
                            ..NoiseParams::default()
                        },
                        ridge_sharpness: 2.9,
                        range_width: 0.34,
                        crest_detail: 68.0,
                        ..MountainParams::default()
                    }),
                ),
                (
                    "Rock".into(),
                    LayerKind::Materials(MaterialsParams::alpine_peak()),
                ),
                (
                    "Stream Power".into(),
                    LayerKind::StreamPowerErosion(StreamPowerParams {
                        iterations: 34,
                        k: 0.0008,
                        dt: 0.25,
                        dendritic_seed: 0.85,
                        refill_each_iter: true,
                        hardness_source: terra_core::mask::MaskSource::Hardness,
                        level_count: 3,
                        level_step_strength: 1.3,
                        ..StreamPowerParams::default()
                    }),
                ),
                (
                    "Fans".into(),
                    LayerKind::HydraulicErosion({
                        let mut p = HydraulicErosionParams::depositional();
                        p.hardness_source = terra_core::mask::MaskSource::Hardness;
                        p.level_count = 3;
                        p.level_step_strength = 1.15;
                        p
                    }),
                ),
                (
                    "Drainage Detail".into(),
                    LayerKind::MultiScaleAmplify(fidelity_amplify(0.05, 1.5, 0.42)),
                ),
                (
                    "Talus".into(),
                    LayerKind::ThermalErosion(ThermalErosionParams {
                        iterations: 40,
                        strength: 0.5,
                        hardness_source: terra_core::mask::MaskSource::Hardness,
                        level_step_strength: 1.1,
                        ..ThermalErosionParams::default()
                    }),
                ),
                (
                    "Soft Flows".into(),
                    LayerKind::EffectFilter(EffectFilterParams::soft_flows()),
                ),
                (
                    "Ridged Flows".into(),
                    LayerKind::EffectFilter(EffectFilterParams::ridged_flows()),
                ),
                (
                    "Sediment Fill".into(),
                    LayerKind::EffectFilter(EffectFilterParams::sediment_fill_soft()),
                ),
            ],
        },
        LayerPreset {
            name: "Desert Mesa".into(),
            description: "High-fidelity caprock mesa with headcuts, particle gullies, fans and layered talus".into(),
            layers: vec![
                (
                    "Base".into(),
                    LayerKind::SculptBase(SculptParams::filled(512, 55.0)),
                ),
                (
                    "Plains".into(),
                    LayerKind::Fbm(FbmParams {
                        base: NoiseParams {
                            seed: 21,
                            frequency: 0.001,
                            amplitude: 35.0,
                            octaves: 3,
                            ..NoiseParams::default()
                        },
                        noise: FractalNoiseType::Perlin,
                    }),
                ),
                (
                    "Mesa".into(),
                    LayerKind::Mesa(MesaParams {
                        radius: 0.26,
                        height: 260.0,
                        edge_steepness: 3.6,
                        soft: 0.2,
                        cap_noise: 5.0,
                        ..MesaParams::default()
                    }),
                ),
                (
                    "Caprock".into(),
                    LayerKind::Materials(MaterialsParams {
                        default_hardness: 0.7,
                        ..MaterialsParams::soft_over_hard(18.0)
                    }),
                ),
                (
                    "Rocky Plateaus".into(),
                    LayerKind::EffectFilter(EffectFilterParams::rocky_plateaus()),
                ),
                (
                    "Rocky Layers".into(),
                    LayerKind::EffectFilter(EffectFilterParams::rocky_layers()),
                ),
                (
                    "Cliff Reinforce".into(),
                    LayerKind::EffectFilter(EffectFilterParams::cliff_reinforce()),
                ),
                (
                    "Stream Power".into(),
                    LayerKind::StreamPowerErosion(StreamPowerParams {
                        iterations: 28,
                        k: 0.0006,
                        dt: 0.22,
                        dendritic_seed: 0.4,
                        hardness_source: terra_core::mask::MaskSource::Hardness,
                        level_count: 3,
                        level_step_strength: 1.2,
                        ..StreamPowerParams::default()
                    }),
                ),
                (
                    "Gullies".into(),
                    LayerKind::HydraulicErosion(HydraulicErosionParams {
                        iterations: 84,
                        rainfall: 0.035,
                        erosion: 0.46,
                        deposition: 0.38,
                        fan_boost: 0.9,
                        bank_slip: 0.58,
                        particle_density: 0.18,
                        particle_lifetime: 48,
                        particle_inertia: 0.28,
                        particle_radius: 2,
                        hardness_source: terra_core::mask::MaskSource::Hardness,
                        level_count: 3,
                        level_step_strength: 1.15,
                        ..HydraulicErosionParams::default()
                    }),
                ),
                (
                    "Weathered Detail".into(),
                    LayerKind::MultiScaleAmplify(fidelity_amplify(0.04, 1.38, 0.34)),
                ),
                (
                    "Talus".into(),
                    LayerKind::ThermalErosion(ThermalErosionParams {
                        talus_angle_deg: 34.0,
                        iterations: 42,
                        strength: 0.58,
                        hardness_source: terra_core::mask::MaskSource::Hardness,
                        ..ThermalErosionParams::default()
                    }),
                ),
            ],
        },
        LayerPreset {
            name: "Butte Country".into(),
            description: "Several buttes with talus skirts and soft desert floor".into(),
            layers: vec![
                (
                    "Base".into(),
                    LayerKind::SculptBase(SculptParams::filled(512, 40.0)),
                ),
                (
                    "Floor".into(),
                    LayerKind::Fbm(FbmParams {
                        base: NoiseParams {
                            seed: 9,
                            frequency: 0.0014,
                            amplitude: 22.0,
                            octaves: 3,
                            ..NoiseParams::default()
                        },
                        noise: FractalNoiseType::Perlin,
                    }),
                ),
                (
                    "Butte A".into(),
                    LayerKind::Mesa(MesaParams {
                        center_u: 0.35,
                        center_v: 0.42,
                        ..MesaParams::butte()
                    }),
                ),
                (
                    "Butte B".into(),
                    LayerKind::Mesa(MesaParams {
                        center_u: 0.62,
                        center_v: 0.55,
                        seed: 19,
                        height: 300.0,
                        ..MesaParams::butte()
                    }),
                ),
                (
                    "Butte C".into(),
                    LayerKind::Mesa(MesaParams {
                        center_u: 0.48,
                        center_v: 0.68,
                        seed: 27,
                        radius: 0.08,
                        height: 280.0,
                        ..MesaParams::butte()
                    }),
                ),
                (
                    "Caprock".into(),
                    LayerKind::Materials(MaterialsParams::soft_over_hard(12.0)),
                ),
                (
                    "Talus".into(),
                    LayerKind::ThermalErosion(ThermalErosionParams {
                        talus_angle_deg: 33.0,
                        iterations: 45,
                        strength: 0.62,
                        hardness_source: terra_core::mask::MaskSource::Hardness,
                        level_step_strength: 1.15,
                        ..ThermalErosionParams::default()
                    }),
                ),
            ],
        },
        LayerPreset {
            name: "Wind Dunes".into(),
            description: "Basin dunes with deep troughs and sparse scrub".into(),
            layers: vec![
                (
                    "Base".into(),
                    LayerKind::SculptBase(SculptParams::filled(512, 14.0)),
                ),
                (
                    "Dunes".into(),
                    LayerKind::Dunes(DuneParams {
                        base: NoiseParams {
                            seed: 55,
                            frequency: 0.0032,
                            amplitude: 52.0,
                            octaves: 4,
                            ..NoiseParams::default()
                        },
                        wave_frequency: 0.012,
                        asymmetry: 0.74,
                        trough_depth: 20.0,
                        basin_floor: 0.07,
                        dune_scale: 0.012,
                        dune_height: 52.0,
                        crest_sharpness: 0.74,
                        sand_supply: 0.9,
                        linearity: 0.75,
                        ..DuneParams::default()
                    }),
                ),
                (
                    "Warp".into(),
                    LayerKind::DomainWarp(DomainWarpParams {
                        warp_strength: 52.0,
                        warp_frequency: 0.002,
                        ..DomainWarpParams::default()
                    }),
                ),
                (
                    "Scrub".into(),
                    LayerKind::Vegetation(VegetationParams {
                        density: 0.08,
                        min_distance: 10.0,
                        max_slope_deg: 18.0,
                        biome_id: None,
                        ..VegetationParams::default()
                    }),
                ),
            ],
        },
        LayerPreset {
            name: "Desert Dunes".into(),
            description: "Wind-aligned dunes over arid flats".into(),
            layers: vec![
                (
                    "Base".into(),
                    LayerKind::SculptBase(SculptParams::filled(512, 12.0)),
                ),
                (
                    "Dunes".into(),
                    LayerKind::Dunes(DuneParams {
                        base: NoiseParams {
                            seed: 55,
                            frequency: 0.0038,
                            amplitude: 42.0,
                            octaves: 4,
                            ..NoiseParams::default()
                        },
                        wave_frequency: 0.014,
                        asymmetry: 0.72,
                        trough_depth: 12.0,
                        basin_floor: 0.04,
                        dune_scale: 0.014,
                        dune_height: 42.0,
                        crest_sharpness: 0.72,
                        sand_supply: 0.85,
                        linearity: 0.82,
                        ..DuneParams::default()
                    }),
                ),
                (
                    "Warp".into(),
                    LayerKind::DomainWarp(DomainWarpParams {
                        warp_strength: 48.0,
                        warp_frequency: 0.0022,
                        ..DomainWarpParams::default()
                    }),
                ),
            ],
        },
        LayerPreset {
            name: "Tropical Island".into(),
            description: "High volcanic island with closed coast, shelf, reef, mature drainage, tropical climate and jungle".into(),
            layers: vec![
                (
                    "Island Landmass + Bathymetry".into(),
                    LayerKind::Island(IslandParams {
                        seed: 63,
                        archetype: IslandArchetype::VolcanicHighIsland,
                        center_u: 0.49,
                        center_v: 0.51,
                        rotation_deg: 24.0,
                        radius: 0.66,
                        aspect: 1.22,
                        sea_level: 0.0,
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
                ),
                (
                    "Upland Fractal Detail".into(),
                    LayerKind::Fbm(FbmParams {
                        base: NoiseParams {
                            seed: 63,
                            frequency: 0.0021,
                            amplitude: 48.0,
                            octaves: 7,
                            ..NoiseParams::default()
                        },
                        noise: FractalNoiseType::Perlin,
                    }),
                ),
                (
                    "Asymmetric Volcanic Summit".into(),
                    LayerKind::Volcano(VolcanoParams {
                        center_u: 0.48,
                        center_v: 0.52,
                        radius: 0.18,
                        height: 175.0,
                        flank_power: 1.4,
                        crater_radius: 0.07,
                        crater_depth: 18.0,
                        roughness: 10.0,
                        seed: 12,
                    }),
                ),
                (
                    "Island Design Constraints".into(),
                    LayerKind::TerrainConstraints(TerrainConstraintParams {
                        preview_strength: 0.55,
                        constraints: vec![
                            TerrainConstraint {
                                kind: TerrainConstraintKind::Ridge,
                                points: vec![
                                    SculptPoint { u: 0.34, v: 0.58, pressure: 0.7 },
                                    SculptPoint { u: 0.47, v: 0.48, pressure: 1.0 },
                                    SculptPoint { u: 0.62, v: 0.43, pressure: 0.65 },
                                ],
                                width_m: 145.0,
                                value: 42.0,
                                strength: 0.9,
                            },
                            TerrainConstraint {
                                kind: TerrainConstraintKind::River,
                                points: vec![
                                    SculptPoint { u: 0.47, v: 0.49, pressure: 1.0 },
                                    SculptPoint { u: 0.42, v: 0.62, pressure: 0.9 },
                                    SculptPoint { u: 0.32, v: 0.72, pressure: 0.5 },
                                ],
                                width_m: 72.0,
                                value: 18.0,
                                strength: 0.78,
                            },
                            TerrainConstraint {
                                kind: TerrainConstraintKind::River,
                                points: vec![
                                    SculptPoint { u: 0.52, v: 0.48, pressure: 1.0 },
                                    SculptPoint { u: 0.63, v: 0.54, pressure: 0.8 },
                                    SculptPoint { u: 0.72, v: 0.61, pressure: 0.45 },
                                ],
                                width_m: 64.0,
                                value: 15.0,
                                strength: 0.72,
                            },
                            TerrainConstraint {
                                kind: TerrainConstraintKind::Protect,
                                points: vec![SculptPoint { u: 0.48, v: 0.51, pressure: 1.0 }],
                                width_m: 190.0,
                                value: 0.0,
                                strength: 0.52,
                            },
                        ],
                    }),
                ),
                (
                    "Seamless Constraint Reconstruction".into(),
                    LayerKind::GradientReconstruct(GradientReconstructParams {
                        iterations: 96,
                        screening: 0.12,
                        constraint_strength: 8.5,
                        gradient_smoothing: 0.22,
                    }),
                ),
                (
                    "Volcanic Geology".into(),
                    LayerKind::Materials(MaterialsParams {
                        default_hardness: 0.42,
                        rules: vec![
                            MaterialRule {
                                name: "Weathered Basalt".into(),
                                id: 1,
                                min_slope_deg: 28.0,
                                max_slope_deg: 90.0,
                                min_height: 0.0,
                                max_height: OPEN_HEIGHT_MAX,
                                mask: terra_core::mask::MaskSource::None,
                                hardness: 0.78,
                                tint: [0.16, 0.18, 0.17],
                                roughness: 0.86,
                                metalness: 0.0,
                                albedo_path: None,
                            },
                            MaterialRule {
                                name: "Deep Tropical Regolith".into(),
                                id: 2,
                                min_slope_deg: 0.0,
                                max_slope_deg: 31.0,
                                min_height: 2.0,
                                max_height: OPEN_HEIGHT_MAX,
                                mask: terra_core::mask::MaskSource::None,
                                hardness: 0.17,
                                tint: [0.16, 0.27, 0.08],
                                roughness: 0.96,
                                metalness: 0.0,
                                albedo_path: None,
                            },
                        ],
                        ..MaterialsParams::default()
                    }),
                ),
                (
                    "Coupled Tropical Landscape Evolution".into(),
                    LayerKind::LandscapeEvolution(LandscapeEvolutionParams {
                        iterations: 28,
                        uplift_rate: 0.028,
                        incision_k: 0.00042,
                        area_exponent: 0.52,
                        slope_exponent: 1.05,
                        hillslope_diffusion: 0.22,
                        talus_angle_deg: 33.0,
                        sediment_transport: 0.48,
                        constraint_preservation: 0.82,
                        base_level: 0.0,
                        use_dinfinity: true,
                        geological_age: 0.55,
                        rainfall: 1.8,
                        erosion: 0.75,
                        uplift: 0.65,
                        river_incision: 0.7,
                        drainage_scale: 0.7,
                        solver: terra_core::layer::EvolutionSolverMode::Fast,
                        ..LandscapeEvolutionParams::default()
                    }),
                ),
                (
                    "Slope Weathering".into(),
                    LayerKind::ThermalErosion(ThermalErosionParams {
                        iterations: 36,
                        talus_angle_deg: 34.0,
                        strength: 0.46,
                        hardness: 0.28,
                        level_count: 3,
                        level_step_strength: 0.92,
                        ..ThermalErosionParams::default()
                    }),
                ),
                (
                    "Tropical Rainfall Erosion".into(),
                    LayerKind::HydraulicErosion(HydraulicErosionParams {
                        iterations: 76,
                        rainfall: 0.038,
                        rainfall_source: terra_core::mask::MaskSource::Named(terra_core::fields::keys::LAND_MASK.into()),
                        protection_source: terra_core::mask::MaskSource::Named(terra_core::fields::keys::MOUNTAIN_MASK.into()),
                        protection_strength: 0.18,
                        evaporation: 0.014,
                        capacity: 0.15,
                        erosion: 0.46,
                        deposition: 0.42,
                        hardness: 0.22,
                        fan_boost: 0.85,
                        floodplain_bias: 0.48,
                        bank_slip: 0.24,
                        sediment_softness: 0.42,
                        particle_density: 0.0,
                        level_count: 3,
                        level_step_strength: 1.08,
                        ..HydraulicErosionParams::default()
                    }),
                ),
                (
                    "Local Drainage Repair".into(),
                    LayerKind::HydrologyRepair(HydrologyRepairParams {
                        iterations: 7,
                        incision: 0.014,
                        repair_radius_m: 420.0,
                        constraint_preservation: 0.92,
                        stream_threshold: 26.0,
                    }),
                ),
                (
                    "Flow-Aligned Island Detail".into(),
                    LayerKind::GeomorphicDetail(GeomorphicDetailParams {
                        amplitude: 3.8,
                        scale_m: 58.0,
                        octaves: 5,
                        flow_alignment: 0.9,
                        slope_gate: 0.12,
                        seed: 631,
                        preserve_drainage: 0.86,
                        ..Default::default()
                    }),
                ),
                ("Micro Geomorphology".into(), LayerKind::MultiScaleAmplify(fidelity_amplify(0.035, 1.28, 0.32))),
                (
                    "Beach Sand".into(),
                    LayerKind::Materials(MaterialsParams {
                        rules: vec![
                            MaterialRule {
                                name: "Tropical Volcanic Soil".into(),
                                id: 2,
                                min_slope_deg: 0.0,
                                max_slope_deg: 30.0,
                                min_height: 4.0,
                                max_height: OPEN_HEIGHT_MAX,
                                mask: terra_core::mask::MaskSource::None,
                                hardness: 0.2,
                                tint: [0.12, 0.30, 0.09],
                                roughness: 0.94,
                                metalness: 0.0,
                                albedo_path: None,
                            },
                            MaterialRule {
                                name: "Coral Sand".into(),
                                id: 3,
                                min_slope_deg: 91.0,
                                max_slope_deg: 90.0,
                                min_height: 1.0,
                                max_height: 14.0,
                                mask: terra_core::mask::MaskSource::Named(terra_core::fields::keys::BEACH.into()),
                                hardness: 0.08,
                                tint: [0.78, 0.70, 0.49],
                                roughness: 0.9,
                                metalness: 0.0,
                                albedo_path: None,
                            },
                            MaterialRule {
                                name: "Basalt Cliffs".into(),
                                id: 1,
                                min_slope_deg: 22.0,
                                max_slope_deg: 90.0,
                                min_height: OPEN_HEIGHT_MIN,
                                max_height: OPEN_HEIGHT_MAX,
                                mask: terra_core::mask::MaskSource::None,
                                hardness: 0.8,
                                tint: [0.18, 0.20, 0.19],
                                roughness: 0.82,
                                metalness: 0.0,
                                albedo_path: None,
                            },
                            MaterialRule {
                                name: "Shallow Reef".into(),
                                id: 5,
                                min_slope_deg: 91.0,
                                max_slope_deg: 90.0,
                                min_height: OPEN_HEIGHT_MIN,
                                max_height: 0.0,
                                mask: terra_core::mask::MaskSource::Named(terra_core::fields::keys::REEF.into()),
                                hardness: 0.72,
                                tint: [0.22, 0.58, 0.48],
                                roughness: 0.7,
                                metalness: 0.0,
                                albedo_path: None,
                            },
                        ],
                        ..MaterialsParams::default()
                    }),
                ),
                (
                    "Climate".into(),
                    LayerKind::Biomes(BiomesParams {
                        use_climate: true,
                        climate: ClimateParams {
                            sea_level_temp: 0.94,
                            latitude: 0.08,
                            orographic_strength: 1.35,
                            sea_level: 0.0,
                            ..ClimateParams::default()
                        },
                        ..BiomesParams::default()
                    }),
                ),
                (
                    "Jungle".into(),
                    LayerKind::Vegetation(VegetationParams {
                        density: 0.78,
                        min_distance: 2.4,
                        max_slope_deg: 42.0,
                        biome_id: None,
                        root_cohesion: 0.28,
                        ..VegetationParams::default()
                    }),
                ),
                (
                    "Forest-Soil Feedback".into(),
                    LayerKind::EcosystemFeedback(EcosystemFeedbackParams {
                        passes: 4,
                        root_cohesion: 0.62,
                        rainfall_interception: 0.31,
                        weathering: 0.11,
                        sediment_capture: 0.42,
                        strength: 0.38,
                    }),
                ),
            ],
        },
        LayerPreset {
            name: "Eroded Highlands".into(),
            description: "High-fidelity mature highlands with staged weathering, dendritic incision, particle rills and PBR strata".into(),
            layers: vec![
                (
                    "Base".into(),
                    LayerKind::SculptBase(SculptParams::filled(512, 120.0)),
                ),
                (
                    "Highlands".into(),
                    LayerKind::Fbm(FbmParams {
                        base: NoiseParams {
                            seed: 73,
                            frequency: 0.0018,
                            amplitude: 280.0,
                            octaves: 7,
                            ..NoiseParams::default()
                        },
                        noise: FractalNoiseType::Perlin,
                    }),
                ),
                (
                    "Highland Materials".into(),
                    LayerKind::Materials(MaterialsParams::default()),
                ),
                (
                    "Thermal Weathering".into(),
                    LayerKind::ThermalErosion(ThermalErosionParams {
                        iterations: 52,
                        strength: 0.58,
                        hardness_source: terra_core::mask::MaskSource::Hardness,
                        level_count: 4,
                        level_step_strength: 1.12,
                        ..ThermalErosionParams::default()
                    }),
                ),
                (
                    "Drainage Incision".into(),
                    LayerKind::StreamPowerErosion(StreamPowerParams {
                        iterations: 34,
                        k: 0.0007,
                        dt: 0.25,
                        dendritic_seed: 0.72,
                        refill_each_iter: true,
                        hardness_source: terra_core::mask::MaskSource::Hardness,
                        level_count: 4,
                        level_step_strength: 1.22,
                        ..StreamPowerParams::default()
                    }),
                ),
                (
                    "Particle Weathering".into(),
                    LayerKind::HydraulicErosion(HydraulicErosionParams {
                        iterations: 92,
                        rainfall: 0.032,
                        capacity: 0.16,
                        erosion: 0.42,
                        deposition: 0.34,
                        fan_boost: 0.65,
                        floodplain_bias: 0.35,
                        bank_slip: 0.32,
                        particle_density: 0.16,
                        particle_lifetime: 46,
                        particle_inertia: 0.25,
                        particle_radius: 2,
                        hardness_source: terra_core::mask::MaskSource::Hardness,
                        level_count: 4,
                        level_step_strength: 1.18,
                        ..HydraulicErosionParams::default()
                    }),
                ),
                (
                    "Micro Geomorphology".into(),
                    LayerKind::MultiScaleAmplify(fidelity_amplify(0.05, 1.52, 0.38)),
                ),
            ],
        },
        LayerPreset {
            name: "River Valley".into(),
            description: "High-fidelity watershed with D-infinity tributaries, particle rills, alluvial floodplain, climate and vegetation".into(),
            layers: vec![
                (
                    "Base".into(),
                    LayerKind::SculptBase(SculptParams::filled(512, 45.0)),
                ),
                (
                    "Uplift".into(),
                    LayerKind::Uplift(UpliftParams {
                        amplitude: 360.0,
                        corridor_width: 0.4,
                        ..UpliftParams::default()
                    }),
                ),
                (
                    "Valley Materials".into(),
                    LayerKind::Materials(MaterialsParams::default()),
                ),
                (
                    "Watershed Incision".into(),
                    LayerKind::StreamPowerErosion(StreamPowerParams {
                        iterations: 38,
                        k: 0.0007,
                        dt: 0.28,
                        dendritic_seed: 0.62,
                        refill_each_iter: true,
                        hardness_source: terra_core::mask::MaskSource::Hardness,
                        level_count: 4,
                        level_step_strength: 1.2,
                        ..StreamPowerParams::default()
                    }),
                ),
                (
                    "Alluvial Fill & Rills".into(),
                    LayerKind::HydraulicErosion({
                        let mut p = HydraulicErosionParams::depositional();
                        p.iterations = 110;
                        p.particle_density = 0.16;
                        p.particle_lifetime = 48;
                        p.particle_inertia = 0.3;
                        p.particle_radius = 3;
                        p.hardness_source = terra_core::mask::MaskSource::Hardness;
                        p.level_count = 4;
                        p.level_step_strength = 1.2;
                        p
                    }),
                ),
                (
                    "Sediment Fill".into(),
                    LayerKind::EffectFilter(EffectFilterParams::sediment_fill_soft()),
                ),
                (
                    "Floodplain Detail".into(),
                    LayerKind::MultiScaleAmplify(fidelity_amplify(0.04, 1.38, 0.55)),
                ),
                (
                    "River Carve".into(),
                    LayerKind::RiverCarve(RiverCarveParams {
                        accumulation_threshold: 32.0,
                        depth: 30.0,
                        width: 7.0,
                        bank_smooth: 3.0,
                        use_dinfinity: true,
                        ..RiverCarveParams::default()
                    }),
                ),
                (
                    "Coastal Edge".into(),
                    LayerKind::Coastal(CoastalParams {
                        sea_level: 30.0,
                        beach_width: 42.0,
                        shelf_depth: 22.0,
                        ..CoastalParams::default()
                    }),
                ),
                (
                    "Temperate Climate".into(),
                    LayerKind::Biomes(BiomesParams {
                        use_climate: true,
                        climate: ClimateParams {
                            sea_level_temp: 0.58,
                            latitude: 0.46,
                            orographic_strength: 0.95,
                            rain_shadow_strength: 0.35,
                            wind_dir_deg: 225.0,
                            sea_level: 30.0,
                            ..ClimateParams::default()
                        },
                        ..BiomesParams::default()
                    }),
                ),
                (
                    "Riparian Vegetation".into(),
                    LayerKind::Vegetation(VegetationParams {
                        density: 0.34,
                        min_distance: 4.0,
                        max_slope_deg: 30.0,
                        root_cohesion: 0.08,
                        ..VegetationParams::default()
                    }),
                ),
            ],
        },
        LayerPreset {
            name: super::recipe::VALLEY_WITH_CREEKS_NAME.into(),
            description: super::recipe::VALLEY_WITH_CREEKS_DESCRIPTION.into(),
            layers: {
                let mut layers = vec![(
                    "Base".into(),
                    LayerKind::SculptBase(SculptParams::filled(512, 45.0)),
                )];
                layers.extend(super::recipe::valley_with_creeks_layers());
                layers
            },
        },
        LayerPreset {
            name: "Layered Plateau".into(),
            description: "Soft clamp plateau with soft-over-hard strata and SPE".into(),
            layers: vec![
                (
                    "Base".into(),
                    LayerKind::SculptBase(SculptParams::filled(512, 60.0)),
                ),
                (
                    "Mass".into(),
                    LayerKind::Fbm(FbmParams {
                        base: NoiseParams {
                            seed: 14,
                            frequency: 0.0012,
                            amplitude: 160.0,
                            octaves: 4,
                            ..NoiseParams::default()
                        },
                        noise: FractalNoiseType::Perlin,
                    }),
                ),
                (
                    "Plateau".into(),
                    LayerKind::Plateau(PlateauParams {
                        low: 80.0,
                        high: 160.0,
                        soft: 12.0,
                    }),
                ),
                (
                    "Strata".into(),
                    LayerKind::Materials(MaterialsParams::soft_over_hard(22.0)),
                ),
                (
                    "Stream Power".into(),
                    LayerKind::StreamPowerErosion(StreamPowerParams {
                        iterations: 24,
                        k: 0.07,
                        dendritic_seed: 0.35,
                        hardness_source: terra_core::mask::MaskSource::Hardness,
                        level_step_strength: 1.1,
                        ..StreamPowerParams::default()
                    }),
                ),
                (
                    "Thermal".into(),
                    LayerKind::ThermalErosion(ThermalErosionParams {
                        hardness_source: terra_core::mask::MaskSource::Hardness,
                        ..ThermalErosionParams::default()
                    }),
                ),
            ],
        },
    ]
}

pub fn project_templates() -> Vec<ProjectTemplate> {
    WorldTemplate::all()
        .iter()
        .copied()
        .map(|t| {
            let layers = if matches!(t, WorldTemplate::Blank) {
                vec![(
                    "Base".into(),
                    LayerKind::SculptBase(SculptParams::filled(512, 20.0)),
                )]
            } else {
                vec![]
            };
            ProjectTemplate {
                id: t.id(),
                name: t.label(),
                description: t.description(),
                default_file_name: match t {
                    WorldTemplate::Blank => "Untitled.json",
                    WorldTemplate::TropicalIsland => "Island.json",
                    WorldTemplate::Alpine => "Alpine.json",
                    WorldTemplate::Desert => "Desert.json",
                    WorldTemplate::RiverValley => "RiverValley.json",
                    WorldTemplate::Badlands => "Badlands.json",
                    WorldTemplate::YoungMountains => "YoungMountains.json",
                    WorldTemplate::OldMountains => "OldMountains.json",
                    WorldTemplate::DuneField => "DuneField.json",
                    WorldTemplate::Coastal => "Coastal.json",
                },
                layers,
            }
        })
        .collect()
}

pub fn project_template_by_id(id: &str) -> Option<ProjectTemplate> {
    project_templates().into_iter().find(|t| t.id == id)
}

pub fn layers_from_project_template(id: &str) -> Option<Vec<Layer>> {
    project_template_by_id(id).map(|t| {
        t.layers
            .into_iter()
            .map(|(n, k)| Layer::new(n, k))
            .collect()
    })
}

pub fn layers_from_preset(name: &str) -> Option<Vec<Layer>> {
    builtin_presets()
        .into_iter()
        .find(|p| p.name == name)
        .map(|p| {
            p.layers
                .into_iter()
                .map(|(n, k)| Layer::new(n, k))
                .collect()
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn valley_with_creeks_preset_and_recipe_share_canonical_content() {
        let preset = builtin_presets()
            .into_iter()
            .find(|preset| preset.name == super::super::recipe::VALLEY_WITH_CREEKS_NAME)
            .expect("Valley with Creeks preset");
        let recipe = super::super::recipe::builtin_recipes()
            .into_iter()
            .find(|recipe| recipe.name == super::super::recipe::VALLEY_WITH_CREEKS_NAME)
            .expect("Valley with Creeks recipe");
        assert_eq!(preset.description, recipe.description);
        assert!(matches!(preset.layers[0].1, LayerKind::SculptBase(_)));
        assert_eq!(
            format!("{:?}", &preset.layers[1..]),
            format!("{:?}", recipe.layers)
        );
        assert!(recipe.layers.iter().any(|(_, kind)| matches!(
            kind,
            LayerKind::StreamPowerErosion(params) if params.iterations == 28
        )));
        assert!(recipe.layers.iter().any(|(_, kind)| matches!(
            kind,
            LayerKind::Path(params) if params.nodes.len() == 3
        )));
        assert!(recipe.layers.iter().any(|(_, kind)| matches!(
            kind,
            LayerKind::RiverNetwork(params) if params.springs.len() == 4
        )));
    }

    #[test]
    fn authored_material_and_biome_preset_bounds_are_finite() {
        for preset in builtin_presets() {
            for (_, kind) in preset.layers {
                match kind {
                    LayerKind::Materials(params) => {
                        assert!(
                            params.rules.iter().all(|rule| {
                                rule.min_height.is_finite() && rule.max_height.is_finite()
                            }),
                            "{} has non-finite material bounds",
                            preset.name
                        );
                    }
                    LayerKind::Biomes(params) => {
                        assert!(
                            params.bands.iter().all(|band| {
                                band.min_height.is_finite() && band.max_height.is_finite()
                            }),
                            "{} has non-finite biome bounds",
                            preset.name
                        );
                    }
                    _ => {}
                }
            }
        }
    }

    #[test]
    fn representative_surface_presets_round_trip_as_documents() {
        for name in ["Alpine Range", "Tropical Island"] {
            let layers = layers_from_preset(name).unwrap_or_else(|| panic!("missing {name}"));
            let doc = terra_core::document::TerrainDocument::from_flat_layers(layers);
            let json = doc
                .to_json()
                .unwrap_or_else(|error| panic!("{name} must serialize: {error}"));
            assert!(!json.contains("\"min_height\":null"), "{name}");
            assert!(!json.contains("\"max_height\":null"), "{name}");
            terra_core::document::TerrainDocument::from_json(&json)
                .unwrap_or_else(|error| panic!("{name} must reload: {error}"));
        }
    }

    #[test]
    fn fidelity_showcases_keep_required_stages() {
        for name in [
            "Alpine Range",
            "Mountain Range (Uplift)",
            "Dendritic Range",
            "Desert Mesa",
            "Eroded Highlands",
            "River Valley",
        ] {
            let preset = builtin_presets()
                .into_iter()
                .find(|preset| preset.name == name)
                .unwrap_or_else(|| panic!("missing showcase preset {name}"));
            let material_index = preset
                .layers
                .iter()
                .position(|(_, kind)| matches!(kind, LayerKind::Materials(_)))
                .expect("showcase requires authored materials");
            let first_simulation = preset
                .layers
                .iter()
                .position(|(_, kind)| {
                    matches!(
                        kind,
                        LayerKind::ThermalErosion(_)
                            | LayerKind::HydraulicErosion(_)
                            | LayerKind::StreamPowerErosion(_)
                            | LayerKind::MultiScaleAmplify(_)
                    )
                })
                .expect("showcase requires geomorphology");
            assert!(
                material_index < first_simulation,
                "{name}: materials must drive erosion hardness"
            );
            assert!(
                preset.layers.iter().any(|(_, kind)| matches!(
                    kind,
                    LayerKind::HydraulicErosion(params) if params.particle_density >= 0.1
                )),
                "{name}: missing high-detail particle erosion"
            );
            assert!(
                preset
                    .layers
                    .iter()
                    .any(|(_, kind)| matches!(kind, LayerKind::MultiScaleAmplify(_))),
                "{name}: missing multiscale detail stage"
            );
        }
    }

    #[test]
    fn project_template_ids_are_unique_and_world_designs_resolve() {
        let templates = project_templates();
        let mut ids = HashSet::new();
        for t in &templates {
            assert!(ids.insert(t.id), "duplicate project template id {}", t.id);
            assert!(!t.name.is_empty());
            assert!(!t.default_file_name.is_empty());
            assert!(
                t.default_file_name.ends_with(".json"),
                "template {} default file should be .json",
                t.id
            );
        }
        for id in [
            "blank",
            "tropical_island",
            "alpine",
            "desert",
            "river_valley",
            "badlands",
            "young_mountains",
            "old_mountains",
            "dune_field",
            "coastal",
        ] {
            assert!(
                project_template_by_id(id).is_some(),
                "missing template {id}"
            );
            assert!(
                layers_from_project_template(id).is_some(),
                "layers_from_project_template missing {id}"
            );
        }
        assert_eq!(templates.len(), 10);
        assert_eq!(world_design_templates().len(), 10);
        assert_eq!(
            world_design_templates().first().map(|t| t.id),
            Some("blank"),
            "Blank should be first in World Design strip"
        );
    }

    #[test]
    fn fidelity_showcases_evaluate_finite_with_relief() {
        let metrics = terra_core::heightfield::HeightfieldMetrics::new(64, 64, 4096.0, 4096.0);
        for name in [
            "Alpine Range",
            "Mountain Range (Uplift)",
            "Dendritic Range",
            "Desert Mesa",
            "Eroded Highlands",
            "River Valley",
        ] {
            let layers = layers_from_preset(name).expect("showcase layers");
            let mut stack = terra_core::layer::LayerStack::new();
            for layer in layers {
                stack.push(layer);
            }
            let mut evaluator = terra_cpu_eval::StackEvaluator::new();
            let mut context = terra_cpu_eval::EvalContext::new(metrics);
            let height = evaluator
                .rebuild_all(&stack, &mut context)
                .unwrap_or_else(|error| panic!("{name} failed to evaluate: {error}"));
            let values = height.to_dense();
            assert!(
                values.iter().all(|value| value.is_finite()),
                "{name}: non-finite height"
            );
            let min = values.iter().copied().fold(f32::INFINITY, f32::min);
            let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert!(
                max - min > 40.0,
                "{name}: insufficient relief {}",
                max - min
            );
        }
    }

    #[test]
    fn tropical_island_has_closed_coast_zones_relief_and_land_only_jungle() {
        let metrics = terra_core::heightfield::HeightfieldMetrics::new(64, 64, 4096.0, 4096.0);
        let layers = layers_from_preset("Tropical Island").expect("tropical island layers");
        assert!(layers
            .iter()
            .any(|layer| matches!(layer.kind, LayerKind::TerrainConstraints(_))));
        assert!(layers
            .iter()
            .any(|layer| matches!(layer.kind, LayerKind::GradientReconstruct(_))));
        assert!(layers
            .iter()
            .any(|layer| matches!(layer.kind, LayerKind::LandscapeEvolution(_))));
        assert!(layers
            .iter()
            .any(|layer| matches!(layer.kind, LayerKind::HydrologyRepair(_))));
        assert!(layers
            .iter()
            .any(|layer| matches!(layer.kind, LayerKind::GeomorphicDetail(_))));
        assert!(layers
            .iter()
            .any(|layer| matches!(layer.kind, LayerKind::EcosystemFeedback(_))));
        assert!(matches!(layers[0].kind, LayerKind::Island(_)));
        assert!(layers
            .iter()
            .any(|layer| matches!(layer.kind, LayerKind::HydraulicErosion(_))));
        assert!(layers
            .iter()
            .any(|layer| matches!(layer.kind, LayerKind::MultiScaleAmplify(_))));

        let mut stack = terra_core::layer::LayerStack::new();
        for layer in layers {
            stack.push(layer);
        }
        let mut evaluator = terra_cpu_eval::StackEvaluator::new();
        let mut context = terra_cpu_eval::EvalContext::new(metrics);
        context.quality = terra_core::quality::PreviewQuality::Full;
        let height = evaluator
            .rebuild_all(&stack, &mut context)
            .expect("island evaluates");
        let values = height.to_dense();
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(min < -100.0, "missing deep bathymetry: {min}");
        assert!(max > 300.0, "missing high-island relief: {max}");

        for i in 0..metrics.width {
            assert!(height.get(i, 0) < 0.0 && height.get(i, metrics.height - 1) < 0.0);
        }
        for j in 0..metrics.height {
            assert!(height.get(0, j) < 0.0 && height.get(metrics.width - 1, j) < 0.0);
        }
        let land_fraction =
            values.iter().filter(|&&v| v > 0.0).count() as f32 / values.len() as f32;
        assert!(
            (0.08..0.62).contains(&land_fraction),
            "implausible land fraction {land_fraction}"
        );

        for key in [
            terra_core::fields::keys::LAND_MASK,
            terra_core::fields::keys::BEACH,
            terra_core::fields::keys::REEF,
            terra_core::fields::keys::BATHYMETRY,
            terra_core::fields::keys::EROSION,
            terra_core::fields::keys::MATERIALS,
            terra_core::fields::keys::VEGETATION,
            terra_core::fields::keys::CONSTRAINT_ERROR,
            terra_core::fields::keys::FLOW_ACCUMULATION,
            terra_core::fields::keys::REPAIR_REGION,
            terra_core::fields::keys::DETAIL_MASK,
            terra_core::fields::keys::ROOT_COHESION,
            terra_core::fields::keys::SEDIMENT_THICKNESS,
        ] {
            assert!(
                context.aux_maps.get(key).is_some(),
                "missing semantic field {key}"
            );
        }
        let beach = context
            .aux_maps
            .get(terra_core::fields::keys::BEACH)
            .unwrap();
        let reef = context
            .aux_maps
            .get(terra_core::fields::keys::REEF)
            .unwrap();
        assert!(
            beach.data().iter().any(|&v| v > 0.5),
            "no usable beach zone"
        );
        assert!(reef.data().iter().any(|&v| v > 0.5), "no usable reef zone");

        let materials = context
            .aux_maps
            .get(terra_core::fields::keys::MATERIALS)
            .unwrap();
        let coral_sand_id = 3.0 / 16.0;
        let shallow_reef_id = 5.0 / 16.0;
        let has_coral_sand = beach
            .data()
            .iter()
            .zip(materials.data())
            .any(|(&zone, &material)| zone > 0.5 && (material - coral_sand_id).abs() < 1e-5);
        let has_shallow_reef = reef
            .data()
            .iter()
            .zip(materials.data())
            .any(|(&zone, &material)| zone > 0.5 && (material - shallow_reef_id).abs() < 1e-5);
        assert!(
            has_coral_sand,
            "beach field did not drive coral-sand material"
        );
        assert!(
            has_shallow_reef,
            "reef field did not drive shallow-reef material"
        );

        let land = context
            .aux_maps
            .get(terra_core::fields::keys::LAND_MASK)
            .unwrap();
        let vegetation = context
            .aux_maps
            .get(terra_core::fields::keys::VEGETATION)
            .unwrap();
        let ocean_vegetation = vegetation
            .data()
            .iter()
            .zip(land.data())
            .filter(|(_, &l)| l < 0.1)
            .map(|(&v, _)| v)
            .fold(0.0f32, f32::max);
        assert!(
            ocean_vegetation < 1e-5,
            "vegetation leaked offshore: {ocean_vegetation}"
        );
    }
}
