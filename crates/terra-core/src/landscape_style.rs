//! Landscape style presets — Phase 11 Rule 8.
//!
//! Presets are **parameter sets** over real algorithms (uplift, SPE / landscape
//! evolution, thermal / debris, materials strata). They never bake unique
//! hardcoded generators per look.

use crate::analyze::{
    DebrisFlowParams, HydraulicErosionParams, MultiScaleAmplifyParams, ThermalErosionParams,
};
use crate::authoring::{GeomorphicDetailParams, LandscapeEvolutionParams};
use crate::generators::UpliftParams;
use crate::hydro::StreamPowerParams;
use crate::landscape_evolution::EvolutionSolverMode;
use crate::material_schema::MaterialsParams;

/// Named landscape look built from shared process parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LandscapeStyle {
    Alpine,
    Himalayan,
    ScottishHighlands,
    Mesa,
    Badlands,
    Canyon,
    Volcanic,
    RollingHills,
    YoungMountains,
    OldMountains,
    DuneField,
    Coastal,
    TropicalIsland,
    RiverValley,
}

impl LandscapeStyle {
    pub fn all() -> &'static [LandscapeStyle] {
        &[
            Self::Alpine,
            Self::Himalayan,
            Self::ScottishHighlands,
            Self::Mesa,
            Self::Badlands,
            Self::Canyon,
            Self::Volcanic,
            Self::RollingHills,
            Self::YoungMountains,
            Self::OldMountains,
            Self::DuneField,
            Self::Coastal,
            Self::TropicalIsland,
            Self::RiverValley,
        ]
    }

    pub fn id(self) -> &'static str {
        match self {
            Self::Alpine => "alpine",
            Self::Himalayan => "himalayan",
            Self::ScottishHighlands => "scottish_highlands",
            Self::Mesa => "mesa",
            Self::Badlands => "badlands",
            Self::Canyon => "canyon",
            Self::Volcanic => "volcanic",
            Self::RollingHills => "rolling_hills",
            Self::YoungMountains => "young_mountains",
            Self::OldMountains => "old_mountains",
            Self::DuneField => "dune_field",
            Self::Coastal => "coastal",
            Self::TropicalIsland => "tropical_island",
            Self::RiverValley => "river_valley",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Alpine => "Alpine",
            Self::Himalayan => "Himalayan",
            Self::ScottishHighlands => "Scottish Highlands",
            Self::Mesa => "Mesa",
            Self::Badlands => "Badlands",
            Self::Canyon => "Canyon",
            Self::Volcanic => "Volcanic",
            Self::RollingHills => "Rolling Hills",
            Self::YoungMountains => "Young Mountains",
            Self::OldMountains => "Old Mountains",
            Self::DuneField => "Dune Field",
            Self::Coastal => "Coastal",
            Self::TropicalIsland => "Tropical Island",
            Self::RiverValley => "River Valley",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Alpine => "Tectonic uplift, mature drainage, debris scars and talus.",
            Self::Himalayan => "Extreme uplift and relief with aggressive stream-power incision.",
            Self::ScottishHighlands => "Long-evolved rounded relief with dense drainage.",
            Self::Mesa => "Layered geology, plateau, canyon incision, differential erosion.",
            Self::Badlands => "Soft strata, dense drainage, strong differential erosion.",
            Self::Canyon => "Deep fluvial incision through layered beds.",
            Self::Volcanic => "Radial construction with flank drainage and ash-soft cover.",
            Self::RollingHills => "Low relief, broad valleys, mild weathering.",
            Self::YoungMountains => "Strong uplift, steep relief, immature valleys.",
            Self::OldMountains => "Long landscape evolution, broad valleys, reduced relief.",
            Self::DuneField => "Wind-driven sand transport over a hard substrate.",
            Self::Coastal => "Shore profile with inland drainage toward the sea.",
            Self::TropicalIsland => "Volcanic island landmass with wet drainage and shore profile.",
            Self::RiverValley => "Valley uplift, fluvial incision, floodplain fill and carve.",
        }
    }

    pub fn params(self) -> LandscapeStyleParams {
        match self {
            Self::Alpine => LandscapeStyleParams::alpine(),
            Self::Himalayan => LandscapeStyleParams::himalayan(),
            Self::ScottishHighlands => LandscapeStyleParams::scottish_highlands(),
            Self::Mesa => LandscapeStyleParams::mesa(),
            Self::Badlands => LandscapeStyleParams::badlands(),
            Self::Canyon => LandscapeStyleParams::canyon(),
            Self::Volcanic => LandscapeStyleParams::volcanic(),
            Self::RollingHills => LandscapeStyleParams::rolling_hills(),
            Self::YoungMountains => LandscapeStyleParams::young_mountains(),
            Self::OldMountains => LandscapeStyleParams::old_mountains(),
            Self::DuneField => LandscapeStyleParams::dune_field(),
            Self::Coastal => LandscapeStyleParams::coastal(),
            Self::TropicalIsland => LandscapeStyleParams::tropical_island(),
            Self::RiverValley => LandscapeStyleParams::river_valley(),
        }
    }
}

/// Parameter bundle applied to real stack operators.
#[derive(Debug, Clone)]
pub struct LandscapeStyleParams {
    pub geological_age: f32,
    pub rainfall: f32,
    pub drainage_density: f32,
    pub ridge_sharpness: f32,
    pub uplift: UpliftParams,
    pub evolution: LandscapeEvolutionParams,
    pub thermal: ThermalErosionParams,
    pub debris: DebrisFlowParams,
    pub hydraulic: HydraulicErosionParams,
    pub stream_power: StreamPowerParams,
    pub materials: MaterialsParams,
    pub geomorphic_detail: GeomorphicDetailParams,
    pub amplify: MultiScaleAmplifyParams,
}

impl LandscapeStyleParams {
    pub fn alpine() -> Self {
        let mut evolution = LandscapeEvolutionParams::default();
        evolution.geological_age = 0.35;
        evolution.rainfall = 1.2;
        evolution.drainage_scale = 0.65;
        evolution.iterations = 22;
        evolution.solver = EvolutionSolverMode::Fast;

        let mut thermal = ThermalErosionParams::default();
        thermal.iterations = 18;
        thermal.talus_angle_deg = 36.0;

        let mut debris = DebrisFlowParams::default();
        debris.iterations = 12;

        Self {
            geological_age: 0.35,
            rainfall: 1.2,
            drainage_density: 0.65,
            ridge_sharpness: 0.85,
            uplift: UpliftParams {
                amplitude: 420.0,
                corridor_width: 0.35,
                ridge_power: 2.0,
                ..UpliftParams::default()
            },
            evolution,
            thermal,
            debris,
            hydraulic: HydraulicErosionParams::default(),
            stream_power: StreamPowerParams {
                iterations: 28,
                k: 0.09,
                ..StreamPowerParams::default()
            },
            materials: MaterialsParams::alpine_peak(),
            geomorphic_detail: GeomorphicDetailParams::default(),
            amplify: MultiScaleAmplifyParams::default(),
        }
    }

    pub fn himalayan() -> Self {
        let mut p = Self::alpine();
        p.geological_age = 0.22;
        p.rainfall = 1.55;
        p.ridge_sharpness = 0.95;
        p.uplift.amplitude = 780.0;
        p.uplift.corridor_width = 0.22;
        p.uplift.ridge_power = 2.6;
        p.evolution.geological_age = 0.22;
        p.evolution.rainfall = 1.55;
        p.evolution.iterations = 30;
        p.stream_power.iterations = 40;
        p.stream_power.k = 0.12;
        p.thermal.talus_angle_deg = 40.0;
        p
    }

    pub fn scottish_highlands() -> Self {
        let mut p = Self::old_mountains();
        p.rainfall = 1.8;
        p.drainage_density = 0.8;
        p.evolution.rainfall = 1.8;
        p.evolution.drainage_scale = 0.8;
        p.ridge_sharpness = 0.4;
        p
    }

    pub fn mesa() -> Self {
        let mut evolution = LandscapeEvolutionParams::default();
        evolution.geological_age = 0.7;
        evolution.rainfall = 0.25;
        evolution.drainage_scale = 0.3;
        evolution.iterations = 18;

        Self {
            geological_age: 0.7,
            rainfall: 0.25,
            drainage_density: 0.3,
            ridge_sharpness: 0.55,
            uplift: UpliftParams {
                amplitude: 180.0,
                corridor_width: 0.55,
                ..UpliftParams::default()
            },
            evolution,
            thermal: ThermalErosionParams {
                iterations: 22,
                talus_angle_deg: 32.0,
                ..ThermalErosionParams::default()
            },
            debris: DebrisFlowParams::default(),
            hydraulic: HydraulicErosionParams::default(),
            stream_power: StreamPowerParams {
                iterations: 20,
                k: 0.07,
                ..StreamPowerParams::default()
            },
            materials: MaterialsParams::soft_over_hard(28.0),
            geomorphic_detail: GeomorphicDetailParams::default(),
            amplify: MultiScaleAmplifyParams::default(),
        }
    }

    pub fn badlands() -> Self {
        let mut p = Self::mesa();
        p.geological_age = 0.55;
        p.rainfall = 0.45;
        p.drainage_density = 0.9;
        p.evolution.geological_age = 0.55;
        p.evolution.rainfall = 0.45;
        p.evolution.drainage_scale = 0.9;
        p.evolution.iterations = 26;
        p.stream_power.iterations = 32;
        p.stream_power.k = 0.11;
        p.materials = MaterialsParams::soft_over_hard(14.0);
        p.thermal.iterations = 28;
        p
    }

    pub fn canyon() -> Self {
        let mut p = Self::mesa();
        p.stream_power.iterations = 48;
        p.stream_power.k = 0.14;
        p.drainage_density = 0.55;
        p.evolution.drainage_scale = 0.55;
        p
    }

    pub fn volcanic() -> Self {
        let mut p = Self::young_mountains();
        p.rainfall = 1.1;
        p.materials = MaterialsParams::soft_over_hard(18.0);
        p.thermal.talus_angle_deg = 34.0;
        p
    }

    pub fn rolling_hills() -> Self {
        let mut p = Self::old_mountains();
        p.uplift.amplitude = 120.0;
        p.ridge_sharpness = 0.25;
        p.evolution.iterations = 14;
        p.stream_power.k = 0.04;
        p.thermal.iterations = 10;
        p
    }

    pub fn young_mountains() -> Self {
        let mut evolution = LandscapeEvolutionParams::default();
        evolution.geological_age = 0.18;
        evolution.rainfall = 1.35;
        evolution.drainage_scale = 0.45;
        evolution.iterations = 16;
        evolution.solver = EvolutionSolverMode::Fast;

        Self {
            geological_age: 0.18,
            rainfall: 1.35,
            drainage_density: 0.45,
            ridge_sharpness: 0.9,
            uplift: UpliftParams {
                amplitude: 640.0,
                corridor_width: 0.28,
                ridge_power: 2.4,
                ..UpliftParams::default()
            },
            evolution,
            thermal: ThermalErosionParams {
                iterations: 14,
                talus_angle_deg: 38.0,
                ..ThermalErosionParams::default()
            },
            debris: DebrisFlowParams {
                iterations: 16,
                ..DebrisFlowParams::default()
            },
            hydraulic: HydraulicErosionParams::default(),
            stream_power: StreamPowerParams {
                iterations: 18,
                k: 0.1,
                dendritic_seed: 0.35,
                ..StreamPowerParams::default()
            },
            materials: MaterialsParams::alpine_peak(),
            geomorphic_detail: GeomorphicDetailParams::default(),
            amplify: MultiScaleAmplifyParams::default(),
        }
    }

    pub fn old_mountains() -> Self {
        let mut evolution = LandscapeEvolutionParams::default();
        evolution.geological_age = 0.88;
        evolution.rainfall = 1.05;
        evolution.drainage_scale = 0.7;
        evolution.iterations = 36;
        evolution.solver = EvolutionSolverMode::Fast;

        Self {
            geological_age: 0.88,
            rainfall: 1.05,
            drainage_density: 0.7,
            ridge_sharpness: 0.35,
            uplift: UpliftParams {
                amplitude: 240.0,
                corridor_width: 0.55,
                ridge_power: 1.1,
                ..UpliftParams::default()
            },
            evolution,
            thermal: ThermalErosionParams {
                iterations: 24,
                talus_angle_deg: 28.0,
                ..ThermalErosionParams::default()
            },
            debris: DebrisFlowParams {
                iterations: 6,
                ..DebrisFlowParams::default()
            },
            hydraulic: HydraulicErosionParams::depositional(),
            stream_power: StreamPowerParams {
                iterations: 22,
                k: 0.05,
                dendritic_seed: 0.7,
                ..StreamPowerParams::default()
            },
            materials: MaterialsParams::soft_over_hard(22.0),
            geomorphic_detail: GeomorphicDetailParams::default(),
            amplify: MultiScaleAmplifyParams::default(),
        }
    }

    pub fn dune_field() -> Self {
        let mut p = Self::mesa();
        p.rainfall = 0.08;
        p.drainage_density = 0.1;
        p.evolution.rainfall = 0.08;
        p.evolution.drainage_scale = 0.1;
        p.evolution.iterations = 8;
        p.stream_power.iterations = 4;
        p.thermal.iterations = 8;
        p
    }

    pub fn coastal() -> Self {
        let mut evolution = LandscapeEvolutionParams::default();
        evolution.geological_age = 0.5;
        evolution.rainfall = 1.4;
        evolution.drainage_scale = 0.75;
        evolution.iterations = 20;

        Self {
            geological_age: 0.5,
            rainfall: 1.4,
            drainage_density: 0.75,
            ridge_sharpness: 0.45,
            uplift: UpliftParams {
                amplitude: 220.0,
                corridor_width: 0.48,
                ..UpliftParams::default()
            },
            evolution,
            thermal: ThermalErosionParams::default(),
            debris: DebrisFlowParams::default(),
            hydraulic: HydraulicErosionParams::depositional(),
            stream_power: StreamPowerParams {
                iterations: 24,
                k: 0.07,
                ..StreamPowerParams::default()
            },
            materials: MaterialsParams::soft_over_hard(16.0),
            geomorphic_detail: GeomorphicDetailParams::default(),
            amplify: MultiScaleAmplifyParams::default(),
        }
    }

    pub fn tropical_island() -> Self {
        let mut p = Self::coastal();
        p.geological_age = 0.55;
        p.rainfall = 2.08;
        p.drainage_density = 0.7;
        p.ridge_sharpness = 0.65;
        p.evolution.geological_age = 0.55;
        p.evolution.rainfall = 2.08;
        p.evolution.drainage_scale = 0.7;
        p.evolution.iterations = 22;
        p.uplift.amplitude = 280.0;
        p.materials = MaterialsParams::soft_over_hard(20.0);
        p.thermal.talus_angle_deg = 34.0;
        p
    }

    pub fn river_valley() -> Self {
        let mut p = Self::coastal();
        p.geological_age = 0.5;
        p.rainfall = 1.6;
        p.drainage_density = 0.85;
        p.ridge_sharpness = 0.45;
        p.evolution.geological_age = 0.5;
        p.evolution.rainfall = 1.6;
        p.evolution.drainage_scale = 0.85;
        p.evolution.iterations = 24;
        p.uplift.amplitude = 360.0;
        p.uplift.corridor_width = 0.42;
        p.stream_power.iterations = 28;
        p.stream_power.k = 0.08;
        p.stream_power.dendritic_seed = 0.55;
        p.hydraulic = HydraulicErosionParams::depositional();
        p.materials = MaterialsParams::soft_over_hard(18.0);
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_style_has_stable_id() {
        let mut seen = std::collections::HashSet::new();
        for s in LandscapeStyle::all() {
            assert!(seen.insert(s.id()));
            assert!(!s.label().is_empty());
            let p = s.params();
            assert!((0.0..=1.0).contains(&p.geological_age));
        }
    }
}
