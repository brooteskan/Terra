//! EffectFilter → GPU preview mode mapping (CPU remains export oracle).

use terra_core::layer::EffectFilterKind;
use terra_core::layer::EffectFilterParams;

/// Spatial execution contract for one admitted EffectFilter configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectFilterGpuScope {
    /// The output depends only on the same input texel.
    LocalPointwise,
    /// A bounded neighbourhood reaches `halo_per_pass` texels per executed pass.
    LocalExpanding { halo_per_pass: u32 },
    /// A reduction, arbitrary resample, or globally coupled solver requires the field.
    FullField,
}

/// How the executor derives the number of shader passes for a filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectFilterGpuPasses {
    /// Preserve the existing Smooth/Inflate/Denoise interactive preview contract.
    LegacyQualityScaled,
    /// Execute the kernel exactly once. Most CPU EffectFilter arms are single passes.
    Once,
}

/// Planner/executor shared description of an admitted EffectFilter configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectFilterGpuSpec {
    pub mode: u32,
    pub scope: EffectFilterGpuScope,
    pub passes: EffectFilterGpuPasses,
    /// True when the kernel needs the exact entering-field min/max reduction.
    pub needs_height_range: bool,
}

/// Return the executable GPU contract for an admitted configuration.
///
/// This is deliberately the only admission list. The planner and executor consume
/// the same descriptor so mode selection, spatial reach, and pass count cannot drift.
pub fn effect_filter_gpu_spec(p: &EffectFilterParams) -> Option<EffectFilterGpuSpec> {
    use EffectFilterGpuPasses::{LegacyQualityScaled, Once};
    use EffectFilterGpuScope::{FullField, LocalExpanding, LocalPointwise};
    use EffectFilterKind::*;

    let radius = p.radius.clamp(1, crate::graph::EFFECT_FILTER_MAX_RADIUS);
    let finite = p.strength.is_finite()
        && p.amount.is_finite()
        && p.frequency.is_finite()
        && p.sea_level.is_finite()
        && p.beach_width.is_finite()
        && p.crater_radius.is_finite()
        && p.slope_min.is_finite()
        && p.slope_max.is_finite()
        && p.flow_threshold.is_finite()
        && p.rock_hardness.is_finite()
        && p.wall_steepness.is_finite()
        && p.valley_floor.is_finite()
        && p.talus_mix.is_finite()
        && p.terrace_height.is_finite()
        && p.terrace_offset.is_finite()
        && p.top_smoothness.is_finite()
        && p.riser_sharpness.is_finite()
        && p.rotation_deg.is_finite()
        && p.anisotropy.is_finite()
        && p.warp_strength.is_finite()
        && p.warp_frequency.is_finite()
        && p.lacunarity.is_finite()
        && p.persistence.is_finite()
        && p.scale_m.is_finite();
    let seed32 = p.seed <= u64::from(u32::MAX);
    let radius_supported = p.radius <= crate::graph::EFFECT_FILTER_MAX_RADIUS;
    let warp_seed32 = seed32 && (p.warp_strength.abs() <= 1.0e-5 || p.seed < u64::from(u32::MAX));
    let seed_stream32 = |seed: u64, octaves: u32, stride: u64| {
        seed.checked_add(u64::from(octaves.saturating_sub(1)).saturating_mul(stride))
            .is_some_and(|last| last <= u64::from(u32::MAX))
    };
    let spec = match p.kind {
        Smooth | Inflate | Denoise if finite && radius_supported => EffectFilterGpuSpec {
            mode: resolve_effect_mode(p.kind),
            scope: LocalExpanding {
                halo_per_pass: radius,
            },
            passes: LegacyQualityScaled,
            needs_height_range: false,
        },
        AddSet if finite => EffectFilterGpuSpec {
            mode: 24,
            scope: LocalPointwise,
            passes: Once,
            needs_height_range: false,
        },
        Deflate if finite && radius_supported => EffectFilterGpuSpec {
            mode: 6,
            scope: LocalExpanding {
                halo_per_pass: radius,
            },
            passes: Once,
            needs_height_range: false,
        },
        Curve if finite => EffectFilterGpuSpec {
            mode: 25,
            scope: FullField,
            passes: Once,
            needs_height_range: true,
        },
        Cutoff if finite => EffectFilterGpuSpec {
            mode: 26,
            scope: FullField,
            passes: Once,
            needs_height_range: true,
        },
        TerraceSimple if finite => EffectFilterGpuSpec {
            mode: 27,
            scope: FullField,
            passes: Once,
            needs_height_range: true,
        },
        Shore if finite => EffectFilterGpuSpec {
            mode: 31,
            scope: LocalPointwise,
            passes: Once,
            needs_height_range: false,
        },
        Blocks if finite => EffectFilterGpuSpec {
            mode: 32,
            scope: LocalPointwise,
            passes: Once,
            needs_height_range: false,
        },
        ZeroEdge if finite => EffectFilterGpuSpec {
            mode: 33,
            scope: FullField,
            passes: Once,
            needs_height_range: true,
        },
        Squeeze if finite => EffectFilterGpuSpec {
            mode: 34,
            scope: FullField,
            passes: Once,
            needs_height_range: true,
        },
        DirectionalBlur if finite && radius_supported => EffectFilterGpuSpec {
            mode: 35,
            scope: LocalExpanding {
                halo_per_pass: radius,
            },
            passes: Once,
            needs_height_range: false,
        },
        AngleBlur if finite && radius_supported => EffectFilterGpuSpec {
            mode: 36,
            scope: LocalExpanding {
                halo_per_pass: radius.max(1),
            },
            passes: Once,
            needs_height_range: false,
        },
        Swirl if finite && seed32 => EffectFilterGpuSpec {
            mode: 37,
            scope: FullField,
            passes: Once,
            needs_height_range: false,
        },
        Crater if finite && seed32 && p.iterations <= 1 => EffectFilterGpuSpec {
            mode: 38,
            scope: LocalPointwise,
            passes: Once,
            needs_height_range: false,
        },
        Distortion if finite && seed32 => EffectFilterGpuSpec {
            mode: 39,
            scope: FullField,
            passes: Once,
            needs_height_range: false,
        },
        Balloon if finite && radius_supported => EffectFilterGpuSpec {
            mode: 40,
            scope: LocalExpanding {
                halo_per_pass: radius.max(1),
            },
            passes: Once,
            needs_height_range: false,
        },
        NoisePerlin | NoiseValue
            if finite
                && warp_seed32
                && p.octaves.clamp(1, 12) == p.octaves
                && seed_stream32(p.seed, p.octaves, 1013) =>
        {
            EffectFilterGpuSpec {
                mode: if p.kind == NoisePerlin { 41 } else { 42 },
                scope: LocalPointwise,
                passes: Once,
                needs_height_range: false,
            }
        }
        NoiseWhite if finite && seed32 => EffectFilterGpuSpec {
            mode: 43,
            scope: LocalPointwise,
            passes: Once,
            needs_height_range: false,
        },
        NoiseWave if finite && warp_seed32 => EffectFilterGpuSpec {
            mode: 44,
            scope: LocalPointwise,
            passes: Once,
            needs_height_range: false,
        },
        ScatterDetail if finite && warp_seed32 => EffectFilterGpuSpec {
            mode: 45,
            scope: LocalPointwise,
            passes: Once,
            needs_height_range: false,
        },
        SpikeRemoval if finite && p.radius == 1 => EffectFilterGpuSpec {
            mode: 46,
            scope: LocalExpanding { halo_per_pass: 1 },
            passes: Once,
            needs_height_range: false,
        },
        NoiseBillow
            if finite
                && p.octaves.clamp(1, 12) == p.octaves
                && warp_seed32
                && seed_stream32(p.seed, p.octaves, 1301) =>
        {
            EffectFilterGpuSpec {
                mode: 47,
                scope: LocalPointwise,
                passes: Once,
                needs_height_range: false,
            }
        }
        NoiseRidged
            if finite
                && p.octaves.clamp(1, 12) == p.octaves
                && warp_seed32
                && seed_stream32(p.seed, p.octaves, 9173) =>
        {
            EffectFilterGpuSpec {
                mode: 48,
                scope: LocalPointwise,
                passes: Once,
                needs_height_range: false,
            }
        }
        Ridged if finite && seed_stream32(p.seed, 5, 9173) => EffectFilterGpuSpec {
            mode: 49,
            scope: LocalPointwise,
            passes: Once,
            needs_height_range: false,
        },
        Rugged if finite && seed_stream32(p.seed ^ 0xA06D, 6, 9173) => EffectFilterGpuSpec {
            mode: 50,
            scope: LocalPointwise,
            passes: Once,
            needs_height_range: false,
        },
        BorderBlend if finite && p.sea_level.abs() > 1.0e-3 => EffectFilterGpuSpec {
            mode: 51,
            scope: LocalPointwise,
            passes: Once,
            needs_height_range: false,
        },
        FlattenFilter if finite && p.sea_level.abs() > 1.0e-3 => EffectFilterGpuSpec {
            mode: 52,
            scope: LocalPointwise,
            passes: Once,
            needs_height_range: false,
        },
        Hexagons if finite => EffectFilterGpuSpec {
            mode: 53,
            scope: FullField,
            passes: Once,
            needs_height_range: false,
        },
        TerraceSteep if finite && seed32 => EffectFilterGpuSpec {
            mode: 54,
            scope: FullField,
            passes: Once,
            needs_height_range: true,
        },
        _ => return None,
    };
    Some(spec)
}

/// Map artist filter kinds onto `effect_filter.wgsl` mode indices.
pub fn effect_filter_mode(kind: EffectFilterKind) -> u32 {
    use EffectFilterKind::*;
    match kind {
        Smooth => 0,
        Distortion | Swirl => 1,
        SpikeRemoval => 2,
        Shore => 3,
        Denoise | Kuwahara => 4,
        Inflate => 5,
        Deflate => 6,
        Balloon => 7,
        TerraceSimple | TerraceIrregular | TerraceSteep => 8,
        Curve => 25,
        Cutoff => 26,
        AddSet => 24,
        RockySharp | RockyWide | RockyLayers | CliffReinforce | RockyPlateaus | RockyCliffs
        | RockyHard | Rocky | Cliffs | Chipped | Rugged | Ridged | SmoothRidges | Canyon
        | AngleBreak | WindCarve | Squeeze => 11,
        NoiseBillow | NoiseGabor | NoisePerlin | NoiseValue | NoiseSimplex | NoiseWhite
        | NoisePhasor | NoiseVoronoi | DesignVoronoi | Hexagons | ScatterDetail | NoiseRidged
        | NoiseWave => 12,
        AngleBlur | DirectionalBlur => 13,
        ZeroEdge | BorderBlend | WashedOff => 14,
        FlattenFilter => 15,
        Strata => 16,
        Crater => 17,
        TalusFill | SedimentFillSoft | MudSettle | HydraulicSediment | SoftFlows | ThinFlows
        | RidgedFlows | WideFlows | SedimentFlows => 18,
        // Sharpen-style leftover / blocks
        Blocks => 23,
    }
}

/// Some kinds share the "flows" downhill smear kernel.
pub fn effect_filter_mode_override(kind: EffectFilterKind) -> Option<u32> {
    use EffectFilterKind::*;
    match kind {
        SoftFlows | ThinFlows | RidgedFlows | WideFlows | SedimentFlows | HydraulicSediment => {
            Some(21)
        }
        Swirl => Some(22),
        Chipped | RockySharp => Some(19),
        Kuwahara => Some(20),
        _ => None,
    }
}

pub fn resolve_effect_mode(kind: EffectFilterKind) -> u32 {
    effect_filter_mode_override(kind).unwrap_or_else(|| effect_filter_mode(kind))
}
