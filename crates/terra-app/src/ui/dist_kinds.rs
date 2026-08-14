//! Shared distribution generator / modifier catalogs (inspector + Quick Add).

use terra_core::mask::DistNodeKind;
use terra_gui::Icon;

/// Distribution base kinds (WC generators). Excludes MaskAsset.
pub fn dist_base_kinds() -> Vec<(&'static str, DistNodeKind)> {
    vec![
        ("Fill", DistNodeKind::Fill { value: 1.0 }),
        (
            "Noise",
            DistNodeKind::Noise {
                seed: 1,
                frequency: 0.05,
            },
        ),
        (
            "Perlin",
            DistNodeKind::NoisePerlin {
                seed: 1,
                frequency: 0.05,
                octaves: 4,
            },
        ),
        (
            "Ridged",
            DistNodeKind::NoiseRidged {
                seed: 1,
                frequency: 0.05,
                octaves: 4,
            },
        ),
        (
            "Worley",
            DistNodeKind::NoiseWorley {
                seed: 1,
                frequency: 0.05,
            },
        ),
        (
            "Billow",
            DistNodeKind::NoiseBillow {
                seed: 1,
                frequency: 0.05,
                octaves: 4,
            },
        ),
        (
            "Slope",
            DistNodeKind::Slope {
                min_deg: 15.0,
                max_deg: 45.0,
            },
        ),
        (
            "Steepness",
            DistNodeKind::Steepness {
                min_deg: 20.0,
                max_deg: 70.0,
            },
        ),
        (
            "Height",
            DistNodeKind::Height {
                min: 0.0,
                max: 500.0,
            },
        ),
        (
            "Curvature",
            DistNodeKind::Curvature {
                min: -1.0,
                max: 1.0,
            },
        ),
        ("Flow", DistNodeKind::Flow { min: 0.0, max: 1.0 }),
        (
            "Sea Level",
            DistNodeKind::SeaLevel {
                level: 0.0,
                width: 20.0,
            },
        ),
        (
            "Occlusion",
            DistNodeKind::Occlusion {
                radius: 4,
                strength: 1.0,
            },
        ),
        ("Cavity", DistNodeKind::Cavity { strength: 1.0 }),
        (
            "Angle",
            DistNodeKind::Angle {
                degrees: 45.0,
                spread: 30.0,
            },
        ),
        (
            "Roughness",
            DistNodeKind::Roughness {
                radius: 2,
                strength: 1.0,
            },
        ),
        (
            "Rocks",
            DistNodeKind::Rocks {
                density: 0.5,
                threshold: 0.4,
            },
        ),
        (
            "Rocky Edges",
            DistNodeKind::RockyEdges {
                width: 2.0,
                strength: 1.0,
            },
        ),
    ]
}

/// Distribution effect kinds (chained onto the last base node / WC modifiers).
pub fn dist_effect_kinds() -> Vec<(&'static str, DistNodeKind)> {
    vec![
        ("Invert", DistNodeKind::EffectInvert),
        ("Blur", DistNodeKind::EffectBlur { radius: 2 }),
        (
            "Remap",
            DistNodeKind::EffectRemap {
                in_min: 0.0,
                in_max: 1.0,
            },
        ),
        ("Contrast", DistNodeKind::EffectContrast { amount: 1.5 }),
        ("Clamp", DistNodeKind::EffectClamp { min: 0.0, max: 1.0 }),
        (
            "Smoothstep",
            DistNodeKind::EffectSmoothstep {
                edge0: 0.2,
                edge1: 0.8,
            },
        ),
        ("Curve", DistNodeKind::EffectCurve { contrast: 1.5 }),
        ("Edge", DistNodeKind::EffectEdge { strength: 1.0 }),
        (
            "Distortion",
            DistNodeKind::EffectDistortion {
                seed: 1,
                frequency: 0.05,
                amount: 4.0,
            },
        ),
        (
            "Simple Flow",
            DistNodeKind::EffectSimpleFlow {
                iterations: 4,
                strength: 0.5,
            },
        ),
    ]
}

pub fn dist_kind_icon(kind: &DistNodeKind) -> Icon {
    match kind {
        DistNodeKind::Fill { .. } => Icon::CircleDot,
        DistNodeKind::Noise { .. }
        | DistNodeKind::NoisePerlin { .. }
        | DistNodeKind::NoiseRidged { .. }
        | DistNodeKind::NoiseWorley { .. }
        | DistNodeKind::NoiseBillow { .. } => Icon::Sparkles,
        DistNodeKind::Slope { .. }
        | DistNodeKind::Steepness { .. }
        | DistNodeKind::Angle { .. } => Icon::Activity,
        DistNodeKind::Height { .. } | DistNodeKind::SeaLevel { .. } => Icon::Mountain,
        DistNodeKind::Curvature { .. }
        | DistNodeKind::Cavity { .. }
        | DistNodeKind::Roughness { .. } => Icon::Blend,
        DistNodeKind::Flow { .. } => Icon::Waves,
        DistNodeKind::Occlusion { .. } => Icon::Eye,
        DistNodeKind::Rocks { .. } | DistNodeKind::RockyEdges { .. } => Icon::Box,
        DistNodeKind::EffectInvert
        | DistNodeKind::EffectBlur { .. }
        | DistNodeKind::EffectRemap { .. }
        | DistNodeKind::EffectContrast { .. }
        | DistNodeKind::EffectClamp { .. }
        | DistNodeKind::EffectSmoothstep { .. }
        | DistNodeKind::EffectCurve { .. }
        | DistNodeKind::EffectEdge { .. }
        | DistNodeKind::EffectDistortion { .. }
        | DistNodeKind::EffectSimpleFlow { .. } => Icon::SlidersHorizontal,
        _ => Icon::Layers,
    }
}

pub fn dist_kind_description(label: &str, is_effect: bool) -> &'static str {
    if is_effect {
        match label {
            "Invert" => "Invert the parent distribution.",
            "Blur" => "Soften the parent distribution.",
            "Remap" => "Remap input range of the parent.",
            "Contrast" => "Increase contrast of the parent.",
            "Clamp" => "Clamp the parent to a range.",
            "Smoothstep" => "Smoothstep falloff on the parent.",
            "Curve" => "Contrast curve on the parent.",
            "Edge" => "Edge detect on the parent.",
            "Distortion" => "Distort the parent with noise.",
            "Simple Flow" => "Flow-like warp of the parent.",
            _ => "Modifier applied to the last distribution.",
        }
    } else {
        match label {
            "Fill" => "Constant coverage fill.",
            "Noise" | "Perlin" | "Ridged" | "Worley" | "Billow" => "Procedural noise distribution.",
            "Slope" | "Steepness" => "Mask from terrain slope.",
            "Height" => "Mask from a height range.",
            "Curvature" => "Mask from terrain curvature.",
            "Flow" => "Mask from flow accumulation.",
            "Sea Level" => "Band around sea level.",
            "Occlusion" => "Ambient occlusion–style mask.",
            "Cavity" => "Concave cavity mask.",
            "Angle" => "Aspect / sun-facing angle mask.",
            "Roughness" => "Local roughness mask.",
            "Rocks" | "Rocky Edges" => "Rocky feature mask.",
            _ => "Terrain distribution generator.",
        }
    }
}
