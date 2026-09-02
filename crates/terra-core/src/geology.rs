//! Shared geological value types and depth-query kernels.

use serde::{Deserialize, Serialize};

/// Lithology class for a geological stratum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum StratumMaterial {
    #[default]
    Sedimentary,
    Igneous,
    Metamorphic,
    Unconsolidated,
    Soil,
    Ice,
}

impl StratumMaterial {
    pub fn stability(self) -> f32 {
        match self {
            Self::Igneous => 0.92,
            Self::Metamorphic => 0.88,
            Self::Sedimentary => 0.55,
            Self::Unconsolidated => 0.18,
            Self::Soil => 0.22,
            Self::Ice => 0.12,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Sedimentary => "Sedimentary",
            Self::Igneous => "Igneous",
            Self::Metamorphic => "Metamorphic",
            Self::Unconsolidated => "Unconsolidated",
            Self::Soil => "Soil",
            Self::Ice => "Ice",
        }
    }

    pub fn cycle(self) -> Self {
        match self {
            Self::Sedimentary => Self::Igneous,
            Self::Igneous => Self::Metamorphic,
            Self::Metamorphic => Self::Unconsolidated,
            Self::Unconsolidated => Self::Soil,
            Self::Soil => Self::Ice,
            Self::Ice => Self::Sedimentary,
        }
    }
}

/// Attitude of a geological bed stack, independent of the free surface.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BedGeometry {
    #[default]
    Horizontal,
    Tilted {
        dip_deg: f32,
        azimuth_deg: f32,
    },
    Folded {
        amplitude_m: f32,
        wavelength_m: f32,
        seed: u64,
    },
    Warped {
        frequency: f32,
        amplitude_m: f32,
        seed: u64,
    },
}

impl BedGeometry {
    pub fn depth_warp(self, x: f32, z: f32) -> f32 {
        match self {
            Self::Horizontal => 0.0,
            Self::Tilted {
                dip_deg,
                azimuth_deg,
            } => {
                let dip = dip_deg.to_radians();
                let az = azimuth_deg.to_radians();
                (x * az.cos() + z * az.sin()) * dip.sin()
            }
            Self::Folded {
                amplitude_m,
                wavelength_m,
                seed,
            } => {
                let wl = wavelength_m.max(1.0);
                let fold = (x / wl * std::f32::consts::TAU).sin() * amplitude_m;
                let n = crate::noise::sample_noise(
                    crate::noise::FractalNoiseType::Perlin,
                    x / wl,
                    z / wl,
                    seed,
                );
                fold + n * amplitude_m * 0.35
            }
            Self::Warped {
                frequency,
                amplitude_m,
                seed,
            } => {
                let f = frequency.max(1e-5);
                crate::noise::sample_noise(
                    crate::noise::FractalNoiseType::Perlin,
                    x * f,
                    z * f,
                    seed,
                ) * amplitude_m
            }
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Horizontal => "Horizontal",
            Self::Tilted { .. } => "Tilted",
            Self::Folded { .. } => "Folded",
            Self::Warped { .. } => "Warped",
        }
    }

    pub fn cycle(self) -> Self {
        match self {
            Self::Horizontal => Self::Tilted {
                dip_deg: 12.0,
                azimuth_deg: 45.0,
            },
            Self::Tilted { .. } => Self::Folded {
                amplitude_m: 25.0,
                wavelength_m: 180.0,
                seed: 11,
            },
            Self::Folded { .. } => Self::Warped {
                frequency: 0.012,
                amplitude_m: 22.0,
                seed: 11,
            },
            Self::Warped { .. } => Self::Horizontal,
        }
    }
}

/// One layer in a practical material stack, ordered surface to bedrock.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stratum {
    pub name: String,
    pub id: u32,
    #[serde(default = "default_material_hardness")]
    pub hardness: f32,
    #[serde(default = "default_stratum_thickness")]
    pub thickness: f32,
    #[serde(default = "default_stratum_erodibility_sentinel")]
    pub erodibility: f32,
    #[serde(default)]
    pub material_type: StratumMaterial,
}

fn default_material_hardness() -> f32 {
    0.5
}

fn default_stratum_thickness() -> f32 {
    1.0e6
}

fn default_stratum_erodibility_sentinel() -> f32 {
    -1.0
}

impl Stratum {
    pub fn soft_cap(thickness: f32) -> Self {
        Self {
            name: "Soft Cap".into(),
            id: 3,
            hardness: 0.08,
            thickness,
            erodibility: 0.92,
            material_type: StratumMaterial::Unconsolidated,
        }
    }

    pub fn hard_base() -> Self {
        Self {
            name: "Hard Base".into(),
            id: 1,
            hardness: 0.92,
            thickness: default_stratum_thickness(),
            erodibility: 0.08,
            material_type: StratumMaterial::Igneous,
        }
    }

    pub fn effective_erodibility(&self) -> f32 {
        if self.erodibility < 0.0 {
            (1.0 - self.hardness).clamp(0.0, 1.0)
        } else {
            self.erodibility.clamp(0.0, 1.0)
        }
    }

    pub fn material_stability(&self) -> f32 {
        let base = self.material_type.stability();
        (base * (0.35 + 0.65 * self.hardness.clamp(0.0, 1.0))).clamp(0.0, 1.0)
    }
}

/// Depth below the material reference surface, warped by bed geometry.
pub fn strata_depth_m(h_ref: f32, h: f32, x: f32, z: f32, geom: &BedGeometry) -> f32 {
    (h_ref - h + geom.depth_warp(x, z)).max(0.0)
}

pub fn stratum_at_depth(strata: &[Stratum], depth: f32) -> Option<&Stratum> {
    let mut remaining = depth.max(0.0);
    for stratum in strata {
        let thickness = stratum.thickness.max(0.0);
        if remaining <= thickness || thickness >= 1.0e5 {
            return Some(stratum);
        }
        remaining -= thickness;
    }
    strata.last()
}

pub fn hardness_at_strata_depth(strata: &[Stratum], depth: f32, fallback: f32) -> f32 {
    stratum_at_depth(strata, depth)
        .map(|s| s.hardness.clamp(0.0, 1.0))
        .unwrap_or_else(|| fallback.clamp(0.0, 1.0))
}

pub fn erodibility_at_strata_depth(strata: &[Stratum], depth: f32, fallback: f32) -> f32 {
    stratum_at_depth(strata, depth)
        .map(Stratum::effective_erodibility)
        .unwrap_or_else(|| (1.0 - fallback).clamp(0.0, 1.0))
}

pub fn stability_at_strata_depth(strata: &[Stratum], depth: f32, fallback: f32) -> f32 {
    stratum_at_depth(strata, depth)
        .map(Stratum::material_stability)
        .unwrap_or_else(|| fallback.clamp(0.0, 1.0))
}

pub fn material_id_at_strata_depth(strata: &[Stratum], depth: f32) -> f32 {
    stratum_at_depth(strata, depth)
        .map(|s| (s.id as f32 / 16.0).clamp(0.0, 1.0))
        .unwrap_or(0.0)
}
