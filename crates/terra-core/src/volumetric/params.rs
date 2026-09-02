//! Persisted parameters for local volumetric geology operations.

use serde::{Deserialize, Serialize};

/// Dual-height cliff undercut / shelf stamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverhangStampParams {
    #[serde(default = "default_overhang_u")]
    pub u: f32,
    #[serde(default = "default_overhang_v")]
    pub v: f32,
    #[serde(default = "default_overhang_radius")]
    pub radius_uv: f32,
    #[serde(default = "default_overhang_depth")]
    pub depth: f32,
    #[serde(default = "default_overhang_lip")]
    pub lip_height: f32,
    #[serde(default = "default_overhang_entrance")]
    pub entrance_angle_deg: f32,
    #[serde(default = "default_overhang_falloff")]
    pub falloff: f32,
    #[serde(default = "default_overhang_seed")]
    pub seed: u64,
    #[serde(default = "default_overhang_noise")]
    pub noise_amplitude: f32,
}

fn default_overhang_u() -> f32 {
    0.5
}

fn default_overhang_v() -> f32 {
    0.5
}

fn default_overhang_radius() -> f32 {
    0.08
}

fn default_overhang_depth() -> f32 {
    18.0
}

fn default_overhang_lip() -> f32 {
    2.0
}

fn default_overhang_entrance() -> f32 {
    180.0
}

fn default_overhang_falloff() -> f32 {
    0.35
}

fn default_overhang_seed() -> u64 {
    11
}

fn default_overhang_noise() -> f32 {
    0.25
}

impl Default for OverhangStampParams {
    fn default() -> Self {
        Self {
            u: default_overhang_u(),
            v: default_overhang_v(),
            radius_uv: default_overhang_radius(),
            depth: default_overhang_depth(),
            lip_height: default_overhang_lip(),
            entrance_angle_deg: default_overhang_entrance(),
            falloff: default_overhang_falloff(),
            seed: default_overhang_seed(),
            noise_amplitude: default_overhang_noise(),
        }
    }
}

impl OverhangStampParams {
    /// Preset aimed at a mid-world cliff face opening westward.
    pub fn cliff_overhang() -> Self {
        Self {
            u: 0.52,
            v: 0.5,
            radius_uv: 0.1,
            depth: 22.0,
            lip_height: 3.0,
            entrance_angle_deg: 180.0,
            falloff: 0.3,
            seed: 19,
            noise_amplitude: 0.3,
        }
    }
}

/// Local analytic SDF cave pocket projected onto a dual-height result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalSdfParams {
    #[serde(default = "default_sdf_u")]
    pub u: f32,
    #[serde(default = "default_sdf_v")]
    pub v: f32,
    #[serde(default = "default_sdf_size_x")]
    pub size_x: f32,
    #[serde(default = "default_sdf_size_y")]
    pub size_y: f32,
    #[serde(default = "default_sdf_size_z")]
    pub size_z: f32,
    #[serde(default = "default_sdf_depth")]
    pub depth: f32,
    #[serde(default = "default_sdf_entrance_r")]
    pub entrance_radius: f32,
    #[serde(default = "default_sdf_entrance_ang")]
    pub entrance_angle_deg: f32,
    #[serde(default)]
    pub lip_height: f32,
    #[serde(default = "default_sdf_seed")]
    pub seed: u64,
    #[serde(default = "default_sdf_noise")]
    pub noise_amplitude: f32,
    #[serde(default = "default_sdf_samples")]
    pub vertical_samples: u32,
}

fn default_sdf_u() -> f32 {
    0.55
}

fn default_sdf_v() -> f32 {
    0.5
}

fn default_sdf_size_x() -> f32 {
    28.0
}

fn default_sdf_size_y() -> f32 {
    14.0
}

fn default_sdf_size_z() -> f32 {
    22.0
}

fn default_sdf_depth() -> f32 {
    20.0
}

fn default_sdf_entrance_r() -> f32 {
    5.0
}

fn default_sdf_entrance_ang() -> f32 {
    180.0
}

fn default_sdf_seed() -> u64 {
    23
}

fn default_sdf_noise() -> f32 {
    0.2
}

fn default_sdf_samples() -> u32 {
    24
}

impl Default for LocalSdfParams {
    fn default() -> Self {
        Self {
            u: default_sdf_u(),
            v: default_sdf_v(),
            size_x: default_sdf_size_x(),
            size_y: default_sdf_size_y(),
            size_z: default_sdf_size_z(),
            depth: default_sdf_depth(),
            entrance_radius: default_sdf_entrance_r(),
            entrance_angle_deg: default_sdf_entrance_ang(),
            lip_height: 0.5,
            seed: default_sdf_seed(),
            noise_amplitude: default_sdf_noise(),
            vertical_samples: default_sdf_samples(),
        }
    }
}

impl LocalSdfParams {
    /// Compact karst / pocket cave preset.
    pub fn karst_pocket() -> Self {
        Self {
            u: 0.58,
            v: 0.48,
            size_x: 24.0,
            size_y: 12.0,
            size_z: 18.0,
            depth: 18.0,
            entrance_radius: 4.5,
            entrance_angle_deg: 200.0,
            lip_height: 1.0,
            seed: 31,
            noise_amplitude: 0.28,
            vertical_samples: 28,
        }
    }
}
