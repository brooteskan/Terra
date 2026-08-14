//! Noise parameter vocabulary.
//!
//! These param/enum types describe the noise this module generates, so they
//! live beside the algorithms (A3-B1). `layer::kinds` re-exports them, keeping
//! `terra_core::layer::NoiseParams` and friends valid for documents and the app.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseParams {
    pub seed: u64,
    pub frequency: f32,
    pub amplitude: f32,
    pub octaves: u32,
    pub lacunarity: f32,
    pub persistence: f32,
    pub offset_x: f32,
    pub offset_z: f32,
    pub remap_min: f32,
    pub remap_max: f32,
}

impl Default for NoiseParams {
    fn default() -> Self {
        Self {
            seed: 1,
            frequency: 0.002,
            amplitude: 120.0,
            octaves: 1,
            lacunarity: 2.0,
            persistence: 0.5,
            offset_x: 0.0,
            offset_z: 0.0,
            remap_min: -1.0,
            remap_max: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorleyParams {
    pub base: NoiseParams,
    pub distance_metric: WorleyMetric,
    pub feature: WorleyFeature,
}

impl Default for WorleyParams {
    fn default() -> Self {
        Self {
            base: NoiseParams {
                octaves: 1,
                frequency: 0.004,
                ..NoiseParams::default()
            },
            distance_metric: WorleyMetric::Euclidean,
            feature: WorleyFeature::F1,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum WorleyMetric {
    #[default]
    Euclidean,
    Manhattan,
    Chebyshev,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum WorleyFeature {
    #[default]
    F1,
    F2,
    F2MinusF1,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum FractalNoiseType {
    Value,
    #[default]
    Perlin,
    OpenSimplex,
}
