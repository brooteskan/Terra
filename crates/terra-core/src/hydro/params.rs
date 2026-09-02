//! Persisted parameters for hydrology algorithms.

use crate::mask_types::MaskSource;
use serde::{Deserialize, Serialize};

/// Stream-power erosion (SPE) parameters.
///
/// Incision follows \(E = K\,A^{m}\,S^{n}\) modulated by softness \(1-K_{\mathrm{hard}}\).
/// Drainage uses Priority-Flood + D8/D∞ on the CPU export oracle; Draft/Medium GPU
/// runs an approximate multi-pass D8 SPE.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamPowerParams {
    pub iterations: u32,
    /// Erodibility coefficient \(K\) in the stream-power law.
    ///
    /// Acts on **grid-relative** D8 slope (drop per 1 / √2 cells) and world-m²
    /// drainage area — the convention baked into [`crate::hydro::spe_increment`]
    /// on the `hydro` path. This is **not** numerically comparable with
    /// [`crate::landscape_evolution::LandscapeEvolutionParams`]'s `k`, which acts
    /// on world-metric slope and rain-scaled discharge. Do not silently unify the
    /// two: it would retune every saved project and needs a versioned document
    /// migration (station D1's territory).
    pub k: f32,
    /// Drainage-area exponent \(m\) (classic ~0.5).
    pub m: f32,
    /// Local-slope exponent \(n\) (classic ~1).
    pub n: f32,
    /// Optional tectonic uplift per iteration (meters). Usually 0 when uplift is a prior layer.
    pub uplift_rate: f32,
    /// Heights cannot cut below this floor (meters).
    pub base_level: f32,
    /// Incision time-step / scale factor.
    pub dt: f32,
    /// Prefer D∞ accumulation when true; else D8.
    pub use_dinfinity: bool,
    /// Re-run Priority-Flood every iteration (costlier, fewer trapped pits).
    pub refill_each_iter: bool,
    /// Recompute D8/D∞ drainage every N SPE iterations (1 = every iter).
    /// Draft preview may raise this to skip barriers; Full/Export keeps 1.
    #[serde(default = "default_drainage_reuse_stride")]
    pub drainage_reuse_stride: u32,
    /// Constant bedrock hardness \(K \in [0,1]\); soft rock incises faster.
    #[serde(default = "default_soft_hardness")]
    pub hardness: f32,
    #[serde(default)]
    pub hardness_source: MaskSource,
    /// 0 = off; >0 applies a light ridge-distance valley prime before SPE (Dendry-inspired).
    #[serde(default)]
    pub dendritic_seed: f32,
    /// Accumulation threshold used when baking the stream-order aux map.
    #[serde(default = "default_stream_threshold")]
    pub stream_threshold: f32,
    /// 0 = quality default schedule length; else clamp multilevel count.
    #[serde(default)]
    pub level_count: u32,
    /// Skip this many coarse levels (WC Start Level).
    #[serde(default)]
    pub start_level: u32,
    /// Scales fine-level effect (WC Level Step Strength).
    #[serde(default = "default_level_step_strength")]
    pub level_step_strength: f32,
    /// Per-level strength bars (empty = scalar only).
    #[serde(default)]
    pub level_step_curve: Vec<f32>,
}

fn default_stream_threshold() -> f32 {
    40.0
}

fn default_drainage_reuse_stride() -> u32 {
    1
}

fn default_soft_hardness() -> f32 {
    0.0
}

fn default_level_step_strength() -> f32 {
    1.0
}

impl Default for StreamPowerParams {
    fn default() -> Self {
        Self {
            iterations: 24,
            k: 0.08,
            m: 0.5,
            n: 1.0,
            uplift_rate: 0.0,
            base_level: 0.0,
            dt: 1.0,
            use_dinfinity: false,
            refill_each_iter: false,
            drainage_reuse_stride: 1,
            hardness: 0.0,
            hardness_source: MaskSource::None,
            dendritic_seed: 0.0,
            stream_threshold: 40.0,
            level_count: 0,
            start_level: 0,
            level_step_strength: 1.0,
            level_step_curve: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiverCarveParams {
    pub accumulation_threshold: f32,
    pub depth: f32,
    pub width: f32,
    pub bank_smooth: f32,
    pub use_dinfinity: bool,
    #[serde(default)]
    pub guide: crate::mask_types::MaskSource,
    #[serde(default = "default_river_guide_boost")]
    pub guide_boost: f32,
}

fn default_river_guide_boost() -> f32 {
    3.0
}

impl Default for RiverCarveParams {
    fn default() -> Self {
        Self {
            accumulation_threshold: 50.0,
            depth: 25.0,
            width: 4.0,
            bank_smooth: 1.5,
            use_dinfinity: true,
            guide: crate::mask_types::MaskSource::None,
            guide_boost: 3.0,
        }
    }
}
