//! Global World Creator–style level-step schedule (upsample chain).

use serde::{Deserialize, Serialize};

use super::level_step::{author_sim_levels, SimLevel};

/// How high-detail preview is scoped for large terrains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum HighDetailMode {
    #[default]
    None,
    Zone,
    Camera,
}

impl HighDetailMode {
    pub fn label(self) -> &'static str {
        match self {
            HighDetailMode::None => "None",
            HighDetailMode::Zone => "Zone",
            HighDetailMode::Camera => "Camera",
        }
    }

    pub fn cycle(self) -> Self {
        match self {
            HighDetailMode::None => HighDetailMode::Zone,
            HighDetailMode::Zone => HighDetailMode::Camera,
            HighDetailMode::Camera => HighDetailMode::None,
        }
    }
}

/// Document-level level-step / precision settings (WC Terrain panel).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelStepSettings {
    /// Pixel-to-metre relation (WC Precision). Higher = more metres per pixel.
    #[serde(default = "default_precision")]
    pub precision: f32,
    /// Scales base noises / distributions / filters (WC Noise World Scale).
    #[serde(default = "default_world_scale")]
    pub world_scale: f32,
    /// Max upsample level index (0-based). 0 = derive from resolution.
    #[serde(default)]
    pub max_level: u32,
    /// High-detail preview mode for large terrains.
    #[serde(default)]
    pub high_detail: HighDetailMode,
    /// Show outline around high-detail zone.
    #[serde(default)]
    pub show_hd_outline: bool,
    /// UV rect for Zone HD preview (u0, v0, u1, v1) in \[0,1\].
    #[serde(default = "default_hd_zone")]
    pub hd_zone: [f32; 4],
    /// Artist-facing curve: minimum physical wavelength represented at preview (metres).
    /// Operations finer than this are truncated on Draft/Medium.
    #[serde(default = "default_min_wavelength")]
    pub min_preview_wavelength_m: f32,
}

fn default_precision() -> f32 {
    1.0
}

fn default_world_scale() -> f32 {
    1.0
}

fn default_hd_zone() -> [f32; 4] {
    [0.3, 0.3, 0.7, 0.7]
}

fn default_min_wavelength() -> f32 {
    1.0
}

impl Default for LevelStepSettings {
    fn default() -> Self {
        Self {
            precision: 1.0,
            world_scale: 1.0,
            max_level: 0,
            high_detail: HighDetailMode::None,
            show_hd_outline: false,
            hd_zone: default_hd_zone(),
            min_preview_wavelength_m: default_min_wavelength(),
        }
    }
}

impl LevelStepSettings {
    /// Number of upsample levels for a target resolution (2, 4, 8, …, ≥ target).
    pub fn level_count_for_resolution(target_res: u32) -> u32 {
        let mut n = 1u32;
        let mut r = 2u32;
        while r < target_res.max(2) {
            r = r.saturating_mul(2);
            n += 1;
            if n > 16 {
                break;
            }
        }
        n
    }

    /// Resolution at level index `level` (0 → 2×2, 1 → 4×4, …).
    pub fn resolution_at_level(level: u32) -> u32 {
        2u32.saturating_pow(level.saturating_add(1).min(16))
    }

    /// Build the full upsample resolution chain up to `target_res`.
    pub fn upsample_chain(target_res: u32) -> Vec<u32> {
        let mut chain = Vec::new();
        let mut r = 2u32;
        while r < target_res {
            chain.push(r);
            r = r.saturating_mul(2);
            if chain.len() > 16 {
                break;
            }
        }
        if chain.last().copied() != Some(target_res) {
            chain.push(target_res.max(2));
        }
        chain
    }

    /// Effective max level for this document given preview/export resolution.
    pub fn effective_max_level(&self, target_res: u32) -> u32 {
        let derived = Self::level_count_for_resolution(target_res).saturating_sub(1);
        if self.max_level == 0 {
            derived
        } else {
            self.max_level.min(derived)
        }
    }

    /// Truncate a sim schedule for progressive preview quality.
    ///
    /// Draft → coarsest third, Medium → two thirds, Full/Export → all levels.
    pub fn truncate_for_preview(
        levels: Vec<SimLevel>,
        quality: crate::quality::PreviewQuality,
    ) -> Vec<SimLevel> {
        if levels.is_empty() {
            return levels;
        }
        let n = levels.len();
        let keep = match quality {
            crate::quality::PreviewQuality::Draft => (n + 2) / 3,
            crate::quality::PreviewQuality::Medium => (2 * n + 2) / 3,
            crate::quality::PreviewQuality::Full | crate::quality::PreviewQuality::Export => n,
        }
        .max(1)
        .min(n);
        levels.into_iter().take(keep).collect()
    }

    /// Drop levels whose wavelength is finer than `min_wavelength_m` (preview truncation).
    pub fn filter_by_wavelength(levels: Vec<SimLevel>, min_wavelength_m: f32) -> Vec<SimLevel> {
        let min_w = min_wavelength_m.max(0.1);
        let filtered: Vec<_> = levels
            .into_iter()
            .filter(|l| l.physical_wavelength_m >= min_w * 0.5)
            .collect();
        if filtered.is_empty() {
            // Always keep at least the coarsest representation.
            vec![SimLevel::new(64, 1.0, 1.0).with_wavelength(min_w.max(16.0))]
        } else {
            filtered
        }
    }

    /// Apply author controls then truncate for preview quality.
    pub fn schedule_for_filter(
        &self,
        base: Vec<SimLevel>,
        level_count: u32,
        start_level: u32,
        strength: f32,
        strength_curve: &[f32],
        quality: crate::quality::PreviewQuality,
    ) -> Vec<SimLevel> {
        let mut levels = author_sim_levels(base, level_count, start_level, strength);
        // Precision is metres per simulation pixel. Values above one solve on
        // a coarser grid; values below one retain more fine structure.
        let precision = self.precision.clamp(0.25, 4.0);
        for level in &mut levels {
            level.resolution = ((level.resolution as f32 / precision).round() as u32)
                .max(32)
                .min(16_384);
        }
        levels.dedup_by(|a, b| a.resolution == b.resolution);

        if !strength_curve.is_empty() {
            for (i, level) in levels.iter_mut().enumerate() {
                let s = strength_curve
                    .get(i)
                    .or_else(|| strength_curve.last())
                    .copied()
                    .unwrap_or(1.0)
                    .clamp(0.0, 2.0);
                level.effect_scale = (level.effect_scale * s).clamp(0.05, 2.5);
                level.iter_scale = (level.iter_scale * (0.5 + 0.5 * s)).clamp(0.05, 1.5);
            }
        }
        // Honour document max_level cap.
        if self.max_level > 0 && levels.len() as u32 > self.max_level {
            levels.truncate(self.max_level as usize);
        }
        Self::truncate_for_preview(levels, quality)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyze::default_sim_levels;
    use crate::quality::PreviewQuality;

    #[test]
    fn upsample_chain_doubles() {
        let chain = LevelStepSettings::upsample_chain(256);
        assert_eq!(chain.first(), Some(&2));
        assert_eq!(chain.last(), Some(&256));
        assert!(chain.windows(2).all(|w| w[1] == w[0] * 2 || w[1] == 256));
    }

    #[test]
    fn preview_truncates_draft() {
        let base = default_sim_levels(512);
        let draft = LevelStepSettings::truncate_for_preview(base.clone(), PreviewQuality::Draft);
        let full = LevelStepSettings::truncate_for_preview(base, PreviewQuality::Full);
        assert!(draft.len() <= full.len());
        assert!(!draft.is_empty());
    }

    #[test]
    fn precision_changes_solver_resolution_without_changing_authored_strength() {
        let base = default_sim_levels(512);
        let coarse = LevelStepSettings {
            precision: 2.0,
            ..Default::default()
        }
        .schedule_for_filter(base.clone(), 0, 0, 1.0, &[], PreviewQuality::Full);
        let fine = LevelStepSettings {
            precision: 0.5,
            ..Default::default()
        }
        .schedule_for_filter(base, 0, 0, 1.0, &[], PreviewQuality::Full);
        assert!(coarse.last().unwrap().resolution < fine.last().unwrap().resolution);
        assert!((coarse[0].effect_scale - fine[0].effect_scale).abs() < 1e-6);
    }
}
