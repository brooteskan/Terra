//! Placement paint/rule combination schema.

use super::MaskCombine;
use serde::{Deserialize, Serialize};

/// How manual paint combines with procedural placement rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PlacementCombineMode {
    #[default]
    PaintOnly,
    RulesOnly,
    PaintMulRules,
    PaintAddRules,
    PaintOverridesRules,
    RulesOutsidePaint,
}

impl PlacementCombineMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::PaintOnly => "Paint Only",
            Self::RulesOnly => "Rules Only",
            Self::PaintMulRules => "Paint x Rules",
            Self::PaintAddRules => "Paint + Rules",
            Self::PaintOverridesRules => "Paint Overrides Rules",
            Self::RulesOutsidePaint => "Rules Outside Painted Area",
        }
    }

    /// Plain-language ownership control for artists.
    pub fn artist_label(self) -> &'static str {
        match self {
            Self::PaintOverridesRules | Self::RulesOutsidePaint => "Paint owns · rules fill gaps",
            Self::PaintMulRules => "Guided by rules",
            Self::PaintAddRules => "Paint + rules",
            Self::PaintOnly => "Paint only",
            Self::RulesOnly => "Rules only",
        }
    }

    /// Toggle between ownership paint (default) and guided multiply.
    pub fn cycle_artist(self) -> Self {
        match self {
            Self::PaintOverridesRules | Self::RulesOutsidePaint | Self::PaintOnly => {
                Self::PaintMulRules
            }
            _ => Self::PaintOverridesRules,
        }
    }

    pub fn combine(self, manual: f32, procedural: f32) -> f32 {
        let m = manual.clamp(0.0, 1.0);
        let p = procedural.clamp(0.0, 1.0);
        match self {
            Self::PaintOnly => m,
            Self::RulesOnly => p,
            Self::PaintMulRules => m * p,
            Self::PaintAddRules => (m + p).clamp(0.0, 1.0),
            Self::PaintOverridesRules | Self::RulesOutsidePaint => {
                if m > 1e-4 {
                    m
                } else {
                    p
                }
            }
        }
    }

    /// Map to DistNode / mask stack combine when rules are DistNodes and paint is a mask ref.
    /// Paint is applied *after* DistNodes in `bake_distribution_with_context`.
    pub fn mask_combine(self) -> super::MaskCombine {
        match self {
            Self::PaintMulRules => MaskCombine::Multiply,
            Self::PaintAddRules => MaskCombine::Add,
            Self::PaintOnly => MaskCombine::Replace,
            Self::PaintOverridesRules | Self::RulesOutsidePaint => MaskCombine::PaintOverride,
            // Paint entry should be omitted for RulesOnly; Multiply is a harmless fallback.
            Self::RulesOnly => MaskCombine::Multiply,
        }
    }

    /// Whether manual paint should be attached to the biome distribution.
    pub fn uses_manual_paint(self) -> bool {
        !matches!(self, Self::RulesOnly)
    }
}
