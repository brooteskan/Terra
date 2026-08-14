//! Mask fields, sources, and compositing ops.

mod bake;
mod dist_nodes;
mod distribution;
mod field;
mod geom;
mod ops;
mod paint;
mod placement;

pub use bake::{bake_mask_assets, bake_mask_assets_resolved};
pub use dist_nodes::{
    apply_effect_public, bake_dist_node_base, bake_dist_nodes, ClimateMaskChannel, DistBakeContext,
    DistNode, DistNodeId, DistNodeKind,
};
pub use distribution::{
    bake_distribution, bake_distribution_with_context, Distribution, DistributionEntry, MaskCombine,
};
pub use field::MaskField;
pub use geom::{dist_point_segment, point_in_polygon};
pub use ops::{apply_mask_ops, MaskOp};
pub use paint::{MaskPaintTool, PaintBuffer, PaintStroke};
pub use placement::{
    coverage_estimate, CompareOp, Condition, ConditionChannel, CoverageTerm,
    PlacementCoordinateSpace, PlacementDefinition, PlacementRefinement, PlacementSource, RuleGroup,
    RuleGroupMode, RuleNode, MAX_RULE_NEST_DEPTH,
};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MaskId(pub Uuid);

impl MaskId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn nil() -> Self {
        Self(Uuid::nil())
    }
}

impl Default for MaskId {
    fn default() -> Self {
        Self::new()
    }
}

/// Reference from a layer to a mask with strength.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskRef {
    pub id: MaskId,
    pub strength: f32,
    pub invert: bool,
}

impl MaskRef {
    pub fn new(id: MaskId) -> Self {
        Self {
            id,
            strength: 1.0,
            invert: false,
        }
    }
}

/// Where a mask's values come from.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum MaskSource {
    #[default]
    None,
    Constant(f32),
    Height {
        min: f32,
        max: f32,
    },
    Slope {
        min_deg: f32,
        max_deg: f32,
    },
    Aspect {
        center_deg: f32,
        width_deg: f32,
    },
    Curvature {
        min: f32,
        max: f32,
    },
    Convexity,
    Concavity,
    AmbientOcclusion {
        radius: u32,
        strength: f32,
    },
    DistanceField {
        threshold: f32,
    },
    Noise {
        seed: u64,
        frequency: f32,
    },
    Painted {
        mask_id: MaskId,
    },
    /// Filled by simulation layers (schema-ready).
    FlowDirection,
    FlowAccumulation {
        min: f32,
        max: f32,
    },
    Wetness,
    Sediment,
    Erosion,
    Deposition,
    /// Bedrock hardness K ∈ \[0,1\] from sims / materials bake.
    Hardness,
    /// Phase H climate aux (normalized \[0,1\]).
    Temperature,
    Rainfall,
    Humidity,
    Snow,
    SoilMoisture,
    WindExposure,
    Named(String),
    /// Reference a published layer/group output by stable id.
    LayerOutput {
        output_id: crate::ids::OutputId,
    },
}

/// Default viewport overlay colour for masks (visual only — not used in eval).
pub fn default_mask_display_color() -> [f32; 3] {
    [0.20, 0.75, 0.95]
}

/// Stable pastel overlay colour derived from a mask id (visual only).
pub fn display_color_for_mask_id(id: MaskId) -> [f32; 3] {
    let h = (id.0.as_u128() % 360) as f32;
    let s = 0.62;
    let v = 0.92;
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    [r + m, g + m, b + m]
}

/// A named mask asset in the project.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskAsset {
    pub id: MaskId,
    pub name: String,
    pub source: MaskSource,
    pub ops: Vec<MaskOp>,
    /// Optional editable UV-space paint data. Missing data remains compatible with old projects.
    #[serde(default)]
    pub paint: Option<PaintBuffer>,
    /// Viewport overlay colour (RGB 0–1). Visual only — does not affect evaluation.
    #[serde(default = "default_mask_display_color")]
    pub display_color: [f32; 3],
}

impl MaskAsset {
    pub fn new(id: MaskId, name: impl Into<String>, source: MaskSource) -> Self {
        Self {
            id,
            name: name.into(),
            source,
            ops: Vec::new(),
            paint: None,
            display_color: display_color_for_mask_id(id),
        }
    }

    pub fn new_painted(id: MaskId, name: impl Into<String>, resolution: u32) -> Self {
        Self {
            id,
            name: name.into(),
            source: MaskSource::Painted { mask_id: id },
            ops: Vec::new(),
            paint: Some(PaintBuffer::new(resolution, resolution)),
            display_color: display_color_for_mask_id(id),
        }
    }

    pub fn is_painted(&self) -> bool {
        matches!(self.source, MaskSource::Painted { .. })
    }

    /// Fix painted self-id and ensure a paint buffer before inserting into the document.
    pub fn prepare_for_document(&mut self) {
        if let MaskSource::Painted { mask_id } = &mut self.source {
            *mask_id = self.id;
            if self.paint.is_none() {
                self.paint = Some(PaintBuffer::new(512, 512));
            }
        }
    }
}
