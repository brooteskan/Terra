//! Spatial reach contributed by mask distributions and mask assets.

use crate::invalidation::Reach;
use crate::mask_ir::{
    DistNode, DistNodeKind, Distribution, MaskAsset, MaskOp, MaskRef, MaskSource,
};

/// Reach contributed by a mask distribution. Empty distribution is local.
pub fn distribution_reach(dist: &Distribution, mask_assets: &[MaskAsset]) -> Reach {
    let mut reach = Reach::LOCAL;
    for entry in &dist.entries {
        reach = reach.combine(mask_ref_reach(&entry.mask, mask_assets));
        if reach.is_full() {
            return Reach::Full;
        }
    }
    for node in &dist.nodes {
        reach = reach.combine(node_reach(node, mask_assets));
        if reach.is_full() {
            return Reach::Full;
        }
    }
    reach
}

fn node_reach(node: &DistNode, mask_assets: &[MaskAsset]) -> Reach {
    if !node.enabled {
        return Reach::LOCAL;
    }
    let mut reach = dist_node_reach(&node.kind, mask_assets);
    for child in &node.children {
        reach = reach.combine(node_reach(child, mask_assets));
        if reach.is_full() {
            return Reach::Full;
        }
    }
    reach
}

fn dist_node_reach(kind: &DistNodeKind, mask_assets: &[MaskAsset]) -> Reach {
    use DistNodeKind::*;
    match kind {
        Fill { .. }
        | Noise { .. }
        | NoisePerlin { .. }
        | NoiseRidged { .. }
        | NoiseWorley { .. }
        | NoiseBillow { .. }
        | Height { .. }
        | Flow { .. }
        | SeaLevel { .. }
        | Climate { .. }
        | Voronoi { .. }
        | Polygon { .. }
        | Spline { .. }
        | GroupAll
        | GroupAny
        | EffectInvert
        | EffectLevels { .. }
        | EffectRemap { .. }
        | EffectContrast { .. }
        | EffectClamp { .. }
        | EffectCurve { .. }
        | EffectSmoothstep { .. } => Reach::LOCAL,
        Slope { .. }
        | Curvature { .. }
        | Cavity { .. }
        | Steepness { .. }
        | Angle { .. }
        | Rocks { .. }
        | RockyEdges { .. }
        | EffectEdge { .. } => Reach::Localized { halo_samples: 1 },
        Occlusion { radius, .. } | Roughness { radius, .. } | EffectBlur { radius } => {
            Reach::Localized {
                halo_samples: *radius,
            }
        }
        EffectSimpleFlow { iterations, .. } => Reach::Localized {
            halo_samples: *iterations,
        },
        EffectDistortion { .. } | Distance { .. } | EffectDilate { .. } | EffectErode { .. } => {
            Reach::Full
        }
        MaskAsset { mask } | Paint { mask } | ImportedMask { mask } => {
            mask_ref_reach(mask, mask_assets)
        }
    }
}

fn mask_ref_reach(mask: &MaskRef, mask_assets: &[MaskAsset]) -> Reach {
    let Some(asset) = mask_assets.iter().find(|asset| asset.id == mask.id) else {
        return Reach::Full;
    };
    let mut reach = mask_source_reach(&asset.source);
    for op in &asset.ops {
        reach = reach.combine(mask_op_reach(op));
        if reach.is_full() {
            return Reach::Full;
        }
    }
    reach
}

fn mask_source_reach(source: &MaskSource) -> Reach {
    use MaskSource::*;
    match source {
        None | Constant(_) | Noise { .. } | Painted { .. } | Height { .. } => Reach::LOCAL,
        Slope { .. } | Aspect { .. } | Curvature { .. } | Convexity | Concavity => {
            Reach::Localized { halo_samples: 1 }
        }
        AmbientOcclusion { radius, .. } => Reach::Localized {
            halo_samples: *radius,
        },
        DistanceField { .. } => Reach::Full,
        FlowDirection
        | FlowAccumulation { .. }
        | Wetness
        | Sediment
        | Erosion
        | Deposition
        | Hardness
        | Temperature
        | Rainfall
        | Humidity
        | Snow
        | SoilMoisture
        | WindExposure
        | Named(_)
        | LayerOutput { .. } => Reach::LOCAL,
    }
}

fn mask_op_reach(op: &MaskOp) -> Reach {
    match op {
        MaskOp::Blur { radius } => Reach::Localized {
            halo_samples: *radius,
        },
        MaskOp::Add { .. }
        | MaskOp::Subtract { .. }
        | MaskOp::Multiply { .. }
        | MaskOp::Min { .. }
        | MaskOp::Max { .. }
        | MaskOp::Invert
        | MaskOp::Clamp { .. }
        | MaskOp::Levels { .. }
        | MaskOp::Smoothstep { .. }
        | MaskOp::Remap { .. } => Reach::LOCAL,
    }
}
