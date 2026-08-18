//! Effective spatial reach of a configured layer.
//!
//! [`crate::layer::LayerKind::spatial_dependency`] and
//! [`crate::layer::LayerKind::intrinsic_reach`] describe an operator's *height
//! kernel* in isolation. A real layer carries more coupling than its kind: the
//! parameter bindings that sample the whole field into a scalar, the masks that
//! gate its blend, and the auxiliary fields it publishes. [`effective_reach`]
//! folds all of that into the one question phase-2 sub-region recompute (#100)
//! asks — can this configured layer confine a localized edit, and if so by how
//! much?
//!
//! Correctness bias: a [`Reach::Full`] answer is never wrong, only unoptimised,
//! so every rule here errs toward `Full` when it cannot *prove* a bound. The
//! phase-2 equivalence oracle (partial recompute must equal a whole-field
//! rebuild) is the backstop that would catch an over-optimistic `Localized`.

use crate::invalidation::{AuxReach, Reach};
use crate::layer::Layer;
use crate::mask_ir::{
    DistNode, DistNodeKind, Distribution, MaskAsset, MaskOp, MaskRef, MaskSource,
};

/// Effective reach of `layer` given the project's mask assets.
///
/// Considers the layer in isolation — its kind, parameters, masks, and published
/// aux. It deliberately does *not* consider stack position or group membership:
/// an isolated (non-pass-through) group evaluates whole-field and never calls
/// this (its private-composite cache reuse is a separate, coarser mechanism).
///
/// `mask_assets` resolves the layer's mask references (usually `&ctx.mask_assets`);
/// pass an empty slice when the layer has no mask references.
pub fn effective_reach(layer: &Layer, mask_assets: &[MaskAsset]) -> Reach {
    // Parameter bindings drive `mean_mask`, which subsamples the whole field into
    // a scalar that mutates this layer's params — a global input coupling no
    // per-kernel halo can express.
    if !layer.common.param_bindings.is_empty() {
        return Reach::Full;
    }

    // Globally-derived aux (jump-flood distance, whole-field normalize, basin
    // routing) lives in a flat, un-tiled buffer; a localized recompute would
    // leave it stale. Per-texel aux is patchable and does not force a downgrade.
    if layer.kind.aux_reach() == AuxReach::Global {
        return Reach::Full;
    }

    let intrinsic = layer.kind.intrinsic_reach();
    if intrinsic.is_full() {
        return Reach::Full;
    }

    intrinsic.combine(distribution_reach(&layer.common.masks, mask_assets))
}

/// Reach contributed by a layer's mask [`Distribution`]. Empty distribution =
/// no constraint = [`Reach::LOCAL`].
fn distribution_reach(dist: &Distribution, mask_assets: &[MaskAsset]) -> Reach {
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

/// Per-side sample halo of one distribution node kind. Exhaustive: a new node
/// kind must declare its reach here.
fn dist_node_reach(kind: &DistNodeKind, mask_assets: &[MaskAsset]) -> Reach {
    use DistNodeKind::*;
    match kind {
        // Per-texel: constant, procedural noise, per-pixel height/flow/climate
        // selectors, UV shapes, and the per-pixel remap effects.
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

        // Gradient-derived (first-difference stencil) — one sample.
        Slope { .. }
        | Curvature { .. }
        | Cavity { .. }
        | Steepness { .. }
        | Angle { .. }
        | Rocks { .. }
        | RockyEdges { .. }
        | EffectEdge { .. } => Reach::Localized { halo_samples: 1 },

        // Explicit sample-radius neighbourhood nodes.
        Occlusion { radius, .. } | Roughness { radius, .. } | EffectBlur { radius } => {
            Reach::Localized {
                halo_samples: *radius,
            }
        }

        // Iterative smear — one sample per iteration.
        EffectSimpleFlow { iterations, .. } => Reach::Localized {
            halo_samples: *iterations,
        },

        // Whole-field at this granularity: a domain-warp sample offset is
        // unbounded, a distance field is global, and the morphological
        // expand/contract radii are world-space metres this sample-space walk
        // can't convert without the field metrics — so force whole-field.
        EffectDistortion { .. } | Distance { .. } | EffectDilate { .. } | EffectErode { .. } => {
            Reach::Full
        }

        // Asset references resolve to the referenced mask's own reach.
        MaskAsset { mask } | Paint { mask } | ImportedMask { mask } => {
            mask_ref_reach(mask, mask_assets)
        }
    }
}

/// Reach of a referenced mask asset: its source reach folded with its op stack.
/// An unresolved reference is treated as [`Reach::Full`] — we can't prove it local.
fn mask_ref_reach(mask: &MaskRef, mask_assets: &[MaskAsset]) -> Reach {
    let Some(asset) = mask_assets.iter().find(|a| a.id == mask.id) else {
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

/// Per-side sample halo of a mask source. Exhaustive.
fn mask_source_reach(source: &MaskSource) -> Reach {
    use MaskSource::*;
    match source {
        // Input-independent or per-texel data.
        None | Constant(_) | Noise { .. } | Painted { .. } | Height { .. } => Reach::LOCAL,

        // Gradient-derived selectors — one-sample stencil.
        Slope { .. } | Aspect { .. } | Curvature { .. } | Convexity | Concavity => {
            Reach::Localized { halo_samples: 1 }
        }

        // Explicit neighbourhood radius.
        AmbientOcclusion { radius, .. } => Reach::Localized {
            halo_samples: *radius,
        },

        // Global distance transform.
        DistanceField { .. } => Reach::Full,

        // Aux-read selectors contribute no stencil of their own; the consistency
        // of the aux they read is guaranteed by the producing layer's
        // `AuxReach::Global` forcing that producer whole-field.
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

/// Per-side sample halo added by one mask op. Only `Blur` widens; the rest are
/// per-pixel value remaps.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter_params::{EffectFilterKind, EffectFilterParams};
    use crate::layer::{BindingSource, Layer, LayerKind, ParamBinding};
    use crate::mask_ir::{DistNode, DistNodeKind, Distribution, MaskId, MaskRef};

    fn layer(kind: LayerKind) -> Layer {
        Layer::new("test", kind)
    }

    #[test]
    fn plain_generator_is_local() {
        let l = layer(LayerKind::VoronoiRegions(Default::default()));
        assert_eq!(effective_reach(&l, &[]), Reach::LOCAL);
    }

    #[test]
    fn basin_kind_is_full() {
        let l = layer(LayerKind::ThermalErosion(Default::default()));
        assert_eq!(effective_reach(&l, &[]), Reach::Full);
    }

    #[test]
    fn param_bindings_force_full() {
        let mut l = layer(LayerKind::VoronoiRegions(Default::default()));
        l.common
            .param_bindings
            .push(ParamBinding::new("frequency", BindingSource::Constant(0.5)));
        assert_eq!(effective_reach(&l, &[]), Reach::Full);
    }

    #[test]
    fn global_aux_producer_forces_full_despite_local_height_kernel() {
        // Island's height kernel is Local, but it publishes jump-flood aux.
        let l = layer(LayerKind::Island(Default::default()));
        assert_eq!(
            l.kind.spatial_dependency(),
            crate::invalidation::DirtyClass::Local
        );
        assert_eq!(effective_reach(&l, &[]), Reach::Full);
    }

    #[test]
    fn bounded_effect_filter_keeps_a_halo() {
        let p = EffectFilterParams {
            kind: EffectFilterKind::Smooth,
            radius: 3,
            iterations: 2,
            ..EffectFilterParams::default()
        };
        let l = layer(LayerKind::EffectFilter(p));
        assert_eq!(
            effective_reach(&l, &[]),
            Reach::Localized { halo_samples: 6 }
        );
    }

    #[test]
    fn per_texel_aux_stays_localizable() {
        // SculptStrokes publishes only per-texel stamp aux, so it keeps its
        // one-sample reconcile halo.
        let l = layer(LayerKind::SculptStrokes(Default::default()));
        assert_eq!(
            effective_reach(&l, &[]),
            Reach::Localized { halo_samples: 1 }
        );
    }

    #[test]
    fn smooth_stroke_with_reconcile_reaches_two_samples() {
        // A Smooth stroke reads a 3x3 of the layer input; with reconcile on, the
        // stamped result is re-read at 3x3, so the effective reach is two samples.
        use crate::authoring::{SculptStroke, SculptStrokeKind, SculptStrokeParams};
        let l = layer(LayerKind::SculptStrokes(SculptStrokeParams {
            strokes: vec![SculptStroke {
                kind: SculptStrokeKind::Smooth,
                ..SculptStroke::default()
            }],
            reconcile: 0.15,
        }));
        assert_eq!(
            effective_reach(&l, &[]),
            Reach::Localized { halo_samples: 2 }
        );
    }

    #[test]
    fn slope_mask_contributes_one_sample_halo() {
        let mut l = layer(LayerKind::VoronoiRegions(Default::default()));
        l.common.masks = Distribution {
            entries: Vec::new(),
            nodes: vec![DistNode::new(DistNodeKind::Slope {
                min_deg: 10.0,
                max_deg: 40.0,
            })],
        };
        assert_eq!(
            effective_reach(&l, &[]),
            Reach::Localized { halo_samples: 1 }
        );
    }

    #[test]
    fn distance_field_node_forces_full() {
        let mut l = layer(LayerKind::VoronoiRegions(Default::default()));
        l.common.masks = Distribution {
            entries: Vec::new(),
            nodes: vec![DistNode::new(DistNodeKind::Distance {
                mask: MaskRef::new(MaskId::new()),
                max_distance: 32.0,
            })],
        };
        assert_eq!(effective_reach(&l, &[]), Reach::Full);
    }

    #[test]
    fn blur_node_widens_halo_by_its_radius() {
        let mut l = layer(LayerKind::VoronoiRegions(Default::default()));
        l.common.masks = Distribution {
            entries: Vec::new(),
            nodes: vec![DistNode::new(DistNodeKind::EffectBlur { radius: 5 })],
        };
        assert_eq!(
            effective_reach(&l, &[]),
            Reach::Localized { halo_samples: 5 }
        );
    }

    #[test]
    fn unresolved_mask_asset_reference_is_full() {
        let mut l = layer(LayerKind::VoronoiRegions(Default::default()));
        l.common.masks = Distribution {
            entries: Vec::new(),
            nodes: vec![DistNode::new(DistNodeKind::MaskAsset {
                mask: MaskRef::new(MaskId::new()),
            })],
        };
        // No assets provided -> reference cannot be proven local.
        assert_eq!(effective_reach(&l, &[]), Reach::Full);
    }
}
