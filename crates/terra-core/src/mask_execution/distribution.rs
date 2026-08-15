//! Distribution evaluation against resolved mask fields.

use super::dist_nodes::{bake_dist_nodes, DistBakeContext};
use crate::heightfield::HeightfieldMetrics;
use crate::mask_field::MaskField;
use crate::mask_ir::Distribution;
use crate::mask_types::MaskId;
use std::collections::HashMap;

/// Bake a distribution against already-resolved mask fields (legacy entries only).
pub fn bake_distribution(
    dist: &Distribution,
    masks: &HashMap<MaskId, MaskField>,
    metrics: HeightfieldMetrics,
) -> MaskField {
    bake_distribution_with_context(dist, metrics, &DistBakeContext::masks_only(masks))
}

/// Bake distribution: DistNodes first (when present), then legacy mask entries.
pub fn bake_distribution_with_context(
    dist: &Distribution,
    metrics: HeightfieldMetrics,
    ctx: &DistBakeContext<'_>,
) -> MaskField {
    if dist.is_empty() {
        return MaskField::ones(metrics);
    }

    let mut acc = if dist.nodes.is_empty() {
        MaskField::ones(metrics)
    } else {
        bake_dist_nodes(&dist.nodes, metrics, ctx)
    };

    for entry in &dist.entries {
        let field = ctx
            .masks
            .get(&entry.mask.id)
            .map(|field| field.resampled_nearest(metrics))
            .unwrap_or_else(|| MaskField::ones(metrics));
        for j in 0..metrics.height {
            for i in 0..metrics.width {
                let mut v = field.get(i, j) * entry.mask.strength;
                if entry.mask.invert {
                    v = 1.0 - v;
                }
                let combined = entry.combine.apply(acc.get(i, j), v);
                acc.set(i, j, combined);
            }
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heightfield::HeightfieldMetrics;
    use crate::mask_ir::{DistNode, DistNodeKind, DistributionEntry, MaskCombine};
    use crate::mask_types::MaskRef;

    #[test]
    fn legacy_vec_deserializes() {
        let json =
            r#"[{"id":"00000000-0000-0000-0000-000000000001","strength":0.5,"invert":true}]"#;
        let dist: Distribution = serde_json::from_str(json).unwrap();
        assert_eq!(dist.entries.len(), 1);
        assert!((dist.entries[0].mask.strength - 0.5).abs() < 1e-6);
        assert!(dist.entries[0].mask.invert);
        assert!(matches!(dist.entries[0].combine, MaskCombine::Multiply));
    }

    #[test]
    fn nested_entries_roundtrip() {
        let mut dist = Distribution::new();
        dist.push(MaskRef::new(MaskId::new()));
        dist.entries[0].combine = MaskCombine::Max;
        let json = serde_json::to_string(&dist).unwrap();
        let back: Distribution = serde_json::from_str(&json).unwrap();
        assert_eq!(back.entries.len(), 1);
        assert!(matches!(back.entries[0].combine, MaskCombine::Max));
    }

    #[test]
    fn nodes_roundtrip() {
        let mut dist = Distribution::new();
        let mut slope = DistNode::slope(10.0, 40.0);
        slope
            .children
            .push(DistNode::new(DistNodeKind::EffectBlur { radius: 2 }));
        dist.push_node(slope);
        dist.push_node(DistNode::height(0.0, 500.0));
        let json = serde_json::to_string(&dist).unwrap();
        let back: Distribution = serde_json::from_str(&json).unwrap();
        assert_eq!(back.nodes.len(), 2);
        assert_eq!(back.nodes[0].children.len(), 1);
    }

    #[test]
    fn bake_multiply_then_max() {
        let metrics = HeightfieldMetrics::new(4, 4, 100.0, 100.0);
        let id_a = MaskId::new();
        let id_b = MaskId::new();
        let mut masks = HashMap::new();
        masks.insert(id_a, MaskField::filled(metrics, 0.5));
        masks.insert(id_b, MaskField::filled(metrics, 0.8));
        let dist = Distribution {
            entries: vec![
                DistributionEntry::new(MaskRef::new(id_a)),
                DistributionEntry {
                    mask: MaskRef::new(id_b),
                    combine: MaskCombine::Max,
                },
            ],
            nodes: Vec::new(),
        };
        let baked = bake_distribution(&dist, &masks, metrics);
        // ones * 0.5 = 0.5, then max(0.5, 0.8) = 0.8
        assert!((baked.get(0, 0) - 0.8).abs() < 1e-5);
    }

    #[test]
    fn legacy_entry_resamples_referenced_mask_to_target_metrics() {
        let source = HeightfieldMetrics::new(2, 2, 40.0, 40.0);
        let target = HeightfieldMetrics::new(4, 4, 40.0, 40.0);
        let id = MaskId::new();
        let masks = HashMap::from([(id, MaskField::filled(source, 0.375))]);
        let dist = Distribution::from_refs(vec![MaskRef::new(id)]);

        let baked = bake_distribution(&dist, &masks, target);

        assert_eq!(baked.metrics.width, 4);
        assert_eq!(baked.metrics.height, 4);
        assert_eq!(baked.get(3, 3), 0.375);
    }
}
