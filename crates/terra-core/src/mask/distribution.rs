//! Nested distribution stacks (WC-style mask composition).

use super::dist_nodes::{bake_dist_nodes, DistBakeContext, DistNode};
use super::{MaskField, MaskId, MaskRef};
use crate::heightfield::HeightfieldMetrics;
use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// How a distribution entry combines with the accumulator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MaskCombine {
    #[default]
    Multiply,
    Add,
    Subtract,
    Min,
    Max,
    Replace,
    Invert,
    /// Prefer `b` wherever it is painted; otherwise keep `a` (rules outside paint).
    PaintOverride,
}

impl MaskCombine {
    pub fn apply(self, a: f32, b: f32) -> f32 {
        match self {
            MaskCombine::Multiply => (a * b).clamp(0.0, 1.0),
            MaskCombine::Add => (a + b).clamp(0.0, 1.0),
            MaskCombine::Subtract => (a - b).clamp(0.0, 1.0),
            MaskCombine::Min => a.min(b),
            MaskCombine::Max => a.max(b),
            MaskCombine::Replace => b.clamp(0.0, 1.0),
            MaskCombine::Invert => (1.0 - a).clamp(0.0, 1.0),
            MaskCombine::PaintOverride => {
                if b > 1e-4 {
                    b.clamp(0.0, 1.0)
                } else {
                    a.clamp(0.0, 1.0)
                }
            }
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            MaskCombine::Multiply => "Multiply",
            MaskCombine::Add => "Add",
            MaskCombine::Subtract => "Subtract",
            MaskCombine::Min => "Minimum",
            MaskCombine::Max => "Maximum",
            MaskCombine::Replace => "Replace",
            MaskCombine::Invert => "Invert",
            MaskCombine::PaintOverride => "Paint Override",
        }
    }

    pub fn cycle(self) -> Self {
        match self {
            MaskCombine::Multiply => MaskCombine::Add,
            MaskCombine::Add => MaskCombine::Subtract,
            MaskCombine::Subtract => MaskCombine::Min,
            MaskCombine::Min => MaskCombine::Max,
            MaskCombine::Max => MaskCombine::Replace,
            MaskCombine::Replace => MaskCombine::Invert,
            MaskCombine::Invert => MaskCombine::PaintOverride,
            MaskCombine::PaintOverride => MaskCombine::Multiply,
        }
    }
}

/// One mask binding inside a [`Distribution`] (legacy path).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionEntry {
    pub mask: MaskRef,
    #[serde(default)]
    pub combine: MaskCombine,
}

impl DistributionEntry {
    pub fn new(mask: MaskRef) -> Self {
        Self {
            mask,
            combine: MaskCombine::Multiply,
        }
    }
}

/// Ordered mask stack used for layer/group scoping and materials/veg coverage.
///
/// Supports:
/// - Legacy `Vec<MaskRef>` / `{ "entries": [...] }` mask asset refs
/// - WC DistNode stack via `{ "nodes": [...] }` (Stage B)
#[derive(Debug, Clone, Default, Serialize)]
pub struct Distribution {
    pub entries: Vec<DistributionEntry>,
    /// Procedural / terrain-feature distribution nodes (evaluated when non-empty).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nodes: Vec<DistNode>,
}

impl Distribution {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_refs(refs: Vec<MaskRef>) -> Self {
        Self {
            entries: refs.into_iter().map(DistributionEntry::new).collect(),
            nodes: Vec::new(),
        }
    }

    pub fn from_nodes(nodes: Vec<DistNode>) -> Self {
        Self {
            entries: Vec::new(),
            nodes,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.nodes.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len() + self.nodes.len()
    }

    pub fn first(&self) -> Option<&DistributionEntry> {
        self.entries.first()
    }

    pub fn push(&mut self, mask: MaskRef) {
        self.entries.push(DistributionEntry::new(mask));
    }

    pub fn push_node(&mut self, node: DistNode) {
        self.nodes.push(node);
    }

    pub fn retain<F: FnMut(&DistributionEntry) -> bool>(&mut self, mut f: F) {
        self.entries.retain(|e| f(e));
    }

    pub fn iter(&self) -> impl Iterator<Item = &DistributionEntry> {
        self.entries.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut DistributionEntry> {
        self.entries.iter_mut()
    }
}

impl<'de> Deserialize<'de> for Distribution {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct DistVisitor;

        impl<'de> Visitor<'de> for DistVisitor {
            type Value = Distribution;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a Distribution object or a legacy MaskRef array")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut refs = Vec::new();
                while let Some(m) = seq.next_element::<MaskRef>()? {
                    refs.push(m);
                }
                Ok(Distribution::from_refs(refs))
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut entries: Option<Vec<DistributionEntry>> = None;
                let mut nodes: Option<Vec<DistNode>> = None;
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "entries" => {
                            entries = Some(map.next_value()?);
                        }
                        "nodes" => {
                            nodes = Some(map.next_value()?);
                        }
                        _ => {
                            let _: de::IgnoredAny = map.next_value()?;
                        }
                    }
                }
                Ok(Distribution {
                    entries: entries.unwrap_or_default(),
                    nodes: nodes.unwrap_or_default(),
                })
            }
        }

        deserializer.deserialize_any(DistVisitor)
    }
}

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
    use super::super::dist_nodes::DistNodeKind;
    use super::*;
    use crate::heightfield::HeightfieldMetrics;

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
