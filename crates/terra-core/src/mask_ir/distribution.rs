//! Persisted nested distribution stacks.

use super::DistNode;
use crate::mask_types::MaskRef;
use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
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
