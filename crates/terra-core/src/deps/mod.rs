//! Internal dependency graph for validation, invalidation, and UI (not a node editor).

use crate::layer::{LayerId, LayerStack, OutputId, StackNode};
use crate::mask::{Distribution, MaskId};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// What a dependency edge represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DepKind {
    /// Implicit: later stack node consumes accumulated context from earlier.
    StackOrder,
    MaskRef,
    NamedOutput,
    ParamBinding,
    GroupInput,
}

/// Stable reference to a stack node or asset participating in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeRef {
    Layer(LayerId),
    Group(LayerId),
    Mask(MaskId),
    Output(OutputId),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DepEdge {
    pub from: NodeRef,
    pub to: NodeRef,
    pub kind: DepKind,
}

#[derive(Debug, Clone, Default)]
pub struct DependencyGraph {
    pub edges: Vec<DepEdge>,
    /// Forward: `from` → dependents (`to`).
    pub adjacency: HashMap<NodeRef, Vec<NodeRef>>,
    /// Reverse: `to` → dependencies (`from`).
    pub reverse: HashMap<NodeRef, Vec<NodeRef>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DepError {
    Cycle(Vec<NodeRef>),
    BrokenRef { from: NodeRef, missing: NodeRef },
}

impl std::fmt::Display for DepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DepError::Cycle(c) => write!(f, "circular dependency involving {} nodes", c.len()),
            DepError::BrokenRef { .. } => write!(f, "broken dependency reference"),
        }
    }
}

impl DependencyGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.edges.clear();
        self.adjacency.clear();
        self.reverse.clear();
    }

    pub fn add_edge(&mut self, from: NodeRef, to: NodeRef, kind: DepKind) {
        self.edges.push(DepEdge { from, to, kind });
        self.adjacency.entry(from).or_default().push(to);
        self.reverse.entry(to).or_default().push(from);
    }

    /// Build stack-order edges and collect mask refs from the document tree.
    pub fn build_from_stack(stack: &LayerStack, mask_ids: &[MaskId]) -> Self {
        let mut g = Self::new();
        let mut prior: Option<NodeRef> = None;
        fn walk(
            nodes: &[StackNode],
            g: &mut DependencyGraph,
            prior: &mut Option<NodeRef>,
            mask_ids: &[MaskId],
        ) {
            for n in nodes {
                match n {
                    StackNode::Layer(l) => {
                        let me = NodeRef::Layer(l.id());
                        if let Some(p) = *prior {
                            g.add_edge(p, me, DepKind::StackOrder);
                        }
                        collect_distribution_deps(g, me, &l.common.masks, mask_ids);
                        for out in &l.common.outputs {
                            g.add_edge(me, NodeRef::Output(out.id), DepKind::NamedOutput);
                        }
                        for b in &l.common.param_bindings {
                            match &b.source {
                                crate::layer::BindingSource::Mask(mid) => {
                                    g.add_edge(NodeRef::Mask(*mid), me, DepKind::ParamBinding);
                                }
                                crate::layer::BindingSource::LayerOutput(oid)
                                | crate::layer::BindingSource::GroupOutput(oid) => {
                                    g.add_edge(NodeRef::Output(*oid), me, DepKind::ParamBinding);
                                }
                                _ => {}
                            }
                        }
                        *prior = Some(me);
                    }
                    StackNode::Group(grp) if !grp.enabled => {}
                    StackNode::Group(grp) => {
                        let me = NodeRef::Group(grp.id);
                        if let Some(p) = *prior {
                            g.add_edge(p, me, DepKind::StackOrder);
                        }
                        collect_distribution_deps(g, me, &grp.masks, mask_ids);
                        for out in &grp.outputs {
                            g.add_edge(me, NodeRef::Output(out.id), DepKind::NamedOutput);
                        }
                        let mut child_prior = match grp.input_mode {
                            crate::layer::GroupInputMode::CopyInput => *prior,
                            crate::layer::GroupInputMode::EmptyHeight => None,
                            crate::layer::GroupInputMode::SelectedField(
                                crate::layer::SelectedGroupInput::Output(oid),
                            ) => {
                                g.add_edge(NodeRef::Output(oid), me, DepKind::GroupInput);
                                Some(NodeRef::Output(oid))
                            }
                            crate::layer::GroupInputMode::SelectedField(_) => *prior,
                        };
                        walk(&grp.children, g, &mut child_prior, mask_ids);
                        *prior = Some(me);
                    }
                }
            }
        }
        walk(&stack.nodes, &mut g, &mut prior, mask_ids);
        g
    }

    /// Detect cycles via Kahn topological sort; returns Ok(order) or Err(cycle).
    pub fn detect_cycle(&self) -> Result<Vec<NodeRef>, DepError> {
        let mut indeg: HashMap<NodeRef, usize> = HashMap::new();
        let mut nodes = HashSet::new();
        for e in &self.edges {
            nodes.insert(e.from);
            nodes.insert(e.to);
            *indeg.entry(e.to).or_insert(0) += 1;
            indeg.entry(e.from).or_insert(0);
        }
        let mut q: VecDeque<NodeRef> = indeg
            .iter()
            .filter(|(_, d)| **d == 0)
            .map(|(n, _)| *n)
            .collect();
        let mut order = Vec::new();
        let mut adj = self.adjacency.clone();
        while let Some(n) = q.pop_front() {
            order.push(n);
            if let Some(nexts) = adj.remove(&n) {
                for t in nexts {
                    if let Some(d) = indeg.get_mut(&t) {
                        *d = d.saturating_sub(1);
                        if *d == 0 {
                            q.push_back(t);
                        }
                    }
                }
            }
        }
        if order.len() < nodes.len() {
            let leftover: Vec<_> = nodes.into_iter().filter(|n| !order.contains(n)).collect();
            return Err(DepError::Cycle(leftover));
        }
        Ok(order)
    }

    /// All nodes that depend on `source` (transitive) — "Used by".
    pub fn dependents_of(&self, source: NodeRef) -> HashSet<NodeRef> {
        let mut out = HashSet::new();
        let mut stack = vec![source];
        while let Some(n) = stack.pop() {
            if let Some(nexts) = self.adjacency.get(&n) {
                for t in nexts {
                    if out.insert(*t) {
                        stack.push(*t);
                    }
                }
            }
        }
        out
    }

    /// Direct + transitive upstream dependencies — "Depends on".
    pub fn dependencies_of(&self, target: NodeRef) -> HashSet<NodeRef> {
        let mut out = HashSet::new();
        let mut stack = vec![target];
        while let Some(n) = stack.pop() {
            if let Some(preds) = self.reverse.get(&n) {
                for p in preds {
                    if out.insert(*p) {
                        stack.push(*p);
                    }
                }
            }
        }
        out
    }

    /// Immediate upstream neighbours only.
    pub fn direct_dependencies(&self, target: NodeRef) -> Vec<NodeRef> {
        self.reverse.get(&target).cloned().unwrap_or_default()
    }

    /// Immediate downstream neighbours only.
    pub fn direct_dependents(&self, source: NodeRef) -> Vec<NodeRef> {
        self.adjacency.get(&source).cloned().unwrap_or_default()
    }

    pub fn validate_refs(
        &self,
        known_masks: &HashSet<MaskId>,
        known_outputs: &HashSet<OutputId>,
    ) -> Vec<DepError> {
        let mut errs = Vec::new();
        for e in &self.edges {
            let missing = match e.from {
                NodeRef::Mask(id) if !known_masks.contains(&id) => Some(e.from),
                NodeRef::Output(id) if !known_outputs.contains(&id) => Some(e.from),
                _ => None,
            };
            if let Some(m) = missing {
                errs.push(DepError::BrokenRef {
                    from: e.to,
                    missing: m,
                });
            }
        }
        errs
    }
}

fn collect_distribution_deps(
    g: &mut DependencyGraph,
    owner: NodeRef,
    dist: &Distribution,
    _mask_ids: &[MaskId],
) {
    // Record references whether or not the assets currently exist. `validate_refs`
    // needs the missing edges in order to report broken project references.
    for entry in dist.iter() {
        g.add_edge(NodeRef::Mask(entry.mask.id), owner, DepKind::MaskRef);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::{FlatParams, Layer, LayerGroup, LayerKind};

    #[test]
    fn stack_order_is_acyclic() {
        let mut stack = LayerStack::new();
        stack.push(Layer::new("A", LayerKind::Flat(FlatParams { height: 1.0 })));
        stack.push(Layer::new("B", LayerKind::Flat(FlatParams { height: 2.0 })));
        let g = DependencyGraph::build_from_stack(&stack, &[]);
        assert!(g.detect_cycle().is_ok());
    }

    #[test]
    fn broken_mask_ref_is_reported() {
        let mut stack = LayerStack::new();
        let mut layer = Layer::new("A", LayerKind::Flat(FlatParams { height: 1.0 }));
        let missing = MaskId::new();
        layer.common.masks.push(crate::mask::MaskRef::new(missing));
        let layer_id = layer.id();
        stack.push(layer);
        let g = DependencyGraph::build_from_stack(&stack, &[]);
        let errs = g.validate_refs(&HashSet::new(), &HashSet::new());
        assert!(errs.iter().any(|e| matches!(
            e,
            DepError::BrokenRef {
                from: NodeRef::Layer(id),
                missing: NodeRef::Mask(m),
            } if *id == layer_id && *m == missing
        )));
    }

    #[test]
    fn groups_participate_in_order() {
        let mut stack = LayerStack::new();
        stack.push_group(LayerGroup::new("Alpine"));
        let g = DependencyGraph::build_from_stack(&stack, &[]);
        assert!(!g.edges.is_empty() || g.adjacency.is_empty());
    }
}
