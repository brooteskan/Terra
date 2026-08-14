//! Artist-facing PlacementDefinition — compiles into existing DistNode / Distribution stacks.
//!
//! Authoring IR only. Coverage bake remains [`super::bake_dist_nodes`] /
//! [`super::bake_distribution_with_context`].

use super::dist_nodes::{ClimateMaskChannel, DistBakeContext, DistNode, DistNodeId, DistNodeKind};
use super::distribution::{
    bake_distribution_with_context, Distribution, DistributionEntry, MaskCombine,
};
use super::{MaskId, MaskRef};
use crate::heightfield::HeightfieldMetrics;
use crate::ids::LayerId;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use uuid::Uuid;

/// Maximum nesting depth for rule groups (root = 0).
pub const MAX_RULE_NEST_DEPTH: u8 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PlacementCoordinateSpace {
    /// Painted / imported coverage in world meters — stable when height changes.
    WorldSpace,
    /// Surface-locked UV paint — follows terrain where supported.
    SurfaceSpace,
    /// Procedural rules that reevaluate as terrain / climate change.
    #[default]
    RuleBased,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PlacementSource {
    #[default]
    Rules,
    /// Artist edited the compiled DistNode stack — do not auto-recompile.
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RuleGroupMode {
    #[default]
    All,
    Any,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CompareOp {
    #[default]
    Above,
    Below,
    Between,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConditionChannel {
    Height,
    Slope,
    Curvature,
    Flow,
    WaterDistance,
    CoastDistance,
    Temperature,
    Rainfall,
    WindExposure,
    Noise,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Condition {
    pub channel: ConditionChannel,
    pub op: CompareOp,
    pub a: f32,
    #[serde(default)]
    pub b: f32,
    #[serde(default)]
    pub falloff: f32,
    #[serde(default)]
    pub seed: u64,
    #[serde(default = "default_noise_freq")]
    pub frequency: f32,
}

fn default_noise_freq() -> f32 {
    0.02
}

impl ConditionChannel {
    pub fn label(self) -> &'static str {
        match self {
            Self::Height => "Height",
            Self::Slope => "Slope",
            Self::Curvature => "Curvature",
            Self::Flow => "Flow",
            Self::WaterDistance => "Distance to water",
            Self::CoastDistance => "Distance to coast",
            Self::Temperature => "Temperature",
            Self::Rainfall => "Rainfall",
            Self::WindExposure => "Wind exposure",
            Self::Noise => "Noise",
        }
    }
}

impl CompareOp {
    pub fn label(self) -> &'static str {
        match self {
            Self::Above => "above",
            Self::Below => "below",
            Self::Between => "between",
        }
    }
}

impl Condition {
    /// Artist-readable Where phrase (e.g. "Height above 1200").
    pub fn where_phrase(&self) -> String {
        match self.op {
            CompareOp::Between => format!(
                "{} between {:.0}–{:.0}",
                self.channel.label(),
                self.a.min(self.b),
                self.a.max(self.b)
            ),
            _ => format!("{} {} {:.0}", self.channel.label(), self.op.label(), self.a),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CoverageTerm {
    PaintedWorld { mask: MaskId },
    PaintedSurface { mask: MaskId },
    Polygon { points: Vec<[f32; 2]>, soft: f32 },
    Spline { points: Vec<[f32; 2]>, width: f32 },
    ImportedMask { mask: MaskId },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RuleNode {
    Group(RuleGroup),
    Condition(Condition),
    CoverageRef { index: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleGroup {
    pub mode: RuleGroupMode,
    #[serde(default)]
    pub invert: bool,
    #[serde(default)]
    pub children: Vec<RuleNode>,
}

impl Default for RuleGroup {
    fn default() -> Self {
        Self {
            mode: RuleGroupMode::All,
            invert: false,
            children: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PlacementRefinement {
    ExcludeBiome {
        biome_group: LayerId,
        mask: Option<MaskId>,
    },
    ExcludeRiverCorridor {
        flow_min: f32,
    },
    ExcludeRoads {
        mask: MaskId,
    },
    Expand {
        radius_m: f32,
    },
    Contract {
        radius_m: f32,
    },
    Smooth {
        radius_samples: u32,
    },
    BoundaryNoise {
        seed: u64,
        frequency: f32,
        amount: f32,
    },
    Falloff {
        edge0: f32,
        edge1: f32,
    },
}

/// Artist placement IR. Compiles into [`Distribution`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementDefinition {
    #[serde(default)]
    pub space: PlacementCoordinateSpace,
    #[serde(default)]
    pub source: PlacementSource,
    #[serde(default)]
    pub coverage: Vec<CoverageTerm>,
    #[serde(default)]
    pub root: RuleGroup,
    #[serde(default)]
    pub refinements: Vec<PlacementRefinement>,
    #[serde(default)]
    pub paint_combine: crate::biome_definition::PlacementCombineMode,
    #[serde(default)]
    pub paint_mask: Option<MaskId>,
    #[serde(default)]
    pub content_hash: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_stack: Option<Distribution>,
}

impl Default for PlacementDefinition {
    fn default() -> Self {
        let mut p = Self {
            space: PlacementCoordinateSpace::RuleBased,
            source: PlacementSource::Rules,
            coverage: Vec::new(),
            root: RuleGroup::default(),
            refinements: Vec::new(),
            paint_combine: crate::biome_definition::PlacementCombineMode::default(),
            paint_mask: None,
            content_hash: 0,
            custom_stack: None,
        };
        p.recompute_hash();
        p
    }
}

impl PlacementDefinition {
    pub fn recompute_hash(&mut self) {
        self.content_hash = hash_placement_rules(self);
    }

    pub fn mark_custom(&mut self, stack: Distribution) {
        self.source = PlacementSource::Custom;
        self.custom_stack = Some(stack);
    }

    pub fn reset_to_rules(&mut self) -> Distribution {
        self.source = PlacementSource::Rules;
        self.custom_stack = None;
        self.recompute_hash();
        self.compile()
    }

    pub fn active_distribution(&self) -> Distribution {
        match self.source {
            PlacementSource::Custom => self.custom_stack.clone().unwrap_or_else(|| self.compile()),
            PlacementSource::Rules => self.compile(),
        }
    }

    pub fn compile(&self) -> Distribution {
        let mut ctx = CompileCtx {
            seed: (self.content_hash as u128) | 0x504C_4143_4500_0000,
            ordinal: 0,
            depth: 0,
            coverage: &self.coverage,
        };
        let mut nodes = Vec::new();

        if !self.root.children.is_empty() || self.root.invert {
            nodes.push(ctx.compile_group(&self.root));
        }

        for (i, term) in self.coverage.iter().enumerate() {
            if !rule_references_coverage(&self.root, i) {
                let mut n = ctx.coverage_node(term);
                n.combine = MaskCombine::Multiply;
                nodes.push(n);
            }
        }

        for r in &self.refinements {
            nodes.extend(ctx.compile_refinement(r));
        }

        let mut dist = Distribution::from_nodes(nodes);
        if let Some(mid) = self.paint_mask {
            if self.paint_combine.uses_manual_paint() {
                dist.entries.push(DistributionEntry {
                    mask: MaskRef::new(mid),
                    combine: self.paint_combine.mask_combine(),
                });
            }
        }
        dist
    }

    pub fn try_decompile(dist: &Distribution) -> Option<Self> {
        if dist.nodes.is_empty() && dist.entries.is_empty() {
            return Some(Self::default());
        }

        let mut coverage = Vec::new();
        let mut refinements = Vec::new();
        let mut root = RuleGroup::default();
        let mut paint_mask = None;
        let mut paint_combine = crate::biome_definition::PlacementCombineMode::default();

        if let Some(entry) = dist.entries.first() {
            paint_mask = Some(entry.mask.id);
            paint_combine = match entry.combine {
                MaskCombine::Multiply => {
                    crate::biome_definition::PlacementCombineMode::PaintMulRules
                }
                MaskCombine::Add => crate::biome_definition::PlacementCombineMode::PaintAddRules,
                MaskCombine::Replace => crate::biome_definition::PlacementCombineMode::PaintOnly,
                MaskCombine::PaintOverride => {
                    crate::biome_definition::PlacementCombineMode::PaintOverridesRules
                }
                _ => crate::biome_definition::PlacementCombineMode::PaintMulRules,
            };
        }

        if dist.nodes.len() == 1
            && matches!(
                dist.nodes[0].kind,
                DistNodeKind::GroupAll | DistNodeKind::GroupAny
            )
        {
            root = decompile_group(&dist.nodes[0], &mut coverage)?;
        } else {
            root.mode = RuleGroupMode::All;
            for n in &dist.nodes {
                if let Some(node) = decompile_node(n, &mut coverage, &mut refinements) {
                    root.children.push(node);
                } else {
                    match &n.kind {
                        DistNodeKind::EffectBlur { radius } => {
                            refinements.push(PlacementRefinement::Smooth {
                                radius_samples: *radius,
                            });
                        }
                        DistNodeKind::EffectDilate { radius_m } => {
                            refinements.push(PlacementRefinement::Expand {
                                radius_m: *radius_m,
                            });
                        }
                        DistNodeKind::EffectErode { radius_m } => {
                            refinements.push(PlacementRefinement::Contract {
                                radius_m: *radius_m,
                            });
                        }
                        _ => return None,
                    }
                }
            }
        }

        let mut p = Self {
            space: PlacementCoordinateSpace::RuleBased,
            source: PlacementSource::Rules,
            coverage,
            root,
            refinements,
            paint_combine,
            paint_mask,
            content_hash: 0,
            custom_stack: None,
        };
        p.recompute_hash();
        Some(p)
    }

    pub fn from_legacy_dist_node(node: DistNode) -> Self {
        let mut p = Self::default();
        p.source = PlacementSource::Custom;
        p.custom_stack = Some(Distribution::from_nodes(vec![node]));
        p.recompute_hash();
        p
    }
}

struct CompileCtx<'a> {
    seed: u128,
    ordinal: u32,
    depth: u8,
    coverage: &'a [CoverageTerm],
}

impl<'a> CompileCtx<'a> {
    fn next_id(&mut self) -> DistNodeId {
        let id = DistNodeId(Uuid::from_u128(
            self.seed ^ ((self.ordinal as u128) << 16) ^ 0xD157_4E0D_0001,
        ));
        self.ordinal = self.ordinal.wrapping_add(1);
        id
    }

    fn node(&mut self, kind: DistNodeKind, combine: MaskCombine) -> DistNode {
        DistNode {
            id: self.next_id(),
            enabled: true,
            opacity: 1.0,
            combine,
            kind,
            children: Vec::new(),
        }
    }

    fn compile_group(&mut self, group: &RuleGroup) -> DistNode {
        if group.invert {
            return compile_inverted_group(self, group);
        }
        let depth = self.depth;
        if depth >= MAX_RULE_NEST_DEPTH {
            return self.node(DistNodeKind::Fill { value: 1.0 }, MaskCombine::Multiply);
        }
        self.depth = depth + 1;
        let kind = match group.mode {
            RuleGroupMode::All => DistNodeKind::GroupAll,
            RuleGroupMode::Any => DistNodeKind::GroupAny,
        };
        let mut n = self.node(kind, MaskCombine::Multiply);
        for child in &group.children {
            n.children.push(self.compile_rule_node(child));
        }
        self.depth = depth;
        n
    }

    fn compile_rule_node(&mut self, node: &RuleNode) -> DistNode {
        match node {
            RuleNode::Group(g) => self.compile_group(g),
            RuleNode::Condition(c) => self.compile_condition(c),
            RuleNode::CoverageRef { index } => {
                if let Some(term) = self.coverage.get(*index) {
                    let mut n = self.coverage_node(term);
                    n.combine = MaskCombine::Multiply;
                    n
                } else {
                    self.node(DistNodeKind::Fill { value: 1.0 }, MaskCombine::Multiply)
                }
            }
        }
    }

    fn compile_condition(&mut self, c: &Condition) -> DistNode {
        let (min, max) = match c.op {
            CompareOp::Above => (c.a, f32::MAX),
            CompareOp::Below => (f32::MIN, c.a),
            CompareOp::Between => (c.a.min(c.b), c.a.max(c.b)),
        };
        let falloff = c.falloff.max(0.0);
        let mut node = match c.channel {
            ConditionChannel::Height => {
                let lo = if min == f32::MIN { -1.0e6 } else { min };
                let hi = if max == f32::MAX { 1.0e6 } else { max };
                self.node(
                    DistNodeKind::Height {
                        min: lo - falloff,
                        max: hi + falloff,
                    },
                    MaskCombine::Multiply,
                )
            }
            ConditionChannel::Slope => {
                let lo = if min == f32::MIN { 0.0 } else { min.max(0.0) };
                let hi = if max == f32::MAX {
                    90.0
                } else {
                    max.clamp(0.0, 90.0)
                };
                self.node(
                    DistNodeKind::Slope {
                        min_deg: (lo - falloff).max(0.0),
                        max_deg: (hi + falloff).min(90.0),
                    },
                    MaskCombine::Multiply,
                )
            }
            ConditionChannel::Curvature => self.node(
                DistNodeKind::Curvature {
                    min: if min == f32::MIN { -10.0 } else { min },
                    max: if max == f32::MAX { 10.0 } else { max },
                },
                MaskCombine::Multiply,
            ),
            ConditionChannel::Flow => self.node(
                DistNodeKind::Flow {
                    min: if min == f32::MIN {
                        0.0
                    } else {
                        min.clamp(0.0, 1.0)
                    },
                    max: if max == f32::MAX {
                        1.0
                    } else {
                        max.clamp(0.0, 1.0)
                    },
                },
                MaskCombine::Multiply,
            ),
            ConditionChannel::WaterDistance | ConditionChannel::CoastDistance => {
                let width = match c.op {
                    CompareOp::Between => ((c.b - c.a).abs()).max(1.0),
                    _ => c.a.max(1.0),
                };
                let mut n = self.node(
                    DistNodeKind::SeaLevel { level: 0.0, width },
                    MaskCombine::Multiply,
                );
                if matches!(c.op, CompareOp::Above) {
                    n.children
                        .push(self.node(DistNodeKind::EffectInvert, MaskCombine::Multiply));
                }
                n
            }
            ConditionChannel::Temperature => {
                climate_band(self, ClimateMaskChannel::Temperature, min, max, falloff)
            }
            ConditionChannel::Rainfall => {
                climate_band(self, ClimateMaskChannel::Rainfall, min, max, falloff)
            }
            ConditionChannel::WindExposure => {
                climate_band(self, ClimateMaskChannel::WindExposure, min, max, falloff)
            }
            ConditionChannel::Noise => {
                let mut n = self.node(
                    DistNodeKind::Noise {
                        seed: c.seed,
                        frequency: c.frequency.max(1e-6),
                    },
                    MaskCombine::Multiply,
                );
                let lo = if min == f32::MIN {
                    0.0
                } else {
                    min.clamp(0.0, 1.0)
                };
                let hi = if max == f32::MAX {
                    1.0
                } else {
                    max.clamp(0.0, 1.0)
                };
                n.children.push(self.node(
                    DistNodeKind::EffectRemap {
                        in_min: lo,
                        in_max: hi.max(lo + 1e-5),
                    },
                    MaskCombine::Multiply,
                ));
                n
            }
        };
        if falloff > 1e-5
            && !matches!(
                c.channel,
                ConditionChannel::WaterDistance | ConditionChannel::CoastDistance
            )
        {
            node.children.push(self.node(
                DistNodeKind::EffectSmoothstep {
                    edge0: 0.0,
                    edge1: (falloff / (falloff + 1.0)).clamp(0.05, 0.95),
                },
                MaskCombine::Multiply,
            ));
        }
        node
    }

    fn coverage_node(&mut self, term: &CoverageTerm) -> DistNode {
        match term {
            CoverageTerm::PaintedWorld { mask }
            | CoverageTerm::PaintedSurface { mask }
            | CoverageTerm::ImportedMask { mask } => self.node(
                DistNodeKind::MaskAsset {
                    mask: MaskRef::new(*mask),
                },
                MaskCombine::Multiply,
            ),
            CoverageTerm::Polygon { points, soft } => self.node(
                DistNodeKind::Polygon {
                    points: points.clone(),
                    soft: *soft,
                },
                MaskCombine::Multiply,
            ),
            CoverageTerm::Spline { points, width } => self.node(
                DistNodeKind::Spline {
                    points: points.clone(),
                    width: *width,
                },
                MaskCombine::Multiply,
            ),
        }
    }

    fn compile_refinement(&mut self, r: &PlacementRefinement) -> Vec<DistNode> {
        match r {
            PlacementRefinement::ExcludeBiome { mask, .. } => {
                let mid = mask.unwrap_or_else(MaskId::new);
                vec![self.node(
                    DistNodeKind::MaskAsset {
                        mask: MaskRef::new(mid),
                    },
                    MaskCombine::Subtract,
                )]
            }
            PlacementRefinement::ExcludeRiverCorridor { flow_min } => {
                vec![self.node(
                    DistNodeKind::Flow {
                        min: flow_min.clamp(0.0, 1.0),
                        max: 1.0,
                    },
                    MaskCombine::Subtract,
                )]
            }
            PlacementRefinement::ExcludeRoads { mask } => {
                vec![self.node(
                    DistNodeKind::MaskAsset {
                        mask: MaskRef::new(*mask),
                    },
                    MaskCombine::Subtract,
                )]
            }
            PlacementRefinement::Expand { radius_m } => {
                vec![self.node(
                    DistNodeKind::EffectDilate {
                        radius_m: *radius_m,
                    },
                    MaskCombine::Multiply,
                )]
            }
            PlacementRefinement::Contract { radius_m } => {
                vec![self.node(
                    DistNodeKind::EffectErode {
                        radius_m: *radius_m,
                    },
                    MaskCombine::Multiply,
                )]
            }
            PlacementRefinement::Smooth { radius_samples } => {
                vec![self.node(
                    DistNodeKind::EffectBlur {
                        radius: (*radius_samples).max(1),
                    },
                    MaskCombine::Multiply,
                )]
            }
            PlacementRefinement::BoundaryNoise {
                seed,
                frequency,
                amount,
            } => {
                let mut n = self.node(
                    DistNodeKind::Noise {
                        seed: *seed,
                        frequency: frequency.max(1e-6),
                    },
                    MaskCombine::Multiply,
                );
                n.opacity = amount.clamp(0.0, 1.0);
                vec![n]
            }
            PlacementRefinement::Falloff { edge0, edge1 } => {
                vec![self.node(
                    DistNodeKind::EffectSmoothstep {
                        edge0: *edge0,
                        edge1: *edge1,
                    },
                    MaskCombine::Multiply,
                )]
            }
        }
    }
}

fn climate_band(
    ctx: &mut CompileCtx<'_>,
    channel: ClimateMaskChannel,
    min: f32,
    max: f32,
    falloff: f32,
) -> DistNode {
    let lo = if min == f32::MIN { 0.0 } else { min };
    let hi = if max == f32::MAX { 1.0 } else { max };
    let mut n = ctx.node(DistNodeKind::Climate { channel }, MaskCombine::Multiply);
    n.children.push(ctx.node(
        DistNodeKind::EffectRemap {
            in_min: lo - falloff,
            in_max: (hi + falloff).max(lo - falloff + 1e-5),
        },
        MaskCombine::Multiply,
    ));
    n
}

fn compile_inverted_group(ctx: &mut CompileCtx<'_>, group: &RuleGroup) -> DistNode {
    let flipped = RuleGroup {
        mode: match group.mode {
            RuleGroupMode::All => RuleGroupMode::Any,
            RuleGroupMode::Any => RuleGroupMode::All,
        },
        invert: false,
        children: group.children.iter().map(invert_rule_node).collect(),
    };
    ctx.compile_group(&flipped)
}

fn invert_rule_node(node: &RuleNode) -> RuleNode {
    match node {
        RuleNode::Group(g) => RuleNode::Group(RuleGroup {
            mode: g.mode,
            invert: !g.invert,
            children: g.children.clone(),
        }),
        RuleNode::Condition(c) => RuleNode::Condition(Condition {
            channel: c.channel,
            op: match c.op {
                CompareOp::Above => CompareOp::Below,
                CompareOp::Below => CompareOp::Above,
                CompareOp::Between => CompareOp::Between,
            },
            a: c.a,
            b: c.b,
            falloff: c.falloff,
            seed: c.seed,
            frequency: c.frequency,
        }),
        RuleNode::CoverageRef { index } => RuleNode::Group(RuleGroup {
            mode: RuleGroupMode::All,
            invert: true,
            children: vec![RuleNode::CoverageRef { index: *index }],
        }),
    }
}

fn rule_references_coverage(group: &RuleGroup, index: usize) -> bool {
    for c in &group.children {
        match c {
            RuleNode::CoverageRef { index: i } if *i == index => return true,
            RuleNode::Group(g) if rule_references_coverage(g, index) => return true,
            _ => {}
        }
    }
    false
}

fn hash_placement_rules(p: &PlacementDefinition) -> u64 {
    let mut h = DefaultHasher::new();
    #[derive(Serialize)]
    struct Key<'a> {
        space: PlacementCoordinateSpace,
        coverage: &'a [CoverageTerm],
        root: &'a RuleGroup,
        refinements: &'a [PlacementRefinement],
        paint_combine: crate::biome_definition::PlacementCombineMode,
        paint_mask: Option<MaskId>,
    }
    if let Ok(bytes) = serde_json::to_vec(&Key {
        space: p.space,
        coverage: &p.coverage,
        root: &p.root,
        refinements: &p.refinements,
        paint_combine: p.paint_combine,
        paint_mask: p.paint_mask,
    }) {
        bytes.hash(&mut h);
    }
    h.finish()
}

/// Fraction of samples with coverage > 0.05.
pub fn coverage_estimate(
    placement: &PlacementDefinition,
    metrics: HeightfieldMetrics,
    ctx: &DistBakeContext<'_>,
) -> f32 {
    let dist = placement.active_distribution();
    let field = bake_distribution_with_context(&dist, metrics, ctx);
    let total = (metrics.width * metrics.height).max(1) as f32;
    let mut hit = 0u32;
    for j in 0..metrics.height {
        for i in 0..metrics.width {
            if field.get(i, j) > 0.05 {
                hit += 1;
            }
        }
    }
    hit as f32 / total
}

fn decompile_group(node: &DistNode, coverage: &mut Vec<CoverageTerm>) -> Option<RuleGroup> {
    let mode = match node.kind {
        DistNodeKind::GroupAll => RuleGroupMode::All,
        DistNodeKind::GroupAny => RuleGroupMode::Any,
        _ => return None,
    };
    let mut refinements = Vec::new();
    let mut children = Vec::new();
    for c in &node.children {
        if let Some(rn) = decompile_node(c, coverage, &mut refinements) {
            children.push(rn);
        } else {
            return None;
        }
    }
    if !refinements.is_empty() {
        return None;
    }
    Some(RuleGroup {
        mode,
        invert: false,
        children,
    })
}

fn decompile_node(
    node: &DistNode,
    coverage: &mut Vec<CoverageTerm>,
    refinements: &mut Vec<PlacementRefinement>,
) -> Option<RuleNode> {
    match &node.kind {
        DistNodeKind::GroupAll | DistNodeKind::GroupAny => {
            decompile_group(node, coverage).map(RuleNode::Group)
        }
        DistNodeKind::Height { min, max } => Some(RuleNode::Condition(Condition {
            channel: ConditionChannel::Height,
            op: band_op(*min, *max, -1.0e5, 1.0e5),
            a: if *min <= -1.0e5 + 1.0 { *max } else { *min },
            b: *max,
            falloff: 0.0,
            seed: 0,
            frequency: 0.02,
        })),
        DistNodeKind::Slope { min_deg, max_deg } => Some(RuleNode::Condition(Condition {
            channel: ConditionChannel::Slope,
            op: band_op(*min_deg, *max_deg, 0.0, 90.0),
            a: if *min_deg <= 0.01 { *max_deg } else { *min_deg },
            b: *max_deg,
            falloff: 0.0,
            seed: 0,
            frequency: 0.02,
        })),
        DistNodeKind::Curvature { min, max } => Some(RuleNode::Condition(Condition {
            channel: ConditionChannel::Curvature,
            op: CompareOp::Between,
            a: *min,
            b: *max,
            falloff: 0.0,
            seed: 0,
            frequency: 0.02,
        })),
        DistNodeKind::Flow { min, max } => Some(RuleNode::Condition(Condition {
            channel: ConditionChannel::Flow,
            op: band_op(*min, *max, 0.0, 1.0),
            a: if *min <= 0.01 { *max } else { *min },
            b: *max,
            falloff: 0.0,
            seed: 0,
            frequency: 0.02,
        })),
        DistNodeKind::SeaLevel { width, .. } => {
            let inverted = node
                .children
                .iter()
                .any(|c| matches!(c.kind, DistNodeKind::EffectInvert));
            Some(RuleNode::Condition(Condition {
                channel: ConditionChannel::WaterDistance,
                op: if inverted {
                    CompareOp::Above
                } else {
                    CompareOp::Below
                },
                a: *width,
                b: 0.0,
                falloff: 0.0,
                seed: 0,
                frequency: 0.02,
            }))
        }
        DistNodeKind::MaskAsset { mask }
        | DistNodeKind::Paint { mask }
        | DistNodeKind::ImportedMask { mask } => {
            let idx = coverage.len();
            coverage.push(CoverageTerm::ImportedMask { mask: mask.id });
            Some(RuleNode::CoverageRef { index: idx })
        }
        DistNodeKind::Polygon { points, soft } => {
            let idx = coverage.len();
            coverage.push(CoverageTerm::Polygon {
                points: points.clone(),
                soft: *soft,
            });
            Some(RuleNode::CoverageRef { index: idx })
        }
        DistNodeKind::EffectDilate { radius_m } => {
            refinements.push(PlacementRefinement::Expand {
                radius_m: *radius_m,
            });
            None
        }
        DistNodeKind::EffectErode { radius_m } => {
            refinements.push(PlacementRefinement::Contract {
                radius_m: *radius_m,
            });
            None
        }
        DistNodeKind::EffectBlur { radius } => {
            refinements.push(PlacementRefinement::Smooth {
                radius_samples: *radius,
            });
            None
        }
        _ => None,
    }
}

fn band_op(min: f32, max: f32, lo_bound: f32, hi_bound: f32) -> CompareOp {
    if min <= lo_bound + 1e-2 {
        CompareOp::Below
    } else if max >= hi_bound - 1e-2 {
        CompareOp::Above
    } else {
        CompareOp::Between
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heightfield::HeightfieldMetrics;
    use std::collections::HashMap;

    fn sample_rules() -> PlacementDefinition {
        let mut p = PlacementDefinition::default();
        p.root = RuleGroup {
            mode: RuleGroupMode::All,
            invert: false,
            children: vec![
                RuleNode::Condition(Condition {
                    channel: ConditionChannel::Height,
                    op: CompareOp::Above,
                    a: 1800.0,
                    b: 0.0,
                    falloff: 0.0,
                    seed: 0,
                    frequency: 0.02,
                }),
                RuleNode::Condition(Condition {
                    channel: ConditionChannel::Slope,
                    op: CompareOp::Below,
                    a: 55.0,
                    b: 0.0,
                    falloff: 0.0,
                    seed: 0,
                    frequency: 0.02,
                }),
            ],
        };
        p.recompute_hash();
        p
    }

    #[test]
    fn compile_all_height_slope() {
        let p = sample_rules();
        let dist = p.compile();
        assert_eq!(dist.nodes.len(), 1);
        assert!(matches!(dist.nodes[0].kind, DistNodeKind::GroupAll));
        assert_eq!(dist.nodes[0].children.len(), 2);
    }

    #[test]
    fn compile_any_group() {
        let mut p = PlacementDefinition::default();
        p.root.mode = RuleGroupMode::Any;
        p.root.children = vec![
            RuleNode::Condition(Condition {
                channel: ConditionChannel::Height,
                op: CompareOp::Below,
                a: 100.0,
                b: 0.0,
                falloff: 0.0,
                seed: 0,
                frequency: 0.02,
            }),
            RuleNode::Condition(Condition {
                channel: ConditionChannel::Flow,
                op: CompareOp::Above,
                a: 0.5,
                b: 0.0,
                falloff: 0.0,
                seed: 0,
                frequency: 0.02,
            }),
        ];
        p.recompute_hash();
        let dist = p.compile();
        assert!(matches!(dist.nodes[0].kind, DistNodeKind::GroupAny));
    }

    #[test]
    fn stable_compile() {
        let p = sample_rules();
        let a = p.compile();
        let b = p.compile();
        assert_eq!(a.nodes.len(), b.nodes.len());
        for (x, y) in a.nodes.iter().zip(b.nodes.iter()) {
            assert_eq!(x.id.0, y.id.0);
            assert_eq!(format!("{:?}", x.kind), format!("{:?}", y.kind));
        }
    }

    #[test]
    fn serde_round_trip() {
        let p = sample_rules();
        let json = serde_json::to_string(&p).unwrap();
        let back: PlacementDefinition = serde_json::from_str(&json).unwrap();
        assert_eq!(back.root.children.len(), 2);
        assert_eq!(back.content_hash, p.content_hash);
    }

    #[test]
    fn custom_preserves_stack() {
        let mut p = sample_rules();
        let mut edited = p.compile();
        edited
            .nodes
            .push(DistNode::new(DistNodeKind::EffectBlur { radius: 3 }));
        p.mark_custom(edited.clone());
        assert_eq!(p.source, PlacementSource::Custom);
        assert_eq!(p.active_distribution().nodes.len(), edited.nodes.len());
        let restored = p.reset_to_rules();
        assert_eq!(p.source, PlacementSource::Rules);
        assert_eq!(restored.nodes.len(), 1);
    }

    #[test]
    fn empty_compile_is_full_coverage() {
        let p = PlacementDefinition::default();
        let dist = p.compile();
        assert!(dist.nodes.is_empty());
        let metrics = HeightfieldMetrics::new(8, 8, 1000.0, 1000.0);
        let masks = HashMap::new();
        let ctx = DistBakeContext::masks_only(&masks);
        let field = bake_distribution_with_context(&dist, metrics, &ctx);
        assert!((field.get(0, 0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn painted_plus_conditions() {
        let mid = MaskId::new();
        let mut p = PlacementDefinition::default();
        p.coverage.push(CoverageTerm::PaintedWorld { mask: mid });
        p.root.children.push(RuleNode::CoverageRef { index: 0 });
        p.root.children.push(RuleNode::Condition(Condition {
            channel: ConditionChannel::Slope,
            op: CompareOp::Below,
            a: 40.0,
            b: 0.0,
            falloff: 0.0,
            seed: 0,
            frequency: 0.02,
        }));
        p.space = PlacementCoordinateSpace::WorldSpace;
        p.recompute_hash();
        let dist = p.compile();
        assert!(matches!(dist.nodes[0].kind, DistNodeKind::GroupAll));
        assert!(dist.nodes[0]
            .children
            .iter()
            .any(|c| matches!(c.kind, DistNodeKind::MaskAsset { .. })));
    }

    #[test]
    fn hash_changes_on_edit() {
        let mut p = sample_rules();
        let h0 = p.content_hash;
        p.root.children.push(RuleNode::Condition(Condition {
            channel: ConditionChannel::Flow,
            op: CompareOp::Above,
            a: 0.2,
            b: 0.0,
            falloff: 0.0,
            seed: 0,
            frequency: 0.02,
        }));
        p.recompute_hash();
        assert_ne!(p.content_hash, h0);
    }

    #[test]
    fn try_decompile_group_all() {
        let p = sample_rules();
        let dist = p.compile();
        let back = PlacementDefinition::try_decompile(&dist).expect("decompile");
        assert_eq!(back.root.mode, RuleGroupMode::All);
        assert_eq!(back.root.children.len(), 2);
    }
}
