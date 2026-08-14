//! Ordered layer stack (user-facing model). No node-graph UI.

mod binding;
mod blend;
mod cache_policy;
mod group_mode;
mod kinds;
mod metadata;
mod operation;
mod output;
mod registry;
mod scale;
mod stack;
mod workflow;

pub use crate::ids::{LayerId, OutputId};
pub use binding::{
    BindingCombine, BindingCurve, BindingSource, ParamBinding, ParamPath, RemapRange,
};
pub use blend::{blend_heights, blend_pair, blend_weights, BlendMode};
pub use cache_policy::{CachePolicy, CacheState};
pub use group_mode::{
    default_preview_color, is_default_preview_color, palette_preview_color, BiomeSection,
    FalloffCurve, GroupEvalMode, GroupInputMode, GroupKind, SelectedGroupInput, StackCategory,
    TransitionSettings,
};
pub use kinds::*;
pub use metadata::{
    AccentCategory, BuildStatus, ChannelRequirements, LayerCapabilities, LayerInstanceMeta,
    LayerTypeMeta, MaskCompatibility,
};
pub use operation::{FieldContract, OperationCategory};
pub use output::{NamedOutputDecl, OutputRef, PublishedOutput};
pub use registry::LayerTypeRegistry;
pub use scale::ScaleBand;
pub use stack::{biome_destination_section, is_shape_kind, LayerGroup, LayerStack, StackNode};
pub use workflow::WorkflowStage;

use crate::mask::Distribution;
use crate::operation_placement::OperationPlacement;
use serde::{Deserialize, Serialize};

/// Parameters shared by every layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerCommon {
    pub id: LayerId,
    pub name: String,
    pub enabled: bool,
    /// Opacity in \[0, 1\].
    pub opacity: f32,
    pub blend: BlendMode,
    #[serde(default)]
    pub masks: Distribution,
    /// When true, parameters and paint are read-only.
    #[serde(default)]
    pub locked: bool,
    /// Solo this layer for preview (others dimmed / skipped in UI).
    #[serde(default)]
    pub solo: bool,
    /// Optional colour tag index (0 = none, 1–7 = palette).
    #[serde(default)]
    pub color_tag: u8,
    /// Intermediate output is cached / baked (legacy bool; prefer [`Self::cache_policy`]).
    #[serde(default)]
    pub cached: bool,
    /// Explicit cache policy (v3+). When absent, derived from `cached`.
    #[serde(default)]
    pub cache_policy: Option<CachePolicy>,
    /// Named outputs published for later mask / binding references.
    #[serde(default)]
    pub outputs: Vec<NamedOutputDecl>,
    /// Spatial parameter modulation bindings.
    #[serde(default)]
    pub param_bindings: Vec<ParamBinding>,
    /// Optional spatial transform for shape-like layers (stamp / polygon).
    #[serde(default)]
    pub shape_transform: Option<crate::biome_paint::ShapeTransform>,
    /// Develop / Biome-local Apply Where (defaults to Entire Biome).
    #[serde(default)]
    pub operation_placement: OperationPlacement,
    /// Artist Develop category when under a Biome (Vegetation vs Objects, etc.).
    #[serde(default)]
    pub develop_category: Option<crate::operation_placement::DevelopCategory>,
}

impl LayerCommon {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: LayerId::new(),
            name: name.into(),
            enabled: true,
            opacity: 1.0,
            blend: BlendMode::Normal,
            masks: Distribution::new(),
            locked: false,
            solo: false,
            color_tag: 0,
            cached: false,
            cache_policy: None,
            outputs: Vec::new(),
            param_bindings: Vec::new(),
            shape_transform: None,
            operation_placement: OperationPlacement::entire_biome(),
            develop_category: None,
        }
    }

    pub fn resolved_cache_policy(&self) -> CachePolicy {
        self.cache_policy
            .unwrap_or_else(|| CachePolicy::from_legacy_cached(self.cached))
    }

    pub fn set_cache_policy(&mut self, policy: CachePolicy) {
        self.cache_policy = Some(policy);
        self.cached = policy.to_legacy_cached();
    }
}

/// A single terrain layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    pub common: LayerCommon,
    pub kind: LayerKind,
}

impl Layer {
    pub fn new(name: impl Into<String>, kind: LayerKind) -> Self {
        let blend = kind.default_blend();
        let mut common = LayerCommon::new(name);
        common.blend = blend;
        // Default published outputs from the operation contract.
        for field in kind.produced_fields() {
            if field != crate::fields::FieldId::Height {
                common
                    .outputs
                    .push(NamedOutputDecl::new(field.display_name(), field));
            }
        }
        Self { common, kind }
    }

    /// Sync compiled local masks from [`LayerCommon::operation_placement`].
    pub fn sync_operation_placement_masks(&mut self) {
        self.common
            .operation_placement
            .sync_definition_from_apply_where();
        self.common
            .operation_placement
            .write_to_layer_masks(&mut self.common.masks);
    }

    pub fn id(&self) -> LayerId {
        self.common.id
    }

    pub fn category(&self) -> OperationCategory {
        self.kind.category()
    }

    pub fn duplicate(&self) -> Self {
        let mut clone = self.clone();
        clone.common.id = LayerId::new();
        clone.common.name = format!("{} Copy", self.common.name);
        // New output ids so references don't silently retarget.
        for out in &mut clone.common.outputs {
            out.id = OutputId::new();
        }
        clone
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_gets_new_id() {
        let a = Layer::new("Base", LayerKind::Flat(FlatParams::default()));
        let b = a.duplicate();
        assert_ne!(a.id(), b.id());
        assert!(b.common.name.contains("Copy"));
    }

    #[test]
    fn generator_defaults_to_add() {
        let fbm = Layer::new("H", LayerKind::Fbm(FbmParams::default()));
        assert_eq!(fbm.common.blend, BlendMode::Add);
        let perlin = Layer::new("P", LayerKind::NoisePerlin(NoiseParams::default()));
        assert_eq!(perlin.common.blend, BlendMode::Add);
    }

    #[test]
    fn filter_defaults_to_normal() {
        let thermal = Layer::new(
            "T",
            LayerKind::ThermalErosion(ThermalErosionParams::default()),
        );
        assert_eq!(thermal.common.blend, BlendMode::Normal);
        let flat = Layer::new("B", LayerKind::Flat(FlatParams::default()));
        assert_eq!(flat.common.blend, BlendMode::Normal);
    }

    #[test]
    fn category_from_kind() {
        let m = Layer::new("M", LayerKind::Mountains(MountainParams::default()));
        assert_eq!(m.category(), OperationCategory::Generator);
        let h = Layer::new(
            "H",
            LayerKind::HydraulicErosion(HydraulicErosionParams::default()),
        );
        assert_eq!(h.category(), OperationCategory::Simulation);
    }
}
