//! Non-destructive Shape history helpers for the Sculpt workspace.
//!
//! Viewport sculpt tools auto-create / update [`LayerKind::SculptStrokes`] layers
//! on the terrain stack. Brush coverage lives on the stroke IR — artists never
//! manually add a Shape Layer, add a Mask, or paint a Mask separately.

use crate::authoring::{SculptStrokeKind, SculptStrokeParams};
use crate::layer::{BrushEditable, EditSupport, Layer, LayerId, LayerKind, LayerStack};
use serde::{Deserialize, Serialize};

/// Whether the next stroke session creates a new Shape Layer or appends to the selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ShapeEditMode {
    /// First dab of a stroke session creates a new named Shape Layer.
    #[default]
    NewLayerPerSession,
    /// Append strokes to the currently selected Shape Layer (`SculptStrokes`).
    ContinueSelected,
}

impl ShapeEditMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::NewLayerPerSession => "New layer per stroke",
            Self::ContinueSelected => "Continue editing selected",
        }
    }
}

/// Artist-facing sculpt / stamp tool that participates in Shape history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShapeTool {
    Raise,
    Lower,
    Smooth,
    Flatten,
    Terrace,
    Pinch,
    Inflate,
    ErodeBrush,
    MountainStamp,
    ValleyStamp,
    PlateauStamp,
    CraterStamp,
    Coastline,
    RiverPath,
    HeightStamp,
    NoiseBrush,
}

impl ShapeTool {
    pub fn all() -> &'static [ShapeTool] {
        &[
            Self::Raise,
            Self::Lower,
            Self::Smooth,
            Self::Flatten,
            Self::Terrace,
            Self::Pinch,
            Self::Inflate,
            Self::ErodeBrush,
            Self::MountainStamp,
            Self::ValleyStamp,
            Self::PlateauStamp,
            Self::CraterStamp,
            Self::Coastline,
            Self::RiverPath,
            Self::HeightStamp,
            Self::NoiseBrush,
        ]
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Raise => "Raise",
            Self::Lower => "Lower",
            Self::Smooth => "Smooth",
            Self::Flatten => "Flatten",
            Self::Terrace => "Terrace",
            Self::Pinch => "Pinch",
            Self::Inflate => "Inflate",
            Self::ErodeBrush => "Erode Brush",
            Self::MountainStamp => "Mountain Stamp",
            Self::ValleyStamp => "Valley Stamp",
            Self::PlateauStamp => "Plateau Stamp",
            Self::CraterStamp => "Crater Stamp",
            Self::Coastline => "Coastline Tool",
            Self::RiverPath => "River Path",
            Self::HeightStamp => "Height Stamp",
            Self::NoiseBrush => "Noise Brush",
        }
    }

    /// Default Shape Layer name for a new stroke session.
    pub fn default_layer_name(self) -> &'static str {
        match self {
            Self::Raise => "Uplift",
            Self::Lower => "Lowering",
            Self::Smooth => "Smoothing",
            Self::Flatten => "Flatten",
            Self::Terrace => "Terraces",
            Self::Pinch => "Pinch",
            Self::Inflate => "Inflate",
            Self::ErodeBrush => "Erosion Brush",
            Self::MountainStamp => "Mountains",
            Self::ValleyStamp => "Valley",
            Self::PlateauStamp => "Plateau",
            Self::CraterStamp => "Crater",
            Self::Coastline => "Coastal Smoothing",
            Self::RiverPath => "River Valley",
            Self::HeightStamp => "Height Stamp",
            Self::NoiseBrush => "Noise Detail",
        }
    }

    pub fn stroke_kind(self) -> SculptStrokeKind {
        match self {
            Self::Raise => SculptStrokeKind::Raise,
            Self::Lower => SculptStrokeKind::Lower,
            Self::Smooth => SculptStrokeKind::Smooth,
            Self::Flatten => SculptStrokeKind::Flatten,
            Self::Terrace => SculptStrokeKind::Terrace,
            Self::Pinch => SculptStrokeKind::Pinch,
            Self::Inflate => SculptStrokeKind::Inflate,
            Self::ErodeBrush => SculptStrokeKind::Erode,
            Self::MountainStamp => SculptStrokeKind::MountainStamp,
            Self::ValleyStamp => SculptStrokeKind::ValleyStamp,
            Self::PlateauStamp => SculptStrokeKind::PlateauStamp,
            Self::CraterStamp => SculptStrokeKind::CraterStamp,
            Self::Coastline => SculptStrokeKind::Coastline,
            Self::RiverPath => SculptStrokeKind::RiverPath,
            Self::HeightStamp => SculptStrokeKind::HeightStamp,
            Self::NoiseBrush => SculptStrokeKind::Noise,
        }
    }

    /// Single-click stamp tools (do not continuous-drag append by default).
    pub fn is_stamp(self) -> bool {
        matches!(
            self,
            Self::MountainStamp
                | Self::ValleyStamp
                | Self::PlateauStamp
                | Self::CraterStamp
                | Self::HeightStamp
        )
    }

    /// Classification for viewport authoring readiness.
    pub fn authoring_class(self) -> ShapeAuthoringClass {
        match self {
            Self::Raise
            | Self::Lower
            | Self::Smooth
            | Self::Flatten
            | Self::Terrace
            | Self::NoiseBrush
            | Self::ErodeBrush
            | Self::MountainStamp
            | Self::ValleyStamp
            | Self::PlateauStamp
            | Self::CraterStamp
            | Self::HeightStamp
            | Self::Coastline
            | Self::RiverPath => ShapeAuthoringClass::Ready,
            Self::Pinch | Self::Inflate => ShapeAuthoringClass::Ready,
        }
    }
}

/// Whether a tool can participate in direct viewport Shape history authoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShapeAuthoringClass {
    /// Stroke / stamp IR on a Shape Layer today.
    Ready,
    /// Existing generator/processor; wrap with auto Shape Layer + coverage.
    Adaptable,
    /// Needs significant new IR / GPU / interaction.
    Deferred,
}

/// Resolve which Shape Layer to stamp into.
pub fn resolve_shape_target(
    stack: &LayerStack,
    selected: Option<LayerId>,
    mode: ShapeEditMode,
    session_layer: Option<LayerId>,
    tool: ShapeTool,
) -> ShapeTargetDecision {
    let brush = tool.stroke_kind();

    // Explicit Semantic Sculpt selection wins so artists can deliberately
    // continue an existing non-destructive stroke layer. Foundation is the one
    // other explicit edit target; an unsupported brush stays unavailable rather
    // than silently redirecting away from the artist's selection.
    if let Some(id) = selected {
        if let Some(layer) = stack.find(id) {
            if matches!(layer.kind, LayerKind::SculptStrokes(_)) {
                return ShapeTargetDecision::UseExisting(id);
            }
            if layer.kind.is_sculpt_base() {
                return if layer.brush_support(brush) == EditSupport::Unsupported {
                    ShapeTargetDecision::UnavailableOnFoundation { layer: id, tool }
                } else {
                    ShapeTargetDecision::UseExisting(id)
                };
            }
        }
    }
    if let Some(id) = session_layer {
        if stack
            .find(id)
            .is_some_and(|layer| matches!(layer.kind, LayerKind::SculptStrokes(_)))
        {
            return ShapeTargetDecision::UseExisting(id);
        }
    }
    if mode == ShapeEditMode::ContinueSelected {
        if let Some(id) = selected {
            if stack
                .find(id)
                .is_some_and(|layer| matches!(layer.kind, LayerKind::SculptStrokes(_)))
            {
                return ShapeTargetDecision::UseExisting(id);
            }
        }
    }
    ShapeTargetDecision::CreateNew {
        name: unique_shape_name(stack, tool.default_layer_name()),
        tool,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeTargetDecision {
    UseExisting(LayerId),
    CreateNew {
        name: String,
        tool: ShapeTool,
    },
    /// Foundation was explicitly selected but cannot represent this brush.
    UnavailableOnFoundation {
        layer: LayerId,
        tool: ShapeTool,
    },
}

fn unique_shape_name(stack: &LayerStack, base: &str) -> String {
    let names: Vec<String> = stack
        .flatten_layers()
        .iter()
        .map(|l| l.common.name.clone())
        .collect();
    if !names.iter().any(|n| n == base) {
        return base.to_string();
    }
    for i in 2..200 {
        let candidate = format!("{base} {i}");
        if !names.iter().any(|n| n == &candidate) {
            return candidate;
        }
    }
    format!("{base} {}", LayerId::new().0.as_simple())
}

/// Create an empty Shape history layer (`SculptStrokes`) ready for viewport stamps.
pub fn create_shape_layer(name: impl Into<String>) -> Layer {
    Layer::new(
        name,
        LayerKind::SculptStrokes(SculptStrokeParams::default()),
    )
}

/// Merge stroke lists from `sources` into `keep` (non-destructive append).
pub fn merge_sculpt_stroke_layers(keep: &mut SculptStrokeParams, sources: &[&SculptStrokeParams]) {
    for src in sources {
        keep.strokes.extend(src.strokes.iter().cloned());
        keep.world_strokes
            .extend(src.world_strokes.iter().cloned().map(|mut stroke| {
                stroke.id = terra_world::AuthoredFeatureId::new();
                stroke
            }));
    }
}

/// Editing target label: `Stack → Shape Layer`.
pub fn editing_target_label(scope_name: &str, layer_name: &str) -> String {
    format!("Editing:\n{scope_name} → {layer_name}")
}

/// Compact one-line target for chrome.
pub fn editing_target_line(scope_name: &str, layer_name: &str) -> String {
    format!("Editing: {scope_name} → {layer_name}")
}

/// True when `layer` participates in Shape history (non-destructive sculpt).
pub fn is_shape_history_layer(kind: &LayerKind) -> bool {
    matches!(kind, LayerKind::SculptStrokes(_))
}

/// Append or extend a stroke on params (coverage stored on the layer).
// Sculpt-stroke stamp: the stroke params plus the brush kind, uv position,
// radius/strength/target and a continuing flag, each used once. Kept flat.
#[allow(clippy::too_many_arguments)]
pub fn stamp_stroke(
    params: &mut SculptStrokeParams,
    kind: SculptStrokeKind,
    u: f32,
    v: f32,
    radius_m: f32,
    strength: f32,
    target_height: f32,
    continuing: bool,
) {
    params.stamp_stroke(kind, u, v, radius_m, strength, target_height, continuing);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authoring::{SculptPoint, SculptStroke};

    #[test]
    fn new_session_creates_named_layer_decision() {
        let stack = LayerStack::new();
        let d = resolve_shape_target(
            &stack,
            None,
            ShapeEditMode::NewLayerPerSession,
            None,
            ShapeTool::MountainStamp,
        );
        match d {
            ShapeTargetDecision::CreateNew { name, tool } => {
                assert_eq!(name, "Mountains");
                assert_eq!(tool, ShapeTool::MountainStamp);
            }
            _ => panic!("expected create"),
        }
    }

    #[test]
    fn continue_uses_selected_sculpt_strokes() {
        let mut stack = LayerStack::new();
        let layer = create_shape_layer("Western Mountains");
        let id = layer.id();
        stack.push(layer);
        let d = resolve_shape_target(
            &stack,
            Some(id),
            ShapeEditMode::ContinueSelected,
            None,
            ShapeTool::Raise,
        );
        assert_eq!(d, ShapeTargetDecision::UseExisting(id));
    }

    #[test]
    fn selected_sculpt_strokes_wins_over_new_layer_mode() {
        let mut stack = LayerStack::new();
        let layer = create_shape_layer("Existing Raise");
        let id = layer.id();
        stack.push(layer);
        let d = resolve_shape_target(
            &stack,
            Some(id),
            ShapeEditMode::NewLayerPerSession,
            None,
            ShapeTool::Raise,
        );
        assert_eq!(
            d,
            ShapeTargetDecision::UseExisting(id),
            "selecting a Shape Layer must keep strokes on that layer"
        );
    }

    fn base_stack() -> (LayerStack, LayerId) {
        let mut stack = LayerStack::new();
        let layer = Layer::new(
            "Base",
            LayerKind::SculptBase(crate::layer::SculptParams::filled(8, 0.0)),
        );
        let id = layer.id();
        stack.push(layer);
        (stack, id)
    }

    fn constraints_stack() -> (LayerStack, LayerId) {
        let mut stack = LayerStack::new();
        let layer = Layer::new(
            "Constraints",
            LayerKind::TerrainConstraints(Default::default()),
        );
        let id = layer.id();
        stack.push(layer);
        (stack, id)
    }

    #[test]
    fn foundation_keeps_supported_brush() {
        // A brush the legacy foundation raster implements stays on the Base layer.
        let (stack, id) = base_stack();
        let d = resolve_shape_target(
            &stack,
            Some(id),
            ShapeEditMode::NewLayerPerSession,
            None,
            ShapeTool::Lower,
        );
        assert_eq!(d, ShapeTargetDecision::UseExisting(id));
    }

    #[test]
    fn foundation_keeps_approximate_brush() {
        let (stack, id) = base_stack();
        let d = resolve_shape_target(
            &stack,
            Some(id),
            ShapeEditMode::NewLayerPerSession,
            None,
            ShapeTool::Pinch,
        );
        assert_eq!(d, ShapeTargetDecision::UseExisting(id));
    }

    #[test]
    fn constraints_selection_creates_semantic_sculpt_layer() {
        let (stack, id) = constraints_stack();
        let d = resolve_shape_target(
            &stack,
            Some(id),
            ShapeEditMode::NewLayerPerSession,
            None,
            ShapeTool::MountainStamp,
        );
        assert!(matches!(d, ShapeTargetDecision::CreateNew { .. }));
    }

    #[test]
    fn constraints_redirect_unsupported_brush() {
        let (stack, id) = constraints_stack();
        let d = resolve_shape_target(
            &stack,
            Some(id),
            ShapeEditMode::NewLayerPerSession,
            None,
            ShapeTool::Raise,
        );
        assert!(matches!(d, ShapeTargetDecision::CreateNew { .. }));
    }

    #[test]
    fn foundation_makes_unsupported_brushes_unavailable() {
        // Foundation is an explicit edit request. Brushes its raster cannot
        // represent must stay unavailable instead of redirecting elsewhere.
        let (stack, base) = base_stack();
        for tool in [
            ShapeTool::Terrace,
            ShapeTool::Inflate,
            ShapeTool::MountainStamp,
        ] {
            let d = resolve_shape_target(
                &stack,
                Some(base),
                ShapeEditMode::NewLayerPerSession,
                None,
                tool,
            );
            assert_eq!(
                d,
                ShapeTargetDecision::UnavailableOnFoundation { layer: base, tool }
            );
        }
    }

    #[test]
    fn foundation_restriction_wins_over_live_session_shape_layer() {
        let (mut stack, base) = base_stack();
        let shape = create_shape_layer("Terraces");
        let shape_id = shape.id();
        stack.push(shape);
        let d = resolve_shape_target(
            &stack,
            Some(base),
            ShapeEditMode::NewLayerPerSession,
            Some(shape_id),
            ShapeTool::Terrace,
        );
        assert_eq!(
            d,
            ShapeTargetDecision::UnavailableOnFoundation {
                layer: base,
                tool: ShapeTool::Terrace,
            }
        );
    }

    #[test]
    fn merge_appends_strokes() {
        let mut keep = SculptStrokeParams {
            strokes: vec![SculptStroke {
                kind: SculptStrokeKind::Raise,
                points: vec![SculptPoint::default()],
                ..Default::default()
            }],
            ..Default::default()
        };
        let other = SculptStrokeParams {
            strokes: vec![SculptStroke {
                kind: SculptStrokeKind::Smooth,
                points: vec![SculptPoint::default()],
                ..Default::default()
            }],
            ..Default::default()
        };
        merge_sculpt_stroke_layers(&mut keep, &[&other]);
        assert_eq!(keep.strokes.len(), 2);
    }

    #[test]
    fn editing_label_format() {
        assert_eq!(
            editing_target_line("Main Continent", "Western Mountains"),
            "Editing: Main Continent → Western Mountains"
        );
    }

    #[test]
    fn stamp_stroke_appends_when_continuing() {
        let mut p = SculptStrokeParams::default();
        stamp_stroke(
            &mut p,
            SculptStrokeKind::Raise,
            0.1,
            0.1,
            50.0,
            10.0,
            0.0,
            false,
        );
        stamp_stroke(
            &mut p,
            SculptStrokeKind::Raise,
            0.12,
            0.11,
            50.0,
            10.0,
            0.0,
            true,
        );
        assert_eq!(p.strokes.len(), 1);
        assert_eq!(p.strokes[0].points.len(), 2);
    }
}
