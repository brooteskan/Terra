//! Brush-to-layer edit capability.
//!
//! This is the single admission predicate for contextual brush UI and edit
//! routing. Structural registry capabilities remain in `metadata`.

use super::LayerKind;
use crate::authoring::SculptStrokeKind;

/// Fidelity of the code path that stores a brush edit on a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EditSupport {
    /// The layer stores and evaluates the brush with its intended semantics.
    Native,
    /// The layer has a real edit path, but currently maps to a substitute primitive.
    Approximate,
    /// The layer has no code path that edits its stored content with this brush.
    Unsupported,
}

/// Return the edit support for a brush on a layer kind.
///
/// This deliberately depends only on the two finite domain enums. Instance state
/// such as locking and visibility is a separate UI concern.
pub fn brush_support(layer: &LayerKind, brush: SculptStrokeKind) -> EditSupport {
    match layer {
        LayerKind::SculptStrokes(_) => EditSupport::Native,
        LayerKind::SculptBase(_) => match brush.foundation_mode() {
            Some(_) if matches!(brush, SculptStrokeKind::Erode | SculptStrokeKind::Pinch) => {
                EditSupport::Approximate
            }
            Some(_) => EditSupport::Native,
            None => EditSupport::Unsupported,
        },
        LayerKind::TerrainConstraints(_) => match brush.terrain_constraint_kind() {
            Some(_)
                if matches!(
                    brush,
                    SculptStrokeKind::Hardness | SculptStrokeKind::Sediment
                ) =>
            {
                EditSupport::Approximate
            }
            Some(_) => EditSupport::Native,
            None => EditSupport::Unsupported,
        },
        _ => EditSupport::Unsupported,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::{
        FlatParams, LayerKind, SculptParams, SculptStrokeParams, TerrainConstraintParams,
    };

    const ALL_BRUSHES: &[SculptStrokeKind] = &[
        SculptStrokeKind::Raise,
        SculptStrokeKind::Lower,
        SculptStrokeKind::Smooth,
        SculptStrokeKind::Flatten,
        SculptStrokeKind::Ridge,
        SculptStrokeKind::Valley,
        SculptStrokeKind::Terrace,
        SculptStrokeKind::Roughness,
        SculptStrokeKind::Uplift,
        SculptStrokeKind::Hardness,
        SculptStrokeKind::Sediment,
        SculptStrokeKind::Protect,
        SculptStrokeKind::EncourageErosion,
        SculptStrokeKind::Pinch,
        SculptStrokeKind::Inflate,
        SculptStrokeKind::Erode,
        SculptStrokeKind::Noise,
        SculptStrokeKind::MountainStamp,
        SculptStrokeKind::ValleyStamp,
        SculptStrokeKind::PlateauStamp,
        SculptStrokeKind::CraterStamp,
        SculptStrokeKind::Coastline,
        SculptStrokeKind::RiverPath,
        SculptStrokeKind::HeightStamp,
    ];

    #[test]
    fn sculpt_strokes_natively_support_every_brush() {
        let layer = LayerKind::SculptStrokes(SculptStrokeParams::default());
        for &brush in ALL_BRUSHES {
            assert_eq!(
                brush_support(&layer, brush),
                EditSupport::Native,
                "{brush:?}"
            );
        }
    }

    #[test]
    fn sculpt_base_support_matches_foundation_modes_and_fidelity() {
        let layer = LayerKind::SculptBase(SculptParams::filled(8, 0.0));
        for &brush in ALL_BRUSHES {
            let expected = match brush {
                SculptStrokeKind::Erode | SculptStrokeKind::Pinch => EditSupport::Approximate,
                _ if brush.foundation_mode().is_some() => EditSupport::Native,
                _ => EditSupport::Unsupported,
            };
            assert_eq!(brush_support(&layer, brush), expected, "{brush:?}");
        }
    }

    #[test]
    fn terrain_constraints_use_only_the_explicit_mapping() {
        let layer = LayerKind::TerrainConstraints(TerrainConstraintParams::default());
        for &brush in ALL_BRUSHES {
            let expected = match brush {
                SculptStrokeKind::Hardness | SculptStrokeKind::Sediment => EditSupport::Approximate,
                _ if brush.terrain_constraint_kind().is_some() => EditSupport::Native,
                _ => EditSupport::Unsupported,
            };
            assert_eq!(brush_support(&layer, brush), expected, "{brush:?}");
        }
        assert_eq!(
            brush_support(&layer, SculptStrokeKind::Raise),
            EditSupport::Unsupported,
            "Raise was previously swallowed by the Roughness catch-all"
        );
    }

    #[test]
    fn parametric_layers_support_no_brushes() {
        let layer = LayerKind::Flat(FlatParams::default());
        for &brush in ALL_BRUSHES {
            assert_eq!(
                brush_support(&layer, brush),
                EditSupport::Unsupported,
                "{brush:?}"
            );
        }
    }
}
