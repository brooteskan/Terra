//! Brush-to-layer edit capability and dispatch.
//!
//! This is the single admission and mutation surface for contextual brush UI
//! and edit routing. Structural registry capabilities remain in `metadata`.

use super::{Layer, LayerKind, SculptParams};
use crate::authoring::{
    SculptPoint, SculptStrokeKind, SculptStrokeParams, TerrainConstraint, TerrainConstraintKind,
    TerrainConstraintParams,
};

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

/// Parameters for one brush sample.
///
/// Both radius units are carried because raster foundation edits operate in UV
/// space while resolution-independent strokes and constraints store metres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BrushDab {
    pub u: f32,
    pub v: f32,
    pub radius_uv: f32,
    pub radius_m: f32,
    pub strength: f32,
    pub target_height: f32,
    pub falloff: f32,
    pub continuing: bool,
}

/// A layer payload that can report and apply brush edits.
///
/// Callers must check [`Self::brush_support`] before applying. Unsupported
/// brushes are nevertheless required to be no-ops so programmatic callers
/// cannot silently coerce an edit into a different primitive.
pub trait BrushEditable {
    fn brush_support(&self, brush: SculptStrokeKind) -> EditSupport;
    fn apply_brush(&mut self, brush: SculptStrokeKind, dab: BrushDab);
}

impl BrushEditable for SculptStrokeParams {
    fn brush_support(&self, _brush: SculptStrokeKind) -> EditSupport {
        EditSupport::Native
    }

    fn apply_brush(&mut self, brush: SculptStrokeKind, dab: BrushDab) {
        self.stamp_stroke(
            brush,
            dab.u,
            dab.v,
            dab.radius_m,
            dab.strength,
            dab.target_height,
            dab.continuing,
        );
        if let Some(last) = self.strokes.last_mut() {
            last.strength = dab.strength;
            last.falloff = dab.falloff;
            last.target_height = dab.target_height;
        }
    }
}

impl BrushEditable for SculptParams {
    fn brush_support(&self, brush: SculptStrokeKind) -> EditSupport {
        match brush.foundation_mode() {
            Some(_) if matches!(brush, SculptStrokeKind::Erode | SculptStrokeKind::Pinch) => {
                EditSupport::Approximate
            }
            Some(_) => EditSupport::Native,
            None => EditSupport::Unsupported,
        }
    }

    fn apply_brush(&mut self, brush: SculptStrokeKind, dab: BrushDab) {
        if let Some(mode) = brush.foundation_mode() {
            self.stamp_circle(dab.u, dab.v, dab.radius_uv, dab.strength, mode);
        }
    }
}

impl BrushEditable for TerrainConstraintParams {
    fn brush_support(&self, brush: SculptStrokeKind) -> EditSupport {
        match brush.terrain_constraint_kind() {
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
        }
    }

    fn apply_brush(&mut self, brush: SculptStrokeKind, dab: BrushDab) {
        let Some(constraint_kind) = brush.terrain_constraint_kind() else {
            return;
        };
        let radius_m = dab.radius_m.max(1.0);
        let point = SculptPoint {
            u: dab.u,
            v: dab.v,
            pressure: 1.0,
        };
        let append = dab.continuing
            && self.constraints.last().is_some_and(|last| {
                last.kind == constraint_kind
                    && (last.width_m - dab.radius_m).abs() <= radius_m * 0.05
            });
        if append {
            self.constraints.last_mut().unwrap().points.push(point);
        } else {
            self.constraints.push(TerrainConstraint {
                kind: constraint_kind,
                points: vec![point],
                width_m: radius_m,
                value: dab.strength,
                strength: if matches!(constraint_kind, TerrainConstraintKind::Protect) {
                    dab.strength.clamp(0.0, 1.0)
                } else {
                    1.0
                },
            });
        }
    }
}

impl BrushEditable for LayerKind {
    fn brush_support(&self, brush: SculptStrokeKind) -> EditSupport {
        match self {
            Self::SculptStrokes(params) => params.brush_support(brush),
            Self::SculptBase(params) => params.brush_support(brush),
            Self::TerrainConstraints(params) => params.brush_support(brush),
            _ => EditSupport::Unsupported,
        }
    }

    fn apply_brush(&mut self, brush: SculptStrokeKind, dab: BrushDab) {
        match self {
            Self::SculptStrokes(params) => params.apply_brush(brush, dab),
            Self::SculptBase(params) => params.apply_brush(brush, dab),
            Self::TerrainConstraints(params) => params.apply_brush(brush, dab),
            _ => {}
        }
    }
}

impl BrushEditable for Layer {
    fn brush_support(&self, brush: SculptStrokeKind) -> EditSupport {
        self.kind.brush_support(brush)
    }

    fn apply_brush(&mut self, brush: SculptStrokeKind, dab: BrushDab) {
        self.kind.apply_brush(brush, dab);
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

    fn dab(continuing: bool) -> BrushDab {
        BrushDab {
            u: 0.5,
            v: 0.5,
            radius_uv: 0.2,
            radius_m: 80.0,
            strength: 12.0,
            target_height: 25.0,
            falloff: 2.5,
            continuing,
        }
    }

    #[test]
    fn sculpt_strokes_natively_support_every_brush() {
        let layer = LayerKind::SculptStrokes(SculptStrokeParams::default());
        for &brush in ALL_BRUSHES {
            assert_eq!(layer.brush_support(brush), EditSupport::Native, "{brush:?}");
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
            assert_eq!(layer.brush_support(brush), expected, "{brush:?}");
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
            assert_eq!(layer.brush_support(brush), expected, "{brush:?}");
        }
        assert_eq!(
            layer.brush_support(SculptStrokeKind::Raise),
            EditSupport::Unsupported,
            "Raise was previously swallowed by the Roughness catch-all"
        );
    }

    #[test]
    fn parametric_layers_support_no_brushes() {
        let layer = LayerKind::Flat(FlatParams::default());
        for &brush in ALL_BRUSHES {
            assert_eq!(
                layer.brush_support(brush),
                EditSupport::Unsupported,
                "{brush:?}"
            );
        }
    }

    #[test]
    fn sculpt_strokes_apply_and_refresh_live_dab_settings() {
        let mut params = SculptStrokeParams::default();
        params.apply_brush(SculptStrokeKind::Raise, dab(false));

        let mut next = dab(true);
        next.u = 0.55;
        next.strength = 30.0;
        next.target_height = 40.0;
        next.falloff = 4.0;
        params.apply_brush(SculptStrokeKind::Raise, next);

        assert_eq!(params.strokes.len(), 1);
        let stroke = &params.strokes[0];
        assert_eq!(stroke.points.len(), 2);
        assert_eq!(stroke.strength, 30.0);
        assert_eq!(stroke.target_height, 40.0);
        assert_eq!(stroke.falloff, 4.0);
    }

    #[test]
    fn sculpt_base_dispatches_supported_brush_and_refuses_unsupported_brush() {
        let mut params = SculptParams::filled(16, 20.0);
        params.apply_brush(SculptStrokeKind::Lower, dab(false));
        assert!(params.samples.iter().any(|&sample| sample < 20.0));

        let after_lower = params.samples.clone();
        params.apply_brush(SculptStrokeKind::Terrace, dab(false));
        assert_eq!(params.samples, after_lower);
    }

    #[test]
    fn terrain_constraints_create_and_continue_explicit_primitives() {
        let mut params = TerrainConstraintParams::default();
        let mut first = dab(false);
        first.strength = 1.5;
        params.apply_brush(SculptStrokeKind::Protect, first);

        let mut next = dab(true);
        next.u = 0.55;
        params.apply_brush(SculptStrokeKind::Protect, next);

        assert_eq!(params.constraints.len(), 1);
        let constraint = &params.constraints[0];
        assert_eq!(constraint.kind, TerrainConstraintKind::Protect);
        assert_eq!(constraint.points.len(), 2);
        assert_eq!(constraint.value, 1.5);
        assert_eq!(constraint.strength, 1.0);
    }

    #[test]
    fn layer_delegation_keeps_unsupported_edits_as_no_ops() {
        let mut layer = Layer::new(
            "Constraints",
            LayerKind::TerrainConstraints(TerrainConstraintParams::default()),
        );
        assert_eq!(
            layer.brush_support(SculptStrokeKind::Raise),
            EditSupport::Unsupported
        );
        layer.apply_brush(SculptStrokeKind::Raise, dab(false));

        let LayerKind::TerrainConstraints(params) = &layer.kind else {
            panic!("expected constraints layer");
        };
        assert!(params.constraints.is_empty());
    }
}
