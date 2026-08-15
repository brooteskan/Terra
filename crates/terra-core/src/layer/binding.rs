//! Parameter modulation bindings (mask / field → numeric params).

use crate::field_data::FieldId;
use crate::layer::OutputId;
use crate::mask::MaskId;
use serde::{Deserialize, Serialize};

/// Where a modulated parameter reads its spatial influence from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BindingSource {
    Mask(MaskId),
    Field(FieldId),
    LayerOutput(OutputId),
    GroupOutput(OutputId),
    Constant(f32),
}

/// How a binding combines with the base parameter value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BindingCombine {
    #[default]
    Multiply,
    Add,
    Replace,
    Interpolate,
}

/// Remap a source field value into a parameter range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemapRange {
    pub in_min: f32,
    pub in_max: f32,
    pub out_min: f32,
    pub out_max: f32,
}

impl Default for RemapRange {
    fn default() -> Self {
        Self {
            in_min: 0.0,
            in_max: 1.0,
            out_min: 0.0,
            out_max: 1.0,
        }
    }
}

impl RemapRange {
    pub fn map(&self, t: f32) -> f32 {
        let span = (self.in_max - self.in_min).max(1e-6);
        let n = ((t - self.in_min) / span).clamp(0.0, 1.0);
        self.out_min + n * (self.out_max - self.out_min)
    }
}

/// Simple curve shaping (identity / smoothstep / invert).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BindingCurve {
    #[default]
    Linear,
    Smoothstep,
    Invert,
}

impl BindingCurve {
    pub fn apply(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            BindingCurve::Linear => t,
            BindingCurve::Smoothstep => t * t * (3.0 - 2.0 * t),
            BindingCurve::Invert => 1.0 - t,
        }
    }
}

/// Path to a numeric parameter on a layer (string key for serde / UI).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ParamPath(pub String);

impl ParamPath {
    pub fn new(path: impl Into<String>) -> Self {
        Self(path.into())
    }
}

/// Drive a numeric operation parameter from a spatial source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamBinding {
    pub target: ParamPath,
    pub source: BindingSource,
    #[serde(default)]
    pub remap: RemapRange,
    #[serde(default)]
    pub curve: BindingCurve,
    #[serde(default = "default_strength")]
    pub strength: f32,
    #[serde(default)]
    pub combine: BindingCombine,
}

fn default_strength() -> f32 {
    1.0
}

impl ParamBinding {
    pub fn new(target: impl Into<String>, source: BindingSource) -> Self {
        Self {
            target: ParamPath::new(target),
            source,
            remap: RemapRange::default(),
            curve: BindingCurve::Linear,
            strength: 1.0,
            combine: BindingCombine::Multiply,
        }
    }

    /// Combine base parameter with a sampled source value in \[0,1\] (pre-remap).
    pub fn apply_scalar(&self, base: f32, sample: f32) -> f32 {
        let t = self.curve.apply(sample);
        let v = self.remap.map(t);
        let s = self.strength.clamp(0.0, 1.0);
        match self.combine {
            BindingCombine::Multiply => base * (1.0 - s + s * v),
            BindingCombine::Add => base + v * s,
            BindingCombine::Replace => base * (1.0 - s) + v * s,
            BindingCombine::Interpolate => base * (1.0 - t * s) + v * (t * s),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiply_binding_scales() {
        let mut b = ParamBinding::new("height", BindingSource::Constant(1.0));
        b.combine = BindingCombine::Multiply;
        b.remap = RemapRange {
            in_min: 0.0,
            in_max: 1.0,
            out_min: 0.0,
            out_max: 1.0,
        };
        let out = b.apply_scalar(100.0, 0.5);
        assert!((out - 50.0).abs() < 1e-3);
    }
}
