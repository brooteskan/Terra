use crate::mask_field::MaskField;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MaskOp {
    Add {
        amount: f32,
    },
    Subtract {
        amount: f32,
    },
    Multiply {
        amount: f32,
    },
    Min {
        value: f32,
    },
    Max {
        value: f32,
    },
    Invert,
    Clamp {
        min: f32,
        max: f32,
    },
    Levels {
        in_black: f32,
        in_white: f32,
        gamma: f32,
    },
    Smoothstep {
        edge0: f32,
        edge1: f32,
    },
    Blur {
        radius: u32,
    },
    Remap {
        out_min: f32,
        out_max: f32,
    },
}

impl MaskOp {
    /// Binary-ish apply when combining two fields; unary ops ignore `b`.
    pub fn apply(self, a: f32, b: f32) -> f32 {
        match self {
            MaskOp::Add { amount } => (a + b * amount).clamp(0.0, 1.0),
            MaskOp::Subtract { amount } => (a - b * amount).clamp(0.0, 1.0),
            MaskOp::Multiply { amount } => (a * (b * amount + (1.0 - amount))).clamp(0.0, 1.0),
            MaskOp::Min { value: _ } => a.min(b),
            MaskOp::Max { value: _ } => a.max(b),
            MaskOp::Invert => 1.0 - a,
            MaskOp::Clamp { min, max } => a.clamp(min, max),
            MaskOp::Levels {
                in_black,
                in_white,
                gamma,
            } => {
                let t = ((a - in_black) / (in_white - in_black).max(1e-6)).clamp(0.0, 1.0);
                t.powf(1.0 / gamma.max(1e-6))
            }
            MaskOp::Smoothstep { edge0, edge1 } => smoothstep(edge0, edge1, a),
            MaskOp::Blur { .. } => a,
            MaskOp::Remap { out_min, out_max } => out_min + a * (out_max - out_min),
        }
    }

    pub fn apply_unary(self, a: f32) -> f32 {
        self.apply(a, a)
    }
}

impl MaskField {
    pub fn combine(&self, other: &Self, op: MaskOp) -> Self {
        assert_eq!(self.metrics.width, other.metrics.width);
        let mut out = self.clone();
        for (a, b) in out.data_mut().iter_mut().zip(other.data().iter()) {
            *a = op.apply(*a, *b);
        }
        out
    }
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub fn apply_mask_ops(field: &mut MaskField, ops: &[MaskOp]) {
    for op in ops {
        match *op {
            MaskOp::Blur { radius } => blur_inplace(field, radius),
            MaskOp::Invert => {
                for v in field.data_mut() {
                    *v = 1.0 - *v;
                }
            }
            MaskOp::Clamp { min, max } => {
                crate::simd_ops::clamp_slice_in_place(field.data_mut(), min, max);
            }
            other => {
                for v in field.data_mut() {
                    *v = other.apply_unary(*v).clamp(0.0, 1.0);
                }
            }
        }
    }
}

fn blur_inplace(field: &mut MaskField, radius: u32) {
    if radius == 0 {
        return;
    }
    let w = field.metrics.width;
    let h = field.metrics.height;
    let src = field.data().to_vec();
    let r = radius as i32;
    for j in 0..h {
        for i in 0..w {
            let mut sum: f32 = 0.0;
            let mut count: f32 = 0.0;
            for dj in -r..=r {
                for di in -r..=r {
                    let ii = i as i32 + di;
                    let jj = j as i32 + dj;
                    if ii >= 0 && jj >= 0 && ii < w as i32 && jj < h as i32 {
                        sum += src[(jj as u32 * w + ii as u32) as usize];
                        count += 1.0;
                    }
                }
            }
            field.set(i, j, sum / count.max(1.0));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heightfield::HeightfieldMetrics;

    #[test]
    fn invert_and_clamp() {
        let m = HeightfieldMetrics::new(4, 4, 4.0, 4.0);
        let mut f = MaskField::filled(m, 0.25);
        apply_mask_ops(
            &mut f,
            &[MaskOp::Invert, MaskOp::Clamp { min: 0.0, max: 0.5 }],
        );
        assert!((f.get(0, 0) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn levels_bounds() {
        let op = MaskOp::Levels {
            in_black: 0.0,
            in_white: 1.0,
            gamma: 1.0,
        };
        assert!((op.apply_unary(0.0) - 0.0).abs() < 1e-5);
        assert!((op.apply_unary(1.0) - 1.0).abs() < 1e-5);
    }
}
