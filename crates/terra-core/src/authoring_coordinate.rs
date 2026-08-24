//! Explicit coordinate frames for viewport-authored terrain actions.
//!
//! Bounded projects author in normalized UV while Infinite projects author in
//! fixed-origin world metres. Keeping those representations in distinct enum
//! variants prevents a normalized value from being interpreted as metres.

use serde::{de::Error as _, Deserialize, Deserializer, Serialize};
use terra_world::WorldPosition;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum AuthoringCoordinateError {
    #[error("bounded terrain UV must be finite and inside [0, 1]")]
    InvalidBoundedUv,
    #[error("surface height must be finite")]
    InvalidHeight,
    #[error("brush radius must be finite and positive")]
    InvalidRadius,
}

/// Validated normalized coordinate in a bounded heightfield.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct BoundedUv {
    u: f32,
    v: f32,
}

impl BoundedUv {
    pub fn try_new(u: f32, v: f32) -> Result<Self, AuthoringCoordinateError> {
        if !u.is_finite()
            || !v.is_finite()
            || !(0.0..=1.0).contains(&u)
            || !(0.0..=1.0).contains(&v)
        {
            return Err(AuthoringCoordinateError::InvalidBoundedUv);
        }
        Ok(Self { u, v })
    }

    pub const fn u(self) -> f32 {
        self.u
    }

    pub const fn v(self) -> f32 {
        self.v
    }

    pub const fn tuple(self) -> (f32, f32) {
        (self.u, self.v)
    }
}

impl<'de> Deserialize<'de> for BoundedUv {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct WireUv {
            u: f32,
            v: f32,
        }

        let wire = WireUv::deserialize(deserializer)?;
        Self::try_new(wire.u, wire.v).map_err(D::Error::custom)
    }
}

/// Explicitly framed terrain point carried by authoring actions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AuthoringPoint {
    Bounded { uv: BoundedUv, height_m: f32 },
    Infinite { world: WorldPosition, height_m: f32 },
}

impl AuthoringPoint {
    pub fn bounded(uv: BoundedUv, height_m: f32) -> Result<Self, AuthoringCoordinateError> {
        validate_height(height_m)?;
        Ok(Self::Bounded { uv, height_m })
    }

    pub fn infinite(world: WorldPosition, height_m: f32) -> Result<Self, AuthoringCoordinateError> {
        validate_height(height_m)?;
        Ok(Self::Infinite { world, height_m })
    }

    pub const fn height_m(self) -> f32 {
        match self {
            Self::Bounded { height_m, .. } | Self::Infinite { height_m, .. } => height_m,
        }
    }

    pub const fn bounded_uv(self) -> Option<BoundedUv> {
        match self {
            Self::Bounded { uv, .. } => Some(uv),
            Self::Infinite { .. } => None,
        }
    }

    pub const fn world_position(self) -> Option<WorldPosition> {
        match self {
            Self::Bounded { .. } => None,
            Self::Infinite { world, .. } => Some(world),
        }
    }

    pub fn horizontal_distance(self, other: Self) -> Option<f64> {
        match (self, other) {
            (Self::Bounded { uv: a, .. }, Self::Bounded { uv: b, .. }) => {
                Some(f64::from((b.u() - a.u()).hypot(b.v() - a.v())))
            }
            (Self::Infinite { world: a, .. }, Self::Infinite { world: b, .. }) => {
                Some((b.x_m() - a.x_m()).hypot(b.z_m() - a.z_m()))
            }
            _ => None,
        }
    }

    pub fn lerp(self, other: Self, t: f64) -> Option<Self> {
        let t = t.clamp(0.0, 1.0);
        match (self, other) {
            (
                Self::Bounded {
                    uv: a,
                    height_m: ah,
                },
                Self::Bounded {
                    uv: b,
                    height_m: bh,
                },
            ) => Self::bounded(
                BoundedUv::try_new(
                    a.u() + (b.u() - a.u()) * t as f32,
                    a.v() + (b.v() - a.v()) * t as f32,
                )
                .ok()?,
                ah + (bh - ah) * t as f32,
            )
            .ok(),
            (
                Self::Infinite {
                    world: a,
                    height_m: ah,
                },
                Self::Infinite {
                    world: b,
                    height_m: bh,
                },
            ) => Self::infinite(
                WorldPosition::try_new(
                    a.x_m() + (b.x_m() - a.x_m()) * t,
                    a.z_m() + (b.z_m() - a.z_m()) * t,
                )
                .ok()?,
                ah + (bh - ah) * t as f32,
            )
            .ok(),
            _ => None,
        }
    }
}

/// Project-framed brush footprint. Radius units are fixed by the variant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AuthoringBrushStamp {
    Bounded {
        uv: BoundedUv,
        radius_uv: f32,
        height_m: f32,
    },
    Infinite {
        world: WorldPosition,
        radius_m: f64,
        height_m: f32,
    },
}

impl AuthoringBrushStamp {
    pub fn bounded(
        uv: BoundedUv,
        radius_uv: f32,
        height_m: f32,
    ) -> Result<Self, AuthoringCoordinateError> {
        validate_radius(f64::from(radius_uv))?;
        validate_height(height_m)?;
        Ok(Self::Bounded {
            uv,
            radius_uv,
            height_m,
        })
    }

    pub fn infinite(
        world: WorldPosition,
        radius_m: f64,
        height_m: f32,
    ) -> Result<Self, AuthoringCoordinateError> {
        validate_radius(radius_m)?;
        validate_height(height_m)?;
        Ok(Self::Infinite {
            world,
            radius_m,
            height_m,
        })
    }

    pub const fn point(self) -> AuthoringPoint {
        match self {
            Self::Bounded { uv, height_m, .. } => AuthoringPoint::Bounded { uv, height_m },
            Self::Infinite {
                world, height_m, ..
            } => AuthoringPoint::Infinite { world, height_m },
        }
    }

    pub fn from_point(
        point: AuthoringPoint,
        bounded_radius_uv: f32,
        infinite_radius_m: f64,
    ) -> Result<Self, AuthoringCoordinateError> {
        match point {
            AuthoringPoint::Bounded { uv, height_m } => {
                Self::bounded(uv, bounded_radius_uv, height_m)
            }
            AuthoringPoint::Infinite { world, height_m } => {
                Self::infinite(world, infinite_radius_m, height_m)
            }
        }
    }
}

/// Picker result containing authoritative absolute X/Z and optional bounded UV.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerrainSurfaceHit {
    pub world: WorldPosition,
    pub height_m: f32,
    pub bounded_uv: Option<BoundedUv>,
}

impl TerrainSurfaceHit {
    pub fn try_new(
        world: WorldPosition,
        height_m: f32,
        bounded_uv: Option<BoundedUv>,
    ) -> Result<Self, AuthoringCoordinateError> {
        validate_height(height_m)?;
        Ok(Self {
            world,
            height_m,
            bounded_uv,
        })
    }

    pub fn authoring_point(self, infinite: bool) -> AuthoringPoint {
        if infinite {
            AuthoringPoint::Infinite {
                world: self.world,
                height_m: self.height_m,
            }
        } else {
            AuthoringPoint::Bounded {
                uv: self
                    .bounded_uv
                    .expect("bounded picker results always carry normalized UV"),
                height_m: self.height_m,
            }
        }
    }
}

fn validate_height(height_m: f32) -> Result<(), AuthoringCoordinateError> {
    height_m
        .is_finite()
        .then_some(())
        .ok_or(AuthoringCoordinateError::InvalidHeight)
}

fn validate_radius(radius: f64) -> Result<(), AuthoringCoordinateError> {
    (radius.is_finite() && radius > 0.0)
        .then_some(())
        .ok_or(AuthoringCoordinateError::InvalidRadius)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_uv_rejects_unframed_values() {
        assert!(BoundedUv::try_new(0.25, 0.75).is_ok());
        assert!(BoundedUv::try_new(-0.001, 0.5).is_err());
        assert!(BoundedUv::try_new(0.5, 1.001).is_err());
        assert!(BoundedUv::try_new(f32::NAN, 0.5).is_err());
    }

    #[test]
    fn brush_stamp_variant_fixes_radius_units() {
        let uv = BoundedUv::try_new(0.25, 0.75).unwrap();
        let bounded = AuthoringBrushStamp::bounded(uv, 0.05, 12.0).unwrap();
        assert!(matches!(
            bounded,
            AuthoringBrushStamp::Bounded {
                radius_uv: 0.05,
                ..
            }
        ));

        let world = WorldPosition::try_new(-5_000_000.25, 5_000_000.5).unwrap();
        let infinite = AuthoringBrushStamp::infinite(world, 32.0, 12.0).unwrap();
        assert!(matches!(
            infinite,
            AuthoringBrushStamp::Infinite { radius_m: 32.0, .. }
        ));
    }

    #[test]
    fn world_space_interpolation_stays_absolute_at_large_coordinates() {
        let start = AuthoringPoint::infinite(
            WorldPosition::try_new(-5_000_000.25, 5_000_000.5).unwrap(),
            10.0,
        )
        .unwrap();
        let end = AuthoringPoint::infinite(
            WorldPosition::try_new(-4_999_936.25, 5_000_128.5).unwrap(),
            30.0,
        )
        .unwrap();

        assert_eq!(start.horizontal_distance(end), Some(128.0_f64.hypot(64.0)));
        let midpoint = start.lerp(end, 0.5).unwrap();
        let AuthoringPoint::Infinite { world, height_m } = midpoint else {
            panic!("world interpolation changed coordinate frame");
        };
        assert_eq!(world.x_m(), -4_999_968.25);
        assert_eq!(world.z_m(), 5_000_064.5);
        assert_eq!(height_m, 20.0);
    }

    #[test]
    fn coordinate_frames_cannot_be_interpolated_together() {
        let bounded = AuthoringPoint::bounded(BoundedUv::try_new(0.5, 0.5).unwrap(), 0.0).unwrap();
        let infinite =
            AuthoringPoint::infinite(WorldPosition::try_new(0.5, 0.5).unwrap(), 0.0).unwrap();

        assert_eq!(bounded.horizontal_distance(infinite), None);
        assert_eq!(bounded.lerp(infinite, 0.5), None);
    }
}
