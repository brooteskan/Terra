use crate::WorldError;
use serde::{de::Error as _, Deserialize, Deserializer, Serialize};

/// Authoritative fixed-origin CPU position in metres.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct WorldPosition {
    x_m: f64,
    z_m: f64,
}

impl WorldPosition {
    pub const ORIGIN: Self = Self { x_m: 0.0, z_m: 0.0 };

    pub fn try_new(x_m: f64, z_m: f64) -> Result<Self, WorldError> {
        if !x_m.is_finite() || !z_m.is_finite() {
            return Err(WorldError::InvalidWorldCoordinate);
        }
        Ok(Self { x_m, z_m })
    }

    pub const fn x_m(self) -> f64 {
        self.x_m
    }

    pub const fn z_m(self) -> f64 {
        self.z_m
    }
}

impl<'de> Deserialize<'de> for WorldPosition {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct WirePosition {
            x_m: f64,
            z_m: f64,
        }

        let wire = WirePosition::deserialize(deserializer)?;
        Self::try_new(wire.x_m, wire.z_m).map_err(D::Error::custom)
    }
}

/// A finite, non-empty, half-open world-space rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct WorldRect {
    min: WorldPosition,
    max: WorldPosition,
}

impl WorldRect {
    pub fn try_new(min: WorldPosition, max: WorldPosition) -> Result<Self, WorldError> {
        if !min.x_m.is_finite()
            || !min.z_m.is_finite()
            || !max.x_m.is_finite()
            || !max.z_m.is_finite()
            || min.x_m >= max.x_m
            || min.z_m >= max.z_m
        {
            return Err(WorldError::InvalidWorldRect);
        }
        Ok(Self { min, max })
    }

    pub fn width(self) -> f64 {
        self.max.x_m - self.min.x_m
    }

    pub fn height(self) -> f64 {
        self.max.z_m - self.min.z_m
    }

    pub const fn min(self) -> WorldPosition {
        self.min
    }

    pub const fn max(self) -> WorldPosition {
        self.max
    }
}

impl<'de> Deserialize<'de> for WorldRect {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct WireRect {
            min: WorldPosition,
            max: WorldPosition,
        }

        let wire = WireRect::deserialize(deserializer)?;
        Self::try_new(wire.min, wire.max).map_err(D::Error::custom)
    }
}
