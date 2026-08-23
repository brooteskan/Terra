use crate::WorldError;
use serde::{Deserialize, Serialize};

/// Authoritative fixed-origin CPU position in metres.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WorldPosition {
    pub x_m: f64,
    pub z_m: f64,
}

impl WorldPosition {
    pub const ORIGIN: Self = Self { x_m: 0.0, z_m: 0.0 };

    pub fn try_new(x_m: f64, z_m: f64) -> Result<Self, WorldError> {
        if !x_m.is_finite() || !z_m.is_finite() {
            return Err(WorldError::InvalidWorldCoordinate);
        }
        Ok(Self { x_m, z_m })
    }
}

/// A finite, non-empty, half-open world-space rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WorldRect {
    pub min: WorldPosition,
    pub max: WorldPosition,
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
}
