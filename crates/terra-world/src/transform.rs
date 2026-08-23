use crate::{SampleCoord, WorldError, WorldPosition};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleSpacing {
    x_m: f64,
    z_m: f64,
}

impl SampleSpacing {
    pub fn try_new(x_m: f64, z_m: f64) -> Result<Self, WorldError> {
        if !x_m.is_finite() || !z_m.is_finite() || x_m <= 0.0 || z_m <= 0.0 {
            return Err(WorldError::InvalidSpacing);
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

/// Fixed-origin transform between the global integer sample lattice and metres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleWorldTransform {
    origin: WorldPosition,
    spacing: SampleSpacing,
}

impl SampleWorldTransform {
    pub const fn new(origin: WorldPosition, spacing: SampleSpacing) -> Self {
        Self { origin, spacing }
    }

    pub const fn origin(self) -> WorldPosition {
        self.origin
    }

    pub const fn spacing(self) -> SampleSpacing {
        self.spacing
    }

    pub fn sample_center(self, sample: SampleCoord) -> Result<WorldPosition, WorldError> {
        WorldPosition::try_new(
            self.origin.x_m() + (sample.x as f64 + 0.5) * self.spacing.x_m(),
            self.origin.z_m() + (sample.z as f64 + 0.5) * self.spacing.z_m(),
        )
    }

    pub fn world_to_sample_floor(self, world: WorldPosition) -> Result<SampleCoord, WorldError> {
        let x = ((world.x_m() - self.origin.x_m()) / self.spacing.x_m()).floor();
        let z = ((world.z_m() - self.origin.z_m()) / self.spacing.z_m()).floor();
        if !(I64_MIN_F64..I64_MAX_EXCLUSIVE_F64).contains(&x)
            || !(I64_MIN_F64..I64_MAX_EXCLUSIVE_F64).contains(&z)
        {
            return Err(WorldError::ArithmeticOverflow);
        }
        Ok(SampleCoord {
            x: x as i64,
            z: z as i64,
        })
    }
}

const I64_MIN_F64: f64 = -9_223_372_036_854_775_808.0;
const I64_MAX_EXCLUSIVE_F64: f64 = 9_223_372_036_854_775_808.0;
