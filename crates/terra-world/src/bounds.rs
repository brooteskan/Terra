use crate::{WorldError, WorldPosition};
use serde::{de::Error as _, Deserialize, Deserializer, Serialize};

/// Finite, closed world-space X/Z bounds for sparse authored data.
///
/// Unlike [`crate::WorldRect`], authored bounds include both their minimum and
/// maximum edges and may be degenerate on either axis. Closed edges ensure a
/// point or path on a tile seam is discoverable from both adjacent evaluation
/// domains.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct WorldBounds {
    min: WorldPosition,
    max: WorldPosition,
}

impl WorldBounds {
    pub fn try_new(min: WorldPosition, max: WorldPosition) -> Result<Self, WorldError> {
        if min.x_m() > max.x_m() || min.z_m() > max.z_m() {
            return Err(WorldError::InvalidWorldBounds);
        }
        Ok(Self { min, max })
    }

    pub const fn from_point(point: WorldPosition) -> Self {
        Self {
            min: point,
            max: point,
        }
    }

    pub fn try_from_points(
        points: impl IntoIterator<Item = WorldPosition>,
    ) -> Result<Option<Self>, WorldError> {
        let mut points = points.into_iter();
        let Some(first) = points.next() else {
            return Ok(None);
        };
        let mut min_x = first.x_m();
        let mut min_z = first.z_m();
        let mut max_x = first.x_m();
        let mut max_z = first.z_m();
        for point in points {
            min_x = min_x.min(point.x_m());
            min_z = min_z.min(point.z_m());
            max_x = max_x.max(point.x_m());
            max_z = max_z.max(point.z_m());
        }
        let min = WorldPosition::try_new(min_x, min_z)?;
        let max = WorldPosition::try_new(max_x, max_z)?;
        Self::try_new(min, max).map(Some)
    }

    pub const fn min(self) -> WorldPosition {
        self.min
    }

    pub const fn max(self) -> WorldPosition {
        self.max
    }

    pub fn checked_union(self, other: Self) -> Result<Self, WorldError> {
        let min = WorldPosition::try_new(
            self.min.x_m().min(other.min.x_m()),
            self.min.z_m().min(other.min.z_m()),
        )?;
        let max = WorldPosition::try_new(
            self.max.x_m().max(other.max.x_m()),
            self.max.z_m().max(other.max.z_m()),
        )?;
        Self::try_new(min, max)
    }

    pub fn checked_expand(self, pad_x_m: f64, pad_z_m: f64) -> Result<Self, WorldError> {
        if !pad_x_m.is_finite() || !pad_z_m.is_finite() || pad_x_m < 0.0 || pad_z_m < 0.0 {
            return Err(WorldError::InvalidWorldExpansion);
        }
        let min = WorldPosition::try_new(self.min.x_m() - pad_x_m, self.min.z_m() - pad_z_m)
            .map_err(|_| WorldError::InvalidWorldExpansion)?;
        let max = WorldPosition::try_new(self.max.x_m() + pad_x_m, self.max.z_m() + pad_z_m)
            .map_err(|_| WorldError::InvalidWorldExpansion)?;
        Self::try_new(min, max).map_err(|_| WorldError::InvalidWorldExpansion)
    }

    pub fn checked_intersection(self, other: Self) -> Result<Option<Self>, WorldError> {
        let min_x = self.min.x_m().max(other.min.x_m());
        let min_z = self.min.z_m().max(other.min.z_m());
        let max_x = self.max.x_m().min(other.max.x_m());
        let max_z = self.max.z_m().min(other.max.z_m());
        if min_x > max_x || min_z > max_z {
            return Ok(None);
        }
        let min = WorldPosition::try_new(min_x, min_z)?;
        let max = WorldPosition::try_new(max_x, max_z)?;
        Self::try_new(min, max).map(Some)
    }

    pub fn intersects(self, other: Self) -> bool {
        self.min.x_m() <= other.max.x_m()
            && self.max.x_m() >= other.min.x_m()
            && self.min.z_m() <= other.max.z_m()
            && self.max.z_m() >= other.min.z_m()
    }
}

impl<'de> Deserialize<'de> for WorldBounds {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct WireBounds {
            min: WorldPosition,
            max: WorldPosition,
        }

        let wire = WireBounds::deserialize(deserializer)?;
        Self::try_new(wire.min, wire.max).map_err(D::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(x: f64, z: f64) -> WorldPosition {
        WorldPosition::try_new(x, z).unwrap()
    }

    #[test]
    fn closed_bounds_allow_points_and_touching_intersections() {
        let point_bounds = WorldBounds::try_new(point(1.0, 2.0), point(1.0, 2.0)).unwrap();
        let area = WorldBounds::try_new(point(-1.0, -1.0), point(1.0, 2.0)).unwrap();
        assert!(point_bounds.intersects(area));
        assert_eq!(
            point_bounds.checked_intersection(area).unwrap(),
            Some(point_bounds)
        );
    }

    #[test]
    fn union_expansion_and_disjoint_intersection_are_checked() {
        let left = WorldBounds::try_new(point(-2.0, -1.0), point(-1.0, 1.0)).unwrap();
        let right = WorldBounds::try_new(point(2.0, 0.0), point(3.0, 4.0)).unwrap();
        let union = left.checked_union(right).unwrap();
        assert_eq!(union.min(), point(-2.0, -1.0));
        assert_eq!(union.max(), point(3.0, 4.0));
        assert_eq!(left.checked_intersection(right).unwrap(), None);
        assert_eq!(
            left.checked_expand(1.0, 2.0).unwrap(),
            WorldBounds::try_new(point(-3.0, -3.0), point(0.0, 3.0)).unwrap()
        );
        assert_eq!(
            left.checked_expand(-1.0, 0.0),
            Err(WorldError::InvalidWorldExpansion)
        );
        assert_eq!(
            WorldBounds::from_point(point(f64::MAX, 0.0)).checked_expand(f64::MAX, 0.0),
            Err(WorldError::InvalidWorldExpansion)
        );
    }

    #[test]
    fn constructors_and_deserialization_preserve_invariants() {
        assert_eq!(
            WorldBounds::try_new(point(2.0, 0.0), point(1.0, 1.0)),
            Err(WorldError::InvalidWorldBounds)
        );
        assert!(serde_json::from_str::<WorldBounds>(
            r#"{"min":{"x_m":2.0,"z_m":0.0},"max":{"x_m":1.0,"z_m":1.0}}"#,
        )
        .is_err());
        let bounds = WorldBounds::try_new(point(-4.0, 3.0), point(5.0, 8.0)).unwrap();
        let json = serde_json::to_string(&bounds).unwrap();
        assert_eq!(serde_json::from_str::<WorldBounds>(&json).unwrap(), bounds);
    }

    #[test]
    fn point_fold_is_empty_or_finite() {
        assert_eq!(WorldBounds::try_from_points([]).unwrap(), None);
        let bounds = WorldBounds::try_from_points([point(4.0, -2.0), point(-1.0, 3.0)])
            .unwrap()
            .unwrap();
        assert_eq!(bounds.min(), point(-1.0, -2.0));
        assert_eq!(bounds.max(), point(4.0, 3.0));
    }
}
