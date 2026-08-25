use crate::{SpatialIndex, SpatialIndexError};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Stable identity shared by a serialized authored record and its runtime index entry.
///
/// This compatibility type remains in `terra-world` while current callers move
/// to `terra-authoring`. New spatial-index code must not depend on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AuthoredFeatureId(pub Uuid);

impl AuthoredFeatureId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for AuthoredFeatureId {
    fn default() -> Self {
        Self::new()
    }
}

/// Temporary source-compatible index name for current authored-feature callers.
pub type AuthoredFeatureIndex = SpatialIndex<AuthoredFeatureId>;

/// Temporary source-compatible error name for current authored-feature callers.
pub type FeatureIndexError = SpatialIndexError<AuthoredFeatureId>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatibility_identity_round_trips_through_serde() {
        let feature_id = AuthoredFeatureId(Uuid::from_u128(0x1234));
        let json = serde_json::to_string(&feature_id).unwrap();

        assert_eq!(
            serde_json::from_str::<AuthoredFeatureId>(&json).unwrap(),
            feature_id
        );
    }
}
