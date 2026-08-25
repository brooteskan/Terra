use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Stable identity shared by a persisted authored record and runtime indexes.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_round_trips_as_a_transparent_uuid() {
        let id = AuthoredFeatureId(Uuid::from_u128(0x222));
        let json = serde_json::to_string(&id).unwrap();

        assert_eq!(json, r#""00000000-0000-0000-0000-000000000222""#);
        assert_eq!(
            serde_json::from_str::<AuthoredFeatureId>(&json).unwrap(),
            id
        );
    }
}
