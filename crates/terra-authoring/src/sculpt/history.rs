use super::{
    BoundedSculptStore, SculptStoreError, SculptStroke, WorldSculptStore, WorldSculptStroke,
    DEFAULT_RECONCILE,
};
use crate::FeatureChange;
use serde::{de::Error as _, Deserialize, Deserializer, Serialize, Serializer};

/// A sculpt history whose persisted coordinate space is explicit and exclusive.
#[derive(Debug, Clone, PartialEq)]
pub enum SculptHistory {
    BoundedUvV1(BoundedSculptStore),
    WorldMetresV1(WorldSculptStore),
}

impl Default for SculptHistory {
    fn default() -> Self {
        Self::BoundedUvV1(BoundedSculptStore::default())
    }
}

impl SculptHistory {
    pub fn empty_bounded() -> Self {
        Self::BoundedUvV1(BoundedSculptStore::default())
    }

    pub fn empty_world() -> Self {
        Self::WorldMetresV1(WorldSculptStore::default())
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::BoundedUvV1(store) => store.is_empty(),
            Self::WorldMetresV1(store) => store.is_empty(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::BoundedUvV1(store) => store.len(),
            Self::WorldMetresV1(store) => store.len(),
        }
    }

    pub fn reconcile(&self) -> f32 {
        match self {
            Self::BoundedUvV1(store) => store.reconcile(),
            Self::WorldMetresV1(store) => store.reconcile(),
        }
    }

    pub fn set_reconcile(&mut self, reconcile: f32) {
        match self {
            Self::BoundedUvV1(store) => store.set_reconcile(reconcile),
            Self::WorldMetresV1(store) => store.set_reconcile(reconcile),
        }
    }

    pub fn as_bounded(&self) -> Option<&BoundedSculptStore> {
        match self {
            Self::BoundedUvV1(store) => Some(store),
            Self::WorldMetresV1(_) => None,
        }
    }

    pub fn as_bounded_mut(&mut self) -> Option<&mut BoundedSculptStore> {
        match self {
            Self::BoundedUvV1(store) => Some(store),
            Self::WorldMetresV1(_) => None,
        }
    }

    pub fn as_world(&self) -> Option<&WorldSculptStore> {
        match self {
            Self::BoundedUvV1(_) => None,
            Self::WorldMetresV1(store) => Some(store),
        }
    }

    pub fn as_world_mut(&mut self) -> Option<&mut WorldSculptStore> {
        match self {
            Self::BoundedUvV1(_) => None,
            Self::WorldMetresV1(store) => Some(store),
        }
    }

    pub fn merge(&mut self, source: &Self) -> Result<Vec<FeatureChange>, SculptStoreError> {
        match (self, source) {
            (Self::BoundedUvV1(destination), Self::BoundedUvV1(source)) => {
                destination.merge(source);
                Ok(Vec::new())
            }
            (Self::WorldMetresV1(destination), Self::WorldMetresV1(source)) => {
                destination.merge(source)
            }
            _ => Err(SculptStoreError::CoordinateSpaceMismatch),
        }
    }

    pub fn reseed_feature_ids(&mut self) {
        if let Self::WorldMetresV1(store) = self {
            store.reseed_feature_ids();
        }
    }
}

#[derive(Deserialize)]
struct SculptHistoryWire {
    #[serde(default)]
    strokes: Vec<SculptStroke>,
    #[serde(default, rename = "world_metres_v1")]
    world_strokes: Option<Vec<WorldSculptStroke>>,
    #[serde(default = "default_reconcile")]
    reconcile: f32,
}

#[derive(Serialize)]
struct BoundedHistoryWire<'a> {
    strokes: &'a [SculptStroke],
    reconcile: f32,
}

#[derive(Serialize)]
struct WorldHistoryWire<'a> {
    strokes: &'a [SculptStroke],
    #[serde(rename = "world_metres_v1")]
    world_strokes: &'a [WorldSculptStroke],
    reconcile: f32,
}

impl Serialize for SculptHistory {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::BoundedUvV1(store) => BoundedHistoryWire {
                strokes: store.records(),
                reconcile: store.reconcile(),
            }
            .serialize(serializer),
            Self::WorldMetresV1(store) => WorldHistoryWire {
                strokes: &[],
                world_strokes: store.records(),
                reconcile: store.reconcile(),
            }
            .serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for SculptHistory {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SculptHistoryWire::deserialize(deserializer)?;
        if !wire.strokes.is_empty()
            && wire
                .world_strokes
                .as_ref()
                .is_some_and(|records| !records.is_empty())
        {
            return Err(D::Error::custom(SculptStoreError::CoordinateSpaceMismatch));
        }
        if wire.strokes.is_empty() {
            if let Some(records) = wire.world_strokes {
                return WorldSculptStore::try_from_records(records, wire.reconcile)
                    .map(Self::WorldMetresV1)
                    .map_err(D::Error::custom);
            }
        }
        Ok(Self::BoundedUvV1(BoundedSculptStore::from_records(
            wire.strokes,
            wire.reconcile,
        )))
    }
}

fn default_reconcile() -> f32 {
    DEFAULT_RECONCILE
}
