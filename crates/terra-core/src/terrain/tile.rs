use crate::fields::FieldId;
use crate::heightfield::TileId;
use crate::layer::LayerId;
use serde::{Deserialize, Serialize};

/// Canonical identity for a terrain tile payload.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TerrainTileKey {
    pub layer: Option<LayerId>,
    pub field: FieldId,
    pub level: u8,
    pub tile: TileId,
}
