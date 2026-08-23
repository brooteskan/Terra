use crate::field_data::FieldId;
use crate::layer::LayerId;
use serde::{Deserialize, Serialize};
use terra_world::TileAddress;

/// Canonical identity for a terrain tile payload.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TerrainTileKey {
    pub layer: Option<LayerId>,
    pub field: FieldId,
    pub address: TileAddress,
}

impl TerrainTileKey {
    pub const fn new(layer: Option<LayerId>, field: FieldId, address: TileAddress) -> Self {
        Self {
            layer,
            field,
            address,
        }
    }

    pub const fn height(address: TileAddress) -> Self {
        Self::new(None, FieldId::Height, address)
    }
}
