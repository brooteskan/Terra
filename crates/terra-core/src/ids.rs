//! Stable UUID identities used across the crate.
//!
//! `LayerId` and `OutputId` are plain newtype identifiers with no dependency on
//! the layer stack, so they live in their own leaf module rather than in
//! `layer`. `layer` re-exports both, keeping `terra_core::layer::{LayerId,
//! OutputId}` and every serialized document valid.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Stable layer identity for undo/reorder/caching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LayerId(pub Uuid);

impl LayerId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Stable sentinel id for UI virtual rows (not persisted as stack nodes).
    pub fn from_u128(v: u128) -> Self {
        Self(Uuid::from_u128(v))
    }
}

impl Default for LayerId {
    fn default() -> Self {
        Self::new()
    }
}

/// Stable identity for a published output (survives display-name renames).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OutputId(pub Uuid);

impl OutputId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for OutputId {
    fn default() -> Self {
        Self::new()
    }
}
