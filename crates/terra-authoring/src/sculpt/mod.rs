//! Persisted, coordinate-space-tagged sculpt histories.

mod bounded;
mod history;
mod query;
mod records;
mod world;

pub use bounded::BoundedSculptStore;
pub use history::SculptHistory;
pub use query::{SculptQueryError, WorldSculptQuery};
pub use records::{
    SculptPoint, SculptStroke, SculptStrokeKind, WorldSculptPoint, WorldSculptStroke,
};
pub use world::{DeletedWorldSculpt, SculptStoreError, WorldSculptStore};

pub(crate) const DEFAULT_RECONCILE: f32 = 0.15;
