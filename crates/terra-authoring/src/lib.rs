//! Backend-neutral contracts for persisted authored-terrain data.
//!
//! `terra-authoring` sits between [`terra_world`] and `terra-core`: it owns
//! stable authored identity, persisted feature stores, mutation results, and
//! read-only spatial-query adapters. It deliberately has no knowledge of
//! documents, layers, evaluators, rendering, IO, or application state.

mod feature_change;
mod identity;
pub mod sculpt;

pub use feature_change::FeatureChange;
pub use identity::AuthoredFeatureId;
