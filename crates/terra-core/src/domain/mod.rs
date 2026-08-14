//! Shape / Biome domain model: roles, ownership view, and diagnostics.

mod authoring_eval;
mod classify;
mod ownership;
mod role;
mod warnings;

pub use authoring_eval::{
    authoring_order_is_arbitrary, evaluation_eval_stage_order, incomplete_project_diagnostics,
    workflow_stage_metadata_order, world_eval_outline, SoftDiagnostic,
};
pub use classify::{classify_in_context, classify_layer_kind};
pub use ownership::{DomainBiomeRef, DomainLayerRef, DomainView};
pub use role::{DomainParent, DomainRole};
pub use warnings::behavioural_differences;
