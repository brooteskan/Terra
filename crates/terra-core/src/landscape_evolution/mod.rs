//! Phase 3 — Physically-based landscape evolution.
//!
//! Stream-power landscape evolution after Cordonnier et al. (uplift + fluvial
//! incision) and Tzathas et al. 2024 (analytical / fast evaluation).
//!
//! This is **not** droplet hydraulic erosion.

mod analytical;
mod cache;
mod iterative;
mod operator;
mod params;
mod uplift;

use crate::mask::MaskField;

#[derive(Debug, Clone)]
pub(super) struct BoundaryMasks {
    pub(super) routing_outlets: MaskField,
    pub(super) elevation_locks: MaskField,
}

pub use operator::{
    evaluate_landscape_evolution, LandscapeEvolutionInput, LandscapeEvolutionOperator,
    LandscapeEvolutionOutput,
};
pub use params::{BoundaryMode, EvolutionSolverMode, LandscapeEvolutionParams, UpliftMode};
pub use uplift::{asymmetric_belt, synthesise_uplift};
