//! Reusable geological operators: terraces and stratigraphy.
//!
//! EffectFilters (Simple / Irregular / Steep Terrace, Strata) and Materials strata
//! share these kernels so rock filters and erosion see one lithology model.

mod strata;
mod terrace;

pub use strata::{
    expose_strata_height, sample_bed, strata_band_displace, strata_depth_m, strata_fields,
    strata_fields_with, stratum_at_depth, BedSample, StrataFieldParams,
};
pub use terrace::{terrace_irregular, terrace_simple, terrace_steep, TerraceControls, TerraceMode};
