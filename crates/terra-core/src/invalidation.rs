//! Dependency-free dirty-region policy vocabulary.

/// How a process dirty region should expand for incremental recomputation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirtyClass {
    /// Local stencil (blur / single-pass thermal) — pad by stencil radius only.
    Local,
    /// Multi-iteration neighbourhood ops — expand by tile radius from iters.
    Expanding,
    /// Drainage / SPE / amplify — basin-coupled; prefer full field or large expand.
    BasinDependent,
}

/// Tile Chebyshev expand radius for a dirty class (not sample halo).
pub fn expand_radius_for(class: DirtyClass, stencil: u32, iterations: u32) -> u32 {
    match class {
        DirtyClass::Local => stencil.max(1).saturating_sub(1).max(1),
        DirtyClass::Expanding => {
            let batches = (iterations.max(1) + 7) / 8;
            batches.max(1).min(4)
        }
        DirtyClass::BasinDependent => ((iterations.max(1) + 3) / 4).max(2).min(8),
    }
}

impl DirtyClass {
    /// Sample-space support radius for cache keys and invalidation expansion.
    ///
    /// Basin-coupled processes are treated as global — callers should prefer
    /// `mark_all` when this returns `None`.
    pub fn support_radius_samples(
        self,
        tile_size_samples: u32,
        stencil: u32,
        iterations: u32,
    ) -> Option<u32> {
        match self {
            Self::BasinDependent => None,
            Self::Local | Self::Expanding => Some(
                expand_radius_for(self, stencil, iterations)
                    .saturating_mul(tile_size_samples.max(1)),
            ),
        }
    }
}
