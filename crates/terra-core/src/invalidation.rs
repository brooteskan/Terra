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
            let batches = iterations.max(1).div_ceil(8);
            batches.clamp(1, 4)
        }
        DirtyClass::BasinDependent => iterations.max(1).div_ceil(4).clamp(2, 8),
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

/// Effective spatial reach of a *configured* pass: either whole-field, or
/// localizable with a per-side sample halo.
///
/// [`DirtyClass`] is the coarse per-kind bucket; `Reach` is the resolved answer
/// sub-region recompute (#100 phase 2) actually asks — folding a layer's kind
/// together with its parameters, masks, and published aux into "can this confine
/// to a dirty region, and if so by how much?". A `Full` answer is never wrong,
/// only unoptimised, so every rule that produces it errs toward `Full` when
/// uncertain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reach {
    /// The pass observes or writes the whole field; a localized edit cannot be
    /// confined and the entire field must be recomputed.
    Full,
    /// The pass is localizable: a dirty region expands by `halo_samples` on every
    /// side before recompute. `halo_samples == 0` is a pure per-texel pass.
    Localized { halo_samples: u32 },
}

impl Reach {
    /// A per-texel pass: localizable with no neighbourhood halo.
    pub const LOCAL: Reach = Reach::Localized { halo_samples: 0 };

    /// Combine two reaches conservatively: `Full` absorbs; otherwise keep the
    /// larger halo (a pass constrained by two couplings needs the wider one).
    #[must_use]
    pub fn combine(self, other: Reach) -> Reach {
        match (self, other) {
            (Reach::Full, _) | (_, Reach::Full) => Reach::Full,
            (Reach::Localized { halo_samples: a }, Reach::Localized { halo_samples: b }) => {
                Reach::Localized {
                    halo_samples: a.max(b),
                }
            }
        }
    }

    pub fn is_full(self) -> bool {
        matches!(self, Reach::Full)
    }

    /// Per-side halo for a localizable operation. `None` means the operation
    /// requires a complete field and cannot be evaluated over sparse tiles.
    pub const fn halo_samples(self) -> Option<u32> {
        match self {
            Self::Full => None,
            Self::Localized { halo_samples } => Some(halo_samples),
        }
    }
}

/// Why an otherwise known terrain operation cannot execute directly over
/// sparse Infinite-world tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpatialRejectReason {
    RequiresCompleteField,
    FullFieldNormalization,
    BasinDependent,
    GlobalAuxiliary,
    GlobalParameterReduction,
    BoundedAuthoredData,
    MissingDomainCoordinateContract,
    UnclassifiedOperation,
    RegionBakedNotImplemented,
    HierarchicalNotImplemented,
}

impl std::fmt::Display for SpatialRejectReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::RequiresCompleteField => "requires complete-field evaluation",
            Self::FullFieldNormalization => "uses full-field normalization",
            Self::BasinDependent => "depends on basin-wide terrain state",
            Self::GlobalAuxiliary => "consumes a globally-derived auxiliary field",
            Self::GlobalParameterReduction => "reduces a complete field into a parameter",
            Self::BoundedAuthoredData => "uses authored data tied to a bounded project",
            Self::MissingDomainCoordinateContract => {
                "does not yet have an Infinite-world coordinate contract"
            }
            Self::UnclassifiedOperation => "has no spatial capability classification",
            Self::RegionBakedNotImplemented => "requires region-baked evaluation (not in Slice 1)",
            Self::HierarchicalNotImplemented => "requires hierarchical evaluation (not in Slice 1)",
        })
    }
}

/// Sparse-world strategy declared by one configured operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InfiniteOperationCapability {
    /// The operation can evaluate directly over a tile plus its [`Reach`] halo.
    Direct,
    /// Reserved for a future finite-region bake strategy.
    RegionBaked,
    /// Reserved for a future multilevel/global-summary strategy.
    Hierarchical,
    Unsupported(SpatialRejectReason),
}

impl InfiniteOperationCapability {
    pub const fn rejection(self) -> Option<SpatialRejectReason> {
        match self {
            Self::Direct => None,
            Self::RegionBaked => Some(SpatialRejectReason::RegionBakedNotImplemented),
            Self::Hierarchical => Some(SpatialRejectReason::HierarchicalNotImplemented),
            Self::Unsupported(reason) => Some(reason),
        }
    }
}

/// Complete, machine-readable spatial contract for one compiled operation.
/// Reach remains the single source of truth for halo calculations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperationSpatialContract {
    pub reach: Reach,
    pub aux_reach: AuxReach,
    pub infinite: InfiniteOperationCapability,
}

/// How a layer's published auxiliary fields (everything it emits besides height)
/// behave under a localized edit.
///
/// Aux lives in flat, un-tiled `MaskField` buffers, so a partial recompute can
/// only stay correct when the aux is a per-texel function of the edited region.
/// Height is always per-texel; this classifies the *rest* of a layer's outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuxReach {
    /// Publishes no downstream-consumed aux, or only height.
    HeightOnly,
    /// Aux is a per-texel function of the edited footprint (max-merged stamp
    /// fields, height snapshots) — patchable over just the dirty texels.
    PerTexel,
    /// Aux derives from a global reduction (jump-flood distance, whole-field
    /// normalize, basin routing); a localized recompute would leave it stale, so
    /// the layer must recompute whole-field.
    Global,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_absorbs_in_combine() {
        assert_eq!(Reach::Full.combine(Reach::LOCAL), Reach::Full);
        assert_eq!(Reach::LOCAL.combine(Reach::Full), Reach::Full);
        assert_eq!(Reach::Full.combine(Reach::Full), Reach::Full);
    }

    #[test]
    fn combine_keeps_the_larger_halo() {
        let a = Reach::Localized { halo_samples: 2 };
        let b = Reach::Localized { halo_samples: 7 };
        assert_eq!(a.combine(b), Reach::Localized { halo_samples: 7 });
        assert_eq!(b.combine(a), Reach::Localized { halo_samples: 7 });
    }

    #[test]
    fn local_is_zero_halo() {
        assert_eq!(Reach::LOCAL, Reach::Localized { halo_samples: 0 });
        assert!(!Reach::LOCAL.is_full());
        assert!(Reach::Full.is_full());
    }
}
