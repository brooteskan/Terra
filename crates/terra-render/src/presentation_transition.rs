//! Typed terrain presentation baseline and shadow-mode transition validation.

use std::fmt;
use terra_core::tiling::SampleRect;
use terra_gpu::output_identity::{
    GpuOutputCompleteness, GpuOutputCoverage, GpuOutputId, GpuTerrainOutputIdentity,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerrainPresentationMode {
    Shared,
    FullCopy,
    RegionalCopy,
    CpuUpload,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerrainPresentationDecisionCode {
    Accepted,
    RefusedStaleGeneration,
    RefusedNoOutput,
}

impl TerrainPresentationDecisionCode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::RefusedStaleGeneration => "refused_stale_generation",
            Self::RefusedNoOutput => "refused_no_output",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerrainTransitionDiagnosticCode {
    NonFinalCurrentOutput,
    RegionalWithoutCompleteBaseline,
    PlanRevisionMismatch,
    DeviceGenerationMismatch,
    ResourceIncarnationMismatch,
    SourceRendererExtentMismatch,
    RegionalBaseMismatch,
    SourceGenerationStale,
    SourceGenerationSuperseded,
    PartialSourceIncomplete,
    HeightNormalLineageMismatch,
    OutsideDirtyRegionChanged,
}

impl TerrainTransitionDiagnosticCode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NonFinalCurrentOutput => "non_final_current_output",
            Self::RegionalWithoutCompleteBaseline => "regional_without_complete_baseline",
            Self::PlanRevisionMismatch => "plan_revision_mismatch",
            Self::DeviceGenerationMismatch => "device_generation_mismatch",
            Self::ResourceIncarnationMismatch => "resource_incarnation_mismatch",
            Self::SourceRendererExtentMismatch => "source_renderer_extent_mismatch",
            Self::RegionalBaseMismatch => "regional_base_mismatch",
            Self::SourceGenerationStale => "source_generation_stale",
            Self::SourceGenerationSuperseded => "source_generation_superseded",
            Self::PartialSourceIncomplete => "partial_source_incomplete",
            Self::HeightNormalLineageMismatch => "height_normal_lineage_mismatch",
            Self::OutsideDirtyRegionChanged => "outside_dirty_region_changed",
        }
    }
}

impl fmt::Display for TerrainTransitionDiagnosticCode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresentedTerrainBaseline {
    pub identity: GpuTerrainOutputIdentity,
    pub mode: TerrainPresentationMode,
    pub complete: bool,
    pub local_slots_coherent: bool,
    pub local_slot_epoch: u64,
    pub last_full_generation: Option<u64>,
    pub height_lineage: GpuOutputId,
    pub normal_lineage: GpuOutputId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerrainPresentationExpectations {
    pub plan_revision: u64,
    pub generation: u64,
    pub extent: (u32, u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerrainPresentationRecord {
    pub candidate: GpuTerrainOutputIdentity,
    pub expectations: TerrainPresentationExpectations,
    pub baseline_before: Option<PresentedTerrainBaseline>,
    pub baseline_after: PresentedTerrainBaseline,
    pub requested_mode: TerrainPresentationMode,
    pub actual_mode: TerrainPresentationMode,
    pub requested_rect: Option<SampleRect>,
    pub actual_rect: Option<SampleRect>,
    pub local_slots_coherent_before: bool,
    pub local_slots_coherent_after: bool,
    pub decision: TerrainPresentationDecisionCode,
    pub shadow_diagnostic: Option<TerrainTransitionDiagnosticCode>,
}

pub fn validate_transition_shadow(
    candidate: GpuTerrainOutputIdentity,
    baseline: Option<PresentedTerrainBaseline>,
    expected: TerrainPresentationExpectations,
    actual_mode: TerrainPresentationMode,
) -> Option<TerrainTransitionDiagnosticCode> {
    if matches!(candidate.completeness, GpuOutputCompleteness::Complete)
        && candidate.selected_field.selected != candidate.selected_field.expected_final
    {
        return Some(TerrainTransitionDiagnosticCode::NonFinalCurrentOutput);
    }
    if candidate.plan_revision != expected.plan_revision {
        return Some(TerrainTransitionDiagnosticCode::PlanRevisionMismatch);
    }
    if candidate.extent != expected.extent {
        return Some(TerrainTransitionDiagnosticCode::SourceRendererExtentMismatch);
    }
    if candidate.generation < expected.generation {
        return Some(TerrainTransitionDiagnosticCode::SourceGenerationStale);
    }
    if candidate.generation > expected.generation {
        return Some(TerrainTransitionDiagnosticCode::SourceGenerationSuperseded);
    }

    let GpuOutputCoverage::Patch { expected_base, .. } = candidate.coverage else {
        if baseline.is_some_and(|value| value.height_lineage != value.normal_lineage) {
            return Some(TerrainTransitionDiagnosticCode::HeightNormalLineageMismatch);
        }
        return None;
    };
    // Shared and full-copy presentations replace the complete visible source and
    // therefore do not consume the renderer-local baseline. The patch's expected
    // base matters only when the renderer actually applies a bounded copy.
    if actual_mode != TerrainPresentationMode::RegionalCopy {
        return None;
    }
    if !matches!(candidate.completeness, GpuOutputCompleteness::Complete) {
        return Some(TerrainTransitionDiagnosticCode::PartialSourceIncomplete);
    }
    let Some(baseline) = baseline else {
        return Some(TerrainTransitionDiagnosticCode::RegionalWithoutCompleteBaseline);
    };
    if !baseline.complete {
        return Some(TerrainTransitionDiagnosticCode::RegionalWithoutCompleteBaseline);
    }
    if expected_base != Some(baseline.identity.output) {
        return Some(TerrainTransitionDiagnosticCode::RegionalBaseMismatch);
    }
    if candidate.output_resource.device_generation
        != baseline.identity.output_resource.device_generation
    {
        return Some(TerrainTransitionDiagnosticCode::DeviceGenerationMismatch);
    }
    if candidate.output_resource.incarnation != baseline.identity.output_resource.incarnation {
        return Some(TerrainTransitionDiagnosticCode::ResourceIncarnationMismatch);
    }
    if baseline.height_lineage != baseline.normal_lineage {
        return Some(TerrainTransitionDiagnosticCode::HeightNormalLineageMismatch);
    }
    None
}

/// Choose whether a requested regional copy can consume the renderer-local
/// baseline or must rebuild that baseline from the complete source texture.
///
/// A full copy can repair local lineage failures (for example, when an
/// intermediate GPU evaluation was superseded before presentation). Candidate
/// failures that a full copy cannot repair remain regional so the original
/// diagnostic is preserved instead of being disguised as recovery.
pub fn recoverable_regional_presentation_mode(
    candidate: GpuTerrainOutputIdentity,
    baseline: Option<PresentedTerrainBaseline>,
    expected: TerrainPresentationExpectations,
    local_slots_coherent: bool,
) -> TerrainPresentationMode {
    if !local_slots_coherent {
        return TerrainPresentationMode::FullCopy;
    }
    let regional_diagnostic = validate_transition_shadow(
        candidate,
        baseline,
        expected,
        TerrainPresentationMode::RegionalCopy,
    );
    let full_copy_repairs_transition = regional_diagnostic.is_some()
        && validate_transition_shadow(
            candidate,
            baseline,
            expected,
            TerrainPresentationMode::FullCopy,
        )
        .is_none();
    if full_copy_repairs_transition {
        TerrainPresentationMode::FullCopy
    } else {
        TerrainPresentationMode::RegionalCopy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use terra_core::{quality::PreviewQuality, terrain_plan::FieldSlot};
    use terra_gpu::output_identity::*;

    fn identity(output: u64, coverage: GpuOutputCoverage) -> GpuTerrainOutputIdentity {
        GpuTerrainOutputIdentity {
            output: GpuOutputId(output),
            frame_id: 2,
            generation: 3,
            evaluation_id: 4,
            plan_revision: 5,
            requested_quality: PreviewQuality::Full,
            actual_quality: PreviewQuality::Full,
            intent: GpuEvaluationIntent::InteractiveLocal,
            selected_field: GpuSelectedFieldIdentity {
                selected: FieldSlot::from_index(1),
                expected_final: FieldSlot::from_index(1),
                resource_incarnation: GpuResourceIncarnation(8),
                physical_allocation: 0,
            },
            output_resource: GpuOutputResourceIdentity {
                device_generation: 7,
                incarnation: GpuResourceIncarnation(9),
                slot: GpuOutputSlot::Ping,
            },
            extent: (64, 64),
            coverage,
            completeness: GpuOutputCompleteness::Complete,
            invalidation: GpuInvalidationKind::Regional,
            last_write: GpuLastWriteIdentity {
                serial: GpuSubmissionSerial(10),
                completion: GpuSubmissionCompletion::Submitted,
            },
        }
    }

    fn baseline(identity: GpuTerrainOutputIdentity) -> PresentedTerrainBaseline {
        PresentedTerrainBaseline {
            identity,
            mode: TerrainPresentationMode::Shared,
            complete: true,
            local_slots_coherent: false,
            local_slot_epoch: 1,
            last_full_generation: Some(identity.generation),
            height_lineage: identity.output,
            normal_lineage: identity.output,
        }
    }

    fn expectations() -> TerrainPresentationExpectations {
        TerrainPresentationExpectations {
            plan_revision: 5,
            generation: 3,
            extent: (64, 64),
        }
    }

    #[test]
    fn diagnostic_codes_are_stable() {
        assert_eq!(
            TerrainTransitionDiagnosticCode::NonFinalCurrentOutput.as_str(),
            "non_final_current_output"
        );
        assert_eq!(
            TerrainTransitionDiagnosticCode::RegionalBaseMismatch.as_str(),
            "regional_base_mismatch"
        );
        assert_eq!(
            TerrainTransitionDiagnosticCode::OutsideDirtyRegionChanged.as_str(),
            "outside_dirty_region_changed"
        );
        assert_eq!(
            TerrainPresentationDecisionCode::RefusedStaleGeneration.as_str(),
            "refused_stale_generation"
        );
        assert_eq!(
            TerrainPresentationDecisionCode::RefusedNoOutput.as_str(),
            "refused_no_output"
        );
    }

    #[test]
    fn regional_patch_requires_exact_complete_base() {
        let base = identity(1, GpuOutputCoverage::WholeField);
        let patch = identity(
            2,
            GpuOutputCoverage::Patch {
                rect: SampleRect {
                    x: 8,
                    y: 8,
                    w: 4,
                    h: 4,
                },
                expected_base: Some(GpuOutputId(99)),
            },
        );
        assert_eq!(
            validate_transition_shadow(
                patch,
                Some(baseline(base)),
                expectations(),
                TerrainPresentationMode::RegionalCopy
            ),
            Some(TerrainTransitionDiagnosticCode::RegionalBaseMismatch)
        );
    }

    #[test]
    fn whole_source_presentations_do_not_consume_the_local_base() {
        let patch = identity(
            2,
            GpuOutputCoverage::Patch {
                rect: SampleRect {
                    x: 8,
                    y: 8,
                    w: 4,
                    h: 4,
                },
                expected_base: Some(GpuOutputId(99)),
            },
        );
        for mode in [
            TerrainPresentationMode::Shared,
            TerrainPresentationMode::FullCopy,
        ] {
            assert_eq!(
                validate_transition_shadow(patch, None, expectations(), mode),
                None,
                "{mode:?} replaces the complete visible source"
            );
        }
    }

    #[test]
    fn recoverable_regional_base_gap_promotes_to_full_copy() {
        let presented = identity(37, GpuOutputCoverage::WholeField);
        let candidate = identity(
            39,
            GpuOutputCoverage::Patch {
                rect: SampleRect {
                    x: 8,
                    y: 8,
                    w: 4,
                    h: 4,
                },
                expected_base: Some(GpuOutputId(38)),
            },
        );

        assert_eq!(
            recoverable_regional_presentation_mode(
                candidate,
                Some(baseline(presented)),
                expectations(),
                true,
            ),
            TerrainPresentationMode::FullCopy
        );
        assert_eq!(
            recoverable_regional_presentation_mode(
                identity(
                    38,
                    GpuOutputCoverage::Patch {
                        rect: SampleRect {
                            x: 8,
                            y: 8,
                            w: 4,
                            h: 4,
                        },
                        expected_base: Some(presented.output),
                    },
                ),
                Some(baseline(presented)),
                expectations(),
                true,
            ),
            TerrainPresentationMode::RegionalCopy
        );
    }

    #[test]
    fn every_shadow_invariant_has_a_stable_first_code() {
        let base_identity = identity(1, GpuOutputCoverage::WholeField);
        let base = baseline(base_identity);
        let patch_coverage = GpuOutputCoverage::Patch {
            rect: SampleRect {
                x: 8,
                y: 8,
                w: 4,
                h: 4,
            },
            expected_base: Some(base_identity.output),
        };
        let validate = |candidate, baseline| {
            validate_transition_shadow(
                candidate,
                baseline,
                expectations(),
                TerrainPresentationMode::RegionalCopy,
            )
        };

        let mut candidate = identity(2, patch_coverage);
        candidate.selected_field.selected = FieldSlot::from_index(2);
        assert_eq!(
            validate(candidate, Some(base)),
            Some(TerrainTransitionDiagnosticCode::NonFinalCurrentOutput)
        );

        let mut candidate = identity(2, patch_coverage);
        candidate.plan_revision = 6;
        assert_eq!(
            validate(candidate, Some(base)),
            Some(TerrainTransitionDiagnosticCode::PlanRevisionMismatch)
        );

        let mut candidate = identity(2, patch_coverage);
        candidate.extent = (32, 64);
        assert_eq!(
            validate(candidate, Some(base)),
            Some(TerrainTransitionDiagnosticCode::SourceRendererExtentMismatch)
        );

        let mut candidate = identity(2, patch_coverage);
        candidate.generation = 2;
        assert_eq!(
            validate(candidate, Some(base)),
            Some(TerrainTransitionDiagnosticCode::SourceGenerationStale)
        );

        let mut candidate = identity(2, patch_coverage);
        candidate.generation = 4;
        assert_eq!(
            validate(candidate, Some(base)),
            Some(TerrainTransitionDiagnosticCode::SourceGenerationSuperseded)
        );

        let mut candidate = identity(2, patch_coverage);
        candidate.completeness = GpuOutputCompleteness::DeferredSuffix;
        assert_eq!(
            validate(candidate, Some(base)),
            Some(TerrainTransitionDiagnosticCode::PartialSourceIncomplete)
        );

        let candidate = identity(2, patch_coverage);
        assert_eq!(
            validate(candidate, None),
            Some(TerrainTransitionDiagnosticCode::RegionalWithoutCompleteBaseline)
        );

        let mut incomplete = base;
        incomplete.complete = false;
        assert_eq!(
            validate(candidate, Some(incomplete)),
            Some(TerrainTransitionDiagnosticCode::RegionalWithoutCompleteBaseline)
        );

        let mut wrong_base = candidate;
        wrong_base.coverage = GpuOutputCoverage::Patch {
            rect: SampleRect {
                x: 8,
                y: 8,
                w: 4,
                h: 4,
            },
            expected_base: Some(GpuOutputId(99)),
        };
        assert_eq!(
            validate(wrong_base, Some(base)),
            Some(TerrainTransitionDiagnosticCode::RegionalBaseMismatch)
        );

        let mut candidate = candidate;
        candidate.output_resource.device_generation += 1;
        assert_eq!(
            validate(candidate, Some(base)),
            Some(TerrainTransitionDiagnosticCode::DeviceGenerationMismatch)
        );

        let mut candidate = identity(2, patch_coverage);
        candidate.output_resource.incarnation = GpuResourceIncarnation(10);
        assert_eq!(
            validate(candidate, Some(base)),
            Some(TerrainTransitionDiagnosticCode::ResourceIncarnationMismatch)
        );

        let mut mismatched_lineage = base;
        mismatched_lineage.normal_lineage = GpuOutputId(77);
        let candidate = identity(2, patch_coverage);
        assert_eq!(
            validate(candidate, Some(mismatched_lineage)),
            Some(TerrainTransitionDiagnosticCode::HeightNormalLineageMismatch)
        );
    }
}
