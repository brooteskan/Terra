//! Frame graph: the frame's pass plan and dense timestamp allocation.
//!
//! [`FrameSchedule`] is built once per frame from the active backend and decides
//! which passes run. As the frame path records each pass it logs it via
//! [`FrameGraph::mark`]; at submit time [`FrameGraph::end_frame`] debug-asserts
//! the recorded sequence equals [`FrameSchedule::passes`], so the plan and the
//! recording cannot silently drift apart.

use crate::gpu_timing::GpuTimestampTimer;

/// Logical passes scheduled each frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassKind {
    Begin,
    Shadow,
    RasterLit,
    ProgressivePt,
    ProgressivePost,
    Overlays,
    ResolveTimestamps,
}

/// Which presentation backend owns the main color/depth this frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentationBackendId {
    RasterLit,
    ProgressivePt,
}

impl PresentationBackendId {
    pub fn from_mode(mode: crate::ViewportRendererMode) -> Self {
        if mode.uses_progressive_path_tracer() {
            Self::ProgressivePt
        } else {
            Self::RasterLit
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::RasterLit => "RasterLit",
            Self::ProgressivePt => "ProgressivePt",
        }
    }
}

/// Planned pass list for one frame (built before recording).
#[derive(Debug, Clone, Default)]
pub struct FrameSchedule {
    pub backend: Option<PresentationBackendId>,
    pub shadow: bool,
    /// Whether the path tracer actually dispatches this frame. False on converged
    /// progressive frames (spp 0), which still present the last HDR via post.
    pub pt_dispatch: bool,
    pub progressive_post: bool,
    pub overlays: bool,
}

impl FrameSchedule {
    pub fn for_backend(
        backend: PresentationBackendId,
        shadows_enabled: bool,
        pt_dispatch: bool,
    ) -> Self {
        match backend {
            // Raster never path-traces; ignore the caller's dispatch hint.
            PresentationBackendId::RasterLit => Self {
                backend: Some(backend),
                shadow: shadows_enabled,
                pt_dispatch: false,
                progressive_post: false,
                overlays: true,
            },
            PresentationBackendId::ProgressivePt => Self {
                backend: Some(backend),
                shadow: false,
                pt_dispatch,
                progressive_post: true,
                overlays: true,
            },
        }
    }

    pub fn passes(&self) -> Vec<PassKind> {
        let mut out = vec![PassKind::Begin];
        if self.shadow {
            out.push(PassKind::Shadow);
        }
        match self.backend {
            Some(PresentationBackendId::RasterLit) => out.push(PassKind::RasterLit),
            Some(PresentationBackendId::ProgressivePt) if self.pt_dispatch => {
                out.push(PassKind::ProgressivePt)
            }
            // Converged progressive frames skip the dispatch but still post + present.
            Some(PresentationBackendId::ProgressivePt) => {}
            None => {}
        }
        if self.progressive_post {
            out.push(PassKind::ProgressivePost);
        }
        if self.overlays {
            out.push(PassKind::Overlays);
        }
        out.push(PassKind::ResolveTimestamps);
        out
    }
}

/// Thin frame-graph host: the plan plus the passes actually recorded this frame.
#[derive(Debug, Default)]
pub struct FrameGraph {
    pub schedule: FrameSchedule,
    recorded: Vec<PassKind>,
}

impl FrameGraph {
    /// Install the plan for a new frame and seed the recording with `Begin`.
    pub fn begin(&mut self, schedule: FrameSchedule) {
        self.schedule = schedule;
        self.recorded.clear();
        self.recorded.push(PassKind::Begin);
    }

    /// Log a pass as the frame path records it, in recording order.
    pub fn mark(&mut self, pass: PassKind) {
        self.recorded.push(pass);
    }

    /// Resolve timestamps and, in debug builds, assert the recorded passes match
    /// the plan. Call immediately before submitting the frame's encoder.
    pub fn end_frame(
        &mut self,
        timer: Option<&mut GpuTimestampTimer>,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        self.recorded.push(PassKind::ResolveTimestamps);
        if let Some(timer) = timer {
            timer.resolve(encoder);
        }
        debug_assert_eq!(
            self.recorded,
            self.schedule.passes(),
            "frame graph drift: recorded passes do not match the planned schedule"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raster_schedule_orders_passes() {
        let s = FrameSchedule::for_backend(PresentationBackendId::RasterLit, true, false);
        assert_eq!(
            s.passes(),
            vec![
                PassKind::Begin,
                PassKind::Shadow,
                PassKind::RasterLit,
                PassKind::Overlays,
                PassKind::ResolveTimestamps,
            ]
        );
    }

    #[test]
    fn raster_without_shadows_skips_shadow_pass() {
        let s = FrameSchedule::for_backend(PresentationBackendId::RasterLit, false, false);
        assert_eq!(
            s.passes(),
            vec![
                PassKind::Begin,
                PassKind::RasterLit,
                PassKind::Overlays,
                PassKind::ResolveTimestamps,
            ]
        );
    }

    #[test]
    fn progressive_schedule_includes_dispatch_when_sampling() {
        let s = FrameSchedule::for_backend(PresentationBackendId::ProgressivePt, false, true);
        assert_eq!(
            s.passes(),
            vec![
                PassKind::Begin,
                PassKind::ProgressivePt,
                PassKind::ProgressivePost,
                PassKind::Overlays,
                PassKind::ResolveTimestamps,
            ]
        );
    }

    #[test]
    fn converged_progressive_frame_skips_dispatch() {
        // spp 0: no path-trace dispatch, but the frame still posts and presents.
        let s = FrameSchedule::for_backend(PresentationBackendId::ProgressivePt, false, false);
        assert_eq!(
            s.passes(),
            vec![
                PassKind::Begin,
                PassKind::ProgressivePost,
                PassKind::Overlays,
                PassKind::ResolveTimestamps,
            ]
        );
    }

    #[test]
    fn progressive_backend_ignores_shadow_request() {
        // The raster shadow pass never runs under the path tracer.
        let s = FrameSchedule::for_backend(PresentationBackendId::ProgressivePt, true, true);
        assert!(!s.shadow);
        assert!(!s.passes().contains(&PassKind::Shadow));
    }

    #[test]
    fn raster_backend_forces_pt_dispatch_off() {
        // Even if the caller passes a stale dispatch hint, raster clears it.
        let s = FrameSchedule::for_backend(PresentationBackendId::RasterLit, false, true);
        assert!(!s.pt_dispatch);
        assert!(!s.passes().contains(&PassKind::ProgressivePt));
    }
}
