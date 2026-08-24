use std::collections::VecDeque;
use std::sync::Arc;

use terra_jobs::{JobError, JobHandle, Pending, Pollable};
use terra_render::{
    PresentationPipelineBundle, PresentationPipelineCompiler, PresentationPipelineFeature,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PipelineCompileRequest {
    pub(crate) id: u64,
    pub(crate) feature: PresentationPipelineFeature,
    pub(crate) project_generation: u64,
    pub(crate) device_generation: u64,
}

impl PipelineCompileRequest {
    pub(crate) fn matches_live(self, project_generation: u64, device_generation: u64) -> bool {
        self.project_generation == project_generation && self.device_generation == device_generation
    }

    fn same_context(
        self,
        feature: PresentationPipelineFeature,
        project_generation: u64,
        device_generation: u64,
    ) -> bool {
        self.feature == feature
            && self.project_generation == project_generation
            && self.device_generation == device_generation
    }
}

pub(crate) struct PipelineCompileCompletion {
    pub(crate) request: PipelineCompileRequest,
    pub(crate) result: Result<PresentationPipelineBundle, String>,
}

struct DesiredCompile {
    request: PipelineCompileRequest,
    compiler: PresentationPipelineCompiler,
}

struct ActiveCompile {
    request: PipelineCompileRequest,
    handle: JobHandle<Result<PresentationPipelineBundle, String>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PipelineCompileStatus {
    Idle,
    Pending(PipelineCompileRequest),
    Failed {
        request: PipelineCompileRequest,
        message: String,
    },
}

pub(crate) struct PipelineCompileSubmission {
    pub(crate) request: PipelineCompileRequest,
    pub(crate) new_request: bool,
}

pub(crate) struct PresentationPipelineCompileCoordinator {
    desired: VecDeque<DesiredCompile>,
    active: Option<ActiveCompile>,
    completions: VecDeque<PipelineCompileCompletion>,
    failures: Vec<(PipelineCompileRequest, String)>,
    status: PipelineCompileStatus,
    next_request_id: u64,
    completion_waker: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl Default for PresentationPipelineCompileCoordinator {
    fn default() -> Self {
        Self {
            desired: VecDeque::new(),
            active: None,
            completions: VecDeque::new(),
            failures: Vec::new(),
            status: PipelineCompileStatus::Idle,
            next_request_id: 1,
            completion_waker: None,
        }
    }
}

impl PresentationPipelineCompileCoordinator {
    pub(crate) fn set_completion_waker(&mut self, waker: Arc<dyn Fn() + Send + Sync>) {
        self.completion_waker = Some(waker);
    }

    pub(crate) fn request(
        &mut self,
        feature: PresentationPipelineFeature,
        project_generation: u64,
        device_generation: u64,
        compiler: PresentationPipelineCompiler,
    ) -> PipelineCompileSubmission {
        self.failures
            .retain(|(request, _)| request.feature != feature);
        if let Some(request) = self
            .active
            .as_ref()
            .map(|active| active.request)
            .filter(|request| request.same_context(feature, project_generation, device_generation))
        {
            return PipelineCompileSubmission {
                request,
                new_request: false,
            };
        }
        if let Some(request) = self
            .desired
            .iter()
            .map(|desired| desired.request)
            .find(|request| request.same_context(feature, project_generation, device_generation))
        {
            return PipelineCompileSubmission {
                request,
                new_request: false,
            };
        }

        self.desired
            .retain(|desired| desired.request.feature != feature);
        let request = PipelineCompileRequest {
            id: self.next_request_id,
            feature,
            project_generation,
            device_generation,
        };
        self.next_request_id = self.next_request_id.wrapping_add(1).max(1);
        self.desired.push_back(DesiredCompile { request, compiler });
        if self.active.is_none() {
            self.start_next();
        }
        PipelineCompileSubmission {
            request,
            new_request: true,
        }
    }

    pub(crate) fn cancel_all(&mut self) {
        self.desired.clear();
        self.completions.clear();
        self.failures.clear();
        if let Some(active) = &self.active {
            active.handle.cancel();
        }
        self.status = PipelineCompileStatus::Idle;
    }

    pub(crate) fn take_completion(&mut self) -> Option<PipelineCompileCompletion> {
        self.completions.pop_front()
    }

    pub(crate) fn status(&self) -> &PipelineCompileStatus {
        &self.status
    }

    pub(crate) fn failed_request(&self) -> Option<PipelineCompileRequest> {
        self.failures.last().map(|(request, _)| *request)
    }

    pub(crate) fn retained_failure(&self) -> Option<(PipelineCompileRequest, String)> {
        self.failures
            .last()
            .map(|(request, message)| (*request, message.clone()))
    }

    pub(crate) fn record_install_failure(
        &mut self,
        request: PipelineCompileRequest,
        message: String,
    ) {
        self.failures.push((request, message.clone()));
        self.status = PipelineCompileStatus::Failed { request, message };
    }

    fn start_next(&mut self) {
        let Some(desired) = self.desired.pop_front() else {
            if self.active.is_none() {
                self.status = PipelineCompileStatus::Idle;
            }
            return;
        };
        let request = desired.request;
        let compiler = desired.compiler;
        let waker = self.completion_waker.clone();
        let handle = terra_jobs::spawn_one_shot_with_notify(
            "terra-presentation-pipeline",
            move |_ctx| compiler.compile(request.feature),
            move || {
                if let Some(waker) = waker {
                    waker();
                }
            },
        );
        self.active = Some(ActiveCompile { request, handle });
        self.status = PipelineCompileStatus::Pending(request);
    }

    fn finish_active(
        &mut self,
        outcome: Result<Result<PresentationPipelineBundle, String>, JobError>,
    ) {
        let active = self.active.take().expect("active pipeline compile exists");
        let result = match outcome {
            Ok(result) => result,
            Err(JobError::Cancelled) => {
                self.start_next();
                return;
            }
            Err(JobError::Panicked(message)) => Err(format!("pipeline worker panicked: {message}")),
        };
        if let Err(message) = &result {
            self.failures.push((active.request, message.clone()));
            self.status = PipelineCompileStatus::Failed {
                request: active.request,
                message: message.clone(),
            };
        }
        self.completions.push_back(PipelineCompileCompletion {
            request: active.request,
            result,
        });
        if !self.desired.is_empty() {
            self.start_next();
        } else if !matches!(self.status, PipelineCompileStatus::Failed { .. }) {
            self.status = PipelineCompileStatus::Idle;
        }
    }
}

impl Pollable for PresentationPipelineCompileCoordinator {
    fn pump(&mut self) -> Pending {
        let mut redraw = false;
        if let Some(outcome) = self
            .active
            .as_ref()
            .and_then(|active| active.handle.try_take())
        {
            self.finish_active(outcome);
            redraw = true;
        }
        Pending {
            busy: false,
            animate: false,
            redraw,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_identity_includes_feature_and_generations() {
        let base = PipelineCompileRequest {
            id: 1,
            feature: PresentationPipelineFeature::Progressive,
            project_generation: 3,
            device_generation: 5,
        };
        assert!(base.matches_live(3, 5));
        assert!(!base.matches_live(4, 5));
        assert!(!base.same_context(PresentationPipelineFeature::Brush, 3, 5));
    }
}
