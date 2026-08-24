use terra_jobs::{JobError, JobHandle, Pending, Pollable};
use terra_render::{TerrainPipelineBundle, TerrainPipelineCompiler, TerrainShaderVariant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TerrainPipelineRequest {
    pub(crate) variant: TerrainShaderVariant,
    pub(crate) project_generation: u64,
    pub(crate) device_generation: u64,
}

impl TerrainPipelineRequest {
    pub(crate) fn matches_live(
        self,
        variant: TerrainShaderVariant,
        project_generation: u64,
        device_generation: u64,
    ) -> bool {
        self.variant == variant
            && self.project_generation == project_generation
            && self.device_generation == device_generation
    }
}

pub(crate) struct TerrainPipelineCompletion {
    pub(crate) request: TerrainPipelineRequest,
    pub(crate) result: Result<TerrainPipelineBundle, String>,
}

struct DesiredCompile {
    request: TerrainPipelineRequest,
    compiler: TerrainPipelineCompiler,
}

struct ActiveCompile {
    request: TerrainPipelineRequest,
    handle: JobHandle<Result<TerrainPipelineBundle, String>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TerrainPipelineCompileStatus {
    Idle,
    Pending(TerrainPipelineRequest),
    Failed {
        request: TerrainPipelineRequest,
        message: String,
    },
}

pub(crate) struct TerrainPipelineCompileCoordinator {
    desired: Option<DesiredCompile>,
    active: Option<ActiveCompile>,
    completion: Option<TerrainPipelineCompletion>,
    status: TerrainPipelineCompileStatus,
}

impl Default for TerrainPipelineCompileCoordinator {
    fn default() -> Self {
        Self {
            desired: None,
            active: None,
            completion: None,
            status: TerrainPipelineCompileStatus::Idle,
        }
    }
}

impl TerrainPipelineCompileCoordinator {
    pub(crate) fn request(
        &mut self,
        request: TerrainPipelineRequest,
        compiler: TerrainPipelineCompiler,
    ) {
        if self
            .active
            .as_ref()
            .is_some_and(|active| active.request == request)
            || self
                .desired
                .as_ref()
                .is_some_and(|desired| desired.request == request)
        {
            return;
        }
        self.completion = None;
        self.desired = Some(DesiredCompile { request, compiler });
        if let Some(active) = &self.active {
            active.handle.cancel();
        } else {
            self.start_desired();
        }
    }

    pub(crate) fn cancel_all(&mut self) {
        self.desired = None;
        self.completion = None;
        if let Some(active) = &self.active {
            active.handle.cancel();
        }
        self.status = TerrainPipelineCompileStatus::Idle;
    }

    pub(crate) fn take_completion(&mut self) -> Option<TerrainPipelineCompletion> {
        self.completion.take()
    }

    pub(crate) fn status(&self) -> &TerrainPipelineCompileStatus {
        &self.status
    }

    fn start_desired(&mut self) {
        let Some(desired) = self.desired.take() else {
            return;
        };
        let request = desired.request;
        let compiler = desired.compiler;
        let handle = terra_jobs::spawn_one_shot("terra-infinite-pipeline", move |_ctx| {
            compiler
                .compile(request.variant)
                .map_err(|error| error.to_string())
        });
        self.active = Some(ActiveCompile { request, handle });
        self.status = TerrainPipelineCompileStatus::Pending(request);
    }

    fn finish_active(&mut self, outcome: Result<Result<TerrainPipelineBundle, String>, JobError>) {
        let active = self.active.take().expect("active pipeline compile exists");
        let still_desired = self.desired.is_none();
        if still_desired {
            let result = match outcome {
                Ok(result) => result,
                Err(JobError::Cancelled) => {
                    self.status = TerrainPipelineCompileStatus::Idle;
                    return;
                }
                Err(JobError::Panicked(message)) => {
                    Err(format!("pipeline worker panicked: {message}"))
                }
            };
            match &result {
                Ok(_) => self.status = TerrainPipelineCompileStatus::Idle,
                Err(message) => {
                    self.status = TerrainPipelineCompileStatus::Failed {
                        request: active.request,
                        message: message.clone(),
                    };
                }
            }
            self.completion = Some(TerrainPipelineCompletion {
                request: active.request,
                result,
            });
        }
    }
}

impl Pollable for TerrainPipelineCompileCoordinator {
    fn pump(&mut self) -> Pending {
        let mut redraw = false;
        if let Some(outcome) = self
            .active
            .as_ref()
            .and_then(|active| active.handle.try_take())
        {
            self.finish_active(outcome);
            redraw = true;
            if self.active.is_none() && self.desired.is_some() {
                self.start_desired();
            }
        }
        Pending {
            busy: self.active.is_some(),
            animate: self.active.is_some(),
            redraw,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wait_for_completion(coordinator: &mut TerrainPipelineCompileCoordinator) {
        for _ in 0..10_000 {
            coordinator.pump();
            if coordinator.completion.is_some() {
                return;
            }
            std::thread::yield_now();
        }
        panic!("pipeline coordinator did not finish test job");
    }

    #[test]
    fn request_identity_includes_project_and_device_generations() {
        let base = TerrainPipelineRequest {
            variant: TerrainShaderVariant::Infinite,
            project_generation: 3,
            device_generation: 5,
        };
        assert_ne!(
            base,
            TerrainPipelineRequest {
                project_generation: 4,
                ..base
            }
        );
        assert_ne!(
            base,
            TerrainPipelineRequest {
                device_generation: 6,
                ..base
            }
        );
        assert_ne!(
            base,
            TerrainPipelineRequest {
                variant: TerrainShaderVariant::Bounded,
                ..base
            }
        );
        assert!(base.matches_live(TerrainShaderVariant::Infinite, 3, 5));
        assert!(!base.matches_live(TerrainShaderVariant::Infinite, 4, 5));
    }

    #[test]
    fn worker_failure_becomes_persistent_retryable_status() {
        let request = TerrainPipelineRequest {
            variant: TerrainShaderVariant::Infinite,
            project_generation: 7,
            device_generation: 11,
        };
        let handle = terra_jobs::spawn_one_shot("pipeline-failure-test", |_| {
            Err::<TerrainPipelineBundle, _>("synthetic compile failure".to_string())
        });
        let mut coordinator = TerrainPipelineCompileCoordinator {
            active: Some(ActiveCompile { request, handle }),
            status: TerrainPipelineCompileStatus::Pending(request),
            ..Default::default()
        };
        wait_for_completion(&mut coordinator);
        assert!(matches!(
            coordinator.status(),
            TerrainPipelineCompileStatus::Failed { request: failed, message }
                if *failed == request && message.contains("synthetic compile failure")
        ));
        let completion = coordinator.take_completion().expect("failed completion");
        assert_eq!(completion.request, request);
        match completion.result {
            Err(message) => assert!(message.contains("synthetic compile failure")),
            Ok(_) => panic!("synthetic failure unexpectedly produced a bundle"),
        }
    }
}
