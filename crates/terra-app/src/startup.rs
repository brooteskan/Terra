//! Startup error handling: typed errors, diagnostics, and the boot-failure
//! splash policy.
//!
//! ## Smoke-test procedure (debug builds)
//!
//! Set `TERRA_STARTUP_FAULT` to one of `event-loop`, `window`, `gpu-init`, or
//! `boot-worker` and launch the debug binary. Each triggers the corresponding
//! failure path:
//!
//! | Value          | Expected behavior                                         |
//! |----------------|-----------------------------------------------------------|
//! | `event-loop`   | stderr + dialog, exit 1                                   |
//! | `window`       | stderr + dialog, exit 1                                   |
//! | `gpu-init`     | stderr + dialog, exit 1                                   |
//! | `boot-worker`  | splash flips to failure frame; key/click/Alt+F4 → exit 1  |

use std::path::Path;

use thiserror::Error;

/// Why Terra could not start.
#[derive(Debug, Error)]
pub enum StartupError {
    #[error("failed to create event loop: {0}")]
    EventLoop(winit::error::EventLoopError),

    #[error("event loop exited with error: {0}")]
    RunApp(winit::error::EventLoopError),

    #[error("failed to create window: {0}")]
    Window(winit::error::OsError),

    #[error("GPU initialization failed: {0}")]
    Gpu(terra_render::RenderError),

    #[error("GPU pipeline build failed: {0}")]
    BootWorker(terra_jobs::JobError),
}

impl StartupError {
    pub fn advice(&self) -> &'static str {
        match self {
            StartupError::EventLoop(_) | StartupError::RunApp(_) => {
                "The OS event loop could not be created or ran into an unrecoverable error. \
                 Try restarting your session or updating your display drivers."
            }
            StartupError::Window(_) => {
                "The application window could not be created. Check that a display is \
                 connected and your desktop session is running."
            }
            StartupError::Gpu(_) => {
                "No compatible GPU adapter or device was found. Update your graphics \
                 drivers and ensure a DirectX 12 or Vulkan capable GPU is available. \
                 You can try setting WGPU_BACKEND=vulkan or WGPU_BACKEND=dx12 to force \
                 a specific backend."
            }
            StartupError::BootWorker(_) => {
                "The GPU shader/pipeline build crashed. This usually indicates a driver \
                 bug. Update your graphics drivers and try again."
            }
        }
    }
}

/// Report a startup failure through all available channels.
///
/// - Always: `log::error!` + `eprintln!` (with log-file path when available).
/// - When `show_dialog` is true (no GUI surface to show it on): an `rfd` native
///   message dialog blocks until dismissed. When false the caller is responsible
///   for a visible diagnostic (the boot-failure splash).
pub fn report_failure(error: &StartupError, log_file: Option<&Path>, show_dialog: bool) {
    let detail = format!("{error}");
    let advice = error.advice();
    let log_hint = match log_file {
        Some(path) => format!("\n\nLog file: {}", path.display()),
        None => String::new(),
    };

    log::error!("{detail}\n{advice}{log_hint}");
    eprintln!("Terra: {detail}\n{advice}{log_hint}");

    if show_dialog {
        rfd::MessageDialog::new()
            .set_level(rfd::MessageLevel::Error)
            .set_title("Terra — startup failed")
            .set_description(format!("{detail}\n\n{advice}{log_hint}"))
            .show();
    }
}

/// Result of polling `BootState::job` through the typed policy layer.
pub(crate) enum BootPoll<T> {
    /// Worker still running.
    Pending,
    /// Worker produced a value.
    Ready(T),
    /// Worker failed — the error should be surfaced to the user.
    Failed(StartupError),
    /// Worker was cancelled — treat as clean shutdown.
    Shutdown,
}

/// Classify a `JobHandle::try_take` result into the app's startup policy.
pub(crate) fn classify_boot_poll<T>(poll: Option<Result<T, terra_jobs::JobError>>) -> BootPoll<T> {
    match poll {
        None => BootPoll::Pending,
        Some(Ok(value)) => BootPoll::Ready(value),
        Some(Err(terra_jobs::JobError::Cancelled)) => BootPoll::Shutdown,
        Some(Err(error)) => BootPoll::Failed(StartupError::BootWorker(error)),
    }
}

/// Build the text lines for the boot-failure splash frame.
pub(crate) fn failure_splash_lines_with_telemetry(
    error: &StartupError,
    log_file: Option<&Path>,
    telemetry: Option<&terra_telemetry::CompilationSnapshot>,
) -> Vec<String> {
    let mut lines = Vec::with_capacity(5);
    lines.push(format!("{error}"));
    if let Some(failed) = telemetry.and_then(|snapshot| snapshot.last_failure()) {
        lines.push(format!(
            "Failed during {}: {} ({})",
            failed.kind.as_str(),
            failed.label,
            format_elapsed(failed.duration)
        ));
    }
    lines.push(String::new());
    lines.push(error.advice().to_string());
    if let Some(path) = log_file {
        lines.push(String::new());
        lines.push(format!("Log file: {}", path.display()));
    }
    lines.push(String::new());
    lines.push("Press any key or click to close.".to_string());
    lines
}

/// User-facing live startup status. Pipeline identity takes precedence over a
/// containing coarse stage so a long synchronous compile is never hidden.
pub(crate) fn startup_status(snapshot: &terra_telemetry::CompilationSnapshot) -> String {
    if let Some(active) = snapshot.active_pipeline() {
        let pipeline_count = snapshot
            .active
            .iter()
            .filter(|entry| entry.kind.is_pipeline())
            .count();
        let kind = match active.kind {
            terra_telemetry::CompilationKind::RenderPipeline => "render pipeline",
            terra_telemetry::CompilationKind::ComputePipeline => "compute pipeline",
            terra_telemetry::CompilationKind::StartupStage => unreachable!(),
        };
        let more = if pipeline_count > 1 {
            format!(" (+{} more)", pipeline_count - 1)
        } else {
            String::new()
        };
        return format!(
            "Compiling {kind}: {} ({}){more}",
            active.label,
            format_elapsed(active.elapsed)
        );
    }
    if let Some(stage) = snapshot.active_stage() {
        return format!("{}...", stage.label);
    }
    if snapshot.completed.is_empty() {
        "Preparing GPU startup...".to_string()
    } else {
        "Finalizing GPU startup...".to_string()
    }
}

fn format_elapsed(elapsed: std::time::Duration) -> String {
    if elapsed.as_secs_f64() < 1.0 {
        format!("{} ms", elapsed.as_millis())
    } else {
        format!("{:.1} s", elapsed.as_secs_f64())
    }
}

/// Debug-only fault injection for the smoke-test matrix.
///
/// Returns `true` when `TERRA_STARTUP_FAULT` matches `stage`. Compiled out of
/// release builds entirely.
#[cfg(debug_assertions)]
pub fn injected_fault(stage: &str) -> bool {
    std::env::var("TERRA_STARTUP_FAULT")
        .ok()
        .is_some_and(|v| v.eq_ignore_ascii_case(stage))
}

#[cfg(not(debug_assertions))]
pub fn injected_fault(_stage: &str) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};
    use terra_jobs::JobError;

    fn telemetry_isolated() -> std::sync::MutexGuard<'static, ()> {
        static TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        TEST_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[test]
    fn startup_error_variants_are_distinguishable() {
        // OsError::new is pub(crate) in winit, so Window is tested only via
        // the smoke matrix. The other three are constructible.
        let event_loop = StartupError::EventLoop(winit::error::EventLoopError::ExitFailure(1));
        let gpu = StartupError::Gpu(terra_render::RenderError::Msg("no adapter".into()));
        let boot = StartupError::BootWorker(JobError::Panicked("boom".into()));

        let msgs: Vec<String> = [&event_loop, &gpu, &boot]
            .iter()
            .map(|e| format!("{e}"))
            .collect();
        assert!(msgs[0].contains("event loop"));
        assert!(msgs[1].contains("GPU initialization"));
        assert!(msgs[2].contains("pipeline build"));
        assert!(msgs[1].contains("no adapter"));
        assert!(msgs[2].contains("boom"));
    }

    #[test]
    fn classify_pending() {
        let poll: Option<Result<u32, JobError>> = None;
        assert!(matches!(classify_boot_poll(poll), BootPoll::Pending));
    }

    #[test]
    fn classify_ready() {
        let poll = Some(Ok(42u32));
        match classify_boot_poll(poll) {
            BootPoll::Ready(v) => assert_eq!(v, 42),
            other => panic!("expected Ready, got {:?}", variant_name(&other)),
        }
    }

    #[test]
    fn classify_panicked_is_failed() {
        let poll: Option<Result<u32, _>> = Some(Err(JobError::Panicked("segfault".into())));
        match classify_boot_poll(poll) {
            BootPoll::Failed(StartupError::BootWorker(JobError::Panicked(msg))) => {
                assert!(msg.contains("segfault"));
            }
            other => panic!(
                "expected Failed(BootWorker(Panicked)), got {:?}",
                variant_name(&other)
            ),
        }
    }

    #[test]
    fn classify_cancelled_is_shutdown() {
        let poll: Option<Result<u32, _>> = Some(Err(JobError::Cancelled));
        assert!(matches!(classify_boot_poll(poll), BootPoll::Shutdown));
    }

    #[test]
    fn failure_splash_lines_include_error_and_advice() {
        let error = StartupError::BootWorker(JobError::Panicked("oops".into()));
        let lines = failure_splash_lines_with_telemetry(&error, None, None);
        let joined = lines.join("\n");
        assert!(joined.contains("oops"), "error text missing");
        assert!(joined.contains("driver"), "advice missing");
        assert!(joined.contains("Press any key"), "dismiss hint missing");
    }

    #[test]
    fn failure_splash_lines_include_log_path() {
        let error = StartupError::Gpu(terra_render::RenderError::Msg("nope".into()));
        let lines =
            failure_splash_lines_with_telemetry(&error, Some(Path::new("/tmp/terra.log")), None);
        let joined = lines.join("\n");
        assert!(joined.contains("/tmp/terra.log"));
    }

    #[test]
    fn startup_status_prefers_exact_pipeline_over_coarse_stage() {
        let _isolated = telemetry_isolated();
        terra_telemetry::reset();
        let stage = terra_telemetry::begin_stage("Building renderer");
        let pipeline = terra_telemetry::begin(
            terra_telemetry::CompilationKind::RenderPipeline,
            "terrain-bounded-pipe",
        );
        let status = startup_status(&terra_telemetry::snapshot());
        assert!(status.contains("render pipeline: terrain-bounded-pipe"));
        drop(pipeline);
        drop(stage);
    }

    #[test]
    fn startup_status_uses_coarse_stage_when_no_pipeline_is_active() {
        let _isolated = telemetry_isolated();
        terra_telemetry::reset();
        let stage = terra_telemetry::begin_stage("Building terrain evaluator");
        assert_eq!(
            startup_status(&terra_telemetry::snapshot()),
            "Building terrain evaluator..."
        );
        stage.complete();
    }

    #[test]
    fn failure_lines_retain_failed_pipeline_label() {
        let _isolated = telemetry_isolated();
        terra_telemetry::reset();
        terra_telemetry::begin(
            terra_telemetry::CompilationKind::ComputePipeline,
            "pyramid-error",
        )
        .fail();
        let snapshot = terra_telemetry::snapshot();
        let error = StartupError::BootWorker(JobError::Panicked("boom".into()));
        let joined = failure_splash_lines_with_telemetry(&error, None, Some(&snapshot)).join("\n");
        assert!(joined.contains("pyramid-error"));
    }

    fn variant_name<T>(poll: &BootPoll<T>) -> &'static str {
        match poll {
            BootPoll::Pending => "Pending",
            BootPoll::Ready(_) => "Ready",
            BootPoll::Failed(_) => "Failed",
            BootPoll::Shutdown => "Shutdown",
        }
    }
}
