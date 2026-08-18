//! Terra editor application shell.

use std::process::ExitCode;

use terra_app::app::TerraApp;
use terra_app::startup::{self, StartupError};
use winit::event_loop::{ControlFlow, EventLoop};

fn main() -> ExitCode {
    if let Some(code) = handle_release_cli() {
        return code;
    }

    let logging = terra_app::logging::init();
    harden_gpu_env();

    if startup::injected_fault("event-loop") {
        let error = StartupError::EventLoop(winit::error::EventLoopError::ExitFailure(1));
        startup::report_failure(&error, logging.log_file(), true);
        return ExitCode::FAILURE;
    }

    let event_loop = match EventLoop::new() {
        Ok(el) => el,
        Err(error) => {
            let error = StartupError::EventLoop(error);
            startup::report_failure(&error, logging.log_file(), true);
            return ExitCode::FAILURE;
        }
    };
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = TerraApp::default();
    let run_result = event_loop.run_app(&mut app);

    // Check for a startup failure stored by the ApplicationHandler (surfaces
    // 3–5: window, init_gpu, boot worker). `presented` is true when the
    // boot-failure splash already showed the error — skip the dialog.
    if let Some((error, presented)) = app.take_startup_failure() {
        startup::report_failure(&error, logging.log_file(), !presented);
        return ExitCode::FAILURE;
    }

    match run_result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            let error = StartupError::RunApp(error);
            startup::report_failure(&error, logging.log_file(), true);
            ExitCode::FAILURE
        }
    }
}

fn handle_release_cli() -> Option<ExitCode> {
    let mut args = std::env::args().skip(1);
    let first = args.next()?;
    if args.next().is_some() {
        eprintln!("unexpected extra arguments");
        return Some(ExitCode::from(2));
    }

    match first.as_str() {
        "--version" | "-V" => {
            println!("Terra {}", env!("CARGO_PKG_VERSION"));
            Some(ExitCode::SUCCESS)
        }
        "--self-check" => {
            if let Err(err) = run_self_check() {
                eprintln!("Terra self-check failed: {err}");
                Some(ExitCode::FAILURE)
            } else {
                println!("Terra {} self-check ok", env!("CARGO_PKG_VERSION"));
                Some(ExitCode::SUCCESS)
            }
        }
        "--help" | "-h" => {
            println!("Terra {}", env!("CARGO_PKG_VERSION"));
            println!("Usage: terra [--version] [--self-check]");
            Some(ExitCode::SUCCESS)
        }
        _ => None,
    }
}

fn run_self_check() -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|err| format!("current exe unavailable: {err}"))?;
    if !exe.exists() {
        return Err(format!(
            "current exe path does not exist: {}",
            exe.display()
        ));
    }

    if env!("CARGO_PKG_VERSION").trim().is_empty() {
        return Err("package version is empty".into());
    }

    Ok(())
}

/// Soften Vulkan capture-overlay damage when `WGPU_BACKEND=vulkan` is forced.
///
/// OBS / Overwolf / Medal register implicit layers that have stack-overflowed
/// `vkCreateDevice` on this machine. DX12 is the Windows default; these env
/// vars only matter on the Vulkan path.
fn harden_gpu_env() {
    const CAPTURE_DISABLE: &[(&str, &str)] = &[
        ("DISABLE_VULKAN_OBS_CAPTURE", "1"),
        ("DISABLE_VULKAN_OW_OBS_CAPTURE", "1"),
        ("DISABLE_VULKAN_OW_OVERLAY_LAYER", "1"),
        ("DISABLE_VULKAN_MEDAL_OBS_CAPTURE", "1"),
    ];
    for &(key, value) in CAPTURE_DISABLE {
        if std::env::var_os(key).is_none() {
            // SAFETY: called once at process start before other threads.
            unsafe { std::env::set_var(key, value) };
        }
    }
    if std::env::var_os("VK_LOADER_LAYERS_DISABLE").is_none() {
        unsafe {
            std::env::set_var(
                "VK_LOADER_LAYERS_DISABLE",
                "~VK_LAYER_OBS_HOOK:~VK_LAYER_OW_OBS_HOOK:~VK_LAYER_OW_Overlay:~VK_LAYER_MEDAL_HOOK",
            );
        }
    }
}
