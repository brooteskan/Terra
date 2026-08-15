//! Terra editor application shell.

use terra_app::app::TerraApp;
use winit::event_loop::{ControlFlow, EventLoop};

fn main() {
    if let Some(exit_code) = handle_release_cli() {
        std::process::exit(exit_code);
    }

    let _logging = terra_app::logging::init();
    harden_gpu_env();

    let event_loop = EventLoop::new().expect("event loop");
    // Wait on OS events; about_to_wait arms WaitUntil only while refining.
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = TerraApp::default();
    event_loop.run_app(&mut app).expect("run app");
}

fn handle_release_cli() -> Option<i32> {
    let mut args = std::env::args().skip(1);
    let first = args.next()?;
    if args.next().is_some() {
        eprintln!("unexpected extra arguments");
        return Some(2);
    }

    match first.as_str() {
        "--version" | "-V" => {
            println!("Terra {}", env!("CARGO_PKG_VERSION"));
            Some(0)
        }
        "--self-check" => {
            if let Err(err) = run_self_check() {
                eprintln!("Terra self-check failed: {err}");
                Some(1)
            } else {
                println!("Terra {} self-check ok", env!("CARGO_PKG_VERSION"));
                Some(0)
            }
        }
        "--help" | "-h" => {
            println!("Terra {}", env!("CARGO_PKG_VERSION"));
            println!("Usage: terra [--version] [--self-check]");
            Some(0)
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
