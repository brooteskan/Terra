//! Source-level guard for the capture-only winit input boundary.

use std::fs;
use std::path::{Path, PathBuf};

#[test]
fn window_input_callback_does_not_start_application_or_evaluation_work() {
    let source = fs::read_to_string(manifest_dir().join("src/app/lifecycle.rs"))
        .expect("read app lifecycle source");
    let callback = source
        .split_once("fn window_event(")
        .and_then(|(_, rest)| rest.split_once("fn about_to_wait("))
        .map(|(body, _)| body)
        .expect("locate window_event callback");

    for forbidden in [
        "paint_at_cursor(",
        "flush_live_paint_preview(",
        "run_eval_step(",
        "run_eval_step_with_intent(",
        "request_rebuild(",
        "apply_actions(",
        "dispatch_command(",
    ] {
        assert!(
            !callback.contains(forbidden),
            "window_event must remain capture-only; found `{forbidden}`"
        );
    }
}

fn manifest_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}
