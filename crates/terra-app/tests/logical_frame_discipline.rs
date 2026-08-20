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
        "request_redraw(",
        "renderer.resize(",
        "renderer.reconfigure(",
    ] {
        assert!(
            !callback.contains(forbidden),
            "window_event must remain capture-only; found `{forbidden}`"
        );
    }
}

#[test]
fn redraw_callback_is_presentation_only() {
    let source =
        fs::read_to_string(manifest_dir().join("src/app/redraw.rs")).expect("read redraw source");
    let callback = source
        .split_once("pub(crate) fn redraw(")
        .and_then(|(_, rest)| rest.split_once("pub(crate) fn apply_pending_ui_effects("))
        .map(|(body, _)| body)
        .expect("locate redraw callback body");

    for forbidden in [
        "run_eval_step(",
        "run_eval_step_with_intent(",
        "apply_actions(",
        "perform_project_action(",
        "request_rebuild(",
        "request_rebuild_immediate(",
        "notify_invalidation(",
        "set_renderer_mode(",
        "set_display_aids(",
        "request_redraw(",
        "renderer.resize(",
        "renderer.reconfigure(",
        "prepare_presentation(",
    ] {
        assert!(
            !callback.contains(forbidden),
            "redraw must remain presentation-only; found `{forbidden}`"
        );
    }
}

#[test]
fn direct_redraw_requests_are_confined_to_the_lifecycle_adapter_and_bootstrap() {
    let app_dir = manifest_dir().join("src/app");
    for entry in fs::read_dir(&app_dir).expect("read app source directory") {
        let path = entry.expect("directory entry").path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        let source = fs::read_to_string(&path).expect("read app source file");
        let count = source.matches("request_redraw(").count();
        if path.file_name().and_then(|name| name.to_str()) == Some("lifecycle.rs") {
            assert_eq!(
                count, 3,
                "lifecycle keeps two bootstrap requests and one adapter"
            );
        } else {
            assert_eq!(
                count,
                0,
                "direct redraw request escaped into {}",
                path.display()
            );
        }
    }
}

#[test]
fn refinement_rechecks_input_and_generation_before_advance_and_publication() {
    let source =
        fs::read_to_string(manifest_dir().join("src/app/eval.rs")).expect("read eval source");
    let advance = source
        .split_once("pub(crate) fn advance_gpu_refinement(")
        .and_then(|(_, rest)| rest.split_once("pub(crate) fn supersede_gpu_refinement("))
        .map(|(body, _)| body)
        .expect("locate refinement advancement body");

    assert!(
        advance.matches("self.input.has_pending()").count() >= 2,
        "input must gate both refinement advancement and final publication"
    );
    assert!(
        advance.matches("job.is_fresh(self.eval_token)").count() >= 2,
        "generation freshness must be rechecked before advancement and publication"
    );
    assert!(
        advance.contains("publish_compiled_refinement("),
        "the guarded path must include final candidate publication"
    );
}

fn manifest_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}
