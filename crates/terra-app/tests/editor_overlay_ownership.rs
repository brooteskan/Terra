use std::path::PathBuf;

fn source(relative: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative);
    std::fs::read_to_string(path).expect("source file")
}

#[test]
fn editor_overlays_are_app_owned_and_submitted_before_gui() {
    let renderer = source("../terra-render/src/lib.rs");
    assert!(!renderer.contains("pub brush: BrushOverlay"));
    assert!(!renderer.contains("pub guides: GuideOverlay"));
    assert!(!renderer.contains("PassKind::Overlays"));

    let app = source("src/app/mod.rs");
    assert!(app.contains("editor_overlays: Option<editor_overlays::EditorOverlays>"));

    let redraw = source("src/app/redraw.rs");
    let terrain = redraw
        .find("renderer.render_terrain()")
        .expect("terrain submit");
    let overlays = redraw
        .find("editor_overlays.render(gpu, renderer, &view)")
        .expect("editor overlay submit");
    let gui = redraw
        .find("gui_renderer.render(")
        .expect("GUI submit after overlays");
    assert!(terrain < overlays && overlays < gui);
}
