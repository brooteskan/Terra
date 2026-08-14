//! On-disk compatibility ratchet for the editor prefs file (A2-B1).
//!
//! `EditorPrefs` `#[serde(flatten)]`s `terra_gui::LayoutPrefs`, so the file stays
//! byte-compatible with the pre-fix blob where dock geometry and app/render prefs
//! shared one flat object. Dropping the flatten shim, or moving the workspace /
//! render fields back into the toolkit, fails these tests.

use std::collections::BTreeSet;

use terra_app::app::prefs::EditorPrefs;

/// A prefs file captured in the pre-fix format: every legacy key at top level, all
/// set to non-default values so a field that lands in the wrong place is visible.
const LEGACY_FIXTURE: &str = r#"{
  "tool_panel_w": 200.0,
  "right_panel_w": 400.0,
  "layers_frac": 0.6,
  "tool_panel_collapsed": true,
  "inspector_collapsed": true,
  "hide_mode_rail": true,
  "hide_layers_panel": true,
  "mask_editor_panel_w": 300.0,
  "preferred_workspace": "simulation",
  "auto_switch_workspace_on_create": true,
  "viewport_render": {
    "mode": "ProgressiveRayTraced",
    "preset": "Quality",
    "target_fps": 30.0,
    "max_spp": 256,
    "dynamic_resolution": false,
    "denoise": false,
    "interactive_spp": 3,
    "settling_spp": 5,
    "refining_spp": 7,
    "max_bounces_interactive": 6,
    "max_bounces_refining": 8,
    "min_internal_scale": 0.25,
    "max_internal_scale": 0.9,
    "history_clamp_k": 2.0,
    "converge_fraction": 0.8,
    "debug_viz_mode": 2
  }
}"#;

#[test]
fn legacy_flat_file_parses_into_editor_prefs() {
    let prefs: EditorPrefs = serde_json::from_str(LEGACY_FIXTURE).unwrap();

    // Flattened dock geometry (owned by terra-gui).
    assert_eq!(prefs.layout.tool_panel_w, 200.0);
    assert_eq!(prefs.layout.right_panel_w, 400.0);
    assert_eq!(prefs.layout.layers_frac, 0.6);
    assert!(prefs.layout.tool_panel_collapsed);
    assert!(prefs.layout.inspector_collapsed);
    assert!(prefs.layout.hide_mode_rail);
    assert!(prefs.layout.hide_layers_panel);
    assert_eq!(prefs.layout.mask_editor_panel_w, 300.0);

    // App-domain fields (relocated out of the toolkit).
    assert_eq!(prefs.preferred_workspace, "simulation");
    assert!(prefs.auto_switch_workspace_on_create);

    // Render-domain fields (relocated out of the toolkit). Numbers cross the
    // serde flatten buffer, so check several of each width.
    assert_eq!(prefs.viewport_render.mode, "ProgressiveRayTraced");
    assert_eq!(prefs.viewport_render.preset, "Quality");
    assert_eq!(prefs.viewport_render.target_fps, 30.0);
    assert_eq!(prefs.viewport_render.max_spp, 256);
    assert!(!prefs.viewport_render.dynamic_resolution);
    assert!(!prefs.viewport_render.denoise);
    assert_eq!(prefs.viewport_render.max_bounces_refining, 8);
    assert_eq!(prefs.viewport_render.min_internal_scale, 0.25);
    assert_eq!(prefs.viewport_render.debug_viz_mode, 2);
}

#[test]
fn editor_prefs_default_serializes_legacy_flat_key_set() {
    let json = serde_json::to_value(EditorPrefs::default()).unwrap();
    let keys: BTreeSet<&str> = json
        .as_object()
        .expect("EditorPrefs serializes to a flat JSON object")
        .keys()
        .map(String::as_str)
        .collect();

    let expected: BTreeSet<&str> = [
        // Flattened dock geometry.
        "tool_panel_w",
        "right_panel_w",
        "layers_frac",
        "tool_panel_collapsed",
        "inspector_collapsed",
        "hide_mode_rail",
        "hide_layers_panel",
        "mask_editor_panel_w",
        // App/render prefs at the same flat level (flatten shim intact).
        "preferred_workspace",
        "auto_switch_workspace_on_create",
        "viewport_render",
    ]
    .into_iter()
    .collect();

    assert_eq!(
        keys, expected,
        "EditorPrefs must serialize to the pre-fix flat key set"
    );
}
