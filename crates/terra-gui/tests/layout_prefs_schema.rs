//! Schema ratchet: `LayoutPrefs` serializes to dock geometry only.
//!
//! terra-gui is an app-agnostic UI toolkit. Re-adding an app- or render-domain
//! field (workspace prefs, viewport render settings) to `LayoutPrefs` — reverting
//! the A2-B1 relocation — changes this key set and fails the test.

use std::collections::BTreeSet;

use terra_gui::LayoutPrefs;

#[test]
fn layout_prefs_serializes_dock_geometry_only() {
    let json = serde_json::to_value(LayoutPrefs::default()).unwrap();
    let keys: BTreeSet<&str> = json
        .as_object()
        .expect("LayoutPrefs serializes to a JSON object")
        .keys()
        .map(String::as_str)
        .collect();

    let expected: BTreeSet<&str> = [
        "tool_panel_w",
        "right_panel_w",
        "layers_frac",
        "tool_panel_collapsed",
        "inspector_collapsed",
        "hide_mode_rail",
        "hide_layers_panel",
        "mask_editor_panel_w",
    ]
    .into_iter()
    .collect();

    assert_eq!(
        keys, expected,
        "LayoutPrefs must serialize dock geometry only; app/render prefs \
         belong in terra-app's EditorPrefs"
    );
}
