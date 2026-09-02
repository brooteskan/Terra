//! App-owned editor preferences.
//!
//! The application owns its persisted-preferences schema; the `terra-gui` toolkit
//! owns only dock geometry ([`terra_gui::LayoutPrefs`]). [`EditorPrefs`] is the disk
//! root: it flattens the toolkit's dock geometry alongside Terra's app- and
//! render-domain settings, so the on-disk JSON stays a flat key set while ownership
//! of each domain lives in the crate that defines it.

use std::path::PathBuf;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use terra_gui::LayoutPrefs;

use crate::ui::ViewportRenderPrefs;

/// Persisted editor preferences (dock geometry + workspace + viewport render).
///
/// `layout` is `#[serde(flatten)]`ed so the file remains a single flat object with
/// the same keys the pre-split `terra_gui::LayoutPrefs` produced — no migration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditorPrefs {
    #[serde(flatten)]
    pub layout: LayoutPrefs,
    /// Preferred task workspace id (`sculpt`, `biomes`, …). Editor preference only.
    #[serde(default = "default_preferred_workspace")]
    pub preferred_workspace: String,
    /// When true, creating an entity may switch to its home workspace.
    /// Default false — artists stay in the current workspace unless they opt in.
    #[serde(default)]
    pub auto_switch_workspace_on_create: bool,
    /// Persisted viewport render quality settings (user preference).
    #[serde(default)]
    pub viewport_render: ViewportRenderPrefs,
}

fn default_preferred_workspace() -> String {
    "sculpt".into()
}

impl Default for EditorPrefs {
    fn default() -> Self {
        Self {
            layout: LayoutPrefs::default(),
            preferred_workspace: default_preferred_workspace(),
            auto_switch_workspace_on_create: false,
            viewport_render: ViewportRenderPrefs::default(),
        }
    }
}

pub(crate) fn editor_prefs_path() -> PathBuf {
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("terra_layout.json")
}

pub(crate) fn load_editor_prefs() -> EditorPrefs {
    let path = editor_prefs_path();
    if let Ok(bytes) = std::fs::read(&path) {
        if let Ok(prefs) = serde_json::from_slice::<EditorPrefs>(&bytes) {
            let mut prefs = prefs;
            prefs.layout.clamp_mut();
            return prefs;
        }
    }
    EditorPrefs::default()
}

/// Queue editor-prefs persistence without placing filesystem latency on the render/UI thread.
/// A [`terra_jobs::Debounced`] worker coalesces queued snapshots to the latest before each
/// write, so a splitter drag's burst of saves collapses into a single file write.
pub(crate) fn save_editor_prefs(prefs: &EditorPrefs) {
    static SAVER: OnceLock<terra_jobs::Debounced<EditorPrefs>> = OnceLock::new();
    SAVER
        .get_or_init(|| {
            terra_jobs::Debounced::spawn("terra-layout-save", |latest: EditorPrefs| {
                let path = editor_prefs_path();
                if let Ok(json) = serde_json::to_vec_pretty(&latest) {
                    let _ = std::fs::write(path, json);
                }
            })
        })
        .submit(prefs.clone());
}
