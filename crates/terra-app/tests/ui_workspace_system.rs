//! Non-linear workspace system: presentation focus without project/eval mutation.

use terra_app::app::prefs::EditorPrefs;
use terra_app::ui::{
    all_tools, tools_for_workspace, workspace_definition, UiState, WorkspaceId, WorkspaceMode,
};
use terra_core::document::TerrainDocument;
use terra_core::eval::EvalScheduler;
use terra_core::layer::{LayerKind, MaterialsParams};

fn selection_snapshot(
    doc: &TerrainDocument,
) -> (
    Option<terra_core::layer::LayerId>,
    Option<terra_core::biome_paint::BiomeLayerId>,
) {
    (doc.selected, doc.selected_biome_layer)
}

#[test]
fn switch_all_workspaces_preserves_project_and_selection() {
    let mut doc = TerrainDocument::new_default();
    doc.selected = doc.stack.layer_ids().first().copied().or(doc.selected);
    let before_json = doc.to_json().unwrap();
    let before_sel = selection_snapshot(&doc);

    let mut ui = UiState::default();
    ui.camera_xz = (0.33, 0.66);
    ui.camera_yaw = 0.8;
    ui.camera_pitch = 0.4;
    let cam = (ui.camera_xz, ui.camera_yaw, ui.camera_pitch);

    for id in WorkspaceId::ALL {
        ui.switch_workspace(id);
        assert_eq!(ui.active_workspace, id);
        assert_eq!(doc.to_json().unwrap(), before_json);
        assert_eq!(selection_snapshot(&doc), before_sel);
        assert_eq!(ui.camera_xz, cam.0);
        assert!((ui.camera_yaw - cam.1).abs() < 1e-6);
        assert!((ui.camera_pitch - cam.2).abs() < 1e-6);
    }
}

#[test]
fn switch_does_not_invalidate_terrain() {
    let mut ui = UiState::default();
    let mut scheduler = EvalScheduler::new();
    let token = scheduler.request_rebuild();
    for id in WorkspaceId::ALL {
        ui.switch_workspace(id);
    }
    assert_eq!(scheduler.current_token, token);
}

#[test]
fn preferred_workspace_persists_in_editor_prefs() {
    let mut ui = UiState::default();
    ui.switch_workspace(WorkspaceId::Simulation);
    assert_eq!(ui.preferred_workspace, "simulation");
    assert!(ui.layout_dirty);

    // Assemble the on-disk prefs the way the app's save seam does, then round-trip.
    let prefs = EditorPrefs {
        layout: ui.layout.clone(),
        preferred_workspace: ui.preferred_workspace.clone(),
        auto_switch_workspace_on_create: ui.auto_switch_workspace_on_create,
        viewport_render: ui.viewport_render.to_prefs(),
    };
    let json = serde_json::to_string(&prefs).unwrap();
    let back: EditorPrefs = serde_json::from_str(&json).unwrap();
    assert_eq!(back.preferred_workspace, "simulation");

    let mut ui2 = UiState::default();
    ui2.preferred_workspace = back.preferred_workspace;
    ui2.apply_preferred_workspace_from_prefs();
    assert_eq!(ui2.active_workspace, WorkspaceId::Simulation);
}

#[test]
fn legacy_all_tools_preference_loads_as_objects() {
    let prefs = EditorPrefs {
        preferred_workspace: "all_tools".to_string(),
        ..EditorPrefs::default()
    };
    let json = serde_json::to_string(&prefs).unwrap();
    let loaded: EditorPrefs = serde_json::from_str(&json).unwrap();

    let mut ui = UiState::default();
    ui.preferred_workspace = loaded.preferred_workspace;
    ui.apply_preferred_workspace_from_prefs();

    assert_eq!(ui.active_workspace, WorkspaceId::Objects);
}

#[test]
fn all_tools_removed_from_rail() {
    assert!(!WorkspaceId::ALL.contains(&WorkspaceId::AllTools));
    // Legacy catalog path still resolves for tests / remaps.
    let visible = tools_for_workspace(WorkspaceId::AllTools).len();
    assert_eq!(visible, all_tools().len());
}

#[test]
fn unrelated_item_remains_editable_from_any_workspace() {
    // Selecting a Materials layer while in Sculpt must not be blocked by workspace.
    let mut doc = TerrainDocument::new_default();
    let mut mats = MaterialsParams::default();
    mats.rules.clear();
    let layer = terra_core::layer::Layer::new("Mat", LayerKind::Materials(mats));
    let mat_id = layer.id();
    doc.stack.push(layer);
    doc.selected = Some(mat_id);

    let mut ui = UiState::default();
    ui.switch_workspace(WorkspaceId::Sculpt);
    assert_eq!(doc.selected, Some(mat_id));
    // Inspector path is selection-driven; workspace must not clear selection.
    ui.switch_workspace(WorkspaceId::Biomes);
    assert_eq!(doc.selected, Some(mat_id));

    // Simulation selected from Biomes focus — still selected.
    let sim =
        terra_core::layer::Layer::new("Hydro", LayerKind::HydraulicErosion(Default::default()));
    let sim_id = sim.id();
    doc.stack.push(sim);
    doc.selected = Some(sim_id);
    ui.switch_workspace(WorkspaceId::Biomes);
    assert_eq!(doc.selected, Some(sim_id));
}

#[test]
fn workspace_definitions_have_no_wizard_copy() {
    for def in terra_app::ui::all_workspace_definitions() {
        let blob = format!("{} {} {}", def.name, def.description, def.tools_heading());
        let lower = blob.to_ascii_lowercase();
        for banned in [
            "step",
            "stage complete",
            "continue",
            "required next",
            "finish this",
        ] {
            assert!(!lower.contains(banned), "banned wording in {}", def.name);
        }
    }
}

#[test]
fn legacy_mode_maps_into_workspace_id() {
    assert_eq!(
        WorkspaceMode::Terrain.to_workspace_id(),
        WorkspaceId::Sculpt
    );
    assert_eq!(
        WorkspaceMode::Materials.to_workspace_id(),
        WorkspaceId::Surface
    );
    assert_eq!(WorkspaceMode::Masks.to_workspace_id(), WorkspaceId::Rules);
    let _ = workspace_definition(WorkspaceId::Develop);
}
