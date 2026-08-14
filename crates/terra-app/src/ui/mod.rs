//! Terra product UI — editor chrome built on `terra-gui`.

mod actions;
mod bookmarks_gui;
mod brand;
mod chrome_gui;
mod chrome_layout;
mod command_palette;
mod command_registry;
mod contextual_create_gui;
mod dist_kinds;
mod dock_gui;
mod hierarchy;
mod hierarchy_view;
mod history_gui;
mod inspector;
mod inspector_schema;
mod panels;
mod pipeline_gui;
mod presets;
mod project_home_gui;
mod quick_add;
mod recipe;
mod style;
mod thumbnails;
mod tool_catalog;
mod tool_thumbs;
mod tools_gui;
mod viewport_gui;
mod workspace;

/// Poll completion of lazily decoded tool thumbnails.
pub fn take_tool_thumbnail_ready_signal() -> bool {
    tool_thumbs::take_ready_signal()
}
/// Keep the event loop awake only while lazy thumbnail work is outstanding.
pub fn tool_thumbnails_pending() -> bool {
    tool_thumbs::has_pending_work()
}
/// Warm the tool-thumb decode pool (call once after the window is up).
pub fn prefetch_tool_thumbnails() {
    tool_thumbs::prefetch_all();
}

pub use actions::{MaskEditAction, PanelAction, TerrainSettingsUpdate};
pub use chrome_gui::{
    apply_borderless_window_frame, caption_controls_width, draw_caption_controls,
    draw_export_unsupported_modal, ChromeGuiState,
};
pub use command_palette::{draw_command_palette, CommandPaletteState, PaletteAction};
pub(crate) use command_registry::resolve_workspace_command;
pub use command_registry::{
    commands, format_shortcuts, fuzzy_match, resolve_shortcut, resolve_shortcut_for_input,
    BindingVisibility, CommandCategory, CommandDef, CommandId, ShortcutBinding, ShortcutChord,
    ShortcutModifiers,
};
pub use contextual_create_gui::{
    create_to_workspace, draw_viewport_context_menu, hierarchy_add_actions, workspace_to_create,
    ViewportContextMenu,
};
pub use dock_gui::DockGuiState;
pub use hierarchy::{
    draw_layers_gui, hierarchy_presentation_snapshot, LayerDragSource, LayerPresentationState,
    LayersGuiState,
};
pub use hierarchy_view::{
    advanced_placement_id, biome_section_artist_label, hierarchy_identity_ids, mask_stack_row_id,
    world_rule_entity_meta, world_rules_meta, ArtistConcept, TERRAIN_ROOT, WORLD_SECTION,
};
pub use inspector::{draw_inspector_gui, InspectorGuiState};
pub use inspector_schema::InspectorSection;
pub use panels::{draw_windows, WindowsGuiState};
pub use presets::{
    builtin_presets, contextual_presets, layers_from_preset, layers_from_project_template,
    project_template_by_id, project_templates, world_design_templates, ContextualPreset,
    LayerPreset, ProjectTemplate,
};
pub use project_home_gui::{
    draw_discard_confirm, draw_new_project_templates, draw_project_home, DiscardConfirmChoice,
    NewProjectTemplateChoice, NewWorldSettings, ProjectHomeAction, ProjectHomeGuiState,
    ProjectPrefs, RecentProject, RecentViewMode,
};
pub use quick_add::{draw_quick_add, QuickAddState};
pub use recipe::{builtin_recipes, instantiate_recipe, GroupRecipe, RecipeId, RecipeInstance};
pub use tool_catalog::{
    all_tools, quick_add_entries, tools_for_mode, tools_for_workspace, ToolAction, ToolDef,
};
pub use tools_gui::{draw_tools_gui, LayerToolDrag, ToolsGuiState};
pub use viewport_gui::ViewportToolMode;
pub use workspace::{
    all_workspace_definitions, hierarchy_dim_for_kind, workspace_definition, AppWorkspace,
    BrushWorkspaceState, ContextualActionId, HierarchyEmphasis, TempSoloState, WorkspaceDefinition,
    WorkspaceId, WorkspaceMode, WorkspaceState, WorkspaceToolFilter,
};

use serde::{Deserialize, Serialize};
use terra_core::document::TerrainDocument;
use terra_core::eval::PreviewQuality;
use terra_core::layer::{LayerId, LayerKind};
use terra_core::mask::MaskId;
use terra_gui::GuiContext;

use crate::ui::chrome_gui::{draw_menu_bar, draw_menu_overlays};
use crate::ui::dock_gui::draw_bottom_dock;
use crate::ui::viewport_gui::draw_viewport_overlays;

#[derive(Default)]
pub struct UiState {
    pub show_mask_editor: bool,
    /// Set when leaving Mask view so the app can apply/close the edit session.
    pub pending_close_mask_editor: bool,
    pub show_content_browser: bool,
    pub show_export: bool,
    /// Stub dialog when Export is clicked (feature not ready).
    pub show_export_unsupported: bool,
    pub show_2d_preview: bool,
    pub show_profiler: bool,
    pub show_widget_lab: bool,
    pub preview_mode: Preview2dMode,
    pub editor_tool: EditorTool,
    /// Last height/sculpt brush armed from the catalog or tool bar (restored by Sculpt).
    pub last_sculpt_tool: Option<EditorTool>,
    /// Active task workspace (non-linear focus — not a workflow step).
    pub active_workspace: WorkspaceId,
    /// Legacy tool-catalog category (derived from active workspace).
    pub workspace_mode: WorkspaceMode,
    /// Application workspace tab (Terrain / Materials / â€¦).
    pub app_workspace: AppWorkspace,
    /// When false, inspector shows a short Simple param set (default).
    pub inspector_advanced: bool,
    /// Sculpt brush radius in normalized UV (Base raise/lower/smooth).
    pub sculpt_radius: f32,
    /// Raise/lower peak meters per stamp, or smooth blend 0â€“1.
    pub sculpt_strength: f32,
    /// Brush edge hardness: 0 is soft and 1 is hard.
    pub brush_falloff: f32,
    /// Fraction of the radius travelled between brush stamps.
    pub brush_spacing: f32,
    /// Continuous paint multiplier used by brush tools.
    pub brush_flow: f32,
    /// Reverse the active brush operation while painting.
    pub invert_brush: bool,
    /// Visibility switches for non-destructive viewport helpers.
    pub viewport_overlays: ViewportOverlayFlags,
    /// Dragging a layer tool from the left strip toward the viewport.
    pub tool_drag: Option<LayerToolDrag>,
    /// RGBA pixels supplied by the app from the latest successful evaluation.
    pub preview_rgba: Option<(u32, u32, Vec<u8>)>,
    pub export_path: Option<String>,
    /// `Some(0..=1)` while a background export is running.
    pub export_progress: Option<f32>,
    pub status: String,
    pub refining: bool,
    /// Best-effort progressive-build completion, from 0.0 through 1.0.
    pub build_progress: Option<f32>,
    /// Layer or pass currently being refined, when the evaluator can identify one.
    pub refining_layer_name: Option<String>,
    /// True when the visible terrain is a Draft or Medium evaluation.
    pub draft_displayed: bool,
    pub quality: PreviewQuality,
    pub profile: FrameProfile,
    pub selected_mask: Option<MaskId>,
    /// Active viewport mask paint target; `None` when sculpting Base or camera-free.
    pub paint_mask: Option<MaskId>,
    /// Paint / erase / smooth mode for the active reusable mask.
    pub mask_paint_tool: terra_core::mask::MaskPaintTool,
    /// When true (Shift held), layer click opens context menu.
    pub shift_context: bool,
    /// Quick Add popup open.
    pub show_quick_add: bool,
    /// When set, Quick Add only lists tools for this WC category.
    pub quick_add_category: Option<terra_core::layer::StackCategory>,
    /// When set, Quick Add lists tools for this artist concept folder only.
    pub quick_add_concept: Option<ArtistConcept>,
    /// When set, new layers are inserted into this group (biome / section).
    pub quick_add_into: Option<terra_core::layer::LayerId>,
    /// When set with [`Self::quick_add_into`], limit the catalog to this biome section.
    /// `None` while targeting a biome root means “any biome-routable content”.
    pub quick_add_biome_section: Option<terra_core::layer::BiomeSection>,
    /// When set, Quick Add lists DistNode generators/modifiers for this biome.
    pub quick_add_distribution: Option<terra_core::layer::LayerId>,
    /// Most recently selected Quick Add tool IDs, newest first.
    pub recent_tools: Vec<String>,
    pub quick_add: QuickAddState,
    /// Command palette open.
    pub show_command_palette: bool,
    pub command_palette: CommandPaletteState,
    /// History panel open.
    pub show_history: bool,
    /// Terrain Recipe view open (execution-order overview).
    pub show_pipeline: bool,
    /// Current camera look target expressed in terrain UV coordinates.
    pub camera_xz: (f32, f32),
    /// Current orbit-camera yaw in radians.
    pub camera_yaw: f32,
    /// Current orbit-camera pitch in radians (view gizmo).
    pub camera_pitch: f32,
    /// When Lit, distinguish the mode-bar Lighting pill from Final.
    pub viewport_lighting_selected: bool,
    /// Camera locations saved with Ctrl+1 through Ctrl+9.
    pub bookmarks: [Option<CameraBookmark>; 9],
    /// Read-only history labels supplied by the app, oldest to newest.
    pub history_descriptions: Vec<String>,
    /// Environment lighting preset for the 3D viewport (presentation only).
    pub lighting_preset: LightingPreset,
    /// True when the lighting values have been edited away from `lighting_preset`,
    /// so the preset field renders blank. Cleared when a preset is (re)selected.
    pub lighting_customized: bool,
    /// Converged samples in Progressive RT mode (read-only renderer telemetry).
    pub progressive_samples: u32,
    /// Progressive path-tracer active this frame (synced from renderer mode).
    pub progressive_renderer_active: bool,
    /// Artist-facing viewport render controls (synced to renderer each frame).
    pub viewport_render: ViewportRenderSettings,
    /// Viewport display lighting preset menu open.
    pub lighting_menu_open: bool,
    /// Camera speed dropdown on the bottom viewport bar.
    pub camera_speed_menu_open: bool,
    /// Multiplier for orbit / pan / look / zoom mouse deltas.
    pub camera_speed: f32,
    /// Synced from the OS window each frame (custom caption maximize/restore icon).
    pub window_maximized: bool,
    /// Tool search query in the contextual tool panel.
    pub tool_search: String,
    /// Brush symmetry toggle (presentation / future stroke mirror).
    pub brush_symmetry: bool,
    /// Dock layout (resizable / collapsible panels).
    pub layout: terra_gui::LayoutPrefs,
    /// Preferred task workspace id (`sculpt`, `biomes`, …). Editor preference,
    /// persisted in the app's editor prefs file (not the dock-geometry blob).
    pub preferred_workspace: String,
    /// When true, creating an entity may switch to its home workspace.
    /// Default false — artists stay in the current workspace unless they opt in.
    pub auto_switch_workspace_on_create: bool,
    /// Biome Focus — definition currently being polished (WHAT + WHERE).
    pub biome_focus: Option<terra_core::biome_definition::BiomeDefinitionId>,
    /// Bookmarks floating window visible.
    pub show_bookmarks: bool,
    /// Layer-kind clipboard for Copy/Paste Settings in the inspector.
    pub settings_clipboard: Option<LayerKind>,
    /// Pending camera bookmark save slot (0..8), consumed by the app.
    pub pending_bookmark_save: Option<usize>,
    /// Pending camera bookmark recall slot (0..8), consumed by the app.
    pub pending_bookmark_recall: Option<usize>,
    /// Dirty flag — editor prefs (dock geometry + workspace + render settings)
    /// should be written to disk.
    pub layout_dirty: bool,
    /// Snapshot of dirty tile IDs `(tx, tz)` for the dirty-tiles overlay.
    pub dirty_tile_ids: Vec<(u32, u32)>,
    /// Tile grid dimensions matching the last GPU/CPU eval metrics.
    pub dirty_tile_grid: (u32, u32),
    /// Opt-in Phase 2 geomorph debug overlay (command palette). Not part of
    /// the default artist mode bar — clears when set back to `None`.
    pub geomorph_debug_field: Option<terra_core::GeomorphDebugField>,
    /// Set by command palette when debug overlay changes; consumed by preview bake.
    pub force_preview_refresh: bool,
    /// Active brush mode for the Paint Biome tool (Paint / Erase / Raise+Paint / â€¦).
    pub biome_paint_tool: terra_core::biome_paint::BiomePaintTool,
    /// Shape history: new layer per stroke session vs continue selected.
    pub shape_edit_mode: terra_core::shape_history::ShapeEditMode,
    /// Active Shape Layer for the current stroke session (cleared on tool change).
    pub shape_session_layer: Option<terra_core::layer::LayerId>,
    /// When true, next full eval after stroke end is requested (draft while painting).
    pub shape_commit_full: bool,
    /// Workspace-owned biome colour overlay (presentation). Prefer over
    /// `BiomeLayer.show_biome_colors` — switching workspace must not write project data.
    pub biome_color_preview: bool,
    /// Temporary editor-only isolate. Must never write project `Region.solo` / layer solo.
    pub temp_solo: crate::ui::workspace::TempSoloState,
    /// Viewport / hierarchy Create-here context menu.
    pub viewport_context_menu: Option<crate::ui::contextual_create_gui::ViewportContextMenu>,
    /// Compact â€œUpdating: â€¦â€ dependency feedback (non-blocking).
    pub affected_feedback: Option<String>,
    /// Expanded â€œWhy outdated?â€ text for the inspector / dock.
    pub why_rebuild_text: Option<String>,
    /// Layer ids with outdated cached results (synced from session).
    pub outdated_layer_ids: Vec<terra_core::layer::LayerId>,
    /// Cap 1 soft incomplete-project diagnostic for the dock (non-blocking).
    pub soft_project_diag: Option<String>,
}

/// Presentation-only environment lighting for the 3D viewport.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LightingPreset {
    #[default]
    Studio,
    Midday,
    Sunset,
    Overcast,
    HighContrast,
    Neutral,
    /// Stochastic heightfield rays with temporal accumulation and denoising.
    Progressive,
}

impl LightingPreset {
    /// Presets exposed in the viewport Lighting menu (all wired to the renderer).
    pub const ALL: [LightingPreset; 7] = [
        LightingPreset::Studio,
        LightingPreset::Midday,
        LightingPreset::Sunset,
        LightingPreset::Overcast,
        LightingPreset::HighContrast,
        LightingPreset::Neutral,
        LightingPreset::Progressive,
    ];

    pub fn label(self) -> &'static str {
        match self {
            LightingPreset::Studio => "Studio",
            LightingPreset::Midday => "Midday",
            LightingPreset::Sunset => "Sunset",
            LightingPreset::Overcast => "Overcast",
            LightingPreset::HighContrast => "High Contrast",
            LightingPreset::Neutral => "Neutral",
            LightingPreset::Progressive => "Progressive RT",
        }
    }

    /// Returns (light_dir_xyz + intensity_w, exposure, clear_rgb).
    pub fn params(self) -> ([f32; 4], f32, [f32; 3]) {
        match self {
            LightingPreset::Studio => ([-0.25, -0.95, -0.15, 1.05], 1.05, [0.18, 0.20, 0.24]),
            LightingPreset::Midday => ([-0.40, -0.88, -0.25, 1.15], 1.10, [0.42, 0.52, 0.62]),
            LightingPreset::Sunset => ([-0.75, -0.45, -0.20, 1.20], 1.15, [0.55, 0.38, 0.32]),
            LightingPreset::Overcast => ([-0.15, -0.98, -0.10, 0.75], 1.00, [0.45, 0.48, 0.52]),
            LightingPreset::HighContrast => ([-0.55, -0.75, -0.35, 1.35], 1.05, [0.12, 0.14, 0.18]),
            LightingPreset::Neutral => ([-0.35, -0.90, -0.20, 1.00], 1.00, [0.28, 0.32, 0.38]),
            LightingPreset::Progressive => ([-0.46, -0.82, -0.34, 1.18], 1.08, [0.36, 0.44, 0.54]),
        }
    }

    pub fn is_progressive(self) -> bool {
        matches!(self, LightingPreset::Progressive)
    }

    /// Suggested viewport renderer mode for this lighting preset.
    pub fn suggested_renderer_mode(self) -> terra_render::ViewportRendererMode {
        if self.is_progressive() {
            terra_render::ViewportRendererMode::ProgressiveRayTraced
        } else {
            terra_render::ViewportRendererMode::Raster
        }
    }
}

/// Convert a sun azimuth (degrees, compass bearing around +Y) and elevation
/// (degrees above the horizon) into a `light_dir` xyz — the direction *from* the
/// light *toward* the scene, matching `EnvironmentLighting::light_dir`. The
/// result is unit length (the renderer normalizes it anyway); sun intensity is
/// carried separately in `light_dir.w`.
pub fn sun_dir_from_az_el(azimuth_deg: f32, elevation_deg: f32) -> [f32; 3] {
    let az = azimuth_deg.to_radians();
    let el = elevation_deg.to_radians();
    let ce = el.cos();
    // toward-sun = (ce·cos az, sin el, ce·sin az); light_dir points the other way.
    [-(ce * az.cos()), -el.sin(), -(ce * az.sin())]
}

/// Inverse of [`sun_dir_from_az_el`]: recover `(azimuth, elevation)` in degrees
/// from a `light_dir` xyz. Azimuth is wrapped to `[0, 360)`.
pub fn sun_az_el_from_dir(light_dir: [f32; 3]) -> (f32, f32) {
    // toward-sun is the negated light direction.
    let s = [-light_dir[0], -light_dir[1], -light_dir[2]];
    let len = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt().max(1e-6);
    let (x, y, z) = (s[0] / len, s[1] / len, s[2] / len);
    let elevation = y.clamp(-1.0, 1.0).asin().to_degrees();
    let mut azimuth = z.atan2(x).to_degrees();
    if azimuth < 0.0 {
        azimuth += 360.0;
    }
    (azimuth, elevation)
}

#[cfg(test)]
mod sun_dir_tests {
    use super::*;

    #[test]
    fn az_el_round_trips_through_dir() {
        for &(az, el) in &[(0.0f32, 45.0f32), (90.0, 10.0), (215.0, 72.0), (359.0, 1.0)] {
            let dir = sun_dir_from_az_el(az, el);
            let (az2, el2) = sun_az_el_from_dir(dir);
            assert!((el - el2).abs() < 1e-2, "elevation {el} -> {el2}");
            // Compare azimuth modulo 360 so 359° vs 359° does not read as a 360° gap.
            let daz = ((az - az2 + 540.0) % 360.0 - 180.0).abs();
            assert!(daz < 1e-2, "azimuth {az} -> {az2}");
        }
    }

    #[test]
    fn studio_preset_reads_as_a_high_sun() {
        let (ld, _, _) = LightingPreset::Studio.params();
        let (_az, el) = sun_az_el_from_dir([ld[0], ld[1], ld[2]]);
        assert!(el > 60.0, "Studio should read as a high sun, got {el}");
    }
}

/// Persisted, serde-stable form of the viewport render controls (user preference).
///
/// This is the on-disk vocabulary — render `mode`/`preset` are stored as strings so
/// the editor prefs file does not depend on `terra_render`'s enums. The runtime source
/// of truth is [`ViewportRenderSettings`]; convert with [`ViewportRenderSettings::from_prefs`]
/// / [`ViewportRenderSettings::to_prefs`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewportRenderPrefs {
    #[serde(default = "default_render_mode")]
    pub mode: String,
    #[serde(default = "default_render_preset")]
    pub preset: String,
    #[serde(default = "default_target_fps")]
    pub target_fps: f32,
    #[serde(default = "default_max_spp")]
    pub max_spp: u32,
    #[serde(default = "default_true")]
    pub dynamic_resolution: bool,
    #[serde(default = "default_true")]
    pub denoise: bool,
    #[serde(default = "default_interactive_spp")]
    pub interactive_spp: u32,
    #[serde(default = "default_settling_spp")]
    pub settling_spp: u32,
    #[serde(default = "default_refining_spp")]
    pub refining_spp: u32,
    #[serde(default = "default_bounces_interactive")]
    pub max_bounces_interactive: u32,
    #[serde(default = "default_bounces_refining")]
    pub max_bounces_refining: u32,
    #[serde(default = "default_min_scale")]
    pub min_internal_scale: f32,
    #[serde(default = "default_max_scale")]
    pub max_internal_scale: f32,
    #[serde(default = "default_history_clamp")]
    pub history_clamp_k: f32,
    #[serde(default = "default_converge")]
    pub converge_fraction: f32,
    #[serde(default)]
    pub debug_viz_mode: u32,
}

fn default_render_mode() -> String {
    "Raster".into()
}
fn default_render_preset() -> String {
    "Balanced".into()
}
fn default_target_fps() -> f32 {
    60.0
}
fn default_max_spp() -> u32 {
    128
}
fn default_true() -> bool {
    true
}
fn default_interactive_spp() -> u32 {
    1
}
fn default_settling_spp() -> u32 {
    2
}
fn default_refining_spp() -> u32 {
    4
}
fn default_bounces_interactive() -> u32 {
    2
}
fn default_bounces_refining() -> u32 {
    4
}
fn default_min_scale() -> f32 {
    0.5
}
fn default_max_scale() -> f32 {
    1.0
}
fn default_history_clamp() -> f32 {
    1.5
}
fn default_converge() -> f32 {
    0.95
}

impl Default for ViewportRenderPrefs {
    fn default() -> Self {
        Self {
            mode: default_render_mode(),
            preset: default_render_preset(),
            target_fps: default_target_fps(),
            max_spp: default_max_spp(),
            dynamic_resolution: true,
            denoise: true,
            interactive_spp: default_interactive_spp(),
            settling_spp: default_settling_spp(),
            refining_spp: default_refining_spp(),
            max_bounces_interactive: default_bounces_interactive(),
            max_bounces_refining: default_bounces_refining(),
            min_internal_scale: default_min_scale(),
            max_internal_scale: default_max_scale(),
            history_clamp_k: default_history_clamp(),
            converge_fraction: default_converge(),
            debug_viz_mode: 0,
        }
    }
}

/// Primary viewport rendering controls (synced to [`terra_render::ViewportQualityManager`]).
#[derive(Debug, Clone)]
pub struct ViewportRenderSettings {
    pub mode: terra_render::ViewportRendererMode,
    pub preset: terra_render::QualityPreset,
    pub target_fps: f32,
    pub max_spp: u32,
    pub dynamic_resolution: bool,
    pub denoise: bool,
    pub menu_open: bool,
    pub advanced_open: bool,
    /// Persisted scroll offset for the render panel (survives frames; reset would
    /// snap the window back to the top every frame).
    pub menu_scroll: f32,
    pub interactive_spp: u32,
    pub settling_spp: u32,
    pub refining_spp: u32,
    pub max_bounces_interactive: u32,
    pub max_bounces_refining: u32,
    pub min_internal_scale: f32,
    pub max_internal_scale: f32,
    pub history_clamp_k: f32,
    pub converge_fraction: f32,
    /// Developer debug visualization (0 = final). Hidden from artists by default.
    pub debug_viz_mode: u32,
    /// Editable sun direction, shared by Raster and RT. Azimuth is a compass
    /// bearing around +Y (degrees); elevation is degrees above the horizon.
    /// Seeded from the lighting preset when it changes, then overrides the
    /// preset's baked direction so the sliders move the sun.
    pub sun_azimuth_deg: f32,
    pub sun_elevation_deg: f32,
    /// Editable sun intensity (`light_dir.w`) and tonemap exposure, shared by
    /// both renderers. Seeded from the preset on change, then overridden.
    pub sun_intensity: f32,
    pub exposure: f32,
    /// Editable sky / clear color, shared by both renderers (raster atmosphere +
    /// fog, RT sky). Seeded from the preset on change, then overridden.
    pub sky_color: [f32; 3],
    /// Raster shading controls (inert in the RT backend). Ambient and fog default
    /// to 1.0 (the current look); shadow_strength 0 keeps cast shadows off until
    /// dialed up. These are not seeded from presets — they are raster tweaks.
    pub ambient_strength: f32,
    pub shadow_strength: f32,
    pub fog_strength: f32,
}

impl Default for ViewportRenderSettings {
    fn default() -> Self {
        let cfg = terra_render::RenderQualityConfig::default();
        // Seed the editable sun angle from the default lighting preset so the
        // initial state matches what selecting Studio would produce.
        let (studio_dir, studio_exposure, studio_clear) = LightingPreset::Studio.params();
        let (sun_azimuth_deg, sun_elevation_deg) =
            sun_az_el_from_dir([studio_dir[0], studio_dir[1], studio_dir[2]]);
        Self {
            mode: cfg.mode,
            preset: cfg.preset,
            target_fps: cfg.target_fps,
            max_spp: cfg.max_accumulated_spp,
            dynamic_resolution: cfg.dynamic_resolution_enabled,
            denoise: cfg.denoise_enabled,
            menu_open: false,
            advanced_open: false,
            menu_scroll: 0.0,
            interactive_spp: cfg.interactive_spp,
            settling_spp: cfg.settling_spp,
            refining_spp: cfg.refining_spp,
            max_bounces_interactive: cfg.max_bounces_interactive,
            max_bounces_refining: cfg.max_bounces_refining,
            min_internal_scale: cfg.min_internal_scale,
            max_internal_scale: cfg.max_internal_scale,
            history_clamp_k: cfg.history_clamp_k,
            converge_fraction: cfg.converge_fraction,
            debug_viz_mode: 0,
            sun_azimuth_deg,
            sun_elevation_deg,
            sun_intensity: studio_dir[3],
            exposure: studio_exposure,
            sky_color: studio_clear,
            ambient_strength: 1.0,
            shadow_strength: 0.0,
            fog_strength: 1.0,
        }
    }
}

impl ViewportRenderSettings {
    pub fn from_prefs(prefs: &ViewportRenderPrefs) -> Self {
        let mut s = Self::default();
        s.mode = match prefs.mode.as_str() {
            "Fast" => terra_render::ViewportRendererMode::Fast,
            "Raster" => terra_render::ViewportRendererMode::Raster,
            "Final" => terra_render::ViewportRendererMode::Final,
            _ => terra_render::ViewportRendererMode::ProgressiveRayTraced,
        };
        s.preset = match prefs.preset.as_str() {
            "Performance" => terra_render::QualityPreset::Performance,
            "Quality" => terra_render::QualityPreset::Quality,
            _ => terra_render::QualityPreset::Balanced,
        };
        s.target_fps = prefs.target_fps;
        s.max_spp = prefs.max_spp;
        s.dynamic_resolution = prefs.dynamic_resolution;
        s.denoise = prefs.denoise;
        s.interactive_spp = prefs.interactive_spp;
        s.settling_spp = prefs.settling_spp;
        s.refining_spp = prefs.refining_spp;
        s.max_bounces_interactive = prefs.max_bounces_interactive;
        s.max_bounces_refining = prefs.max_bounces_refining;
        s.min_internal_scale = prefs.min_internal_scale;
        s.max_internal_scale = prefs.max_internal_scale;
        s.history_clamp_k = prefs.history_clamp_k;
        s.converge_fraction = prefs.converge_fraction;
        s.debug_viz_mode = prefs.debug_viz_mode;
        s
    }

    pub fn to_prefs(&self) -> ViewportRenderPrefs {
        ViewportRenderPrefs {
            mode: match self.mode {
                terra_render::ViewportRendererMode::Fast => "Fast".into(),
                terra_render::ViewportRendererMode::Raster => "Raster".into(),
                terra_render::ViewportRendererMode::ProgressiveRayTraced => {
                    "ProgressiveRayTraced".into()
                }
                terra_render::ViewportRendererMode::Final => "Final".into(),
            },
            preset: match self.preset {
                terra_render::QualityPreset::Performance => "Performance".into(),
                terra_render::QualityPreset::Balanced => "Balanced".into(),
                terra_render::QualityPreset::Quality => "Quality".into(),
            },
            target_fps: self.target_fps,
            max_spp: self.max_spp,
            dynamic_resolution: self.dynamic_resolution,
            denoise: self.denoise,
            interactive_spp: self.interactive_spp,
            settling_spp: self.settling_spp,
            refining_spp: self.refining_spp,
            max_bounces_interactive: self.max_bounces_interactive,
            max_bounces_refining: self.max_bounces_refining,
            min_internal_scale: self.min_internal_scale,
            max_internal_scale: self.max_internal_scale,
            history_clamp_k: self.history_clamp_k,
            converge_fraction: self.converge_fraction,
            debug_viz_mode: self.debug_viz_mode,
        }
    }
}

/// Persisted orbit-camera state for a numbered navigation bookmark.
#[derive(Debug, Clone, Copy)]
pub struct CameraBookmark {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}

impl UiState {
    pub fn ensure_sculpt_defaults(&mut self) {
        if self.sculpt_radius <= 0.0 {
            self.sculpt_radius = 0.04;
        }
        if self.sculpt_strength <= 0.0 {
            self.sculpt_strength = 4.0;
        }
        if self.brush_falloff <= 0.0 {
            self.brush_falloff = 0.5;
        }
        if self.brush_spacing <= 0.0 {
            self.brush_spacing = 0.1;
        }
        if self.brush_flow <= 0.0 {
            self.brush_flow = 1.0;
        }
        if self.camera_speed <= 0.0 {
            self.camera_speed = 1.0;
        }
    }

    /// Remember and arm an editor tool. Sculpt brushes update [`Self::last_sculpt_tool`].
    pub fn set_editor_tool(&mut self, tool: EditorTool) {
        if tool.is_sculpt() {
            self.last_sculpt_tool = Some(tool);
        }
        self.editor_tool = tool;
    }

    /// Brush restored when the viewport Sculpt mode button is pressed.
    pub fn remembered_sculpt_tool(&self) -> EditorTool {
        self.last_sculpt_tool
            .filter(|t| t.is_sculpt())
            .unwrap_or(EditorTool::Raise)
    }

    /// Capture formal editor-only [`WorkspaceState`] (no terrain-generation data).
    pub fn workspace_state(&self) -> WorkspaceState {
        WorkspaceState {
            active: self.active_workspace,
            mode: self.workspace_mode,
            app_workspace: self.app_workspace,
            editor_tool: self.editor_tool,
            viewport_mode: self.preview_mode,
            viewport_overlays: self.viewport_overlays,
            lighting_preset: self.lighting_preset,
            lighting_customized: self.lighting_customized,
            inspector_advanced: self.inspector_advanced,
            brush: BrushWorkspaceState {
                radius: self.sculpt_radius,
                strength: self.sculpt_strength,
                falloff: self.brush_falloff,
                spacing: self.brush_spacing,
                flow: self.brush_flow,
                invert: self.invert_brush,
                symmetry: self.brush_symmetry,
            },
            biome_paint_tool: self.biome_paint_tool,
            mask_paint_tool: self.mask_paint_tool,
            selected_mask: self.selected_mask,
            paint_mask: self.paint_mask,
            biome_color_preview: self.biome_color_preview,
            biome_focus: self.biome_focus,
            temp_solo: self.temp_solo.clone(),
            show_pipeline: self.show_pipeline,
            show_history: self.show_history,
            build_progress: self.build_progress,
            quality: self.quality,
            draft_displayed: self.draft_displayed,
            refining: self.refining,
            refining_layer_name: self.refining_layer_name.clone(),
            camera_xz: self.camera_xz,
            camera_yaw: self.camera_yaw,
            camera_pitch: self.camera_pitch,
        }
    }

    /// Apply a [`WorkspaceState`] snapshot onto this UI session (presentation only).
    pub fn apply_workspace_state(&mut self, ws: &WorkspaceState) {
        self.active_workspace = ws.active;
        self.workspace_mode = ws.mode;
        self.app_workspace = ws.app_workspace;
        self.editor_tool = ws.editor_tool;
        self.preview_mode = ws.viewport_mode;
        self.viewport_overlays = ws.viewport_overlays;
        self.lighting_preset = ws.lighting_preset;
        // Carry the customized flag with the preset so switching workspaces preserves
        // an edited look, instead of clearing it and letting the redraw seed snap the
        // viewport back to the preset (the "entering Sculpt resets lighting" bug).
        self.lighting_customized = ws.lighting_customized;
        self.inspector_advanced = ws.inspector_advanced;
        self.sculpt_radius = ws.brush.radius;
        self.sculpt_strength = ws.brush.strength;
        self.brush_falloff = ws.brush.falloff;
        self.brush_spacing = ws.brush.spacing;
        self.brush_flow = ws.brush.flow;
        self.invert_brush = ws.brush.invert;
        self.brush_symmetry = ws.brush.symmetry;
        self.biome_paint_tool = ws.biome_paint_tool;
        self.mask_paint_tool = ws.mask_paint_tool;
        self.selected_mask = ws.selected_mask;
        self.paint_mask = ws.paint_mask;
        self.biome_color_preview = ws.biome_color_preview;
        self.biome_focus = ws.biome_focus;
        self.temp_solo = ws.temp_solo.clone();
        self.show_pipeline = ws.show_pipeline;
        self.show_history = ws.show_history;
        self.build_progress = ws.build_progress;
        self.quality = ws.quality;
        self.draft_displayed = ws.draft_displayed;
        self.refining = ws.refining;
        self.refining_layer_name = ws.refining_layer_name.clone();
        // Camera is intentionally applied so switch_workspace can preserve it.
        self.camera_xz = ws.camera_xz;
        self.camera_yaw = ws.camera_yaw;
        self.camera_pitch = ws.camera_pitch;
    }

    /// Switch task workspace (presentation only). Does not mutate project or eval.
    ///
    /// Preserves selection (on document), camera, brush numerics, and hierarchy
    /// expansion (owned by layers GUI). Persists preferred workspace into layout prefs.
    pub fn switch_workspace(&mut self, id: WorkspaceId) {
        // All Tools removed from the rail — remap legacy prefs / commands.
        let id = if matches!(id, WorkspaceId::AllTools) {
            WorkspaceId::Objects
        } else {
            id
        };
        let mut ws = self.workspace_state();
        ws.switch_workspace(id);
        self.apply_workspace_state(&ws);
        self.tool_drag = None;
        self.preferred_workspace = id.as_str().to_string();
        self.layout_dirty = true;
        let def = workspace_definition(id);
        self.status = format!("Workspace: {} — {}", def.name, def.description);
    }

    /// Switch tool-domain mode without mutating project or evaluation state.
    pub fn switch_workspace_mode(&mut self, mode: WorkspaceMode) {
        self.switch_workspace(mode.to_workspace_id());
    }

    /// Switch application workspace intent without mutating project or evaluation state.
    pub fn switch_app_workspace(&mut self, workspace: AppWorkspace) {
        self.switch_workspace(workspace.to_workspace_id());
    }

    /// Toggle biome colour overlay (workspace-only; does not create placement layers).
    pub fn set_biome_color_preview(&mut self, on: bool) {
        self.biome_color_preview = on;
    }

    /// Active workspace metadata.
    pub fn workspace_def(&self) -> &'static WorkspaceDefinition {
        workspace_definition(self.active_workspace)
    }

    /// Apply persisted preferred workspace from editor prefs (editor-only).
    pub fn apply_preferred_workspace_from_prefs(&mut self) {
        if let Some(id) = WorkspaceId::parse(&self.preferred_workspace) {
            // Direct field sync without re-dirtying layout on load.
            let mut ws = self.workspace_state();
            ws.switch_workspace(id);
            self.apply_workspace_state(&ws);
        }
    }

    /// Arm mask painting without entering full Mask view (keeps TOOLS rail visible).
    pub fn arm_mask_paint(&mut self) {
        self.set_editor_tool(EditorTool::PaintMask);
        self.tool_drag = None;
        self.viewport_overlays.mask_overlay = true;
        self.show_2d_preview = true;
        self.mask_paint_tool = terra_core::mask::MaskPaintTool::Paint;
        // Do not call enter_mask_view — inspector hosts mask chrome while PaintMask is armed.
    }

    /// Switch workspace only when needed (avoids resetting tools when already there).
    pub fn ensure_workspace(&mut self, id: WorkspaceId) {
        if self.active_workspace != id {
            self.switch_workspace(id);
        }
    }

    /// After creating a shape / landform layer: Sculpt workspace + brush (or path/polygon tool).
    pub fn focus_created_shape(&mut self, authoring_tool: Option<EditorTool>) {
        self.leave_mask_view();
        self.ensure_workspace(WorkspaceId::Sculpt);
        let tool = authoring_tool.unwrap_or_else(|| self.remembered_sculpt_tool());
        self.set_editor_tool(tool);
        self.paint_mask = None;
        self.viewport_overlays.brush_preview = true;
    }

    /// After creating a project mask: Mask workspace + Mask view (+ paint when painted).
    pub fn focus_created_mask(&mut self, painted: bool) {
        self.ensure_workspace(WorkspaceId::Rules);
        self.enter_mask_view();
        self.viewport_overlays.mask_overlay = true;
        if painted {
            self.arm_mask_paint();
        } else if self.editor_tool == EditorTool::PaintMask {
            self.set_editor_tool(EditorTool::Move);
        }
    }

    /// After creating a biome container: Biomes workspace + paint tool.
    pub fn focus_created_biome(&mut self) {
        self.leave_mask_view();
        self.ensure_workspace(WorkspaceId::Biomes);
        self.set_editor_tool(EditorTool::PaintBiome);
        self.biome_color_preview = true;
        self.paint_mask = None;
    }

    /// Focus the authoring view for a newly created stack layer.
    pub fn focus_view_for_new_layer(&mut self, kind: &terra_core::layer::LayerKind) {
        use terra_core::layer::{biome_destination_section, BiomeSection, StackCategory};

        match kind {
            terra_core::layer::LayerKind::Path(_) => {
                self.focus_created_shape(Some(EditorTool::EditPath));
                self.status = "Path: click terrain to add spline nodes".into();
                return;
            }
            terra_core::layer::LayerKind::PolygonHeight(_) => {
                self.focus_created_shape(Some(EditorTool::EditPolygon));
                self.status =
                    "Polygon: click terrain to add vertices; use Finish Drawing when done".into();
                return;
            }
            terra_core::layer::LayerKind::RiverNetwork(_) => {
                self.leave_mask_view();
                self.ensure_workspace(WorkspaceId::Simulation);
                self.set_editor_tool(EditorTool::EditRiverSpring);
                self.paint_mask = None;
                self.status = "River Network: click terrain to place springs".into();
                return;
            }
            _ => {}
        }

        if let Some(section) = biome_destination_section(kind) {
            self.leave_mask_view();
            self.paint_mask = None;
            match section {
                BiomeSection::Filters => self.ensure_workspace(WorkspaceId::Develop),
                BiomeSection::Materials => self.ensure_workspace(WorkspaceId::Surface),
                BiomeSection::Objects => self.ensure_workspace(WorkspaceId::Objects),
                BiomeSection::LocalSims => self.ensure_workspace(WorkspaceId::Simulation),
            }
            self.set_editor_tool(EditorTool::Move);
            return;
        }

        match StackCategory::from_operation(kind.category()) {
            StackCategory::Shape | StackCategory::Foundation => {
                self.focus_created_shape(None);
            }
            StackCategory::Simulation => {
                self.leave_mask_view();
                self.ensure_workspace(WorkspaceId::Simulation);
                self.set_editor_tool(EditorTool::Move);
                self.paint_mask = None;
            }
            StackCategory::Surface => {
                self.leave_mask_view();
                self.ensure_workspace(WorkspaceId::Biomes);
                self.set_editor_tool(EditorTool::Move);
                self.paint_mask = None;
            }
            StackCategory::Mask => {
                self.focus_created_mask(false);
            }
        }
    }

    /// Viewport Mask view (primary Views bar) — drives left dock + mask viz.
    pub fn is_mask_view(&self) -> bool {
        matches!(
            self.preview_mode,
            Preview2dMode::Mask | Preview2dMode::Masks
        )
    }

    /// Enter Mask view: mask preview, Show Mask overlay, left mask panel.
    pub fn enter_mask_view(&mut self) {
        self.preview_mode = Preview2dMode::Masks;
        self.show_mask_editor = true;
        self.viewport_lighting_selected = false;
        self.pending_close_mask_editor = false;
        let _ov = &mut self.viewport_overlays;
    }

    /// Leave Mask view and restore Terrain chrome (tool rail). Schedules session close.
    pub fn leave_mask_view(&mut self) {
        if self.is_mask_view() || self.show_mask_editor {
            self.pending_close_mask_editor = true;
        }
        if self.is_mask_view() {
            self.preview_mode = Preview2dMode::Lit;
        }
        self.show_mask_editor = false;
        self.viewport_lighting_selected = false;
    }
}

/// Optional viewport aids. These flags affect editor presentation, not terrain data.
#[derive(Debug, Clone, Copy)]
pub struct ViewportOverlayFlags {
    pub grid: bool,
    pub world_bounds: bool,
    pub water_level: bool,
    pub contours: bool,
    pub wireframe: bool,
    pub brush_preview: bool,
    pub mask_overlay: bool,
    pub perf_info: bool,
    /// Phase I dirty-tile debug tint (consumes `TileScheduler::dirty` snapshot).
    pub dirty_tiles: bool,
}

impl Default for ViewportOverlayFlags {
    fn default() -> Self {
        Self {
            grid: false,
            world_bounds: false,
            // Match prior always-on ocean when Island/Coastal sea level exists.
            water_level: true,
            contours: false,
            wireframe: false,
            brush_preview: false,
            mask_overlay: false,
            perf_info: false,
            dirty_tiles: false,
        }
    }
}

impl ViewportOverlayFlags {
    pub const EMPTY: Self = Self {
        grid: false,
        world_bounds: false,
        water_level: true,
        contours: false,
        wireframe: false,
        brush_preview: false,
        mask_overlay: false,
        perf_info: false,
        dirty_tiles: false,
    };
}

/// Left-sidebar / sculpt tool selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EditorTool {
    /// Photoshop-style hand: orbit / pan / zoom in the viewport.
    #[default]
    Move,
    Raise,
    Lower,
    Smooth,
    /// Mask paint (upper layers).
    PaintMask,
    /// Paint biome splat weights (WC Biome Layers).
    PaintBiome,
    /// Click terrain to append Path control points on the selected Path layer.
    EditPath,
    /// Click terrain to append vertices on the selected Polygon layer.
    EditPolygon,
    /// Semantic, resolution-independent sculpt operations.
    Ridge,
    Valley,
    Roughness,
    UpliftBrush,
    Protect,
    Hardness,
    Sediment,
    RiverConstraint,
    /// Click terrain to place RiverNetwork springs on the selected River Network layer.
    EditRiverSpring,
    /// Viewport diagnostic overlays (also set via viewport pills).
    Height,
    Slope,
    Flow,
    Masks,
    /// Shape history brushes / stamps (Sculpt workspace).
    Flatten,
    Terrace,
    Pinch,
    Inflate,
    ErodeBrush,
    MountainStamp,
    ValleyStamp,
    PlateauStamp,
    CraterStamp,
    CoastlineTool,
    RiverPathTool,
    HeightStamp,
    NoiseBrush,
    /// Click two points to measure world-space distance.
    Measure,
}

impl EditorTool {
    pub fn as_preview_mode(self) -> Option<Preview2dMode> {
        match self {
            EditorTool::Height => Some(Preview2dMode::Height),
            EditorTool::Slope => Some(Preview2dMode::Slope),
            EditorTool::Flow => Some(Preview2dMode::Flow),
            EditorTool::Masks => Some(Preview2dMode::Masks),
            _ => None,
        }
    }

    pub fn is_move(self) -> bool {
        matches!(self, EditorTool::Move)
    }

    pub fn is_sculpt(self) -> bool {
        self.shape_tool().is_some()
            || matches!(
                self,
                EditorTool::Ridge
                    | EditorTool::Valley
                    | EditorTool::Roughness
                    | EditorTool::UpliftBrush
                    | EditorTool::Protect
                    | EditorTool::Hardness
                    | EditorTool::Sediment
                    | EditorTool::RiverConstraint
            )
    }

    /// Maps to Shape history [`terra_core::shape_history::ShapeTool`] when applicable.
    pub fn shape_tool(self) -> Option<terra_core::shape_history::ShapeTool> {
        use terra_core::shape_history::ShapeTool;
        match self {
            EditorTool::Raise => Some(ShapeTool::Raise),
            EditorTool::Lower => Some(ShapeTool::Lower),
            EditorTool::Smooth => Some(ShapeTool::Smooth),
            EditorTool::Flatten => Some(ShapeTool::Flatten),
            EditorTool::Terrace => Some(ShapeTool::Terrace),
            EditorTool::Pinch => Some(ShapeTool::Pinch),
            EditorTool::Inflate => Some(ShapeTool::Inflate),
            EditorTool::ErodeBrush => Some(ShapeTool::ErodeBrush),
            EditorTool::MountainStamp => Some(ShapeTool::MountainStamp),
            EditorTool::ValleyStamp => Some(ShapeTool::ValleyStamp),
            EditorTool::PlateauStamp => Some(ShapeTool::PlateauStamp),
            EditorTool::CraterStamp => Some(ShapeTool::CraterStamp),
            EditorTool::CoastlineTool => Some(ShapeTool::Coastline),
            EditorTool::RiverPathTool => Some(ShapeTool::RiverPath),
            EditorTool::HeightStamp => Some(ShapeTool::HeightStamp),
            EditorTool::NoiseBrush => Some(ShapeTool::NoiseBrush),
            EditorTool::Ridge => Some(ShapeTool::MountainStamp), // ridge brush â‰ˆ mountain stroke
            EditorTool::Valley => Some(ShapeTool::ValleyStamp),
            EditorTool::Roughness => Some(ShapeTool::NoiseBrush),
            EditorTool::UpliftBrush => Some(ShapeTool::Raise),
            EditorTool::RiverConstraint => Some(ShapeTool::RiverPath),
            _ => None,
        }
    }

    /// Click-to-place Path nodes / River springs / Measure endpoints.
    pub fn is_place_point(self) -> bool {
        matches!(
            self,
            EditorTool::EditPath
                | EditorTool::EditPolygon
                | EditorTool::EditRiverSpring
                | EditorTool::Measure
        )
    }

    /// Raise / Lower / Smooth / Paint Mask / Paint Biome — own left-drag paint and wheel size.
    pub fn is_brush(self) -> bool {
        self.is_sculpt() || matches!(self, EditorTool::PaintMask | EditorTool::PaintBiome)
    }

    pub fn is_view(self) -> bool {
        self.as_preview_mode().is_some()
    }

    /// Legacy sculpt mode byte for SculptBase stamps (raise/lower/smooth only).
    pub fn sculpt_mode(self) -> Option<u8> {
        match self {
            EditorTool::Raise | EditorTool::UpliftBrush => Some(0),
            EditorTool::Lower | EditorTool::ErodeBrush => Some(1),
            EditorTool::Smooth | EditorTool::Pinch => Some(2),
            EditorTool::Ridge | EditorTool::MountainStamp => Some(3),
            EditorTool::Valley | EditorTool::ValleyStamp | EditorTool::RiverPathTool => Some(4),
            EditorTool::Roughness | EditorTool::NoiseBrush => Some(5),
            EditorTool::Protect => Some(7),
            EditorTool::Hardness => Some(8),
            EditorTool::Sediment => Some(9),
            EditorTool::RiverConstraint => Some(10),
            EditorTool::Flatten | EditorTool::HeightStamp | EditorTool::PlateauStamp => Some(0),
            EditorTool::Terrace => Some(2),
            EditorTool::Inflate | EditorTool::CraterStamp | EditorTool::CoastlineTool => Some(0),
            _ => None,
        }
    }
}

/// Per-frame timings in microseconds for the profiler overlay.
#[derive(Debug, Clone, Default)]
pub struct FrameProfile {
    pub eval_us: u64,
    pub upload_us: u64,
    pub render_us: u64,
    pub ui_us: u64,
    pub frame_us: u64,
    pub gen_id: u64,
    pub tex_w: u32,
    pub tex_h: u32,
    pub quality: &'static str,
    /// `"GPU"` preview engine or `"CPU"` fallback.
    pub path: &'static str,
    pub terrain_grid_size: u32,
    pub tiles_x: u32,
    pub tiles_z: u32,
    pub cache_hits: u32,
    pub computed_layers: u32,
    pub disabled_layers: u32,
    pub slowest_layer: String,
    pub slowest_layer_us: u64,
    pub tile_cache_resident: usize,
    pub tile_cache_pinned: usize,
    pub tile_cache_used_mb: f32,
    pub tile_cache_budget_mb: f32,
    pub tile_cache_evictions: u64,
    pub tile_uploads_pending: usize,
    pub visible_tiles_exact: usize,
    pub visible_tiles_fallback: usize,
    pub visible_tiles_missing: usize,
    /// GPU terrain pass microseconds (0 if TIMESTAMP_QUERY unsupported).
    pub gpu_terrain_us: u64,
    /// GPU shadow pass microseconds.
    pub gpu_shadow_us: u64,
    pub gpu_timestamps_supported: bool,
    // Progressive path-tracer debug (Phase 12)
    pub camera_version: u64,
    pub terrain_version: u64,
    pub lighting_version: u64,
    pub viewport_version: u64,
    pub material_version: u64,
    pub geometry_version: u64,
    pub last_invalidation: &'static str,
    pub accum_frame: u32,
    pub global_frame: u64,
    pub max_spp: u32,
    pub spp_this_frame: u32,
    pub bounce_count: u32,
    pub internal_scale: f32,
    pub smoothed_gpu_ms: f32,
    pub last_gpu_ms: f32,
    pub convergence_fraction: f32,
    pub active_tiles: u32,
    pub reduced_tiles: u32,
    pub converged_tiles: u32,
    pub path_trace_us: u64,
    pub temporal_us: u64,
    pub denoise_us: u64,
    pub interaction_state: &'static str,
    pub renderer_mode: &'static str,
}

impl FrameProfile {
    pub fn update_layer_timings(&mut self, timings: &[terra_core::eval::LayerEvalTiming]) {
        use terra_core::eval::LayerEvalStatus;

        self.cache_hits = 0;
        self.computed_layers = 0;
        self.disabled_layers = 0;
        self.slowest_layer.clear();
        self.slowest_layer_us = 0;
        for timing in timings {
            match timing.status {
                LayerEvalStatus::Disabled => {
                    self.disabled_layers = self.disabled_layers.saturating_add(1)
                }
                LayerEvalStatus::CacheHit => self.cache_hits = self.cache_hits.saturating_add(1),
                LayerEvalStatus::Computed => {
                    self.computed_layers = self.computed_layers.saturating_add(1)
                }
            }
            if timing.elapsed_us >= self.slowest_layer_us {
                self.slowest_layer_us = timing.elapsed_us;
                self.slowest_layer = timing.layer_name.clone();
            }
        }
    }

    pub fn update_tile_cache(&mut self, stats: terra_core::TileCacheStats, pending_uploads: usize) {
        const MIB: f32 = 1024.0 * 1024.0;
        self.tile_cache_resident = stats.resident_tiles;
        self.tile_cache_pinned = stats.pinned_tiles;
        self.tile_cache_used_mb = stats.used_bytes as f32 / MIB;
        self.tile_cache_budget_mb = stats.budget_bytes as f32 / MIB;
        self.tile_cache_evictions = stats.evictions;
        self.tile_uploads_pending = pending_uploads;
    }

    pub fn update_visible_tiles(&mut self, exact: usize, fallback: usize, missing: usize) {
        self.visible_tiles_exact = exact;
        self.visible_tiles_fallback = fallback;
        self.visible_tiles_missing = missing;
    }

    pub fn update_progressive(
        &mut self,
        versions: terra_render::SceneVersions,
        last_invalidation: terra_render::InvalidationReason,
        accum_frame: u32,
        global_frame: u64,
        quality: &terra_render::ViewportQualityManager,
        gpu: &terra_render::GpuTimings,
        interaction: terra_core::EditorRefinementState,
        mode: terra_render::ViewportRendererMode,
        progressive_samples: u32,
    ) {
        self.camera_version = versions.camera_version;
        self.terrain_version = versions.terrain_version;
        self.lighting_version = versions.lighting_version;
        self.viewport_version = versions.viewport_version;
        self.material_version = versions.material_version;
        self.geometry_version = versions.geometry_version;
        self.last_invalidation = last_invalidation.label();
        self.accum_frame = accum_frame;
        self.global_frame = global_frame;
        self.max_spp = quality.config.max_accumulated_spp;
        self.spp_this_frame = quality.spp_this_frame;
        self.bounce_count = quality.bounce_count;
        self.internal_scale = quality.internal_scale;
        self.smoothed_gpu_ms = quality.smoothed_gpu_ms();
        self.last_gpu_ms = quality.last_gpu_ms();
        self.convergence_fraction = quality.convergence_fraction;
        self.active_tiles = quality.active_sampling_tiles;
        self.reduced_tiles = quality.reduced_sampling_tiles;
        self.converged_tiles = quality.converged_sampling_tiles;
        self.path_trace_us = gpu.path_trace_us;
        self.temporal_us = gpu.temporal_us;
        self.denoise_us = gpu.denoise_us;
        self.interaction_state = match interaction {
            terra_core::EditorRefinementState::Interactive => "Interactive",
            terra_core::EditorRefinementState::Settling => "Settling",
            terra_core::EditorRefinementState::Refining => "Refining",
            terra_core::EditorRefinementState::Converged => "Converged",
            terra_core::EditorRefinementState::Export => "Export",
        };
        self.renderer_mode = mode.label();
        let _ = progressive_samples;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Preview2dMode {
    #[default]
    Lit,
    Unlit,
    Height,
    Slope,
    Curvature,
    Flow,
    Convexity,
    Concavity,
    Normals,
    Wireframe,
    AmbientOcclusion,
    Material,
    Biome,
    /// Legacy singular spelling retained for integrations using the Phase 3 API.
    Mask,
    Masks,
    VegetationDensity,
    /// Hydraulic wetness / water accumulation (normalized).
    Water,
    Sediment,
    Hardness,
    Erosion,
    Deposition,
    /// Strahler-like stream order from SPE / river routing.
    StreamOrder,
    /// Cumulative stream-power incision depth.
    SpeIncision,
    /// Phase H climate overlays.
    Temperature,
    Rainfall,
    Snow,
    SoilMoisture,
    /// Phase J overhang / cave region mask.
    Overhang,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UiCursor {
    #[default]
    Default,
    /// Open hand — hover over a layer grip.
    Grab,
    /// Closed hand — actively dragging a layer.
    Grabbing,
    NResize,
    SResize,
    EResize,
    WResize,
    NeResize,
    NwResize,
    SeResize,
    SwResize,
}

/// Borderless-window edge used with `Window::drag_resize_window`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowResizeEdge {
    East,
    North,
    NorthEast,
    NorthWest,
    South,
    SouthEast,
    SouthWest,
    West,
}

impl WindowResizeEdge {
    pub fn cursor(self) -> UiCursor {
        match self {
            Self::East => UiCursor::EResize,
            Self::North => UiCursor::NResize,
            Self::NorthEast => UiCursor::NeResize,
            Self::NorthWest => UiCursor::NwResize,
            Self::South => UiCursor::SResize,
            Self::SouthEast => UiCursor::SeResize,
            Self::SouthWest => UiCursor::SwResize,
            Self::West => UiCursor::WResize,
        }
    }
}

#[derive(Debug, Default)]
pub struct FrameUiOutput {
    pub actions: Vec<PanelAction>,
    pub selected: Option<LayerId>,
    pub request_undo: bool,
    pub request_redo: bool,
    pub request_save: bool,
    pub request_save_as: bool,
    pub request_load_path: bool,
    pub request_new_project: bool,
    pub request_close_project: bool,
    /// Pick an export directory without starting the job.
    pub request_export_path: bool,
    /// Start export to the current path (or pick one first if unset).
    pub request_start_export: bool,
    pub camera_reset: bool,
    pub camera_top_view: bool,
    pub camera_frame_selection: bool,
    /// Cancel the in-flight evaluation / refine job.
    pub request_cancel_build: bool,
    /// Force a full-quality rebuild (EXPORT button).
    pub request_full_build: bool,
    /// Save the current camera into the next free bookmark slot.
    pub request_save_bookmark: bool,
    /// Save into a specific bookmark slot (0..8).
    pub request_save_bookmark_slot: Option<usize>,
    /// Recall a bookmark slot (0..8).
    pub request_recall_bookmark: Option<usize>,
    /// Custom caption: minimize the OS window.
    pub request_window_minimize: bool,
    /// Custom caption: toggle maximized / restored.
    pub request_window_toggle_maximize: bool,
    /// Custom caption: quit the application.
    pub request_window_close: bool,
    /// Start an OS window move drag (title-bar empty space).
    pub request_window_drag: bool,
    /// Start an OS border resize drag.
    pub request_window_drag_resize: Option<WindowResizeEdge>,
    /// Preferred OS cursor for this frame.
    pub cursor: UiCursor,
}

/// Draw all editor chrome with terra-gui.
pub fn draw_editor_gui(
    ui: &mut GuiContext<'_>,
    doc: &TerrainDocument,
    ui_state: &mut UiState,
    chrome: &mut ChromeGuiState,
    tools: &mut ToolsGuiState,
    layers: &mut LayersGuiState,
    inspector: &mut InspectorGuiState,
    windows: &mut WindowsGuiState,
    dock: &mut DockGuiState,
) -> FrameUiOutput {
    ui_state.ensure_sculpt_defaults();

    let mut out = FrameUiOutput {
        selected: doc.selected,
        ..Default::default()
    };

    // Leaving Mask view (Views bar / Done / Escape).
    if ui_state.pending_close_mask_editor {
        ui_state.pending_close_mask_editor = false;
        ui_state.show_mask_editor = false;
    }

    let mask_view = ui_state.is_mask_view();
    if mask_view {
        ui_state.layout.hide_mode_rail = true;
        ui_state.layout.left_dock_w = Some(ui_state.layout.mask_editor_panel_w.clamp(
            terra_gui::LayoutPrefs::MASK_EDITOR_MIN,
            terra_gui::LayoutPrefs::MASK_EDITOR_MAX,
        ));
        ui_state.show_mask_editor = true;
    } else {
        ui_state.layout.left_dock_w = None;
        ui_state.layout.hide_mode_rail = !ui_state.app_workspace.shows_mode_rail();
    }
    ui_state.layout.hide_layers_panel = !ui_state.app_workspace.shows_layers_tree();
    // Sync live layout into the GUI state used by rect helpers.
    ui.state.layout = ui_state.layout.clone();

    draw_menu_bar(ui, doc, ui_state, chrome, &mut out);

    // Open menus/popups must not leak press/release into chrome underneath.
    if chrome.blocks_background_input()
        || ui_state.show_quick_add
        || ui_state.show_export_unsupported
        || layers.context_menu.is_some()
        || layers.add_menu_open
        || inspector.more_menu_open
        || ui_state.lighting_menu_open
        || ui_state.camera_speed_menu_open
        || ui_state.viewport_context_menu.is_some()
        || ui.state.open_combo.is_some()
    {
        ui.suspend_pointer_edges();
    }

    if !mask_view {
        out.actions.extend(draw_tools_gui(ui, doc, ui_state, tools));
    }
    if ui_state.app_workspace.shows_layers_tree() {
        out.actions
            .extend(draw_layers_gui(ui, doc, ui_state, layers));
        if !matches!(
            out.cursor,
            UiCursor::NResize
                | UiCursor::SResize
                | UiCursor::EResize
                | UiCursor::WResize
                | UiCursor::NeResize
                | UiCursor::NwResize
                | UiCursor::SeResize
                | UiCursor::SwResize
        ) {
            out.cursor = layers.cursor_hint;
        }
    }
    out.actions
        .extend(draw_inspector_gui(ui, doc, ui_state, inspector));
    out.actions.extend(draw_viewport_overlays(
        ui,
        ui_state,
        doc,
        doc.metrics.world_size_x.max(doc.metrics.world_size_z),
    ));
    draw_bottom_dock(ui, doc, ui_state, dock, &mut out);
    draw_windows(ui, doc, ui_state, windows, &mut out);
    pipeline_gui::draw_pipeline_gui(ui, doc, ui_state, &mut windows.recipe, &mut out.actions);
    history_gui::draw_history_gui(ui, ui_state, &mut windows.history_scroll, &mut out);
    bookmarks_gui::draw_bookmarks_gui(ui, ui_state, &mut windows.bookmarks_scroll, &mut out);

    if ui.draw_layout_splitters() {
        ui_state.layout = ui.state.layout.clone();
        ui_state.layout_dirty = true;
    }

    // Menus last so they stack above panels and receive hit-testing.
    draw_menu_overlays(ui, ui_state, chrome, &mut out);
    // Global popups are drawn after menus and other chrome.
    let mut quick_add = std::mem::take(&mut ui_state.quick_add);
    out.actions
        .extend(ui.with_menu_input(|ui| draw_quick_add(ui, doc, ui_state, &mut quick_add)));
    ui_state.quick_add = quick_add;
    out.actions
        .extend(draw_viewport_context_menu(ui, doc, ui_state));
    let mut command_palette = std::mem::take(&mut ui_state.command_palette);
    for action in draw_command_palette(ui, doc, ui_state, &mut command_palette) {
        match action {
            PaletteAction::Panel(action) => out.actions.push(action),
            PaletteAction::Undo => out.request_undo = true,
            PaletteAction::Redo => out.request_redo = true,
            PaletteAction::Export => ui_state.show_export = true,
            PaletteAction::Save => out.request_save = true,
            PaletteAction::SaveAs => out.request_save_as = true,
            PaletteAction::NewProject => out.request_new_project = true,
            PaletteAction::OpenProject => out.request_load_path = true,
            PaletteAction::CloseProject => out.request_close_project = true,
            PaletteAction::CameraReset => out.camera_reset = true,
            PaletteAction::CameraTopView => out.camera_top_view = true,
            PaletteAction::CameraFrameSelection => out.camera_frame_selection = true,
        }
    }
    ui_state.command_palette = command_palette;
    draw_export_unsupported_modal(ui, &mut ui_state.show_export_unsupported);
    out.selected = doc.selected;
    out
}
