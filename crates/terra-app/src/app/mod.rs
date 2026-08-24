//! Application shell: window, eval, project I/O, paint, and UI action dispatch.

mod actions;
mod editor_overlays;
mod eval;
mod export;
mod frame_trace;
mod helpers;
mod input;
mod lifecycle;
mod logical_frame;
mod paint;
mod pipeline_compile;
pub mod prefs;
mod project;
mod redraw;
mod refinement_job;
mod shapes;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::ui::{
    layers_from_project_template, ChromeGuiState, DockGuiState, InspectorGuiState, LayersGuiState,
    NewWorldSettings, Preview2dMode, ProjectHomeGuiState, ProjectPrefs, ToolsGuiState, UiState,
    WindowsGuiState,
};
use terra_core::document::EditorSession;
use terra_core::heightfield::Heightfield;
use terra_core::layer::LayerId;
use terra_core::quality::PreviewQuality;
use terra_core::tiling::UvRect;
use terra_cpu_eval::{EvalScheduler, EvalWorker};
use terra_gpu::{
    GpuHeightPyramid, GpuHeightPyramidMaterializer, GpuPyramidErrorReadback,
    GpuPyramidPlanningMetadata, GpuTileAtlas,
};
use terra_gpu_eval::{GpuCompiledTileProducer, GpuTerrainEngine, GpuTileEvaluationJob};
use terra_gui::{GuiRenderer, GuiState, Rect, WidgetLabState};
use terra_io::{BackgroundExporter, BackgroundProjectIo};
use terra_render::TerrainRenderer;
use winit::event::MouseButton;
use winit::event_loop::EventLoopProxy;
use winit::window::Window;

pub(crate) const REFINE_INTERVAL_MS: u128 = 80;
/// Required quiet time after input before optional Medium/Full work may start.
pub(crate) const POST_INPUT_REFINE_GRACE_MS: u64 = 80;
/// Host-side logical-frame budget. This is a start/defer gate, never a promise
/// that an already-started GPU submission can be preempted.
pub(crate) const LOGICAL_FRAME_HOST_BUDGET_MS: u64 = 8;
pub(crate) const FULL_FIELD_SETTLE_MS: u64 = 75;
pub(crate) const FULL_FIELD_REFINE_MS: u64 = 225;

struct CompiledTileWorkJob {
    lease: terra_core::TerrainTileWorkLease,
    engine: GpuTileEvaluationJob,
}

#[derive(Debug, Clone)]
pub enum RuntimeEvent {
    DeviceLost {
        reason: wgpu::DeviceLostReason,
        message: String,
    },
}

/// GPU objects built off the main thread during startup, handed back through
/// [`BootState::job`]. All three are `wgpu`-backed and therefore `Send`.
pub(crate) struct BootResult {
    renderer: TerrainRenderer,
    tile_atlas: Option<GpuTileAtlas>,
    gpu_engine: GpuTerrainEngine,
    gpu_pyramid_materializer: GpuHeightPyramidMaterializer,
}

/// Startup state held while the renderer's pipelines/shaders compile on a worker
/// thread. The main thread keeps the surface (in `pending`) so it can animate the
/// splash every frame; when the worker sends its [`BootResult`], the surface is
/// attached and the objects installed into the app.
pub(crate) struct BootState {
    pub(crate) gpu: terra_render::GpuContext,
    pub(crate) pending: terra_render::PendingSurface,
    pub(crate) job: terra_jobs::JobHandle<BootResult>,
    pub(crate) started: Instant,
    /// Boot worker failure, parked here while the failure splash is shown.
    pub(crate) failure: Option<crate::startup::StartupError>,
}

/// Continuous fly keys for the game-engine-style viewport camera.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct CameraKeys {
    pub w: bool,
    pub a: bool,
    pub s: bool,
    pub d: bool,
    pub q: bool,
    pub e: bool,
}

impl CameraKeys {
    pub fn any(self) -> bool {
        self.w || self.a || self.s || self.d || self.q || self.e
    }

    pub fn set(&mut self, code: winit::keyboard::KeyCode, pressed: bool) {
        use winit::keyboard::KeyCode::*;
        match code {
            KeyW => self.w = pressed,
            KeyA => self.a = pressed,
            KeyS => self.s = pressed,
            KeyD => self.d = pressed,
            KeyQ => self.q = pressed,
            KeyE => self.e = pressed,
            _ => {}
        }
    }
}

/// Discrete stage completion for Draft / Medium / Full.
pub(crate) fn quality_stage_progress(quality: PreviewQuality) -> f32 {
    match quality {
        PreviewQuality::Draft => 1.0 / 3.0,
        PreviewQuality::Medium => 2.0 / 3.0,
        PreviewQuality::Full | PreviewQuality::Export => 1.0,
    }
}

/// Progress bar fill while a refine job for `quality` is in flight.
/// Asymptotically approaches the stage ceiling so a long Medium/Full never looks frozen.
pub(crate) fn quality_in_flight_progress(quality: PreviewQuality, elapsed_secs: f32) -> f32 {
    let (lo, hi) = match quality {
        PreviewQuality::Draft => (0.0, 1.0 / 3.0),
        PreviewQuality::Medium => (1.0 / 3.0, 2.0 / 3.0),
        PreviewQuality::Full | PreviewQuality::Export => (2.0 / 3.0, 1.0),
    };
    let fill = 1.0 - (-elapsed_secs * 0.4).exp();
    let wobble = (elapsed_secs * 2.2).sin() * 0.02;
    (lo + (hi - lo) * (0.12 + 0.82 * fill) + wobble).clamp(lo + 0.02, hi - 0.01)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum AppScreen {
    #[default]
    Home,
    Editor,
}

/// Deferred project action waiting on a dirty-document confirm.
#[derive(Debug, Clone)]
pub(crate) enum PendingProjectAction {
    New,
    Open,
    Close,
    OpenPath(PathBuf),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayerPointKind {
    Path,
    Polygon,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct LayerPointDrag {
    layer: LayerId,
    index: usize,
    kind: LayerPointKind,
    start_cursor: (f64, f64),
    start_u: f32,
    start_v: f32,
    start_height: f32,
}

#[derive(Debug, Clone)]
pub(crate) struct DeferredFullField {
    pub generation: u64,
    pub layer_name: String,
    pub deferred_layers: usize,
    /// `None` while the bounded terrain gesture remains active.
    pub settle_at: Option<Instant>,
}

impl DeferredFullField {
    fn hold_during_gesture(&mut self) {
        self.settle_at = None;
    }

    /// Arm exactly once after the gesture ends. Further idle ticks do not debounce
    /// the deadline, so the suffix starts 75 ms after mouse-up rather than 75 ms
    /// after whichever event loop tick happened most recently.
    fn arm_after_gesture(&mut self, now: Instant) -> bool {
        if self.settle_at.is_some() {
            return false;
        }
        self.settle_at = Some(now + Duration::from_millis(FULL_FIELD_SETTLE_MS));
        true
    }

    fn ready(&self, generation: u64, now: Instant) -> bool {
        self.generation == generation && self.settle_at.is_some_and(|deadline| now >= deadline)
    }
}

pub struct TerraApp {
    runtime_event_proxy: Option<EventLoopProxy<RuntimeEvent>>,
    window: Option<Arc<Window>>,
    /// Set only during startup while GPU pipelines compile on a worker thread;
    /// `None` once the renderer is installed. Drives the animated splash.
    boot: Option<BootState>,
    renderer: Option<TerrainRenderer>,
    editor_overlays: Option<editor_overlays::EditorOverlays>,
    /// App-owned GPU handles. Every GPU consumer (renderer, tile atlas, terrain
    /// engine, GUI) shares clones of this instead of sourcing device/queue
    /// through the renderer.
    gpu: Option<terra_render::GpuContext>,
    pipeline_compile: pipeline_compile::TerrainPipelineCompileCoordinator,
    project_generation: u64,
    device_generation: u64,
    session: EditorSession,
    ui_state: UiState,
    scheduler: EvalScheduler,
    /// Final-output tile residency, revisioning, and progressive refinement state.
    terrain_runtime: terra_core::TerrainRuntime,
    runtime_started: Instant,

    /// Dedicated CPU evaluator for Medium/Full progressive refinement.
    eval_worker: EvalWorker,
    eval_token: u64,
    /// OS events accumulated until the next event-loop scheduling boundary.
    input: input::InputAccumulator,
    /// Owned effects emitted by immediate-mode UI presentation. They are applied
    /// during the next logical frame's application-update phase, never from the
    /// `RedrawRequested` callback that produced them.
    pending_ui_effects: VecDeque<redraw::PendingUiEffects>,
    /// Latest OS surface size captured by `window_event`; repeated resize events
    /// coalesce here until the next logical frame applies the final size.
    pending_surface_resize: Option<winit::dpi::PhysicalSize<u32>>,
    pending_surface_reconfigure: bool,
    /// Logical scheduling/diagnostic lifecycle; deliberately app-owned and non-global.
    logical_frames: logical_frame::LogicalFrameCoordinator,
    /// Bounded, frame/generation-correlated observability for the Base brush path.
    frame_trace: frame_trace::FrameTraceRecorder,
    /// Optional Medium/Full GPU work, resumable only at safe allocation and
    /// compiled-operation submission boundaries.
    refinement_job: Option<refinement_job::RefinementJob>,
    next_refinement_job_id: u64,
    /// Most recent complete terrain generation made available for presentation.
    last_complete_generation: logical_frame::EditGeneration,
    last_accepted_evaluation_id: u64,
    last_reported_presentation_timing_frame: u64,
    /// A matching Medium/Full job is queued or executing on the worker.
    worker_refine_pending: bool,
    /// Earliest edited layer not yet communicated to the persistent worker cache.
    worker_dirty_from: Option<LayerId>,
    /// Spatial scope (normalized UV) of the pending bounded suffix dirty, carried
    /// to the worker beside `worker_dirty_from`. `Some` only while a bounded suffix
    /// is pending; any footprint-less trigger escalates it to `None` (a whole-field
    /// suffix, today's behavior). Cleared with the other accumulators when a fresh
    /// result is consumed.
    worker_dirty_region: Option<UvRect>,
    /// Resolution the worker's persistent layer cache currently holds (the last
    /// fresh result's width), or `None` when unknown/invalidated. Gates the
    /// straight-to-Full sculpt policy: a bounded scope may skip the Draft/Medium
    /// CPU rungs only when the Full-res checkpoints it would reuse already exist.
    worker_cache_res: Option<u32>,
    /// Project/stack changes require the worker to discard every cached suffix.
    worker_mark_all_dirty: bool,

    last_height: Option<Heightfield>,
    project_path: Option<PathBuf>,
    exporter: BackgroundExporter,
    height_pyramid_export: export::HeightPyramidExportController,
    project_io: BackgroundProjectIo,
    /// Zero-sized registry adapter for the tool-thumbnail decode pool.
    tool_thumbs: crate::ui::ToolThumbPump,
    /// Per-frame poll registry: one `about_to_wait` tick pumps the background
    /// subsystems above (exporter, project IO, thumbnails) and aggregates their
    /// pending/wake facts. Behind an `Arc` so the tick can take `&mut self` while
    /// the entry list — fixed at construction — is read through the shared handle.
    jobs: Arc<terra_jobs::JobRegistry<TerraApp>>,
    /// After async save of a newly created project, enter the editor with this doc.
    pending_enter_after_save: Option<(terra_core::document::TerrainDocument, PathBuf)>,
    mouse_pressed: Option<MouseButton>,
    last_cursor: Option<(f64, f64)>,
    /// Cursor position when the current mouse button was pressed (for click-vs-drag).
    mouse_press_cursor: Option<(f64, f64)>,
    /// Accumulated movement while RMB held (to distinguish pan from context-click).
    right_drag_distance: f32,
    modifiers_shift: bool,
    modifiers_alt: bool,
    modifiers_ctrl: bool,
    modifiers_super: bool,
    /// Held WASD/QE for continuous camera fly (game-engine viewport).
    camera_keys: CameraKeys,
    /// Last time WASD fly was applied (for dt).
    last_camera_move: Instant,
    needs_height_upload: bool,
    last_refine: Instant,
    last_edit: Instant,
    pending_eval: bool,
    /// Required evaluation requested with no edit debounce. This is an explicit
    /// work intent consumed only by the coordinator's interactive-work phase.
    pending_eval_immediate: bool,
    /// When true, next eval starts from Draft even if already refining.
    force_draft: bool,
    /// Bounded edit scope waiting for the next GPU evaluation. UV is retained
    /// until quality/metrics are known, then converted to texels.
    pending_gpu_dirty_region: Option<UvRect>,
    /// Missing globally coupled suffix for the latest local edit generation.
    deferred_full_field: Option<DeferredFullField>,
    /// Wave C GPU layer preview engine (shares renderer device).
    gpu_engine: Option<GpuTerrainEngine>,
    /// Backend-neutral structural plan and pending semantic edit batch.
    terrain_plan_cache: terra_core::terrain_plan::TerrainPlanCache,
    pending_plan_edits: Vec<terra_core::terrain_plan::TerrainEditClass>,
    pending_plan_invalidation: Option<terra_core::terrain_plan::PlanInvalidation>,
    /// The last interactive path can complete entirely on the GPU. A locally
    /// truthful result may still have a deferred full-field GPU suffix; that is
    /// distinct from requiring CPU fallback and must not demote the next dab.
    last_eval_gpu_supported: bool,
    /// Progressive final-output tile atlas used by the LOD renderer migration.
    tile_atlas: Option<GpuTileAtlas>,
    /// Immutable GPU content pyramid for the latest accepted complete output.
    gpu_height_pyramid: Option<GpuHeightPyramid>,
    gpu_pyramid_materializer: Option<GpuHeightPyramidMaterializer>,
    /// Compact measured-error transfer and immutable CPU planning snapshot. These
    /// describe pyramid content and demand only; neither mirrors atlas residency.
    gpu_pyramid_error_readback: Option<GpuPyramidErrorReadback>,
    gpu_pyramid_planning_metadata: Option<GpuPyramidPlanningMetadata>,
    terrain_demand_planner: terra_core::TerrainDemandPlanner,
    latest_terrain_demand: Option<terra_core::TerrainDemandPlan>,
    /// Bounded, revision-aware demand awaiting concrete CPU/GPU atlas publication.
    terrain_tile_scheduler: terra_core::TerrainTileWorkScheduler,
    compiled_tile_producer: GpuCompiledTileProducer,
    compiled_tile_jobs: Vec<CompiledTileWorkJob>,
    /// Monotonic identity for accepted CPU heightfields within an output revision.
    next_cpu_tile_content_revision: u64,
    gui_renderer: Option<GuiRenderer>,
    gui_state: GuiState,
    widget_lab: WidgetLabState,
    chrome_gui: ChromeGuiState,
    tools_gui: ToolsGuiState,
    inspector_gui: InspectorGuiState,
    layers_gui: LayersGuiState,
    windows_gui: WindowsGuiState,
    dock_gui: DockGuiState,
    /// Consumed each frame by terra-gui (mouse wheel).
    gui_scroll_delta: f32,
    /// Text/key edits consumed by the UI search popups on the next redraw.
    gui_text: String,
    gui_backspace: bool,
    gui_escape: bool,
    gui_enter: bool,
    /// Pointer edges accumulated from sealed input snapshots until GUI presentation.
    gui_primary_pressed: bool,
    gui_primary_released: bool,
    gui_secondary_pressed: bool,
    gui_secondary_released: bool,
    /// Last frame: custom UI captured the pointer (blocks camera/paint).
    gui_wants_pointer: bool,
    /// Quit requested from the custom caption close button.
    pending_exit: bool,
    /// Startup failure stored for reporting after the event loop exits.
    startup_failure: Option<crate::startup::StartupError>,
    /// The failure was already shown on the boot-failure splash (skip dialog).
    failure_presented: bool,
    /// True while GUI has active pointer capture (drag/scroll/text) â€” not hover.
    gui_interacting: bool,
    /// A sculpt gesture changed the base buffer; represented in History as an annotation.
    sculpt_stroke_active: bool,
    /// Last terrain UV stamped this stroke â€” used to fill gaps on fast brush moves.
    last_paint_uv: Option<(f32, f32)>,
    /// App-owned 3D viewport in logical pixels.
    viewport_rect: Rect,
    /// Rebuild live 2D preview only when height/mode changes (not every frame).
    preview_dirty: bool,
    /// Rebuild region viz overlay only when world/eval/mode changes.
    /// Cached UI fingerprints to avoid per-frame clones.
    ui_history_fp: (usize, usize),
    ui_outdated_fp: usize,
    ui_soft_diag_fp: u64,
    last_preview_mode: Preview2dMode,
    /// Home (project manager) vs full editor.
    screen: AppScreen,
    /// True when the document has unsaved edits.
    document_dirty: bool,
    project_prefs: ProjectPrefs,
    project_home: ProjectHomeGuiState,
    /// When set, show discard confirm before performing this action.
    pending_project_action: Option<PendingProjectAction>,
    /// New Project template picker overlay.
    show_new_template_picker: bool,
    new_template_selected: String,
    new_world_settings: NewWorldSettings,
    /// Landforms: dragging a shape control point (shape id + point index).
    dragging_shape_point: Option<(terra_core::shape_object::ShapeObjectId, usize)>,
    /// Layer-native Path / Polygon direct-manipulation session.
    dragging_layer_point: Option<LayerPointDrag>,
    /// Polygon fill vertices while BiomePaintTool::PolygonFill is active.
    biome_polygon_points: Vec<(f32, f32)>,
    /// Region boundary polygon vertices (DrawRegionPolygon tool).
    /// First corner UV for DrawRegionRect (second click commits).
    /// Measure tool: first click UV; second click reports distance.
    measure_anchor: Option<(f32, f32)>,
    /// Re-upload placement tint to the 3D viewport.
    placement_tint_dirty: bool,
    /// Re-upload active mask paint overlay (shares placement-tint GPU slot).
    mask_overlay_dirty: bool,
    /// Skip redundant aux / vegetation / overhang GPU syncs.
    aux_upload_fp: u64,
    veg_upload_fp: u64,
    overhang_upload_fp: u64,
    /// Detect lighting preset changes for progressive invalidation.
    last_lighting_preset: crate::ui::LightingPreset,
    last_lighting_customized: bool,
    /// Last mask id uploaded to the viewport overlay (detect enter/leave paint mode).
    last_mask_overlay_id: Option<terra_core::mask::MaskId>,
    /// Region Mask Editor session â€” paint/op edits invalidate mask cache only.
    /// Terrain rebuild deferred until Apply / editor close / explicit apply.
    /// Paint buffer snapshot at the start of a mask paint stroke (for undo).
    mask_paint_stroke_before: Option<(terra_core::mask::MaskId, Vec<f32>, u32, u32)>,
}

impl Default for TerraApp {
    fn default() -> Self {
        let now = Instant::now();
        let prefs = crate::app::prefs::load_editor_prefs();
        let mut gui_state = GuiState::default();
        gui_state.layout = prefs.layout.clone();
        let mut ui_state = UiState::default();
        ui_state.viewport_render =
            crate::ui::ViewportRenderSettings::from_prefs(&prefs.viewport_render);
        ui_state.preferred_workspace = prefs.preferred_workspace;
        ui_state.auto_switch_workspace_on_create = prefs.auto_switch_workspace_on_create;
        ui_state.layout = prefs.layout;
        ui_state.apply_preferred_workspace_from_prefs();
        let session = EditorSession::new();
        let bounded = session
            .document
            .bounded_settings()
            .expect("default editor session is bounded");
        let metrics = bounded.metrics;
        let terrain_runtime = terra_core::TerrainRuntime::new(terra_core::PyramidConfig::new(
            bounded.preview_resolution,
            metrics.world_size_x,
            metrics.world_size_z,
        ));
        // Register the background subsystems that `about_to_wait` pumps each
        // frame, in the order it used to poll them by hand: export, project IO,
        // then the tool-thumbnail pool.
        let mut jobs = terra_jobs::JobRegistry::<TerraApp>::new();
        jobs.register(|app| &mut app.exporter);
        jobs.register(|app| &mut app.project_io);
        jobs.register(|app| &mut app.tool_thumbs);
        jobs.register(|app| &mut app.pipeline_compile);
        Self {
            runtime_event_proxy: None,
            window: None,
            boot: None,
            renderer: None,
            editor_overlays: None,
            gpu: None,
            pipeline_compile: pipeline_compile::TerrainPipelineCompileCoordinator::default(),
            project_generation: 0,
            device_generation: 0,
            session,
            ui_state,
            scheduler: EvalScheduler::new(),
            terrain_runtime,
            runtime_started: now,
            eval_worker: EvalWorker::spawn(),
            eval_token: 0,
            input: input::InputAccumulator::default(),
            pending_ui_effects: VecDeque::new(),
            pending_surface_resize: None,
            pending_surface_reconfigure: false,
            logical_frames: logical_frame::LogicalFrameCoordinator::default(),
            frame_trace: frame_trace::FrameTraceRecorder::default(),
            refinement_job: None,
            next_refinement_job_id: 0,
            last_complete_generation: logical_frame::EditGeneration::default(),
            last_accepted_evaluation_id: 0,
            last_reported_presentation_timing_frame: 0,
            worker_refine_pending: false,
            worker_dirty_from: None,
            worker_dirty_region: None,
            worker_cache_res: None,
            worker_mark_all_dirty: true,
            last_height: None,
            project_path: None,
            exporter: BackgroundExporter::new(),
            height_pyramid_export: export::HeightPyramidExportController::default(),
            project_io: BackgroundProjectIo::new(),
            tool_thumbs: crate::ui::ToolThumbPump,
            jobs: Arc::new(jobs),
            pending_enter_after_save: None,
            mouse_pressed: None,
            last_cursor: None,
            mouse_press_cursor: None,
            right_drag_distance: 0.0,
            modifiers_shift: false,
            modifiers_alt: false,
            modifiers_ctrl: false,
            modifiers_super: false,
            camera_keys: CameraKeys::default(),
            last_camera_move: now,
            needs_height_upload: false,
            last_refine: now,
            last_edit: now,
            pending_eval: false,
            pending_eval_immediate: false,
            force_draft: false,
            pending_gpu_dirty_region: None,
            deferred_full_field: None,
            gpu_engine: None,
            terrain_plan_cache: terra_core::terrain_plan::TerrainPlanCache::new(),
            pending_plan_edits: vec![terra_core::terrain_plan::TerrainEditClass::Structure],
            pending_plan_invalidation: None,
            last_eval_gpu_supported: false,
            tile_atlas: None,
            gpu_height_pyramid: None,
            gpu_pyramid_materializer: None,
            gpu_pyramid_error_readback: None,
            gpu_pyramid_planning_metadata: None,
            terrain_demand_planner: terra_core::TerrainDemandPlanner::default(),
            latest_terrain_demand: None,
            terrain_tile_scheduler: terra_core::TerrainTileWorkScheduler::default(),
            compiled_tile_producer: GpuCompiledTileProducer::new(),
            compiled_tile_jobs: Vec::new(),
            next_cpu_tile_content_revision: 0,
            gui_renderer: None,
            gui_state,
            widget_lab: WidgetLabState::default(),
            chrome_gui: ChromeGuiState::default(),
            tools_gui: ToolsGuiState::default(),
            inspector_gui: InspectorGuiState::default(),
            layers_gui: LayersGuiState::default(),
            windows_gui: WindowsGuiState::default(),
            dock_gui: DockGuiState,
            gui_scroll_delta: 0.0,
            gui_text: String::new(),
            gui_backspace: false,
            gui_escape: false,
            gui_enter: false,
            gui_primary_pressed: false,
            gui_primary_released: false,
            gui_secondary_pressed: false,
            gui_secondary_released: false,
            gui_wants_pointer: false,
            pending_exit: false,
            startup_failure: None,
            failure_presented: false,
            gui_interacting: false,
            sculpt_stroke_active: false,
            last_paint_uv: None,
            viewport_rect: Rect::from_min_max(88.0, 44.0, 1300.0, 690.0),
            preview_dirty: true,
            ui_history_fp: (usize::MAX, usize::MAX),
            ui_outdated_fp: usize::MAX,
            ui_soft_diag_fp: u64::MAX,
            last_preview_mode: Preview2dMode::Height,
            screen: AppScreen::Home,
            document_dirty: false,
            project_prefs: load_project_prefs(),
            project_home: ProjectHomeGuiState::default(),
            pending_project_action: None,
            show_new_template_picker: false,
            new_template_selected: "blank".into(),
            new_world_settings: NewWorldSettings::default(),
            dragging_shape_point: None,
            dragging_layer_point: None,
            biome_polygon_points: Vec::new(),
            measure_anchor: None,
            placement_tint_dirty: true,
            mask_overlay_dirty: true,
            aux_upload_fp: u64::MAX,
            veg_upload_fp: u64::MAX,
            overhang_upload_fp: u64::MAX,
            last_lighting_preset: crate::ui::LightingPreset::Studio,
            last_lighting_customized: false,
            last_mask_overlay_id: None,
            mask_paint_stroke_before: None,
        }
    }
}

impl TerraApp {
    pub fn set_runtime_event_proxy(&mut self, proxy: EventLoopProxy<RuntimeEvent>) {
        self.runtime_event_proxy = Some(proxy);
    }
}

impl TerraApp {
    pub fn take_startup_failure(&mut self) -> Option<(crate::startup::StartupError, bool)> {
        self.startup_failure
            .take()
            .map(|e| (e, self.failure_presented))
    }
}

pub(crate) fn project_prefs_path() -> PathBuf {
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("terra_projects.json")
}

pub(crate) fn load_project_prefs() -> ProjectPrefs {
    let path = project_prefs_path();
    if let Ok(bytes) = std::fs::read(&path) {
        if let Ok(prefs) = serde_json::from_slice::<ProjectPrefs>(&bytes) {
            return prefs;
        }
    }
    ProjectPrefs::default()
}

pub(crate) fn save_project_prefs(prefs: &ProjectPrefs) {
    let path = project_prefs_path();
    if let Ok(json) = serde_json::to_vec_pretty(prefs) {
        let _ = std::fs::write(path, json);
    }
}

/// `Documents/Terra` (falls back to `./Terra` if Documents is unavailable).
pub(crate) fn default_terra_projects_dir() -> PathBuf {
    let docs = directories::UserDirs::new()
        .and_then(|dirs| dirs.document_dir().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    docs.join("Terra")
}

pub(crate) fn sanitize_project_name(raw: &str) -> String {
    let trimmed = raw.trim();
    let mut name: String = trimmed
        .chars()
        .map(|c| match c {
            '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*' => '_',
            c if c.is_control() => '_',
            c => c,
        })
        .collect();
    while name.ends_with('.') || name.ends_with(' ') {
        name.pop();
    }
    if name.is_empty() {
        "Untitled".into()
    } else {
        name
    }
}

pub(crate) fn project_name_from_path(path: &std::path::Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Untitled");
    sanitize_project_name(stem)
}

/// `Documents/Terra/<name>/<name>.json`, creating the project folder as needed.
pub(crate) fn prepare_project_path(
    projects_root: &std::path::Path,
    name: &str,
) -> std::io::Result<PathBuf> {
    let project_dir = projects_root.join(name);
    std::fs::create_dir_all(&project_dir)?;
    Ok(project_dir.join(format!("{name}.json")))
}

pub(crate) fn document_from_template(
    template_id: &str,
) -> Option<terra_core::document::TerrainDocument> {
    match template_id {
        "blank" => Some(terra_core::blank_world_design(8192.0, 512)),
        "tropical_island" => Some(terra_core::tropical_island_world(10_000.0, 512)),
        "alpine" => Some(terra_core::alpine_world(10_000.0, 512)),
        "desert" => Some(terra_core::desert_world(10_000.0, 512)),
        "river_valley" => Some(terra_core::river_valley_world(10_000.0, 512)),
        "badlands" => Some(terra_core::badlands_world(10_000.0, 512)),
        "young_mountains" => Some(terra_core::young_mountains_world(10_000.0, 512)),
        "old_mountains" => Some(terra_core::old_mountains_world(10_000.0, 512)),
        "dune_field" => Some(terra_core::dune_field_world(10_000.0, 512)),
        "coastal" => Some(terra_core::coastal_world(10_000.0, 512)),
        other => {
            let layers = layers_from_project_template(other)?;
            Some(terra_core::document::TerrainDocument::from_flat_layers(
                layers,
            ))
        }
    }
}

pub(crate) fn document_from_world_settings(
    template_id: &str,
    world_size_m: f32,
    sea_level: f32,
) -> terra_core::document::TerrainDocument {
    let preview_res = terra_core::preview_resolution_for_world_size(world_size_m);
    let mut doc = match template_id {
        "tropical_island" => terra_core::tropical_island_world(world_size_m, preview_res),
        "alpine" => terra_core::alpine_world(world_size_m, preview_res),
        "desert" => terra_core::desert_world(world_size_m, preview_res),
        "river_valley" => terra_core::river_valley_world(world_size_m, preview_res),
        "badlands" => terra_core::badlands_world(world_size_m, preview_res),
        "young_mountains" => terra_core::young_mountains_world(world_size_m, preview_res),
        "old_mountains" => terra_core::old_mountains_world(world_size_m, preview_res),
        "dune_field" => terra_core::dune_field_world(world_size_m, preview_res),
        "coastal" => terra_core::coastal_world(world_size_m, preview_res),
        "blank" => terra_core::blank_world_design(world_size_m, preview_res),
        other => document_from_template(other)
            .unwrap_or_else(|| terra_core::blank_world_design(world_size_m, preview_res)),
    };
    doc.blueprint.sea_level = sea_level;
    doc.blueprint.world_size_m = world_size_m;
    apply_blueprint_to_stack(&mut doc);
    doc
}

/// Map artist blueprint controls onto stack process layers + shape widths.
pub(crate) fn apply_blueprint_to_stack(doc: &mut terra_core::document::TerrainDocument) {
    let evo_iters = doc.blueprint.evolution_iterations();
    let ridge_w = doc.blueprint.ridge_width_m();
    let rainfall = doc.blueprint.rainfall_scale();
    for shape in &mut doc.shapes.shapes {
        if matches!(
            shape.kind,
            terra_core::ShapeKind::MountainSpine | terra_core::ShapeKind::RidgeSpline
        ) {
            shape.width_m = ridge_w;
        }
    }
    if !doc.shapes.shapes.is_empty() || doc.shapes.managed_constraints_layer.is_some() {
        doc.compile_shapes_into_stack();
    }
    for layer in doc.stack.flatten_layers_mut() {
        match &mut layer.kind {
            terra_core::LayerKind::LandscapeEvolution(p) => {
                p.iterations = evo_iters;
                p.geological_age = doc.blueprint.geological_age_norm();
                p.rainfall = rainfall;
                p.drainage_scale = doc.blueprint.drainage_density.clamp(0.0, 1.0);
                p.incision_k = (0.00035 * rainfall).clamp(1e-5, 0.01);
                p.erosion = (0.35 + 0.45 * rainfall.min(2.0)).clamp(0.1, 1.5);
            }
            terra_core::LayerKind::Island(p) => {
                p.sea_level = doc.blueprint.sea_level;
            }
            _ => {}
        }
    }
}

pub(crate) fn ensure_hydrology_processes(doc: &mut terra_core::document::TerrainDocument) {
    use terra_core::authoring::{
        GeomorphicDetailParams, HydrologyRepairParams, LandscapeEvolutionParams,
    };
    use terra_core::layer::{Layer, LayerKind};
    let layers = doc.stack.flatten_layers();
    let has_evo = layers
        .iter()
        .any(|l| matches!(l.kind, LayerKind::LandscapeEvolution(_)));
    let has_repair = layers
        .iter()
        .any(|l| matches!(l.kind, LayerKind::HydrologyRepair(_)));
    let has_detail = layers
        .iter()
        .any(|l| matches!(l.kind, LayerKind::GeomorphicDetail(_)));
    drop(layers);
    if !has_evo {
        let mut evo = LandscapeEvolutionParams::default();
        evo.iterations = doc.blueprint.evolution_iterations();
        evo.geological_age = doc.blueprint.geological_age_norm();
        evo.rainfall = doc.blueprint.rainfall_scale();
        evo.drainage_scale = doc.blueprint.drainage_density.clamp(0.0, 1.0);
        doc.stack.push_into_category(Layer::new(
            "Landscape Evolution",
            LayerKind::LandscapeEvolution(evo),
        ));
    }
    if !has_repair {
        doc.stack.push_into_category(Layer::new(
            "Hydrology Repair",
            LayerKind::HydrologyRepair(HydrologyRepairParams::default()),
        ));
    }
    if !has_detail {
        doc.stack.push_into_category(Layer::new(
            "Geomorphic Detail",
            LayerKind::GeomorphicDetail(GeomorphicDetailParams::default()),
        ));
    }
}

pub(crate) fn ensure_surface_processes(doc: &mut terra_core::document::TerrainDocument) {
    use terra_core::layer::{Layer, LayerKind, MaterialsParams, VegetationParams};
    let layers = doc.stack.flatten_layers();
    let has_mat = layers
        .iter()
        .any(|l| matches!(l.kind, LayerKind::Materials(_)));
    let has_veg = layers
        .iter()
        .any(|l| matches!(l.kind, LayerKind::Vegetation(_)));
    drop(layers);
    if !has_mat {
        doc.stack.push_into_category(Layer::new(
            "Materials",
            LayerKind::Materials(MaterialsParams::default()),
        ));
    }
    if !has_veg {
        doc.stack.push_into_category(Layer::new(
            "Vegetation",
            LayerKind::Vegetation(VegetationParams::default()),
        ));
    }
}
