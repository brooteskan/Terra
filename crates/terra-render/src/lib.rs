//! wgpu terrain viewport — GPU height textures + world-fixed grid displacement.
//!
//! Long-term layout: [`frame_graph`], [`backends`].
//! [`TerrainRenderer`] remains the strangler host until Phase E cleanup.
//!
//! # Frame seam
//!
//! Terrain + GUI compositing. A presented frame is composited by two crates
//! writing the *same* swapchain texture in sequence, ordered only by queue
//! submission. The protocol is:
//!
//! 1. [`TerrainRenderer::render_terrain`] acquires the swapchain frame in-crate
//!    (the sole `get_current_texture`), records and submits the terrain encoder,
//!    and returns the still-**un-presented** [`wgpu::SurfaceTexture`].
//! 2. The app creates a [`wgpu::TextureView`] of *that returned frame* and
//!    builds the GUI against it.
//! 3. `terra_gui::GuiRenderer::render` records and submits a **second** encoder
//!    whose pass uses `LoadOp::Load` — it composites over the terrain output
//!    instead of clearing it.
//! 4. The app presents (`SurfaceTexture::present`) — the sole present — only
//!    *after* both encoders are submitted.
//!
//! Nothing in the type system enforces steps 2–4: a `LoadOp::Clear` in the GUI
//! pass would erase the viewport, and a present before the GUI submit would show
//! a stale frame, both without a compile error. Two tests stand in for that
//! missing pushback — `terra-gui/tests/frame_seam.rs` locks the GUI pass's
//! `LoadOp::Load` against a plain backdrop, and `terra-app/tests/`
//! `frame_compositing.rs` drives the real terrain and GUI renderers into one
//! offscreen target and asserts the GUI lands while the terrain outside it
//! survives. Acquire-error semantics (`Lost`/`Outdated` recovery, `OutOfMemory`
//! exit) are typed through [`RenderError::Surface`]; see `render_terrain`.

pub mod adaptive_sampling;
pub mod backends;
pub mod brush;
pub mod camera;
pub mod clipmap;
pub mod frame_graph;
pub mod gpu_timing;
pub mod grid;
pub mod guides;
pub mod height_gpu;
mod integrity_probe;
mod optional_resource;
pub mod overhang;
pub mod path_tracer;
mod presentation_pipeline;
pub mod presentation_transition;
pub mod progressive;
pub mod render_quality;
pub mod retirement;
pub mod scene_versions;
pub mod shadows;
pub mod staging;
pub mod terrain_mesh;
mod terrain_pipeline;
mod terrain_shader;
pub mod vegetation;

pub use adaptive_sampling::{AdaptiveSamplingState, TileState, VarianceTileSummary, TILE_SIZE};
pub use backends::{
    GBufferViews, HdrFrame, PresentationBackendId, ProgressivePostPipeline, ProgressivePtOutput,
};
pub use brush::{pick_terrain_uv, pick_terrain_uv_on_surface, BrushOverlay, SurfacePick};
pub use camera::OrbitCamera;
pub use clipmap::{
    ClipmapConfig, ClipmapPresentInput, ClipmapPresentPlan, ClipmapRingDraw, ClipmapRingLevel,
    ClipmapTraversalBounds, WorldGridConfig,
};
pub use frame_graph::{FrameGraph, FrameSchedule, PassKind};
pub use gpu_timing::{GpuPresentationTraceContext, GpuTimings};
pub use grid::TerrainGrid;
pub use guides::{GuideOverlay, GuideState};
pub use height_gpu::{AuxMaps, HeightGpu, HeightPresentGeom};
pub use integrity_probe::TerrainIntegrityProbeResult;
pub use optional_resource::{OptionalResource, OptionalResourceState};
pub use overhang::OverhangOverlay;
pub use path_tracer::{PathTraceUniforms, PathTracer};
pub use presentation_pipeline::{
    PresentationPipelineBundle, PresentationPipelineCompiler, PresentationPipelineFeature,
    ProgressivePresentationBundle,
};
pub use presentation_transition::{
    PresentedTerrainBaseline, TerrainPresentationDecisionCode, TerrainPresentationExpectations,
    TerrainPresentationMode, TerrainPresentationRecord, TerrainTransitionDiagnosticCode,
};
pub use render_quality::{
    QualityPreset, RenderQualityConfig, ViewportQualityManager, ViewportRendererMode,
};
pub use scene_versions::{
    CameraChangeThresholds, CameraSnapshot, InvalidationReason, SceneVersionRegistry, SceneVersions,
};
pub use terra_core::EditorRefinementState;
pub use terrain_pipeline::{
    OceanPipelineBundle, TerrainPipelineBundle, TerrainPipelineCompileError,
    TerrainPipelineCompiler, WireframePipelineBundle,
};
pub use terrain_shader::TerrainShaderVariant;
pub use vegetation::VegetationOverlay;

use bytemuck::{Pod, Zeroable};
use terra_core::heightfield::Heightfield;
use terra_core::layer::MaterialsParams;
use terra_core::mask::MaskField;
use terra_core::tiling::SampleRect;
use thiserror::Error;
use winit::window::Window;

#[derive(Debug, Error)]
pub enum RenderError {
    /// Surface acquire failed. Returned unmapped so the app can match on the
    /// `wgpu::SurfaceError` variant and recover (reconfigure on Lost/Outdated,
    /// exit on OutOfMemory) rather than only log a stringified message.
    #[error(transparent)]
    Surface(#[from] wgpu::SurfaceError),
    #[error("{0}")]
    Msg(String),
}

/// Camera and clipmap traversal policy selected by the active project topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TerrainTraversalMode {
    #[default]
    Bounded,
    Infinite,
}

/// Fixed-origin spatial inputs required to derive a transient Infinite render frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InfinitePresentationConfig {
    pub topology: terra_core::InfiniteTopologyConfig,
    pub horizon_m: f64,
}

impl TerrainTraversalMode {
    fn constrain_camera(self, camera: &mut OrbitCamera, world_size: (f32, f32)) {
        if self == Self::Bounded {
            camera.clamp_to_world(world_size);
        }
    }
}

fn terrain_patch_index_count(grid: &TerrainGrid, traversal: TerrainTraversalMode) -> u32 {
    if traversal == TerrainTraversalMode::Infinite {
        grid.surface_index_count
    } else {
        grid.index_count
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniforms {
    view_proj: [[f32; 4]; 4],
    /// xyz = light direction (from light toward scene), w = sun intensity
    light_dir: [f32; 4],
    world: [f32; 4],
    /// x=tex_w, y=tex_h, z=ocean_level, w=slab base height
    grid: [f32; 4],
    /// World-fixed mesh: origin_x, origin_z, spacing_x (unused when UV*world), grid_size
    clipmap: [f32; 4],
    /// xyz = camera eye, w = exposure
    eye: [f32; 4],
    /// x = stochastic frame seed, y = progressive enabled, z = accumulated samples, w = biome tint
    render: [f32; 4],
    /// x = shading_mode, y = contours on, z = contour interval m, w = clipmap hole half-extent
    viz: [f32; 4],
    light_view_proj: [[f32; 4]; 4],
    /// x=use_tile_stream, y=tile_size, z=halo, w=max_pages
    stream: [f32; 4],
    /// x=fog_density, y=height_falloff, z=max_amount, w=sun_scatter
    fog: [f32; 4],
    /// x=shadow_enabled, y=depth_bias, z=stream_level, w=soft_scale
    shadow: [f32; 4],
    /// Raster shading controls: x=ambient_strength, y=shadow_strength, z=fog_strength, w=unused
    raster: [f32; 4],
    /// Document and plan revision halves for complete streamed-content identity.
    stream2: [u32; 4],
    /// Output/content revision halves, bitcast as raw u32 values.
    stream3: [u32; 4],
    /// x=level_count, y=target_level, z=current_frame_lo bits, w=transition_frames.
    stream4: [u32; 4],
    /// x=terminal monolithic allowed, y=stream debug mode, z/w reserved.
    stream5: [f32; 4],
    /// x=Infinite sparse addressing, y=sparse directory capacity, z=max LOD.
    stream6: [u32; 4],
    /// Signed finest-tile render anchor: x low/high, z low/high.
    stream7: [u32; 4],
    /// x=finest spacing metres, y=finest tile span metres,
    /// z/w=camera-relative X/Z centre of this draw's exclusion hole.
    stream8: [f32; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TerrainTerminalFallback {
    #[default]
    RootRequired,
    MonolithicMigration,
}

/// Complete shader-facing tile-stream resource set. Demand is deliberately not
/// represented here: rendered selection comes only from these GPU tables.
pub struct TerrainTileStreamResources {
    pub atlas_view: wgpu::TextureView,
    pub physical_page_table: wgpu::Buffer,
    pub virtual_page_table: wgpu::Buffer,
    pub level_table: wgpu::Buffer,
    pub tile_size: u32,
    pub halo: u32,
    pub max_pages: u32,
    pub level_count: u32,
    pub target_level: u8,
    pub target_resolution: u32,
    pub content: terra_core::TerrainContentStamp,
    pub transition_frames: u32,
    pub terminal_fallback: TerrainTerminalFallback,
    pub virtual_page_count: u32,
    pub infinite_topology: Option<terra_core::InfiniteTopologyConfig>,
    pub enable: bool,
}

/// Viewport false-color / analysis shading (mode bar).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum ViewportShadingMode {
    #[default]
    Lit = 0,
    Height = 1,
    Slope = 2,
    Flow = 3,
}

/// Display-aid flags pushed from the editor chrome each frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct ViewportDisplayAids {
    pub wireframe: bool,
    pub grid: bool,
    pub world_bounds: bool,
    pub contours: bool,
    pub shading: ViewportShadingMode,
}

const MATERIAL_SLOT_COUNT: usize = 17;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaterialGpu {
    albedo_roughness: [f32; 4],
    metalness_valid: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaterialPalette {
    entries: [MaterialGpu; MATERIAL_SLOT_COUNT],
}

impl MaterialPalette {
    fn from_params(
        params: Option<&terra_core::layer::MaterialsParams>,
        albedo_layers: &[i32; MATERIAL_SLOT_COUNT],
    ) -> Self {
        let colors = [
            [0.42, 0.40, 0.36],
            [0.48, 0.36, 0.24],
            [0.30, 0.42, 0.22],
            [0.62, 0.54, 0.36],
            [0.78, 0.80, 0.82],
        ];
        let mut entries = [MaterialGpu {
            albedo_roughness: [0.45, 0.42, 0.38, 0.85],
            metalness_valid: [0.0, 0.0, 0.0, 0.0],
        }; MATERIAL_SLOT_COUNT];
        for (id, color) in colors.into_iter().enumerate() {
            entries[id] = MaterialGpu {
                albedo_roughness: [color[0], color[1], color[2], 0.82],
                metalness_valid: [0.0, 1.0, (albedo_layers[id] + 1) as f32, 0.0],
            };
        }
        if let Some(params) = params {
            for rule in &params.rules {
                let id = (rule.id as usize).min(MATERIAL_SLOT_COUNT - 1);
                entries[id] = MaterialGpu {
                    albedo_roughness: [
                        rule.tint[0].max(0.0),
                        rule.tint[1].max(0.0),
                        rule.tint[2].max(0.0),
                        rule.roughness.clamp(0.04, 1.0),
                    ],
                    metalness_valid: [
                        rule.metalness.clamp(0.0, 1.0),
                        1.0,
                        (albedo_layers[id] + 1) as f32,
                        0.0,
                    ],
                };
            }
        }
        Self { entries }
    }
}

/// wgpu/Vulkan drivers often expect uniform bindings sized to 256-byte alignment.
const FRAME_UNIFORM_BUF_SIZE: u64 = 512;

#[derive(Clone, Copy)]
struct GpuPresentationAttempt {
    requested_mode: TerrainPresentationMode,
    actual_mode: TerrainPresentationMode,
    requested_rect: Option<SampleRect>,
    actual_rect: Option<SampleRect>,
    coherent_before: bool,
}

pub struct TerrainRenderer {
    /// Presentation surface. `None` for headless renderers built via `new_headless`.
    ///
    /// These four GPU handles are the renderer's private property: the app owns
    /// the [`GpuContext`] and hands it *in*, so nothing acquires a device, queue,
    /// or surface config back *through* the renderer. Keeping them private makes
    /// that reach-through a compile error rather than a convention (A1-G1).
    surface: Option<wgpu::Surface<'static>>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: std::sync::Arc<terra_gpu::PipelineCacheRegistry>,
    config: wgpu::SurfaceConfiguration,
    bounded_pipeline_bundle: TerrainPipelineBundle,
    infinite_pipeline_bundle: OptionalResource<TerrainPipelineBundle>,
    bounded_ocean_pipeline: OptionalResource<OceanPipelineBundle>,
    infinite_ocean_pipeline: OptionalResource<OceanPipelineBundle>,
    bounded_wireframe_pipeline: OptionalResource<WireframePipelineBundle>,
    infinite_wireframe_pipeline: OptionalResource<WireframePipelineBundle>,
    active_pipeline_variant: TerrainShaderVariant,
    pipeline_family: std::sync::Arc<()>,
    /// Single frame-uniform buffer for the world-fixed terrain grid.
    pub uniform_buf: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    /// Authored tint/roughness/metalness indexed by the material-ID map.
    material_palette_buf: wgpu::Buffer,
    /// RGBA8 albedo texture array — one layer per material slot (neutral grey when unused).
    albedo_array: wgpu::Texture,
    albedo_array_view: wgpu::TextureView,
    albedo_sampler: wgpu::Sampler,
    /// Per material-ID albedo array layer (-1 = tint only).
    albedo_layers: [i32; MATERIAL_SLOT_COUNT],
    pub depth: wgpu::TextureView,
    /// Unit UV grid spanning the full world (coarse fallback).
    pub grid: TerrainGrid,
    /// Camera-centered nested LOD rings (fine → coarse draw order).
    pub ring_grids: Vec<TerrainGrid>,
    /// Per-ring uniform buffers so mid-pass `write_buffer` is not required.
    ring_uniform_bufs: Vec<wgpu::Buffer>,
    /// Bind groups mirroring [`Self::bind_group`] but pointing at ring uniforms.
    ring_bind_groups: Vec<wgpu::BindGroup>,
    pub clipmap: ClipmapConfig,
    /// Backward-compatible alias for [`Self::clipmap`].fallback.
    pub world_grid: WorldGridConfig,
    pub heights: HeightGpu,
    pub camera: OrbitCamera,
    traversal_mode: TerrainTraversalMode,
    infinite_presentation: Option<InfinitePresentationConfig>,
    pub size: winit::dpi::PhysicalSize<u32>,
    /// Last CPU→GPU height upload microseconds.
    pub last_upload_us: u64,
    /// Last resolved GPU pass timings (0 when TIMESTAMP_QUERY unavailable).
    pub last_gpu_timings: GpuTimings,
    pending_presentation_trace: GpuPresentationTraceContext,
    presentation_baseline: Option<PresentedTerrainBaseline>,
    last_presentation_record: Option<TerrainPresentationRecord>,
    presentation_slot_epoch: u64,
    integrity_probe: Option<integrity_probe::TerrainIntegrityProbe>,
    /// Terrain mesh resolution drawn last frame (profiler).
    pub last_grid_resolution: u32,
    /// After first height present, leave orbit target alone so uploads don't fight the user.
    camera_framed: bool,
    /// Changes whenever the sampled height view is replaced, allowing app-owned
    /// editor overlays to refresh their bind groups without reaching into internals.
    height_binding_revision: u64,
    /// Phase J dual-height overhang / cave roof proxy (opt-in layers).
    overhang: OptionalResource<OverhangOverlay>,
    /// Instanced vegetation driven by the evaluated vegetation-density field.
    vegetation: OptionalResource<VegetationOverlay>,
    /// Presentation lighting (does not affect height data).
    pub lighting: EnvironmentLighting,
    /// Active ocean height; None disables the water surface.
    ocean_level: Option<f32>,
    /// Strength of painted biome placement colour overlay (0 = off).
    biome_tint_strength: f32,
    /// Display aids / analysis shading from the viewport chrome.
    display_aids: ViewportDisplayAids,
    /// Progressive stochastic lighting, temporal reprojection, and denoising.
    progressive: OptionalResource<ProgressivePresentationBundle>,
    /// Scene generation counters and invalidation tracking.
    scene_versions: SceneVersionRegistry,
    /// Adaptive quality / resolution budgeting.
    quality: ViewportQualityManager,
    /// Per-tile adaptive sampling state (Phase 11).
    adaptive: AdaptiveSamplingState,
    /// Monotonic frame counter — never reset on invalidation.
    global_frame_index: u64,
    /// Last internal render scale — detects resolution invalidation.
    last_internal_scale: f32,
    /// Editor interaction state for quality budgeting (set via [`Self::set_interaction_state`]).
    last_interaction_state: EditorRefinementState,
    /// Debug visualization mode (0 = final composite).
    debug_viz_mode: u32,
    /// Optional GPU timestamp queries.
    gpu_timer: Option<gpu_timing::GpuTimestampTimer>,
    /// Planned pass graph for the current frame.
    frame_graph: FrameGraph,
    /// Directional shadow map (stable Fast Lit shadows).
    shadow_map: shadows::ShadowMap,
    /// CPU→GPU staging ring for large height uploads.
    staging: staging::StagingRing,
    /// Keep dummy tile atlas texture alive for bind group 15 when streaming is off.
    #[allow(dead_code)]
    tile_atlas_texture: wgpu::Texture,
    tile_atlas_view: wgpu::TextureView,
    page_table_buf: wgpu::Buffer,
    virtual_page_table_buf: wgpu::Buffer,
    tile_level_table_buf: wgpu::Buffer,
    use_tile_stream: bool,
    tile_stream_tile_size: f32,
    tile_stream_halo: f32,
    tile_stream_max_pages: f32,
    tile_stream_level_count: u32,
    tile_stream_target_level: u8,
    tile_stream_target_resolution: u32,
    tile_stream_content: terra_core::TerrainContentStamp,
    tile_stream_transition_frames: u32,
    tile_stream_terminal_fallback: TerrainTerminalFallback,
    tile_stream_debug_mode: u32,
    tile_stream_virtual_page_count: u32,
    tile_stream_infinite_topology: Option<terra_core::InfiniteTopologyConfig>,
}

/// Environment lighting used for Lit viewport presentation.
#[derive(Debug, Clone, Copy)]
pub struct EnvironmentLighting {
    pub light_dir: [f32; 4],
    pub exposure: f32,
    pub clear: [f32; 3],
    /// Raster fill-light multiplier (1.0 = current look).
    pub ambient_strength: f32,
    /// Raster cast-shadow darkness in [0, 1]; 0 disables the shadow pass.
    pub shadow_strength: f32,
    /// Raster aerial-perspective fog multiplier (1.0 = current look).
    pub fog_strength: f32,
}

impl Default for EnvironmentLighting {
    fn default() -> Self {
        Self {
            light_dir: [-0.35, -0.90, -0.20, 1.00],
            exposure: 1.00,
            clear: [0.28, 0.32, 0.38],
            ambient_strength: 1.0,
            shadow_strength: 0.0,
            fog_strength: 1.0,
        }
    }
}

/// Owned GPU handles shared across the whole app.
///
/// `wgpu::Device`/`Queue` are internally ref-counted (`Clone`), so cloning a
/// `GpuContext` shares one device rather than creating another. The app hands
/// clones to the tile atlas, terrain engine, and GUI so every consumer draws
/// on the same device the renderer uses.
#[derive(Clone)]
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// Color format the renderer's targets are built for — negotiated from the
    /// window surface at runtime, or chosen directly for headless/offscreen use.
    pub surface_format: wgpu::TextureFormat,
    pipelines: std::sync::Arc<terra_gpu::PipelineCacheRegistry>,
    adapter_metadata: std::sync::Arc<AdapterMetadata>,
}

/// Stable adapter facts included in startup logs and benchmark reports.
#[derive(Debug, Clone, Default)]
pub struct AdapterMetadata {
    pub name: String,
    pub vendor: u32,
    pub device: u32,
    pub device_type: String,
    pub backend: String,
    pub driver: String,
    pub driver_info: String,
    pub downlevel_shader_model: String,
    pub pipeline_cache_supported: bool,
    pub pipeline_cache_enabled: bool,
}

impl GpuContext {
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        Self::with_adapter_metadata(device, queue, surface_format, AdapterMetadata::default())
    }

    fn with_adapter_metadata(
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        adapter_metadata: AdapterMetadata,
    ) -> Self {
        let pipelines = std::sync::Arc::new(terra_gpu::PipelineCacheRegistry::new(&device));
        Self {
            device,
            queue,
            surface_format,
            pipelines,
            adapter_metadata: std::sync::Arc::new(adapter_metadata),
        }
    }

    pub fn pipeline_registry(&self) -> &terra_gpu::PipelineCacheRegistry {
        &self.pipelines
    }

    pub fn adapter_metadata(&self) -> &AdapterMetadata {
        &self.adapter_metadata
    }
}

/// The window surface plus its negotiated configuration, produced alongside a
/// [`GpuContext`] by [`init_gpu`] and consumed by [`TerrainRenderer::new`].
///
/// Fields are crate-private: the surface can be handed *to* the renderer, never
/// sourced back *through* it.
pub struct SurfaceTarget {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
}

/// Bring up the instance → surface → adapter → device chain for `window`.
///
/// This is the app's single runtime GPU initialization and owns the only
/// non-test `request_device` in the workspace. The returned [`GpuContext`] is
/// the app's GPU handle; the [`SurfaceTarget`] is passed straight into
/// [`TerrainRenderer::new`].
pub async fn init_gpu(
    window: std::sync::Arc<Window>,
) -> Result<(GpuContext, SurfaceTarget), RenderError> {
    let size = window.inner_size();
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        // Windows: DX12 avoids OBS/Overwolf/Medal Vulkan implicit layers that
        // STATUS_STACK_OVERFLOW in vkCreateDevice (esp. debug + large shaders).
        backends: preferred_backends(),
        ..Default::default()
    });
    let surface = instance
        .create_surface(window.clone())
        .map_err(|e| RenderError::Msg(e.to_string()))?;
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .ok_or_else(|| RenderError::Msg("no adapter".into()))?;
    let adapter_info = adapter.get_info();
    let downlevel = adapter.get_downlevel_capabilities();
    let pipeline_cache_supported = adapter.features().contains(wgpu::Features::PIPELINE_CACHE);

    let mut limits = wgpu::Limits::default();
    let adapter_limits = adapter.limits();
    // Path tracer uses 4 storage textures; request headroom when the adapter allows it.
    limits.max_storage_textures_per_shader_stage = adapter_limits
        .max_storage_textures_per_shader_stage
        .clamp(4, 16);
    limits.max_storage_buffers_per_shader_stage = adapter_limits
        .max_storage_buffers_per_shader_stage
        .max(limits.max_storage_buffers_per_shader_stage);
    limits.max_compute_workgroup_storage_size = adapter_limits
        .max_compute_workgroup_storage_size
        .max(limits.max_compute_workgroup_storage_size);
    limits.max_compute_invocations_per_workgroup = adapter_limits
        .max_compute_invocations_per_workgroup
        .max(limits.max_compute_invocations_per_workgroup);
    limits.max_compute_workgroups_per_dimension = adapter_limits
        .max_compute_workgroups_per_dimension
        .max(limits.max_compute_workgroups_per_dimension);
    limits.max_buffer_size = adapter_limits.max_buffer_size.max(limits.max_buffer_size);
    limits.max_texture_dimension_2d = adapter_limits
        .max_texture_dimension_2d
        .max(limits.max_texture_dimension_2d);

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("terra-render"),
                required_features: gpu_timing::requested_timestamp_features(&adapter)
                    | (adapter.features() & wgpu::Features::PIPELINE_CACHE),
                required_limits: limits,
                memory_hints: Default::default(),
            },
            None,
        )
        .await
        .map_err(|e| RenderError::Msg(e.to_string()))?;

    let caps = surface.get_capabilities(&adapter);
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|f| {
            matches!(
                f,
                wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm
            )
        })
        .or_else(|| caps.formats.iter().copied().find(|f| !f.is_srgb()))
        .unwrap_or(caps.formats[0]);

    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);
    let adapter_metadata = AdapterMetadata {
        name: adapter_info.name,
        vendor: adapter_info.vendor,
        device: adapter_info.device,
        device_type: format!("{:?}", adapter_info.device_type),
        backend: format!("{:?}", adapter_info.backend),
        driver: adapter_info.driver,
        driver_info: adapter_info.driver_info,
        downlevel_shader_model: format!("{:?}", downlevel.shader_model),
        pipeline_cache_supported,
        pipeline_cache_enabled: device.features().contains(wgpu::Features::PIPELINE_CACHE),
    };
    Ok((
        GpuContext::with_adapter_metadata(device, queue, format, adapter_metadata),
        SurfaceTarget {
            surface,
            config,
            size,
        },
    ))
}

impl SurfaceTarget {
    /// Hand the surface to the main thread as a [`PendingSurface`] so the
    /// renderer's pipelines can be built off-thread (see [`TerrainRenderer::new_detached`])
    /// while the main thread keeps presenting splash frames. The surface must
    /// stay on the thread that presents; only `config`/`size` cross to the worker.
    pub fn into_pending(self) -> PendingSurface {
        PendingSurface {
            surface: self.surface,
            config: self.config,
            size: self.size,
        }
    }
}

/// The window surface held on the main thread during startup, decoupled from the
/// renderer whose pipelines are compiling on a worker.
///
/// This is the seam that keeps the window responsive while shaders build: the
/// surface (which must be presented from the thread that owns the window) stays
/// here so the app can animate a splash via [`Self::present_splash`], while the
/// heavy [`TerrainRenderer::new_detached`] runs elsewhere. When that returns,
/// [`Self::attach`] hands the surface to the finished renderer.
pub struct PendingSurface {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
}

impl PendingSurface {
    /// Surface configuration the detached renderer must be built for (format,
    /// size, present/alpha modes). Clone it to hand to the worker.
    pub fn config(&self) -> &wgpu::SurfaceConfiguration {
        &self.config
    }

    /// Physical size the surface was configured at.
    pub fn size(&self) -> winit::dpi::PhysicalSize<u32> {
        self.size
    }

    /// Present one splash frame: clear the swapchain to `clear`, let `overlay`
    /// draw over the acquired view, then present — without consuming the surface.
    ///
    /// Cosmetic, so a failed surface acquire (e.g. transient `Outdated` during a
    /// resize) is logged and skipped, never fatal.
    pub fn present_splash(
        &self,
        ctx: &GpuContext,
        clear: [f32; 3],
        overlay: impl FnOnce(&wgpu::TextureView),
    ) {
        let frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(err) => {
                log::warn!("splash: surface acquire failed, skipping: {err}");
                return;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("splash-clear"),
            });
        {
            // Clear pass: the GUI renderer draws with LoadOp::Load, so the
            // attachment must be defined before it runs (freshly acquired
            // swapchain contents are otherwise undefined).
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("splash-clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: clear[0] as f64,
                            g: clear[1] as f64,
                            b: clear[2] as f64,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));
        overlay(&view);
        frame.present();
    }

    /// Attach this surface to a renderer that was built detached (surface `None`)
    /// on a worker thread, finishing it into a windowed renderer. Reconfigures
    /// the surface against the renderer's device to be safe.
    pub fn attach(self, renderer: &mut TerrainRenderer) {
        self.surface.configure(&renderer.device, &self.config);
        renderer.config = self.config;
        renderer.size = self.size;
        renderer.surface = Some(self.surface);
    }
}

impl TerrainRenderer {
    /// Build every device-only pipeline and render target for a *windowed*
    /// renderer, but without the surface — so this (the expensive shader/pipeline
    /// compile) can run on a worker thread while the main thread animates the
    /// startup splash. Pass `config`/`size` cloned from the [`PendingSurface`];
    /// finish on the main thread with [`PendingSurface::attach`].
    pub fn new_detached(
        ctx: &GpuContext,
        config: wgpu::SurfaceConfiguration,
        size: winit::dpi::PhysicalSize<u32>,
    ) -> Self {
        Self::init(
            ctx.device.clone(),
            ctx.queue.clone(),
            ctx.pipelines.clone(),
            None,
            config,
            size,
        )
    }
}

impl TerrainRenderer {
    /// Build the windowed renderer from the app-owned [`GpuContext`] and the
    /// [`SurfaceTarget`] that [`init_gpu`] produced together. The device and
    /// queue handles are cloned (cheap refcount bumps, shared not duplicated);
    /// the surface is consumed.
    pub fn new(ctx: &GpuContext, target: SurfaceTarget) -> Self {
        Self::init(
            ctx.device.clone(),
            ctx.queue.clone(),
            ctx.pipelines.clone(),
            Some(target.surface),
            target.config,
            target.size,
        )
    }

    /// Current swapchain dimensions in physical pixels (each always ≥ 1).
    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    pub fn set_presentation_trace_context(&mut self, context: GpuPresentationTraceContext) {
        self.pending_presentation_trace = context;
    }

    /// Swapchain color format the surface was configured with.
    pub fn format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// Capture the immutable GPU/layout capability needed to build another
    /// terrain variant on a worker thread.
    pub fn terrain_pipeline_compiler(&self) -> TerrainPipelineCompiler {
        TerrainPipelineCompiler::new(
            self.device.clone(),
            self.pipelines.clone(),
            self.bind_group_layout.clone(),
            self.config.format,
            std::sync::Arc::clone(&self.pipeline_family),
        )
    }

    pub fn presentation_pipeline_compiler(&self) -> PresentationPipelineCompiler {
        PresentationPipelineCompiler::new(
            self.device.clone(),
            self.queue.clone(),
            self.pipelines.clone(),
            self.terrain_pipeline_compiler(),
            self.config.format,
            self.config.width,
            self.config.height,
            self.quality.internal_scale,
        )
    }

    pub fn has_pipeline_variant(&self, variant: TerrainShaderVariant) -> bool {
        match variant {
            TerrainShaderVariant::Bounded => true,
            TerrainShaderVariant::Infinite => self.infinite_pipeline_bundle.is_ready(),
        }
    }

    /// Install a fully compiled bundle without changing the active presentation.
    pub fn install_pipeline_bundle(&mut self, bundle: TerrainPipelineBundle) -> Result<(), String> {
        if bundle.format != self.config.format {
            return Err(format!(
                "terrain pipeline format mismatch: bundle={:?}, renderer={:?}",
                bundle.format, self.config.format
            ));
        }
        if !std::sync::Arc::ptr_eq(&bundle.family, &self.pipeline_family) {
            return Err("terrain pipeline bundle belongs to another renderer".into());
        }
        match bundle.variant {
            TerrainShaderVariant::Bounded => self.bounded_pipeline_bundle = bundle,
            TerrainShaderVariant::Infinite => self.infinite_pipeline_bundle.set_ready(bundle),
        }
        Ok(())
    }

    pub fn activate_pipeline_variant(
        &mut self,
        variant: TerrainShaderVariant,
    ) -> Result<(), String> {
        if !self.has_pipeline_variant(variant) {
            return Err(format!(
                "{} terrain pipeline is not installed",
                variant.name()
            ));
        }
        self.active_pipeline_variant = variant;
        Ok(())
    }

    pub const fn active_pipeline_variant(&self) -> TerrainShaderVariant {
        self.active_pipeline_variant
    }

    fn active_pipeline_bundle(&self) -> &TerrainPipelineBundle {
        match self.active_pipeline_variant {
            TerrainShaderVariant::Bounded => &self.bounded_pipeline_bundle,
            TerrainShaderVariant::Infinite => self
                .infinite_pipeline_bundle
                .ready()
                .expect("active Infinite terrain pipeline is installed"),
        }
    }

    fn ocean_slot(&self, variant: TerrainShaderVariant) -> &OptionalResource<OceanPipelineBundle> {
        match variant {
            TerrainShaderVariant::Bounded => &self.bounded_ocean_pipeline,
            TerrainShaderVariant::Infinite => &self.infinite_ocean_pipeline,
        }
    }

    fn ocean_slot_mut(
        &mut self,
        variant: TerrainShaderVariant,
    ) -> &mut OptionalResource<OceanPipelineBundle> {
        match variant {
            TerrainShaderVariant::Bounded => &mut self.bounded_ocean_pipeline,
            TerrainShaderVariant::Infinite => &mut self.infinite_ocean_pipeline,
        }
    }

    fn wireframe_slot(
        &self,
        variant: TerrainShaderVariant,
    ) -> &OptionalResource<WireframePipelineBundle> {
        match variant {
            TerrainShaderVariant::Bounded => &self.bounded_wireframe_pipeline,
            TerrainShaderVariant::Infinite => &self.infinite_wireframe_pipeline,
        }
    }

    fn wireframe_slot_mut(
        &mut self,
        variant: TerrainShaderVariant,
    ) -> &mut OptionalResource<WireframePipelineBundle> {
        match variant {
            TerrainShaderVariant::Bounded => &mut self.bounded_wireframe_pipeline,
            TerrainShaderVariant::Infinite => &mut self.infinite_wireframe_pipeline,
        }
    }

    pub fn optional_pipeline_state(
        &self,
        feature: PresentationPipelineFeature,
    ) -> &OptionalResourceState {
        match feature {
            PresentationPipelineFeature::InfiniteTerrain => self.infinite_pipeline_bundle.state(),
            PresentationPipelineFeature::Ocean(variant) => self.ocean_slot(variant).state(),
            PresentationPipelineFeature::Wireframe(variant) => self.wireframe_slot(variant).state(),
            PresentationPipelineFeature::Progressive => self.progressive.state(),
            PresentationPipelineFeature::Overhang => self.overhang.state(),
            PresentationPipelineFeature::Vegetation => self.vegetation.state(),
            PresentationPipelineFeature::Guides | PresentationPipelineFeature::Brush => {
                panic!("editor overlay state is application-owned")
            }
        }
    }

    pub fn begin_optional_pipeline_compile(
        &mut self,
        feature: PresentationPipelineFeature,
        request_id: u64,
    ) -> bool {
        match feature {
            PresentationPipelineFeature::InfiniteTerrain => {
                self.infinite_pipeline_bundle.begin(request_id)
            }
            PresentationPipelineFeature::Ocean(variant) => {
                self.ocean_slot_mut(variant).begin(request_id)
            }
            PresentationPipelineFeature::Wireframe(variant) => {
                self.wireframe_slot_mut(variant).begin(request_id)
            }
            PresentationPipelineFeature::Progressive => self.progressive.begin(request_id),
            PresentationPipelineFeature::Overhang => self.overhang.begin(request_id),
            PresentationPipelineFeature::Vegetation => self.vegetation.begin(request_id),
            PresentationPipelineFeature::Guides | PresentationPipelineFeature::Brush => false,
        }
    }

    pub fn fail_optional_pipeline_compile(
        &mut self,
        feature: PresentationPipelineFeature,
        request_id: u64,
        message: String,
    ) -> bool {
        match feature {
            PresentationPipelineFeature::InfiniteTerrain => {
                self.infinite_pipeline_bundle.fail(request_id, message)
            }
            PresentationPipelineFeature::Ocean(variant) => {
                self.ocean_slot_mut(variant).fail(request_id, message)
            }
            PresentationPipelineFeature::Wireframe(variant) => {
                self.wireframe_slot_mut(variant).fail(request_id, message)
            }
            PresentationPipelineFeature::Progressive => self.progressive.fail(request_id, message),
            PresentationPipelineFeature::Overhang => self.overhang.fail(request_id, message),
            PresentationPipelineFeature::Vegetation => self.vegetation.fail(request_id, message),
            PresentationPipelineFeature::Guides | PresentationPipelineFeature::Brush => false,
        }
    }

    pub fn install_optional_pipeline_bundle(
        &mut self,
        request_id: u64,
        bundle: PresentationPipelineBundle,
    ) -> Result<(), String> {
        match bundle {
            PresentationPipelineBundle::Terrain(bundle) => {
                if bundle.format != self.config.format
                    || !std::sync::Arc::ptr_eq(&bundle.family, &self.pipeline_family)
                    || bundle.variant != TerrainShaderVariant::Infinite
                {
                    return Err("incompatible Infinite terrain pipeline bundle".into());
                }
                self.infinite_pipeline_bundle
                    .install(request_id, bundle)
                    .map_err(|_| "stale Infinite terrain pipeline bundle".to_string())
            }
            PresentationPipelineBundle::Ocean(bundle) => {
                if bundle.format != self.config.format
                    || !std::sync::Arc::ptr_eq(&bundle.family, &self.pipeline_family)
                {
                    return Err("incompatible ocean pipeline bundle".into());
                }
                self.ocean_slot_mut(bundle.variant)
                    .install(request_id, bundle)
                    .map_err(|_| "stale ocean pipeline bundle".to_string())
            }
            PresentationPipelineBundle::Wireframe(bundle) => {
                if bundle.format != self.config.format
                    || !std::sync::Arc::ptr_eq(&bundle.family, &self.pipeline_family)
                {
                    return Err("incompatible wireframe pipeline bundle".into());
                }
                self.wireframe_slot_mut(bundle.variant)
                    .install(request_id, bundle)
                    .map_err(|_| "stale wireframe pipeline bundle".to_string())
            }
            PresentationPipelineBundle::Progressive(mut bundle) => {
                bundle.progressive.resize(
                    &self.device,
                    &self.pipelines,
                    self.config.width,
                    self.config.height,
                );
                bundle.path_tracer.resize(
                    &self.device,
                    &self.queue,
                    &self.pipelines,
                    self.config.width,
                    self.config.height,
                    self.quality.internal_scale,
                );
                let mask = self.adaptive.prepare_all_active_mask();
                bundle.path_tracer.upload_sample_mask(&self.queue, &mask);
                self.progressive
                    .install(request_id, bundle)
                    .map_err(|_| "stale progressive pipeline bundle".to_string())
            }
            PresentationPipelineBundle::Overhang(bundle) => self
                .overhang
                .install(request_id, bundle)
                .map_err(|_| "stale overhang pipeline bundle".to_string()),
            PresentationPipelineBundle::Vegetation(bundle) => self
                .vegetation
                .install(request_id, bundle)
                .map_err(|_| "stale vegetation pipeline bundle".to_string()),
            PresentationPipelineBundle::Guides(_) | PresentationPipelineBundle::Brush(_) => {
                Err("editor overlay bundle cannot be installed in TerrainRenderer".into())
            }
        }
    }

    pub fn cancel_optional_pipeline_compiles(&mut self) {
        self.infinite_pipeline_bundle.reset_unready();
        self.bounded_ocean_pipeline.reset_unready();
        self.infinite_ocean_pipeline.reset_unready();
        self.bounded_wireframe_pipeline.reset_unready();
        self.infinite_wireframe_pipeline.reset_unready();
        self.progressive.reset_unready();
        self.overhang.reset_unready();
        self.vegetation.reset_unready();
    }

    pub fn invalidate_optional_pipelines(&mut self) {
        self.infinite_pipeline_bundle.reset();
        self.bounded_ocean_pipeline.reset();
        self.infinite_ocean_pipeline.reset();
        self.bounded_wireframe_pipeline.reset();
        self.infinite_wireframe_pipeline.reset();
        self.progressive.reset();
        self.overhang.reset();
        self.vegetation.reset();
        self.active_pipeline_variant = TerrainShaderVariant::Bounded;
    }

    /// Construct a renderer with no window or surface, for offscreen rendering.
    ///
    /// Takes the same app-owned [`GpuContext`] as [`Self::new`] — the device and
    /// queue are cloned (cheap refcount bumps), and the render targets are built
    /// for `ctx.surface_format`. Draw with [`Self::render_to_view`] into
    /// caller-supplied views of that format at exactly `width`×`height`;
    /// [`Self::render_terrain`] returns an error because there is no swapchain to
    /// acquire from. [`Self::resize`] still reallocates the offscreen targets
    /// (surface reconfiguration is skipped).
    ///
    /// Works on a default-limits, feature-less device: the path tracer needs four
    /// storage textures per stage, which `wgpu::Limits::default()` provides, and
    /// without `TIMESTAMP_QUERY` the GPU timer is simply absent.
    pub fn new_headless(ctx: &GpuContext, width: u32, height: u32) -> Self {
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: ctx.surface_format,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        let size = winit::dpi::PhysicalSize::new(config.width, config.height);
        Self::init(
            ctx.device.clone(),
            ctx.queue.clone(),
            ctx.pipelines.clone(),
            None,
            config,
            size,
        )
    }

    /// Shared constructor tail: every device-only resource, after surface and
    /// adapter negotiation. `config` doubles as the render-target description when
    /// `surface` is `None` — `format` bakes the color-target pipelines and
    /// `width`/`height` size the depth/progressive/path-tracer targets; the
    /// present-mode and alpha-mode fields are inert without a surface.
    fn init(
        device: wgpu::Device,
        queue: wgpu::Queue,
        pipelines: std::sync::Arc<terra_gpu::PipelineCacheRegistry>,
        surface: Option<wgpu::Surface<'static>>,
        config: wgpu::SurfaceConfiguration,
        size: winit::dpi::PhysicalSize<u32>,
    ) -> Self {
        let format = config.format;

        log::info!("terra-render: compiling bounded terrain shader/pipelines…");
        let terrain_source = terrain_shader::compose(TerrainShaderVariant::Bounded);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(TerrainShaderVariant::Bounded.shader_label()),
            source: wgpu::ShaderSource::Wgsl(terrain_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    // Vertex displace + fragment height-AO both sample the height field.
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Climate snow / temperature / rainfall (R32Float, non-filterable).
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Material albedo texture array (RGBA8) + filtering sampler.
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Painted biome placement colour overlay (RGBA8).
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Streamed tile atlas (R32Float array) + physical page table.
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 16,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Directional shadow map + comparison sampler.
                wgpu::BindGroupLayoutEntry {
                    binding: 17,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 18,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 19,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 20,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let heights = HeightGpu::new(&device, &pipelines, 256);
        let integrity_probe = integrity_probe::TerrainIntegrityProbe::try_new(&device, &pipelines);
        debug_assert!(std::mem::size_of::<FrameUniforms>() as u64 <= FRAME_UNIFORM_BUF_SIZE);
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame-u"),
            size: FRAME_UNIFORM_BUF_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let material_palette_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("material-palette-u"),
            size: std::mem::size_of::<MaterialPalette>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let albedo_layers = [-1i32; MATERIAL_SLOT_COUNT];
        queue.write_buffer(
            &material_palette_buf,
            0,
            bytemuck::bytes_of(&MaterialPalette::from_params(None, &albedo_layers)),
        );
        let (albedo_array, albedo_array_view, albedo_sampler) =
            create_albedo_array(&device, &queue);

        let (
            tile_atlas_texture,
            tile_atlas_view,
            page_table_buf,
            virtual_page_table_buf,
            tile_level_table_buf,
        ) = create_dummy_tile_stream(&device);
        let shadow_map =
            shadows::ShadowMap::new(&device, &pipelines, heights.display_height_view(), true);
        let staging = staging::StagingRing::new(&device, 3, 4 * 1024 * 1024);
        let gpu_timer = gpu_timing::GpuTimestampTimer::try_new(&device, &queue);

        let bind_group = Self::make_bind_group(
            &device,
            &bind_group_layout,
            &uniform_buf,
            &material_palette_buf,
            &heights,
            &albedo_array_view,
            &albedo_sampler,
            &tile_atlas_view,
            &page_table_buf,
            &virtual_page_table_buf,
            &tile_level_table_buf,
            &shadow_map.view,
            &shadow_map.comparison_sampler,
        );

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = pipelines.render_pipeline(
            TerrainShaderVariant::Bounded.terrain_pipeline_label(),
            format,
            || {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(TerrainShaderVariant::Bounded.terrain_pipeline_label()),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        buffers: &[TerrainGrid::vertex_layout()],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        cull_mode: Some(wgpu::Face::Back),
                        ..Default::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Less,
                        stencil: Default::default(),
                        bias: Default::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: pipelines.driver_cache(),
                })
            },
        );

        let pipeline_family = std::sync::Arc::new(());
        let bounded_pipeline_bundle = TerrainPipelineBundle {
            terrain: pipeline,
            variant: TerrainShaderVariant::Bounded,
            format,
            family: std::sync::Arc::clone(&pipeline_family),
        };
        let depth = create_depth(&device, config.width, config.height);
        let clipmap = ClipmapConfig::for_world(4096.0, 1025);
        let world_grid = clipmap.fallback.clone();
        let grid = TerrainGrid::new(&device, world_grid.grid_size);
        let ring_grids: Vec<TerrainGrid> = clipmap
            .rings
            .iter()
            .map(|ring| TerrainGrid::new(&device, ring.grid_size))
            .collect();
        let ring_uniform_bufs: Vec<wgpu::Buffer> = (0..ring_grids.len())
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("frame-u-ring-{i}")),
                    size: FRAME_UNIFORM_BUF_SIZE,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let ring_bind_groups: Vec<wgpu::BindGroup> = ring_uniform_bufs
            .iter()
            .map(|ring_u| {
                Self::make_bind_group(
                    &device,
                    &bind_group_layout,
                    ring_u,
                    &material_palette_buf,
                    &heights,
                    &albedo_array_view,
                    &albedo_sampler,
                    &tile_atlas_view,
                    &page_table_buf,
                    &virtual_page_table_buf,
                    &tile_level_table_buf,
                    &shadow_map.view,
                    &shadow_map.comparison_sampler,
                )
            })
            .collect();
        let camera = OrbitCamera::default();
        let quality = ViewportQualityManager::default();
        let initial_internal_scale = quality.internal_scale;
        let adaptive = AdaptiveSamplingState::new(config.width, config.height);

        Self {
            surface,
            device,
            queue,
            pipelines,
            config,
            bounded_pipeline_bundle,
            infinite_pipeline_bundle: OptionalResource::default(),
            bounded_ocean_pipeline: OptionalResource::default(),
            infinite_ocean_pipeline: OptionalResource::default(),
            bounded_wireframe_pipeline: OptionalResource::default(),
            infinite_wireframe_pipeline: OptionalResource::default(),
            active_pipeline_variant: TerrainShaderVariant::Bounded,
            pipeline_family,
            uniform_buf,
            bind_group_layout,
            bind_group,
            material_palette_buf,
            albedo_array,
            albedo_array_view,
            albedo_sampler,
            albedo_layers,
            depth,
            grid,
            ring_grids,
            ring_uniform_bufs,
            ring_bind_groups,
            clipmap,
            world_grid,
            heights,
            camera,
            traversal_mode: TerrainTraversalMode::Bounded,
            infinite_presentation: None,
            size,
            last_upload_us: 0,
            last_gpu_timings: GpuTimings::default(),
            pending_presentation_trace: GpuPresentationTraceContext::default(),
            presentation_baseline: None,
            last_presentation_record: None,
            presentation_slot_epoch: 0,
            integrity_probe,
            last_grid_resolution: 0,
            camera_framed: false,
            height_binding_revision: 1,
            overhang: OptionalResource::default(),
            vegetation: OptionalResource::default(),
            lighting: EnvironmentLighting::default(),
            ocean_level: None,
            biome_tint_strength: 0.0,
            display_aids: ViewportDisplayAids::default(),
            progressive: OptionalResource::default(),
            scene_versions: SceneVersionRegistry::default(),
            quality,
            adaptive,
            global_frame_index: 0,
            last_internal_scale: initial_internal_scale,
            last_interaction_state: EditorRefinementState::Interactive,
            debug_viz_mode: 0,
            gpu_timer,
            frame_graph: FrameGraph::default(),
            shadow_map,
            staging,
            tile_atlas_texture,
            tile_atlas_view,
            page_table_buf,
            virtual_page_table_buf,
            tile_level_table_buf,
            use_tile_stream: false,
            tile_stream_tile_size: 256.0,
            tile_stream_halo: 2.0,
            tile_stream_max_pages: 1.0,
            tile_stream_level_count: 0,
            tile_stream_target_level: 0,
            tile_stream_target_resolution: 1,
            tile_stream_content: terra_core::TerrainContentStamp::default(),
            tile_stream_transition_frames: 8,
            tile_stream_terminal_fallback: TerrainTerminalFallback::RootRequired,
            tile_stream_debug_mode: 0,
            tile_stream_virtual_page_count: 0,
            tile_stream_infinite_topology: None,
        }
    }

    // (pipeline compile complete — logged via terrain shader message above)

    // Assembles one wgpu bind group from eleven distinct GPU handles (buffers,
    // views, samplers), each bound once by position to build the descriptor — a
    // params struct would only relocate the same list. Kept flat.
    #[allow(clippy::too_many_arguments)]
    fn make_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        uniform_buf: &wgpu::Buffer,
        material_palette_buf: &wgpu::Buffer,
        heights: &HeightGpu,
        albedo_array_view: &wgpu::TextureView,
        albedo_sampler: &wgpu::Sampler,
        tile_atlas_view: &wgpu::TextureView,
        page_table_buf: &wgpu::Buffer,
        virtual_page_table_buf: &wgpu::Buffer,
        tile_level_table_buf: &wgpu::Buffer,
        shadow_view: &wgpu::TextureView,
        shadow_samp: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain-bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(heights.display_height_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(heights.display_normal_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(heights.sampler()),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(heights.materials_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(heights.wetness_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(heights.vegetation_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(heights.flow_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: material_palette_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(heights.snow_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::TextureView(heights.temperature_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(heights.rainfall_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::TextureView(albedo_array_view),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: wgpu::BindingResource::Sampler(albedo_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: wgpu::BindingResource::TextureView(heights.placement_tint_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: wgpu::BindingResource::TextureView(tile_atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: page_table_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: wgpu::BindingResource::TextureView(shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: wgpu::BindingResource::Sampler(shadow_samp),
                },
                wgpu::BindGroupEntry {
                    binding: 19,
                    resource: virtual_page_table_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 20,
                    resource: tile_level_table_buf.as_entire_binding(),
                },
            ],
        })
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        if let Some(surface) = &self.surface {
            surface.configure(&self.device, &self.config);
        }
        let depth = create_depth(&self.device, self.config.width, self.config.height);
        self.depth = depth;
        self.adaptive.resize(self.config.width, self.config.height);
        let mask = self.adaptive.prepare_all_active_mask();
        if let Some(bundle) = self.progressive.ready_mut() {
            bundle.progressive.resize(
                &self.device,
                &self.pipelines,
                self.config.width,
                self.config.height,
            );
            bundle.path_tracer.resize(
                &self.device,
                &self.queue,
                &self.pipelines,
                self.config.width,
                self.config.height,
                self.quality.internal_scale,
            );
            bundle.path_tracer.upload_sample_mask(&self.queue, &mask);
        }
        self.notify_invalidation(InvalidationReason::ViewportResized);
    }

    /// Re-run `surface.configure` with the current config after the swap chain
    /// was lost or invalidated (`SurfaceError::Lost` / `Outdated`).
    ///
    /// Unlike `resize`, the config is unchanged, so the depth buffer,
    /// progressive history, and path-tracer targets are deliberately left
    /// intact — `render_to_view`'s size/format contract still holds and
    /// accumulation is preserved. Actual size changes route through `resize`
    /// via `WindowEvent::Resized`; if an `Outdated` was caused by a resize
    /// whose event has not arrived yet, reconfiguring at the current size is
    /// idempotent and self-heals once `Resized` lands.
    pub fn reconfigure(&mut self) {
        if let Some(surface) = &self.surface {
            surface.configure(&self.device, &self.config);
        }
    }

    pub fn scene_versions(&self) -> &SceneVersionRegistry {
        &self.scene_versions
    }

    pub fn scene_versions_mut(&mut self) -> &mut SceneVersionRegistry {
        &mut self.scene_versions
    }

    pub fn quality(&self) -> &ViewportQualityManager {
        &self.quality
    }

    pub fn quality_mut(&mut self) -> &mut ViewportQualityManager {
        &mut self.quality
    }

    pub fn set_interaction_state(&mut self, state: EditorRefinementState) {
        self.last_interaction_state = state;
    }

    pub fn interaction_state(&self) -> EditorRefinementState {
        self.last_interaction_state
    }

    pub fn set_debug_viz_mode(&mut self, mode: u32) {
        self.debug_viz_mode = mode;
    }

    pub fn debug_viz_mode(&self) -> u32 {
        self.debug_viz_mode
    }

    pub fn global_frame_index(&self) -> u64 {
        self.global_frame_index
    }

    pub fn progressive_accumulation_frame(&self) -> u32 {
        self.progressive
            .ready()
            .map_or(0, |bundle| bundle.progressive.accumulation_frame_index())
    }

    pub fn progressive_last_invalidation(&self) -> InvalidationReason {
        self.progressive
            .ready()
            .map_or(InvalidationReason::TerrainChanged, |bundle| {
                bundle.progressive.last_invalidation_reason()
            })
    }

    pub fn scene_versions_snapshot(&self) -> SceneVersions {
        self.scene_versions.versions
    }

    pub fn notify_invalidation(&mut self, reason: InvalidationReason) {
        self.scene_versions.notify(reason);
        if reason.resets_accumulation() {
            self.adaptive.reactivate_all();
            let mask = self.adaptive.prepare_all_active_mask();
            if let Some(bundle) = self.progressive.ready_mut() {
                bundle.progressive.invalidate_with_reason(reason);
                bundle.path_tracer.invalidate(&self.queue);
                bundle.path_tracer.upload_sample_mask(&self.queue, &mask);
            }
        }
    }

    /// Select presentation backend from mode (single map — no dual progressive flag).
    pub fn set_renderer_mode(&mut self, mode: ViewportRendererMode) {
        if self.quality.config.mode == mode {
            return;
        }
        self.quality.config.mode = mode;
        let backend = PresentationBackendId::from_mode(mode);
        // Progressive post stack is only armed for the PT backend.
        if let Some(bundle) = self.progressive.ready_mut() {
            bundle
                .progressive
                .set_enabled(matches!(backend, PresentationBackendId::ProgressivePt));
        }
        self.notify_invalidation(InvalidationReason::RenderModeChanged);
    }

    /// Active presentation backend for the current mode.
    pub fn presentation_backend(&self) -> PresentationBackendId {
        if self.traversal_mode == TerrainTraversalMode::Infinite
            || (self.quality.config.mode.uses_progressive_path_tracer()
                && !self.progressive.is_ready())
        {
            PresentationBackendId::RasterLit
        } else {
            PresentationBackendId::from_mode(self.quality.config.mode)
        }
    }

    /// Upload heightfield to GPU textures (no mesh rebuild). Swaps display buffer when done.
    pub fn upload_heightfield(&mut self, hf: &Heightfield) {
        self.upload_heightfield_regions(hf, None);
    }

    /// Upload only dirty sample regions (Wave D); `None` = full field.
    pub fn upload_heightfield_regions(&mut self, hf: &Heightfield, regions: Option<&[SampleRect]>) {
        profiling::scope!("upload_heightfield");
        let t0 = std::time::Instant::now();
        self.staging.begin_frame();
        self.heights.upload_regions_and_swap_with_staging(
            &self.device,
            &self.queue,
            Some(&mut self.staging),
            hf,
            regions,
        );
        self.finish_height_present(t0);
        // CPU ownership has no GPU output identity. Invalidate the typed GPU
        // baseline so the next regional GPU request promotes to a full copy.
        self.presentation_baseline = None;
        self.last_presentation_record = None;
    }

    /// Upload authored tint, roughness, metalness, and optional albedo PNGs.
    pub fn upload_material_palette(&mut self, params: Option<&MaterialsParams>) {
        self.albedo_layers = [-1i32; MATERIAL_SLOT_COUNT];
        if let Some(params) = params {
            for rule in &params.rules {
                let id = (rule.id as usize).min(MATERIAL_SLOT_COUNT - 1);
                if let Some(path) = rule.albedo_path.as_ref().filter(|p| !p.is_empty()) {
                    match load_albedo_png(path) {
                        Ok(rgba) => {
                            write_albedo_layer(&self.queue, &self.albedo_array, id as u32, &rgba);
                            self.albedo_layers[id] = id as i32;
                        }
                        Err(err) => {
                            log::warn!("albedo_path load failed ({path}): {err}");
                        }
                    }
                }
            }
        }
        let palette = MaterialPalette::from_params(params, &self.albedo_layers);
        self.queue
            .write_buffer(&self.material_palette_buf, 0, bytemuck::bytes_of(&palette));
        self.notify_invalidation(InvalidationReason::MaterialChanged);
    }

    /// Upload material-ID and wetness fields produced by the surface layers.
    /// Rebuilds terrain bind groups because the sampled views may have resized.
    pub fn upload_aux_maps(
        &mut self,
        materials: Option<&MaskField>,
        wetness: Option<&MaskField>,
        vegetation: Option<&MaskField>,
    ) {
        self.upload_aux_maps_ex(AuxMaps {
            materials,
            wetness,
            vegetation,
            ..Default::default()
        });
    }

    /// Upload materials/wetness/vegetation plus optional climate R32Float aux maps and flow.
    pub fn upload_aux_maps_ex(&mut self, aux: AuxMaps) {
        self.heights
            .upload_aux_maps_ex(&self.device, &self.queue, aux);
        self.recreate_bind_group();
        self.notify_invalidation(InvalidationReason::MaterialChanged);
    }

    /// Upload artist biome placement colour overlay (RGBA8). Rebuilds bind groups.
    pub fn upload_placement_tint(&mut self, width: u32, height: u32, rgba: &[u8]) {
        self.heights
            .upload_placement_tint(&self.device, &self.queue, width, height, rgba);
        self.recreate_bind_group();
        self.notify_invalidation(InvalidationReason::MaterialChanged);
    }

    /// Present a GPU-resident height texture (Wave C — no CPU readback).
    pub fn present_gpu_height(&mut self, src: &wgpu::Texture, geom: HeightPresentGeom) {
        self.present_gpu_height_region(src, geom, None);
    }

    pub fn present_gpu_height_region(
        &mut self,
        src: &wgpu::Texture,
        geom: HeightPresentGeom,
        region: Option<SampleRect>,
    ) {
        profiling::scope!("present_gpu_height");
        let t0 = std::time::Instant::now();
        self.heights.copy_from_texture_region_and_swap(
            &self.device,
            &self.queue,
            src,
            geom,
            region,
        );
        self.finish_height_present(t0);
    }

    pub fn present_gpu_height_region_traced(
        &mut self,
        src: &wgpu::Texture,
        geom: HeightPresentGeom,
        region: Option<SampleRect>,
        candidate: terra_gpu::output_identity::GpuTerrainOutputIdentity,
        expected: TerrainPresentationExpectations,
    ) -> TerrainPresentationRecord {
        let coherent_before = self.heights.local_slots_coherent();
        let full = SampleRect {
            x: 0,
            y: 0,
            w: geom.width,
            h: geom.height,
        };
        let requested_is_partial = region.is_some_and(|rect| rect != full);
        let actual_mode = if requested_is_partial {
            presentation_transition::recoverable_regional_presentation_mode(
                candidate,
                self.presentation_baseline,
                expected,
                coherent_before,
            )
        } else {
            TerrainPresentationMode::FullCopy
        };
        let actual_rect = match actual_mode {
            TerrainPresentationMode::RegionalCopy => region,
            TerrainPresentationMode::FullCopy => Some(full),
            TerrainPresentationMode::Shared | TerrainPresentationMode::CpuUpload => None,
        };
        // Passing the actual rect is essential when identity recovery promotes a
        // coherent-but-stale local baseline: HeightGpu cannot infer that lineage
        // gap from its texture-slot state alone.
        self.present_gpu_height_region(src, geom, actual_rect);
        let record = self.record_gpu_presentation(
            candidate,
            expected,
            GpuPresentationAttempt {
                requested_mode: if requested_is_partial {
                    TerrainPresentationMode::RegionalCopy
                } else {
                    TerrainPresentationMode::FullCopy
                },
                actual_mode,
                requested_rect: region,
                actual_rect,
                coherent_before,
            },
        );
        if self.integrity_probe.is_some() {
            let source_view = src.create_view(&wgpu::TextureViewDescriptor::default());
            self.submit_integrity_probe(&source_view, record);
        }
        record
    }

    /// Bind a GPU engine height texture directly when formats match (full field).
    /// Partial [`SampleRect`] updates still copy through the double-buffer path; the
    /// first partial after sharing promotes once to a full GPU copy to establish a
    /// renderer-local baseline, then later partials remain region-bounded.
    pub fn present_gpu_height_shared(
        &mut self,
        src: &wgpu::Texture,
        src_view: &wgpu::TextureView,
        geom: HeightPresentGeom,
        region: Option<SampleRect>,
    ) {
        if region.is_some() {
            self.present_gpu_height_region(src, geom, region);
            return;
        }
        profiling::scope!("present_gpu_height_shared");
        let t0 = std::time::Instant::now();
        self.heights
            .present_shared_height(&self.device, &self.queue, src_view, geom);
        self.finish_height_present(t0);
    }

    pub fn present_gpu_height_shared_traced(
        &mut self,
        src: &wgpu::Texture,
        src_view: &wgpu::TextureView,
        geom: HeightPresentGeom,
        region: Option<SampleRect>,
        candidate: terra_gpu::output_identity::GpuTerrainOutputIdentity,
        expected: TerrainPresentationExpectations,
    ) -> TerrainPresentationRecord {
        if region.is_some() {
            return self.present_gpu_height_region_traced(src, geom, region, candidate, expected);
        }
        let coherent_before = self.heights.local_slots_coherent();
        self.present_gpu_height_shared(src, src_view, geom, None);
        let record = self.record_gpu_presentation(
            candidate,
            expected,
            GpuPresentationAttempt {
                requested_mode: TerrainPresentationMode::Shared,
                actual_mode: TerrainPresentationMode::Shared,
                requested_rect: None,
                actual_rect: None,
                coherent_before,
            },
        );
        self.submit_integrity_probe(src_view, record);
        record
    }

    fn record_gpu_presentation(
        &mut self,
        candidate: terra_gpu::output_identity::GpuTerrainOutputIdentity,
        expected: TerrainPresentationExpectations,
        attempt: GpuPresentationAttempt,
    ) -> TerrainPresentationRecord {
        let baseline_before = self.presentation_baseline;
        let shadow_diagnostic = presentation_transition::validate_transition_shadow(
            candidate,
            baseline_before,
            expected,
            attempt.actual_mode,
        );
        self.presentation_slot_epoch = self.presentation_slot_epoch.saturating_add(1);
        let coherent_after = self.heights.local_slots_coherent();
        let last_full_generation = if matches!(
            attempt.actual_mode,
            TerrainPresentationMode::Shared | TerrainPresentationMode::FullCopy
        ) {
            Some(candidate.generation)
        } else {
            baseline_before.and_then(|baseline| baseline.last_full_generation)
        };
        let baseline_after = PresentedTerrainBaseline {
            identity: candidate,
            mode: attempt.actual_mode,
            complete: candidate.is_current_complete_final() && shadow_diagnostic.is_none(),
            local_slots_coherent: coherent_after,
            local_slot_epoch: self.presentation_slot_epoch,
            last_full_generation,
            height_lineage: candidate.output,
            normal_lineage: candidate.output,
        };
        let record = TerrainPresentationRecord {
            candidate,
            expectations: expected,
            baseline_before,
            baseline_after,
            requested_mode: attempt.requested_mode,
            actual_mode: attempt.actual_mode,
            requested_rect: attempt.requested_rect,
            actual_rect: attempt.actual_rect,
            local_slots_coherent_before: attempt.coherent_before,
            local_slots_coherent_after: coherent_after,
            decision: TerrainPresentationDecisionCode::Accepted,
            shadow_diagnostic,
        };
        self.presentation_baseline = Some(baseline_after);
        self.last_presentation_record = Some(record);
        record
    }

    pub const fn presented_terrain_baseline(&self) -> Option<PresentedTerrainBaseline> {
        self.presentation_baseline
    }

    pub const fn last_terrain_presentation_record(&self) -> Option<TerrainPresentationRecord> {
        self.last_presentation_record
    }

    fn submit_integrity_probe(
        &mut self,
        source: &wgpu::TextureView,
        record: TerrainPresentationRecord,
    ) {
        if let Some(probe) = self.integrity_probe.as_mut() {
            probe.submit(
                &self.device,
                &self.queue,
                source,
                record.candidate,
                record.actual_mode,
                record.actual_rect,
            );
        }
    }

    pub fn poll_integrity_probes(&mut self) -> Vec<TerrainIntegrityProbeResult> {
        self.integrity_probe
            .as_mut()
            .map_or_else(Vec::new, |probe| probe.poll(&self.device))
    }

    fn finish_height_present(&mut self, t0: std::time::Instant) {
        self.recreate_bind_group();
        let extent = self.heights.world_size.0.max(self.heights.world_size.1);
        if self.traversal_mode == TerrainTraversalMode::Bounded {
            // Match mesh density to the height texture as closely as device buffer
            // limits allow (256 MiB default). Infinite presentation is derived from
            // its topology and must not be replaced by this bounded-world heuristic.
            let tex = self
                .heights
                .tex_size
                .0
                .max(self.heights.tex_size.1)
                .max(256);
            let max_grid = TerrainGrid::max_resolution_for_device_limits();
            let target_grid = WorldGridConfig::for_world(tex.min(max_grid).max(513)).grid_size;
            let next = ClipmapConfig::for_world_with_height(extent, target_grid, tex);
            let rings_changed = self.clipmap.rings.len() != next.rings.len()
                || self
                    .clipmap
                    .rings
                    .iter()
                    .zip(next.rings.iter())
                    .any(|(a, b)| a.grid_size != b.grid_size)
                || self.clipmap.fallback.grid_size != next.fallback.grid_size;
            if rings_changed {
                self.clipmap = next;
                self.world_grid = self.clipmap.fallback.clone();
                if self.grid.resolution != self.world_grid.grid_size {
                    self.grid = TerrainGrid::new(&self.device, self.world_grid.grid_size);
                }
                self.ring_grids = self
                    .clipmap
                    .rings
                    .iter()
                    .map(|ring| TerrainGrid::new(&self.device, ring.grid_size))
                    .collect();
                self.ensure_ring_uniform_bufs();
                self.recreate_bind_group();
            } else {
                // Keep ring spacings in sync with extent without reallocating meshes.
                self.clipmap = next;
                self.world_grid = self.clipmap.fallback.clone();
            }
        }
        // Frame once (or after explicit reset). Continuous retargeting fights orbit/pan.
        if !self.camera_framed {
            if let Some(config) = self.infinite_presentation {
                self.frame_camera_to_infinite(
                    config.topology.origin.x_m(),
                    config.topology.origin.z_m(),
                    config.horizon_m.clamp(10.0, f64::from(f32::MAX)) as f32,
                );
            } else {
                self.frame_camera_to_terrain();
            }
        } else if self.camera.distance < 10.0 || self.camera.distance > extent * 4.0 {
            self.camera.distance = extent * 1.1;
        }
        self.last_upload_us = t0.elapsed().as_micros() as u64;
        self.notify_invalidation(InvalidationReason::TerrainChanged);
    }

    /// Center orbit on the current heightfield extents (camera-reset / first present).
    pub fn frame_camera_to_terrain(&mut self) {
        let (min_h, max_h) = self.heights.height_range;
        let extent = self.heights.world_size.0.max(self.heights.world_size.1);
        self.camera.target = glam::DVec3::new(
            f64::from(self.heights.world_size.0) * 0.5,
            f64::from((min_h + max_h) * 0.5),
            f64::from(self.heights.world_size.1) * 0.5,
        );
        self.camera.distance = extent * 1.1;
        self.camera_framed = true;
    }

    /// Frame a local Infinite-world preview around the fixed project origin.
    /// Large-coordinate camera-relative presentation is completed by the next
    /// slice; this keeps current traversal centred on the correct signed axes.
    pub fn frame_camera_to_infinite(
        &mut self,
        origin_x: f64,
        origin_z: f64,
        preview_radius_m: f32,
    ) {
        let (min_h, max_h) = self.heights.height_range;
        self.camera.target = glam::DVec3::new(origin_x, f64::from((min_h + max_h) * 0.5), origin_z);
        self.camera.distance = preview_radius_m.max(10.0) * 1.1;
        self.camera_framed = true;
    }

    pub fn request_camera_reframe(&mut self) {
        self.camera_framed = false;
    }

    /// Move the orbit target to a normalized location in the terrain footprint.
    pub fn focus_camera_uv(&mut self, u: f32, v: f32) {
        self.camera.target.x = f64::from(u.clamp(0.0, 1.0) * self.heights.world_size.0);
        self.camera.target.z = f64::from(v.clamp(0.0, 1.0) * self.heights.world_size.1);
        self.constrain_camera();
    }

    /// Switch to a near-vertical overview while retaining the current target.
    pub fn camera_top_view(&mut self) {
        self.camera.pitch = 1.45;
        let extent = self.heights.world_size.0.max(self.heights.world_size.1);
        self.camera.distance = self.camera.distance.max(extent * 0.9);
    }

    /// The terrain document has no spatial selection bounds yet, so frame its
    /// full footprint until selected layers expose one.
    pub fn frame_camera_to_selection(&mut self) {
        self.frame_camera_to_terrain();
    }

    fn recreate_bind_group(&mut self) {
        self.shadow_map
            .recreate_bind_group(&self.device, self.heights.display_height_view());
        self.height_binding_revision = self.height_binding_revision.wrapping_add(1).max(1);
        self.bind_group = Self::make_bind_group(
            &self.device,
            &self.bind_group_layout,
            &self.uniform_buf,
            &self.material_palette_buf,
            &self.heights,
            &self.albedo_array_view,
            &self.albedo_sampler,
            &self.tile_atlas_view,
            &self.page_table_buf,
            &self.virtual_page_table_buf,
            &self.tile_level_table_buf,
            &self.shadow_map.view,
            &self.shadow_map.comparison_sampler,
        );
        self.ensure_ring_uniform_bufs();
        self.ring_bind_groups = self
            .ring_uniform_bufs
            .iter()
            .map(|ring_u| {
                Self::make_bind_group(
                    &self.device,
                    &self.bind_group_layout,
                    ring_u,
                    &self.material_palette_buf,
                    &self.heights,
                    &self.albedo_array_view,
                    &self.albedo_sampler,
                    &self.tile_atlas_view,
                    &self.page_table_buf,
                    &self.virtual_page_table_buf,
                    &self.tile_level_table_buf,
                    &self.shadow_map.view,
                    &self.shadow_map.comparison_sampler,
                )
            })
            .collect();
    }

    fn ensure_ring_uniform_bufs(&mut self) {
        while self.ring_uniform_bufs.len() < self.ring_grids.len() {
            let i = self.ring_uniform_bufs.len();
            self.ring_uniform_bufs
                .push(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("frame-u-ring-{i}")),
                    size: FRAME_UNIFORM_BUF_SIZE,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
        }
        self.ring_uniform_bufs.truncate(self.ring_grids.len());
    }

    /// Bind the atlas plus its dense virtual directory and immutable hierarchy.
    /// Demand is intentionally absent: shader-visible page-table state selects data.
    pub fn set_tile_stream_resources(&mut self, resources: TerrainTileStreamResources) {
        self.tile_atlas_view = resources.atlas_view;
        self.page_table_buf = resources.physical_page_table;
        self.virtual_page_table_buf = resources.virtual_page_table;
        self.tile_level_table_buf = resources.level_table;
        self.tile_stream_tile_size = resources.tile_size.max(1) as f32;
        self.tile_stream_halo = resources.halo as f32;
        self.tile_stream_max_pages = resources.max_pages.max(1) as f32;
        self.tile_stream_level_count = resources.level_count;
        self.tile_stream_target_level = resources.target_level;
        self.tile_stream_target_resolution = resources.target_resolution.max(1);
        self.tile_stream_content = resources.content;
        self.tile_stream_transition_frames = resources.transition_frames;
        self.tile_stream_terminal_fallback = resources.terminal_fallback;
        self.tile_stream_virtual_page_count = resources.virtual_page_count;
        self.tile_stream_infinite_topology = resources.infinite_topology;
        self.use_tile_stream = resources.enable;
        self.recreate_bind_group();
        self.notify_invalidation(InvalidationReason::TerrainChanged);
    }

    pub fn set_use_tile_stream(&mut self, enable: bool) {
        if self.use_tile_stream != enable {
            self.use_tile_stream = enable;
            self.notify_invalidation(InvalidationReason::TerrainChanged);
        }
    }

    pub fn tile_stream_enabled(&self) -> bool {
        self.use_tile_stream
    }

    /// Resolution of the pyramid level the currently-streamed pages were cut at,
    /// mirrored into `FrameUniforms.stream2.zw`. Tests pin this to prove the shader
    /// denormalizes streamed UVs against the page resolution rather than the
    /// monolithic `tex_size` (the corner-artifact revert check).
    pub fn tile_stream_res(&self) -> (u32, u32) {
        (
            self.tile_stream_target_resolution,
            self.tile_stream_target_resolution,
        )
    }

    /// The output revision the currently-streamed pages were stamped with, mirrored
    /// into `FrameUniforms.stream2` and matched by the shader's stale-page gate
    /// (`find_tile_page`). Tests pin this against `TerrainRuntime::output_revision`
    /// to prove the sync path stamps one authoritative revision into both places.
    pub fn tile_stream_revision(&self) -> u64 {
        self.tile_stream_content.output_revision
    }

    pub fn set_tile_stream_debug_mode(&mut self, mode: u32) {
        if self.tile_stream_debug_mode != mode {
            self.tile_stream_debug_mode = mode;
            self.notify_invalidation(InvalidationReason::TerrainChanged);
        }
    }

    pub fn set_shadows_enabled(&mut self, enable: bool) {
        self.shadow_map.set_enabled(enable);
        self.notify_invalidation(InvalidationReason::LightingChanged);
    }

    pub fn height_binding_revision(&self) -> u64 {
        self.height_binding_revision
    }

    /// Upload or clear the Phase J overhang / cave roof proxy mesh.
    pub fn sync_overhang_mesh(&mut self, mesh: Option<&terra_core::volumetric::OverhangMesh>) {
        if let Some(overhang) = self.overhang.ready_mut() {
            match mesh {
                Some(m) if !m.is_empty() => overhang.upload_mesh(&self.device, m),
                _ => overhang.clear(),
            }
        }
        self.notify_invalidation(InvalidationReason::GeometryChanged);
    }

    /// Rebuild sparse viewport instances from the evaluated vegetation field.
    pub fn sync_vegetation_instances(
        &mut self,
        height: &Heightfield,
        density: Option<&MaskField>,
        scale_min: f32,
        scale_max: f32,
        yaw_variation_deg: f32,
    ) {
        if let Some(vegetation) = self.vegetation.ready_mut() {
            vegetation.sync(
                &self.device,
                height,
                density,
                scale_min,
                scale_max,
                yaw_variation_deg,
            );
        }
        self.notify_invalidation(InvalidationReason::GeometryChanged);
    }

    pub fn set_ocean_level(&mut self, level: Option<f32>) {
        let level = level.filter(|value| value.is_finite());
        if self.ocean_level != level {
            self.ocean_level = level;
            self.notify_invalidation(InvalidationReason::GeometryChanged);
        }
    }

    pub fn ocean_enabled(&self) -> bool {
        self.ocean_level.is_some()
    }

    /// Clear viewport GPU state that belongs to the previous document.
    pub fn reset_project_state(
        &mut self,
        world_size: (f32, f32),
        ocean_level: Option<f32>,
        traversal_mode: TerrainTraversalMode,
    ) {
        self.traversal_mode = traversal_mode;
        self.infinite_presentation = None;
        self.use_tile_stream = false;
        self.tile_stream_virtual_page_count = 0;
        self.tile_stream_infinite_topology = None;
        self.presentation_baseline = None;
        self.last_presentation_record = None;
        self.heights
            .reset_project_state(&self.device, &self.queue, world_size);
        if let Some(vegetation) = self.vegetation.ready_mut() {
            let blank = terra_core::heightfield::Heightfield::zeros(
                terra_core::heightfield::HeightfieldMetrics {
                    width: 8,
                    height: 8,
                    world_size_x: world_size.0.max(1.0),
                    world_size_z: world_size.1.max(1.0),
                    tile_size: 8,
                    halo: 0,
                },
            );
            vegetation.sync(&self.device, &blank, None, 1.0, 1.0, 0.0);
        }
        if let Some(overhang) = self.overhang.ready_mut() {
            overhang.clear();
        }
        self.ocean_level = ocean_level.filter(|v| v.is_finite());
        self.notify_invalidation(InvalidationReason::TerrainChanged);
        self.request_camera_reframe();
        let extent = world_size.0.max(world_size.1);
        let next = ClipmapConfig::for_world_with_height(
            extent,
            self.clipmap.fallback.grid_size,
            self.heights
                .tex_size
                .0
                .max(self.heights.tex_size.1)
                .max(513),
        );
        self.clipmap = next;
        self.world_grid = self.clipmap.fallback.clone();
        if self.grid.resolution != self.world_grid.grid_size {
            self.grid = TerrainGrid::new(&self.device, self.world_grid.grid_size);
        }
        self.recreate_bind_group();
    }

    pub const fn traversal_mode(&self) -> TerrainTraversalMode {
        self.traversal_mode
    }

    pub fn configure_infinite_presentation(&mut self, config: InfinitePresentationConfig) {
        debug_assert_eq!(
            self.active_pipeline_variant,
            TerrainShaderVariant::Infinite,
            "Infinite traversal requires the Infinite terrain pipeline bundle"
        );
        self.traversal_mode = TerrainTraversalMode::Infinite;
        self.infinite_presentation = Some(config);
        let next = ClipmapConfig::for_infinite(config.topology, config.horizon_m);
        self.clipmap = next;
        self.world_grid = self.clipmap.fallback.clone();
        if self.grid.resolution != self.world_grid.grid_size {
            self.grid = TerrainGrid::new(&self.device, self.world_grid.grid_size);
        }
        self.ring_grids = self
            .clipmap
            .rings
            .iter()
            .map(|ring| TerrainGrid::new(&self.device, ring.grid_size))
            .collect();
        self.ensure_ring_uniform_bufs();
        self.recreate_bind_group();
        self.request_camera_reframe();
    }

    /// Camera-relative frame origin snapped to the finest Infinite tile lattice.
    pub fn render_origin_xz(&self) -> glam::DVec2 {
        let Some(config) = self.infinite_presentation else {
            return glam::DVec2::ZERO;
        };
        let origin = config.topology.origin;
        let span = config.topology.finest_spacing_m * f64::from(config.topology.tile_size);
        if !span.is_finite() || span <= 0.0 {
            return glam::DVec2::new(origin.x_m(), origin.z_m());
        }
        glam::DVec2::new(
            origin.x_m() + ((self.camera.target.x - origin.x_m()) / span).floor() * span,
            origin.z_m() + ((self.camera.target.z - origin.z_m()) / span).floor() * span,
        )
    }

    pub fn camera_view_proj(&self, aspect: f32) -> glam::Mat4 {
        self.camera
            .view_proj_relative_to(aspect, self.render_origin_xz())
    }

    /// Apply the active topology's camera constraint. Infinite traversal keeps
    /// the rig unchanged, including across negative fixed-origin coordinates.
    pub fn constrain_camera(&mut self) {
        self.traversal_mode
            .constrain_camera(&mut self.camera, self.heights.world_size);
    }

    fn clipmap_traversal_bounds(&self) -> ClipmapTraversalBounds {
        match self.traversal_mode {
            TerrainTraversalMode::Bounded => ClipmapTraversalBounds::Bounded {
                world_x: self.heights.world_size.0,
                world_z: self.heights.world_size.1,
            },
            TerrainTraversalMode::Infinite => ClipmapTraversalBounds::Infinite,
        }
    }

    /// Push viewport chrome display aids (wireframe / grid / bounds / contours / shading).
    pub fn set_display_aids(&mut self, aids: ViewportDisplayAids) {
        if self.display_aids.wireframe != aids.wireframe
            || self.display_aids.grid != aids.grid
            || self.display_aids.world_bounds != aids.world_bounds
            || self.display_aids.contours != aids.contours
            || self.display_aids.shading != aids.shading
        {
            self.notify_invalidation(InvalidationReason::RenderModeChanged);
        }
        self.display_aids = aids;
    }

    /// 0 = hide placement tint; ~0.55–0.7 is a readable artist overlay.
    pub fn set_biome_tint_strength(&mut self, strength: f32) {
        let strength = strength.clamp(0.0, 1.0);
        if (self.biome_tint_strength - strength).abs() > 1e-4 {
            self.biome_tint_strength = strength;
            self.notify_invalidation(InvalidationReason::MaterialChanged);
        }
    }

    /// Deprecated: prefer [`Self::set_renderer_mode`]. Only applies when mode is ProgressivePt.
    pub fn set_progressive_enabled(&mut self, enabled: bool) {
        let backend = self.presentation_backend();
        if let Some(bundle) = self.progressive.ready_mut() {
            bundle
                .progressive
                .set_enabled(enabled && matches!(backend, PresentationBackendId::ProgressivePt));
        }
    }

    pub fn progressive_samples(&self) -> u32 {
        self.progressive
            .ready()
            .map_or(0, |bundle| bundle.progressive.samples())
    }

    /// Acquire the swapchain frame, render terrain into it, and return it
    /// **un-presented** for the caller to composite the GUI onto. This is the
    /// windowed entry point; the terrain half of the [frame seam](crate#frame-seam).
    ///
    /// Caller obligations (nothing here enforces them):
    /// - Build the GUI view from the returned frame's texture — the same frame,
    ///   not a fresh `get_current_texture` — so both passes target one surface.
    /// - Run exactly one GUI pass into that view with `LoadOp::Load`. Queue
    ///   submission order is the only thing sequencing GUI-after-terrain, so the
    ///   GUI encoder must be submitted after this call returns.
    /// - Call [`wgpu::SurfaceTexture::present`] only after the GUI submit. The
    ///   returned frame is not presented here.
    ///
    /// # Errors
    ///
    /// - [`RenderError::Surface`] wraps `wgpu::SurfaceError` unmapped so the app
    ///   can match and recover: `Timeout` → skip the frame; `Outdated`/`Lost` →
    ///   [`reconfigure`](Self::reconfigure) and repaint; `OutOfMemory` → exit;
    ///   `Other` → skip.
    /// - [`RenderError::Msg`] only when called on a headless renderer (no
    ///   surface) — use [`render_to_view`](Self::render_to_view) instead.
    pub fn render_terrain(&mut self) -> Result<wgpu::SurfaceTexture, RenderError> {
        let Some(surface) = &self.surface else {
            return Err(RenderError::Msg(
                "headless renderer has no surface; render via render_to_view".into(),
            ));
        };
        let frame = surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let (width, height) = (self.config.width, self.config.height);
        self.render_to_view(&view, width, height);
        Ok(frame)
    }

    /// Record and submit one terrain frame (shadow, backend, scene composite, post) into `view`.
    ///
    /// Target contract, until sizing is decoupled from the surface configuration:
    /// - `view` must be a RENDER_ATTACHMENT-usable view whose texture format equals
    ///   `self.config.format` — the terrain/ocean/wireframe/overlay/composite
    ///   pipelines were baked against that format at construction. wgpu 24 exposes
    ///   no format getter on `TextureView`, so this cannot be asserted here.
    /// - `width`/`height` must equal `self.config.width`/`.height`: the depth
    ///   buffer, progressive history, and path-tracer targets are allocated at the
    ///   configured size (see `resize`), and the passes attach them alongside `view`.
    pub fn render_to_view(&mut self, view: &wgpu::TextureView, width: u32, height: u32) {
        debug_assert_eq!(
            (width, height),
            (self.config.width, self.config.height),
            "render_to_view target size must match the configured size; call resize() first"
        );

        self.scene_versions.begin_frame();

        let aspect = width as f32 / height.max(1) as f32;
        if let Some(reason) = self.scene_versions.update_camera(&self.camera, aspect) {
            self.notify_invalidation(reason);
        }

        self.quality.update_for_state(self.last_interaction_state);
        if let Some(bundle) = self.progressive.ready_mut() {
            bundle
                .progressive
                .set_max_samples(self.quality.config.max_accumulated_spp);
            bundle.progressive.set_history_cap(self.quality.history_cap);
        }

        let internal_scale = self.quality.internal_scale;
        if (self.last_internal_scale - internal_scale).abs() > 1e-4 {
            let mask = self.adaptive.prepare_all_active_mask();
            if let Some(bundle) = self.progressive.ready_mut() {
                bundle.path_tracer.resize(
                    &self.device,
                    &self.queue,
                    &self.pipelines,
                    width,
                    height,
                    internal_scale,
                );
                bundle.path_tracer.upload_sample_mask(&self.queue, &mask);
            }
            self.notify_invalidation(InvalidationReason::ViewportResized);
            self.last_internal_scale = internal_scale;
        }

        let backend = self.presentation_backend();
        // Raster cast shadows are driven by the lighting shadow-strength control;
        // 0 keeps the depth pass off, matching the historical "no shadows" look.
        self.shadow_map
            .set_enabled(self.lighting.shadow_strength > 1e-4);
        let shadows_for_schedule = self.traversal_mode == TerrainTraversalMode::Bounded
            && self.shadow_map.enabled()
            && matches!(backend, PresentationBackendId::RasterLit);
        // Converged progressive frames (spp 0) present the last HDR without a new
        // dispatch; the schedule records that so its plan matches what runs.
        let pt_dispatch = self.quality.spp_this_frame > 0;
        self.frame_graph.begin(FrameSchedule::for_backend(
            backend,
            shadows_for_schedule,
            pt_dispatch,
        ));
        // The schedule is now the single source of truth for the frame path.
        let backend = self
            .frame_graph
            .schedule
            .backend
            .expect("frame schedule always records a backend");
        let path_trace_mode = matches!(backend, PresentationBackendId::ProgressivePt);
        // Keep progressive post armed whenever the schedule expects it.
        if let Some(bundle) = self.progressive.ready_mut() {
            if self.frame_graph.schedule.progressive_post && !bundle.progressive.enabled() {
                bundle.progressive.set_enabled(true);
            } else if !self.frame_graph.schedule.progressive_post && bundle.progressive.enabled() {
                bundle.progressive.set_enabled(false);
            }
        }

        self.constrain_camera();
        let render_origin = self.render_origin_xz();
        let view_proj = self.camera_view_proj(aspect);
        let (min_h, max_h) = self.heights.height_range;
        let (tw, th) = self.heights.tex_size;
        self.last_grid_resolution = self.grid.resolution;

        let lighting_signature = [
            self.lighting.light_dir[0],
            self.lighting.light_dir[1],
            self.lighting.light_dir[2],
            self.lighting.light_dir[3],
            self.lighting.exposure,
            self.lighting.clear[0],
            self.lighting.clear[1],
            self.lighting.clear[2],
        ];
        if self
            .progressive
            .ready()
            .is_some_and(|bundle| bundle.progressive.signature_changed(lighting_signature))
        {
            self.notify_invalidation(InvalidationReason::LightingChanged);
        }
        if let Some(bundle) = self.progressive.ready_mut() {
            bundle.progressive.prepare_signature(lighting_signature);
        }

        let progressive_seed = self.global_frame_index as u32;
        let contour_interval = {
            let span = (max_h - min_h).max(1.0);
            (span / 20.0).clamp(5.0, 100.0)
        };
        let slab_base = {
            let span = (max_h - min_h).max(1.0);
            let extent = self.heights.world_size.0.max(self.heights.world_size.1);
            let thickness = span.max(extent * 0.03).max(40.0);
            min_h - thickness
        };
        let (atm_clear, fog) = shadows::atmosphere_from_sun(
            [
                self.lighting.light_dir[0],
                self.lighting.light_dir[1],
                self.lighting.light_dir[2],
            ],
            self.lighting.clear,
        );
        let light_view_proj = self.shadow_map.update_light(
            &self.queue,
            [
                self.lighting.light_dir[0],
                self.lighting.light_dir[1],
                self.lighting.light_dir[2],
            ],
            self.heights.world_size,
            self.heights.height_range,
        );
        let absolute_eye = self.camera.eye();
        let eye = glam::Vec3::new(
            (absolute_eye.x - render_origin.x) as f32,
            absolute_eye.y as f32,
            (absolute_eye.z - render_origin.y) as f32,
        );
        let infinite_topology = self.tile_stream_infinite_topology.or_else(|| {
            self.infinite_presentation
                .map(|presentation| presentation.topology)
        });
        let (stream6, stream7, stream8) = if let Some(topology) = infinite_topology {
            let tile_span = topology.finest_spacing_m * f64::from(topology.tile_size);
            let anchor_x = ((render_origin.x - topology.origin.x_m()) / tile_span).floor() as i64;
            let anchor_z = ((render_origin.y - topology.origin.z_m()) / tile_span).floor() as i64;
            let x = anchor_x as u64;
            let z = anchor_z as u64;
            (
                [
                    1,
                    if self.use_tile_stream {
                        self.tile_stream_virtual_page_count
                    } else {
                        0
                    },
                    u32::from(topology.max_lod.get()),
                    0,
                ],
                [x as u32, (x >> 32) as u32, z as u32, (z >> 32) as u32],
                [topology.finest_spacing_m as f32, tile_span as f32, 0.0, 0.0],
            )
        } else {
            ([0; 4], [0; 4], [0.0; 4])
        };
        let base_uniforms = FrameUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            light_dir: self.lighting.light_dir,
            world: [
                self.heights.world_size.0,
                self.heights.world_size.1,
                min_h,
                max_h.max(min_h + 1e-3),
            ],
            grid: [
                tw as f32,
                th as f32,
                self.ocean_level.unwrap_or(min_h - 1.0),
                slab_base,
            ],
            clipmap: [0.0; 4],
            eye: [eye.x, eye.y, eye.z, self.lighting.exposure],
            render: [
                progressive_seed as f32,
                if path_trace_mode { 1.0 } else { 0.0 },
                self.progressive
                    .ready()
                    .map_or(0, |bundle| bundle.progressive.samples()) as f32,
                self.biome_tint_strength,
            ],
            viz: [
                self.display_aids.shading as u32 as f32,
                if self.display_aids.contours { 1.0 } else { 0.0 },
                contour_interval,
                0.0,
            ],
            light_view_proj: light_view_proj.to_cols_array_2d(),
            stream: [
                if self.use_tile_stream { 1.0 } else { 0.0 },
                self.tile_stream_tile_size,
                self.tile_stream_halo,
                self.tile_stream_max_pages,
            ],
            fog,
            shadow: [
                if self.shadow_map.enabled() { 1.0 } else { 0.0 },
                0.0015,
                self.tile_stream_target_level as f32,
                1.25,
            ],
            raster: [
                self.lighting.ambient_strength,
                self.lighting.shadow_strength,
                self.lighting.fog_strength,
                0.0,
            ],
            stream2: [
                self.tile_stream_content.document_revision as u32,
                (self.tile_stream_content.document_revision >> 32) as u32,
                self.tile_stream_content.plan_revision as u32,
                (self.tile_stream_content.plan_revision >> 32) as u32,
            ],
            stream3: [
                self.tile_stream_content.output_revision as u32,
                (self.tile_stream_content.output_revision >> 32) as u32,
                self.tile_stream_content.content_revision as u32,
                (self.tile_stream_content.content_revision >> 32) as u32,
            ],
            stream4: [
                self.tile_stream_level_count,
                u32::from(self.tile_stream_target_level),
                self.global_frame_index as u32,
                self.tile_stream_transition_frames,
            ],
            stream5: [
                if self.tile_stream_terminal_fallback
                    == TerrainTerminalFallback::MonolithicMigration
                {
                    1.0
                } else {
                    0.0
                },
                self.tile_stream_debug_mode as f32,
                0.0,
                0.0,
            ],
            stream6,
            stream7,
            stream8,
        };
        let world_x = self.heights.world_size.0;
        let world_z = self.heights.world_size.1;
        let fallback_spacing = self
            .clipmap
            .fallback
            .spacing_for_extent(world_x.max(world_z));

        if let Some(timer) = self.gpu_timer.as_mut() {
            timer.poll_readback(&self.device);
            self.last_gpu_timings = timer.last();
            timer.begin_frame(std::mem::take(&mut self.pending_presentation_trace));
        }
        self.staging.begin_frame();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("terrain-enc"),
            });
        if let Some(overhang) = self.overhang.ready_mut() {
            overhang.upload_view_proj(&self.queue, view_proj, self.lighting.light_dir);
        }
        if let Some(vegetation) = self.vegetation.ready_mut() {
            vegetation.upload_view_proj(&self.queue, view_proj, self.lighting.light_dir);
        }

        // Depth-only directional shadow pass (RasterLit only).
        if self.frame_graph.schedule.shadow {
            self.frame_graph.mark(PassKind::Shadow);
            let shadow_ts = self
                .gpu_timer
                .as_mut()
                .and_then(|t| t.shadow_timestamp_writes());
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("shadow-pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.shadow_map.view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: shadow_ts,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&self.shadow_map.pipeline);
                pass.set_bind_group(0, &self.shadow_map.bind_group, &[]);
                pass.set_vertex_buffer(0, self.grid.vertex_buf.slice(..));
                pass.set_index_buffer(self.grid.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.grid.index_count, 0, 0..1);
            }
        }

        match backend {
            PresentationBackendId::ProgressivePt => {
                // Converged ⇒ spp 0 still presents last HDR via progressive post.
                if self.frame_graph.schedule.pt_dispatch {
                    self.frame_graph.mark(PassKind::ProgressivePt);
                    let mask = self.adaptive.prepare_all_active_mask();
                    self.progressive
                        .ready_mut()
                        .expect("progressive backend requires installed bundle")
                        .path_tracer
                        .upload_sample_mask(&self.queue, &mask);

                    let target = glam::Vec3::new(
                        (self.camera.target.x - render_origin.x) as f32,
                        self.camera.target.y as f32,
                        (self.camera.target.z - render_origin.y) as f32,
                    );
                    let view_mat = glam::Mat4::look_at_rh(eye, target, glam::Vec3::Y);
                    let view_inv = view_mat.inverse();
                    let dx = world_x / tw.max(1) as f32;
                    let dz = world_z / th.max(1) as f32;
                    let cfg = self.quality.config;
                    let pt_uniforms = PathTracer::uniforms_from_scene(
                        view_inv,
                        aspect,
                        self.camera.fov_y,
                        self.camera.near,
                        self.camera.far,
                        self.lighting.light_dir,
                        self.lighting.clear,
                        self.lighting.exposure,
                        self.heights.world_size,
                        self.heights.height_range,
                        (tw as f32, th as f32),
                        (dx, dz),
                        cfg.direct_luminance_clamp,
                        cfg.indirect_luminance_clamp,
                        cfg.sun_angular_radius_rad,
                        self.quality.bounce_count,
                        self.quality.spp_this_frame,
                    );
                    let path_ts = self
                        .gpu_timer
                        .as_mut()
                        .and_then(|t| t.path_trace_timestamp_writes());
                    self.progressive
                        .ready_mut()
                        .expect("progressive backend requires installed bundle")
                        .path_tracer
                        .dispatch(
                            &self.device,
                            &self.queue,
                            &mut encoder,
                            self.heights.display_height_view(),
                            self.heights.display_normal_view(),
                            self.heights.materials_view(),
                            pt_uniforms,
                            self.quality.spp_this_frame,
                            path_ts,
                        );
                }
            }
            PresentationBackendId::RasterLit => {
                let present = backends::raster_lit::plan_raster_present(
                    &self.clipmap,
                    ClipmapPresentInput {
                        camera_x: absolute_eye.x,
                        camera_z: absolute_eye.z,
                        world_x,
                        world_z,
                        height_tex_w: tw,
                        height_tex_h: th,
                        traversal_bounds: self.clipmap_traversal_bounds(),
                    },
                );

                // Upload all per-draw uniforms before the pass (queue writes are
                // illegal while a buffer is bound in an active render pass).
                if present.use_single_grid {
                    let mut uniforms = base_uniforms;
                    uniforms.clipmap = [
                        (present.fallback_origin_x - render_origin.x) as f32,
                        (present.fallback_origin_z - render_origin.y) as f32,
                        present.fallback_spacing,
                        self.grid.resolution as f32,
                    ];
                    uniforms.viz[3] = 0.0;
                    self.queue
                        .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
                } else {
                    if present.draw_fallback {
                        let mut uniforms = base_uniforms;
                        uniforms.clipmap = [
                            (present.fallback_origin_x - render_origin.x) as f32,
                            (present.fallback_origin_z - render_origin.y) as f32,
                            present.fallback_spacing,
                            self.grid.resolution as f32,
                        ];
                        uniforms.viz[3] = present.fallback_exclude_half_extent;
                        uniforms.stream8[2] =
                            (present.fallback_exclude_center_x - render_origin.x) as f32;
                        uniforms.stream8[3] =
                            (present.fallback_exclude_center_z - render_origin.y) as f32;
                        self.queue.write_buffer(
                            &self.uniform_buf,
                            0,
                            bytemuck::bytes_of(&uniforms),
                        );
                    }
                    for draw in &present.rings {
                        let Some(ring_u) = self.ring_uniform_bufs.get(draw.ring_index) else {
                            continue;
                        };
                        let mut uniforms = base_uniforms;
                        uniforms.clipmap = [
                            (draw.origin_x - render_origin.x) as f32,
                            (draw.origin_z - render_origin.y) as f32,
                            draw.spacing,
                            draw.grid_size as f32,
                        ];
                        uniforms.viz[3] = draw.exclude_half_extent;
                        uniforms.stream8[2] = (draw.exclude_center_x - render_origin.x) as f32;
                        uniforms.stream8[3] = (draw.exclude_center_z - render_origin.y) as f32;
                        self.queue
                            .write_buffer(ring_u, 0, bytemuck::bytes_of(&uniforms));
                    }
                }

                let color_view = view;
                self.frame_graph.mark(PassKind::RasterLit);
                let terrain_ts = self
                    .gpu_timer
                    .as_mut()
                    .and_then(|t| t.terrain_timestamp_writes());
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("raster-lit-pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: color_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: atm_clear[0] as f64,
                                    g: atm_clear[1] as f64,
                                    b: atm_clear[2] as f64,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &self.depth,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: terrain_ts,
                        occlusion_query_set: None,
                    });
                    pass.set_pipeline(&self.active_pipeline_bundle().terrain);
                    // Infinite clipmaps are transient surface patches, not solid terrain
                    // blocks. Drawing each grid's skirt and underside creates one nested
                    // wall set per LOD; the clipmap holes then leave only repeated corners.
                    if present.use_single_grid {
                        pass.set_bind_group(0, &self.bind_group, &[]);
                        pass.set_vertex_buffer(0, self.grid.vertex_buf.slice(..));
                        pass.set_index_buffer(
                            self.grid.index_buf.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        let index_count =
                            terrain_patch_index_count(&self.grid, self.traversal_mode);
                        pass.draw_indexed(0..index_count, 0, 0..1);
                    } else {
                        if present.draw_fallback {
                            pass.set_bind_group(0, &self.bind_group, &[]);
                            pass.set_vertex_buffer(0, self.grid.vertex_buf.slice(..));
                            pass.set_index_buffer(
                                self.grid.index_buf.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            let index_count =
                                terrain_patch_index_count(&self.grid, self.traversal_mode);
                            pass.draw_indexed(0..index_count, 0, 0..1);
                        }
                        for draw in &present.rings {
                            let Some(ring_grid) = self.ring_grids.get(draw.ring_index) else {
                                continue;
                            };
                            let Some(ring_bg) = self.ring_bind_groups.get(draw.ring_index) else {
                                continue;
                            };
                            pass.set_bind_group(0, ring_bg, &[]);
                            pass.set_vertex_buffer(0, ring_grid.vertex_buf.slice(..));
                            pass.set_index_buffer(
                                ring_grid.index_buf.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            let index_count =
                                terrain_patch_index_count(ring_grid, self.traversal_mode);
                            pass.draw_indexed(0..index_count, 0, 0..1);
                        }
                    }
                }

                if self.frame_graph.schedule.scene_composite {
                    self.frame_graph.mark(PassKind::SceneComposite);
                    // Hole-free uniforms for terrain-scene composition.
                    {
                        let mut uniforms = base_uniforms;
                        uniforms.clipmap = [
                            (present.fallback_origin_x - render_origin.x) as f32,
                            (present.fallback_origin_z - render_origin.y) as f32,
                            fallback_spacing,
                            self.grid.resolution as f32,
                        ];
                        uniforms.viz[3] = 0.0;
                        self.queue.write_buffer(
                            &self.uniform_buf,
                            0,
                            bytemuck::bytes_of(&uniforms),
                        );
                    }
                    {
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("raster-lit-scene-composite"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: color_view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: &self.depth,
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });
                        if self.traversal_mode == TerrainTraversalMode::Bounded
                            && self.ocean_level.is_some()
                        {
                            if let Some(ocean) =
                                self.ocean_slot(self.active_pipeline_variant).ready()
                            {
                                pass.set_pipeline(&ocean.pipeline);
                                pass.set_bind_group(0, &self.bind_group, &[]);
                                pass.set_vertex_buffer(0, self.grid.vertex_buf.slice(..));
                                pass.set_index_buffer(
                                    self.grid.index_buf.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                pass.draw_indexed(0..self.grid.surface_index_count, 0, 0..1);
                            }
                        }
                        if self.display_aids.wireframe {
                            if let Some(wireframe) =
                                self.wireframe_slot(self.active_pipeline_variant).ready()
                            {
                                pass.set_pipeline(&wireframe.pipeline);
                                pass.set_bind_group(0, &self.bind_group, &[]);
                                pass.set_vertex_buffer(0, self.grid.vertex_buf.slice(..));
                                pass.set_index_buffer(
                                    self.grid.edge_index_buf.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                pass.draw_indexed(0..self.grid.edge_index_count, 0, 0..1);
                            }
                        }
                        if self.traversal_mode == TerrainTraversalMode::Bounded {
                            if let Some(vegetation) = self.vegetation.ready() {
                                vegetation.draw(&mut pass);
                            }
                            if let Some(overhang) = self.overhang.ready() {
                                overhang.draw(&mut pass);
                            }
                        }
                    }
                }
            }
        }

        if self.frame_graph.schedule.progressive_post {
            self.frame_graph.mark(PassKind::ProgressivePost);
            let (temporal_ts, denoise_ts) = self
                .gpu_timer
                .as_mut()
                .map(|t| t.progressive_timestamp_writes())
                .unwrap_or((None, None));
            let bundle = self
                .progressive
                .ready_mut()
                .expect("progressive post requires installed bundle");
            let ProgressivePresentationBundle {
                progressive,
                path_tracer,
            } = bundle;
            ProgressivePostPipeline::resolve_hdr(
                progressive,
                &self.device,
                &self.queue,
                &mut encoder,
                HdrFrame {
                    color: path_tracer.radiance_view(),
                    width,
                    height,
                },
                GBufferViews {
                    depth: path_tracer.depth_view(),
                    normal: Some(path_tracer.normal_view()),
                },
                view,
                view_proj,
                self.quality.config.depth_rel_tol,
                self.quality.config.history_clamp_k,
                self.quality.atrous_iterations,
                temporal_ts,
                denoise_ts,
                self.debug_viz_mode,
            );
        }

        self.frame_graph
            .end_frame(self.gpu_timer.as_mut(), &mut encoder);
        self.queue.submit(Some(encoder.finish()));

        let pixels = width as u64 * height as u64;
        let spp = self
            .quality
            .spp_this_frame
            .max(if path_trace_mode { 1 } else { 0 });
        let bounces = self.quality.bounce_count.max(1);
        self.quality.approx_rays_this_frame = pixels * u64::from(spp) * u64::from(bounces);
        let (active, reduced, converged) = self.adaptive.count_by_state();
        self.quality.active_sampling_tiles = active;
        self.quality.reduced_sampling_tiles = reduced;
        self.quality.converged_sampling_tiles = converged;
        if self.adaptive.tile_count() > 0 {
            self.quality.convergence_fraction =
                converged as f32 / self.adaptive.tile_count() as f32;
        }

        let gpu_ms = (self.last_gpu_timings.terrain_us
            + self.last_gpu_timings.shadow_us
            + self.last_gpu_timings.path_trace_us
            + self.last_gpu_timings.temporal_us
            + self.last_gpu_timings.denoise_us) as f32
            / 1000.0;
        self.quality.observe_gpu_frame_ms(gpu_ms);
        // Adaptive variance gating: keep tiles active until a real GPU variance path exists.
        // Do not drive the hot path from fake sample-count variance.
        self.heights.tick_retirement(self.global_frame_index);
        self.global_frame_index = self.global_frame_index.wrapping_add(1);
    }

    /// Bootstrap adaptive tile states from accumulated sample count (debug / offline only).
    #[allow(dead_code)]
    fn update_adaptive_from_progressive(&mut self) {
        let samples = self
            .progressive
            .ready()
            .map_or(0, |bundle| bundle.progressive.samples()) as f32;
        if samples <= 0.0 {
            return;
        }
        let min = self.quality.config.min_samples_before_converge.max(1) as f32;
        let variance = if samples >= min { 0.001 } else { 0.05 };
        let mut summaries = Vec::with_capacity(self.adaptive.tile_count() as usize);
        for ty in 0..self.adaptive.tiles_y {
            for tx in 0..self.adaptive.tiles_x {
                summaries.push(VarianceTileSummary {
                    tile_x: tx,
                    tile_y: ty,
                    mean_luminance: 0.5,
                    variance,
                    sample_count: samples,
                });
            }
        }
        self.adaptive
            .update_from_variance_summaries(&summaries, &self.quality.config);
    }
}

const ALBEDO_TEX_SIZE: u32 = 256;

fn create_dummy_tile_stream(
    device: &wgpu::Device,
) -> (
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Buffer,
    wgpu::Buffer,
    wgpu::Buffer,
) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("dummy-tile-atlas"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("dummy-tile-atlas-view"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    let page_table = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dummy-page-table"),
        size: std::mem::size_of::<terra_gpu::GpuPageTableEntry>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let virtual_page_table = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dummy-virtual-page-table"),
        size: std::mem::size_of::<terra_gpu::GpuVirtualPageEntry>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let level_table = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dummy-terrain-level-table"),
        size: std::mem::size_of::<terra_gpu::GpuTerrainLevelEntry>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    (texture, view, page_table, virtual_page_table, level_table)
}

fn create_albedo_array(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::Texture, wgpu::TextureView, wgpu::Sampler) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("material-albedo-array"),
        size: wgpu::Extent3d {
            width: ALBEDO_TEX_SIZE,
            height: ALBEDO_TEX_SIZE,
            depth_or_array_layers: MATERIAL_SLOT_COUNT as u32,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    // Neutral mid-grey fill so unbound layers don't flash black.
    let grey = vec![180u8; (ALBEDO_TEX_SIZE * ALBEDO_TEX_SIZE * 4) as usize];
    for layer in 0..MATERIAL_SLOT_COUNT as u32 {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &grey,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(ALBEDO_TEX_SIZE * 4),
                rows_per_image: Some(ALBEDO_TEX_SIZE),
            },
            wgpu::Extent3d {
                width: ALBEDO_TEX_SIZE,
                height: ALBEDO_TEX_SIZE,
                depth_or_array_layers: 1,
            },
        );
    }
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("material-albedo-array-view"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("material-albedo-samp"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    (texture, view, sampler)
}

fn load_albedo_png(path: &str) -> Result<image::RgbaImage, String> {
    let img = image::open(path).map_err(|e| e.to_string())?;
    let rgba = img.to_rgba8();
    Ok(image::imageops::resize(
        &rgba,
        ALBEDO_TEX_SIZE,
        ALBEDO_TEX_SIZE,
        image::imageops::FilterType::Triangle,
    ))
}

fn write_albedo_layer(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    layer: u32,
    rgba: &image::RgbaImage,
) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: 0,
                y: 0,
                z: layer,
            },
            aspect: wgpu::TextureAspect::All,
        },
        rgba.as_raw(),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(ALBEDO_TEX_SIZE * 4),
            rows_per_image: Some(ALBEDO_TEX_SIZE),
        },
        wgpu::Extent3d {
            width: ALBEDO_TEX_SIZE,
            height: ALBEDO_TEX_SIZE,
            depth_or_array_layers: 1,
        },
    );
}

/// Backend selection for instance creation.
///
/// Explicit `backends` in `InstanceDescriptor` ignores `WGPU_BACKEND`, so we parse it
/// ourselves. On Windows, default to DX12 — Vulkan + OBS/Overwolf/Medal implicit layers
/// has been observed to STATUS_STACK_OVERFLOW inside `vkCreateDevice`.
fn preferred_backends() -> wgpu::Backends {
    if let Ok(raw) = std::env::var("WGPU_BACKEND") {
        let mut backends = wgpu::Backends::empty();
        for part in raw.split([',', '|']).map(|p| p.trim().to_ascii_lowercase()) {
            match part.as_str() {
                "vulkan" | "vk" => backends |= wgpu::Backends::VULKAN,
                "dx12" | "d3d12" => backends |= wgpu::Backends::DX12,
                "metal" => backends |= wgpu::Backends::METAL,
                "gl" | "gles" => backends |= wgpu::Backends::GL,
                "primary" => return wgpu::Backends::PRIMARY,
                "all" => return wgpu::Backends::all(),
                _ => {}
            }
        }
        if !backends.is_empty() {
            return backends;
        }
    }
    if cfg!(target_os = "windows") {
        wgpu::Backends::DX12
    } else {
        wgpu::Backends::PRIMARY
    }
}

fn create_depth(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    depth_tex.create_view(&wgpu::TextureViewDescriptor {
        label: Some("depth-attach"),
        ..Default::default()
    })
}

#[cfg(test)]
mod shader_tests {
    #[test]
    fn terrain_and_ocean_shader_parses() {
        let source = crate::terrain_shader::compose(crate::TerrainShaderVariant::Bounded);
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|error| panic!("terrain WGSL parse failed: {error}"));
        assert!(module
            .entry_points
            .iter()
            .any(|entry| entry.name == "vs_ocean"));
        assert!(module
            .entry_points
            .iter()
            .any(|entry| entry.name == "fs_ocean"));
    }

    /// Return the `{ ... }` body of the first WGSL function named `name`.
    fn fn_body<'a>(src: &'a str, name: &str) -> &'a str {
        let sig = format!("fn {name}(");
        let start = src
            .find(&sig)
            .unwrap_or_else(|| panic!("fn {name} not found in shader"));
        let open = start
            + src[start..]
                .find('{')
                .unwrap_or_else(|| panic!("fn {name} has no body brace"));
        let mut depth = 0i32;
        for (offset, byte) in src[open..].bytes().enumerate() {
            match byte {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return &src[open..=open + offset];
                    }
                }
                _ => {}
            }
        }
        panic!("unbalanced braces in fn {name}");
    }

    /// Revert check for multilevel addressing: lookup is dense by virtual tile,
    /// sampling uses the selected level's resolution, and no physical-page scan
    /// can return as atlas capacity grows.
    #[test]
    fn streamed_sampling_uses_page_resolution_not_monolithic_grid() {
        let source = crate::terrain_shader::compose(crate::TerrainShaderVariant::Bounded);

        let lookup = fn_body(&source, "lookup_tile_page");
        assert!(
            lookup.contains("metadata_offset") && lookup.contains("virtual_page_table"),
            "lookup must directly index the dense virtual page table"
        );
        assert!(
            !lookup.contains("for ("),
            "lookup must not scan physical pages"
        );

        let page = fn_body(&source, "sample_page_bilinear");
        assert!(
            page.contains("terrain_levels") && page.contains("resolution"),
            "page sampling must denormalize with the selected level resolution"
        );
        assert!(
            !page.contains("u.grid"),
            "streamed page sampling must not scale by the monolithic grid"
        );

        let resolve = fn_body(&source, "resolve_from_level");
        assert!(
            resolve.contains("level = level - 1") && resolve.contains("sample_height_monolithic"),
            "resolution must walk resident ancestors before terminal fallback"
        );
    }

    #[test]
    fn infinite_streaming_uses_signed_sparse_lookup_and_never_monolithic_fallback() {
        let source = crate::terrain_shader::compose(crate::TerrainShaderVariant::Infinite);
        let lookup = fn_body(&source, "lookup_tile_page_sparse");
        assert!(lookup.contains("sparse_hash") && lookup.contains("mapping.tile_x_hi"));
        assert!(lookup.contains("mapping.tile_z_hi") && lookup.contains("probe < capacity"));
        let address = fn_body(&source, "infinite_address");
        assert!(address.contains("signed64_add_i32") && address.contains("signed64_shift_right"));
        let resolve = fn_body(&source, "resolve_height_infinite_from");
        assert!(resolve.contains("lookup_tile_page_sparse"));
        assert!(
            resolve.contains("u.stream.x <= 0.5") && resolve.contains("STREAM_TERMINAL"),
            "unready Infinite presentation must be empty, never monolithic or partially resident"
        );
        assert!(
            !resolve.contains("sample_height_monolithic"),
            "Infinite page misses must not expose the finite monolithic heightfield"
        );
    }

    #[test]
    fn streamed_debug_distinguishes_exact_ancestor_blend_and_terminal() {
        let source = crate::terrain_shader::compose(crate::TerrainShaderVariant::Bounded);
        for class in [
            "STREAM_EXACT",
            "STREAM_ANCESTOR",
            "STREAM_BLEND",
            "STREAM_TERMINAL",
        ] {
            assert!(
                source.contains(class),
                "missing stream sample class {class}"
            );
        }
        let fragment = fn_body(&source, "fs_main");
        assert!(
            fragment.contains("stream5.y")
                && fragment.contains("STREAM_EXACT")
                && fragment.contains("STREAM_ANCESTOR")
                && fragment.contains("STREAM_BLEND"),
            "debug rendering must expose the actual shader resolution class"
        );
    }

    #[test]
    fn vegetation_shader_parses() {
        let source = include_str!("shaders/vegetation.wgsl");
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("vegetation WGSL parse failed: {error}"));
        assert!(module
            .entry_points
            .iter()
            .any(|entry| entry.name == "vs_main"));
        assert!(module
            .entry_points
            .iter()
            .any(|entry| entry.name == "fs_main"));
    }

    #[test]
    fn progressive_shaders_parse() {
        for (name, source) in [
            (
                "temporal",
                include_str!("shaders/progressive_temporal.wgsl"),
            ),
            ("atrous", include_str!("shaders/progressive_atrous.wgsl")),
            (
                "composite",
                include_str!("shaders/progressive_composite.wgsl"),
            ),
        ] {
            naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|error| panic!("{name} WGSL parse failed: {error}"));
        }
    }
}

#[cfg(test)]
mod render_error_tests {
    use super::*;

    // Guards the render seam (A1-G2): a surface acquire failure must cross
    // `RenderError` as the typed `Surface(wgpu::SurfaceError)` variant so the
    // app can match and recover. Collapsing the enum back to `Msg(String)`, or
    // dropping the `Surface` variant, fails to compile here.
    #[test]
    fn surface_error_crosses_the_seam_typed() {
        let err = RenderError::from(wgpu::SurfaceError::Lost);
        assert!(matches!(
            err,
            RenderError::Surface(wgpu::SurfaceError::Lost)
        ));
    }

    #[test]
    fn infinite_traversal_does_not_clamp_the_camera_rig() {
        let mut camera = OrbitCamera::default();
        camera.target.x = -1_000_000.0;
        camera.target.z = 1_000_000.0;
        let before = camera.target;
        TerrainTraversalMode::Infinite.constrain_camera(&mut camera, (4096.0, 4096.0));
        assert_eq!(camera.target, before);

        TerrainTraversalMode::Bounded.constrain_camera(&mut camera, (4096.0, 4096.0));
        assert_ne!(camera.target, before);
    }
}

#[cfg(test)]
mod pipeline_cache_lifecycle_tests {
    use super::GpuContext;

    #[test]
    fn pipeline_registry_drops_with_its_gpu_context_owners() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let context = GpuContext::new(
            gpu.device.clone(),
            gpu.queue.clone(),
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let weak = std::sync::Arc::downgrade(&context.pipelines);
        let clone = context.clone();
        drop(context);
        assert!(weak.upgrade().is_some());
        drop(clone);
        assert!(weak.upgrade().is_none());
    }
}
