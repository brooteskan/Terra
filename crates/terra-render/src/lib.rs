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
pub mod overhang;
pub mod path_tracer;
pub mod progressive;
pub mod render_quality;
pub mod retirement;
pub mod scene_versions;
pub mod shadows;
pub mod staging;
pub mod terrain_mesh;
pub mod vegetation;

pub use adaptive_sampling::{AdaptiveSamplingState, TileState, VarianceTileSummary, TILE_SIZE};
pub use backends::{
    GBufferViews, HdrFrame, PresentationBackendId, ProgressivePostPipeline, ProgressivePtOutput,
};
pub use brush::{pick_terrain_uv, pick_terrain_uv_on_surface, BrushGizmo, BrushOverlay};
pub use camera::OrbitCamera;
pub use clipmap::{
    plan_resident_tiles, projected_error_px, ClipmapConfig, ClipmapPresentPlan, ClipmapRingDraw,
    ClipmapRingLevel, ResidentTileSelection, ViewportTilePlan, WorldGridConfig,
};
pub use frame_graph::{FrameGraph, FrameSchedule, PassKind};
pub use gpu_timing::GpuTimings;
pub use grid::TerrainGrid;
pub use guides::{GuideOverlay, GuideState};
pub use height_gpu::HeightGpu;
pub use overhang::OverhangOverlay;
pub use path_tracer::{PathTraceUniforms, PathTracer};
pub use render_quality::{
    QualityPreset, RenderQualityConfig, ViewportQualityManager, ViewportRendererMode,
};
pub use scene_versions::{
    CameraChangeThresholds, CameraSnapshot, InvalidationReason, SceneVersionRegistry, SceneVersions,
};
pub use terra_core::EditorRefinementState;
pub use vegetation::VegetationOverlay;

use bytemuck::{Pod, Zeroable};
use terra_core::heightfield::Heightfield;
use terra_core::layer::MaterialsParams;
use terra_core::mask::MaskField;
use terra_core::tiling::SampleRect;
use terra_core::{FieldId, NormalizedRect, TerrainPyramid};
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
    config: wgpu::SurfaceConfiguration,
    pub pipeline: wgpu::RenderPipeline,
    /// Grid edge LineList overlay for the Wireframe display aid.
    pub wireframe_pipeline: wgpu::RenderPipeline,
    /// Single frame-uniform buffer for the world-fixed terrain grid.
    pub uniform_buf: wgpu::Buffer,
    /// Transparent sea-level surface rendered after opaque terrain.
    pub ocean_pipeline: wgpu::RenderPipeline,
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
    pub size: winit::dpi::PhysicalSize<u32>,
    /// Last CPU→GPU height upload microseconds.
    pub last_upload_us: u64,
    /// Last resolved GPU pass timings (0 when TIMESTAMP_QUERY unavailable).
    pub last_gpu_timings: GpuTimings,
    /// Terrain mesh resolution drawn last frame (profiler).
    pub last_grid_resolution: u32,
    /// Desired visible pages resident at the requested pyramid level.
    pub last_tile_plan_exact: usize,
    /// Desired visible pages currently covered by a coarser resident ancestor.
    pub last_tile_plan_fallback: usize,
    /// Desired visible pages with no resident ancestor yet.
    pub last_tile_plan_missing: usize,
    /// After first height present, leave orbit target alone so uploads don't fight the user.
    camera_framed: bool,
    /// Sculpt / mask brush ring drawn on the height surface.
    pub brush: BrushOverlay,
    /// World grid + AABB bounds line guides.
    pub guides: GuideOverlay,
    /// Phase J dual-height overhang / cave roof proxy (opt-in layers).
    pub overhang: OverhangOverlay,
    /// Instanced vegetation driven by the evaluated vegetation-density field.
    pub vegetation: VegetationOverlay,
    /// Presentation lighting (does not affect height data).
    pub lighting: EnvironmentLighting,
    /// Active ocean height; None disables the water surface.
    ocean_level: Option<f32>,
    /// Strength of painted biome placement colour overlay (0 = off).
    biome_tint_strength: f32,
    /// Display aids / analysis shading from the viewport chrome.
    display_aids: ViewportDisplayAids,
    /// Progressive stochastic lighting, temporal reprojection, and denoising.
    progressive: progressive::ProgressiveRenderer,
    /// GPU heightfield path tracer (progressive ray-traced modes).
    path_tracer: PathTracer,
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
    use_tile_stream: bool,
    tile_stream_tile_size: f32,
    tile_stream_halo: f32,
    tile_stream_max_pages: f32,
    tile_stream_level: f32,
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

    let mut limits = wgpu::Limits::default();
    let adapter_limits = adapter.limits();
    // Path tracer uses 4 storage textures; request headroom when the adapter allows it.
    limits.max_storage_textures_per_shader_stage = adapter_limits
        .max_storage_textures_per_shader_stage
        .max(4)
        .min(16);
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
                required_features: gpu_timing::requested_timestamp_features(&adapter),
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
    Ok((
        GpuContext {
            device,
            queue,
            surface_format: format,
        },
        SurfaceTarget {
            surface,
            config,
            size,
        },
    ))
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
            Some(target.surface),
            target.config,
            target.size,
        )
    }

    /// Current swapchain dimensions in physical pixels (each always ≥ 1).
    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Swapchain color format the surface was configured with.
    pub fn format(&self) -> wgpu::TextureFormat {
        self.config.format
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
        Self::init(ctx.device.clone(), ctx.queue.clone(), None, config, size)
    }

    /// Shared constructor tail: every device-only resource, after surface and
    /// adapter negotiation. `config` doubles as the render-target description when
    /// `surface` is `None` — `format` bakes the color-target pipelines and
    /// `width`/`height` size the depth/progressive/path-tracer targets; the
    /// present-mode and alpha-mode fields are inert without a surface.
    fn init(
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface: Option<wgpu::Surface<'static>>,
        config: wgpu::SurfaceConfiguration,
        size: winit::dpi::PhysicalSize<u32>,
    ) -> Self {
        let format = config.format;

        log::info!("terra-render: compiling terrain shader/pipelines…");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/terrain.wgsl").into()),
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
            ],
        });

        let heights = HeightGpu::new(&device, 256);
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

        let (tile_atlas_texture, tile_atlas_view, page_table_buf) =
            create_dummy_tile_stream(&device);
        let shadow_map = shadows::ShadowMap::new(&device, heights.display_height_view(), true);
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
            &shadow_map.view,
            &shadow_map.comparison_sampler,
        );

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain-pipe"),
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
            cache: None,
        });

        let ocean_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ocean-pipe"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_ocean"),
                buffers: &[TerrainGrid::vertex_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_ocean"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
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
            cache: None,
        });

        let wireframe_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("wireframe-pipe"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[TerrainGrid::vertex_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_wireframe"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: wgpu::DepthBiasState {
                    constant: -4,
                    slope_scale: -2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
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
                    &shadow_map.view,
                    &shadow_map.comparison_sampler,
                )
            })
            .collect();
        let camera = OrbitCamera::default();
        let brush = BrushOverlay::new(&device, format);
        let guides = GuideOverlay::new(&device, format);
        let overhang = OverhangOverlay::new(&device, format);
        let vegetation = VegetationOverlay::new(&device, format);
        let progressive =
            progressive::ProgressiveRenderer::new(&device, config.width, config.height, format);
        let quality = ViewportQualityManager::default();
        let initial_internal_scale = quality.internal_scale;
        let mut path_tracer = PathTracer::new(
            &device,
            &queue,
            config.width,
            config.height,
            initial_internal_scale,
        );
        let adaptive = AdaptiveSamplingState::new(config.width, config.height);
        path_tracer.upload_sample_mask(&queue, &adaptive.prepare_all_active_mask());

        Self {
            surface,
            device,
            queue,
            config,
            pipeline,
            wireframe_pipeline,
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
            size,
            ocean_pipeline,
            last_upload_us: 0,
            last_gpu_timings: GpuTimings::default(),
            last_grid_resolution: 0,
            last_tile_plan_exact: 0,
            last_tile_plan_fallback: 0,
            last_tile_plan_missing: 0,
            camera_framed: false,
            brush,
            guides,
            overhang,
            vegetation,
            lighting: EnvironmentLighting::default(),
            ocean_level: None,
            biome_tint_strength: 0.0,
            display_aids: ViewportDisplayAids::default(),
            progressive,
            path_tracer,
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
            use_tile_stream: false,
            tile_stream_tile_size: 256.0,
            tile_stream_halo: 2.0,
            tile_stream_max_pages: 1.0,
            tile_stream_level: 0.0,
        }
    }

    // (pipeline compile complete — logged via terrain shader message above)

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
        self.progressive
            .resize(&self.device, self.config.width, self.config.height);
        self.path_tracer.resize(
            &self.device,
            &self.queue,
            self.config.width,
            self.config.height,
            self.quality.internal_scale,
        );
        self.adaptive.resize(self.config.width, self.config.height);
        let mask = self.adaptive.prepare_all_active_mask();
        self.path_tracer.upload_sample_mask(&self.queue, &mask);
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
        self.progressive.accumulation_frame_index()
    }

    pub fn progressive_last_invalidation(&self) -> InvalidationReason {
        self.progressive.last_invalidation_reason()
    }

    pub fn scene_versions_snapshot(&self) -> SceneVersions {
        self.scene_versions.versions
    }

    pub fn notify_invalidation(&mut self, reason: InvalidationReason) {
        self.scene_versions.notify(reason);
        if reason.resets_accumulation() {
            self.progressive.invalidate_with_reason(reason);
            self.path_tracer.invalidate(&self.queue);
            self.adaptive.reactivate_all();
            let mask = self.adaptive.prepare_all_active_mask();
            self.path_tracer.upload_sample_mask(&self.queue, &mask);
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
        self.progressive
            .set_enabled(matches!(backend, PresentationBackendId::ProgressivePt));
        self.notify_invalidation(InvalidationReason::RenderModeChanged);
    }

    /// Active presentation backend for the current mode.
    pub fn presentation_backend(&self) -> PresentationBackendId {
        PresentationBackendId::from_mode(self.quality.config.mode)
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
        self.upload_aux_maps_ex(
            materials, wetness, vegetation, None, None, None, None, None, None,
        );
    }

    /// Upload materials/wetness/vegetation plus optional climate R32Float aux maps and flow.
    pub fn upload_aux_maps_ex(
        &mut self,
        materials: Option<&MaskField>,
        wetness: Option<&MaskField>,
        vegetation: Option<&MaskField>,
        temperature: Option<&MaskField>,
        rainfall: Option<&MaskField>,
        snow: Option<&MaskField>,
        soil_moisture: Option<&MaskField>,
        biomes: Option<&MaskField>,
        flow: Option<&MaskField>,
    ) {
        self.heights.upload_aux_maps_ex(
            &self.device,
            &self.queue,
            materials,
            wetness,
            vegetation,
            temperature,
            rainfall,
            snow,
            soil_moisture,
            biomes,
            flow,
        );
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
    pub fn present_gpu_height(
        &mut self,
        src: &wgpu::Texture,
        width: u32,
        height: u32,
        world_size: (f32, f32),
        height_range: (f32, f32),
        dx: f32,
        dz: f32,
    ) {
        self.present_gpu_height_region(src, width, height, world_size, height_range, dx, dz, None);
    }

    pub fn present_gpu_height_region(
        &mut self,
        src: &wgpu::Texture,
        width: u32,
        height: u32,
        world_size: (f32, f32),
        height_range: (f32, f32),
        dx: f32,
        dz: f32,
        region: Option<SampleRect>,
    ) {
        profiling::scope!("present_gpu_height");
        let t0 = std::time::Instant::now();
        self.heights.copy_from_texture_region_and_swap(
            &self.device,
            &self.queue,
            src,
            width,
            height,
            world_size,
            height_range,
            dx,
            dz,
            region,
        );
        self.finish_height_present(t0);
    }

    /// Bind a GPU engine height texture directly when formats match (full field).
    /// Partial [`SampleRect`] updates still copy through the double-buffer path.
    pub fn present_gpu_height_shared(
        &mut self,
        src: &wgpu::Texture,
        src_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        world_size: (f32, f32),
        height_range: (f32, f32),
        dx: f32,
        dz: f32,
        region: Option<SampleRect>,
    ) {
        if region.is_some() {
            self.present_gpu_height_region(
                src,
                width,
                height,
                world_size,
                height_range,
                dx,
                dz,
                region,
            );
            return;
        }
        profiling::scope!("present_gpu_height_shared");
        let t0 = std::time::Instant::now();
        self.heights.present_shared_height(
            &self.device,
            &self.queue,
            src_view,
            width,
            height,
            world_size,
            height_range,
            dx,
            dz,
        );
        self.finish_height_present(t0);
    }

    fn finish_height_present(&mut self, t0: std::time::Instant) {
        self.recreate_bind_group();
        let extent = self.heights.world_size.0.max(self.heights.world_size.1);
        // Match mesh density to the height texture as closely as device buffer
        // limits allow (256 MiB default). Height/normal textures still carry Full
        // 1 m detail; mesh is capped so create_buffer does not exceed max_buffer_size.
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
        // Frame once (or after explicit reset). Continuous retargeting fights orbit/pan.
        if !self.camera_framed {
            self.frame_camera_to_terrain();
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
        self.camera.target = glam::Vec3::new(
            self.heights.world_size.0 * 0.5,
            (min_h + max_h) * 0.5,
            self.heights.world_size.1 * 0.5,
        );
        self.camera.distance = extent * 1.1;
        self.camera_framed = true;
    }

    pub fn request_camera_reframe(&mut self) {
        self.camera_framed = false;
    }

    /// Move the orbit target to a normalized location in the terrain footprint.
    pub fn focus_camera_uv(&mut self, u: f32, v: f32) {
        self.camera.target.x = u.clamp(0.0, 1.0) * self.heights.world_size.0;
        self.camera.target.z = v.clamp(0.0, 1.0) * self.heights.world_size.1;
        self.camera.clamp_to_world(self.heights.world_size);
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

    /// Bind a live tile atlas + page table so the terrain shader can sample
    /// resident pages (falling back to the monolithic height texture on misses).
    ///
    /// `atlas_view` / `page_table` must remain valid while streaming is enabled
    /// (typically owned by `GpuTileAtlas` in the app).
    pub fn set_tile_stream_resources(
        &mut self,
        atlas_view: wgpu::TextureView,
        page_table: wgpu::Buffer,
        tile_size: u32,
        halo: u32,
        max_pages: u32,
        level: u8,
        enable: bool,
    ) {
        self.tile_atlas_view = atlas_view;
        self.page_table_buf = page_table;
        self.tile_stream_tile_size = tile_size.max(1) as f32;
        self.tile_stream_halo = halo as f32;
        self.tile_stream_max_pages = max_pages.max(1) as f32;
        self.tile_stream_level = level as f32;
        // Streaming samples resident pages; the shader falls back to the monolithic
        // height texture on page misses so presentation stays continuous.
        self.use_tile_stream = enable;
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

    pub fn set_shadows_enabled(&mut self, enable: bool) {
        self.shadow_map.set_enabled(enable);
        self.notify_invalidation(InvalidationReason::LightingChanged);
    }

    pub fn set_brush_gizmo(&mut self, gizmo: Option<BrushGizmo>) {
        self.brush.set_gizmo(gizmo);
    }

    /// Update brush ring mesh to match current gizmo + optional ring height samples.
    pub fn sync_brush_geometry(&mut self, ring_heights: Option<&[f32]>) {
        self.brush
            .sync_geometry(&self.queue, self.heights.world_size, ring_heights);
    }

    /// Upload or clear the Phase J overhang / cave roof proxy mesh.
    pub fn sync_overhang_mesh(&mut self, mesh: Option<&terra_core::volumetric::OverhangMesh>) {
        match mesh {
            Some(m) if !m.is_empty() => self.overhang.upload_mesh(&self.device, m),
            _ => self.overhang.clear(),
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
        self.vegetation.sync(
            &self.device,
            height,
            density,
            scale_min,
            scale_max,
            yaw_variation_deg,
        );
        self.notify_invalidation(InvalidationReason::GeometryChanged);
    }

    pub fn set_ocean_level(&mut self, level: Option<f32>) {
        let level = level.filter(|value| value.is_finite());
        if self.ocean_level != level {
            self.ocean_level = level;
            self.notify_invalidation(InvalidationReason::GeometryChanged);
        }
    }

    /// Clear viewport GPU state that belongs to the previous document.
    pub fn reset_project_state(&mut self, world_size: (f32, f32), ocean_level: Option<f32>) {
        self.use_tile_stream = false;
        self.heights
            .reset_project_state(&self.device, &self.queue, world_size);
        // Empty vegetation overlay (no density → clear instances).
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
        self.vegetation
            .sync(&self.device, &blank, None, 1.0, 1.0, 0.0);
        self.overhang.clear();
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
        self.guides.set_state(GuideState {
            grid: aids.grid,
            bounds: aids.world_bounds,
        });
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
        let backend = PresentationBackendId::from_mode(self.quality.config.mode);
        if matches!(backend, PresentationBackendId::ProgressivePt) {
            self.progressive.set_enabled(enabled);
        } else {
            self.progressive.set_enabled(false);
        }
    }

    pub fn progressive_samples(&self) -> u32 {
        self.progressive.samples()
    }

    /// Reconcile the full-world footprint with the streamed terrain pyramid.
    /// When tile-stream sampling is enabled, missing pages fall back to the
    /// monolithic height texture; these counts describe streamed coverage.
    pub fn update_visible_tile_plan(&mut self, pyramid: &TerrainPyramid) {
        let visible = NormalizedRect::new(0.0, 0.0, 1.0, 1.0).expect("unit world rect");
        let plan = plan_resident_tiles(
            pyramid,
            None,
            &FieldId::Height,
            pyramid.max_level(),
            visible,
        );
        self.last_tile_plan_exact = plan.exact_tiles;
        self.last_tile_plan_fallback = plan.fallback_tiles;
        self.last_tile_plan_missing = plan.missing_tiles;
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

    /// Record and submit one full frame (shadow, backend, overlays, post) into `view`.
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
        self.progressive
            .set_max_samples(self.quality.config.max_accumulated_spp);
        self.progressive.set_history_cap(self.quality.history_cap);

        let internal_scale = self.quality.internal_scale;
        if (self.last_internal_scale - internal_scale).abs() > 1e-4 {
            self.path_tracer
                .resize(&self.device, &self.queue, width, height, internal_scale);
            let mask = self.adaptive.prepare_all_active_mask();
            self.path_tracer.upload_sample_mask(&self.queue, &mask);
            self.notify_invalidation(InvalidationReason::ViewportResized);
            self.last_internal_scale = internal_scale;
        }

        let backend = PresentationBackendId::from_mode(self.quality.config.mode);
        // Raster cast shadows are driven by the lighting shadow-strength control;
        // 0 keeps the depth pass off, matching the historical "no shadows" look.
        self.shadow_map
            .set_enabled(self.lighting.shadow_strength > 1e-4);
        let shadows_for_schedule =
            self.shadow_map.enabled() && matches!(backend, PresentationBackendId::RasterLit);
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
        if self.frame_graph.schedule.progressive_post && !self.progressive.enabled() {
            self.progressive.set_enabled(true);
        } else if !self.frame_graph.schedule.progressive_post && self.progressive.enabled() {
            self.progressive.set_enabled(false);
        }

        let view_proj = self.camera.view_proj(aspect);
        let (min_h, max_h) = self.heights.height_range;
        let (tw, th) = self.heights.tex_size;
        self.camera.clamp_to_world(self.heights.world_size);
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
        if self.progressive.signature_changed(lighting_signature) {
            self.notify_invalidation(InvalidationReason::LightingChanged);
        }
        self.progressive.prepare_signature(lighting_signature);

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
        let eye = self.camera.eye();
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
                self.progressive.samples() as f32,
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
                self.tile_stream_level,
                1.25,
            ],
            raster: [
                self.lighting.ambient_strength,
                self.lighting.shadow_strength,
                self.lighting.fog_strength,
                0.0,
            ],
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
            timer.begin_frame();
        }
        self.staging.begin_frame();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("terrain-enc"),
            });
        self.brush.upload_view_proj(&self.queue, view_proj);
        self.guides.upload_view_proj(&self.queue, view_proj);
        self.guides.sync_geometry(
            &self.queue,
            self.heights.world_size,
            self.heights.height_range,
            (self.camera.target.x, self.camera.target.z),
        );
        self.overhang
            .upload_view_proj(&self.queue, view_proj, self.lighting.light_dir);
        self.vegetation
            .upload_view_proj(&self.queue, view_proj, self.lighting.light_dir);

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
                    self.path_tracer.upload_sample_mask(&self.queue, &mask);

                    let view_mat = glam::Mat4::look_at_rh(eye, self.camera.target, glam::Vec3::Y);
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
                    self.path_tracer.dispatch(
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
                    self.camera.target.x,
                    self.camera.target.z,
                    world_x,
                    world_z,
                    tw,
                    th,
                );

                // Upload all per-draw uniforms before the pass (queue writes are
                // illegal while a buffer is bound in an active render pass).
                if present.use_single_grid {
                    let mut uniforms = base_uniforms;
                    uniforms.clipmap = [
                        0.0,
                        0.0,
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
                            0.0,
                            0.0,
                            present.fallback_spacing,
                            self.grid.resolution as f32,
                        ];
                        uniforms.viz[3] = present.fallback_exclude_half_extent;
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
                            draw.origin_x,
                            draw.origin_z,
                            draw.spacing,
                            draw.grid_size as f32,
                        ];
                        uniforms.viz[3] = draw.exclude_half_extent;
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
                    pass.set_pipeline(&self.pipeline);

                    if present.use_single_grid {
                        pass.set_bind_group(0, &self.bind_group, &[]);
                        pass.set_vertex_buffer(0, self.grid.vertex_buf.slice(..));
                        pass.set_index_buffer(
                            self.grid.index_buf.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        pass.draw_indexed(0..self.grid.index_count, 0, 0..1);
                    } else {
                        if present.draw_fallback {
                            pass.set_bind_group(0, &self.bind_group, &[]);
                            pass.set_vertex_buffer(0, self.grid.vertex_buf.slice(..));
                            pass.set_index_buffer(
                                self.grid.index_buf.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            pass.draw_indexed(0..self.grid.index_count, 0, 0..1);
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
                            pass.draw_indexed(0..ring_grid.index_count, 0, 0..1);
                        }
                    }
                }

                if self.frame_graph.schedule.overlays {
                    self.frame_graph.mark(PassKind::Overlays);
                    // Hole-free uniforms for ocean / overlays (shared main buffer).
                    {
                        let mut uniforms = base_uniforms;
                        uniforms.clipmap =
                            [0.0, 0.0, fallback_spacing, self.grid.resolution as f32];
                        uniforms.viz[3] = 0.0;
                        self.queue.write_buffer(
                            &self.uniform_buf,
                            0,
                            bytemuck::bytes_of(&uniforms),
                        );
                    }
                    {
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("raster-lit-overlays"),
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
                        if self.ocean_level.is_some() {
                            pass.set_pipeline(&self.ocean_pipeline);
                            pass.set_bind_group(0, &self.bind_group, &[]);
                            pass.set_vertex_buffer(0, self.grid.vertex_buf.slice(..));
                            pass.set_index_buffer(
                                self.grid.index_buf.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            pass.draw_indexed(0..self.grid.surface_index_count, 0, 0..1);
                        }
                        if self.display_aids.wireframe {
                            pass.set_pipeline(&self.wireframe_pipeline);
                            pass.set_bind_group(0, &self.bind_group, &[]);
                            pass.set_vertex_buffer(0, self.grid.vertex_buf.slice(..));
                            pass.set_index_buffer(
                                self.grid.edge_index_buf.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            pass.draw_indexed(0..self.grid.edge_index_count, 0, 0..1);
                        }
                        self.vegetation.draw(&mut pass);
                        self.overhang.draw(&mut pass);
                        self.brush.draw(&mut pass);
                        self.guides.draw(&mut pass);
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
            ProgressivePostPipeline::resolve_hdr(
                &mut self.progressive,
                &self.device,
                &self.queue,
                &mut encoder,
                HdrFrame {
                    color: self.path_tracer.radiance_view(),
                    width,
                    height,
                },
                GBufferViews {
                    depth: self.path_tracer.depth_view(),
                    normal: Some(self.path_tracer.normal_view()),
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
            if self.frame_graph.schedule.overlays {
                self.frame_graph.mark(PassKind::Overlays);
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("progressive-overlay-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                self.brush.draw(&mut pass);
                self.guides.draw(&mut pass);
            }
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
        let samples = self.progressive.samples() as f32;
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
) -> (wgpu::Texture, wgpu::TextureView, wgpu::Buffer) {
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
        size: 48,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    (texture, view, page_table)
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
        let source = include_str!("shaders/terrain.wgsl");
        let module = naga::front::wgsl::parse_str(source)
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
}
