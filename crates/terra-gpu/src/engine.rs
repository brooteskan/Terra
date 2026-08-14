//! GPU layer-stack preview engine: texture caches, ping-pong sims, no interactive readback.
//!
//! This module is intentionally a single compilation unit for the wgpu façade
//! (`GpuTerrainEngine`). Prefer extracting pipeline families (noise/blend, erosion,
//! hydro) into sibling files when touching large regions — keep shader
//! `include_str!` paths stable relative to this file.

//! Interactive hard rules (WC): no UI-thread height readback, no mesh rebuild,
//! prefer fully GPU stacks, never present an incomplete prefix as finished Draft.

use crate::effect_filter::resolve_effect_mode;
use crate::graph::{
    compile_gpu_graph, expand_dirty_rect, gpu_blend_mode, gpu_plan_for_layer, layer_gpu_supported,
    GpuComputeGraph, GpuKernel,
};
use crate::{readback_f32, GpuError};
use bytemuck::{Pod, Zeroable};
use std::collections::{HashMap, HashSet};
use terra_core::analyze::{apply_transport_model, clamp_timestep_cfl};
use terra_core::eval::PreviewQuality;
use terra_core::fields::FieldId;
use terra_core::heightfield::{Heightfield, HeightfieldMetrics, TileId};
use terra_core::layer::{
    BlendMode, EffectFilterParams, FractalNoiseType, Layer, LayerId, LayerKind, LayerStack,
    NoiseParams, SculptParams,
};
use terra_core::mask::{MaskAsset, MaskSource};
use terra_core::tiling::{SampleRect, TileScheduler};

/// Small resident texture extent used while no project is active.
/// `ensure_size` restores the next evaluation's document dimensions.
const PROJECT_RESET_TEXTURE_EXTENT: u32 = 8;
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NoiseU {
    width: u32,
    height: u32,
    world_x: f32,
    world_z: f32,
    seed: u32,
    octaves: u32,
    frequency: f32,
    amplitude: f32,
    lacunarity: f32,
    persistence: f32,
    offset_x: f32,
    offset_z: f32,
    remap_min: f32,
    remap_max: f32,
    noise_type: u32,
    mode: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlendU {
    width: u32,
    height: u32,
    opacity: f32,
    mode: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FillU {
    width: u32,
    height: u32,
    value: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ThermalU {
    width: u32,
    height: u32,
    dx: f32,
    talus: f32,
    strength: f32,
    _p2: f32,
    _p3: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HydraulicU {
    width: u32,
    height: u32,
    timestep: f32,
    rainfall: f32,
    evaporation: f32,
    erosion: f32,
    deposition: f32,
    capacity: f32,
    fan_boost: f32,
    floodplain_bias: f32,
    dx: f32,
    incision_bias: f32,
    bedrock_k: f32,
    sediment_k: f32,
    layered: f32,
    _pad1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlurU {
    width: u32,
    height: u32,
    radius: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TerraceU {
    width: u32,
    height: u32,
    levels: u32,
    sharpness: f32,
    min_h: f32,
    max_h: f32,
    _p0: f32,
    _p1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RampU {
    width: u32,
    height: u32,
    world_x: f32,
    world_z: f32,
    height_min: f32,
    height_max: f32,
    direction: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CopyU {
    width: u32,
    height: u32,
    _p0: f32,
    _p1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShapeU {
    width: u32,
    height: u32,
    world_x: f32,
    world_z: f32,
    seed: u32,
    octaves: u32,
    frequency: f32,
    amplitude: f32,
    lacunarity: f32,
    persistence: f32,
    offset_x: f32,
    offset_z: f32,
    ridge_sharpness: f32,
    range_angle: f32,
    range_width: f32,
    wave_frequency: f32,
    asymmetry: f32,
    depth: f32,
    canyon_width: f32,
    meander: f32,
    shape_mode: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RiverAccumU {
    width: u32,
    height: u32,
    _p0: f32,
    _p1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RiverCarveU {
    width: u32,
    height: u32,
    threshold: f32,
    depth: f32,
    channel_width: f32,
    bank_smooth: f32,
    max_radius: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EffectFilterU {
    width: u32,
    height: u32,
    world_x: f32,
    world_z: f32,
    mode: u32,
    radius: u32,
    iterations: u32,
    seed: u32,
    strength: f32,
    amount: f32,
    frequency: f32,
    sea_level: f32,
    beach_width: f32,
    slope_min: f32,
    slope_max: f32,
    rock_hardness: f32,
    terrace_height: f32,
    terrace_offset: f32,
    rotation_deg: f32,
    anisotropy: f32,
    warp_strength: f32,
    warp_frequency: f32,
    dx: f32,
    invert: f32,
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaskBakeU {
    width: u32,
    height: u32,
    mode: u32,
    dz: f32,
    dx: f32,
    value: f32,
    range_min: f32,
    range_max: f32,
    invert: f32,
    strength: f32,
    frequency: f32,
    seed: f32,
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
}

#[derive(Clone, Copy)]
enum TexSlot {
    Ping,
    Pong,
    Layer,
    MaskOnes,
    Hardness,
    WaterA,
    WaterB,
    SedA,
    SedB,
    Rainfall,
    LooseSediment,
    Cache(LayerId),
    /// Pre-blend layer contribution (noise/shape/flat), reusable when only upstream changed.
    Contrib(LayerId),
}

struct HeightTex {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

impl HeightTex {
    fn new(device: &wgpu::Device, label: &str, width: u32, height: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture,
            view,
            width,
            height,
        }
    }
}

/// RGBA float texture for hydraulic outflow fluxes (L,R,D,U).
struct RgbaTex {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl RgbaTex {
    fn new(device: &wgpu::Device, label: &str, width: u32, height: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            _texture: texture,
            view,
        }
    }
}

struct Pipe {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

fn make_pipe(device: &wgpu::Device, label: &str, wgsl: &str, bgl: wgpu::BindGroupLayout) -> Pipe {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pl),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    Pipe { pipeline, bgl }
}

fn storage_write_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::R32Float,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn storage_write_rgba_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba32Float,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn tex_read_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Ring of small uniform buffers so many dispatches can share one submit.
struct UniformPool {
    buffers: Vec<wgpu::Buffer>,
    next: usize,
}

impl UniformPool {
    const SLOT_SIZE: u64 = 256;

    fn new(device: &wgpu::Device, capacity: usize) -> Self {
        let mut buffers = Vec::with_capacity(capacity);
        for i in 0..capacity {
            buffers.push(Self::make_slot(device, i));
        }
        Self { buffers, next: 0 }
    }

    fn make_slot(device: &wgpu::Device, index: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("gpu-engine-u-{index}")),
            size: Self::SLOT_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn reset(&mut self) {
        self.next = 0;
    }

    fn write<T: Pod>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &T,
    ) -> wgpu::Buffer {
        debug_assert!(
            std::mem::size_of::<T>() as u64 <= Self::SLOT_SIZE,
            "uniform larger than pool slot"
        );
        if self.next >= self.buffers.len() {
            let start = self.buffers.len();
            let grow = self.buffers.len().max(8);
            for i in start..start + grow {
                self.buffers.push(Self::make_slot(device, i));
            }
        }
        let buf = self.buffers[self.next].clone();
        self.next += 1;
        queue.write_buffer(&buf, 0, bytemuck::bytes_of(data));
        buf
    }
}

/// Generators whose height field does not depend on the composed input below them.
fn layer_input_independent(kind: &LayerKind) -> bool {
    matches!(
        kind,
        LayerKind::Flat(_)
            | LayerKind::Ramp(_)
            | LayerKind::NoiseValue(_)
            | LayerKind::NoisePerlin(_)
            | LayerKind::Fbm(_)
            | LayerKind::Ridged(_)
            | LayerKind::Mountains(_)
            | LayerKind::Dunes(_)
            | LayerKind::Canyons(_)
            | LayerKind::DomainWarp(_)
    )
}

/// Whether height alone completely represents the enabled prefix before `resume_index`.
///
/// CPU suffix evaluation also observes auxiliary fields and named layer outputs. Until
/// `GpuEvalResult` carries those checkpoints, any prefix that publishes them must restart
/// on the CPU from layer zero rather than borrowing stale state from another generation.
fn cpu_resume_prefix_is_height_only(layers: &[&Layer], resume_index: usize) -> bool {
    layers.iter().take(resume_index).all(|layer| {
        !layer.common.enabled
            || (layer.common.outputs.is_empty()
                && layer
                    .kind
                    .produced_fields()
                    .into_iter()
                    .all(|field| field == FieldId::Height))
    })
}

/// Result of a GPU preview evaluation.
pub struct GpuEvalResult {
    pub width: u32,
    pub height: u32,
    pub world_size: (f32, f32),
    pub height_range: (f32, f32),
    pub fully_gpu: bool,
    pub cpu: Option<Heightfield>,
    /// First flattened layer that must resume on the CPU. When this is `Some(n)`,
    /// `cpu` is the height entering layer `n`; `Some(0)` is a full-CPU restart seed.
    pub resume_cpu_from: Option<usize>,
    /// True when the evaluate loop ran (filters may have been applied). False on seed failure.
    pub did_eval: bool,
}

/// GPU stack evaluator for interactive preview.
pub struct GpuTerrainEngine {
    fill: Pipe,
    noise: Pipe,
    blend: Pipe,
    copy: Pipe,
    thermal: Pipe,
    thermal_apply: Pipe,
    hydraulic_outflow: Pipe,
    hydraulic: Pipe,
    blur: Pipe,
    terrace: Pipe,
    ramp: Pipe,
    shapes: Pipe,
    river_accum: Pipe,
    river_carve: Pipe,
    effect_filter: Pipe,
    mask_bake: Pipe,
    uniform_pool: UniformPool,
    ping: HeightTex,
    pong: HeightTex,
    layer_tex: HeightTex,
    mask_ones: HeightTex,
    hardness: HeightTex,
    water_a: HeightTex,
    water_b: HeightTex,
    delta: HeightTex,
    sed_a: HeightTex,
    sed_b: HeightTex,
    /// Spatial rainfall multiplier (1 = uniform). Stays on GPU across hydraulic iters.
    rainfall: HeightTex,
    /// Loose sediment thickness for layered erodibility (meters).
    loose_sediment: HeightTex,
    outflow: RgbaTex,
    layer_cache: HashMap<LayerId, HeightTex>,
    /// Pre-blend generator output, keyed by layer id.
    layer_contrib: HashMap<LayerId, HeightTex>,
    dirty: HashSet<LayerId>,
    metrics: HeightfieldMetrics,
    approx_range: (f32, f32),
    /// Index of texture holding current composed height: 0=ping, 1=pong.
    current: u8,
    /// Spatial dirty tiles for region present / normal recompute (Wave D).
    tile_sched: TileScheduler,
    /// Optional texel-space edit bounds supplied by interactive painting.
    last_dirty_rect: Option<(u32, u32, u32, u32)>,
    last_quality: Option<PreviewQuality>,
    /// Maximum thermal/hydraulic iterations submitted in one interactive tick.
    pub max_sim_iters_per_tick: u32,
    /// Last compiled GPU pass graph for the evaluated stack.
    pub last_graph: GpuComputeGraph,
}

impl GpuTerrainEngine {
    pub fn new(device: &wgpu::Device, initial: u32) -> Self {
        let w = initial.max(PROJECT_RESET_TEXTURE_EXTENT);
        let fill_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fill-bgl"),
            entries: &[uniform_entry(0), storage_write_entry(1)],
        });
        let noise_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("noise-bgl"),
            entries: &[uniform_entry(0), storage_write_entry(1)],
        });
        let blend_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blend-bgl"),
            entries: &[
                uniform_entry(0),
                tex_read_entry(1),
                tex_read_entry(2),
                tex_read_entry(3),
                storage_write_entry(4),
            ],
        });
        let copy_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("copy-bgl"),
            entries: &[uniform_entry(0), tex_read_entry(1), storage_write_entry(2)],
        });
        let thermal_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("thermal-bgl"),
            entries: &[
                uniform_entry(0),
                tex_read_entry(1),
                storage_write_entry(2),
                tex_read_entry(3),
            ],
        });
        let thermal_apply_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("thermal-apply-bgl"),
            entries: &[
                uniform_entry(0),
                tex_read_entry(1),
                tex_read_entry(2),
                storage_write_entry(3),
            ],
        });
        let hydraulic_outflow_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("hydraulic-outflow-bgl"),
                entries: &[
                    uniform_entry(0),
                    tex_read_entry(1),
                    tex_read_entry(2),
                    storage_write_rgba_entry(3),
                    tex_read_entry(4),
                ],
            });
        let hydraulic_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hydraulic-bgl"),
            entries: &[
                uniform_entry(0),
                tex_read_entry(1),
                tex_read_entry(2),
                tex_read_entry(3),
                tex_read_entry(4),
                storage_write_entry(5),
                storage_write_entry(6),
                storage_write_entry(7),
                tex_read_entry(8),
                tex_read_entry(9),
                tex_read_entry(10),
            ],
        });
        let blur_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blur-bgl"),
            entries: &[uniform_entry(0), tex_read_entry(1), storage_write_entry(2)],
        });
        let terrace_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrace-bgl"),
            entries: &[uniform_entry(0), tex_read_entry(1), storage_write_entry(2)],
        });
        let ramp_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ramp-bgl"),
            entries: &[uniform_entry(0), storage_write_entry(1)],
        });
        let shapes_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shapes-bgl"),
            entries: &[uniform_entry(0), storage_write_entry(1)],
        });
        let river_accum_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("river-accum-bgl"),
            entries: &[
                uniform_entry(0),
                tex_read_entry(1),
                tex_read_entry(2),
                storage_write_entry(3),
            ],
        });
        let river_carve_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("river-carve-bgl"),
            entries: &[
                uniform_entry(0),
                tex_read_entry(1),
                tex_read_entry(2),
                storage_write_entry(3),
            ],
        });

        let fill = make_pipe(device, "fill", include_str!("shaders/fill.wgsl"), fill_bgl);
        let noise = make_pipe(
            device,
            "noise",
            include_str!("shaders/noise.wgsl"),
            noise_bgl,
        );
        let blend = make_pipe(
            device,
            "blend",
            include_str!("shaders/blend.wgsl"),
            blend_bgl,
        );
        let copy = make_pipe(device, "copy", include_str!("shaders/copy.wgsl"), copy_bgl);
        let thermal = make_pipe(
            device,
            "thermal",
            include_str!("shaders/thermal_tex.wgsl"),
            thermal_bgl,
        );
        let thermal_apply = make_pipe(
            device,
            "thermal-apply",
            include_str!("shaders/thermal_apply.wgsl"),
            thermal_apply_bgl,
        );
        let hydraulic_outflow = make_pipe(
            device,
            "hydraulic-outflow",
            include_str!("shaders/hydraulic_outflow.wgsl"),
            hydraulic_outflow_bgl,
        );
        let hydraulic = make_pipe(
            device,
            "hydraulic",
            include_str!("shaders/hydraulic_tex.wgsl"),
            hydraulic_bgl,
        );
        let blur = make_pipe(device, "blur", include_str!("shaders/blur.wgsl"), blur_bgl);
        let terrace = make_pipe(
            device,
            "terrace",
            include_str!("shaders/terrace.wgsl"),
            terrace_bgl,
        );
        let ramp = make_pipe(device, "ramp", include_str!("shaders/ramp.wgsl"), ramp_bgl);
        let shapes = make_pipe(
            device,
            "shapes",
            include_str!("shaders/shapes.wgsl"),
            shapes_bgl,
        );
        let river_accum = make_pipe(
            device,
            "river-accum",
            include_str!("shaders/river_accum.wgsl"),
            river_accum_bgl,
        );
        let river_carve = make_pipe(
            device,
            "river-carve",
            include_str!("shaders/river_carve.wgsl"),
            river_carve_bgl,
        );
        let effect_filter_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("effect-filter-bgl"),
            entries: &[uniform_entry(0), tex_read_entry(1), storage_write_entry(2)],
        });
        let mask_bake_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mask-bake-bgl"),
            entries: &[uniform_entry(0), tex_read_entry(1), storage_write_entry(2)],
        });
        let effect_filter = make_pipe(
            device,
            "effect-filter",
            include_str!("shaders/effect_filter.wgsl"),
            effect_filter_bgl,
        );
        let mask_bake = make_pipe(
            device,
            "mask-bake",
            include_str!("shaders/mask_bake.wgsl"),
            mask_bake_bgl,
        );
        let ping = HeightTex::new(device, "ping", w, w);
        let pong = HeightTex::new(device, "pong", w, w);
        let layer_tex = HeightTex::new(device, "layer", w, w);
        let mask_ones = HeightTex::new(device, "mask-ones", w, w);
        let hardness = HeightTex::new(device, "hardness", w, w);
        let water_a = HeightTex::new(device, "water-a", w, w);
        let water_b = HeightTex::new(device, "water-b", w, w);
        let delta = HeightTex::new(device, "thermal-delta", w, w);
        let sed_a = HeightTex::new(device, "sed-a", w, w);
        let sed_b = HeightTex::new(device, "sed-b", w, w);
        let rainfall = HeightTex::new(device, "rainfall", w, w);
        let loose_sediment = HeightTex::new(device, "loose-sediment", w, w);
        let outflow = RgbaTex::new(device, "hydraulic-outflow", w, w);

        Self {
            fill,
            noise,
            blend,
            copy,
            thermal,
            thermal_apply,
            hydraulic_outflow,
            hydraulic,
            blur,
            terrace,
            ramp,
            shapes,
            river_accum,
            river_carve,
            effect_filter,
            mask_bake,
            uniform_pool: UniformPool::new(device, 64),
            ping,
            pong,
            layer_tex,
            mask_ones,
            hardness,
            water_a,
            water_b,
            delta,
            sed_a,
            sed_b,
            rainfall,
            loose_sediment,
            outflow,
            layer_cache: HashMap::new(),
            layer_contrib: HashMap::new(),
            dirty: HashSet::new(),
            metrics: HeightfieldMetrics {
                width: w,
                height: w,
                world_size_x: 1000.0,
                world_size_z: 1000.0,
                tile_size: w,
                halo: 0,
            },
            approx_range: (0.0, 120.0),
            current: 0,
            tile_sched: TileScheduler::new(),
            last_dirty_rect: None,
            last_quality: None,
            max_sim_iters_per_tick: 8,
            last_graph: GpuComputeGraph::default(),
        }
    }

    /// Bounding sample rect of tiles touched since last clear (padded for normals).
    pub fn dirty_region(&self, pad: u32) -> Option<SampleRect> {
        self.tile_sched.dirty_bounds(&self.metrics, pad)
    }

    /// Snapshot of dirty tile IDs for viewport debug overlay (does not clear).
    pub fn dirty_tiles(&self) -> &[TileId] {
        &self.tile_sched.dirty
    }

    /// Cap thermal/hydraulic iterations for the current interactive refinement phase.
    /// `None` means uncapped (export / full quality).
    pub fn set_simulation_iteration_cap(&mut self, cap: Option<u32>) {
        self.max_sim_iters_per_tick = cap.unwrap_or(u32::MAX);
    }

    pub fn take_dirty_region(&mut self, pad: u32) -> Option<SampleRect> {
        let r = self.dirty_region(pad);
        self.tile_sched.clear();
        r
    }

    fn mark_all_tiles_dirty(&mut self) {
        self.tile_sched.clear();
        for tz in 0..self.metrics.tiles_z() {
            for tx in 0..self.metrics.tiles_x() {
                self.tile_sched.mark_tile(TileId { tx, tz });
            }
        }
    }

    pub fn mark_dirty(&mut self, id: LayerId) {
        self.dirty.insert(id);
    }

    /// Set the texel-space bounds of the most recent local terrain edit.
    pub fn set_dirty_rect(&mut self, rect: Option<(u32, u32, u32, u32)>) {
        self.last_dirty_rect = rect;
    }

    fn mark_tiles_overlapping_rect(&mut self, rect: (u32, u32, u32, u32)) {
        let (x, y, w, h) = rect;
        if w == 0 || h == 0 || self.metrics.width == 0 || self.metrics.height == 0 {
            return;
        }
        let max_x = x
            .saturating_add(w)
            .saturating_sub(1)
            .min(self.metrics.width - 1);
        let max_y = y
            .saturating_add(h)
            .saturating_sub(1)
            .min(self.metrics.height - 1);
        let tx0 = x.min(self.metrics.width - 1) / self.metrics.tile_size;
        let tz0 = y.min(self.metrics.height - 1) / self.metrics.tile_size;
        let tx1 = max_x / self.metrics.tile_size;
        let tz1 = max_y / self.metrics.tile_size;
        self.tile_sched.clear();
        for tz in tz0..=tz1 {
            for tx in tx0..=tx1 {
                self.tile_sched.mark_tile(TileId { tx, tz });
            }
        }
    }

    pub fn mark_dirty_from(&mut self, stack: &LayerStack, id: LayerId) {
        let layers = stack.flatten_layers();
        let mut seen = false;
        for layer in layers {
            if layer.id() == id {
                seen = true;
            }
            if seen {
                self.dirty.insert(layer.id());
            }
        }
    }

    /// First flattened index that is dirty (None = all clean).
    pub fn first_dirty_index(&self, stack: &LayerStack) -> Option<usize> {
        stack
            .flatten_layers()
            .iter()
            .position(|layer| self.dirty.contains(&layer.id()))
    }

    pub fn is_dirty(&self, id: LayerId) -> bool {
        self.dirty.contains(&id)
    }

    pub fn has_layer_cache(&self, id: LayerId, metrics: HeightfieldMetrics) -> bool {
        self.layer_cache
            .get(&id)
            .is_some_and(|t| t.width == metrics.width && t.height == metrics.height)
    }

    /// Upload a CPU heightfield into the layer cache (WC bridge: bake shapes, keep filters live).
    pub fn ingest_height(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: LayerId,
        height: &Heightfield,
        height_range: (f32, f32),
    ) {
        if height.metrics.width == 0 || height.metrics.height == 0 {
            return;
        }
        self.ensure_size(device, height.metrics);
        let w = height.metrics.width;
        let h = height.metrics.height;
        let needs_new = self
            .layer_cache
            .get(&id)
            .map(|t| t.width != w || t.height != h)
            .unwrap_or(true);
        if needs_new {
            self.layer_cache
                .insert(id, HeightTex::new(device, "layer-cache", w, h));
        }
        let dense = height.to_dense();
        let cache = self.layer_cache.get(&id).expect("cache just inserted");
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &cache.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&dense),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.approx_range = height_range;
        self.dirty.remove(&id);
    }

    /// Upload a heightfield into the current ping/pong working buffer.
    fn upload_height_to_current(&mut self, queue: &wgpu::Queue, height: &Heightfield) {
        let w = self.metrics.width;
        let h = self.metrics.height;
        let dense = if height.metrics.width == w && height.metrics.height == h {
            height.to_dense()
        } else {
            resample_height_nearest(height, self.metrics)
        };
        let tex = if self.current == 0 {
            &self.ping.texture
        } else {
            &self.pong.texture
        };
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&dense),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for v in &dense {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
        if lo <= hi {
            self.approx_range = (lo, hi);
        }
    }

    pub fn mark_all_dirty(&mut self, stack: &LayerStack) {
        for layer in stack.flatten_layers() {
            self.dirty.insert(layer.id());
        }
    }

    /// Drop all project-owned GPU caches and replace project-sized working textures
    /// with the small resident baseline so a new/opened document starts clean.
    pub fn reset_project_state(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.layer_cache.clear();
        self.layer_contrib.clear();
        self.dirty.clear();
        self.last_dirty_rect = None;
        self.last_quality = None;
        self.last_graph = crate::graph::GpuComputeGraph::default();
        self.tile_sched = TileScheduler::new();
        self.approx_range = (0.0, 1.0);
        self.current = 0;
        self.uniform_pool.reset();
        let baseline_metrics = HeightfieldMetrics {
            width: PROJECT_RESET_TEXTURE_EXTENT,
            height: PROJECT_RESET_TEXTURE_EXTENT,
            world_size_x: self.metrics.world_size_x,
            world_size_z: self.metrics.world_size_z,
            tile_size: PROJECT_RESET_TEXTURE_EXTENT,
            halo: 0,
        };
        self.ensure_size(device, baseline_metrics);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu-project-reset"),
        });
        self.fill_slot(device, queue, &mut encoder, TexSlot::Ping, 0.0);
        self.fill_slot(device, queue, &mut encoder, TexSlot::Pong, 0.0);
        self.fill_slot(device, queue, &mut encoder, TexSlot::Layer, 0.0);
        self.fill_slot(device, queue, &mut encoder, TexSlot::MaskOnes, 1.0);
        queue.submit(Some(encoder.finish()));
    }

    pub fn output_texture(&self) -> &wgpu::Texture {
        if self.current == 0 {
            &self.ping.texture
        } else {
            &self.pong.texture
        }
    }

    /// Current evaluated height field view (R32Float) — sample directly from the renderer when formats match.
    pub fn output_texture_view(&self) -> &wgpu::TextureView {
        if self.current == 0 {
            &self.ping.view
        } else {
            &self.pong.view
        }
    }

    /// Alias for [`Self::output_texture_view`].
    pub fn height_texture_view(&self) -> &wgpu::TextureView {
        self.output_texture_view()
    }

    fn ensure_size(&mut self, device: &wgpu::Device, metrics: HeightfieldMetrics) {
        let w = metrics.width.max(PROJECT_RESET_TEXTURE_EXTENT);
        let h = metrics.height.max(PROJECT_RESET_TEXTURE_EXTENT);
        if self.ping.width == w
            && self.ping.height == h
            && self.metrics.world_size_x == metrics.world_size_x
        {
            self.metrics = metrics;
            return;
        }
        self.metrics = metrics;
        self.ping = HeightTex::new(device, "ping", w, h);
        self.pong = HeightTex::new(device, "pong", w, h);
        self.layer_tex = HeightTex::new(device, "layer", w, h);
        self.mask_ones = HeightTex::new(device, "mask-ones", w, h);
        self.hardness = HeightTex::new(device, "hardness", w, h);
        self.water_a = HeightTex::new(device, "water-a", w, h);
        self.water_b = HeightTex::new(device, "water-b", w, h);
        self.delta = HeightTex::new(device, "thermal-delta", w, h);
        self.sed_a = HeightTex::new(device, "sed-a", w, h);
        self.sed_b = HeightTex::new(device, "sed-b", w, h);
        self.rainfall = HeightTex::new(device, "rainfall", w, h);
        self.loose_sediment = HeightTex::new(device, "loose-sediment", w, h);
        self.outflow = RgbaTex::new(device, "hydraulic-outflow", w, h);
        self.layer_cache.clear();
        self.layer_contrib.clear();
        self.dirty.clear();
    }

    fn swap_current(&mut self) {
        self.current = 1 - self.current;
    }

    /// Write uniforms into the next pool slot and return that buffer.
    /// Each dispatch must bind its own slot ÔÇö wgpu applies all `queue.write_buffer`
    /// transfers before the command buffer, so a single shared buffer would make
    /// every pass see only the last write.
    fn write_uniform<T: Pod>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &T,
    ) -> wgpu::Buffer {
        self.uniform_pool.write(device, queue, data)
    }

    fn view_of(&self, slot: TexSlot) -> &wgpu::TextureView {
        match slot {
            TexSlot::Ping => &self.ping.view,
            TexSlot::Pong => &self.pong.view,
            TexSlot::Layer => &self.layer_tex.view,
            TexSlot::MaskOnes => &self.mask_ones.view,
            TexSlot::Hardness => &self.hardness.view,
            TexSlot::WaterA => &self.water_a.view,
            TexSlot::WaterB => &self.water_b.view,
            TexSlot::SedA => &self.sed_a.view,
            TexSlot::SedB => &self.sed_b.view,
            TexSlot::Rainfall => &self.rainfall.view,
            TexSlot::LooseSediment => &self.loose_sediment.view,
            TexSlot::Cache(id) => &self.layer_cache.get(&id).expect("cache").view,
            TexSlot::Contrib(id) => &self.layer_contrib.get(&id).expect("contrib").view,
        }
    }

    fn fill_slot(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        slot: TexSlot,
        value: f32,
    ) {
        let u = FillU {
            width: self.metrics.width,
            height: self.metrics.height,
            value,
            _pad: 0.0,
        };
        let u_buf = self.write_uniform(device, queue, &u);
        let view = self.view_of(slot);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fill-bg"),
            layout: &self.fill.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(view),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fill"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fill.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(
                (self.metrics.width + 7) / 8,
                (self.metrics.height + 7) / 8,
                1,
            );
        }
    }

    fn copy_slots(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        src: TexSlot,
        dst: TexSlot,
    ) {
        let u = CopyU {
            width: self.metrics.width,
            height: self.metrics.height,
            _p0: 0.0,
            _p1: 0.0,
        };
        let u_buf = self.write_uniform(device, queue, &u);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("copy-bg"),
            layout: &self.copy.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(self.view_of(src)),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(self.view_of(dst)),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("copy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.copy.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(
                (self.metrics.width + 7) / 8,
                (self.metrics.height + 7) / 8,
                1,
            );
        }
    }

    fn noise_type_u(t: FractalNoiseType) -> Option<u32> {
        match t {
            FractalNoiseType::Value => Some(0),
            FractalNoiseType::Perlin => Some(1),
            FractalNoiseType::OpenSimplex => None,
        }
    }

    fn gen_noise(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        p: &NoiseParams,
        noise_type: u32,
        ridged: bool,
    ) {
        let u = NoiseU {
            width: self.metrics.width,
            height: self.metrics.height,
            world_x: self.metrics.world_size_x,
            world_z: self.metrics.world_size_z,
            seed: (p.seed & 0xFFFF_FFFF) as u32,
            octaves: p.octaves.max(1),
            frequency: p.frequency,
            amplitude: p.amplitude,
            lacunarity: p.lacunarity,
            persistence: p.persistence,
            offset_x: p.offset_x,
            offset_z: p.offset_z,
            remap_min: p.remap_min,
            remap_max: p.remap_max,
            noise_type,
            mode: if ridged { 1 } else { 0 },
        };
        let u_buf = self.write_uniform(device, queue, &u);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("noise-bg"),
            layout: &self.noise.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.layer_tex.view),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("noise"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.noise.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(
                (self.metrics.width + 7) / 8,
                (self.metrics.height + 7) / 8,
                1,
            );
        }
    }

    fn gen_shape(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        u: ShapeU,
    ) {
        let u_buf = self.write_uniform(device, queue, &u);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shapes-bg"),
            layout: &self.shapes.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.layer_tex.view),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("shapes"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.shapes.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(
                (self.metrics.width + 7) / 8,
                (self.metrics.height + 7) / 8,
                1,
            );
        }
    }

    fn run_river_carve(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        p: &terra_core::layer::RiverCarveParams,
        quality: PreviewQuality,
    ) {
        let iters = match quality {
            PreviewQuality::Draft => 12u32,
            PreviewQuality::Medium => 32,
            PreviewQuality::Full | PreviewQuality::Export => {
                (self.metrics.width.min(self.metrics.height) / 4).clamp(48, 160)
            }
        };

        // Seed accumulation with unit rainfall.
        self.fill_slot(device, queue, encoder, TexSlot::WaterA, 1.0);
        self.fill_slot(device, queue, encoder, TexSlot::WaterB, 0.0);

        let height_slot = if self.current == 0 {
            TexSlot::Ping
        } else {
            TexSlot::Pong
        };
        let accum_u = RiverAccumU {
            width: self.metrics.width,
            height: self.metrics.height,
            _p0: 0.0,
            _p1: 0.0,
        };

        let mut src_a = true;
        for _ in 0..iters {
            let u_buf = self.write_uniform(device, queue, &accum_u);
            let (acc_in, acc_out) = if src_a {
                (&self.water_a.view, &self.water_b.view)
            } else {
                (&self.water_b.view, &self.water_a.view)
            };
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("river-accum-bg"),
                layout: &self.river_accum.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: u_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(self.view_of(height_slot)),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(acc_in),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(acc_out),
                    },
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("river-accum"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.river_accum.pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(
                    (self.metrics.width + 7) / 8,
                    (self.metrics.height + 7) / 8,
                    1,
                );
            }
            src_a = !src_a;
        }

        let carve_u = RiverCarveU {
            width: self.metrics.width,
            height: self.metrics.height,
            threshold: p.accumulation_threshold.max(1.0),
            depth: p.depth,
            channel_width: p.width.max(1.0),
            bank_smooth: p.bank_smooth.max(0.0),
            max_radius: match quality {
                PreviewQuality::Draft => 12,
                PreviewQuality::Medium => 20,
                PreviewQuality::Full | PreviewQuality::Export => 32,
            },
            _pad: 0,
        };
        let u_buf = self.write_uniform(device, queue, &carve_u);
        let acc_view = if src_a {
            &self.water_a.view
        } else {
            &self.water_b.view
        };
        let (src, dst) = if self.current == 0 {
            (&self.ping.view, &self.pong.view)
        } else {
            (&self.pong.view, &self.ping.view)
        };
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("river-carve-bg"),
            layout: &self.river_carve.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(acc_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(dst),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("river-carve"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.river_carve.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(
                (self.metrics.width + 7) / 8,
                (self.metrics.height + 7) / 8,
                1,
            );
        }
        self.swap_current();
        self.expand_range(self.approx_range.0 - p.depth * 2.0, self.approx_range.1);
    }

    fn blend_into_current(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        opacity: f32,
        mode: BlendMode,
    ) -> Result<(), GpuError> {
        let u = BlendU {
            width: self.metrics.width,
            height: self.metrics.height,
            opacity,
            mode: gpu_blend_mode(mode).ok_or(GpuError::RequiresCpu)?,
        };
        let u_buf = self.write_uniform(device, queue, &u);
        let src_ping = self.current == 0;
        let (base, dst) = if src_ping {
            (&self.ping.view, &self.pong.view)
        } else {
            (&self.pong.view, &self.ping.view)
        };
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blend-bg"),
            layout: &self.blend.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(base),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.layer_tex.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.mask_ones.view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(dst),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("blend"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.blend.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(
                (self.metrics.width + 7) / 8,
                (self.metrics.height + 7) / 8,
                1,
            );
        }
        self.swap_current();
        Ok(())
    }

    fn scale_iters(quality: PreviewQuality, iters: u32) -> u32 {
        match quality {
            // Draft must still read as a real filter change (WC interactive), not a no-op.
            PreviewQuality::Draft => iters.clamp(2, 8),
            PreviewQuality::Medium => iters.clamp(4, 12),
            PreviewQuality::Full | PreviewQuality::Export => iters.max(1),
        }
    }

    fn dirty_dispatch_extent(&self) -> (u32, u32, u32, u32, u32, u32) {
        // Returns (region_x, region_y, region_w, region_h, groups_x, groups_y)
        if let Some((x, y, w, h)) = self.last_dirty_rect {
            let (x, y, w, h) =
                expand_dirty_rect((x, y, w, h), 8, self.metrics.width, self.metrics.height);
            let gx = (w + 7) / 8;
            let gy = (h + 7) / 8;
            (x, y, w, h, gx.max(1), gy.max(1))
        } else {
            let w = self.metrics.width;
            let h = self.metrics.height;
            (0, 0, 0, 0, (w + 7) / 8, (h + 7) / 8)
        }
    }

    fn run_effect_filter(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        p: &EffectFilterParams,
        quality: PreviewQuality,
    ) {
        let mode = resolve_effect_mode(p.kind);
        let iters = Self::scale_iters(quality, p.iterations.max(1)).min(8);
        let (rx, ry, rw, rh, gx, gy) = self.dirty_dispatch_extent();
        for _ in 0..iters {
            let u = EffectFilterU {
                width: self.metrics.width,
                height: self.metrics.height,
                world_x: self.metrics.world_size_x,
                world_z: self.metrics.world_size_z,
                mode,
                radius: p.radius.max(1).min(16),
                iterations: iters,
                seed: (p.seed & 0xFFFF_FFFF) as u32,
                strength: p.strength.clamp(0.0, 1.0),
                amount: p.amount,
                frequency: p.effective_frequency(),
                sea_level: p.sea_level,
                beach_width: p
                    .beach_width
                    .max(p.crater_radius * self.metrics.world_size_x * 0.5),
                slope_min: p.slope_min,
                slope_max: p.slope_max,
                rock_hardness: p.rock_hardness,
                terrace_height: p.terrace_height,
                terrace_offset: p.terrace_offset,
                rotation_deg: p.rotation_deg,
                anisotropy: p.anisotropy,
                warp_strength: p.warp_strength,
                warp_frequency: p.warp_frequency,
                dx: self.metrics.dx(),
                invert: if p.invert { 1.0 } else { 0.0 },
                region_x: rx,
                region_y: ry,
                region_w: rw,
                region_h: rh,
            };
            let u_buf = self.write_uniform(device, queue, &u);
            let src_ping = self.current == 0;
            let (src, dst) = if src_ping {
                (&self.ping.view, &self.pong.view)
            } else {
                (&self.pong.view, &self.ping.view)
            };
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("effect-filter-bg"),
                layout: &self.effect_filter.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: u_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(src),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(dst),
                    },
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("effect-filter"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.effect_filter.pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
            self.swap_current();
        }
    }

    fn bake_layer_mask_gpu(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        layer: &Layer,
        mask_assets: &[MaskAsset],
    ) -> Result<(), GpuError> {
        self.fill_slot(device, queue, encoder, TexSlot::MaskOnes, 1.0);
        if layer.common.masks.is_empty() {
            return Ok(());
        }
        let [entry] = layer.common.masks.entries.as_slice() else {
            return Err(GpuError::RequiresCpu);
        };
        if !layer.common.masks.nodes.is_empty()
            || entry.combine != terra_core::mask::MaskCombine::Multiply
        {
            return Err(GpuError::RequiresCpu);
        }
        let asset = mask_assets
            .iter()
            .find(|asset| asset.id == entry.mask.id)
            .ok_or(GpuError::RequiresCpu)?;
        if !asset.ops.is_empty() {
            return Err(GpuError::RequiresCpu);
        }
        let (mode, value, range_min, range_max) = match &asset.source {
            MaskSource::Constant(v) => (0u32, *v, 0.0, 1.0),
            MaskSource::Height { min, max } => (1u32, 0.0, *min, *max),
            MaskSource::Slope { min_deg, max_deg } => (2u32, 0.0, *min_deg, *max_deg),
            _ => return Err(GpuError::RequiresCpu),
        };
        let (rx, ry, rw, rh, gx, gy) = self.dirty_dispatch_extent();
        let u = MaskBakeU {
            width: self.metrics.width,
            height: self.metrics.height,
            mode,
            dz: self.metrics.dz(),
            dx: self.metrics.dx(),
            value,
            range_min,
            range_max,
            invert: if entry.mask.invert { 1.0 } else { 0.0 },
            strength: entry.mask.strength,
            frequency: 0.0,
            seed: 0.0,
            region_x: rx,
            region_y: ry,
            region_w: rw,
            region_h: rh,
        };
        let u_buf = self.write_uniform(device, queue, &u);
        let height_view = if self.current == 0 {
            &self.ping.view
        } else {
            &self.pong.view
        };
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mask-bake-bg"),
            layout: &self.mask_bake.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(height_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.mask_ones.view),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mask-bake"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.mask_bake.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        Ok(())
    }

    /// Evaluate the GPU-compatible suffix of a stack, then return a CPU resume point if needed.
    ///
    /// `bridge_prefix` is an optional heightfield representing the stack through the layer
    /// before `first_dirty` (CPU cache / last-good). It lets filters stay live on GPU when
    /// earlier shape layers are not GPU-supported but already baked.
    pub fn evaluate(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        stack: &LayerStack,
        mask_assets: &[MaskAsset],
        metrics: HeightfieldMetrics,
        quality: PreviewQuality,
        want_cpu: bool,
        bridge_prefix: Option<&Heightfield>,
    ) -> Result<GpuEvalResult, GpuError> {
        profiling::scope!("gpu_stack_eval");
        // Flattened GPU evaluation cannot preserve scoped-group composition or solo
        // filtering. Leave all engine state and the last-good texture untouched so the
        // app can route the complete tree to its asynchronous CPU worker.
        if stack.requires_tree_evaluation() {
            return Err(GpuError::RequiresCpu);
        }
        self.ensure_size(device, metrics);
        self.uniform_pool.reset();
        let quality_changed = self.last_quality.replace(quality) != Some(quality);

        let layers = stack.flatten_layers();
        if quality_changed {
            // Drop contrib + wrong-size height caches. When a bridge prefix is supplied the
            // caller already marked the dirty suffix — do not force a full rebuild (that
            // produces the "weird Draft/zero frame" on filter add).
            self.layer_contrib.clear();
            self.layer_cache
                .retain(|_, tex| tex.width == metrics.width && tex.height == metrics.height);
            if bridge_prefix.is_none() {
                self.dirty.extend(layers.iter().map(|layer| layer.id()));
            }
        }
        if layers.is_empty() {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu-empty"),
            });
            self.fill_slot(device, queue, &mut encoder, TexSlot::Ping, 0.0);
            self.current = 0;
            queue.submit(Some(encoder.finish()));
            return Ok(GpuEvalResult {
                width: metrics.width,
                height: metrics.height,
                world_size: (metrics.world_size_x, metrics.world_size_z),
                height_range: (0.0, 0.0),
                fully_gpu: true,
                cpu: if want_cpu {
                    Some(Heightfield::zeros(metrics))
                } else {
                    None
                },
                resume_cpu_from: None,
                did_eval: true,
            });
        }

        let graph = compile_gpu_graph(stack, mask_assets);
        self.last_graph = graph;

        let layer_unsupported =
            |layer: &Layer| layer.common.enabled && !layer_gpu_supported(layer, mask_assets);

        let first_dirty = layers
            .iter()
            .position(|l| self.dirty.contains(&l.id()))
            .unwrap_or(layers.len());

        // All clean and fully GPU-cached: restore top cache (no recompute).
        if first_dirty >= layers.len() && !layers.iter().any(|l| layer_unsupported(l)) {
            if let Some(top) = layers.last() {
                if let Some(cached) = self.layer_cache.get(&top.id()) {
                    if cached.width == metrics.width && cached.height == metrics.height {
                        let mut encoder =
                            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("gpu-cache-hit"),
                            });
                        self.copy_slots(
                            device,
                            queue,
                            &mut encoder,
                            TexSlot::Cache(top.id()),
                            TexSlot::Ping,
                        );
                        self.current = 0;
                        queue.submit(Some(encoder.finish()));
                        let cpu = if want_cpu {
                            Some(self.readback_current(device, queue)?)
                        } else {
                            None
                        };
                        return Ok(GpuEvalResult {
                            width: metrics.width,
                            height: metrics.height,
                            world_size: (metrics.world_size_x, metrics.world_size_z),
                            height_range: self.approx_range,
                            fully_gpu: true,
                            cpu,
                            resume_cpu_from: None,
                            did_eval: true,
                        });
                    }
                }
            }
        }

        // A real CPU checkpoint stops before the first unsupported layer. Interactive
        // preview keeps walking the whole suffix so supported filters above an unsupported
        // layer remain live without forcing a UI-thread readback.
        let execution_end = if want_cpu {
            self.last_graph.cpu_from.unwrap_or(layers.len())
        } else {
            layers.len()
        };
        let first_dirty = first_dirty.min(execution_end);
        // Hybrid resume point (first unsupported we could only passthrough).
        let mut cpu_from = self.last_graph.cpu_from;
        let mut hybrid = false;

        // Seed from previous layer cache when possible.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu-stack"),
        });
        self.fill_slot(device, queue, &mut encoder, TexSlot::MaskOnes, 1.0);

        // A full re-evaluation or quality change affects every sample. Local sculpt
        // edits can instead update only tiles intersecting the supplied stamp bounds.
        // Filter/layer suffix edits also need a full present (shared path), not a stale
        // leftover sculpt rect from a prior stroke.
        let pass_dirty_rect = self.last_dirty_rect;
        if first_dirty == 0 || quality_changed || pass_dirty_rect.is_none() {
            self.mark_all_tiles_dirty();
            self.last_dirty_rect = None;
        } else if let Some(rect) = pass_dirty_rect {
            self.mark_tiles_overlapping_rect(rect);
        }

        let mut seeded = false;
        if first_dirty == 0 {
            self.fill_slot(device, queue, &mut encoder, TexSlot::Ping, 0.0);
            self.current = 0;
            self.approx_range = (0.0, 1.0);
            seeded = true;
        } else {
            let prev_id = layers[first_dirty - 1].id();
            let cached_ok = self
                .layer_cache
                .get(&prev_id)
                .map(|c| c.width == metrics.width && c.height == metrics.height)
                .unwrap_or(false);
            if cached_ok {
                self.copy_slots(
                    device,
                    queue,
                    &mut encoder,
                    TexSlot::Cache(prev_id),
                    TexSlot::Ping,
                );
                self.current = 0;
                // Preserve a sensible range when seeding from cache.
                if self.approx_range.1 <= self.approx_range.0 {
                    self.approx_range = (0.0, 120.0);
                }
                seeded = true;
            }
        }

        if !seeded {
            // Bridge: upload baked prefix (CPU cache / last-good) so dirty GPU filters run.
            let bridge_ok = bridge_prefix.is_some_and(|hf| {
                hf.metrics.width > 0
                    && hf.metrics.height > 0
                    && bridge_prefix_safe(&layers, first_dirty)
            });
            if bridge_ok {
                if let Some(hf) = bridge_prefix {
                    queue.submit(Some(encoder.finish()));
                    self.current = 0;
                    self.upload_height_to_current(queue, hf);
                    if first_dirty > 0 {
                        // Cache the bridged prefix under the previous layer id.
                        let mut enc =
                            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("gpu-bridge-cache"),
                            });
                        self.cache_current(device, queue, &mut enc, layers[first_dirty - 1].id());
                        queue.submit(Some(enc.finish()));
                    }
                    encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("gpu-stack-bridged"),
                    });
                    self.fill_slot(device, queue, &mut encoder, TexSlot::MaskOnes, 1.0);
                    seeded = true;
                }
            }
        }

        if !seeded {
            // Prefix is GPU-supported: dirty from 0 and re-enter.
            let prefix_gpu = layers[..first_dirty]
                .iter()
                .all(|l| !l.common.enabled || layer_gpu_supported(l, mask_assets));
            if prefix_gpu && first_dirty > 0 {
                drop(encoder);
                self.dirty.extend(layers.iter().map(|l| l.id()));
                return self.evaluate(
                    device,
                    queue,
                    stack,
                    mask_assets,
                    metrics,
                    quality,
                    want_cpu,
                    bridge_prefix,
                );
            }
            // Cannot seed — keep last-good on screen; async CPU must rebuild.
            drop(encoder);
            return Ok(GpuEvalResult {
                width: metrics.width,
                height: metrics.height,
                world_size: (metrics.world_size_x, metrics.world_size_z),
                height_range: self.approx_range,
                fully_gpu: false,
                cpu: None,
                resume_cpu_from: Some(0),
                did_eval: false,
            });
        }

        // Walk either the full speculative preview suffix or the exact GPU prefix needed
        // for CPU resume. Unsupported layers use bake cache or passthrough only in the
        // speculative path, so EffectFilters above shapes stay live interactively.
        for (layer_index, layer) in layers
            .iter()
            .enumerate()
            .skip(first_dirty)
            .take(execution_end.saturating_sub(first_dirty))
        {
            if !layer.common.enabled {
                self.cache_current(device, queue, &mut encoder, layer.id());
                self.dirty.remove(&layer.id());
                continue;
            }

            if layer_unsupported(layer) {
                let cached_ok = self
                    .layer_cache
                    .get(&layer.id())
                    .map(|c| c.width == metrics.width && c.height == metrics.height)
                    .unwrap_or(false);
                if cached_ok {
                    self.copy_slots(
                        device,
                        queue,
                        &mut encoder,
                        TexSlot::Cache(layer.id()),
                        TexSlot::Ping,
                    );
                    self.current = 0;
                    self.dirty.remove(&layer.id());
                } else {
                    // Uncached ProceduralShape / Stamp / Path / etc.
                    // Passthrough the working buffer so downstream GPU filters can still
                    // run, but do **not** cache this as the layer bake and do **not**
                    // clear dirty — that poisoned shapes as identity and skipped CPU.
                    hybrid = true;
                    if cpu_from.is_none() {
                        cpu_from = Some(layer_index);
                    }
                }
                continue;
            }

            if layer.common.masks.is_empty() {
                self.fill_slot(device, queue, &mut encoder, TexSlot::MaskOnes, 1.0);
            } else {
                // GPU-resident mask bake from current height prefix — no Maintain::Wait.
                self.bake_layer_mask_gpu(device, queue, &mut encoder, layer, mask_assets)?;
            }
            // Sculpt uploads must land on the queue before later layer_tex fills in this
            // encoder, otherwise a prior fill would overwrite the stamp buffer.
            let id = layer.id();
            let content_dirty = self.dirty.contains(&id);
            let contrib_ok = self
                .layer_contrib
                .get(&id)
                .map(|c| c.width == metrics.width && c.height == metrics.height)
                .unwrap_or(false);
            let reuse_contrib =
                !content_dirty && layer_input_independent(&layer.kind) && contrib_ok;

            if reuse_contrib {
                self.copy_slots(
                    device,
                    queue,
                    &mut encoder,
                    TexSlot::Contrib(id),
                    TexSlot::Layer,
                );
                self.blend_into_current(
                    device,
                    queue,
                    &mut encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            } else {
                if let LayerKind::SculptBase(params) = &layer.kind {
                    queue.submit(Some(encoder.finish()));
                    self.upload_sculpt_to_layer(queue, params);
                    encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("gpu-stack-sculpt"),
                    });
                }
                let plan = gpu_plan_for_layer(layer, mask_assets)
                    .expect("supported layer must retain an executable GPU plan");
                self.eval_layer(device, queue, &mut encoder, layer, plan.kernel, quality)?;
                if layer_input_independent(&layer.kind) {
                    let needs_new = self
                        .layer_contrib
                        .get(&id)
                        .map(|t| t.width != metrics.width || t.height != metrics.height)
                        .unwrap_or(true);
                    if needs_new {
                        self.layer_contrib.insert(
                            id,
                            HeightTex::new(device, "layer-contrib", metrics.width, metrics.height),
                        );
                    }
                    self.copy_slots(
                        device,
                        queue,
                        &mut encoder,
                        TexSlot::Layer,
                        TexSlot::Contrib(id),
                    );
                }
            }
            self.cache_current(device, queue, &mut encoder, id);
            self.dirty.remove(&id);
        }

        queue.submit(Some(encoder.finish()));
        self.last_dirty_rect = None;

        let fully_gpu = cpu_from.is_none() && !hybrid;
        let resume = if fully_gpu {
            None
        } else {
            cpu_from.or(Some(first_dirty))
        };

        // Interactive path (want_cpu=false): present GPU textures at any quality — no
        // Maintain::Wait readback. Export/oracle callers pass want_cpu=true.
        if !want_cpu {
            return Ok(GpuEvalResult {
                width: metrics.width,
                height: metrics.height,
                world_size: (metrics.world_size_x, metrics.world_size_z),
                height_range: self.approx_range,
                fully_gpu,
                cpu: None,
                resume_cpu_from: resume,
                did_eval: true,
            });
        }

        // Height-only prefixes can resume at their exact boundary. Prefixes that publish
        // aux or named outputs cannot be represented by this result type, so give the CPU
        // its canonical layer-zero seed instead of a partial and state-incomplete checkpoint.
        let resume = match resume {
            Some(index) if !cpu_resume_prefix_is_height_only(&layers, index) => Some(0),
            other => other,
        };
        let cpu = if resume == Some(0) {
            Some(Heightfield::zeros(metrics))
        } else {
            Some(self.readback_current(device, queue)?)
        };

        Ok(GpuEvalResult {
            width: metrics.width,
            height: metrics.height,
            world_size: (metrics.world_size_x, metrics.world_size_z),
            height_range: self.approx_range,
            fully_gpu: resume.is_none(),
            cpu,
            resume_cpu_from: resume,
            did_eval: true,
        })
    }

    /// Resample the sculpt paint buffer into `layer_tex` at the current eval resolution.
    fn upload_sculpt_to_layer(&self, queue: &wgpu::Queue, params: &SculptParams) {
        let w = self.metrics.width;
        let h = self.metrics.height;
        let mut dense = vec![0f32; (w as usize).saturating_mul(h as usize)];
        for j in 0..h {
            for i in 0..w {
                let u = (i as f32 + 0.5) / w as f32;
                let v = (j as f32 + 0.5) / h as f32;
                dense[(j * w + i) as usize] = params.sample_bilinear(u, v);
            }
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.layer_tex.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&dense),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
    }

    fn cache_current(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        id: LayerId,
    ) {
        let w = self.metrics.width;
        let h = self.metrics.height;
        let needs_new = self
            .layer_cache
            .get(&id)
            .map(|t| t.width != w || t.height != h)
            .unwrap_or(true);
        if needs_new {
            self.layer_cache
                .insert(id, HeightTex::new(device, "layer-cache", w, h));
        }
        let u = CopyU {
            width: w,
            height: h,
            _p0: 0.0,
            _p1: 0.0,
        };
        let u_buf = self.write_uniform(device, queue, &u);

        // Avoid simultaneous borrows: resolve views by current index + cache entry.
        let src_is_ping = self.current == 0;
        let src_view = if src_is_ping {
            &self.ping.view
        } else {
            &self.pong.view
        };
        let cache = self.layer_cache.get(&id).expect("cache just inserted");
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cache-bg"),
            layout: &self.copy.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&cache.view),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cache-copy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.copy.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((w + 7) / 8, (h + 7) / 8, 1);
        }
    }

    fn eval_layer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        layer: &Layer,
        kernel: GpuKernel,
        quality: PreviewQuality,
    ) -> Result<(), GpuError> {
        match (kernel, &layer.kind) {
            (GpuKernel::Sculpt, LayerKind::SculptBase(p)) => {
                // `layer_tex` was filled by `upload_sculpt_to_layer` just before this call.
                let (lo, hi) = p.sample_range();
                self.expand_range(lo, hi);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Fill, LayerKind::Flat(p)) => {
                self.fill_slot(device, queue, encoder, TexSlot::Layer, p.height);
                self.expand_range(p.height, p.height);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Ramp, LayerKind::Ramp(p)) => {
                let u = RampU {
                    width: self.metrics.width,
                    height: self.metrics.height,
                    world_x: self.metrics.world_size_x,
                    world_z: self.metrics.world_size_z,
                    height_min: p.height_min,
                    height_max: p.height_max,
                    direction: p.direction,
                    _pad: 0.0,
                };
                let u_buf = self.write_uniform(device, queue, &u);
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ramp-bg"),
                    layout: &self.ramp.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: u_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&self.layer_tex.view),
                        },
                    ],
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("ramp"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.ramp.pipeline);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(
                        (self.metrics.width + 7) / 8,
                        (self.metrics.height + 7) / 8,
                        1,
                    );
                }
                self.expand_range(p.height_min, p.height_max);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Noise, LayerKind::NoiseValue(p)) => {
                self.gen_noise(device, queue, encoder, p, 0, false);
                self.expand_range(0.0, p.amplitude);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Noise, LayerKind::NoisePerlin(p)) => {
                self.gen_noise(device, queue, encoder, p, 1, false);
                self.expand_range(0.0, p.amplitude);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Noise, LayerKind::Fbm(p)) => {
                let nt = Self::noise_type_u(p.noise).ok_or(GpuError::RequiresCpu)?;
                self.gen_noise(device, queue, encoder, &p.base, nt, false);
                self.expand_range(0.0, p.base.amplitude);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Noise, LayerKind::Ridged(p)) => {
                let nt = Self::noise_type_u(p.noise).ok_or(GpuError::RequiresCpu)?;
                self.gen_noise(device, queue, encoder, &p.base, nt, true);
                self.expand_range(0.0, p.base.amplitude);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            // Dedicated range-mask / dune asymmetry / canyon meander kernels.
            (GpuKernel::Shape, LayerKind::Mountains(p)) => {
                let u = ShapeU {
                    width: self.metrics.width,
                    height: self.metrics.height,
                    world_x: self.metrics.world_size_x,
                    world_z: self.metrics.world_size_z,
                    seed: (p.base.seed & 0xFFFF_FFFF) as u32,
                    octaves: p.base.octaves.max(1),
                    frequency: p.base.frequency,
                    amplitude: p.base.amplitude,
                    lacunarity: p.base.lacunarity,
                    persistence: p.base.persistence,
                    offset_x: p.base.offset_x,
                    offset_z: p.base.offset_z,
                    ridge_sharpness: p.ridge_sharpness,
                    range_angle: p.range_angle,
                    range_width: p.range_width,
                    wave_frequency: 0.0,
                    asymmetry: 0.0,
                    depth: 0.0,
                    canyon_width: 0.0,
                    meander: 0.0,
                    shape_mode: 0,
                    _pad: 0,
                };
                self.gen_shape(device, queue, encoder, u);
                self.expand_range(0.0, p.base.amplitude);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Shape, LayerKind::Dunes(p)) => {
                let u = ShapeU {
                    width: self.metrics.width,
                    height: self.metrics.height,
                    world_x: self.metrics.world_size_x,
                    world_z: self.metrics.world_size_z,
                    seed: (p.base.seed & 0xFFFF_FFFF) as u32,
                    octaves: p.base.octaves.max(1),
                    frequency: p.base.frequency,
                    amplitude: p.effective_height(),
                    lacunarity: p.base.lacunarity,
                    persistence: p.base.persistence,
                    offset_x: p.base.offset_x,
                    offset_z: p.base.offset_z,
                    ridge_sharpness: p.effective_crest_sharpness(),
                    range_angle: p.direction_deg,
                    range_width: p.linearity,
                    wave_frequency: p.effective_scale(),
                    asymmetry: p.effective_crest_sharpness(),
                    depth: p.trough_depth,
                    canyon_width: p.basin_floor,
                    meander: p.wind_strength,
                    shape_mode: 1,
                    _pad: 0,
                };
                self.gen_shape(device, queue, encoder, u);
                self.expand_range(0.0, p.effective_height());
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Shape, LayerKind::Canyons(p)) => {
                let u = ShapeU {
                    width: self.metrics.width,
                    height: self.metrics.height,
                    world_x: self.metrics.world_size_x,
                    world_z: self.metrics.world_size_z,
                    seed: (p.seed & 0xFFFF_FFFF) as u32,
                    octaves: 1,
                    frequency: 1.0,
                    amplitude: 1.0,
                    lacunarity: 2.0,
                    persistence: 0.5,
                    offset_x: 0.0,
                    offset_z: 0.0,
                    ridge_sharpness: 0.0,
                    range_angle: 0.0,
                    range_width: 0.0,
                    wave_frequency: 0.0,
                    asymmetry: 0.0,
                    depth: p.depth,
                    canyon_width: p.width,
                    meander: p.meander,
                    shape_mode: 2,
                    _pad: 0,
                };
                self.gen_shape(device, queue, encoder, u);
                self.expand_range(-p.depth, 0.0);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Noise, LayerKind::DomainWarp(p)) => {
                self.gen_noise(device, queue, encoder, &p.base, 1, false);
                self.expand_range(0.0, p.base.amplitude);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Thermal, LayerKind::ThermalErosion(p)) => {
                let talus = p.talus_angle_deg.to_radians().tan() * self.metrics.dx();
                let iters = Self::scale_iters(quality, p.iterations).min(match quality {
                    PreviewQuality::Draft => self.max_sim_iters_per_tick.max(1),
                    PreviewQuality::Medium => 24,
                    PreviewQuality::Full | PreviewQuality::Export => u32::MAX,
                });
                self.fill_slot(
                    device,
                    queue,
                    encoder,
                    TexSlot::Hardness,
                    p.hardness.clamp(0.0, 1.0),
                );
                for _ in 0..iters {
                    let strength = p.strength;
                    let talus_v = talus;
                    // Inline thermal step (avoid borrowing self.thermal while mutably borrowing self)
                    let u = ThermalU {
                        width: self.metrics.width,
                        height: self.metrics.height,
                        dx: self.metrics.dx(),
                        talus: talus_v,
                        strength,
                        _p2: 0.0,
                        _p3: 0.0,
                        _pad: 0.0,
                    };
                    let u_buf = self.write_uniform(device, queue, &u);
                    let src_ping = self.current == 0;
                    let (src, dst) = if src_ping {
                        (&self.ping.view, &self.pong.view)
                    } else {
                        (&self.pong.view, &self.ping.view)
                    };
                    let delta_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("thermal-delta-bg"),
                        layout: &self.thermal.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: u_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(&self.delta.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(&self.hardness.view),
                            },
                        ],
                    });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("thermal-delta"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.thermal.pipeline);
                        pass.set_bind_group(0, &delta_bg, &[]);
                        pass.dispatch_workgroups(
                            (self.metrics.width + 7) / 8,
                            (self.metrics.height + 7) / 8,
                            1,
                        );
                    }
                    let apply_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("thermal-apply-bg"),
                        layout: &self.thermal_apply.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: u_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(&self.delta.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(dst),
                            },
                        ],
                    });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("thermal-apply"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.thermal_apply.pipeline);
                        pass.set_bind_group(0, &apply_bg, &[]);
                        pass.dispatch_workgroups(
                            (self.metrics.width + 7) / 8,
                            (self.metrics.height + 7) / 8,
                            1,
                        );
                    }
                    self.swap_current();
                }
            }
            (GpuKernel::Hydraulic, LayerKind::HydraulicErosion(p)) => {
                let p = apply_transport_model(p, p.transport_model);
                self.fill_slot(device, queue, encoder, TexSlot::WaterA, 0.0);
                self.fill_slot(device, queue, encoder, TexSlot::WaterB, 0.0);
                self.fill_slot(device, queue, encoder, TexSlot::SedA, 0.0);
                self.fill_slot(device, queue, encoder, TexSlot::SedB, 0.0);
                self.fill_slot(device, queue, encoder, TexSlot::Rainfall, 1.0);
                self.fill_slot(
                    device,
                    queue,
                    encoder,
                    TexSlot::LooseSediment,
                    if p.layered_materials {
                        p.initial_sediment_thickness.max(0.0)
                    } else {
                        0.0
                    },
                );
                let eff_k = if p.layered_materials {
                    p.sediment_hardness.clamp(0.0, 1.0)
                } else {
                    p.hardness.clamp(0.0, 1.0)
                };
                self.fill_slot(device, queue, encoder, TexSlot::Hardness, eff_k);
                let iters = Self::scale_iters(quality, p.iterations).min(match quality {
                    PreviewQuality::Draft => self.max_sim_iters_per_tick.max(1),
                    PreviewQuality::Medium => 24,
                    PreviewQuality::Full | PreviewQuality::Export => u32::MAX,
                });
                let timestep = clamp_timestep_cfl(p.timestep, self.metrics.dx(), 4.0);
                let mut water_flip = false;
                for _ in 0..iters {
                    let u = HydraulicU {
                        width: self.metrics.width,
                        height: self.metrics.height,
                        timestep,
                        rainfall: p.rainfall,
                        evaporation: p.evaporation,
                        erosion: p.erosion,
                        deposition: p.deposition,
                        capacity: p.capacity,
                        fan_boost: p.fan_boost,
                        floodplain_bias: p.floodplain_bias,
                        dx: self.metrics.dx(),
                        incision_bias: p.incision_bias.max(0.05),
                        bedrock_k: p.bedrock_hardness.clamp(0.0, 1.0),
                        sediment_k: p.sediment_hardness.clamp(0.0, 1.0),
                        layered: if p.layered_materials { 1.0 } else { 0.0 },
                        _pad1: 0.0,
                    };
                    let u_buf = self.write_uniform(device, queue, &u);
                    let src_ping = self.current == 0;
                    let (h_src, h_dst) = if src_ping {
                        (&self.ping.view, &self.pong.view)
                    } else {
                        (&self.pong.view, &self.ping.view)
                    };
                    let (w_src, w_dst) = if water_flip {
                        (&self.water_b.view, &self.water_a.view)
                    } else {
                        (&self.water_a.view, &self.water_b.view)
                    };
                    let (s_src, s_dst) = if water_flip {
                        (&self.sed_b.view, &self.sed_a.view)
                    } else {
                        (&self.sed_a.view, &self.sed_b.view)
                    };
                    let outflow_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("hydraulic-outflow-bg"),
                        layout: &self.hydraulic_outflow.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: u_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(h_src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(w_src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(&self.outflow.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::TextureView(&self.rainfall.view),
                            },
                        ],
                    });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("hydraulic-outflow"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.hydraulic_outflow.pipeline);
                        pass.set_bind_group(0, &outflow_bg, &[]);
                        pass.dispatch_workgroups(
                            (self.metrics.width + 7) / 8,
                            (self.metrics.height + 7) / 8,
                            1,
                        );
                    }
                    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("hydraulic-bg"),
                        layout: &self.hydraulic.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: u_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(h_src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(w_src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(s_src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::TextureView(&self.outflow.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: wgpu::BindingResource::TextureView(h_dst),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: wgpu::BindingResource::TextureView(w_dst),
                            },
                            wgpu::BindGroupEntry {
                                binding: 7,
                                resource: wgpu::BindingResource::TextureView(s_dst),
                            },
                            wgpu::BindGroupEntry {
                                binding: 8,
                                resource: wgpu::BindingResource::TextureView(&self.hardness.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 9,
                                resource: wgpu::BindingResource::TextureView(&self.rainfall.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 10,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.loose_sediment.view,
                                ),
                            },
                        ],
                    });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("hydraulic"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.hydraulic.pipeline);
                        pass.set_bind_group(0, &bg, &[]);
                        pass.dispatch_workgroups(
                            (self.metrics.width + 7) / 8,
                            (self.metrics.height + 7) / 8,
                            1,
                        );
                    }
                    self.swap_current();
                    water_flip = !water_flip;
                }
            }
            (GpuKernel::RiverCarve, LayerKind::RiverCarve(p)) => {
                self.run_river_carve(device, queue, encoder, p, quality);
            }
            (GpuKernel::Blur, LayerKind::Blur(p)) => {
                let iters = p.iterations.max(1).min(8);
                for _ in 0..iters {
                    let u = BlurU {
                        width: self.metrics.width,
                        height: self.metrics.height,
                        radius: p.radius.max(1).min(8),
                        _pad: 0,
                    };
                    let u_buf = self.write_uniform(device, queue, &u);
                    let src_ping = self.current == 0;
                    let (src, dst) = if src_ping {
                        (&self.ping.view, &self.pong.view)
                    } else {
                        (&self.pong.view, &self.ping.view)
                    };
                    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("blur-bg"),
                        layout: &self.blur.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: u_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(src),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(dst),
                            },
                        ],
                    });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("blur"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.blur.pipeline);
                        pass.set_bind_group(0, &bg, &[]);
                        pass.dispatch_workgroups(
                            (self.metrics.width + 7) / 8,
                            (self.metrics.height + 7) / 8,
                            1,
                        );
                    }
                    self.swap_current();
                }
            }
            (GpuKernel::EffectFilter, LayerKind::EffectFilter(p)) => {
                self.run_effect_filter(device, queue, encoder, p, quality);
                let amp = p.amount.abs().max(1.0);
                self.expand_range(self.approx_range.0 - amp, self.approx_range.1 + amp);
            }
            (GpuKernel::Shape, LayerKind::Mesa(p)) => {
                let u = ShapeU {
                    width: self.metrics.width,
                    height: self.metrics.height,
                    world_x: self.metrics.world_size_x,
                    world_z: self.metrics.world_size_z,
                    seed: (p.seed & 0xFFFF_FFFF) as u32,
                    octaves: 3,
                    frequency: 0.001,
                    amplitude: p.height,
                    lacunarity: 2.0,
                    persistence: 0.5,
                    offset_x: p.center_u,
                    offset_z: p.center_v,
                    ridge_sharpness: p.edge_steepness,
                    range_angle: 0.0,
                    range_width: p.radius,
                    wave_frequency: 0.0,
                    asymmetry: 0.0,
                    depth: p.cap_noise,
                    canyon_width: 0.0,
                    meander: p.soft,
                    shape_mode: 5,
                    _pad: 0,
                };
                self.gen_shape(device, queue, encoder, u);
                self.expand_range(0.0, p.height);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Shape, LayerKind::Volcano(p)) => {
                let u = ShapeU {
                    width: self.metrics.width,
                    height: self.metrics.height,
                    world_x: self.metrics.world_size_x,
                    world_z: self.metrics.world_size_z,
                    seed: (p.seed & 0xFFFF_FFFF) as u32,
                    octaves: 3,
                    frequency: 0.001,
                    amplitude: p.height,
                    lacunarity: 2.0,
                    persistence: 0.5,
                    offset_x: p.center_u,
                    offset_z: p.center_v,
                    ridge_sharpness: p.flank_power,
                    range_angle: 0.0,
                    range_width: p.radius,
                    wave_frequency: 0.0,
                    asymmetry: 0.0,
                    depth: p.crater_depth,
                    canyon_width: p.crater_radius,
                    meander: p.roughness,
                    shape_mode: 4,
                    _pad: 0,
                };
                self.gen_shape(device, queue, encoder, u);
                self.expand_range(0.0, p.height);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Shape, LayerKind::Uplift(p)) => {
                let u = ShapeU {
                    width: self.metrics.width,
                    height: self.metrics.height,
                    world_x: self.metrics.world_size_x,
                    world_z: self.metrics.world_size_z,
                    seed: (p.seed & 0xFFFF_FFFF) as u32,
                    octaves: p.detail_octaves.max(1),
                    frequency: p.frequency,
                    amplitude: p.amplitude,
                    lacunarity: 2.0,
                    persistence: 0.5,
                    offset_x: 0.0,
                    offset_z: 0.0,
                    ridge_sharpness: p.ridge_power,
                    range_angle: p.range_angle,
                    range_width: p.corridor_width,
                    wave_frequency: p.detail_frequency,
                    asymmetry: p.altitude_fade,
                    depth: p.detail_amplitude,
                    canyon_width: 0.0,
                    meander: p.warp_strength,
                    shape_mode: 3,
                    _pad: 0,
                };
                self.gen_shape(device, queue, encoder, u);
                self.expand_range(0.0, p.amplitude);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Shape, LayerKind::Island(p)) => {
                // Preview: volcano-like massif + soft shelf (full island profile remains CPU oracle).
                let u = ShapeU {
                    width: self.metrics.width,
                    height: self.metrics.height,
                    world_x: self.metrics.world_size_x,
                    world_z: self.metrics.world_size_z,
                    seed: (p.seed & 0xFFFF_FFFF) as u32,
                    octaves: 4,
                    frequency: p.ridge_frequency.max(0.0001),
                    amplitude: p.mountain_height,
                    lacunarity: 2.0,
                    persistence: 0.5,
                    offset_x: p.center_u,
                    offset_z: p.center_v,
                    ridge_sharpness: p.mountain_power,
                    range_angle: p.rotation_deg,
                    range_width: p.radius,
                    wave_frequency: p.coastline_frequency,
                    asymmetry: p.aspect,
                    depth: p.beach_height,
                    canyon_width: p.lagoon_radius,
                    meander: p.coastline_warp,
                    shape_mode: 4,
                    _pad: 0,
                };
                self.gen_shape(device, queue, encoder, u);
                self.expand_range(p.ocean_floor, p.mountain_height);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Shape, LayerKind::Plateau(p)) => {
                // Preview: hard height clamp via mesa-style flat top across the field.
                let mid = (p.low + p.high) * 0.5;
                let u = ShapeU {
                    width: self.metrics.width,
                    height: self.metrics.height,
                    world_x: self.metrics.world_size_x,
                    world_z: self.metrics.world_size_z,
                    seed: 7,
                    octaves: 2,
                    frequency: 0.0005,
                    amplitude: mid,
                    lacunarity: 2.0,
                    persistence: 0.5,
                    offset_x: 0.5,
                    offset_z: 0.5,
                    ridge_sharpness: 2.5,
                    range_angle: 0.0,
                    range_width: 0.85,
                    wave_frequency: 0.0,
                    asymmetry: 0.0,
                    depth: p.soft,
                    canyon_width: 0.0,
                    meander: 0.15,
                    shape_mode: 5,
                    _pad: 0,
                };
                self.gen_shape(device, queue, encoder, u);
                self.expand_range(p.low, p.high);
                self.blend_into_current(
                    device,
                    queue,
                    encoder,
                    layer.common.opacity,
                    layer.common.blend,
                )?;
            }
            (GpuKernel::Terrace, LayerKind::Terrace(p)) => {
                let u = TerraceU {
                    width: self.metrics.width,
                    height: self.metrics.height,
                    levels: p.levels,
                    sharpness: p.sharpness,
                    min_h: self.approx_range.0 - 1e-3,
                    max_h: self.approx_range.1 + 1e-3,
                    _p0: 0.0,
                    _p1: 0.0,
                };
                let u_buf = self.write_uniform(device, queue, &u);
                let src_ping = self.current == 0;
                let (src, dst) = if src_ping {
                    (&self.ping.view, &self.pong.view)
                } else {
                    (&self.pong.view, &self.ping.view)
                };
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("terrace-bg"),
                    layout: &self.terrace.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: u_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(src),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(dst),
                        },
                    ],
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("terrace"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.terrace.pipeline);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(
                        (self.metrics.width + 7) / 8,
                        (self.metrics.height + 7) / 8,
                        1,
                    );
                }
                self.swap_current();
            }
            (planned, actual) => {
                return Err(GpuError::Wgpu(format!(
                    "GPU support plan {planned:?} does not match layer {actual:?}"
                )));
            }
        }
        Ok(())
    }

    fn expand_range(&mut self, lo: f32, hi: f32) {
        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
        self.approx_range.0 = self.approx_range.0.min(lo);
        self.approx_range.1 = self.approx_range.1.max(hi);
        if self.approx_range.0 > self.approx_range.1 {
            self.approx_range.1 = self.approx_range.0 + 1e-3;
        }
    }

    pub fn readback_current(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Heightfield, GpuError> {
        let w = self.metrics.width;
        let h = self.metrics.height;
        let unpadded = w * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = (unpadded + align - 1) / align * align;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-readback-buf"),
            size: (padded * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu-readback-enc"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: self.output_texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));
        let padded_f32 = readback_f32(device, queue, &buf, (padded * h / 4) as usize)?;
        let mut dense = Vec::with_capacity((w * h) as usize);
        let row_floats = (padded / 4) as usize;
        for y in 0..h as usize {
            let start = y * row_floats;
            dense.extend_from_slice(&padded_f32[start..start + w as usize]);
        }
        Ok(Heightfield::from_dense(self.metrics, &dense))
    }
}

/// `bridge_prefix` is only safe when it is the height *entering* `first_dirty`
/// (GPU/CPU cache of the previous layer). Full-stack last_good is never safe.
fn bridge_prefix_safe(layers: &[&Layer], first_dirty: usize) -> bool {
    first_dirty > 0 && first_dirty <= layers.len()
}

fn resample_height_nearest(src: &Heightfield, dst: HeightfieldMetrics) -> Vec<f32> {
    let w = dst.width as usize;
    let h = dst.height as usize;
    let mut out = vec![0.0f32; w.saturating_mul(h)];
    if src.metrics.width == 0 || src.metrics.height == 0 || w == 0 || h == 0 {
        return out;
    }
    let dense = src.to_dense();
    let sw = src.metrics.width as usize;
    let sh = src.metrics.height as usize;
    for j in 0..h {
        for i in 0..w {
            let u = (i as f32 + 0.5) / w as f32;
            let v = (j as f32 + 0.5) / h as f32;
            let si = ((u * sw as f32) as usize).min(sw - 1);
            let sj = ((v * sh as f32) as usize).min(sh - 1);
            out[j * w + i] = dense[sj * sw + si];
        }
    }
    out
}

#[cfg(test)]
mod smoke_tests {
    use super::*;
    use std::collections::HashMap;
    use terra_core::eval::{EvalContext, PreviewQuality, StackEvaluator};
    use terra_core::heightfield::HeightfieldMetrics;
    use terra_core::layer::{
        BlendMode, BlurParams, CoastalParams, EffectFilterParams, FbmParams, FlatParams,
        FractalNoiseType, GroupInputMode, IslandParams, Layer, LayerGroup, LayerKind, LayerStack,
        MaterialsParams, NamedOutputDecl, NoiseParams, SculptParams, StackNode,
        ThermalErosionParams,
    };
    use terra_core::mask::{
        bake_mask_assets, DistributionEntry, MaskAsset, MaskCombine, MaskId, MaskOp, MaskRef,
        MaskSource,
    };

    fn cpu_oracle(stack: &LayerStack, metrics: HeightfieldMetrics) -> Heightfield {
        let mut evaluator = StackEvaluator::new();
        let mut ctx = EvalContext::new(metrics);
        evaluator
            .rebuild_all(stack, &mut ctx)
            .expect("CPU stack oracle")
    }

    fn cpu_mask_oracle(
        stack: &LayerStack,
        metrics: HeightfieldMetrics,
        assets: &[MaskAsset],
    ) -> Heightfield {
        let mut evaluator = StackEvaluator::new();
        let mut ctx = EvalContext::new(metrics);
        ctx.masks = bake_mask_assets(
            assets,
            &Heightfield::zeros(metrics),
            metrics,
            &HashMap::new(),
        );
        ctx.mask_assets = assets.to_vec();
        evaluator
            .rebuild_all(stack, &mut ctx)
            .expect("CPU mask oracle")
    }

    fn masked_probe_stack(source: MaskSource) -> (LayerStack, MaskAsset, LayerId) {
        let resolution = 32;
        let samples = (0..resolution)
            .flat_map(|j| (0..resolution).map(move |i| i as f32 * 0.5 + j as f32 * 0.25))
            .collect();
        let sculpt = SculptParams {
            resolution,
            samples,
            fill_height: 0.0,
        };
        let asset = MaskAsset::new(MaskId::new(), "probe mask", source);
        let mut probe = Layer::new("unit probe", LayerKind::Flat(FlatParams { height: 1.0 }));
        probe.common.blend = BlendMode::Add;
        let mut binding = MaskRef::new(asset.id);
        binding.strength = 0.65;
        binding.invert = true;
        probe.common.masks.push(binding);

        let mut stack = LayerStack::new();
        let base = Layer::new("authored base", LayerKind::SculptBase(sculpt));
        let base_id = base.id();
        stack.push(base);
        stack.push(probe);
        (stack, asset, base_id)
    }

    fn assert_gpu_fallback(
        gpu: &terra_test_gpu::TestGpu,
        stack: &LayerStack,
        assets: &[MaskAsset],
        metrics: HeightfieldMetrics,
        owner_index: usize,
    ) {
        let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
        let result = engine
            .evaluate(
                &gpu.device,
                &gpu.queue,
                stack,
                assets,
                metrics,
                PreviewQuality::Draft,
                false,
                None,
            )
            .expect("unsupported composite should select CPU fallback");
        assert!(!result.fully_gpu);
        assert_eq!(result.resume_cpu_from, Some(owner_index));
        assert_eq!(engine.last_graph.cpu_from, Some(owner_index));
    }

    #[test]
    fn cpu_resume_prefix_requires_a_complete_height_only_checkpoint() {
        let flat = Layer::new("Flat", LayerKind::Flat(FlatParams { height: 10.0 }));
        let flat_layers = [&flat];
        assert!(cpu_resume_prefix_is_height_only(&flat_layers, 1));

        let island = Layer::new("Island", LayerKind::Island(IslandParams::default()));
        let island_layers = [&island];
        assert!(!cpu_resume_prefix_is_height_only(&island_layers, 1));

        let mut published = Layer::new("Published", LayerKind::Flat(FlatParams::default()));
        published
            .common
            .outputs
            .push(NamedOutputDecl::new("height checkpoint", FieldId::Height));
        let published_layers = [&published];
        assert!(!cpu_resume_prefix_is_height_only(&published_layers, 1));

        let mut disabled_island = island;
        disabled_island.common.enabled = false;
        let disabled_layers = [&disabled_island];
        assert!(cpu_resume_prefix_is_height_only(&disabled_layers, 1));
    }

    /// Revert check for #51: a requested CPU checkpoint must stop before the
    /// unsupported layer, while the no-readback preview may remain speculative.
    #[test]
    fn cpu_resume_readback_stops_before_unsupported_suffix() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "Base",
            LayerKind::Flat(FlatParams { height: 10.0 }),
        ));
        let mut unsupported = Layer::new(
            "Half-strength Add/Set",
            LayerKind::EffectFilter(EffectFilterParams::add_set()),
        );
        unsupported.common.opacity = 0.5;
        stack.push(unsupported);
        let mut downstream = Layer::new(
            "Downstream add",
            LayerKind::Flat(FlatParams { height: 2.0 }),
        );
        downstream.common.blend = BlendMode::Add;
        stack.push(downstream);

        let expected = cpu_oracle(&stack, metrics);
        assert!((expected.get(8, 8) - 17.0).abs() < 1.0e-4);

        let mut preview_engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
        let preview = preview_engine
            .evaluate(
                &gpu.device,
                &gpu.queue,
                &stack,
                &[],
                metrics,
                PreviewQuality::Draft,
                false,
                None,
            )
            .expect("speculative preview");
        assert_eq!(preview.resume_cpu_from, Some(1));
        assert!(preview.cpu.is_none());
        let speculative = preview_engine
            .readback_current(&gpu.device, &gpu.queue)
            .expect("speculative preview readback for test");
        assert!((speculative.get(8, 8) - 12.0).abs() < 0.01);

        let mut resume_engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
        let result = resume_engine
            .evaluate(
                &gpu.device,
                &gpu.queue,
                &stack,
                &[],
                metrics,
                PreviewQuality::Draft,
                true,
                None,
            )
            .expect("CPU checkpoint evaluation");
        assert!(!result.fully_gpu);
        assert_eq!(result.resume_cpu_from, Some(1));
        let checkpoint = result.cpu.expect("height entering unsupported layer");
        assert!((checkpoint.get(8, 8) - 10.0).abs() < 0.01);

        let mut evaluator = StackEvaluator::new();
        let mut ctx = EvalContext::new(metrics);
        let completed = evaluator
            .evaluate_suffix(&stack, &mut ctx, 1, checkpoint)
            .expect("CPU suffix");
        let max_error = completed
            .to_dense()
            .iter()
            .zip(expected.to_dense())
            .map(|(actual, oracle)| (actual - oracle).abs())
            .fold(0.0f32, f32::max);
        assert!(max_error < 0.01, "hybrid vs CPU max error {max_error}");
    }

    #[test]
    fn aux_producing_gpu_prefix_forces_full_cpu_restart() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
        let land_mask = MaskAsset::new(
            MaskId::new(),
            "Island land",
            MaskSource::Named(terra_core::fields::keys::LAND_MASK.into()),
        );
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "Island",
            LayerKind::Island(IslandParams::default()),
        ));
        let mut upper = Layer::new("Land-only add", LayerKind::Flat(FlatParams { height: 5.0 }));
        upper.common.blend = BlendMode::Add;
        upper.common.masks.push(MaskRef::new(land_mask.id));
        stack.push(upper);
        let assets = vec![land_mask];

        let expected = cpu_mask_oracle(&stack, metrics, &assets);
        let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
        let result = engine
            .evaluate(
                &gpu.device,
                &gpu.queue,
                &stack,
                &assets,
                metrics,
                PreviewQuality::Draft,
                true,
                None,
            )
            .expect("conservative CPU restart");
        assert_eq!(engine.last_graph.cpu_from, Some(1));
        assert_eq!(result.resume_cpu_from, Some(0));
        let checkpoint = result.cpu.expect("layer-zero seed");
        assert!(checkpoint.to_dense().iter().all(|height| *height == 0.0));

        let mut evaluator = StackEvaluator::new();
        let mut ctx = EvalContext::new(metrics);
        ctx.mask_assets = assets.clone();
        ctx.masks = bake_mask_assets(&assets, &checkpoint, metrics, &HashMap::new());
        let completed = evaluator
            .evaluate_suffix(&stack, &mut ctx, 0, checkpoint)
            .expect("full CPU restart");
        assert!(ctx
            .aux_maps
            .get(terra_core::fields::keys::LAND_MASK)
            .is_some());
        assert_eq!(completed.to_dense(), expected.to_dense());
    }

    /// Revert check for #47: scoped groups must never reach the flattened GPU evaluator.
    #[test]
    fn scoped_group_requires_cpu_tree_evaluation() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "Base",
            LayerKind::Flat(FlatParams { height: 10.0 }),
        ));
        let mut group = LayerGroup::isolated("Scoped");
        group.input_mode = GroupInputMode::EmptyHeight;
        group.opacity = 0.5;
        group.children.push(StackNode::Layer(Layer::new(
            "Feature",
            LayerKind::Flat(FlatParams { height: 20.0 }),
        )));
        stack.push_group(group);

        let expected = cpu_oracle(&stack, metrics);
        assert!((expected.get(8, 8) - 15.0).abs() < 1.0e-4);

        let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
        let result = engine.evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        );
        assert!(matches!(result, Err(GpuError::RequiresCpu)));
        assert!(
            engine.last_quality.is_none(),
            "preflight must not mutate GPU state"
        );
    }

    /// Revert check for #47: solo filtering is a tree operation, not a flat GPU stack.
    #[test]
    fn solo_stack_requires_cpu_tree_evaluation() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "Base",
            LayerKind::Flat(FlatParams { height: 100.0 }),
        ));
        let mut solo = Layer::new("Solo", LayerKind::Flat(FlatParams { height: 20.0 }));
        solo.common.blend = BlendMode::Add;
        solo.common.solo = true;
        stack.push(solo);
        let mut sibling = Layer::new("Sibling", LayerKind::Flat(FlatParams { height: 50.0 }));
        sibling.common.blend = BlendMode::Add;
        stack.push(sibling);

        let expected = cpu_oracle(&stack, metrics);
        assert!((expected.get(8, 8) - 20.0).abs() < 1.0e-4);

        let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
        let result = engine.evaluate(
            &gpu.device,
            &gpu.queue,
            &stack,
            &[],
            metrics,
            PreviewQuality::Draft,
            true,
            None,
        );
        assert!(matches!(result, Err(GpuError::RequiresCpu)));
    }

    #[test]
    fn pass_through_group_remains_fully_gpu() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "Base",
            LayerKind::Flat(FlatParams { height: 10.0 }),
        ));
        let mut folder = LayerGroup::new("Folder");
        let mut child = Layer::new("Child", LayerKind::Flat(FlatParams { height: 5.0 }));
        child.common.blend = BlendMode::Add;
        folder.children.push(StackNode::Layer(child));
        stack.push_group(folder);

        let expected = cpu_oracle(&stack, metrics);
        let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
        let result = engine
            .evaluate(
                &gpu.device,
                &gpu.queue,
                &stack,
                &[],
                metrics,
                PreviewQuality::Draft,
                true,
                None,
            )
            .expect("pass-through folders are flattenable");
        assert!(result.fully_gpu);
        assert_eq!(result.resume_cpu_from, None);
        let actual = result.cpu.expect("GPU readback");
        assert!((actual.get(8, 8) - expected.get(8, 8)).abs() < 0.01);
    }

    /// Revert check for #48: Coastal must request CPU instead of completing as identity.
    #[test]
    fn coastal_marks_gpu_preview_incomplete() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "Base",
            LayerKind::Flat(FlatParams { height: 20.0 }),
        ));
        stack.push(Layer::new(
            "Coastal",
            LayerKind::Coastal(CoastalParams::default()),
        ));

        let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
        let result = engine
            .evaluate(
                &gpu.device,
                &gpu.queue,
                &stack,
                &[],
                metrics,
                PreviewQuality::Draft,
                false,
                None,
            )
            .expect("unsupported layer should select fallback");
        assert!(!result.fully_gpu);
        assert_eq!(result.resume_cpu_from, Some(1));
        assert_eq!(engine.last_graph.cpu_from, Some(1));
    }

    /// Revert check for #48: OpenSimplex fractals must not error or become Perlin.
    #[test]
    fn open_simplex_fractals_mark_gpu_preview_incomplete() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
        for kind in [
            LayerKind::Fbm(FbmParams {
                noise: FractalNoiseType::OpenSimplex,
                ..FbmParams::default()
            }),
            LayerKind::Ridged(FbmParams {
                noise: FractalNoiseType::OpenSimplex,
                ..FbmParams::default()
            }),
        ] {
            let mut stack = LayerStack::new();
            stack.push(Layer::new("OpenSimplex", kind));
            let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
            let result = engine
                .evaluate(
                    &gpu.device,
                    &gpu.queue,
                    &stack,
                    &[],
                    metrics,
                    PreviewQuality::Draft,
                    false,
                    None,
                )
                .expect("unsupported noise should select fallback");
            assert!(!result.fully_gpu);
            assert_eq!(result.resume_cpu_from, Some(0));
        }
    }

    /// A cached height does not make Materials' missing aux outputs GPU-complete.
    #[test]
    fn cached_materials_height_keeps_cpu_boundary() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "Base",
            LayerKind::Flat(FlatParams { height: 20.0 }),
        ));
        let materials = Layer::new(
            "Materials",
            LayerKind::Materials(MaterialsParams::default()),
        );
        let materials_id = materials.id();
        stack.push(materials);
        let mut upper = Layer::new("Upper", LayerKind::Flat(FlatParams { height: 2.0 }));
        upper.common.blend = BlendMode::Add;
        stack.push(upper);

        let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
        let cached = Heightfield::filled(metrics, 20.0);
        engine.ingest_height(&gpu.device, &gpu.queue, materials_id, &cached, (20.0, 20.0));
        let result = engine
            .evaluate(
                &gpu.device,
                &gpu.queue,
                &stack,
                &[],
                metrics,
                PreviewQuality::Draft,
                false,
                None,
            )
            .expect("cached unsupported height can be presented speculatively");
        assert!(!result.fully_gpu);
        assert_eq!(result.resume_cpu_from, Some(1));
        assert_eq!(engine.last_graph.cpu_from, Some(1));
    }

    #[test]
    fn materials_aux_affects_cpu_suffix_fixture() {
        let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
        let hardness_id = MaskId::new();
        let hardness_asset = MaskAsset::new(hardness_id, "Hardness", MaskSource::Hardness);
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "Base",
            LayerKind::Flat(FlatParams { height: 20.0 }),
        ));
        stack.push(Layer::new(
            "Materials",
            LayerKind::Materials(MaterialsParams::default()),
        ));
        let mut upper = Layer::new("Upper", LayerKind::Flat(FlatParams { height: 10.0 }));
        upper.common.blend = BlendMode::Add;
        upper.common.masks.push(MaskRef::new(hardness_id));
        stack.push(upper);

        let graph = compile_gpu_graph(&stack, &[hardness_asset.clone()]);
        assert_eq!(graph.cpu_from, Some(1));

        let mut evaluator = StackEvaluator::new();
        let mut ctx = EvalContext::new(metrics);
        ctx.mask_assets = vec![hardness_asset];
        let result = evaluator
            .rebuild_all(&stack, &mut ctx)
            .expect("CPU fallback oracle");
        assert!(ctx.aux_maps.hardness.is_some());
        assert!(
            (result.get(8, 8) - 22.0).abs() < 0.1,
            "downstream hardness mask must observe Materials aux"
        );
    }

    #[test]
    fn constant_height_and_slope_masks_match_cpu_oracle() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        // Equal texture dimensions but unequal world spacing catches dx/dz substitution.
        let metrics = HeightfieldMetrics::new(32, 32, 320.0, 80.0);
        for source in [
            MaskSource::Constant(0.35),
            MaskSource::Height {
                min: 4.0,
                max: 16.0,
            },
            MaskSource::Slope {
                min_deg: 2.0,
                max_deg: 18.0,
            },
        ] {
            let (stack, asset, base_id) = masked_probe_stack(source);
            let expected = cpu_mask_oracle(&stack, metrics, std::slice::from_ref(&asset));
            let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
            engine.mark_dirty(base_id);
            let result = engine
                .evaluate(
                    &gpu.device,
                    &gpu.queue,
                    &stack,
                    std::slice::from_ref(&asset),
                    metrics,
                    PreviewQuality::Draft,
                    true,
                    None,
                )
                .expect("supported GPU mask");
            assert!(result.fully_gpu);
            assert_eq!(result.resume_cpu_from, None);
            let actual = result.cpu.expect("GPU mask readback");
            let max_error = actual
                .to_dense()
                .iter()
                .zip(expected.to_dense())
                .map(|(gpu, cpu)| (gpu - cpu).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_error <= 1.0e-3,
                "GPU mask exceeded documented tolerance: {max_error}"
            );
        }
    }

    /// Revert check for #50: a mask can be GPU-bakeable while the in-place filter
    /// that owns it cannot apply the outer composite. That owner must fall back.
    #[test]
    fn masked_blur_selects_cpu_fallback_at_filter() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
        let samples = (0..16 * 16)
            .map(|index| if index % 2 == 0 { 0.0 } else { 100.0 })
            .collect();
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "varying base",
            LayerKind::SculptBase(SculptParams {
                resolution: 16,
                samples,
                fill_height: 0.0,
            }),
        ));
        let asset = MaskAsset::new(MaskId::new(), "half", MaskSource::Constant(0.5));
        let mut blur = Layer::new("masked blur", LayerKind::Blur(BlurParams::default()));
        blur.common.masks.push(MaskRef::new(asset.id));
        stack.push(blur);

        let expected = cpu_mask_oracle(&stack, metrics, std::slice::from_ref(&asset));
        assert!(
            expected
                .to_dense()
                .iter()
                .any(|height| *height > 1.0 && *height < 99.0),
            "fixture must exercise partial masked filtering"
        );
        assert_gpu_fallback(&gpu, &stack, std::slice::from_ref(&asset), metrics, 1);
    }

    /// Revert check for #50: a simulation result must be composited with
    /// LayerCommon opacity rather than mutating the entering field directly.
    #[test]
    fn partial_opacity_simulation_selects_cpu_fallback() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let metrics = HeightfieldMetrics::new(16, 16, 16.0, 16.0);
        let mut samples = vec![0.0; 16 * 16];
        samples[8 * 16 + 8] = 100.0;
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "peaked base",
            LayerKind::SculptBase(SculptParams {
                resolution: 16,
                samples,
                fill_height: 0.0,
            }),
        ));
        let mut simulation = Layer::new(
            "partial thermal",
            LayerKind::ThermalErosion(ThermalErosionParams {
                iterations: 2,
                ..ThermalErosionParams::default()
            }),
        );
        simulation.common.opacity = 0.5;
        stack.push(simulation);

        let expected = cpu_oracle(&stack, metrics);
        assert!(
            expected.get(8, 8) < 99.0,
            "fixture must exercise partial outer compositing"
        );
        assert_gpu_fallback(&gpu, &stack, &[], metrics, 1);
    }

    /// Revert check for #50: unsupported blend equations must never be mapped to
    /// Overlay or hard min/max operations by the runtime.
    #[test]
    fn unsupported_generator_blends_select_cpu_fallback() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
        for mode in [BlendMode::HeightBlend, BlendMode::SmoothUnion] {
            let mut stack = LayerStack::new();
            stack.push(Layer::new(
                "base",
                LayerKind::Flat(FlatParams { height: 10.0 }),
            ));
            let mut contribution =
                Layer::new("contribution", LayerKind::Flat(FlatParams { height: 20.0 }));
            contribution.common.blend = mode;
            stack.push(contribution);

            let expected = cpu_oracle(&stack, metrics);
            assert!(expected.get(8, 8).is_finite(), "CPU oracle for {mode:?}");
            assert_gpu_fallback(&gpu, &stack, &[], metrics, 1);
        }
    }

    #[test]
    fn complex_masks_mark_gpu_preview_incomplete() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);

        let first = MaskAsset::new(MaskId::new(), "first", MaskSource::Constant(0.8));
        let second = MaskAsset::new(MaskId::new(), "second", MaskSource::Constant(0.25));
        let mut ordered = Layer::new("ordered", LayerKind::Flat(FlatParams { height: 1.0 }));
        ordered.common.masks.push(MaskRef::new(first.id));
        ordered.common.masks.entries.push(DistributionEntry {
            mask: MaskRef::new(second.id),
            combine: MaskCombine::Subtract,
        });

        let mut operated = MaskAsset::new(MaskId::new(), "operated", MaskSource::Constant(0.2));
        operated.ops.push(MaskOp::Invert);
        let operated_layer = {
            let mut layer = Layer::new("operated", LayerKind::Flat(FlatParams { height: 1.0 }));
            layer.common.masks.push(MaskRef::new(operated.id));
            layer
        };

        for (layer, assets) in [
            (ordered, vec![first, second]),
            (operated_layer, vec![operated]),
        ] {
            let layer_id = layer.id();
            let mut stack = LayerStack::new();
            stack.push(layer);
            let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
            engine.mark_dirty(layer_id);
            let result = engine
                .evaluate(
                    &gpu.device,
                    &gpu.queue,
                    &stack,
                    &assets,
                    metrics,
                    PreviewQuality::Draft,
                    false,
                    None,
                )
                .expect("unsupported mask should select fallback");
            assert!(!result.fully_gpu);
            assert_eq!(result.resume_cpu_from, Some(0));
            assert_eq!(engine.last_graph.cpu_from, Some(0));
        }
    }

    #[test]
    fn cached_mask_result_cannot_hide_new_asset_operations() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
        let mut asset = MaskAsset::new(MaskId::new(), "mask", MaskSource::Constant(0.5));
        let mut layer = Layer::new("masked", LayerKind::Flat(FlatParams { height: 10.0 }));
        layer.common.masks.push(MaskRef::new(asset.id));
        let layer_id = layer.id();
        let mut stack = LayerStack::new();
        stack.push(layer);

        let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
        engine.mark_dirty(layer_id);
        let first = engine
            .evaluate(
                &gpu.device,
                &gpu.queue,
                &stack,
                std::slice::from_ref(&asset),
                metrics,
                PreviewQuality::Draft,
                false,
                None,
            )
            .expect("initial supported mask");
        assert!(first.fully_gpu);

        asset.ops.push(MaskOp::Invert);
        let second = engine
            .evaluate(
                &gpu.device,
                &gpu.queue,
                &stack,
                std::slice::from_ref(&asset),
                metrics,
                PreviewQuality::Draft,
                false,
                None,
            )
            .expect("cached unsupported mask should select fallback");
        assert!(!second.fully_gpu);
        assert_eq!(second.resume_cpu_from, Some(0));
        assert_eq!(engine.last_graph.cpu_from, Some(0));
    }

    /// Regression for uniform isolation via pool slots: Draft must composite layers
    /// even when blend and cache share one submit (each pass gets its own uniform buffer).
    #[test]
    fn draft_eval_composites_sculpt_and_noise() {
        let Some(gpu) = terra_test_gpu::headless() else {
            // Headless CI without a GPU adapter.
            return;
        };
        let metrics = HeightfieldMetrics::new(64, 64, 64.0, 64.0);
        let mut stack = LayerStack::new();
        let base = Layer::new(
            "Base",
            LayerKind::SculptBase(SculptParams::filled(64, 20.0)),
        );
        let base_id = base.id();
        stack.push(base);
        stack.push(Layer::new(
            "Hills",
            LayerKind::NoiseValue(NoiseParams {
                seed: 1,
                frequency: 0.05,
                amplitude: 10.0,
                octaves: 1,
                lacunarity: 2.0,
                persistence: 0.5,
                ..NoiseParams::default()
            }),
        ));

        let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
        engine.mark_dirty(base_id);
        let before = engine
            .evaluate(
                &gpu.device,
                &gpu.queue,
                &stack,
                &[],
                metrics,
                PreviewQuality::Draft,
                true,
                None,
            )
            .expect("draft eval");
        let hf0 = before.cpu.expect("cpu readback");
        let center0 = hf0.get(32, 32);
        // Without per-pass submit, blends see opacity 0 and the field stays ~0.
        assert!(
            center0 > 15.0,
            "expected sculpt base (~20) through Draft blend, got {center0}"
        );

        // Live raise: stamp then re-eval Draft without waiting for CPU refine.
        {
            let mut layers = stack.flatten_layers_mut();
            if let LayerKind::SculptBase(params) = &mut layers[0].kind {
                params.stamp_circle(0.5, 0.5, 0.15, 25.0, 0);
            }
        }
        engine.mark_dirty_from(&stack, base_id);
        let after = engine
            .evaluate(
                &gpu.device,
                &gpu.queue,
                &stack,
                &[],
                metrics,
                PreviewQuality::Draft,
                true,
                None,
            )
            .expect("draft eval after stamp");
        let hf1 = after.cpu.expect("cpu readback");
        let center1 = hf1.get(32, 32);
        assert!(
            center1 > center0 + 5.0,
            "live Raise should lift Draft heights while held; before={center0} after={center1}"
        );
    }

    #[test]
    fn draft_eval_applies_effect_filter() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let metrics = HeightfieldMetrics::new(64, 64, 64.0, 64.0);
        let mut stack = LayerStack::new();
        let base = Layer::new(
            "Base",
            LayerKind::SculptBase(SculptParams::filled(64, 20.0)),
        );
        let base_id = base.id();
        stack.push(base);
        let filter = Layer::new(
            "Inflate",
            LayerKind::EffectFilter(terra_core::layer::EffectFilterParams::inflate()),
        );
        let filter_id = filter.id();
        stack.push(filter);

        let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
        engine.mark_dirty(base_id);
        let before = engine
            .evaluate(
                &gpu.device,
                &gpu.queue,
                &stack,
                &[],
                metrics,
                PreviewQuality::Draft,
                true,
                None,
            )
            .expect("draft with filter");
        assert!(before.did_eval);
        assert!(before.fully_gpu);
        let h0 = before.cpu.expect("cpu").get(32, 32);

        // Disable filter and compare — inflate should have raised the surface.
        if let Some(layer) = stack.find_mut(filter_id) {
            layer.common.enabled = false;
        }
        engine.mark_dirty_from(&stack, base_id);
        let after = engine
            .evaluate(
                &gpu.device,
                &gpu.queue,
                &stack,
                &[],
                metrics,
                PreviewQuality::Draft,
                true,
                None,
            )
            .expect("draft without filter");
        let h1 = after.cpu.expect("cpu").get(32, 32);
        assert!(
            h0 > h1 + 0.5,
            "Inflate EffectFilter should raise Draft heights; with={h0} without={h1}"
        );
    }

    #[test]
    fn draft_eval_flat_survives_cache_copy_uniform() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let metrics = HeightfieldMetrics::new(32, 32, 32.0, 32.0);
        let mut stack = LayerStack::new();
        let layer = Layer::new("Flat", LayerKind::Flat(FlatParams { height: 50.0 }));
        let id = layer.id();
        stack.push(layer);

        let mut engine = GpuTerrainEngine::new(&gpu.device, metrics.width);
        engine.mark_dirty(id);
        let result = engine
            .evaluate(
                &gpu.device,
                &gpu.queue,
                &stack,
                &[],
                metrics,
                PreviewQuality::Draft,
                true,
                None,
            )
            .expect("flat draft");
        let hf = result.cpu.expect("cpu");
        let mid = hf.get(16, 16);
        assert!(
            (mid - 50.0).abs() < 0.01,
            "Flat blend must not be clobbered by cache CopyU; got {mid}"
        );
    }

    fn assert_working_texture_dimensions(engine: &GpuTerrainEngine, width: u32, height: u32) {
        for texture in [
            &engine.ping,
            &engine.pong,
            &engine.layer_tex,
            &engine.mask_ones,
            &engine.hardness,
            &engine.water_a,
            &engine.water_b,
            &engine.delta,
            &engine.sed_a,
            &engine.sed_b,
            &engine.rainfall,
            &engine.loose_sediment,
        ] {
            assert_eq!((texture.width, texture.height), (width, height));
            assert_eq!(
                (texture.texture.width(), texture.texture.height()),
                (width, height)
            );
        }
        assert_eq!(
            (
                engine.outflow._texture.width(),
                engine.outflow._texture.height()
            ),
            (width, height)
        );
    }

    /// Revert check for #35: reset must release project-sized evaluator textures,
    /// and the existing evaluation size check must restore the next document size.
    #[test]
    fn project_reset_shrinks_working_set_and_evaluate_restores_size() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let mut engine = GpuTerrainEngine::new(&gpu.device, 64);
        let cached_layer = Layer::new("Cached", LayerKind::Flat(FlatParams { height: 1.0 }));
        let cached_id = cached_layer.id();
        engine.layer_cache.insert(
            cached_id,
            HeightTex::new(&gpu.device, "reset-test-cache", 64, 64),
        );
        engine.layer_contrib.insert(
            cached_id,
            HeightTex::new(&gpu.device, "reset-test-contrib", 64, 64),
        );
        engine.mark_dirty(cached_id);

        engine.reset_project_state(&gpu.device, &gpu.queue);

        assert_working_texture_dimensions(
            &engine,
            PROJECT_RESET_TEXTURE_EXTENT,
            PROJECT_RESET_TEXTURE_EXTENT,
        );
        assert_eq!(
            (engine.metrics.width, engine.metrics.height),
            (PROJECT_RESET_TEXTURE_EXTENT, PROJECT_RESET_TEXTURE_EXTENT)
        );
        assert!(engine.layer_cache.is_empty());
        assert!(engine.layer_contrib.is_empty());
        assert!(engine.dirty.is_empty());
        assert!(engine.dirty_tiles().is_empty());
        assert_eq!(engine.current, 0);

        let next_metrics = HeightfieldMetrics::new(32, 48, 320.0, 480.0);
        let result = engine
            .evaluate(
                &gpu.device,
                &gpu.queue,
                &LayerStack::new(),
                &[],
                next_metrics,
                PreviewQuality::Draft,
                false,
                None,
            )
            .expect("empty evaluation after reset");

        assert_eq!((result.width, result.height), (32, 48));
        assert_working_texture_dimensions(&engine, 32, 48);
    }
}
