//! GPU pipeline construction and shader-side data contracts.

use super::*;

mod amplify;
mod common;
mod dispatch;
mod filters;
mod generators;
mod sculpt;
mod simulations;

#[allow(clippy::too_many_arguments)]
pub(super) fn record_copy_views_region(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pipe: &Pipe,
    source: &wgpu::TextureView,
    destination: &wgpu::TextureView,
    width: u32,
    height: u32,
    region: (u32, u32, u32, u32),
) {
    let (region_x, region_y, region_w, region_h) = region;
    if region_w == 0 || region_h == 0 {
        return;
    }
    let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("compiled-plan-legacy-copy-uniform"),
        contents: bytemuck::bytes_of(&CopyU {
            width,
            height,
            region_x,
            region_y,
            region_w,
            region_h,
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compiled-plan-legacy-copy-bind-group"),
        layout: &pipe.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(source),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(destination),
            },
        ],
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("compiled-plan-legacy-copy"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipe.pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(region_w.div_ceil(8), region_h.div_ceil(8), 1);
}
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
    warp_strength: f32,
    warp_frequency: f32,
    cell_jitter: f32,
    height_per_cell: f32,
}

#[repr(u32)]
#[derive(Clone, Copy)]
enum NoiseKernelMode {
    /// Existing portable value-noise preview. Its output is already public and
    /// parity-ratcheted, so the new CPU-aligned modes must not rewrite it.
    LegacyValue = 0,
    Perlin = 1,
    Fbm = 2,
    Ridged = 3,
    DomainWarp = 4,
    VoronoiRegions = 5,
}

#[derive(Clone, Copy)]
struct NoiseDispatch {
    noise_type: u32,
    mode: NoiseKernelMode,
    warp_strength: f32,
    warp_frequency: f32,
    cell_jitter: f32,
    height_per_cell: f32,
}

impl NoiseDispatch {
    const fn new(noise_type: u32, mode: NoiseKernelMode) -> Self {
        Self {
            noise_type,
            mode,
            warp_strength: 0.0,
            warp_frequency: 0.0,
            cell_jitter: 0.0,
            height_per_cell: 0.0,
        }
    }

    const fn domain_warp(warp_strength: f32, warp_frequency: f32) -> Self {
        Self {
            noise_type: 1,
            mode: NoiseKernelMode::DomainWarp,
            warp_strength,
            warp_frequency,
            cell_jitter: 0.0,
            height_per_cell: 0.0,
        }
    }

    const fn voronoi_regions(cell_jitter: f32, height_per_cell: f32) -> Self {
        Self {
            noise_type: 0,
            mode: NoiseKernelMode::VoronoiRegions,
            warp_strength: 0.0,
            warp_frequency: 0.0,
            cell_jitter,
            height_per_cell,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlendU {
    width: u32,
    height: u32,
    opacity: f32,
    mode: u32,
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
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
    _p0: f32,
    _p1: f32,
    _p2: f32,
    _p3: f32,
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
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SculptStrokesU {
    width: u32,
    height: u32,
    world_x: f32,
    world_z: f32,
    /// Contiguous stroke range this dispatch stamps: `[stroke_lo, stroke_hi)`.
    /// The edited pass shares the layout with `lo = 0`, `hi = stroke_count`.
    stroke_lo: u32,
    stroke_hi: u32,
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SculptSmoothU {
    width: u32,
    height: u32,
    world_x: f32,
    world_z: f32,
    stroke_index: u32,
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
    _pad0: u32,
}

/// Reduce pass uniform: measures one Flatten stroke's footprint mean. `stroke_index`
/// selects the header; workgroups address their partial slot via `num_workgroups`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SculptReduceU {
    width: u32,
    height: u32,
    world_x: f32,
    world_z: f32,
    stroke_index: u32,
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
}

/// Resolve pass uniform: folds the reduce partials into `targets[target_index]`,
/// falling back to `fallback` (the stroke's `target_height`) when the footprint
/// carried no weight — the exact CPU `flatten_target_for` degenerate branch.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SculptResolveU {
    num_partials: u32,
    target_index: u32,
    fallback: f32,
    _p0: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SculptReconcileU {
    width: u32,
    height: u32,
    reconcile: f32,
    _p0: f32,
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PathU {
    width: u32,
    height: u32,
    world_x: f32,
    world_z: f32,
    point_count: u32,
    carve: u32,
    seed: u32,
    _pad0: u32,
    base_width: f32,
    falloff: f32,
    noise_strength: f32,
    noise_scale: f32,
    height_offset: f32,
    profile: f32,
    _pad1: f32,
    _pad2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PolygonHeightU {
    width: u32,
    height: u32,
    world_x: f32,
    world_z: f32,
    point_count: u32,
    mode: u32,
    carve: u32,
    _pad0: u32,
    target_height: f32,
    falloff: f32,
    _pad1: f32,
    _pad2: f32,
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
struct PlateauU {
    width: u32,
    height: u32,
    low: f32,
    high: f32,
    soft: f32,
    _pad: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct IslandU {
    width: u32,
    height: u32,
    world_x: f32,
    world_z: f32,
    seed: u32,
    archetype: u32,
    _pad_u: [u32; 2],
    center_u: f32,
    center_v: f32,
    rotation_deg: f32,
    radius: f32,
    aspect: f32,
    sea_level: f32,
    ocean_floor: f32,
    mountain_height: f32,
    shelf_width: f32,
    shelf_depth: f32,
    beach_width: f32,
    beach_height: f32,
    reef_width: f32,
    reef_depth: f32,
    coastline_warp: f32,
    coastline_frequency: f32,
    mountain_power: f32,
    ridge_strength: f32,
    ridge_frequency: f32,
    lagoon_radius: f32,
    _pad_f: [f32; 4],
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
struct StreamPowerU {
    width: u32,
    height: u32,
    k: f32,
    m: f32,
    n: f32,
    dt: f32,
    uplift: f32,
    base_level: f32,
    cell_area: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AmplifyDownsampleU {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    area_mode: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AmplifyBlendU {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    hardness: f32,
    ridge_lock: f32,
    lock_strength: f32,
    detail_boost: f32,
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
    flow_threshold: f32,
    wall_steepness: f32,
    valley_floor: f32,
    talus_mix: f32,
    top_smoothness: f32,
    riser_sharpness: f32,
    lacunarity: f32,
    persistence: f32,
    octaves: u32,
    voronoi_feature: u32,
    tileable: u32,
    _pad_params: u32,
    crater_radius: f32,
    dz: f32,
    _pad_metric0: f32,
    _pad_metric1: f32,
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EffectRangeU {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HeightmapSampleU {
    width: u32,
    height: u32,
    source_width: u32,
    source_height: u32,
    mode: u32,
    _pad0: u32,
    height_scale: f32,
    height_offset: f32,
    world_x: f32,
    world_z: f32,
    offset_x: f32,
    offset_z: f32,
    inv_scale: f32,
    sin_t: f32,
    cos_t: f32,
    blend_size: f32,
    blend_roundness: f32,
    _pad1: [f32; 3],
}

pub(super) struct Pipe {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

pub(super) fn make_pipe(
    device: &wgpu::Device,
    label: &str,
    wgsl: &str,
    bgl: wgpu::BindGroupLayout,
) -> Pipe {
    terra_core::shader_progress::record_shader_compiled();
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

pub(super) fn storage_write_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

pub(super) fn storage_write_rgba_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

pub(super) fn tex_read_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

pub(super) fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

pub(super) fn storage_read_buffer_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(super) fn storage_rw_buffer_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
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
                tex_read_entry(4),
                storage_write_entry(5),
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
                storage_rw_buffer_entry(11),
            ],
        });
        let blur_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blur-bgl"),
            entries: &[uniform_entry(0), tex_read_entry(1), storage_write_entry(2)],
        });
        let terrace_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrace-bgl"),
            entries: &[
                uniform_entry(0),
                tex_read_entry(1),
                storage_write_entry(2),
                storage_read_buffer_entry(3),
            ],
        });
        let ramp_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ramp-bgl"),
            entries: &[uniform_entry(0), storage_write_entry(1)],
        });
        let shapes_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shapes-bgl"),
            entries: &[uniform_entry(0), storage_write_entry(1)],
        });
        let island_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("island-bgl"),
            entries: &[uniform_entry(0), storage_write_entry(1)],
        });
        let plateau_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("plateau-bgl"),
            entries: &[uniform_entry(0), tex_read_entry(1), storage_write_entry(2)],
        });
        let path_height_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("path-height-bgl"),
            entries: &[
                uniform_entry(0),
                tex_read_entry(1),
                storage_read_buffer_entry(2),
                storage_write_entry(3),
            ],
        });
        let polygon_height_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("polygon-height-bgl"),
                entries: &[
                    uniform_entry(0),
                    tex_read_entry(1),
                    storage_read_buffer_entry(2),
                    storage_write_entry(3),
                ],
            });
        let heightmap_sample_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("heightmap-sample-bgl"),
                entries: &[
                    uniform_entry(0),
                    tex_read_entry(1),
                    storage_write_entry(2),
                    storage_write_entry(3),
                ],
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
        let stream_power_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stream-power-bgl"),
            entries: &[
                uniform_entry(0),
                tex_read_entry(1),
                tex_read_entry(2),
                tex_read_entry(3),
                storage_write_entry(4),
            ],
        });
        let amplify_downsample_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("amplify-downsample-bgl"),
                entries: &[uniform_entry(0), tex_read_entry(1), storage_write_entry(2)],
            });
        let amplify_upsample_blend_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("amplify-upsample-blend-bgl"),
                entries: &[
                    uniform_entry(0),
                    tex_read_entry(1),
                    tex_read_entry(2),
                    storage_write_entry(3),
                ],
            });

        let fill = make_pipe(
            device,
            "fill",
            include_str!("../../shaders/fill.wgsl"),
            fill_bgl,
        );
        let noise = make_pipe(
            device,
            "noise",
            include_str!("../../shaders/noise.wgsl"),
            noise_bgl,
        );
        let blend = make_pipe(
            device,
            "blend",
            include_str!("../../shaders/blend.wgsl"),
            blend_bgl,
        );
        let copy = make_pipe(
            device,
            "copy",
            include_str!("../../shaders/copy.wgsl"),
            copy_bgl,
        );
        let thermal = make_pipe(
            device,
            "thermal",
            include_str!("../../shaders/thermal_tex.wgsl"),
            thermal_bgl,
        );
        let thermal_apply = make_pipe(
            device,
            "thermal-apply",
            include_str!("../../shaders/thermal_apply.wgsl"),
            thermal_apply_bgl,
        );
        let hydraulic_outflow = make_pipe(
            device,
            "hydraulic-outflow",
            include_str!("../../shaders/hydraulic_outflow.wgsl"),
            hydraulic_outflow_bgl,
        );
        let hydraulic = make_pipe(
            device,
            "hydraulic",
            include_str!("../../shaders/hydraulic_tex.wgsl"),
            hydraulic_bgl,
        );
        let blur = make_pipe(
            device,
            "blur",
            include_str!("../../shaders/blur.wgsl"),
            blur_bgl,
        );
        let terrace = make_pipe(
            device,
            "terrace",
            include_str!("../../shaders/terrace.wgsl"),
            terrace_bgl,
        );
        let ramp = make_pipe(
            device,
            "ramp",
            include_str!("../../shaders/ramp.wgsl"),
            ramp_bgl,
        );
        let shapes = make_pipe(
            device,
            "shapes",
            include_str!("../../shaders/shapes.wgsl"),
            shapes_bgl,
        );
        let island = make_pipe(
            device,
            "island",
            include_str!("../../shaders/island.wgsl"),
            island_bgl,
        );
        let plateau = make_pipe(
            device,
            "plateau",
            include_str!("../../shaders/plateau.wgsl"),
            plateau_bgl,
        );
        let path_height = make_pipe(
            device,
            "path-height",
            include_str!("../../shaders/path_height.wgsl"),
            path_height_bgl,
        );
        let polygon_height = make_pipe(
            device,
            "polygon-height",
            include_str!("../../shaders/polygon_height.wgsl"),
            polygon_height_bgl,
        );
        let heightmap_sample = make_pipe(
            device,
            "heightmap-sample",
            include_str!("../../shaders/heightmap_sample.wgsl"),
            heightmap_sample_bgl,
        );
        let river_accum = make_pipe(
            device,
            "river-accum",
            include_str!("../../shaders/river_accum.wgsl"),
            river_accum_bgl,
        );
        let river_carve = make_pipe(
            device,
            "river-carve",
            include_str!("../../shaders/river_carve.wgsl"),
            river_carve_bgl,
        );
        let stream_power = make_pipe(
            device,
            "stream-power-incision",
            include_str!("../../shaders/stream_power_incision.wgsl"),
            stream_power_bgl,
        );
        let amplify_downsample = make_pipe(
            device,
            "amplify-downsample",
            include_str!("../../shaders/amplify_downsample.wgsl"),
            amplify_downsample_bgl,
        );
        let amplify_upsample_blend = make_pipe(
            device,
            "amplify-upsample-blend",
            include_str!("../../shaders/amplify_upsample_blend.wgsl"),
            amplify_upsample_blend_bgl,
        );
        let effect_filter_range_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("effect-filter-range-bgl"),
                entries: &[
                    uniform_entry(0),
                    tex_read_entry(1),
                    storage_rw_buffer_entry(2),
                ],
            });
        let effect_filter_range = make_pipe(
            device,
            "effect-filter-range",
            include_str!("../../shaders/effect_filter_range.wgsl"),
            effect_filter_range_bgl,
        );
        let effect_filter_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("effect-filter-bgl"),
            entries: &[
                uniform_entry(0),
                tex_read_entry(1),
                storage_write_entry(2),
                storage_read_buffer_entry(3),
            ],
        });
        let effect_filter = make_pipe(
            device,
            "effect-filter",
            include_str!("../../shaders/effect_filter.wgsl"),
            effect_filter_bgl,
        );
        let sculpt_strokes_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sculpt-strokes-bgl"),
                entries: &[
                    uniform_entry(0),
                    tex_read_entry(1), // src_original (base neighborhood)
                    tex_read_entry(2), // running_in (chained height)
                    storage_read_buffer_entry(3), // headers
                    storage_read_buffer_entry(4), // points
                    storage_read_buffer_entry(5), // targets (Flatten footprint means)
                    storage_write_entry(6), // stamp_out
                ],
            });
        let sculpt_strokes_edited_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sculpt-strokes-edited-bgl"),
                entries: &[
                    uniform_entry(0),
                    storage_read_buffer_entry(1), // headers
                    storage_read_buffer_entry(2), // points
                    storage_write_entry(3),       // edited_out
                ],
            });
        let sculpt_strokes_smooth_curvature_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sculpt-strokes-smooth-curvature-bgl"),
                entries: &[
                    uniform_entry(0),
                    tex_read_entry(1),            // running height
                    storage_read_buffer_entry(2), // headers
                    storage_read_buffer_entry(3), // points
                    storage_write_entry(4),       // horizontal blur
                ],
            });
        let sculpt_strokes_smooth_apply_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sculpt-strokes-smooth-apply-bgl"),
                entries: &[
                    uniform_entry(0),
                    tex_read_entry(1),            // running height
                    tex_read_entry(2),            // horizontal blur
                    storage_read_buffer_entry(3), // headers
                    storage_read_buffer_entry(4), // points
                    storage_write_entry(5),       // smoothed height
                ],
            });
        let sculpt_strokes_flatten_reduce_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sculpt-strokes-flatten-reduce-bgl"),
                entries: &[
                    uniform_entry(0),
                    tex_read_entry(1), // running field entering the stroke
                    storage_read_buffer_entry(2), // headers
                    storage_read_buffer_entry(3), // points
                    storage_rw_buffer_entry(4), // partials
                ],
            });
        let sculpt_strokes_flatten_resolve_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sculpt-strokes-flatten-resolve-bgl"),
                entries: &[
                    uniform_entry(0),
                    storage_read_buffer_entry(1), // partials
                    storage_rw_buffer_entry(2),   // targets
                ],
            });
        let sculpt_strokes_reconcile_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sculpt-strokes-reconcile-bgl"),
                entries: &[
                    uniform_entry(0),
                    tex_read_entry(1),
                    tex_read_entry(2),
                    storage_write_entry(3),
                ],
            });
        let sculpt_strokes = make_pipe(
            device,
            "sculpt-strokes",
            include_str!("../../shaders/sculpt_strokes.wgsl"),
            sculpt_strokes_bgl,
        );
        let sculpt_strokes_edited = make_pipe(
            device,
            "sculpt-strokes-edited",
            include_str!("../../shaders/sculpt_strokes_edited.wgsl"),
            sculpt_strokes_edited_bgl,
        );
        let sculpt_strokes_smooth_curvature = make_pipe(
            device,
            "sculpt-strokes-smooth-curvature",
            include_str!("../../shaders/sculpt_strokes_smooth_curvature.wgsl"),
            sculpt_strokes_smooth_curvature_bgl,
        );
        let sculpt_strokes_smooth_apply = make_pipe(
            device,
            "sculpt-strokes-smooth-apply",
            include_str!("../../shaders/sculpt_strokes_smooth_apply.wgsl"),
            sculpt_strokes_smooth_apply_bgl,
        );
        let sculpt_strokes_flatten_reduce = make_pipe(
            device,
            "sculpt-strokes-flatten-reduce",
            include_str!("../../shaders/sculpt_strokes_flatten_reduce.wgsl"),
            sculpt_strokes_flatten_reduce_bgl,
        );
        let sculpt_strokes_flatten_resolve = make_pipe(
            device,
            "sculpt-strokes-flatten-resolve",
            include_str!("../../shaders/sculpt_strokes_flatten_resolve.wgsl"),
            sculpt_strokes_flatten_resolve_bgl,
        );
        let sculpt_strokes_reconcile = make_pipe(
            device,
            "sculpt-strokes-reconcile",
            include_str!("../../shaders/sculpt_strokes_reconcile.wgsl"),
            sculpt_strokes_reconcile_bgl,
        );
        let ping = HeightTex::new(device, "ping", w, w);
        let pong = HeightTex::new(device, "pong", w, w);
        let published_height = HeightTex::new(device, "published-height", w, w);
        let layer_tex = HeightTex::new(device, "layer", w, w);
        let mask_ones = HeightTex::new(device, "mask-ones", w, w);
        let unit_mask = HeightTex::new(device, "unit-mask", w, w);
        let stamp_mask = HeightTex::new(device, "stamp-mask", w, w);
        let sim_side = w;
        let hardness = HeightTex::new(device, "hardness", sim_side, sim_side);
        let water_a = HeightTex::new(device, "water-a", sim_side, sim_side);
        let water_b = HeightTex::new(device, "water-b", sim_side, sim_side);
        let delta = HeightTex::new(device, "thermal-delta", sim_side, sim_side);
        let sed_a = HeightTex::new(device, "sed-a", sim_side, sim_side);
        let sed_b = HeightTex::new(device, "sed-b", sim_side, sim_side);
        let rainfall = HeightTex::new(device, "rainfall", sim_side, sim_side);
        let loose_sediment = HeightTex::new(device, "loose-sediment", sim_side, sim_side);
        let outflow = RgbaTex::new(device, "hydraulic-outflow", sim_side, sim_side);
        let amplify_a = HeightTex::new(device, "amplify-a", sim_side, sim_side);
        let amplify_b = HeightTex::new(device, "amplify-b", sim_side, sim_side);
        let sculpt_stamp = HeightTex::new(device, "sculpt-stamp", w, w);
        let sculpt_stamp_b = HeightTex::new(device, "sculpt-stamp-b", w, w);
        let sculpt_edited = HeightTex::new(device, "sculpt-edited", w, w);
        let sculpt_smooth_curvature = HeightTex::new(device, "sculpt-smooth-curvature", w, w);
        let effect_filter_range_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("effect-filter-range"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let simulation_invalid_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simulation-invalid-state"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self {
            plan_operations: GpuPlanOperations::new(device),
            plan_resources: GpuPlanResourceCache::default(),
            device_generation: NEXT_DEVICE_GENERATION.fetch_add(1, Ordering::Relaxed),
            output_resource_incarnation: GpuResourceIncarnation(1),
            next_output_id: 1,
            next_submission_serial: 1,
            last_output_identity: None,
            active_plan_revision: None,
            deferred_plan_resume: None,
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
            island,
            plateau,
            river_accum,
            river_carve,
            stream_power,
            amplify_downsample,
            amplify_upsample_blend,
            effect_filter_range,
            effect_filter,
            sculpt_strokes,
            sculpt_strokes_edited,
            sculpt_strokes_smooth_curvature,
            sculpt_strokes_smooth_apply,
            sculpt_strokes_flatten_reduce,
            sculpt_strokes_flatten_resolve,
            sculpt_strokes_reconcile,
            path_height,
            polygon_height,
            heightmap_sample,
            uniform_pool: UniformPool::new(device, 64),
            ping,
            pong,
            published_height,
            layer_tex,
            mask_ones,
            unit_mask,
            stamp_mask,
            hardness,
            water_a,
            water_b,
            delta,
            sed_a,
            sed_b,
            rainfall,
            loose_sediment,
            outflow,
            amplify_a,
            amplify_b,
            sculpt_stamp,
            sculpt_stamp_b,
            sculpt_edited,
            sculpt_smooth_curvature,
            effect_filter_range_buffer,
            simulation_invalid_state_buffer,
            layer_cache: HashMap::new(),
            stamp_mask_cache: HashMap::new(),
            source_rasters: HashMap::new(),
            stroke_runtime: HashMap::new(),
            #[cfg(test)]
            source_upload_count: 0,
            dirty: HashSet::new(),
            metrics: HeightfieldMetrics {
                width: w,
                height: w,
                world_size_x: 1000.0,
                world_size_z: 1000.0,
                tile_size: w,
                halo: 0,
            },
            tile_sample_window: None,
            approx_range: (0.0, 120.0),
            current: 0,
            tile_sched: TileScheduler::new(),
            last_dirty_rect: None,
            last_quality: None,
            max_sim_iters_per_tick: 8,
            last_graph: GpuComputeGraph::default(),
            #[cfg(test)]
            executed_kernels: Vec::new(),
            last_eval_stats: GpuEvalStats::default(),
            last_plan_operation_trace: Vec::new(),
            evaluation_timer: GpuEvaluationTimer::try_new(device),
            pending_evaluation_trace: None,
            retired_refinement_completions: Vec::new(),
            #[cfg(test)]
            executed_plan_operations: Vec::new(),
        }
    }
}
