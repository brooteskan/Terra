//! GPU evaluator resources implementation.

use super::*;

/// One stroke's GPU header. Layout mirrors `StrokeHeader` in
/// `shaders/sculpt_strokes.wgsl` (48 bytes, 8-byte aligned for the trailing
/// `vec2<f32>` bbox fields); `points` are uploaded separately as `vec4<f32>`.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Pod, Zeroable)]
pub(super) struct StrokeHeaderGpu {
    pub(super) kind: u32,
    pub(super) first_point: u32,
    pub(super) point_count: u32,
    pub(super) _pad0: u32,
    pub(super) radius_m: f32,
    pub(super) strength: f32,
    pub(super) target_height: f32,
    pub(super) falloff: f32,
    pub(super) bbox_min: [f32; 2],
    pub(super) bbox_max: [f32; 2],
}

pub(super) struct StrokeRuntimeBuffers {
    pub(super) headers: wgpu::Buffer,
    pub(super) points: wgpu::Buffer,
    pub(super) header_capacity: usize,
    pub(super) point_capacity: usize,
    pub(super) uploaded_headers: Vec<StrokeHeaderGpu>,
    pub(super) uploaded_points: Vec<[f32; 4]>,
}

pub(super) const INITIAL_STROKE_HEADER_CAPACITY: usize = 8;
pub(super) const INITIAL_STROKE_POINT_CAPACITY: usize = 64;

/// Alias-collapsed kind id shared with `shaders/sculpt_strokes.wgsl`. Only kinds
/// the planner admits are ever uploaded — the per-sample maps, the base-3x3
/// Smooth/Pinch/Coastline, and the footprint-mean Flatten (#117), whose per-stroke
/// target the reduce/resolve passes precompute into the `targets` buffer.
pub(super) fn stroke_kind_gpu_id(kind: SculptStrokeKind) -> u32 {
    match kind {
        SculptStrokeKind::Raise => 0,
        SculptStrokeKind::Lower => 1,
        SculptStrokeKind::Ridge | SculptStrokeKind::MountainStamp => 2,
        SculptStrokeKind::Valley | SculptStrokeKind::ValleyStamp | SculptStrokeKind::RiverPath => 3,
        SculptStrokeKind::Terrace => 4,
        SculptStrokeKind::Roughness | SculptStrokeKind::Noise => 5,
        SculptStrokeKind::Inflate => 6,
        SculptStrokeKind::PlateauStamp => 7,
        SculptStrokeKind::CraterStamp => 8,
        SculptStrokeKind::HeightStamp => 9,
        SculptStrokeKind::Erode | SculptStrokeKind::EncourageErosion => 10,
        // Uplift / Hardness / Sediment / Protect contribute only aux on the CPU;
        // their height is unchanged, but they still mark the edit region.
        SculptStrokeKind::Uplift
        | SculptStrokeKind::Hardness
        | SculptStrokeKind::Sediment
        | SculptStrokeKind::Protect => 11,
        // Smooth pulls each sample toward the clamped 3x3 mean of the layer input
        // (`src`); Pinch applies a bounded, strength-weighted 1.25 gain;
        // Coastline lowers the sample and blends it toward that mean under a
        // weight gate. The stamp kernel reads that neighborhood directly
        // (#114, #115, #116).
        SculptStrokeKind::Smooth => 12,
        SculptStrokeKind::Pinch => 13,
        SculptStrokeKind::Coastline => 14,
        // Flatten settles toward the brush-weighted mean of the running field over
        // its footprint; the reduce/resolve passes compute that scalar per stroke
        // into `targets`, and the stamp arm applies `h + (target - h) * w` (#117).
        SculptStrokeKind::Flatten => 15,
    }
}

/// Flatten a stroke set into GPU headers + a shared `vec4` point pool. Each
/// header carries a world-space footprint (point bbox padded by `radius_m`) so
/// the stamp kernel can cull texts outside the brush; correctness still comes
/// from the per-texel weight test. Both vectors are kept non-empty so the storage
/// bindings are valid even for an empty stroke set (`stroke_count` gates reads).
pub(super) fn build_stroke_buffers(
    strokes: &[&SculptStroke],
    m: &HeightfieldMetrics,
) -> (Vec<StrokeHeaderGpu>, Vec<[f32; 4]>) {
    let sx = m.world_size_x;
    let sz = m.world_size_z;
    let mut headers = Vec::with_capacity(strokes.len());
    let mut points: Vec<[f32; 4]> = Vec::new();
    for stroke in strokes {
        let first_point = points.len() as u32;
        let (mut min_x, mut min_z) = (f32::INFINITY, f32::INFINITY);
        let (mut max_x, mut max_z) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for pt in &stroke.points {
            let wx = pt.u * sx;
            let wz = pt.v * sz;
            min_x = min_x.min(wx);
            max_x = max_x.max(wx);
            min_z = min_z.min(wz);
            max_z = max_z.max(wz);
            points.push([pt.u, pt.v, pt.pressure, 0.0]);
        }
        let r = stroke.radius_m;
        // A stroke with no points has no footprint: an inverted bbox (min > max)
        // culls every texel, exactly as the CPU produces no contribution.
        let (bbox_min, bbox_max) = if stroke.points.is_empty() {
            ([1.0, 1.0], [-1.0, -1.0])
        } else {
            ([min_x - r, min_z - r], [max_x + r, max_z + r])
        };
        headers.push(StrokeHeaderGpu {
            kind: stroke_kind_gpu_id(stroke.kind),
            first_point,
            point_count: stroke.points.len() as u32,
            _pad0: 0,
            radius_m: stroke.radius_m,
            strength: stroke.strength,
            target_height: stroke.target_height,
            falloff: stroke.falloff,
            bbox_min,
            bbox_max,
        });
    }
    if headers.is_empty() {
        // Placeholder so the storage buffer is bindable; never read (count == 0).
        headers.push(StrokeHeaderGpu {
            kind: 11,
            first_point: 0,
            point_count: 0,
            _pad0: 0,
            radius_m: 0.0,
            strength: 0.0,
            target_height: 0.0,
            falloff: 0.0,
            bbox_min: [1.0, 1.0],
            bbox_max: [-1.0, -1.0],
        });
    }
    if points.is_empty() {
        points.push([0.0, 0.0, 0.0, 0.0]);
    }
    (headers, points)
}

pub(super) fn make_storage_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    bytes: &[u8],
) -> wgpu::Buffer {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (bytes.len() as u64).max(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, bytes);
    buf
}

pub(super) fn make_runtime_storage_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    element_size: usize,
    capacity: usize,
    bytes: &[u8],
) -> wgpu::Buffer {
    let size = element_size.saturating_mul(capacity.max(1)).max(16) as u64;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !bytes.is_empty() {
        queue.write_buffer(&buffer, 0, bytes);
    }
    buffer
}

#[derive(Clone, Copy)]
pub(super) enum TexSlot {
    Ping,
    Pong,
    Layer,
    MaskOnes,
    UnitMask,
    StampMask,
    Hardness,
    WaterA,
    WaterB,
    SedA,
    SedB,
    Rainfall,
    LooseSediment,
    SculptStamp,
}

pub(super) struct HeightTex {
    pub(super) texture: wgpu::Texture,
    pub(super) view: wgpu::TextureView,
    pub(super) width: u32,
    pub(super) height: u32,
}

impl HeightTex {
    pub(super) fn new(device: &wgpu::Device, label: &str, width: u32, height: u32) -> Self {
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

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct SourceFingerprint {
    pub(super) len: u64,
    pub(super) modified: Option<SystemTime>,
}

pub(super) struct SourceRasterTex {
    pub(super) tex: HeightTex,
    pub(super) fingerprint: Option<SourceFingerprint>,
}

/// RGBA float texture for hydraulic outflow fluxes (L,R,D,U).
pub(super) struct RgbaTex {
    pub(super) _texture: wgpu::Texture,
    pub(super) view: wgpu::TextureView,
}

impl RgbaTex {
    pub(super) fn new(device: &wgpu::Device, label: &str, width: u32, height: u32) -> Self {
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

/// Ring of small uniform buffers so many dispatches can share one submit.
pub(super) struct UniformPool {
    pub(super) buffers: Vec<wgpu::Buffer>,
    pub(super) next: usize,
}

impl UniformPool {
    const SLOT_SIZE: u64 = 256;

    pub(super) fn new(device: &wgpu::Device, capacity: usize) -> Self {
        let mut buffers = Vec::with_capacity(capacity);
        for i in 0..capacity {
            buffers.push(Self::make_slot(device, i));
        }
        Self { buffers, next: 0 }
    }

    pub(super) fn make_slot(device: &wgpu::Device, index: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("gpu-engine-u-{index}")),
            size: Self::SLOT_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub(super) fn reset(&mut self) {
        self.next = 0;
    }

    pub(super) fn write<T: Pod>(
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

impl GpuTerrainEngine {
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

    /// Drop all project-owned GPU caches and replace project-sized working textures
    /// with the small resident baseline so a new/opened document starts clean.
    pub fn reset_project_state(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.plan_resources.clear_current();
        self.active_plan_revision = None;
        self.deferred_plan_resume = None;
        self.layer_cache.clear();
        self.stamp_mask_cache.clear();
        self.source_rasters.clear();
        self.stroke_runtime.clear();
        self.dirty.clear();
        self.last_dirty_rect = None;
        self.last_quality = None;
        self.last_graph = terra_gpu::graph::GpuComputeGraph::default();
        self.last_eval_stats = GpuEvalStats::default();
        self.last_plan_operation_trace.clear();
        self.last_output_identity = None;
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
        self.fill_slot(device, queue, &mut encoder, TexSlot::UnitMask, 1.0);
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

    pub(super) fn ensure_size(&mut self, device: &wgpu::Device, metrics: HeightfieldMetrics) {
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
        self.output_resource_incarnation = GpuResourceIncarnation(
            self.output_resource_incarnation
                .0
                .checked_add(1)
                .expect("GPU output resource incarnation exhausted"),
        );
        self.layer_tex = HeightTex::new(device, "layer", w, h);
        self.mask_ones = HeightTex::new(device, "mask-ones", w, h);
        self.unit_mask = HeightTex::new(device, "unit-mask", w, h);
        self.stamp_mask = HeightTex::new(device, "stamp-mask", w, h);
        self.hardness = HeightTex::new(device, "hardness", w, h);
        self.water_a = HeightTex::new(device, "water-a", w, h);
        self.water_b = HeightTex::new(device, "water-b", w, h);
        self.delta = HeightTex::new(device, "thermal-delta", w, h);
        self.sed_a = HeightTex::new(device, "sed-a", w, h);
        self.sed_b = HeightTex::new(device, "sed-b", w, h);
        self.rainfall = HeightTex::new(device, "rainfall", w, h);
        self.loose_sediment = HeightTex::new(device, "loose-sediment", w, h);
        self.outflow = RgbaTex::new(device, "hydraulic-outflow", w, h);
        self.amplify_a = HeightTex::new(device, "amplify-a", w, h);
        self.amplify_b = HeightTex::new(device, "amplify-b", w, h);
        self.sculpt_stamp = HeightTex::new(device, "sculpt-stamp", w, h);
        self.sculpt_stamp_b = HeightTex::new(device, "sculpt-stamp-b", w, h);
        self.sculpt_edited = HeightTex::new(device, "sculpt-edited", w, h);
        self.layer_cache.clear();
        self.stroke_runtime.clear();
        self.dirty.clear();
    }

    pub(super) fn swap_current(&mut self) {
        self.current = 1 - self.current;
    }

    /// Write uniforms into the next pool slot and return that buffer.
    /// Each dispatch must bind its own slot ÔÇö wgpu applies all `queue.write_buffer`
    /// transfers before the command buffer, so a single shared buffer would make
    /// every pass see only the last write.
    pub(super) fn write_uniform<T: Pod>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &T,
    ) -> wgpu::Buffer {
        self.uniform_pool.write(device, queue, data)
    }

    pub(super) fn view_of(&self, slot: TexSlot) -> &wgpu::TextureView {
        match slot {
            TexSlot::Ping => &self.ping.view,
            TexSlot::Pong => &self.pong.view,
            TexSlot::Layer => &self.layer_tex.view,
            TexSlot::MaskOnes => &self.mask_ones.view,
            TexSlot::UnitMask => &self.unit_mask.view,
            TexSlot::StampMask => &self.stamp_mask.view,
            TexSlot::Hardness => &self.hardness.view,
            TexSlot::WaterA => &self.water_a.view,
            TexSlot::WaterB => &self.water_b.view,
            TexSlot::SedA => &self.sed_a.view,
            TexSlot::SedB => &self.sed_b.view,
            TexSlot::Rainfall => &self.rainfall.view,
            TexSlot::LooseSediment => &self.loose_sediment.view,
            TexSlot::SculptStamp => &self.sculpt_stamp.view,
        }
    }
}
