//! GPU-resident height + normal textures with double buffering.
//!
//! Large slot reallocations should retire old textures via [`crate::retirement::DeferredGpuRetirement`]
//! instead of dropping them immediately on the upload path.

use bytemuck::{Pod, Zeroable};
use terra_core::heightfield::Heightfield;
use terra_core::mask::MaskField;
use terra_core::tiling::SampleRect;

/// Small resident texture extent used while no project is active.
/// Existing upload/presentation paths restore the next document's dimensions.
const PROJECT_RESET_TEXTURE_EXTENT: u32 = 8;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NormalUniforms {
    width: u32,
    height: u32,
    dx: f32,
    dz: f32,
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
}

struct HeightSlot {
    height: wgpu::Texture,
    height_view: wgpu::TextureView,
    /// Kept alive; sampled via `normal_view`.
    normal: wgpu::Texture,
    normal_view: wgpu::TextureView,
    /// Cached normal-pass bind group for this slot (rebuilt on resize).
    normal_bind: Option<wgpu::BindGroup>,
    width: u32,
    height_px: u32,
}

impl HeightSlot {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let height_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("height-r32"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let normal_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("normal-rgba16"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        Self {
            height_view: height_tex.create_view(&wgpu::TextureViewDescriptor::default()),
            normal_view: normal_tex.create_view(&wgpu::TextureViewDescriptor::default()),
            height: height_tex,
            normal: normal_tex,
            normal_bind: None,
            width,
            height_px: height,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use terra_core::heightfield::HeightfieldMetrics;

    fn assert_slot_dimensions(height: &HeightGpu, width: u32, height_px: u32) {
        assert_eq!(height.tex_size, (width, height_px));
        for slot in &height.slots {
            assert_eq!((slot.width, slot.height_px), (width, height_px));
            assert_eq!(
                (slot.height.width(), slot.height.height()),
                (width, height_px)
            );
            assert_eq!(
                (slot.normal.width(), slot.normal.height()),
                (width, height_px)
            );
        }
    }

    /// Revert check for #35: reset must replace, rather than retain, project-sized slots.
    #[test]
    fn project_reset_shrinks_slots_and_defers_old_textures() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let mut height = HeightGpu::new(&gpu.device, 64);

        height.reset_project_state(&gpu.device, &gpu.queue, (1000.0, 750.0));

        assert_slot_dimensions(
            &height,
            PROJECT_RESET_TEXTURE_EXTENT,
            PROJECT_RESET_TEXTURE_EXTENT,
        );
        assert!(height.shared_height_view.is_none());
        assert_eq!(height.retirement.pending(), 4);
        height.tick_retirement(2);
        assert_eq!(height.retirement.pending(), 4);
        height.tick_retirement(3);
        assert_eq!(height.retirement.pending(), 0);
    }

    #[test]
    fn upload_after_project_reset_restores_requested_size() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let mut height = HeightGpu::new(&gpu.device, 64);
        height.reset_project_state(&gpu.device, &gpu.queue, (1000.0, 750.0));

        let metrics = HeightfieldMetrics::new(32, 48, 320.0, 480.0);
        height.upload_and_swap(&gpu.device, &gpu.queue, &Heightfield::zeros(metrics));

        assert_slot_dimensions(&height, 32, 48);
        assert_eq!(height.world_size, (320.0, 480.0));
    }
}

/// Single-channel auxiliary surface map sampled by the terrain material shader.
struct AuxMap {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

impl AuxMap {
    fn new(device: &wgpu::Device, width: u32, height: u32, label: &'static str) -> Self {
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
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            _texture: texture,
            view,
            width,
            height,
        }
    }
}

/// RGBA8 artist placement / biome colour overlay (filterable).
struct RgbaAuxMap {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

impl RgbaAuxMap {
    fn new(device: &wgpu::Device, width: u32, height: u32, label: &'static str) -> Self {
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
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            _texture: texture,
            view,
            width,
            height,
        }
    }
}

/// Double-buffered GPU height/normal pair. Display index is what the viewport samples.
pub struct HeightGpu {
    slots: [HeightSlot; 2],
    pub display: usize,
    write: usize,
    sampler: wgpu::Sampler,
    normal_pipeline: wgpu::ComputePipeline,
    normal_bgl: wgpu::BindGroupLayout,
    normal_uniform: wgpu::Buffer,
    pub height_range: (f32, f32),
    pub world_size: (f32, f32),
    pub tex_size: (u32, u32),
    materials: AuxMap,
    wetness: AuxMap,
    vegetation: AuxMap,
    /// Drainage / SPE flow accumulation for lit channel tinting.
    flow: AuxMap,
    /// Phase H climate overlays uploaded as R32Float (CPU bake / hybrid).
    temperature: AuxMap,
    rainfall: AuxMap,
    snow: AuxMap,
    soil_moisture: AuxMap,
    biomes: AuxMap,
    /// Painted biome placement colours (RGBA8) for 3D tint overlay.
    placement_tint: RgbaAuxMap,
    /// External R32Float height view (GPU engine output) when sharing without copy.
    shared_height_view: Option<wgpu::TextureView>,
    retirement: crate::retirement::DeferredGpuRetirement,
    current_frame: u64,
}

impl HeightGpu {
    pub fn new(device: &wgpu::Device, initial: u32) -> Self {
        let w = initial.max(PROJECT_RESET_TEXTURE_EXTENT);
        let slots = [HeightSlot::new(device, w, w), HeightSlot::new(device, w, w)];
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("height-samp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("normals"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/normals.wgsl").into()),
        });
        let normal_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("normal-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("normal-pl"),
            bind_group_layouts: &[&normal_bgl],
            push_constant_ranges: &[],
        });
        let normal_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("normal-pipe"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let normal_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("normal-u"),
            size: std::mem::size_of::<NormalUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let materials = AuxMap::new(device, 1, 1, "materials-r32");
        let wetness = AuxMap::new(device, 1, 1, "wetness-r32");
        let vegetation = AuxMap::new(device, 1, 1, "vegetation-r32");
        let flow = AuxMap::new(device, 1, 1, "flow-r32");
        let temperature = AuxMap::new(device, 1, 1, "temperature-r32");
        let rainfall = AuxMap::new(device, 1, 1, "rainfall-r32");
        let snow = AuxMap::new(device, 1, 1, "snow-r32");
        let soil_moisture = AuxMap::new(device, 1, 1, "soil-moisture-r32");
        let biomes = AuxMap::new(device, 1, 1, "biomes-r32");
        let placement_tint = RgbaAuxMap::new(device, 1, 1, "placement-tint-rgba8");

        Self {
            slots,
            display: 0,
            write: 1,
            sampler,
            normal_pipeline,
            normal_bgl,
            normal_uniform,
            height_range: (0.0, 1.0),
            world_size: (1.0, 1.0),
            tex_size: (w, w),
            materials,
            wetness,
            vegetation,
            flow,
            temperature,
            rainfall,
            snow,
            soil_moisture,
            biomes,
            placement_tint,
            shared_height_view: None,
            retirement: crate::retirement::DeferredGpuRetirement::new(3),
            current_frame: 0,
        }
    }

    /// Advance deferred texture retirement for the given frame index.
    pub fn tick_retirement(&mut self, frame: u64) {
        self.current_frame = frame;
        self.retirement.tick(frame);
    }

    fn retire_slot(&mut self, slot: HeightSlot) {
        self.retirement.queue(self.current_frame, slot.height);
        self.retirement.queue(self.current_frame, slot.normal);
    }

    fn ensure_size(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.slots[self.write].width == width && self.slots[self.write].height_px == height {
            return;
        }
        let old_write = std::mem::replace(
            &mut self.slots[self.write],
            HeightSlot::new(device, width, height),
        );
        self.retire_slot(old_write);
        if self.slots[self.display].width != width || self.slots[self.display].height_px != height {
            let old_display = std::mem::replace(
                &mut self.slots[self.display],
                HeightSlot::new(device, width, height),
            );
            self.retire_slot(old_display);
        }
        self.tex_size = (width, height);
    }

    /// Upload CPU heightfield into the write slot, compute normals, then swap for display.
    pub fn upload_and_swap(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        hf: &Heightfield,
    ) {
        self.upload_regions_and_swap(device, queue, hf, None);
    }

    /// Upload full field or only `regions` (Wave D dirty tiles), then recompute normals.
    pub fn upload_regions_and_swap(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        hf: &Heightfield,
        regions: Option<&[SampleRect]>,
    ) {
        self.upload_regions_and_swap_with_staging(device, queue, None, hf, regions);
    }

    /// Same as [`Self::upload_regions_and_swap`] but routes large rects through a staging ring.
    pub fn upload_regions_and_swap_with_staging(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mut staging: Option<&mut crate::staging::StagingRing>,
        hf: &Heightfield,
        regions: Option<&[SampleRect]>,
    ) {
        // CPU owns the viewport again — drop any borrowed GPU-engine view so
        // display_height_view() samples the slot we are about to write.
        // Without this, interactive filter stamps / async CPU results upload
        // invisibly while the terrain keeps showing a stale shared texture.
        self.shared_height_view = None;
        let w = hf.metrics.width;
        let h = hf.metrics.height;
        self.ensure_size(device, w, h);
        self.world_size = (hf.metrics.world_size_x, hf.metrics.world_size_z);
        self.height_range = hf.min_max();

        let full = SampleRect { x: 0, y: 0, w, h };
        let rects: Vec<SampleRect> = match regions {
            Some(r) if !r.is_empty() => r.to_vec(),
            _ => vec![full],
        };

        // Prefer tiling samples over a full dense flatten when uploading partial rects.
        let use_dense = rects.len() == 1 && rects[0].w == w && rects[0].h == h;

        // A partial upload writes only its dirty rects into the write slot, which
        // still holds stale content from two frames ago (zeros on its first reuse).
        // Seed the write slot from the current display slot first — mirroring the
        // GPU-to-GPU path's "seed write slot from last display" — so texels outside
        // the dirty rects survive the display swap. Normals are then recomputed over
        // the whole reconstructed field (below) rather than just the dirty union.
        let partial = !use_dense
            && self.slots[self.display].width == w
            && self.slots[self.display].height_px == h;
        if partial {
            let mut seed = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("height-seed-enc"),
            });
            seed.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.slots[self.display].height,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &self.slots[self.write].height,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
            queue.submit(Some(seed.finish()));
        }

        let slot = &self.slots[self.write];
        let dense = if use_dense { Some(hf.to_dense()) } else { None };

        let mut encoder = if staging.is_some() {
            Some(
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("height-staging-enc"),
                }),
            )
        } else {
            None
        };

        for rect in &rects {
            if rect.is_empty() {
                continue;
            }
            let mut packed = Vec::with_capacity((rect.w * rect.h) as usize);
            if let Some(data) = dense.as_ref() {
                for row in rect.y..rect.y + rect.h {
                    let start = (row * w + rect.x) as usize;
                    packed.extend_from_slice(&data[start..start + rect.w as usize]);
                }
            } else {
                for row in rect.y..rect.y + rect.h {
                    for col in rect.x..rect.x + rect.w {
                        packed.push(hf.get(col, row));
                    }
                }
            }
            let origin = wgpu::Origin3d {
                x: rect.x,
                y: rect.y,
                z: 0,
            };
            if let (Some(ring), Some(enc)) = (staging.as_mut(), encoder.as_mut()) {
                ring.write_r32_region(
                    device,
                    queue,
                    Some(enc),
                    &slot.height,
                    origin,
                    rect.w,
                    rect.h,
                    &packed,
                );
            } else {
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &slot.height,
                        mip_level: 0,
                        origin,
                        aspect: wgpu::TextureAspect::All,
                    },
                    bytemuck::cast_slice(&packed),
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(rect.w * 4),
                        rows_per_image: Some(rect.h),
                    },
                    wgpu::Extent3d {
                        width: rect.w,
                        height: rect.h,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }

        if let Some(enc) = encoder.take() {
            queue.submit(Some(enc.finish()));
        }

        let normal_region = if partial {
            // The write slot was seeded from display and patched, so it now holds
            // the complete field — recompute every normal from it, not just the
            // dirty union (whose neighbours would otherwise stay stale).
            full
        } else {
            regions
                .and_then(|r| {
                    let mut iter = r.iter().copied();
                    let first = iter.next()?;
                    Some(iter.fold(first, |a, b| a.union(b)).padded(&hf.metrics, 1))
                })
                .unwrap_or(full)
        };

        self.compute_normals_region_and_swap(
            device,
            queue,
            w,
            h,
            hf.metrics.dx(),
            hf.metrics.dz(),
            normal_region,
        );
    }

    /// GPU→GPU path: copy a height texture into the write slot (no CPU readback).
    pub fn copy_from_texture_and_swap(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src: &wgpu::Texture,
        width: u32,
        height: u32,
        world_size: (f32, f32),
        height_range: (f32, f32),
        dx: f32,
        dz: f32,
    ) {
        self.copy_from_texture_region_and_swap(
            device,
            queue,
            src,
            width,
            height,
            world_size,
            height_range,
            dx,
            dz,
            None,
        );
    }

    pub fn copy_from_texture_region_and_swap(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src: &wgpu::Texture,
        width: u32,
        height: u32,
        world_size: (f32, f32),
        height_range: (f32, f32),
        dx: f32,
        dz: f32,
        region: Option<SampleRect>,
    ) {
        self.shared_height_view = None;
        self.ensure_size(device, width, height);
        self.world_size = world_size;
        let (lo, hi) = if height_range.0 <= height_range.1 {
            height_range
        } else {
            (height_range.1, height_range.0)
        };
        self.height_range = (lo, hi.max(lo + 1e-3));

        let full = SampleRect {
            x: 0,
            y: 0,
            w: width,
            h: height,
        };
        let rect = region.unwrap_or(full);
        let partial = rect.w != width || rect.h != height;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("height-copy-enc"),
        });
        if partial
            && self.slots[self.display].width == width
            && self.slots[self.display].height_px == height
        {
            // Seed write slot from last display, then overlay dirty region.
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.slots[self.display].height,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &self.slots[self.write].height,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: src,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: rect.x,
                    y: rect.y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.slots[self.write].height,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: rect.x,
                    y: rect.y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: rect.w,
                height: rect.h,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        let normal_rect = if partial {
            SampleRect {
                x: rect.x.saturating_sub(1),
                y: rect.y.saturating_sub(1),
                w: (rect.x + rect.w + 1).min(width) - rect.x.saturating_sub(1),
                h: (rect.y + rect.h + 1).min(height) - rect.y.saturating_sub(1),
            }
        } else {
            rect
        };

        self.compute_normals_region_and_swap(device, queue, width, height, dx, dz, normal_rect);
    }

    fn compute_normals_region_and_swap(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        w: u32,
        h: u32,
        dx: f32,
        dz: f32,
        region: SampleRect,
    ) {
        let uniforms = NormalUniforms {
            width: w,
            height: h,
            dx,
            dz,
            region_x: region.x,
            region_y: region.y,
            region_w: region.w.max(1),
            region_h: region.h.max(1),
        };
        queue.write_buffer(&self.normal_uniform, 0, bytemuck::bytes_of(&uniforms));

        let write = self.write;
        if self.slots[write].normal_bind.is_none() {
            let slot = &self.slots[write];
            self.slots[write].normal_bind =
                Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("normal-bg"),
                    layout: &self.normal_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.normal_uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&slot.height_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&slot.normal_view),
                        },
                    ],
                }));
        }
        let bind = self.slots[write]
            .normal_bind
            .as_ref()
            .expect("normal bind group");

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("normal-enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("normals"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.normal_pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups((region.w + 7) / 8, (region.h + 7) / 8, 1);
        }
        queue.submit(Some(encoder.finish()));
        std::mem::swap(&mut self.display, &mut self.write);
    }

    /// Sample an external R32Float height texture in place; normals are computed locally.
    pub fn present_shared_height(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        world_size: (f32, f32),
        height_range: (f32, f32),
        dx: f32,
        dz: f32,
    ) {
        self.shared_height_view = Some(src_view.clone());
        self.tex_size = (width, height);
        self.world_size = world_size;
        let (lo, hi) = if height_range.0 <= height_range.1 {
            height_range
        } else {
            (height_range.1, height_range.0)
        };
        self.height_range = (lo, hi.max(lo + 1e-3));
        self.ensure_size(device, width, height);
        let region = SampleRect {
            x: 0,
            y: 0,
            w: width,
            h: height,
        };
        self.dispatch_normals_for_view(device, queue, src_view, width, height, dx, dz, region);
    }

    fn dispatch_normals_for_view(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        height_view: &wgpu::TextureView,
        w: u32,
        h: u32,
        dx: f32,
        dz: f32,
        region: SampleRect,
    ) {
        let uniforms = NormalUniforms {
            width: w,
            height: h,
            dx,
            dz,
            region_x: region.x,
            region_y: region.y,
            region_w: region.w.max(1),
            region_h: region.h.max(1),
        };
        queue.write_buffer(&self.normal_uniform, 0, bytemuck::bytes_of(&uniforms));

        let normal_out = &self.slots[self.display].normal_view;
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("normal-bg-shared"),
            layout: &self.normal_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.normal_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(height_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(normal_out),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("normal-enc-shared"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("normals-shared"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.normal_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((region.w + 7) / 8, (region.h + 7) / 8, 1);
        }
        queue.submit(Some(encoder.finish()));
    }

    pub fn display_height_view(&self) -> &wgpu::TextureView {
        if let Some(view) = &self.shared_height_view {
            view
        } else {
            &self.slots[self.display].height_view
        }
    }

    /// Drop any borrowed engine view and replace project-sized slots with the
    /// small blank baseline. Replaced textures keep the normal deferred-retirement path.
    pub fn reset_project_state(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        world_size: (f32, f32),
    ) {
        self.shared_height_view = None;
        self.world_size = world_size;
        self.height_range = (0.0, 1.0);
        let w = PROJECT_RESET_TEXTURE_EXTENT;
        let h = PROJECT_RESET_TEXTURE_EXTENT;
        self.ensure_size(device, w, h);
        let zeros = vec![0f32; (w as usize).saturating_mul(h as usize)];
        for slot in &self.slots {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &slot.height,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&zeros),
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
        // Rebuild normals from the blank height so lighting does not keep old slopes.
        self.compute_normals_region_and_swap(
            device,
            queue,
            w,
            h,
            world_size.0 / w.max(1) as f32,
            world_size.1 / h.max(1) as f32,
            SampleRect { x: 0, y: 0, w, h },
        );
    }

    pub fn display_normal_view(&self) -> &wgpu::TextureView {
        &self.slots[self.display].normal_view
    }

    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }

    /// Upload surface maps emitted by CPU or hybrid evaluation. Missing maps are
    /// replaced with a 1×1 zero map so old evaluation results cannot leak through.
    pub fn upload_aux_maps(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        materials: Option<&MaskField>,
        wetness: Option<&MaskField>,
        vegetation: Option<&MaskField>,
    ) {
        self.upload_aux_maps_ex(
            device, queue, materials, wetness, vegetation, None, None, None, None, None, None,
        );
    }

    /// Upload materials/wetness/vegetation plus optional climate aux and flow (Phase H polish).
    pub fn upload_aux_maps_ex(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
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
        Self::upload_aux_map(
            device,
            queue,
            &mut self.materials,
            materials,
            "materials-r32",
        );
        Self::upload_aux_map(device, queue, &mut self.wetness, wetness, "wetness-r32");
        Self::upload_aux_map(
            device,
            queue,
            &mut self.vegetation,
            vegetation,
            "vegetation-r32",
        );
        Self::upload_aux_map(device, queue, &mut self.flow, flow, "flow-r32");
        Self::upload_aux_map(
            device,
            queue,
            &mut self.temperature,
            temperature,
            "temperature-r32",
        );
        Self::upload_aux_map(device, queue, &mut self.rainfall, rainfall, "rainfall-r32");
        Self::upload_aux_map(device, queue, &mut self.snow, snow, "snow-r32");
        Self::upload_aux_map(
            device,
            queue,
            &mut self.soil_moisture,
            soil_moisture,
            "soil-moisture-r32",
        );
        Self::upload_aux_map(device, queue, &mut self.biomes, biomes, "biomes-r32");
    }

    fn upload_aux_map(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &mut AuxMap,
        source: Option<&MaskField>,
        label: &'static str,
    ) {
        let (width, height, data): (u32, u32, &[f32]) = match source {
            Some(map) => (map.metrics.width, map.metrics.height, map.data()),
            None => (1, 1, &[0.0]),
        };
        if target.width != width || target.height != height {
            *target = AuxMap::new(device, width, height, label);
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &target._texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    pub fn materials_view(&self) -> &wgpu::TextureView {
        &self.materials.view
    }

    pub fn wetness_view(&self) -> &wgpu::TextureView {
        &self.wetness.view
    }

    pub fn vegetation_view(&self) -> &wgpu::TextureView {
        &self.vegetation.view
    }

    pub fn flow_view(&self) -> &wgpu::TextureView {
        &self.flow.view
    }

    pub fn temperature_view(&self) -> &wgpu::TextureView {
        &self.temperature.view
    }

    pub fn rainfall_view(&self) -> &wgpu::TextureView {
        &self.rainfall.view
    }

    pub fn snow_view(&self) -> &wgpu::TextureView {
        &self.snow.view
    }

    pub fn soil_moisture_view(&self) -> &wgpu::TextureView {
        &self.soil_moisture.view
    }

    pub fn biomes_view(&self) -> &wgpu::TextureView {
        &self.biomes.view
    }

    pub fn placement_tint_view(&self) -> &wgpu::TextureView {
        &self.placement_tint.view
    }

    /// Upload painted biome placement colours (RGBA8, row-major). Empty → 1×1 transparent.
    pub fn upload_placement_tint(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) {
        let (w, h, bytes): (u32, u32, Vec<u8>) =
            if width == 0 || height == 0 || rgba.len() < (width * height * 4) as usize {
                (1, 1, vec![0, 0, 0, 0])
            } else {
                (width, height, rgba.to_vec())
            };
        if self.placement_tint.width != w || self.placement_tint.height != h {
            self.placement_tint = RgbaAuxMap::new(device, w, h, "placement-tint-rgba8");
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.placement_tint._texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
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
}
