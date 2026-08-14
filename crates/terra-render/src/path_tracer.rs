//! Compute heightfield path tracer (Phase 3–6).
//!
//! GPU compilation requires a wgpu device; see `shader_tests::path_trace_shader_parses`.

use bytemuck::{Pod, Zeroable};
use glam::Mat4;

const UNIFORM_ALIGN: u64 = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PathTraceUniforms {
    pub view_inv: [[f32; 4]; 4],
    /// aspect, tan_half_fov_y, near, far
    pub proj_params: [f32; 4],
    /// xyz toward scene, w intensity
    pub sun_dir_intensity: [f32; 4],
    /// rgb clear, w exposure
    pub clear_exposure: [f32; 4],
    /// world_x, world_z, min_h, max_h
    pub world_height: [f32; 4],
    /// spp, max_bounces, frame_seed, accum_index
    pub trace_params: [f32; 4],
    /// direct_clamp, indirect_clamp, sun_angular_radius, _
    pub clamps_radius: [f32; 4],
    /// out_w, out_h, internal_w, internal_h
    pub resolution_scale: [f32; 4],
    /// tex_w, tex_h, dx, dz
    pub tex_dims: [f32; 4],
}

struct OutputTextures {
    #[allow(dead_code)]
    radiance: [wgpu::Texture; 2],
    radiance_view: [wgpu::TextureView; 2],
    #[allow(dead_code)]
    depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
    #[allow(dead_code)]
    normal: wgpu::Texture,
    normal_view: wgpu::TextureView,
    #[allow(dead_code)]
    albedo: wgpu::Texture,
    albedo_view: wgpu::TextureView,
    /// Index of the buffer last written (display / next read).
    write_index: usize,
}

fn make_storage_view(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn clear_texture_zeros(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
    bytes_per_pixel: u32,
) {
    let w = width.max(1);
    let h = height.max(1);
    let row_bytes = (w * bytes_per_pixel).next_multiple_of(256);
    let data = vec![0u8; (row_bytes * h) as usize];
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(row_bytes),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
}

fn create_outputs(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
) -> OutputTextures {
    let (r0, rv0) = make_storage_view(
        device,
        "path-radiance-0",
        width,
        height,
        wgpu::TextureFormat::Rgba16Float,
    );
    let (r1, rv1) = make_storage_view(
        device,
        "path-radiance-1",
        width,
        height,
        wgpu::TextureFormat::Rgba16Float,
    );
    clear_texture_zeros(queue, &r0, width, height, 8);
    clear_texture_zeros(queue, &r1, width, height, 8);

    let (depth, depth_view) = make_storage_view(
        device,
        "path-depth",
        width,
        height,
        wgpu::TextureFormat::R32Float,
    );
    let (normal, normal_view) = make_storage_view(
        device,
        "path-normal",
        width,
        height,
        wgpu::TextureFormat::Rgba16Float,
    );
    let (albedo, albedo_view) = make_storage_view(
        device,
        "path-albedo",
        width,
        height,
        wgpu::TextureFormat::Rgba8Unorm,
    );

    OutputTextures {
        radiance: [r0, r1],
        radiance_view: [rv0, rv1],
        depth,
        depth_view,
        normal,
        normal_view,
        albedo,
        albedo_view,
        write_index: 0,
    }
}

fn create_sample_mask(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("path-sample-mask"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Owns the path-tracing compute pipeline and G-buffer outputs.
pub struct PathTracer {
    width: u32,
    height: u32,
    internal_width: u32,
    internal_height: u32,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,
    outputs: OutputTextures,
    sample_mask_tex: wgpu::Texture,
    sample_mask_view: wgpu::TextureView,
    accum_index: u32,
    frame_seed: u32,
}

impl PathTracer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        internal_scale: f32,
    ) -> Self {
        let internal_scale = internal_scale.clamp(0.25, 1.0);
        let internal_width = ((width as f32) * internal_scale).round().max(1.0) as u32;
        let internal_height = ((height as f32) * internal_scale).round().max(1.0) as u32;

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("path-trace-bgl"),
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
                sampled_tex(1, false),
                sampled_tex(2, false),
                sampled_tex(3, false),
                storage_tex(
                    4,
                    wgpu::TextureFormat::Rgba16Float,
                    wgpu::StorageTextureAccess::WriteOnly,
                ),
                storage_tex(
                    5,
                    wgpu::TextureFormat::R32Float,
                    wgpu::StorageTextureAccess::WriteOnly,
                ),
                storage_tex(
                    6,
                    wgpu::TextureFormat::Rgba16Float,
                    wgpu::StorageTextureAccess::WriteOnly,
                ),
                storage_tex(
                    7,
                    wgpu::TextureFormat::Rgba8Unorm,
                    wgpu::StorageTextureAccess::WriteOnly,
                ),
                sampled_tex(8, false),
                sampled_tex(9, false),
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("path-trace"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/path_trace.wgsl").into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("path-trace-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("path-trace-pipe"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("path-trace-u"),
            size: UNIFORM_ALIGN,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let outputs = create_outputs(device, queue, width, height);
        let (sample_mask_tex, sample_mask_view) = create_sample_mask(device, width, height);

        Self {
            width,
            height,
            internal_width,
            internal_height,
            pipeline,
            bind_group_layout,
            uniform_buf,
            outputs,
            sample_mask_tex,
            sample_mask_view,
            accum_index: 0,
            frame_seed: 0,
        }
    }

    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        internal_scale: f32,
    ) {
        let accum = self.accum_index;
        let seed = self.frame_seed;
        *self = Self::new(device, queue, width, height, internal_scale);
        self.accum_index = accum;
        self.frame_seed = seed;
    }

    pub fn destroy(&mut self) {
        self.accum_index = 0;
        self.frame_seed = 0;
        self.width = 0;
        self.height = 0;
    }

    pub fn invalidate(&mut self, queue: &wgpu::Queue) {
        self.accum_index = 0;
        self.frame_seed = 0;
        self.outputs.write_index = 0;
        if self.width == 0 || self.height == 0 {
            return;
        }
        for tex in &self.outputs.radiance {
            clear_texture_zeros(queue, tex, self.width, self.height, 8);
        }
    }

    /// Upload an R8 adaptive sampling mask (0 = skip / keep previous radiance).
    pub fn upload_sample_mask(&mut self, queue: &wgpu::Queue, mask: &[u8]) {
        let expected = (self.width * self.height) as usize;
        if mask.len() < expected || self.width == 0 || self.height == 0 {
            return;
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.sample_mask_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &mask[..expected],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.width),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }

    pub fn accum_index(&self) -> u32 {
        self.accum_index
    }

    pub fn radiance_view(&self) -> &wgpu::TextureView {
        &self.outputs.radiance_view[self.outputs.write_index]
    }

    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.outputs.depth_view
    }

    pub fn normal_view(&self) -> &wgpu::TextureView {
        &self.outputs.normal_view
    }

    pub fn albedo_view(&self) -> &wgpu::TextureView {
        &self.outputs.albedo_view
    }

    /// Sample count is packed into radiance.a (no separate texture).
    pub fn sample_count_view(&self) -> &wgpu::TextureView {
        self.radiance_view()
    }

    /// Roughness is packed into normal.w; material id into albedo.a.
    pub fn material_roughness_view(&self) -> &wgpu::TextureView {
        &self.outputs.normal_view
    }

    pub fn dispatch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        height_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        material_view: &wgpu::TextureView,
        uniforms: PathTraceUniforms,
        spp: u32,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        if spp == 0 {
            return;
        }

        let read = self.outputs.write_index;
        let write = 1 - read;

        let mut u = uniforms;
        u.trace_params[0] = spp as f32;
        u.trace_params[2] = self.frame_seed as f32;
        u.trace_params[3] = self.accum_index as f32;
        u.resolution_scale = [
            self.width as f32,
            self.height as f32,
            self.internal_width as f32,
            self.internal_height as f32,
        ];
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("path-trace-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(height_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(material_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(
                        &self.outputs.radiance_view[write],
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&self.outputs.depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&self.outputs.normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&self.outputs.albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&self.sample_mask_view),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(&self.outputs.radiance_view[read]),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("path-trace-pass"),
                timestamp_writes,
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (self.width + 7) / 8;
            let wg_y = (self.height + 7) / 8;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        self.outputs.write_index = write;
        self.accum_index = self.accum_index.saturating_add(spp);
        self.frame_seed = self.frame_seed.wrapping_add(1);
    }

    /// Convenience builder from camera matrices and lighting.
    #[allow(clippy::too_many_arguments)]
    pub fn uniforms_from_scene(
        view_inv: Mat4,
        aspect: f32,
        fov_y: f32,
        near: f32,
        far: f32,
        sun_dir: [f32; 4],
        clear: [f32; 3],
        exposure: f32,
        world_size: (f32, f32),
        height_range: (f32, f32),
        tex_size: (f32, f32),
        dx_dz: (f32, f32),
        direct_clamp: f32,
        indirect_clamp: f32,
        sun_angular_radius: f32,
        max_bounces: u32,
        spp: u32,
    ) -> PathTraceUniforms {
        PathTraceUniforms {
            view_inv: view_inv.to_cols_array_2d(),
            proj_params: [aspect, (fov_y * 0.5).tan(), near, far],
            sun_dir_intensity: sun_dir,
            clear_exposure: [clear[0], clear[1], clear[2], exposure],
            world_height: [world_size.0, world_size.1, height_range.0, height_range.1],
            trace_params: [spp as f32, max_bounces as f32, 0.0, 0.0],
            clamps_radius: [direct_clamp, indirect_clamp, sun_angular_radius, 0.0],
            resolution_scale: [0.0; 4],
            tex_dims: [tex_size.0, tex_size.1, dx_dz.0, dx_dz.1],
        }
    }
}

fn sampled_tex(binding: u32, filterable: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_tex(
    binding: u32,
    format: wgpu::TextureFormat,
    access: wgpu::StorageTextureAccess,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

#[cfg(test)]
mod shader_tests {
    #[test]
    fn path_trace_shader_parses() {
        let source = include_str!("shaders/path_trace.wgsl");
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("path_trace WGSL parse failed: {error}"));
        assert!(module.entry_points.iter().any(|entry| entry.name == "main"));
    }
}
