//! Instanced crossed-card vegetation for the 3D viewport.

use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use terra_core::{heightfield::Heightfield, mask::MaskField};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PlantVertex {
    local_pos: [f32; 3],
    local_uv: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PlantInstance {
    position_scale: [f32; 4],
    color_phase: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VegetationUniforms {
    view_proj: [[f32; 4]; 4],
    light_dir: [f32; 4],
}

pub struct VegetationOverlay {
    pipeline: wgpu::RenderPipeline,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    vertex_buf: wgpu::Buffer,
    instance_buf: wgpu::Buffer,
    vertex_count: u32,
    instance_count: u32,
}

impl VegetationOverlay {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        terra_core::shader_progress::record_shader_compiled();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vegetation-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/vegetation.wgsl").into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vegetation-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vegetation-u"),
            size: 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vegetation-bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vegetation-pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = terra_gpu::cached_render_pipeline(device, "vegetation-pipe", format, || {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("vegetation-pipe"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[
                        wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<PlantVertex>() as u64,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &[
                                wgpu::VertexAttribute {
                                    format: wgpu::VertexFormat::Float32x3,
                                    offset: 0,
                                    shader_location: 0,
                                },
                                wgpu::VertexAttribute {
                                    format: wgpu::VertexFormat::Float32x2,
                                    offset: 12,
                                    shader_location: 1,
                                },
                            ],
                        },
                        wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<PlantInstance>() as u64,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes: &[
                                wgpu::VertexAttribute {
                                    format: wgpu::VertexFormat::Float32x4,
                                    offset: 0,
                                    shader_location: 2,
                                },
                                wgpu::VertexAttribute {
                                    format: wgpu::VertexFormat::Float32x4,
                                    offset: 16,
                                    shader_location: 3,
                                },
                            ],
                        },
                    ],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
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
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: terra_gpu::shared_pipeline_cache(device).as_ref(),
            })
        });

        // Two crossed tapered cards. Each triangle is already double-sided.
        let vertices = [
            PlantVertex {
                local_pos: [-1.0, 0.0, 0.0],
                local_uv: [0.0, 0.0],
            },
            PlantVertex {
                local_pos: [1.0, 0.0, 0.0],
                local_uv: [1.0, 0.0],
            },
            PlantVertex {
                local_pos: [0.0, 1.0, 0.0],
                local_uv: [0.5, 1.0],
            },
            PlantVertex {
                local_pos: [0.0, 0.0, -1.0],
                local_uv: [0.0, 0.0],
            },
            PlantVertex {
                local_pos: [0.0, 0.0, 1.0],
                local_uv: [1.0, 0.0],
            },
            PlantVertex {
                local_pos: [0.0, 1.0, 0.0],
                local_uv: [0.5, 1.0],
            },
        ];
        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vegetation-v"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let placeholder = [PlantInstance {
            position_scale: [0.0; 4],
            color_phase: [0.0; 4],
        }];
        let instance_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vegetation-instances"),
            contents: bytemuck::cast_slice(&placeholder),
            usage: wgpu::BufferUsages::VERTEX,
        });
        Self {
            pipeline,
            uniform_buf,
            bind_group,
            vertex_buf,
            instance_buf,
            vertex_count: 6,
            instance_count: 0,
        }
    }

    pub fn sync(
        &mut self,
        device: &wgpu::Device,
        height: &Heightfield,
        density: Option<&MaskField>,
        scale_min: f32,
        scale_max: f32,
        yaw_variation_deg: f32,
    ) {
        let Some(density) = density else {
            self.instance_count = 0;
            return;
        };
        let scale_lo = scale_min.min(scale_max).max(0.05);
        let scale_hi = scale_min.max(scale_max).max(scale_lo);
        let yaw_span = yaw_variation_deg.to_radians();
        let w = height.metrics.width;
        let h = height.metrics.height;
        let step = (w.max(h) / 256).max(1);
        let mut instances = Vec::new();
        for j in (0..h).step_by(step as usize) {
            for i in (0..w).step_by(step as usize) {
                let d = density.get(i, j).clamp(0.0, 1.0);
                if d <= 0.002 {
                    continue;
                }
                let hash = hash_u32(i.wrapping_mul(73856093) ^ j.wrapping_mul(19349663));
                if hash > (d * 8.0).clamp(0.0, 0.82) {
                    continue;
                }
                let x = height.metrics.world_x(i);
                let z = height.metrics.world_z(j);
                let y = height.get(i, j) + 0.15;
                let base = height
                    .metrics
                    .dx()
                    .min(height.metrics.dz())
                    .clamp(2.0, 18.0);
                let scale = base * (scale_lo + (scale_hi - scale_lo) * hash);
                let yaw_hash =
                    hash_u32(i.wrapping_mul(19349663) ^ j.wrapping_mul(83492791) ^ 0x9e37_79b9);
                let yaw = (yaw_hash * 2.0 - 1.0) * yaw_span;
                let lush = 0.7 + 0.3 * d;
                instances.push(PlantInstance {
                    position_scale: [x, y, z, scale],
                    color_phase: [0.045 * lush, 0.26 * lush, 0.055 * lush, yaw],
                });
                if instances.len() >= 30_000 {
                    break;
                }
            }
            if instances.len() >= 30_000 {
                break;
            }
        }
        self.instance_count = instances.len() as u32;
        if instances.is_empty() {
            return;
        }
        self.instance_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vegetation-instances"),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX,
        });
    }

    pub fn upload_view_proj(&self, queue: &wgpu::Queue, view_proj: Mat4, light_dir: [f32; 4]) {
        let uniforms = VegetationUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            light_dir,
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        if self.instance_count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        pass.set_vertex_buffer(1, self.instance_buf.slice(..));
        pass.draw(0..self.vertex_count, 0..self.instance_count);
    }
}

fn hash_u32(mut value: u32) -> f32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846ca68b);
    value ^= value >> 16;
    (value & 0x00ff_ffff) as f32 / 16_777_216.0
}
