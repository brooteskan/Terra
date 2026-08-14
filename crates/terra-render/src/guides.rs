//! World grid + AABB bounds line overlays for the viewport.

use bytemuck::{Pod, Zeroable};
use glam::Mat4;

const VERT_CAP: usize = 4096;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GuideVertex {
    pos: [f32; 3],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GuideUniforms {
    view_proj: [[f32; 4]; 4],
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GuideState {
    pub grid: bool,
    pub bounds: bool,
}

pub struct GuideOverlay {
    pipeline: wgpu::RenderPipeline,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    vertex_buf: wgpu::Buffer,
    vertex_count: u32,
    pub state: GuideState,
    last_key: (bool, bool, u32, u32, i32, i32, i32, i32),
}

impl GuideOverlay {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("guides-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/guides.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("guides-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("guides-u"),
            size: 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("guides-bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("guides-pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("guides-pipe"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GuideVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 12,
                            shader_location: 1,
                        },
                    ],
                }],
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
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: wgpu::DepthBiasState {
                    constant: -2,
                    slope_scale: -1.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("guides-verts"),
            size: (VERT_CAP * std::mem::size_of::<GuideVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            uniform_buf,
            bind_group,
            vertex_buf,
            vertex_count: 0,
            state: GuideState::default(),
            last_key: (false, false, 0, 0, 0, 0, 0, 0),
        }
    }

    pub fn set_state(&mut self, state: GuideState) {
        self.state = state;
    }

    pub fn sync_geometry(
        &mut self,
        queue: &wgpu::Queue,
        world_size: (f32, f32),
        height_range: (f32, f32),
        focus_xz: (f32, f32),
    ) {
        let wx = world_size.0.max(1.0);
        let wz = world_size.1.max(1.0);
        let y0 = height_range.0;
        let y1 = height_range.1.max(y0 + 1.0);
        let extent = wx.max(wz);
        let raw_step = (extent / 24.0).max(1.0);
        let step = nice_step(raw_step);
        // Snap focus so the grid only rebuilds when the camera crosses a cell.
        let snap_x = (focus_xz.0 / step).round() as i32;
        let snap_z = (focus_xz.1 / step).round() as i32;
        let key = (
            self.state.grid,
            self.state.bounds,
            wx.to_bits(),
            wz.to_bits(),
            y0.to_bits() as i32,
            y1.to_bits() as i32,
            snap_x,
            snap_z,
        );
        if key == self.last_key && (self.state.grid || self.state.bounds) == (self.vertex_count > 0)
        {
            if !self.state.grid && !self.state.bounds {
                self.vertex_count = 0;
            }
            return;
        }
        self.last_key = key;

        let mut verts = Vec::with_capacity(512);
        let lift = 0.75;
        let y_grid = y0 + lift;

        if self.state.grid {
            // Camera-centered plane that extends well past the terrain footprint.
            let half_span = (extent * 4.0).max(step * 48.0);
            let cx = snap_x as f32 * step;
            let cz = snap_z as f32 * step;
            let x0 = cx - half_span;
            let x1 = cx + half_span;
            let z0 = cz - half_span;
            let z1 = cz + half_span;
            let grid_col = [0.55, 0.58, 0.62, 0.35];
            let major_col = [0.75, 0.78, 0.82, 0.55];

            let mut x = x0;
            while x <= x1 + step * 0.01 {
                let major = (x / step).round() as i32 % 5 == 0;
                let c = if major { major_col } else { grid_col };
                push_line(&mut verts, [x, y_grid, z0], [x, y_grid, z1], c);
                x += step;
            }
            let mut z = z0;
            while z <= z1 + step * 0.01 {
                let major = (z / step).round() as i32 % 5 == 0;
                let c = if major { major_col } else { grid_col };
                push_line(&mut verts, [x0, y_grid, z], [x1, y_grid, z], c);
                z += step;
            }
        }

        if self.state.bounds {
            let c = [0.95, 0.85, 0.35, 0.85];
            let corners = [
                [0.0, y0, 0.0],
                [wx, y0, 0.0],
                [wx, y0, wz],
                [0.0, y0, wz],
                [0.0, y1, 0.0],
                [wx, y1, 0.0],
                [wx, y1, wz],
                [0.0, y1, wz],
            ];
            let edges = [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
            ];
            for (a, b) in edges {
                push_line(&mut verts, corners[a], corners[b], c);
            }
        }

        if verts.len() > VERT_CAP {
            verts.truncate(VERT_CAP - (VERT_CAP % 2));
        }
        if verts.is_empty() {
            self.vertex_count = 0;
            return;
        }
        queue.write_buffer(&self.vertex_buf, 0, bytemuck::cast_slice(&verts));
        self.vertex_count = verts.len() as u32;
    }

    pub fn upload_view_proj(&self, queue: &wgpu::Queue, view_proj: Mat4) {
        let u = GuideUniforms {
            view_proj: view_proj.to_cols_array_2d(),
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u));
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        if self.vertex_count < 2 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        pass.draw(0..self.vertex_count, 0..1);
    }
}

fn push_line(verts: &mut Vec<GuideVertex>, a: [f32; 3], b: [f32; 3], color: [f32; 4]) {
    if verts.len() + 2 > VERT_CAP {
        return;
    }
    verts.push(GuideVertex { pos: a, color });
    verts.push(GuideVertex { pos: b, color });
}

fn nice_step(raw: f32) -> f32 {
    let exp = raw.log10().floor();
    let base = 10f32.powf(exp);
    let mant = raw / base;
    let nice = if mant < 1.5 {
        1.0
    } else if mant < 3.5 {
        2.0
    } else if mant < 7.5 {
        5.0
    } else {
        10.0
    };
    nice * base
}
