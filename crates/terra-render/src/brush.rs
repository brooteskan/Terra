//! World-space brush ring overlay for sculpt / mask painting.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use terra_core::heightfield::Heightfield;

use crate::camera::OrbitCamera;

const RING_SEGMENTS: usize = 64;
const VERT_CAP: usize = RING_SEGMENTS + 1; // closed loop

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BrushVertex {
    pos: [f32; 3],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BrushUniforms {
    view_proj: [[f32; 4]; 4],
}

/// CPU description of the brush footprint in terrain UV space.
#[derive(Debug, Clone, Copy)]
pub struct BrushGizmo {
    pub u: f32,
    pub v: f32,
    pub radius_uv: f32,
    pub color: [f32; 4],
}

pub struct BrushOverlay {
    pipeline: wgpu::RenderPipeline,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    vertex_buf: wgpu::Buffer,
    vertex_count: u32,
    pub gizmo: Option<BrushGizmo>,
}

impl BrushOverlay {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        terra_core::shader_progress::record_shader_compiled();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brush-gizmo-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/brush.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brush-bgl"),
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
            label: Some("brush-u"),
            size: 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brush-bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("brush-pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("brush-pipe"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<BrushVertex>() as u64,
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
                topology: wgpu::PrimitiveTopology::LineStrip,
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
            label: Some("brush-verts"),
            size: (VERT_CAP * std::mem::size_of::<BrushVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            uniform_buf,
            bind_group,
            vertex_buf,
            vertex_count: 0,
            gizmo: None,
        }
    }

    pub fn set_gizmo(&mut self, gizmo: Option<BrushGizmo>) {
        self.gizmo = gizmo;
    }

    /// Rebuild ring vertices from UV footprint.
    /// `ring_heights` has `RING_SEGMENTS+1` samples around the circle (same order as verts).
    pub fn sync_geometry(
        &mut self,
        queue: &wgpu::Queue,
        world_size: (f32, f32),
        ring_heights: Option<&[f32]>,
    ) {
        let Some(g) = self.gizmo else {
            self.vertex_count = 0;
            return;
        };
        let wx = world_size.0.max(1.0);
        let wz = world_size.1.max(1.0);
        let cx = g.u * wx;
        let cz = g.v * wz;
        let rx = g.radius_uv.max(0.002) * wx;
        let rz = g.radius_uv.max(0.002) * wz;
        let lift = 1.5;

        let mut verts = Vec::with_capacity(VERT_CAP);
        for i in 0..=RING_SEGMENTS {
            let t = (i as f32 / RING_SEGMENTS as f32) * std::f32::consts::TAU;
            let x = cx + t.cos() * rx;
            let z = cz + t.sin() * rz;
            let y = ring_heights.and_then(|h| h.get(i).copied()).unwrap_or(0.0) + lift;
            verts.push(BrushVertex {
                pos: [x, y, z],
                color: g.color,
            });
        }

        queue.write_buffer(&self.vertex_buf, 0, bytemuck::cast_slice(&verts));
        self.vertex_count = verts.len() as u32;
    }

    pub fn sample_ring_heights(
        heights: Option<&Heightfield>,
        world_size: (f32, f32),
        u: f32,
        v: f32,
        radius_uv: f32,
    ) -> [f32; VERT_CAP] {
        let wx = world_size.0.max(1.0);
        let wz = world_size.1.max(1.0);
        let cx = u * wx;
        let cz = v * wz;
        let rx = radius_uv.max(0.002) * wx;
        let rz = radius_uv.max(0.002) * wz;
        let mut out = [0.0f32; VERT_CAP];
        // `i` drives the ring angle (`i as f32 / RING_SEGMENTS`), not just the
        // `out[i]` write, and the inclusive range walks one past the segment
        // count to close the ring. Kept indexed.
        #[allow(clippy::needless_range_loop)]
        for i in 0..=RING_SEGMENTS {
            let t = (i as f32 / RING_SEGMENTS as f32) * std::f32::consts::TAU;
            let x = cx + t.cos() * rx;
            let z = cz + t.sin() * rz;
            out[i] = sample_height(heights, x, z);
        }
        out
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

    pub fn upload_view_proj(&self, queue: &wgpu::Queue, view_proj: Mat4) {
        let u = BrushUniforms {
            view_proj: view_proj.to_cols_array_2d(),
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u));
    }
}

fn sample_height(heights: Option<&Heightfield>, x: f32, z: f32) -> f32 {
    let Some(hf) = heights else {
        return 0.0;
    };
    let (i, j) = hf.metrics.sample_index(x, z);
    hf.get(i, j)
}

/// Raycast cursor through the camera onto the terrain XZ plane → UV.
///
/// `cursor` / `screen` are logical pixels (same space as terra-gui).
/// Returns `None` if the ray misses the world footprint.
pub fn pick_terrain_uv(
    camera: &OrbitCamera,
    aspect: f32,
    cursor: (f32, f32),
    screen: (f32, f32),
    world_size: (f32, f32),
) -> Option<(f32, f32)> {
    if screen.0 < 1.0 || screen.1 < 1.0 {
        return None;
    }
    let ndc_x = (cursor.0 / screen.0) * 2.0 - 1.0;
    let ndc_y = 1.0 - (cursor.1 / screen.1) * 2.0;
    let inv = camera.view_proj(aspect).inverse();
    let near = inv.project_point3(Vec3::new(ndc_x, ndc_y, 0.0));
    let far = inv.project_point3(Vec3::new(ndc_x, ndc_y, 1.0));
    let dir = far - near;
    if dir.y.abs() < 1e-6 {
        return None;
    }
    // Prefer hit near camera target height so gizmo tracks orbit better on hills.
    let plane_y = camera.target.y;
    let t = (plane_y - near.y) / dir.y;
    if t < 0.0 {
        return None;
    }
    let hit = near + dir * t;
    let u = hit.x / world_size.0.max(1.0);
    let v = hit.z / world_size.1.max(1.0);
    if (0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v) {
        Some((u, v))
    } else {
        None
    }
}

/// Refine pick onto the height surface with a few height-plane iterations.
pub fn pick_terrain_uv_on_surface(
    camera: &OrbitCamera,
    aspect: f32,
    cursor: (f32, f32),
    screen: (f32, f32),
    world_size: (f32, f32),
    heights: Option<&Heightfield>,
) -> Option<(f32, f32)> {
    let mut plane_y = camera.target.y;
    let mut best = None;
    for _ in 0..4 {
        let Some((u, v)) = pick_at_plane(camera, aspect, cursor, screen, world_size, plane_y)
        else {
            return best;
        };
        best = Some((u, v));
        let x = u * world_size.0;
        let z = v * world_size.1;
        let h = sample_height(heights, x, z);
        if (h - plane_y).abs() < 0.25 {
            break;
        }
        plane_y = h;
    }
    best
}

fn pick_at_plane(
    camera: &OrbitCamera,
    aspect: f32,
    cursor: (f32, f32),
    screen: (f32, f32),
    world_size: (f32, f32),
    plane_y: f32,
) -> Option<(f32, f32)> {
    if screen.0 < 1.0 || screen.1 < 1.0 {
        return None;
    }
    let ndc_x = (cursor.0 / screen.0) * 2.0 - 1.0;
    let ndc_y = 1.0 - (cursor.1 / screen.1) * 2.0;
    let inv = camera.view_proj(aspect).inverse();
    let near = inv.project_point3(Vec3::new(ndc_x, ndc_y, 0.0));
    let far = inv.project_point3(Vec3::new(ndc_x, ndc_y, 1.0));
    let dir = far - near;
    if dir.y.abs() < 1e-6 {
        return None;
    }
    let t = (plane_y - near.y) / dir.y;
    if t < 0.0 {
        return None;
    }
    let hit = near + dir * t;
    let u = hit.x / world_size.0.max(1.0);
    let v = hit.z / world_size.1.max(1.0);
    if (0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v) {
        Some((u, v))
    } else {
        None
    }
}
