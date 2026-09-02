use bytemuck::{Pod, Zeroable};
use terra_core::heightfield::Heightfield;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TerrainVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

pub struct TerrainMesh {
    pub vertex_buf: wgpu::Buffer,
    pub index_buf: wgpu::Buffer,
    pub index_count: u32,
    pub height_range: (f32, f32),
}

impl TerrainMesh {
    pub fn empty(device: &wgpu::Device) -> Self {
        let v = [TerrainVertex {
            position: [0.0; 3],
            normal: [0.0, 1.0, 0.0],
        }];
        let i = [0u32, 0, 0];
        Self {
            vertex_buf: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("empty-v"),
                contents: bytemuck::cast_slice(&v),
                usage: wgpu::BufferUsages::VERTEX,
            }),
            index_buf: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("empty-i"),
                contents: bytemuck::cast_slice(&i),
                usage: wgpu::BufferUsages::INDEX,
            }),
            index_count: 0,
            height_range: (0.0, 1.0),
        }
    }

    pub fn vertex_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<TerrainVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }

    pub fn from_heightfield(device: &wgpu::Device, hf: &Heightfield) -> Self {
        let w = hf.metrics.width;
        let h = hf.metrics.height;
        // Decimate for viewport if huge
        let step = (w.max(h) / 512).max(1);
        let gw = w.div_ceil(step);
        let gh = h.div_ceil(step);

        let mut vertices = Vec::with_capacity((gw * gh) as usize);
        let (min_h, max_h) = hf.min_max();
        let dx = hf.metrics.dx();
        let dz = hf.metrics.dz();

        for jz in 0..gh {
            for ix in 0..gw {
                let i = (ix * step).min(w - 1);
                let j = (jz * step).min(h - 1);
                let y = hf.get(i, j);
                let il = i.saturating_sub(step);
                let ir = (i + step).min(w - 1);
                let jd = j.saturating_sub(step);
                let ju = (j + step).min(h - 1);
                let gx = (hf.get(ir, j) - hf.get(il, j)) / ((ir - il).max(1) as f32 * dx);
                let gz = (hf.get(i, ju) - hf.get(i, jd)) / ((ju - jd).max(1) as f32 * dz);
                let n = glam::Vec3::new(-gx, 1.0, -gz).normalize_or_zero();
                vertices.push(TerrainVertex {
                    position: [hf.metrics.world_x(i), y, hf.metrics.world_z(j)],
                    normal: n.to_array(),
                });
            }
        }

        let mut indices = Vec::with_capacity(((gw - 1) * (gh - 1) * 6) as usize);
        for jz in 0..gh - 1 {
            for ix in 0..gw - 1 {
                let i0 = jz * gw + ix;
                let i1 = i0 + 1;
                let i2 = i0 + gw;
                let i3 = i2 + 1;
                indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
            }
        }

        Self {
            vertex_buf: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("terrain-v"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }),
            index_buf: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("terrain-i"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            }),
            index_count: indices.len() as u32,
            height_range: (min_h, max_h.max(min_h + 1e-3)),
        }
    }
}
