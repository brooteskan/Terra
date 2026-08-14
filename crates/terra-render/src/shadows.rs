//! Directional shadow map for stable Fast Lit sun shadows.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

const SHADOW_MAP_SIZE: u32 = 2048;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ShadowUniforms {
    pub light_view_proj: [[f32; 4]; 4],
    /// x=enabled, y=bias, z=soft_scale, w=unused
    pub params: [f32; 4],
}

/// Orthographic depth map from the sun toward the terrain AABB.
pub struct ShadowMap {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub comparison_sampler: wgpu::Sampler,
    pub uniform_buf: wgpu::Buffer,
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    enabled: bool,
}

impl ShadowMap {
    pub fn new(
        device: &wgpu::Device,
        height_view: &wgpu::TextureView,
        height_bgl_entry_compatible: bool,
    ) -> Self {
        let _ = height_bgl_entry_compatible;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain-shadow-map"),
            size: wgpu::Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("terrain-shadow-view"),
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        let comparison_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain-shadow-cmp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow-pass-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow-uniforms"),
            size: std::mem::size_of::<ShadowUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group =
            make_shadow_bind_group(device, &bind_group_layout, &uniform_buf, height_view);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow-depth"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shadow.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow-depth-pipe"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_shadow"),
                buffers: &[crate::grid::TerrainGrid::vertex_layout()],
                compilation_options: Default::default(),
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 1.5,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            texture,
            view,
            comparison_sampler,
            uniform_buf,
            pipeline,
            bind_group_layout,
            bind_group,
            enabled: false,
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn recreate_bind_group(&mut self, device: &wgpu::Device, height_view: &wgpu::TextureView) {
        self.bind_group = make_shadow_bind_group(
            device,
            &self.bind_group_layout,
            &self.uniform_buf,
            height_view,
        );
    }

    /// Fit an orthographic light frustum around the terrain AABB and upload it.
    pub fn update_light(
        &self,
        queue: &wgpu::Queue,
        light_dir_from_light: [f32; 3],
        world_size: (f32, f32),
        height_range: (f32, f32),
    ) -> Mat4 {
        let light_view_proj = fit_shadow_ortho(light_dir_from_light, world_size, height_range);
        let uniforms = ShadowUniforms {
            light_view_proj: light_view_proj.to_cols_array_2d(),
            params: [
                if self.enabled { 1.0 } else { 0.0 },
                0.0015,
                world_size.0.max(1.0),
                world_size.1.max(1.0),
            ],
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        light_view_proj
    }
}

/// Orthographic light view-projection covering the entire terrain AABB for any sun
/// direction. A fixed half-extent leaves an uncovered wedge whose edge rotates with
/// the sun azimuth, so fit the bounds to the eight AABB corners transformed into
/// light-view space. A free function so the coverage is unit-testable without a GPU.
pub fn fit_shadow_ortho(
    light_dir_from_light: [f32; 3],
    world_size: (f32, f32),
    height_range: (f32, f32),
) -> Mat4 {
    let center = Vec3::new(
        world_size.0 * 0.5,
        (height_range.0 + height_range.1) * 0.5,
        world_size.1 * 0.5,
    );
    let mut dir = Vec3::from_array(light_dir_from_light);
    if dir.length_squared() < 1e-6 {
        dir = Vec3::new(-0.35, -0.9, -0.2);
    }
    dir = dir.normalize();
    // light_dir is "from light toward scene"; the light camera looks along it.
    let extent = world_size.0.max(world_size.1).max(1.0);
    let eye = center - dir * extent * 1.6;
    let up = if dir.y.abs() > 0.95 { Vec3::X } else { Vec3::Y };
    let view = Mat4::look_at_rh(eye, center, up);

    let (x0, x1) = (0.0, world_size.0.max(1.0));
    let (z0, z1) = (0.0, world_size.1.max(1.0));
    let (y0, y1) = (height_range.0, height_range.1.max(height_range.0 + 1.0));
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for &cx in &[x0, x1] {
        for &cy in &[y0, y1] {
            for &cz in &[z0, z1] {
                let v = view.transform_point3(Vec3::new(cx, cy, cz));
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
    }
    // Small margin so PCF taps at the border stay inside the map.
    let pad = (hi - lo).length() * 0.02 + 1.0;
    // View looks down -Z, so AABB points have negative z; near/far are the positive
    // distances -hi.z (closest to the light) and -lo.z (farthest).
    let near = (-hi.z - pad).max(0.01);
    let far = (-lo.z + pad).max(near + 1.0);
    Mat4::orthographic_rh(lo.x - pad, hi.x + pad, lo.y - pad, hi.y + pad, near, far) * view
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ortho_covers_whole_terrain_for_any_sun() {
        let world = (4096.0, 2048.0);
        let heights = (-120.0, 900.0);
        for az_i in 0..12 {
            let az = az_i as f32 / 12.0 * std::f32::consts::TAU;
            for &el_deg in &[5.0f32, 25.0, 55.0, 85.0] {
                let el = el_deg.to_radians();
                let ce = el.cos();
                // Direction from light toward scene (sun above the horizon).
                let dir = [-(ce * az.cos()), -el.sin(), -(ce * az.sin())];
                let m = fit_shadow_ortho(dir, world, heights);
                for &x in &[0.0, world.0] {
                    for &y in &[heights.0, heights.1] {
                        for &z in &[0.0, world.1] {
                            let c = m * glam::Vec4::new(x, y, z, 1.0);
                            assert!(
                                c.x.abs() <= 1.0 + 1e-3
                                    && c.y.abs() <= 1.0 + 1e-3
                                    && c.z >= -1e-3
                                    && c.z <= 1.0 + 1e-3,
                                "corner ({x},{y},{z}) left clip space at az_i={az_i} el={el_deg}: {c:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}

fn make_shadow_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buf: &wgpu::Buffer,
    height_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shadow-pass-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(height_view),
            },
        ],
    })
}

/// Atmosphere helpers derived from sun direction for clear colour / fog.
pub fn atmosphere_from_sun(
    light_dir_from_light: [f32; 3],
    base_clear: [f32; 3],
) -> ([f32; 3], [f32; 4]) {
    let mut dir = Vec3::from_array(light_dir_from_light);
    if dir.length_squared() < 1e-6 {
        dir = Vec3::new(-0.35, -0.9, -0.2);
    }
    dir = dir.normalize();
    // Elevation of the sun above the horizon (light points down when y < 0).
    let sun_up = (-dir.y).clamp(0.0, 1.0);
    let dusk = (1.0 - sun_up).clamp(0.0, 1.0);
    let zenith = [
        (base_clear[0] * 0.55 + 0.20 * sun_up).clamp(0.0, 1.0),
        (base_clear[1] * 0.60 + 0.28 * sun_up).clamp(0.0, 1.0),
        (base_clear[2] * 0.70 + 0.40 * sun_up).clamp(0.0, 1.0),
    ];
    let horizon = [
        (0.55 + 0.35 * dusk).clamp(0.0, 1.0),
        (0.42 + 0.15 * dusk).clamp(0.0, 1.0),
        (0.32 + 0.05 * (1.0 - dusk)).clamp(0.0, 1.0),
    ];
    let clear = [
        zenith[0] * 0.65 + horizon[0] * 0.35,
        zenith[1] * 0.65 + horizon[1] * 0.35,
        zenith[2] * 0.65 + horizon[2] * 0.35,
    ];
    // fog: density, height_falloff, max_amount, sun_scatter
    let fog = [
        0.00012 + 0.00008 * dusk,
        0.35 + 0.25 * sun_up,
        0.28 + 0.12 * dusk,
        0.35 + 0.40 * sun_up,
    ];
    (clear, fog)
}
