//! GPU-authoritative world-space brush ring and terrain surface picking.

use std::sync::mpsc::{self, Receiver, TryRecvError};

use bytemuck::{Pod, Zeroable};
use glam::{DVec2, Mat4, Vec3};
use terra_core::{heightfield::Heightfield, BoundedUv, TerrainSurfaceHit, WorldPosition};

use crate::{camera::OrbitCamera, TerrainTraversalMode};

const RING_SEGMENTS: u32 = 64;
const RING_VERTEX_COUNT: u32 = RING_SEGMENTS + 1;
const READBACK_SLOTS: usize = 3;
const UNIFORM_BUFFER_SIZE: u64 = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BrushUniforms {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    /// xy = world size, zw = displayed minimum/maximum height.
    world_height: [f32; 4],
    /// xy = cursor NDC, z = radius in the active frame (UV or metres), w = visible flag.
    cursor_radius: [f32; 4],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct RawSurfacePick {
    /// x = hit flag, yz = UV, w = surface height.
    hit_uv_height: [f32; 4],
    /// xyz = bounded world position or camera-relative Infinite position.
    world_pos_request: [f32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PickContext {
    serial: u64,
    cursor: (f32, f32),
    screen: (f32, f32),
    view_proj: [f32; 16],
    camera_target: [f64; 3],
    render_origin: [f64; 2],
    traversal: TerrainTraversalMode,
    binding_revision: u64,
    surface_revision: u64,
}

enum ReadbackState {
    Idle,
    Submitted(PickContext),
    Mapping {
        context: PickContext,
        receiver: Receiver<Result<(), wgpu::BufferAsyncError>>,
    },
}

struct ReadbackSlot {
    buffer: wgpu::Buffer,
    state: ReadbackState,
}

/// Completed non-blocking hit against the exact height texture used by the viewport.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfacePick {
    pub hit: TerrainSurfaceHit,
    pub local_position: [f32; 3],
    context: PickContext,
}

pub struct BrushOverlay {
    depth_pipeline: wgpu::RenderPipeline,
    overlay_pipeline: wgpu::RenderPipeline,
    compute_pipeline: wgpu::ComputePipeline,
    infinite_compute_pipeline: wgpu::ComputePipeline,
    infinite_depth_pipeline: wgpu::RenderPipeline,
    infinite_overlay_pipeline: wgpu::RenderPipeline,
    render_bgl: wgpu::BindGroupLayout,
    compute_bgl: wgpu::BindGroupLayout,
    infinite_compute_bgl: wgpu::BindGroupLayout,
    render_bind_group: Option<wgpu::BindGroup>,
    compute_bind_group: Option<wgpu::BindGroup>,
    infinite_compute_bind_group: Option<wgpu::BindGroup>,
    uniform_buf: wgpu::Buffer,
    result_buf: wgpu::Buffer,
    uniforms: BrushUniforms,
    readback: Vec<ReadbackSlot>,
    next_readback: usize,
    next_serial: u64,
    binding_revision: u64,
    traversal: TerrainTraversalMode,
    latest_pick: Option<SurfacePick>,
}

impl BrushOverlay {
    pub fn new(
        device: &wgpu::Device,
        pipelines: &terra_gpu::PipelineCacheRegistry,
        format: wgpu::TextureFormat,
    ) -> Self {
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brush-gizmo-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/brush.wgsl").into()),
        });
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brush-pick-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/brush_pick.wgsl").into()),
        });
        let infinite_compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brush-pick-infinite-shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/brush_pick_infinite.wgsl").into(),
            ),
        });
        let infinite_render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brush-gizmo-infinite-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/brush_infinite.wgsl").into()),
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brush-uniforms"),
            size: UNIFORM_BUFFER_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let result_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brush-surface-pick"),
            size: std::mem::size_of::<RawSurfacePick>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let entries = |read_only: bool, visibility| {
            [
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ]
        };
        let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brush-render-bgl"),
            entries: &entries(true, wgpu::ShaderStages::VERTEX),
        });
        let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brush-compute-bgl"),
            entries: &entries(false, wgpu::ShaderStages::COMPUTE),
        });
        let infinite_compute_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("brush-infinite-compute-bgl"),
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
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let render_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("brush-render-layout"),
            bind_group_layouts: &[&render_bgl],
            push_constant_ranges: &[],
        });
        let make_render_pipeline = |label: &'static str, depth_stencil| {
            pipelines.render_pipeline(label, format, || {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&render_layout),
                    vertex: wgpu::VertexState {
                        module: &render_shader,
                        entry_point: Some("vs_main"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &render_shader,
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
                    depth_stencil,
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: pipelines.driver_cache(),
                })
            })
        };
        let depth_pipeline = make_render_pipeline(
            "brush-depth-pipeline",
            Some(wgpu::DepthStencilState {
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
        );
        // The progressive backend owns a linear R32 depth texture rather than the
        // raster Depth32 attachment. Its cursor is composited after post and must
        // not be tested against the unrelated, stale raster attachment.
        let overlay_pipeline = make_render_pipeline("brush-overlay-pipeline", None);
        let make_infinite_render_pipeline = |label: &'static str, depth_stencil| {
            pipelines.render_pipeline(label, format, || {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&render_layout),
                    vertex: wgpu::VertexState {
                        module: &infinite_render_shader,
                        entry_point: Some("vs_main"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &infinite_render_shader,
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
                    depth_stencil,
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: pipelines.driver_cache(),
                })
            })
        };
        let infinite_depth_pipeline = make_infinite_render_pipeline(
            "brush-infinite-depth-pipeline",
            Some(wgpu::DepthStencilState {
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
        );
        let infinite_overlay_pipeline =
            make_infinite_render_pipeline("brush-infinite-overlay-pipeline", None);

        let compute_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("brush-compute-layout"),
            bind_group_layouts: &[&compute_bgl],
            push_constant_ranges: &[],
        });
        let compute_pipeline = pipelines.compute_pipeline("brush-pick-pipeline", || {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("brush-pick-pipeline"),
                layout: Some(&compute_layout),
                module: &compute_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: pipelines.driver_cache(),
            })
        });
        let infinite_compute_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("brush-infinite-compute-layout"),
                bind_group_layouts: &[&infinite_compute_bgl],
                push_constant_ranges: &[],
            });
        let infinite_compute_pipeline =
            pipelines.compute_pipeline("brush-pick-infinite-pipeline", || {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("brush-pick-infinite-pipeline"),
                    layout: Some(&infinite_compute_layout),
                    module: &infinite_compute_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: pipelines.driver_cache(),
                })
            });

        let readback = (0..READBACK_SLOTS)
            .map(|index| ReadbackSlot {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("brush-pick-readback-{index}")),
                    size: std::mem::size_of::<RawSurfacePick>() as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                state: ReadbackState::Idle,
            })
            .collect();

        Self {
            depth_pipeline,
            overlay_pipeline,
            compute_pipeline,
            infinite_compute_pipeline,
            infinite_depth_pipeline,
            infinite_overlay_pipeline,
            render_bgl,
            compute_bgl,
            infinite_compute_bgl,
            render_bind_group: None,
            compute_bind_group: None,
            infinite_compute_bind_group: None,
            uniform_buf,
            result_buf,
            uniforms: BrushUniforms::zeroed(),
            readback,
            next_readback: 0,
            next_serial: 1,
            binding_revision: 0,
            traversal: TerrainTraversalMode::Bounded,
            latest_pick: None,
        }
    }

    pub fn rebind_height(&mut self, device: &wgpu::Device, height_view: &wgpu::TextureView) {
        let entries = [
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
                resource: self.result_buf.as_entire_binding(),
            },
        ];
        self.render_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brush-render-bind"),
            layout: &self.render_bgl,
            entries: &entries,
        }));
        self.compute_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brush-compute-bind"),
            layout: &self.compute_bgl,
            entries: &entries,
        }));
        self.binding_revision = self.binding_revision.wrapping_add(1).max(1);
        self.latest_pick = None;
    }

    pub fn rebind_depth(&mut self, device: &wgpu::Device, depth_view: &wgpu::TextureView) {
        self.infinite_compute_bind_group =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("brush-infinite-compute-bind"),
                layout: &self.infinite_compute_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.result_buf.as_entire_binding(),
                    },
                ],
            }));
        self.binding_revision = self.binding_revision.wrapping_add(1).max(1);
        self.latest_pick = None;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn request_surface_pick(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera: &OrbitCamera,
        aspect: f32,
        cursor: (f32, f32),
        screen: (f32, f32),
        world_size: (f32, f32),
        height_range: (f32, f32),
        radius: f32,
        color: [f32; 4],
        visible: bool,
        traversal: TerrainTraversalMode,
        render_origin: DVec2,
        surface_revision: u64,
    ) {
        let ready = match traversal {
            TerrainTraversalMode::Bounded => self.compute_bind_group.is_some(),
            TerrainTraversalMode::Infinite => self.infinite_compute_bind_group.is_some(),
        };
        if screen.0 < 1.0 || screen.1 < 1.0 || !ready {
            self.hide(queue);
            return;
        }
        self.poll(device);
        self.traversal = traversal;
        let ndc_x = (cursor.0 / screen.0) * 2.0 - 1.0;
        let ndc_y = 1.0 - (cursor.1 / screen.1) * 2.0;
        let view_proj = match traversal {
            TerrainTraversalMode::Bounded => camera.view_proj(aspect),
            TerrainTraversalMode::Infinite => camera.view_proj_relative_to(aspect, render_origin),
        };
        let mut color = color;
        if !visible {
            color[3] = 0.0;
        }
        self.uniforms = BrushUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            inv_view_proj: view_proj.inverse().to_cols_array_2d(),
            world_height: [
                world_size.0.max(1.0),
                world_size.1.max(1.0),
                height_range.0,
                height_range.1,
            ],
            cursor_radius: [ndc_x, ndc_y, radius, 1.0],
            color,
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&self.uniforms));

        let serial = self.next_serial;
        self.next_serial = self.next_serial.wrapping_add(1).max(1);
        let context = PickContext {
            serial,
            cursor,
            screen,
            view_proj: view_proj.to_cols_array(),
            camera_target: camera.target.to_array(),
            render_origin: render_origin.to_array(),
            traversal,
            binding_revision: self.binding_revision,
            surface_revision,
        };
        let available = (0..self.readback.len())
            .map(|offset| (self.next_readback + offset) % self.readback.len())
            .find(|index| matches!(self.readback[*index].state, ReadbackState::Idle));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("brush-pick-encoder"),
        });
        encoder.clear_buffer(&self.result_buf, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("brush-surface-pick"),
                timestamp_writes: None,
            });
            match traversal {
                TerrainTraversalMode::Bounded => {
                    pass.set_pipeline(&self.compute_pipeline);
                    pass.set_bind_group(
                        0,
                        self.compute_bind_group.as_ref().expect("checked above"),
                        &[],
                    );
                }
                TerrainTraversalMode::Infinite => {
                    pass.set_pipeline(&self.infinite_compute_pipeline);
                    pass.set_bind_group(
                        0,
                        self.infinite_compute_bind_group
                            .as_ref()
                            .expect("checked above"),
                        &[],
                    );
                }
            }
            pass.dispatch_workgroups(1, 1, 1);
        }
        if let Some(index) = available {
            encoder.copy_buffer_to_buffer(
                &self.result_buf,
                0,
                &self.readback[index].buffer,
                0,
                std::mem::size_of::<RawSurfacePick>() as u64,
            );
            self.readback[index].state = ReadbackState::Submitted(context);
            self.next_readback = (index + 1) % self.readback.len();
        }
        queue.submit(Some(encoder.finish()));
    }

    pub fn hide(&mut self, queue: &wgpu::Queue) {
        self.uniforms.cursor_radius[3] = 0.0;
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&self.uniforms));
    }

    pub fn poll(&mut self, device: &wgpu::Device) {
        let _ = device.poll(wgpu::Maintain::Poll);
        for slot in &mut self.readback {
            let state = std::mem::replace(&mut slot.state, ReadbackState::Idle);
            slot.state = match state {
                ReadbackState::Idle => ReadbackState::Idle,
                ReadbackState::Submitted(context) => {
                    let (sender, receiver) = mpsc::channel();
                    slot.buffer
                        .slice(..)
                        .map_async(wgpu::MapMode::Read, move |result| {
                            let _ = sender.send(result);
                        });
                    ReadbackState::Mapping { context, receiver }
                }
                ReadbackState::Mapping { context, receiver } => match receiver.try_recv() {
                    Ok(Ok(())) => {
                        let bytes = slot.buffer.slice(..).get_mapped_range();
                        let raw = *bytemuck::from_bytes::<RawSurfacePick>(&bytes);
                        drop(bytes);
                        slot.buffer.unmap();
                        if raw.hit_uv_height[0] > 0.5
                            && context.binding_revision == self.binding_revision
                            && self
                                .latest_pick
                                .is_none_or(|pick| context.serial >= pick.context.serial)
                        {
                            let local_position = [
                                raw.world_pos_request[0],
                                raw.world_pos_request[1],
                                raw.world_pos_request[2],
                            ];
                            let world =
                                world_from_render_local(context.render_origin, local_position);
                            let bounded_uv = match context.traversal {
                                TerrainTraversalMode::Bounded => {
                                    BoundedUv::try_new(raw.hit_uv_height[1], raw.hit_uv_height[2])
                                        .map(Some)
                                }
                                TerrainTraversalMode::Infinite => Ok(None),
                            };
                            if let (Ok(world), Ok(bounded_uv)) = (world, bounded_uv) {
                                if let Ok(hit) = TerrainSurfaceHit::try_new(
                                    world,
                                    raw.hit_uv_height[3],
                                    bounded_uv,
                                ) {
                                    self.latest_pick = Some(SurfacePick {
                                        hit,
                                        local_position,
                                        context,
                                    });
                                }
                            }
                        }
                        ReadbackState::Idle
                    }
                    Ok(Err(_)) | Err(TryRecvError::Disconnected) => ReadbackState::Idle,
                    Err(TryRecvError::Empty) => ReadbackState::Mapping { context, receiver },
                },
            };
        }
    }

    pub fn latest_pick_for(
        &self,
        camera: &OrbitCamera,
        aspect: f32,
        cursor: (f32, f32),
        screen: (f32, f32),
        traversal: TerrainTraversalMode,
        render_origin: DVec2,
        surface_revision: u64,
    ) -> Option<SurfacePick> {
        let pick = self.latest_pick?;
        if pick.context.binding_revision != self.binding_revision
            || pick.context.surface_revision != surface_revision
            || pick.context.traversal != traversal
            || pick.context.render_origin != render_origin.to_array()
            || pick.context.camera_target != camera.target.to_array()
            || (pick.context.cursor.0 - cursor.0).abs() > 0.75
            || (pick.context.cursor.1 - cursor.1).abs() > 0.75
            || (pick.context.screen.0 - screen.0).abs() > 0.5
            || (pick.context.screen.1 - screen.1).abs() > 0.5
        {
            return None;
        }
        let current = match traversal {
            TerrainTraversalMode::Bounded => camera.view_proj(aspect),
            TerrainTraversalMode::Infinite => camera.view_proj_relative_to(aspect, render_origin),
        }
        .to_cols_array();
        current
            .iter()
            .zip(pick.context.view_proj)
            .all(|(a, b)| (*a - b).abs() <= 1.0e-4)
            .then_some(pick)
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, depth_tested: bool) {
        let Some(bind_group) = &self.render_bind_group else {
            return;
        };
        pass.set_pipeline(match self.traversal {
            TerrainTraversalMode::Infinite if depth_tested => &self.infinite_depth_pipeline,
            TerrainTraversalMode::Infinite => &self.infinite_overlay_pipeline,
            TerrainTraversalMode::Bounded if depth_tested => &self.depth_pipeline,
            TerrainTraversalMode::Bounded => &self.overlay_pipeline,
        });
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..RING_VERTEX_COUNT, 0..1);
    }

    pub fn upload_view_proj(&mut self, queue: &wgpu::Queue, view_proj: Mat4) {
        self.uniforms.view_proj = view_proj.to_cols_array_2d();
        self.uniforms.inv_view_proj = view_proj.inverse().to_cols_array_2d();
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&self.uniforms));
    }
}

/// Raycast the cursor onto the camera target plane. Retained for callers that
/// intentionally need a plane rather than the terrain surface.
pub fn pick_terrain_uv(
    camera: &OrbitCamera,
    aspect: f32,
    cursor: (f32, f32),
    screen: (f32, f32),
    world_size: (f32, f32),
) -> Option<(f32, f32)> {
    pick_at_plane(
        camera,
        aspect,
        cursor,
        screen,
        world_size,
        camera.target.y as f32,
    )
}

/// Raycast onto a CPU heightfield using a bounded surface traversal and bisection.
/// The application normally prefers the asynchronously completed GPU pick; this
/// is the non-blocking fallback while that result is in flight.
pub fn pick_terrain_uv_on_surface(
    camera: &OrbitCamera,
    aspect: f32,
    cursor: (f32, f32),
    screen: (f32, f32),
    world_size: (f32, f32),
    heights: Option<&Heightfield>,
) -> Option<(f32, f32)> {
    let Some(heights) = heights else {
        return pick_at_plane(camera, aspect, cursor, screen, world_size, 0.0);
    };
    let (origin, direction, mut t_min, mut t_max) = cursor_ray(camera, aspect, cursor, screen)?;
    let wx = world_size.0.max(1.0);
    let wz = world_size.1.max(1.0);
    clip_axis(origin.x, direction.x, 0.0, wx, &mut t_min, &mut t_max)?;
    clip_axis(origin.z, direction.z, 0.0, wz, &mut t_min, &mut t_max)?;
    if t_min > t_max {
        return None;
    }

    let sample_delta = |t: f32| {
        let p = origin + direction * t;
        p.y - sample_height_bilinear(heights, p.x / wx, p.z / wz)
    };
    let mut previous_t = t_min;
    let mut previous_delta = sample_delta(previous_t);
    if previous_delta <= 0.0 {
        let p = origin + direction * previous_t;
        return Some(((p.x / wx).clamp(0.0, 1.0), (p.z / wz).clamp(0.0, 1.0)));
    }
    let steps = heights
        .metrics
        .width
        .max(heights.metrics.height)
        .clamp(128, 8192);
    for step in 1..=steps {
        let t = t_min + (t_max - t_min) * (step as f32 / steps as f32);
        let delta = sample_delta(t);
        if delta <= 0.0 && previous_delta > 0.0 {
            let mut lo = previous_t;
            let mut hi = t;
            for _ in 0..12 {
                let mid = (lo + hi) * 0.5;
                if sample_delta(mid) > 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let p = origin + direction * hi;
            return Some(((p.x / wx).clamp(0.0, 1.0), (p.z / wz).clamp(0.0, 1.0)));
        }
        previous_t = t;
        previous_delta = delta;
    }
    None
}

fn cursor_ray(
    camera: &OrbitCamera,
    aspect: f32,
    cursor: (f32, f32),
    screen: (f32, f32),
) -> Option<(Vec3, Vec3, f32, f32)> {
    if screen.0 < 1.0 || screen.1 < 1.0 {
        return None;
    }
    let ndc_x = (cursor.0 / screen.0) * 2.0 - 1.0;
    let ndc_y = 1.0 - (cursor.1 / screen.1) * 2.0;
    let inv = camera.view_proj(aspect).inverse();
    let near = inv.project_point3(Vec3::new(ndc_x, ndc_y, 0.0));
    let far = inv.project_point3(Vec3::new(ndc_x, ndc_y, 1.0));
    let segment = far - near;
    let length = segment.length();
    (length.is_finite() && length > 1.0e-6).then_some((near, segment / length, 0.0, length))
}

fn clip_axis(
    origin: f32,
    direction: f32,
    min: f32,
    max: f32,
    t_min: &mut f32,
    t_max: &mut f32,
) -> Option<()> {
    if direction.abs() <= 1.0e-8 {
        return (origin >= min && origin <= max).then_some(());
    }
    let t0 = (min - origin) / direction;
    let t1 = (max - origin) / direction;
    *t_min = (*t_min).max(t0.min(t1));
    *t_max = (*t_max).min(t0.max(t1));
    (*t_min <= *t_max).then_some(())
}

fn pick_at_plane(
    camera: &OrbitCamera,
    aspect: f32,
    cursor: (f32, f32),
    screen: (f32, f32),
    world_size: (f32, f32),
    plane_y: f32,
) -> Option<(f32, f32)> {
    let (near, direction, _, _) = cursor_ray(camera, aspect, cursor, screen)?;
    if direction.y.abs() < 1.0e-6 {
        return None;
    }
    let t = (plane_y - near.y) / direction.y;
    if t < 0.0 {
        return None;
    }
    let hit = near + direction * t;
    let u = hit.x / world_size.0.max(1.0);
    let v = hit.z / world_size.1.max(1.0);
    ((0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v)).then_some((u, v))
}

fn sample_height_bilinear(heights: &Heightfield, u: f32, v: f32) -> f32 {
    let width = heights.metrics.width.max(1);
    let height = heights.metrics.height.max(1);
    let x = u.clamp(0.0, 1.0) * width.saturating_sub(1) as f32;
    let y = v.clamp(0.0, 1.0) * height.saturating_sub(1) as f32;
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let tx = x.fract();
    let ty = y.fract();
    let h0 = heights.get(x0, y0) * (1.0 - tx) + heights.get(x1, y0) * tx;
    let h1 = heights.get(x0, y1) * (1.0 - tx) + heights.get(x1, y1) * tx;
    h0 * (1.0 - ty) + h1 * ty
}

fn world_from_render_local(
    render_origin: [f64; 2],
    local_position: [f32; 3],
) -> Result<WorldPosition, terra_core::WorldError> {
    WorldPosition::try_new(
        render_origin[0] + f64::from(local_position[0]),
        render_origin[1] + f64::from(local_position[2]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use terra_core::heightfield::HeightfieldMetrics;

    fn cursor_for_world(camera: &OrbitCamera, screen: (f32, f32), point: Vec3) -> (f32, f32) {
        let clip = camera.view_proj(screen.0 / screen.1).project_point3(point);
        (
            (clip.x + 1.0) * 0.5 * screen.0,
            (1.0 - clip.y) * 0.5 * screen.1,
        )
    }

    #[test]
    fn elevated_surface_pick_round_trips_projected_world_point() {
        let metrics = HeightfieldMetrics::new(256, 256, 4096.0, 4096.0);
        let heights = Heightfield::filled(metrics, 700.0);
        let camera = OrbitCamera {
            target: Vec3::new(2048.0, 700.0, 2048.0).into(),
            distance: 4800.0,
            yaw: 0.7,
            pitch: 0.6,
            ..OrbitCamera::default()
        };
        let screen = (1600.0, 900.0);
        let expected = Vec3::new(3100.0, 700.0, 900.0);
        let cursor = cursor_for_world(&camera, screen, expected);
        let actual = pick_terrain_uv_on_surface(
            &camera,
            screen.0 / screen.1,
            cursor,
            screen,
            (4096.0, 4096.0),
            Some(&heights),
        )
        .expect("projected terrain point should be picked");
        assert!((actual.0 - expected.x / 4096.0).abs() < 1.0e-3);
        assert!((actual.1 - expected.z / 4096.0).abs() < 1.0e-3);
    }

    #[test]
    fn elevated_edge_hit_survives_target_plane_miss() {
        let metrics = HeightfieldMetrics::new(256, 256, 4096.0, 4096.0);
        let heights = Heightfield::filled(metrics, 800.0);
        let camera = OrbitCamera {
            target: Vec3::new(-800.0, 0.0, 2048.0).into(),
            distance: 4000.0,
            yaw: 0.0,
            pitch: 0.4,
            ..OrbitCamera::default()
        };
        let screen = (1600.0, 900.0);
        let expected = Vec3::new(100.0, 800.0, 2048.0);
        let cursor = cursor_for_world(&camera, screen, expected);
        assert_eq!(
            pick_at_plane(
                &camera,
                screen.0 / screen.1,
                cursor,
                screen,
                (4096.0, 4096.0),
                camera.target.y as f32,
            ),
            None,
            "the old target-plane seed rejects this visible edge hit"
        );
        let actual = pick_terrain_uv_on_surface(
            &camera,
            screen.0 / screen.1,
            cursor,
            screen,
            (4096.0, 4096.0),
            Some(&heights),
        )
        .expect("surface traversal must retain the visible edge hit");
        assert!((actual.0 - expected.x / 4096.0).abs() < 1.0e-3);
        assert!((actual.1 - 0.5).abs() < 1.0e-3);
    }

    #[test]
    fn render_origin_reconstruction_preserves_large_absolute_coordinates() {
        let expected = WorldPosition::try_new(5_000_000.25, -5_000_000.5).unwrap();
        let before_shift =
            world_from_render_local([4_999_936.0, -5_000_064.0], [64.25, 12.0, 63.5]).unwrap();
        let after_shift =
            world_from_render_local([5_000_000.0, -5_000_000.0], [0.25, 12.0, -0.5]).unwrap();

        assert_eq!(before_shift, expected);
        assert_eq!(after_shift, expected);
    }
}
