//! Immutable GPU height-pyramid materialization and measured error metadata.

use bytemuck::{Pod, Zeroable};
use std::sync::mpsc::{self, TryRecvError};
use terra_core::{TerrainContentStamp, TerrainPyramid, TerrainTileKey};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::output_identity::GpuOutputId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuPyramidContentIdentity {
    pub output_revision: u64,
    pub output: GpuOutputId,
    pub generation: u64,
    pub plan_revision: u64,
}

#[derive(Debug, Error)]
pub enum GpuPyramidError {
    #[error("source extent {width}x{height} is not a square pyramid level")]
    SourceLevelMissing { width: u32, height: u32 },
    #[error("pyramid level {0} is not materialized")]
    LevelMissing(u8),
    #[error("tile {key:?} does not belong to this pyramid")]
    InvalidTile { key: TerrainTileKey },
    #[error("geometric-error metadata readback failed: {0}")]
    MetadataReadback(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct GpuPyramidPlanningMetadata {
    pub identity: GpuPyramidContentIdentity,
    pub geometric_errors: Vec<f32>,
}

/// One-shot, non-blocking transfer of the compact geometric-error buffer. This
/// never maps height textures and is polled with `Maintain::Poll` on frame work.
pub struct GpuPyramidErrorReadback {
    identity: GpuPyramidContentIdentity,
    buffer: wgpu::Buffer,
    len: usize,
    receiver: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

impl GpuPyramidErrorReadback {
    pub fn identity(&self) -> GpuPyramidContentIdentity {
        self.identity
    }

    pub fn poll(
        &mut self,
        device: &wgpu::Device,
    ) -> Result<Option<GpuPyramidPlanningMetadata>, GpuPyramidError> {
        let Some(receiver) = self.receiver.as_ref() else {
            return Ok(None);
        };
        let _ = device.poll(wgpu::Maintain::Poll);
        match receiver.try_recv() {
            Ok(Ok(())) => {
                let mapped = self.buffer.slice(..).get_mapped_range();
                let bits: &[u32] = bytemuck::cast_slice(&mapped[..self.len * 4]);
                let geometric_errors = bits
                    .iter()
                    .map(|bits| {
                        let value = f32::from_bits(*bits);
                        if value.is_finite() && value >= 0.0 {
                            value
                        } else {
                            f32::MAX
                        }
                    })
                    .collect();
                drop(mapped);
                self.buffer.unmap();
                self.receiver = None;
                Ok(Some(GpuPyramidPlanningMetadata {
                    identity: self.identity,
                    geometric_errors,
                }))
            }
            Ok(Err(error)) => {
                self.receiver = None;
                Err(GpuPyramidError::MetadataReadback(error.to_string()))
            }
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => {
                self.receiver = None;
                Err(GpuPyramidError::MetadataReadback(
                    "map callback disconnected".to_string(),
                ))
            }
        }
    }
}

struct GpuPyramidLevel {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

/// Immutable snapshot of one accepted GPU output and all coarser levels.
///
/// The dense metadata buffer is content metadata indexed by
/// `TerrainPyramid::tile_metadata_index`; it contains no residency flags or
/// physical page handles.
pub struct GpuHeightPyramid {
    descriptor: TerrainPyramid,
    identity: GpuPyramidContentIdentity,
    source_level: u8,
    levels: Vec<GpuPyramidLevel>,
    error_bits: wgpu::Buffer,
}

impl GpuHeightPyramid {
    pub fn descriptor(&self) -> &TerrainPyramid {
        &self.descriptor
    }

    pub fn identity(&self) -> GpuPyramidContentIdentity {
        self.identity
    }

    pub fn source_level(&self) -> u8 {
        self.source_level
    }

    pub fn level_texture(&self, level: u8) -> Result<&wgpu::Texture, GpuPyramidError> {
        self.levels
            .get(level as usize)
            .map(|entry| &entry.texture)
            .ok_or(GpuPyramidError::LevelMissing(level))
    }

    pub fn level_view(&self, level: u8) -> Result<&wgpu::TextureView, GpuPyramidError> {
        self.levels
            .get(level as usize)
            .map(|entry| &entry.view)
            .ok_or(GpuPyramidError::LevelMissing(level))
    }

    pub fn error_buffer(&self) -> &wgpu::Buffer {
        &self.error_bits
    }

    pub fn tile_error_index(&self, key: &TerrainTileKey) -> Result<u32, GpuPyramidError> {
        if key.level > self.source_level {
            return Err(GpuPyramidError::InvalidTile { key: key.clone() });
        }
        self.descriptor
            .tile_metadata_index(key.level, key.tile)
            .ok_or_else(|| GpuPyramidError::InvalidTile { key: key.clone() })
    }

    /// Begin the compact, once-per-content metadata transfer required by the CPU
    /// camera-demand walk. Queue ordering makes the copy observe materialization's
    /// completed error writes without blocking the interactive thread.
    pub fn begin_error_readback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> GpuPyramidErrorReadback {
        let len = self.descriptor.metadata_len() as usize;
        let size = (len.max(1) * 4) as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain-pyramid-planning-metadata"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("terrain-pyramid-planning-metadata-copy"),
        });
        encoder.copy_buffer_to_buffer(&self.error_bits, 0, &buffer, 0, size);
        queue.submit(Some(encoder.finish()));
        let slice = buffer.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        GpuPyramidErrorReadback {
            identity: self.identity,
            buffer,
            len,
            receiver: Some(receiver),
        }
    }

    /// Test observability only. Production generation and publication never map
    /// height or metadata resources to the CPU.
    #[doc(hidden)]
    pub fn read_level_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        level: u8,
    ) -> Result<Vec<f32>, GpuPyramidError> {
        let metrics = self
            .descriptor
            .level_metrics(level)
            .ok_or(GpuPyramidError::LevelMissing(level))?;
        let texture = self.level_texture(level)?;
        Ok(read_r32_texture_blocking(
            device,
            queue,
            texture,
            metrics.width,
            metrics.height,
        ))
    }

    /// Test observability only; see [`Self::read_level_blocking`].
    #[doc(hidden)]
    pub fn read_error_bits_blocking(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<u32> {
        read_u32_buffer_blocking(
            device,
            queue,
            &self.error_bits,
            self.descriptor.metadata_len() as usize,
        )
    }
}

impl GpuPyramidContentIdentity {
    pub fn content_stamp(self) -> TerrainContentStamp {
        TerrainContentStamp {
            document_revision: self.generation,
            plan_revision: self.plan_revision,
            output_revision: self.output_revision,
            content_revision: self.output.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DownsampleUniforms {
    child_width: u32,
    child_height: u32,
    parent_width: u32,
    parent_height: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ErrorUniforms {
    child_width: u32,
    child_height: u32,
    parent_width: u32,
    parent_height: u32,
    tile_size: u32,
    metadata_offset: u32,
    tiles_x: u32,
    _pad: u32,
}

/// Reusable compute pipelines for generating immutable height pyramids.
pub struct GpuHeightPyramidMaterializer {
    downsample_layout: wgpu::BindGroupLayout,
    downsample: wgpu::ComputePipeline,
    error_layout: wgpu::BindGroupLayout,
    error: wgpu::ComputePipeline,
}

impl GpuHeightPyramidMaterializer {
    pub fn new(device: &wgpu::Device) -> Self {
        let downsample_layout = texture_compute_layout(device, "pyramid-downsample-bgl", false);
        let error_layout = texture_compute_layout(device, "pyramid-error-bgl", true);
        let downsample = compute_pipeline(
            device,
            "pyramid-downsample",
            include_str!("shaders/terrain_pyramid_downsample.wgsl"),
            &downsample_layout,
        );
        let error = compute_pipeline(
            device,
            "pyramid-error",
            include_str!("shaders/terrain_pyramid_error.wgsl"),
            &error_layout,
        );
        Self {
            downsample_layout,
            downsample,
            error_layout,
            error,
        }
    }

    pub fn materialize(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        descriptor: &TerrainPyramid,
        source: &wgpu::Texture,
        source_extent: (u32, u32),
        identity: GpuPyramidContentIdentity,
    ) -> Result<GpuHeightPyramid, GpuPyramidError> {
        let source_level = descriptor
            .levels
            .iter()
            .find(|level| {
                level.resolution == source_extent.0 && level.resolution == source_extent.1
            })
            .map(|level| level.index)
            .ok_or(GpuPyramidError::SourceLevelMissing {
                width: source_extent.0,
                height: source_extent.1,
            })?;

        let mut levels = Vec::with_capacity(source_level as usize + 1);
        for level in descriptor.levels.iter().take(source_level as usize + 1) {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("terrain-pyramid-level"),
                size: wgpu::Extent3d {
                    width: level.resolution,
                    height: level.resolution,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            levels.push(GpuPyramidLevel { texture, view });
        }

        let error_bits = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain-pyramid-error-bits"),
            size: u64::from(descriptor.metadata_len().max(1)) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("terrain-pyramid-materialize"),
        });
        encoder.clear_buffer(&error_bits, 0, None);
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: source,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &levels[source_level as usize].texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: source_extent.0,
                height: source_extent.1,
                depth_or_array_layers: 1,
            },
        );

        for child_level in (1..=source_level).rev() {
            let parent_level = child_level - 1;
            let child_metrics = descriptor.level_metrics(child_level).unwrap();
            let parent_metrics = descriptor.level_metrics(parent_level).unwrap();
            let downsample_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("terrain-pyramid-downsample-uniform"),
                contents: bytemuck::bytes_of(&DownsampleUniforms {
                    child_width: child_metrics.width,
                    child_height: child_metrics.height,
                    parent_width: parent_metrics.width,
                    parent_height: parent_metrics.height,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let downsample_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("terrain-pyramid-downsample-bg"),
                layout: &self.downsample_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: downsample_uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &levels[child_level as usize].view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &levels[parent_level as usize].view,
                        ),
                    },
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("terrain-pyramid-downsample-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.downsample);
                pass.set_bind_group(0, &downsample_bind, &[]);
                pass.dispatch_workgroups(
                    parent_metrics.width.div_ceil(8),
                    parent_metrics.height.div_ceil(8),
                    1,
                );
            }

            let metadata_offset = descriptor
                .tile_metadata_index(child_level, terra_core::TileId { tx: 0, tz: 0 })
                .unwrap();
            let error_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("terrain-pyramid-error-uniform"),
                contents: bytemuck::bytes_of(&ErrorUniforms {
                    child_width: child_metrics.width,
                    child_height: child_metrics.height,
                    parent_width: parent_metrics.width,
                    parent_height: parent_metrics.height,
                    tile_size: child_metrics.tile_size,
                    metadata_offset,
                    tiles_x: child_metrics.tiles_x(),
                    _pad: 0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let error_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("terrain-pyramid-error-bg"),
                layout: &self.error_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: error_uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &levels[child_level as usize].view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &levels[parent_level as usize].view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: error_bits.as_entire_binding(),
                    },
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("terrain-pyramid-error-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.error);
                pass.set_bind_group(0, &error_bind, &[]);
                pass.dispatch_workgroups(
                    child_metrics.width.div_ceil(8),
                    child_metrics.height.div_ceil(8),
                    1,
                );
            }
        }
        queue.submit(Some(encoder.finish()));

        Ok(GpuHeightPyramid {
            descriptor: descriptor.clone(),
            identity,
            source_level,
            levels,
            error_bits,
        })
    }
}

fn texture_compute_layout(
    device: &wgpu::Device,
    label: &str,
    with_error_buffer: bool,
) -> wgpu::BindGroupLayout {
    let mut entries = vec![
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
            ty: if with_error_buffer {
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                }
            } else {
                wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::R32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                }
            },
            count: None,
        },
    ];
    if with_error_buffer {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

fn compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    terra_core::shader_progress::record_shader_compiled();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn read_r32_texture_blocking(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Vec<f32> {
    let unpadded = width * 4;
    let padded =
        unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("terrain-pyramid-level-readback"),
        size: u64::from(padded) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("terrain-pyramid-level-readback"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));
    let bytes = map_buffer_blocking(device, &buffer);
    let mut result = Vec::with_capacity((width * height) as usize);
    for row in 0..height as usize {
        let offset = row * padded as usize;
        result.extend_from_slice(bytemuck::cast_slice(
            &bytes[offset..offset + unpadded as usize],
        ));
    }
    drop(bytes);
    buffer.unmap();
    result
}

fn read_u32_buffer_blocking(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    len: usize,
) -> Vec<u32> {
    let size = (len.max(1) * 4) as u64;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("terrain-pyramid-error-readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("terrain-pyramid-error-readback"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &buffer, 0, size);
    queue.submit(Some(encoder.finish()));
    let bytes = map_buffer_blocking(device, &buffer);
    let result = bytemuck::cast_slice(&bytes[..len * 4]).to_vec();
    drop(bytes);
    buffer.unmap();
    result
}

fn map_buffer_blocking<'a>(
    device: &wgpu::Device,
    buffer: &'a wgpu::Buffer,
) -> wgpu::BufferView<'a> {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().expect("map callback").expect("GPU readback map");
    slice.get_mapped_range()
}
