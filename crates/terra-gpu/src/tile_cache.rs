use bytemuck::{Pod, Zeroable};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use terra_core::{
    FieldId, HeightTile, TerrainContentStamp, TerrainEvaluationDomain, TerrainPyramid,
    TerrainTileKey, TileCacheError, TileCacheInsert, TilePageHandle, TileResidencyCache,
};
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::pyramid::{GpuHeightPyramid, GpuPyramidError};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuPageTableEntry {
    pub key_hash_lo: u32,
    pub key_hash_hi: u32,
    pub generation: u32,
    pub valid: u32,
    pub level: u32,
    pub tile_x: u32,
    pub tile_z: u32,
    pub width: u32,
    pub height: u32,
    pub halo: u32,
    pub document_revision_lo: u32,
    pub document_revision_hi: u32,
    pub plan_revision_lo: u32,
    pub plan_revision_hi: u32,
    pub output_revision_lo: u32,
    pub output_revision_hi: u32,
    pub content_revision_lo: u32,
    pub content_revision_hi: u32,
    pub published_frame_lo: u32,
    pub published_frame_hi: u32,
}

impl GpuPageTableEntry {
    fn resident(
        key: &TerrainTileKey,
        handle: TilePageHandle,
        width: u32,
        height: u32,
        halo: u32,
        content: TerrainContentStamp,
        published_frame: u64,
    ) -> Self {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        Self {
            key_hash_lo: hash as u32,
            key_hash_hi: (hash >> 32) as u32,
            generation: handle.generation,
            valid: 1,
            level: key.level as u32,
            tile_x: key.tile.tx,
            tile_z: key.tile.tz,
            width,
            height,
            halo,
            document_revision_lo: content.document_revision as u32,
            document_revision_hi: (content.document_revision >> 32) as u32,
            plan_revision_lo: content.plan_revision as u32,
            plan_revision_hi: (content.plan_revision >> 32) as u32,
            output_revision_lo: content.output_revision as u32,
            output_revision_hi: (content.output_revision >> 32) as u32,
            content_revision_lo: content.content_revision as u32,
            content_revision_hi: (content.content_revision >> 32) as u32,
            published_frame_lo: published_frame as u32,
            published_frame_hi: (published_frame >> 32) as u32,
        }
    }

    fn invalid(generation: u32) -> Self {
        Self {
            generation,
            ..Zeroable::zeroed()
        }
    }
}

/// Dense virtual-to-physical page mapping. The index is supplied by
/// `TerrainPyramid::tile_metadata_index`; this is shader-visible residency state,
/// not a CPU-side residency mirror.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuVirtualPageEntry {
    pub physical_slot: u32,
    pub generation: u32,
    pub valid: u32,
    pub _pad: u32,
}

impl GpuVirtualPageEntry {
    fn resident(handle: TilePageHandle) -> Self {
        Self {
            physical_slot: handle.slot,
            generation: handle.generation,
            valid: 1,
            _pad: 0,
        }
    }
}

/// Immutable level addressing metadata consumed by the terrain shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuTerrainLevelEntry {
    pub resolution: u32,
    pub tiles_x: u32,
    pub tiles_z: u32,
    pub metadata_offset: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuTileUpload {
    pub handle: TilePageHandle,
    pub evicted: Vec<TerrainTileKey>,
}

#[derive(Debug, Error)]
pub enum GpuTileCacheError {
    #[error("tile atlas needs at least one page")]
    EmptyAtlas,
    #[error("tile page extent {requested} exceeds device limit {limit}")]
    PageExtentUnsupported { requested: u32, limit: u32 },
    #[error("tile page count {requested} exceeds device limit {limit}")]
    PageCountUnsupported { requested: u32, limit: u32 },
    #[error("tile payload {width}x{height} exceeds atlas page extent {page_extent}")]
    TileTooLarge {
        width: u32,
        height: u32,
        page_extent: u32,
    },
    #[error("tile residency failed: {0:?}")]
    Residency(TileCacheError),
    #[error("pyramid content revision {content} is stale; live output revision is {live}")]
    StalePyramid { content: u64, live: u64 },
    #[error("pyramid content identity is stale")]
    StalePyramidIdentity,
    #[error("evaluated tile content identity is stale")]
    StaleEvaluatedTileIdentity,
    #[error("tile {0:?} is not addressable by the configured height hierarchy")]
    UnaddressableTile(TerrainTileKey),
    #[error("pyramid tile publication failed: {0}")]
    Pyramid(#[from] GpuPyramidError),
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PackUniforms {
    source_width: u32,
    source_height: u32,
    origin_x: u32,
    origin_z: u32,
    interior_width: u32,
    interior_height: u32,
    halo: u32,
    page_extent: u32,
}

/// R32Float texture-array atlas plus a shader-readable page table.
///
/// Each physical array layer is a reusable tile page. The CPU residency policy supplies a
/// generation-checked handle; shaders can reject stale handles using the matching page-table row.
pub struct GpuTileAtlas {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    page_table: wgpu::Buffer,
    virtual_page_table: wgpu::Buffer,
    level_table: wgpu::Buffer,
    hierarchy: Option<TerrainPyramid>,
    level_count: u32,
    virtual_page_count: u32,
    residency: TileResidencyCache,
    page_extent: u32,
    tile_size: u32,
    halo: u32,
    max_pages: u32,
    page_bytes: u64,
    pack_layout: wgpu::BindGroupLayout,
    pack_pipeline: wgpu::ComputePipeline,
}

impl GpuTileAtlas {
    pub fn new(
        device: &wgpu::Device,
        tile_size: u32,
        halo: u32,
        max_pages: u32,
    ) -> Result<Self, GpuTileCacheError> {
        if max_pages == 0 {
            return Err(GpuTileCacheError::EmptyAtlas);
        }
        let page_extent = tile_size.saturating_add(halo.saturating_mul(2)).max(1);
        let limits = device.limits();
        if page_extent > limits.max_texture_dimension_2d {
            return Err(GpuTileCacheError::PageExtentUnsupported {
                requested: page_extent,
                limit: limits.max_texture_dimension_2d,
            });
        }
        if max_pages > limits.max_texture_array_layers {
            return Err(GpuTileCacheError::PageCountUnsupported {
                requested: max_pages,
                limit: limits.max_texture_array_layers,
            });
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain-tile-atlas-r32"),
            size: wgpu::Extent3d {
                width: page_extent,
                height: page_extent,
                depth_or_array_layers: max_pages,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("terrain-tile-atlas-view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let zero_entries = vec![GpuPageTableEntry::zeroed(); max_pages as usize];
        let page_table = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain-tile-page-table"),
            contents: bytemuck::cast_slice(&zero_entries),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let virtual_page_table = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain-virtual-page-table"),
            contents: bytemuck::bytes_of(&GpuVirtualPageEntry::zeroed()),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let level_table = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain-level-table"),
            contents: bytemuck::bytes_of(&GpuTerrainLevelEntry::zeroed()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let page_bytes = u64::from(page_extent) * u64::from(page_extent) * 4;
        let pack_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain-pyramid-pack-bgl"),
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
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        terra_core::shader_progress::record_shader_compiled();
        let pack_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain-pyramid-pack"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/terrain_pyramid_pack.wgsl").into(),
            ),
        });
        let pack_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain-pyramid-pack-pl"),
            bind_group_layouts: &[&pack_layout],
            push_constant_ranges: &[],
        });
        let pack_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("terrain-pyramid-pack"),
            layout: Some(&pack_pipeline_layout),
            module: &pack_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self {
            texture,
            view,
            page_table,
            virtual_page_table,
            level_table,
            hierarchy: None,
            level_count: 0,
            virtual_page_count: 0,
            residency: TileResidencyCache::new(page_bytes * u64::from(max_pages)),
            page_extent,
            tile_size,
            halo,
            max_pages,
            page_bytes,
            pack_layout,
            pack_pipeline,
        })
    }

    /// Configure the immutable virtual address space used by streamed height pages.
    /// Reconfiguration retires all prior residency before replacing shader-visible
    /// directory buffers, so old document mappings cannot survive a shape change.
    pub fn configure_hierarchy(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        hierarchy: &TerrainPyramid,
    ) {
        self.clear(queue);
        let mut offset = 0u32;
        let levels: Vec<_> = hierarchy
            .levels
            .iter()
            .map(|level| {
                let metrics = hierarchy
                    .level_metrics(level.index)
                    .expect("descriptor level must have metrics");
                let entry = GpuTerrainLevelEntry {
                    resolution: level.resolution,
                    tiles_x: metrics.tiles_x(),
                    tiles_z: metrics.tiles_z(),
                    metadata_offset: offset,
                };
                offset = offset.saturating_add(metrics.tiles_x().saturating_mul(metrics.tiles_z()));
                entry
            })
            .collect();
        let virtual_count = hierarchy.metadata_len().max(1);
        let virtual_entries = vec![GpuVirtualPageEntry::zeroed(); virtual_count as usize];
        self.virtual_page_table = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain-virtual-page-table"),
            contents: bytemuck::cast_slice(&virtual_entries),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let level_entries = if levels.is_empty() {
            vec![GpuTerrainLevelEntry::zeroed()]
        } else {
            levels
        };
        self.level_table = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain-level-table"),
            contents: bytemuck::cast_slice(&level_entries),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        self.level_count = hierarchy.levels.len() as u32;
        self.virtual_page_count = hierarchy.metadata_len();
        self.hierarchy = Some(hierarchy.clone());
    }

    fn virtual_index(&self, key: &TerrainTileKey) -> Option<u32> {
        (key.layer.is_none() && key.field == FieldId::Height)
            .then(|| {
                self.hierarchy
                    .as_ref()?
                    .tile_metadata_index(key.level, key.tile)
            })
            .flatten()
    }

    fn require_virtual_index(&self, key: &TerrainTileKey) -> Result<u32, GpuTileCacheError> {
        self.virtual_index(key)
            .ok_or_else(|| GpuTileCacheError::UnaddressableTile(key.clone()))
    }

    fn invalidate_key(&self, queue: &wgpu::Queue, key: &TerrainTileKey) {
        if let Some(index) = self.virtual_index(key) {
            self.write_virtual_entry(queue, index, GpuVirtualPageEntry::zeroed());
        }
    }

    fn allocate_page(
        &mut self,
        queue: &wgpu::Queue,
        key: &TerrainTileKey,
        revision: u64,
        input_revision_hash: u64,
        content: Option<TerrainContentStamp>,
    ) -> Result<TileCacheInsert, GpuTileCacheError> {
        if self.hierarchy.is_some() {
            self.require_virtual_index(key)?;
        }
        let insert = self
            .residency
            .insert_with_content(
                key.clone(),
                self.page_bytes,
                revision,
                input_revision_hash,
                content,
            )
            .map_err(GpuTileCacheError::Residency)?;
        for evicted in &insert.evicted {
            self.invalidate_key(queue, &evicted.key);
            self.write_page_entry(
                queue,
                evicted.handle.slot,
                GpuPageTableEntry::invalid(evicted.handle.generation),
            );
        }
        // Replacing an existing virtual key keeps its physical slot but must not
        // leave the old mapping visible while the payload is being overwritten.
        self.invalidate_key(queue, key);
        Ok(insert)
    }

    pub fn upload_height_tile(
        &mut self,
        queue: &wgpu::Queue,
        key: TerrainTileKey,
        tile: &HeightTile,
        revision: u64,
        input_revision_hash: u64,
    ) -> Result<GpuTileUpload, GpuTileCacheError> {
        self.upload_height_tile_with_content(
            queue,
            key,
            tile,
            revision,
            input_revision_hash,
            None,
            0,
        )
    }

    pub fn upload_height_tile_current(
        &mut self,
        queue: &wgpu::Queue,
        key: TerrainTileKey,
        tile: &HeightTile,
        content: TerrainContentStamp,
    ) -> Result<GpuTileUpload, GpuTileCacheError> {
        self.upload_height_tile_current_at_frame(queue, key, tile, content, 0)
    }

    pub fn upload_height_tile_current_at_frame(
        &mut self,
        queue: &wgpu::Queue,
        key: TerrainTileKey,
        tile: &HeightTile,
        content: TerrainContentStamp,
        published_frame: u64,
    ) -> Result<GpuTileUpload, GpuTileCacheError> {
        self.upload_height_tile_with_content(
            queue,
            key,
            tile,
            content.output_revision,
            content.content_revision,
            Some(content),
            published_frame,
        )
    }

    // Shared implementation keeps the legacy revision/hash API and the full
    // content/frame API on one publication-ordering path.
    #[allow(clippy::too_many_arguments)]
    fn upload_height_tile_with_content(
        &mut self,
        queue: &wgpu::Queue,
        key: TerrainTileKey,
        tile: &HeightTile,
        revision: u64,
        input_revision_hash: u64,
        content: Option<TerrainContentStamp>,
        published_frame: u64,
    ) -> Result<GpuTileUpload, GpuTileCacheError> {
        if tile.stride() > self.page_extent || tile.stride_z() > self.page_extent {
            return Err(GpuTileCacheError::TileTooLarge {
                width: tile.stride(),
                height: tile.stride_z(),
                page_extent: self.page_extent,
            });
        }
        let insert = self.allocate_page(queue, &key, revision, input_revision_hash, content)?;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: insert.handle.slot,
                },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(tile.data()),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(tile.stride() * 4),
                rows_per_image: Some(tile.stride_z()),
            },
            wgpu::Extent3d {
                width: tile.stride(),
                height: tile.stride_z(),
                depth_or_array_layers: 1,
            },
        );
        self.write_page_entry(
            queue,
            insert.handle.slot,
            GpuPageTableEntry::resident(
                &key,
                insert.handle,
                tile.interior_width,
                tile.interior_height,
                tile.halo,
                content.unwrap_or(TerrainContentStamp {
                    output_revision: revision,
                    content_revision: input_revision_hash,
                    ..TerrainContentStamp::default()
                }),
                published_frame,
            ),
        );
        if let Some(virtual_index) = self.virtual_index(&key) {
            self.write_virtual_entry(
                queue,
                virtual_index,
                GpuVirtualPageEntry::resident(insert.handle),
            );
        }
        Ok(GpuTileUpload {
            handle: insert.handle,
            evicted: insert
                .evicted
                .into_iter()
                .map(|eviction| eviction.key)
                .collect(),
        })
    }

    /// Publish one immutable pyramid tile directly from GPU level storage.
    ///
    /// Page payload work is submitted before the valid page-table row is queued,
    /// so shader-visible residency cannot precede its texture contents.
    pub fn publish_pyramid_tile(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pyramid: &GpuHeightPyramid,
        key: TerrainTileKey,
        live_output_revision: u64,
    ) -> Result<GpuTileUpload, GpuTileCacheError> {
        let identity = pyramid.identity();
        if identity.output_revision != live_output_revision {
            return Err(GpuTileCacheError::StalePyramid {
                content: identity.output_revision,
                live: live_output_revision,
            });
        }
        self.publish_pyramid_tile_with_identity(device, queue, pyramid, key, identity, 0)
    }

    pub fn publish_pyramid_tile_current(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pyramid: &GpuHeightPyramid,
        key: TerrainTileKey,
        live_content: TerrainContentStamp,
    ) -> Result<GpuTileUpload, GpuTileCacheError> {
        self.publish_pyramid_tile_current_at_frame(device, queue, pyramid, key, live_content, 0)
    }

    pub fn publish_pyramid_tile_current_at_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pyramid: &GpuHeightPyramid,
        key: TerrainTileKey,
        live_content: TerrainContentStamp,
        published_frame: u64,
    ) -> Result<GpuTileUpload, GpuTileCacheError> {
        let identity = pyramid.identity();
        if identity.content_stamp() != live_content {
            return Err(GpuTileCacheError::StalePyramidIdentity);
        }
        self.publish_pyramid_tile_with_identity(
            device,
            queue,
            pyramid,
            key,
            identity,
            published_frame,
        )
    }

    fn publish_pyramid_tile_with_identity(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pyramid: &GpuHeightPyramid,
        key: TerrainTileKey,
        identity: crate::pyramid::GpuPyramidContentIdentity,
        published_frame: u64,
    ) -> Result<GpuTileUpload, GpuTileCacheError> {
        pyramid.tile_error_index(&key)?;
        let metrics = pyramid
            .descriptor()
            .level_metrics(key.level)
            .ok_or_else(|| GpuPyramidError::InvalidTile { key: key.clone() })?;
        let extent = pyramid
            .descriptor()
            .tile_extent(key.level, key.tile)
            .ok_or_else(|| GpuPyramidError::InvalidTile { key: key.clone() })?;
        if extent.width + self.halo * 2 > self.page_extent
            || extent.height + self.halo * 2 > self.page_extent
        {
            return Err(GpuTileCacheError::TileTooLarge {
                width: extent.width + self.halo * 2,
                height: extent.height + self.halo * 2,
                page_extent: self.page_extent,
            });
        }
        let source_view = pyramid.level_view(key.level)?;
        let insert = self.allocate_page(
            queue,
            &key,
            identity.output_revision,
            identity.output.0,
            Some(identity.content_stamp()),
        )?;
        // Keep this slot invalid until its new payload has been submitted. This
        // also protects replacement of an existing virtual key in the same slot.
        self.write_page_entry(
            queue,
            insert.handle.slot,
            GpuPageTableEntry::invalid(insert.handle.generation),
        );

        let destination_view = self.texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("terrain-pyramid-atlas-slot-storage"),
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_array_layer: insert.handle.slot,
            array_layer_count: Some(1),
            ..Default::default()
        });
        let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain-pyramid-pack-uniform"),
            contents: bytemuck::bytes_of(&PackUniforms {
                source_width: metrics.width,
                source_height: metrics.height,
                origin_x: extent.origin_x,
                origin_z: extent.origin_z,
                interior_width: extent.width,
                interior_height: extent.height,
                halo: self.halo,
                page_extent: self.page_extent,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain-pyramid-pack-bg"),
            layout: &self.pack_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&destination_view),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("terrain-pyramid-pack-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("terrain-pyramid-pack-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pack_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                self.page_extent.div_ceil(8),
                self.page_extent.div_ceil(8),
                1,
            );
        }
        queue.submit(Some(encoder.finish()));
        self.write_page_entry(
            queue,
            insert.handle.slot,
            GpuPageTableEntry::resident(
                &key,
                insert.handle,
                extent.width,
                extent.height,
                self.halo,
                identity.content_stamp(),
                published_frame,
            ),
        );
        if let Some(virtual_index) = self.virtual_index(&key) {
            self.write_virtual_entry(
                queue,
                virtual_index,
                GpuVirtualPageEntry::resident(insert.handle),
            );
        }
        Ok(GpuTileUpload {
            handle: insert.handle,
            evicted: insert
                .evicted
                .into_iter()
                .map(|eviction| eviction.key)
                .collect(),
        })
    }

    /// Publish a completed tile-domain evaluation. Residency is not allocated
    /// until the complete content stamp has been revalidated by the caller.
    pub fn publish_evaluated_tile_current_at_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: &wgpu::TextureView,
        domain: &TerrainEvaluationDomain,
        live_content: TerrainContentStamp,
        published_frame: u64,
    ) -> Result<GpuTileUpload, GpuTileCacheError> {
        if domain.content != live_content {
            return Err(GpuTileCacheError::StaleEvaluatedTileIdentity);
        }
        let hierarchy = self
            .hierarchy
            .as_ref()
            .ok_or_else(|| GpuTileCacheError::UnaddressableTile(domain.key.clone()))?;
        let expected = hierarchy
            .tile_extent(domain.key.level, domain.key.tile)
            .ok_or_else(|| GpuTileCacheError::UnaddressableTile(domain.key.clone()))?;
        if expected != domain.interior {
            return Err(GpuTileCacheError::UnaddressableTile(domain.key.clone()));
        }
        if expected.width + self.halo * 2 > self.page_extent
            || expected.height + self.halo * 2 > self.page_extent
        {
            return Err(GpuTileCacheError::TileTooLarge {
                width: expected.width + self.halo * 2,
                height: expected.height + self.halo * 2,
                page_extent: self.page_extent,
            });
        }
        if domain.publication_halo < self.halo {
            return Err(GpuTileCacheError::TileTooLarge {
                width: expected.width + self.halo * 2,
                height: expected.height + self.halo * 2,
                page_extent: expected.width + domain.publication_halo * 2,
            });
        }

        let key = domain.key.clone();
        let insert = self.allocate_page(
            queue,
            &key,
            live_content.output_revision,
            live_content.content_revision,
            Some(live_content),
        )?;
        self.write_page_entry(
            queue,
            insert.handle.slot,
            GpuPageTableEntry::invalid(insert.handle.generation),
        );
        let destination_view = self.texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("terrain-evaluated-atlas-slot-storage"),
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_array_layer: insert.handle.slot,
            array_layer_count: Some(1),
            ..Default::default()
        });
        let interior_local_x = domain.interior.origin_x - domain.evaluation.origin_x;
        let interior_local_z = domain.interior.origin_z - domain.evaluation.origin_z;
        let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain-evaluated-pack-uniform"),
            contents: bytemuck::bytes_of(&PackUniforms {
                source_width: domain.evaluation.width,
                source_height: domain.evaluation.height,
                origin_x: interior_local_x,
                origin_z: interior_local_z,
                interior_width: expected.width,
                interior_height: expected.height,
                halo: self.halo,
                page_extent: self.page_extent,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain-evaluated-pack-bg"),
            layout: &self.pack_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(source),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&destination_view),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("terrain-evaluated-pack-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("terrain-evaluated-pack-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pack_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                self.page_extent.div_ceil(8),
                self.page_extent.div_ceil(8),
                1,
            );
        }
        queue.submit(Some(encoder.finish()));
        self.write_page_entry(
            queue,
            insert.handle.slot,
            GpuPageTableEntry::resident(
                &key,
                insert.handle,
                expected.width,
                expected.height,
                self.halo,
                live_content,
                published_frame,
            ),
        );
        if let Some(virtual_index) = self.virtual_index(&key) {
            self.write_virtual_entry(
                queue,
                virtual_index,
                GpuVirtualPageEntry::resident(insert.handle),
            );
        }
        Ok(GpuTileUpload {
            handle: insert.handle,
            evicted: insert
                .evicted
                .into_iter()
                .map(|eviction| eviction.key)
                .collect(),
        })
    }

    /// Invalidate all document-owned residency without reallocating atlas resources.
    ///
    /// Texture contents may remain in physical pages because the page table is made
    /// entirely invalid before streaming can be enabled for the next document.
    pub fn clear(&mut self, queue: &wgpu::Queue) {
        self.residency.clear();
        let invalid_entries = vec![GpuPageTableEntry::zeroed(); self.max_pages as usize];
        queue.write_buffer(&self.page_table, 0, bytemuck::cast_slice(&invalid_entries));
        if self.virtual_page_count > 0 {
            let virtual_entries =
                vec![GpuVirtualPageEntry::zeroed(); self.virtual_page_count as usize];
            queue.write_buffer(
                &self.virtual_page_table,
                0,
                bytemuck::cast_slice(&virtual_entries),
            );
        }
    }

    /// Remove one page from both the CPU policy cache and shader-visible tables.
    pub fn unpublish(&mut self, queue: &wgpu::Queue, key: &TerrainTileKey) -> bool {
        let Some(handle) = self.residency.peek(key).map(|entry| entry.handle) else {
            return false;
        };
        if !self.residency.remove(key) {
            return false;
        }
        self.invalidate_key(queue, key);
        self.write_page_entry(
            queue,
            handle.slot,
            GpuPageTableEntry::invalid(handle.generation.wrapping_add(1).max(1)),
        );
        true
    }

    pub fn lookup(&mut self, key: &TerrainTileKey) -> Option<TilePageHandle> {
        self.residency.get(key).map(|entry| entry.handle)
    }

    pub fn is_current(&self, key: &TerrainTileKey, content: TerrainContentStamp) -> bool {
        self.residency.is_current(key, content)
    }

    pub fn pin(&mut self, key: &TerrainTileKey) -> bool {
        self.residency.pin(key)
    }

    pub fn unpin(&mut self, key: &TerrainTileKey) -> bool {
        self.residency.unpin(key)
    }

    pub fn texture_view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Fresh view for bind-group ownership (atlas texture remains authoritative).
    pub fn create_texture_view(&self) -> wgpu::TextureView {
        self.texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("terrain-tile-atlas-view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        })
    }

    pub fn page_table_buffer(&self) -> &wgpu::Buffer {
        &self.page_table
    }

    pub fn page_table_buffer_cloned(&self) -> wgpu::Buffer {
        self.page_table.clone()
    }

    pub fn virtual_page_table_buffer_cloned(&self) -> wgpu::Buffer {
        self.virtual_page_table.clone()
    }

    pub fn level_table_buffer_cloned(&self) -> wgpu::Buffer {
        self.level_table.clone()
    }

    pub fn level_count(&self) -> u32 {
        self.level_count
    }

    pub fn virtual_page_count(&self) -> u32 {
        self.virtual_page_count
    }

    /// Read the shader-visible page table back to the CPU. Test observability only.
    ///
    /// Blocks on a GPU readback (`map_async` + `poll(Wait)`), so it must never run
    /// on a frame path — it exists so tests can assert what the shader would see
    /// (the authoritative residency source), not the CPU mirrors that can silently
    /// diverge from it. `#[doc(hidden)]` keeps it off the public API surface while
    /// still letting downstream crates' tests reach it across the crate boundary.
    #[doc(hidden)]
    pub fn read_page_table_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Vec<GpuPageTableEntry> {
        let size = std::mem::size_of::<GpuPageTableEntry>() as u64 * u64::from(self.max_pages);
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile-page-table-readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tile-page-table-readback-encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.page_table, 0, &staging, 0, size);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("map callback")
            .expect("page-table readback mapping");
        let mapped = slice.get_mapped_range();
        let entries = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        entries
    }

    #[doc(hidden)]
    pub fn read_virtual_page_table_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Vec<GpuVirtualPageEntry> {
        if self.virtual_page_count == 0 {
            return Vec::new();
        }
        let size =
            std::mem::size_of::<GpuVirtualPageEntry>() as u64 * u64::from(self.virtual_page_count);
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("virtual-page-table-readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("virtual-page-table-readback"),
        });
        encoder.copy_buffer_to_buffer(&self.virtual_page_table, 0, &staging, 0, size);
        queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("map callback")
            .expect("virtual page-table mapping");
        let mapped = slice.get_mapped_range();
        let entries = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        entries
    }

    /// Read one physical atlas page for seam and publication tests only.
    #[doc(hidden)]
    pub fn read_page_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        slot: u32,
    ) -> Vec<f32> {
        assert!(slot < self.max_pages);
        let unpadded = self.page_extent * 4;
        let padded = unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain-tile-page-readback"),
            size: u64::from(padded) * u64::from(self.page_extent),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("terrain-tile-page-readback"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: slot,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(self.page_extent),
                },
            },
            wgpu::Extent3d {
                width: self.page_extent,
                height: self.page_extent,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("map callback")
            .expect("atlas page readback map");
        let mapped = slice.get_mapped_range();
        let mut result = Vec::with_capacity((self.page_extent * self.page_extent) as usize);
        for row in 0..self.page_extent as usize {
            let start = row * padded as usize;
            result.extend_from_slice(bytemuck::cast_slice(
                &mapped[start..start + unpadded as usize],
            ));
        }
        drop(mapped);
        staging.unmap();
        result
    }

    pub fn page_extent(&self) -> u32 {
        self.page_extent
    }

    pub fn tile_size(&self) -> u32 {
        self.tile_size
    }

    pub fn halo(&self) -> u32 {
        self.halo
    }

    pub fn max_pages(&self) -> u32 {
        self.max_pages
    }

    pub fn residency(&self) -> &TileResidencyCache {
        &self.residency
    }

    fn write_page_entry(&self, queue: &wgpu::Queue, slot: u32, entry: GpuPageTableEntry) {
        let offset = u64::from(slot) * std::mem::size_of::<GpuPageTableEntry>() as u64;
        queue.write_buffer(&self.page_table, offset, bytemuck::bytes_of(&entry));
    }

    fn write_virtual_entry(&self, queue: &wgpu::Queue, index: u32, entry: GpuVirtualPageEntry) {
        let offset = u64::from(index) * std::mem::size_of::<GpuVirtualPageEntry>() as u64;
        queue.write_buffer(&self.virtual_page_table, offset, bytemuck::bytes_of(&entry));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use terra_core::{FieldId, LayerId, TileId};

    #[test]
    fn page_entry_carries_generation_and_virtual_identity() {
        let key = TerrainTileKey {
            layer: Some(LayerId::new()),
            field: FieldId::Height,
            level: 5,
            tile: TileId { tx: 1, tz: 1 },
        };
        let metrics = terra_core::heightfield::HeightfieldMetrics {
            width: 256,
            height: 256,
            world_size_x: 1000.0,
            world_size_z: 1000.0,
            tile_size: 128,
            halo: 2,
        };
        let tile = HeightTile::new(key.tile, &metrics);
        let entry = GpuPageTableEntry::resident(
            &key,
            TilePageHandle {
                slot: 4,
                generation: 11,
            },
            tile.interior_width,
            tile.interior_height,
            tile.halo,
            TerrainContentStamp {
                document_revision: 3,
                plan_revision: 4,
                output_revision: 0x1_0000_0002,
                content_revision: 5,
            },
            9,
        );
        assert_eq!(entry.generation, 11);
        assert_eq!(entry.level, 5);
        assert_eq!((entry.tile_x, entry.tile_z), (1, 1));
        assert_eq!((entry.output_revision_hi, entry.output_revision_lo), (1, 2));
        assert_eq!(entry.document_revision_lo, 3);
        assert_eq!(entry.published_frame_lo, 9);
    }

    #[test]
    fn clear_invalidates_page_table_and_atlas_remains_uploadable() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let mut atlas = GpuTileAtlas::new(&gpu.device, 8, 1, 2).unwrap();
        let metrics = terra_core::heightfield::HeightfieldMetrics {
            width: 8,
            height: 8,
            world_size_x: 8.0,
            world_size_z: 8.0,
            tile_size: 8,
            halo: 1,
        };
        let old_key = TerrainTileKey {
            layer: Some(LayerId::new()),
            field: FieldId::Height,
            level: 0,
            tile: TileId { tx: 0, tz: 0 },
        };
        let tile = HeightTile::new(old_key.tile, &metrics);
        let old = atlas
            .upload_height_tile(&gpu.queue, old_key, &tile, 1, 1)
            .unwrap()
            .handle;
        assert_eq!(
            atlas.read_page_table_blocking(&gpu.device, &gpu.queue)[old.slot as usize].valid,
            1
        );

        atlas.clear(&gpu.queue);

        assert_eq!(atlas.residency().stats().resident_tiles, 0);
        assert_eq!(atlas.residency().stats().used_bytes, 0);
        assert_eq!(atlas.residency().resolve_handle(old), None);
        assert!(atlas
            .read_page_table_blocking(&gpu.device, &gpu.queue)
            .iter()
            .all(|entry| entry.valid == 0));

        let new_key = TerrainTileKey {
            layer: Some(LayerId::new()),
            field: FieldId::Height,
            level: 0,
            tile: TileId { tx: 0, tz: 0 },
        };
        let new = atlas
            .upload_height_tile(&gpu.queue, new_key.clone(), &tile, 2, 2)
            .unwrap()
            .handle;
        assert_eq!(new.slot, old.slot);
        assert_ne!(new.generation, old.generation);
        assert_eq!(atlas.residency().resolve_handle(old), None);
        assert_eq!(atlas.residency().resolve_handle(new), Some(&new_key));
        let entries = atlas.read_page_table_blocking(&gpu.device, &gpu.queue);
        assert_eq!(entries[new.slot as usize].valid, 1);
        assert_eq!(entries[new.slot as usize].generation, new.generation);
    }

    #[test]
    fn eviction_keeps_cache_and_shader_visible_page_counts_in_lockstep() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let mut atlas = GpuTileAtlas::new(&gpu.device, 8, 1, 2).unwrap();
        let metrics = terra_core::heightfield::HeightfieldMetrics {
            width: 24,
            height: 8,
            world_size_x: 24.0,
            world_size_z: 8.0,
            tile_size: 8,
            halo: 1,
        };
        let layer = LayerId::new();
        let make_key = |tx| TerrainTileKey {
            layer: Some(layer),
            field: FieldId::Height,
            level: 0,
            tile: TileId { tx, tz: 0 },
        };
        let first = make_key(0);
        let second = make_key(1);
        let third = make_key(2);
        let first_handle = atlas
            .upload_height_tile(
                &gpu.queue,
                first.clone(),
                &HeightTile::new(first.tile, &metrics),
                7,
                7,
            )
            .unwrap()
            .handle;
        atlas
            .upload_height_tile(
                &gpu.queue,
                second.clone(),
                &HeightTile::new(second.tile, &metrics),
                7,
                7,
            )
            .unwrap();

        let before = atlas.read_page_table_blocking(&gpu.device, &gpu.queue);
        assert_eq!(before.iter().filter(|entry| entry.valid != 0).count(), 2);
        assert_eq!(atlas.residency().stats().resident_tiles, 2);

        let replacement = atlas
            .upload_height_tile(
                &gpu.queue,
                third.clone(),
                &HeightTile::new(third.tile, &metrics),
                8,
                8,
            )
            .unwrap();
        assert_eq!(replacement.evicted, vec![first]);
        assert_eq!(replacement.handle.slot, first_handle.slot);
        assert_ne!(replacement.handle.generation, first_handle.generation);
        assert_eq!(atlas.residency().resolve_handle(first_handle), None);
        assert_eq!(
            atlas.residency().resolve_handle(replacement.handle),
            Some(&third)
        );

        let after = atlas.read_page_table_blocking(&gpu.device, &gpu.queue);
        assert_eq!(after.iter().filter(|entry| entry.valid != 0).count(), 2);
        assert_eq!(atlas.residency().stats().resident_tiles, 2);
        let row = after[replacement.handle.slot as usize];
        assert_eq!(row.valid, 1);
        assert_eq!(row.generation, replacement.handle.generation);
        assert_eq!((row.output_revision_hi, row.output_revision_lo), (0, 8));

        atlas.clear(&gpu.queue);
        assert_eq!(atlas.residency().stats().resident_tiles, 0);
        assert!(atlas
            .read_page_table_blocking(&gpu.device, &gpu.queue)
            .iter()
            .all(|entry| entry.valid == 0));
    }

    #[test]
    fn configured_hierarchy_maps_multiple_levels_and_unpublishes_child() {
        let Some(gpu) = terra_test_gpu::headless() else {
            return;
        };
        let hierarchy = TerrainPyramid::new(terra_core::PyramidConfig {
            target_resolution: 64,
            world_size_x: 1024.0,
            world_size_z: 1024.0,
            tile_size: 16,
            halo: 1,
        });
        let mut atlas = GpuTileAtlas::new(&gpu.device, 16, 1, 4).unwrap();
        atlas.configure_hierarchy(&gpu.device, &gpu.queue, &hierarchy);
        assert_eq!(atlas.level_count(), hierarchy.levels.len() as u32);
        assert_eq!(atlas.virtual_page_count(), hierarchy.metadata_len());

        let content = TerrainContentStamp {
            document_revision: 3,
            plan_revision: 5,
            output_revision: 7,
            content_revision: 11,
        };
        let root_key = TerrainTileKey {
            layer: None,
            field: FieldId::Height,
            level: 0,
            tile: TileId { tx: 0, tz: 0 },
        };
        let child_key = TerrainTileKey {
            layer: None,
            field: FieldId::Height,
            level: hierarchy.max_level(),
            tile: TileId { tx: 1, tz: 2 },
        };
        let root_metrics = hierarchy.level_metrics(0).unwrap();
        let child_metrics = hierarchy.level_metrics(hierarchy.max_level()).unwrap();
        let root = terra_core::Heightfield::filled(root_metrics, 10.0);
        let child = terra_core::Heightfield::filled(child_metrics, 20.0);
        atlas
            .upload_height_tile_current(
                &gpu.queue,
                root_key.clone(),
                root.tile(root_key.tile).unwrap(),
                content,
            )
            .unwrap();
        atlas
            .upload_height_tile_current(
                &gpu.queue,
                child_key.clone(),
                child.tile(child_key.tile).unwrap(),
                content,
            )
            .unwrap();

        let mappings = atlas.read_virtual_page_table_blocking(&gpu.device, &gpu.queue);
        let root_index = hierarchy.tile_metadata_index(0, root_key.tile).unwrap() as usize;
        let child_index = hierarchy
            .tile_metadata_index(child_key.level, child_key.tile)
            .unwrap() as usize;
        assert_eq!(mappings[root_index].valid, 1);
        assert_eq!(mappings[child_index].valid, 1);
        assert_ne!(
            mappings[root_index].physical_slot,
            mappings[child_index].physical_slot
        );

        assert!(atlas.unpublish(&gpu.queue, &child_key));
        let mappings = atlas.read_virtual_page_table_blocking(&gpu.device, &gpu.queue);
        assert_eq!(mappings[root_index].valid, 1);
        assert_eq!(mappings[child_index].valid, 0);
        assert!(atlas.is_current(&root_key, content));
        assert!(!atlas.is_current(&child_key, content));
    }
}
