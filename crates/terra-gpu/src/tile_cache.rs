use bytemuck::{Pod, Zeroable};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use terra_core::{
    HeightTile, TerrainTileKey, TileCacheError, TileCacheInsert, TilePageHandle, TileResidencyCache,
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
    pub revision_lo: u32,
    pub revision_hi: u32,
}

impl GpuPageTableEntry {
    fn resident(
        key: &TerrainTileKey,
        handle: TilePageHandle,
        width: u32,
        height: u32,
        halo: u32,
        revision: u64,
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
            revision_lo: revision as u32,
            revision_hi: (revision >> 32) as u32,
        }
    }

    fn invalid(generation: u32) -> Self {
        Self {
            generation,
            ..Zeroable::zeroed()
        }
    }
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

    fn allocate_page(
        &mut self,
        queue: &wgpu::Queue,
        key: &TerrainTileKey,
        revision: u64,
        input_revision_hash: u64,
    ) -> Result<TileCacheInsert, GpuTileCacheError> {
        let insert = self
            .residency
            .insert(key.clone(), self.page_bytes, revision, input_revision_hash)
            .map_err(GpuTileCacheError::Residency)?;
        for evicted in &insert.evicted {
            self.write_page_entry(
                queue,
                evicted.handle.slot,
                GpuPageTableEntry::invalid(evicted.handle.generation),
            );
        }
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
        if tile.stride() > self.page_extent || tile.stride_z() > self.page_extent {
            return Err(GpuTileCacheError::TileTooLarge {
                width: tile.stride(),
                height: tile.stride_z(),
                page_extent: self.page_extent,
            });
        }
        let insert = self.allocate_page(queue, &key, revision, input_revision_hash)?;
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
                revision,
            ),
        );
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
        let insert = self.allocate_page(queue, &key, live_output_revision, identity.output.0)?;
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
                live_output_revision,
            ),
        );
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
    }

    pub fn lookup(&mut self, key: &TerrainTileKey) -> Option<TilePageHandle> {
        self.residency.get(key).map(|entry| entry.handle)
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
            0x1_0000_0002,
        );
        assert_eq!(entry.generation, 11);
        assert_eq!(entry.level, 5);
        assert_eq!((entry.tile_x, entry.tile_z), (1, 1));
        assert_eq!((entry.revision_hi, entry.revision_lo), (1, 2));
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
        assert_eq!((row.revision_hi, row.revision_lo), (0, 8));

        atlas.clear(&gpu.queue);
        assert_eq!(atlas.residency().stats().resident_tiles, 0);
        assert!(atlas
            .read_page_table_blocking(&gpu.device, &gpu.queue)
            .iter()
            .all(|entry| entry.valid == 0));
    }
}
