use bytemuck::{Pod, Zeroable};
use std::collections::{hash_map::DefaultHasher, HashMap};
use std::hash::{Hash, Hasher};
use terra_core::{HeightTile, TerrainTileKey, TileCacheError, TilePageHandle, TileResidencyCache};
use thiserror::Error;
use wgpu::util::DeviceExt;

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
        tile: &HeightTile,
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
            width: tile.interior_width,
            height: tile.interior_height,
            halo: tile.halo,
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
    handles: HashMap<TerrainTileKey, TilePageHandle>,
    page_extent: u32,
    tile_size: u32,
    halo: u32,
    max_pages: u32,
    page_bytes: u64,
    memory_budget: crate::memory::MemoryBudget,
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
        let mut memory_budget = crate::memory::MemoryBudget::new(
            ((page_bytes * u64::from(max_pages)) / (1024 * 1024)).max(1) + 64,
        );
        let _ = memory_budget.try_alloc(page_bytes * u64::from(max_pages));
        Ok(Self {
            texture,
            view,
            page_table,
            residency: TileResidencyCache::new(page_bytes * u64::from(max_pages)),
            handles: HashMap::new(),
            page_extent,
            tile_size,
            halo,
            max_pages,
            page_bytes,
            memory_budget,
        })
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
        let insert = self
            .residency
            .insert(key.clone(), self.page_bytes, revision, input_revision_hash)
            .map_err(GpuTileCacheError::Residency)?;

        for evicted in &insert.evicted {
            if let Some(handle) = self.handles.remove(evicted) {
                self.write_page_entry(
                    queue,
                    handle.slot,
                    GpuPageTableEntry::invalid(handle.generation),
                );
            }
        }
        self.handles.insert(key.clone(), insert.handle);
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
            GpuPageTableEntry::resident(&key, insert.handle, tile, revision),
        );
        Ok(GpuTileUpload {
            handle: insert.handle,
            evicted: insert.evicted,
        })
    }

    /// Invalidate all document-owned residency without reallocating atlas resources.
    ///
    /// Texture contents may remain in physical pages because the page table is made
    /// entirely invalid before streaming can be enabled for the next document.
    pub fn clear(&mut self, queue: &wgpu::Queue) {
        self.residency.clear();
        self.handles.clear();
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

    pub fn memory_budget(&self) -> &crate::memory::MemoryBudget {
        &self.memory_budget
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

    fn read_page_table(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        atlas: &GpuTileAtlas,
    ) -> Vec<GpuPageTableEntry> {
        let size = std::mem::size_of::<GpuPageTableEntry>() as u64 * u64::from(atlas.max_pages());
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile-page-table-test-readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tile-page-table-test-readback-encoder"),
        });
        encoder.copy_buffer_to_buffer(atlas.page_table_buffer(), 0, &staging, 0, size);
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
            &tile,
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
            read_page_table(&gpu.device, &gpu.queue, &atlas)[old.slot as usize].valid,
            1
        );

        atlas.clear(&gpu.queue);

        assert_eq!(atlas.residency().stats().resident_tiles, 0);
        assert_eq!(atlas.residency().resolve_handle(old), None);
        assert!(read_page_table(&gpu.device, &gpu.queue, &atlas)
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
        let entries = read_page_table(&gpu.device, &gpu.queue, &atlas);
        assert_eq!(entries[new.slot as usize].valid, 1);
        assert_eq!(entries[new.slot as usize].generation, new.generation);
    }
}
