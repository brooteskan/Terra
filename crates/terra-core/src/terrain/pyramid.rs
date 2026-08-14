use super::{TerrainTileKey, TilePageHandle};
use crate::fields::FieldId;
use crate::heightfield::{HeightfieldMetrics, TileId, DEFAULT_HALO, DEFAULT_TILE_SIZE};
use crate::layer::LayerId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TileRecord {
    /// Backend payload backing this residency record.
    pub handle: TilePageHandle,
    pub revision: u64,
    pub input_revision_hash: u64,
    /// Min height in metres for resident payload (0 when unknown).
    pub height_min: f32,
    /// Max height in metres for resident payload (0 when unknown).
    pub height_max: f32,
    /// World-space geometric error at this pyramid level (metres).
    pub geometric_error: f32,
    pub last_used_frame: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PyramidConfig {
    pub target_resolution: u32,
    pub world_size_x: f32,
    pub world_size_z: f32,
    pub tile_size: u32,
    pub halo: u32,
}

impl PyramidConfig {
    pub fn new(target_resolution: u32, world_size_x: f32, world_size_z: f32) -> Self {
        Self {
            target_resolution: target_resolution.max(2),
            world_size_x,
            world_size_z,
            tile_size: DEFAULT_TILE_SIZE,
            halo: DEFAULT_HALO,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TerrainLevel {
    pub index: u8,
    pub resolution: u32,
    pub metrics: HeightfieldMetrics,
}

/// Sparse, payload-backed residency pyramid for viewport terrain tiles.
#[derive(Debug, Clone)]
pub struct TerrainPyramid {
    pub config: PyramidConfig,
    pub levels: Vec<TerrainLevel>,
    records: HashMap<TerrainTileKey, TileRecord>,
}

impl TerrainPyramid {
    pub fn new(config: PyramidConfig) -> Self {
        let mut resolutions = Vec::new();
        let mut resolution = 2u32;
        while resolution < config.target_resolution {
            resolutions.push(resolution);
            resolution = resolution.saturating_mul(2);
            if resolution == u32::MAX {
                break;
            }
        }
        if resolutions.last().copied() != Some(config.target_resolution) {
            resolutions.push(config.target_resolution);
        }
        let levels = resolutions
            .into_iter()
            .enumerate()
            .map(|(index, resolution)| TerrainLevel {
                index: index as u8,
                resolution,
                metrics: HeightfieldMetrics {
                    width: resolution,
                    height: resolution,
                    world_size_x: config.world_size_x,
                    world_size_z: config.world_size_z,
                    tile_size: config.tile_size.min(resolution),
                    halo: config.halo,
                },
            })
            .collect();
        Self {
            config,
            levels,
            records: HashMap::new(),
        }
    }

    pub fn max_level(&self) -> u8 {
        self.levels.len().saturating_sub(1) as u8
    }

    pub fn level_metrics(&self) -> Vec<HeightfieldMetrics> {
        self.levels.iter().map(|level| level.metrics).collect()
    }

    pub fn record(&self, key: &TerrainTileKey) -> Option<&TileRecord> {
        self.records.get(key)
    }

    /// Forget all residency metadata when a new final-output revision begins.
    pub fn clear_residency(&mut self) {
        self.records.clear();
    }

    pub fn remove_resident(&mut self, key: &TerrainTileKey) -> bool {
        self.records.remove(key).is_some()
    }

    /// Publish metadata only after a backend has stored the corresponding payload.
    pub fn publish_resident(
        &mut self,
        key: TerrainTileKey,
        handle: TilePageHandle,
        revision: u64,
        input_revision_hash: u64,
        frame: u64,
    ) {
        self.records.insert(
            key,
            TileRecord {
                handle,
                revision,
                input_revision_hash,
                height_min: 0.0,
                height_max: 0.0,
                geometric_error: 0.0,
                last_used_frame: frame,
            },
        );
    }

    /// Find the finest resident tile at or below the desired level covering the same world point.
    pub fn best_resident_ancestor(
        &self,
        layer: Option<LayerId>,
        field: &FieldId,
        desired_level: u8,
        desired_tile: TileId,
    ) -> Option<TerrainTileKey> {
        let desired = self.levels.get(desired_level as usize)?;
        let center_u = ((desired_tile.tx * desired.metrics.tile_size) as f32
            + desired.metrics.tile_size as f32 * 0.5)
            / desired.metrics.width as f32;
        let center_v = ((desired_tile.tz * desired.metrics.tile_size) as f32
            + desired.metrics.tile_size as f32 * 0.5)
            / desired.metrics.height as f32;
        for level_index in (0..=desired_level).rev() {
            let level = &self.levels[level_index as usize];
            let sx = (center_u.clamp(0.0, 0.999_999) * level.metrics.width as f32) as u32;
            let sz = (center_v.clamp(0.0, 0.999_999) * level.metrics.height as f32) as u32;
            let key = TerrainTileKey {
                layer,
                field: field.clone(),
                level: level_index,
                tile: TileId {
                    tx: sx / level.metrics.tile_size,
                    tz: sz / level.metrics.tile_size,
                },
            };
            if self.records.contains_key(&key) {
                return Some(key);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pyramid_builds_complete_upsample_chain() {
        let pyramid = TerrainPyramid::new(PyramidConfig::new(1000, 4096.0, 4096.0));
        assert_eq!(pyramid.levels.first().unwrap().resolution, 2);
        assert_eq!(pyramid.levels.last().unwrap().resolution, 1000);
        assert!(pyramid.levels.windows(2).all(|pair| {
            pair[1].resolution == pair[0].resolution * 2 || pair[1].resolution == 1000
        }));
    }

    #[test]
    fn renderer_can_fall_back_to_resident_coarse_tile() {
        let mut pyramid = TerrainPyramid::new(PyramidConfig::new(1024, 4096.0, 4096.0));
        let coarse = TerrainTileKey {
            layer: None,
            field: FieldId::Height,
            level: 0,
            tile: TileId { tx: 0, tz: 0 },
        };
        let handle = TilePageHandle {
            slot: 4,
            generation: 2,
        };
        pyramid.publish_resident(coarse.clone(), handle, 1, 1, 0);
        assert_eq!(pyramid.record(&coarse).unwrap().handle, handle);
        let found = pyramid.best_resident_ancestor(
            None,
            &FieldId::Height,
            pyramid.max_level(),
            TileId { tx: 3, tz: 2 },
        );
        assert_eq!(found, Some(coarse));
    }

    #[test]
    fn evicted_payload_is_not_reported_resident() {
        let mut pyramid = TerrainPyramid::new(PyramidConfig::new(256, 1024.0, 1024.0));
        let key = TerrainTileKey {
            layer: Some(LayerId::new()),
            field: FieldId::Height,
            level: 0,
            tile: TileId { tx: 0, tz: 0 },
        };
        pyramid.publish_resident(
            key.clone(),
            TilePageHandle {
                slot: 1,
                generation: 1,
            },
            1,
            1,
            0,
        );
        assert!(pyramid.remove_resident(&key));
        assert!(pyramid.record(&key).is_none());
    }
}
