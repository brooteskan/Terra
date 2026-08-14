use crate::heightfield::{HeightfieldMetrics, TileId};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Normalized, inclusive-exclusive terrain footprint.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NormalizedRect {
    pub u0: f32,
    pub v0: f32,
    pub u1: f32,
    pub v1: f32,
}

impl NormalizedRect {
    pub fn new(u0: f32, v0: f32, u1: f32, v1: f32) -> Option<Self> {
        let rect = Self {
            u0: u0.min(u1).clamp(0.0, 1.0),
            v0: v0.min(v1).clamp(0.0, 1.0),
            u1: u0.max(u1).clamp(0.0, 1.0),
            v1: v0.max(v1).clamp(0.0, 1.0),
        };
        (!rect.is_empty()).then_some(rect)
    }

    pub fn is_empty(self) -> bool {
        self.u1 <= self.u0 || self.v1 <= self.v0
    }
}

/// A viewport footprint that can be enumerated as terrain tiles.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegionSet {
    rects: Vec<NormalizedRect>,
}

impl RegionSet {
    pub fn from_rect(rect: NormalizedRect) -> Self {
        Self { rects: vec![rect] }
    }

    pub fn tiles(&self, metrics: &HeightfieldMetrics) -> Vec<TileId> {
        let mut ids = HashSet::new();
        if metrics.width == 0 || metrics.height == 0 || metrics.tile_size == 0 {
            return Vec::new();
        }
        for rect in &self.rects {
            let x0 = (rect.u0 * metrics.width as f32).floor() as u32;
            let z0 = (rect.v0 * metrics.height as f32).floor() as u32;
            let x1 = ((rect.u1 * metrics.width as f32).ceil() as u32)
                .saturating_sub(1)
                .min(metrics.width - 1);
            let z1 = ((rect.v1 * metrics.height as f32).ceil() as u32)
                .saturating_sub(1)
                .min(metrics.height - 1);
            let tx0 = x0.min(metrics.width - 1) / metrics.tile_size;
            let tz0 = z0.min(metrics.height - 1) / metrics.tile_size;
            let tx1 = x1 / metrics.tile_size;
            let tz1 = z1 / metrics.tile_size;
            for tz in tz0..=tz1 {
                for tx in tx0..=tx1 {
                    ids.insert(TileId { tx, tz });
                }
            }
        }
        let mut ids: Vec<_> = ids.into_iter().collect();
        ids.sort_by_key(|id| (id.tz, id.tx));
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_tiles_are_deterministic_and_local() {
        let metrics = HeightfieldMetrics::new(1024, 1024, 4096.0, 4096.0);
        let region = RegionSet::from_rect(NormalizedRect::new(0.1, 0.1, 0.2, 0.2).unwrap());
        let tiles = region.tiles(&metrics);
        assert!(!tiles.is_empty());
        assert!(tiles.len() < metrics.tile_count() as usize);
        assert!(tiles
            .windows(2)
            .all(|pair| (pair[0].tz, pair[0].tx) <= (pair[1].tz, pair[1].tx)));
    }
}
