//! Sparse world-space paint pages for biome placement.

use crate::biome_paint::BiomeLayerId;
use crate::ids::LayerId;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use uuid::Uuid;

/// Page coordinate in world paint space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PaintPageCoord {
    pub x: i32,
    pub y: i32,
}

/// Identity for a sparse paint channel (placement layer + biome target).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct SparsePaintChannelKey {
    pub placement_id: BiomeLayerId,
    /// Biome group LayerId or (after migration) definition-linked group id.
    pub biome_id: LayerId,
}

/// One dense page of paint samples in world metres.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaintPage {
    pub coord: PaintPageCoord,
    pub width: u32,
    pub height: u32,
    /// Origin of page corner in world metres (X, Z).
    pub origin_x: f32,
    pub origin_z: f32,
    /// Metres per sample.
    pub metres_per_sample: f32,
    pub samples: Vec<f32>,
    /// Content hash for incremental export / cache.
    #[serde(default)]
    pub content_hash: u64,
}

impl PaintPage {
    pub fn empty(
        coord: PaintPageCoord,
        page_samples: u32,
        metres_per_sample: f32,
        origin_x: f32,
        origin_z: f32,
    ) -> Self {
        let n = (page_samples * page_samples) as usize;
        Self {
            coord,
            width: page_samples,
            height: page_samples,
            origin_x,
            origin_z,
            metres_per_sample,
            samples: vec![0.0; n],
            content_hash: 0,
        }
    }

    pub fn rehash(&mut self) {
        let mut h: u64 = 0xcbf29ce484222325;
        for s in &self.samples {
            let bits = s.to_bits() as u64;
            h ^= bits;
            h = h.wrapping_mul(0x100000001b3);
        }
        self.content_hash = h;
    }

    pub fn stamp_circle_world(
        &mut self,
        wx: f32,
        wz: f32,
        radius_m: f32,
        strength: f32,
        erase: bool,
    ) -> bool {
        let mut dirty = false;
        let r = radius_m.max(1e-3);
        for j in 0..self.height {
            for i in 0..self.width {
                let sx = self.origin_x + (i as f32 + 0.5) * self.metres_per_sample;
                let sz = self.origin_z + (j as f32 + 0.5) * self.metres_per_sample;
                let d = ((sx - wx).hypot(sz - wz)) / r;
                if d <= 1.0 {
                    let amount = (1.0 - d * d) * strength;
                    let sample = &mut self.samples[(j * self.width + i) as usize];
                    let next = if erase {
                        *sample - amount
                    } else {
                        *sample + amount
                    }
                    .clamp(0.0, 1.0);
                    if (next - *sample).abs() > 1e-6 {
                        *sample = next;
                        dirty = true;
                    }
                }
            }
        }
        if dirty {
            self.rehash();
        }
        dirty
    }

    pub fn sample_world(&self, wx: f32, wz: f32) -> f32 {
        if self.width == 0 || self.height == 0 || self.metres_per_sample <= 0.0 {
            return 0.0;
        }
        let u = (wx - self.origin_x) / self.metres_per_sample;
        let v = (wz - self.origin_z) / self.metres_per_sample;
        if u < 0.0 || v < 0.0 || u >= self.width as f32 || v >= self.height as f32 {
            return 0.0;
        }
        let i = u.floor() as u32;
        let j = v.floor() as u32;
        self.samples[(j.min(self.height - 1) * self.width + i.min(self.width - 1)) as usize]
    }
}

/// World-space sparse paint store.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SparsePaintStore {
    /// Samples per page edge (default 256).
    #[serde(default = "default_page_samples")]
    pub page_samples: u32,
    /// Metres per sample inside pages.
    #[serde(default = "default_mps")]
    pub metres_per_sample: f32,
    /// Serializable channel list (JSON-friendly; rebuilt into a map at runtime).
    #[serde(default)]
    pub channel_list: Vec<SparsePaintChannel>,
    /// Pages dirtied since last eval consume.
    #[serde(skip)]
    pub dirty_pages: HashSet<(SparsePaintChannelKey, PaintPageCoord)>,
}

fn default_page_samples() -> u32 {
    256
}

fn default_mps() -> f32 {
    4.0
}

/// One serializable paint channel with its pages.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SparsePaintChannel {
    pub key: SparsePaintChannelKey,
    pub pages: Vec<PaintPage>,
}

impl SparsePaintStore {
    pub fn new(metres_per_sample: f32, page_samples: u32) -> Self {
        Self {
            page_samples: page_samples.max(16),
            metres_per_sample: metres_per_sample.max(0.1),
            channel_list: Vec::new(),
            dirty_pages: HashSet::new(),
        }
    }

    fn channel_mut(&mut self, key: SparsePaintChannelKey) -> &mut SparsePaintChannel {
        if let Some(idx) = self.channel_list.iter().position(|c| c.key == key) {
            return &mut self.channel_list[idx];
        }
        self.channel_list.push(SparsePaintChannel {
            key,
            pages: Vec::new(),
        });
        self.channel_list.last_mut().unwrap()
    }

    fn channel(&self, key: SparsePaintChannelKey) -> Option<&SparsePaintChannel> {
        self.channel_list.iter().find(|c| c.key == key)
    }

    pub fn page_extent_m(&self) -> f32 {
        self.page_samples as f32 * self.metres_per_sample
    }

    pub fn coord_for_world(&self, wx: f32, wz: f32) -> PaintPageCoord {
        let e = self.page_extent_m().max(1e-3);
        PaintPageCoord {
            x: (wx / e).floor() as i32,
            y: (wz / e).floor() as i32,
        }
    }

    fn ensure_page(&mut self, key: SparsePaintChannelKey, coord: PaintPageCoord) -> &mut PaintPage {
        let e = self.page_extent_m();
        let mps = self.metres_per_sample;
        let ps = self.page_samples;
        let ch = self.channel_mut(key);
        if let Some(idx) = ch.pages.iter().position(|p| p.coord == coord) {
            return &mut ch.pages[idx];
        }
        ch.pages.push(PaintPage::empty(
            coord,
            ps,
            mps,
            coord.x as f32 * e,
            coord.y as f32 * e,
        ));
        ch.pages.last_mut().unwrap()
    }

    /// Stamp a soft circle in world metres. Returns dirty page coords.
    pub fn stamp_circle(
        &mut self,
        key: SparsePaintChannelKey,
        wx: f32,
        wz: f32,
        radius_m: f32,
        strength: f32,
        erase: bool,
    ) -> Vec<PaintPageCoord> {
        let e = self.page_extent_m().max(1e-3);
        let r = radius_m.max(0.0);
        let min_x = ((wx - r) / e).floor() as i32;
        let max_x = ((wx + r) / e).floor() as i32;
        let min_y = ((wz - r) / e).floor() as i32;
        let max_y = ((wz + r) / e).floor() as i32;
        let mut dirty = Vec::new();
        for py in min_y..=max_y {
            for px in min_x..=max_x {
                let coord = PaintPageCoord { x: px, y: py };
                let changed = {
                    let page = self.ensure_page(key, coord);
                    page.stamp_circle_world(wx, wz, radius_m, strength, erase)
                };
                if changed {
                    self.dirty_pages.insert((key, coord));
                    dirty.push(coord);
                }
            }
        }
        dirty
    }

    pub fn sample(&self, key: SparsePaintChannelKey, wx: f32, wz: f32) -> f32 {
        let coord = self.coord_for_world(wx, wz);
        self.channel(key)
            .and_then(|c| c.pages.iter().find(|p| p.coord == coord))
            .map(|p| p.sample_world(wx, wz))
            .unwrap_or(0.0)
    }

    pub fn take_dirty(&mut self) -> HashSet<(SparsePaintChannelKey, PaintPageCoord)> {
        std::mem::take(&mut self.dirty_pages)
    }

    /// Snapshot pages intersecting a world AABB for undo.
    pub fn snapshot_region(
        &self,
        key: SparsePaintChannelKey,
        wx: f32,
        wz: f32,
        radius_m: f32,
    ) -> Vec<PaintPage> {
        let e = self.page_extent_m().max(1e-3);
        let r = radius_m.max(0.0);
        let min_x = ((wx - r) / e).floor() as i32;
        let max_x = ((wx + r) / e).floor() as i32;
        let min_y = ((wz - r) / e).floor() as i32;
        let max_y = ((wz + r) / e).floor() as i32;
        let mut out = Vec::new();
        if let Some(ch) = self.channel(key) {
            for py in min_y..=max_y {
                for px in min_x..=max_x {
                    let coord = PaintPageCoord { x: px, y: py };
                    if let Some(p) = ch.pages.iter().find(|p| p.coord == coord) {
                        out.push(p.clone());
                    }
                }
            }
        }
        out
    }

    pub fn restore_pages(&mut self, key: SparsePaintChannelKey, pages: &[PaintPage]) {
        for p in pages {
            self.dirty_pages.insert((key, p.coord));
        }
        let ch = self.channel_mut(key);
        for p in pages {
            if let Some(existing) = ch.pages.iter_mut().find(|x| x.coord == p.coord) {
                *existing = p.clone();
            } else {
                ch.pages.push(p.clone());
            }
        }
    }

    pub fn clear_channel(&mut self, key: SparsePaintChannelKey) {
        if let Some(ch) = self.channel_list.iter_mut().find(|c| c.key == key) {
            ch.pages.clear();
        }
    }

    pub fn channel_pages(&self, key: SparsePaintChannelKey) -> Vec<PaintPage> {
        self.channel(key)
            .map(|c| c.pages.clone())
            .unwrap_or_default()
    }

    pub fn has_channel(&self, key: SparsePaintChannelKey) -> bool {
        self.channel(key).is_some_and(|c| !c.pages.is_empty())
    }

    /// Bake channel into a dense UV buffer for legacy mask bridge.
    pub fn bake_uv(
        &self,
        key: SparsePaintChannelKey,
        width: u32,
        height: u32,
        world_size_x: f32,
        world_size_z: f32,
    ) -> Vec<f32> {
        let mut samples = vec![0.0; (width * height) as usize];
        for j in 0..height {
            for i in 0..width {
                let u = (i as f32 + 0.5) / width as f32;
                let v = (j as f32 + 0.5) / height as f32;
                let wx = u * world_size_x;
                let wz = v * world_size_z;
                samples[(j * width + i) as usize] = self.sample(key, wx, wz);
            }
        }
        samples
    }

    pub fn channel_content_hash(&self, key: SparsePaintChannelKey) -> u64 {
        let mut h: u64 = 0x84222325;
        if let Some(ch) = self.channel(key) {
            let mut pages = ch.pages.clone();
            pages.sort_by(|a, b| a.coord.x.cmp(&b.coord.x).then(a.coord.y.cmp(&b.coord.y)));
            for p in pages {
                h ^= p.content_hash;
                h = h.wrapping_mul(0x100000001b3);
                h ^= (p.coord.x as u64).wrapping_shl(32) ^ (p.coord.y as u64);
            }
        }
        h
    }
}

/// Stable stroke id for undo coalescing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PaintStrokeId(pub Uuid);

impl PaintStrokeId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for PaintStrokeId {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::biome_paint::BiomeLayerId;
    use crate::ids::LayerId;

    fn key() -> SparsePaintChannelKey {
        SparsePaintChannelKey {
            placement_id: BiomeLayerId::new(),
            biome_id: LayerId::new(),
        }
    }

    #[test]
    fn stamp_grows_with_painted_area() {
        let mut store = SparsePaintStore::new(4.0, 64);
        let k = key();
        store.stamp_circle(k, 100.0, 100.0, 20.0, 1.0, false);
        assert_eq!(store.channel_pages(k).len(), 1);
        store.stamp_circle(k, 5000.0, 5000.0, 20.0, 1.0, false);
        assert_eq!(store.channel_pages(k).len(), 2);
        assert!(store.sample(k, 100.0, 100.0) > 0.5);
        assert!(store.sample(k, 5000.0, 5000.0) > 0.5);
        assert_eq!(store.sample(k, 2000.0, 2000.0), 0.0);
    }

    #[test]
    fn snapshot_restore_undo() {
        let mut store = SparsePaintStore::new(4.0, 64);
        let k = key();
        let before = store.snapshot_region(k, 50.0, 50.0, 30.0);
        store.stamp_circle(k, 50.0, 50.0, 25.0, 1.0, false);
        assert!(store.sample(k, 50.0, 50.0) > 0.5);
        store.clear_channel(k);
        store.restore_pages(k, &before);
        assert_eq!(store.sample(k, 50.0, 50.0), 0.0);
    }

    #[test]
    fn content_hash_stable() {
        let mut store = SparsePaintStore::new(4.0, 32);
        let k = key();
        store.stamp_circle(k, 10.0, 10.0, 8.0, 0.8, false);
        let h1 = store.channel_content_hash(k);
        let h2 = store.channel_content_hash(k);
        assert_eq!(h1, h2);
        store.stamp_circle(k, 10.0, 10.0, 8.0, 0.5, false);
        assert_ne!(h1, store.channel_content_hash(k));
    }
}
