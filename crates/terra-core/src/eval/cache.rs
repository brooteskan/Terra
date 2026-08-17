use super::smart_cache::DiskSmartCache;
use crate::heightfield::{Heightfield, HeightfieldMetrics, TileId};
use crate::layer::LayerId;
use crate::mask::MaskField;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct CachedOutput {
    pub height: Heightfield,
    pub generation: u64,
    pub dirty: bool,
    pub aux: HashMap<String, MaskField>,
    /// Materials strata (not stored in the aux HashMap).
    pub strata: Option<Vec<crate::layer::Stratum>>,
}

/// Per-tile dirty seed for a layer (#100 phase 2). Distinguishes a clean layer
/// from one dirtied over the whole field from one dirtied over an enumerable set
/// of *seed* tiles (pre-reach-expansion). Seeds are in-memory metadata only; they
/// never persist to a spill and are dropped whenever a fresh output is stored.
///
/// `AllTiles` is the conservative default: a missing entry, a spilled entry, or a
/// plain [`LayerCache::mark_dirty`] all resolve to it, so any path that forgets to
/// record a seed degrades to a whole-field recompute — never to a stale result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeedState {
    /// The layer's cache entry is clean; its own contribution has not changed.
    Clean,
    /// The whole field is dirty (plain mark, spilled, or missing entry).
    AllTiles,
    /// Exactly these seed tiles are dirty. May be empty: an input-only change
    /// (a layer above the edit) has no own-contribution seed but is still dirty.
    Tiles(Vec<TileId>),
}

/// One layer's cache slot. A pinned (`baked`) output that spilled to disk is
/// dropped from memory and kept only as a lightweight `Spilled` descriptor until
/// something reloads it — this is where the "memory can be reclaimed" contract is
/// actually honoured (B1-D8). Non-baked outputs, and baked outputs whose spill
/// failed or whose cache has no disk, stay `Resident` with their full buffers.
#[derive(Debug)]
enum CacheEntry {
    Resident(CachedOutput),
    Spilled {
        metrics: HeightfieldMetrics,
        dirty: bool,
    },
}

impl CacheEntry {
    fn dirty(&self) -> bool {
        match self {
            CacheEntry::Resident(output) => output.dirty,
            CacheEntry::Spilled { dirty, .. } => *dirty,
        }
    }

    fn metrics(&self) -> HeightfieldMetrics {
        match self {
            CacheEntry::Resident(output) => output.height.metrics,
            CacheEntry::Spilled { metrics, .. } => *metrics,
        }
    }

    /// Clean and dimensionally usable for `metrics`.
    fn clean_match(&self, metrics: HeightfieldMetrics) -> bool {
        !self.dirty()
            && self.metrics().width == metrics.width
            && self.metrics().height == metrics.height
    }
}

#[derive(Debug)]
pub struct LayerCache {
    entries: HashMap<LayerId, CacheEntry>,
    pub generation: u64,
    /// Optional on-disk spill for baked (`cached`) layer checkpoints.
    disk: Option<DiskSmartCache>,
    /// Seed (pre-reach-expansion) dirty tiles for entries that were dirtied over a
    /// bounded region. A record exists only while its entry is `Resident` and
    /// dirty; its absence on a dirty entry means the whole field is dirty. Kept off
    /// [`CachedOutput`] so the B1-D8 spill format and every `CachedOutput` literal
    /// stay untouched (#103 bar).
    seed_dirty: HashMap<LayerId, HashSet<TileId>>,
}

impl Default for LayerCache {
    fn default() -> Self {
        Self::new()
    }
}

impl LayerCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            generation: 0,
            disk: Some(DiskSmartCache::owned_instance()),
            seed_dirty: HashMap::new(),
        }
    }

    pub fn with_disk(root: impl Into<PathBuf>) -> Self {
        Self {
            entries: HashMap::new(),
            generation: 0,
            disk: Some(DiskSmartCache::new(root)),
            seed_dirty: HashMap::new(),
        }
    }

    pub fn without_disk() -> Self {
        Self {
            entries: HashMap::new(),
            generation: 0,
            disk: None,
            seed_dirty: HashMap::new(),
        }
    }

    pub fn enable_disk(&mut self, root: impl Into<PathBuf>) {
        self.disk = Some(DiskSmartCache::new(root));
    }

    pub fn disable_disk(&mut self) {
        self.disk = None;
    }

    /// Detach the disk spill store (its per-instance root travels with it). Used
    /// by the worker restart to carry pinned bakes into the replacement evaluator.
    pub fn take_disk(&mut self) -> Option<DiskSmartCache> {
        self.disk.take()
    }

    /// Install a disk spill store, replacing (and dropping) any current one.
    pub fn set_disk(&mut self, disk: Option<DiskSmartCache>) {
        self.disk = disk;
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.seed_dirty.clear();
        self.generation = self.generation.wrapping_add(1);
    }

    pub fn insert(&mut self, id: LayerId, output: CachedOutput) {
        // A freshly stored output is clean over the whole field.
        self.seed_dirty.remove(&id);
        self.entries.insert(id, CacheEntry::Resident(output));
    }

    /// Insert and write a durable disk checkpoint. On a successful spill the heavy
    /// buffers are reclaimed from memory, leaving only a `Spilled` descriptor; if
    /// there is no disk or the spill fails, the entry stays `Resident`.
    pub fn insert_baked(&mut self, id: LayerId, output: CachedOutput) {
        // A freshly baked output is clean over the whole field.
        self.seed_dirty.remove(&id);
        if let Some(disk) = &self.disk {
            if disk.spill(id, &output).is_ok() {
                self.entries.insert(
                    id,
                    CacheEntry::Spilled {
                        metrics: output.height.metrics,
                        dirty: output.dirty,
                    },
                );
                return;
            }
        }
        self.entries.insert(id, CacheEntry::Resident(output));
    }

    /// In-memory checkpoint only. Returns `None` for a spilled entry (its buffers
    /// are on disk) — use [`Self::get_or_load`] to materialize it.
    pub fn get(&self, id: LayerId) -> Option<&CachedOutput> {
        match self.entries.get(&id) {
            Some(CacheEntry::Resident(output)) => Some(output),
            _ => None,
        }
    }

    /// Memory hit, else reload a spilled/disk bake matching `metrics`. A reloaded
    /// bake is promoted back to `Resident` for the reference this returns.
    pub fn get_or_load(
        &mut self,
        id: LayerId,
        metrics: HeightfieldMetrics,
    ) -> Option<&CachedOutput> {
        enum Plan {
            UseResident,
            LoadDisk { spilled_claim: bool },
            GiveUp,
        }
        let plan = match self.entries.get(&id) {
            Some(entry @ CacheEntry::Resident(_)) => {
                if entry.clean_match(metrics) {
                    Plan::UseResident
                } else if entry.dirty() {
                    // Dirty entries invalidated their disk file on mark; a load
                    // attempt mirrors the previous behavior and simply misses.
                    Plan::LoadDisk {
                        spilled_claim: false,
                    }
                } else {
                    // Clean but wrong size: recompute (no disk probe, as before).
                    Plan::GiveUp
                }
            }
            Some(entry @ CacheEntry::Spilled { .. }) => {
                if entry.clean_match(metrics) {
                    Plan::LoadDisk {
                        spilled_claim: true,
                    }
                } else if entry.dirty() {
                    Plan::LoadDisk {
                        spilled_claim: false,
                    }
                } else {
                    Plan::GiveUp
                }
            }
            None => Plan::LoadDisk {
                spilled_claim: false,
            },
        };

        match plan {
            Plan::UseResident => {}
            Plan::GiveUp => return None,
            Plan::LoadDisk { spilled_claim } => {
                let loaded = self
                    .disk
                    .as_ref()
                    .and_then(|disk| disk.load(id, metrics).ok().flatten());
                if let Some(loaded) = loaded {
                    self.entries.insert(id, CacheEntry::Resident(loaded));
                } else if spilled_claim {
                    // A clean `Spilled` entry whose backing file is gone or failed
                    // validation: drop it so the caller recomputes instead of
                    // trusting a checkpoint we can no longer produce.
                    self.entries.remove(&id);
                    self.seed_dirty.remove(&id);
                }
            }
        }

        match self.entries.get(&id) {
            Some(CacheEntry::Resident(output))
                if !output.dirty
                    && output.height.metrics.width == metrics.width
                    && output.height.metrics.height == metrics.height =>
            {
                Some(output)
            }
            _ => None,
        }
    }

    /// Whether a clean checkpoint for `id` at `metrics` is available without
    /// materializing it — answered from the entry state, or a header-only disk
    /// probe that adopts a surviving spill as a reclaimed `Spilled` entry. Used by
    /// the incremental scan so probing a pin never inflates its buffers.
    pub fn has_clean(&mut self, id: LayerId, metrics: HeightfieldMetrics) -> bool {
        if let Some(entry) = self.entries.get(&id) {
            return entry.clean_match(metrics);
        }
        // Missing in memory: a prior instance may have left a valid spill (e.g. a
        // worker restart adopting the previous thread's bakes). Probe the header
        // only and adopt a lightweight descriptor.
        if let Some(disk) = &self.disk {
            if disk.probe(id, metrics) {
                self.entries.insert(
                    id,
                    CacheEntry::Spilled {
                        metrics,
                        dirty: false,
                    },
                );
                return true;
            }
        }
        false
    }

    pub fn mark_dirty(&mut self, id: LayerId) {
        // Plain dirty escalates any bounded seed record to whole-field: dropping
        // the record makes `seed_state` report `AllTiles`.
        self.seed_dirty.remove(&id);
        match self.entries.get_mut(&id) {
            Some(CacheEntry::Resident(output)) => output.dirty = true,
            Some(CacheEntry::Spilled { dirty, .. }) => *dirty = true,
            None => {}
        }
        if let Some(disk) = &self.disk {
            disk.invalidate(id);
        }
        self.generation = self.generation.wrapping_add(1);
    }

    /// Dirty `id` over a bounded seed tile set (#100 phase 2).
    ///
    /// Unions `tiles` into the entry's seed set when the mark can stay bounded —
    /// the entry is `Resident` and either clean or already carries a seed record.
    /// A `Spilled` or missing entry (no in-memory buffer to carry forward), or an
    /// entry already plain-dirtied over the whole field, escalates to a full
    /// [`Self::mark_dirty`] instead (the B1-D8 spill/salvage contract is untouched:
    /// any partial-region dirty on a spilled entry invalidates its disk file).
    ///
    /// An empty `tiles` slice on a bounded entry records an *empty* seed set — the
    /// input-only case for a layer above the edited one: dirty, but with no
    /// own-contribution change of its own.
    pub fn mark_dirty_region(&mut self, id: LayerId, tiles: &[TileId]) {
        let can_bound = match self.entries.get(&id) {
            Some(CacheEntry::Resident(output)) => {
                !output.dirty || self.seed_dirty.contains_key(&id)
            }
            // Spilled buffers live on disk; a missing entry has nothing to seed
            // from. Either way there is no clean in-memory checkpoint to patch.
            _ => false,
        };
        if can_bound {
            self.seed_dirty
                .entry(id)
                .or_default()
                .extend(tiles.iter().copied());
            match self.entries.get_mut(&id) {
                Some(CacheEntry::Resident(output)) => output.dirty = true,
                Some(CacheEntry::Spilled { dirty, .. }) => *dirty = true,
                None => {}
            }
            if let Some(disk) = &self.disk {
                disk.invalidate(id);
            }
            self.generation = self.generation.wrapping_add(1);
        } else {
            self.mark_dirty(id);
        }
    }

    /// Resolve the per-tile dirty state of `id`. See [`SeedState`]. Consulted by
    /// the tile-scoped suffix walk to decide scoped-vs-whole-field recompute.
    pub fn seed_state(&self, id: LayerId) -> SeedState {
        match self.entries.get(&id) {
            Some(CacheEntry::Resident(output)) => {
                if !output.dirty {
                    SeedState::Clean
                } else if let Some(tiles) = self.seed_dirty.get(&id) {
                    SeedState::Tiles(tiles.iter().copied().collect())
                } else {
                    SeedState::AllTiles
                }
            }
            // A spilled entry keeps no in-memory buffer to seed a partial patch
            // from, so a dirty one is treated as whole-field.
            Some(CacheEntry::Spilled { dirty, .. }) => {
                if *dirty {
                    SeedState::AllTiles
                } else {
                    SeedState::Clean
                }
            }
            None => SeedState::AllTiles,
        }
    }

    pub fn is_dirty(&self, id: LayerId) -> bool {
        match self.entries.get(&id) {
            Some(entry) => entry.dirty(),
            None => true,
        }
    }

    pub fn get_mut(&mut self, id: LayerId) -> Option<&mut CachedOutput> {
        match self.entries.get_mut(&id) {
            Some(CacheEntry::Resident(output)) => Some(output),
            _ => None,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// True when `id` is present but its buffers have been spilled to disk and
    /// reclaimed from memory (observability / tests).
    pub fn is_spilled(&self, id: LayerId) -> bool {
        matches!(self.entries.get(&id), Some(CacheEntry::Spilled { .. }))
    }

    pub fn disk_root(&self) -> Option<&std::path::Path> {
        self.disk.as_ref().map(|d| d.root())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heightfield::HeightfieldMetrics;
    use std::path::PathBuf;

    fn scratch_dir(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("terra_layer_cache_{tag}_{}", uuid::Uuid::new_v4()))
    }

    fn output(metrics: HeightfieldMetrics, value: f32) -> CachedOutput {
        CachedOutput {
            height: Heightfield::filled(metrics, value),
            generation: 1,
            dirty: false,
            aux: HashMap::new(),
            strata: None,
        }
    }

    /// B1-D8 revert check: `insert_baked` spills and reclaims memory, and the
    /// entry reloads on demand. If eviction is reverted the entry stays resident
    /// and `is_spilled` fails.
    #[test]
    fn insert_baked_spills_and_reclaims_memory() {
        let dir = scratch_dir("evict");
        let mut cache = LayerCache::with_disk(&dir);
        let metrics = HeightfieldMetrics::new(8, 8, 80.0, 80.0);
        let id = LayerId::new();
        let mut out = output(metrics, 0.0);
        out.height.set(2, 3, 9.0);

        cache.insert_baked(id, out);
        assert!(cache.is_spilled(id), "baked entry should be spilled");
        assert!(
            cache.get(id).is_none(),
            "a spilled entry holds no in-memory buffers"
        );

        let loaded = cache.get_or_load(id, metrics).expect("reload from disk");
        assert_eq!(loaded.height.get(2, 3), 9.0);
        assert!(
            !cache.is_spilled(id),
            "after reload the entry is resident again"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A fresh cache on a root that already holds a spill (the worker-restart
    /// shape) adopts it via `has_clean` without loading its blobs.
    #[test]
    fn has_clean_adopts_a_surviving_spill() {
        let dir = scratch_dir("adopt");
        let metrics = HeightfieldMetrics::new(8, 8, 80.0, 80.0);
        let id = LayerId::new();
        {
            let mut writer = LayerCache::with_disk(&dir);
            writer.insert_baked(id, output(metrics, 4.0));
        }
        let mut reader = LayerCache::with_disk(&dir);
        assert!(reader.has_clean(id, metrics), "surviving spill is clean");
        assert!(reader.is_spilled(id), "adopted as a reclaimed descriptor");
        let loaded = reader.get_or_load(id, metrics).expect("reload");
        assert_eq!(loaded.height.get(0, 0), 4.0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn marking_a_spilled_entry_dirty_invalidates_the_disk_file() {
        let dir = scratch_dir("dirty");
        let metrics = HeightfieldMetrics::new(8, 8, 80.0, 80.0);
        let id = LayerId::new();
        let mut cache = LayerCache::with_disk(&dir);
        cache.insert_baked(id, output(metrics, 2.0));
        assert!(cache.is_spilled(id));
        cache.mark_dirty(id);
        assert!(cache.is_dirty(id));
        assert!(
            cache.get_or_load(id, metrics).is_none(),
            "a dirtied bake must not reload"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn tiles(ids: &[(u32, u32)]) -> Vec<TileId> {
        ids.iter().map(|&(tx, tz)| TileId { tx, tz }).collect()
    }

    /// A clean resident entry reports `Clean`; a plain `mark_dirty` reports the
    /// conservative `AllTiles`; a `mark_dirty_region` reports exactly its seeds.
    #[test]
    fn seed_state_tracks_clean_all_and_bounded() {
        let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
        let id = LayerId::new();
        let mut cache = LayerCache::without_disk();
        cache.insert(id, output(metrics, 0.0));
        assert_eq!(cache.seed_state(id), SeedState::Clean);

        cache.mark_dirty(id);
        assert_eq!(cache.seed_state(id), SeedState::AllTiles);

        // Re-store (clean again), then region-mark a bounded set.
        cache.insert(id, output(metrics, 0.0));
        cache.mark_dirty_region(id, &tiles(&[(0, 0), (1, 0)]));
        match cache.seed_state(id) {
            SeedState::Tiles(mut got) => {
                got.sort_by_key(|t| (t.tz, t.tx));
                assert_eq!(got, tiles(&[(0, 0), (1, 0)]));
            }
            other => panic!("expected bounded seeds, got {other:?}"),
        }
        assert!(cache.is_dirty(id));
    }

    /// A missing entry is `AllTiles`: there is no previous output to seed a
    /// partial patch from.
    #[test]
    fn seed_state_missing_entry_is_all_tiles() {
        let cache = LayerCache::without_disk();
        assert_eq!(cache.seed_state(LayerId::new()), SeedState::AllTiles);
    }

    /// Region-marks union across calls, growing the seed set.
    #[test]
    fn mark_dirty_region_unions_seeds() {
        let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
        let id = LayerId::new();
        let mut cache = LayerCache::without_disk();
        cache.insert(id, output(metrics, 0.0));
        cache.mark_dirty_region(id, &tiles(&[(0, 0)]));
        cache.mark_dirty_region(id, &tiles(&[(1, 1)]));
        match cache.seed_state(id) {
            SeedState::Tiles(mut got) => {
                got.sort_by_key(|t| (t.tz, t.tx));
                assert_eq!(got, tiles(&[(0, 0), (1, 1)]));
            }
            other => panic!("expected unioned seeds, got {other:?}"),
        }
    }

    /// Ordering hazard: a plain `mark_dirty` before a region-mark keeps the entry
    /// at `AllTiles` — the whole-field mark must not be narrowed to a region.
    #[test]
    fn plain_dirty_then_region_stays_all_tiles() {
        let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
        let id = LayerId::new();
        let mut cache = LayerCache::without_disk();
        cache.insert(id, output(metrics, 0.0));
        cache.mark_dirty(id);
        cache.mark_dirty_region(id, &tiles(&[(0, 0)]));
        assert_eq!(cache.seed_state(id), SeedState::AllTiles);
    }

    /// Ordering hazard, the other direction: a plain `mark_dirty` after a
    /// region-mark escalates back to `AllTiles`.
    #[test]
    fn region_then_plain_dirty_escalates_to_all_tiles() {
        let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
        let id = LayerId::new();
        let mut cache = LayerCache::without_disk();
        cache.insert(id, output(metrics, 0.0));
        cache.mark_dirty_region(id, &tiles(&[(0, 0)]));
        cache.mark_dirty(id);
        assert_eq!(cache.seed_state(id), SeedState::AllTiles);
    }

    /// An empty region-mark on a clean entry is the input-only case: dirty, but
    /// with an empty (not absent) seed set.
    #[test]
    fn empty_region_mark_is_dirty_with_empty_seeds() {
        let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
        let id = LayerId::new();
        let mut cache = LayerCache::without_disk();
        cache.insert(id, output(metrics, 0.0));
        cache.mark_dirty_region(id, &[]);
        assert!(cache.is_dirty(id));
        assert_eq!(cache.seed_state(id), SeedState::Tiles(Vec::new()));
    }

    /// Storing a fresh output drops the seed record: the layer is clean again.
    #[test]
    fn store_clears_seed_record() {
        let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
        let id = LayerId::new();
        let mut cache = LayerCache::without_disk();
        cache.insert(id, output(metrics, 0.0));
        cache.mark_dirty_region(id, &tiles(&[(0, 0)]));
        cache.insert(id, output(metrics, 1.0));
        assert_eq!(cache.seed_state(id), SeedState::Clean);
    }

    /// A region-mark on a spilled entry escalates to whole-field (no in-memory
    /// buffer to patch) and invalidates the disk file.
    #[test]
    fn region_mark_on_spilled_escalates() {
        let dir = scratch_dir("region_spill");
        let metrics = HeightfieldMetrics::new(8, 8, 80.0, 80.0);
        let id = LayerId::new();
        let mut cache = LayerCache::with_disk(&dir);
        cache.insert_baked(id, output(metrics, 2.0));
        assert!(cache.is_spilled(id));
        cache.mark_dirty_region(id, &tiles(&[(0, 0)]));
        assert_eq!(cache.seed_state(id), SeedState::AllTiles);
        assert!(
            cache.get_or_load(id, metrics).is_none(),
            "a region-dirtied bake must not reload"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
