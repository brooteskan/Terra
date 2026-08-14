use super::TerrainTileKey;
use crate::heightfield::TileId;
use crate::layer::LayerId;
use std::collections::HashMap;

/// Composite cache identity for a resident terrain tile payload.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TerrainCacheKey {
    pub layer_id: Option<LayerId>,
    pub generation: u32,
    pub tile: TileId,
    pub level: u8,
    pub param_hash: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TilePageHandle {
    pub slot: u32,
    pub generation: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentTile {
    pub handle: TilePageHandle,
    pub bytes: u64,
    pub revision: u64,
    pub input_revision_hash: u64,
    pub last_used_tick: u64,
    pub pin_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileCacheInsert {
    pub handle: TilePageHandle,
    pub evicted: Vec<TerrainTileKey>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileCacheError {
    EntryExceedsBudget {
        bytes: u64,
        budget_bytes: u64,
    },
    AllCandidatesPinned {
        required_bytes: u64,
        budget_bytes: u64,
    },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TileCacheStats {
    pub budget_bytes: u64,
    pub used_bytes: u64,
    pub resident_tiles: usize,
    pub pinned_tiles: usize,
    pub evictions: u64,
    pub hits: u64,
    pub misses: u64,
}

#[derive(Debug, Clone)]
struct PageSlot {
    generation: u32,
    key: Option<TerrainTileKey>,
}

/// Backend-neutral byte-budget and page-table policy for terrain payloads.
/// Payload memory stays backend-owned; generation handles prevent stale page references.
#[derive(Debug, Clone)]
pub struct TileResidencyCache {
    budget_bytes: u64,
    used_bytes: u64,
    tick: u64,
    entries: HashMap<TerrainTileKey, ResidentTile>,
    slots: Vec<PageSlot>,
    free_slots: Vec<u32>,
    evictions: u64,
    hits: u64,
    misses: u64,
}

impl TileResidencyCache {
    pub fn new(budget_bytes: u64) -> Self {
        Self {
            budget_bytes,
            used_bytes: 0,
            tick: 0,
            entries: HashMap::new(),
            slots: Vec::new(),
            free_slots: Vec::new(),
            evictions: 0,
            hits: 0,
            misses: 0,
        }
    }

    pub fn get(&mut self, key: &TerrainTileKey) -> Option<&ResidentTile> {
        self.tick = self.tick.wrapping_add(1);
        match self.entries.get_mut(key) {
            Some(entry) => {
                entry.last_used_tick = self.tick;
                self.hits = self.hits.saturating_add(1);
            }
            None => {
                self.misses = self.misses.saturating_add(1);
                return None;
            }
        }
        self.entries.get(key)
    }

    pub fn peek(&self, key: &TerrainTileKey) -> Option<&ResidentTile> {
        self.entries.get(key)
    }

    pub fn resolve_handle(&self, handle: TilePageHandle) -> Option<&TerrainTileKey> {
        let slot = self.slots.get(handle.slot as usize)?;
        (slot.generation == handle.generation)
            .then_some(slot.key.as_ref())
            .flatten()
    }

    pub fn insert(
        &mut self,
        key: TerrainTileKey,
        bytes: u64,
        revision: u64,
        input_revision_hash: u64,
    ) -> Result<TileCacheInsert, TileCacheError> {
        if bytes > self.budget_bytes {
            return Err(TileCacheError::EntryExceedsBudget {
                bytes,
                budget_bytes: self.budget_bytes,
            });
        }
        let old_bytes = self.entries.get(&key).map_or(0, |entry| entry.bytes);
        let projected = self
            .used_bytes
            .saturating_sub(old_bytes)
            .saturating_add(bytes);
        let required = projected.saturating_sub(self.budget_bytes);
        let victims = self.lru_victims(Some(&key), required);
        let reclaimed = victims
            .iter()
            .filter_map(|victim| self.entries.get(victim))
            .map(|entry| entry.bytes)
            .sum::<u64>();
        if reclaimed < required {
            return Err(TileCacheError::AllCandidatesPinned {
                required_bytes: required,
                budget_bytes: self.budget_bytes,
            });
        }
        let evicted = self.evict(victims);
        self.tick = self.tick.wrapping_add(1);
        if let Some(entry) = self.entries.get_mut(&key) {
            self.used_bytes = self
                .used_bytes
                .saturating_sub(entry.bytes)
                .saturating_add(bytes);
            entry.bytes = bytes;
            entry.revision = revision;
            entry.input_revision_hash = input_revision_hash;
            entry.last_used_tick = self.tick;
            return Ok(TileCacheInsert {
                handle: entry.handle,
                evicted,
            });
        }
        let handle = self.allocate(key.clone());
        self.entries.insert(
            key,
            ResidentTile {
                handle,
                bytes,
                revision,
                input_revision_hash,
                last_used_tick: self.tick,
                pin_count: 0,
            },
        );
        self.used_bytes = self.used_bytes.saturating_add(bytes);
        Ok(TileCacheInsert { handle, evicted })
    }

    pub fn remove(&mut self, key: &TerrainTileKey) -> bool {
        let Some(entry) = self.entries.remove(key) else {
            return false;
        };
        self.used_bytes = self.used_bytes.saturating_sub(entry.bytes);
        let slot = &mut self.slots[entry.handle.slot as usize];
        slot.key = None;
        slot.generation = slot.generation.wrapping_add(1).max(1);
        self.free_slots.push(entry.handle.slot);
        true
    }

    /// Remove all residency while preserving the cache budget and slot generations.
    ///
    /// Advancing every allocated slot is required at document boundaries: rebuilding
    /// the cache from scratch would allow an old generation-1 handle to resolve again
    /// when the same physical slot is reused by the next document.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.used_bytes = 0;
        self.free_slots.clear();
        self.free_slots.reserve(self.slots.len());
        for (index, slot) in self.slots.iter_mut().enumerate() {
            slot.key = None;
            slot.generation = slot.generation.wrapping_add(1).max(1);
            self.free_slots.push(index as u32);
        }
    }

    pub fn pin(&mut self, key: &TerrainTileKey) -> bool {
        self.entries.get_mut(key).is_some_and(|entry| {
            entry.pin_count = entry.pin_count.saturating_add(1);
            true
        })
    }

    pub fn unpin(&mut self, key: &TerrainTileKey) -> bool {
        self.entries.get_mut(key).is_some_and(|entry| {
            entry.pin_count = entry.pin_count.saturating_sub(1);
            true
        })
    }

    pub fn stats(&self) -> TileCacheStats {
        TileCacheStats {
            budget_bytes: self.budget_bytes,
            used_bytes: self.used_bytes,
            resident_tiles: self.entries.len(),
            pinned_tiles: self
                .entries
                .values()
                .filter(|entry| entry.pin_count > 0)
                .count(),
            evictions: self.evictions,
            hits: self.hits,
            misses: self.misses,
        }
    }

    fn lru_victims(
        &self,
        protected: Option<&TerrainTileKey>,
        required: u64,
    ) -> Vec<TerrainTileKey> {
        let mut candidates: Vec<_> = self
            .entries
            .iter()
            .filter(|(key, entry)| Some(*key) != protected && entry.pin_count == 0)
            .map(|(key, entry)| (entry.last_used_tick, key.clone(), entry.bytes))
            .collect();
        candidates.sort_by_key(|candidate| candidate.0);
        let mut reclaimed = 0;
        let mut victims = Vec::new();
        for (_, key, bytes) in candidates {
            if reclaimed >= required {
                break;
            }
            reclaimed = reclaimed.saturating_add(bytes);
            victims.push(key);
        }
        victims
    }

    fn evict(&mut self, keys: Vec<TerrainTileKey>) -> Vec<TerrainTileKey> {
        keys.into_iter()
            .filter(|key| {
                let removed = self.remove(key);
                self.evictions += u64::from(removed);
                removed
            })
            .collect()
    }

    fn allocate(&mut self, key: TerrainTileKey) -> TilePageHandle {
        if let Some(slot_index) = self.free_slots.pop() {
            let slot = &mut self.slots[slot_index as usize];
            slot.key = Some(key);
            TilePageHandle {
                slot: slot_index,
                generation: slot.generation,
            }
        } else {
            let slot = self.slots.len() as u32;
            self.slots.push(PageSlot {
                generation: 1,
                key: Some(key),
            });
            TilePageHandle {
                slot,
                generation: 1,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FieldId, LayerId, TileId};

    fn key(layer: LayerId, tx: u32) -> TerrainTileKey {
        TerrainTileKey {
            layer: Some(layer),
            field: FieldId::Height,
            level: 3,
            tile: TileId { tx, tz: 0 },
        }
    }

    #[test]
    fn layer_identity_prevents_page_collisions() {
        let mut cache = TileResidencyCache::new(32);
        let a = key(LayerId::new(), 0);
        let b = key(LayerId::new(), 0);
        let ah = cache.insert(a.clone(), 8, 1, 1).unwrap().handle;
        let bh = cache.insert(b.clone(), 8, 1, 1).unwrap().handle;
        assert_ne!(ah, bh);
        assert_eq!(cache.resolve_handle(ah), Some(&a));
        assert_eq!(cache.resolve_handle(bh), Some(&b));
    }

    #[test]
    fn evicts_lru_unpinned_tile() {
        let layer = LayerId::new();
        let mut cache = TileResidencyCache::new(16);
        let a = key(layer, 0);
        let b = key(layer, 1);
        let c = key(layer, 2);
        cache.insert(a.clone(), 8, 1, 1).unwrap();
        cache.insert(b.clone(), 8, 1, 1).unwrap();
        cache.get(&a);
        assert_eq!(cache.insert(c, 8, 1, 1).unwrap().evicted, vec![b]);
    }

    #[test]
    fn stale_handle_cannot_resolve_reused_page() {
        let layer = LayerId::new();
        let mut cache = TileResidencyCache::new(8);
        let old = cache.insert(key(layer, 0), 8, 1, 1).unwrap().handle;
        let b = key(layer, 1);
        let new = cache.insert(b.clone(), 8, 2, 2).unwrap().handle;
        assert_eq!(old.slot, new.slot);
        assert_ne!(old.generation, new.generation);
        assert_eq!(cache.resolve_handle(old), None);
        assert_eq!(cache.resolve_handle(new), Some(&b));
    }

    #[test]
    fn clear_invalidates_handles_and_preserves_reusable_capacity() {
        let layer = LayerId::new();
        let mut cache = TileResidencyCache::new(16);
        let a = key(layer, 0);
        let b = key(layer, 1);
        let ah = cache.insert(a.clone(), 8, 1, 1).unwrap().handle;
        let bh = cache.insert(b.clone(), 8, 1, 1).unwrap().handle;
        assert!(cache.pin(&a));

        cache.clear();

        let stats = cache.stats();
        assert_eq!(stats.budget_bytes, 16);
        assert_eq!(stats.used_bytes, 0);
        assert_eq!(stats.resident_tiles, 0);
        assert_eq!(stats.pinned_tiles, 0);
        assert_eq!(cache.resolve_handle(ah), None);
        assert_eq!(cache.resolve_handle(bh), None);

        let replacement = key(layer, 2);
        let new = cache.insert(replacement.clone(), 8, 2, 2).unwrap().handle;
        assert!(new.slot == ah.slot || new.slot == bh.slot);
        assert_ne!(
            new.generation,
            if new.slot == ah.slot {
                ah.generation
            } else {
                bh.generation
            }
        );
        assert_eq!(cache.resolve_handle(ah), None);
        assert_eq!(cache.resolve_handle(bh), None);
        assert_eq!(cache.resolve_handle(new), Some(&replacement));
    }

    #[test]
    fn pinned_tiles_fail_explicitly_when_budget_is_full() {
        let layer = LayerId::new();
        let mut cache = TileResidencyCache::new(8);
        let a = key(layer, 0);
        cache.insert(a.clone(), 8, 1, 1).unwrap();
        cache.pin(&a);
        assert!(matches!(
            cache.insert(key(layer, 1), 8, 1, 1),
            Err(TileCacheError::AllCandidatesPinned { .. })
        ));
    }

    #[test]
    fn terrain_cache_key_distinguishes_generations() {
        let layer = LayerId::new();
        let a = TerrainCacheKey {
            layer_id: Some(layer),
            generation: 1,
            tile: TileId { tx: 0, tz: 0 },
            level: 2,
            param_hash: 42,
        };
        let b = TerrainCacheKey {
            generation: 2,
            ..a.clone()
        };
        assert_ne!(a, b);
    }
}
