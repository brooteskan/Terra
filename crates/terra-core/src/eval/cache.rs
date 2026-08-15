use super::smart_cache::DiskSmartCache;
use crate::heightfield::{Heightfield, HeightfieldMetrics};
use crate::layer::LayerId;
use crate::mask::MaskField;
use std::collections::HashMap;
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

#[derive(Debug)]
pub struct LayerCache {
    entries: HashMap<LayerId, CachedOutput>,
    pub generation: u64,
    /// Optional on-disk spill for baked (`cached`) layer checkpoints.
    disk: Option<DiskSmartCache>,
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
            disk: Some(DiskSmartCache::new(DiskSmartCache::default_location())),
        }
    }

    pub fn with_disk(root: impl Into<PathBuf>) -> Self {
        Self {
            entries: HashMap::new(),
            generation: 0,
            disk: Some(DiskSmartCache::new(root)),
        }
    }

    pub fn without_disk() -> Self {
        Self {
            entries: HashMap::new(),
            generation: 0,
            disk: None,
        }
    }

    pub fn enable_disk(&mut self, root: impl Into<PathBuf>) {
        self.disk = Some(DiskSmartCache::new(root));
    }

    pub fn disable_disk(&mut self) {
        self.disk = None;
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.generation = self.generation.wrapping_add(1);
    }

    pub fn insert(&mut self, id: LayerId, output: CachedOutput) {
        self.entries.insert(id, output);
    }

    /// Insert and write a durable disk checkpoint.
    pub fn insert_baked(&mut self, id: LayerId, output: CachedOutput) {
        if let Some(disk) = &self.disk {
            let _ = disk.spill(id, &output);
        }
        self.entries.insert(id, output);
    }

    pub fn get(&self, id: LayerId) -> Option<&CachedOutput> {
        self.entries.get(&id)
    }

    /// Memory hit, else try disk reload for a clean bake matching `metrics`.
    pub fn get_or_load(
        &mut self,
        id: LayerId,
        metrics: HeightfieldMetrics,
    ) -> Option<&CachedOutput> {
        let mem_ok = self.entries.get(&id).is_some_and(|e| {
            !e.dirty
                && e.height.metrics.width == metrics.width
                && e.height.metrics.height == metrics.height
        });
        if !mem_ok {
            // Dirty / missing / wrong size: try disk (dirty entries invalidate disk on mark).
            if self.entries.get(&id).is_none_or(|e| e.dirty) {
                if let Some(disk) = &self.disk {
                    if let Ok(Some(loaded)) = disk.load(id, metrics) {
                        self.entries.insert(id, loaded);
                    }
                }
            }
        }
        self.entries.get(&id).filter(|e| {
            !e.dirty
                && e.height.metrics.width == metrics.width
                && e.height.metrics.height == metrics.height
        })
    }

    pub fn mark_dirty(&mut self, id: LayerId) {
        if let Some(e) = self.entries.get_mut(&id) {
            e.dirty = true;
        }
        if let Some(disk) = &self.disk {
            disk.invalidate(id);
        }
        self.generation = self.generation.wrapping_add(1);
    }

    pub fn is_dirty(&self, id: LayerId) -> bool {
        match self.entries.get(&id) {
            Some(e) => e.dirty,
            None => true,
        }
    }

    pub fn get_mut(&mut self, id: LayerId) -> Option<&mut CachedOutput> {
        self.entries.get_mut(&id)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn disk_root(&self) -> Option<&std::path::Path> {
        self.disk.as_ref().map(|d| d.root())
    }
}
