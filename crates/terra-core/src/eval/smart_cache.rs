//! Session-scoped disk spill for baked layer checkpoints (Gaea-style Smart Cache).
//!
//! When a layer is pinned (`cached`), its height + aux maps can be written to a
//! single atomic bake file so the pinned output is reclaimed from memory and a
//! later rebuild in the *same session* skips re-running the processor. This is a
//! per-instance spill, **not** a cross-session pin store: each live [`LayerCache`]
//! owns a private root under [`DiskSmartCache::default_location`] that is deleted
//! when the cache drops, and project-open still marks every layer dirty. The one
//! deliberate cross-instance handoff is the worker restart in
//! [`super::EvalWorker`], which carries this store into its replacement evaluator
//! so bakes written before a panic survive.
//!
//! Each bake is one `<layer-uuid>.bake` file written temp-then-rename, so a reader
//! sees either the previous bake or the new one — never a torn mix — and `load`
//! additionally validates the payload length against the header, so a truncated or
//! corrupt file degrades to a miss rather than a false hit.

use super::cache::CachedOutput;
use super::EvalError;
use crate::heightfield::{Heightfield, HeightfieldMetrics};
use crate::layer::LayerId;
use crate::mask::MaskField;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Once};
use std::time::{Duration, SystemTime};

/// Magic for the single-file bake format introduced by B1-D8. The prior format
/// used separate `.meta.json` / `.height.bin` / `.aux.*.bin` files; those are
/// swept, never read, so a new magic is enough to invalidate them.
const MAGIC: &[u8; 4] = b"TCB1";
// Bump whenever terrain processors change in a way that makes baked outputs stale.
// Version 2 invalidated checkpoints from the pre-fidelity procedural generators.
// Version 3 invalidated procedural outputs created before 64-bit seed canonicalization.
// Version 4 invalidated checkpoints where loose_sediment could contain coarse debris
// while sediment_depth held the actual fine-sediment inventory.
// Version 5 moved to the single-file atomic bake layout (B1-D8); older multi-file
// spills use a different on-disk shape and are swept, not loaded.
const VERSION: u32 = 5;

/// Age past which an orphaned per-instance root (whose `RootGuard` never ran,
/// e.g. a crashed session) is swept. Live instances keep a recent mtime.
const STALE_INSTANCE_AGE: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// Header prefix of a `.bake` file: identity + validation metadata, followed by
/// the raw f32 blobs (height, then each aux by sorted name).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BakeHeader {
    version: u32,
    metrics: HeightfieldMetrics,
    generation: u64,
    aux_names: Vec<String>,
    #[serde(default)]
    strata: Option<Vec<crate::layer::Stratum>>,
}

/// On-disk bake store keyed by [`LayerId`], writing one atomic file per layer.
#[derive(Debug, Clone)]
pub struct DiskSmartCache {
    root: PathBuf,
    /// RAII cleanup for an owned per-instance root; `None` for borrowed roots.
    /// Held only for its `Drop`, so it is never read directly.
    #[allow(dead_code)]
    guard: Option<Arc<RootGuard>>,
}

#[derive(Debug)]
struct RootGuard {
    root: PathBuf,
}

impl Drop for RootGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

impl DiskSmartCache {
    /// Borrowed-root store: the caller owns `root` and its lifetime. Used by
    /// tests and by explicit `LayerCache::with_disk` / `enable_disk` roots.
    /// Dropping this store does **not** delete `root`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let _ = fs::create_dir_all(&root);
        Self { root, guard: None }
    }

    /// Private per-instance store under [`Self::default_location`]. Each call gets
    /// its own `<parent>/<uuid>` directory, removed when the last clone drops.
    /// This is the single-writer root every live [`LayerCache`] owns; instances
    /// never share a directory. Constructing the first instance in a process also
    /// sweeps the parent of legacy files and crashed-session leftovers.
    pub fn owned_instance() -> Self {
        let parent = Self::default_location();
        let _ = fs::create_dir_all(&parent);
        sweep_parent_once(&parent);
        let root = parent.join(uuid::Uuid::new_v4().to_string());
        let _ = fs::create_dir_all(&root);
        Self {
            guard: Some(Arc::new(RootGuard { root: root.clone() })),
            root,
        }
    }

    /// Shared parent directory `temp_dir()/terra_smart_cache`. Each evaluator
    /// instance owns a private `<this>/<uuid>` subdirectory; nothing writes bakes
    /// directly under this parent.
    pub fn default_location() -> PathBuf {
        std::env::temp_dir().join("terra_smart_cache")
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn bake_path(&self, id: LayerId) -> PathBuf {
        self.root.join(format!("{}.bake", id.0))
    }

    fn bake_tmp_path(&self, id: LayerId) -> PathBuf {
        self.root.join(format!("{}.bake.tmp", id.0))
    }

    pub fn invalidate(&self, id: LayerId) {
        let _ = fs::remove_file(self.bake_path(id));
        let _ = fs::remove_file(self.bake_tmp_path(id));
    }

    pub fn clear_all(&self) {
        let _ = fs::remove_dir_all(&self.root);
        let _ = fs::create_dir_all(&self.root);
    }

    /// Write `output` as one atomic `<id>.bake` file (temp + rename).
    pub fn spill(&self, id: LayerId, output: &CachedOutput) -> Result<(), EvalError> {
        fs::create_dir_all(&self.root).map_err(io_err)?;

        let mut aux_names: Vec<String> = output.aux.keys().cloned().collect();
        aux_names.sort();

        let header = BakeHeader {
            version: VERSION,
            metrics: output.height.metrics,
            generation: output.generation,
            aux_names: aux_names.clone(),
            strata: output.strata.clone(),
        };
        let header_json = serde_json::to_vec(&header).map_err(io_err)?;

        let cells = (output.height.metrics.width as usize)
            .saturating_mul(output.height.metrics.height as usize);
        let mut buf = Vec::with_capacity(8 + header_json.len() + (1 + aux_names.len()) * cells * 4);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&(header_json.len() as u32).to_le_bytes());
        buf.extend_from_slice(&header_json);
        append_f32_le(&mut buf, &output.height.to_dense());
        for name in &aux_names {
            // Every aux blob is exactly width*height f32s; a missing key would
            // desync the fixed-stride payload, so emit zeros rather than skip.
            match output.aux.get(name) {
                Some(field) => append_f32_le(&mut buf, field.data()),
                None => buf.resize(buf.len() + cells * 4, 0),
            }
        }

        let tmp = self.bake_tmp_path(id);
        let final_path = self.bake_path(id);
        fs::write(&tmp, &buf).map_err(io_err)?;
        // Atomic publish: a reader sees either the old bake or the new one, never
        // a partially written file. On Windows `rename` maps to MoveFileEx with
        // REPLACE_EXISTING, so this also overwrites any prior bake for this id.
        fs::rename(&tmp, &final_path).map_err(|e| {
            let _ = fs::remove_file(&tmp);
            io_err(e)
        })?;
        Ok(())
    }

    /// Load a baked checkpoint if present, metrics match, and the payload length
    /// is exactly what the header implies. A truncated / torn / mismatched file
    /// returns `Ok(None)` (a miss), never partial data as a clean hit.
    pub fn load(
        &self,
        id: LayerId,
        expected: HeightfieldMetrics,
    ) -> Result<Option<CachedOutput>, EvalError> {
        let bytes = match fs::read(self.bake_path(id)) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(io_err(e)),
        };
        let Some((header, payload)) = parse_validated(&bytes, expected) else {
            return Ok(None);
        };

        let cells = (expected.width * expected.height) as usize;
        let blob_bytes = cells * 4;
        let height = Heightfield::from_dense(expected, &read_f32_le(&payload[..blob_bytes]));

        let mut aux = HashMap::new();
        let mut offset = blob_bytes;
        for name in &header.aux_names {
            let data = read_f32_le(&payload[offset..offset + blob_bytes]);
            let mut field = MaskField::zeros(expected);
            field.data_mut().copy_from_slice(&data);
            aux.insert(name.clone(), field);
            offset += blob_bytes;
        }

        Ok(Some(CachedOutput {
            height,
            generation: header.generation,
            dirty: false,
            aux,
            strata: header.strata,
        }))
    }

    /// Whether a valid bake for `id` at `expected` exists — no blob load. Used to
    /// adopt a surviving spill (worker restart) as a reclaimed entry. Applies the
    /// same validation as [`Self::load`], so a torn file is not claimed clean.
    pub fn probe(&self, id: LayerId, expected: HeightfieldMetrics) -> bool {
        let Ok(bytes) = fs::read(self.bake_path(id)) else {
            return false;
        };
        parse_validated(&bytes, expected).is_some()
    }
}

fn io_err<E: ToString>(e: E) -> EvalError {
    EvalError::Io(e.to_string())
}

/// Parse and validate a `.bake` buffer. Returns the header and the payload slice
/// only when magic, version, metrics, and — critically — the exact payload length
/// all check out. Any shortfall or mismatch yields `None`, i.e. a cache miss.
fn parse_validated(bytes: &[u8], expected: HeightfieldMetrics) -> Option<(BakeHeader, &[u8])> {
    if bytes.len() < 8 || &bytes[0..4] != MAGIC {
        return None;
    }
    let header_len = u32::from_le_bytes(bytes[4..8].try_into().ok()?) as usize;
    let header_end = 8usize.checked_add(header_len)?;
    let header_bytes = bytes.get(8..header_end)?;
    let header: BakeHeader = serde_json::from_slice(header_bytes).ok()?;
    if header.version != VERSION {
        return None;
    }
    if header.metrics.width != expected.width
        || header.metrics.height != expected.height
        || (header.metrics.world_size_x - expected.world_size_x).abs() > 1e-3
        || (header.metrics.world_size_z - expected.world_size_z).abs() > 1e-3
    {
        return None;
    }

    let cells = (expected.width as usize).checked_mul(expected.height as usize)?;
    let blob_bytes = cells.checked_mul(4)?;
    let n_blobs = 1usize.checked_add(header.aux_names.len())?;
    let payload_len = blob_bytes.checked_mul(n_blobs)?;
    let payload = bytes.get(header_end..)?;
    if payload.len() != payload_len {
        // Truncated (interrupted write) or oversized (mixed / corrupt) — the very
        // torn-file case that must never load as a clean checkpoint.
        return None;
    }
    Some((header, payload))
}

fn append_f32_le(buf: &mut Vec<u8>, data: &[f32]) {
    buf.reserve(data.len() * 4);
    for &v in data {
        buf.extend_from_slice(&v.to_bits().to_le_bytes());
    }
}

fn read_f32_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect()
}

fn sweep_parent_once(parent: &Path) {
    static SWEEP: Once = Once::new();
    SWEEP.call_once(|| sweep_parent(parent));
}

/// Best-effort reclaim under the shared parent: delete pre-B1-D8 multi-file spills
/// (which accumulated here until the OS cleared temp) and per-instance roots left
/// by crashed sessions (whose `RootGuard` never ran).
fn sweep_parent(parent: &Path) {
    let Ok(entries) = fs::read_dir(parent) else {
        return;
    };
    let now = SystemTime::now();
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            let stale = entry
                .metadata()
                .and_then(|m| m.modified())
                .map(|modified| {
                    now.duration_since(modified).unwrap_or_default() > STALE_INSTANCE_AGE
                })
                .unwrap_or(false);
            if stale {
                let _ = fs::remove_dir_all(&path);
            }
        } else if is_legacy_bake_file(&path) {
            let _ = fs::remove_file(&path);
        }
    }
}

fn is_legacy_bake_file(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    name.ends_with(".meta.json")
        || name.ends_with(".height.bin")
        || name.ends_with(".strata.json")
        || (name.contains(".aux.") && name.ends_with(".bin"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field_data::{keys, AuxMaps};
    use crate::heightfield::HeightfieldMetrics;
    use std::collections::HashMap;

    fn scratch_dir(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("terra_smart_cache_{tag}_{}", uuid::Uuid::new_v4()))
    }

    #[test]
    fn spill_and_reload_roundtrip() {
        let dir = scratch_dir("test");
        let cache = DiskSmartCache::new(&dir);
        let metrics = HeightfieldMetrics::new(8, 8, 80.0, 80.0);
        let mut height = Heightfield::zeros(metrics);
        height.set(3, 4, 12.5);
        let mut aux = HashMap::new();
        let mut flow = MaskField::zeros(metrics);
        flow.set(1, 1, 0.75);
        aux.insert("flow_acc".into(), flow);

        let id = LayerId::new();
        let output = CachedOutput {
            height,
            generation: 7,
            dirty: false,
            aux,
            strata: None,
        };
        cache.spill(id, &output).unwrap();
        let loaded = cache.load(id, metrics).unwrap().expect("disk hit");
        assert!((loaded.height.get(3, 4) - 12.5).abs() < 1e-5);
        assert!((loaded.aux["flow_acc"].get(1, 1) - 0.75).abs() < 1e-5);
        assert!(!loaded.dirty);
        assert_eq!(loaded.generation, 7);
        cache.clear_all();
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn layered_inventories_roundtrip_with_canonical_cache_keys() {
        let dir = scratch_dir("layers");
        let cache = DiskSmartCache::new(&dir);
        let metrics = HeightfieldMetrics::new(4, 4, 40.0, 40.0);
        let mut maps = AuxMaps::new();
        maps.insert(
            keys::BEDROCK_HEIGHT,
            MaskField::from_raw(metrics, &[6.0; 16]),
        );
        maps.insert(keys::DEBRIS_DEPTH, MaskField::from_raw(metrics, &[2.0; 16]));
        maps.insert(
            keys::SEDIMENT_DEPTH,
            MaskField::from_raw(metrics, &[3.0; 16]),
        );
        let aux = maps.to_hashmap();
        assert!(aux.contains_key(keys::SEDIMENT_THICKNESS));
        assert!(!aux.contains_key(keys::SEDIMENT_DEPTH));
        assert!(!aux.contains_key(keys::LOOSE_SEDIMENT));

        let id = LayerId::new();
        cache
            .spill(
                id,
                &CachedOutput {
                    height: Heightfield::filled(metrics, 11.0),
                    generation: 9,
                    dirty: false,
                    aux,
                    strata: None,
                },
            )
            .unwrap();
        let loaded = cache.load(id, metrics).unwrap().expect("disk hit");
        let restored = AuxMaps::from_hashmap(&loaded.aux);
        assert_eq!(restored.bedrock_height.as_ref().unwrap().get(0, 0), 6.0);
        assert_eq!(restored.get(keys::DEBRIS_DEPTH).unwrap().get(0, 0), 2.0);
        assert_eq!(restored.sediment_thickness.as_ref().unwrap().get(0, 0), 3.0);

        cache.clear_all();
        let _ = fs::remove_dir_all(&dir);
    }

    /// B1-D8 revert check: a structurally plausible bake (valid magic + header)
    /// whose payload is short must load as a miss, never as a clean hit.
    #[test]
    fn torn_bake_with_valid_header_is_a_miss() {
        let dir = scratch_dir("torn");
        let cache = DiskSmartCache::new(&dir);
        let metrics = HeightfieldMetrics::new(8, 8, 80.0, 80.0);
        let id = LayerId::new();

        let header = BakeHeader {
            version: VERSION,
            metrics,
            generation: 1,
            aux_names: Vec::new(),
            strata: None,
        };
        let header_json = serde_json::to_vec(&header).unwrap();
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&(header_json.len() as u32).to_le_bytes());
        buf.extend_from_slice(&header_json);
        // Header promises 8*8 f32 = 256 payload bytes; write far fewer.
        buf.extend_from_slice(&[0u8; 64]);
        fs::write(cache.bake_path(id), &buf).unwrap();

        assert!(
            cache.load(id, metrics).unwrap().is_none(),
            "a short payload must be a miss"
        );
        assert!(
            !cache.probe(id, metrics),
            "probe must reject the torn file too"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// B1-D8 revert check: two owned-instance stores never share a directory, so a
    /// spill in one is invisible to the other for the same `LayerId`.
    #[test]
    fn instance_roots_are_isolated() {
        let a = DiskSmartCache::owned_instance();
        let b = DiskSmartCache::owned_instance();
        assert_ne!(a.root(), b.root());

        let metrics = HeightfieldMetrics::new(4, 4, 40.0, 40.0);
        let id = LayerId::new();
        a.spill(
            id,
            &CachedOutput {
                height: Heightfield::filled(metrics, 5.0),
                generation: 1,
                dirty: false,
                aux: HashMap::new(),
                strata: None,
            },
        )
        .unwrap();

        assert!(a.load(id, metrics).unwrap().is_some());
        assert!(
            b.load(id, metrics).unwrap().is_none(),
            "a sibling instance must not see another instance's bake"
        );
    }

    /// B1-D8 revert check: an owned-instance root is removed when the store drops.
    #[test]
    fn dropping_owned_instance_removes_root() {
        let metrics = HeightfieldMetrics::new(4, 4, 40.0, 40.0);
        let id = LayerId::new();
        let root;
        {
            let cache = DiskSmartCache::owned_instance();
            root = cache.root().to_path_buf();
            cache
                .spill(
                    id,
                    &CachedOutput {
                        height: Heightfield::filled(metrics, 5.0),
                        generation: 1,
                        dirty: false,
                        aux: HashMap::new(),
                        strata: None,
                    },
                )
                .unwrap();
            assert!(root.exists());
        }
        assert!(!root.exists(), "owned root should be swept on drop");
    }
}
