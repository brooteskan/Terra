//! Disk spill for baked layer checkpoints (Gaea-style Smart Cache).
//!
//! When a layer is marked `cached`, its height + aux maps can be written to disk so
//! memory can be reclaimed and later rebuilds skip re-running the processor.

use super::cache::CachedOutput;
use super::EvalError;
use crate::heightfield::{Heightfield, HeightfieldMetrics};
use crate::layer::LayerId;
use crate::mask::MaskField;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 4] = b"TCS1";
// Bump whenever terrain processors change in a way that makes baked outputs stale.
// Version 2 invalidates checkpoints produced by the pre-fidelity procedural generators.
// Version 3 invalidates procedural outputs created before 64-bit seed canonicalization.
// Version 4 invalidates checkpoints where loose_sediment could contain coarse debris
// while sediment_depth held the actual fine-sediment inventory.
const VERSION: u32 = 4;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DiskMeta {
    version: u32,
    metrics: HeightfieldMetrics,
    generation: u64,
    aux_names: Vec<String>,
}

/// On-disk bake store keyed by [`LayerId`].
#[derive(Debug, Clone)]
pub struct DiskSmartCache {
    root: PathBuf,
}

impl DiskSmartCache {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let _ = fs::create_dir_all(&root);
        Self { root }
    }

    /// `temp_dir()/terra_smart_cache`
    pub fn default_location() -> PathBuf {
        std::env::temp_dir().join("terra_smart_cache")
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn stem(&self, id: LayerId) -> PathBuf {
        self.root.join(id.0.to_string())
    }

    fn meta_path(&self, id: LayerId) -> PathBuf {
        self.stem(id).with_extension("meta.json")
    }

    fn height_path(&self, id: LayerId) -> PathBuf {
        self.stem(id).with_extension("height.bin")
    }

    fn aux_path(&self, id: LayerId, name: &str) -> PathBuf {
        // Sanitize aux keys for filesystem safety.
        let safe: String = name
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        self.stem(id).with_extension(format!("aux.{safe}.bin"))
    }

    pub fn invalidate(&self, id: LayerId) {
        let _ = fs::remove_file(self.meta_path(id));
        let _ = fs::remove_file(self.height_path(id));
        // Best-effort wipe of aux blobs for this layer.
        if let Ok(entries) = fs::read_dir(&self.root) {
            let prefix = format!("{}.aux.", id.0);
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with(&prefix) && name.ends_with(".bin") {
                    let _ = fs::remove_file(entry.path());
                }
            }
        }
    }

    pub fn clear_all(&self) {
        let _ = fs::remove_dir_all(&self.root);
        let _ = fs::create_dir_all(&self.root);
    }

    pub fn spill(&self, id: LayerId, output: &CachedOutput) -> Result<(), EvalError> {
        fs::create_dir_all(&self.root).map_err(|e| EvalError::Io(e.to_string()))?;

        let mut aux_names: Vec<String> = output.aux.keys().cloned().collect();
        aux_names.sort();

        let meta = DiskMeta {
            version: VERSION,
            metrics: output.height.metrics,
            generation: output.generation,
            aux_names: aux_names.clone(),
        };
        let meta_json =
            serde_json::to_vec_pretty(&meta).map_err(|e| EvalError::Io(e.to_string()))?;
        fs::write(self.meta_path(id), meta_json).map_err(|e| EvalError::Io(e.to_string()))?;

        write_f32_blob(&self.height_path(id), &output.height.to_dense())?;

        for name in &aux_names {
            if let Some(field) = output.aux.get(name) {
                write_f32_blob(&self.aux_path(id, name), field.data())?;
            }
        }
        if let Some(strata) = &output.strata {
            let path = self.stem(id).with_extension("strata.json");
            let json =
                serde_json::to_vec_pretty(strata).map_err(|e| EvalError::Io(e.to_string()))?;
            fs::write(path, json).map_err(|e| EvalError::Io(e.to_string()))?;
        }
        Ok(())
    }

    /// Load a baked checkpoint if present and metrics match.
    pub fn load(
        &self,
        id: LayerId,
        expected: HeightfieldMetrics,
    ) -> Result<Option<CachedOutput>, EvalError> {
        let meta_path = self.meta_path(id);
        if !meta_path.exists() {
            return Ok(None);
        }
        let meta_bytes = fs::read(&meta_path).map_err(|e| EvalError::Io(e.to_string()))?;
        let meta: DiskMeta =
            serde_json::from_slice(&meta_bytes).map_err(|e| EvalError::Io(e.to_string()))?;
        if meta.version != VERSION {
            return Ok(None);
        }
        if meta.metrics.width != expected.width
            || meta.metrics.height != expected.height
            || (meta.metrics.world_size_x - expected.world_size_x).abs() > 1e-3
            || (meta.metrics.world_size_z - expected.world_size_z).abs() > 1e-3
        {
            return Ok(None);
        }

        let height_data = read_f32_blob(&self.height_path(id))?;
        let expected_len = (expected.width * expected.height) as usize;
        if height_data.len() != expected_len {
            return Ok(None);
        }
        let height = Heightfield::from_dense(expected, &height_data);

        let mut aux = HashMap::new();
        for name in &meta.aux_names {
            let path = self.aux_path(id, name);
            if !path.exists() {
                continue;
            }
            let data = read_f32_blob(&path)?;
            if data.len() != expected_len {
                continue;
            }
            let mut field = MaskField::zeros(expected);
            field.data_mut().copy_from_slice(&data);
            aux.insert(name.clone(), field);
        }

        Ok(Some(CachedOutput {
            height,
            generation: meta.generation,
            dirty: false,
            aux,
            strata: {
                let path = self.stem(id).with_extension("strata.json");
                if path.exists() {
                    fs::read(&path)
                        .ok()
                        .and_then(|b| serde_json::from_slice(&b).ok())
                } else {
                    None
                }
            },
        }))
    }
}

fn write_f32_blob(path: &Path, data: &[f32]) -> Result<(), EvalError> {
    let mut file = File::create(path).map_err(|e| EvalError::Io(e.to_string()))?;
    file.write_all(MAGIC)
        .map_err(|e| EvalError::Io(e.to_string()))?;
    file.write_all(&(data.len() as u32).to_le_bytes())
        .map_err(|e| EvalError::Io(e.to_string()))?;
    for v in data {
        file.write_all(&v.to_bits().to_le_bytes())
            .map_err(|e| EvalError::Io(e.to_string()))?;
    }
    Ok(())
}

fn read_f32_blob(path: &Path) -> Result<Vec<f32>, EvalError> {
    let mut file = File::open(path).map_err(|e| EvalError::Io(e.to_string()))?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)
        .map_err(|e| EvalError::Io(e.to_string()))?;
    if &magic != MAGIC {
        return Err(EvalError::Io("bad smart-cache magic".into()));
    }
    let mut len_buf = [0u8; 4];
    file.read_exact(&mut len_buf)
        .map_err(|e| EvalError::Io(e.to_string()))?;
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut out = Vec::with_capacity(len);
    let mut word = [0u8; 4];
    for _ in 0..len {
        file.read_exact(&mut word)
            .map_err(|e| EvalError::Io(e.to_string()))?;
        out.push(f32::from_le_bytes(word));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fields::{keys, AuxMaps};
    use crate::heightfield::HeightfieldMetrics;
    use std::collections::HashMap;

    #[test]
    fn spill_and_reload_roundtrip() {
        let dir =
            std::env::temp_dir().join(format!("terra_smart_cache_test_{}", uuid::Uuid::new_v4()));
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
        cache.clear_all();
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn layered_inventories_roundtrip_with_canonical_cache_keys() {
        let dir = std::env::temp_dir().join(format!(
            "terra_smart_cache_layers_test_{}",
            uuid::Uuid::new_v4()
        ));
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
}
