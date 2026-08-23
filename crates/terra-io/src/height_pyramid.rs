use crate::IoError;
use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use terra_core::{
    conservative_geometric_errors, measure_tile_geometric_error, FieldId, PyramidConfig,
    TerrainPyramid, TerrainTileKey, TileId,
};

const FORMAT: &str = "terra.height-pyramid";
const VERSION: u32 = 1;
const POINTER_FILE: &str = "height-pyramid.current";
const PACKAGES_DIR: &str = "packages";
static STAGING_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn bounded_key(pyramid: &TerrainPyramid, key: &TerrainTileKey) -> Option<(u8, TileId)> {
    pyramid.level_and_tile(key.address)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct HeightPyramidWorld {
    pub size_x: f32,
    pub size_z: f32,
    pub horizontal_unit: String,
    pub vertical_unit: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct HeightPyramidEncoding {
    pub sample: String,
    pub byte_order: String,
    pub row_order: String,
    pub sample_location: String,
    pub stored_halo: u32,
    pub world_edge: String,
    pub unused_texels: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct HeightPyramidLevelManifest {
    pub level: u8,
    pub width_samples: u32,
    pub height_samples: u32,
    pub tiles_x: u32,
    pub tiles_z: u32,
    pub spacing_x: f32,
    pub spacing_z: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct HeightPyramidTileManifest {
    pub level: u8,
    pub tx: u32,
    pub tz: u32,
    pub origin_x: u32,
    pub origin_z: u32,
    pub interior_width: u32,
    pub interior_height: u32,
    pub world_min_x: f32,
    pub world_min_z: f32,
    pub world_max_x: f32,
    pub world_max_z: f32,
    pub packed_width: u32,
    pub packed_height: u32,
    pub valid_width: u32,
    pub valid_height: u32,
    pub byte_length: u64,
    pub payload: String,
    pub payload_hash: String,
    pub local_geometric_error_m: f32,
    pub geometric_error_m: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct HeightPyramidManifest {
    pub format: String,
    pub version: u32,
    pub content_id: String,
    pub world: HeightPyramidWorld,
    pub encoding: HeightPyramidEncoding,
    pub tile_size: u32,
    pub levels: Vec<HeightPyramidLevelManifest>,
    pub tiles: Vec<HeightPyramidTileManifest>,
}

#[derive(Debug, Clone)]
pub struct HeightPyramidPackageResult {
    pub package_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub content_id: String,
    pub tile_count: usize,
}

/// Incremental deterministic package writer. It retains only the previous level
/// of packed pages so child errors can be measured without reconstructing a
/// complete finest-level heightfield.
pub struct HeightPyramidPackageBuilder {
    root: PathBuf,
    staging: PathBuf,
    pyramid: TerrainPyramid,
    manifest: HeightPyramidManifest,
    next_index: usize,
    retained_level: Option<u8>,
    retained_tiles: HashMap<TileId, Vec<f32>>,
    current_level: Option<u8>,
    current_tiles: HashMap<TileId, Vec<f32>>,
    committed: bool,
}

impl HeightPyramidPackageBuilder {
    pub fn new(root: impl Into<PathBuf>, pyramid: TerrainPyramid) -> Result<Self, IoError> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        let sequence = STAGING_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let staging = root.join(format!(
            ".terra-height-pyramid-{}-{sequence}.partial",
            std::process::id()
        ));
        if staging.exists() {
            std::fs::remove_dir_all(&staging)?;
        }
        std::fs::create_dir_all(staging.join("height"))?;
        let levels = pyramid
            .levels()
            .iter()
            .map(|level| {
                let metrics = pyramid
                    .level_metrics(level.index)
                    .expect("pyramid owns valid level metadata");
                HeightPyramidLevelManifest {
                    level: level.index,
                    width_samples: metrics.width,
                    height_samples: metrics.height,
                    tiles_x: metrics.tiles_x(),
                    tiles_z: metrics.tiles_z(),
                    spacing_x: metrics.dx(),
                    spacing_z: metrics.dz(),
                }
            })
            .collect();
        let page_extent = pyramid.config.tile_size + pyramid.config.halo * 2;
        let manifest = HeightPyramidManifest {
            format: FORMAT.into(),
            version: VERSION,
            content_id: String::new(),
            world: HeightPyramidWorld {
                size_x: pyramid.config.world_size_x,
                size_z: pyramid.config.world_size_z,
                horizontal_unit: "metre".into(),
                vertical_unit: "metre".into(),
            },
            encoding: HeightPyramidEncoding {
                sample: "ieee754-f32".into(),
                byte_order: "little-endian".into(),
                row_order: "z-major-x-minor".into(),
                sample_location: "normalized-cell-center".into(),
                stored_halo: pyramid.config.halo,
                world_edge: "clamp".into(),
                unused_texels: "zero".into(),
            },
            tile_size: pyramid.config.tile_size,
            levels,
            tiles: Vec::with_capacity(pyramid.metadata_len() as usize),
        };
        debug_assert!(page_extent > 0);
        Ok(Self {
            root,
            staging,
            pyramid,
            manifest,
            next_index: 0,
            retained_level: None,
            retained_tiles: HashMap::new(),
            current_level: None,
            current_tiles: HashMap::new(),
            committed: false,
        })
    }

    pub fn expected_tile(&self) -> Option<TerrainTileKey> {
        self.pyramid.height_tiles().nth(self.next_index)
    }

    pub fn progress(&self) -> f32 {
        self.next_index as f32 / self.pyramid.metadata_len().max(1) as f32
    }

    pub fn write_tile(&mut self, key: &TerrainTileKey, packed: &[f32]) -> Result<(), IoError> {
        let expected = self
            .expected_tile()
            .ok_or_else(|| IoError::Msg("height pyramid already has complete coverage".into()))?;
        if &expected != key || key.field != FieldId::Height || key.layer.is_some() {
            return Err(IoError::Msg(format!(
                "out-of-order height tile: expected {expected:?}, got {key:?}"
            )));
        }
        let page_extent = self.pyramid.config.tile_size + self.pyramid.config.halo * 2;
        let expected_len = page_extent as usize * page_extent as usize;
        if packed.len() != expected_len {
            return Err(IoError::Msg(format!(
                "packed tile has {} samples; expected {expected_len}",
                packed.len()
            )));
        }
        let (level, tile) = bounded_key(&self.pyramid, key)
            .ok_or_else(|| IoError::Msg(format!("invalid pyramid tile {key:?}")))?;
        self.advance_level(level);
        let extent = self
            .pyramid
            .tile_extent(level, tile)
            .ok_or_else(|| IoError::Msg(format!("invalid pyramid tile {key:?}")))?;
        let level_metrics = self
            .pyramid
            .level_metrics(level)
            .expect("valid tile belongs to a level");
        let local_error = if level == 0 {
            0.0
        } else {
            measure_tile_geometric_error(&self.pyramid, key, |level, x, z| {
                if level
                    == self
                        .pyramid
                        .topology()
                        .level_index(key.address)
                        .unwrap_or(u8::MAX)
                {
                    packed_sample(
                        packed,
                        page_extent,
                        self.pyramid.config.halo,
                        extent.origin_x,
                        extent.origin_z,
                        x,
                        z,
                    )
                } else {
                    retained_sample(
                        &self.pyramid,
                        &self.retained_tiles,
                        level,
                        x,
                        z,
                        page_extent,
                    )
                }
            })
            .ok_or_else(|| IoError::Msg(format!("cannot measure geometric error for {key:?}")))?
        };
        let bytes = floats_to_le_bytes(packed);
        let payload_hash = blake3::hash(&bytes).to_hex().to_string();
        let relative = format!(
            "height/l{:02}/{:06}_{:06}.{}.r32",
            level, tile.tx, tile.tz, payload_hash
        );
        let path = self
            .staging
            .join(relative.replace('/', std::path::MAIN_SEPARATOR_STR));
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::File::create(&path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&bytes)?;
        writer.flush()?;
        self.manifest.tiles.push(HeightPyramidTileManifest {
            level,
            tx: tile.tx,
            tz: tile.tz,
            origin_x: extent.origin_x,
            origin_z: extent.origin_z,
            interior_width: extent.width,
            interior_height: extent.height,
            world_min_x: extent.origin_x as f32 / level_metrics.width as f32
                * level_metrics.world_size_x,
            world_min_z: extent.origin_z as f32 / level_metrics.height as f32
                * level_metrics.world_size_z,
            world_max_x: (extent.origin_x + extent.width) as f32 / level_metrics.width as f32
                * level_metrics.world_size_x,
            world_max_z: (extent.origin_z + extent.height) as f32 / level_metrics.height as f32
                * level_metrics.world_size_z,
            packed_width: page_extent,
            packed_height: page_extent,
            valid_width: extent.width + self.pyramid.config.halo * 2,
            valid_height: extent.height + self.pyramid.config.halo * 2,
            byte_length: bytes.len() as u64,
            payload: relative,
            payload_hash,
            local_geometric_error_m: local_error,
            geometric_error_m: local_error,
        });
        if level < self.pyramid.max_level() {
            self.current_tiles.insert(tile, packed.to_vec());
        }
        self.next_index += 1;
        Ok(())
    }

    pub fn finish(mut self) -> Result<HeightPyramidPackageResult, IoError> {
        if self.next_index != self.pyramid.metadata_len() as usize {
            return Err(IoError::Msg(format!(
                "incomplete height pyramid: wrote {} of {} tiles",
                self.next_index,
                self.pyramid.metadata_len()
            )));
        }
        let local = self
            .manifest
            .tiles
            .iter()
            .map(|tile| tile.local_geometric_error_m)
            .collect::<Vec<_>>();
        let conservative = conservative_geometric_errors(&self.pyramid, &local)
            .map_err(|error| IoError::Msg(error.to_string()))?;
        for (tile, error) in self.manifest.tiles.iter_mut().zip(conservative) {
            tile.geometric_error_m = error;
        }
        let identity_bytes = serde_json::to_vec(&self.manifest)?;
        let content_id = blake3::hash(&identity_bytes).to_hex().to_string();
        self.manifest.content_id.clone_from(&content_id);
        let manifest_bytes = serde_json::to_vec_pretty(&self.manifest)?;
        let manifest_path = self.staging.join("manifest.json");
        let mut file = std::fs::File::create(&manifest_path)?;
        file.write_all(&manifest_bytes)?;
        file.sync_all()?;
        drop(file);

        let packages = self.root.join(PACKAGES_DIR);
        std::fs::create_dir_all(&packages)?;
        let package_dir = packages.join(&content_id);
        if package_dir.exists() {
            let existing = std::fs::read(package_dir.join("manifest.json"))?;
            if existing != manifest_bytes {
                return Err(IoError::Msg(format!(
                    "content-address collision for {content_id}"
                )));
            }
            std::fs::remove_dir_all(&self.staging)?;
        } else {
            std::fs::rename(&self.staging, &package_dir)?;
        }
        let pointer_tmp = self.root.join(format!("{POINTER_FILE}.tmp"));
        {
            let mut pointer = std::fs::File::create(&pointer_tmp)?;
            writeln!(pointer, "{content_id}")?;
            pointer.sync_all()?;
        }
        let pointer_path = self.root.join(POINTER_FILE);
        if pointer_path.exists() {
            std::fs::remove_file(&pointer_path)?;
        }
        std::fs::rename(pointer_tmp, pointer_path)?;
        self.committed = true;
        Ok(HeightPyramidPackageResult {
            manifest_path: package_dir.join("manifest.json"),
            package_dir,
            content_id,
            tile_count: self.manifest.tiles.len(),
        })
    }

    fn advance_level(&mut self, level: u8) {
        if self.current_level == Some(level) {
            return;
        }
        if let Some(previous) = self.current_level.replace(level) {
            self.retained_level = Some(previous);
            self.retained_tiles = std::mem::take(&mut self.current_tiles);
        }
    }
}

impl Drop for HeightPyramidPackageBuilder {
    fn drop(&mut self) {
        if !self.committed && self.staging.exists() {
            let _ = std::fs::remove_dir_all(&self.staging);
        }
    }
}

#[derive(Debug, Clone)]
pub struct HeightPyramidPackage {
    root: PathBuf,
    pub manifest: HeightPyramidManifest,
}

impl HeightPyramidPackage {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, IoError> {
        let path = path.as_ref();
        let root = if path.join(POINTER_FILE).is_file() {
            let content_id = std::fs::read_to_string(path.join(POINTER_FILE))?;
            let content_id = content_id.trim();
            if content_id.len() != 64 || !content_id.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err(IoError::Msg(
                    "invalid height-pyramid current pointer".into(),
                ));
            }
            path.join(PACKAGES_DIR).join(content_id)
        } else {
            path.to_path_buf()
        };
        let manifest: HeightPyramidManifest =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        let package = Self { root, manifest };
        package.validate()?;
        Ok(package)
    }

    pub fn read_tile(&self, level: u8, tile: TileId) -> Result<Vec<f32>, IoError> {
        let entry = self
            .tile_entry(level, tile)
            .ok_or_else(|| IoError::Msg(format!("missing tile l{level}/{tile:?}")))?;
        let path = safe_payload_path(&self.root, &entry.payload)?;
        let mut reader = BufReader::new(std::fs::File::open(path)?);
        let mut bytes = Vec::with_capacity(entry.byte_length as usize);
        reader.read_to_end(&mut bytes)?;
        if bytes.len() as u64 != entry.byte_length {
            return Err(IoError::Msg(format!(
                "wrong payload length for {}",
                entry.payload
            )));
        }
        if blake3::hash(&bytes).to_hex().as_str() != entry.payload_hash {
            return Err(IoError::Msg(format!(
                "payload hash mismatch for {}",
                entry.payload
            )));
        }
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect())
    }

    pub fn sample(&self, level: u8, x: u32, z: u32) -> Result<f32, IoError> {
        let descriptor = self.descriptor()?;
        let metrics = descriptor
            .level_metrics(level)
            .ok_or_else(|| IoError::Msg(format!("invalid level {level}")))?;
        if x >= metrics.width || z >= metrics.height {
            return Err(IoError::Msg(format!(
                "sample ({x},{z}) outside level {level}"
            )));
        }
        let tile = TileId {
            tx: x / metrics.tile_size,
            tz: z / metrics.tile_size,
        };
        let entry = self.tile_entry(level, tile).expect("validated coverage");
        let data = self.read_tile(level, tile)?;
        let local_x = x - entry.origin_x + self.manifest.encoding.stored_halo;
        let local_z = z - entry.origin_z + self.manifest.encoding.stored_halo;
        Ok(data[(local_z * entry.packed_width + local_x) as usize])
    }

    /// Resolve an exact sample when its tile is available, otherwise walk to the
    /// finest available ancestor using normalized cell-center coordinates.
    pub fn sample_exact_or_ancestor(
        &self,
        requested_level: u8,
        x: u32,
        z: u32,
        mut available: impl FnMut(u8, TileId) -> bool,
    ) -> Result<(u8, f32), IoError> {
        let descriptor = self.descriptor()?;
        let requested = descriptor
            .level_metrics(requested_level)
            .ok_or_else(|| IoError::Msg(format!("invalid level {requested_level}")))?;
        if x >= requested.width || z >= requested.height {
            return Err(IoError::Msg(format!(
                "sample ({x},{z}) outside level {requested_level}"
            )));
        }
        let uv_x = (x as f32 + 0.5) / requested.width as f32;
        let uv_z = (z as f32 + 0.5) / requested.height as f32;
        for level in (0..=requested_level).rev() {
            let metrics = descriptor
                .level_metrics(level)
                .expect("valid ancestor level");
            let sample_x = (uv_x * metrics.width as f32)
                .floor()
                .clamp(0.0, metrics.width.saturating_sub(1) as f32)
                as u32;
            let sample_z = (uv_z * metrics.height as f32)
                .floor()
                .clamp(0.0, metrics.height.saturating_sub(1) as f32)
                as u32;
            let tile = TileId {
                tx: sample_x / metrics.tile_size,
                tz: sample_z / metrics.tile_size,
            };
            if available(level, tile) {
                return Ok((level, self.sample(level, sample_x, sample_z)?));
            }
        }
        Err(IoError::Msg(
            "no exact tile or ancestor is available".into(),
        ))
    }

    pub fn reconstruct_region(
        &self,
        level: u8,
        origin_x: u32,
        origin_z: u32,
        width: u32,
        height: u32,
    ) -> Result<Vec<f32>, IoError> {
        let mut values = Vec::with_capacity(width as usize * height as usize);
        for z in origin_z..origin_z.saturating_add(height) {
            for x in origin_x..origin_x.saturating_add(width) {
                values.push(self.sample(level, x, z)?);
            }
        }
        Ok(values)
    }

    pub fn tile_entry(&self, level: u8, tile: TileId) -> Option<&HeightPyramidTileManifest> {
        self.manifest
            .tiles
            .iter()
            .find(|entry| entry.level == level && entry.tx == tile.tx && entry.tz == tile.tz)
    }

    pub fn descriptor(&self) -> Result<TerrainPyramid, IoError> {
        let target = self
            .manifest
            .levels
            .last()
            .ok_or_else(|| IoError::Msg("pyramid has no levels".into()))?
            .width_samples;
        let mut config = PyramidConfig::new(
            target,
            self.manifest.world.size_x,
            self.manifest.world.size_z,
        );
        config.tile_size = self.manifest.tile_size;
        config.halo = self.manifest.encoding.stored_halo;
        TerrainPyramid::try_new(config)
            .map_err(|error| IoError::Msg(format!("invalid height-pyramid geometry: {error}")))
    }

    pub fn validate(&self) -> Result<(), IoError> {
        if self.manifest.format != FORMAT || self.manifest.version != VERSION {
            return Err(IoError::Msg("unsupported height-pyramid manifest".into()));
        }
        let pyramid = self.descriptor()?;
        if self.manifest.levels.len() != pyramid.levels().len()
            || self.manifest.tiles.len() != pyramid.metadata_len() as usize
        {
            return Err(IoError::Msg("incomplete pyramid metadata".into()));
        }
        for (actual, expected) in self.manifest.levels.iter().zip(pyramid.levels()) {
            let metrics = pyramid
                .level_metrics(expected.index)
                .expect("descriptor owns valid level");
            if actual.level != expected.index
                || actual.width_samples != metrics.width
                || actual.height_samples != metrics.height
                || actual.tiles_x != metrics.tiles_x()
                || actual.tiles_z != metrics.tiles_z()
            {
                return Err(IoError::Msg(format!(
                    "invalid level metadata for level {}",
                    expected.index
                )));
            }
        }
        for (index, key) in pyramid.height_tiles().enumerate() {
            let entry = self
                .manifest
                .tiles
                .get(index)
                .ok_or_else(|| IoError::Msg("missing ordered tile entry".into()))?;
            let (level, tile) = bounded_key(&pyramid, &key).expect("valid bounded key");
            let extent = pyramid.tile_extent(level, tile).expect("valid key");
            if entry.level != level
                || entry.tx != tile.tx
                || entry.tz != tile.tz
                || entry.origin_x != extent.origin_x
                || entry.origin_z != extent.origin_z
                || entry.interior_width != extent.width
                || entry.interior_height != extent.height
                || !entry.world_min_x.is_finite()
                || !entry.world_min_z.is_finite()
                || !entry.world_max_x.is_finite()
                || !entry.world_max_z.is_finite()
                || entry.world_min_x > entry.world_max_x
                || entry.world_min_z > entry.world_max_z
                || !entry.local_geometric_error_m.is_finite()
                || entry.local_geometric_error_m < 0.0
                || !entry.geometric_error_m.is_finite()
                || entry.geometric_error_m < 0.0
            {
                return Err(IoError::Msg(format!("invalid tile entry at index {index}")));
            }
            let path = safe_payload_path(&self.root, &entry.payload)?;
            let bytes = std::fs::read(path)?;
            if bytes.len() as u64 != entry.byte_length {
                return Err(IoError::Msg(format!(
                    "invalid payload size for {}",
                    entry.payload
                )));
            }
            if blake3::hash(&bytes).to_hex().as_str() != entry.payload_hash {
                return Err(IoError::Msg(format!(
                    "payload hash mismatch for {}",
                    entry.payload
                )));
            }
        }
        let mut identity = self.manifest.clone();
        let expected = std::mem::take(&mut identity.content_id);
        let actual = blake3::hash(&serde_json::to_vec(&identity)?)
            .to_hex()
            .to_string();
        if expected != actual {
            return Err(IoError::Msg("package content identity mismatch".into()));
        }
        Ok(())
    }
}

fn retained_sample(
    pyramid: &TerrainPyramid,
    pages: &HashMap<TileId, Vec<f32>>,
    level: u8,
    x: u32,
    z: u32,
    page_extent: u32,
) -> f32 {
    let metrics = pyramid.level_metrics(level).expect("valid retained level");
    let tile = TileId {
        tx: x / metrics.tile_size,
        tz: z / metrics.tile_size,
    };
    let extent = pyramid
        .tile_extent(level, tile)
        .expect("valid retained tile");
    packed_sample(
        pages.get(&tile).expect("complete retained parent level"),
        page_extent,
        pyramid.config.halo,
        extent.origin_x,
        extent.origin_z,
        x,
        z,
    )
}

fn packed_sample(
    data: &[f32],
    stride: u32,
    halo: u32,
    origin_x: u32,
    origin_z: u32,
    x: u32,
    z: u32,
) -> f32 {
    let local_x = x - origin_x + halo;
    let local_z = z - origin_z + halo;
    data[(local_z * stride + local_x) as usize]
}

fn floats_to_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn safe_payload_path(root: &Path, relative: &str) -> Result<PathBuf, IoError> {
    let relative = Path::new(relative);
    if relative.is_absolute()
        || relative
            .components()
            .any(|part| !matches!(part, std::path::Component::Normal(_)))
    {
        return Err(IoError::Msg(format!(
            "unsafe payload path: {}",
            relative.display()
        )));
    }
    Ok(root.join(relative))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packed_page(pyramid: &TerrainPyramid, key: &TerrainTileKey) -> Vec<f32> {
        let (level, tile) = bounded_key(pyramid, key).unwrap();
        let extent = pyramid.tile_extent(level, tile).unwrap();
        let metrics = pyramid.level_metrics(level).unwrap();
        let halo = pyramid.config.halo;
        let stride = pyramid.config.tile_size + halo * 2;
        let mut page = vec![0.0; (stride * stride) as usize];
        for pz in 0..extent.height + halo * 2 {
            for px in 0..extent.width + halo * 2 {
                let x = (extent.origin_x as i64 + px as i64 - halo as i64)
                    .clamp(0, metrics.width as i64 - 1) as u32;
                let z = (extent.origin_z as i64 + pz as i64 - halo as i64)
                    .clamp(0, metrics.height as i64 - 1) as u32;
                page[(pz * stride + px) as usize] = x as f32 * 0.25 + z as f32 * 0.5;
            }
        }
        page
    }

    #[test]
    fn package_round_trip_is_deterministic_and_reconstructs_regions() {
        let root = std::env::temp_dir().join(format!(
            "terra-height-pyramid-package-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let mut config = PyramidConfig::new(17, 170.0, 90.0);
        config.tile_size = 8;
        config.halo = 2;
        let pyramid = TerrainPyramid::new(config);
        let mut builder = HeightPyramidPackageBuilder::new(&root, pyramid.clone()).unwrap();
        for key in pyramid.height_tiles() {
            builder
                .write_tile(&key, &packed_page(&pyramid, &key))
                .unwrap();
        }
        let first = builder.finish().unwrap();
        let first_manifest = std::fs::read(&first.manifest_path).unwrap();
        let package = HeightPyramidPackage::open(&root).unwrap();
        assert_eq!(
            package.manifest.tiles.len(),
            pyramid.metadata_len() as usize
        );
        assert_eq!(
            package
                .reconstruct_region(pyramid.max_level(), 6, 6, 5, 4)
                .unwrap(),
            (6..10)
                .flat_map(|z| (6..11).map(move |x| x as f32 * 0.25 + z as f32 * 0.5))
                .collect::<Vec<_>>()
        );
        let finest = pyramid.max_level();
        let (resolved, _) = package
            .sample_exact_or_ancestor(finest, 12, 12, |level, _| level < finest)
            .unwrap();
        assert_eq!(resolved, finest - 1);
        let left = package.read_tile(finest, TileId { tx: 0, tz: 0 }).unwrap();
        let right = package.read_tile(finest, TileId { tx: 1, tz: 0 }).unwrap();
        let stride = pyramid.config.tile_size + pyramid.config.halo * 2;
        for global_z in 1..7 {
            for global_x in 6..10 {
                let left_x = global_x + pyramid.config.halo;
                let right_x = global_x + pyramid.config.halo - pyramid.config.tile_size;
                let row = global_z + pyramid.config.halo;
                assert_eq!(
                    left[(row * stride + left_x) as usize],
                    right[(row * stride + right_x) as usize]
                );
            }
        }

        let mut repeat = HeightPyramidPackageBuilder::new(&root, pyramid.clone()).unwrap();
        for key in pyramid.height_tiles() {
            repeat
                .write_tile(&key, &packed_page(&pyramid, &key))
                .unwrap();
        }
        let second = repeat.finish().unwrap();
        assert_eq!(first.content_id, second.content_id);
        assert_eq!(first_manifest, std::fs::read(second.manifest_path).unwrap());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn dropping_an_incomplete_builder_never_publishes() {
        let root = std::env::temp_dir().join(format!(
            "terra-height-pyramid-cancel-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let pyramid = TerrainPyramid::new(PyramidConfig::new(8, 8.0, 8.0));
        {
            let mut builder = HeightPyramidPackageBuilder::new(&root, pyramid.clone()).unwrap();
            let key = builder.expected_tile().unwrap();
            builder
                .write_tile(&key, &packed_page(&pyramid, &key))
                .unwrap();
        }
        assert!(!root.join(POINTER_FILE).exists());
        assert!(std::fs::read_dir(&root).unwrap().all(|entry| !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains("partial")));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn malformed_package_geometry_is_a_load_error_not_a_panic() {
        let root = std::env::temp_dir().join(format!(
            "terra-height-pyramid-invalid-geometry-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let pyramid = TerrainPyramid::new(PyramidConfig::new(2, 8.0, 8.0));
        let mut builder = HeightPyramidPackageBuilder::new(&root, pyramid.clone()).unwrap();
        for key in pyramid.height_tiles() {
            builder
                .write_tile(&key, &packed_page(&pyramid, &key))
                .unwrap();
        }
        let published = builder.finish().unwrap();
        let mut manifest: HeightPyramidManifest =
            serde_json::from_slice(&std::fs::read(&published.manifest_path).unwrap()).unwrap();
        manifest.tile_size = 0;
        std::fs::write(
            &published.manifest_path,
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let result = HeightPyramidPackage::open(published.manifest_path.parent().unwrap());
        assert!(
            matches!(result, Err(IoError::Msg(message)) if message.contains("invalid height-pyramid geometry"))
        );
        let _ = std::fs::remove_dir_all(root);
    }
}
