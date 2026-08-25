use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

pub const TERRA_PIPELINE_CACHE_SCHEMA: u32 = 1;
pub const WGPU_PIPELINE_CACHE_COMPAT: &str = "24.0.5";
const MAGIC: &[u8; 8] = b"TERRAPC\0";
const CHECKSUM_BYTES: usize = 32;
const FIXED_HEADER_BYTES: usize = MAGIC.len() + 4 + 4 + 8 + CHECKSUM_BYTES;
const MAX_KEY_BYTES: usize = 4 * 1024;
const MAX_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;
const MAX_FILE_BYTES: u64 = (FIXED_HEADER_BYTES + MAX_KEY_BYTES + MAX_PAYLOAD_BYTES) as u64;
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct PipelineCacheConfig {
    pub directory: PathBuf,
}

impl PipelineCacheConfig {
    pub fn new(directory: PathBuf) -> Self {
        Self { directory }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineCacheBlobStatus {
    Unsupported,
    Miss,
    BlobLoaded,
}

impl PipelineCacheBlobStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unsupported => "unsupported",
            Self::Miss => "miss",
            Self::BlobLoaded => "blob-loaded",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineCacheSaveResult {
    NotAttempted,
    Unsupported,
    NoData,
    Saved,
    Oversized,
    IoError,
}

impl PipelineCacheSaveResult {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::NotAttempted => "not-attempted",
            Self::Unsupported => "unsupported",
            Self::NoData => "no-data",
            Self::Saved => "saved",
            Self::Oversized => "oversized",
            Self::IoError => "io-error",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PipelineCacheReport {
    pub schema: u32,
    pub key: Option<String>,
    pub blob_status: PipelineCacheBlobStatus,
    pub blob_bytes: u64,
    pub detail: String,
    pub save_result: PipelineCacheSaveResult,
    pub saved_bytes: u64,
}

impl Default for PipelineCacheReport {
    fn default() -> Self {
        Self {
            schema: TERRA_PIPELINE_CACHE_SCHEMA,
            key: None,
            blob_status: PipelineCacheBlobStatus::Unsupported,
            blob_bytes: 0,
            detail: "pipeline cache is unsupported".into(),
            save_result: PipelineCacheSaveResult::NotAttempted,
            saved_bytes: 0,
        }
    }
}

#[derive(Debug, Clone)]
struct PersistenceTarget {
    key: String,
    path: PathBuf,
}

/// Pipeline reuse and persistence scoped to one owning GPU context.
pub struct PipelineCacheRegistry {
    driver: Option<wgpu::PipelineCache>,
    persistence: Option<PersistenceTarget>,
    report: Mutex<PipelineCacheReport>,
    render: Mutex<HashMap<(&'static str, wgpu::TextureFormat), wgpu::RenderPipeline>>,
    compute: Mutex<HashMap<&'static str, wgpu::ComputePipeline>>,
}

impl PipelineCacheRegistry {
    /// Create an in-process registry without disk persistence.
    pub fn new(device: &wgpu::Device) -> Self {
        Self::create(device, None, None)
    }

    /// Create a registry whose persistent identity is derived from the selected adapter.
    pub fn persistent(
        device: &wgpu::Device,
        adapter_info: &wgpu::AdapterInfo,
        config: Option<PipelineCacheConfig>,
    ) -> Self {
        Self::create(device, Some(adapter_info), config)
    }

    fn create(
        device: &wgpu::Device,
        adapter_info: Option<&wgpu::AdapterInfo>,
        config: Option<PipelineCacheConfig>,
    ) -> Self {
        let feature_enabled = device.features().contains(wgpu::Features::PIPELINE_CACHE);
        let wgpu_key = adapter_info.and_then(wgpu::util::pipeline_cache_key);
        let identity = adapter_info
            .zip(wgpu_key.as_deref())
            .map(|(info, key)| canonical_key(info.backend, key));

        let mut report = PipelineCacheReport::default();
        report.key = identity.clone();
        let persistence =
            identity
                .as_ref()
                .zip(config.as_ref())
                .map(|(key, config)| PersistenceTarget {
                    key: key.clone(),
                    path: config.directory.join(format!("{key}.bin")),
                });

        let payload = if !feature_enabled || identity.is_none() {
            report.detail = "adapter/backend does not support application pipeline caches".into();
            None
        } else if let Some(target) = persistence.as_ref() {
            match load_payload(target) {
                Ok(Some(bytes)) => {
                    report.blob_status = PipelineCacheBlobStatus::BlobLoaded;
                    report.blob_bytes = bytes.len() as u64;
                    report.detail = "validated Terra cache blob supplied to wgpu".into();
                    Some(bytes)
                }
                Ok(None) => {
                    report.blob_status = PipelineCacheBlobStatus::Miss;
                    report.detail = "cache file is missing".into();
                    None
                }
                Err(detail) => {
                    report.blob_status = PipelineCacheBlobStatus::Miss;
                    report.detail = detail;
                    None
                }
            }
        } else {
            report.blob_status = PipelineCacheBlobStatus::Miss;
            report.detail = "persistent cache directory is unavailable".into();
            None
        };

        let driver = feature_enabled.then(|| {
            // SAFETY: non-empty data reaches this call only after Terra's complete
            // envelope, identity, bounds, and checksum validation succeeds. Terra
            // writes envelopes only around bytes returned by `PipelineCache::get_data`.
            unsafe {
                device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                    label: Some("terra-context-pipeline-cache"),
                    data: payload.as_deref(),
                    fallback: true,
                })
            }
        });

        log::info!(
            "pipeline_cache schema={} key={:?} blob_status={} bytes={} detail={:?}",
            report.schema,
            report.key,
            report.blob_status.as_str(),
            report.blob_bytes,
            report.detail
        );
        Self {
            driver,
            persistence,
            report: Mutex::new(report),
            render: Mutex::new(HashMap::new()),
            compute: Mutex::new(HashMap::new()),
        }
    }

    pub fn driver_cache(&self) -> Option<&wgpu::PipelineCache> {
        self.driver.as_ref()
    }

    pub fn report(&self) -> PipelineCacheReport {
        self.report
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    pub fn save(&self) -> PipelineCacheSaveResult {
        let outcome = self.save_inner();
        let mut report = self
            .report
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        report.save_result = outcome.result.clone();
        report.saved_bytes = outcome.bytes;
        if let Some(detail) = outcome.detail {
            report.detail = detail;
        }
        log::info!(
            "pipeline_cache_save schema={} key={:?} blob_status={} loaded_bytes={} result={} saved_bytes={} detail={:?}",
            report.schema,
            report.key,
            report.blob_status.as_str(),
            report.blob_bytes,
            report.save_result.as_str(),
            report.saved_bytes,
            report.detail
        );
        outcome.result
    }

    fn save_inner(&self) -> SaveOutcome {
        let Some(driver) = self.driver.as_ref() else {
            return SaveOutcome::new(PipelineCacheSaveResult::Unsupported, 0, None);
        };
        let Some(target) = self.persistence.as_ref() else {
            return SaveOutcome::new(
                PipelineCacheSaveResult::Unsupported,
                0,
                Some("persistent cache destination is unavailable".into()),
            );
        };
        let Some(payload) = driver.get_data() else {
            return SaveOutcome::new(
                PipelineCacheSaveResult::NoData,
                0,
                Some("wgpu returned no pipeline cache data".into()),
            );
        };
        if payload.len() > MAX_PAYLOAD_BYTES {
            return SaveOutcome::new(
                PipelineCacheSaveResult::Oversized,
                payload.len() as u64,
                Some(format!(
                    "wgpu cache payload exceeds {} byte limit",
                    MAX_PAYLOAD_BYTES
                )),
            );
        }
        let envelope = encode_envelope(&target.key, &payload);
        match atomic_write(&target.path, &envelope) {
            Ok(()) => SaveOutcome::new(
                PipelineCacheSaveResult::Saved,
                payload.len() as u64,
                Some("pipeline cache saved atomically".into()),
            ),
            Err(error) => SaveOutcome::new(
                PipelineCacheSaveResult::IoError,
                payload.len() as u64,
                Some(format!("pipeline cache save failed: {error}")),
            ),
        }
    }

    pub fn render_pipeline(
        &self,
        label: &'static str,
        format: wgpu::TextureFormat,
        create: impl FnOnce() -> wgpu::RenderPipeline,
    ) -> wgpu::RenderPipeline {
        let mut pipelines = self
            .render
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let key = (label, format);
        pipelines
            .entry(key)
            .or_insert_with(|| {
                terra_telemetry::measure(
                    terra_telemetry::CompilationKind::RenderPipeline,
                    label,
                    create,
                )
            })
            .clone()
    }

    pub fn compute_pipeline(
        &self,
        label: &'static str,
        create: impl FnOnce() -> wgpu::ComputePipeline,
    ) -> wgpu::ComputePipeline {
        let mut pipelines = self
            .compute
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        pipelines
            .entry(label)
            .or_insert_with(|| {
                terra_telemetry::measure(
                    terra_telemetry::CompilationKind::ComputePipeline,
                    label,
                    create,
                )
            })
            .clone()
    }
}

struct SaveOutcome {
    result: PipelineCacheSaveResult,
    bytes: u64,
    detail: Option<String>,
}

impl SaveOutcome {
    fn new(result: PipelineCacheSaveResult, bytes: u64, detail: Option<String>) -> Self {
        Self {
            result,
            bytes,
            detail,
        }
    }
}

fn canonical_key(backend: wgpu::Backend, wgpu_key: &str) -> String {
    format!(
        "terra-pipeline-schema-{}-wgpu-{}-{}-{}",
        TERRA_PIPELINE_CACHE_SCHEMA,
        WGPU_PIPELINE_CACHE_COMPAT,
        backend.to_str(),
        wgpu_key
    )
}

fn checksum(key: &[u8], payload: &[u8]) -> [u8; CHECKSUM_BYTES] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(key);
    hasher.update(payload);
    *hasher.finalize().as_bytes()
}

fn encode_envelope(key: &str, payload: &[u8]) -> Vec<u8> {
    let key = key.as_bytes();
    let mut bytes = Vec::with_capacity(FIXED_HEADER_BYTES + key.len() + payload.len());
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&TERRA_PIPELINE_CACHE_SCHEMA.to_le_bytes());
    bytes.extend_from_slice(&(key.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&checksum(key, payload));
    bytes.extend_from_slice(key);
    bytes.extend_from_slice(payload);
    bytes
}

fn decode_envelope<'a>(expected_key: &str, bytes: &'a [u8]) -> Result<&'a [u8], String> {
    if bytes.len() < FIXED_HEADER_BYTES {
        return Err("cache envelope is truncated".into());
    }
    if &bytes[..MAGIC.len()] != MAGIC {
        return Err("cache envelope magic mismatch".into());
    }
    let mut cursor = MAGIC.len();
    let schema = read_u32(bytes, &mut cursor)?;
    if schema != TERRA_PIPELINE_CACHE_SCHEMA {
        return Err(format!("cache schema mismatch: found {schema}"));
    }
    let key_len = read_u32(bytes, &mut cursor)? as usize;
    let payload_len = usize::try_from(read_u64(bytes, &mut cursor)?)
        .map_err(|_| "cache payload length does not fit this platform".to_string())?;
    let checksum_end = cursor
        .checked_add(CHECKSUM_BYTES)
        .ok_or_else(|| "cache envelope length overflow".to_string())?;
    let stored_checksum: [u8; CHECKSUM_BYTES] = bytes
        .get(cursor..checksum_end)
        .ok_or_else(|| "cache checksum is truncated".to_string())?
        .try_into()
        .map_err(|_| "cache checksum is malformed".to_string())?;
    cursor = checksum_end;
    if key_len > MAX_KEY_BYTES {
        return Err("cache key is oversized".into());
    }
    if payload_len > MAX_PAYLOAD_BYTES {
        return Err("cache payload is oversized".into());
    }
    let key_end = cursor
        .checked_add(key_len)
        .ok_or_else(|| "cache key length overflow".to_string())?;
    let payload_end = key_end
        .checked_add(payload_len)
        .ok_or_else(|| "cache payload length overflow".to_string())?;
    if payload_end != bytes.len() {
        return Err("cache envelope length mismatch".into());
    }
    let key = bytes
        .get(cursor..key_end)
        .ok_or_else(|| "cache key is truncated".to_string())?;
    if key != expected_key.as_bytes() {
        return Err("cache identity mismatch".into());
    }
    let payload = bytes
        .get(key_end..payload_end)
        .ok_or_else(|| "cache payload is truncated".to_string())?;
    if checksum(key, payload) != stored_checksum {
        return Err("cache checksum mismatch".into());
    }
    Ok(payload)
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, String> {
    let end = cursor
        .checked_add(4)
        .ok_or_else(|| "cache header length overflow".to_string())?;
    let value = bytes
        .get(*cursor..end)
        .ok_or_else(|| "cache header is truncated".to_string())?;
    *cursor = end;
    Ok(u32::from_le_bytes(value.try_into().unwrap()))
}

fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64, String> {
    let end = cursor
        .checked_add(8)
        .ok_or_else(|| "cache header length overflow".to_string())?;
    let value = bytes
        .get(*cursor..end)
        .ok_or_else(|| "cache header is truncated".to_string())?;
    *cursor = end;
    Ok(u64::from_le_bytes(value.try_into().unwrap()))
}

fn load_payload(target: &PersistenceTarget) -> Result<Option<Vec<u8>>, String> {
    let mut file = match File::open(&target.path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("cache file open failed: {error}")),
    };
    let length = file
        .metadata()
        .map_err(|error| format!("cache metadata failed: {error}"))?
        .len();
    if length > MAX_FILE_BYTES {
        return Err(format!("cache file is oversized: {length} bytes"));
    }
    let capacity = usize::try_from(length).map_err(|_| "cache file length overflow".to_string())?;
    let mut bytes = Vec::with_capacity(capacity);
    (&mut file)
        .take(MAX_FILE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("cache file read failed: {error}"))?;
    if bytes.len() as u64 > MAX_FILE_BYTES {
        return Err(format!(
            "cache file grew beyond the {MAX_FILE_BYTES} byte limit while reading"
        ));
    }
    let payload = decode_envelope(&target.key, &bytes)?;
    Ok(Some(payload.to_vec()))
}

fn atomic_write(destination: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let parent = destination.parent().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "cache path has no parent")
    })?;
    std::fs::create_dir_all(parent)?;
    let file_name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "cache filename is invalid",
            )
        })?;
    let mut last_collision = None;
    for _ in 0..32 {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temporary = parent.join(format!(
            ".{file_name}.{}.{}.tmp",
            std::process::id(),
            sequence
        ));
        let mut file = match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                last_collision = Some(error);
                continue;
            }
            Err(error) => return Err(error),
        };
        let result = (|| {
            file.write_all(bytes)?;
            file.flush()?;
            file.sync_all()?;
            drop(file);
            replace_file(&temporary, destination)
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&temporary);
        }
        return result;
    }
    Err(last_collision.unwrap_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "could not allocate unique cache temporary file",
        )
    }))
}

#[cfg(not(windows))]
fn replace_file(temporary: &Path, destination: &Path) -> std::io::Result<()> {
    std::fs::rename(temporary, destination)?;
    if let Some(parent) = destination.parent() {
        let _ = File::open(parent).and_then(|directory| directory.sync_all());
    }
    Ok(())
}

#[cfg(windows)]
fn replace_file(temporary: &Path, destination: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };

    let source: Vec<u16> = temporary
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let target: Vec<u16> = destination
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let moved = unsafe {
        MoveFileExW(
            source.as_ptr(),
            target.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if moved == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "terra-pipeline-cache-test-{}-{sequence}",
                std::process::id()
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn adapter(backend: wgpu::Backend) -> wgpu::AdapterInfo {
        wgpu::AdapterInfo {
            name: "test".into(),
            vendor: 4318,
            device: 12345,
            device_type: wgpu::DeviceType::DiscreteGpu,
            driver: "driver".into(),
            driver_info: "info".into(),
            backend,
        }
    }

    #[test]
    fn identity_is_deterministic_and_versioned() {
        let info = adapter(wgpu::Backend::Vulkan);
        let wgpu_key = wgpu::util::pipeline_cache_key(&info).unwrap();
        let key = canonical_key(info.backend, &wgpu_key);
        assert_eq!(
            key,
            "terra-pipeline-schema-1-wgpu-24.0.5-vulkan-wgpu_pipeline_cache_vulkan_4318_12345"
        );
        assert!(wgpu::util::pipeline_cache_key(&adapter(wgpu::Backend::Dx12)).is_none());
        let workspace_manifest = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .unwrap()
            .join("Cargo.toml");
        let manifest = std::fs::read_to_string(workspace_manifest).unwrap();
        assert!(manifest.contains(&format!("wgpu = \"={WGPU_PIPELINE_CACHE_COMPAT}\"")));
    }

    #[test]
    fn envelope_round_trip_rejects_corruption_and_mismatch() {
        let key = "cache-key";
        let payload = b"driver-owned-bytes";
        let envelope = encode_envelope(key, payload);
        assert_eq!(decode_envelope(key, &envelope).unwrap(), payload);

        let mut corrupt = envelope.clone();
        *corrupt.last_mut().unwrap() ^= 0xff;
        assert!(decode_envelope(key, &corrupt)
            .unwrap_err()
            .contains("checksum"));
        assert!(decode_envelope("other-key", &envelope)
            .unwrap_err()
            .contains("identity"));
        assert!(decode_envelope(key, &envelope[..envelope.len() - 1]).is_err());
        let mut trailing = envelope;
        trailing.push(0);
        assert!(decode_envelope(key, &trailing).is_err());
    }

    #[test]
    fn envelope_rejects_header_schema_and_size_failures() {
        let key = "cache-key";
        let mut bad_magic = encode_envelope(key, b"data");
        bad_magic[0] ^= 1;
        assert!(decode_envelope(key, &bad_magic)
            .unwrap_err()
            .contains("magic"));

        let mut bad_schema = encode_envelope(key, b"data");
        bad_schema[MAGIC.len()..MAGIC.len() + 4].copy_from_slice(&2u32.to_le_bytes());
        assert!(decode_envelope(key, &bad_schema)
            .unwrap_err()
            .contains("schema"));

        let mut oversized = encode_envelope(key, b"data");
        let payload_length_offset = MAGIC.len() + 4 + 4;
        oversized[payload_length_offset..payload_length_offset + 8]
            .copy_from_slice(&((MAX_PAYLOAD_BYTES as u64) + 1).to_le_bytes());
        assert!(decode_envelope(key, &oversized)
            .unwrap_err()
            .contains("oversized"));
    }

    #[test]
    fn missing_corrupt_and_valid_files_have_expected_results() {
        let directory = TestDirectory::new();
        let target = PersistenceTarget {
            key: "cache-key".into(),
            path: directory.path().join("cache.bin"),
        };
        assert!(load_payload(&target).unwrap().is_none());
        std::fs::write(&target.path, b"bad").unwrap();
        assert!(load_payload(&target).is_err());
        std::fs::write(&target.path, encode_envelope(&target.key, b"payload")).unwrap();
        assert_eq!(load_payload(&target).unwrap().unwrap(), b"payload");
    }

    #[test]
    fn oversized_file_is_rejected_before_reading() {
        let directory = TestDirectory::new();
        let target = PersistenceTarget {
            key: "cache-key".into(),
            path: directory.path().join("cache.bin"),
        };
        let file = File::create(&target.path).unwrap();
        file.set_len(MAX_FILE_BYTES + 1).unwrap();
        drop(file);
        assert!(load_payload(&target).unwrap_err().contains("oversized"));
    }

    #[test]
    fn atomic_write_replaces_existing_destination() {
        let directory = TestDirectory::new();
        let destination = directory.path().join("cache.bin");
        atomic_write(&destination, b"first").unwrap();
        assert_eq!(std::fs::read(&destination).unwrap(), b"first");
        atomic_write(&destination, b"second").unwrap();
        assert_eq!(std::fs::read(&destination).unwrap(), b"second");
        let leftovers: Vec<_> = std::fs::read_dir(directory.path())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty());
    }
}
