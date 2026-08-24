//! JSON report schema for separate-process startup benchmarks.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const OUTPUT_ENV: &str = "TERRA_STARTUP_BENCH_OUTPUT";
pub const PROFILE_ENV: &str = "TERRA_STARTUP_BENCH_PROFILE";
pub const RUN_KIND_ENV: &str = "TERRA_STARTUP_BENCH_RUN_KIND";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartupBenchmarkReport {
    pub schema_version: u32,
    pub profile: String,
    pub run_kind: String,
    pub pid: u32,
    pub generation: u64,
    pub app_version: String,
    pub success: bool,
    pub error: Option<String>,
    pub adapter: AdapterReport,
    pub boot_duration_ms: u64,
    pub dominant_pipeline: Option<PipelineReport>,
    pub pipelines: Vec<PipelineReport>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AdapterReport {
    pub name: String,
    pub vendor: u32,
    pub device: u32,
    pub device_type: String,
    pub backend: String,
    pub driver: String,
    pub driver_info: String,
    pub downlevel_shader_model: String,
    pub pipeline_cache_supported: bool,
    pub pipeline_cache_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineReport {
    pub kind: String,
    pub label: String,
    pub status: String,
    pub started_ms: u64,
    pub duration_ms: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StartupBenchmarkSuite {
    pub schema_version: u32,
    pub cold_warm_definition: String,
    pub reports: Vec<StartupBenchmarkReport>,
}

pub fn output_path() -> Option<PathBuf> {
    std::env::var_os(OUTPUT_ENV).map(PathBuf::from)
}

pub fn is_requested() -> bool {
    output_path().is_some()
}

pub fn write_requested_report(
    gpu: Option<&terra_render::GpuContext>,
    telemetry: &terra_telemetry::CompilationSnapshot,
    error: Option<String>,
) -> Result<bool, String> {
    let Some(path) = output_path() else {
        return Ok(false);
    };
    let adapter = gpu.map_or_else(AdapterReport::default, |gpu| {
        let metadata = gpu.adapter_metadata();
        AdapterReport {
            name: metadata.name.clone(),
            vendor: metadata.vendor,
            device: metadata.device,
            device_type: metadata.device_type.clone(),
            backend: metadata.backend.clone(),
            driver: metadata.driver.clone(),
            driver_info: metadata.driver_info.clone(),
            downlevel_shader_model: metadata.downlevel_shader_model.clone(),
            pipeline_cache_supported: metadata.pipeline_cache_supported,
            pipeline_cache_enabled: metadata.pipeline_cache_enabled,
        }
    });
    let pipelines: Vec<_> = telemetry
        .completed
        .iter()
        .filter(|record| record.kind.is_pipeline())
        .map(pipeline_report)
        .collect();
    let dominant_pipeline = telemetry.dominant_pipeline().map(pipeline_report);
    let report = StartupBenchmarkReport {
        schema_version: 1,
        profile: std::env::var(PROFILE_ENV).unwrap_or_else(|_| "unknown".into()),
        run_kind: std::env::var(RUN_KIND_ENV).unwrap_or_else(|_| "unknown".into()),
        pid: std::process::id(),
        generation: telemetry.generation,
        app_version: env!("CARGO_PKG_VERSION").to_string(),
        success: error.is_none(),
        error,
        adapter,
        boot_duration_ms: millis(telemetry.elapsed),
        dominant_pipeline,
        pipelines,
    };
    let json = serde_json::to_vec_pretty(&report)
        .map_err(|error| format!("serialize startup benchmark report: {error}"))?;
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("create benchmark output directory: {error}"))?;
    }
    std::fs::write(&path, json)
        .map_err(|error| format!("write benchmark report {}: {error}", path.display()))?;
    Ok(true)
}

fn pipeline_report(record: &terra_telemetry::CompilationRecord) -> PipelineReport {
    PipelineReport {
        kind: record.kind.as_str().to_string(),
        label: record.label.clone(),
        status: record.status.as_str().to_string(),
        started_ms: millis(record.started_after_reset),
        duration_ms: millis(record.duration),
    }
}

fn millis(duration: std::time::Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}
