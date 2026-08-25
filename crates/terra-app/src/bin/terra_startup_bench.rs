use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::{Duration, Instant};

use terra_app::startup_benchmark::{
    StartupBenchmarkComparison, StartupBenchmarkReport, StartupBenchmarkSuite, OUTPUT_ENV,
    PROFILE_ENV, RUN_KIND_ENV,
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Terra startup benchmark failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let options = Options::parse()?;
    let mut reports = Vec::with_capacity(4);
    let mut comparisons = Vec::with_capacity(2);
    let cache_root = benchmark_cache_root(&options.output)?;
    for (profile, executable) in [
        ("debug", options.debug_exe.as_path()),
        ("release", options.release_exe.as_path()),
    ] {
        let cache_directory = cache_root.join(profile);
        if cache_directory.exists() {
            std::fs::remove_dir_all(&cache_directory).map_err(|error| {
                format!(
                    "clear benchmark cache directory {}: {error}",
                    cache_directory.display()
                )
            })?;
        }
        std::fs::create_dir_all(&cache_directory).map_err(|error| {
            format!(
                "create benchmark cache directory {}: {error}",
                cache_directory.display()
            )
        })?;
        let pair_start = reports.len();
        for run_kind in ["cold", "warm"] {
            reports.push(run_child(
                executable,
                profile,
                run_kind,
                &options.output,
                &cache_directory,
                options.timeout,
            )?);
        }
        let cold = &reports[pair_start];
        let warm = &reports[pair_start + 1];
        validate_cache_pair(cold, warm)?;
        let speedup_ms = cold.boot_duration_ms as i128 - warm.boot_duration_ms as i128;
        let speedup_percent = if cold.boot_duration_ms == 0 {
            0.0
        } else {
            speedup_ms as f64 * 100.0 / cold.boot_duration_ms as f64
        };
        comparisons.push(StartupBenchmarkComparison {
            profile: profile.into(),
            cold_ms: cold.boot_duration_ms,
            warm_ms: warm.boot_duration_ms,
            speedup_ms: speedup_ms.clamp(i64::MIN as i128, i64::MAX as i128) as i64,
            speedup_percent,
        });
        println!(
            "{profile}: cold={}ms warm={}ms speedup={speedup_ms}ms ({speedup_percent:.1}%)",
            cold.boot_duration_ms, warm.boot_duration_ms
        );
    }
    let pids: BTreeSet<_> = reports.iter().map(|report| report.pid).collect();
    if pids.len() != reports.len() {
        return Err("a benchmark child PID was reused; fresh-process isolation is unproven".into());
    }
    let suite = StartupBenchmarkSuite {
        schema_version: 2,
        cold_warm_definition: "cold is a fresh child process with an empty benchmark-owned Terra pipeline-cache directory; warm is the immediately following fresh child using the cache saved by cold. Blob-loaded means Terra supplied validated bytes to wgpu, not that the driver reported a hit.".into(),
        comparisons,
        reports,
    };
    if let Some(parent) = options
        .output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("create output directory: {error}"))?;
    }
    let json = serde_json::to_vec_pretty(&suite)
        .map_err(|error| format!("serialize benchmark suite: {error}"))?;
    std::fs::write(&options.output, json)
        .map_err(|error| format!("write {}: {error}", options.output.display()))?;
    println!("wrote {}", options.output.display());
    Ok(())
}

fn benchmark_cache_root(output: &Path) -> Result<PathBuf, String> {
    let file_name = output
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty() && *name != "." && *name != "..")
        .ok_or_else(|| {
            "--output must name a file, not a directory or filesystem root".to_string()
        })?;
    let parent = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    Ok(parent.join(format!("{file_name}.pipeline-cache")))
}

fn run_child(
    executable: &Path,
    profile: &str,
    run_kind: &str,
    suite_output: &Path,
    cache_directory: &Path,
    timeout: Duration,
) -> Result<StartupBenchmarkReport, String> {
    if !executable.is_file() {
        return Err(format!(
            "{} executable not found: {}",
            profile,
            executable.display()
        ));
    }
    let child_output = suite_output.with_extension(format!("{profile}-{run_kind}.child.json"));
    let mut child = Command::new(executable)
        .env(OUTPUT_ENV, &child_output)
        .env(PROFILE_ENV, profile)
        .env(RUN_KIND_ENV, run_kind)
        .env(terra_app::pipeline_cache::DIRECTORY_ENV, cache_directory)
        .spawn()
        .map_err(|error| format!("launch {}: {error}", executable.display()))?;
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|error| format!("poll benchmark child: {error}"))?
        {
            break status;
        }
        if started.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("{profile} {run_kind} run exceeded {timeout:?}"));
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    let bytes = std::fs::read(&child_output)
        .map_err(|error| format!("read {}: {error}", child_output.display()))?;
    let report: StartupBenchmarkReport = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse {}: {error}", child_output.display()))?;
    if status.success() != report.success {
        return Err(format!(
            "{profile} {run_kind} exit/report disagreement: exit={status}, report success={}",
            report.success
        ));
    }
    Ok(report)
}

fn validate_cache_pair(
    cold: &StartupBenchmarkReport,
    warm: &StartupBenchmarkReport,
) -> Result<(), String> {
    if cold.adapter.pipeline_cache_enabled {
        if cold.pipeline_cache.blob_status != "miss" {
            return Err(format!(
                "supported cold run reported cache status {:?}",
                cold.pipeline_cache.blob_status
            ));
        }
        if cold.pipeline_cache.save_result != "saved" {
            return Err(format!(
                "supported cold run did not save cache data: {:?}",
                cold.pipeline_cache.save_result
            ));
        }
        if warm.pipeline_cache.blob_status != "blob-loaded" {
            return Err(format!(
                "supported warm run did not load Terra cache data: {:?}",
                warm.pipeline_cache.blob_status
            ));
        }
        if warm.pipeline_cache.save_result != "saved" {
            return Err(format!(
                "supported warm run did not update cache data: {:?}",
                warm.pipeline_cache.save_result
            ));
        }
    } else if cold.pipeline_cache.blob_status != "unsupported"
        || warm.pipeline_cache.blob_status != "unsupported"
    {
        return Err("unsupported backend did not report unsupported cache status".into());
    }
    Ok(())
}

struct Options {
    debug_exe: PathBuf,
    release_exe: PathBuf,
    output: PathBuf,
    timeout: Duration,
}

impl Options {
    fn parse() -> Result<Self, String> {
        let mut debug_exe = None;
        let mut release_exe = None;
        let mut output = None;
        let mut timeout = Duration::from_secs(900);
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--debug-exe" => debug_exe = Some(PathBuf::from(value)),
                "--release-exe" => release_exe = Some(PathBuf::from(value)),
                "--output" => output = Some(PathBuf::from(value)),
                "--timeout-seconds" => {
                    timeout = Duration::from_secs(
                        value
                            .parse()
                            .map_err(|_| format!("invalid timeout: {value}"))?,
                    );
                }
                _ => return Err(format!("unknown option: {flag}")),
            }
        }
        Ok(Self {
            debug_exe: debug_exe.ok_or("--debug-exe is required")?,
            release_exe: release_exe.ok_or("--release-exe is required")?,
            output: output.ok_or("--output is required")?,
            timeout,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_cache_root_is_a_dedicated_output_sibling() {
        assert_eq!(
            benchmark_cache_root(Path::new("reports/startup.json")).unwrap(),
            Path::new("reports/startup.json.pipeline-cache")
        );
        assert!(benchmark_cache_root(Path::new("/")).is_err());
    }
}
