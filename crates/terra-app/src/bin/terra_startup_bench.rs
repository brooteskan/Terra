use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::{Duration, Instant};

use terra_app::startup_benchmark::{
    StartupBenchmarkReport, StartupBenchmarkSuite, OUTPUT_ENV, PROFILE_ENV, RUN_KIND_ENV,
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
    for (profile, executable) in [
        ("debug", options.debug_exe.as_path()),
        ("release", options.release_exe.as_path()),
    ] {
        for run_kind in ["cold", "warm"] {
            reports.push(run_child(
                executable,
                profile,
                run_kind,
                &options.output,
                options.timeout,
            )?);
        }
    }
    let pids: BTreeSet<_> = reports.iter().map(|report| report.pid).collect();
    if pids.len() != reports.len() {
        return Err("a benchmark child PID was reused; fresh-process isolation is unproven".into());
    }
    let suite = StartupBenchmarkSuite {
        schema_version: 1,
        cold_warm_definition: "cold is the first fresh child process for a profile; warm is the immediately following fresh child. Driver-managed caches may persist, but in-process wgpu pipeline handles cannot.".into(),
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

fn run_child(
    executable: &Path,
    profile: &str,
    run_kind: &str,
    suite_output: &Path,
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
