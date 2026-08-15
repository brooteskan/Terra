//! Application-owned logging initialization and diagnostic context.

use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};

use flexi_logger::{
    DeferredNow, Duplicate, FileSpec, LogSpecification, Logger, LoggerHandle, Record,
};
use terra_core::eval::PreviewQuality;

pub const RETAINED_LOG_FILES: usize = 6;
pub const DEFAULT_LOG_FILTER: &str = "warn,terra_app=info,terra_core=info,terra_gpu=info,terra_render=info,terra_io=info,terra_gui=info";
pub const LOG_FILE_PREFIX: &str = "log-";

/// Keeps the active logger alive until application shutdown.
pub struct LoggingGuard {
    _handle: Option<LoggerHandle>,
    log_file: Option<PathBuf>,
}

#[derive(Debug, PartialEq, Eq)]
enum LogDestination {
    File(PathBuf),
    ConsoleOnly(String),
}

impl LoggingGuard {
    pub fn log_file(&self) -> Option<&Path> {
        self.log_file.as_deref()
    }
}

/// Initialize console + rotating file logging, degrading to console-only on failure.
pub fn init() -> LoggingGuard {
    let mut startup_warning = None;
    let file_logger = match prepare_log_destination(log_directory()) {
        LogDestination::File(directory) => start_file_logger(&directory)
            .map(|(handle, filter_warning, log_file)| {
                startup_warning = filter_warning;
                (handle, log_file)
            })
            .map_err(|error| format!("could not open persistent log file: {error}")),
        LogDestination::ConsoleOnly(error) => Err(error),
    };
    let (handle, log_file) = match file_logger {
        Ok((handle, log_file)) => (Some(handle), Some(log_file)),
        Err(file_error) => {
            let (handle, filter_warning) = match start_console_logger() {
                Ok(result) => result,
                Err(console_error) => {
                    eprintln!(
                        "Terra logging unavailable: {file_error}; console logger failed: {console_error}"
                    );
                    install_panic_hook();
                    return LoggingGuard {
                        _handle: None,
                        log_file: None,
                    };
                }
            };
            startup_warning = filter_warning;
            log::warn!("persistent logging unavailable; using console only: {file_error}");
            (Some(handle), None)
        }
    };

    install_panic_hook();

    let active_filter = if startup_warning.is_some() {
        DEFAULT_LOG_FILTER.to_string()
    } else {
        std::env::var("RUST_LOG").unwrap_or_else(|_| DEFAULT_LOG_FILTER.to_string())
    };
    if let Some(warning) = startup_warning {
        log::warn!("{warning}; using default filter {DEFAULT_LOG_FILTER:?}");
    }
    if let Some(path) = log_file.as_deref() {
        log::info!(
            target: "terra_app::logging",
            "Terra {} starting; log_file={}; filter={active_filter:?}; retained_launch_logs={}",
            env!("CARGO_PKG_VERSION"),
            path.display(),
            RETAINED_LOG_FILES
        );
    } else {
        log::info!(
            target: "terra_app::logging",
            "Terra {} starting with console-only logging",
            env!("CARGO_PKG_VERSION")
        );
    }

    LoggingGuard {
        _handle: handle,
        log_file,
    }
}

/// Return the directory where Terra writes its rotating application logs.
pub fn log_directory() -> Result<PathBuf, String> {
    directories::BaseDirs::new()
        .map(|dirs| log_directory_from_local_data(dirs.data_local_dir()))
        .ok_or_else(|| "per-user local application-data directory is unavailable".to_string())
}

fn log_directory_from_local_data(local_data: &Path) -> PathBuf {
    local_data.join("Terra").join("logs")
}

fn prepare_log_directory(directory: &Path) -> Result<(), String> {
    std::fs::create_dir_all(directory)
        .map_err(|error| format!("could not create {}: {error}", directory.display()))
}

fn prepare_log_destination(directory: Result<PathBuf, String>) -> LogDestination {
    match directory {
        Ok(directory) => match prepare_log_directory(&directory) {
            Ok(()) => LogDestination::File(directory),
            Err(error) => LogDestination::ConsoleOnly(error),
        },
        Err(error) => LogDestination::ConsoleOnly(error),
    }
}

fn configured_logger() -> (Logger, Option<String>) {
    match Logger::try_with_env_or_str(DEFAULT_LOG_FILTER) {
        Ok(logger) => (logger, None),
        Err(error) => (
            Logger::with(
                LogSpecification::parse(DEFAULT_LOG_FILTER)
                    .unwrap_or_else(|_| LogSpecification::warn()),
            ),
            Some(format!("invalid RUST_LOG filter: {error}")),
        ),
    }
}

fn launch_timestamp() -> String {
    let mut now = DeferredNow::new();
    now.format("%Y-%m-%d_%H-%M-%S-%3f").to_string()
}

fn unique_log_path(directory: &Path, timestamp: &str) -> PathBuf {
    let initial = directory.join(format!("{LOG_FILE_PREFIX}{timestamp}.log"));
    if !initial.exists() {
        return initial;
    }
    for suffix in 2_u32.. {
        let candidate = directory.join(format!("{LOG_FILE_PREFIX}{timestamp}-{suffix}.log"));
        if !candidate.exists() {
            return candidate;
        }
    }
    unreachable!("the log filename suffix space is inexhaustible")
}

fn is_launch_log(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with(LOG_FILE_PREFIX) && name.ends_with(".log"))
}

fn cleanup_old_launch_logs(directory: &Path, active_log: &Path) -> Result<(), String> {
    let mut previous = std::fs::read_dir(directory)
        .map_err(|error| format!("could not scan {}: {error}", directory.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path != active_log && is_launch_log(path))
        .map(|path| {
            let modified = std::fs::metadata(&path)
                .and_then(|metadata| metadata.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            (modified, path)
        })
        .collect::<Vec<_>>();
    previous.sort();

    let remove_count = (previous.len() + 1).saturating_sub(RETAINED_LOG_FILES);
    for (_, path) in previous.into_iter().take(remove_count) {
        std::fs::remove_file(&path)
            .map_err(|error| format!("could not remove old log {}: {error}", path.display()))?;
    }
    Ok(())
}

fn start_file_logger(directory: &Path) -> Result<(LoggerHandle, Option<String>, PathBuf), String> {
    let (logger, filter_warning) = configured_logger();
    let log_file = unique_log_path(directory, &launch_timestamp());
    let file_spec = FileSpec::try_from(&log_file).map_err(|error| error.to_string())?;
    let handle = logger
        .log_to_file(file_spec)
        .duplicate_to_stderr(Duplicate::All)
        .format(log_format)
        .start()
        .map_err(|error| error.to_string())?;
    if let Err(error) = cleanup_old_launch_logs(directory, &log_file) {
        log::warn!(target: "terra_app::logging", "log retention cleanup failed: {error}");
    }
    Ok((handle, filter_warning, log_file))
}

fn start_console_logger() -> Result<(LoggerHandle, Option<String>), String> {
    let (logger, filter_warning) = configured_logger();
    let handle = logger
        .log_to_stderr()
        .format(log_format)
        .start()
        .map_err(|error| error.to_string())?;
    Ok((handle, filter_warning))
}

fn log_format(
    writer: &mut dyn Write,
    now: &mut DeferredNow,
    record: &Record<'_>,
) -> std::io::Result<()> {
    let thread = std::thread::current();
    let thread_name = thread.name().unwrap_or("<unnamed>");
    writeln!(
        writer,
        "{} {:<5} [{} {:?}] {}: {}",
        now.format_rfc3339(),
        record.level(),
        thread_name,
        thread.id(),
        record.target(),
        record.args()
    )
}

fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let payload = if let Some(message) = info.payload().downcast_ref::<&str>() {
            (*message).to_string()
        } else if let Some(message) = info.payload().downcast_ref::<String>() {
            message.clone()
        } else {
            "non-string panic payload".to_string()
        };
        let location = info
            .location()
            .map(|location| {
                format!(
                    "{}:{}:{}",
                    location.file(),
                    location.line(),
                    location.column()
                )
            })
            .unwrap_or_else(|| "<unknown>".to_string());
        log::error!(
            target: "terra_app::panic",
            "panic at {location}: {payload}"
        );
        log::logger().flush();
        previous(info);
    }));
}

/// Consistent context attached to fallible application operations.
#[derive(Debug, Clone, Copy)]
pub struct OperationContext<'a> {
    operation: &'static str,
    token: Option<u64>,
    quality: Option<PreviewQuality>,
    layer: Option<&'a str>,
    project_path: Option<&'a Path>,
}

impl<'a> OperationContext<'a> {
    pub fn evaluation(token: u64, quality: PreviewQuality) -> Self {
        Self {
            operation: "evaluation",
            token: Some(token),
            quality: Some(quality),
            layer: None,
            project_path: None,
        }
    }

    pub fn with_layer(mut self, layer: Option<&'a str>) -> Self {
        self.layer = layer;
        self
    }

    pub fn with_project_path(mut self, project_path: Option<&'a Path>) -> Self {
        self.project_path = project_path;
        self
    }
}

impl fmt::Display for OperationContext<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "operation={}", self.operation)?;
        if let Some(token) = self.token {
            write!(formatter, " token={token}")?;
        }
        if let Some(quality) = self.quality {
            write!(formatter, " quality={quality:?}")?;
        }
        if let Some(layer) = self.layer {
            write!(formatter, " layer={layer:?}")?;
        }
        if let Some(project_path) = self.project_path {
            write!(formatter, " project={project_path:?}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new(label: &str) -> Self {
            let unique = format!(
                "terra-{label}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("system time")
                    .as_nanos()
            );
            let path = std::env::temp_dir().join(unique);
            std::fs::create_dir_all(&path).expect("create test directory");
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

    #[test]
    fn local_data_path_maps_to_terra_logs() {
        let root = Path::new("local-data-root");
        assert_eq!(
            log_directory_from_local_data(root),
            root.join("Terra").join("logs")
        );
    }

    #[test]
    fn uncreatable_log_directory_is_a_recoverable_error() {
        let temp = TestDirectory::new("logging-fallback");
        let blocker = temp.path().join("blocker");
        std::fs::write(&blocker, b"not a directory").expect("write blocker");
        let destination = prepare_log_destination(Ok(blocker.join("logs")));
        let LogDestination::ConsoleOnly(error) = destination else {
            panic!("an uncreatable file destination must select console fallback");
        };
        assert!(error.contains("could not create"));
    }

    #[test]
    fn operation_context_includes_available_evaluation_fields() {
        let path = Path::new(r"D:\Worlds\Island\Island.json");
        let context = OperationContext::evaluation(42, PreviewQuality::Full)
            .with_layer(Some("Hydraulic Erosion"))
            .with_project_path(Some(path))
            .to_string();
        assert!(context.contains("operation=evaluation"));
        assert!(context.contains("token=42"));
        assert!(context.contains("quality=Full"));
        assert!(context.contains("layer=\"Hydraulic Erosion\""));
        assert!(context.contains("Island.json"));
    }

    #[test]
    fn retention_policy_is_bounded_and_documentable() {
        assert_eq!(RETAINED_LOG_FILES, 6);
        assert!(DEFAULT_LOG_FILTER.starts_with("warn,"));
        assert!(DEFAULT_LOG_FILTER.contains("terra_app=info"));
        let specification =
            LogSpecification::parse(DEFAULT_LOG_FILTER).expect("valid default log filter");
        assert!(specification.enabled(log::Level::Info, "terra_app::logging"));
        assert!(specification.enabled(log::Level::Warn, "wgpu_hal::dx12::device"));
        assert!(!specification.enabled(log::Level::Info, "wgpu_hal::dx12::device"));
    }

    #[test]
    fn formatter_includes_level_target_thread_and_message() {
        let mut bytes = Vec::new();
        let record = log::Record::builder()
            .args(format_args!("format probe"))
            .level(log::Level::Warn)
            .target("terra_app::logging::probe")
            .build();
        log_format(&mut bytes, &mut DeferredNow::new(), &record).expect("format record");
        let text = String::from_utf8(bytes).expect("UTF-8 log line");
        assert!(text.contains("WARN"));
        assert!(text.contains("ThreadId("));
        assert!(text.contains("terra_app::logging::probe"));
        assert!(text.contains("format probe"));
        assert!(text.contains('T'), "RFC 3339 timestamp should contain T");
    }

    #[test]
    fn launch_log_paths_are_timestamped_and_collision_safe() {
        let temp = TestDirectory::new("launch-log-name");
        let timestamp = "2026-08-14_20-30-01-123";
        let generated_timestamp = launch_timestamp();
        assert_eq!(generated_timestamp.len(), timestamp.len());
        assert!(generated_timestamp
            .chars()
            .all(|character| character.is_ascii_digit() || character == '-' || character == '_'));
        let first = unique_log_path(temp.path(), timestamp);
        assert_eq!(
            first.file_name().and_then(|name| name.to_str()),
            Some("log-2026-08-14_20-30-01-123.log")
        );
        std::fs::write(&first, b"first launch").expect("write first launch log");
        assert_eq!(
            unique_log_path(temp.path(), timestamp)
                .file_name()
                .and_then(|name| name.to_str()),
            Some("log-2026-08-14_20-30-01-123-2.log")
        );
    }

    #[test]
    fn cleanup_keeps_six_launch_logs_and_ignores_other_files() {
        let temp = TestDirectory::new("launch-log-cleanup");
        let active = temp.path().join("log-2026-08-14_20-30-09-000.log");
        for index in 1..=8 {
            let path = temp
                .path()
                .join(format!("log-2026-08-14_20-30-0{index}-000.log"));
            std::fs::write(path, format!("launch {index}")).expect("write launch log");
        }
        std::fs::write(temp.path().join("notes.txt"), b"keep me").expect("write unrelated file");
        std::fs::write(&active, b"active launch").expect("write active log");

        cleanup_old_launch_logs(temp.path(), &active).expect("cleanup launch logs");

        let launch_logs = std::fs::read_dir(temp.path())
            .expect("read test directory")
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| is_launch_log(path))
            .count();
        assert_eq!(launch_logs, RETAINED_LOG_FILES);
        assert!(active.exists());
        assert!(temp.path().join("notes.txt").exists());
    }
}
