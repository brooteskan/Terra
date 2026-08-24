//! Thread-safe, backend-neutral startup compilation telemetry.
//!
//! The tracker intentionally knows nothing about `wgpu`. GPU-facing crates wrap
//! their synchronous pipeline calls with [`measure`], while the application
//! snapshots the same generation to render live startup status and write reports.

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilationKind {
    RenderPipeline,
    ComputePipeline,
    StartupStage,
}

impl CompilationKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RenderPipeline => "render_pipeline",
            Self::ComputePipeline => "compute_pipeline",
            Self::StartupStage => "startup_stage",
        }
    }

    pub const fn is_pipeline(self) -> bool {
        matches!(self, Self::RenderPipeline | Self::ComputePipeline)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilationStatus {
    Completed,
    Failed,
}

impl CompilationStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ActiveCompilation {
    pub generation: u64,
    pub id: u64,
    pub kind: CompilationKind,
    pub label: String,
    pub started_after_reset: Duration,
    pub elapsed: Duration,
    pub thread_name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CompilationRecord {
    pub generation: u64,
    pub id: u64,
    pub kind: CompilationKind,
    pub label: String,
    pub started_after_reset: Duration,
    pub duration: Duration,
    pub status: CompilationStatus,
    pub thread_name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CompilationSnapshot {
    pub generation: u64,
    pub elapsed: Duration,
    pub active: Vec<ActiveCompilation>,
    pub completed: Vec<CompilationRecord>,
}

impl CompilationSnapshot {
    pub fn active_pipeline(&self) -> Option<&ActiveCompilation> {
        self.active.iter().find(|entry| entry.kind.is_pipeline())
    }

    pub fn active_stage(&self) -> Option<&ActiveCompilation> {
        self.active
            .iter()
            .find(|entry| entry.kind == CompilationKind::StartupStage)
    }

    pub fn last_failure(&self) -> Option<&CompilationRecord> {
        self.completed
            .iter()
            .rev()
            .find(|entry| entry.status == CompilationStatus::Failed && entry.kind.is_pipeline())
            .or_else(|| {
                self.completed
                    .iter()
                    .rev()
                    .find(|entry| entry.status == CompilationStatus::Failed)
            })
    }

    pub fn dominant_pipeline(&self) -> Option<&CompilationRecord> {
        self.completed
            .iter()
            .filter(|entry| entry.kind.is_pipeline())
            .max_by_key(|entry| entry.duration)
    }
}

#[derive(Debug)]
struct ActiveEntry {
    generation: u64,
    id: u64,
    kind: CompilationKind,
    label: String,
    started: Instant,
    thread_name: Option<String>,
}

#[derive(Debug)]
struct TrackerState {
    generation: u64,
    next_id: u64,
    reset_at: Instant,
    active: Vec<ActiveEntry>,
    completed: Vec<CompilationRecord>,
}

impl Default for TrackerState {
    fn default() -> Self {
        Self {
            generation: 0,
            next_id: 0,
            reset_at: Instant::now(),
            active: Vec::new(),
            completed: Vec::new(),
        }
    }
}

static TRACKER: OnceLock<Mutex<TrackerState>> = OnceLock::new();

fn tracker() -> &'static Mutex<TrackerState> {
    TRACKER.get_or_init(|| Mutex::new(TrackerState::default()))
}

fn lock_tracker() -> std::sync::MutexGuard<'static, TrackerState> {
    tracker()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Start a new measurement generation and invalidate guards from older boots.
pub fn reset() -> u64 {
    let mut state = lock_tracker();
    state.generation = state.generation.wrapping_add(1);
    state.next_id = 0;
    state.reset_at = Instant::now();
    state.active.clear();
    state.completed.clear();
    state.generation
}

pub fn generation() -> u64 {
    lock_tracker().generation
}

/// Begin one live operation. Dropping the returned guard without completing it
/// records failure and always clears the active entry.
pub fn begin(kind: CompilationKind, label: impl Into<String>) -> CompilationGuard {
    let started = Instant::now();
    let thread_name = std::thread::current().name().map(str::to_owned);
    let mut state = lock_tracker();
    let generation = state.generation;
    let id = state.next_id;
    state.next_id = state.next_id.wrapping_add(1);
    state.active.push(ActiveEntry {
        generation,
        id,
        kind,
        label: label.into(),
        started,
        thread_name,
    });
    CompilationGuard {
        generation,
        id,
        finished: false,
    }
}

pub fn begin_stage(label: impl Into<String>) -> CompilationGuard {
    begin(CompilationKind::StartupStage, label)
}

/// Measure a synchronous operation. Unwinding through the closure is recorded
/// as failure by the guard's `Drop` implementation.
pub fn measure<T>(
    kind: CompilationKind,
    label: impl Into<String>,
    operation: impl FnOnce() -> T,
) -> T {
    let guard = begin(kind, label);
    let value = operation();
    guard.complete();
    value
}

pub fn snapshot() -> CompilationSnapshot {
    let state = lock_tracker();
    let now = Instant::now();
    let mut active: Vec<_> = state
        .active
        .iter()
        .map(|entry| ActiveCompilation {
            generation: entry.generation,
            id: entry.id,
            kind: entry.kind,
            label: entry.label.clone(),
            started_after_reset: entry.started.saturating_duration_since(state.reset_at),
            elapsed: now.saturating_duration_since(entry.started),
            thread_name: entry.thread_name.clone(),
        })
        .collect();
    active.sort_by_key(|entry| entry.started_after_reset);
    CompilationSnapshot {
        generation: state.generation,
        elapsed: now.saturating_duration_since(state.reset_at),
        active,
        completed: state.completed.clone(),
    }
}

/// Mark every still-active entry in the current generation failed. This is a
/// defensive cleanup for a failed worker handoff or cancellation boundary.
pub fn fail_active() {
    let mut state = lock_tracker();
    let generation = state.generation;
    let now = Instant::now();
    let reset_at = state.reset_at;
    let active = std::mem::take(&mut state.active);
    state
        .completed
        .extend(active.into_iter().filter_map(|entry| {
            (entry.generation == generation).then(|| CompilationRecord {
                generation,
                id: entry.id,
                kind: entry.kind,
                label: entry.label,
                started_after_reset: entry.started.saturating_duration_since(reset_at),
                duration: now.saturating_duration_since(entry.started),
                status: CompilationStatus::Failed,
                thread_name: entry.thread_name,
            })
        }));
}

#[must_use]
pub struct CompilationGuard {
    generation: u64,
    id: u64,
    finished: bool,
}

impl CompilationGuard {
    pub fn complete(mut self) {
        self.finish(CompilationStatus::Completed);
    }

    pub fn fail(mut self) {
        self.finish(CompilationStatus::Failed);
    }

    fn finish(&mut self, status: CompilationStatus) {
        if self.finished {
            return;
        }
        self.finished = true;
        let mut state = lock_tracker();
        if state.generation != self.generation {
            return;
        }
        let Some(index) = state
            .active
            .iter()
            .position(|entry| entry.generation == self.generation && entry.id == self.id)
        else {
            return;
        };
        let entry = state.active.remove(index);
        let now = Instant::now();
        let reset_at = state.reset_at;
        state.completed.push(CompilationRecord {
            generation: entry.generation,
            id: entry.id,
            kind: entry.kind,
            label: entry.label,
            started_after_reset: entry.started.saturating_duration_since(reset_at),
            duration: now.saturating_duration_since(entry.started),
            status,
            thread_name: entry.thread_name,
        });
    }
}

impl Drop for CompilationGuard {
    fn drop(&mut self) {
        self.finish(CompilationStatus::Failed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    fn isolated() -> std::sync::MutexGuard<'static, ()> {
        static TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        TEST_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[test]
    fn reset_invalidates_stale_guards() {
        let _isolated = isolated();
        reset();
        let stale = begin_stage("old boot");
        let generation = reset();
        drop(stale);
        let snapshot = snapshot();
        assert_eq!(snapshot.generation, generation);
        assert!(snapshot.active.is_empty());
        assert!(snapshot.completed.is_empty());
    }

    #[test]
    fn completed_operation_has_a_duration_and_clears_active() {
        let _isolated = isolated();
        reset();
        measure(CompilationKind::ComputePipeline, "compute-a", || {
            assert_eq!(snapshot().active_pipeline().unwrap().label, "compute-a");
        });
        let snapshot = snapshot();
        assert!(snapshot.active.is_empty());
        assert_eq!(snapshot.completed.len(), 1);
        assert_eq!(snapshot.completed[0].status, CompilationStatus::Completed);
    }

    #[test]
    fn panic_records_failure_and_clears_active() {
        let _isolated = isolated();
        reset();
        let result = std::panic::catch_unwind(|| {
            measure(CompilationKind::RenderPipeline, "render-a", || {
                panic!("boom")
            });
        });
        assert!(result.is_err());
        let snapshot = snapshot();
        assert!(snapshot.active.is_empty());
        assert_eq!(snapshot.last_failure().unwrap().label, "render-a");
    }

    #[test]
    fn concurrent_entries_are_all_visible() {
        let _isolated = isolated();
        reset();
        let barrier = Arc::new(Barrier::new(3));
        let mut workers = Vec::new();
        for label in ["one", "two"] {
            let barrier = Arc::clone(&barrier);
            workers.push(std::thread::spawn(move || {
                let guard = begin(CompilationKind::ComputePipeline, label);
                barrier.wait();
                barrier.wait();
                guard.complete();
            }));
        }
        barrier.wait();
        assert_eq!(snapshot().active.len(), 2);
        barrier.wait();
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(snapshot().completed.len(), 2);
    }

    #[test]
    fn fail_active_preserves_failed_labels() {
        let _isolated = isolated();
        reset();
        let guard = begin_stage("worker stage");
        fail_active();
        drop(guard);
        let snapshot = snapshot();
        assert!(snapshot.active.is_empty());
        assert_eq!(snapshot.last_failure().unwrap().label, "worker stage");
    }

    #[test]
    fn failed_pipeline_is_preferred_over_its_unwinding_stage() {
        let _isolated = isolated();
        reset();
        let stage = begin_stage("Building renderer");
        begin(CompilationKind::RenderPipeline, "terrain-bounded-pipe").fail();
        stage.fail();
        assert_eq!(
            snapshot().last_failure().unwrap().label,
            "terrain-bounded-pipe"
        );
    }
}
