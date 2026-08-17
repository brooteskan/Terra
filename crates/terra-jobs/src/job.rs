//! One-shot background jobs: a named worker thread with panic containment by
//! construction, poll-friendly delivery, atomic progress, and cancellation.
//!
//! [`spawn_one_shot`] runs `f(&JobCtx) -> T` on a named thread whose body is
//! wrapped in [`catch_unwind`](std::panic::catch_unwind), so a panicking job
//! resolves to [`JobError::Panicked`] instead of unwinding the worker and
//! silently wedging the caller. The returned [`JobHandle`] is a single-consumer
//! poll surface: [`try_take`](JobHandle::try_take) lifts the finished result out
//! once, [`progress`](JobHandle::progress) reads an atomic `f32`, and
//! [`cancel`](JobHandle::cancel) trips the job's [`CancelToken`]. Dropping the
//! handle detaches the worker — it runs to completion and its result is dropped
//! when the last `Arc` goes (the boot path's receiver-drop discard semantics).

use std::any::Any;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::{CancelFlag, CancelToken};

/// Why a job failed to deliver its value.
///
/// Domain failures belong *inside* `T` (e.g. `T = Result<ExportResult, String>`);
/// this enum covers only the two ways a job fails to produce its `T` at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobError {
    /// The job's token was tripped; any value the body produced is discarded.
    Cancelled,
    /// The job body panicked; the string is the recovered panic message.
    Panicked(String),
}

impl std::fmt::Display for JobError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JobError::Cancelled => write!(f, "job cancelled"),
            JobError::Panicked(message) => write!(f, "job panicked: {message}"),
        }
    }
}

impl std::error::Error for JobError {}

/// The finished-result slot shared between the worker and its [`JobHandle`].
///
/// `finished` is flipped last (with `Release`), so a handle that observes it set
/// is guaranteed to see the stored `slot`. Kept as an alias to stay clear of
/// `clippy::type_complexity` at the struct field.
type ResultSlot<T> = Mutex<Option<Result<T, JobError>>>;

struct Completion<T> {
    finished: AtomicBool,
    slot: ResultSlot<T>,
}

/// Recover a human-readable message from a panic payload.
///
/// Mirrors `panic_payload_message` in terra-core's eval module. terra-jobs sits
/// *below* terra-core in the crate graph and cannot borrow it, so the small
/// helper is duplicated rather than shared. Shared within the crate so the
/// [`crate::latest_wins`] executor reuses it.
pub(crate) fn panic_payload_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".into()
    }
}

/// Handed to a job body: its cancellation token plus a progress sink.
pub struct JobCtx {
    token: CancelToken,
    progress: Arc<AtomicU32>,
}

impl JobCtx {
    /// The job's cancellation token. Poll it inside long loops so
    /// [`JobHandle::cancel`] is observed promptly.
    pub fn token(&self) -> &CancelToken {
        &self.token
    }

    /// Publish fractional progress (conventionally `0.0..=1.0`) for the handle to
    /// read. Stored as the `f32` bit pattern in one atomic — no channel needed.
    pub fn set_progress(&self, progress: f32) {
        self.progress.store(progress.to_bits(), Ordering::Release);
    }
}

/// A poll-friendly handle to a one-shot job.
///
/// Single consumer: [`try_take`](Self::try_take) yields the result exactly once.
/// Dropping the handle detaches the worker (no cancel, no join) — the result is
/// discarded when the worker finishes.
pub struct JobHandle<T> {
    completion: Arc<Completion<T>>,
    progress: Arc<AtomicU32>,
    flag: CancelFlag,
}

impl<T> JobHandle<T> {
    /// Take the finished result if the job has completed, else `None`. Returns
    /// `Some` at most once; subsequent calls return `None` even though
    /// [`is_finished`](Self::is_finished) stays `true`.
    pub fn try_take(&self) -> Option<Result<T, JobError>> {
        if !self.completion.finished.load(Ordering::Acquire) {
            return None;
        }
        self.completion
            .slot
            .lock()
            .expect("job result mutex poisoned")
            .take()
    }

    /// The latest progress the body published via [`JobCtx::set_progress`],
    /// defaulting to `0.0` before the first publish.
    pub fn progress(&self) -> f32 {
        f32::from_bits(self.progress.load(Ordering::Acquire))
    }

    /// Trip the job's cancel token so the body's [`JobCtx::token`] observes it.
    pub fn cancel(&self) {
        self.flag.cancel();
    }

    /// Whether the job has finished (delivered a value, panicked, or cancelled).
    /// Stays `true` after [`try_take`](Self::try_take) has consumed the result.
    pub fn is_finished(&self) -> bool {
        self.completion.finished.load(Ordering::Acquire)
    }
}

/// Spawn `f` on a named worker thread and return a poll handle.
///
/// The body is wrapped in [`catch_unwind`](std::panic::catch_unwind) by
/// construction: a panic becomes [`JobError::Panicked`] rather than unwinding the
/// worker. Cancellation takes precedence over the body's outcome — if the token
/// is tripped by the time the body returns, the job resolves to
/// [`JobError::Cancelled`] and any produced value (or panic) is discarded. This
/// is what lets a cancelled job finish with no delivered result even when it
/// raced to completion.
pub fn spawn_one_shot<T, F>(name: &str, f: F) -> JobHandle<T>
where
    T: Send + 'static,
    F: FnOnce(&JobCtx) -> T + Send + 'static,
{
    let (token, flag) = CancelToken::flag();
    // 0 bits is 0.0f32 — the correct starting progress.
    let progress = Arc::new(AtomicU32::new(0));
    let completion = Arc::new(Completion {
        finished: AtomicBool::new(false),
        slot: Mutex::new(None),
    });

    let worker_token = token;
    let worker_progress = Arc::clone(&progress);
    let worker_completion = Arc::clone(&completion);

    thread::Builder::new()
        .name(name.to_string())
        .spawn(move || {
            let ctx = JobCtx {
                token: worker_token,
                progress: worker_progress,
            };
            let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&ctx)));
            // Cancellation wins over both a produced value and a panic.
            let outcome = if ctx.token.is_cancelled() {
                Err(JobError::Cancelled)
            } else {
                match ran {
                    Ok(value) => Ok(value),
                    Err(payload) => Err(JobError::Panicked(panic_payload_message(payload))),
                }
            };
            *worker_completion
                .slot
                .lock()
                .expect("job result mutex poisoned") = Some(outcome);
            // Publish the slot before flipping `finished`: the handle reads
            // `finished` with Acquire, so this Release orders the store before it.
            worker_completion.finished.store(true, Ordering::Release);
        })
        .expect("spawn one-shot job thread");

    JobHandle {
        completion,
        progress,
        flag,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    /// Spin (yielding) until `cond` holds, with a generous safety timeout so a
    /// regression can't hang the suite. No sleep-based timing is relied on for
    /// correctness — only as an upper bound.
    fn wait_until(mut cond: impl FnMut() -> bool) {
        let start = Instant::now();
        while !cond() {
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "job did not reach the expected state within the timeout"
            );
            thread::yield_now();
        }
    }

    #[test]
    fn delivers_value_once() {
        let handle = spawn_one_shot("test-value", |_| 42u32);
        wait_until(|| handle.is_finished());
        assert_eq!(handle.try_take(), Some(Ok(42)));
        // Single consumer: the result is gone, but the job stays finished.
        assert_eq!(handle.try_take(), None);
        assert!(handle.is_finished());
    }

    #[test]
    fn panic_is_contained_and_reported() {
        let handle = spawn_one_shot("test-panic", |_| -> u32 { panic!("boom") });
        wait_until(|| handle.is_finished());
        match handle.try_take() {
            Some(Err(JobError::Panicked(message))) => assert!(message.contains("boom")),
            other => panic!("expected Panicked, got {other:?}"),
        }
    }

    #[test]
    fn cancel_is_observable_via_token() {
        let handle = spawn_one_shot("test-cancel-token", |ctx| {
            while !ctx.token().is_cancelled() {
                thread::yield_now();
            }
            7u32
        });
        handle.cancel();
        wait_until(|| handle.is_finished());
        assert_eq!(handle.try_take(), Some(Err(JobError::Cancelled)));
    }

    #[test]
    fn cancel_wins_over_a_late_produced_value() {
        // The body never checks the token; it blocks, then returns a value. The
        // cancel arrives before it returns, so the value must be discarded.
        let (tx, rx) = mpsc::channel::<()>();
        let handle = spawn_one_shot("test-cancel-wins", move |_| {
            rx.recv().expect("release signal");
            99u32
        });
        handle.cancel();
        tx.send(()).expect("send release");
        wait_until(|| handle.is_finished());
        assert_eq!(handle.try_take(), Some(Err(JobError::Cancelled)));
    }

    #[test]
    fn progress_is_readable() {
        let (tx, rx) = mpsc::channel::<()>();
        let handle = spawn_one_shot("test-progress", move |ctx| {
            ctx.set_progress(0.25);
            rx.recv().expect("release signal"); // hold so the test observes 0.25
            ctx.set_progress(1.0);
        });
        wait_until(|| handle.progress() >= 0.25);
        assert!((handle.progress() - 0.25).abs() < 1e-6);
        tx.send(()).expect("send release");
        wait_until(|| handle.is_finished());
        assert_eq!(handle.try_take(), Some(Ok(())));
    }

    #[test]
    fn dropping_handle_detaches_without_cancelling() {
        let ran = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel::<()>();
        let worker_ran = Arc::clone(&ran);
        let handle = spawn_one_shot("test-detach", move |_| {
            rx.recv().expect("release signal");
            worker_ran.store(true, Ordering::Release);
        });
        drop(handle); // detach: no cancel, no join
        tx.send(()).expect("send release");
        wait_until(|| ran.load(Ordering::Acquire));
    }
}
