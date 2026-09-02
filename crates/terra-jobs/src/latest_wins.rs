//! A single-thread, latest-generation-wins executor.
//!
//! [`LatestWins`] owns a named worker thread carrying long-lived state `S`, and
//! runs submitted `Req` jobs against it one at a time. The *caller* owns a shared
//! generation counter (`Arc<AtomicU64>`); submitting a job stores its token into
//! that counter, and the executor only runs a dequeued job whose token still
//! matches the live generation — older jobs queued behind a newer one are skipped
//! silently. A standalone supersede is just the owner storing a new value into the
//! counter, so an in-flight job observes cancellation through the [`CancelToken`]
//! handed to its body.
//!
//! This is the eval worker's supersede model used by `terra-cpu-eval`: the app
//! bumps a token to cancel stale refine work, and the persistent evaluator + its
//! layer cache live across jobs. Panic containment is by construction — each job
//! body runs inside [`catch_unwind`](std::panic::catch_unwind), and a panic runs
//! the caller-supplied recovery hook to rebuild `S` (carrying any salvageable
//! parts across) before the next job.
//!
//! The executor stays generic: it never inspects a job's value `T`, so policies
//! like "a cancelled result publishes nothing" belong in the caller that knows
//! what `T` means, not here.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use crate::job::panic_payload_message;
use crate::{CancelToken, JobError};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LatestWinsStatsSnapshot {
    pub submitted: u64,
    pub stale_skipped: u64,
    pub started: u64,
    pub completed: u64,
    pub failed: u64,
}

#[derive(Default)]
struct LatestWinsStats {
    submitted: AtomicU64,
    stale_skipped: AtomicU64,
    started: AtomicU64,
    completed: AtomicU64,
    failed: AtomicU64,
}

impl LatestWinsStats {
    fn snapshot(&self) -> LatestWinsStatsSnapshot {
        LatestWinsStatsSnapshot {
            submitted: self.submitted.load(Ordering::Relaxed),
            stale_skipped: self.stale_skipped.load(Ordering::Relaxed),
            started: self.started.load(Ordering::Relaxed),
            completed: self.completed.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
        }
    }
}

/// A message to the worker thread.
enum Msg<Req> {
    // Boxed so the idle `Shutdown` slot and queued channel entries aren't sized
    // to a potentially large `Req` payload (clippy::large_enum_variant).
    Job { request: Box<Req>, token: u64 },
    Shutdown,
}

/// An event delivered by a [`LatestWins`] executor.
///
/// A stale job (superseded before it was dequeued) produces *no* event — it is
/// dropped on the worker thread. Everything that actually runs produces exactly
/// one of these:
///
/// * [`Completed`](Self::Completed) — the body returned `Ok(value)`.
/// * [`Failed`](Self::Failed) — the body returned `Err` or panicked; the original
///   `request` is handed back so the caller can attach domain context (a job's
///   token is generic, but anything else it needs lives in `Req`).
/// * [`Disconnected`](Self::Disconnected) — the worker thread ended and its event
///   channel closed. Emitted at most once.
#[derive(Debug)]
pub enum JobEvent<Req, T> {
    Completed {
        token: u64,
        value: T,
    },
    Failed {
        token: u64,
        request: Box<Req>,
        error: JobError,
    },
    Disconnected,
}

/// Enqueuing a job failed because the worker thread has ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubmitError;

impl std::fmt::Display for SubmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "job executor is disconnected")
    }
}

impl std::error::Error for SubmitError {}

/// A latest-generation-wins executor over a persistent worker thread.
///
/// `Req` is the job input, `T` the value a successful job produces. The
/// long-lived state `S` never leaves the worker thread, so it appears only on
/// [`spawn`](Self::spawn) and carries no `Send` requirement.
pub struct LatestWins<Req, T> {
    tx: Sender<Msg<Req>>,
    rx: Receiver<JobEvent<Req, T>>,
    /// The shared generation counter, owned jointly with the caller. `submit`
    /// stores into it; the worker reads it to stale-skip and to build cancel
    /// tokens.
    counter: Arc<AtomicU64>,
    stats: Arc<LatestWinsStats>,
    disconnected_reported: bool,
    _handle: JoinHandle<()>,
}

impl<Req, T> LatestWins<Req, T> {
    /// Spawn the worker thread and return the executor.
    ///
    /// * `name` names the thread.
    /// * `counter` is the caller-owned generation counter; keep a clone to
    ///   supersede in-flight work by storing a new value.
    /// * `state` builds the long-lived `S` **on the worker thread**.
    /// * `job` runs one request against `&mut S`, with a [`CancelToken`] that
    ///   trips when `counter` moves off this job's token.
    /// * `recover` rebuilds `S` after a job body panics, and is where salvageable
    ///   parts of the old state are carried across.
    pub fn spawn<S, State, Job, Recover>(
        name: &str,
        counter: Arc<AtomicU64>,
        state: State,
        mut job: Job,
        mut recover: Recover,
    ) -> Self
    where
        Req: Send + 'static,
        T: Send + 'static,
        S: 'static,
        State: FnOnce() -> S + Send + 'static,
        Job: FnMut(&mut S, &Req, &CancelToken) -> Result<T, JobError> + Send + 'static,
        Recover: FnMut(S) -> S + Send + 'static,
    {
        let (job_tx, job_rx) = mpsc::channel::<Msg<Req>>();
        let (event_tx, event_rx) = mpsc::channel::<JobEvent<Req, T>>();
        let worker_counter = Arc::clone(&counter);
        let stats = Arc::new(LatestWinsStats::default());
        let worker_stats = Arc::clone(&stats);

        let handle = thread::Builder::new()
            .name(name.to_string())
            .spawn(move || {
                let mut state = state();
                while let Ok(msg) = job_rx.recv() {
                    let Msg::Job { request, token } = msg else {
                        break; // Shutdown
                    };
                    // Stale-skip on dequeue: a newer submit already moved the
                    // generation on, so this job's output is unwanted. Drop it
                    // silently — no event — before doing any work.
                    if token != worker_counter.load(Ordering::Acquire) {
                        worker_stats.stale_skipped.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    worker_stats.started.fetch_add(1, Ordering::Relaxed);
                    let cancel = CancelToken::generation(Arc::clone(&worker_counter), token);
                    let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        job(&mut state, &request, &cancel)
                    }));
                    let event = match ran {
                        Ok(Ok(value)) => {
                            worker_stats.completed.fetch_add(1, Ordering::Relaxed);
                            JobEvent::Completed { token, value }
                        }
                        Ok(Err(error)) => JobEvent::Failed {
                            token,
                            request,
                            error,
                        },
                        Err(payload) => {
                            // Rebuild the state from the (possibly corrupted) old
                            // one before serving the next job.
                            state = recover(state);
                            JobEvent::Failed {
                                token,
                                request,
                                error: JobError::Panicked(panic_payload_message(payload)),
                            }
                        }
                    };
                    if matches!(event, JobEvent::Failed { .. }) {
                        worker_stats.failed.fetch_add(1, Ordering::Relaxed);
                    }
                    if event_tx.send(event).is_err() {
                        break; // Receiver gone; nothing more to deliver.
                    }
                }
            })
            .expect("spawn LatestWins worker thread");

        Self {
            tx: job_tx,
            rx: event_rx,
            counter,
            stats,
            disconnected_reported: false,
            _handle: handle,
        }
    }

    /// Enqueue `request` under `token`, storing `token` as the live generation.
    ///
    /// The store happens before the enqueue, so the token is the live generation
    /// even if the send fails. A [`SubmitError`] means the worker has ended.
    pub fn submit(&self, request: Req, token: u64) -> Result<(), SubmitError> {
        self.counter.store(token, Ordering::Release);
        let submitted = self
            .tx
            .send(Msg::Job {
                request: Box::new(request),
                token,
            })
            .map_err(|_| SubmitError);
        if submitted.is_ok() {
            self.stats.submitted.fetch_add(1, Ordering::Relaxed);
        }
        submitted
    }

    /// The shared generation counter. Store a new value to supersede in-flight and
    /// queued work without submitting.
    pub fn counter(&self) -> &Arc<AtomicU64> {
        &self.counter
    }

    pub fn stats(&self) -> LatestWinsStatsSnapshot {
        self.stats.snapshot()
    }

    /// Non-blocking poll for one event.
    ///
    /// [`Disconnected`](JobEvent::Disconnected) is returned at most once so a
    /// polling loop cannot flood on a closed channel.
    pub fn try_recv(&mut self) -> Option<JobEvent<Req, T>> {
        match self.rx.try_recv() {
            Ok(event) => Some(event),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) if !self.disconnected_reported => {
                self.disconnected_reported = true;
                Some(JobEvent::Disconnected)
            }
            Err(TryRecvError::Disconnected) => None,
        }
    }

    /// Ask the worker to stop after its current job. Idempotent; also runs on drop.
    pub fn shutdown(&self) {
        let _ = self.tx.send(Msg::Shutdown);
    }
}

impl<Req, T> Drop for LatestWins<Req, T> {
    fn drop(&mut self) {
        let _ = self.tx.send(Msg::Shutdown);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    /// Spin (yielding) until an event arrives, with a generous safety timeout so a
    /// regression can't hang the suite.
    fn wait_for_event<Req, T>(exec: &mut LatestWins<Req, T>) -> JobEvent<Req, T> {
        let start = Instant::now();
        loop {
            if let Some(event) = exec.try_recv() {
                return event;
            }
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "no event arrived within the timeout"
            );
            thread::yield_now();
        }
    }

    fn identity<S>(state: S) -> S {
        state
    }

    #[test]
    fn delivers_completed_with_token_and_value() {
        let counter = Arc::new(AtomicU64::new(0));
        let mut exec = LatestWins::<u64, u64>::spawn(
            "test-lw-complete",
            Arc::clone(&counter),
            || 0u64,
            |_state, req: &u64, _cancel| Ok(*req * 2),
            identity,
        );
        exec.submit(21, 1).expect("submit");
        match wait_for_event(&mut exec) {
            JobEvent::Completed { token, value } => {
                assert_eq!(token, 1);
                assert_eq!(value, 42);
            }
            other => panic!("expected Completed, got {other:?}"),
        }
    }

    #[test]
    fn stale_job_is_skipped_on_dequeue() {
        let counter = Arc::new(AtomicU64::new(0));
        // Block the worker in its state factory until both jobs are queued and the
        // generation has advanced, so the first job is provably stale on dequeue.
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let mut exec = LatestWins::<u64, u64>::spawn(
            "test-lw-stale",
            Arc::clone(&counter),
            move || {
                release_rx.recv().expect("release worker");
                0u64
            },
            |_state, req: &u64, _cancel| Ok(*req),
            identity,
        );

        exec.submit(100, 1).expect("submit stale"); // counter -> 1
        exec.submit(200, 2).expect("submit live"); // counter -> 2
        release_tx.send(()).expect("release");

        // Job token=1 is skipped (live is 2); only job token=2 runs.
        match wait_for_event(&mut exec) {
            JobEvent::Completed { token, value } => {
                assert_eq!(token, 2);
                assert_eq!(value, 200);
            }
            other => panic!("expected only the live job, got {other:?}"),
        }
        assert!(
            exec.try_recv().is_none(),
            "the stale job must not produce an event"
        );
        assert_eq!(
            exec.stats(),
            LatestWinsStatsSnapshot {
                submitted: 2,
                stale_skipped: 1,
                started: 1,
                completed: 1,
                failed: 0,
            }
        );
    }

    #[test]
    fn in_flight_supersede_is_visible_via_cancel_token() {
        let counter = Arc::new(AtomicU64::new(0));
        let (started_tx, started_rx) = mpsc::channel::<()>();
        let (gate_tx, gate_rx) = mpsc::channel::<()>();
        let mut exec = LatestWins::<u64, u64>::spawn(
            "test-lw-supersede",
            Arc::clone(&counter),
            || 0u64,
            move |_state, _req: &u64, cancel| {
                started_tx.send(()).expect("signal start");
                gate_rx.recv().expect("await release");
                if cancel.is_cancelled() {
                    Err(JobError::Cancelled)
                } else {
                    Ok(1)
                }
            },
            identity,
        );

        exec.submit(0, 1).expect("submit"); // counter -> 1, job runs
        started_rx.recv().expect("job started");
        counter.store(2, Ordering::Release); // supersede in-flight
        gate_tx.send(()).expect("release job");

        // A generic executor delivers the domain error; it does not suppress it.
        match wait_for_event(&mut exec) {
            JobEvent::Failed { token, error, .. } => {
                assert_eq!(token, 1);
                assert_eq!(error, JobError::Cancelled);
            }
            other => panic!("expected Failed(Cancelled), got {other:?}"),
        }
    }

    #[test]
    fn panic_runs_recovery_with_salvage_and_returns_request() {
        let counter = Arc::new(AtomicU64::new(0));
        // State is a one-element "disk": recovery carries the survivor across a
        // rebuild, standing in for the eval worker's disk-spill handoff.
        let mut exec = LatestWins::<&'static str, u64>::spawn(
            "test-lw-panic",
            Arc::clone(&counter),
            || vec![7u8],
            |state: &mut Vec<u8>, req: &&'static str, _cancel| {
                if *req == "boom" {
                    panic!("kaboom");
                }
                Ok(state[0] as u64)
            },
            |old: Vec<u8>| vec![old[0]], // carry the survivor across the rebuild
        );

        exec.submit("boom", 1).expect("submit panic");
        match wait_for_event(&mut exec) {
            JobEvent::Failed {
                token,
                request,
                error,
            } => {
                assert_eq!(token, 1);
                assert_eq!(*request, "boom");
                match error {
                    JobError::Panicked(message) => assert!(message.contains("kaboom")),
                    other => panic!("expected Panicked, got {other:?}"),
                }
            }
            other => panic!("expected Failed, got {other:?}"),
        }

        // The rebuilt state still holds the salvaged survivor.
        exec.submit("ok", 2).expect("submit after recovery");
        match wait_for_event(&mut exec) {
            JobEvent::Completed { token, value } => {
                assert_eq!(token, 2);
                assert_eq!(value, 7, "recovery must carry the survivor across");
            }
            other => panic!("expected Completed, got {other:?}"),
        }
    }

    #[test]
    fn stale_completion_is_still_delivered() {
        let counter = Arc::new(AtomicU64::new(0));
        let (started_tx, started_rx) = mpsc::channel::<()>();
        let (gate_tx, gate_rx) = mpsc::channel::<()>();
        // The body ignores its cancel token, so a job superseded after dequeue
        // runs to completion; the executor must deliver that stale completion.
        let mut exec = LatestWins::<u64, u64>::spawn(
            "test-lw-stale-completion",
            Arc::clone(&counter),
            || 0u64,
            move |_state, req: &u64, _cancel| {
                started_tx.send(()).expect("signal start");
                gate_rx.recv().expect("await release");
                Ok(*req)
            },
            identity,
        );

        exec.submit(500, 1).expect("submit"); // counter -> 1, job runs
        started_rx.recv().expect("job started");
        counter.store(2, Ordering::Release); // supersede, but the body ignores it
        gate_tx.send(()).expect("release job");

        match wait_for_event(&mut exec) {
            JobEvent::Completed { token, value } => {
                assert_eq!(token, 1, "the stale completion keeps its own token");
                assert_eq!(value, 500);
            }
            other => panic!("expected the stale Completed, got {other:?}"),
        }
    }

    #[test]
    fn disconnect_reported_once_then_submit_errors() {
        let counter = Arc::new(AtomicU64::new(0));
        let mut exec = LatestWins::<u64, u64>::spawn(
            "test-lw-disconnect",
            Arc::clone(&counter),
            || 0u64,
            |_state, req: &u64, _cancel| Ok(*req),
            identity,
        );
        exec.shutdown();

        // The worker ends and closes its event channel: Disconnected exactly once.
        let start = Instant::now();
        let mut saw_disconnect = false;
        while !saw_disconnect {
            match exec.try_recv() {
                Some(JobEvent::Disconnected) => saw_disconnect = true,
                Some(other) => panic!("expected Disconnected, got {other:?}"),
                None => {
                    assert!(
                        start.elapsed() < Duration::from_secs(5),
                        "worker never reported disconnection"
                    );
                    thread::yield_now();
                }
            }
        }
        assert!(
            exec.try_recv().is_none(),
            "disconnection must be reported only once"
        );

        // With the worker gone, submitting errors.
        assert_eq!(exec.submit(1, 1), Err(SubmitError));
    }
}
