//! A fixed-size worker pool over one shared queue.
//!
//! [`Pool`] spawns `worker_count` named threads that drain a single
//! `mpsc` queue, so a burst of fire-and-forget [`submit`](Pool::submit) calls is
//! spread across the workers. Unlike [`spawn_one_shot`](crate::spawn_one_shot)
//! there is no per-job handle: jobs return `()` and are polled in aggregate via
//! [`pending`](Pool::pending) (queued + executing) and
//! [`take_ready_signal`](Pool::take_ready_signal) (a drained latch that reports
//! "at least one job finished since you last asked"). This is the shape the
//! tool-thumbnail decoder needs — many small independent decodes feeding a UI
//! that only wants to know *whether* to redraw.
//!
//! Panic containment is by construction: each job runs inside
//! [`catch_unwind`](std::panic::catch_unwind), so a panicking job neither
//! unwinds its worker nor poisons the shared receiver mutex, and — critically —
//! never skips the pending-count decrement. A hand-rolled pool that let the
//! panic through would leave the count stuck above zero forever, which is a real
//! bug when a winit loop ties its wake cadence to "is background work pending".
//! The default panic hook still prints the payload to stderr before
//! `catch_unwind` returns, so the panic is contained, not hidden.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};

/// A queued unit of work. Boxed so the channel entry isn't sized to the largest
/// possible closure, and aliased to stay clear of `clippy::type_complexity`.
type PoolJob = Box<dyn FnOnce() + Send + 'static>;

/// A fixed worker count draining one queue, with poll-friendly aggregate signals.
///
/// Dropping the pool drops its [`Sender`](mpsc::Sender); the workers then finish
/// whatever is already queued and exit (mpsc reports the channel closed once the
/// last sender is gone). There is no join on drop — matching the detach-on-drop
/// semantics of the crate's other executors.
pub struct Pool {
    tx: mpsc::Sender<PoolJob>,
    /// Queued **or** executing jobs. Incremented on `submit`, decremented after a
    /// job returns (panic or not).
    pending: Arc<AtomicUsize>,
    /// Set when any job completes; a poll swaps it back to false. Lets the caller
    /// decide to redraw without tracking individual jobs.
    ready: Arc<AtomicBool>,
    _handles: Vec<JoinHandle<()>>,
}

impl Pool {
    /// Spawn a pool of `worker_count` threads named `"{name}-{i}"`.
    ///
    /// Panics if `worker_count` is zero — a pool with no workers would silently
    /// accept jobs that never run.
    pub fn new(name: &str, worker_count: usize) -> Self {
        assert!(worker_count > 0, "Pool needs at least one worker");

        let (tx, rx) = mpsc::channel::<PoolJob>();
        let rx = Arc::new(Mutex::new(rx));
        let pending = Arc::new(AtomicUsize::new(0));
        let ready = Arc::new(AtomicBool::new(false));

        let handles = (0..worker_count)
            .map(|i| {
                let rx = Arc::clone(&rx);
                let pending = Arc::clone(&pending);
                let ready = Arc::clone(&ready);
                thread::Builder::new()
                    .name(format!("{name}-{i}"))
                    .spawn(move || loop {
                        // Hold the receiver mutex only across `recv`, then drop the
                        // guard before running the job. A job never executes under
                        // the lock, so a panicking job can't poison this mutex —
                        // hence the `.expect` rather than a defensive bail-out.
                        let job = {
                            let guard = rx.lock().expect("pool receiver mutex poisoned");
                            guard.recv()
                        };
                        let Ok(job) = job else {
                            break; // all senders dropped; channel closed
                        };
                        // Contain the job so a panic neither unwinds this worker
                        // nor skips the bookkeeping below.
                        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
                        // Order matters: publish `ready` before decrementing
                        // `pending`. A consumer that polls the ready latch *before*
                        // the pending count each frame (as the winit loop does)
                        // must never observe `pending == 0` in a frame whose ready
                        // poll returned false — that would drop the last redraw and
                        // leave the final result unpainted until the next input.
                        ready.store(true, Ordering::Release);
                        pending.fetch_sub(1, Ordering::AcqRel);
                    })
                    .expect("spawn pool worker thread")
            })
            .collect();

        Self {
            tx,
            pending,
            ready,
            _handles: handles,
        }
    }

    /// Enqueue a job to run on some worker. Fire-and-forget: the job returns `()`
    /// and its panic (if any) is contained and logged by the default hook.
    pub fn submit(&self, job: impl FnOnce() + Send + 'static) {
        // Count the job before it can be dequeued, so `pending` never dips below
        // the true in-flight total. Roll back only if the send fails (unreachable
        // while the pool is alive, since it holds the workers and the receiver).
        self.pending.fetch_add(1, Ordering::AcqRel);
        if self.tx.send(Box::new(job)).is_err() {
            self.pending.fetch_sub(1, Ordering::AcqRel);
        }
    }

    /// Jobs currently queued or executing. Returns to `0` once the pool is idle.
    pub fn pending(&self) -> usize {
        self.pending.load(Ordering::Acquire)
    }

    /// True if one or more jobs completed since the last call, then clears.
    ///
    /// A drained latch (swap-false), so a single completed job is reported to
    /// exactly one poll and coalesces bursts into one "please redraw" answer.
    pub fn take_ready_signal(&self) -> bool {
        self.ready.swap(false, Ordering::AcqRel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use std::time::{Duration, Instant};

    /// Spin (yielding) until `cond` holds, with a generous safety timeout so a
    /// regression can't hang the suite. Timing is never relied on for
    /// correctness — only as an upper bound.
    fn wait_until(mut cond: impl FnMut() -> bool) {
        let start = Instant::now();
        while !cond() {
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "pool did not reach the expected state within the timeout"
            );
            thread::yield_now();
        }
    }

    #[test]
    fn completes_jobs_across_workers_and_drains_pending() {
        let pool = Pool::new("test-pool-spread", 4);

        // A Barrier(4) can only be cleared if four jobs run at once, proving the
        // work is genuinely spread across four workers (a smaller pool deadlocks
        // and trips the wait_until timeout).
        let barrier = Arc::new(Barrier::new(4));
        let counter = Arc::new(AtomicUsize::new(0));
        for _ in 0..4 {
            let barrier = Arc::clone(&barrier);
            let counter = Arc::clone(&counter);
            pool.submit(move || {
                barrier.wait();
                counter.fetch_add(1, Ordering::AcqRel);
            });
        }
        wait_until(|| pool.pending() == 0);
        assert_eq!(counter.load(Ordering::Acquire), 4);
    }

    #[test]
    fn panic_is_contained_and_pending_stays_correct() {
        // One worker so a killed thread would be fatal to the jobs after the
        // panic: this is the regression guard for the stuck-pending bug.
        let pool = Pool::new("test-pool-panic", 1);
        let counter = Arc::new(AtomicUsize::new(0));

        pool.submit(|| panic!("boom"));
        for _ in 0..5 {
            let counter = Arc::clone(&counter);
            pool.submit(move || {
                counter.fetch_add(1, Ordering::AcqRel);
            });
        }

        wait_until(|| pool.pending() == 0);
        // The five normal jobs ran on the same worker despite the earlier panic,
        // so the worker survived and the count fully drained.
        assert_eq!(counter.load(Ordering::Acquire), 5);
        assert!(
            pool.take_ready_signal(),
            "a completed job must set the latch"
        );
    }

    #[test]
    fn ready_signal_is_a_drained_latch() {
        let pool = Pool::new("test-pool-latch", 2);
        assert!(!pool.take_ready_signal(), "nothing has completed yet");

        pool.submit(|| {});
        wait_until(|| pool.pending() == 0);
        assert!(pool.take_ready_signal(), "one job completed");
        assert!(
            !pool.take_ready_signal(),
            "the latch drained on the last poll"
        );
    }

    #[test]
    fn pending_counts_queued_and_executing() {
        let pool = Pool::new("test-pool-pending", 1);
        let counter = Arc::new(AtomicUsize::new(0));

        // Gate the first job so it stays executing while the rest queue behind it.
        let (release_tx, release_rx) = mpsc::channel::<()>();
        pool.submit(move || {
            release_rx.recv().expect("release gate");
        });
        for _ in 0..3 {
            let counter = Arc::clone(&counter);
            pool.submit(move || {
                counter.fetch_add(1, Ordering::AcqRel);
            });
        }

        // One executing (blocked) + three queued.
        assert_eq!(pool.pending(), 4);

        release_tx.send(()).expect("send release");
        wait_until(|| pool.pending() == 0);
        assert_eq!(counter.load(Ordering::Acquire), 3);
    }
}
