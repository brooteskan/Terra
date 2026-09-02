//! A coalescing single-worker executor: latest value wins.
//!
//! [`Debounced`] owns one named thread running a caller-supplied `work(T)`. Each
//! [`submit`](Debounced::submit) enqueues a `T`, but before every run the worker
//! drains the queue to its most recent entry — so a fast burst of submissions
//! collapses into one run against the last value. This is exactly the shape of a
//! settings saver fed by a slider drag: dozens of snapshots per second, but only
//! the final one needs to reach disk, and none should sit on the UI thread.
//!
//! Panic containment is by construction: `work` runs inside
//! [`catch_unwind`](std::panic::catch_unwind), so a panicking run neither ends
//! the worker nor wedges the executor — a later `submit` is still serviced. The
//! `work` closure is reused as-is across a contained panic (via
//! [`AssertUnwindSafe`](std::panic::AssertUnwindSafe)); it carries no recovery
//! hook, so state that must be rebuilt after a panic belongs in
//! [`LatestWins`](crate::LatestWins) instead. The default panic hook still prints
//! the payload to stderr, so the panic is contained, not hidden.

use std::sync::mpsc;
use std::thread::{self, JoinHandle};

/// A latest-value-wins executor over a single persistent worker thread.
///
/// Dropping it drops the [`Sender`](mpsc::Sender); the worker then processes any
/// already-queued values (coalescing them as usual) and exits once the channel
/// reports closed.
pub struct Debounced<T> {
    tx: mpsc::Sender<T>,
    _handle: JoinHandle<()>,
}

impl<T: Send + 'static> Debounced<T> {
    /// Spawn the worker thread named `name`, running `work` against the latest
    /// submitted value each time it wakes.
    pub fn spawn<W>(name: &str, mut work: W) -> Self
    where
        W: FnMut(T) + Send + 'static,
    {
        let (tx, rx) = mpsc::channel::<T>();
        let handle = thread::Builder::new()
            .name(name.to_string())
            .spawn(move || {
                while let Ok(mut latest) = rx.recv() {
                    // Coalesce: skip ahead to the newest queued value so a burst
                    // runs `work` once, on the last submission.
                    while let Ok(newer) = rx.try_recv() {
                        latest = newer;
                    }
                    // Contain the run so a panic doesn't end the worker: the next
                    // submit is still serviced.
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        work(latest);
                    }));
                }
            })
            .expect("spawn debounced worker thread");

        Self {
            tx,
            _handle: handle,
        }
    }

    /// Enqueue `value`. Fire-and-forget: if a burst is queued, only the most
    /// recent value survives to the next run. Dropping the executor after the
    /// last sender would make this a no-op, but the executor holds the sender, so
    /// a send only fails if the worker thread has already ended.
    pub fn submit(&self, value: T) {
        let _ = self.tx.send(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    /// Spin (yielding) until `cond` holds, with a generous safety timeout.
    fn wait_until(mut cond: impl FnMut() -> bool) {
        let start = Instant::now();
        while !cond() {
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "debounced worker did not reach the expected state within the timeout"
            );
            thread::yield_now();
        }
    }

    #[test]
    fn burst_coalesces_to_latest() {
        let runs = Arc::new(Mutex::new(Vec::<u32>::new()));
        let (started_tx, started_rx) = mpsc::channel::<()>();
        let (gate_tx, gate_rx) = mpsc::channel::<()>();

        let worker_runs = Arc::clone(&runs);
        let mut first = true;
        let debounced = Debounced::spawn("test-debounce-coalesce", move |value: u32| {
            if first {
                first = false;
                // Block the first run so the burst below queues up behind it.
                started_tx.send(()).expect("signal first run started");
                gate_rx.recv().expect("await release");
            }
            worker_runs.lock().expect("runs mutex").push(value);
        });

        debounced.submit(1);
        started_rx.recv().expect("first run started");

        // These queue while run #1 is gated; they must collapse into one run.
        for v in 2..=10 {
            debounced.submit(v);
        }
        gate_tx.send(()).expect("release first run");

        wait_until(|| runs.lock().expect("runs mutex").len() == 2);
        // Exactly two runs: value 1 (which triggered run #1), then the coalesced
        // burst delivering only the last value, 10.
        assert_eq!(*runs.lock().expect("runs mutex"), vec![1, 10]);
    }

    #[test]
    fn panicking_run_keeps_worker_alive() {
        const SENTINEL: u32 = 0;
        let attempts = Arc::new(AtomicUsize::new(0));
        let last = Arc::new(AtomicUsize::new(usize::MAX));

        let worker_attempts = Arc::clone(&attempts);
        let worker_last = Arc::clone(&last);
        let debounced = Debounced::spawn("test-debounce-panic", move |value: u32| {
            worker_attempts.fetch_add(1, Ordering::AcqRel);
            if value == SENTINEL {
                panic!("boom");
            }
            worker_last.store(value as usize, Ordering::Release);
        });

        debounced.submit(SENTINEL);
        // Wait until the sentinel run has been attempted (and thus panicked and
        // been contained) before submitting the next value — this guarantees the
        // two are separate runs rather than coalesced into one.
        wait_until(|| attempts.load(Ordering::Acquire) >= 1);

        debounced.submit(42);
        wait_until(|| last.load(Ordering::Acquire) == 42);
        // The worker survived the panic and serviced the later submit.
        assert_eq!(attempts.load(Ordering::Acquire), 2);
    }
}
