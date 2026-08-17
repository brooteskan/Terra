//! A per-frame poll registry: pump every background subsystem once and
//! aggregate its wakefulness.
//!
//! Phases 2–4 moved each background subsystem onto a terra-jobs executor, but
//! the winit `about_to_wait` loop still polled them one by one and assembled
//! "is anything pending?" by hand from a handful of ad-hoc reads. [`JobRegistry`]
//! collapses that into one [`tick`](JobRegistry::tick): it pumps each registered
//! [`Pollable`] once and OR-folds the [`Pending`] facts they report into a single
//! [`Tick`].
//!
//! **Facts only, never typed results.** The registry aggregates pending/wake
//! booleans; it deliberately does not own typed result handling. Unifying
//! heterogeneous result types (eval results, export outcomes, loaded documents)
//! into one event enum here would be over-abstraction — each subsystem's `pump`
//! advances its own poll surface and reports booleans, and the caller drains the
//! typed results exactly where it always has, with the concrete types in hand.
//!
//! **Ownership stays with the host.** Rather than move subsystems into the
//! registry — which would force every non-tick access (`exporter.start(…)`,
//! `job.result.take()`) through a downcast — entries are *projection functions*
//! (`fn(&mut Host) -> &mut dyn Pollable`), registered once. Each is a zero-sized
//! function pointer with no captured state, so the subsystems remain plain fields
//! on the host and `tick` merely borrows each in turn to pump it.
//!
//! **What is intentionally not registered.** Not every executor belongs here.
//! Progress/refine executors whose "pending" is really app policy state
//! (`pending_eval`, `refining`) and whose drain needs typed access to the host
//! stay app-side; fire-and-forget workers (a debounced settings saver) never
//! influence wakefulness and are absent too. The registry is for subsystems whose
//! only tick-time contribution is "am I busy / should we wake / should we
//! repaint".

/// A projection from the host to one of its [`Pollable`] subsystems.
///
/// A bare `fn` pointer, so it captures nothing and the subsystem it points at
/// stays owned by the host.
type Project<Host> = fn(&mut Host) -> &mut dyn Pollable;

/// Wakefulness facts one [`Pollable`] reports from a single
/// [`pump`](Pollable::pump).
///
/// All-false is idle: no queued or in-flight work, and nothing changed this pump.
#[derive(Debug, Clone, Copy, Default)]
pub struct Pending {
    /// Work is queued or executing. While any subsystem is busy the event loop
    /// must not fall through to `ControlFlow::Wait`.
    pub busy: bool,
    /// This subsystem wants animation-cadence wakes (~16 ms) while busy — a
    /// progress bar to advance, a warmup to animate in. Plain `busy` work (an
    /// export streaming to disk) leaves this false and is served by the caller's
    /// own debounce/refine cadence instead.
    pub animate: bool,
    /// Something observable changed this pump (a completion latch fired, a
    /// progress value moved) and the frame should repaint.
    pub redraw: bool,
}

/// A background subsystem the [`JobRegistry`] can pump once per frame.
///
/// `pump` is called exactly once per [`tick`](JobRegistry::tick), in registration
/// order. Implementations advance their poll surface (drain a channel, read an
/// atomic, swap a latch) and report [`Pending`] facts. Because a pump may have
/// side effects — a drained ready-latch reports "completed" to exactly one
/// caller — the once-per-tick, in-order contract is load-bearing.
pub trait Pollable {
    /// Advance this subsystem's poll surface once and report its facts.
    fn pump(&mut self) -> Pending;
}

/// The OR-fold of every entry's [`Pending`] facts across one
/// [`tick`](JobRegistry::tick).
#[derive(Debug, Clone, Copy, Default)]
pub struct Tick {
    /// Any registered subsystem is busy.
    pub any_pending: bool,
    /// Any busy subsystem wants animation-cadence wakes.
    pub animate: bool,
    /// Any subsystem changed this tick and the frame should repaint.
    pub redraw: bool,
}

/// A fixed set of background subsystems, pumped together once per frame.
///
/// Generic over the `Host` that owns the subsystems (Terra's `TerraApp`). Holds
/// only projection functions, so it carries no subsystem state of its own and is
/// cheap to keep behind a shared handle.
pub struct JobRegistry<Host> {
    entries: Vec<Project<Host>>,
}

impl<Host> JobRegistry<Host> {
    /// An empty registry. Add subsystems with [`register`](Self::register).
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Register one subsystem by a projection from the host to it. Entries are
    /// pumped in registration order.
    pub fn register(&mut self, project: Project<Host>) {
        self.entries.push(project);
    }

    /// Pump every registered subsystem once, in order, and OR-fold their facts.
    ///
    /// Takes `&self`: the entry list never changes after construction, which lets
    /// a caller hold the registry behind a shared handle and still pass its own
    /// `&mut Host` here without a borrow conflict.
    pub fn tick(&self, host: &mut Host) -> Tick {
        let mut acc = Tick::default();
        for project in &self.entries {
            let facts = project(host).pump();
            acc.any_pending |= facts.busy;
            acc.animate |= facts.animate;
            acc.redraw |= facts.redraw;
        }
        acc
    }
}

impl<Host> Default for JobRegistry<Host> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// A pollable that returns scripted facts and logs each pump into a shared
    /// order log, so tests can assert once-per-tick, in-order pumping.
    struct Probe {
        name: &'static str,
        facts: Pending,
        pumps: usize,
        log: Rc<RefCell<Vec<&'static str>>>,
    }

    impl Probe {
        fn new(name: &'static str, facts: Pending, log: &Rc<RefCell<Vec<&'static str>>>) -> Self {
            Self {
                name,
                facts,
                pumps: 0,
                log: Rc::clone(log),
            }
        }
    }

    impl Pollable for Probe {
        fn pump(&mut self) -> Pending {
            self.pumps += 1;
            self.log.borrow_mut().push(self.name);
            self.facts
        }
    }

    struct Host {
        a: Probe,
        b: Probe,
    }

    #[test]
    fn pumps_each_entry_once_per_tick_in_registration_order() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut host = Host {
            a: Probe::new("a", Pending::default(), &log),
            b: Probe::new("b", Pending::default(), &log),
        };
        let mut reg = JobRegistry::<Host>::new();
        reg.register(|h| &mut h.a);
        reg.register(|h| &mut h.b);

        reg.tick(&mut host);
        assert_eq!(*log.borrow(), vec!["a", "b"]);
        assert_eq!(host.a.pumps, 1);
        assert_eq!(host.b.pumps, 1);

        reg.tick(&mut host);
        assert_eq!(*log.borrow(), vec!["a", "b", "a", "b"], "order repeats");
        assert_eq!(host.a.pumps, 2);
        assert_eq!(host.b.pumps, 2);
    }

    #[test]
    fn facts_or_fold_across_entries() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut host = Host {
            a: Probe::new(
                "a",
                Pending {
                    busy: true,
                    animate: false,
                    redraw: false,
                },
                &log,
            ),
            b: Probe::new(
                "b",
                Pending {
                    busy: false,
                    animate: true,
                    redraw: true,
                },
                &log,
            ),
        };
        let mut reg = JobRegistry::<Host>::new();
        reg.register(|h| &mut h.a);
        reg.register(|h| &mut h.b);

        let tick = reg.tick(&mut host);
        assert!(tick.any_pending, "a is busy");
        assert!(tick.animate, "b wants animation");
        assert!(tick.redraw, "b changed");
    }

    #[test]
    fn empty_registry_reports_all_false_and_pumps_nothing() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut host = Host {
            a: Probe::new("a", Pending::default(), &log),
            b: Probe::new("b", Pending::default(), &log),
        };
        let reg = JobRegistry::<Host>::new();

        let tick = reg.tick(&mut host);
        assert!(!tick.any_pending);
        assert!(!tick.animate);
        assert!(!tick.redraw);
        assert_eq!(host.a.pumps, 0, "no entries registered — nothing pumped");
        assert!(log.borrow().is_empty());
    }

    /// A pollable modelling a drained latch: `redraw` true on the first pump,
    /// false thereafter — the tool-thumbnail ready-signal shape.
    struct Latch {
        ready: bool,
    }

    impl Pollable for Latch {
        fn pump(&mut self) -> Pending {
            let redraw = self.ready;
            self.ready = false;
            Pending {
                busy: false,
                animate: false,
                redraw,
            }
        }
    }

    struct LatchHost {
        latch: Latch,
    }

    #[test]
    fn a_drained_latch_reports_redraw_on_exactly_one_tick() {
        let mut host = LatchHost {
            latch: Latch { ready: true },
        };
        let mut reg = JobRegistry::<LatchHost>::new();
        reg.register(|h| &mut h.latch);

        assert!(reg.tick(&mut host).redraw, "first tick observes the latch");
        assert!(
            !reg.tick(&mut host).redraw,
            "latch drained on the first tick — not reported again"
        );
    }
}
