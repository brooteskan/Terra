//! Ordered OS-input accumulation and immutable logical-frame snapshots.

use std::time::Instant;

use winit::event::{ElementState, MouseButton};
use winit::keyboard::KeyCode;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct InputModifiers {
    pub(crate) shift: bool,
    pub(crate) alt: bool,
    pub(crate) ctrl: bool,
    pub(crate) super_key: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum PointerCancelReason {
    FocusLost,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum InputEvent {
    Keyboard {
        code: Option<KeyCode>,
        state: ElementState,
    },
    Modifiers(InputModifiers),
    PointerButton {
        state: ElementState,
        button: MouseButton,
    },
    PointerMoved {
        x: f64,
        y: f64,
    },
    Wheel {
        delta: f32,
    },
    Focused(bool),
    CursorEntered,
    CursorLeft,
    PointerCancelled(PointerCancelReason),
}

impl InputEvent {
    fn is_pointer_sample(self) -> bool {
        matches!(
            self,
            Self::PointerButton { .. } | Self::PointerMoved { .. } | Self::PointerCancelled(_)
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct StampedInputEvent {
    sequence: u64,
    received_at: Instant,
    event: InputEvent,
}

impl StampedInputEvent {
    pub(crate) const fn sequence(&self) -> u64 {
        self.sequence
    }

    pub(crate) const fn received_at(&self) -> Instant {
        self.received_at
    }

    pub(crate) const fn event(&self) -> InputEvent {
        self.event
    }
}

#[derive(Debug, Default)]
pub(crate) struct InputAccumulator {
    next_sequence: u64,
    pending: Vec<StampedInputEvent>,
}

impl InputAccumulator {
    pub(crate) fn record(&mut self, event: InputEvent, received_at: Instant) {
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .expect("input sequence exhausted");
        self.pending.push(StampedInputEvent {
            sequence: self.next_sequence,
            received_at,
            event,
        });
    }

    pub(crate) fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    pub(crate) fn seal(&mut self) -> InputSnapshot {
        InputSnapshot {
            events: std::mem::take(&mut self.pending),
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct InputSnapshot {
    events: Vec<StampedInputEvent>,
}

impl InputSnapshot {
    pub(crate) fn events(&self) -> &[StampedInputEvent] {
        &self.events
    }

    pub(crate) fn len(&self) -> usize {
        self.events.len()
    }

    pub(crate) fn pointer_sample_count(&self) -> usize {
        self.events
            .iter()
            .filter(|event| event.event.is_pointer_sample())
            .count()
    }

    pub(crate) fn first_primary_press_receipt(&self) -> Option<Instant> {
        self.events.iter().find_map(|event| {
            matches!(
                event.event,
                InputEvent::PointerButton {
                    state: ElementState::Pressed,
                    button: MouseButton::Left,
                }
            )
            .then_some(event.received_at)
        })
    }

    pub(crate) fn has_primary_pointer_edge(&self) -> bool {
        self.events.iter().any(|event| {
            matches!(
                event.event,
                InputEvent::PointerButton {
                    button: MouseButton::Left,
                    ..
                }
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moved(x: f64) -> InputEvent {
        InputEvent::PointerMoved { x, y: 0.0 }
    }

    #[test]
    fn sealing_preserves_order_without_loss_or_duplication() {
        let now = Instant::now();
        let mut input = InputAccumulator::default();
        input.record(
            InputEvent::PointerButton {
                state: ElementState::Pressed,
                button: MouseButton::Left,
            },
            now,
        );
        input.record(moved(1.0), now);
        input.record(moved(2.0), now);
        input.record(
            InputEvent::PointerButton {
                state: ElementState::Released,
                button: MouseButton::Left,
            },
            now,
        );
        let snapshot = input.seal();
        assert_eq!(snapshot.len(), 4);
        assert_eq!(snapshot.pointer_sample_count(), 4);
        assert_eq!(
            snapshot
                .events()
                .iter()
                .map(StampedInputEvent::sequence)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
        assert!(!input.has_pending());
    }

    #[test]
    fn events_after_seal_are_retained_for_the_next_snapshot() {
        let now = Instant::now();
        let mut input = InputAccumulator::default();
        input.record(moved(1.0), now);
        let first = input.seal();
        input.record(moved(2.0), now);
        let second = input.seal();
        assert_eq!(first.events()[0].sequence(), 1);
        assert_eq!(second.events()[0].sequence(), 2);
    }

    #[test]
    fn frame_boundaries_do_not_change_pointer_sample_order() {
        let now = Instant::now();
        let mut input = InputAccumulator::default();
        input.record(
            InputEvent::PointerButton {
                state: ElementState::Pressed,
                button: MouseButton::Left,
            },
            now,
        );
        input.record(moved(1.0), now);
        let first = input.seal();
        input.record(moved(2.0), now);
        input.record(
            InputEvent::PointerButton {
                state: ElementState::Released,
                button: MouseButton::Left,
            },
            now,
        );
        let second = input.seal();
        let all = first
            .events()
            .iter()
            .chain(second.events())
            .map(StampedInputEvent::sequence)
            .collect::<Vec<_>>();
        assert_eq!(all, vec![1, 2, 3, 4]);
    }

    #[test]
    fn modifier_focus_and_cancellation_order_is_explicit() {
        let now = Instant::now();
        let mut input = InputAccumulator::default();
        input.record(
            InputEvent::Modifiers(InputModifiers {
                shift: true,
                ..InputModifiers::default()
            }),
            now,
        );
        input.record(InputEvent::Focused(false), now);
        input.record(
            InputEvent::PointerCancelled(PointerCancelReason::FocusLost),
            now,
        );
        let snapshot = input.seal();
        assert!(matches!(
            snapshot.events()[0].event(),
            InputEvent::Modifiers(InputModifiers { shift: true, .. })
        ));
        assert_eq!(snapshot.events()[1].event(), InputEvent::Focused(false));
        assert_eq!(
            snapshot.events()[2].event(),
            InputEvent::PointerCancelled(PointerCancelReason::FocusLost)
        );
    }
}
