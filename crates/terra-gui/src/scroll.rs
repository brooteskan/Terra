//! Scrollbar drawing + drag interaction for overflow panels.

use crate::context::GuiContext;
use crate::id::Id;
use crate::state::ScrollDrag;
use crate::style::{self, SCROLLBAR_PAD, SCROLLBAR_W};
use crate::types::Rect;

/// Vertical scrollbar on the right edge of `viewport`.
///
/// `content_h` is the full content height in logical px (including padding),
/// measured from `viewport.min_y`.
pub fn scrollbar_y(
    ui: &mut GuiContext<'_>,
    id: Id,
    viewport: Rect,
    content_h: f32,
    scroll_y: &mut f32,
) {
    scrollbar(ui, id, viewport, content_h, scroll_y, ScrollAxis::Vertical);
}

/// Horizontal scrollbar along the bottom of `viewport`.
pub fn scrollbar_x(
    ui: &mut GuiContext<'_>,
    id: Id,
    viewport: Rect,
    content_w: f32,
    scroll_x: &mut f32,
) {
    scrollbar(
        ui,
        id,
        viewport,
        content_w,
        scroll_x,
        ScrollAxis::Horizontal,
    );
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ScrollAxis {
    Horizontal,
    Vertical,
}

impl ScrollAxis {
    fn vertical(self) -> bool {
        self == Self::Vertical
    }

    fn viewport_extent(self, viewport: Rect) -> f32 {
        if self.vertical() {
            viewport.height()
        } else {
            viewport.width()
        }
    }

    fn track(self, viewport: Rect, view_extent: f32) -> Rect {
        if self.vertical() {
            Rect::from_pos_size(
                viewport.max_x - SCROLLBAR_W - SCROLLBAR_PAD,
                viewport.min_y + SCROLLBAR_PAD,
                SCROLLBAR_W,
                (view_extent - SCROLLBAR_PAD * 2.0).max(8.0),
            )
        } else {
            Rect::from_pos_size(
                viewport.min_x + SCROLLBAR_PAD,
                viewport.max_y - SCROLLBAR_W - SCROLLBAR_PAD,
                (view_extent - SCROLLBAR_PAD * 2.0).max(8.0),
                SCROLLBAR_W,
            )
        }
    }

    fn extent(self, rect: Rect) -> f32 {
        if self.vertical() {
            rect.height()
        } else {
            rect.width()
        }
    }

    fn start(self, rect: Rect) -> f32 {
        if self.vertical() {
            rect.min_y
        } else {
            rect.min_x
        }
    }

    fn pointer(self, pointer: (f32, f32)) -> f32 {
        if self.vertical() {
            pointer.1
        } else {
            pointer.0
        }
    }

    fn thumb(self, track: Rect, offset: f32, extent: f32) -> Rect {
        if self.vertical() {
            Rect::from_pos_size(track.min_x, track.min_y + offset, track.width(), extent)
        } else {
            Rect::from_pos_size(track.min_x + offset, track.min_y, extent, track.height())
        }
    }
}

fn scrollbar(
    ui: &mut GuiContext<'_>,
    id: Id,
    viewport: Rect,
    content_extent: f32,
    scroll: &mut f32,
    axis: ScrollAxis,
) {
    let view_extent = axis.viewport_extent(viewport).max(1.0);
    let content_extent = content_extent.max(view_extent);
    let max_scroll = (content_extent - view_extent).max(0.0);
    *scroll = scroll.clamp(0.0, max_scroll);
    if max_scroll < 1.0 {
        return;
    }

    let track = axis.track(viewport, view_extent);
    let track_extent = axis.extent(track);
    // Never clamp with min > max — short panels can have a very short track.
    let minimum_thumb = if axis.vertical() { 22.0_f32 } else { 28.0_f32 }.min(track_extent);
    let thumb_extent =
        (track_extent * (view_extent / content_extent)).clamp(minimum_thumb, track_extent);
    let travel = (track_extent - thumb_extent).max(0.0);
    let mut thumb = axis.thumb(track, travel * (*scroll / max_scroll), thumb_extent);

    let thumb_id = id.child("thumb");
    let track_id = id.child("track");

    // Continue thumb drag only while the primary button stays down.
    if let Some(drag) = ui.state.scroll_drag {
        if drag.id == id && drag.vertical == axis.vertical() {
            if !ui.input.primary_down {
                ui.state.scroll_drag = None;
                if ui.state.is_active(thumb_id) {
                    ui.state.active = None;
                }
            } else if let Some(pointer) = ui.input.pointer {
                let delta = axis.pointer(pointer) - drag.start_pointer;
                *scroll =
                    (drag.start_scroll + delta * drag.scroll_per_pixel).clamp(0.0, max_scroll);
                thumb = axis.thumb(track, travel * (*scroll / max_scroll), thumb_extent);
                ui.state.set_hot(thumb_id);
                ui.state.active = Some(thumb_id);
            }
        }
    }

    let pointer_in_track = ui
        .input
        .pointer
        .map(|(x, y)| track.contains(x, y))
        .unwrap_or(false);
    let pointer_in_thumb = ui
        .input
        .pointer
        .map(|(x, y)| thumb.contains(x, y))
        .unwrap_or(false);

    if pointer_in_thumb {
        ui.state.set_hot(thumb_id);
    } else if pointer_in_track {
        ui.state.set_hot(track_id);
    }

    if pointer_in_thumb && ui.input.primary_pressed {
        ui.state.active = Some(thumb_id);
        if let Some(pointer) = ui.input.pointer {
            ui.state.scroll_drag = Some(ScrollDrag {
                id,
                vertical: axis.vertical(),
                start_scroll: *scroll,
                start_pointer: axis.pointer(pointer),
                scroll_per_pixel: if travel > 0.5 {
                    max_scroll / travel
                } else {
                    0.0
                },
            });
        }
    } else if pointer_in_track && ui.input.primary_pressed && !pointer_in_thumb {
        // Jump so the thumb centers on the click.
        if let Some(pointer) = ui.input.pointer {
            let center = axis.pointer(pointer) - thumb_extent * 0.5;
            let t = ((center - axis.start(track)) / travel.max(1.0)).clamp(0.0, 1.0);
            *scroll = t * max_scroll;
            ui.state.active = Some(thumb_id);
            ui.state.scroll_drag = Some(ScrollDrag {
                id,
                vertical: axis.vertical(),
                start_scroll: *scroll,
                start_pointer: axis.pointer(pointer),
                scroll_per_pixel: if travel > 0.5 {
                    max_scroll / travel
                } else {
                    0.0
                },
            });
        }
    }

    let thumb_hot = ui.state.is_hot(thumb_id) || ui.state.is_active(thumb_id);
    ui.panel(track, style::SCROLLBAR_TRACK);
    ui.panel(
        thumb,
        if thumb_hot {
            style::SCROLLBAR_THUMB_HOVER
        } else {
            style::SCROLLBAR_THUMB
        },
    );
}
