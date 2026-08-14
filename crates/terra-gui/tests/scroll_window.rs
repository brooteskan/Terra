//! Reproduction: does the "Viewport Rendering" window scroll when its content
//! overflows (advanced options expanded)?
//!
//! Faithfully mirrors how the app draws that window: inside `begin_overlay`
//! (the viewport mode bar) and `with_menu_input` (the open-menu wrapper), with a
//! `begin_window` whose content is taller than the window.

use terra_gui::{GuiContext, GuiInput, GuiState, Id, Rect};

const SW: f32 = 900.0;
const SH: f32 = 700.0;

/// One frame of the render window with `n_rows` fixed-height rows of content.
/// Returns the scroll id used internally so tests can read `scroll_max`.
fn render_window_frame(
    state: &mut GuiState,
    input: GuiInput,
    win: Rect,
    open: &mut bool,
    scroll_y: &mut f32,
    n_rows: usize,
) {
    let mut ctx = GuiContext::begin(SW, SH, 1.0, input, state);
    // Mode bar draws viewport chrome inside an overlay layer …
    ctx.begin_overlay();
    // … and the open render menu is drawn through `with_menu_input`.
    ctx.with_menu_input(|ctx| {
        if ctx.begin_window(Id::new("vr"), "Viewport Rendering", win, open, scroll_y) {
            for _ in 0..n_rows {
                let _ = ctx.allocate(32.0);
            }
            ctx.end_window(scroll_y);
        }
    });
    ctx.end_overlay();
    ctx.end();
}

fn scroll_id() -> Id {
    Id::new("vr").child("scroll")
}

#[test]
fn advanced_window_overflows_and_reports_scroll_range() {
    // 520-tall window matches the app's advanced-open menu height.
    let win = Rect::from_pos_size(300.0, 60.0, 300.0, 520.0);
    let center = (win.center_x(), win.center_y());
    let mut state = GuiState::default();
    let mut open = true;
    let mut scroll_y = 0.0;

    // 20 rows * 32 = 640 px content vs ~492 px viewport → must overflow.
    render_window_frame(
        &mut state,
        GuiInput {
            pointer: Some(center),
            ..Default::default()
        },
        win,
        &mut open,
        &mut scroll_y,
        20,
    );

    let max = state.scroll_max.get(&scroll_id().0).copied().unwrap_or(0.0);
    assert!(
        max > 1.0,
        "overflowing advanced content should report a positive scroll range, got {max}"
    );
}

#[test]
fn advanced_window_wheel_scrolls_content() {
    let win = Rect::from_pos_size(300.0, 60.0, 300.0, 520.0);
    let center = (win.center_x(), win.center_y());
    let mut state = GuiState::default();
    let mut open = true;
    let mut scroll_y = 0.0;

    // Frame 1: settle so `scroll_max` is known for the wheel gate next frame.
    render_window_frame(
        &mut state,
        GuiInput {
            pointer: Some(center),
            ..Default::default()
        },
        win,
        &mut open,
        &mut scroll_y,
        20,
    );

    // Frame 2: wheel down (negative delta scrolls content down) over the content.
    render_window_frame(
        &mut state,
        GuiInput {
            pointer: Some(center),
            scroll_delta: -3.0,
            ..Default::default()
        },
        win,
        &mut open,
        &mut scroll_y,
        20,
    );

    assert!(
        scroll_y > 0.0,
        "wheel over the overflowing window should move content, scroll_y = {scroll_y}"
    );
}

#[test]
fn advanced_window_thumb_drag_scrolls_content() {
    let win = Rect::from_pos_size(300.0, 60.0, 300.0, 520.0);
    let center = (win.center_x(), win.center_y());
    // Scrollbar thumb sits at the content's right edge, near the top when scroll=0.
    // viewport = (300, 88)-(600, 580); track.min_x = max_x - 8 - 3 = 589.
    let thumb_point = (593.0, 110.0);
    let mut state = GuiState::default();
    let mut open = true;
    let mut scroll_y = 0.0;

    // Frame 1: settle (button up) so scroll range is known.
    render_window_frame(
        &mut state,
        GuiInput {
            pointer: Some(center),
            ..Default::default()
        },
        win,
        &mut open,
        &mut scroll_y,
        20,
    );

    // Frame 2: press on the thumb (down edge) — starts the drag.
    render_window_frame(
        &mut state,
        GuiInput {
            pointer: Some(thumb_point),
            primary_down: true,
            ..Default::default()
        },
        win,
        &mut open,
        &mut scroll_y,
        20,
    );

    // Frame 3: hold and move the pointer down 120 px — content should scroll.
    render_window_frame(
        &mut state,
        GuiInput {
            pointer: Some((thumb_point.0, thumb_point.1 + 120.0)),
            primary_down: true,
            ..Default::default()
        },
        win,
        &mut open,
        &mut scroll_y,
        20,
    );

    assert!(
        scroll_y > 0.0,
        "dragging the scrollbar thumb should move content, scroll_y = {scroll_y}"
    );
}
