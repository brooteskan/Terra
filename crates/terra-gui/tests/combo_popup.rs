//! Reproduction: does a combo dropdown stay open after a click?
//!
//! Mirrors the "Viewport Rendering" window, whose Mode / Quality / Debug combos
//! reportedly "appear for a moment then collapse". These tests drive the real
//! `GuiContext` frame lifecycle (press edge, then hold) with no GPU and assert on
//! `open_combo` — the persistent flag that keeps a dropdown visible across frames.

use terra_gui::{combo, GuiContext, GuiInput, GuiState, Id, Rect};

const SW: f32 = 900.0;
const SH: f32 = 700.0;

fn press_frame(pointer: (f32, f32)) -> GuiInput {
    GuiInput {
        pointer: Some(pointer),
        primary_down: true,
        ..Default::default()
    }
}

/// Point inside the first combo's clickable field, given a window at `win`.
/// Field math copied from `widgets::combo` + `context::begin_window`.
fn first_combo_field_point(win: Rect) -> (f32, f32) {
    // begin_window content rect starts WINDOW_TITLE_H (28) below the window top.
    let content_min_x = win.min_x;
    let content_min_y = win.min_y + 28.0;
    // Layout::new pad = 12.
    let pad = 12.0;
    let row_min_x = content_min_x + pad;
    let row_min_y = content_min_y + pad;
    // scrollbar gutter is removed from the right, doesn't affect the field's left half.
    let label_w = 96.0_f32; // CONTROL_LABEL_W, window is wide
    let gap = 8.0; // CONTROL_GAP
    let field_min_x = row_min_x + label_w + gap;
    // Row height 32; vertical centre.
    (field_min_x + 20.0, row_min_y + 16.0)
}

/// One window frame containing a single 3-item combo.
fn window_combo_frame(state: &mut GuiState, input: GuiInput, win: Rect, open: &mut bool) {
    let mut scroll = 0.0;
    let mut ctx = GuiContext::begin(SW, SH, 1.0, input, state);
    if ctx.begin_window(
        Id::new("repro_window"),
        "Viewport Rendering",
        win,
        open,
        &mut scroll,
    ) {
        let mut sel = 0usize;
        combo(
            &mut ctx,
            "Mode",
            &mut sel,
            &["Rasterized", "Hybrid", "Progressive"],
        );
        ctx.end_window(&mut scroll);
    }
    ctx.end();
}

#[test]
fn combo_in_window_stays_open_after_click() {
    let win = Rect::from_pos_size(200.0, 120.0, 300.0, 400.0);
    let field = first_combo_field_point(win);

    let mut state = GuiState::default();
    let mut open = true;

    // Frame 0: settle with the button up so the next frame is a clean press edge.
    window_combo_frame(
        &mut state,
        GuiInput {
            pointer: Some(field),
            primary_down: false,
            ..Default::default()
        },
        win,
        &mut open,
    );
    assert_eq!(state.open_combo, None, "combo starts closed");

    // Frame 1: press on the combo field (down edge) — this should OPEN the dropdown.
    window_combo_frame(&mut state, press_frame(field), win, &mut open);

    assert_eq!(
        state.open_combo,
        Some(Id::new("Mode").child("combo")),
        "after clicking the field the dropdown must stay open across the frame boundary"
    );
}

/// Control: the same combo in a plain scrolled panel (the inspector's situation).
fn panel_combo_frame(state: &mut GuiState, input: GuiInput, panel: Rect) {
    let mut scroll = 0.0;
    let mut ctx = GuiContext::begin(SW, SH, 1.0, input, state);
    ctx.begin_panel_scrolled(
        Id::new("repro_panel"),
        panel,
        terra_gui::Color::rgb(0.1, 0.1, 0.1),
        &mut scroll,
    );
    let mut sel = 0usize;
    combo(
        &mut ctx,
        "Mode",
        &mut sel,
        &["Rasterized", "Hybrid", "Progressive"],
    );
    ctx.end_panel_scrolled(&mut scroll);
    ctx.end();
}

fn first_panel_combo_field_point(panel: Rect) -> (f32, f32) {
    let pad = 12.0;
    let row_min_x = panel.min_x + pad;
    let row_min_y = panel.min_y + pad;
    let label_w = 96.0_f32;
    let gap = 8.0;
    (row_min_x + label_w + gap + 20.0, row_min_y + 16.0)
}

#[test]
fn open_combo_closes_on_press_outside() {
    let win = Rect::from_pos_size(200.0, 120.0, 300.0, 400.0);
    let field = first_combo_field_point(win);

    let mut state = GuiState::default();
    let mut open = true;

    // Settle, then click to open.
    window_combo_frame(
        &mut state,
        GuiInput {
            pointer: Some(field),
            primary_down: false,
            ..Default::default()
        },
        win,
        &mut open,
    );
    window_combo_frame(&mut state, press_frame(field), win, &mut open);
    assert!(
        state.open_combo.is_some(),
        "combo should be open after click"
    );

    // Release, then press somewhere far from the field and its menu.
    window_combo_frame(
        &mut state,
        GuiInput {
            pointer: Some((10.0, 10.0)),
            primary_down: false,
            ..Default::default()
        },
        win,
        &mut open,
    );
    window_combo_frame(&mut state, press_frame((10.0, 10.0)), win, &mut open);

    assert_eq!(
        state.open_combo, None,
        "a press outside the field and menu must dismiss the dropdown"
    );
}

#[test]
fn open_combo_selects_item_on_click() {
    let win = Rect::from_pos_size(200.0, 120.0, 300.0, 400.0);
    let field = first_combo_field_point(win);
    // Menu opens just below the field; second row (index 1) is one ROW_H (28) down.
    let item1 = (field.0, field.1 + 16.0 + 2.0 + 28.0 + 14.0);

    let mut state = GuiState::default();
    let mut open = true;
    let mut picked: Option<usize> = None;

    // Drive frames but capture the selection combo() reports.
    let mut frame = |state: &mut GuiState, input: GuiInput| {
        let mut scroll = 0.0;
        let mut ctx = GuiContext::begin(SW, SH, 1.0, input, state);
        if ctx.begin_window(
            Id::new("repro_window"),
            "Viewport Rendering",
            win,
            &mut open,
            &mut scroll,
        ) {
            let mut sel = 0usize;
            if combo(
                &mut ctx,
                "Mode",
                &mut sel,
                &["Rasterized", "Hybrid", "Progressive"],
            ) {
                picked = Some(sel);
            }
            ctx.end_window(&mut scroll);
        }
        ctx.end();
    };

    // Settle → open.
    frame(
        &mut state,
        GuiInput {
            pointer: Some(field),
            primary_down: false,
            ..Default::default()
        },
    );
    frame(&mut state, press_frame(field));
    assert!(state.open_combo.is_some(), "combo should be open");

    // Release over the item, then press it.
    frame(
        &mut state,
        GuiInput {
            pointer: Some(item1),
            primary_down: false,
            ..Default::default()
        },
    );
    frame(&mut state, press_frame(item1));
    // combo_pick is applied on the following frame.
    frame(
        &mut state,
        GuiInput {
            pointer: Some(item1),
            primary_down: false,
            ..Default::default()
        },
    );

    assert_eq!(
        state.open_combo, None,
        "selecting an item closes the dropdown"
    );
    assert_eq!(picked, Some(1), "clicking row 1 should select index 1");
}

#[test]
fn combo_in_plain_panel_stays_open_after_click() {
    let panel = Rect::from_pos_size(200.0, 120.0, 300.0, 400.0);
    let field = first_panel_combo_field_point(panel);

    let mut state = GuiState::default();

    panel_combo_frame(
        &mut state,
        GuiInput {
            pointer: Some(field),
            primary_down: false,
            ..Default::default()
        },
        panel,
    );
    assert_eq!(state.open_combo, None, "combo starts closed");

    panel_combo_frame(&mut state, press_frame(field), panel);

    assert_eq!(
        state.open_combo,
        Some(Id::new("Mode").child("combo")),
        "plain-panel combo should also stay open after a click"
    );
}
