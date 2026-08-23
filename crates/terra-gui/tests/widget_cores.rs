use terra_gui::{slider_in_rect, GuiContext, GuiInput, GuiState, Id, Rect};

const SCREEN: f32 = 600.0;

fn slider_frame(state: &mut GuiState, input: GuiInput, value: &mut f32) -> bool {
    let id = Id::new("rect-slider");
    let row = Rect::from_pos_size(100.0, 100.0, 400.0, 32.0);
    let track = Rect::from_pos_size(180.0, 114.0, 200.0, 4.0);
    let value_box = Rect::from_pos_size(410.0, 103.0, 70.0, 26.0);
    let mut ctx = GuiContext::begin(SCREEN, SCREEN, 1.0, input, state);
    let changed = slider_in_rect(&mut ctx, id, row, track, value_box, value, 0.0, 1.0, false);
    ctx.end();
    changed
}

#[test]
fn rect_slider_drag_and_text_commit_share_the_toolkit_state_machine() {
    let mut state = GuiState::default();
    let mut value = 0.0;
    slider_frame(
        &mut state,
        GuiInput {
            pointer: Some((280.0, 116.0)),
            ..Default::default()
        },
        &mut value,
    );
    assert!(slider_frame(
        &mut state,
        GuiInput {
            pointer: Some((280.0, 116.0)),
            primary_down: true,
            ..Default::default()
        },
        &mut value,
    ));
    assert!((value - 0.5).abs() < 1e-6);

    let edit_id = Id::new("rect-slider").child("edit");
    state.text_focus = Some(edit_id);
    state.text_buffer = "0.75".into();
    assert!(slider_frame(
        &mut state,
        GuiInput {
            enter_pressed: true,
            ..Default::default()
        },
        &mut value,
    ));
    assert!((value - 0.75).abs() < 1e-6);
    assert_eq!(state.text_focus, None);
}
