//! Custom wgpu immediate-mode UI toolkit (no egui).

mod context;
mod draw;
mod font;
mod icons;
mod id;
mod layout;
mod layout_prefs;
mod renderer;
mod scroll;
mod state;
pub mod style;
mod types;
mod widgets;

pub use context::{
    GuiContext, GuiInput, INSET_BOTTOM, INSET_LEFT, INSET_RIGHT, INSET_TOP, RIGHT_LAYERS_FRAC,
};
pub use draw::{DrawCmd, DrawList};
pub use icons::{Icon, ICON_PX};
pub use id::Id;
pub use layout::Layout;
pub use layout_prefs::{LayoutPrefs, SPLITTER_HIT};
pub use renderer::GuiRenderer;
pub use state::{GuiState, SplitterKind};
pub use types::{Align, Color, Rect};
pub use widgets::{
    accent_button, accent_button_id, button, button_id, checkbox, checkbox_id, chip_button,
    chip_icon_button, collapsible_section, combo, combo_in_rect, icon_button, icon_toggle,
    inspector_tab_bar, label, label_dim, menu_button, radio_toggle, section_header,
    segmented_button, selectable, selection_card, slider_f32, slider_f32_id, slider_i32,
    slider_i32_id, status_pill, widget_lab, WidgetLabState,
};
