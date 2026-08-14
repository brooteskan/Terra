//! Shared free helpers for the app shell.

use crate::ui::UiState;
use terra_core::layer::{LayerId, LayerKind};
use terra_gui::GuiState;
use winit::keyboard::KeyCode;

pub(crate) fn ui_tool_search_focused(ui_state: &UiState, gui_state: &GuiState) -> bool {
    // Tool search uses hot/active on its field id, or any text focus.
    gui_state.wants_text_input()
        || gui_state.hot == Some(terra_gui::Id::new("tool_search"))
        || gui_state.active == Some(terra_gui::Id::new("tool_search"))
        || !ui_state.tool_search.is_empty()
}

pub(crate) fn mask_field_fingerprint(field: &terra_core::mask::MaskField) -> u64 {
    let data = field.data();
    let mut h = data.len() as u64;
    h ^= (field.metrics.width as u64) << 32;
    h ^= field.metrics.height as u64;
    if let Some(&v) = data.first() {
        h ^= v.to_bits() as u64;
    }
    if let Some(&v) = data.last() {
        h ^= (v.to_bits() as u64).rotate_left(17);
    }
    if data.len() > 2 {
        let mid = data[data.len() / 2];
        h ^= (mid.to_bits() as u64).rotate_left(9);
    }
    h
}

pub(crate) fn aux_maps_fingerprint(
    aux: &std::collections::HashMap<String, terra_core::mask::MaskField>,
) -> u64 {
    const KEYS: &[&str] = &[
        "materials",
        "wetness",
        "vegetation",
        "temperature",
        "rainfall",
        "snow",
        "soil_moisture",
        "biomes",
        "flow_accumulation",
    ];
    let mut h = 0u64;
    for (i, key) in KEYS.iter().enumerate() {
        if let Some(field) = aux.get(*key) {
            h ^= mask_field_fingerprint(field).rotate_left((i as u32) * 3);
        }
    }
    h
}

/// Physical-key fallback for the custom GUI's search fields.
pub(crate) fn search_character(code: KeyCode, shift: bool) -> Option<char> {
    let ch = match code {
        KeyCode::KeyA => 'a',
        KeyCode::KeyB => 'b',
        KeyCode::KeyC => 'c',
        KeyCode::KeyD => 'd',
        KeyCode::KeyE => 'e',
        KeyCode::KeyF => 'f',
        KeyCode::KeyG => 'g',
        KeyCode::KeyH => 'h',
        KeyCode::KeyI => 'i',
        KeyCode::KeyJ => 'j',
        KeyCode::KeyK => 'k',
        KeyCode::KeyL => 'l',
        KeyCode::KeyM => 'm',
        KeyCode::KeyN => 'n',
        KeyCode::KeyO => 'o',
        KeyCode::KeyP => 'p',
        KeyCode::KeyQ => 'q',
        KeyCode::KeyR => 'r',
        KeyCode::KeyS => 's',
        KeyCode::KeyT => 't',
        KeyCode::KeyU => 'u',
        KeyCode::KeyV => 'v',
        KeyCode::KeyW => 'w',
        KeyCode::KeyX => 'x',
        KeyCode::KeyY => 'y',
        KeyCode::KeyZ => 'z',
        KeyCode::Space => ' ',
        _ => return None,
    };
    Some(if shift { ch.to_ascii_uppercase() } else { ch })
}

pub(crate) fn randomize_layer_seed(kind: &mut terra_core::layer::LayerKind) {
    use terra_core::layer::LayerKind;
    let seed = (std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(1))
        ^ 0xA5A5_1234;
    match kind {
        LayerKind::NoiseValue(p) | LayerKind::NoisePerlin(p) | LayerKind::NoiseOpenSimplex(p) => {
            p.seed = seed
        }
        LayerKind::NoiseWorley(p) => p.base.seed = seed,
        LayerKind::Fbm(p) | LayerKind::Ridged(p) => p.base.seed = seed,
        LayerKind::DomainWarp(p) => p.base.seed = seed,
        LayerKind::Mountains(p) => p.base.seed = seed,
        LayerKind::Mesa(p) => p.seed = seed,
        LayerKind::Island(p) => p.seed = seed,
        LayerKind::Volcano(p) => p.seed = seed,
        LayerKind::Uplift(p) => p.seed = seed,
        LayerKind::Dunes(p) => p.base.seed = seed,
        LayerKind::Canyons(p) => p.seed = seed,
        LayerKind::VoronoiRegions(p) => p.base.seed = seed,
        LayerKind::Vegetation(p) => p.seed = seed,
        LayerKind::OverhangStamp(p) => p.seed = seed,
        LayerKind::LocalSdf(p) => p.seed = seed,
        _ => {}
    }
}

/// Stable compact key for associating consecutive control updates with a layer.
pub(crate) fn coalesce_layer_id(id: LayerId) -> u64 {
    let raw = id.0.as_u128();
    raw as u64 ^ (raw >> 64) as u64
}

/// Scale/yaw for viewport plant instances from the selected or first Vegetation layer.
pub(crate) fn vegetation_instance_params(
    doc: &terra_core::document::TerrainDocument,
) -> (f32, f32, f32) {
    let pick = |layer: &terra_core::layer::Layer| -> Option<(f32, f32, f32)> {
        match &layer.kind {
            LayerKind::Vegetation(p) => Some((p.scale_min, p.scale_max, p.yaw_variation_deg)),
            _ => None,
        }
    };
    if let Some(id) = doc.selected {
        if let Some(layer) = doc.stack.find(id) {
            if let Some(params) = pick(layer) {
                return params;
            }
        }
    }
    for layer in doc.stack.flatten_layers() {
        if let Some(params) = pick(layer) {
            return params;
        }
    }
    (0.5, 1.5, 180.0)
}
