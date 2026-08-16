//! Clear UI font: IBM Plex Sans + Lucide icons via fontdue.
//!
//! Atlas is rasterized in physical pixels for the active DPI. Glyph metrics are
//! stored in physical pixels so scale=1.0 draws 1:1 with the atlas (sharp).
//! Icons share the same R8 atlas as ASCII glyphs.

use std::sync::{Mutex, OnceLock};

use fontdue::{Font, FontSettings};
use lucide_icons::LUCIDE_FONT_BYTES;

use crate::icons::{Icon, ICON_COUNT, ICON_PX};

/// Nominal logical size at `scale = 1.0`.
pub const FONT_PX: f32 = 14.0;

pub const FIRST_CHAR: u8 = 32;
pub const LAST_CHAR: u8 = 126;
pub const GLYPH_COUNT: usize = (LAST_CHAR - FIRST_CHAR + 1) as usize;

#[derive(Debug, Clone, Copy)]
pub struct Glyph {
    pub uv: [f32; 4],
    /// Horizontal advance in physical pixels at bake size.
    pub advance_px: f32,
    /// Bitmap offset from pen (top-left of line box) in physical pixels.
    pub x0_px: f32,
    pub y0_px: f32,
    /// Exact bitmap size in physical pixels (for 1:1 sampling).
    pub px_w: u32,
    pub px_h: u32,
}

#[derive(Debug, Clone)]
pub struct BakedFont {
    pub atlas: Vec<u8>,
    pub atlas_w: u32,
    pub atlas_h: u32,
    pub glyphs: [Glyph; GLYPH_COUNT],
    pub icons: [Glyph; ICON_COUNT],
    pub ascent: f32,
    pub line_height: f32,
    pub physical_px: u32,
    pub ppp: f32,
}

struct FontCache {
    physical_px: u32,
    baked: BakedFont,
}

static CACHE: OnceLock<Mutex<FontCache>> = OnceLock::new();

fn cache() -> &'static Mutex<FontCache> {
    CACHE.get_or_init(|| {
        let ppp = 1.0;
        let phys = physical_px_for(ppp);
        Mutex::new(FontCache {
            physical_px: phys,
            baked: bake_font(phys, ppp),
        })
    })
}

fn physical_px_for(ppp: f32) -> u32 {
    (FONT_PX * ppp.max(0.5)).round().clamp(1.0, 48.0) as u32
}

fn icon_physical_px_for(ppp: f32) -> u32 {
    (ICON_PX * ppp.max(0.5)).round().clamp(14.0, 64.0) as u32
}

/// Prefer native atlas size — fractional scales resample and look soft.
#[inline]
pub fn snap_text_scale(scale: f32) -> f32 {
    let s = scale.max(0.5);
    if s >= 0.82 {
        1.0
    } else {
        ((s * 4.0).round() / 4.0).max(0.5)
    }
}

/// Ensure the atlas matches the window DPI. Call once per frame before building UI.
pub fn prepare(pixels_per_point: f32) {
    let ppp = pixels_per_point.max(0.5);
    let phys = physical_px_for(ppp);
    let mut guard = cache().lock().expect("font cache");
    if guard.physical_px != phys || (guard.baked.ppp - ppp).abs() > 0.001 {
        guard.physical_px = phys;
        guard.baked = bake_font(phys, ppp);
    }
}

pub fn current() -> BakedFont {
    cache().lock().expect("font cache").baked.clone()
}

pub fn physical_px() -> u32 {
    cache().lock().expect("font cache").physical_px
}

pub fn build_atlas_r8() -> (Vec<u8>, u32, u32, u32) {
    let baked = current();
    (baked.atlas, baked.atlas_w, baked.atlas_h, baked.physical_px)
}

pub fn glyph(ch: u8) -> Glyph {
    let code = ch.clamp(FIRST_CHAR, LAST_CHAR);
    cache().lock().expect("font cache").baked.glyphs[(code - FIRST_CHAR) as usize]
}

pub fn icon_glyph(icon: Icon) -> Glyph {
    cache().lock().expect("font cache").baked.icons[icon.index()]
}

pub fn text_width(text: &str, scale: f32) -> f32 {
    let scale = snap_text_scale(scale);
    let guard = cache().lock().expect("font cache");
    let ppp = guard.baked.ppp.max(0.5);
    let mut pen = 0.0f32;
    for ch in text.chars() {
        let byte = printable_ascii(ch);
        let g = guard.baked.glyphs[(byte - FIRST_CHAR) as usize];
        pen += (g.advance_px * scale).round().max(0.0);
    }
    pen / ppp
}

/// Map a character to a printable ASCII atlas code (space..=tilde), else `?`.
#[inline]
pub fn printable_ascii(ch: char) -> u8 {
    let code = ch as u32;
    if (FIRST_CHAR as u32..=LAST_CHAR as u32).contains(&code) {
        code as u8
    } else {
        b'?'
    }
}

/// Vertically centre a line of text of the given scale inside a row of `row_h`.
/// Result is snapped to the physical pixel grid so glyphs stay sharp.
#[inline]
pub fn text_top_in_row(row_min_y: f32, row_h: f32, scale: f32) -> f32 {
    let scale = snap_text_scale(scale);
    let visual_h = ascent(scale).max(line_height(scale) * 0.72);
    let y = row_min_y + ((row_h - visual_h) * 0.5).max(0.0);
    let ppp = active_ppp().max(0.5);
    (y * ppp).round() / ppp
}

/// Truncate `text` so its rendered width fits `max_w`, appending "..." when needed.
pub fn truncate_to_width(text: &str, scale: f32, max_w: f32) -> String {
    if max_w <= 0.0 {
        return String::new();
    }
    if text_width(text, scale) <= max_w {
        return text.to_string();
    }
    let ellipsis = "...";
    let ell_w = text_width(ellipsis, scale);
    if ell_w >= max_w {
        return ".".to_string();
    }
    let mut out = String::new();
    for ch in text.chars() {
        let candidate = format!("{out}{ch}");
        if text_width(&candidate, scale) + ell_w > max_w {
            break;
        }
        out.push(ch);
    }
    out.push_str(ellipsis);
    out
}

pub fn line_height(scale: f32) -> f32 {
    let scale = snap_text_scale(scale);
    cache().lock().expect("font cache").baked.line_height * scale
}

pub fn ascent(scale: f32) -> f32 {
    let scale = snap_text_scale(scale);
    cache().lock().expect("font cache").baked.ascent * scale
}

pub fn active_ppp() -> f32 {
    cache().lock().expect("font cache").baked.ppp
}

fn empty_glyph() -> Glyph {
    Glyph {
        uv: [0.0; 4],
        advance_px: FONT_PX * 0.5,
        x0_px: 0.0,
        y0_px: 0.0,
        px_w: 0,
        px_h: 0,
    }
}

fn bake_font(physical_px: u32, ppp: f32) -> BakedFont {
    let text_bytes = include_bytes!("../fonts/IBMPlexSans-Regular.ttf");
    let text_font =
        Font::from_bytes(text_bytes.as_slice(), FontSettings::default()).expect("IBM Plex Sans");
    let icon_font =
        Font::from_bytes(LUCIDE_FONT_BYTES, FontSettings::default()).expect("Lucide icons");

    let px = physical_px as f32;
    let icon_px = icon_physical_px_for(ppp) as f32;
    let line = text_font.horizontal_line_metrics(px).expect("line metrics");
    let to_logical = 1.0 / ppp.max(0.5);
    // Snap ascent to a whole physical pixel so every glyph shares one baseline grid.
    let ascent_px = line.ascent.round();
    let ascent = ascent_px * to_logical;
    let line_height =
        (ascent_px - line.descent.round() + line.line_gap.round()).max(px * 1.2) * to_logical;

    enum Slot {
        Ascii(u8),
        Icon(usize),
    }

    let mut rasters: Vec<(Slot, fontdue::Metrics, Vec<u8>)> =
        Vec::with_capacity(GLYPH_COUNT + ICON_COUNT);
    let mut max_w = 1u32;
    let mut max_h = 1u32;

    for code in FIRST_CHAR..=LAST_CHAR {
        let (metrics, bitmap) = text_font.rasterize(code as char, px);
        max_w = max_w.max(metrics.width.max(1) as u32);
        max_h = max_h.max(metrics.height.max(1) as u32);
        rasters.push((Slot::Ascii(code), metrics, bitmap));
    }
    for (i, icon) in Icon::ALL.iter().enumerate() {
        let (metrics, bitmap) = icon_font.rasterize(icon.ch(), icon_px);
        max_w = max_w.max(metrics.width.max(1) as u32);
        max_h = max_h.max(metrics.height.max(1) as u32);
        rasters.push((Slot::Icon(i), metrics, bitmap));
    }

    let pad = 1u32;
    let cell_w = max_w + pad * 2;
    let cell_h = max_h + pad * 2;
    let cols = 16u32;
    let total = rasters.len() as u32;
    let rows = total.div_ceil(cols);
    let atlas_w = (cols * cell_w).next_power_of_two().max(256);
    let atlas_h = (rows * cell_h).next_power_of_two().max(256);
    let mut atlas = vec![0u8; (atlas_w * atlas_h) as usize];

    let mut glyphs = [empty_glyph(); GLYPH_COUNT];
    let mut icons = [empty_glyph(); ICON_COUNT];

    for (i, (slot, metrics, bitmap)) in rasters.into_iter().enumerate() {
        let col = (i as u32) % cols;
        let row = (i as u32) / cols;
        let ox = col * cell_w + pad;
        let oy = row * cell_h + pad;

        let gw = metrics.width as u32;
        let gh = metrics.height as u32;
        for y in 0..gh {
            for x in 0..gw {
                let src = bitmap[(y * gw + x) as usize];
                let dst_x = ox + x;
                let dst_y = oy + y;
                if dst_x < atlas_w && dst_y < atlas_h {
                    atlas[(dst_y * atlas_w + dst_x) as usize] = src;
                }
            }
        }

        // UV at exact texel edges — with 1:1 quads this samples coverage without blur.
        let u0 = ox as f32 / atlas_w as f32;
        let v0 = oy as f32 / atlas_h as f32;
        let u1 = (ox + gw.max(1)) as f32 / atlas_w as f32;
        let v1 = (oy + gh.max(1)) as f32 / atlas_h as f32;

        match slot {
            Slot::Ascii(code) => {
                let y0_px = ascent_px - metrics.ymin as f32 - metrics.height as f32;
                let idx = (code - FIRST_CHAR) as usize;
                glyphs[idx] = Glyph {
                    uv: [u0, v0, u1, v1],
                    advance_px: metrics.advance_width,
                    x0_px: metrics.xmin as f32,
                    y0_px,
                    px_w: gw,
                    px_h: gh,
                };
                if code == b' ' || gw == 0 {
                    glyphs[idx].px_w = 0;
                    glyphs[idx].px_h = 0;
                    glyphs[idx].x0_px = 0.0;
                    glyphs[idx].y0_px = 0.0;
                    glyphs[idx].advance_px = metrics.advance_width.max(px * 0.28);
                }
            }
            Slot::Icon(idx) => {
                let box_px = icon_px;
                let gw_f = gw as f32;
                let gh_f = gh as f32;
                icons[idx] = Glyph {
                    uv: [u0, v0, u1, v1],
                    advance_px: box_px,
                    x0_px: (box_px - gw_f) * 0.5,
                    y0_px: (box_px - gh_f) * 0.5,
                    px_w: gw,
                    px_h: gh,
                };
            }
        }
    }

    BakedFont {
        atlas,
        atlas_w,
        atlas_h,
        glyphs,
        icons,
        ascent,
        line_height,
        physical_px,
        ppp,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bakes_ascii_atlas() {
        prepare(1.0);
        assert!(text_width("Terra", 1.0) > text_width("Ti", 1.0));
        let a = glyph(b'A');
        assert!(a.px_w > 0 && a.px_h > 0);
        // Capitals should sit near the line top (small y0), not far below.
        assert!(
            a.y0_px >= -1.0 && a.y0_px < line_height(1.0) * active_ppp() * 0.5,
            "A y0_px={}",
            a.y0_px
        );
        let g = glyph(b'g');
        assert!(
            g.px_h >= a.px_h,
            "descender should be at least as tall as A"
        );
        prepare(2.0);
        assert_eq!(physical_px(), 28);
    }

    #[test]
    fn atlas_has_ink_for_letters() {
        prepare(1.0);
        let (atlas, w, h, _) = build_atlas_r8();
        assert!(w >= 128 && h >= 128);
        let ink = atlas.iter().filter(|&&p| p > 32).count();
        assert!(ink > 500, "expected substantial ink, got {ink}");
    }

    #[test]
    fn bakes_lucide_icons() {
        prepare(1.0);
        let mountain = icon_glyph(Icon::Mountain);
        assert!(
            mountain.px_w > 0 && mountain.px_h > 0,
            "mountain icon empty"
        );
        let plus = icon_glyph(Icon::Plus);
        assert!(plus.px_w > 0 && plus.px_h > 0, "plus icon empty");
    }

    #[test]
    fn snap_keeps_native_sharp() {
        assert_eq!(snap_text_scale(1.0), 1.0);
        assert_eq!(snap_text_scale(0.92), 1.0);
        assert_eq!(snap_text_scale(0.85), 1.0);
    }
}
