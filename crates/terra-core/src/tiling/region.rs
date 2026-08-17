//! Sample-space rectangles for tiled dirty regions (Wave D).

use crate::heightfield::{HeightfieldMetrics, TileId};
use serde::{Deserialize, Serialize};

/// Inclusive-exclusive sample rectangle `[x, x+w) × [y, y+h)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SampleRect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl SampleRect {
    pub fn from_tile(metrics: &HeightfieldMetrics, id: TileId) -> Self {
        let x = id.tx * metrics.tile_size;
        let y = id.tz * metrics.tile_size;
        let w = (metrics.width - x).min(metrics.tile_size);
        let h = (metrics.height - y).min(metrics.tile_size);
        Self { x, y, w, h }
    }

    /// Expand by `pad` samples on each side (clamped to field).
    pub fn padded(self, metrics: &HeightfieldMetrics, pad: u32) -> Self {
        let x0 = self.x.saturating_sub(pad);
        let y0 = self.y.saturating_sub(pad);
        let x1 = (self.x + self.w + pad).min(metrics.width);
        let y1 = (self.y + self.h + pad).min(metrics.height);
        Self {
            x: x0,
            y: y0,
            w: x1 - x0,
            h: y1 - y0,
        }
    }

    pub fn union(self, other: Self) -> Self {
        let x0 = self.x.min(other.x);
        let y0 = self.y.min(other.y);
        let x1 = (self.x + self.w).max(other.x + other.w);
        let y1 = (self.y + self.h).max(other.y + other.h);
        Self {
            x: x0,
            y: y0,
            w: x1 - x0,
            h: y1 - y0,
        }
    }

    pub fn is_empty(self) -> bool {
        self.w == 0 || self.h == 0
    }
}

/// Union of tile interiors (no padding).
pub fn rects_from_tiles(metrics: &HeightfieldMetrics, tiles: &[TileId]) -> Vec<SampleRect> {
    tiles
        .iter()
        .map(|id| SampleRect::from_tile(metrics, *id))
        .filter(|r| !r.is_empty())
        .collect()
}

/// Single bounding rect covering all tiles, padded for normal/seam stencils.
pub fn bounds_from_tiles(
    metrics: &HeightfieldMetrics,
    tiles: &[TileId],
    pad: u32,
) -> Option<SampleRect> {
    let mut iter = tiles.iter().map(|id| SampleRect::from_tile(metrics, *id));
    let first = iter.next()?;
    let mut acc = first;
    for r in iter {
        acc = acc.union(r);
    }
    Some(acc.padded(metrics, pad))
}

/// Normalized UV rectangle in `[0,1]²` (min/max corners).
///
/// Carries a spatial edit scope that is *resolution-independent*: the same rect
/// maps to a different tile set at each preview resolution (a Draft 512 grid is
/// 2×2 tiles, a Full 2048 grid is 8×8 over the same world). That is why a sculpt
/// edit's scope travels from the app to the CPU worker as UV rather than as
/// texels or tiles — the worker recomputes its own resolution independently and
/// maps the rect through [`tiles_for_uv_rect`] at the job's metrics (#100 phase 4).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UvRect {
    pub min_u: f32,
    pub min_v: f32,
    pub max_u: f32,
    pub max_v: f32,
}

impl UvRect {
    /// The clamped `[0,1]²` bounding box of a stamp centered at UV `(u, v)` with
    /// UV `radius` — the pre-resolution-multiply form of the texel rect the GPU
    /// dirty path derives from the same stamp (floor/ceil parity is applied in
    /// [`tiles_for_uv_rect`]).
    pub fn from_center_radius(u: f32, v: f32, radius: f32) -> Self {
        Self {
            min_u: (u - radius).clamp(0.0, 1.0),
            min_v: (v - radius).clamp(0.0, 1.0),
            max_u: (u + radius).clamp(0.0, 1.0),
            max_v: (v + radius).clamp(0.0, 1.0),
        }
    }

    pub fn union(self, other: Self) -> Self {
        Self {
            min_u: self.min_u.min(other.min_u),
            min_v: self.min_v.min(other.min_v),
            max_u: self.max_u.max(other.max_u),
            max_v: self.max_v.max(other.max_v),
        }
    }
}

/// Map a normalized [`UvRect`] to the set of tiles it touches at `metrics`.
///
/// The UV→sample conversion mirrors the GPU dirty-rect math exactly (floor the
/// min edge, ceil the max edge) so the CPU scope can never be narrower than the
/// region the GPU already presents. The sample span is forced non-empty — a
/// degenerate point rect still marks the tile under it — then mapped to the
/// inclusive tile range it spans, clamped to the field so field-edge partial
/// tiles resolve correctly. A non-finite rect (which would silently truncate to
/// tile (0,0) through the `as u32` saturating cast) escalates to every tile
/// instead of under-marking.
pub fn tiles_for_uv_rect(metrics: &HeightfieldMetrics, rect: UvRect) -> Vec<TileId> {
    if !(rect.min_u.is_finite()
        && rect.min_v.is_finite()
        && rect.max_u.is_finite()
        && rect.max_v.is_finite())
    {
        return all_tiles(metrics);
    }
    let w = metrics.width;
    let h = metrics.height;
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let tile = metrics.tile_size.max(1);

    // Half-open sample span [s0, s1), forced non-empty and clamped in-field.
    let sx0 = ((rect.min_u.clamp(0.0, 1.0) * w as f32).floor() as u32).min(w - 1);
    let sx1 = ((rect.max_u.clamp(0.0, 1.0) * w as f32).ceil() as u32).clamp(sx0 + 1, w);
    let sy0 = ((rect.min_v.clamp(0.0, 1.0) * h as f32).floor() as u32).min(h - 1);
    let sy1 = ((rect.max_v.clamp(0.0, 1.0) * h as f32).ceil() as u32).clamp(sy0 + 1, h);

    let tx0 = sx0 / tile;
    let tx1 = (sx1 - 1) / tile;
    let tz0 = sy0 / tile;
    let tz1 = (sy1 - 1) / tile;

    let cap = ((tx1 - tx0 + 1) as usize) * ((tz1 - tz0 + 1) as usize);
    let mut out = Vec::with_capacity(cap);
    for tz in tz0..=tz1 {
        for tx in tx0..=tx1 {
            out.push(TileId { tx, tz });
        }
    }
    out
}

fn all_tiles(metrics: &HeightfieldMetrics) -> Vec<TileId> {
    let mut out = Vec::with_capacity(metrics.tile_count() as usize);
    for tz in 0..metrics.tiles_z() {
        for tx in 0..metrics.tiles_x() {
            out.push(TileId { tx, tz });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics(res: u32) -> HeightfieldMetrics {
        HeightfieldMetrics {
            width: res,
            height: res,
            world_size_x: res as f32,
            world_size_z: res as f32,
            tile_size: 256,
            halo: 2,
        }
    }

    fn sorted(mut tiles: Vec<TileId>) -> Vec<(u32, u32)> {
        tiles.sort_by_key(|t| (t.tz, t.tx));
        tiles.into_iter().map(|t| (t.tx, t.tz)).collect()
    }

    /// The GPU dirty path's texel-rect corners for the same stamp — the source of
    /// the floor/ceil rounding `tiles_for_uv_rect` must match.
    fn texel_tiles(m: &HeightfieldMetrics, u: f32, v: f32, radius: f32) -> Vec<(u32, u32)> {
        let res = m.width as f32;
        let x0 = ((u - radius).clamp(0.0, 1.0) * res).floor() as u32;
        let y0 = ((v - radius).clamp(0.0, 1.0) * res).floor() as u32;
        let x1 = ((u + radius).clamp(0.0, 1.0) * res).ceil() as u32;
        let y1 = ((v + radius).clamp(0.0, 1.0) * res).ceil() as u32;
        let x1 = x1.max(x0 + 1).min(m.width);
        let y1 = y1.max(y0 + 1).min(m.height);
        let mut set = std::collections::HashSet::new();
        for y in y0..y1 {
            for x in x0..x1 {
                set.insert((x / m.tile_size, y / m.tile_size));
            }
        }
        let mut v: Vec<_> = set.into_iter().collect();
        v.sort_by_key(|&(tx, tz)| (tz, tx));
        v
    }

    #[test]
    fn rect_within_one_tile_marks_one_tile() {
        let m = metrics(1024); // 4x4 tiles of 256
        let got = sorted(tiles_for_uv_rect(
            &m,
            UvRect::from_center_radius(0.3, 0.3, 0.02),
        ));
        assert_eq!(got, vec![(1, 1)]);
    }

    #[test]
    fn rect_crossing_a_tile_boundary_marks_both_tiles() {
        let m = metrics(1024);
        // Straddle the u = 0.25 tile edge (sample 256, the 0|1 boundary).
        let got = sorted(tiles_for_uv_rect(
            &m,
            UvRect::from_center_radius(0.25, 0.3, 0.01),
        ));
        assert_eq!(got, vec![(0, 1), (1, 1)]);
    }

    #[test]
    fn field_edge_partial_tile_resolves_and_clamps() {
        // 70x58 over 32-sample tiles: 3x2 tiles, last column/row partial.
        let m = HeightfieldMetrics {
            width: 70,
            height: 58,
            world_size_x: 70.0,
            world_size_z: 58.0,
            tile_size: 32,
            halo: 2,
        };
        // A stamp at the far corner must land on the partial corner tile (2,1),
        // never off the grid.
        let got = sorted(tiles_for_uv_rect(
            &m,
            UvRect::from_center_radius(0.99, 0.99, 0.005),
        ));
        assert_eq!(got, vec![(2, 1)]);
    }

    #[test]
    fn same_uv_scales_with_resolution() {
        // A small interior stamp (off every tile boundary) is one tile at both
        // resolutions, but a *different* tile — the mapping is resolution-relative.
        let r = UvRect::from_center_radius(0.3, 0.3, 0.01);
        assert_eq!(sorted(tiles_for_uv_rect(&metrics(512), r)), vec![(0, 0)]);
        assert_eq!(sorted(tiles_for_uv_rect(&metrics(2048), r)), vec![(2, 2)]);
        // The same wide rect covers strictly more tiles on the finer grid.
        let straddle = UvRect::from_center_radius(0.5, 0.5, 0.26);
        assert!(
            tiles_for_uv_rect(&metrics(2048), straddle).len()
                > tiles_for_uv_rect(&metrics(512), straddle).len()
        );
    }

    #[test]
    fn degenerate_point_rect_marks_one_tile() {
        let m = metrics(1024);
        let r = UvRect {
            min_u: 0.6,
            min_v: 0.6,
            max_u: 0.6,
            max_v: 0.6,
        };
        assert_eq!(sorted(tiles_for_uv_rect(&m, r)), vec![(2, 2)]);
    }

    #[test]
    fn non_finite_rect_escalates_to_all_tiles() {
        let m = metrics(1024); // 16 tiles
        let r = UvRect {
            min_u: f32::NAN,
            min_v: 0.0,
            max_u: 1.0,
            max_v: 1.0,
        };
        assert_eq!(tiles_for_uv_rect(&m, r).len(), 16);
    }

    #[test]
    fn matches_gpu_texel_rect_tiling() {
        // Rounding parity across resolutions and stamp placements, including
        // boundary-straddling and edge cases.
        for res in [512, 1024, 2048] {
            let m = metrics(res);
            for &(u, v, r) in &[
                (0.5f32, 0.5f32, 0.02f32),
                (0.25, 0.25, 0.03),
                (0.1, 0.9, 0.05),
                (0.5, 0.5, 0.2),
                (0.01, 0.01, 0.02),
                (0.99, 0.5, 0.02),
            ] {
                let got = sorted(tiles_for_uv_rect(&m, UvRect::from_center_radius(u, v, r)));
                assert_eq!(
                    got,
                    texel_tiles(&m, u, v, r),
                    "mismatch at res {res}, stamp ({u},{v},{r})"
                );
            }
        }
    }
}
