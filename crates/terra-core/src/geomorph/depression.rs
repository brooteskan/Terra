//! Depression handling: fill, breach/carve, preserved basins.
//!
//! Priority-Flood (Barnes) is the fill oracle. Breaching carves a least-cost
//! path to the spill instead of raising the pit floor. Preserved-basins mode
//! keeps real lakes above depth / area thresholds.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

use crate::heightfield::Heightfield;
use crate::mask::MaskField;

use super::routing::D8_OFFSETS;

#[derive(Clone, Copy)]
struct FloodCell {
    height: f32,
    index: usize,
}

impl PartialEq for FloodCell {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.height.to_bits() == other.height.to_bits()
    }
}
impl Eq for FloodCell {}
impl PartialOrd for FloodCell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for FloodCell {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .height
            .total_cmp(&self.height)
            .then_with(|| other.index.cmp(&self.index))
    }
}

/// How enclosed depressions are resolved before routing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DepressionMode {
    /// Raise pits to spill elevation (Priority-Flood).
    Fill,
    /// Carve a path from pit to spill (Lindsay-style breach).
    Breach(BreachParams),
    /// Fill tiny pits; keep basins deeper/larger than thresholds as lakes.
    PreserveBasins(PreserveBasinsParams),
    /// Leave DEM unchanged (caller accepts sinks / flats).
    None,
}

impl Default for DepressionMode {
    fn default() -> Self {
        Self::Fill
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BreachParams {
    /// Maximum carve depth (metres) along a breach path.
    pub max_carve_depth: f32,
    /// Optional maximum path length in cells (0 = unlimited).
    pub max_path_cells: u32,
}

impl Default for BreachParams {
    fn default() -> Self {
        Self {
            max_carve_depth: 50.0,
            max_path_cells: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreserveBasinsParams {
    /// Minimum lake depth (m) to preserve.
    pub min_depth_m: f32,
    /// Minimum lake area in cells to preserve.
    pub min_cells: u32,
}

impl Default for PreserveBasinsParams {
    fn default() -> Self {
        Self {
            min_depth_m: 2.0,
            min_cells: 16,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DepressionResult {
    pub height: Heightfield,
    /// Raised amount for fill (or carve depth for breach), metres.
    pub fill_delta: MaskField,
    /// \[0,1\] mask of preserved lake cells (PreserveBasins mode).
    pub lake_mask: MaskField,
}

/// Resolve depressions according to `mode`.
pub fn handle_depressions(hf: &Heightfield, mode: DepressionMode) -> DepressionResult {
    match mode {
        DepressionMode::None => DepressionResult {
            height: hf.clone(),
            fill_delta: MaskField::zeros(hf.metrics),
            lake_mask: MaskField::zeros(hf.metrics),
        },
        DepressionMode::Fill => {
            let filled = priority_flood_fill(hf);
            let delta = height_delta(hf, &filled);
            DepressionResult {
                height: filled,
                fill_delta: delta,
                lake_mask: MaskField::zeros(hf.metrics),
            }
        }
        DepressionMode::Breach(p) => breach_depressions(hf, p),
        DepressionMode::PreserveBasins(p) => preserve_basins(hf, p),
    }
}

/// Classic Priority-Flood fill: boundary cells are outlets; interior pits rise
/// only enough to connect to a spill.
pub fn priority_flood_fill(hf: &Heightfield) -> Heightfield {
    priority_flood_fill_impl(hf, None)
}

/// Priority-Flood seeded by an explicit set of routing outlets.
///
/// Callers must provide at least one marked cell. Unlike [`priority_flood_fill`],
/// this does not silently add the domain rim as an outlet.
pub(crate) fn priority_flood_fill_from_outlets(
    hf: &Heightfield,
    outlets: &MaskField,
) -> Heightfield {
    priority_flood_fill_impl(hf, Some(outlets))
}

fn priority_flood_fill_impl(hf: &Heightfield, outlets: Option<&MaskField>) -> Heightfield {
    let w = hf.metrics.width as usize;
    let h = hf.metrics.height as usize;
    if w == 0 || h == 0 {
        return hf.clone();
    }

    let mut out = hf.clone();
    let mut visited = vec![false; w * h];
    let mut queue = BinaryHeap::new();

    for j in 0..h {
        for i in 0..w {
            let is_outlet = outlets
                .map(|mask| mask.get(i as u32, j as u32) > 0.5)
                .unwrap_or(i == 0 || j == 0 || i + 1 == w || j + 1 == h);
            if !is_outlet {
                continue;
            }
            let index = j * w + i;
            if visited[index] {
                continue;
            }
            visited[index] = true;
            queue.push(FloodCell {
                height: hf.get(i as u32, j as u32),
                index,
            });
        }
    }

    debug_assert!(
        !queue.is_empty(),
        "priority_flood_fill_from_outlets requires a non-empty outlet mask"
    );
    if queue.is_empty() {
        return hf.clone();
    }

    while let Some(cell) = queue.pop() {
        let i = cell.index % w;
        let j = cell.index / w;
        for &(di, dj) in &D8_OFFSETS {
            let ni = i as i32 + di;
            let nj = j as i32 + dj;
            if ni < 0 || nj < 0 || ni >= w as i32 || nj >= h as i32 {
                continue;
            }
            let nindex = nj as usize * w + ni as usize;
            if visited[nindex] {
                continue;
            }
            visited[nindex] = true;
            let filled_height = hf.get(ni as u32, nj as u32).max(cell.height);
            out.set(ni as u32, nj as u32, filled_height);
            queue.push(FloodCell {
                height: filled_height,
                index: nindex,
            });
        }
    }
    out
}

fn height_delta(original: &Heightfield, modified: &Heightfield) -> MaskField {
    let m = original.metrics;
    let mut out = MaskField::zeros(m);
    let data = out.data_mut();
    for j in 0..m.height {
        for i in 0..m.width {
            let d = (modified.get(i, j) - original.get(i, j)).max(0.0);
            data[(j * m.width + i) as usize] = d;
        }
    }
    out
}

/// MaskField::set clamps to \[0,1\]; depression deltas are metres.
fn set_metres(mask: &mut MaskField, i: u32, j: u32, v: f32) {
    let idx = (j * mask.metrics.width + i) as usize;
    mask.data_mut()[idx] = v;
}

/// Breach: for each pit cell that would be raised by fill, carve toward the
/// spill along the Priority-Flood discovery path instead of filling.
fn breach_depressions(hf: &Heightfield, params: BreachParams) -> DepressionResult {
    let w = hf.metrics.width as usize;
    let h = hf.metrics.height as usize;
    let filled = priority_flood_fill(hf);
    let mut out = hf.clone();
    let mut delta = MaskField::zeros(hf.metrics);
    let lake = MaskField::zeros(hf.metrics);

    // Identify pit cells (raised by fill) and carve a descending path to the
    // nearest lower outlet along steepest D8 on the *filled* surface, lowering
    // the original DEM so flow can escape without raising the basin floor.
    let mut pit_cells = Vec::new();
    for j in 0..h {
        for i in 0..w {
            let raise = filled.get(i as u32, j as u32) - hf.get(i as u32, j as u32);
            if raise > 1e-4 {
                pit_cells.push((i, j, filled.get(i as u32, j as u32)));
            }
        }
    }

    for &(si, sj, spill) in &pit_cells {
        let mut ci = si as i32;
        let mut cj = sj as i32;
        let mut steps = 0u32;
        let max_steps = if params.max_path_cells == 0 {
            (w * h) as u32
        } else {
            params.max_path_cells
        };
        // Walk downhill on the filled DEM until we leave the pit (height ≤ spill
        // and original already ≤ spill), carving original cells down to a
        // gently descending profile.
        let mut target = spill;
        loop {
            if steps >= max_steps {
                break;
            }
            let idx_i = ci as u32;
            let idx_j = cj as u32;
            let cur = out.get(idx_i, idx_j);
            if cur > target {
                let carve = (cur - target).min(params.max_carve_depth);
                let next = cur - carve;
                out.set(idx_i, idx_j, next);
                let prev = delta.get(idx_i, idx_j);
                set_metres(&mut delta, idx_i, idx_j, prev.max(carve));
            }
            // Steepest descent on filled surface.
            let mut best = None;
            let mut best_s = 0.0f32;
            let h0 = filled.get(idx_i, idx_j);
            for &(di, dj) in &D8_OFFSETS {
                let ni = ci + di;
                let nj = cj + dj;
                if ni < 0 || nj < 0 || ni >= w as i32 || nj >= h as i32 {
                    continue;
                }
                let dist = if di != 0 && dj != 0 {
                    std::f32::consts::SQRT_2
                } else {
                    1.0
                };
                let slope = (h0 - filled.get(ni as u32, nj as u32)) / dist;
                if slope > best_s {
                    best_s = slope;
                    best = Some((ni, nj));
                }
            }
            let Some((ni, nj)) = best else { break };
            // Stop once we reach a cell that was never a pit on the original.
            if filled.get(ni as u32, nj as u32) - hf.get(ni as u32, nj as u32) <= 1e-4
                && out.get(ni as u32, nj as u32) <= spill + 1e-3
            {
                // Ensure outlet is not higher than path.
                let out_h = out.get(ni as u32, nj as u32);
                if out_h > target {
                    let carve = (out_h - target).min(params.max_carve_depth);
                    out.set(ni as u32, nj as u32, out_h - carve);
                    let prev = delta.get(ni as u32, nj as u32);
                    set_metres(&mut delta, ni as u32, nj as u32, prev.max(carve));
                }
                break;
            }
            ci = ni;
            cj = nj;
            target = (target - 1e-3).max(spill - params.max_carve_depth);
            steps += 1;
        }
    }

    DepressionResult {
        height: out,
        fill_delta: delta,
        lake_mask: lake,
    }
}

/// Fill only small/shallow pits; leave deep/large basins as lakes.
fn preserve_basins(hf: &Heightfield, params: PreserveBasinsParams) -> DepressionResult {
    let w = hf.metrics.width as usize;
    let h = hf.metrics.height as usize;
    let filled = priority_flood_fill(hf);
    let mut out = hf.clone();
    let mut delta = MaskField::zeros(hf.metrics);
    let mut lake = MaskField::zeros(hf.metrics);

    // Label contiguous pit regions on the fill-delta mask.
    let mut labels = vec![0u32; w * h];
    let mut next = 1u32;
    let mut region_cells: Vec<Vec<usize>> = Vec::new();
    let mut region_depth: Vec<f32> = Vec::new();

    for j in 0..h {
        for i in 0..w {
            let idx = j * w + i;
            if labels[idx] != 0 {
                continue;
            }
            let raise = filled.get(i as u32, j as u32) - hf.get(i as u32, j as u32);
            if raise <= 1e-4 {
                continue;
            }
            let lab = next;
            next += 1;
            let mut cells = Vec::new();
            let mut max_depth = 0.0f32;
            let mut q = VecDeque::new();
            q.push_back(idx);
            labels[idx] = lab;
            while let Some(c) = q.pop_front() {
                cells.push(c);
                let ci = c % w;
                let cj = c / w;
                let d = filled.get(ci as u32, cj as u32) - hf.get(ci as u32, cj as u32);
                max_depth = max_depth.max(d);
                for &(di, dj) in &D8_OFFSETS {
                    let ni = ci as i32 + di;
                    let nj = cj as i32 + dj;
                    if ni < 0 || nj < 0 || ni >= w as i32 || nj >= h as i32 {
                        continue;
                    }
                    let nidx = nj as usize * w + ni as usize;
                    if labels[nidx] != 0 {
                        continue;
                    }
                    let nr = filled.get(ni as u32, nj as u32) - hf.get(ni as u32, nj as u32);
                    if nr > 1e-4 {
                        labels[nidx] = lab;
                        q.push_back(nidx);
                    }
                }
            }
            region_cells.push(cells);
            region_depth.push(max_depth);
        }
    }

    for (ri, cells) in region_cells.iter().enumerate() {
        let depth = region_depth[ri];
        let preserve = depth >= params.min_depth_m && cells.len() as u32 >= params.min_cells;
        if preserve {
            for &c in cells {
                let i = (c % w) as u32;
                let j = (c / w) as u32;
                lake.set(i, j, 1.0);
            }
        } else {
            for &c in cells {
                let i = (c % w) as u32;
                let j = (c / w) as u32;
                let raise = filled.get(i, j) - hf.get(i, j);
                out.set(i, j, filled.get(i, j));
                set_metres(&mut delta, i, j, raise);
            }
        }
    }

    DepressionResult {
        height: out,
        fill_delta: delta,
        lake_mask: lake,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heightfield::HeightfieldMetrics;

    #[test]
    fn fill_raises_enclosed_pit() {
        let m = HeightfieldMetrics::new(5, 5, 50.0, 50.0);
        let mut hf = Heightfield::filled(m, 5.0);
        hf.set(2, 2, 1.0);
        let before = hf.get(2, 2);
        let r = handle_depressions(&hf, DepressionMode::Fill);
        let after = r.height.get(2, 2);
        assert!(after > before + 3.0, "before={before} after={after}");
        assert!((after - 5.0).abs() < 1e-5);
        let expected = (after - before).max(0.0);
        assert!(
            (r.fill_delta.get(2, 2) - expected).abs() < 1e-4,
            "delta={} expected={}",
            r.fill_delta.get(2, 2),
            expected
        );
    }

    #[test]
    fn preserve_keeps_deep_basin() {
        let m = HeightfieldMetrics::new(9, 9, 90.0, 90.0);
        let mut hf = Heightfield::filled(m, 10.0);
        for j in 2..7 {
            for i in 2..7 {
                hf.set(i, j, 1.0);
            }
        }
        let r = handle_depressions(
            &hf,
            DepressionMode::PreserveBasins(PreserveBasinsParams {
                min_depth_m: 2.0,
                min_cells: 4,
            }),
        );
        assert!(r.lake_mask.get(4, 4) > 0.5);
        assert!((r.height.get(4, 4) - 1.0).abs() < 1e-5);
    }
}
