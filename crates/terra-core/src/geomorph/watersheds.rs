//! Watershed / drainage basin labelling.

use crate::heightfield::Heightfield;
use crate::mask::MaskField;

use super::routing::{FlowGraph, D8_OFFSETS, NO_FLOW};

/// Watershed extraction options.
#[derive(Debug, Clone)]
pub struct WatershedOptions {
    /// Artist-selected outlet cell (i, j). When set, only that basin is labelled
    /// as primary; others get distinct IDs from auto-discovered outlets.
    pub outlet: Option<(u32, u32)>,
    /// Prefer automatic outlets on the domain boundary.
    pub auto_boundary_outlets: bool,
}

impl Default for WatershedOptions {
    fn default() -> Self {
        Self {
            outlet: None,
            auto_boundary_outlets: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WatershedResult {
    /// Basin ID per cell (0 = unlabeled / ocean edge without sink).
    pub ids: Vec<u32>,
    /// \[0,1\] normalised ID mask for visualisation.
    pub id_mask: MaskField,
    /// Boundary mask where neighbour IDs differ.
    pub boundaries: MaskField,
    /// Outlet cells (i, j, watershed id).
    pub outlets: Vec<(u32, u32, u32)>,
    /// Local contributing area (cell count) per watershed id (index = id).
    pub contributing_area: Vec<f32>,
}

/// Label drainage basins by following D8 flow to sinks / outlets.
pub fn watersheds_from_graph(
    graph: &FlowGraph,
    hf: &Heightfield,
    opts: &WatershedOptions,
) -> WatershedResult {
    let w = graph.width;
    let h = graph.height;
    let n = w * h;
    let mut labels = vec![0u32; n];
    let mut next_label = 1u32;
    let mut outlets = Vec::new();

    // Optional forced outlet seeds first.
    if let Some((oi, oj)) = opts.outlet {
        if (oi as usize) < w && (oj as usize) < h {
            let idx = oj as usize * w + oi as usize;
            labels[idx] = next_label;
            outlets.push((oi, oj, next_label));
            next_label += 1;
            // Flood upstream via donors.
            let mut stack = vec![idx];
            while let Some(c) = stack.pop() {
                for &d in &graph.donors[c] {
                    if labels[d] == 0 {
                        labels[d] = labels[c];
                        stack.push(d);
                    }
                }
            }
        }
    }

    for j in 0..h {
        for i in 0..w {
            let start = j * w + i;
            if labels[start] != 0 {
                continue;
            }
            let mut path = Vec::new();
            let mut ci = i as i32;
            let mut cj = j as i32;
            let mut guard = 0;
            loop {
                let idx = cj as usize * w + ci as usize;
                if labels[idx] != 0 {
                    let lab = labels[idx];
                    for (pi, pj) in path {
                        labels[pj as usize * w + pi as usize] = lab;
                    }
                    break;
                }
                path.push((ci, cj));
                let direction = graph.d8_dir[idx];
                if direction == NO_FLOW || direction as usize >= D8_OFFSETS.len() {
                    let lab = next_label;
                    next_label += 1;
                    outlets.push((ci as u32, cj as u32, lab));
                    for (pi, pj) in &path {
                        labels[*pj as usize * w + *pi as usize] = lab;
                    }
                    break;
                }
                let (di, dj) = D8_OFFSETS[direction as usize];
                let ni = ci + di;
                let nj = cj + dj;
                if ni < 0 || nj < 0 || ni >= w as i32 || nj >= h as i32 || guard > n {
                    let lab = next_label;
                    next_label += 1;
                    outlets.push((ci as u32, cj as u32, lab));
                    for (pi, pj) in &path {
                        labels[*pj as usize * w + *pi as usize] = lab;
                    }
                    break;
                }
                // Local sink / flat trap.
                if hf.get(ni as u32, nj as u32) >= hf.get(ci as u32, cj as u32) {
                    let lab = next_label;
                    next_label += 1;
                    outlets.push((ci as u32, cj as u32, lab));
                    for (pi, pj) in &path {
                        labels[*pj as usize * w + *pi as usize] = lab;
                    }
                    break;
                }
                ci = ni;
                cj = nj;
                guard += 1;
            }
        }
    }

    // If auto boundary outlets requested, keep outlets that sit on the border.
    if opts.auto_boundary_outlets {
        outlets.retain(|&(i, j, _)| {
            i == 0
                || j == 0
                || i + 1 == w as u32
                || j + 1 == h as u32
                || opts.outlet == Some((i, j))
        });
    }

    let max_id = labels.iter().copied().max().unwrap_or(1).max(1);
    let mut id_mask = MaskField::zeros(hf.metrics);
    for j in 0..h {
        for i in 0..w {
            id_mask.set(i as u32, j as u32, labels[j * w + i] as f32 / max_id as f32);
        }
    }

    let boundaries = watershed_boundaries(&labels, w, h, hf.metrics);
    let contributing_area = local_contributing_area(&labels, max_id);

    WatershedResult {
        ids: labels,
        id_mask,
        boundaries,
        outlets,
        contributing_area,
    }
}

/// Discover sink / boundary outlet cells from a flow graph.
pub fn discover_outlets(graph: &FlowGraph) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for idx in 0..graph.width * graph.height {
        if graph.d8_dir[idx] == NO_FLOW || graph.d8_receiver_index(idx).is_none() {
            let (i, j) = graph.coords(idx);
            out.push((i, j));
        }
    }
    out
}

/// Mask of cells where a 4-neighbour has a different watershed id.
pub fn watershed_boundaries(
    ids: &[u32],
    w: usize,
    h: usize,
    metrics: crate::heightfield::HeightfieldMetrics,
) -> MaskField {
    let mut out = MaskField::zeros(metrics);
    for j in 0..h {
        for i in 0..w {
            let id = ids[j * w + i];
            let mut border = false;
            for (di, dj) in [(1i32, 0), (-1, 0), (0, 1), (0, -1)] {
                let ni = i as i32 + di;
                let nj = j as i32 + dj;
                if ni < 0 || nj < 0 || ni >= w as i32 || nj >= h as i32 {
                    border = true;
                    break;
                }
                if ids[nj as usize * w + ni as usize] != id {
                    border = true;
                    break;
                }
            }
            out.set(i as u32, j as u32, if border { 1.0 } else { 0.0 });
        }
    }
    out
}

/// Cell count per watershed id (`contributing_area[id]`).
pub fn local_contributing_area(ids: &[u32], max_id: u32) -> Vec<f32> {
    let mut area = vec![0.0f32; (max_id as usize) + 1];
    for &id in ids {
        if (id as usize) < area.len() {
            area[id as usize] += 1.0;
        }
    }
    area
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geomorph::depression::{handle_depressions, DepressionMode};
    use crate::geomorph::routing::{build_flow_graph, FlowModel};
    use crate::heightfield::HeightfieldMetrics;

    #[test]
    fn two_basins_get_distinct_ids() {
        let m = HeightfieldMetrics::new(16, 8, 160.0, 80.0);
        let hf = crate::geomorph::two_basins(m);
        let filled = handle_depressions(&hf, DepressionMode::Fill).height;
        let g = build_flow_graph(&filled, FlowModel::D8);
        let ws = watersheds_from_graph(&g, &filled, &WatershedOptions::default());
        let left = ws.ids[4 * 16 + 2];
        let right = ws.ids[4 * 16 + 14];
        assert_ne!(left, 0);
        assert_ne!(right, 0);
        assert_ne!(left, right);
    }
}
