//! Flow routing interface: D8 and D∞ with receivers, donors, and topo order.

use std::collections::VecDeque;

use crate::heightfield::Heightfield;
use crate::mask::MaskField;

/// Sentinel for a cell without a downhill receiver (D8).
pub const NO_FLOW: u8 = u8::MAX;

pub const D8_OFFSETS: [(i32, i32); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];

/// Which flow model filters / pipelines should use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FlowModel {
    #[default]
    D8,
    /// Tarboton D-infinity (multi-directional within the winning facet).
    DInfinity,
}

/// Single receiver contribution (D∞ may have two).
#[derive(Debug, Clone, Copy)]
pub struct FlowReceiver {
    pub di: i32,
    pub dj: i32,
    pub fraction: f32,
}

/// Complete routing graph for a DEM.
#[derive(Debug, Clone)]
pub struct FlowGraph {
    pub model: FlowModel,
    pub width: usize,
    pub height: usize,
    /// Per-cell receivers (empty = sink / no flow).
    pub receivers: Vec<Vec<FlowReceiver>>,
    /// D8 direction index for single-receiver convenience (`NO_FLOW` if none).
    pub d8_dir: Vec<u8>,
    /// Donor lists: cells that flow (partly) into this cell.
    pub donors: Vec<Vec<usize>>,
    /// Topological order: upstream → downstream (sources first).
    pub topo_order: Vec<usize>,
    /// Normalised D8 direction mask for AuxMaps / debug.
    pub direction_mask: MaskField,
}

impl FlowGraph {
    #[inline]
    pub fn index(&self, i: u32, j: u32) -> usize {
        j as usize * self.width + i as usize
    }

    #[inline]
    pub fn coords(&self, idx: usize) -> (u32, u32) {
        ((idx % self.width) as u32, (idx / self.width) as u32)
    }

    /// Flat neighbour index for a receiver offset, if in-bounds.
    pub fn receiver_index(&self, idx: usize, fd: FlowReceiver) -> Option<usize> {
        let (i, j) = self.coords(idx);
        let ni = i as i32 + fd.di;
        let nj = j as i32 + fd.dj;
        if ni < 0 || nj < 0 || ni >= self.width as i32 || nj >= self.height as i32 {
            return None;
        }
        Some(nj as usize * self.width + ni as usize)
    }

    /// Single D8 receiver index, or `None` for sinks.
    pub fn d8_receiver_index(&self, idx: usize) -> Option<usize> {
        let d = self.d8_dir[idx];
        if d == NO_FLOW || d as usize >= D8_OFFSETS.len() {
            return None;
        }
        let (di, dj) = D8_OFFSETS[d as usize];
        self.receiver_index(
            idx,
            FlowReceiver {
                di,
                dj,
                fraction: 1.0,
            },
        )
    }
}

/// Build a [`FlowGraph`] for `hf` using `model`.
pub fn build_flow_graph(hf: &Heightfield, model: FlowModel) -> FlowGraph {
    let w = hf.metrics.width as usize;
    let h = hf.metrics.height as usize;
    let (receivers, d8_dir, direction_mask) = match model {
        FlowModel::D8 => {
            let (dirs, mask) = flow_direction_d8(hf);
            let mut receivers = vec![Vec::new(); w * h];
            for idx in 0..w * h {
                let d = dirs[idx];
                if d == NO_FLOW || d as usize >= D8_OFFSETS.len() {
                    continue;
                }
                let (di, dj) = D8_OFFSETS[d as usize];
                receivers[idx].push(FlowReceiver {
                    di,
                    dj,
                    fraction: 1.0,
                });
            }
            (receivers, dirs, mask)
        }
        FlowModel::DInfinity => {
            let din = flow_direction_dinfinity(hf);
            let (d8, mask) = flow_direction_d8(hf);
            (din, d8, mask)
        }
    };

    let donors = build_donors(&receivers, w, h);
    let topo_order = topological_order(&donors, &receivers, w, h);

    FlowGraph {
        model,
        width: w,
        height: h,
        receivers,
        d8_dir,
        donors,
        topo_order,
        direction_mask,
    }
}

/// Lean D8 drainage: one receiver per cell as a flat `Vec<usize>`, no per-cell
/// `Vec`s.
///
/// Produces the same routed products the SPE oracle consumes — `d8_dir`, a flat
/// receiver array, an upstream→downstream topo order, and the direction mask — as
/// [`build_flow_graph`] with [`FlowModel::D8`], but without the general graph's
/// `Vec<Vec<FlowReceiver>>` receivers, `Vec<Vec<usize>>` donor lists, or
/// `Vec`-per-pop topological pass. [`Self::rebuild`] reuses its buffers so a
/// per-iteration SPE refresh does not reallocate the graph each pass (issue #27).
///
/// This is a *consumer* of [`flow_direction_d8`], not a second copy of it: D8 has
/// exactly one receiver per cell, so the general path's per-pop `nexts.sort` is
/// vacuous and this flat Kahn walk yields a **bit-identical** topo order and
/// accumulation. D∞ genuinely needs the multi-receiver [`FlowGraph`]; this is
/// D8-only by construction.
#[derive(Debug, Clone)]
pub struct D8Drainage {
    pub width: usize,
    pub height: usize,
    /// D8 direction index per cell (`NO_FLOW` = sink), from [`flow_direction_d8`].
    pub d8_dir: Vec<u8>,
    /// Flat downstream neighbour index (`usize::MAX` = sink / no receiver).
    pub receiver: Vec<usize>,
    /// Upstream → downstream order; bit-identical to [`FlowGraph::topo_order`].
    pub topo_order: Vec<usize>,
    /// Normalised D8 direction mask (same as [`FlowGraph::direction_mask`]).
    pub direction_mask: MaskField,
    /// Donor counts, reused as the mutable `remaining` counter during the walk.
    indegree: Vec<u32>,
    /// Kahn frontier, reused across rebuilds.
    queue: VecDeque<usize>,
}

impl D8Drainage {
    /// Build a fresh lean D8 drainage for `hf`.
    pub fn build(hf: &Heightfield) -> Self {
        let w = hf.metrics.width as usize;
        let h = hf.metrics.height as usize;
        let n = w * h;
        let mut d = Self {
            width: w,
            height: h,
            d8_dir: Vec::new(),
            receiver: Vec::with_capacity(n),
            topo_order: Vec::with_capacity(n),
            direction_mask: MaskField::zeros(hf.metrics),
            indegree: Vec::with_capacity(n),
            queue: VecDeque::new(),
        };
        d.rebuild(hf);
        d
    }

    /// Recompute in place for `hf`, reusing existing buffers.
    ///
    /// `hf` must share the resolution the cache was built at.
    pub fn rebuild(&mut self, hf: &Heightfield) {
        let w = hf.metrics.width as usize;
        let h = hf.metrics.height as usize;
        let n = w * h;
        debug_assert_eq!(
            (w, h),
            (self.width, self.height),
            "D8Drainage::rebuild resolution mismatch"
        );
        self.width = w;
        self.height = h;

        // Reuses the single-lineage kernel; freshly allocates dirs + mask.
        let (dirs, mask) = flow_direction_d8(hf);

        // Flat single receiver + donor count (indegree) in one pass. With one
        // receiver per cell each donor is pushed once, so this indegree matches
        // the general `build_donors` length after its (here no-op) dedup.
        self.receiver.clear();
        self.receiver.resize(n, usize::MAX);
        self.indegree.clear();
        self.indegree.resize(n, 0);
        for idx in 0..n {
            let d = dirs[idx];
            if d == NO_FLOW || d as usize >= D8_OFFSETS.len() {
                continue;
            }
            let (di, dj) = D8_OFFSETS[d as usize];
            let i = (idx % w) as i32;
            let j = (idx / w) as i32;
            let ni = i + di;
            let nj = j + dj;
            if ni < 0 || nj < 0 || ni >= w as i32 || nj >= h as i32 {
                continue;
            }
            let nidx = nj as usize * w + ni as usize;
            self.receiver[idx] = nidx;
            self.indegree[nidx] += 1;
        }

        // Kahn walk mirroring `topological_order`: seeds in ascending index
        // order, and — because each cell frees at most one successor — that
        // successor is appended directly (the general path's `nexts.sort` over a
        // ≤1-element list is a no-op), so the emitted order is identical.
        self.queue.clear();
        for idx in 0..n {
            if self.indegree[idx] == 0 {
                self.queue.push_back(idx);
            }
        }
        self.topo_order.clear();
        // `indegree` doubles as the mutable `remaining` counter from here on.
        while let Some(idx) = self.queue.pop_front() {
            self.topo_order.push(idx);
            let r = self.receiver[idx];
            if r != usize::MAX && self.indegree[r] > 0 {
                self.indegree[r] -= 1;
                if self.indegree[r] == 0 {
                    self.queue.push_back(r);
                }
            }
        }

        // Cycles / flats: append any unvisited cells in index order (matches
        // `topological_order`'s tail).
        if self.topo_order.len() < n {
            let mut seen = vec![false; n];
            for &idx in &self.topo_order {
                seen[idx] = true;
            }
            for idx in 0..n {
                if !seen[idx] {
                    self.topo_order.push(idx);
                }
            }
        }

        self.d8_dir = dirs;
        self.direction_mask = mask;
    }
}

/// D8 steepest descent. No downhill neighbour → `NO_FLOW`.
///
/// Ties break toward the lowest direction index for determinism.
pub fn flow_direction_d8(hf: &Heightfield) -> (Vec<u8>, MaskField) {
    use rayon::prelude::*;

    let w = hf.metrics.width as usize;
    let h = hf.metrics.height as usize;
    let mut dirs = vec![NO_FLOW; w * h];
    let mut mask_data = vec![0.0f32; w * h];
    dirs.par_iter_mut()
        .zip(mask_data.par_iter_mut())
        .enumerate()
        .for_each(|(idx, (dir_slot, mask_slot))| {
            let i = idx % w;
            let j = idx / w;
            let h0 = hf.get(i as u32, j as u32);
            let mut best = NO_FLOW;
            let mut best_s = 0.0f32;
            for (k, &(di, dj)) in D8_OFFSETS.iter().enumerate() {
                let ni = i as i32 + di;
                let nj = j as i32 + dj;
                if ni < 0 || nj < 0 || ni >= w as i32 || nj >= h as i32 {
                    continue;
                }
                let dist = if di != 0 && dj != 0 {
                    std::f32::consts::SQRT_2
                } else {
                    1.0
                };
                let slope = (h0 - hf.get(ni as u32, nj as u32)) / dist;
                if slope > best_s + 1e-12 {
                    best_s = slope;
                    best = k as u8;
                } else if (slope - best_s).abs() <= 1e-12 && slope > 0.0 && (k as u8) < best {
                    best = k as u8;
                }
            }
            *dir_slot = best;
            // Match prior MaskField::set clamping (NO_FLOW sentinel → 0).
            *mask_slot = if best == NO_FLOW {
                0.0
            } else {
                (best as f32 / 7.0).clamp(0.0, 1.0)
            };
        });
    (dirs, MaskField::from_raw(hf.metrics, &mask_data))
}

/// D∞ (Tarboton): partition flow between the two cells of the winning facet.
pub fn flow_direction_dinfinity(hf: &Heightfield) -> Vec<Vec<FlowReceiver>> {
    let w = hf.metrics.width as usize;
    let h = hf.metrics.height as usize;
    let mut out = vec![Vec::new(); w * h];
    let dx = hf.metrics.dx();
    let dz = hf.metrics.dz();

    for j in 0..h {
        for i in 0..w {
            let h0 = hf.get(i as u32, j as u32);
            let mut best_slope = 0.0f32;
            let mut best_dirs = Vec::new();

            for facet in 0..8 {
                let (di1, dj1) = D8_OFFSETS[facet];
                let (di2, dj2) = D8_OFFSETS[(facet + 1) % 8];
                let ni1 = i as i32 + di1;
                let nj1 = j as i32 + dj1;
                let ni2 = i as i32 + di2;
                let nj2 = j as i32 + dj2;
                if ni1 < 0
                    || nj1 < 0
                    || ni2 < 0
                    || nj2 < 0
                    || ni1 >= w as i32
                    || nj1 >= h as i32
                    || ni2 >= w as i32
                    || nj2 >= h as i32
                {
                    continue;
                }

                let v1 = (di1 as f32 * dx, dj1 as f32 * dz);
                let v2 = (di2 as f32 * dx, dj2 as f32 * dz);
                let z1 = hf.get(ni1 as u32, nj1 as u32) - h0;
                let z2 = hf.get(ni2 as u32, nj2 as u32) - h0;
                let drop1 = -z1;
                let drop2 = -z2;
                let det = v1.0 * v2.1 - v1.1 * v2.0;
                if det.abs() <= f32::EPSILON {
                    continue;
                }

                let gx = (z1 * v2.1 - z2 * v1.1) / det;
                let gz = (v1.0 * z2 - v2.0 * z1) / det;
                let downhill = (-gx, -gz);
                let c1 = (downhill.0 * v2.1 - downhill.1 * v2.0) / det;
                let c2 = (v1.0 * downhill.1 - v1.1 * downhill.0) / det;
                let mut facet_dirs = Vec::new();
                let mut facet_slope = 0.0;

                if c1 >= -1e-6 && c2 >= -1e-6 {
                    let mut weight1 = c1.max(0.0) * v1.0.hypot(v1.1);
                    let mut weight2 = c2.max(0.0) * v2.0.hypot(v2.1);
                    if drop1 <= 0.0 {
                        weight1 = 0.0;
                    }
                    if drop2 <= 0.0 {
                        weight2 = 0.0;
                    }
                    let total = weight1 + weight2;
                    if total > f32::EPSILON {
                        facet_slope = downhill.0.hypot(downhill.1);
                        if weight1 > 0.0 {
                            facet_dirs.push(FlowReceiver {
                                di: di1,
                                dj: dj1,
                                fraction: weight1 / total,
                            });
                        }
                        if weight2 > 0.0 {
                            facet_dirs.push(FlowReceiver {
                                di: di2,
                                dj: dj2,
                                fraction: weight2 / total,
                            });
                        }
                    }
                }

                if facet_dirs.is_empty() {
                    let slope1 = (drop1 / v1.0.hypot(v1.1)).max(0.0);
                    let slope2 = (drop2 / v2.0.hypot(v2.1)).max(0.0);
                    if slope1 > 0.0 || slope2 > 0.0 {
                        if slope1 >= slope2 {
                            facet_slope = slope1;
                            facet_dirs.push(FlowReceiver {
                                di: di1,
                                dj: dj1,
                                fraction: 1.0,
                            });
                        } else {
                            facet_slope = slope2;
                            facet_dirs.push(FlowReceiver {
                                di: di2,
                                dj: dj2,
                                fraction: 1.0,
                            });
                        }
                    }
                }

                if facet_slope > best_slope {
                    best_slope = facet_slope;
                    best_dirs = facet_dirs;
                }
            }
            out[j * w + i] = best_dirs;
        }
    }
    out
}

fn build_donors(receivers: &[Vec<FlowReceiver>], w: usize, h: usize) -> Vec<Vec<usize>> {
    let n = w * h;
    let mut donors = vec![Vec::new(); n];
    for idx in 0..n {
        let i = (idx % w) as i32;
        let j = (idx / w) as i32;
        for fd in &receivers[idx] {
            let ni = i + fd.di;
            let nj = j + fd.dj;
            if ni < 0 || nj < 0 || ni >= w as i32 || nj >= h as i32 {
                continue;
            }
            let nidx = nj as usize * w + ni as usize;
            donors[nidx].push(idx);
        }
    }
    for d in &mut donors {
        d.sort_unstable();
        d.dedup();
    }
    donors
}

/// Upstream → downstream topological order (sources first).
fn topological_order(
    donors: &[Vec<usize>],
    receivers: &[Vec<FlowReceiver>],
    w: usize,
    h: usize,
) -> Vec<usize> {
    let n = w * h;
    let mut indegree = vec![0u32; n];
    for i in 0..n {
        indegree[i] = donors[i].len() as u32;
    }

    let mut queue: VecDeque<usize> = (0..n).filter(|&i| indegree[i] == 0).collect();
    let mut seeds: Vec<usize> = queue.drain(..).collect();
    seeds.sort_unstable();
    queue.extend(seeds);

    let mut order = Vec::with_capacity(n);
    let mut remaining = indegree;
    while let Some(i) = queue.pop_front() {
        order.push(i);
        let i_x = (i % w) as i32;
        let i_y = (i / w) as i32;
        let mut nexts = Vec::new();
        for fd in &receivers[i] {
            let ni = i_x + fd.di;
            let nj = i_y + fd.dj;
            if ni < 0 || nj < 0 || ni >= w as i32 || nj >= h as i32 {
                continue;
            }
            let nidx = nj as usize * w + ni as usize;
            if remaining[nidx] > 0 {
                remaining[nidx] -= 1;
                if remaining[nidx] == 0 {
                    nexts.push(nidx);
                }
            }
        }
        nexts.sort_unstable();
        queue.extend(nexts);
    }

    // Cycles / flats: append any unvisited cells in index order.
    if order.len() < n {
        let mut seen = vec![false; n];
        for &i in &order {
            seen[i] = true;
        }
        for i in 0..n {
            if !seen[i] {
                order.push(i);
            }
        }
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geomorph::{accumulate_drainage_area, Precipitation};
    use crate::heightfield::HeightfieldMetrics;

    #[test]
    fn d8_deterministic_ties() {
        let m = HeightfieldMetrics::new(3, 3, 3.0, 3.0);
        let mut hf = Heightfield::filled(m, 2.0);
        hf.set(1, 1, 3.0);
        // Equal drop to all 8 → prefer dir 0 (1,0).
        let (dirs, _) = flow_direction_d8(&hf);
        assert_eq!(dirs[4], 0);
        let a = build_flow_graph(&hf, FlowModel::D8);
        let b = build_flow_graph(&hf, FlowModel::D8);
        assert_eq!(a.d8_dir, b.d8_dir);
        assert_eq!(a.topo_order, b.topo_order);
    }

    #[test]
    fn pyramid_flows_outward_down() {
        let m = HeightfieldMetrics::new(9, 9, 9.0, 9.0);
        let mut hf = Heightfield::zeros(m);
        for j in 0..9 {
            for i in 0..9 {
                let d = (i as i32 - 4).abs() + (j as i32 - 4).abs();
                hf.set(i, j, 10.0 - d as f32);
            }
        }
        let g = build_flow_graph(&hf, FlowModel::D8);
        let acc = accumulate_drainage_area(&g, &Precipitation::uniform(1.0));
        // Corners seed at least their own cell; some path collects several upstream.
        assert!(acc[0] >= 1.0);
        assert!(acc.iter().any(|&a| a > 5.0));
    }

    #[test]
    fn d8_marks_local_sink_as_no_flow() {
        let m = HeightfieldMetrics::new(3, 3, 3.0, 3.0);
        let mut hf = Heightfield::filled(m, 2.0);
        hf.set(1, 1, 1.0);
        let g = build_flow_graph(&hf, FlowModel::D8);
        let acc = accumulate_drainage_area(&g, &Precipitation::uniform(1.0));
        // Center pit has no lower neighbour → sink; all 8 neighbours drain into it.
        assert_eq!(g.d8_dir[4], NO_FLOW);
        assert_eq!(acc[4], 9.0);
    }

    #[test]
    fn downhill_flow_single_outlet_sanity() {
        let m = HeightfieldMetrics::new(9, 9, 90.0, 90.0);
        let mut hf = Heightfield::zeros(m);
        for j in 0..9 {
            for i in 0..9 {
                hf.set(i, j, 20.0 - j as f32);
            }
        }
        let g = build_flow_graph(&hf, FlowModel::D8);
        let acc = accumulate_drainage_area(&g, &Precipitation::uniform(1.0));
        // Bottom row collects the most water on a south-draining plane.
        let top: f32 = (0..9).map(|i| acc[i]).sum();
        let bot: f32 = (0..9).map(|i| acc[8 * 9 + i]).sum();
        assert!(bot > top);
        // Interior cells flow south (dir index 2 = (0, 1)).
        let south = g.d8_dir.iter().filter(|&&d| d == 2).count();
        assert!(south > 20);
    }

    #[test]
    fn dinfinity_routes_within_the_winning_facet() {
        let m = HeightfieldMetrics::new(5, 5, 5.0, 5.0);
        let mut hf = Heightfield::zeros(m);
        for j in 0..5 {
            for i in 0..5 {
                hf.set(i, j, 100.0 - i as f32 - 0.5 * j as f32);
            }
        }
        let routed = flow_direction_dinfinity(&hf);
        let center = &routed[2 * 5 + 2];
        assert_eq!(center.len(), 2);
        assert!(center.iter().any(|d| (d.di, d.dj) == (1, 0)));
        assert!(center.iter().any(|d| (d.di, d.dj) == (1, 1)));
        let total: f32 = center.iter().map(|d| d.fraction).sum();
        assert!((total - 1.0).abs() < 1e-6);
        assert!(center.iter().all(|d| d.fraction > 0.0));
    }

    #[test]
    fn dinfinity_cardinal_plane_does_not_split_sideways() {
        let m = HeightfieldMetrics::new(5, 5, 5.0, 5.0);
        let mut hf = Heightfield::zeros(m);
        for j in 0..5 {
            for i in 0..5 {
                hf.set(i, j, 20.0 - i as f32);
            }
        }
        let routed = flow_direction_dinfinity(&hf);
        let center = &routed[2 * 5 + 2];
        assert_eq!(center.len(), 1);
        assert_eq!((center[0].di, center[0].dj), (1, 0));
        assert!((center[0].fraction - 1.0).abs() < 1e-6);
    }
}
