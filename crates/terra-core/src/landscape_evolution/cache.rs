//! Drainage topology cache for landscape evolution.

use crate::geomorph::{build_flow_graph, priority_flood_fill_from_outlets, FlowModel};
use crate::heightfield::Heightfield;
use crate::mask::MaskField;

/// Cached drainage products shared across Fast / Accurate passes.
#[derive(Debug, Clone)]
pub struct DrainageCache {
    /// Downstream neighbour index (`usize::MAX` = outlet / sink).
    pub receiver: Vec<usize>,
    /// Topological order: downstream → upstream (outlets first).
    pub topo_down_to_up: Vec<usize>,
    /// Drainage area in cell counts (precipitation-scaled later).
    pub accumulation: Vec<f32>,
    /// Normalised direction mask for aux publish.
    pub direction_mask: MaskField,
    /// Height snapshot used to build this cache.
    pub height_fingerprint: Vec<f32>,
    pub max_abs_delta: f32,
}

impl DrainageCache {
    pub fn build(height: &Heightfield, use_fill: bool, routing_outlets: &MaskField) -> Self {
        let filled = if use_fill {
            priority_flood_fill_from_outlets(height, routing_outlets)
        } else {
            height.clone()
        };
        let graph = build_flow_graph(&filled, FlowModel::D8);
        let n = graph.width * graph.height;

        // Single D8 receiver per cell (`usize::MAX` = outlet / sink).
        let mut receiver = vec![usize::MAX; n];
        for idx in 0..n {
            if let Some(r) = graph.d8_receiver_index(idx) {
                receiver[idx] = r;
            }
        }

        // Explicit routing outlets terminate drainage even when their local
        // gradient would otherwise point back into the domain.
        let mut direction_mask = graph.direction_mask.clone();
        for j in 0..graph.height {
            for i in 0..graph.width {
                if routing_outlets.get(i as u32, j as u32) > 0.5 {
                    receiver[j * graph.width + i] = usize::MAX;
                    direction_mask.set(i as u32, j as u32, 0.0);
                }
            }
        }

        // Re-accumulate over the cut receiver graph so discharge stops at an
        // explicit outlet instead of leaking into its former receiver.
        let mut accumulation = vec![1.0f32; n];
        for &idx in &graph.topo_order {
            let r = receiver[idx];
            if r != usize::MAX {
                accumulation[r] += accumulation[idx];
            }
        }

        // The graph's topo order runs upstream → downstream; reverse it for the
        // outlets-first order the analytical / iterative solvers consume.
        let mut topo_down_to_up = graph.topo_order.clone();
        topo_down_to_up.reverse();

        let height_fingerprint = height.to_dense();

        Self {
            receiver,
            topo_down_to_up,
            accumulation,
            direction_mask,
            height_fingerprint,
            max_abs_delta: 0.0,
        }
    }

    /// Whether height change since cache build warrants a topology rebuild.
    pub fn needs_rebuild(&self, height: &Heightfield, threshold_m: f32) -> bool {
        let dense = height.to_dense();
        if dense.len() != self.height_fingerprint.len() {
            return true;
        }
        let mut max_d = 0.0f32;
        for (a, b) in dense.iter().zip(self.height_fingerprint.iter()) {
            max_d = max_d.max((a - b).abs());
            if max_d > threshold_m {
                return true;
            }
        }
        false
    }

    pub fn refresh_fingerprint(&mut self, height: &Heightfield) {
        self.height_fingerprint = height.to_dense();
        self.max_abs_delta = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heightfield::HeightfieldMetrics;

    #[test]
    fn open_rim_outlets_terminate_receiver_chains_at_the_rim() {
        let m = HeightfieldMetrics::new(12, 12, 120.0, 120.0);
        let mut data = vec![0.0f32; (m.width * m.height) as usize];
        for j in 0..m.height {
            for i in 0..m.width {
                data[(j * m.width + i) as usize] = (m.width - i) as f32 + (m.height - j) as f32;
            }
        }
        let height = Heightfield::from_dense(m, &data);
        let mut outlets = MaskField::zeros(m);
        for j in 0..m.height {
            for i in 0..m.width {
                if i == 0 || j == 0 || i + 1 == m.width || j + 1 == m.height {
                    outlets.set(i, j, 1.0);
                }
            }
        }

        let cache = DrainageCache::build(&height, true, &outlets);
        let w = m.width as usize;
        let n = cache.receiver.len();
        for start in 0..n {
            let mut current = start;
            let mut terminated = false;
            for _ in 0..=n {
                let receiver = cache.receiver[current];
                if receiver == usize::MAX {
                    let i = (current % w) as u32;
                    let j = (current / w) as u32;
                    assert!(
                        outlets.get(i, j) > 0.5,
                        "chain from {start} ended at unmarked sink ({i},{j})"
                    );
                    terminated = true;
                    break;
                }
                current = receiver;
            }
            assert!(terminated, "receiver chain from {start} contains a cycle");
        }
    }

    #[test]
    fn authored_interior_outlet_cuts_receiver_and_accumulation() {
        let m = HeightfieldMetrics::new(11, 11, 110.0, 110.0);
        let (cx, cy) = (5u32, 5u32);
        let mut data = vec![0.0f32; (m.width * m.height) as usize];
        for j in 0..m.height {
            for i in 0..m.width {
                data[(j * m.width + i) as usize] =
                    (i as f32 - cx as f32).hypot(j as f32 - cy as f32);
            }
        }
        let height = Heightfield::from_dense(m, &data);
        let mut outlets = MaskField::zeros(m);
        outlets.set(cx, cy, 1.0);

        let cache = DrainageCache::build(&height, true, &outlets);
        let idx = (cy * m.width + cx) as usize;
        assert_eq!(cache.receiver[idx], usize::MAX);
        assert_eq!(cache.accumulation[idx], (m.width * m.height) as f32);
    }
}
