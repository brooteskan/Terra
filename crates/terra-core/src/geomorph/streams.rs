//! Channel network extraction and Horton–Strahler ordering.

use crate::analyze::jump_flood_distance;
use crate::heightfield::HeightfieldMetrics;
use crate::mask::MaskField;

use super::routing::FlowGraph;

#[derive(Debug, Clone)]
pub struct StreamExtractParams {
    /// Minimum drainage area (cell-weighted) to form a channel.
    pub accumulation_threshold: f32,
    /// Channel width scale in cells: width ≈ scale * sqrt(A / threshold).
    pub width_scale: f32,
}

impl Default for StreamExtractParams {
    fn default() -> Self {
        Self {
            accumulation_threshold: 8.0,
            width_scale: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StreamNetwork {
    pub channel_mask: MaskField,
    pub channel_width: MaskField,
    /// Raw Strahler order (integer as f32).
    pub order: MaskField,
    /// Normalised order \[0,1\].
    pub order_normalised: MaskField,
    pub distance_to_channel: MaskField,
    /// Hierarchy weight: higher-order channels dominate.
    pub hierarchy: MaskField,
}

/// Extract channels from accumulation and compute Strahler order on the D8 graph.
pub fn extract_streams(
    graph: &FlowGraph,
    drainage_area: &[f32],
    metrics: HeightfieldMetrics,
    params: &StreamExtractParams,
) -> StreamNetwork {
    let w = graph.width;
    let h = graph.height;
    let thr = params.accumulation_threshold.max(1.0);

    let mut channel_mask = MaskField::zeros(metrics);
    let mut channel_width = MaskField::zeros(metrics);
    for j in 0..h {
        for i in 0..w {
            let idx = j * w + i;
            let a = drainage_area[idx];
            if a >= thr {
                channel_mask.set(i as u32, j as u32, 1.0);
                let width = params.width_scale * (a / thr).sqrt();
                channel_width.set(i as u32, j as u32, width);
            }
        }
    }

    let order_raw = strahler_order(graph, drainage_area, thr);
    let max_order = order_raw.iter().copied().max().unwrap_or(1).max(1) as f32;
    let mut order = MaskField::zeros(metrics);
    let mut order_normalised = MaskField::zeros(metrics);
    let mut hierarchy = MaskField::zeros(metrics);
    for j in 0..h {
        for i in 0..w {
            let o = order_raw[j * w + i] as f32;
            order.set(i as u32, j as u32, o);
            order_normalised.set(i as u32, j as u32, o / max_order);
            hierarchy.set(
                i as u32,
                j as u32,
                if o > 0.0 { o / max_order } else { 0.0 },
            );
        }
    }

    let dist = distance_to_channel(&channel_mask);

    StreamNetwork {
        channel_mask,
        channel_width,
        order,
        order_normalised,
        distance_to_channel: dist,
        hierarchy,
    }
}

/// Classic Horton–Strahler order along the D8 donor/receiver graph.
///
/// Only cells with `drainage_area >= threshold` participate. Sources start at
/// order 1; when two+ inflows of equal max order meet, order increments.
pub fn strahler_order(graph: &FlowGraph, drainage_area: &[f32], threshold: f32) -> Vec<u32> {
    let n = graph.width * graph.height;
    let thr = threshold.max(1e-6);
    let mut order = vec![0u32; n];

    // Upstream → downstream.
    for &idx in &graph.topo_order {
        if drainage_area[idx] < thr {
            continue;
        }
        let mut max_in = 0u32;
        let mut max_count = 0u32;
        for &d in &graph.donors[idx] {
            if drainage_area[d] < thr {
                continue;
            }
            // Only count donors that actually list this cell as a receiver.
            let flows_here = graph.receivers[d].iter().any(|fd| {
                graph
                    .receiver_index(d, *fd)
                    .map(|r| r == idx)
                    .unwrap_or(false)
            });
            if !flows_here {
                continue;
            }
            let o = order[d];
            if o > max_in {
                max_in = o;
                max_count = 1;
            } else if o == max_in && o > 0 {
                max_count += 1;
            }
        }
        order[idx] = if max_in == 0 {
            1
        } else if max_count >= 2 {
            max_in + 1
        } else {
            max_in
        };
    }
    order
}

/// Estimated channel half-width in cells from accumulation.
pub fn channel_width_estimate(drainage_area: &[f32], threshold: f32, scale: f32) -> Vec<f32> {
    let thr = threshold.max(1e-6);
    drainage_area
        .iter()
        .map(|&a| {
            if a >= thr {
                scale * (a / thr).sqrt()
            } else {
                0.0
            }
        })
        .collect()
}

/// Normalised distance to nearest channel cell (jump-flood).
pub fn distance_to_channel(channel_mask: &MaskField) -> MaskField {
    jump_flood_distance(channel_mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geomorph::accumulation::{accumulate_drainage_area, Precipitation};
    use crate::geomorph::depression::{handle_depressions, DepressionMode};
    use crate::geomorph::routing::{build_flow_graph, FlowModel};
    use crate::heightfield::HeightfieldMetrics;

    #[test]
    fn valley_produces_ordered_channel() {
        let m = HeightfieldMetrics::new(24, 24, 240.0, 240.0);
        let hf = super::super::fixtures::single_valley(m);
        let filled = handle_depressions(&hf, DepressionMode::Fill).height;
        let g = build_flow_graph(&filled, FlowModel::D8);
        let acc = accumulate_drainage_area(&g, &Precipitation::uniform(1.0));
        let net = extract_streams(
            &g,
            &acc,
            m,
            &StreamExtractParams {
                accumulation_threshold: 6.0,
                width_scale: 1.0,
            },
        );
        let max_o = net.order.data().iter().copied().fold(0.0f32, f32::max);
        assert!(max_o >= 1.0, "max order={max_o}");
        assert!(net.channel_mask.data().iter().any(|&v| v > 0.5));
    }
}
