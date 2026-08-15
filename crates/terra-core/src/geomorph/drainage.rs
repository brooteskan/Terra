//! Drainage-aware ridge / valley analysis.

use crate::heightfield::Heightfield;
use crate::mask::MaskField;
use crate::spatial_kernels::jump_flood_distance;

use super::derivatives::ridge_valley_likelihood;
use super::routing::{FlowGraph, D8_OFFSETS};
use super::streams::StreamNetwork;

#[derive(Debug, Clone)]
pub struct DrainageAnalysis {
    pub ridge_mask: MaskField,
    pub valley_mask: MaskField,
    pub distance_to_ridge: MaskField,
    pub distance_to_valley: MaskField,
    pub drainage_density: MaskField,
    pub ridge_likelihood: MaskField,
    pub valley_likelihood: MaskField,
}

/// Combined drainage-aware morphometry.
pub fn drainage_analysis(
    hf: &Heightfield,
    graph: &FlowGraph,
    streams: &StreamNetwork,
    drainage_area: &[f32],
) -> DrainageAnalysis {
    let (ridge_l, valley_l) = ridge_valley_likelihood(hf, 0.0);
    let ridge = ridge_mask(hf, graph);
    let valley = valley_mask(streams, &valley_l);
    let distance_to_ridge = jump_flood_distance(&ridge);
    let distance_to_valley = jump_flood_distance(&valley);
    let density = drainage_density(streams, hf.metrics, 16);

    // Boost likelihoods with drainage cues.
    let m = hf.metrics;
    let mut ridge_likelihood = ridge_l;
    let mut valley_likelihood = valley_l;
    let max_a = drainage_area.iter().copied().fold(1.0f32, f32::max);
    for j in 0..m.height {
        for i in 0..m.width {
            let idx = (j * m.width + i) as usize;
            let a = (drainage_area[idx] / max_a).clamp(0.0, 1.0);
            let r = ridge_likelihood.get(i, j) * (1.0 - 0.5 * a) + 0.35 * ridge.get(i, j);
            let v = valley_likelihood.get(i, j) * (0.5 + 0.5 * a) + 0.35 * valley.get(i, j);
            ridge_likelihood.set(i, j, r.clamp(0.0, 1.0));
            valley_likelihood.set(i, j, v.clamp(0.0, 1.0));
        }
    }

    DrainageAnalysis {
        ridge_mask: ridge,
        valley_mask: valley,
        distance_to_ridge,
        distance_to_valley,
        drainage_density: density,
        ridge_likelihood,
        valley_likelihood,
    }
}

/// Ridge graph approximation: local maxima along D8 drainage divides
/// (cells that are higher than all downhill neighbours and not channels).
pub fn ridge_mask(hf: &Heightfield, graph: &FlowGraph) -> MaskField {
    use rayon::prelude::*;

    let m = hf.metrics;
    let w = m.width as usize;
    let h = m.height as usize;
    let n = w * h;
    let data: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|idx| {
            let i = idx % w;
            let j = idx / w;
            let h0 = hf.get(i as u32, j as u32);
            let mut is_ridge = true;
            let mut has_nb = false;
            for &(di, dj) in &D8_OFFSETS {
                let ni = i as i32 + di;
                let nj = j as i32 + dj;
                if ni < 0 || nj < 0 || ni >= w as i32 || nj >= h as i32 {
                    continue;
                }
                has_nb = true;
                if hf.get(ni as u32, nj as u32) >= h0 {
                    is_ridge = false;
                    break;
                }
            }
            let divide = graph.donors[idx].is_empty() && graph.receivers[idx].len() <= 1;
            if (is_ridge && has_nb) || (divide && is_ridge) {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    MaskField::from_raw(m, &data)
}

/// Valley graph: channel cells unioned with strong concavity seeds.
pub fn valley_mask(streams: &StreamNetwork, valley_likelihood: &MaskField) -> MaskField {
    let m = streams.channel_mask.metrics;
    let mut out = MaskField::zeros(m);
    for j in 0..m.height {
        for i in 0..m.width {
            let v = if streams.channel_mask.get(i, j) > 0.5 {
                1.0
            } else if valley_likelihood.get(i, j) > 0.65 {
                1.0
            } else {
                0.0
            };
            out.set(i, j, v);
        }
    }
    out
}

pub fn distance_to_ridge(ridge: &MaskField) -> MaskField {
    jump_flood_distance(ridge)
}

pub fn distance_to_valley(valley: &MaskField) -> MaskField {
    jump_flood_distance(valley)
}

/// Local channel density in a window of `radius` cells.
pub fn drainage_density(
    streams: &StreamNetwork,
    metrics: crate::heightfield::HeightfieldMetrics,
    radius: i32,
) -> MaskField {
    let m = metrics;
    let mut out = MaskField::zeros(m);
    let r = radius.max(1);
    let channel = &streams.channel_mask;
    for j in 0..m.height as i32 {
        for i in 0..m.width as i32 {
            let mut sum = 0.0f32;
            let mut n = 0.0f32;
            for dj in -r..=r {
                for di in -r..=r {
                    let ni = i + di;
                    let nj = j + dj;
                    if ni < 0 || nj < 0 || ni >= m.width as i32 || nj >= m.height as i32 {
                        continue;
                    }
                    sum += channel.get(ni as u32, nj as u32);
                    n += 1.0;
                }
            }
            out.set(i as u32, j as u32, (sum / n.max(1.0)).clamp(0.0, 1.0));
        }
    }
    out
}
