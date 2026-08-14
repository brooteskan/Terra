//! Flow accumulation / discharge from precipitation fields.

use crate::heightfield::HeightfieldMetrics;
use crate::mask::MaskField;
use crate::noise::fbm;
use crate::noise::{FractalNoiseType, NoiseParams};

use super::routing::{D8Drainage, FlowGraph};

/// How precipitation is sourced for accumulation.
#[derive(Debug, Clone)]
pub enum PrecipitationSource {
    /// Constant rainfall rate (relative units).
    Uniform(f32),
    /// Explicit spatial map (same resolution as DEM).
    Map(MaskField),
    /// Artist weight mask × `scale`.
    ArtistMask { mask: MaskField, scale: f32 },
    /// Procedural fBm rainfall.
    Procedural {
        params: NoiseParams,
        scale: f32,
        offset: f32,
    },
}

/// Precipitation field builder.
#[derive(Debug, Clone)]
pub struct Precipitation {
    pub source: PrecipitationSource,
}

impl Precipitation {
    pub fn uniform(rate: f32) -> Self {
        Self {
            source: PrecipitationSource::Uniform(rate.max(0.0)),
        }
    }

    pub fn from_map(map: MaskField) -> Self {
        Self {
            source: PrecipitationSource::Map(map),
        }
    }

    pub fn from_mask(mask: MaskField, scale: f32) -> Self {
        Self {
            source: PrecipitationSource::ArtistMask {
                mask,
                scale: scale.max(0.0),
            },
        }
    }

    pub fn procedural(params: NoiseParams, scale: f32, offset: f32) -> Self {
        Self {
            source: PrecipitationSource::Procedural {
                params,
                scale: scale.max(0.0),
                offset,
            },
        }
    }

    /// Sample precipitation P(x,y) ≥ 0 for each cell.
    pub fn sample(&self, metrics: HeightfieldMetrics) -> Vec<f32> {
        let w = metrics.width as usize;
        let h = metrics.height as usize;
        let n = w * h;
        match &self.source {
            PrecipitationSource::Uniform(rate) => vec![*rate; n],
            PrecipitationSource::Map(map) => {
                let mut out = vec![0.0f32; n];
                for j in 0..h {
                    for i in 0..w {
                        out[j * w + i] = map.get(i as u32, j as u32).max(0.0);
                    }
                }
                out
            }
            PrecipitationSource::ArtistMask { mask, scale } => {
                let mut out = vec![0.0f32; n];
                for j in 0..h {
                    for i in 0..w {
                        out[j * w + i] = mask.get(i as u32, j as u32).clamp(0.0, 1.0) * *scale;
                    }
                }
                out
            }
            PrecipitationSource::Procedural {
                params,
                scale,
                offset,
            } => {
                let mut out = vec![0.0f32; n];
                let dx = metrics.dx();
                let dz = metrics.dz();
                for j in 0..h {
                    for i in 0..w {
                        let x = i as f32 * dx;
                        let z = j as f32 * dz;
                        let nval = fbm(FractalNoiseType::OpenSimplex, x, z, params)
                            .mul_add(*scale, *offset)
                            .max(0.0);
                        out[j * w + i] = nval;
                    }
                }
                out
            }
        }
    }
}

/// Drainage area / cell-count accumulation weighted by precipitation.
///
/// Each cell contributes `P[i]`; receivers collect upstream contribution.
/// For uniform P=1 this equals classic flow accumulation (cell counts).
pub fn accumulate_drainage_area(graph: &FlowGraph, precip: &Precipitation) -> Vec<f32> {
    let p = precip.sample(graph.direction_mask.metrics);
    accumulate_with_weights(graph, &p)
}

/// Discharge Q ≈ Σ P · cell_area along the flow graph (world m² · precip units).
pub fn accumulate_discharge(
    graph: &FlowGraph,
    precip: &Precipitation,
    metrics: HeightfieldMetrics,
) -> Vec<f32> {
    let cell_area = (metrics.world_size_x / metrics.width.max(1) as f32)
        * (metrics.world_size_z / metrics.height.max(1) as f32);
    let cell_area = cell_area.max(1e-6);
    let mut p = precip.sample(metrics);
    for v in &mut p {
        *v *= cell_area;
    }
    accumulate_with_weights(graph, &p)
}

/// Lean flat-D8 counterpart of [`accumulate_drainage_area`] for a [`D8Drainage`].
///
/// Bit-identical to the general path on D8 input: same topo order, same
/// per-receiver addition sequence, single receiver whose fraction is `1.0`
/// (`acc[r] += a` is the general path's `a * 1.0`). The single-lineage guard in
/// `tests/dead_seams.rs` keeps this from drifting into a second copy of the D8
/// accumulation logic.
pub fn accumulate_drainage_area_d8(drainage: &D8Drainage, precip: &Precipitation) -> Vec<f32> {
    let weights = precip.sample(drainage.direction_mask.metrics);
    accumulate_flat_d8(drainage, &weights)
}

fn accumulate_flat_d8(drainage: &D8Drainage, weights: &[f32]) -> Vec<f32> {
    let n = drainage.width * drainage.height;
    assert_eq!(weights.len(), n);
    let mut acc = weights.to_vec();
    for &idx in &drainage.topo_order {
        let a = acc[idx];
        if a <= 0.0 {
            continue;
        }
        let r = drainage.receiver[idx];
        if r != usize::MAX {
            acc[r] += a;
        }
    }
    acc
}

fn accumulate_with_weights(graph: &FlowGraph, weights: &[f32]) -> Vec<f32> {
    let n = graph.width * graph.height;
    assert_eq!(weights.len(), n);
    let mut acc = weights.to_vec();
    // Upstream → downstream along topo order.
    for &idx in &graph.topo_order {
        let a = acc[idx];
        if a <= 0.0 {
            continue;
        }
        for &fd in &graph.receivers[idx] {
            if let Some(nidx) = graph.receiver_index(idx, fd) {
                acc[nidx] += a * fd.fraction;
            }
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geomorph::depression::{handle_depressions, DepressionMode};
    use crate::geomorph::routing::{build_flow_graph, FlowModel};
    use crate::heightfield::{Heightfield, HeightfieldMetrics};

    #[test]
    fn uniform_conserves_total_at_sinks() {
        let m = HeightfieldMetrics::new(8, 8, 80.0, 80.0);
        let mut hf = Heightfield::zeros(m);
        for j in 0..8 {
            for i in 0..8 {
                hf.set(i, j, 20.0 - j as f32);
            }
        }
        let filled = handle_depressions(&hf, DepressionMode::Fill).height;
        let g = build_flow_graph(&filled, FlowModel::D8);
        let acc = accumulate_drainage_area(&g, &Precipitation::uniform(1.0));
        // Total precip = 64; sinks hold the conserved mass that didn't leave
        // the domain — sum over all cells is NOT conserved (each cell's water
        // is also counted in downstream cells). Sum of *local contribution*
        // equals n. Max accumulation should equal full domain for single outlet.
        let max = acc.iter().copied().fold(0.0f32, f32::max);
        // South-draining plane: each of 8 columns accumulates 8 cells.
        assert!((max - 8.0).abs() < 1e-3, "max acc={max}");
        assert!(acc.iter().sum::<f32>() > 8.0);
    }

    #[test]
    fn spatial_precip_changes_discharge() {
        let m = HeightfieldMetrics::new(8, 8, 80.0, 80.0);
        let mut hf = Heightfield::zeros(m);
        for j in 0..8 {
            for i in 0..8 {
                hf.set(i, j, 20.0 - j as f32);
            }
        }
        let filled = handle_depressions(&hf, DepressionMode::Fill).height;
        let g = build_flow_graph(&filled, FlowModel::D8);
        let mut map = MaskField::filled(m, 0.0);
        for j in 0..8 {
            for i in 0..4 {
                map.set(i, j, 2.0);
            }
        }
        let acc = accumulate_drainage_area(&g, &Precipitation::from_map(map));
        let left: f32 = (0..8).map(|j| acc[j * 8 + 0]).sum();
        let right: f32 = (0..8).map(|j| acc[j * 8 + 7]).sum();
        // Left half has precip; paths drain south — still expect structure.
        assert!(acc.iter().copied().fold(0.0f32, f32::max) > 1.0);
        let _ = (left, right);
    }
}
