//! Geological stratum field — independent of the final terrain surface.
//!
//! Beds may be horizontal, tilted, or folded/warped. Each sample yields
//! thickness-band hardness, erodibility, material type, and layer id so
//! erosion and rock filters share one lithology query.

use crate::heightfield::Heightfield;
use crate::layer::{BedGeometry, Stratum, StratumMaterial};
use crate::mask::MaskField;

/// Parameters for synthesising a stratum field from height (filter path).
#[derive(Debug, Clone, Copy)]
pub struct StrataFieldParams {
    pub frequency: f32,
    pub seed: u64,
    pub hardness_contrast: f32,
    pub geometry: BedGeometry,
}

impl Default for StrataFieldParams {
    fn default() -> Self {
        Self {
            frequency: 0.04,
            seed: 7,
            hardness_contrast: 0.65,
            geometry: BedGeometry::Warped {
                frequency: 0.012,
                amplitude_m: 18.0,
                seed: 7,
            },
        }
    }
}

/// One lithology sample at a world position / height.
#[derive(Debug, Clone, Copy)]
pub struct BedSample {
    pub hardness: f32,
    pub erodibility: f32,
    pub stability: f32,
    pub material: StratumMaterial,
    /// Normalised layer id \[0,1\] for aux / visualisation.
    pub layer_id: f32,
    /// Stratigraphic coordinate (meters-like).
    pub strat_z: f32,
}

/// Stratigraphic scalar at (x, z, h). Beds are iso-surfaces of this field.
#[inline]
pub fn strat_coord(x: f32, z: f32, h: f32, geom: &BedGeometry) -> f32 {
    h + geom.depth_warp(x, z)
}

/// Sample synthetic beds from warped elevation (arid / Strata filter path).
pub fn sample_bed(
    x: f32,
    z: f32,
    h: f32,
    h_min: f32,
    _span: f32,
    p: &StrataFieldParams,
) -> BedSample {
    let freq = p.frequency.max(1e-5);
    let contrast = p.hardness_contrast.clamp(0.0, 1.0);
    let strat_z = strat_coord(x, z, h, &p.geometry);
    let phase = (strat_z - h_min) * freq * 2.5;
    let layer = phase.floor().rem_euclid(6.0);
    let bed = (layer * 0.17 + p.geometry.depth_warp(x, z) * 0.02).fract();

    let (hardness, material) = if bed < 0.45 {
        (0.55 + contrast * 0.4, StratumMaterial::Sedimentary)
    } else if bed < 0.75 {
        (0.35 + contrast * 0.15, StratumMaterial::Sedimentary)
    } else {
        (
            0.12 + (1.0 - contrast) * 0.2,
            StratumMaterial::Unconsolidated,
        )
    };
    let hardness = hardness.clamp(0.05, 0.98);
    let erodibility = (1.0 - hardness).clamp(0.02, 0.98);
    let stability = material.stability() * (0.35 + 0.65 * hardness);

    BedSample {
        hardness,
        erodibility,
        stability: stability.clamp(0.0, 1.0),
        material,
        layer_id: (layer / 5.0).clamp(0.0, 1.0),
        strat_z,
    }
}

/// Build hardness + layer-id fields from height (shared by Rocky Layers / Strata).
pub fn strata_fields(
    hf: &Heightfield,
    frequency: f32,
    seed: u64,
    hardness_contrast: f32,
) -> (MaskField, MaskField) {
    let p = StrataFieldParams {
        frequency,
        seed,
        hardness_contrast,
        geometry: BedGeometry::Warped {
            frequency: frequency.max(1e-5) * 0.35,
            amplitude_m: 12.0 + hardness_contrast * 20.0,
            seed: seed ^ 0x57A7A,
        },
    };
    strata_fields_with(hf, &p)
}

pub fn strata_fields_with(hf: &Heightfield, p: &StrataFieldParams) -> (MaskField, MaskField) {
    let m = hf.metrics;
    let mut hardness = MaskField::zeros(m);
    let mut layer_id = MaskField::zeros(m);

    let mut h_min = f32::INFINITY;
    let mut h_max = f32::NEG_INFINITY;
    for j in 0..m.height {
        for i in 0..m.width {
            let z = hf.get(i, j);
            h_min = h_min.min(z);
            h_max = h_max.max(z);
        }
    }
    let span = (h_max - h_min).max(1.0);

    for j in 0..m.height {
        for i in 0..m.width {
            let x = m.world_x(i);
            let z = m.world_z(j);
            let h0 = hf.get(i, j);
            let s = sample_bed(x, z, h0, h_min, span, p);
            hardness.set(i, j, s.hardness);
            layer_id.set(i, j, s.layer_id);
        }
    }
    (hardness, layer_id)
}

/// Preferentially recess soft beds on steep faces — exposes strata on cliffs/canyons.
pub fn expose_strata_height(
    hf: &Heightfield,
    hardness: &MaskField,
    amount: f32,
    cliff_mask: &MaskField,
) -> Heightfield {
    let mut out = hf.clone();
    let amt = amount.max(0.0);
    for j in 0..hf.metrics.height {
        for i in 0..hf.metrics.width {
            let face = cliff_mask.get(i, j);
            if face < 1e-4 {
                continue;
            }
            let soft = 1.0 - hardness.get(i, j);
            let h0 = hf.get(i, j);
            let recess = soft * amt * face;
            let ledge = if soft < 0.35 {
                soft * amt * 0.15 * face
            } else {
                0.0
            };
            out.set(i, j, h0 - recess + ledge);
        }
    }
    out
}

/// Look up an authored [`Stratum`] stack at stratigraphic depth (meters into subsurface).
pub fn stratum_at_depth<'a>(strata: &'a [Stratum], depth: f32) -> Option<&'a Stratum> {
    if strata.is_empty() {
        return None;
    }
    let mut remaining = depth.max(0.0);
    for s in strata {
        let t = s.thickness.max(0.0);
        if remaining <= t || t >= 1.0e5 {
            return Some(s);
        }
        remaining -= t;
    }
    strata.last()
}

/// Depth below Materials reference, warped by bed geometry.
pub fn strata_depth_m(h_ref: f32, h: f32, x: f32, z: f32, geom: &BedGeometry) -> f32 {
    let warp = geom.depth_warp(x, z);
    (h_ref - h + warp).max(0.0)
}

/// Folded-bed helper used by the cosmetic Strata EffectFilter.
pub fn strata_band_displace(
    hf: &Heightfield,
    frequency: f32,
    amount: f32,
    seed: u64,
) -> Heightfield {
    let mut out = hf.clone();
    let freq = frequency.max(1e-5);
    let amp = amount.max(0.0);
    let geom = BedGeometry::Folded {
        amplitude_m: amp * 0.35,
        wavelength_m: (1.0 / freq).clamp(8.0, 400.0),
        seed,
    };
    let p = StrataFieldParams {
        frequency: freq,
        seed,
        hardness_contrast: 0.7,
        geometry: geom,
    };
    let (hardness, _) = strata_fields_with(hf, &p);
    // Convert soft/hard beds into subtle ledge displacement (not random noise).
    for j in 0..hf.metrics.height {
        for i in 0..hf.metrics.width {
            let h0 = hf.get(i, j);
            let k = hardness.get(i, j);
            let band = (k - 0.5) * 2.0;
            out.set(i, j, h0 + band * amp * 0.5);
        }
    }
    // Keep a light sinusoidal component for fine bedding read when amount is small.
    if amp > 1e-4 {
        for j in 0..hf.metrics.height {
            for i in 0..hf.metrics.width {
                let h0 = out.get(i, j);
                let x = hf.metrics.world_x(i);
                let z = hf.metrics.world_z(j);
                let s = strat_coord(x, z, hf.get(i, j), &p.geometry);
                let fine = ((s * freq).sin() * 0.5 + 0.5) * amp * 0.35;
                out.set(i, j, h0 + fine - amp * 0.175);
            }
        }
    }
    out
}
