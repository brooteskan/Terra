//! Multi-scale terrain amplification (geological realism).
//!
//! Turns low/medium-resolution authored landforms into hydrologically consistent
//! high-resolution relief by adding **drainage-conditioned** meso/micro structure.
//!
//! Research anchors:
//! - Schott et al. 2023/2024 — interactive erosion / multi-scale amplify (silhouette
//!   preservation, wavelength-aware detail)
//! - Grenier et al. 2024 — cascaded structured procedural patterns aligned to slope
//!   and flow (no isotropic fBm soup)
//!
//! Frequency bands (metres, scale-aware defaults from world extent):
//! - **Macro** — artist silhouette (preserved; never randomly noised)
//! - **Meso** — ridges, tributaries, secondary valleys
//! - **Micro** — gullies, ridge breakup, fine erosion texture
//!
//! Every height delta answers a geomorphological question (ridge / gully / rock /
//! deposition). Fine channels nest inside broader drainage organisation.

use crate::geomorph::{
    accumulate_drainage_area, build_flow_graph, gradient_components, mean_curvature,
    priority_flood_fill, ridge_valley_likelihood, slope_magnitude, FlowModel, Precipitation,
};
use crate::heightfield::{Heightfield, HeightfieldMetrics};
use crate::mask::MaskField;
use crate::noise::value_noise2;

use serde::{Deserialize, Serialize};

/// Wavelength ranges in metres for the three amplification bands.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AmplificationBands {
    /// Artist landform / silhouette (detail must not rewrite this band).
    pub macro_m: (f32, f32),
    /// Ridges, tributaries, secondary valleys.
    pub meso_m: (f32, f32),
    /// Gullies, surface breakup, fine erosion texture.
    pub micro_m: (f32, f32),
}

impl AmplificationBands {
    /// Scale-aware defaults from world diagonal (not hardcoded absolute ranges).
    ///
    /// Example on a ~4 km world: macro ≈ 500–5 km, meso ≈ 50–500 m, micro ≈ 2–50 m.
    pub fn from_world(metrics: HeightfieldMetrics) -> Self {
        let diag = metrics.world_size_x.hypot(metrics.world_size_z).max(64.0);
        let cell = metrics.dx().max(metrics.dz()).max(0.25);
        // Macro owns the coarsest quarter of the world diagonal upward.
        let macro_lo = (diag * 0.12).clamp(cell * 16.0, diag * 0.5);
        let macro_hi = diag.max(macro_lo * 1.5);
        // Meso sits between ~1% and ~12% of diagonal.
        let meso_hi = macro_lo * 0.95;
        let meso_lo = (diag * 0.012).clamp(cell * 4.0, meso_hi * 0.35);
        // Micro nests under meso, down to a few cells.
        let micro_hi = meso_lo * 0.95;
        let micro_lo = (diag * 0.0005).clamp(cell * 1.5, micro_hi * 0.4);
        Self {
            macro_m: (macro_lo, macro_hi),
            meso_m: (meso_lo, meso_hi),
            micro_m: (micro_lo, micro_hi),
        }
    }

    pub fn meso_centre_m(self) -> f32 {
        (self.meso_m.0 * self.meso_m.1).sqrt()
    }

    pub fn micro_centre_m(self) -> f32 {
        (self.micro_m.0 * self.micro_m.1).sqrt()
    }

    pub fn silhouette_wavelength_m(self) -> f32 {
        self.macro_m.0
    }
}

/// Parameters for drainage-conditioned multi-scale amplification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainAmplificationParams {
    /// Peak meso amplitude in metres (ridge / tributary relief).
    #[serde(default = "default_meso_amp")]
    pub meso_amplitude_m: f32,
    /// Peak micro amplitude in metres (gullies / breakup).
    #[serde(default = "default_micro_amp")]
    pub micro_amplitude_m: f32,
    /// Cascade depth for structured patterns (2–6). Narrower patterns nest in broader ones.
    #[serde(default = "default_cascades")]
    pub cascade_levels: u32,
    /// How strongly patterns align to flow / aspect (0 = isotropic frame, 1 = full).
    #[serde(default = "default_flow_align")]
    pub flow_alignment: f32,
    /// Minimum slope (rise/run) before rock/ridge detail activates.
    #[serde(default = "default_slope_gate")]
    pub slope_gate: f32,
    /// Preserve major drainage corridors (suppress random incision on high accumulation).
    #[serde(default = "default_preserve_drainage")]
    pub preserve_drainage: f32,
    /// Strength of macro-silhouette lock (high-pass the amplify delta).
    #[serde(default = "default_silhouette")]
    pub silhouette_lock: f32,
    /// Ridge-conditioned breakup amount (0–1 multiplier on meso/micro ridge terms).
    #[serde(default = "default_ridge_breakup")]
    pub ridge_breakup: f32,
    /// Flow-conditioned gully / micro-channel amount.
    #[serde(default = "default_gully")]
    pub gully_strength: f32,
    /// Slope/lithology-conditioned rock roughness.
    #[serde(default = "default_rock")]
    pub rock_roughness: f32,
    /// Optional override bands; `None` → [`AmplificationBands::from_world`].
    #[serde(default)]
    pub bands: Option<AmplificationBands>,
    #[serde(default = "default_amp_seed")]
    pub seed: u64,
}

fn default_meso_amp() -> f32 {
    8.0
}
fn default_micro_amp() -> f32 {
    1.8
}
fn default_cascades() -> u32 {
    4
}
fn default_flow_align() -> f32 {
    0.85
}
fn default_slope_gate() -> f32 {
    0.12
}
fn default_preserve_drainage() -> f32 {
    0.7
}
fn default_silhouette() -> f32 {
    0.92
}
fn default_ridge_breakup() -> f32 {
    0.75
}
fn default_gully() -> f32 {
    0.85
}
fn default_rock() -> f32 {
    0.45
}
fn default_amp_seed() -> u64 {
    91
}

impl Default for TerrainAmplificationParams {
    fn default() -> Self {
        Self {
            meso_amplitude_m: default_meso_amp(),
            micro_amplitude_m: default_micro_amp(),
            cascade_levels: default_cascades(),
            flow_alignment: default_flow_align(),
            slope_gate: default_slope_gate(),
            preserve_drainage: default_preserve_drainage(),
            silhouette_lock: default_silhouette(),
            ridge_breakup: default_ridge_breakup(),
            gully_strength: default_gully(),
            rock_roughness: default_rock(),
            bands: None,
            seed: default_amp_seed(),
        }
    }
}

/// Products of a terrain amplification pass.
#[derive(Debug, Clone)]
pub struct TerrainAmplificationResult {
    /// Enhanced elevation (hydrologically organised detail; macro silhouette preserved).
    pub height: Heightfield,
    /// Fine-scale flow organisation used to drive nested channels \[0,1\].
    pub fine_flow: MaskField,
    /// Nested micro-channel / gully map \[0,1\].
    pub micro_channel: MaskField,
    /// Ridge-conditioned breakup intensity \[0,1\].
    pub ridge_breakup: MaskField,
    /// Fine erosion / incision intensity \[0,1\].
    pub fine_erosion: MaskField,
    /// Combined detail application weight (compat / overlays).
    pub detail_mask: MaskField,
}

/// Drainage-conditioned multi-scale amplification.
///
/// Anti-procedural-soup: never applies `height += generic_noise * amount`.
/// Each delta is gated by ridge / flow / slope / hardness / deposition cues.
pub fn amplify_terrain(
    input: &Heightfield,
    p: &TerrainAmplificationParams,
    flow_accumulation: Option<&MaskField>,
    hardness: Option<&MaskField>,
    protection: Option<&MaskField>,
) -> TerrainAmplificationResult {
    let m = input.metrics;
    let bands = p.bands.unwrap_or_else(|| AmplificationBands::from_world(m));
    let cascades = p.cascade_levels.clamp(2, 6);
    let align = p.flow_alignment.clamp(0.0, 1.0);
    let slope_gate = p.slope_gate.max(0.02);
    let preserve = p.preserve_drainage.clamp(0.0, 1.0);
    let silhouette = p.silhouette_lock.clamp(0.0, 1.0);

    let (gx, gz, gmag) = gradient_components(input, 0.0);
    let slope = slope_magnitude(input, 0.0);
    let curvature = mean_curvature(input, 0.0);
    let (ridge_l, valley_l) = ridge_valley_likelihood(input, 0.0);

    let flow = match flow_accumulation {
        Some(f) => normalize_mask(f),
        None => compute_flow_norm(input),
    };
    // Fine flow: emphasise tributary corridors (mid accumulation) for nesting.
    let fine_flow = build_fine_flow(&flow, &valley_l, &slope);

    let mut delta = MaskField::zeros(m);
    let mut ridge_breakup = MaskField::zeros(m);
    let mut micro_channel = MaskField::zeros(m);
    let mut fine_erosion = MaskField::zeros(m);
    let mut detail_mask = MaskField::zeros(m);

    // Parent envelopes so narrower cascades nest inside broader organisation.
    let mut meso_org = MaskField::zeros(m);
    let mut meso_channel_org = MaskField::zeros(m);

    // --- MESO: ridges / tributaries / secondary valleys ---
    apply_cascade_band(
        BandRole::Meso,
        input,
        &gx,
        &gz,
        &gmag,
        &slope,
        &curvature,
        &ridge_l,
        &valley_l,
        &flow,
        &fine_flow,
        hardness,
        protection,
        &bands,
        p,
        cascades,
        align,
        slope_gate,
        preserve,
        None,
        None,
        p.meso_amplitude_m,
        &mut delta,
        &mut ridge_breakup,
        &mut micro_channel,
        &mut fine_erosion,
        &mut detail_mask,
        &mut meso_org,
        &mut meso_channel_org,
    );

    // --- MICRO: gullies / ridge breakup / surface erosion (nested) ---
    apply_cascade_band(
        BandRole::Micro,
        input,
        &gx,
        &gz,
        &gmag,
        &slope,
        &curvature,
        &ridge_l,
        &valley_l,
        &flow,
        &fine_flow,
        hardness,
        protection,
        &bands,
        p,
        cascades,
        align,
        slope_gate,
        preserve,
        Some(&meso_org),
        Some(&meso_channel_org),
        p.micro_amplitude_m,
        &mut delta,
        &mut ridge_breakup,
        &mut micro_channel,
        &mut fine_erosion,
        &mut detail_mask,
        &mut MaskField::zeros(m), // unused parent write-back
        &mut MaskField::zeros(m),
    );

    // High-pass the amplify delta so macro silhouette stays artist-defined.
    let locked = if silhouette > 1e-4 {
        let smooth_r = wavelength_to_radius_texels(m, bands.silhouette_wavelength_m());
        let low = box_blur_mask(&delta, smooth_r);
        let mut hp = MaskField::zeros(m);
        for j in 0..m.height {
            for i in 0..m.width {
                let d = delta.get(i, j);
                let keep = d - low.get(i, j) * silhouette;
                hp.set(i, j, keep);
            }
        }
        hp
    } else {
        delta
    };

    let mut out = input.clone();
    for j in 0..m.height {
        for i in 0..m.width {
            out.set(i, j, input.get(i, j) + locked.get(i, j));
        }
    }

    // Renormalise diagnostic maps.
    ridge_breakup = clamp01_mask(&ridge_breakup);
    micro_channel = clamp01_mask(&micro_channel);
    fine_erosion = clamp01_mask(&fine_erosion);
    detail_mask = clamp01_mask(&detail_mask);

    TerrainAmplificationResult {
        height: out,
        fine_flow,
        micro_channel,
        ridge_breakup,
        fine_erosion,
        detail_mask,
    }
}

#[derive(Clone, Copy)]
enum BandRole {
    Meso,
    Micro,
}

#[allow(clippy::too_many_arguments)]
fn apply_cascade_band(
    role: BandRole,
    input: &Heightfield,
    gx: &MaskField,
    gz: &MaskField,
    gmag: &MaskField,
    slope: &MaskField,
    curvature: &MaskField,
    ridge_l: &MaskField,
    valley_l: &MaskField,
    flow: &MaskField,
    fine_flow: &MaskField,
    hardness: Option<&MaskField>,
    protection: Option<&MaskField>,
    bands: &AmplificationBands,
    p: &TerrainAmplificationParams,
    cascades: u32,
    align: f32,
    slope_gate: f32,
    preserve: f32,
    parent_org: Option<&MaskField>,
    parent_channel: Option<&MaskField>,
    amplitude_m: f32,
    delta: &mut MaskField,
    ridge_breakup: &mut MaskField,
    micro_channel: &mut MaskField,
    fine_erosion: &mut MaskField,
    detail_mask: &mut MaskField,
    org_out: &mut MaskField,
    channel_org_out: &mut MaskField,
) {
    let m = input.metrics;
    let (lo, hi) = match role {
        BandRole::Meso => bands.meso_m,
        BandRole::Micro => bands.micro_m,
    };
    let base_lambda = (lo * hi).sqrt().max(lo);
    let amp = amplitude_m.max(0.0);
    if amp < 1e-8 {
        return;
    }

    for j in 0..m.height {
        for i in 0..m.width {
            let protected = protection
                .map(|f| f.get(i, j))
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);
            if protected > 0.995 {
                continue;
            }
            let s = slope.get(i, j);
            let gate = ((s - slope_gate) / slope_gate).clamp(0.0, 1.0);
            let mag = gmag.get(i, j).max(1e-5);
            let gxi = gx.get(i, j);
            let gzi = gz.get(i, j);
            // Down-slope and across-slope frame (Grenier flow/slope alignment).
            let down_x = -gxi / mag;
            let down_z = -gzi / mag;
            let across_x = -gzi / mag;
            let across_z = gxi / mag;

            let x = m.world_x(i);
            let z = m.world_z(j);
            let ridge = ridge_l.get(i, j);
            let valley = valley_l.get(i, j);
            let f = flow.get(i, j);
            let ff = fine_flow.get(i, j);
            let hard = hardness.map(|h| h.get(i, j)).unwrap_or(0.0).clamp(0.0, 1.0);
            let curv = curvature.get(i, j); // ~[0,1] mean curvature map
                                            // Convex ridges sit toward high curvature in Terra's normalised map;
                                            // use ridge/valley fields as primary organisers.

            // Geomorphological gates (anti-soup).
            let drain_suppress = 1.0 - (f * preserve).clamp(0.0, 1.0);
            // Tributary window: enough water to organise channels, not trunk rivers.
            let tributary = (ff * (1.0 - f * 0.65)).clamp(0.0, 1.0);
            let ridge_gate = ridge * (1.0 - f * 0.55) * gate * p.ridge_breakup.clamp(0.0, 1.0);
            // Valleys organise gullies even when mid-accumulation is weak; tributaries boost.
            let gully_gate = ((valley * 0.7 + tributary * 0.9).min(1.0)
                * (0.35 + 0.65 * drain_suppress)
                * (0.25 + 0.75 * gate)
                * p.gully_strength.clamp(0.0, 1.0))
            .clamp(0.0, 1.0);
            let rock_gate = gate
                * hard.max(0.15)
                * (1.0 - valley * 0.7)
                * (1.0 - f * 0.4)
                * p.rock_roughness.clamp(0.0, 1.0);
            // Soft depositional texture only on low-slope, high-flow flats.
            let deposit_gate = ((0.18 - s) / 0.18).clamp(0.0, 1.0) * f * (1.0 - hard);

            let nest = match role {
                BandRole::Meso => 1.0,
                BandRole::Micro => {
                    let po = parent_org.map(|o| o.get(i, j)).unwrap_or(0.35);
                    let pc = parent_channel.map(|c| c.get(i, j)).unwrap_or(0.25);
                    // Micro must live inside meso organisation (cascaded nesting).
                    (0.2 + 0.8 * po.max(pc)).clamp(0.0, 1.0)
                }
            };

            let unlock = (1.0 - protected) * nest;
            let mut cell_delta = 0.0;
            let mut cell_ridge = 0.0;
            let mut cell_channel = 0.0;
            let mut cell_erosion = 0.0;
            let mut cell_org = 0.0;
            let mut cell_ch_org = 0.0;
            let mut amp_oct = 1.0;
            let mut norm = 0.0;
            let mut lambda = base_lambda;

            for o in 0..cascades {
                if lambda < lo * 0.85 {
                    break;
                }
                if lambda > hi * 1.15 && o > 0 {
                    lambda *= 0.55;
                    amp_oct *= 0.55;
                    continue;
                }

                let stretch = 1.0 + align * 5.0;
                let u = (x * across_x + z * across_z) / lambda;
                let v = (x * down_x + z * down_z) / (lambda * stretch);
                let seed = p.seed
                    ^ match role {
                        BandRole::Meso => 0xA11CE_u64,
                        BandRole::Micro => 0xBEEF_u64,
                    }
                    ^ ((o as u64) << 17);

                // Structured patterns — each answers a feature question.
                // Ridge breakup: anisotropic ridged noise across-slope on divides.
                let ridge_n = ridged_aniso(u, v * (0.35 + 0.65 * (1.0 - align)), seed);
                // Gullies: channel-like valleys elongated down-flow.
                let gully_n = channel_aniso(u * (0.4 + 0.6 * align), v, seed.wrapping_add(19));
                // Rock roughness: higher-frequency slope/lithology texture.
                let rock_n = value_noise2(u * 1.7, v * 1.7, seed.wrapping_add(41));
                // Depositional ripples: gentle across-flow undulation on flats.
                let dep_n = value_noise2(u * 0.8, v * 2.5, seed.wrapping_add(73));

                let ridge_term = ridge_n * ridge_gate * amp_oct;
                let gully_term = -gully_n.abs() * gully_gate * amp_oct; // incision
                let rock_term = rock_n * rock_gate * amp_oct * 0.55;
                let dep_term = dep_n * deposit_gate * amp_oct * 0.25;

                let contrib = ridge_term + gully_term + rock_term + dep_term;
                cell_delta += contrib;
                cell_ridge += ridge_term.abs();
                cell_channel += gully_n.abs() * gully_gate * amp_oct;
                cell_erosion += (-gully_term).max(0.0) + rock_term.abs() * 0.35;
                cell_org += ridge_gate * amp_oct + rock_gate * amp_oct * 0.5;
                cell_ch_org += gully_gate * amp_oct;
                norm += amp_oct;

                lambda *= 0.5;
                amp_oct *= 0.55;
            }

            if norm < 1e-6 {
                continue;
            }
            let scale = amp * unlock / norm;
            let d = cell_delta * scale;
            // Mild curvature feedback: reinforce existing concavity for gullies.
            let curv_bias = 1.0 + (valley - ridge) * (curv - 0.5) * 0.35;
            let d = d * curv_bias;

            delta.set(i, j, delta.get(i, j) + d);
            ridge_breakup.set(i, j, ridge_breakup.get(i, j) + cell_ridge * unlock);
            micro_channel.set(i, j, micro_channel.get(i, j) + cell_channel * unlock);
            fine_erosion.set(i, j, fine_erosion.get(i, j) + cell_erosion * unlock);
            let w = (ridge_gate + gully_gate + rock_gate + deposit_gate) * unlock;
            detail_mask.set(i, j, detail_mask.get(i, j).max(w));
            org_out.set(i, j, (cell_org / norm).clamp(0.0, 1.0));
            channel_org_out.set(i, j, (cell_ch_org / norm).clamp(0.0, 1.0));
        }
    }
}

fn ridged_aniso(u: f32, v: f32, seed: u64) -> f32 {
    // value_noise2 ∈ [-1,1] → ridged peaks near zero-crossings.
    let n = value_noise2(u, v, seed);
    1.0 - n.abs() * 2.0
}

fn channel_aniso(u: f32, v: f32, seed: u64) -> f32 {
    // Absolute value elongated down-flow → rill/gully profiles ∈ [-1,1].
    let n = value_noise2(u, v, seed);
    n.abs() * 2.0 - 1.0
}

fn build_fine_flow(flow: &MaskField, valley: &MaskField, slope: &MaskField) -> MaskField {
    let m = flow.metrics;
    let mut out = MaskField::zeros(m);
    for j in 0..m.height {
        for i in 0..m.width {
            let f = flow.get(i, j);
            // Log-shaped mid window so nearly-uniform accumulation still organises.
            let fl = (f * 8.0 + 1.0).ln() / (8.0f32 + 1.0).ln();
            let mid = 1.0 - ((fl - 0.35) * 2.8).abs().min(1.0);
            let v = valley.get(i, j);
            let s = (slope.get(i, j) / 0.45).clamp(0.0, 1.0);
            let organised = mid.max(0.0) * (0.35 + 0.65 * v) * (0.25 + 0.75 * s);
            // Valley/slope fallback so fine flow is never empty on coherent relief.
            let fallback = v * (0.2 + 0.8 * s) * 0.55;
            out.set(i, j, organised.max(fallback).clamp(0.0, 1.0));
        }
    }
    out
}

fn compute_flow_norm(input: &Heightfield) -> MaskField {
    let filled = priority_flood_fill(input);
    let graph = build_flow_graph(&filled, FlowModel::D8);
    let acc = accumulate_drainage_area(&graph, &Precipitation::uniform(1.0));
    let m = input.metrics;
    let mut field = MaskField::zeros(m);
    for j in 0..m.height {
        for i in 0..m.width {
            field.set(i, j, acc[(j * m.width + i) as usize]);
        }
    }
    normalize_mask(&field)
}

fn normalize_mask(src: &MaskField) -> MaskField {
    let m = src.metrics;
    let mut max_v = 1e-6f32;
    for j in 0..m.height {
        for i in 0..m.width {
            max_v = max_v.max(src.get(i, j));
        }
    }
    let mut out = MaskField::zeros(m);
    let inv = 1.0 / max_v;
    for j in 0..m.height {
        for i in 0..m.width {
            out.set(i, j, (src.get(i, j) * inv).clamp(0.0, 1.0));
        }
    }
    out
}

fn clamp01_mask(src: &MaskField) -> MaskField {
    let m = src.metrics;
    let mut max_v = 1e-8f32;
    for j in 0..m.height {
        for i in 0..m.width {
            max_v = max_v.max(src.get(i, j));
        }
    }
    let mut out = MaskField::zeros(m);
    let inv = 1.0 / max_v;
    for j in 0..m.height {
        for i in 0..m.width {
            out.set(i, j, (src.get(i, j) * inv).clamp(0.0, 1.0));
        }
    }
    out
}

fn wavelength_to_radius_texels(metrics: HeightfieldMetrics, wavelength_m: f32) -> u32 {
    let cell = metrics.dx().max(metrics.dz()).max(1e-3);
    ((wavelength_m / cell) * 0.5).round().clamp(1.0, 64.0) as u32
}

fn box_blur_mask(src: &MaskField, radius: u32) -> MaskField {
    let m = src.metrics;
    let r = radius.max(1) as i32;
    let mut tmp = MaskField::zeros(m);
    let mut out = MaskField::zeros(m);
    let w = m.width as i32;
    let h = m.height as i32;
    // Separable box blur.
    for j in 0..h {
        for i in 0..w {
            let mut sum = 0.0;
            let mut n = 0.0;
            for di in -r..=r {
                let x = (i + di).clamp(0, w - 1) as u32;
                sum += src.get(x, j as u32);
                n += 1.0;
            }
            tmp.set(i as u32, j as u32, sum / n);
        }
    }
    for j in 0..h {
        for i in 0..w {
            let mut sum = 0.0;
            let mut n = 0.0;
            for dj in -r..=r {
                let y = (j + dj).clamp(0, h - 1) as u32;
                sum += tmp.get(i as u32, y);
                n += 1.0;
            }
            out.set(i as u32, j as u32, sum / n);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geomorph::{noisy_mountain, single_valley};

    fn metrics() -> HeightfieldMetrics {
        HeightfieldMetrics::new(64, 64, 2000.0, 2000.0)
    }

    fn mask_any_above(m: &MaskField, t: f32) -> bool {
        for j in 0..m.metrics.height {
            for i in 0..m.metrics.width {
                if m.get(i, j) > t {
                    return true;
                }
            }
        }
        false
    }

    #[test]
    fn bands_are_scale_aware_and_ordered() {
        let small = AmplificationBands::from_world(HeightfieldMetrics::new(64, 64, 500.0, 500.0));
        let large = AmplificationBands::from_world(HeightfieldMetrics::new(64, 64, 8000.0, 8000.0));
        assert!(small.macro_m.0 < large.macro_m.0);
        assert!(small.meso_m.1 <= small.macro_m.0 * 1.01);
        assert!(small.micro_m.1 <= small.meso_m.0 * 1.01);
        assert!(small.micro_m.0 < small.micro_m.1);
    }

    #[test]
    fn amplify_changes_height_and_is_deterministic() {
        let hf = noisy_mountain(metrics());
        let p = TerrainAmplificationParams::default();
        let a = amplify_terrain(&hf, &p, None, None, None);
        let b = amplify_terrain(&hf, &p, None, None, None);
        assert_eq!(a.height.to_dense(), b.height.to_dense());
        let mut changed = 0usize;
        for (x, y) in a.height.to_dense().iter().zip(hf.to_dense().iter()) {
            if (*x - *y).abs() > 1e-5 {
                changed += 1;
            }
        }
        assert!(
            changed > 64,
            "expected structured detail, changed={changed}"
        );
        assert!(mask_any_above(&a.fine_flow, 0.01), "fine flow organisation");
        assert!(
            mask_any_above(&a.micro_channel, 0.01)
                || mask_any_above(&a.ridge_breakup, 0.01)
                || mask_any_above(&a.fine_erosion, 0.01),
            "expected at least one conditioned detail map"
        );
        assert!(mask_any_above(&a.detail_mask, 0.01));
    }

    #[test]
    fn silhouette_lock_limits_low_frequency_rewrite() {
        let hf = noisy_mountain(metrics());
        let mut unlocked = TerrainAmplificationParams::default();
        unlocked.silhouette_lock = 0.0;
        unlocked.meso_amplitude_m = 12.0;
        unlocked.micro_amplitude_m = 3.0;
        let mut locked = unlocked.clone();
        locked.silhouette_lock = 1.0;

        let u = amplify_terrain(&hf, &unlocked, None, None, None);
        let l = amplify_terrain(&hf, &locked, None, None, None);
        let bands = AmplificationBands::from_world(hf.metrics);
        let r = wavelength_to_radius_texels(hf.metrics, bands.silhouette_wavelength_m());

        let mut du = MaskField::zeros(hf.metrics);
        let mut dl = MaskField::zeros(hf.metrics);
        for j in 0..hf.metrics.height {
            for i in 0..hf.metrics.width {
                du.set(i, j, u.height.get(i, j) - hf.get(i, j));
                dl.set(i, j, l.height.get(i, j) - hf.get(i, j));
            }
        }
        let lu = box_blur_mask(&du, r);
        let ll = box_blur_mask(&dl, r);
        let mut sum_u = 0.0f32;
        let mut sum_l = 0.0f32;
        for j in 0..hf.metrics.height {
            for i in 0..hf.metrics.width {
                sum_u += lu.get(i, j).abs();
                sum_l += ll.get(i, j).abs();
            }
        }
        assert!(
            sum_l < sum_u * 0.85,
            "silhouette lock should reduce low-frequency rewrite ({sum_l} vs {sum_u})"
        );
    }

    #[test]
    fn protection_mask_blocks_detail() {
        let hf = single_valley(metrics());
        let mut prot = MaskField::zeros(hf.metrics);
        for j in 0..hf.metrics.height {
            for i in 0..hf.metrics.width / 2 {
                prot.set(i, j, 1.0);
            }
        }
        let p = TerrainAmplificationParams {
            meso_amplitude_m: 10.0,
            micro_amplitude_m: 2.5,
            ..Default::default()
        };
        let out = amplify_terrain(&hf, &p, None, None, Some(&prot));
        let mut left = 0.0f32;
        let mut right = 0.0f32;
        for j in 0..hf.metrics.height {
            for i in 0..hf.metrics.width {
                let d = (out.height.get(i, j) - hf.get(i, j)).abs();
                if i < hf.metrics.width / 2 {
                    left += d;
                } else {
                    right += d;
                }
            }
        }
        assert!(right > left * 4.0, "protected half should stay near input");
    }
}
