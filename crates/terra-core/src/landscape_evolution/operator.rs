//! Reusable [`LandscapeEvolutionOperator`] — Phase 3 physically-based evolution.

use crate::analyze;
use crate::fields::keys;
use crate::heightfield::Heightfield;
use crate::hydro;
use crate::layer::ThermalErosionParams;
use crate::mask::{MaskField, MaskSource};
use std::collections::HashMap;

use super::analytical::evolve_analytical;
use super::iterative::evolve_iterative;
use super::params::{BoundaryMode, EvolutionSolverMode, LandscapeEvolutionParams};
use super::uplift::synthesise_uplift;
use super::BoundaryMasks;

/// Inputs consumed by the landscape evolution operator.
#[derive(Debug, Clone)]
pub struct LandscapeEvolutionInput<'a> {
    pub elevation: &'a Heightfield,
    pub painted_uplift: Option<&'a MaskField>,
    pub precipitation: Option<&'a MaskField>,
    pub erodibility: Option<&'a MaskField>,
    pub lithology_hardness: Option<&'a MaskField>,
    pub outlet_mask: Option<&'a MaskField>,
    pub protection: Option<&'a MaskField>,
}

/// Outputs produced by the operator.
#[derive(Debug, Clone)]
pub struct LandscapeEvolutionOutput {
    pub elevation: Heightfield,
    /// Smooth tectonic / uplift structure retained separately from the eroded surface.
    pub tectonic_base: Heightfield,
    pub flow_direction: MaskField,
    pub flow_accumulation: MaskField,
    pub stream_order: MaskField,
    pub erosion_rate: MaskField,
    pub incision: MaskField,
    pub uplift_field: MaskField,
    pub deposition: Option<MaskField>,
    pub discharge: MaskField,
}

impl LandscapeEvolutionOutput {
    pub fn into_fields(self) -> (Heightfield, HashMap<&'static str, MaskField>) {
        let mut fields = HashMap::new();
        fields.insert(keys::FLOW_DIRECTION, self.flow_direction);
        fields.insert(keys::FLOW_ACCUMULATION, self.flow_accumulation);
        fields.insert(keys::STREAM_ORDER, self.stream_order);
        fields.insert(keys::SPE_INCISION, self.incision.clone());
        fields.insert(keys::EROSION, self.erosion_rate);
        fields.insert(keys::UPLIFT_RATE, self.uplift_field);
        fields.insert(keys::TECTONIC_BASE, self.tectonic_base.into_mask());
        fields.insert(keys::WATER_DISCHARGE, self.discharge);
        if let Some(d) = self.deposition {
            fields.insert(keys::DEPOSITION, d.clone());
            fields.insert(keys::SEDIMENT_THICKNESS, d);
        }
        (self.elevation, fields)
    }
}

/// Physically-based large-scale landscape evolution (Cordonnier / Tzathas SPE).
///
/// Not droplet hydraulic erosion — fluvial incision follows the stream-power law
/// coupled with tectonic uplift.
#[derive(Debug, Clone)]
pub struct LandscapeEvolutionOperator {
    pub params: LandscapeEvolutionParams,
}

impl LandscapeEvolutionOperator {
    pub fn new(params: LandscapeEvolutionParams) -> Self {
        Self { params }
    }

    pub fn evaluate(&self, input: LandscapeEvolutionInput<'_>) -> LandscapeEvolutionOutput {
        let p = &self.params;
        let m = input.elevation.metrics;

        // ── Hardness / erodibility ───────────────────────────────────────
        let mut hardness = input
            .lithology_hardness
            .cloned()
            .unwrap_or_else(|| MaskField::zeros(m));
        if let Some(ero) = input.erodibility {
            for j in 0..m.height {
                for i in 0..m.width {
                    let soft = ero.get(i, j).clamp(0.0, 1.0);
                    let h = (hardness.get(i, j) * (1.0 - soft * 0.85)).clamp(0.0, 1.0);
                    hardness.set(i, j, h);
                }
            }
        }

        // ── Uplift field (tectonic) ──────────────────────────────────────
        let boundaries = build_boundary_masks(input.elevation, p, input.outlet_mask);
        let mut uplift = synthesise_uplift(m, p, input.painted_uplift);
        zero_locked_mask_values(&mut uplift, &boundaries.elevation_locks);
        // Rainfall field modulates local uplift-driven discharge later via params.rainfall;
        // optional precip map scales rainfall spatially into a discharge weight.
        let rain_weight = input
            .precipitation
            .cloned()
            .unwrap_or_else(|| MaskField::filled(m, p.rainfall.max(0.0)));

        // ── Boundary / outlet mask ───────────────────────────────────────
        // ── Tectonic base structure (preserved large forms) ──────────────
        // Seed tectonic base from input + a short uplift impulse so large forms
        // exist before fluvial organisation — erosion cannot freely destroy this.
        let mut tectonic = input.elevation.clone();
        let seed_time = (p.evolution_time() * 0.15).max(p.dt);
        for j in 0..m.height {
            for i in 0..m.width {
                if boundaries.elevation_locks.get(i, j) > 0.5 {
                    continue;
                }
                let u = uplift.get(i, j) * seed_time;
                tectonic.set(i, j, tectonic.get(i, j) + u);
            }
        }

        // Initial surface for evolution: blend input toward tectonic seed by age.
        // Low age → mostly input + mild uplift; high age → more tectonic structure.
        let age = p.geological_age.clamp(0.0, 1.0);
        let mut working = input.elevation.clone();
        let seed_blend = (0.2 + 0.55 * age) * p.uplift.clamp(0.0, 2.0).min(1.0);
        for j in 0..m.height {
            for i in 0..m.width {
                if boundaries.elevation_locks.get(i, j) > 0.5 {
                    continue;
                }
                let a = seed_blend.clamp(0.0, 1.0);
                working.set(i, j, working.get(i, j) * (1.0 - a) + tectonic.get(i, j) * a);
            }
        }

        // Apply spatial rainfall into a temporary hardness-softness boost for wet cells.
        // (Keeps precip as a first-class input without changing SPE units mid-solve.)
        if input.precipitation.is_some() {
            let mean = mean_mask(&rain_weight).max(1e-4);
            for j in 0..m.height {
                for i in 0..m.width {
                    let w = (rain_weight.get(i, j) / mean).clamp(0.25, 2.5);
                    // Wetter → slightly softer effective rock for incision.
                    let h = hardness.get(i, j) / w.sqrt();
                    hardness.set(i, j, h.clamp(0.0, 1.0));
                }
            }
        }

        // Artist resistance is a global hardness floor and is applied after
        // all spatial softening. The union form leaves lithology unchanged at
        // zero and makes resistance=1 an exact no-incision limit.
        let global_resistance = p.terrain_resistance.clamp(0.0, 1.0);
        if global_resistance > 0.0 {
            for j in 0..m.height {
                for i in 0..m.width {
                    let local = hardness.get(i, j).clamp(0.0, 1.0);
                    let combined = 1.0 - (1.0 - local) * (1.0 - global_resistance);
                    hardness.set(i, j, combined.clamp(0.0, 1.0));
                }
            }
        }

        // Light Dendry-inspired valley prime so Fast analytical has immature
        // drainage seeds (Cordonnier networks need an initial flow direction).
        let prime = 0.08 + 0.12 * (1.0 - age);
        if prime > 1e-4 {
            working = crate::hydro::ridge_distance_valley_seed_with_locks(
                &working,
                prime,
                &boundaries.elevation_locks,
            );
            restore_locked_height(&mut working, input.elevation, &boundaries.elevation_locks);
        }

        let (mut evolved, cache, incision_raw, erosion_raw) = match p.solver {
            EvolutionSolverMode::Fast => {
                let passes = p.fast_passes();
                let (z, cache, incision) = evolve_analytical(
                    &working,
                    &tectonic,
                    &uplift,
                    &hardness,
                    &boundaries,
                    p,
                    passes,
                );
                let erosion = incision
                    .iter()
                    .map(|v| v / p.evolution_time().max(1.0))
                    .collect::<Vec<_>>();
                (z, cache, incision, erosion)
            }
            EvolutionSolverMode::Accurate => {
                let steps = p.accurate_steps();
                let (z, cache, incision, erosion) =
                    evolve_iterative(&working, &uplift, &hardness, &boundaries, p, steps);
                (z, cache, incision, erosion)
            }
        };

        // Hillslope / thermal companion (ridge singularity regularisation).
        if p.hillslope_diffusion > 1e-4 {
            let thermal = ThermalErosionParams {
                talus_angle_deg: p.talus_angle_deg,
                iterations: ((p.accurate_steps() / 4).max(1)).min(24),
                strength: p.hillslope_diffusion.clamp(0.0, 1.0),
                hardness: 0.0,
                hardness_source: MaskSource::None,
                layered_materials: false,
                ..ThermalErosionParams::default()
            };
            let hillslope_hardness = hardness_with_locks(&hardness, &boundaries.elevation_locks);
            let (hillslope, _e, deposit) =
                analyze::thermal_erode_with_hardness(&evolved, &thermal, &hillslope_hardness);
            let transport = p.sediment_transport.clamp(0.0, 1.0);
            evolved = hillslope;
            restore_locked_height(&mut evolved, input.elevation, &boundaries.elevation_locks);
            let deposition = if transport > 1e-4 || p.deposition_coeff > 1e-6 {
                let data: Vec<f32> = deposit
                    .data()
                    .iter()
                    .map(|d| d * transport.max(p.deposition_coeff))
                    .collect();
                Some(MaskField::from_raw(m, &data))
            } else {
                None
            };

            // Preserve large tectonic forms: blend protected / locked cells toward tectonic base.
            apply_tectonic_preservation(
                &mut evolved,
                &tectonic,
                input.protection,
                &boundaries.elevation_locks,
                p.constraint_preservation,
            );

            return finalize(
                evolved,
                tectonic,
                cache,
                &incision_raw,
                &erosion_raw,
                uplift,
                deposition,
                m,
                p,
                input.elevation,
                &boundaries.elevation_locks,
            );
        }

        apply_tectonic_preservation(
            &mut evolved,
            &tectonic,
            input.protection,
            &boundaries.elevation_locks,
            p.constraint_preservation,
        );

        finalize(
            evolved,
            tectonic,
            cache,
            &incision_raw,
            &erosion_raw,
            uplift,
            None,
            m,
            p,
            input.elevation,
            &boundaries.elevation_locks,
        )
    }
}

fn finalize(
    mut evolved: Heightfield,
    mut tectonic: Heightfield,
    cache: super::cache::DrainageCache,
    incision_raw: &[f32],
    erosion_raw: &[f32],
    uplift: MaskField,
    mut deposition: Option<MaskField>,
    m: crate::heightfield::HeightfieldMetrics,
    p: &LandscapeEvolutionParams,
    original: &Heightfield,
    elevation_locks: &MaskField,
) -> LandscapeEvolutionOutput {
    restore_locked_height(&mut evolved, original, elevation_locks);
    restore_locked_height(&mut tectonic, original, elevation_locks);
    if let Some(field) = deposition.as_mut() {
        zero_locked_mask_values(field, elevation_locks);
    }
    let w = m.width as usize;
    let h = m.height as usize;
    let max_acc = cache
        .accumulation
        .iter()
        .copied()
        .fold(0.0f32, f32::max)
        .max(1.0);
    let mut acc_mask = MaskField::zeros(m);
    let mut discharge = MaskField::zeros(m);
    let rain = p.rainfall.max(0.05);
    for j in 0..h {
        for i in 0..w {
            let a = cache.accumulation[j * w + i];
            acc_mask.set(i as u32, j as u32, a / max_acc);
            discharge.set(i as u32, j as u32, a * rain);
        }
    }
    let order = hydro::stream_order_log2(
        &cache.accumulation,
        w,
        h,
        20.0 + 60.0 * (1.0 - p.drainage_scale.clamp(0.0, 1.0)),
    );
    let max_order = order.iter().copied().max().unwrap_or(1).max(1) as f32;
    let mut order_mask = MaskField::zeros(m);
    let mut incision = MaskField::zeros(m);
    let mut erosion_rate = MaskField::zeros(m);
    for j in 0..h {
        for i in 0..w {
            let idx = j * w + i;
            order_mask.set(i as u32, j as u32, order[idx] as f32 / max_order);
            if elevation_locks.get(i as u32, j as u32) <= 0.5 {
                incision.set(i as u32, j as u32, incision_raw[idx]);
                erosion_rate.set(i as u32, j as u32, erosion_raw[idx]);
            }
        }
    }

    LandscapeEvolutionOutput {
        elevation: evolved,
        tectonic_base: tectonic,
        flow_direction: cache.direction_mask,
        flow_accumulation: acc_mask,
        stream_order: order_mask,
        erosion_rate,
        incision,
        uplift_field: uplift,
        deposition,
        discharge,
    }
}

fn apply_tectonic_preservation(
    evolved: &mut Heightfield,
    tectonic: &Heightfield,
    protection: Option<&MaskField>,
    elevation_locks: &MaskField,
    preservation: f32,
) {
    let m = evolved.metrics;
    let base_lock = (preservation.clamp(0.0, 1.0) * 0.35).clamp(0.0, 1.0);
    for j in 0..m.height {
        for i in 0..m.width {
            if elevation_locks.get(i, j) > 0.5 {
                continue;
            }
            let painted = protection
                .map(|f| f.get(i, j))
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);
            let a = (base_lock + painted * preservation.clamp(0.0, 1.0)).clamp(0.0, 1.0);
            if a <= 1e-5 {
                continue;
            }
            // Only pull toward tectonic base where erosion over-cut large forms.
            let e = evolved.get(i, j);
            let t = tectonic.get(i, j);
            if e < t {
                evolved.set(i, j, e + (t - e) * a);
            }
        }
    }
}

fn build_boundary_masks(
    elevation: &Heightfield,
    p: &LandscapeEvolutionParams,
    outlet_mask: Option<&MaskField>,
) -> BoundaryMasks {
    let m = elevation.metrics;
    let mut routing_outlets = MaskField::zeros(m);
    let mut elevation_locks = MaskField::zeros(m);
    match p.boundary {
        BoundaryMode::OutletMask => {
            if let Some(mask) = outlet_mask {
                copy_thresholded_mask(mask, &mut routing_outlets);
                copy_thresholded_mask(mask, &mut elevation_locks);
            } else {
                mark_sea_level(elevation, &mut elevation_locks, p.base_level);
                union_mask(&elevation_locks, &mut routing_outlets);
                mark_border(&mut routing_outlets, 1);
            }
        }
        BoundaryMode::SeaLevel => {
            mark_sea_level(elevation, &mut elevation_locks, p.base_level);
            union_mask(&elevation_locks, &mut routing_outlets);
            mark_border(&mut routing_outlets, 1);
        }
        BoundaryMode::Fixed => {
            mark_border(&mut routing_outlets, 1);
            mark_border(&mut elevation_locks, 1);
        }
        BoundaryMode::OpenDrainage => {
            mark_border(&mut routing_outlets, 1);
        }
    }

    // A missing routing outlet gets an explicit open-rim fallback. It never
    // silently becomes an elevation lock.
    if !has_marked_cells(&routing_outlets) {
        mark_border(&mut routing_outlets, 1);
    }

    BoundaryMasks {
        routing_outlets,
        elevation_locks,
    }
}

fn copy_thresholded_mask(source: &MaskField, target: &mut MaskField) {
    let m = target.metrics;
    for j in 0..m.height {
        for i in 0..m.width {
            if source.get(i, j) > 0.5 {
                target.set(i, j, 1.0);
            }
        }
    }
}

fn union_mask(source: &MaskField, target: &mut MaskField) {
    copy_thresholded_mask(source, target);
}

fn has_marked_cells(mask: &MaskField) -> bool {
    mask.data().iter().any(|&v| v > 0.5)
}

fn mark_sea_level(elevation: &Heightfield, boundary: &mut MaskField, base: f32) {
    let m = elevation.metrics;
    for j in 0..m.height {
        for i in 0..m.width {
            if elevation.get(i, j) <= base + 0.25 {
                boundary.set(i, j, 1.0);
            }
        }
    }
}

fn mark_border(boundary: &mut MaskField, margin: u32) {
    let m = boundary.metrics;
    for j in 0..m.height {
        for i in 0..m.width {
            if i < margin || j < margin || i + margin >= m.width || j + margin >= m.height {
                boundary.set(i, j, 1.0);
            }
        }
    }
}

fn restore_locked_height(
    target: &mut Heightfield,
    original: &Heightfield,
    elevation_locks: &MaskField,
) {
    let m = target.metrics;
    for j in 0..m.height {
        for i in 0..m.width {
            if elevation_locks.get(i, j) > 0.5 {
                target.set(i, j, original.get(i, j));
            }
        }
    }
}

fn zero_locked_mask_values(target: &mut MaskField, elevation_locks: &MaskField) {
    let m = target.metrics;
    for j in 0..m.height {
        for i in 0..m.width {
            if elevation_locks.get(i, j) > 0.5 {
                target.set(i, j, 0.0);
            }
        }
    }
}

fn hardness_with_locks(hardness: &MaskField, elevation_locks: &MaskField) -> MaskField {
    let mut out = hardness.clone();
    let m = out.metrics;
    for j in 0..m.height {
        for i in 0..m.width {
            if elevation_locks.get(i, j) > 0.5 {
                out.set(i, j, 1.0);
            }
        }
    }
    out
}

fn mean_mask(field: &MaskField) -> f32 {
    let d = field.data();
    if d.is_empty() {
        return 0.0;
    }
    d.iter().sum::<f32>() / d.len() as f32
}

trait HeightfieldMaskExt {
    fn into_mask(self) -> MaskField;
}

impl HeightfieldMaskExt for Heightfield {
    fn into_mask(self) -> MaskField {
        MaskField::from_raw(self.metrics, &self.to_dense())
    }
}

/// Run the operator and return authoring-compatible fields.
pub fn evaluate_landscape_evolution(
    input: &Heightfield,
    p: &LandscapeEvolutionParams,
    hardness: Option<&MaskField>,
    uplift: Option<&MaskField>,
    protection: Option<&MaskField>,
    precipitation: Option<&MaskField>,
    outlet_mask: Option<&MaskField>,
) -> (Heightfield, HashMap<&'static str, MaskField>) {
    let op = LandscapeEvolutionOperator::new(p.clone());
    let out = op.evaluate(LandscapeEvolutionInput {
        elevation: input,
        painted_uplift: uplift,
        precipitation,
        erodibility: None,
        lithology_hardness: hardness,
        outlet_mask,
        protection,
    });
    out.into_fields()
}
