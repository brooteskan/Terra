use super::{EvalContext, EvalError};
use crate::analyze;
use crate::authoring;
use crate::fields::{
    bake_hardness_from_materials_ex, bake_hardness_from_strata_ex, has_depth_aware_strata, keys,
    resolve_hardness,
};
use crate::generators;
use crate::heightfield::Heightfield;
use crate::hydro;
use crate::layer::*;
use crate::mask::{MaskField, MaskSource};
use crate::surface;
use crate::volumetric;

/// Evaluates built-in layer kinds by matching on [`LayerKind`].
///
/// There is no processor registration and no pluggable seam: every built-in
/// kind is dispatched by the single `match` in [`ProcessorRegistry::evaluate`],
/// which keeps the set closed and serde-friendly. If a pluggable seam is ever
/// wanted, it should arrive together with its first implementor rather than as
/// an empty abstraction.
pub struct ProcessorRegistry {
    // No state: built-in kinds are dispatched by the match in `evaluate`.
}

impl ProcessorRegistry {
    pub fn builtin() -> Self {
        Self {}
    }

    /// Evaluate a single layer by matching on its [`LayerKind`].
    ///
    /// This `match` is the real dispatch mechanism — the one place every
    /// built-in kind becomes a height contribution. Adding a kind means adding
    /// an arm here.
    pub fn evaluate(
        &self,
        ctx: &mut EvalContext,
        input: &Heightfield,
        layer: &Layer,
    ) -> Result<Heightfield, EvalError> {
        match &layer.kind {
            LayerKind::SculptBase(p) => Ok(generators::sculpt_base(ctx.metrics, p)),
            LayerKind::SculptStrokes(p) => {
                let result = authoring::apply_sculpt_strokes(input, p);
                Ok(publish_authoring_merge(
                    ctx,
                    result,
                    &[
                        keys::SCULPT_PROTECTION,
                        keys::UPLIFT_RATE,
                        keys::HARDNESS,
                        keys::SEDIMENT_THICKNESS,
                        keys::EDIT_REGION,
                    ],
                ))
            }
            LayerKind::TerrainConstraints(p) => {
                let result = authoring::apply_constraints(input, p);
                Ok(publish_authoring_merge(
                    ctx,
                    result,
                    &[
                        keys::SCULPT_PROTECTION,
                        keys::UPLIFT_RATE,
                        keys::EDIT_REGION,
                        keys::CONSTRAINT_WEIGHT,
                    ],
                ))
            }
            LayerKind::GradientReconstruct(p) => {
                let target = ctx.aux_maps.get(keys::CONSTRAINT_TARGET).cloned();
                let weight = ctx.aux_maps.get(keys::CONSTRAINT_WEIGHT).cloned();
                let mut authored = p.clone();
                authored.iterations = match ctx.quality {
                    crate::eval::PreviewQuality::Draft => authored.iterations.min(18),
                    crate::eval::PreviewQuality::Medium => authored.iterations.min(42),
                    _ => authored.iterations,
                }
                .max(1);
                Ok(publish_authoring(
                    ctx,
                    authoring::gradient_reconstruct(
                        input,
                        &authored,
                        target.as_ref(),
                        weight.as_ref(),
                    ),
                ))
            }
            LayerKind::LandscapeEvolution(p) => {
                let hardness = ctx.aux_maps.hardness.clone();
                let uplift = ctx.aux_maps.get(keys::UPLIFT_RATE).cloned();
                let protection = ctx.aux_maps.get(keys::SCULPT_PROTECTION).cloned();
                let precip = ctx.aux_maps.rainfall.clone();
                let mut authored = p.clone();
                match ctx.quality {
                    crate::eval::PreviewQuality::Draft => {
                        authored.iterations = authored.iterations.min(6);
                        authored.fixed_point_iters = authored.fixed_point_iters.min(3);
                        authored.solver = crate::landscape_evolution::EvolutionSolverMode::Fast;
                    }
                    crate::eval::PreviewQuality::Medium => {
                        authored.iterations = authored.iterations.min(16);
                        authored.fixed_point_iters = authored.fixed_point_iters.min(5);
                    }
                    _ => {}
                }
                authored.iterations = authored.iterations.max(1);
                authored.fixed_point_iters = authored.fixed_point_iters.max(1);
                let (height, fields) = crate::landscape_evolution::evaluate_landscape_evolution(
                    input,
                    &authored,
                    hardness.as_ref(),
                    uplift.as_ref(),
                    protection.as_ref(),
                    precip.as_ref(),
                    None,
                );
                Ok(publish_authoring(
                    ctx,
                    authoring::AuthoringResult { height, fields },
                ))
            }
            LayerKind::HydrologyRepair(p) => {
                let edit = ctx.aux_maps.get(keys::EDIT_REGION).cloned();
                let hardness = ctx.aux_maps.hardness.clone();
                let protection = ctx.aux_maps.get(keys::SCULPT_PROTECTION).cloned();
                let mut authored = p.clone();
                authored.iterations = match ctx.quality {
                    crate::eval::PreviewQuality::Draft => authored.iterations.min(3),
                    crate::eval::PreviewQuality::Medium => authored.iterations.min(6),
                    _ => authored.iterations,
                }
                .max(1);
                Ok(publish_authoring(
                    ctx,
                    authoring::repair_hydrology(
                        input,
                        &authored,
                        edit.as_ref(),
                        hardness.as_ref(),
                        protection.as_ref(),
                    ),
                ))
            }
            LayerKind::GeomorphicDetail(p) => {
                let flow = ctx.aux_maps.flow_accumulation.clone();
                let protection = ctx.aux_maps.get(keys::SCULPT_PROTECTION).cloned();
                let hardness = ctx.aux_maps.hardness.clone();
                Ok(publish_authoring(
                    ctx,
                    authoring::geomorphic_detail_with_hardness(
                        input,
                        p,
                        flow.as_ref(),
                        hardness.as_ref(),
                        protection.as_ref(),
                    ),
                ))
            }
            LayerKind::EcosystemFeedback(p) => {
                let vegetation = ctx.aux_maps.vegetation.clone();
                let moisture = ctx
                    .aux_maps
                    .soil_moisture
                    .clone()
                    .or_else(|| ctx.aux_maps.wetness.clone());
                let hardness = ctx.aux_maps.hardness.clone();
                let sediment = ctx
                    .aux_maps
                    .sediment_thickness
                    .clone()
                    .or_else(|| ctx.aux_maps.sediment.clone());
                Ok(publish_authoring(
                    ctx,
                    authoring::ecosystem_feedback(
                        input,
                        p,
                        vegetation.as_ref(),
                        moisture.as_ref(),
                        hardness.as_ref(),
                        sediment.as_ref(),
                    ),
                ))
            }
            LayerKind::Flat(p) => Ok(generators::flat(ctx.metrics, p.height)),
            LayerKind::Ramp(p) => Ok(generators::ramp(ctx.metrics, p)),
            LayerKind::NoiseValue(p) => Ok(generators::noise_field(
                ctx.metrics,
                p,
                FractalNoiseType::Value,
            )),
            LayerKind::NoisePerlin(p) => Ok(generators::noise_field(
                ctx.metrics,
                p,
                FractalNoiseType::Perlin,
            )),
            LayerKind::NoiseOpenSimplex(p) => Ok(generators::noise_field(
                ctx.metrics,
                p,
                FractalNoiseType::OpenSimplex,
            )),
            LayerKind::NoiseWorley(p) => Ok(generators::worley_field(ctx.metrics, p)),
            LayerKind::Fbm(p) => Ok(generators::fbm_field(ctx.metrics, p)),
            LayerKind::Ridged(p) => Ok(generators::ridged_field(ctx.metrics, p)),
            LayerKind::DomainWarp(p) => Ok(generators::domain_warp_field(ctx.metrics, p)),
            LayerKind::Terrace(p) => Ok(generators::terrace(input, p)),
            LayerKind::Plateau(p) => Ok(generators::plateau(input, p)),
            LayerKind::Mesa(p) => Ok(generators::mesa(ctx.metrics, p)),
            LayerKind::Island(p) => {
                let result = generators::island(ctx.metrics, p);
                ctx.aux_insert(keys::LAND_MASK, result.land);
                ctx.aux_insert(keys::SHORE_DISTANCE, result.shore_distance);
                ctx.aux_insert(keys::BATHYMETRY, result.bathymetry);
                ctx.aux_insert(keys::SHELF, result.shelf);
                ctx.aux_insert(keys::BEACH, result.beach);
                ctx.aux_insert(keys::REEF, result.reef);
                ctx.aux_insert(keys::MOUNTAIN_MASK, result.mountain);
                Ok(result.height)
            }
            LayerKind::Mountains(p) => Ok(generators::mountains(ctx.metrics, p)),
            LayerKind::Volcano(p) => Ok(generators::volcano(ctx.metrics, p)),
            LayerKind::Uplift(p) => Ok(generators::uplift(ctx.metrics, p)),
            LayerKind::Dunes(p) => {
                let (height, aeolian) = generators::dunes_with_aux(ctx.metrics, p);
                publish_aeolian_aux(ctx, &aeolian);
                Ok(height)
            }
            LayerKind::Canyons(p) => Ok(generators::canyons(ctx.metrics, p)),
            LayerKind::VoronoiRegions(p) => Ok(generators::voronoi_regions(ctx.metrics, p)),
            LayerKind::ImportHeightmap(p) => generators::import_heightmap(ctx.metrics, p),
            LayerKind::ProceduralShape(p) => Ok(generators::procedural_shape(ctx.metrics, p)),
            LayerKind::Stamp2d(p) => generators::import_heightmap(ctx.metrics, &p.heightmap),
            LayerKind::Stamp3d(p) => generators::stamp_3d(ctx.metrics, p),
            LayerKind::PolygonHeight(p) => Ok(generators::polygon_height(input, p)),
            LayerKind::ThermalErosion(p) => {
                let base = match ctx.quality {
                    crate::eval::PreviewQuality::Draft => {
                        analyze::draft_sim_levels(ctx.metrics.width)
                    }
                    _ => analyze::default_sim_levels(ctx.metrics.width),
                };
                let levels = ctx.level_steps.schedule_for_filter(
                    base,
                    p.level_count,
                    p.start_level,
                    p.level_step_strength,
                    &p.level_step_curve,
                    ctx.quality,
                );
                if use_strata_hardness(ctx, &p.hardness_source) {
                    let reference = ctx.aux_maps.strata_reference.clone().unwrap();
                    let strata = ctx.aux_maps.strata.clone().unwrap();
                    let geom = ctx.aux_maps.bed_geometry;
                    let default_k = p.hardness;
                    let (hf, erosion, deposit) = analyze::thermal_erode_with_strata_ex(
                        input, p, &reference, &strata, default_k, &geom,
                    );
                    let hardness =
                        bake_hardness_from_strata_ex(&reference, &hf, &strata, default_k, &geom);
                    ctx.aux_insert(keys::HARDNESS, hardness);
                    let mats = surface::material_weights_at_depth(&reference, &hf, &strata);
                    ctx.aux_insert(keys::MATERIALS, mats);
                    ctx.aux_insert(keys::EROSION, erosion);
                    ctx.aux_insert(keys::DEPOSITION, deposit.clone());
                    ctx.aux_insert(keys::DEBRIS_DEPTH, deposit);
                    ctx.ensure_derived_fields(&hf);
                    return Ok(hf);
                }
                let hardness = bake_layer_hardness(ctx, input, p.hardness, &p.hardness_source);
                ctx.aux_insert(keys::HARDNESS, hardness.clone());
                if p.layered_materials {
                    let initial = analyze::MassWastingState::with_optional_layers(
                        input,
                        ctx.aux_maps.bedrock_height.as_ref(),
                        ctx.aux_maps.get(keys::DEBRIS_DEPTH),
                        ctx.aux_maps.sediment_thickness.as_ref(),
                        0.0,
                        0.0,
                    );
                    let result =
                        analyze::thermal_erode_layered(input, p, &hardness, Some(&initial));
                    ctx.aux_insert(keys::EROSION, result.erosion);
                    ctx.aux_insert(keys::DEPOSITION, result.deposition.clone());
                    ctx.aux_insert(keys::DEBRIS_DEPTH, result.loose_debris.clone());
                    ctx.aux_insert(keys::BEDROCK_HEIGHT, result.bedrock);
                    ctx.aux_insert(keys::SEDIMENT_THICKNESS, result.sediment);
                    ctx.aux_insert(keys::TALUS_STABILITY, result.talus_stability);
                    ctx.aux_insert(keys::INSTABILITY, result.instability);
                    ctx.ensure_derived_fields(&result.height);
                    return Ok(result.height);
                }
                let (hf, erosion, deposit) = if hardness_is_uniform(&hardness, p.hardness) {
                    analyze::thermal_erode_leveled(input, p, &levels)
                } else {
                    analyze::thermal_erode_leveled_with_hardness(input, p, &hardness, &levels)
                };
                ctx.aux_insert(keys::EROSION, erosion);
                ctx.aux_insert(keys::DEPOSITION, deposit.clone());
                ctx.aux_insert(keys::DEBRIS_DEPTH, deposit);
                ctx.ensure_derived_fields(&hf);
                Ok(hf)
            }
            LayerKind::DebrisFlow(p) => {
                let hardness = bake_layer_hardness(ctx, input, p.hardness, &p.hardness_source);
                ctx.aux_insert(keys::HARDNESS, hardness.clone());
                let initial = analyze::MassWastingState::with_optional_layers(
                    input,
                    ctx.aux_maps.bedrock_height.as_ref(),
                    ctx.aux_maps.get(keys::DEBRIS_DEPTH),
                    ctx.aux_maps.sediment_thickness.as_ref(),
                    p.initial_debris_thickness,
                    p.initial_sediment_thickness,
                );
                let result = analyze::debris_flow_erode(input, p, &hardness, Some(&initial));
                ctx.aux_insert(keys::EROSION, result.erosion);
                ctx.aux_insert(keys::DEPOSITION, result.deposition);
                ctx.aux_insert(keys::BEDROCK_HEIGHT, result.bedrock);
                ctx.aux_insert(keys::DEBRIS_DEPTH, result.debris);
                ctx.aux_insert(keys::SEDIMENT_THICKNESS, result.sediment);
                ctx.aux_insert(keys::SLIDE_PATH, result.slide_path);
                ctx.aux_insert(keys::INSTABILITY, result.instability);
                ctx.aux_insert(keys::FLOW_ACCUMULATION, result.flow_accumulation);
                ctx.ensure_derived_fields(&result.height);
                Ok(result.height)
            }
            LayerKind::HydraulicErosion(p) => {
                let base = match ctx.quality {
                    crate::eval::PreviewQuality::Draft => {
                        analyze::draft_sim_levels(ctx.metrics.width)
                    }
                    _ => analyze::default_sim_levels(ctx.metrics.width),
                };
                let levels = ctx.level_steps.schedule_for_filter(
                    base,
                    p.level_count,
                    p.start_level,
                    p.level_step_strength,
                    &p.level_step_curve,
                    ctx.quality,
                );
                let p = analyze::apply_transport_model(p, p.transport_model);
                if use_strata_hardness(ctx, &p.hardness_source) {
                    let reference = ctx.aux_maps.strata_reference.clone().unwrap();
                    let strata = ctx.aux_maps.strata.clone().unwrap();
                    let geom = ctx.aux_maps.bed_geometry;
                    let default_k = p.hardness;
                    let mut result = analyze::hydraulic_erode_with_strata_ex(
                        input, &p, &reference, &strata, default_k, &geom,
                    );
                    // Use the currently exposed stratum as the droplet resistance,
                    // then rebake after fine erosion reveals any deeper material.
                    let particle_hardness = bake_hardness_from_strata_ex(
                        &reference,
                        &result.height,
                        &strata,
                        default_k,
                        &geom,
                    );
                    result = analyze::apply_particle_erosion(result, &p, &particle_hardness);
                    let mut hardness = bake_hardness_from_strata_ex(
                        &reference,
                        &result.height,
                        &strata,
                        default_k,
                        &geom,
                    );
                    if p.sediment_softness > 0.0 {
                        hardness = analyze::apply_sediment_softness(
                            &hardness,
                            &result.deposition_raw,
                            p.sediment_softness,
                        );
                    }
                    ctx.aux_insert(keys::HARDNESS, hardness);
                    let mut mats =
                        surface::material_weights_at_depth(&reference, &result.height, &strata);
                    if p.sediment_softness > 0.0 {
                        mats = analyze::apply_sediment_material_ids(
                            &mats,
                            &result.deposition_raw,
                            p.sediment_softness,
                        );
                    }
                    ctx.aux_insert(keys::MATERIALS, mats);
                    ctx.aux_insert(keys::WETNESS, result.wetness);
                    ctx.aux_insert(keys::SEDIMENT, result.sediment);
                    ctx.aux_insert(keys::EROSION, result.erosion);
                    ctx.aux_insert(keys::DEPOSITION, result.deposition);
                    ctx.ensure_derived_fields(&result.height);
                    return Ok(result.height);
                }
                let hardness = bake_layer_hardness(ctx, input, p.hardness, &p.hardness_source);
                let rainfall = bake_optional_mask(ctx, input, &p.rainfall_source);
                let protection = bake_optional_mask(ctx, input, &p.protection_source);

                // Prefer HydraulicErosionCore for shared rain / ridge / layered / outputs.
                // Multilevel schedules still use the leveled oracle with the same biases.
                let (result, outputs) = if levels.len() <= 1 {
                    let core = analyze::HydraulicErosionCore::new(p.clone());
                    let (result, outputs, _diag, _state) = core.simulate(
                        input,
                        &hardness,
                        rainfall.as_ref(),
                        protection.as_ref(),
                        None,
                        None,
                    );
                    (result, Some(outputs))
                } else {
                    let result = analyze::hydraulic_erode_leveled_with_fields(
                        input,
                        &p,
                        &hardness,
                        rainfall.as_ref(),
                        protection.as_ref(),
                        &levels,
                    );
                    (result, None)
                };

                let mut hardness_out = hardness;
                if p.sediment_softness > 0.0 {
                    hardness_out = analyze::apply_sediment_softness(
                        &hardness_out,
                        &result.deposition_raw,
                        p.sediment_softness,
                    );
                    if let Some(mats) = ctx.aux_maps.materials.clone() {
                        let mats = analyze::apply_sediment_material_ids(
                            &mats,
                            &result.deposition_raw,
                            p.sediment_softness,
                        );
                        ctx.aux_insert(keys::MATERIALS, mats);
                    }
                }
                ctx.aux_insert(keys::HARDNESS, hardness_out);
                if p.output_erosion {
                    ctx.aux_insert(keys::EROSION, result.erosion.clone());
                }
                if p.output_deposition {
                    ctx.aux_insert(keys::DEPOSITION, result.deposition.clone());
                }
                if p.output_sediment {
                    ctx.aux_insert(keys::SEDIMENT, result.sediment.clone());
                }
                ctx.aux_insert(keys::WETNESS, result.wetness.clone());
                if let Some(outputs) = outputs {
                    if let Some(rain) = outputs.rainfall {
                        ctx.aux_insert(keys::RAINFALL, rain);
                    }
                    if p.output_water {
                        if let Some(water) = outputs.water {
                            ctx.aux_insert(keys::WATER_DEPTH, water);
                        }
                        if let Some(vel) = outputs.velocity {
                            ctx.aux_insert(keys::WATER_VELOCITY, vel);
                        }
                    }
                    if p.output_flow {
                        if let Some(flow) = outputs.flow {
                            ctx.aux_insert(keys::FLOW_ACCUMULATION, flow);
                        }
                    }
                    if p.output_channels {
                        if let Some(ch) = outputs.channels {
                            ctx.aux_insert(keys::CHANNEL_MASK, ch);
                        }
                    }
                    if p.output_sediment {
                        if let Some(bedrock) = outputs.bedrock {
                            ctx.aux_insert(keys::BEDROCK_HEIGHT, bedrock);
                        }
                        if let Some(loose) = outputs.loose_sediment {
                            ctx.aux_insert(keys::SEDIMENT_THICKNESS, loose);
                        }
                    }
                } else if p.output_water {
                    ctx.aux_insert(keys::WATER_DEPTH, result.water_raw.clone());
                }
                ctx.ensure_derived_fields(&result.height);
                Ok(result.height)
            }
            LayerKind::RiverCarve(p) => {
                let guide = bake_optional_mask(ctx, input, &p.guide);
                let (hf, flow, acc, wet) = hydro::carve_rivers(input, p, guide.as_ref());
                ctx.aux_insert(keys::FLOW_DIRECTION, flow);
                ctx.aux_insert(keys::FLOW_ACCUMULATION, acc);
                merge_wetness_max(ctx, wet);
                Ok(hf)
            }
            LayerKind::StreamPowerErosion(p) => {
                let mut p = p.clone();
                match ctx.quality {
                    crate::eval::PreviewQuality::Draft => {
                        p.iterations = p.iterations.min(8).max(1);
                        // Fewer Priority-Flood / D8 barriers between incision steps.
                        p.drainage_reuse_stride = p.drainage_reuse_stride.max(2);
                    }
                    crate::eval::PreviewQuality::Medium => {
                        p.iterations = p.iterations.min(16).max(1);
                        p.drainage_reuse_stride = p.drainage_reuse_stride.max(1);
                    }
                    _ => {
                        // Export oracle: refresh drainage every iteration.
                        p.drainage_reuse_stride = 1;
                    }
                }
                // SPE stays single-res; author level-step scales K / iters from the schedule.
                let base = match ctx.quality {
                    crate::eval::PreviewQuality::Draft => {
                        analyze::draft_sim_levels(ctx.metrics.width)
                    }
                    _ => analyze::default_sim_levels(ctx.metrics.width),
                };
                let levels = ctx.level_steps.schedule_for_filter(
                    base,
                    p.level_count,
                    p.start_level,
                    p.level_step_strength,
                    &p.level_step_curve,
                    ctx.quality,
                );
                if !levels.is_empty() {
                    let n = levels.len() as f32;
                    let iter_scale: f32 = levels.iter().map(|l| l.iter_scale).sum::<f32>() / n;
                    let effect: f32 = levels.iter().map(|l| l.effect_scale).sum::<f32>() / n;
                    p.iterations = ((p.iterations as f32 * iter_scale).round() as u32).max(1);
                    p.k *= effect;
                }
                if use_strata_hardness(ctx, &p.hardness_source) {
                    let reference = ctx.aux_maps.strata_reference.clone().unwrap();
                    let strata = ctx.aux_maps.strata.clone().unwrap();
                    let geom = ctx.aux_maps.bed_geometry;
                    let default_k = p.hardness;
                    let result = hydro::stream_power_erode_with_strata(
                        input, &p, &reference, &strata, default_k,
                    );
                    let hardness = bake_hardness_from_strata_ex(
                        &reference,
                        &result.height,
                        &strata,
                        default_k,
                        &geom,
                    );
                    ctx.aux_insert(keys::HARDNESS, hardness);
                    let mats =
                        surface::material_weights_at_depth(&reference, &result.height, &strata);
                    ctx.aux_insert(keys::MATERIALS, mats);
                    ctx.aux_insert(keys::FLOW_DIRECTION, result.flow_direction);
                    ctx.aux_insert(keys::FLOW_ACCUMULATION, result.flow_accumulation);
                    ctx.aux_insert(keys::STREAM_ORDER, result.stream_order.clone());
                    ctx.aux_insert(keys::SPE_INCISION, result.spe_incision.clone());
                    ctx.aux_insert(keys::EROSION, result.spe_incision);
                    ctx.ensure_derived_fields(&result.height);
                    return Ok(result.height);
                }
                let hardness = bake_layer_hardness(ctx, input, p.hardness, &p.hardness_source);
                ctx.aux_insert(keys::HARDNESS, hardness.clone());
                let result = hydro::stream_power_erode(input, &p, &hardness);
                ctx.aux_insert(keys::FLOW_DIRECTION, result.flow_direction);
                ctx.aux_insert(keys::FLOW_ACCUMULATION, result.flow_accumulation);
                ctx.aux_insert(keys::STREAM_ORDER, result.stream_order.clone());
                ctx.aux_insert(keys::SPE_INCISION, result.spe_incision.clone());
                // Reuse erosion overlay for SPE incision magnitude.
                ctx.aux_insert(keys::EROSION, result.spe_incision);
                ctx.ensure_derived_fields(&result.height);
                Ok(result.height)
            }
            LayerKind::MultiScaleAmplify(p) => {
                let mut p = p.clone();
                match ctx.quality {
                    crate::eval::PreviewQuality::Draft => {
                        p.thermal_iters = p.thermal_iters.min(6).max(1);
                        p.spe_iters = p.spe_iters.min(2);
                        p.level_count = if p.level_count == 0 {
                            2
                        } else {
                            p.level_count.min(2)
                        };
                    }
                    crate::eval::PreviewQuality::Medium => {
                        p.thermal_iters = p.thermal_iters.min(10).max(1);
                        p.spe_iters = p.spe_iters.min(4);
                    }
                    _ => {}
                }
                let hardness = bake_layer_hardness(ctx, input, p.hardness, &p.hardness_source);
                let ridge_lock = bake_optional_mask(ctx, input, &p.ridge_lock);
                let levels = match ctx.quality {
                    crate::eval::PreviewQuality::Draft => {
                        analyze::amplify_sim_levels(ctx.metrics.width, p.level_count.max(1).min(2))
                    }
                    _ => analyze::amplify_sim_levels(ctx.metrics.width, p.level_count),
                };
                let result = analyze::multi_scale_amplify(
                    input,
                    &p,
                    &hardness,
                    ridge_lock.as_ref(),
                    &levels,
                );
                ctx.aux_insert(keys::HARDNESS, hardness);
                ctx.aux_insert(keys::EROSION, result.erosion);
                ctx.aux_insert(keys::DEPOSITION, result.deposition);
                ctx.ensure_derived_fields(&result.height);
                Ok(result.height)
            }
            LayerKind::Blur(p) => Ok(generators::blur(input, p)),
            LayerKind::Coastal(p) => Ok(generators::coastal(input, p)),
            LayerKind::EffectFilter(p) => Ok(generators::effect_filter(input, p)),
            LayerKind::Path(p) => {
                let (hf, wet) = generators::path_stamp(input, p);
                merge_wetness_max(ctx, wet);
                Ok(hf)
            }
            LayerKind::RiverNetwork(p) => {
                let (hf, wet) = generators::river_network(input, p);
                merge_wetness_max(ctx, wet);
                Ok(hf)
            }
            LayerKind::SandSimulation(p) => {
                let mut authored = p.clone();
                let quality_cap = match ctx.quality {
                    // Keep Draft interactive; Medium/Full run on the eval worker.
                    crate::eval::PreviewQuality::Draft => 32,
                    crate::eval::PreviewQuality::Medium => 256,
                    crate::eval::PreviewQuality::Full => authored.iterations.max(1),
                    crate::eval::PreviewQuality::Export => authored.iterations.max(1),
                };
                let steps = analyze::transport_steps_for_quality(authored.iterations, quality_cap);
                // Cap per-step avalanching on Draft so edits stay responsive.
                if matches!(ctx.quality, crate::eval::PreviewQuality::Draft) {
                    authored.avalanche_iters = authored.avalanche_iters.min(4);
                } else if matches!(ctx.quality, crate::eval::PreviewQuality::Medium) {
                    authored.avalanche_iters = authored.avalanche_iters.min(12);
                }
                let aeolian = generators::sand_simulation_full(input, &authored, steps);
                publish_aeolian_aux(ctx, &aeolian);
                let mut sand_mask = MaskField::zeros(input.metrics);
                let raw = aeolian.sand_depth.data();
                let mut max_s = 1e-6f32;
                for &v in raw {
                    max_s = max_s.max(v);
                }
                for j in 0..input.metrics.height {
                    for i in 0..input.metrics.width {
                        sand_mask.set(i, j, (aeolian.sand_depth.get(i, j) / max_s).clamp(0.0, 1.0));
                    }
                }
                ctx.aux_insert(keys::SAND_MATERIAL_MASK, sand_mask);
                Ok(aeolian.height)
            }
            LayerKind::FluidSimulation(p) => {
                let (height, mask) = generators::fluid_simulation(input, p);
                merge_wetness_max(ctx, mask.clone());
                ctx.aux_insert(keys::WATER_DEPTH, mask);
                Ok(height)
            }
            LayerKind::Materials(p) => {
                // Height passthrough: materials publish aux maps (weights/hardness), not height.
                let (weights, hardness) = surface::bake_materials(
                    input,
                    p,
                    ctx.aux_maps.wetness.as_ref(),
                    &ctx.aux,
                    &ctx.mask_assets,
                    Some(&ctx.masks),
                );
                // Also keep the explicit ex-bake path consistent when only rules exist.
                let hardness = if p.strata.is_empty() {
                    bake_hardness_from_materials_ex(
                        &weights,
                        &p.rules,
                        &p.strata,
                        p.default_hardness,
                    )
                } else {
                    hardness
                };
                ctx.aux_insert(keys::MATERIALS, weights);
                ctx.aux_insert(keys::HARDNESS, hardness);
                if !p.strata.is_empty() {
                    // Store meters as a MaskField for depth-aware erosion consumers.
                    let reference = MaskField::from_raw(input.metrics, &input.to_dense());
                    ctx.aux_insert(keys::STRATA_REFERENCE, reference);
                    ctx.aux_maps.strata = Some(p.strata.clone());
                    ctx.aux_maps.bed_geometry = p.bed_geometry;
                }
                Ok(input.clone())
            }
            LayerKind::Biomes(p) => {
                // Height passthrough: climate/biome classification writes aux maps only.
                let maps = surface::bake_biomes_climate(
                    input,
                    p,
                    ctx.aux_maps.wetness.as_ref(),
                    ctx.aux_maps.sediment.as_ref(),
                );
                ctx.aux_insert(keys::TEMPERATURE, maps.temperature);
                ctx.aux_insert(keys::RAINFALL, maps.rainfall);
                ctx.aux_insert(keys::HUMIDITY, maps.humidity);
                ctx.aux_insert(keys::ARIDITY, maps.aridity);
                ctx.aux_insert(keys::SNOW, maps.snow);
                ctx.aux_insert(keys::SOIL_MOISTURE, maps.soil_moisture);
                ctx.aux_insert(keys::WIND_EXPOSURE, maps.wind_exposure);
                ctx.aux_insert(keys::BIOMES, maps.biomes);
                Ok(input.clone())
            }
            LayerKind::Vegetation(p) => {
                // Height passthrough: vegetation density/aux; optional hardness boost only.
                let density = surface::vegetation_density(
                    input,
                    p,
                    ctx.aux_maps.biomes.as_ref(),
                    Some(&ctx.masks),
                    ctx.aux_maps.get(keys::LAND_MASK),
                );
                ctx.aux_insert(keys::VEGETATION, density.clone());
                // Optional Cordonnier one-way: roots slightly boost hardness cohesion.
                if p.root_cohesion > 1e-6 {
                    if let Some(hardness) = ctx.aux_maps.hardness.as_ref() {
                        let boosted = crate::climate::apply_root_cohesion(
                            hardness,
                            &density,
                            p.root_cohesion,
                        );
                        ctx.aux_insert(keys::HARDNESS, boosted);
                    }
                }
                Ok(input.clone())
            }
            LayerKind::OverhangStamp(p) => {
                let result = volumetric::apply_overhang_stamp(input, p);
                ctx.aux_insert(keys::OVERHANG_CEILING, result.ceiling);
                ctx.aux_insert(keys::OVERHANG_MASK, result.mask);
                Ok(result.height)
            }
            LayerKind::LocalSdf(p) => {
                let result = volumetric::apply_local_sdf(input, p);
                ctx.aux_insert(keys::OVERHANG_CEILING, result.ceiling);
                ctx.aux_insert(keys::OVERHANG_MASK, result.mask);
                Ok(result.height)
            }
        }
    }
}

fn hardness_is_uniform(field: &MaskField, constant: f32) -> bool {
    let c = constant.clamp(0.0, 1.0);
    field.data().iter().all(|&v| (v - c).abs() < 1e-5)
}

/// Depth-aware strata path when Materials left a stack and the layer opts into
/// `MaskSource::Hardness` (baked Materials \(K\) / depth peeling).
fn use_strata_hardness(ctx: &EvalContext, source: &MaskSource) -> bool {
    has_depth_aware_strata(&ctx.aux_maps) && matches!(source, MaskSource::Hardness)
}

fn bake_layer_hardness(
    ctx: &EvalContext,
    input: &Heightfield,
    constant: f32,
    source: &MaskSource,
) -> MaskField {
    let painted = match source {
        MaskSource::None => None,
        MaskSource::Constant(v) => Some(MaskField::filled(input.metrics, *v)),
        MaskSource::Painted { mask_id } => ctx.masks.get(mask_id).cloned(),
        MaskSource::Hardness => ctx.aux_maps.hardness.clone(),
        MaskSource::Named(name) => ctx.aux_maps.get(name).cloned(),
        MaskSource::LayerOutput { output_id } => ctx.published_outputs.get(output_id).cloned(),
        other => {
            // Bake procedural sources against the input heightfield.
            let baked = crate::mask::bake_mask_assets(
                &[crate::mask::MaskAsset {
                    id: crate::mask::MaskId::new(),
                    name: "hardness_tmp".into(),
                    source: other.clone(),
                    ops: Vec::new(),
                    paint: None,
                    display_color: crate::mask::default_mask_display_color(),
                }],
                input,
                input.metrics,
                &ctx.aux,
            );
            baked.into_values().next()
        }
    };
    resolve_hardness(input.metrics, constant, painted.as_ref(), &ctx.aux_maps)
}

fn publish_authoring(ctx: &mut EvalContext, result: authoring::AuthoringResult) -> Heightfield {
    for (key, field) in result.fields {
        ctx.aux_insert(key, field);
    }
    ctx.ensure_derived_fields(&result.height);
    result.height
}
fn publish_authoring_merge(
    ctx: &mut EvalContext,
    mut result: authoring::AuthoringResult,
    merge_keys: &[&'static str],
) -> Heightfield {
    for key in merge_keys {
        let Some(previous) = ctx.aux_maps.get(key).cloned() else {
            continue;
        };
        let Some(incoming) = result.fields.get_mut(key) else {
            continue;
        };
        if previous.metrics.width != incoming.metrics.width
            || previous.metrics.height != incoming.metrics.height
        {
            continue;
        }
        for (value, old) in incoming.data_mut().iter_mut().zip(previous.data()) {
            *value = value.max(*old);
        }
    }
    publish_authoring(ctx, result)
}

/// Max-blend channel wetness so Path / RiverNetwork / RiverCarve stack with fluids.
fn merge_wetness_max(ctx: &mut EvalContext, incoming: MaskField) {
    if let Some(prev) = ctx.aux_maps.wetness.as_ref() {
        let mut merged = prev.clone();
        let w = incoming.metrics.width;
        let h = incoming.metrics.height;
        for j in 0..h {
            for i in 0..w {
                let a = merged.get(i, j);
                let b = incoming.get(i, j);
                if b > a {
                    merged.set(i, j, b);
                }
            }
        }
        ctx.aux_insert(keys::WETNESS, merged);
    } else {
        ctx.aux_insert(keys::WETNESS, incoming);
    }
}

fn publish_aeolian_aux(ctx: &mut EvalContext, aeolian: &analyze::AeolianResult) {
    ctx.aux_insert(keys::SAND_DEPTH, aeolian.sand_depth.clone());
    ctx.aux_insert(keys::BEDROCK_HEIGHT, aeolian.bedrock.clone());
    ctx.aux_insert(keys::WIND_DIRECTION, aeolian.wind_direction.clone());
    ctx.aux_insert(keys::WIND_SPEED, aeolian.wind_speed.clone());
    ctx.aux_insert(keys::SAND_FLUX, aeolian.sand_flux.clone());
    ctx.aux_insert(keys::EROSION, aeolian.erosion.clone());
    ctx.aux_insert(keys::DEPOSITION, aeolian.deposition.clone());
    ctx.aux_insert(keys::SHELTERING, aeolian.sheltering.clone());
    ctx.aux_insert(keys::DUNE_CREST, aeolian.dune_crest.clone());
    // Transport direction alias for matter / overlays that already consume flow_direction.
    ctx.aux_insert(keys::FLOW_DIRECTION, aeolian.wind_direction.clone());
}

fn bake_optional_mask(
    ctx: &EvalContext,
    input: &Heightfield,
    source: &MaskSource,
) -> Option<MaskField> {
    match source {
        MaskSource::None => None,
        MaskSource::Constant(v) => Some(MaskField::filled(input.metrics, *v)),
        MaskSource::Painted { mask_id } => ctx.masks.get(mask_id).cloned(),
        MaskSource::Hardness => ctx.aux_maps.hardness.clone(),
        MaskSource::LayerOutput { output_id } => ctx.published_outputs.get(output_id).cloned(),
        MaskSource::Named(name) => ctx.aux_maps.get(name).cloned(),
        other => {
            let baked = crate::mask::bake_mask_assets(
                &[crate::mask::MaskAsset {
                    id: crate::mask::MaskId::new(),
                    name: "lock_tmp".into(),
                    source: other.clone(),
                    ops: Vec::new(),
                    paint: None,
                    display_color: crate::mask::default_mask_display_color(),
                }],
                input,
                input.metrics,
                &ctx.aux,
            );
            baked.into_values().next()
        }
    }
}
