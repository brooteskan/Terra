//! Layer evaluation, caching, and dirty propagation.

mod cache;
mod processors;
mod scheduler;
mod smart_cache;
mod worker;

pub use crate::quality::PreviewQuality;
pub use cache::{CachedOutput, LayerCache};
pub use processors::ProcessorRegistry;
pub use scheduler::{EvalJob, EvalScheduler};
pub use smart_cache::DiskSmartCache;
pub use worker::{EvalWorkRequest, EvalWorkResult, EvalWorker};

use crate::fields::AuxMaps;
use crate::heightfield::{Heightfield, HeightfieldMetrics};
use crate::layer::{blend_heights, Layer, LayerId, LayerStack, StackNode};
use crate::mask::{MaskAsset, MaskField, MaskId};
use std::collections::{HashMap, HashSet};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use std::time::Instant;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EvalError {
    #[error("unknown layer processor for kind")]
    UnknownProcessor,
    #[error("cancelled")]
    Cancelled,
    #[error("io: {0}")]
    Io(String),
}

/// How a layer contributed to a particular evaluation pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerEvalStatus {
    Disabled,
    CacheHit,
    Computed,
}

/// CPU timing and cache provenance for one artist-visible layer.
#[derive(Debug, Clone)]
pub struct LayerEvalTiming {
    pub layer: LayerId,
    pub layer_name: String,
    pub layer_kind: &'static str,
    pub elapsed_us: u64,
    pub status: LayerEvalStatus,
}

pub struct EvalContext {
    pub metrics: HeightfieldMetrics,
    /// Project-wide progressive evaluation controls. Keeping this on the
    /// context makes CPU, worker, and hybrid evaluation use the document's
    /// authored world scale and level schedule instead of hidden defaults.
    pub level_steps: crate::analyze::LevelStepSettings,
    pub masks: HashMap<MaskId, MaskField>,
    pub mask_assets: Vec<MaskAsset>,
    /// Typed aux maps (preferred). Processors should read/write these.
    pub aux_maps: AuxMaps,
    /// String-key adapter kept in sync with [`Self::aux_maps`] for cache / IO / masks.
    pub aux: HashMap<String, MaskField>,
    /// Stable outputs published by layers already evaluated below the current layer.
    pub published_outputs: HashMap<crate::layer::OutputId, MaskField>,
    pub cancelled: bool,
    /// Shared worker generation; a mismatch cancels at the next layer boundary.
    pub(crate) cancellation_generation: Option<(Arc<AtomicU64>, u64)>,
    pub quality: PreviewQuality,
    /// Timings for the current pass, in actual layer evaluation order.
    pub layer_timings: Vec<LayerEvalTiming>,
}

impl EvalContext {
    pub fn new(metrics: HeightfieldMetrics) -> Self {
        Self {
            metrics,
            level_steps: crate::analyze::LevelStepSettings::default(),
            masks: HashMap::new(),
            mask_assets: Vec::new(),
            aux_maps: AuxMaps::new(),
            aux: HashMap::new(),
            published_outputs: HashMap::new(),
            cancelled: false,
            quality: PreviewQuality::Full,
            cancellation_generation: None,
            layer_timings: Vec::new(),
        }
    }

    pub fn check_cancelled(&self) -> Result<(), EvalError> {
        let generation_changed = self
            .cancellation_generation
            .as_ref()
            .is_some_and(|(generation, expected)| generation.load(Ordering::Acquire) != *expected);
        if self.cancelled || generation_changed {
            Err(EvalError::Cancelled)
        } else {
            Ok(())
        }
    }

    pub fn set_cancellation_generation(&mut self, generation: Arc<AtomicU64>, expected: u64) {
        self.cancellation_generation = Some((generation, expected));
    }

    /// Insert an aux map into both typed and string stores.
    pub fn aux_insert(&mut self, key: impl Into<String>, field: MaskField) {
        let key = key.into();
        let canonical = crate::fields::keys::canonical(&key).to_string();
        self.aux_maps.insert(canonical.clone(), field.clone());
        if canonical == crate::fields::keys::SEDIMENT_THICKNESS {
            self.aux.remove(crate::fields::keys::SEDIMENT_DEPTH);
            self.aux.remove(crate::fields::keys::LOOSE_SEDIMENT);
        }
        self.aux.insert(canonical, field);
    }

    /// Replace string aux and rebuild typed maps (worker / scheduler ingest).
    /// Preserves any strata already on `aux_maps` when the HashMap has none.
    pub fn set_aux_hashmap(&mut self, aux: HashMap<String, MaskField>) {
        let keep_strata = self.aux_maps.strata.take();
        self.aux_maps = AuxMaps::from_hashmap_preserving_strata(&aux, keep_strata);
        self.sync_aux_hashmap();
    }

    /// Push typed maps into the string HashMap adapter (strata stays on `aux_maps`).
    pub fn sync_aux_hashmap(&mut self) {
        self.aux = self.aux_maps.to_hashmap();
    }

    /// Ensure slope/curvature derived caches exist for the current heightfield.
    pub fn ensure_derived_fields(&mut self, hf: &Heightfield) {
        self.aux_maps.ensure_derived(hf);
        self.sync_aux_hashmap();
    }
}

pub struct StackEvaluator {
    pub registry: ProcessorRegistry,
    pub cache: LayerCache,
}

impl Default for StackEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl StackEvaluator {
    pub fn new() -> Self {
        Self {
            registry: ProcessorRegistry::builtin(),
            cache: LayerCache::new(),
        }
    }

    pub fn mark_dirty_from(&mut self, stack: &LayerStack, id: LayerId) {
        let ids = stack.layer_ids();
        if let Some(start) = ids.iter().position(|&x| x == id) {
            for &dep in &ids[start..] {
                self.cache.mark_dirty(dep);
            }
        } else {
            // Unknown id: dirty everything
            for &dep in &ids {
                self.cache.mark_dirty(dep);
            }
        }
    }

    /// Stage-aware dirty: only invalidate this layer and later EvalStages.
    ///
    /// Material edits do not rebuild height (Blueprint / PreBiome / Hydro).
    /// Vegetation / scatter edits do not rebuild materials.
    pub fn mark_dirty_from_stage(&mut self, stack: &LayerStack, id: LayerId) {
        let Some(layer) = stack.find(id) else {
            self.mark_dirty_from(stack, id);
            return;
        };
        let min_order = layer.kind.eval_stage().order();
        self.cache.mark_dirty(id);
        for lid in stack.layer_ids() {
            if lid == id {
                continue;
            }
            if let Some(other) = stack.find(lid) {
                if other.kind.eval_stage().order() >= min_order {
                    self.cache.mark_dirty(lid);
                }
            }
        }
    }

    /// Dirty all layers at or after an EvalStage (World Rule selective invalidation).
    pub fn mark_dirty_from_eval_stage(
        &mut self,
        stack: &LayerStack,
        stage: crate::landscape_blueprint::EvalStage,
    ) {
        let min_order = stage.order();
        for lid in stack.layer_ids() {
            if let Some(layer) = stack.find(lid) {
                if layer.kind.eval_stage().order() >= min_order {
                    self.cache.mark_dirty(lid);
                }
            }
        }
    }

    pub fn mark_all_dirty(&mut self, stack: &LayerStack) {
        for id in stack.layer_ids() {
            self.cache.mark_dirty(id);
        }
    }

    /// Discard every layer cache entry (project switch / hard reset).
    pub fn clear_project_caches(&mut self) {
        self.cache.clear();
    }

    /// Full rebuild (Phase 1 path) — tree walk so scoped groups compose correctly.
    pub fn rebuild_all(
        &mut self,
        stack: &LayerStack,
        ctx: &mut EvalContext,
    ) -> Result<Heightfield, EvalError> {
        profiling::scope!("rebuild_all");
        self.cache.clear();
        let seed = Heightfield::zeros(ctx.metrics);
        self.evaluate_nodes(&stack.nodes, ctx, &seed)
    }

    /// Incremental rebuild from first dirty layer (Phase 4).
    ///
    /// Flat stacks use suffix-only evaluation from the first dirty layer. Scoped
    /// groups and solo mode use the same dirty-aware tree walk as [`evaluate_nodes`].
    pub fn rebuild_incremental(
        &mut self,
        stack: &LayerStack,
        ctx: &mut EvalContext,
    ) -> Result<Heightfield, EvalError> {
        profiling::scope!("rebuild_incremental");
        if stack.requires_tree_evaluation() {
            let seed = Heightfield::zeros(ctx.metrics);
            return self.evaluate_nodes(&stack.nodes, ctx, &seed);
        }

        let layers: Vec<&Layer> = stack.flatten_layers();
        if layers.is_empty() {
            return Ok(Heightfield::zeros(ctx.metrics));
        }

        let first_dirty = layers.iter().position(|l| {
            if l.common.cached {
                self.cache.get_or_load(l.id(), ctx.metrics).is_none()
            } else {
                self.cache.is_dirty(l.id())
            }
        });

        // All clean: return cached top if dimensions match.
        if first_dirty.is_none() {
            if let Some(top) = self.cache.get(layers.last().unwrap().id()) {
                if top.height.metrics.width == ctx.metrics.width
                    && top.height.metrics.height == ctx.metrics.height
                    && !top.dirty
                {
                    for layer in &layers {
                        record_reused_layer(ctx, layer);
                    }
                    return Ok(top.height.clone());
                }
            }
        }

        let first_dirty = first_dirty.unwrap_or(0);
        for layer in &layers[..first_dirty] {
            record_reused_layer(ctx, layer);
        }

        let mut current = if first_dirty == 0 {
            Heightfield::zeros(ctx.metrics)
        } else {
            let prev_id = layers[first_dirty - 1].id();
            self.cache
                .get_or_load(prev_id, ctx.metrics)
                .map(|c| c.height.clone())
                .unwrap_or_else(|| Heightfield::zeros(ctx.metrics))
        };

        for layer in &layers[first_dirty..] {
            ctx.check_cancelled()?;
            current = self.evaluate_layer(ctx, &current, layer)?;
            self.store_cached(layer.id(), &current, ctx, layer.common.cached);
        }

        Ok(current)
    }

    /// Evaluate a node list bottom→top, composing scoped groups as units.
    pub fn evaluate_nodes(
        &mut self,
        nodes: &[StackNode],
        ctx: &mut EvalContext,
        input: &Heightfield,
    ) -> Result<Heightfield, EvalError> {
        let mut current = input.clone();
        let soloing = nodes.iter().any(node_contains_solo);
        for node in nodes {
            ctx.check_cancelled()?;
            if soloing && !node_contains_solo(node) {
                continue;
            }
            match node {
                StackNode::Layer(layer) => {
                    current = self.evaluate_layer(ctx, &current, layer)?;
                    self.store_cached(layer.id(), &current, ctx, layer.common.cached);
                }
                StackNode::Group(group) if !group.enabled => {}
                StackNode::Group(group) => {
                    use crate::layer::{GroupEvalMode, GroupInputMode};

                    refresh_point_of_use_masks(ctx, &current);
                    let pass_through =
                        matches!(group.eval_mode, GroupEvalMode::PassThrough) && !group.is_scoped();

                    if pass_through {
                        // Organisational folder: children mutate the live context.
                        current = self.evaluate_nodes(&group.children, ctx, &current)?;
                    } else {
                        // Isolated composite: private working height, then mix back.
                        let private_seed = match &group.input_mode {
                            GroupInputMode::CopyInput => current.clone(),
                            GroupInputMode::EmptyHeight => Heightfield::zeros(ctx.metrics),
                            GroupInputMode::SelectedField(_) => current.clone(),
                        };
                        // Snapshot aux so child sims don't leak into the parent
                        // until after the group composite.
                        let aux_snapshot = ctx.aux_maps.clone();
                        let aux_hash_snapshot = ctx.aux.clone();
                        let descendant_ids = collect_descendant_layer_ids(&group.children);
                        let (group_out, child_aux) = if let Some((height, aux)) = self
                            .try_reuse_group_cache(group.id, ctx, &descendant_ids, &private_seed)
                        {
                            record_subtree_cache_hits(ctx, &group.children);
                            (height, aux)
                        } else {
                            let group_out =
                                self.evaluate_nodes(&group.children, ctx, &private_seed)?;
                            let child_aux = ctx.aux_maps.clone();
                            self.store_group_cached(
                                group.id,
                                &group_out,
                                &child_aux,
                                &private_seed,
                                ctx,
                                group.cache_policy.to_legacy_cached(),
                            );
                            (group_out, child_aux)
                        };
                        // Restore parent aux, then selectively merge published child aux
                        // under the group mask after height composite.
                        ctx.aux_maps = aux_snapshot;
                        ctx.aux = aux_hash_snapshot;
                        ctx.sync_aux_hashmap();

                        let mask = effective_layer_mask(ctx, &group.masks, &current);
                        // Biome Filters blend toward lower biomes at `filter_blending`
                        // (1.0 = full mix, 0.0 = no contribution) rather than a hard cut.
                        // Height-delta semantics: with CopyInput, Normal mix is equivalent to
                        //   H = shared + w * (biome_result - shared)
                        // which avoids blending unrelated absolute heights.
                        let mix_opacity =
                            if matches!(group.group_kind, crate::layer::GroupKind::Biome) {
                                group.opacity * group.filter_blending
                            } else {
                                group.opacity
                            };
                        current = if matches!(group.group_kind, crate::layer::GroupKind::Biome)
                            && matches!(group.input_mode, GroupInputMode::CopyInput)
                        {
                            mix_height_delta(
                                &current,
                                &private_seed,
                                &group_out,
                                mix_opacity,
                                &mask,
                            )
                        } else {
                            mix_heightfields(&current, &group_out, group.blend, mix_opacity, &mask)
                        };
                        // Merge child aux weighted by group mask (non-destructive leak fix).
                        merge_aux_masked(ctx, &child_aux, &mask, mix_opacity);
                    }
                }
            }
        }
        Ok(current)
    }

    /// Continue evaluating a flattened stack from a precomputed heightfield.
    ///
    /// `current` must be the height entering `start_index`, and `ctx` must contain the
    /// equivalent auxiliary and published-output state produced by the skipped prefix.
    /// Callers that cannot supply that complete checkpoint must restart from layer zero.
    pub fn evaluate_suffix(
        &mut self,
        stack: &LayerStack,
        ctx: &mut EvalContext,
        start_index: usize,
        mut current: Heightfield,
    ) -> Result<Heightfield, EvalError> {
        let layers = stack.flatten_layers();
        for layer in layers.into_iter().skip(start_index) {
            ctx.check_cancelled()?;
            current = self.evaluate_layer(ctx, &current, layer)?;
            self.store_cached(layer.id(), &current, ctx, layer.common.cached);
        }
        Ok(current)
    }

    fn store_cached(&mut self, id: LayerId, height: &Heightfield, ctx: &EvalContext, baked: bool) {
        let output = CachedOutput {
            height: height.clone(),
            generation: self.cache.generation,
            dirty: false,
            aux: ctx.aux_maps.to_hashmap(),
            strata: ctx.aux_maps.strata.clone(),
        };
        if baked {
            self.cache.insert_baked(id, output);
        } else {
            self.cache.insert(id, output);
        }
    }

    /// Cache an isolated group's private composite keyed by its input fingerprint.
    fn store_group_cached(
        &mut self,
        id: LayerId,
        height: &Heightfield,
        child_aux: &crate::fields::AuxMaps,
        input: &Heightfield,
        _ctx: &EvalContext,
        baked: bool,
    ) {
        let output = CachedOutput {
            height: height.clone(),
            generation: height_fingerprint(input),
            dirty: false,
            aux: child_aux.to_hashmap(),
            strata: child_aux.strata.clone(),
        };
        if baked {
            self.cache.insert_baked(id, output);
        } else {
            self.cache.insert(id, output);
        }
    }

    fn evaluate_layer(
        &mut self,
        ctx: &mut EvalContext,
        input: &Heightfield,
        layer: &Layer,
    ) -> Result<Heightfield, EvalError> {
        let timing_started = Instant::now();
        if !layer.common.enabled {
            record_layer_timing(ctx, layer, timing_started, LayerEvalStatus::Disabled);
            return Ok(input.clone());
        }

        // Terrain-aware and runtime masks are evaluated against the exact field
        // entering their owner. This makes placement deterministic in preview,
        // export, and cold evaluation instead of depending on a previous frame.
        refresh_point_of_use_masks(ctx, input);

        // Any clean cached checkpoint reuses height + aux without re-invoking the processor.
        if !self.cache.is_dirty(layer.id()) {
            if let Some(cached) = self.cache.get_or_load(layer.id(), ctx.metrics) {
                ctx.aux_maps.extend_hashmap(&cached.aux);
                if cached.strata.is_some() {
                    ctx.aux_maps.strata = cached.strata.clone();
                }
                ctx.sync_aux_hashmap();
                publish_layer_outputs(ctx, layer, &cached.height);
                record_layer_timing(ctx, layer, timing_started, LayerEvalStatus::CacheHit);
                return Ok(cached.height.clone());
            }
        }

        let scaled_layer = layer_with_world_scale(layer, ctx.level_steps.world_scale);
        let mut bound_layer = apply_param_bindings(ctx, &scaled_layer);
        let generated = self.registry.evaluate(ctx, input, &bound_layer)?;
        // Avoid unused-mut warning if future passes mutate further.
        let _ = &mut bound_layer;
        let mask = effective_layer_mask(ctx, &layer.common.masks, input);
        // Gate materials / vegetation aux by local placement (Biome × Local at group+layer).
        if matches!(
            layer.kind,
            crate::layer::LayerKind::Materials(_) | crate::layer::LayerKind::Vegetation(_)
        ) {
            gate_aux_by_mask(ctx, &mask);
        }
        let mut out = input.clone();
        let w = input.metrics.width;
        let h = input.metrics.height;
        for j in 0..h {
            for i in 0..w {
                let hin = input.get(i, j);
                let hlayer = generated.get(i, j);
                let m = mask.get(i, j);
                let v = blend_heights(layer.common.blend, hin, hlayer, layer.common.opacity, m);
                out.set(i, j, v);
            }
        }
        out.refresh_halos();
        publish_layer_outputs(ctx, layer, &out);
        record_layer_timing(ctx, layer, timing_started, LayerEvalStatus::Computed);
        Ok(out)
    }

    fn try_reuse_group_cache(
        &mut self,
        group_id: LayerId,
        ctx: &EvalContext,
        descendant_ids: &[LayerId],
        input: &Heightfield,
    ) -> Option<(Heightfield, crate::fields::AuxMaps)> {
        if self.cache.is_dirty(group_id) {
            return None;
        }
        if descendant_ids.iter().any(|&id| self.cache.is_dirty(id)) {
            return None;
        }
        let cached = self.cache.get_or_load(group_id, ctx.metrics)?;
        if cached.generation != height_fingerprint(input) {
            return None;
        }
        let child_aux = crate::fields::AuxMaps::from_hashmap_preserving_strata(
            &cached.aux,
            cached.strata.clone(),
        );
        Some((cached.height.clone(), child_aux))
    }
}

fn collect_descendant_layer_ids(nodes: &[StackNode]) -> Vec<LayerId> {
    let mut ids = Vec::new();
    collect_descendant_layer_ids_into(nodes, &mut ids);
    ids
}

fn collect_descendant_layer_ids_into(nodes: &[StackNode], out: &mut Vec<LayerId>) {
    for node in nodes {
        match node {
            StackNode::Layer(layer) => out.push(layer.id()),
            StackNode::Group(group) if group.enabled => {
                collect_descendant_layer_ids_into(&group.children, out);
            }
            StackNode::Group(_) => {}
        }
    }
}

fn record_subtree_cache_hits(ctx: &mut EvalContext, nodes: &[StackNode]) {
    let soloing = nodes.iter().any(node_contains_solo);
    for node in nodes {
        if soloing && !node_contains_solo(node) {
            continue;
        }
        match node {
            StackNode::Layer(layer) if layer.common.enabled => record_reused_layer(ctx, layer),
            StackNode::Group(group) if group.enabled => {
                record_subtree_cache_hits(ctx, &group.children);
            }
            _ => {}
        }
    }
}

fn record_layer_timing(
    ctx: &mut EvalContext,
    layer: &Layer,
    started: Instant,
    status: LayerEvalStatus,
) {
    ctx.layer_timings.push(LayerEvalTiming {
        layer: layer.id(),
        layer_name: layer.common.name.clone(),
        layer_kind: layer.kind.type_display_name(),
        elapsed_us: started.elapsed().as_micros() as u64,
        status,
    });
}

fn record_reused_layer(ctx: &mut EvalContext, layer: &Layer) {
    ctx.layer_timings.push(LayerEvalTiming {
        layer: layer.id(),
        layer_name: layer.common.name.clone(),
        layer_kind: layer.kind.type_display_name(),
        elapsed_us: 0,
        status: LayerEvalStatus::CacheHit,
    });
}

fn node_contains_solo(node: &StackNode) -> bool {
    match node {
        StackNode::Layer(layer) => layer.common.solo,
        StackNode::Group(group) => group.children.iter().any(node_contains_solo),
    }
}

fn refresh_point_of_use_masks(ctx: &mut EvalContext, input: &Heightfield) {
    let assets: Vec<_> = ctx
        .mask_assets
        .iter()
        .filter(|asset| mask_source_is_point_of_use(&asset.source))
        .cloned()
        .collect();
    if assets.is_empty() {
        return;
    }
    let rebaked = crate::mask::bake_mask_assets_resolved(
        &assets,
        input,
        input.metrics,
        &ctx.aux,
        &ctx.published_outputs,
    );
    ctx.masks.extend(rebaked);
}

fn mask_source_is_point_of_use(source: &crate::mask::MaskSource) -> bool {
    use crate::mask::MaskSource::*;
    matches!(
        source,
        Height { .. }
            | Slope { .. }
            | Aspect { .. }
            | Curvature { .. }
            | Convexity
            | Concavity
            | AmbientOcclusion { .. }
            | DistanceField { .. }
            | Named(_)
            | FlowDirection
            | FlowAccumulation { .. }
            | Wetness
            | Sediment
            | Erosion
            | Deposition
            | Hardness
            | Temperature
            | Rainfall
            | Humidity
            | Snow
            | SoilMoisture
            | WindExposure
            | LayerOutput { .. }
    )
}

fn apply_param_bindings(ctx: &EvalContext, layer: &Layer) -> Layer {
    if layer.common.param_bindings.is_empty() {
        return layer.clone();
    }
    let mut out = layer.clone();
    for binding in &layer.common.param_bindings {
        let sample = sample_binding_source(ctx, &binding.source);
        if binding.target.0 == "opacity" {
            out.common.opacity = binding
                .apply_scalar(layer.common.opacity, sample)
                .clamp(0.0, 1.0);
        } else {
            out.kind.apply_param_binding(binding, sample);
        }
    }
    out
}

fn sample_binding_source(ctx: &EvalContext, source: &crate::layer::BindingSource) -> f32 {
    use crate::layer::BindingSource;
    match source {
        BindingSource::Constant(v) => v.clamp(0.0, 1.0),
        BindingSource::Mask(id) => mean_mask(ctx.masks.get(id)),
        BindingSource::LayerOutput(id) | BindingSource::GroupOutput(id) => {
            mean_mask(ctx.published_outputs.get(id))
        }
        BindingSource::Field(field) => mean_mask(ctx.aux_maps.get(&field.cache_key())),
    }
}

fn mean_mask(field: Option<&MaskField>) -> f32 {
    let Some(f) = field else {
        return 0.0;
    };
    let w = f.metrics.width;
    let h = f.metrics.height;
    if w == 0 || h == 0 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    let mut n = 0u32;
    // Subsample for speed — binding modulation uses mean influence, not every cell.
    let step = ((w.max(h) / 64).max(1)) as u32;
    let mut j = 0u32;
    while j < h {
        let mut i = 0u32;
        while i < w {
            sum += f.get(i, j);
            n += 1;
            i += step;
        }
        j += step;
    }
    if n == 0 {
        0.0
    } else {
        (sum / n as f32).clamp(0.0, 1.0)
    }
}

fn layer_with_world_scale(layer: &Layer, world_scale: f32) -> Layer {
    let scale = world_scale.clamp(0.05, 20.0);
    if (scale - 1.0).abs() < 1e-6 {
        return layer.clone();
    }
    let mut layer = layer.clone();
    let scale_noise = |noise: &mut crate::layer::NoiseParams| {
        noise.frequency /= scale;
        noise.offset_x *= scale;
        noise.offset_z *= scale;
    };
    match &mut layer.kind {
        crate::layer::LayerKind::NoiseValue(p)
        | crate::layer::LayerKind::NoisePerlin(p)
        | crate::layer::LayerKind::NoiseOpenSimplex(p) => scale_noise(p),
        crate::layer::LayerKind::NoiseWorley(p) => scale_noise(&mut p.base),
        crate::layer::LayerKind::Fbm(p) | crate::layer::LayerKind::Ridged(p) => {
            scale_noise(&mut p.base)
        }
        crate::layer::LayerKind::DomainWarp(p) => {
            scale_noise(&mut p.base);
            p.warp_frequency /= scale;
        }
        crate::layer::LayerKind::Mountains(p) => scale_noise(&mut p.base),
        crate::layer::LayerKind::Dunes(p) => {
            scale_noise(&mut p.base);
            p.wave_frequency /= scale;
        }
        crate::layer::LayerKind::Uplift(p) => {
            p.frequency /= scale;
            p.detail_frequency /= scale;
        }
        crate::layer::LayerKind::Island(p) => {
            p.coastline_frequency /= scale;
            p.ridge_frequency /= scale;
        }
        crate::layer::LayerKind::VoronoiRegions(p) => scale_noise(&mut p.base),
        _ => {}
    }
    layer
}

fn publish_layer_outputs(ctx: &mut EvalContext, layer: &Layer, height: &Heightfield) {
    for output in &layer.common.outputs {
        if !output.enabled {
            continue;
        }
        let field = if output.field == crate::fields::FieldId::Height {
            MaskField::from_raw(height.metrics, &height.to_dense())
        } else {
            let key = output.field.cache_key();
            let Some(field) = ctx.aux_maps.get(&key).cloned() else {
                continue;
            };
            field
        };
        ctx.published_outputs.insert(output.id, field);
    }
}

fn mix_heightfields(
    h_in: &Heightfield,
    h_layer: &Heightfield,
    blend: crate::layer::BlendMode,
    opacity: f32,
    mask: &MaskField,
) -> Heightfield {
    let mut out = h_in.clone();
    for j in 0..h_in.metrics.height {
        for i in 0..h_in.metrics.width {
            let v = blend_heights(
                blend,
                h_in.get(i, j),
                h_layer.get(i, j),
                opacity,
                mask.get(i, j),
            );
            out.set(i, j, v);
        }
    }
    out.refresh_halos();
    out
}

/// Biome height-delta composite: `H += w * (biome_result - shared_input)`.
///
/// `h_parent` is the stack accumulator below this biome, `shared` is the biome's
/// CopyInput seed, and `biome_result` is the biome group's private output.
fn mix_height_delta(
    h_parent: &Heightfield,
    shared: &Heightfield,
    biome_result: &Heightfield,
    opacity: f32,
    mask: &MaskField,
) -> Heightfield {
    let mut out = h_parent.clone();
    for j in 0..h_parent.metrics.height {
        for i in 0..h_parent.metrics.width {
            let w = (mask.get(i, j) * opacity).clamp(0.0, 1.0);
            let delta = biome_result.get(i, j) - shared.get(i, j);
            out.set(i, j, h_parent.get(i, j) + w * delta);
        }
    }
    out.refresh_halos();
    out
}

/// Merge child aux maps into the parent context, weighted by the group mask.
fn merge_aux_masked(
    ctx: &mut EvalContext,
    child: &crate::fields::AuxMaps,
    mask: &MaskField,
    opacity: f32,
) {
    let child_map = child.to_hashmap();
    for (key, child_field) in child_map {
        let mut out = ctx
            .aux_maps
            .get(&key)
            .cloned()
            .unwrap_or_else(|| MaskField::zeros(ctx.metrics));
        for j in 0..ctx.metrics.height {
            for i in 0..ctx.metrics.width {
                let w = (mask.get(i, j) * opacity).clamp(0.0, 1.0);
                let v = out.get(i, j) * (1.0 - w) + child_field.get(i, j) * w;
                out.set(i, j, v);
            }
        }
        ctx.aux_insert(key, out);
    }
    if child.strata.is_some() {
        ctx.aux_maps.strata = child.strata.clone();
    }
}

fn composite_distribution(
    ctx: &EvalContext,
    dist: &crate::mask::Distribution,
    input: &Heightfield,
) -> MaskField {
    use crate::mask::DistBakeContext;
    let slope = ctx.aux.get("slope").map(|m| m.data());
    let curv = ctx.aux.get("curvature").map(|m| m.data());
    let flow = ctx
        .aux
        .get("flow_accumulation")
        .or_else(|| ctx.aux.get("flow"))
        .map(|m| m.data());
    let bake_ctx = DistBakeContext {
        height: Some(input),
        slope_deg: slope,
        curvature: curv,
        flow,
        masks: &ctx.masks,
        aux: Some(&ctx.aux),
    };
    crate::mask::bake_distribution_with_context(dist, input.metrics, &bake_ctx)
}

/// Effective contribution mask from the layer's local distribution.
fn effective_layer_mask(
    ctx: &EvalContext,
    local: &crate::mask::Distribution,
    input: &Heightfield,
) -> MaskField {
    composite_distribution(ctx, local, input)
}

/// Fingerprint of height data for group-cache keyed reuse.
fn height_fingerprint(h: &Heightfield) -> u64 {
    let m = h.metrics;
    let mut state = (m.width as u64)
        .wrapping_mul(0x1000_0000_1b3)
        .wrapping_add(m.height as u64);
    if m.width == 0 || m.height == 0 {
        return state;
    }
    let corners = [
        (0, 0),
        (m.width - 1, 0),
        (0, m.height - 1),
        (m.width - 1, m.height - 1),
        (m.width / 2, m.height / 2),
    ];
    for (i, j) in corners {
        state ^= (h.get(i, j).to_bits() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        state = state.rotate_left(13);
    }
    let step_i = (m.width / 8).max(1);
    let step_j = (m.height / 8).max(1);
    let mut j = 0;
    while j < m.height {
        let mut i = 0;
        while i < m.width {
            state = state
                .wrapping_mul(0x1000_0000_1b3)
                .wrapping_add(h.get(i, j).to_bits() as u64);
            i += step_i;
        }
        j += step_j;
    }
    state
}

/// Multiply recent materials / vegetation aux fields by a placement mask.
fn gate_aux_by_mask(ctx: &mut EvalContext, mask: &MaskField) {
    use crate::fields::keys;
    let mul = |field: &mut MaskField| {
        let w = field.metrics.width;
        let h = field.metrics.height;
        for j in 0..h {
            for i in 0..w {
                let v = field.get(i, j) * mask.get(i, j);
                field.set(i, j, v);
            }
        }
    };
    for slot in [
        &mut ctx.aux_maps.materials,
        &mut ctx.aux_maps.hardness,
        &mut ctx.aux_maps.vegetation,
    ] {
        if let Some(field) = slot.as_mut() {
            mul(field);
        }
    }
    for key in [keys::MATERIALS, keys::HARDNESS, keys::VEGETATION] {
        if let Some(field) = ctx.aux.get_mut(key) {
            mul(field);
        }
    }
}

/// Helper used by tests to count processor invocations.
pub fn dirty_suffix_ids(stack: &LayerStack, from: LayerId) -> HashSet<LayerId> {
    let ids = stack.layer_ids();
    let mut set = HashSet::new();
    if let Some(start) = ids.iter().position(|&x| x == from) {
        for &id in &ids[start..] {
            set.insert(id);
        }
    }
    set
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::{BlendMode, FlatParams, LayerKind, NoiseParams};

    #[test]
    fn disabled_layer_noop() {
        let mut stack = LayerStack::new();
        let mut flat = Layer::new("Flat", LayerKind::Flat(FlatParams { height: 50.0 }));
        flat.common.enabled = false;
        stack.push(flat);
        let mut eval = StackEvaluator::new();
        let metrics = HeightfieldMetrics::new(16, 16, 64.0, 64.0);
        let mut ctx = EvalContext::new(metrics);
        let out = eval.rebuild_all(&stack, &mut ctx).unwrap();
        assert_eq!(out.get(0, 0), 0.0);
    }

    #[test]
    fn mark_dirty_from_suffix() {
        let mut stack = LayerStack::new();
        let a = Layer::new("A", LayerKind::Flat(FlatParams { height: 1.0 }));
        let b = Layer::new("B", LayerKind::Flat(FlatParams { height: 2.0 }));
        let c = Layer::new("C", LayerKind::NoiseValue(NoiseParams::default()));
        let id_a = a.id();
        let id_b = b.id();
        let id_c = c.id();
        stack.push(a);
        stack.push(b);
        stack.push(c);
        let mut eval = StackEvaluator::new();
        eval.mark_all_dirty(&stack);
        // clear dirty artificially
        for id in [id_a, id_b, id_c] {
            eval.cache.insert(
                id,
                CachedOutput {
                    height: Heightfield::zeros(HeightfieldMetrics::new(4, 4, 4.0, 4.0)),
                    generation: 0,
                    dirty: false,
                    aux: HashMap::new(),
                    strata: None,
                },
            );
        }
        eval.mark_dirty_from(&stack, id_b);
        assert!(!eval.cache.is_dirty(id_a));
        assert!(eval.cache.is_dirty(id_b));
        assert!(eval.cache.is_dirty(id_c));
        let suffix = dirty_suffix_ids(&stack, id_b);
        assert!(suffix.contains(&id_b) && suffix.contains(&id_c) && !suffix.contains(&id_a));
    }

    #[test]
    fn baked_lower_layer_is_reused_when_upper_layer_is_dirty() {
        let mut stack = LayerStack::new();
        let mut baked = Layer::new("Baked", LayerKind::Flat(FlatParams { height: 1.0 }));
        baked.common.cached = true;
        let baked_id = baked.id();
        let mut upper = Layer::new("Upper", LayerKind::Flat(FlatParams { height: 3.0 }));
        upper.common.blend = BlendMode::Add;
        let upper_id = upper.id();
        stack.push(baked);
        stack.push(upper);

        let metrics = HeightfieldMetrics::new(8, 8, 8.0, 8.0);
        let mut eval = StackEvaluator::new();
        eval.cache.insert(
            baked_id,
            CachedOutput {
                // Deliberately differs from the processor output. Rebuilding only
                // the upper layer must seed from this frozen checkpoint.
                height: Heightfield::filled(metrics, 7.0),
                generation: 0,
                dirty: false,
                aux: HashMap::new(),
                strata: None,
            },
        );
        eval.mark_dirty_from(&stack, upper_id);

        let mut ctx = EvalContext::new(metrics);
        let out = eval.rebuild_incremental(&stack, &mut ctx).unwrap();
        assert_eq!(out.get(0, 0), 10.0);
        assert!(!eval.cache.is_dirty(baked_id));
    }

    #[test]
    fn deterministic_noise_layer() {
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "N",
            LayerKind::NoiseValue(NoiseParams {
                seed: 123,
                frequency: 0.05,
                amplitude: 10.0,
                octaves: 1,
                ..NoiseParams::default()
            }),
        ));
        let metrics = HeightfieldMetrics::new(32, 32, 128.0, 128.0);
        let mut eval = StackEvaluator::new();
        let mut ctx = EvalContext::new(metrics);
        let a = eval.rebuild_all(&stack, &mut ctx).unwrap().to_dense();
        let mut eval2 = StackEvaluator::new();
        let mut ctx2 = EvalContext::new(metrics);
        let b = eval2.rebuild_all(&stack, &mut ctx2).unwrap().to_dense();
        assert_eq!(a, b);
    }

    #[test]
    fn add_blend_merges_with_base() {
        use crate::layer::BlendMode;
        let metrics = HeightfieldMetrics::new(16, 16, 64.0, 64.0);
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "Base",
            LayerKind::Flat(FlatParams { height: 10.0 }),
        ));
        let noise = Layer::new(
            "Hills",
            LayerKind::NoiseValue(NoiseParams {
                seed: 7,
                frequency: 0.08,
                amplitude: 5.0,
                octaves: 1,
                ..NoiseParams::default()
            }),
        );
        assert_eq!(noise.common.blend, BlendMode::Add);
        stack.push(noise);

        // Noise-only for comparison
        let mut noise_only = LayerStack::new();
        let mut n = Layer::new(
            "Hills",
            LayerKind::NoiseValue(NoiseParams {
                seed: 7,
                frequency: 0.08,
                amplitude: 5.0,
                octaves: 1,
                ..NoiseParams::default()
            }),
        );
        n.common.blend = BlendMode::Normal;
        noise_only.push(n);

        let mut eval = StackEvaluator::new();
        let mut ctx = EvalContext::new(metrics);
        let merged = eval.rebuild_all(&stack, &mut ctx).unwrap();

        let mut eval2 = StackEvaluator::new();
        let mut ctx2 = EvalContext::new(metrics);
        let only = eval2.rebuild_all(&noise_only, &mut ctx2).unwrap();

        let sample = merged.get(8, 8);
        let noise_sample = only.get(8, 8);
        assert!(
            (sample - (10.0 + noise_sample)).abs() < 1e-3,
            "expected base+noise {sample} vs {}",
            10.0 + noise_sample
        );
        assert!(
            (sample - noise_sample).abs() > 1.0,
            "merged should not equal noise alone"
        );
    }

    #[test]
    fn normal_blend_replaces_base() {
        use crate::layer::BlendMode;
        let metrics = HeightfieldMetrics::new(16, 16, 64.0, 64.0);
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "Base",
            LayerKind::Flat(FlatParams { height: 10.0 }),
        ));
        let mut noise = Layer::new(
            "Hills",
            LayerKind::NoiseValue(NoiseParams {
                seed: 7,
                frequency: 0.08,
                amplitude: 5.0,
                octaves: 1,
                ..NoiseParams::default()
            }),
        );
        noise.common.blend = BlendMode::Normal;
        stack.push(noise);

        let mut noise_only = LayerStack::new();
        let mut n = Layer::new(
            "Hills",
            LayerKind::NoiseValue(NoiseParams {
                seed: 7,
                frequency: 0.08,
                amplitude: 5.0,
                octaves: 1,
                ..NoiseParams::default()
            }),
        );
        n.common.blend = BlendMode::Normal;
        noise_only.push(n);

        let mut eval = StackEvaluator::new();
        let mut ctx = EvalContext::new(metrics);
        let replaced = eval.rebuild_all(&stack, &mut ctx).unwrap();
        let mut eval2 = StackEvaluator::new();
        let mut ctx2 = EvalContext::new(metrics);
        let only = eval2.rebuild_all(&noise_only, &mut ctx2).unwrap();

        assert!((replaced.get(8, 8) - only.get(8, 8)).abs() < 1e-4);
    }

    #[test]
    fn solo_skips_non_solo_layers_in_the_same_stack() {
        let metrics = HeightfieldMetrics::new(8, 8, 80.0, 80.0);
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "Base",
            LayerKind::Flat(FlatParams { height: 100.0 }),
        ));
        let mut solo = Layer::new("Solo", LayerKind::Flat(FlatParams { height: 20.0 }));
        solo.common.blend = BlendMode::Add;
        solo.common.solo = true;
        stack.push(solo);
        let mut after = Layer::new("After", LayerKind::Flat(FlatParams { height: 50.0 }));
        after.common.blend = BlendMode::Add;
        stack.push(after);

        let mut eval = StackEvaluator::new();
        let mut ctx = EvalContext::new(metrics);
        let out = eval.rebuild_all(&stack, &mut ctx).unwrap();
        assert!((out.get(4, 4) - 20.0).abs() < 1.0e-4);
    }

    #[test]
    fn height_mask_is_rebaked_against_the_owning_layers_input() {
        use crate::mask::{bake_mask_assets, MaskAsset, MaskId, MaskRef, MaskSource};

        let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
        let mask_id = MaskId::new();
        let asset = MaskAsset::new(
            mask_id,
            "High ground",
            MaskSource::Height {
                min: 50.0,
                max: 100.0,
            },
        );

        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "Base",
            LayerKind::Flat(FlatParams { height: 100.0 }),
        ));
        let mut raise = Layer::new(
            "Raise high ground",
            LayerKind::Flat(FlatParams { height: 25.0 }),
        );
        raise.common.blend = BlendMode::Add;
        raise.common.masks.push(MaskRef::new(mask_id));
        stack.push(raise);

        let mut ctx = EvalContext::new(metrics);
        // Simulate a cold export / stale preview bake. The zero-height reference
        // produces an empty mask, but evaluation must replace it at point of use.
        ctx.masks = bake_mask_assets(
            std::slice::from_ref(&asset),
            &Heightfield::zeros(metrics),
            metrics,
            &HashMap::new(),
        );
        ctx.mask_assets.push(asset);

        let mut eval = StackEvaluator::new();
        let out = eval.rebuild_all(&stack, &mut ctx).unwrap();
        assert!((out.get(8, 8) - 125.0).abs() < 1.0e-4);
    }

    #[test]
    fn scoped_group_mask_limits_child_normal_filter() {
        use crate::layer::LayerGroup;
        use crate::mask::{bake_mask_assets, MaskAsset, MaskId, MaskRef, MaskSource};

        let metrics = HeightfieldMetrics::new(32, 32, 320.0, 320.0);
        let mask_id = MaskId::new();
        let asset = MaskAsset {
            id: mask_id,
            name: "Right".into(),
            source: MaskSource::Height {
                min: 0.0,
                max: 50.0,
            },
            ops: Vec::new(),
            paint: None,
            display_color: crate::mask::default_mask_display_color(),
        };
        let mut reference = Heightfield::zeros(metrics);
        for j in 0..32 {
            for i in 0..32 {
                reference.set(i, j, if i >= 16 { 100.0 } else { 0.0 });
            }
        }

        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "Base",
            LayerKind::Flat(FlatParams { height: 10.0 }),
        ));
        let mut group = LayerGroup::new("Scoped");
        group.masks.push(MaskRef::new(mask_id));
        group.children.push(StackNode::Layer(Layer::new(
            "Raise",
            LayerKind::Flat(FlatParams { height: 80.0 }),
        )));
        stack.push_group(group);

        let mut eval = StackEvaluator::new();
        let mut ctx = EvalContext::new(metrics);
        ctx.masks = bake_mask_assets(&[asset], &reference, metrics, &HashMap::new());
        let out = eval.rebuild_all(&stack, &mut ctx).unwrap();
        assert!(out.get(24, 16) > 60.0, "inside group mask should raise");
        assert!(
            (out.get(8, 16) - 10.0).abs() < 1e-3,
            "outside group mask should keep base"
        );
    }

    fn push_flat_to_biome_filters(biome: &mut crate::layer::LayerGroup, height: f32) -> LayerId {
        use crate::layer::{BiomeSection, FlatParams, LayerKind};
        biome.ensure_biome_sections();
        let layer = Layer::new("Flat", LayerKind::Flat(FlatParams { height }));
        let id = layer.id();
        if let Some(sec) = biome.find_section_mut(BiomeSection::Filters) {
            sec.children.push(StackNode::Layer(layer));
        } else {
            biome.children.push(StackNode::Layer(layer));
        }
        id
    }

    #[test]
    fn incremental_scoped_groups_reuse_clean_sibling_biome() {
        use crate::layer::LayerGroup;

        let metrics = HeightfieldMetrics::new(16, 16, 160.0, 160.0);
        let mut stack = LayerStack::new();
        stack.push(Layer::new(
            "Base",
            LayerKind::Flat(FlatParams { height: 0.0 }),
        ));

        let mut biome_a = LayerGroup::biome("Alpine");
        let layer_a = push_flat_to_biome_filters(&mut biome_a, 10.0);
        let biome_a_id = biome_a.id;

        let mut biome_b = LayerGroup::biome("Desert");
        let layer_b = push_flat_to_biome_filters(&mut biome_b, 20.0);

        stack.push_group(biome_a);
        stack.push_group(biome_b);

        let mut eval = StackEvaluator::new();
        let mut ctx = EvalContext::new(metrics);
        let _ = eval.rebuild_all(&stack, &mut ctx).unwrap();
        assert!(
            eval.cache.get(biome_a_id).is_some(),
            "scoped group output should be cached"
        );

        eval.cache.mark_dirty(layer_a);
        let mut ctx = EvalContext::new(metrics);
        let _ = eval.rebuild_incremental(&stack, &mut ctx).unwrap();

        let b_timing = ctx
            .layer_timings
            .iter()
            .find(|t| t.layer == layer_b)
            .expect("biome B layer should appear in timings");
        assert_eq!(
            b_timing.status,
            LayerEvalStatus::CacheHit,
            "clean sibling biome layer must not recompute"
        );
        assert!(
            !eval.cache.is_dirty(layer_b),
            "sibling layer cache should stay clean"
        );
    }
}
