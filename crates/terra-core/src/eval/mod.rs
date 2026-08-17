//! Layer evaluation, caching, and dirty propagation.

mod cache;
mod processors;
pub mod reach;
mod scheduler;
mod smart_cache;
mod worker;

pub use crate::quality::PreviewQuality;
pub use cache::{CachedOutput, LayerCache, SeedState};
pub use processors::ProcessorRegistry;
pub use scheduler::EvalScheduler;
pub use smart_cache::DiskSmartCache;
pub use worker::{
    EvalWorkFailure, EvalWorkRequest, EvalWorkResult, EvalWorker, EvalWorkerEvent,
    EvalWorkerSubmitError,
};

use crate::field_data::AuxMaps;
use crate::heightfield::{Heightfield, HeightfieldMetrics, TileId};
use crate::layer::{blend_heights, Layer, LayerId, LayerStack, StackNode};
use crate::mask::{MaskAsset, MaskField, MaskId};
use crate::tiling::TileScheduler;
use std::collections::{HashMap, HashSet};
use std::sync::{atomic::AtomicU64, Arc};
use std::time::Instant;
use terra_jobs::CancelToken;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EvalError {
    #[error("unknown layer processor for kind")]
    UnknownProcessor,
    #[error("cancelled")]
    Cancelled,
    #[error("io: {0}")]
    Io(String),
    #[error("layer \"{layer}\" panicked: {message}")]
    LayerPanicked { layer: String, message: String },
    #[error("evaluation panicked: {0}")]
    Panicked(String),
}

impl EvalError {
    pub fn layer_name(&self) -> Option<&str> {
        match self {
            Self::LayerPanicked { layer, .. } => Some(layer),
            _ => None,
        }
    }
}

/// Source-import failures from `generators` surface to evaluator callers as
/// `Io`, keeping the public `EvalError` surface unchanged. The path/cause detail
/// carried by `SourceImportError` is preserved in the message.
impl From<crate::generators::SourceImportError> for EvalError {
    fn from(err: crate::generators::SourceImportError) -> Self {
        Self::Io(err.to_string())
    }
}

fn panic_payload_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".into()
    }
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
    /// Tiles recomputed when this layer ran tile-scoped (#100 phase 2). `None`
    /// means whole-field, a cache hit, or disabled; `Some(n)` is the size of the
    /// expanded dirty set the scoped recompute actually touched. The recompute
    /// counter the phase-2 equivalence oracle asserts against.
    pub tiles_recomputed: Option<u32>,
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
    /// Cooperative cancel signal. Checked between layers and, since #101, inside
    /// the `fill_world` generators so a superseding edit interrupts a long fill
    /// mid-flight instead of waiting it out.
    pub(crate) cancel: CancelToken,
    pub quality: PreviewQuality,
    /// Timings for the current pass, in actual layer evaluation order.
    pub layer_timings: Vec<LayerEvalTiming>,
    /// Initial tile-scoped recompute region for the flat suffix walk (#100 phase
    /// 2). `None` (the default) recomputes the whole field, exactly as before.
    /// `Some(tiles)` seeds the cumulative dirty set so a localized edit recomputes
    /// only the touched tiles plus each downstream layer's reach. Driven directly
    /// by headless tests this phase; `run_cpu_job` threads it in phase 4.
    pub initial_scope: Option<Vec<TileId>>,
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
            cancel: CancelToken::never(),
            layer_timings: Vec::new(),
            initial_scope: None,
        }
    }

    pub fn check_cancelled(&self) -> Result<(), EvalError> {
        if self.cancelled || self.cancel.is_cancelled() {
            Err(EvalError::Cancelled)
        } else {
            Ok(())
        }
    }

    /// Install a generation-based cancel token, as the eval worker does when it
    /// runs a CPU job. Kept as a distinct helper so `worker.rs` need not name
    /// [`CancelToken`] directly.
    pub fn set_cancellation_generation(&mut self, generation: Arc<AtomicU64>, expected: u64) {
        self.cancel = CancelToken::generation(generation, expected);
    }

    /// Install an arbitrary cancel token (e.g. the one-shot flag the phase-2
    /// export executors carry).
    pub fn set_cancel_token(&mut self, token: CancelToken) {
        self.cancel = token;
    }

    /// A cheap clone of the current cancel token for handing to generators.
    ///
    /// Note this does not fold in the plain [`cancelled`](Self::cancelled) flag,
    /// which is set before eval begins and stays covered by the between-layer
    /// [`check_cancelled`](Self::check_cancelled).
    pub fn cancel_token(&self) -> CancelToken {
        self.cancel.clone()
    }

    /// Insert an aux map into both typed and string stores.
    pub fn aux_insert(&mut self, key: impl Into<String>, field: MaskField) {
        let key = key.into();
        let canonical = crate::field_data::keys::canonical(&key).to_string();
        self.aux_maps.insert(canonical.clone(), field.clone());
        if canonical == crate::field_data::keys::SEDIMENT_THICKNESS {
            self.aux.remove(crate::field_data::keys::SEDIMENT_DEPTH);
            self.aux.remove(crate::field_data::keys::LOOSE_SEDIMENT);
        }
        self.aux.insert(canonical, field);
    }

    /// Replace string aux and rebuild typed maps (worker / scheduler ingest).
    /// Preserves any strata already on `aux_maps` when the HashMap has none.
    pub fn set_aux_hashmap(&mut self, aux: HashMap<String, MaskField>) {
        let aux = aux
            .into_iter()
            .map(|(key, field)| (key, field.into_resampled_nearest(self.metrics)))
            .collect::<HashMap<_, _>>();
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

    /// Tile-scoped dirty from a layer (#100 phase 2): mark `id` and its suffix
    /// dirty, but with a bounded seed region instead of the whole field.
    ///
    /// The edited layer `id` seeds with the actual edit region `tiles`. Layers
    /// *above* it have no own-contribution change — only their input moves — so
    /// they seed empty (dirty, input-driven only). The tile-scoped suffix walk in
    /// [`Self::rebuild_incremental`] accumulates and reach-expands these seeds; an
    /// unknown id falls back to whole-field [`Self::mark_dirty_from`] semantics.
    pub fn mark_dirty_from_region(&mut self, stack: &LayerStack, id: LayerId, tiles: &[TileId]) {
        let ids = stack.layer_ids();
        if let Some(start) = ids.iter().position(|&x| x == id) {
            self.cache.mark_dirty_region(id, tiles);
            for &dep in &ids[start + 1..] {
                self.cache.mark_dirty_region(dep, &[]);
            }
        } else {
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

        let metrics = ctx.metrics;
        let first_dirty = layers.iter().position(|l| {
            if l.common.cached {
                // has_clean answers from the entry state (or a header-only disk
                // probe) without inflating a spilled checkpoint into memory.
                !self.cache.has_clean(l.id(), metrics)
            } else {
                self.cache.is_dirty(l.id())
            }
        });

        // All clean: reuse the cached top, reloading it from a spill if the pinned
        // top was reclaimed from memory. `get_or_load` returns `Some` only for a
        // clean, dimensionally matching checkpoint.
        if first_dirty.is_none() {
            if let Some(top) = self.cache.get_or_load(layers.last().unwrap().id(), metrics) {
                let height = top.height.clone();
                for layer in &layers {
                    record_reused_layer(ctx, layer);
                }
                return Ok(height);
            }
        }

        let mut first_dirty = first_dirty.unwrap_or(0);
        let mut fell_back = false;
        let mut current = if first_dirty == 0 {
            Heightfield::zeros(metrics)
        } else {
            let prev_id = layers[first_dirty - 1].id();
            match self
                .cache
                .get_or_load(prev_id, metrics)
                .map(|c| c.height.clone())
            {
                Some(height) => height,
                None => {
                    // The seed checkpoint for the clean prefix could not be
                    // reloaded (its spill was removed or failed validation).
                    // Zero-seeding would silently truncate the stack, so fall back
                    // to a full rebuild from the base instead (B1-D8). The cache is
                    // incoherent here, so force whole-field — never tile-scoped.
                    first_dirty = 0;
                    fell_back = true;
                    Heightfield::zeros(metrics)
                }
            }
        };
        for layer in &layers[..first_dirty] {
            record_reused_layer(ctx, layer);
        }

        // Tile-scoped suffix accumulation (#100 phase 2). `cum` is the cumulative
        // set of tiles whose composed height differs from the previous cached
        // composite at the current level; `None` means we escalated to whole-field
        // and stay there (sticky) for the rest of the walk — reproducing today's
        // behavior exactly. It seeds from the caller's `initial_scope`.
        let tile_count = metrics.tile_count();
        let mut cum: Option<HashSet<TileId>> = if fell_back {
            None
        } else {
            Some(ctx.initial_scope.iter().flatten().copied().collect())
        };

        for layer in &layers[first_dirty..] {
            ctx.check_cancelled()?;

            let scope = self.plan_layer_scope(ctx, &current, layer, &mut cum, tile_count);
            current = match scope {
                Some(tiles) => self.evaluate_layer_scoped(ctx, &current, layer, &tiles)?,
                None => self.evaluate_layer(ctx, &current, layer)?,
            };
            self.store_cached(layer.id(), &current, ctx, layer.common.cached);
        }

        Ok(current)
    }

    /// Decide how the next suffix layer recomputes: `Some(tiles)` for a tile-scoped
    /// recompute over `tiles`, or `None` for a whole-field / cache-hit / escalated
    /// recompute. Mutates `cum` in place — folding in the layer's own seeds and
    /// reach expansion, or clearing it to `None` on escalation (sticky).
    fn plan_layer_scope(
        &self,
        ctx: &EvalContext,
        current: &Heightfield,
        layer: &Layer,
        cum: &mut Option<HashSet<TileId>>,
        tile_count: u32,
    ) -> Option<Vec<TileId>> {
        // Already whole-field (sticky), or a clean layer whose cached output is
        // unchanged (its downstream difference set is unchanged): recompute the
        // normal way and leave `cum` untouched.
        if cum.is_none() || !self.cache.is_dirty(layer.id()) {
            return None;
        }

        // Disabled layers are an identity passthrough: no reach, no escalation (a
        // disabled basin-coupled layer must not force whole-field). Fold any own
        // seeds into `cum` — a no-op superset at worst — and pass through.
        if !layer.common.enabled {
            if let SeedState::Tiles(seeds) = self.cache.seed_state(layer.id()) {
                cum.as_mut().unwrap().extend(seeds);
            }
            return None;
        }

        let reach = reach::effective_reach(layer, &ctx.mask_assets);
        let prev_dims_ok = self.cache.get(layer.id()).is_some_and(|c| {
            c.height.metrics.width == current.metrics.width
                && c.height.metrics.height == current.metrics.height
        });

        match self.cache.seed_state(layer.id()) {
            // Bounded own-seed, a bounded reach, and a dimensionally usable previous
            // output: union the seeds, reach-expand the whole cumulative set, and
            // recompute exactly those tiles — unless expansion filled the field.
            SeedState::Tiles(seeds) if !reach.is_full() && prev_dims_ok => {
                let mut set = cum.take().unwrap();
                set.extend(seeds);
                let mut sched = TileScheduler {
                    dirty: set.into_iter().collect(),
                };
                sched.expand_for_reach(current, reach);
                let expanded: HashSet<TileId> = sched.dirty.into_iter().collect();
                if expanded.len() as u32 >= tile_count {
                    *cum = None;
                    None
                } else {
                    let scope = expanded.iter().copied().collect();
                    *cum = Some(expanded);
                    Some(scope)
                }
            }
            // Whole-field seed (AllTiles), a full reach, or a previous output that
            // cannot be reused (absent / wrong dims): escalate to whole-field, and
            // every layer above inherits it (sticky).
            _ => {
                *cum = None;
                None
            }
        }
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
        child_aux: &crate::field_data::AuxMaps,
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
        let layer_name = layer.common.name.clone();
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.evaluate_layer_inner(ctx, input, layer)
        })) {
            Ok(result) => result,
            Err(payload) => Err(EvalError::LayerPanicked {
                layer: layer_name,
                message: panic_payload_message(payload),
            }),
        }
    }

    fn evaluate_layer_inner(
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

    /// Tile-scoped recompute of one layer (#100 phase 2), panic-contained exactly
    /// like [`Self::evaluate_layer`]. Caller (`plan_layer_scope`) guarantees the
    /// preconditions: the layer is enabled and dirty, its reach is bounded, and a
    /// dimensionally-matching previous output exists to carry clean tiles from.
    fn evaluate_layer_scoped(
        &mut self,
        ctx: &mut EvalContext,
        input: &Heightfield,
        layer: &Layer,
        scope: &[TileId],
    ) -> Result<Heightfield, EvalError> {
        let layer_name = layer.common.name.clone();
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.evaluate_layer_scoped_inner(ctx, input, layer, scope)
        })) {
            Ok(result) => result,
            Err(payload) => Err(EvalError::LayerPanicked {
                layer: layer_name,
                message: panic_payload_message(payload),
            }),
        }
    }

    fn evaluate_layer_scoped_inner(
        &mut self,
        ctx: &mut EvalContext,
        input: &Heightfield,
        layer: &Layer,
        scope: &[TileId],
    ) -> Result<Heightfield, EvalError> {
        let timing_started = Instant::now();
        let metrics = ctx.metrics;

        // Seed the output from the previous clean composite; recomputed tiles
        // overwrite it, clean tiles are carried forward unchanged. If the previous
        // output is somehow gone, we cannot carry clean tiles — recompute whole.
        let Some(prev) = self.cache.get(layer.id()).map(|c| c.height.clone()) else {
            return self.evaluate_layer_inner(ctx, input, layer);
        };

        // Point-of-use masks rebake against the exact field entering this layer.
        refresh_point_of_use_masks(ctx, input);

        let scaled_layer = layer_with_world_scale(layer, ctx.level_steps.world_scale);
        let bound_layer = apply_param_bindings(ctx, &scaled_layer);
        let generated = self.generate_scoped(ctx, input, &bound_layer, scope)?;
        let mask = effective_layer_mask(ctx, &layer.common.masks, input);

        let mut out = prev;
        for &id in scope {
            let (ox, oz, iw, ih) = {
                let Some(tile) = out.tile(id) else {
                    continue;
                };
                let (ox, oz) = tile.interior_origin(&metrics);
                (ox, oz, tile.interior_width, tile.interior_height)
            };
            for lz in 0..ih {
                for lx in 0..iw {
                    let i = ox + lx;
                    let j = oz + lz;
                    let hin = input.get(i, j);
                    let hlayer = generated.get(i, j);
                    let m = mask.get(i, j);
                    let v = blend_heights(
                        layer.common.blend,
                        hin,
                        hlayer,
                        layer.common.opacity,
                        m,
                    );
                    out.set(i, j, v);
                }
            }
        }

        // Refresh only the ghosts of the recomputed tiles' 8-neighbour ring. A
        // full `refresh_halos()` would be correct but whole-field; this is the
        // scoped equivalent and its seam metric must read zero.
        let seam = TileScheduler {
            dirty: scope.to_vec(),
        }
        .sync_dirty(&mut out);
        debug_assert!(
            seam.abs() < 1.0e-3,
            "scoped halo refresh left a seam of {seam}"
        );

        publish_layer_outputs(ctx, layer, &out);
        record_scoped_layer_timing(ctx, layer, timing_started, scope.len() as u32);
        Ok(out)
    }

    /// Produce a height field whose `scope` tiles hold this layer's freshly
    /// generated contribution. Values outside `scope` are never read by the
    /// caller's blend, so only the scope tiles need be correct. Arms without a
    /// tile-sliced entry fall back to a whole-field generate (correct, no win).
    fn generate_scoped(
        &self,
        ctx: &mut EvalContext,
        input: &Heightfield,
        layer: &Layer,
        scope: &[TileId],
    ) -> Result<Heightfield, EvalError> {
        use crate::layer::LayerKind;
        let metrics = ctx.metrics;
        match &layer.kind {
            LayerKind::SculptBase(p) => {
                let mut g = Heightfield::zeros(metrics);
                for &id in scope {
                    if let Some(dst) = g.tile_mut(id) {
                        *dst = crate::generators::sculpt_base_tile(metrics, p, id);
                    }
                }
                Ok(g)
            }
            LayerKind::PolygonHeight(p) => {
                let mut g = input.clone();
                for &id in scope {
                    if let Some(tile) = crate::generators::polygon_height_tile(input, p, id) {
                        if let Some(dst) = g.tile_mut(id) {
                            *dst = tile;
                        }
                    }
                }
                Ok(g)
            }
            LayerKind::Plateau(p) => {
                let mut g = input.clone();
                for &id in scope {
                    if let Some(dst) = g.tile_mut(id) {
                        dst.map_interior(|h| crate::generators::plateau_sample(p, h));
                    }
                }
                Ok(g)
            }
            LayerKind::Coastal(p) => {
                let mut g = input.clone();
                for &id in scope {
                    if let Some(dst) = g.tile_mut(id) {
                        dst.map_interior(|h| crate::generators::coastal_sample(p, h));
                    }
                }
                Ok(g)
            }
            LayerKind::Path(p) => self.generate_scoped_path(ctx, input, p, layer.id(), scope),
            // Not yet tile-wired (input-independent generators, etc.): whole-field
            // generate is still correct because the caller blends only `scope`.
            _ => self.registry.evaluate(ctx, input, layer),
        }
    }

    /// Scoped Path stamp: patches height per tile and re-maxes the per-texel
    /// wetness aux over just the recomputed tiles, seeding clean tiles from the
    /// previous cached merge — the tile-scoped equivalent of `merge_wetness_max`.
    fn generate_scoped_path(
        &self,
        ctx: &mut EvalContext,
        input: &Heightfield,
        p: &crate::layer::PathParams,
        layer_id: LayerId,
        scope: &[TileId],
    ) -> Result<Heightfield, EvalError> {
        use crate::field_data::keys;
        let metrics = ctx.metrics;
        let mut g = input.clone();
        // Wetness entering from below (already in ctx), and the previous merged
        // wetness (correct on clean tiles). Re-max the fresh stamp over `below` on
        // scope tiles only; clean tiles keep the previous merge.
        let below = ctx.aux_maps.wetness.clone();
        let mut wet = self
            .cache
            .get(layer_id)
            .and_then(|c| c.aux.get(keys::WETNESS).cloned())
            .unwrap_or_else(|| below.clone().unwrap_or_else(|| MaskField::zeros(metrics)));
        for &id in scope {
            let Some(crate::generators::PathStampTile { height, wetness }) =
                crate::generators::path_stamp_tile(input, p, id)
            else {
                continue;
            };
            let (ox, oz) = height.interior_origin(&metrics);
            let iw = height.interior_width;
            let ih = height.interior_height;
            for lz in 0..ih {
                for lx in 0..iw {
                    let stamp = wetness[(lz * iw + lx) as usize];
                    let base = below
                        .as_ref()
                        .map(|b| b.get(ox + lx, oz + lz))
                        .unwrap_or(0.0);
                    wet.set(ox + lx, oz + lz, base.max(stamp));
                }
            }
            if let Some(dst) = g.tile_mut(id) {
                *dst = height;
            }
        }
        ctx.aux_insert(keys::WETNESS, wet);
        Ok(g)
    }

    fn try_reuse_group_cache(
        &mut self,
        group_id: LayerId,
        ctx: &EvalContext,
        descendant_ids: &[LayerId],
        input: &Heightfield,
    ) -> Option<(Heightfield, crate::field_data::AuxMaps)> {
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
        let child_aux = crate::field_data::AuxMaps::from_hashmap_preserving_strata(
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
        tiles_recomputed: None,
    });
}

/// Timing for a layer that ran tile-scoped, tagged with the size of the tile set
/// it recomputed (the phase-2 recompute counter).
fn record_scoped_layer_timing(
    ctx: &mut EvalContext,
    layer: &Layer,
    started: Instant,
    tiles: u32,
) {
    ctx.layer_timings.push(LayerEvalTiming {
        layer: layer.id(),
        layer_name: layer.common.name.clone(),
        layer_kind: layer.kind.type_display_name(),
        elapsed_us: started.elapsed().as_micros() as u64,
        status: LayerEvalStatus::Computed,
        tiles_recomputed: Some(tiles),
    });
}

fn record_reused_layer(ctx: &mut EvalContext, layer: &Layer) {
    ctx.layer_timings.push(LayerEvalTiming {
        layer: layer.id(),
        layer_name: layer.common.name.clone(),
        layer_kind: layer.kind.type_display_name(),
        elapsed_us: 0,
        status: LayerEvalStatus::CacheHit,
        tiles_recomputed: None,
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
    let step = (w.max(h) / 64).max(1);
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
        let field = if output.field == crate::field_data::FieldId::Height {
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

/// Borrow `mask` unchanged when its grid matches `target`; nearest-resample otherwise.
///
/// Cross-field samplers must not raw-index a mask baked on a different grid (#94).
/// The equal-dimensions path only pays a compare and borrows; a mismatch degrades
/// gracefully to a world-aligned resample instead of an out-of-bounds panic.
fn mask_at(mask: &MaskField, target: HeightfieldMetrics) -> std::borrow::Cow<'_, MaskField> {
    if mask.metrics.width == target.width && mask.metrics.height == target.height {
        std::borrow::Cow::Borrowed(mask)
    } else {
        log::warn!(
            "mask grid {}x{} does not match target {}x{}; resampling defensively",
            mask.metrics.width,
            mask.metrics.height,
            target.width,
            target.height
        );
        std::borrow::Cow::Owned(mask.resampled_nearest(target))
    }
}

/// Merge child aux maps into the parent context, weighted by the group mask.
fn merge_aux_masked(
    ctx: &mut EvalContext,
    child: &crate::field_data::AuxMaps,
    mask: &MaskField,
    opacity: f32,
) {
    // Normalize every foreign-grid field to `ctx.metrics` so the inner loop
    // cannot raw-index a mask, child, or parent aux baked on a different grid (#94).
    let mask = mask_at(mask, ctx.metrics);
    let child_map = child.to_hashmap();
    for (key, child_field) in child_map {
        let child_field = child_field.into_resampled_nearest(ctx.metrics);
        let mut out = ctx
            .aux_maps
            .get(&key)
            .cloned()
            .unwrap_or_else(|| MaskField::zeros(ctx.metrics))
            .into_resampled_nearest(ctx.metrics);
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
        .wrapping_mul(0x0100_0000_01b3)
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
                .wrapping_mul(0x0100_0000_01b3)
                .wrapping_add(h.get(i, j).to_bits() as u64);
            i += step_i;
        }
        j += step_j;
    }
    state
}

/// Multiply recent materials / vegetation aux fields by a placement mask.
fn gate_aux_by_mask(ctx: &mut EvalContext, mask: &MaskField) {
    use crate::field_data::keys;
    let mul = |field: &mut MaskField| {
        // Match the mask to this field's grid so `get` stays in-range by local
        // construction rather than depending on the producer's metrics (#94).
        let mask = mask_at(mask, field.metrics);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::{BlendMode, FlatParams, LayerKind, NoiseParams};
    use std::collections::HashSet;

    /// Layer ids from `from` to the top of the stack (inclusive). Test-only
    /// mirror of the dirty suffix `mark_dirty_from` propagates over.
    fn dirty_suffix_ids(stack: &LayerStack, from: LayerId) -> HashSet<LayerId> {
        let ids = stack.layer_ids();
        let mut set = HashSet::new();
        if let Some(start) = ids.iter().position(|&x| x == from) {
            for &id in &ids[start..] {
                set.insert(id);
            }
        }
        set
    }

    /// #94: gating an 8x8 aux field with a 4x4 mask must not panic (raw
    /// `mask.get(4, 3)` indexed 16 == len before the clamp/resample) and must
    /// keep the mask world-aligned. Reverting the `mask_at` guard fails this.
    #[test]
    fn gate_aux_by_mask_survives_smaller_mask() {
        use crate::field_data::keys;
        let metrics = HeightfieldMetrics::new(8, 8, 80.0, 80.0);
        let mut ctx = EvalContext::new(metrics);
        ctx.aux_insert(keys::MATERIALS, MaskField::ones(metrics));

        // Half-resolution mask: left half 0, right half 1.
        let mut mask = MaskField::zeros(HeightfieldMetrics::new(4, 4, 80.0, 80.0));
        for j in 0..4 {
            for i in 2..4 {
                mask.set(i, j, 1.0);
            }
        }

        gate_aux_by_mask(&mut ctx, &mask);

        let gated = ctx.aux.get(keys::MATERIALS).expect("materials aux present");
        assert_eq!(gated.metrics.width, 8);
        assert_eq!(gated.get(0, 0), 0.0, "left half gated off");
        assert_eq!(gated.get(7, 7), 1.0, "right half kept, world-aligned");
    }

    /// #94: `merge_aux_masked` raw-indexes three cross-context fields (mask,
    /// child, and the cloned parent aux). A 4x4 for any of them against an 8x8
    /// `ctx.metrics` panicked before the resample; all three are hardened here.
    #[test]
    fn merge_aux_masked_survives_mismatched_mask_child_and_parent_aux() {
        use crate::field_data::keys;
        let metrics = HeightfieldMetrics::new(8, 8, 80.0, 80.0);
        let small = HeightfieldMetrics::new(4, 4, 80.0, 80.0);
        let mut ctx = EvalContext::new(metrics);

        // Foreign-resolution parent aux (the merge target `out`).
        ctx.aux_insert(keys::MATERIALS, MaskField::zeros(small));

        // Foreign-resolution child field.
        let mut child = AuxMaps::default();
        child.materials = Some(MaskField::ones(small));

        // Foreign-resolution mask: left half 0, right half 1.
        let mut mask = MaskField::zeros(small);
        for j in 0..4 {
            for i in 2..4 {
                mask.set(i, j, 1.0);
            }
        }

        merge_aux_masked(&mut ctx, &child, &mask, 1.0);

        let merged = ctx.aux.get(keys::MATERIALS).expect("materials aux present");
        assert_eq!(merged.metrics.width, 8, "renormalized to ctx.metrics");
        assert_eq!(merged.get(0, 0), 0.0, "w=0 keeps parent (zeros)");
        assert_eq!(merged.get(7, 7), 1.0, "w=1 adopts child (ones)");
    }

    #[test]
    fn source_import_error_maps_to_io_keeping_path_and_cause() {
        let src = crate::generators::SourceImportError::Image {
            path: "textures/ridge.png".into(),
            message: "decode failed".into(),
        };
        let err = EvalError::from(src);
        assert!(matches!(err, EvalError::Io(_)));
        let text = err.to_string();
        assert!(text.contains("textures/ridge.png"), "lost the path: {text}");
        assert!(text.contains("decode failed"), "lost the cause: {text}");
    }

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

    /// B1-D8 revert check: a pinned layer's checkpoint survives being reclaimed
    /// from memory (spilled) and is reused via reload when an upper layer edits,
    /// instead of being recomputed.
    #[test]
    fn pinned_layer_survives_reload_through_spill() {
        let dir = std::env::temp_dir().join(format!("terra_pin_reload_{}", uuid::Uuid::new_v4()));
        let metrics = HeightfieldMetrics::new(8, 8, 80.0, 80.0);
        let mut stack = LayerStack::new();
        let mut base = Layer::new("Base", LayerKind::Flat(FlatParams { height: 50.0 }));
        base.common.cached = true;
        let base_id = base.id();
        let mut upper = Layer::new("Upper", LayerKind::Flat(FlatParams { height: 3.0 }));
        upper.common.blend = BlendMode::Add;
        let upper_id = upper.id();
        stack.push(base);
        stack.push(upper);

        let mut eval = StackEvaluator::new();
        eval.cache.enable_disk(&dir);
        eval.mark_all_dirty(&stack);
        let mut ctx = EvalContext::new(metrics);
        let _ = eval.rebuild_incremental(&stack, &mut ctx).unwrap();
        assert!(
            eval.cache.is_spilled(base_id),
            "pinned base should be spilled to disk after bake"
        );

        eval.cache.mark_dirty(upper_id);
        let mut ctx2 = EvalContext::new(metrics);
        let out = eval.rebuild_incremental(&stack, &mut ctx2).unwrap();
        assert_eq!(out.get(0, 0), 53.0);
        let base_timing = ctx2
            .layer_timings
            .iter()
            .find(|t| t.layer == base_id)
            .expect("base appears in timings");
        assert_eq!(
            base_timing.status,
            LayerEvalStatus::CacheHit,
            "pinned base must be reused from its spill, not recomputed"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// B1-D8 revert check: when a pinned seed checkpoint cannot be reloaded, the
    /// incremental rebuild recomputes from the base rather than zero-seeding
    /// (which would silently truncate the stack).
    #[test]
    fn lost_pinned_seed_recomputes_instead_of_zero_seeding() {
        let dir = std::env::temp_dir().join(format!("terra_lost_seed_{}", uuid::Uuid::new_v4()));
        let metrics = HeightfieldMetrics::new(8, 8, 80.0, 80.0);
        let mut stack = LayerStack::new();
        let mut base = Layer::new("Base", LayerKind::Flat(FlatParams { height: 50.0 }));
        base.common.cached = true;
        let base_id = base.id();
        let mut upper = Layer::new("Upper", LayerKind::Flat(FlatParams { height: 3.0 }));
        upper.common.blend = BlendMode::Add;
        let upper_id = upper.id();
        stack.push(base);
        stack.push(upper);

        let mut eval = StackEvaluator::new();
        eval.cache.enable_disk(&dir);
        eval.mark_all_dirty(&stack);
        let mut ctx = EvalContext::new(metrics);
        let _ = eval.rebuild_incremental(&stack, &mut ctx).unwrap();
        assert!(eval.cache.is_spilled(base_id));

        // Delete the pinned base's spill behind the cache's back: its entry still
        // claims clean, but the seed can no longer be reloaded.
        let bake = dir.join(format!("{}.bake", base_id.0));
        std::fs::remove_file(&bake).expect("remove spilled base");

        eval.cache.mark_dirty(upper_id);
        let mut ctx2 = EvalContext::new(metrics);
        let out = eval.rebuild_incremental(&stack, &mut ctx2).unwrap();
        // A zero-seed fallback would drop the base and yield 3.0; a correct full
        // rebuild recomputes the base and yields 53.0.
        assert_eq!(out.get(0, 0), 53.0);
        let _ = std::fs::remove_dir_all(&dir);
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
