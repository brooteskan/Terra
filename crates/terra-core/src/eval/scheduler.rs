use super::StackEvaluator;
use crate::heightfield::Heightfield;
use crate::mask::MaskField;
use crate::quality::PreviewQuality;
use std::collections::HashMap;
use std::sync::Arc;

/// Per-session evaluation state for the interactive app.
///
/// Despite the name, this is *not* an orchestrator: it schedules no work of its
/// own. It is the state the interactive eval paths read and write between
/// frames — the rebuild token mint, the progressive [`PreviewQuality`] ladder,
/// the last-good composed DEM plus its aux/strata/timing side-channels, and the
/// UI-thread [`StackEvaluator`] whose dirty-state bookkeeping the interactive
/// paths maintain across edits (`mark_dirty_from` / `mark_all_dirty`) and whose
/// layer cache the GPU bridge-prefix and ingest paths read opportunistically.
///
/// No stack eval runs on the UI thread: every full-stack composed-height rebuild
/// runs off-thread on the background [`EvalWorker`](super::EvalWorker), which
/// owns its own `StackEvaluator`. The app is the integrator that ties these
/// together; this struct just holds the shared eval-session state.
pub struct EvalScheduler {
    pub evaluator: StackEvaluator,
    pub current_token: u64,
    pub last_good: Option<Arc<Heightfield>>,
    pub last_aux: HashMap<String, MaskField>,
    /// Materials strata preserved across HashMap aux round-trips.
    pub last_strata: Option<Vec<crate::layer::Stratum>>,
    /// Most recent per-layer CPU timing and cache provenance.
    pub last_layer_timings: Vec<super::LayerEvalTiming>,
    pub quality: PreviewQuality,
}

impl Default for EvalScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl EvalScheduler {
    pub fn new() -> Self {
        // The UI-thread evaluator's baked checkpoints have no disk reader:
        // nothing on the UI thread reloads spills, and lifecycle GPU ingest reads
        // in-memory checkpoints. Spilling here only churned the shared cache
        // directory (deleting/overwriting the worker's bakes), so this evaluator
        // runs memory-only (B1-D8).
        let mut evaluator = StackEvaluator::new();
        evaluator.cache.disable_disk();
        Self {
            evaluator,
            current_token: 0,
            last_good: None,
            last_aux: HashMap::new(),
            last_strata: None,
            last_layer_timings: Vec::new(),
            quality: PreviewQuality::Draft,
        }
    }

    pub fn request_rebuild(&mut self) -> u64 {
        self.current_token = self.current_token.wrapping_add(1);
        self.quality = PreviewQuality::Draft;
        self.current_token
    }

    pub fn advance_quality(&mut self) -> bool {
        if let Some(next) = self.quality.next_refine() {
            self.quality = next;
            true
        } else {
            false
        }
    }
}
