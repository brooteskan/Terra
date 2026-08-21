//! CPU adapter for backend-neutral realism benchmark documents.

use terra_core::analyze::TerrainStatistics;
use terra_core::document::TerrainDocument;
use terra_core::heightfield::Heightfield;
use terra_core::quality::PreviewQuality;

use crate::{EvalContext, StackEvaluator};

/// Evaluate a document to a heightfield and collect morphometrics.
pub fn measure_document(doc: &TerrainDocument) -> Result<(Heightfield, TerrainStatistics), String> {
    let mut ctx = EvalContext::new(doc.metrics);
    ctx.quality = PreviewQuality::Draft;
    let mut eval = StackEvaluator::new();
    let hf = eval
        .rebuild_all(&doc.stack, &mut ctx)
        .map_err(|e| format!("benchmark eval failed: {e}"))?;
    let stats = TerrainStatistics::compute(&hf);
    Ok((hf, stats))
}
