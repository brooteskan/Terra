//! Repro + timing for issue #98: Full CPU eval of a stack that contains a
//! GPU-unsupported generator (`VoronoiRegions`) plus a Flatten Shape layer.
//!
//! Usage: `cargo run [--release] --example voronoi_flatten_full_eval -- [max_res]`
//!
//! Neither `VoronoiRegions` nor `SculptStrokes` has a GPU kernel, so the
//! interactive Full refine of this stack cannot stay on the GPU — it resumes on
//! the CPU. The point of this binary is to establish that that CPU eval is
//! *finite* and scales O(pixels): it is a long synchronous computation, not an
//! infinite loop. The reported wall time is what the background eval worker
//! spends per Full build (and what a mid-layer cancel must be able to interrupt).
//!
//! `max_res` (optional) caps the resolution sweep so a debug run stays quick;
//! the default sweep is 256/512/1024/2048 to match the table in the issue.

use std::collections::HashMap;
use std::time::Instant;

use terra_core::eval::{EvalContext, PreviewQuality, StackEvaluator};
use terra_core::generators::VoronoiParams;
use terra_core::heightfield::{Heightfield, HeightfieldMetrics};
use terra_core::layer::{Layer, LayerKind, LayerStack};
use terra_core::mask::bake_mask_assets;
use terra_core::shape_history::{create_shape_layer, stamp_stroke, ShapeTool};

fn eval_at(stack: &LayerStack, m: HeightfieldMetrics, q: PreviewQuality) {
    let mut ctx = EvalContext::new(m);
    ctx.quality = q;
    let seed = Heightfield::zeros(m);
    ctx.masks = bake_mask_assets(&[], &seed, m, &HashMap::new());
    let mut eval = StackEvaluator::new();
    let t = Instant::now();
    let _ = eval.rebuild_all(stack, &mut ctx).expect("eval");
    eprintln!(
        "  eval {:?} @ {}x{} done in {:?}",
        q,
        m.width,
        m.height,
        t.elapsed()
    );
}

fn main() {
    let max_res: u32 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(2048);

    let mut stack = LayerStack::new();
    stack.push(Layer::new(
        "Voronoi",
        LayerKind::VoronoiRegions(VoronoiParams::default()),
    ));
    // create_shape_layer already carries default SculptStrokeParams; stamp one
    // Flatten stroke at the centre so the Shape layer contributes real work.
    let mut flat = create_shape_layer("Flatten");
    if let LayerKind::SculptStrokes(p) = &mut flat.kind {
        stamp_stroke(
            p,
            ShapeTool::Flatten.stroke_kind(),
            0.5,
            0.5,
            200.0,
            4.0,
            0.0,
            false,
        );
    }
    stack.push(flat);

    for res in [256u32, 512, 1024, 2048] {
        if res > max_res {
            break;
        }
        let m = HeightfieldMetrics::new(res, res, 4096.0, 4096.0);
        eval_at(&stack, m, PreviewQuality::Full);
    }
}
