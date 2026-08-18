// Interactive SculptStrokes preview — Flatten footprint-mean resolve pass (#117).
//
// Folds the per-workgroup `(h*w, w)` partials from the reduce pass into a single
// scalar `targets[target_index] = sum(h*w) / sum(w)`. Dispatched as one workgroup;
// each thread strides over the partials, then a tree reduction combines them. When
// the footprint carried no weight the target degenerates to `fallback` (the stroke's
// `target_height`), matching the CPU `flatten_target_for` `wsum == 0` branch.

struct Uniforms {
    num_partials: u32,
    target_index: u32,
    fallback: f32,
    _p0: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> partials: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> targets: array<f32>;

const GROUP: u32 = 256u;
var<workgroup> sdata: array<vec2<f32>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lid: u32) {
    var acc = vec2<f32>(0.0, 0.0);
    var i = lid;
    loop {
        if (i >= u.num_partials) { break; }
        acc = acc + partials[i];
        i = i + GROUP;
    }

    sdata[lid] = acc;
    workgroupBarrier();
    var stride = GROUP >> 1u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) {
            sdata[lid] = sdata[lid] + sdata[lid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid == 0u) {
        let sum = sdata[0];
        var mean = u.fallback;
        if (sum.y > 0.0) {
            mean = sum.x / sum.y;
        }
        targets[u.target_index] = mean;
    }
}
