// Interactive SculptStrokes preview — reconcile pass (#113).
//
// Relaxes the stamped field toward its 3x3 neighborhood mean, weighted by the
// per-texel brush coverage, exactly as the tail of `apply_sculpt_strokes`:
//
//   a   = clamp(reconcile, 0, 1) * edited * 0.35
//   out = src + (avg3x3(src) - src) * a
//
// It always runs (identity where `edited` or `reconcile` is zero) and writes the
// layer contribution consumed by the standard blend, so there is a single code
// path whether or not reconcile is enabled.

struct Uniforms {
    width: u32,
    height: u32,
    reconcile: f32,
    _p0: f32,
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var stamp: texture_2d<f32>;
@group(0) @binding(2) var edited: texture_2d<f32>;
@group(0) @binding(3) var layer_out: texture_storage_2d<r32float, write>;

fn sample_clamped(i: i32, j: i32) -> f32 {
    let ii = clamp(i, 0, i32(u.width) - 1);
    let jj = clamp(j, 0, i32(u.height) - 1);
    return textureLoad(stamp, vec2<i32>(ii, jj), 0).r;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.region_w || gid.y >= u.region_h) { return; }
    let x = u.region_x + gid.x;
    let y = u.region_y + gid.y;
    if (x >= u.width || y >= u.height) { return; }
    let p = vec2<i32>(i32(x), i32(y));
    let center = textureLoad(stamp, p, 0).r;

    var sum = 0.0;
    for (var dj = -1; dj <= 1; dj = dj + 1) {
        for (var di = -1; di <= 1; di = di + 1) {
            sum = sum + sample_clamped(p.x + di, p.y + dj);
        }
    }
    let avg = sum / 9.0;

    let e = textureLoad(edited, p, 0).r;
    let a = clamp(u.reconcile, 0.0, 1.0) * e * 0.35;
    let out_h = center + (avg - center) * a;
    textureStore(layer_out, p, vec4<f32>(out_h, 0.0, 0.0, 0.0));
}
