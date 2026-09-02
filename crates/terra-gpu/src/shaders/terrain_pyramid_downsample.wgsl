struct Uniforms {
    child_width: u32,
    child_height: u32,
    parent_width: u32,
    parent_height: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var child: texture_2d<f32>;
@group(0) @binding(2) var parent: texture_storage_2d<r32float, write>;

fn overlap(a0: f32, a1: f32, b0: f32, b1: f32) -> f32 {
    return max(0.0, min(a1, b1) - max(a0, b0));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.parent_width || gid.y >= u.parent_height) { return; }
    let px0 = f32(gid.x) / f32(u.parent_width);
    let px1 = f32(gid.x + 1u) / f32(u.parent_width);
    let py0 = f32(gid.y) / f32(u.parent_height);
    let py1 = f32(gid.y + 1u) / f32(u.parent_height);
    let cx0 = u32(floor(px0 * f32(u.child_width)));
    let cy0 = u32(floor(py0 * f32(u.child_height)));
    let cx1 = min(u32(ceil(px1 * f32(u.child_width))), u.child_width);
    let cy1 = min(u32(ceil(py1 * f32(u.child_height))), u.child_height);
    var total = 0.0;
    var weight_sum = 0.0;
    var first = 0.0;
    var min_value = 3.402823466e+38;
    var max_value = -3.402823466e+38;
    var sample_count = 0u;
    for (var cy = cy0; cy < cy1; cy = cy + 1u) {
        let sy0 = f32(cy) / f32(u.child_height);
        let sy1 = f32(cy + 1u) / f32(u.child_height);
        let wy = overlap(py0, py1, sy0, sy1);
        for (var cx = cx0; cx < cx1; cx = cx + 1u) {
            let sx0 = f32(cx) / f32(u.child_width);
            let sx1 = f32(cx + 1u) / f32(u.child_width);
            let weight = overlap(px0, px1, sx0, sx1) * wy;
            let value = textureLoad(child, vec2<i32>(i32(cx), i32(cy)), 0).r;
            if (sample_count == 0u) { first = value; }
            sample_count = sample_count + 1u;
            min_value = min(min_value, value);
            max_value = max(max_value, value);
            total = total + value * weight;
            weight_sum = weight_sum + weight;
        }
    }
    var value = select(0.0, total / weight_sum, weight_sum > 0.0);
    if (sample_count > 0u && min_value == max_value) { value = first; }
    textureStore(parent, vec2<i32>(gid.xy), vec4<f32>(value, 0.0, 0.0, 0.0));
}
