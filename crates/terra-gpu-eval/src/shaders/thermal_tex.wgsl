// Pass A: write outbound sediment mass into `delta` (positive = leaving cell).
// Softness = 1 - K from hardness texture (R32Float).
struct Uniforms {
    width: u32,
    height: u32,
    dx: f32,
    talus: f32,
    strength: f32,
    _p2: f32,
    _p3: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var delta: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var hardness: texture_2d<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    if (i >= u.width || j >= u.height) { return; }
    let p = vec2<i32>(i32(i), i32(j));
    let h0 = textureLoad(src, p, 0).r;
    let k = clamp(textureLoad(hardness, p, 0).r, 0.0, 1.0);
    let soft = 1.0 - k;
    var sum = 0.0;
    let ii = i32(i);
    let jj = i32(j);
    let d0 = vec2<i32>(ii - 1, jj);
    if (d0.x >= 0) {
        let diff = h0 - textureLoad(src, d0, 0).r - u.talus;
        if (diff > 0.0) { sum += diff; }
    }
    let d1 = vec2<i32>(ii + 1, jj);
    if (d1.x < i32(u.width)) {
        let diff = h0 - textureLoad(src, d1, 0).r - u.talus;
        if (diff > 0.0) { sum += diff; }
    }
    let d2 = vec2<i32>(ii, jj - 1);
    if (d2.y >= 0) {
        let diff = h0 - textureLoad(src, d2, 0).r - u.talus;
        if (diff > 0.0) { sum += diff; }
    }
    let d3 = vec2<i32>(ii, jj + 1);
    if (d3.y < i32(u.height)) {
        let diff = h0 - textureLoad(src, d3, 0).r - u.talus;
        if (diff > 0.0) { sum += diff; }
    }
    let move_amt = select(0.0, sum * u.strength * 0.25 * soft, sum > 0.0);
    textureStore(delta, p, vec4<f32>(move_amt, 0.0, 0.0, 0.0));
}
