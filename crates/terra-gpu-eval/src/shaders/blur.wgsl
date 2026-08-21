struct Uniforms {
    width: u32,
    height: u32,
    radius: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = i32(gid.x);
    let j = i32(gid.y);
    if (gid.x >= u.width || gid.y >= u.height) { return; }
    let r = i32(u.radius);
    var sum = 0.0;
    var count = 0.0;
    for (var dj = -r; dj <= r; dj++) {
        for (var di = -r; di <= r; di++) {
            let ni = clamp(i + di, 0, i32(u.width) - 1);
            let nj = clamp(j + dj, 0, i32(u.height) - 1);
            sum += textureLoad(src, vec2<i32>(ni, nj), 0).r;
            count += 1.0;
        }
    }
    textureStore(dst, vec2<i32>(i, j), vec4<f32>(sum / max(count, 1.0), 0.0, 0.0, 0.0));
}
