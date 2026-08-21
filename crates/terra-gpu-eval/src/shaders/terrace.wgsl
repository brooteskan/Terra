struct Uniforms {
    width: u32,
    height: u32,
    levels: u32,
    sharpness: f32,
    min_h: f32,
    max_h: f32,
    _p0: f32,
    _p1: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) { return; }
    let p = vec2<i32>(i32(gid.x), i32(gid.y));
    let h = textureLoad(src, p, 0).r;
    let levels = f32(max(u.levels, 2u));
    let span = max(u.max_h - u.min_h, 1e-6);
    let t = clamp((h - u.min_h) / span, 0.0, 1.0);
    let stepped = floor(t * levels) / max(levels - 1.0, 1.0);
    let k = clamp(u.sharpness, 0.0, 1.0);
    let mixed = t * (1.0 - k) + stepped * k;
    let out_h = u.min_h + mixed * span;
    textureStore(dst, p, vec4<f32>(out_h, 0.0, 0.0, 0.0));
}
