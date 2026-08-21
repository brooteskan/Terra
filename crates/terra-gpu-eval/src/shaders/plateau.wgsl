struct Uniforms {
    width: u32,
    height: u32,
    low: f32,
    high: f32,
    soft: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<r32float, write>;

fn plateau_sample(h: f32) -> f32 {
    let soft = max(u.soft, 1e-3);
    if (h < u.low) {
        let t = clamp((h - (u.low - soft)) / soft, 0.0, 1.0);
        return (u.low - soft) + t * soft;
    }
    if (h > u.high) {
        let t = clamp((h - u.high) / soft, 0.0, 1.0);
        return u.high + t * soft * 0.25;
    }
    let mid = (u.low + u.high) * 0.5;
    return h * 0.25 + mid * 0.75;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) { return; }
    let p = vec2<i32>(i32(gid.x), i32(gid.y));
    let h = textureLoad(src, p, 0).r;
    textureStore(dst, p, vec4<f32>(plateau_sample(h), 0.0, 0.0, 0.0));
}
