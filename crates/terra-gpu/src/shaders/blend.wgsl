struct Uniforms {
    width: u32,
    height: u32,
    opacity: f32,
    mode: u32, // 0 Normal … 6 Overlay
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var src_base: texture_2d<f32>;
@group(0) @binding(2) var src_layer: texture_2d<f32>;
@group(0) @binding(3) var src_mask: texture_2d<f32>;
@group(0) @binding(4) var src_extra_mask: texture_2d<f32>;
@group(0) @binding(5) var dst: texture_storage_2d<r32float, write>;

fn blend_pair(mode: u32, a: f32, b: f32) -> f32 {
    switch mode {
        case 0u: { return b; }
        case 1u: { return a + b; }
        case 2u: { return a - b; }
        case 3u: { return a * b; }
        case 4u: { return min(a, b); }
        case 5u: { return max(a, b); }
        default: {
            if (a < 0.0) {
                return 2.0 * a * b;
            }
            return a + b - a * b / (abs(a) + 1.0);
        }
    }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) { return; }
    let p = vec2<i32>(i32(gid.x), i32(gid.y));
    let hin = textureLoad(src_base, p, 0).r;
    let hlayer = textureLoad(src_layer, p, 0).r;
    let m = textureLoad(src_mask, p, 0).r * textureLoad(src_extra_mask, p, 0).r;
    let w = clamp(u.opacity * m, 0.0, 1.0);
    let blended = blend_pair(u.mode, hin, hlayer);
    let out_h = hin * (1.0 - w) + blended * w;
    textureStore(dst, p, vec4<f32>(out_h, 0.0, 0.0, 0.0));
}
