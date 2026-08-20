struct Uniforms {
    width: u32,
    height: u32,
    opacity: f32,
    blend_mode: u32,
    composite_mode: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var parent_field: texture_2d<f32>;
@group(0) @binding(2) var private_seed_field: texture_2d<f32>;
@group(0) @binding(3) var child_field: texture_2d<f32>;
@group(0) @binding(4) var mask_field: texture_2d<f32>;
@group(0) @binding(5) var output_field: texture_storage_2d<r32float, write>;

fn blend_pair(mode: u32, a: f32, b: f32) -> f32 {
    let smooth_k = 8.0;
    switch mode {
        case 0u: { return b; }
        case 1u: { return a + b; }
        case 2u: { return a - b; }
        case 3u: { return a * b; }
        case 4u: { return min(a, b); }
        case 5u: { return max(a, b); }
        case 6u: {
            if (a < 0.0) { return 2.0 * a * b; }
            return a + b - a * b / (abs(a) + 1.0);
        }
        case 7u: {
            let t = clamp((b - a) * 0.05 + 0.5, 0.0, 1.0);
            return a * (1.0 - t) + max(b, a) * t;
        }
        case 8u: {
            let h = clamp(0.5 + 0.5 * (a - b) / smooth_k, 0.0, 1.0);
            return b * (1.0 - h) + a * h + smooth_k * h * (1.0 - h);
        }
        case 9u: {
            let h = clamp(0.5 + 0.5 * (b - a) / smooth_k, 0.0, 1.0);
            return b * (1.0 - h) + a * h - smooth_k * h * (1.0 - h);
        }
        case 10u: {
            let h = clamp(0.5 + 0.5 * (a - b) / smooth_k, 0.0, 1.0);
            return b * (1.0 - h) + a * h + smooth_k * h * (1.0 - h);
        }
        default: {
            let h = clamp(0.5 + 0.5 * (a + b) / smooth_k, 0.0, 1.0);
            return (-b) * (1.0 - h) + a * h + smooth_k * h * (1.0 - h);
        }
    }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) { return; }
    let p = vec2<i32>(gid.xy);
    let parent = textureLoad(parent_field, p, 0).r;
    let seed = textureLoad(private_seed_field, p, 0).r;
    let child = textureLoad(child_field, p, 0).r;
    let weight = clamp(textureLoad(mask_field, p, 0).r * u.opacity, 0.0, 1.0);
    var result: f32;
    if (u.composite_mode == 1u) {
        result = parent + weight * (child - seed);
    } else {
        let blended = blend_pair(u.blend_mode, parent, child);
        result = parent * (1.0 - weight) + blended * weight;
    }
    textureStore(output_field, p, vec4<f32>(result, 0.0, 0.0, 0.0));
}
