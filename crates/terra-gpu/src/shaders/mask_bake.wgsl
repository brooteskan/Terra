// Bake interactive masks from the current GPU height prefix (no CPU readback).

struct Uniforms {
    width: u32,
    height: u32,
    mode: u32, // 0 constant, 1 height, 2 slope
    dz: f32,
    dx: f32,
    value: f32,       // constant value OR height mid
    range_min: f32,   // height/slope min
    range_max: f32,   // height/slope max
    invert: f32,
    strength: f32,
    frequency: f32,
    seed: f32,
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var height_tex: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<r32float, write>;

fn sample_h(i: i32, j: i32) -> f32 {
    let ii = clamp(i, 0, i32(u.width) - 1);
    let jj = clamp(j, 0, i32(u.height) - 1);
    return textureLoad(height_tex, vec2<i32>(ii, jj), 0).r;
}

fn range_mask(v: f32, lo: f32, hi: f32, epsilon: f32) -> f32 {
    return clamp((v - lo) / max(hi - lo, epsilon), 0.0, 1.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let has_region = u.region_w > 0u && u.region_h > 0u;
    let i = select(gid.x, u.region_x + gid.x, has_region);
    let j = select(gid.y, u.region_y + gid.y, has_region);
    if (has_region) {
        if (gid.x >= u.region_w || gid.y >= u.region_h) { return; }
    } else if (gid.x >= u.width || gid.y >= u.height) {
        return;
    }
    if (i >= u.width || j >= u.height) { return; }

    let ii = i32(i);
    let jj = i32(j);
    var m = 1.0;
    switch u.mode {
        case 0u: {
            m = clamp(u.value, 0.0, 1.0);
        }
        case 1u: {
            m = range_mask(sample_h(ii, jj), u.range_min, u.range_max, 1e-6);
        }
        case 2u: {
            let dx = max(u.dx, 1e-4);
            let dz = max(u.dz, 1e-4);
            let hx = sample_h(ii + 1, jj) - sample_h(ii - 1, jj);
            let hz = sample_h(ii, jj + 1) - sample_h(ii, jj - 1);
            let g = sqrt((hx * 0.5 / dx) * (hx * 0.5 / dx) + (hz * 0.5 / dz) * (hz * 0.5 / dz));
            let deg = atan(g) * 57.2957795;
            m = range_mask(deg, u.range_min, u.range_max, 1e-3);
        }
        default: {
            m = 1.0;
        }
    }
    m = clamp(m * u.strength, 0.0, 1.0);
    if (u.invert > 0.5) {
        m = 1.0 - m;
    }
    textureStore(dst, vec2<i32>(ii, jj), vec4<f32>(m, 0.0, 0.0, 0.0));
}
