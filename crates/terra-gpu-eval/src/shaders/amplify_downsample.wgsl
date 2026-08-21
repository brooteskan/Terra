// CPU-compatible area downsample / bilinear resize for MultiScaleAmplify levels.
struct Uniforms {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    area_mode: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<r32float, write>;

fn bilinear(i: u32, j: u32) -> f32 {
    let fu = (f32(i) + 0.5) / f32(u.dst_width);
    let fv = (f32(j) + 0.5) / f32(u.dst_height);
    let x = clamp(fu * f32(u.src_width) - 0.5, 0.0, f32(u.src_width - 1u));
    let y = clamp(fv * f32(u.src_height) - 0.5, 0.0, f32(u.src_height - 1u));
    let x0 = u32(floor(x));
    let y0 = u32(floor(y));
    let x1 = min(x0 + 1u, u.src_width - 1u);
    let y1 = min(y0 + 1u, u.src_height - 1u);
    let fx = x - f32(x0);
    let fy = y - f32(y0);
    let h00 = textureLoad(src, vec2<i32>(i32(x0), i32(y0)), 0).r;
    let h10 = textureLoad(src, vec2<i32>(i32(x1), i32(y0)), 0).r;
    let h01 = textureLoad(src, vec2<i32>(i32(x0), i32(y1)), 0).r;
    let h11 = textureLoad(src, vec2<i32>(i32(x1), i32(y1)), 0).r;
    return h00 * (1.0 - fx) * (1.0 - fy)
        + h10 * fx * (1.0 - fy)
        + h01 * (1.0 - fx) * fy
        + h11 * fx * fy;
}

fn area_sample(i: u32, j: u32) -> f32 {
    let x0 = f32(i) * f32(u.src_width) / f32(u.dst_width);
    let x1 = f32(i + 1u) * f32(u.src_width) / f32(u.dst_width);
    let y0 = f32(j) * f32(u.src_height) / f32(u.dst_height);
    let y1 = f32(j + 1u) * f32(u.src_height) / f32(u.dst_height);
    let sx0 = u32(floor(x0));
    let sx1 = min(u32(ceil(x1)), u.src_width);
    let sy0 = u32(floor(y0));
    let sy1 = min(u32(ceil(y1)), u.src_height);
    var sum = 0.0;
    var weight = 0.0;
    for (var sy = sy0; sy < sy1; sy++) {
        let wy = max(min(y1, f32(sy + 1u)) - max(y0, f32(sy)), 0.0);
        for (var sx = sx0; sx < sx1; sx++) {
            let wx = max(min(x1, f32(sx + 1u)) - max(x0, f32(sx)), 0.0);
            let w = wx * wy;
            sum += textureLoad(src, vec2<i32>(i32(sx), i32(sy)), 0).r * w;
            weight += w;
        }
    }
    return select(0.0, sum / weight, weight > 1.1920929e-7);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.dst_width || gid.y >= u.dst_height) { return; }
    let value = select(bilinear(gid.x, gid.y), area_sample(gid.x, gid.y), u.area_mode != 0u);
    textureStore(dst, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(value, 0.0, 0.0, 0.0));
}
