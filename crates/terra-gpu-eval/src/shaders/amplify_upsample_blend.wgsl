// Bilinearly lift one processed SimLevel and retarget it toward the pre-level field.
struct Uniforms {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    hardness: f32,
    ridge_lock: f32,
    lock_strength: f32,
    detail_boost: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var processed: texture_2d<f32>;
@group(0) @binding(2) var before: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<r32float, write>;

fn processed_at(i: u32, j: u32) -> f32 {
    // `upsample_to_metrics` routes through downsample_height(src, target.width),
    // whose logical destination is square even when only dst_height rows are read.
    let logical_height = u.dst_width;
    let fu = (f32(i) + 0.5) / f32(u.dst_width);
    let fv = (f32(j) + 0.5) / f32(logical_height);
    let x = clamp(fu * f32(u.src_width) - 0.5, 0.0, f32(u.src_width - 1u));
    let y = clamp(fv * f32(u.src_height) - 0.5, 0.0, f32(u.src_height - 1u));
    let x0 = u32(floor(x));
    let y0 = u32(floor(y));
    let x1 = min(x0 + 1u, u.src_width - 1u);
    let y1 = min(y0 + 1u, u.src_height - 1u);
    let fx = x - f32(x0);
    let fy = y - f32(y0);
    let h00 = textureLoad(processed, vec2<i32>(i32(x0), i32(y0)), 0).r;
    let h10 = textureLoad(processed, vec2<i32>(i32(x1), i32(y0)), 0).r;
    let h01 = textureLoad(processed, vec2<i32>(i32(x0), i32(y1)), 0).r;
    let h11 = textureLoad(processed, vec2<i32>(i32(x1), i32(y1)), 0).r;
    return h00 * (1.0 - fx) * (1.0 - fy)
        + h10 * fx * (1.0 - fy)
        + h01 * (1.0 - fx) * fy
        + h11 * fx * fy;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.dst_width || gid.y >= u.dst_height) { return; }
    let p = vec2<i32>(i32(gid.x), i32(gid.y));
    let base = textureLoad(before, p, 0).r;
    let amp = processed_at(gid.x, gid.y);
    let k = clamp(u.hardness, 0.0, 1.0);
    let lock = clamp(u.ridge_lock, 0.0, 1.0);
    let lock_strength = clamp(u.lock_strength, 0.0, 1.0);
    let soft = max(1.0 - k, 0.0);
    let unlock = max(1.0 - lock * lock_strength, 0.0);
    let accept = clamp(clamp(soft * unlock * max(u.detail_boost, 0.0), 0.0, 1.5) / 1.5, 0.0, 1.0);
    let preserve = clamp(max(k, lock) * lock_strength, 0.0, 1.0);
    let t = clamp(accept * (1.0 - preserve), 0.0, 1.0);
    textureStore(dst, p, vec4<f32>(base * (1.0 - t) + amp * t, 0.0, 0.0, 0.0));
}
