// Gather-style channel carve from flow accumulation (RiverCarve GPU preview).

struct Uniforms {
    width: u32,
    height: u32,
    threshold: f32,
    depth: f32,
    channel_width: f32,
    bank_smooth: f32,
    max_radius: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var height_tex: texture_2d<f32>;
@group(0) @binding(2) var acc_tex: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = i32(gid.x);
    let j = i32(gid.y);
    if (gid.x >= u.width || gid.y >= u.height) { return; }

    let h0 = textureLoad(height_tex, vec2<i32>(i, j), 0).r;
    let bank_smooth = max(u.bank_smooth, 0.0);
    let base_w = max(u.channel_width, 1.0);
    // Search radius covers the widest bank we may carve from a high-acc cell.
    let max_scale = 4.0;
    let max_bank = base_w * max_scale * (1.0 + bank_smooth * 0.75);
    let radius = min(i32(ceil(max_bank)), i32(u.max_radius));

    var carve = 0.0;
    for (var dj = -radius; dj <= radius; dj++) {
        for (var di = -radius; di <= radius; di++) {
            let ni = i + di;
            let nj = j + dj;
            if (ni < 0 || nj < 0 || ni >= i32(u.width) || nj >= i32(u.height)) {
                continue;
            }
            let acc = textureLoad(acc_tex, vec2<i32>(ni, nj), 0).r;
            if (acc < u.threshold) {
                continue;
            }
            let accumulation_scale = clamp(sqrt(acc / max(u.threshold, 1e-3)), 1.0, 4.0);
            let channel_width = base_w * accumulation_scale;
            let channel_depth = u.depth * sqrt(accumulation_scale);
            let bank_radius = channel_width * (1.0 + bank_smooth * 0.75);
            let dist = sqrt(f32(di * di + dj * dj));
            if (dist > bank_radius) {
                continue;
            }
            let sigma = channel_width * max(0.35 + bank_smooth * 0.2, 0.1);
            let falloff = exp(-0.5 * (dist / sigma) * (dist / sigma));
            // CPU RiverCarve subtracts every overlapping source contribution.
            // This gather visits the same source-cell order for one destination.
            carve += channel_depth * falloff;
        }
    }

    textureStore(dst, vec2<i32>(i, j), vec4<f32>(h0 - carve, 0.0, 0.0, 0.0));
}
