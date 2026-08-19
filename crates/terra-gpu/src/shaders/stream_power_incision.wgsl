// Approximate stream-power incision for the GPU viewport preview.
// Uses multi-pass D8 accumulation (no Priority-Flood). CPU export stays authoritative.

struct Uniforms {
    width: u32,
    height: u32,
    k: f32,
    m: f32,
    n: f32,
    dt: f32,
    uplift: f32,
    base_level: f32,
    cell_area: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var height_tex: texture_2d<f32>;
@group(0) @binding(2) var acc_tex: texture_2d<f32>;
@group(0) @binding(3) var hardness_tex: texture_2d<f32>;
@group(0) @binding(4) var dst: texture_storage_2d<r32float, write>;

fn h_at(i: i32, j: i32) -> f32 {
    let ii = clamp(i, 0, i32(u.width) - 1);
    let jj = clamp(j, 0, i32(u.height) - 1);
    return textureLoad(height_tex, vec2<i32>(ii, jj), 0).r;
}

/// Steepest D8 downhill slope in the CPU hydro convention: height drop per
/// cardinal/diagonal grid step, not per world-space metre.
fn steepest_slope(i: i32, j: i32) -> f32 {
    let h0 = h_at(i, j);
    var best = 0.0;
    for (var dj = -1; dj <= 1; dj++) {
        for (var di = -1; di <= 1; di++) {
            if (di == 0 && dj == 0) { continue; }
            let ni = i + di;
            let nj = j + dj;
            if (ni < 0 || nj < 0 || ni >= i32(u.width) || nj >= i32(u.height)) {
                continue;
            }
            let d = select(1.0, 1.41421356, di != 0 && dj != 0);
            let dh = (h0 - h_at(ni, nj)) / d;
            best = max(best, dh);
        }
    }
    return best;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = i32(gid.x);
    let j = i32(gid.y);
    if (gid.x >= u.width || gid.y >= u.height) { return; }

    let h0 = textureLoad(height_tex, vec2<i32>(i, j), 0).r;
    let acc = textureLoad(acc_tex, vec2<i32>(i, j), 0).r;
    let khard = clamp(textureLoad(hardness_tex, vec2<i32>(i, j), 0).r, 0.0, 1.0);
    let soft = max(1.0 - khard, 0.0);
    let area = max(acc * u.cell_area, u.cell_area);
    let slope = max(steepest_slope(i, j), 1e-6);
    let power = u.k * pow(area, u.m) * pow(slope, u.n) * soft * u.dt;
    let step = min(power, min(slope * 2.0, 50.0));
    let next = max(h0 - step + u.uplift, u.base_level);
    textureStore(dst, vec2<i32>(i, j), vec4<f32>(next, 0.0, 0.0, 0.0));
}
