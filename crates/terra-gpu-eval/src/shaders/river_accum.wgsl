// One D8 flow-accumulation pull iteration for RiverCarve GPU preview.

struct Uniforms {
    width: u32,
    height: u32,
    _p0: f32,
    _p1: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var height_tex: texture_2d<f32>;
@group(0) @binding(2) var acc_in: texture_2d<f32>;
@group(0) @binding(3) var acc_out: texture_storage_2d<r32float, write>;

fn h_at(i: i32, j: i32) -> f32 {
    let ii = clamp(i, 0, i32(u.width) - 1);
    let jj = clamp(j, 0, i32(u.height) - 1);
    return textureLoad(height_tex, vec2<i32>(ii, jj), 0).r;
}

fn acc_at(i: i32, j: i32) -> f32 {
    let ii = clamp(i, 0, i32(u.width) - 1);
    let jj = clamp(j, 0, i32(u.height) - 1);
    return textureLoad(acc_in, vec2<i32>(ii, jj), 0).r;
}

const D8: array<vec2<i32>, 8> = array<vec2<i32>, 8>(
    vec2<i32>(1, 0), vec2<i32>(1, 1), vec2<i32>(0, 1), vec2<i32>(-1, 1),
    vec2<i32>(-1, 0), vec2<i32>(-1, -1), vec2<i32>(0, -1), vec2<i32>(1, -1),
);

/// Steepest D8 downhill offset from (i,j). The ordering and epsilon tie-break
/// mirror terra-core's `flow_direction_d8`. Returns (0,0) for pits / flats.
fn steepest_down(i: i32, j: i32) -> vec2<i32> {
    let h0 = h_at(i, j);
    var best_dh = 0.0;
    var best = vec2<i32>(0, 0);
    for (var k = 0u; k < 8u; k++) {
        let offset = D8[k];
        let ni = i + offset.x;
        let nj = j + offset.y;
        if (ni < 0 || nj < 0 || ni >= i32(u.width) || nj >= i32(u.height)) {
            continue;
        }
        let dist = select(1.0, 1.41421356, offset.x != 0 && offset.y != 0);
        let dh = (h0 - h_at(ni, nj)) / dist;
        if (dh > best_dh + 1e-12) {
            best_dh = dh;
            best = offset;
        }
    }
    return best;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = i32(gid.x);
    let j = i32(gid.y);
    if (gid.x >= u.width || gid.y >= u.height) { return; }

    // Rain + contributions from neighbors that drain into this cell.
    var sum = 1.0;
    for (var dj = -1; dj <= 1; dj++) {
        for (var di = -1; di <= 1; di++) {
            if (di == 0 && dj == 0) { continue; }
            let ni = i + di;
            let nj = j + dj;
            if (ni < 0 || nj < 0 || ni >= i32(u.width) || nj >= i32(u.height)) {
                continue;
            }
            let flow = steepest_down(ni, nj);
            if (flow.x == -di && flow.y == -dj) {
                sum += acc_at(ni, nj);
            }
        }
    }
    textureStore(acc_out, vec2<i32>(i, j), vec4<f32>(sum, 0.0, 0.0, 0.0));
}
