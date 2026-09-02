struct Params {
    width: u32,
    height: u32,
    rect_x: u32,
    rect_y: u32,
    rect_w: u32,
    rect_h: u32,
    probe_count: u32,
    compare: u32,
    epsilon: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct ProbeResult {
    failed: atomic<u32>,
    max_delta_bits: atomic<u32>,
    first_probe_encoded: atomic<u32>,
    compared: atomic<u32>,
}

@group(0) @binding(0) var source_height: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> baseline: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: ProbeResult;
@group(0) @binding(3) var<uniform> params: Params;

fn probe_coord(index: u32) -> vec2<u32> {
    let x = (index * 2654435761u + 1013904223u) % max(params.width, 1u);
    let y = (index * 2246822519u + 3266489917u) % max(params.height, 1u);
    return vec2<u32>(x, y);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= params.probe_count { return; }
    let coord = probe_coord(index);
    let value = textureLoad(source_height, vec2<i32>(coord), 0).x;
    let outside = coord.x < params.rect_x || coord.y < params.rect_y
        || coord.x >= params.rect_x + params.rect_w
        || coord.y >= params.rect_y + params.rect_h;
    if params.compare != 0u && outside {
        let delta = abs(value - baseline[index]);
        atomicAdd(&result.compared, 1u);
        atomicMax(&result.max_delta_bits, bitcast<u32>(delta));
        if delta != delta || delta > params.epsilon {
            atomicStore(&result.failed, 1u);
            atomicMax(&result.first_probe_encoded, params.probe_count - index);
        }
    }
    baseline[index] = value;
}
