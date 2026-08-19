// Exact entering-height range reduction for EffectFilter remaps.

struct Uniforms {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
};

struct RangeState {
    min_ordered: atomic<u32>,
    max_ordered: atomic<u32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> range_state: RangeState;

fn float_to_ordered(value: f32) -> u32 {
    let bits = bitcast<u32>(value);
    if ((bits & 0x80000000u) != 0u) {
        return bits ^ 0xffffffffu;
    }
    return bits ^ 0x80000000u;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) {
        return;
    }
    let value = textureLoad(src, vec2<i32>(i32(gid.x), i32(gid.y)), 0).r;
    if (value != value) {
        return;
    }
    let ordered = float_to_ordered(value);
    atomicMin(&range_state.min_ordered, ordered);
    atomicMax(&range_state.max_ordered, ordered);
}
