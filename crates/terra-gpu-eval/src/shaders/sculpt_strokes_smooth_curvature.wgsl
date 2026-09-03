// Horizontal half of the tapered separable Sculpt Smooth filter (#227).
//
// The brush mask is deliberately absent from this pass. Filtering through a
// closed radial mask makes displaced terrace height collect at the mask edge.
// The vertical pass computes the brush weight and blends the completed filter
// target into the original terrain once.

struct Uniforms {
    width: u32,
    height: u32,
    world_x: f32,
    world_z: f32,
    stroke_index: u32,
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
};

struct StrokeHeader {
    kind: u32,
    first_point: u32,
    point_count: u32,
    pad0: u32,
    radius_m: f32,
    strength: f32,
    target_height: f32,
    falloff: f32,
    bbox_min: vec2<f32>,
    bbox_max: vec2<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var running_in: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> headers: array<StrokeHeader>;
@group(0) @binding(3) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(4) var horizontal_out: texture_storage_2d<r32float, write>;

const SMOOTH_SPREAD_DEFAULT: u32 = 1u;
const SMOOTH_SPREAD_MAX: u32 = 128u;
const SMOOTH_FILTER_SUPPORT_SCALE: u32 = 2u;

fn smooth_spread(header: StrokeHeader) -> u32 {
    let value = header.target_height;
    if (!(value > 0.0) || value > 3.402823e38) {
        return SMOOTH_SPREAD_DEFAULT;
    }
    return u32(clamp(floor(value + 0.5), 1.0, f32(SMOOTH_SPREAD_MAX)));
}

fn kernel_weight(offset: u32, support: u32) -> f32 {
    let t = 1.0 - f32(offset) / f32(support + 1u);
    return t * t;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.region_w || gid.y >= u.region_h) { return; }
    let x = u.region_x + gid.x;
    let y = u.region_y + gid.y;
    if (x >= u.width || y >= u.height) { return; }
    let p = vec2<i32>(i32(x), i32(y));
    let center = textureLoad(running_in, p, 0).r;
    let header = headers[u.stroke_index];
    if (!(header.strength > 0.0) || header.strength > 3.402823e38) {
        textureStore(horizontal_out, p, vec4<f32>(center, 0.0, 0.0, 0.0));
        return;
    }

    let requested = smooth_spread(header) * SMOOTH_FILTER_SUPPORT_SCALE;
    let support = min(requested, max(u.width - 1u, 1u));
    var weighted_sum = center;
    var weight_sum = 1.0;
    for (var offset = 1u; offset <= support; offset = offset + 1u) {
        let weight = kernel_weight(offset, support);
        let left = vec2<i32>(max(p.x - i32(offset), 0), p.y);
        let right = vec2<i32>(min(p.x + i32(offset), i32(u.width) - 1), p.y);
        weighted_sum = weighted_sum + weight
            * (textureLoad(running_in, left, 0).r + textureLoad(running_in, right, 0).r);
        weight_sum = weight_sum + 2.0 * weight;
    }
    textureStore(horizontal_out, p, vec4<f32>(weighted_sum / weight_sum, 0.0, 0.0, 0.0));
}
