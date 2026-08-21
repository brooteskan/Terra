// Closed-polygon raise/carve preview. Vertices remain in authored normalized UV;
// edge distance is evaluated in world space exactly like the CPU oracle.

struct Uniforms {
    width: u32,
    height: u32,
    world_x: f32,
    world_z: f32,
    point_count: u32,
    mode: u32,
    carve: u32,
    _pad0: u32,
    target_height: f32,
    falloff: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(3) var dst: texture_storage_2d<r32float, write>;

fn point_in_polygon(p: vec2<f32>) -> bool {
    var inside = false;
    var previous = u.point_count - 1u;
    for (var index = 0u; index < u.point_count; index++) {
        let current_point = points[index].xy;
        let previous_point = points[previous].xy;
        let crosses = (current_point.y > p.y) != (previous_point.y > p.y);
        if (crosses) {
            let denominator = max(previous_point.y - current_point.y, 1e-12);
            let boundary = (previous_point.x - current_point.x)
                * (p.y - current_point.y) / denominator + current_point.x;
            if (p.x < boundary) {
                inside = !inside;
            }
        }
        previous = index;
    }
    return inside;
}

fn segment_distance(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let denom = dot(ab, ab);
    var t = 0.0;
    if (denom > 1e-12) {
        t = clamp(dot(p - a, ab) / denom, 0.0, 1.0);
    }
    let delta = p - (a + ab * t);
    return sqrt(dot(delta, delta));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) { return; }
    let texel = vec2<i32>(i32(gid.x), i32(gid.y));
    let h0 = textureLoad(src, texel, 0).r;
    if (u.point_count < 3u) {
        textureStore(dst, texel, vec4<f32>(h0, 0.0, 0.0, 0.0));
        return;
    }

    let uv = vec2<f32>(
        (f32(gid.x) + 0.5) / f32(u.width),
        (f32(gid.y) + 0.5) / f32(u.height),
    );
    let world = uv * vec2<f32>(u.world_x, u.world_z);
    var distance = 1e30;
    for (var index = 0u; index < u.point_count; index++) {
        let next = (index + 1u) % u.point_count;
        let a = points[index].xy * vec2<f32>(u.world_x, u.world_z);
        let b = points[next].xy * vec2<f32>(u.world_x, u.world_z);
        distance = min(distance, segment_distance(world, a, b));
    }
    let edge_t = clamp(distance / u.falloff, 0.0, 1.0);
    var weight = 1.0;
    if (!point_in_polygon(uv)) {
        let t = 1.0 - edge_t;
        weight = t * t * (3.0 - 2.0 * t);
    }
    if (weight <= 1e-4) {
        textureStore(dst, texel, vec4<f32>(h0, 0.0, 0.0, 0.0));
        return;
    }

    var desired_height = h0 + u.target_height;
    if (u.mode == 0u) {
        if (u.carve != 0u) {
            desired_height = h0 - abs(u.target_height);
        }
    } else if (u.carve != 0u) {
        desired_height = h0 - abs(u.target_height);
    } else {
        desired_height = u.target_height;
    }
    let output = h0 * (1.0 - weight) + desired_height * weight;
    textureStore(dst, texel, vec4<f32>(output, 0.0, 0.0, 0.0));
}
