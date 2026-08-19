// Height-only Path preview. The CPU tessellates authored Catmull-Rom nodes and
// uploads the exact world-space polyline consumed here.

struct Uniforms {
    width: u32,
    height: u32,
    world_x: f32,
    world_z: f32,
    point_count: u32,
    carve: u32,
    seed: u32,
    _pad0: u32,
    base_width: f32,
    falloff: f32,
    noise_strength: f32,
    noise_scale: f32,
    height_offset: f32,
    profile: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var src: texture_2d<f32>;
// x, z, interpolated path height, interpolated width scale.
@group(0) @binding(2) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(3) var dst: texture_storage_2d<r32float, write>;

fn hash_u32(x0: u32) -> u32 {
    var x = x0;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

fn hash2(ix: i32, iz: i32, seed: u32) -> u32 {
    var h = seed;
    h ^= bitcast<u32>(ix);
    h = hash_u32(h);
    h ^= bitcast<u32>(iz);
    return hash_u32(h);
}

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn grad_value(h: u32) -> f32 {
    return f32(h) / f32(0xffffffffu) * 2.0 - 1.0;
}

fn value_noise(p: vec2<f32>, seed: u32) -> f32 {
    let cell = vec2<i32>(floor(p));
    let f = p - vec2<f32>(f32(cell.x), f32(cell.y));
    let sx = fade(f.x);
    let sz = fade(f.y);
    let n00 = grad_value(hash2(cell.x, cell.y, seed));
    let n10 = grad_value(hash2(cell.x + 1, cell.y, seed));
    let n01 = grad_value(hash2(cell.x, cell.y + 1, seed));
    let n11 = grad_value(hash2(cell.x + 1, cell.y + 1, seed));
    let nx0 = n00 + (n10 - n00) * sx;
    let nx1 = n01 + (n11 - n01) * sx;
    return nx0 + (nx1 - nx0) * sz;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) { return; }
    let texel = vec2<i32>(i32(gid.x), i32(gid.y));
    let h0 = textureLoad(src, texel, 0).r;
    if (u.point_count < 2u) {
        textureStore(dst, texel, vec4<f32>(h0, 0.0, 0.0, 0.0));
        return;
    }

    let x = (f32(gid.x) + 0.5) / f32(u.width) * u.world_x;
    let z = (f32(gid.y) + 0.5) / f32(u.height) * u.world_z;
    var best = 1e30;
    var width_scale = 1.0;
    var path_height = 0.0;
    for (var index = 0u; index + 1u < u.point_count; index++) {
        let a = points[index];
        let b = points[index + 1u];
        let ab = b.xy - a.xy;
        let ap = vec2<f32>(x, z) - a.xy;
        let ab2 = dot(ab, ab);
        var t = 0.0;
        if (ab2 >= 1e-12) {
            t = clamp(dot(ap, ab) / ab2, 0.0, 1.0);
        }
        let closest = a.xy + ab * t;
        let delta = vec2<f32>(x, z) - closest;
        let distance = sqrt(dot(delta, delta));
        if (distance < best) {
            best = distance;
            width_scale = a.w + (b.w - a.w) * t;
            path_height = a.z + (b.z - a.z) * t;
        }
    }

    let half_width = max(u.base_width * width_scale, 0.1);
    let falloff = max(u.falloff, 0.1);
    if (best > half_width + falloff) {
        textureStore(dst, texel, vec4<f32>(h0, 0.0, 0.0, 0.0));
        return;
    }
    var edge = 1.0;
    if (best > half_width) {
        let t = 1.0 - clamp((best - half_width) / falloff, 0.0, 1.0);
        edge = t * t * (3.0 - 2.0 * t);
    }
    edge = pow(edge, clamp(u.profile, 0.25, 4.0));
    var noise = 0.0;
    if (u.noise_strength > 1e-5) {
        noise = value_noise(vec2<f32>(x, z) * u.noise_scale, u.seed) * u.noise_strength;
    }
    let signed_height = u.height_offset + path_height + noise;
    var path_delta = signed_height * edge;
    if (u.carve != 0u) {
        path_delta = -abs(signed_height) * edge;
    }
    textureStore(dst, texel, vec4<f32>(h0 + path_delta, 0.0, 0.0, 0.0));
}
