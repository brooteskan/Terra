// Vertical half and masked blend of the tapered separable Sculpt Smooth filter.
//
//     h' = mix(h, vertical(horizontal(h)), strength * brush_weight)
//
// Applying the radial weight only after filtering preserves affine slopes and
// avoids the circular no-flux collars produced by brush-gated diffusion.

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
    pad0: u32,
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
@group(0) @binding(2) var horizontal_blur: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> headers: array<StrokeHeader>;
@group(0) @binding(4) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(5) var smooth_out: texture_storage_2d<r32float, write>;

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

fn smoothstep_weight(dist: f32, radius: f32, falloff: f32) -> f32 {
    let t = clamp(1.0 - dist / max(radius, 1e-4), 0.0, 1.0);
    let s = t * t * (3.0 - 2.0 * t);
    return pow(s, max(falloff, 0.1));
}

fn dist_to_polyline(wx: f32, wz: f32, header: StrokeHeader, pressure: ptr<function, f32>) -> f32 {
    let n = header.point_count;
    if (n == 0u) {
        *pressure = 0.0;
        return 1e30;
    }
    let base = header.first_point;
    if (n == 1u) {
        let point = points[base];
        *pressure = clamp(point.z, 0.0, 1.0);
        let dx = wx - point.x * u.world_x;
        let dz = wz - point.y * u.world_z;
        return sqrt(dx * dx + dz * dz);
    }
    var best = 1e30;
    var best_pressure = 0.0;
    for (var k = 0u; k + 1u < n; k = k + 1u) {
        let a = points[base + k];
        let b = points[base + k + 1u];
        let ax = a.x * u.world_x;
        let az = a.y * u.world_z;
        let bx = b.x * u.world_x;
        let bz = b.y * u.world_z;
        let vx = bx - ax;
        let vz = bz - az;
        let t = clamp(((wx - ax) * vx + (wz - az) * vz) / max(vx * vx + vz * vz, 1e-8), 0.0, 1.0);
        let cx = ax + vx * t;
        let cz = az + vz * t;
        let dx = wx - cx;
        let dz = wz - cz;
        let distance = sqrt(dx * dx + dz * dz);
        if (distance < best) {
            best = distance;
            best_pressure = clamp(a.z + (b.z - a.z) * t, 0.0, 1.0);
        }
    }
    *pressure = best_pressure;
    return best;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.region_w || gid.y >= u.region_h) { return; }
    let x = u.region_x + gid.x;
    let y = u.region_y + gid.y;
    if (x >= u.width || y >= u.height) { return; }
    let p = vec2<i32>(i32(x), i32(y));
    let original = textureLoad(running_in, p, 0).r;
    let header = headers[u.stroke_index];
    if (!(header.strength > 0.0) || header.strength > 3.402823e38) {
        textureStore(smooth_out, p, vec4<f32>(original, 0.0, 0.0, 0.0));
        return;
    }

    let requested = smooth_spread(header) * SMOOTH_FILTER_SUPPORT_SCALE;
    let support = min(requested, max(u.height - 1u, 1u));
    var weighted_sum = textureLoad(horizontal_blur, p, 0).r;
    var weight_sum = 1.0;
    for (var offset = 1u; offset <= support; offset = offset + 1u) {
        let weight = kernel_weight(offset, support);
        let down = vec2<i32>(p.x, max(p.y - i32(offset), 0));
        let up = vec2<i32>(p.x, min(p.y + i32(offset), i32(u.height) - 1));
        weighted_sum = weighted_sum + weight
            * (textureLoad(horizontal_blur, down, 0).r + textureLoad(horizontal_blur, up, 0).r);
        weight_sum = weight_sum + 2.0 * weight;
    }
    let blurred = weighted_sum / weight_sum;

    let wx = (f32(x) + 0.5) * u.world_x / f32(u.width);
    let wz = (f32(y) + 0.5) * u.world_z / f32(u.height);
    if (wx < header.bbox_min.x || wx > header.bbox_max.x
        || wz < header.bbox_min.y || wz > header.bbox_max.y) {
        textureStore(smooth_out, p, vec4<f32>(original, 0.0, 0.0, 0.0));
        return;
    }
    var pressure = 0.0;
    let distance = dist_to_polyline(wx, wz, header, &pressure);
    let brush_weight = clamp(
        smoothstep_weight(distance, header.radius_m, header.falloff) * pressure,
        0.0,
        1.0,
    );
    let blend = clamp(header.strength, 0.0, 1.0) * brush_weight;
    let next = original + blend * (blurred - original);
    textureStore(smooth_out, p, vec4<f32>(next, 0.0, 0.0, 0.0));
}
