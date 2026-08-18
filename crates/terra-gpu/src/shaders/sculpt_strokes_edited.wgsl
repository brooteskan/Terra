// Interactive SculptStrokes preview — edited-coverage pass (#117).
//
// Writes the max brush weight touching each texel across the whole stroke set, the
// per-texel coverage the reconcile pass consumes. `edited = max(w)` is independent
// of stroke order and of the Flatten targets, so it is computed once here rather
// than threaded through the segmented stamp passes. The weight is the exact mirror
// of the stamp kernel's (`smoothstep_weight(dist_to_polyline) * pressure`).

struct Uniforms {
    width: u32,
    height: u32,
    world_x: f32,
    world_z: f32,
    stroke_lo: u32, // always 0 here; shared layout with the stamp pass
    stroke_hi: u32, // stroke_count
    _p1: u32,
    _p2: u32,
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
@group(0) @binding(1) var<storage, read> headers: array<StrokeHeader>;
@group(0) @binding(2) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(3) var edited_out: texture_storage_2d<r32float, write>;

fn smoothstep_weight(dist: f32, radius: f32, falloff: f32) -> f32 {
    let t = clamp(1.0 - dist / max(radius, 1e-4), 0.0, 1.0);
    let s = t * t * (3.0 - 2.0 * t);
    return pow(s, max(falloff, 0.1));
}

fn dist_to_polyline(wx: f32, wz: f32, header: StrokeHeader, pressure: ptr<function, f32>) -> f32 {
    let sx = u.world_x;
    let sz = u.world_z;
    let n = header.point_count;
    if (n == 0u) {
        *pressure = 0.0;
        return 1e30;
    }
    let base = header.first_point;
    if (n == 1u) {
        let p = points[base];
        *pressure = clamp(p.z, 0.0, 1.0);
        let dx = wx - p.x * sx;
        let dz = wz - p.y * sz;
        return sqrt(dx * dx + dz * dz);
    }
    var best = 1e30;
    var press = 0.0;
    for (var k = 0u; k + 1u < n; k = k + 1u) {
        let a = points[base + k];
        let b = points[base + k + 1u];
        let ax = a.x * sx;
        let az = a.y * sz;
        let bx = b.x * sx;
        let bz = b.y * sz;
        let vx = bx - ax;
        let vz = bz - az;
        let t = clamp(((wx - ax) * vx + (wz - az) * vz) / max(vx * vx + vz * vz, 1e-8), 0.0, 1.0);
        let cx = ax + vx * t;
        let cz = az + vz * t;
        let dx = wx - cx;
        let dz = wz - cz;
        let d = sqrt(dx * dx + dz * dz);
        if (d < best) {
            best = d;
            press = clamp(a.z + (b.z - a.z) * t, 0.0, 1.0);
        }
    }
    *pressure = press;
    return best;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) { return; }
    let p = vec2<i32>(i32(gid.x), i32(gid.y));
    let dx = u.world_x / f32(u.width);
    let dz = u.world_z / f32(u.height);
    let wx = (f32(gid.x) + 0.5) * dx;
    let wz = (f32(gid.y) + 0.5) * dz;

    var edited = 0.0;
    for (var si = u.stroke_lo; si < u.stroke_hi; si = si + 1u) {
        let header = headers[si];
        if (wx < header.bbox_min.x || wx > header.bbox_max.x
            || wz < header.bbox_min.y || wz > header.bbox_max.y) {
            continue;
        }
        var pressure = 0.0;
        let dist = dist_to_polyline(wx, wz, header, &pressure);
        let weight = smoothstep_weight(dist, header.radius_m, header.falloff) * pressure;
        if (weight <= 0.0) {
            continue;
        }
        edited = max(edited, weight);
    }

    textureStore(edited_out, p, vec4<f32>(edited, 0.0, 0.0, 0.0));
}
