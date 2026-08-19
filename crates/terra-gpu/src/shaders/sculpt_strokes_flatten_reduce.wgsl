// Interactive SculptStrokes preview — Flatten footprint-mean reduce pass (#117).
//
// Partial reduction of one Flatten stroke's brush-weighted footprint mean over the
// running field entering that stroke. Each workgroup tree-reduces its 8x8 tile of
// `(h * w, w)` into one `vec2<f32>` partial; the resolve pass folds the partials
// into the scalar target. The weight mirrors the stamp/edited kernels exactly
// (`smoothstep_weight(dist_to_polyline) * pressure`), and — like the CPU
// `flatten_target_for` — only texels with `w > 0` contribute. Texels outside the
// footprint contribute an exact `(0, 0)`, so the effective sum spans just the
// footprint and its rounding stays proportional to the mean.

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
@group(0) @binding(4) var<storage, read_write> partials: array<vec2<f32>>;

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

var<workgroup> sdata: array<vec2<f32>, 64>;

@compute @workgroup_size(8, 8)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    var contrib = vec2<f32>(0.0, 0.0);
    if (gid.x < u.region_w && gid.y < u.region_h) {
        let x = u.region_x + gid.x;
        let y = u.region_y + gid.y;
        let header = headers[u.stroke_index];
        let dx = u.world_x / f32(u.width);
        let dz = u.world_z / f32(u.height);
        let wx = (f32(x) + 0.5) * dx;
        let wz = (f32(y) + 0.5) * dz;
        if (wx >= header.bbox_min.x && wx <= header.bbox_max.x
            && wz >= header.bbox_min.y && wz <= header.bbox_max.y) {
            var pressure = 0.0;
            let dist = dist_to_polyline(wx, wz, header, &pressure);
            let w = smoothstep_weight(dist, header.radius_m, header.falloff) * pressure;
            if (w > 0.0) {
                let h = textureLoad(running_in, vec2<i32>(i32(x), i32(y)), 0).r;
                contrib = vec2<f32>(h * w, w);
            }
        }
    }

    sdata[lid] = contrib;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) {
            sdata[lid] = sdata[lid] + sdata[lid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid == 0u) {
        let idx = wg.y * nwg.x + wg.x;
        partials[idx] = sdata[0];
    }
}
