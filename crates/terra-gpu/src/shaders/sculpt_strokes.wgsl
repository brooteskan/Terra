// Interactive SculptStrokes preview — stamp pass (#113).
//
// One dispatch stamps every supported per-sample stroke into `stamp_out`, chaining
// strokes at each texel exactly as the CPU `apply_sculpt_strokes` does (stroke k+1
// reads the height stroke k already wrote). `edited_out` receives the max brush
// weight touching the texel, which the reconcile pass consumes. The neighborhood
// and reduction kinds (Smooth / Pinch / Coastline / Flatten) never reach the GPU —
// the planner keeps any layer containing one on the CPU resume.

struct Uniforms {
    width: u32,
    height: u32,
    world_x: f32,   // world_size_x (metres)
    world_z: f32,   // world_size_z (metres)
    stroke_count: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

// Canonical, alias-collapsed stroke kind ids. The engine maps `SculptStrokeKind`
// onto these (e.g. MountainStamp -> RIDGE) so the shader has one arm per behaviour.
const KIND_RAISE: u32 = 0u;
const KIND_LOWER: u32 = 1u;
const KIND_RIDGE: u32 = 2u;
const KIND_VALLEY: u32 = 3u;
const KIND_TERRACE: u32 = 4u;
const KIND_NOISE: u32 = 5u;
const KIND_INFLATE: u32 = 6u;
const KIND_PLATEAU_STAMP: u32 = 7u;
const KIND_CRATER_STAMP: u32 = 8u;
const KIND_HEIGHT_STAMP: u32 = 9u;
const KIND_ERODE: u32 = 10u;
const KIND_AUX_NOOP: u32 = 11u;

struct StrokeHeader {
    kind: u32,
    first_point: u32,
    point_count: u32,
    pad0: u32,
    radius_m: f32,
    strength: f32,
    target_height: f32,
    falloff: f32,
    bbox_min: vec2<f32>,   // world-space footprint (point bbox padded by radius_m)
    bbox_max: vec2<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> headers: array<StrokeHeader>;
@group(0) @binding(3) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(4) var stamp_out: texture_storage_2d<r32float, write>;
@group(0) @binding(5) var edited_out: texture_storage_2d<r32float, write>;

// --- 64-bit unsigned emulation for a bit-exact `hash_noise` port -------------
// vec2<u32> holds (lo, hi). Grid coordinates are non-negative, so the u32 -> u64
// widening in the CPU hash zero-extends and never needs the sign path.

fn mul_full(x: u32, y: u32) -> vec2<u32> {
    let x0 = x & 0xFFFFu;
    let x1 = x >> 16u;
    let y0 = y & 0xFFFFu;
    let y1 = y >> 16u;
    let p00 = x0 * y0;
    let p01 = x0 * y1;
    let p10 = x1 * y0;
    let p11 = x1 * y1;
    let mid = (p00 >> 16u) + (p01 & 0xFFFFu) + (p10 & 0xFFFFu);
    let lo = (p00 & 0xFFFFu) | ((mid & 0xFFFFu) << 16u);
    let hi = p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);
    return vec2<u32>(lo, hi);
}

fn mul64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let ll = mul_full(a.x, b.x);
    let cross = a.x * b.y + a.y * b.x; // low 32 bits of the cross terms (wraps)
    return vec2<u32>(ll.x, ll.y + cross);
}

fn xor64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x ^ b.x, a.y ^ b.y);
}

// Right shift by 1..=31 (the only shifts the hash uses: 30, 27, 31).
fn shr64(a: vec2<u32>, s: u32) -> vec2<u32> {
    let lo = (a.x >> s) | (a.y << (32u - s));
    let hi = a.y >> s;
    return vec2<u32>(lo, hi);
}

fn hash_noise(px: i32, py: i32, seed: u32) -> f32 {
    let x = vec2<u32>(u32(px), 0u);
    let y = vec2<u32>(u32(py), 0u);
    let c0 = vec2<u32>(0x85EBCA87u, 0x9E3779B1u); // 0x9E3779B185EBCA87
    let c1 = vec2<u32>(0x27D4EB4Fu, 0xC2B2AE3Du); // 0xC2B2AE3D27D4EB4F
    let c2 = vec2<u32>(0x1CE4E5B9u, 0xBF58476Du); // 0xBF58476D1CE4E5B9
    let c3 = vec2<u32>(0x133111EBu, 0x94D049BBu); // 0x94D049BB133111EB
    var n = xor64(xor64(mul64(x, c0), mul64(y, c1)), vec2<u32>(seed, 0u));
    n = xor64(n, shr64(n, 30u));
    n = mul64(n, c2);
    n = xor64(n, shr64(n, 27u));
    n = mul64(n, c3);
    n = xor64(n, shr64(n, 31u));
    return f32(n.x) / f32(0xFFFFFFFFu) * 2.0 - 1.0;
}

// --- Brush weight + distance (mirror of authoring.rs) ------------------------

fn smoothstep_weight(dist: f32, radius: f32, falloff: f32) -> f32 {
    let t = clamp(1.0 - dist / max(radius, 1e-4), 0.0, 1.0);
    let s = t * t * (3.0 - 2.0 * t);
    return pow(s, max(falloff, 0.1));
}

// Min distance from (wx, wz) to the stroke polyline; also returns the brush
// pressure interpolated at the closest point. Matches `distance_to_polyline`.
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

// Rust f32::round is half-away-from-zero; WGSL `round` is half-to-even.
fn round_away(x: f32) -> f32 {
    return sign(x) * floor(abs(x) + 0.5);
}

fn apply_kind(header: StrokeHeader, h: f32, dist: f32, w: f32, px: i32, py: i32) -> f32 {
    let s = header.strength * w;
    let r = max(header.radius_m, 1.0);
    switch (header.kind) {
        case 0u: { return h + s; }                                  // RAISE
        case 1u: { return h - s; }                                  // LOWER
        case 2u: {                                                  // RIDGE
            let t = max(1.0 - dist / r, 0.0);
            return h + s * t * t;
        }
        case 3u: {                                                  // VALLEY
            let t = max(1.0 - dist / r, 0.0);
            return h - abs(s) * t;
        }
        case 4u: {                                                  // TERRACE
            let step = max(abs(header.strength), 0.1);
            return h + (round_away(h / step) * step - h) * w;
        }
        case 5u: {                                                  // NOISE
            return h + hash_noise(px, py, 91u) * s;
        }
        case 6u: {                                                  // INFLATE
            let t = max(1.0 - dist / r, 0.0);
            return h + s * t;
        }
        case 7u: {                                                  // PLATEAU_STAMP
            let t = max(1.0 - dist / r, 0.0);
            let plateau = max(header.target_height, h + abs(s));
            return h + (plateau - h) * w * t;
        }
        case 8u: {                                                  // CRATER_STAMP
            let t = dist / r;
            if (t >= 1.0) {
                return h;
            }
            if (t < 0.55) {
                return h - abs(s) * (1.0 - t / 0.55) * w;
            }
            let rim = clamp((t - 0.55) / 0.45, 0.0, 1.0);
            let bump = max(1.0 - abs(rim - 0.5) * 2.0, 0.0);
            return h + abs(s) * 0.45 * bump * w;
        }
        case 9u: {                                                  // HEIGHT_STAMP
            return h + (header.target_height - h) * w;
        }
        case 10u: {                                                 // ERODE / EncourageErosion
            return h - abs(s) * 0.35;
        }
        default: { return h; }                                      // AUX_NOOP
    }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) { return; }
    let p = vec2<i32>(i32(gid.x), i32(gid.y));
    let dx = u.world_x / f32(u.width);
    let dz = u.world_z / f32(u.height);
    let wx = (f32(gid.x) + 0.5) * dx;
    let wz = (f32(gid.y) + 0.5) * dz;

    var h = textureLoad(src, p, 0).r;
    var edited = 0.0;
    for (var si = 0u; si < u.stroke_count; si = si + 1u) {
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
        h = apply_kind(header, h, dist, weight, p.x, p.y);
    }

    textureStore(stamp_out, p, vec4<f32>(h, 0.0, 0.0, 0.0));
    textureStore(edited_out, p, vec4<f32>(edited, 0.0, 0.0, 0.0));
}
