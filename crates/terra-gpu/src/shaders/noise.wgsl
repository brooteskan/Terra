// Procedural height generation. Mode 0 preserves the shipped portable
// NoiseValue preview; modes 1-4 mirror the CPU oracle's arithmetic.

struct Uniforms {
    width: u32,
    height: u32,
    world_x: f32,
    world_z: f32,
    seed: u32,
    octaves: u32,
    frequency: f32,
    amplitude: f32,
    lacunarity: f32,
    persistence: f32,
    offset_x: f32,
    offset_z: f32,
    remap_min: f32,
    remap_max: f32,
    noise_type: u32, // 0 value, 1 perlin
    mode: u32,       // 0 legacy value, 1 perlin, 2 fbm, 3 ridged, 4 domain warp
    warp_strength: f32,
    warp_frequency: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;

// Existing value-noise approximation. Keep this path stable: NoiseValue is
// already admitted and owns a documented portable-hash parity budget.
fn legacy_hash2_seeded(p: vec2<u32>, seed: u32) -> f32 {
    var n = p.x * 374761393u + p.y * 668265263u + seed * 2246822519u;
    n = (n ^ (n >> 13u)) * 1274126177u;
    n = n ^ (n >> 16u);
    return f32(n & 0x00FFFFFFu) / f32(0x01000000u);
}

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn legacy_value_noise_seeded(p: vec2<f32>, seed: u32) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = fract(p);
    let a = legacy_hash2_seeded(vec2<u32>(u32(i.x), u32(i.y)), seed);
    let b = legacy_hash2_seeded(vec2<u32>(u32(i.x + 1), u32(i.y)), seed);
    let c = legacy_hash2_seeded(vec2<u32>(u32(i.x), u32(i.y + 1)), seed);
    let d = legacy_hash2_seeded(vec2<u32>(u32(i.x + 1), u32(i.y + 1)), seed);
    let ux = fade(f.x);
    let uy = fade(f.y);
    return mix(mix(a, b, ux), mix(c, d, ux), uy) * 2.0 - 1.0;
}

fn legacy_fbm(p0: vec2<f32>) -> f32 {
    var p = p0;
    var amplitude = 1.0;
    var sum = 0.0;
    var norm = 0.0;
    let octaves = min(u.octaves, 12u);
    for (var octave = 0u; octave < octaves; octave++) {
        let n = legacy_value_noise_seeded(p, u.seed + octave * 1013u);
        sum += n * amplitude;
        norm += amplitude;
        amplitude *= u.persistence;
        p *= u.lacunarity;
    }
    if (norm > 0.0) {
        sum /= norm;
    }
    return sum;
}

fn legacy_value_height(world: vec2<f32>) -> f32 {
    let p = (world + vec2<f32>(u.offset_x, u.offset_z)) * u.frequency;
    let n = legacy_fbm(p);
    let span = max(u.remap_max - u.remap_min, 1e-5);
    let t = clamp((n - u.remap_min) / span, 0.0, 1.0);
    return (t * 2.0 - 1.0) * u.amplitude;
}

// CPU-compatible hash and noise primitives (`terra_core::noise`). The planner
// admits only seed streams that remain representable as u32, so octave addition
// below cannot differ from the CPU's u64-derived canonical seed.
fn cpu_hash_u32(x0: u32) -> u32 {
    var x = x0;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

fn cpu_hash2(ix: i32, iz: i32, seed: u32) -> u32 {
    var h = seed;
    h ^= bitcast<u32>(ix);
    h = cpu_hash_u32(h);
    h ^= bitcast<u32>(iz);
    return cpu_hash_u32(h);
}

fn cpu_lerp(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

fn cpu_value_noise(p: vec2<f32>, seed: u32) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = p - vec2<f32>(f32(i.x), f32(i.y));
    let ux = fade(f.x);
    let uy = fade(f.y);
    let scale = f32(0xffffffffu);
    let n00 = f32(cpu_hash2(i.x, i.y, seed)) / scale * 2.0 - 1.0;
    let n10 = f32(cpu_hash2(i.x + 1, i.y, seed)) / scale * 2.0 - 1.0;
    let n01 = f32(cpu_hash2(i.x, i.y + 1, seed)) / scale * 2.0 - 1.0;
    let n11 = f32(cpu_hash2(i.x + 1, i.y + 1, seed)) / scale * 2.0 - 1.0;
    return cpu_lerp(cpu_lerp(n00, n10, ux), cpu_lerp(n01, n11, ux), uy);
}

fn cpu_gradient(h: u32, x: f32, z: f32) -> f32 {
    switch (h & 3u) {
        case 0u: { return x + z; }
        case 1u: { return -x + z; }
        case 2u: { return x - z; }
        default: { return -x - z; }
    }
}

fn cpu_perlin_noise(p: vec2<f32>, seed: u32) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = p - vec2<f32>(f32(i.x), f32(i.y));
    let ux = fade(f.x);
    let uy = fade(f.y);
    let g00 = cpu_gradient(cpu_hash2(i.x, i.y, seed), f.x, f.y);
    let g10 = cpu_gradient(cpu_hash2(i.x + 1, i.y, seed), f.x - 1.0, f.y);
    let g01 = cpu_gradient(cpu_hash2(i.x, i.y + 1, seed), f.x, f.y - 1.0);
    let g11 = cpu_gradient(cpu_hash2(i.x + 1, i.y + 1, seed), f.x - 1.0, f.y - 1.0);
    let nx0 = cpu_lerp(g00, g10, ux);
    let nx1 = cpu_lerp(g01, g11, ux);
    return cpu_lerp(nx0, nx1, uy) * 1.4142135623730951;
}

fn cpu_sample_noise(p: vec2<f32>, seed: u32, noise_type: u32) -> f32 {
    if (noise_type == 0u) {
        return cpu_value_noise(p, seed);
    }
    return cpu_perlin_noise(p, seed);
}

fn cpu_remap_height(value: f32) -> f32 {
    let span = max(u.remap_max - u.remap_min, 1e-6);
    let t = clamp((value - u.remap_min) / span, 0.0, 1.0);
    return (t * 2.0 - 1.0) * u.amplitude;
}

fn cpu_fbm_height(world: vec2<f32>, noise_type: u32) -> f32 {
    var amplitude = 1.0;
    var frequency = u.frequency;
    var sum = 0.0;
    var norm = 0.0;
    let octaves = max(u.octaves, 1u);
    for (var octave = 0u; octave < octaves; octave++) {
        let p = (world + vec2<f32>(u.offset_x, u.offset_z)) * frequency;
        let n = cpu_sample_noise(p, u.seed + octave * 1013u, noise_type);
        sum += n * amplitude;
        norm += amplitude;
        amplitude *= u.persistence;
        frequency *= u.lacunarity;
    }
    var value = 0.0;
    if (norm > 0.0) {
        value = sum / norm;
    }
    return cpu_remap_height(value);
}

fn cpu_ridged_height(world: vec2<f32>, noise_type: u32) -> f32 {
    var amplitude = 1.0;
    var frequency = u.frequency;
    var sum = 0.0;
    var norm = 0.0;
    var weight = 1.0;
    let octaves = max(u.octaves, 1u);
    for (var octave = 0u; octave < octaves; octave++) {
        let p = (world + vec2<f32>(u.offset_x, u.offset_z)) * frequency;
        let n = cpu_sample_noise(p, u.seed + octave * 9173u, noise_type);
        let ridge = clamp(1.0 - abs(n), 0.0, 1.0);
        let signal = ridge * ridge * weight;
        weight = clamp(signal * 2.0, 0.0, 1.0);
        sum += signal * amplitude;
        norm += amplitude;
        amplitude *= u.persistence;
        frequency *= u.lacunarity;
    }
    var normalized = 0.0;
    if (norm > 0.0) {
        normalized = sum / norm;
    }
    return clamp(normalized, 0.0, 1.0) * u.amplitude;
}

fn cpu_perlin_height(world: vec2<f32>) -> f32 {
    if (u.octaves <= 1u) {
        let p = (world + vec2<f32>(u.offset_x, u.offset_z)) * u.frequency;
        return cpu_perlin_noise(p, u.seed) * u.amplitude;
    }
    return cpu_fbm_height(world, 1u);
}

fn cpu_domain_warp_height(world: vec2<f32>) -> f32 {
    let offset = vec2<f32>(u.offset_x, u.offset_z);
    let warp_p = (world + offset) * u.warp_frequency;
    let wx = cpu_perlin_noise(warp_p, u.seed) * u.warp_strength;
    let wz = cpu_perlin_noise(warp_p + vec2<f32>(19.1, 7.3), u.seed + 1u)
        * u.warp_strength;
    return cpu_fbm_height(world + vec2<f32>(wx, wz), 1u);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) { return; }
    let uv = vec2<f32>(
        (f32(gid.x) + 0.5) / f32(u.width),
        (f32(gid.y) + 0.5) / f32(u.height),
    );
    let world = vec2<f32>(uv.x * u.world_x, uv.y * u.world_z);
    var height = 0.0;
    switch (u.mode) {
        case 0u: { height = legacy_value_height(world); }
        case 1u: { height = cpu_perlin_height(world); }
        case 2u: { height = cpu_fbm_height(world, u.noise_type); }
        case 3u: { height = cpu_ridged_height(world, u.noise_type); }
        case 4u: { height = cpu_domain_warp_height(world); }
        default: { height = 0.0; }
    }
    textureStore(dst, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(height, 0.0, 0.0, 0.0));
}
