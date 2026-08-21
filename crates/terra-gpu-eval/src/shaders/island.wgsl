struct Uniforms {
    width: u32,
    height: u32,
    world_x: f32,
    world_z: f32,
    seed: u32,
    archetype: u32, // 1 archipelago, 2 atoll
    _pad_u0: u32,
    _pad_u1: u32,
    center_u: f32,
    center_v: f32,
    rotation_deg: f32,
    radius: f32,
    aspect: f32,
    sea_level: f32,
    ocean_floor: f32,
    mountain_height: f32,
    shelf_width: f32,
    shelf_depth: f32,
    beach_width: f32,
    beach_height: f32,
    reef_width: f32,
    reef_depth: f32,
    coastline_warp: f32,
    coastline_frequency: f32,
    mountain_power: f32,
    ridge_strength: f32,
    ridge_frequency: f32,
    lagoon_radius: f32,
    _pad_f0: f32,
    _pad_f1: f32,
    _pad_f2: f32,
    _pad_f3: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;

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

fn smooth01(t0: f32) -> f32 {
    let t = clamp(t0, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

fn grad(h: u32, x: f32, z: f32) -> f32 {
    switch (h & 3u) {
        case 0u: { return x + z; }
        case 1u: { return -x + z; }
        case 2u: { return x - z; }
        default: { return -x - z; }
    }
}

fn perlin(p: vec2<f32>, seed: u32) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = p - vec2<f32>(f32(i.x), f32(i.y));
    let fx = fade(f.x);
    let fz = fade(f.y);
    let g00 = grad(hash2(i.x, i.y, seed), f.x, f.y);
    let g10 = grad(hash2(i.x + 1, i.y, seed), f.x - 1.0, f.y);
    let g01 = grad(hash2(i.x, i.y + 1, seed), f.x, f.y - 1.0);
    let g11 = grad(hash2(i.x + 1, i.y + 1, seed), f.x - 1.0, f.y - 1.0);
    let nx0 = g00 + (g10 - g00) * fx;
    let nx1 = g01 + (g11 - g01) * fx;
    return (nx0 + (nx1 - nx0) * fz) * 1.4142135623730951;
}

fn ridged(x: f32, z: f32, seed: u32, frequency: f32) -> f32 {
    var amp = 1.0;
    var freq = frequency;
    var sum = 0.0;
    var norm = 0.0;
    var weight = 1.0;
    for (var octave = 0u; octave < 5u; octave++) {
        let n = perlin(vec2<f32>(x * freq, z * freq), seed + octave * 9173u);
        let ridge = clamp(1.0 - abs(n), 0.0, 1.0);
        let signal = ridge * ridge * weight;
        weight = clamp(signal * 2.0, 0.0, 1.0);
        sum += signal * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.05;
    }
    return clamp(sum / max(norm, 1e-6), 0.0, 1.0);
}

fn detail_fbm(x: f32, z: f32, seed: u32, frequency: f32) -> f32 {
    var amp = 1.0;
    var freq = frequency;
    var sum = 0.0;
    var norm = 0.0;
    for (var octave = 0u; octave < 3u; octave++) {
        let n = perlin(
            vec2<f32>((x + 7.0) * freq, (z - 13.0) * freq),
            seed + octave * 1013u,
        );
        sum += n * amp;
        norm += amp;
        amp *= 0.48;
        freq *= 2.13;
    }
    return sum / max(norm, 1e-6);
}

fn island_height(x: f32, z: f32) -> f32 {
    let short_axis = max(min(u.world_x, u.world_z), 1.0);
    let base_radius = clamp(u.radius, 0.08, 0.94) * short_axis * 0.5;
    let aspect_root = sqrt(clamp(u.aspect, 0.35, 2.85));
    let radius_x = base_radius * aspect_root;
    let radius_z = base_radius / aspect_root;
    let cx = clamp(u.center_u, 0.05, 0.95) * u.world_x;
    let cz = clamp(u.center_v, 0.05, 0.95) * u.world_z;
    let angle = u.rotation_deg * 0.017453292519943295;
    let ca = cos(angle);
    let sa = sin(angle);
    let dx = x - cx;
    let dz = z - cz;
    let xr = dx * ca + dz * sa;
    let zr = -dx * sa + dz * ca;
    let coast_freq = max(u.coastline_frequency, 1e-6);
    let coast_noise = perlin(vec2<f32>(x * coast_freq, z * coast_freq), u.seed) * 0.68
        + perlin(
            vec2<f32>(x * coast_freq * 2.17 + 31.0, z * coast_freq * 2.17 - 19.0),
            u.seed ^ 0xA17C9E5Du,
        ) * 0.22;
    let theta = atan2(zr / radius_z, xr / radius_x);
    let seed_phase = f32(u.seed);
    let lobes = sin(theta * 3.0 + seed_phase * 0.013) * 0.22
        + sin(theta * 5.0 - seed_phase * 0.007) * 0.12;
    let edge_scale = max(1.0 + clamp(u.coastline_warp, 0.0, 0.42) * (coast_noise + lobes), 0.58);
    var rho = sqrt((xr / radius_x) * (xr / radius_x) + (zr / radius_z) * (zr / radius_z)) / edge_scale;

    if (u.archetype == 1u) {
        let lobe_r = 0.58;
        let offsets = array<vec3<f32>, 2>(vec3<f32>(-0.42, 0.16, 0.62), vec3<f32>(0.38, -0.2, 0.54));
        for (var index = 0u; index < 2u; index++) {
            let item = offsets[index];
            let lx = xr / radius_x - item.x;
            let lz = zr / radius_z - item.y;
            let lr = sqrt(
                (lx / (item.z * lobe_r)) * (lx / (item.z * lobe_r))
                + (lz / (item.z / lobe_r)) * (lz / (item.z / lobe_r)),
            );
            rho = min(rho, lr / max(edge_scale, 0.7));
        }
    }

    var signed_distance = (rho - 1.0) * base_radius;
    if (u.archetype == 2u) {
        signed_distance = max(signed_distance, (clamp(u.lagoon_radius, 0.2, 0.78) - rho) * base_radius);
    }

    let cell = max(u.world_x / f32(u.width), u.world_z / f32(u.height));
    let shelf_width = max(u.shelf_width, cell * 2.0);
    let beach_width = max(u.beach_width, cell);
    let reef_width = max(u.reef_width, cell);
    let reef_center = min(shelf_width * 0.42, max(shelf_width - reef_width * 0.35, 0.0));
    let reef_sigma = max(reef_width * 0.42, cell);
    let ocean_floor = min(u.ocean_floor, u.sea_level - 1.0);

    if (signed_distance >= 0.0) {
        let outward = signed_distance;
        let shelf_t = clamp(outward / shelf_width, 0.0, 1.0);
        let reef_delta = (outward - reef_center) / reef_sigma;
        let reef_band = exp(-(reef_delta * reef_delta));
        let base_depth = max(u.shelf_depth, 1.0) * pow(shelf_t, 1.35);
        let reef_depth = max(u.reef_depth, 0.5);
        let shelf_height = u.sea_level - base_depth;
        let reef_height = u.sea_level - reef_depth - shelf_t * reef_depth * 0.35;
        if (outward <= shelf_width) {
            return max(shelf_height, reef_height * reef_band + shelf_height * (1.0 - reef_band));
        }
        let abyss_t = clamp((outward - shelf_width) / (shelf_width * 2.8), 0.0, 1.0);
        let abyss_smooth = abyss_t * abyss_t * (3.0 - 2.0 * abyss_t);
        return (u.sea_level - max(u.shelf_depth, 1.0)) * (1.0 - abyss_smooth) + ocean_floor * abyss_smooth;
    }

    let inward = -signed_distance;
    if (u.archetype == 2u) {
        let ring_t = clamp(inward / max(beach_width, 1.0), 0.0, 1.0);
        return u.sea_level + max(u.beach_height, 0.5) * sqrt(ring_t);
    }
    if (inward <= beach_width) {
        let t = clamp(inward / beach_width, 0.0, 1.0);
        return u.sea_level + max(u.beach_height, 0.5) * t * t * (3.0 - 2.0 * t);
    }

    let interior = clamp((inward - beach_width) / max(base_radius * 0.82 - beach_width, 1.0), 0.0, 1.0);
    let ridge_noise = ridged(x, z, u.seed ^ 0x51ADE771u, max(u.ridge_frequency, 1e-6));
    let spine_warp = perlin(
        vec2<f32>(x * coast_freq * 0.55 + 7.0, z * coast_freq * 0.55 - 11.0),
        u.seed ^ 0xC04FFEE1u,
    ) * radius_z * 0.18;
    let spine_denom = 2.0 * max(pow(radius_z * 0.24, 2.0), 1.0);
    let spine = exp(-pow(zr + spine_warp, 2.0) / spine_denom);
    let macro_mass = 0.5 + 0.5 * perlin(
        vec2<f32>(x * coast_freq * 0.38 + 3.0, z * coast_freq * 0.38 - 4.0),
        u.seed ^ 0xBA5171C5u,
    );
    let ridge_strength = clamp(u.ridge_strength, 0.0, 0.9);
    let ridge = clamp(
        (1.0 - ridge_strength) + ridge_strength * (ridge_noise * 0.38 + spine * 0.47 + macro_mass * 0.15),
        0.18,
        1.18,
    );
    let drainage_warp = perlin(
        vec2<f32>(x * coast_freq * 0.72, z * coast_freq * 0.72),
        u.seed ^ 0xD2A16E55u,
    ) * 3.2;
    let gully = pow(1.0 - abs(sin(theta * 9.0 + drainage_warp)), 6.0) * smooth01(interior);
    let mountain_height = max(u.mountain_height, 0.0);
    let uplift = pow(interior, max(u.mountain_power, 0.35));
    let detail = detail_fbm(x, z, u.seed ^ 0xD37A11EDu, max(u.ridge_frequency, 1e-6) * 3.4)
        * mountain_height * 0.055 * smooth01(interior);
    let base = u.sea_level + max(u.beach_height, 0.5);
    return max(
        base + mountain_height * uplift * ridge - mountain_height * 0.105 * gully * uplift + detail,
        base + mountain_height * interior * 0.012,
    );
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) { return; }
    let world = vec2<f32>(
        (f32(gid.x) + 0.5) / f32(u.width) * u.world_x,
        (f32(gid.y) + 0.5) / f32(u.height) * u.world_z,
    );
    let h = island_height(world.x, world.y);
    textureStore(dst, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(h, 0.0, 0.0, 0.0));
}
