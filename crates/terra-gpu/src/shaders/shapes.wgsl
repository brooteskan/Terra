// Dedicated Mountains / Dunes / Canyons / Uplift generators (preview; CPU remains bake reference).

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
    ridge_sharpness: f32,
    range_angle: f32,
    range_width: f32,
    wave_frequency: f32,
    asymmetry: f32,
    depth: f32,
    canyon_width: f32,
    meander: f32,
    shape_mode: u32, // 0 mountains, 1 dunes, 2 canyons, 3 uplift, 4 volcano, 5 mesa, 6 legacy island
    _pad: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn smooth01(t0: f32) -> f32 {
    let t = clamp(t0, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

fn legacy_hash2_seeded(p: vec2<u32>, seed: u32) -> f32 {
    var n = p.x * 374761393u + p.y * 668265263u + seed * 2246822519u;
    n = (n ^ (n >> 13u)) * 1274126177u;
    n = n ^ (n >> 16u);
    return f32(n & 0x00FFFFFFu) / f32(0x01000000u);
}

fn legacy_grad(h: f32, x: f32, y: f32) -> f32 {
    let ang = h * 6.2831853;
    return cos(ang) * x + sin(ang) * y;
}

fn legacy_perlin_seeded(p: vec2<f32>, seed: u32) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = fract(p);
    let aa = legacy_hash2_seeded(vec2<u32>(u32(i.x), u32(i.y)), seed);
    let ba = legacy_hash2_seeded(vec2<u32>(u32(i.x + 1), u32(i.y)), seed);
    let ab = legacy_hash2_seeded(vec2<u32>(u32(i.x), u32(i.y + 1)), seed);
    let bb = legacy_hash2_seeded(vec2<u32>(u32(i.x + 1), u32(i.y + 1)), seed);
    let ux = fade(f.x);
    let uy = fade(f.y);
    let x1 = mix(legacy_grad(aa, f.x, f.y), legacy_grad(ba, f.x - 1.0, f.y), ux);
    let x2 = mix(legacy_grad(ab, f.x, f.y - 1.0), legacy_grad(bb, f.x - 1.0, f.y - 1.0), ux);
    return mix(x1, x2, uy);
}

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

fn cpu_grad(h: u32, x: f32, z: f32) -> f32 {
    switch (h & 3u) {
        case 0u: { return x + z; }
        case 1u: { return -x + z; }
        case 2u: { return x - z; }
        default: { return -x - z; }
    }
}

fn perlin_seeded(p: vec2<f32>, seed: u32) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = p - vec2<f32>(f32(i.x), f32(i.y));
    let ux = fade(f.x);
    let uy = fade(f.y);
    let g00 = cpu_grad(cpu_hash2(i.x, i.y, seed), f.x, f.y);
    let g10 = cpu_grad(cpu_hash2(i.x + 1, i.y, seed), f.x - 1.0, f.y);
    let g01 = cpu_grad(cpu_hash2(i.x, i.y + 1, seed), f.x, f.y - 1.0);
    let g11 = cpu_grad(cpu_hash2(i.x + 1, i.y + 1, seed), f.x - 1.0, f.y - 1.0);
    let nx0 = g00 + (g10 - g00) * ux;
    let nx1 = g01 + (g11 - g01) * ux;
    return (nx0 + (nx1 - nx0) * uy) * 1.4142135623730951;
}

/// CPU-style ridged MF (weight cascade, octave seed stride 9173). Returns ≈ [0, amp].
fn ridged_mf(x: f32, z: f32) -> f32 {
    var amp = 1.0;
    var freq = u.frequency;
    var sum = 0.0;
    var norm = 0.0;
    var weight = 1.0;
    let oct = min(u.octaves, 12u);
    for (var o = 0u; o < oct; o++) {
        let ox = (x + u.offset_x) * freq;
        let oz = (z + u.offset_z) * freq;
        var n = perlin_seeded(vec2<f32>(ox, oz), u.seed + o * 9173u);
        n = 1.0 - abs(n);
        n = n * n;
        n = n * weight;
        weight = clamp(n * 2.0, 0.0, 1.0);
        sum += n * amp;
        norm += amp;
        amp *= u.persistence;
        freq *= u.lacunarity;
    }
    if (norm <= 0.0) {
        return 0.0;
    }
    return clamp(sum / norm, 0.0, 1.0) * u.amplitude;
}
/// Ridged MF with explicit amplitude scale (uplift primary uses amp=1 then * u.amplitude).
fn ridged_mf_n(x: f32, z: f32, seed: u32, freq0: f32, amp0: f32, octaves: u32) -> f32 {
    var amp = 1.0;
    var freq = freq0;
    var sum = 0.0;
    var norm = 0.0;
    var weight = 1.0;
    let oct = min(octaves, 12u);
    for (var o = 0u; o < oct; o++) {
        var n = perlin_seeded(vec2<f32>(x * freq, z * freq), seed + o * 9173u);
        n = 1.0 - abs(n);
        n = n * n;
        n = n * weight;
        weight = clamp(n * 2.0, 0.0, 1.0);
        sum += n * amp;
        norm += amp;
        amp *= 0.45;
        freq *= 2.1;
    }
    if (norm <= 0.0) {
        return 0.0;
    }
    return clamp(sum / norm, 0.0, 1.0) * amp0;
}
/// CPU-style fBm with remap to [-1,1] then * amplitude.
fn fbm_amp(x: f32, z: f32) -> f32 {
    var amp = 1.0;
    var freq = u.frequency;
    var sum = 0.0;
    var norm = 0.0;
    let oct = min(u.octaves, 12u);
    for (var o = 0u; o < oct; o++) {
        let ox = (x + u.offset_x) * freq;
        let oz = (z + u.offset_z) * freq;
        let n = perlin_seeded(vec2<f32>(ox, oz), u.seed + o * 1013u);
        sum += n * amp;
        norm += amp;
        amp *= u.persistence;
        freq *= u.lacunarity;
    }
    var v = 0.0;
    if (norm > 0.0) {
        v = sum / norm;
    }
    // Default remap [-1,1] → identity for typical NoiseParams.
    return v * u.amplitude;
}

fn fbm_detail(x: f32, z: f32, seed: u32, freq0: f32, amp0: f32, octaves: u32) -> f32 {
    var amp = 1.0;
    var freq = freq0;
    var sum = 0.0;
    var norm = 0.0;
    let oct = min(octaves, 8u);
    for (var o = 0u; o < oct; o++) {
        let n = perlin_seeded(vec2<f32>((x + 11.0) * freq, (z - 3.0) * freq), seed + o * 1013u);
        sum += n * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    var v = 0.0;
    if (norm > 0.0) {
        v = sum / norm;
    }
    return v * amp0;
}

fn mountains(x: f32, z: f32) -> f32 {
    let world_scale = max(min(u.world_x, u.world_z), 1.0);
    let amplitude = max(u.amplitude, 0.0);
    let nx = x / u.world_x - 0.5;
    let nz = z / u.world_z - 0.5;
    let ca = cos(u.range_angle);
    let sa = sin(u.range_angle);
    let axis = nx * ca + nz * sa;
    let cross_axis = nx * (-sa) + nz * ca;

    let warp_freq = max(u.frequency, 1e-6) * 0.31;
    let warp_cross = perlin_seeded(vec2<f32>(x * warp_freq, z * warp_freq), u.seed ^ 0x7A11CE55u) * 0.085;
    let warp_axis = perlin_seeded(
        vec2<f32>(x * warp_freq + 17.3, z * warp_freq - 9.1),
        u.seed ^ 0x51D390AFu,
    ) * 0.035;
    let warped_axis = axis + warp_axis;
    let warped_cross = cross_axis + warp_cross;
    let corridor_t = 1.0 - abs(warped_cross) / max(u.range_width, 1e-3);
    let range_mask = pow(smooth01(corridor_t), 1.18);
    let ridge_x = warped_axis * world_scale;
    let ridge_z = warped_cross * world_scale;

    let macro_ridge = ridged_mf_n(
        ridge_x,
        ridge_z,
        u.seed,
        max(u.frequency, 1e-6) * 0.52,
        1.0,
        clamp(u.octaves, 2u, 4u),
    );
    var ridge = 0.0;
    if (amplitude > 1e-6) {
        ridge = clamp(ridged_mf(ridge_x, ridge_z) / amplitude, 0.0, 1.0);
    }
    let macro_bias = 0.5 + 0.5 * perlin_seeded(
        vec2<f32>(x * warp_freq * 0.63 + 3.7, z * warp_freq * 0.63 - 5.2),
        u.seed ^ 0xA8314F29u,
    );
    let structure = pow(
        clamp(ridge * 0.72 + clamp(macro_ridge, 0.0, 1.0) * 0.28, 0.0, 1.0),
        max(u.ridge_sharpness, 0.1),
    );
    let foothills = macro_bias * pow(range_mask, 1.8) * 0.13;
    let primary = amplitude * range_mask * (foothills + structure * 0.87);
    var elev_t = 0.0;
    if (amplitude > 1e-6) {
        elev_t = clamp(primary / amplitude, 0.0, 1.0);
    }
    let fine = ridged_mf_n(
        ridge_x,
        ridge_z,
        u.seed ^ 0xC0DEu,
        max(u.frequency, 1e-6) * 4.5,
        1.0,
        4u,
    );
    let detail = (fine - 0.42) * 2.0 * max(u.meander, 0.0) * pow(elev_t, 0.68) * range_mask;
    return max(primary + detail, 0.0);
}
fn dunes(x: f32, z: f32) -> f32 {
    // Preview approx: directional wave bands + asymmetry (CPU runs full aeolian relax).
    let amplitude = max(u.amplitude, 0.0);
    let trough = max(u.depth, 0.0);
    let floor_h = max(clamp(u.canyon_width, 0.0, 1.0) * amplitude, 0.0);
    let freq = max(u.wave_frequency, 1e-5);
    let ang = u.range_angle * 0.01745329252;
    let c = cos(ang);
    let s = sin(ang);
    let along = x * c + z * s;
    let across = -x * s + z * c;
    let lin = clamp(u.range_width, 0.0, 1.0);
    let sharp = clamp(u.asymmetry, 0.0, 1.0);
    let macro_n = fbm_detail(
        x,
        z,
        u.seed,
        max(u.frequency, 1e-6) * 0.42,
        1.0,
        clamp(u.octaves, 2u, 4u),
    );
    let macro01 = clamp(0.5 + 0.5 * macro_n, 0.0, 1.0);
    let warp = perlin_seeded(
        vec2<f32>(across * freq * 0.16, along * freq * 0.11),
        u.seed ^ 0xD0AE7711u,
    ) * (1.7 - lin);
    let tau = 6.2831853;
    var phase = (along * freq + warp) % tau;
    if (phase < 0.0) {
        phase += tau;
    }
    phase /= tau;
    let slip = 0.55 + sharp * 0.3;
    var primary = 0.0;
    if (phase < slip) {
        primary = pow(smooth01(phase / slip), 0.82);
    } else {
        primary = 1.0 - pow(smooth01((phase - slip) / max(1.0 - slip, 1e-3)), 0.55);
    }
    // Low linearity mixes a cross-wind lobe (star-like interference).
    var secondary = 0.0;
    if (lin < 0.85) {
        let ang2 = ang + 1.5707963 * (1.0 - lin);
        let c2 = cos(ang2);
        let s2 = sin(ang2);
        let along2 = x * c2 + z * s2;
        var p2 = (along2 * freq * 0.92) % tau;
        if (p2 < 0.0) {
            p2 += tau;
        }
        p2 /= tau;
        secondary = (0.5 - 0.5 * cos(p2 * tau)) * (1.0 - lin) * 0.55;
    }
    let profile = clamp(primary * (0.65 + 0.35 * lin) + secondary, 0.0, 1.0);
    let height = floor_h
        + amplitude * (0.055 + profile * (0.38 + 0.57 * macro01))
        - trough * pow(1.0 - profile, 1.35) * (0.55 + 0.45 * macro01);
    return max(height, floor_h + amplitude * 0.018 * macro01);
}
fn canyons(x: f32, z: f32) -> f32 {
    let base_width = max(u.canyon_width, 1e-3);
    let longitudinal_freq = 1.0 / max(base_width * 11.0, 1.0);
    let seed_f = f32(u.seed);
    let broad = perlin_seeded(vec2<f32>(z * longitudinal_freq, seed_f * 0.017), u.seed);
    let detail = perlin_seeded(
        vec2<f32>(z * longitudinal_freq * 3.1, seed_f * 0.043),
        u.seed ^ 0x9E37u,
    );
    let meander_n = broad * 0.72 + detail * 0.28;
    let center = u.world_x * 0.5 + meander_n * clamp(u.meander, 0.0, 2.0) * base_width * 2.35;
    let width_noise = perlin_seeded(
        vec2<f32>(z * longitudinal_freq * 0.83 + 13.0, seed_f * 0.007),
        u.seed ^ 0x7F4A71C2u,
    );
    let local_width = base_width * max(1.0 + width_noise * 0.28, 0.45);
    let wall_noise = perlin_seeded(
        vec2<f32>(x / local_width * 0.63, z / local_width * 0.47),
        u.seed ^ 0xC4110A55u,
    );
    let q = max(abs(x - center) / local_width + wall_noise * 0.055, 0.0);
    if (q >= 1.0) {
        return 0.0;
    }
    let wall = pow(smooth01(1.0 - q), 0.72);
    let inner = smooth01(1.0 - q / 0.19);
    let depth_noise = perlin_seeded(
        vec2<f32>(z * longitudinal_freq * 0.57 - 5.0, 8.0),
        u.seed ^ 0xD3371A11u,
    );
    let local_depth = max(u.depth, 0.0) * max(0.88 + depth_noise * 0.16, 0.5);
    let strata = 0.96 + 0.04 * sin(q * 9.0 * 6.2831853 + wall_noise);
    return -local_depth * (wall * 0.78 + inner * 0.22) * strata;
}
/// Schott-inspired corridor uplift (authoring only — not SPE).
/// Uniform mapping: ridge_sharpness=power, range_width=corridor, meander=warp,
/// wave_frequency=detail_freq, asymmetry=altitude_fade, depth=detail_amp, octaves=detail octaves.
fn uplift(x: f32, z: f32) -> f32 {
    let amp = max(u.amplitude, 0.0);
    let corr_w = max(u.range_width, 1e-3);
    let fade = clamp(u.asymmetry, 0.0, 1.0);
    let wx = perlin_seeded(vec2<f32>(x * u.frequency * 0.35, z * u.frequency * 0.35), u.seed)
        * u.meander * u.world_x * 0.08;
    let wz = perlin_seeded(
        vec2<f32>(x * u.frequency * 0.35 + 19.0, z * u.frequency * 0.35 + 7.0),
        u.seed ^ 0xA5A5u,
    ) * u.meander * u.world_z * 0.08;
    let xw = x + wx;
    let zw = z + wz;
    let nx = xw / u.world_x - 0.5;
    let nz = zw / u.world_z - 0.5;
    let ca = cos(u.range_angle);
    let sa = sin(u.range_angle);
    let axis = nx * ca + nz * sa;
    let cross = nx * (-sa) + nz * ca;
    let corridor = pow(1.0 - clamp(abs(cross) / corr_w, 0.0, 1.0), 1.35);
    let ridge_x = axis * u.world_x;
    let ridge_z = cross * u.world_z;
    let ridge = clamp(ridged_mf_n(ridge_x, ridge_z, u.seed, u.frequency, 1.0, 3u), 0.0, 1.0);
    let primary = corridor * pow(ridge, max(u.ridge_sharpness, 0.1)) * amp;
    let detail = fbm_detail(x, z, u.seed ^ 0xC0FFEEu, max(u.wave_frequency, 1e-6), max(u.depth, 0.0), u.octaves);
    var elev_t = 0.0;
    if (amp > 1e-6) {
        elev_t = clamp(primary / amp, 0.0, 1.0);
    }
    let detail_scale = (1.0 - fade) + fade * elev_t;
    return primary + detail * detail_scale;
}

/// Radial cone + crater (CPU volcano authoring primitive).
/// Mapping: amplitude=height, range_width=radius frac, ridge_sharpness=flank_power,
/// canyon_width=crater_radius frac, depth=crater_depth, meander=roughness,
/// offset_x/z = center_u/v.
fn shape_perlin(p: vec2<f32>, seed: u32, legacy: bool) -> f32 {
    if (legacy) {
        return legacy_perlin_seeded(p, seed);
    }
    return perlin_seeded(p, seed);
}

fn volcano_impl(x: f32, z: f32, legacy: bool) -> f32 {
    let short_axis = max(min(u.world_x, u.world_z), 1.0);
    let radius_m = clamp(u.range_width, 0.01, 1.0) * short_axis * 0.5;
    let cx = clamp(u.offset_x, 0.0, 1.0) * u.world_x;
    let cz = clamp(u.offset_z, 0.0, 1.0) * u.world_z;
    let height = max(u.amplitude, 0.0);
    let flank = max(u.ridge_sharpness, 0.2);
    let crater_r = max(clamp(u.canyon_width, 0.0, 0.95) * radius_m, 0.0);
    let crater_d = max(u.depth, 0.0);
    let rough = max(u.meander, 0.0);
    let dx = x - cx;
    let dz = z - cz;
    let dist = sqrt(dx * dx + dz * dz);
    let theta = atan2(dz, dx);
    let macro_freq = 3.2 / max(radius_m, 1.0);
    let macro_noise = shape_perlin(
        vec2<f32>(x * macro_freq, z * macro_freq),
        u.seed ^ 0xA17C31D5u,
        legacy,
    );
    let lobe = sin(theta * 5.0 + macro_noise * 1.7);
    let footprint = max(1.0 + macro_noise * 0.09 + lobe * 0.035, 0.72);
    let radial = dist / (radius_m * footprint);
    if (radial >= 1.0) {
        return 0.0;
    }

    let t = smooth01(1.0 - radial);
    var h = height * pow(t, flank);
    if (rough > 0.0) {
        let fine = shape_perlin(
            vec2<f32>(x * macro_freq * 4.7 + 11.0, z * macro_freq * 4.7 - 7.0),
            u.seed ^ 0xC011A953u,
            legacy,
        );
        let groove_phase = theta * 11.0 + macro_noise * 4.2 + fine * 0.9;
        let groove = pow(1.0 - abs(sin(groove_phase)), 5.0);
        let flank_gate = smooth01(radial / 0.22) * smooth01((1.0 - radial) / 0.18);
        h += fine * rough * t * 0.72;
        h -= groove * rough * 1.35 * flank_gate;
    }

    if (crater_r > 1e-3) {
        let crater_norm = crater_r / radius_m;
        let rim_sigma = max(crater_norm * 0.18, 0.012);
        let rim_d = (radial - crater_norm) / rim_sigma;
        let rim = exp(-(rim_d * rim_d));
        h += rim * crater_d * 0.22;
        if (radial < crater_norm) {
            let bowl = smooth01(1.0 - radial / crater_norm);
            let rim_profile = height * pow(smooth01(1.0 - crater_norm), flank);
            let crater_floor = max(rim_profile - crater_d, 0.0);
            let bowl_blend = pow(bowl, 1.35);
            h = mix(h, crater_floor, bowl_blend);
        }
    }
    return max(h, 0.0);
}

fn volcano(x: f32, z: f32) -> f32 {
    return volcano_impl(x, z, false);
}

fn legacy_volcanic_island(x: f32, z: f32) -> f32 {
    return volcano_impl(x, z, true);
}
/// Hard-cap mesa / butte.
/// Mapping: amplitude=height, range_width=radius frac, ridge_sharpness=edge_steepness,
/// meander=soft skirt frac, depth=cap_noise, offset_x/z = center_u/v.
fn mesa(x: f32, z: f32) -> f32 {
    let short_axis = max(min(u.world_x, u.world_z), 1.0);
    let radius_m = clamp(u.range_width, 0.02, 1.0) * short_axis * 0.5;
    let cx = clamp(u.offset_x, 0.0, 1.0) * u.world_x;
    let cz = clamp(u.offset_z, 0.0, 1.0) * u.world_z;
    let height = max(u.amplitude, 0.0);
    let steep = max(u.ridge_sharpness, 1.0);
    let skirt = max(clamp(u.meander, 0.0, 1.0) * radius_m, 1e-3);
    let outer = radius_m + skirt;
    let cap_noise = max(u.depth, 0.0);
    let cap_r = radius_m * 0.82;
    let dx = x - cx;
    let dz = z - cz;
    let dist = sqrt(dx * dx + dz * dz);
    let theta = atan2(dz, dx);
    let footprint_freq = 2.8 / max(radius_m, 1.0);
    let edge_noise = perlin_seeded(
        vec2<f32>(x * footprint_freq, z * footprint_freq),
        u.seed ^ 0xE06351D3u,
    );
    let lobes = sin(theta * 4.0 + edge_noise) * 0.045
        + sin(theta * 7.0 - edge_noise * 0.7) * 0.025;
    let edge_scale = max(1.0 + edge_noise * 0.075 + lobes, 0.72);
    let warped_dist = dist / edge_scale;
    if (warped_dist >= outer) {
        return 0.0;
    }

    var cap_detail = 0.0;
    if (cap_noise > 0.0) {
        let broad = perlin_seeded(vec2<f32>(x * 0.0031, z * 0.0031), u.seed);
        let fine = perlin_seeded(
            vec2<f32>(x * 0.0097 + 13.0, z * 0.0097 - 7.0),
            u.seed ^ 0xA7115CA1u,
        );
        cap_detail = (broad * 0.72 + fine * 0.28) * cap_noise;
    }
    if (warped_dist <= cap_r) {
        return max(height + cap_detail * 0.85, 0.0);
    }
    if (warped_dist <= radius_m) {
        let t = clamp((warped_dist - cap_r) / max(radius_m - cap_r, 1e-3), 0.0, 1.0);
        let wall = pow(1.0 - t, steep);
        let strata = 0.95 + 0.05 * sin(wall * 11.0 * 6.2831853 + edge_noise * 1.8);
        return max(height * wall * strata + cap_detail * wall * 0.25, 0.0);
    }
    let t = clamp((warped_dist - radius_m) / skirt, 0.0, 1.0);
    let talus_noise = 0.82 + 0.18 * perlin_seeded(
        vec2<f32>(x * footprint_freq * 5.0, z * footprint_freq * 5.0),
        u.seed ^ 0x7A1A5001u,
    );
    let talus = pow(1.0 - smooth01(t), 1.22);
    return max(height * 0.13 * talus * talus_noise, 0.0);
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) { return; }
    let uv = vec2<f32>(
        (f32(gid.x) + 0.5) / f32(u.width),
        (f32(gid.y) + 0.5) / f32(u.height),
    );
    let x = uv.x * u.world_x;
    let z = uv.y * u.world_z;
    var h = 0.0;
    if (u.shape_mode == 0u) {
        h = mountains(x, z);
    } else if (u.shape_mode == 1u) {
        h = dunes(x, z);
    } else if (u.shape_mode == 2u) {
        h = canyons(x, z);
    } else if (u.shape_mode == 3u) {
        h = uplift(x, z);
    } else if (u.shape_mode == 4u) {
        h = volcano(x, z);
    } else if (u.shape_mode == 5u) {
        h = mesa(x, z);
    } else {
        h = legacy_volcanic_island(x, z);
    }
    textureStore(dst, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(h, 0.0, 0.0, 0.0));
}
