// Interactive EffectFilter preview kernels (GPU-resident). CPU remains export oracle.
// mode selects the filter family; strength mixes toward the filtered result.

struct Uniforms {
    width: u32,
    height: u32,
    world_x: f32,
    world_z: f32,
    mode: u32,
    radius: u32,
    iterations: u32,
    seed: u32,
    strength: f32,
    amount: f32,
    frequency: f32,
    sea_level: f32,
    beach_width: f32,
    slope_min: f32,
    slope_max: f32,
    rock_hardness: f32,
    terrace_height: f32,
    terrace_offset: f32,
    rotation_deg: f32,
    anisotropy: f32,
    warp_strength: f32,
    warp_frequency: f32,
    dx: f32,
    invert: f32,
    flow_threshold: f32,
    wall_steepness: f32,
    valley_floor: f32,
    talus_mix: f32,
    top_smoothness: f32,
    riser_sharpness: f32,
    lacunarity: f32,
    persistence: f32,
    octaves: u32,
    voronoi_feature: u32,
    tileable: u32,
    _pad_params: u32,
    crater_radius: f32,
    dz: f32,
    _pad_metric0: f32,
    _pad_metric1: f32,
    // Dirty-region dispatch (full field when region_w/h == 0).
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
};

struct RangeState {
    min_ordered: u32,
    max_ordered: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<storage, read> range_state: RangeState;

fn ordered_to_float(ordered: u32) -> f32 {
    var bits: u32;
    if ((ordered & 0x80000000u) != 0u) {
        bits = ordered ^ 0x80000000u;
    } else {
        bits = ordered ^ 0xffffffffu;
    }
    return bitcast<f32>(bits);
}

fn hash2(p: vec2<u32>, seed: u32) -> f32 {
    var n = p.x * 374761393u + p.y * 668265263u + seed * 2246822519u;
    n = (n ^ (n >> 13u)) * 1274126177u;
    n = n ^ (n >> 16u);
    return f32(n & 0x00FFFFFFu) / f32(0x01000000u);
}

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn grad(h: f32, x: f32, y: f32) -> f32 {
    let ang = h * 6.2831853;
    return cos(ang) * x + sin(ang) * y;
}

fn perlin(p: vec2<f32>, seed: u32) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = fract(p);
    let aa = hash2(vec2<u32>(u32(i.x), u32(i.y)), seed);
    let ba = hash2(vec2<u32>(u32(i.x + 1), u32(i.y)), seed);
    let ab = hash2(vec2<u32>(u32(i.x), u32(i.y + 1)), seed);
    let bb = hash2(vec2<u32>(u32(i.x + 1), u32(i.y + 1)), seed);
    let ux = fade(f.x);
    let uy = fade(f.y);
    let x1 = mix(grad(aa, f.x, f.y), grad(ba, f.x - 1.0, f.y), ux);
    let x2 = mix(grad(ab, f.x, f.y - 1.0), grad(bb, f.x - 1.0, f.y - 1.0), ux);
    return mix(x1, x2, uy);
}

// CPU-compatible hash/noise primitives (`terra_core::noise`). Newly admitted
// EffectFilter kinds use these; the legacy helper above stays untouched for the
// already-ratcheted preview modes.
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

fn cpu_gradient(hash: u32, x: f32, z: f32) -> f32 {
    switch (hash & 3u) {
        case 0u: { return x + z; }
        case 1u: { return -x + z; }
        case 2u: { return x - z; }
        default: { return -x - z; }
    }
}

fn cpu_perlin(p: vec2<f32>, seed: u32) -> f32 {
    let cell = vec2<i32>(floor(p));
    let f = p - vec2<f32>(f32(cell.x), f32(cell.y));
    let ux = fade(f.x);
    let uz = fade(f.y);
    let g00 = cpu_gradient(cpu_hash2(cell.x, cell.y, seed), f.x, f.y);
    let g10 = cpu_gradient(cpu_hash2(cell.x + 1, cell.y, seed), f.x - 1.0, f.y);
    let g01 = cpu_gradient(cpu_hash2(cell.x, cell.y + 1, seed), f.x, f.y - 1.0);
    let g11 = cpu_gradient(cpu_hash2(cell.x + 1, cell.y + 1, seed), f.x - 1.0, f.y - 1.0);
    let x0 = g00 + (g10 - g00) * ux;
    let x1 = g01 + (g11 - g01) * ux;
    return (x0 + (x1 - x0) * uz) * 1.4142135623730951;
}

fn cpu_value(p: vec2<f32>, seed: u32) -> f32 {
    let cell = vec2<i32>(floor(p));
    let f = p - vec2<f32>(f32(cell.x), f32(cell.y));
    let ux = fade(f.x);
    let uz = fade(f.y);
    let scale = f32(0xffffffffu);
    let n00 = f32(cpu_hash2(cell.x, cell.y, seed)) / scale * 2.0 - 1.0;
    let n10 = f32(cpu_hash2(cell.x + 1, cell.y, seed)) / scale * 2.0 - 1.0;
    let n01 = f32(cpu_hash2(cell.x, cell.y + 1, seed)) / scale * 2.0 - 1.0;
    let n11 = f32(cpu_hash2(cell.x + 1, cell.y + 1, seed)) / scale * 2.0 - 1.0;
    let x0 = n00 + (n10 - n00) * ux;
    let x1 = n01 + (n11 - n01) * ux;
    return x0 + (x1 - x0) * uz;
}

fn cpu_fbm(world: vec2<f32>, use_value: bool) -> f32 {
    var amplitude = 1.0;
    var frequency = max(u.frequency, 1e-5);
    var sum = 0.0;
    var norm = 0.0;
    let octave_count = max(u.octaves, 1u);
    for (var octave = 0u; octave < octave_count; octave++) {
        let point = world * frequency;
        let octave_seed = u.seed + octave * 1013u;
        let value = select(cpu_perlin(point, octave_seed), cpu_value(point, octave_seed), use_value);
        sum += value * amplitude;
        norm += amplitude;
        amplitude *= clamp(u.persistence, 0.05, 0.95);
        frequency *= max(u.lacunarity, 1.01);
    }
    return select(0.0, sum / norm, norm > 0.0);
}

fn cpu_fbm_custom(world: vec2<f32>, seed: u32, octaves: u32, frequency_start: f32, lacunarity: f32, persistence: f32) -> f32 {
    var amplitude = 1.0;
    var frequency = frequency_start;
    var sum = 0.0;
    var norm = 0.0;
    for (var octave = 0u; octave < octaves; octave++) {
        sum += cpu_perlin(world * frequency, seed + octave * 1013u) * amplitude;
        norm += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    return sum / max(norm, 1e-20);
}

fn cpu_billow(world: vec2<f32>) -> f32 {
    var amplitude = 1.0;
    var frequency = max(u.frequency, 1e-5);
    var sum = 0.0;
    var norm = 0.0;
    for (var octave = 0u; octave < max(u.octaves, 1u); octave++) {
        sum += abs(cpu_perlin(world * frequency, u.seed + octave * 1301u)) * amplitude;
        norm += amplitude;
        amplitude *= clamp(u.persistence, 0.05, 0.95);
        frequency *= max(u.lacunarity, 1.01);
    }
    return sum / max(norm, 1e-20) * 2.0 - 1.0;
}

fn cpu_ridged_custom(world: vec2<f32>, seed: u32, octaves: u32, frequency_start: f32, lacunarity: f32, persistence: f32) -> f32 {
    var amplitude = 1.0;
    var frequency = frequency_start;
    var sum = 0.0;
    var norm = 0.0;
    var weight = 1.0;
    for (var octave = 0u; octave < octaves; octave++) {
        let noise = cpu_perlin(world * frequency, seed + octave * 9173u);
        let ridge = clamp(1.0 - abs(noise), 0.0, 1.0);
        let signal = ridge * ridge * weight;
        weight = clamp(signal * 2.0, 0.0, 1.0);
        sum += signal * amplitude;
        norm += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    return clamp(sum / max(norm, 1e-20), 0.0, 1.0);
}

fn cpu_world(i: i32, j: i32) -> vec2<f32> {
    return vec2<f32>((f32(i) + 0.5) * u.dx, (f32(j) + 0.5) * u.dz);
}

fn cpu_transform_world(world: vec2<f32>) -> vec2<f32> {
    let angle = u.rotation_deg * 0.017453292519943295;
    let sine = sin(angle);
    let cosine = cos(angle);
    var transformed = vec2<f32>(
        world.x * cosine + world.y * sine,
        -world.x * sine + world.y * cosine,
    );
    let anisotropy = max(u.anisotropy, 0.05);
    transformed.x /= anisotropy;
    transformed.y *= max(sqrt(anisotropy), 0.2);
    if (u.tileable != 0u) {
        let period = max(1.0 / max(u.frequency, 1e-5), 1.0);
        transformed = transformed - floor(transformed / period) * period;
    }
    if (abs(u.warp_strength) > 1e-5) {
        let warp_frequency = max(u.warp_frequency, 1e-5);
        let wx = cpu_perlin(transformed * warp_frequency, u.seed) * u.warp_strength;
        let wz = cpu_perlin(
            transformed * warp_frequency + vec2<f32>(19.1, 7.3),
            u.seed + 1u,
        ) * u.warp_strength;
        transformed += vec2<f32>(wx, wz);
    }
    return transformed;
}

fn smootherstep(value: f32) -> f32 {
    let t = clamp(value, 0.0, 1.0);
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn sample_h(i: i32, j: i32) -> f32 {
    let ii = clamp(i, 0, i32(u.width) - 1);
    let jj = clamp(j, 0, i32(u.height) - 1);
    return textureLoad(src, vec2<i32>(ii, jj), 0).r;
}

fn slope_deg(i: i32, j: i32) -> f32 {
    let dx = u.dx;
    let hx = sample_h(i + 1, j) - sample_h(i - 1, j);
    let hz = sample_h(i, j + 1) - sample_h(i, j - 1);
    let g = sqrt((hx * 0.5 / dx) * (hx * 0.5 / dx) + (hz * 0.5 / dx) * (hz * 0.5 / dx));
    return atan(g) * 57.2957795;
}

fn slope_gate(i: i32, j: i32) -> f32 {
    let s = slope_deg(i, j);
    if (s < u.slope_min || s > u.slope_max) {
        return 0.0;
    }
    return 1.0;
}

fn cpu_slope_forward(i: i32, j: i32) -> f32 {
    let center = sample_h(i, j);
    let gx = (sample_h(i + 1, j) - center) / max(u.dx, 1e-5);
    let gz = (sample_h(i, j + 1) - center) / max(u.dz, 1e-5);
    return atan(sqrt(gx * gx + gz * gz)) * 57.2957795;
}

fn median_exact9(i: i32, j: i32) -> f32 {
    var values: array<f32, 9>;
    var index = 0;
    for (var dj = -1; dj <= 1; dj++) {
        for (var di = -1; di <= 1; di++) {
            values[index] = sample_h(i + di, j + dj);
            index += 1;
        }
    }
    for (var right = 1; right < 9; right++) {
        let value = values[right];
        var left = right;
        loop {
            if (left <= 0 || values[left - 1] <= value) {
                break;
            }
            values[left] = values[left - 1];
            left -= 1;
        }
        values[left] = value;
    }
    return values[4];
}

fn box_blur(i: i32, j: i32, r: i32) -> f32 {
    var sum = 0.0;
    var count = 0.0;
    for (var dj = -r; dj <= r; dj++) {
        for (var di = -r; di <= r; di++) {
            sum += sample_h(i + di, j + dj);
            count += 1.0;
        }
    }
    return sum / max(count, 1.0);
}

fn median9(i: i32, j: i32) -> f32 {
    // Approximate spike removal: clamp to neighbor mid-range.
    var lo = sample_h(i, j);
    var hi = lo;
    for (var dj = -1; dj <= 1; dj++) {
        for (var di = -1; di <= 1; di++) {
            let v = sample_h(i + di, j + dj);
            lo = min(lo, v);
            hi = max(hi, v);
        }
    }
    let c = sample_h(i, j);
    let mid = (lo + hi) * 0.5;
    let span = max(hi - lo, 1e-4);
    // Pull outliers toward mid.
    if (abs(c - mid) > span * 0.35) {
        return mix(c, mid, 0.85);
    }
    return c;
}

fn apply_filter(i: i32, j: i32, h: f32) -> f32 {
    let r = i32(max(u.radius, 1u));
    let wx = (f32(i) + 0.5) / f32(u.width) * u.world_x;
    let wz = (f32(j) + 0.5) / f32(u.height) * u.world_z;
    var out_h = h;

    switch u.mode {
        case 0u: { // Smooth
            out_h = box_blur(i, j, r);
        }
        case 1u: { // Distortion / warp resample
            let freq = max(u.warp_frequency, u.frequency);
            let amp = max(u.warp_strength, u.amount);
            let ox = perlin(vec2<f32>(wx, wz) * freq, u.seed) * amp;
            let oz = perlin(vec2<f32>(wx + 17.0, wz - 9.0) * freq, u.seed + 91u) * amp;
            let si = i32(round(f32(i) + ox / max(u.dx, 1e-3)));
            let sj = i32(round(f32(j) + oz / max(u.dx, 1e-3)));
            out_h = sample_h(si, sj);
        }
        case 2u: { // SpikeRemoval
            out_h = median9(i, j);
        }
        case 3u: { // Shore
            let beach = max(u.beach_width, 1.0);
            let d = h - u.sea_level;
            if (d < beach) {
                let t = clamp(d / beach, 0.0, 1.0);
                out_h = mix(u.sea_level, h, t * t * (3.0 - 2.0 * t));
            }
        }
        case 4u: { // Denoise — edge-preserving bilateral. Mirrors the CPU
            // `filter_kernels::bilateral` with the sigma derivation from
            // `generators::effect_filter`'s Denoise arm: sigma_space = 0.55 * radius
            // (floored at 0.25, the kernel's own clamp), sigma_range = amount in
            // metres (floored at 0.5, the Denoise arm's `amount.max(0.5)`). The range
            // weight underflows to ~0 across a large depth step, so the shelf->basin
            // discontinuity is preserved. CPU remains the export oracle.
            let center = h;
            let ss = max(0.55 * f32(r), 0.25);
            let sr = max(u.amount, 0.5);
            var sum = 0.0;
            var wsum = 0.0;
            for (var dj = -r; dj <= r; dj++) {
                for (var di = -r; di <= r; di++) {
                    let v = sample_h(i + di, j + dj);
                    let ws = exp(-f32(di * di + dj * dj) / (2.0 * ss * ss));
                    let d = v - center;
                    let wr = exp(-(d * d) / (2.0 * sr * sr));
                    let wt = ws * wr;
                    sum += v * wt;
                    wsum += wt;
                }
            }
            out_h = sum / max(wsum, 1e-5);
        }
        case 5u: { // Inflate
            out_h = h + u.amount * u.strength;
        }
        case 6u: { // Deflate — greyscale morphological erosion, amount-limited
            var local_min = h;
            for (var dj = -r; dj <= r; dj++) {
                for (var di = -r; di <= r; di++) {
                    local_min = min(local_min, sample_h(i + di, j + dj));
                }
            }
            out_h = h - min(h - local_min, u.amount);
        }
        case 7u: { // Balloon — inflate gated by low slope
            let g = 1.0 - clamp(slope_deg(i, j) / 45.0, 0.0, 1.0);
            out_h = h + u.amount * g;
        }
        case 8u: { // TerraceSimple
            let step_h = max(u.terrace_height, abs(u.amount));
            let step = max(step_h, 1e-3);
            let phase = u.terrace_offset * step;
            out_h = floor((h - phase) / step + 0.5) * step + phase;
        }
        case 9u: { // Curve — contrast around mid
            let mid = (u.sea_level + h) * 0.5;
            let t = h - mid;
            out_h = mid + t * (1.0 + u.amount);
        }
        case 10u: { // Cutoff
            out_h = clamp(h, u.sea_level - u.amount, u.sea_level + u.amount * 4.0);
            if (u.amount < 0.0) {
                out_h = max(h, u.sea_level);
            }
        }
        case 11u: { // Rocky / rugged detail
            let gate = slope_gate(i, j);
            let n = perlin(vec2<f32>(wx, wz) * max(u.frequency, 0.001), u.seed);
            let hard = mix(1.2, 0.35, clamp(u.rock_hardness, 0.0, 1.0));
            out_h = h + n * u.amount * gate * hard;
        }
        case 12u: { // Noise additive family
            var sum = 0.0;
            var amp = 1.0;
            var freq = max(u.frequency, 0.0001);
            let ang = u.rotation_deg * 0.01745329252;
            let ca = cos(ang);
            let sa = sin(ang);
            let an = max(u.anisotropy, 0.01);
            for (var o = 0u; o < 4u; o++) {
                let rx = (wx * ca - wz * sa) * freq * an;
                let rz = (wx * sa + wz * ca) * freq;
                sum += perlin(vec2<f32>(rx, rz), u.seed + o * 9173u) * amp;
                freq *= 2.0;
                amp *= 0.5;
            }
            out_h = h + sum * u.amount;
        }
        case 13u: { // Directional / angle blur
            let ang = u.rotation_deg * 0.01745329252;
            let dir = vec2<f32>(cos(ang), sin(ang));
            var sum = 0.0;
            var count = 0.0;
            for (var k = -r; k <= r; k++) {
                let di = i32(round(dir.x * f32(k)));
                let dj = i32(round(dir.y * f32(k)));
                sum += sample_h(i + di, j + dj);
                count += 1.0;
            }
            out_h = sum / max(count, 1.0);
        }
        case 14u: { // ZeroEdge / BorderBlend
            let m = min(min(i, j), min(i32(u.width) - 1 - i, i32(u.height) - 1 - j));
            let border = max(r, 1);
            if (m < border) {
                let t = f32(m) / f32(border);
                out_h = mix(u.sea_level, h, t * t * (3.0 - 2.0 * t));
            }
        }
        case 15u: { // FlattenFilter toward sea_level
            out_h = mix(h, u.sea_level, clamp(u.amount, 0.0, 1.0));
        }
        case 16u: { // Strata / layers
            let step = max(abs(u.amount), 0.5);
            let q = floor(h / step);
            let n = perlin(vec2<f32>(wx, wz) * 0.02, u.seed) * 0.15 * step;
            out_h = q * step + n;
        }
        case 17u: { // Crater imprint
            let cx = u.world_x * 0.5;
            let cz = u.world_z * 0.5;
            let rad = max(u.beach_width, u.world_x * 0.05);
            let d = distance(vec2<f32>(wx, wz), vec2<f32>(cx, cz));
            if (d < rad) {
                let t = d / rad;
                let bowl = (1.0 - t * t) * u.amount;
                out_h = h - bowl;
            }
        }
        case 18u: { // Talus / sediment fill soft — blur toward lower
            let b = box_blur(i, j, r);
            out_h = max(h, mix(h, b, clamp(u.amount, 0.0, 1.0)));
        }
        case 19u: { // Sharpen / chipped
            let b = box_blur(i, j, r);
            out_h = h + (h - b) * u.amount;
        }
        case 20u: { // Kuwahara-ish — pick flatter quadrant mean
            let r2 = max(r / 2, 1);
            var best = h;
            var best_var = 1e9;
            let offsets = array<vec2<i32>, 4>(
                vec2<i32>(-r2, -r2),
                vec2<i32>(0, -r2),
                vec2<i32>(-r2, 0),
                vec2<i32>(0, 0)
            );
            for (var q = 0; q < 4; q++) {
                let o = offsets[q];
                var sum = 0.0;
                var sum2 = 0.0;
                var count = 0.0;
                for (var dj = 0; dj <= r2; dj++) {
                    for (var di = 0; di <= r2; di++) {
                        let v = sample_h(i + o.x + di, j + o.y + dj);
                        sum += v;
                        sum2 += v * v;
                        count += 1.0;
                    }
                }
                let mean = sum / count;
                let var_ = sum2 / count - mean * mean;
                if (var_ < best_var) {
                    best_var = var_;
                    best = mean;
                }
            }
            out_h = best;
        }
        case 21u: { // Flows — mild downhill smear
            let hx = sample_h(i + 1, j) - sample_h(i - 1, j);
            let hz = sample_h(i, j + 1) - sample_h(i, j - 1);
            let di = select(-1, 1, hx > 0.0);
            let dj = select(-1, 1, hz > 0.0);
            let down = sample_h(i + di, j + dj);
            out_h = mix(h, min(h, down), clamp(u.amount, 0.0, 1.0));
        }
        case 22u: { // Swirl
            let cx = f32(u.width) * 0.5;
            let cy = f32(u.height) * 0.5;
            let dx = f32(i) - cx;
            let dy = f32(j) - cy;
            let ang = u.amount * 0.02 * exp(-length(vec2<f32>(dx, dy)) * 0.01);
            let ca = cos(ang);
            let sa = sin(ang);
            let si = i32(round(cx + dx * ca - dy * sa));
            let sj = i32(round(cy + dx * sa + dy * ca));
            out_h = sample_h(si, sj);
        }
        case 23u: { // Blocks — quantize XY domain
            let bs = max(r, 2);
            let bi = (i / bs) * bs + bs / 2;
            let bj = (j / bs) * bs + bs / 2;
            out_h = sample_h(bi, bj);
        }
        case 24u: { // AddSet
            out_h = select(h + u.amount, u.amount, u.sea_level >= 0.5);
        }
        case 25u: { // Curve — exact CPU field-range remap
            let min_h = ordered_to_float(range_state.min_ordered);
            let max_h = ordered_to_float(range_state.max_ordered);
            let range = max(max_h - min_h, 1e-5);
            let a = clamp(u.amount, 0.0, 1.0);
            var gamma: f32;
            if (a < 0.5) {
                gamma = 1.0 + (0.5 - a) * 2.0;
            } else {
                gamma = 1.0 / max(1.0 + (a - 0.5) * 1.5, 0.35);
            }
            let n = clamp((h - min_h) / range, 0.0, 1.0);
            var shaped: f32;
            if (a >= 0.5) {
                let t = pow(n, gamma);
                let smooth_v = t * t * (3.0 - 2.0 * t);
                let blend = (a - 0.5) * 2.0;
                shaped = n * (1.0 - blend) + smooth_v * blend;
            } else {
                shaped = pow(n, gamma);
            }
            out_h = min_h + clamp(shaped, 0.0, 1.0) * range;
        }
        case 26u: { // Cutoff — exact CPU field-range shelf/remap
            let min_h = ordered_to_float(range_state.min_ordered);
            let max_h = ordered_to_float(range_state.max_ordered);
            let range = max(max_h - min_h, 1e-5);
            let cut = min_h + clamp(u.sea_level, 0.0, 1.0) * range;
            let soft = max(clamp(u.amount, 0.0, 1.0) * range * 0.15, 1e-4);
            let t = clamp((h - (cut - soft)) / (2.0 * soft), 0.0, 1.0);
            out_h = cut * (1.0 - t) + h * t;
        }
        case 27u: { // TerraceSimple — CPU geology terrace controls
            let min_h = ordered_to_float(range_state.min_ordered);
            let max_h = ordered_to_float(range_state.max_ordered);
            let range = max(max_h - min_h, 1e-5);
            let levels = clamp(round(u.amount), 2.0, 32.0);
            let interval = select(
                max(range / levels, 1e-4),
                max(u.terrace_height, 1e-4),
                u.terrace_height > 1e-4,
            );
            let phase = u.terrace_offset - floor(u.terrace_offset);
            let t = (h - min_h) / interval + phase;
            let band = floor(t);
            let frac = t - band;
            let sharpness = clamp(u.riser_sharpness, 0.0, 1.0);
            let riser_width = clamp(1.0 - sharpness, 0.04, 0.92);
            let tread_end = 1.0 - riser_width;
            var stepped = band;
            if (frac > tread_end) {
                stepped = band + smootherstep((frac - tread_end) / max(riser_width, 1e-4));
            }
            var terrace_h = min_h + (stepped - phase) * interval;
            let top = clamp(u.top_smoothness, 0.0, 1.0);
            if (top > 1e-4) {
                let edge = select(0.0, clamp((tread_end - frac) / max(tread_end, 1e-4), 0.0, 1.0), frac <= tread_end);
                let soften = (1.0 - smootherstep(edge)) * top;
                terrace_h = terrace_h * (1.0 - soften * 0.65) + h * (soften * 0.65);
            }
            out_h = clamp(terrace_h, min_h, min_h + range);
        }
        case 31u: { // Shore — CPU coastal remap
            if (h < u.sea_level) {
                if (u.amount <= 0.0) {
                    out_h = u.sea_level;
                } else {
                    let depth = u.sea_level - h;
                    out_h = u.sea_level - u.amount * (1.0 - exp(-depth / max(u.beach_width, 1e-3)));
                }
            } else if (h < u.sea_level + u.beach_width) {
                let t = clamp((h - u.sea_level) / max(u.beach_width, 1e-3), 0.0, 1.0);
                let softened = t * t * (3.0 - 2.0 * t);
                out_h = u.sea_level + softened * u.beach_width;
            }
        }
        case 32u: { // Blocks — height quantization
            let step = max(u.amount, 1.0);
            out_h = round(h / step) * step;
        }
        case 33u: { // ZeroEdge — exact field-min/world-distance fade
            let min_h = ordered_to_float(range_state.min_ordered);
            let margin = select(
                max(clamp(u.amount, 0.02, 0.45) * min(u.world_x, u.world_z), u.dx),
                u.amount,
                u.amount > 1.0,
            );
            let base = select(min_h, u.sea_level, abs(u.sea_level) > 1e-4);
            let distance_m = min(
                min(f32(i) * u.dx, f32(i32(u.width) - 1 - i) * u.dx),
                min(f32(j) * u.dz, f32(i32(u.height) - 1 - j) * u.dz),
            );
            let t = smoothstep(0.0, margin, distance_m);
            out_h = base * (1.0 - t) + h * t;
        }
        case 34u: { // Squeeze — exact CPU range compression
            let min_h = ordered_to_float(range_state.min_ordered);
            let max_h = ordered_to_float(range_state.max_ordered);
            let range = max(max_h - min_h, 1e-5);
            let amount = select(clamp(u.amount, 0.0, 1.0), clamp(u.amount / (u.amount + 8.0), 0.0, 1.0), u.amount > 1.0);
            let gamma = 1.0 + amount * 2.5;
            let normalized = clamp((h - min_h) / range, 0.0, 1.0);
            let centered = (normalized - 0.5) * 2.0;
            let squeezed = 0.5 + 0.5 * sign(centered) * pow(abs(centered), gamma);
            let mapped = min_h + squeezed * range;
            out_h = h * (1.0 - amount) + mapped * amount;
        }
        case 35u: { // DirectionalBlur — fixed-direction Gaussian
            let angle = u.rotation_deg * 0.017453292519943295;
            let direction = vec2<f32>(cos(angle), sin(angle));
            let sigma = max(f32(r) * 0.5, 0.5);
            var sum = 0.0;
            var weight_sum = 0.0;
            for (var k = -r; k <= r; k++) {
                let fi = i + i32(round(direction.x * f32(k)));
                let fj = j + i32(round(direction.y * f32(k)));
                let weight = exp(-f32(k * k) / (2.0 * sigma * sigma));
                sum += sample_h(fi, fj) * weight;
                weight_sum += weight;
            }
            out_h = sum / max(weight_sum, 1e-5);
        }
        case 36u: { // AngleBlur — gradient-aligned Gaussian
            let center = h;
            let gx0 = (sample_h(i + 1, j) - center) / max(u.dx, 1e-5);
            let gz0 = (sample_h(i, j + 1) - center) / max(u.dz, 1e-5);
            let magnitude = sqrt(gx0 * gx0 + gz0 * gz0);
            let direction = select(vec2<f32>(1.0, 0.0), vec2<f32>(gx0, gz0) / magnitude, magnitude >= 1e-6);
            let sigma = max(f32(r) * 0.5, 0.5);
            var sum = 0.0;
            var weight_sum = 0.0;
            for (var k = -r; k <= r; k++) {
                let fi = i + i32(round(direction.x * f32(k)));
                let fj = j + i32(round(direction.y * f32(k)));
                let weight = exp(-f32(k * k) / (2.0 * sigma * sigma));
                sum += sample_h(fi, fj) * weight;
                weight_sum += weight;
            }
            out_h = sum / max(weight_sum, 1e-5);
        }
        case 37u: { // Swirl — CPU Perlin-driven arbitrary resample
            let uv = vec2<f32>((f32(i) + 0.5) / f32(u.width), (f32(j) + 0.5) / f32(u.height));
            let centered = uv - vec2<f32>(0.5);
            let radial = length(centered);
            let world = cpu_world(i, j);
            let noise = cpu_perlin(world * max(u.frequency, 1e-5), u.seed);
            let angle = (1.0 - radial) * max(u.amount, 0.1) * (0.5 + noise);
            let sine = sin(angle);
            let cosine = cos(angle);
            let sample_uv = clamp(vec2<f32>(
                0.5 + centered.x * cosine - centered.y * sine,
                0.5 + centered.x * sine + centered.y * cosine,
            ), vec2<f32>(0.0), vec2<f32>(1.0));
            let si = i32(round(sample_uv.x * f32(u.width - 1u)));
            let sj = i32(round(sample_uv.y * f32(u.height - 1u)));
            out_h = sample_h(si, sj);
        }
        case 38u: { // Crater — radial bowl, rim, and ejecta
            let world = cpu_world(i, j);
            let center = vec2<f32>(u.world_x * 0.5, u.world_z * 0.5);
            let radius_m = max(clamp(u.crater_radius, 0.02, 0.5) * min(u.world_x, u.world_z), u.dx * 2.0);
            let depth = max(u.amount, 0.0);
            let rim_height = depth * 0.18;
            let warped = cpu_transform_world(world);
            let noise = cpu_perlin(warped * max(u.frequency, 1e-5) * 2.5, u.seed ^ 0x0000c2a7u);
            let distance_m = max(distance(world, center) + noise * radius_m * 0.08, 0.0);
            let normalized = distance_m / radius_m;
            var delta = 0.0;
            if (normalized < 1.0) {
                let t = 1.0 - normalized * normalized;
                delta = -depth * t * t;
            } else if (normalized < 1.25) {
                let t = (normalized - 1.0) / 0.25;
                delta = rim_height * (1.0 - t) * (1.0 - t) * 4.0 * t;
            } else if (normalized < 1.7) {
                let t = (normalized - 1.25) / 0.45;
                delta = rim_height * 0.25 * (1.0 - t) * (0.5 + 0.5 * noise);
            }
            out_h = h + delta;
        }
        case 39u: { // Distortion — CPU Perlin coordinate displacement
            let world = cpu_world(i, j);
            let frequency = max(u.frequency, 1e-5);
            let ox = cpu_perlin(world * frequency, u.seed) * u.amount;
            let oz = cpu_perlin(world * frequency + vec2<f32>(17.0, 9.0), u.seed ^ 0x0000a5a5u) * u.amount;
            let sample_uv = clamp((world + vec2<f32>(ox, oz)) / vec2<f32>(u.world_x, u.world_z), vec2<f32>(0.0), vec2<f32>(1.0));
            let si = i32(sample_uv.x * f32(u.width - 1u));
            let sj = i32(sample_uv.y * f32(u.height - 1u));
            out_h = sample_h(si, sj);
        }
        case 40u: { // Balloon — amount-limited dilation under CPU slope gate
            var local_max = h;
            for (var dj = -r; dj <= r; dj++) {
                for (var di = -r; di <= r; di++) {
                    local_max = max(local_max, sample_h(i + di, j + dj));
                }
            }
            let inflated = h + min(local_max - h, u.amount);
            let slope = cpu_slope_forward(i, j);
            let lo = min(u.slope_min, u.slope_max);
            let hi = max(u.slope_min, u.slope_max);
            let gate = select(clamp((slope - lo) / max(hi - lo, 1e-6), 0.0, 1.0), 1.0, hi <= lo + 1e-3);
            out_h = h * (1.0 - gate) + inflated * gate;
        }
        case 41u: { // NoisePerlin
            let world = cpu_transform_world(cpu_world(i, j));
            out_h = h + cpu_fbm(world, false) * u.amount;
        }
        case 42u: { // NoiseValue
            let world = cpu_transform_world(cpu_world(i, j));
            out_h = h + cpu_fbm(world, true) * u.amount;
        }
        case 43u: { // NoiseWhite
            let noise = f32(cpu_hash2(i, j, u.seed)) / f32(0xffffffffu) * 2.0 - 1.0;
            out_h = h + noise * u.amount;
        }
        case 44u: { // NoiseWave
            let world = cpu_transform_world(cpu_world(i, j));
            let anisotropy = max(u.anisotropy, 0.05);
            let noise = clamp(
                sin(world.x * u.frequency / anisotropy)
                    * cos(world.y * u.frequency * 0.7 * sqrt(anisotropy)),
                -1.0,
                1.0,
            );
            out_h = h + noise * u.amount;
        }
        case 45u: { // ScatterDetail — deterministic sparse impulses
            let world = cpu_transform_world(cpu_world(i, j));
            let cell = max(1.0 / max(u.frequency, 1e-5), 2.0);
            let cx = i32(floor(world.x / cell));
            let cz = i32(floor(world.y / cell));
            let hash = cpu_hash2(cx, cz, u.seed);
            if ((hash % 1000u) <= 120u) {
                let jitter_x = f32((hash >> 8u) & 0xffu) / 255.0;
                let jitter_z = f32((hash >> 16u) & 0xffu) / 255.0;
                let origin = vec2<f32>((f32(cx) + jitter_x) * cell, (f32(cz) + jitter_z) * cell);
                let delta = world - origin;
                let sigma = cell * 0.28;
                let envelope = exp(-dot(delta, delta) / (2.0 * sigma * sigma));
                if (envelope >= 1e-3) {
                    let amplitude = (f32((hash >> 24u) & 0xffu) / 255.0 * 2.0 - 1.0) * u.amount;
                    out_h = h + amplitude * envelope;
                }
            }
        }
        case 46u: { // SpikeRemoval — exact radius-1 median and threshold
            let median = median_exact9(i, j);
            if (abs(h - median) > max(u.amount, 0.5)) {
                out_h = median;
            }
        }
        case 47u: { // NoiseBillow
            let world = cpu_transform_world(cpu_world(i, j));
            out_h = h + cpu_billow(world) * u.amount;
        }
        case 48u: { // NoiseRidged
            let world = cpu_transform_world(cpu_world(i, j));
            let noise = cpu_ridged_custom(
                world, u.seed, max(u.octaves, 1u), max(u.frequency, 1e-5),
                max(u.lacunarity, 1.01), clamp(u.persistence, 0.05, 0.95),
            );
            out_h = h + (noise * 2.0 - 1.0) * u.amount;
        }
        case 49u: { // Ridged detail preset
            let world = cpu_world(i, j);
            let noise = cpu_ridged_custom(world, u.seed, 5u, max(u.frequency, 1e-5), 2.1, 0.55);
            out_h = h + (noise * 2.0 - 1.0) * u.amount;
        }
        case 50u: { // Rugged Perlin/ridged blend
            let world = cpu_world(i, j);
            let seed = u.seed ^ 0x0000a06du;
            let fbm = cpu_fbm_custom(world, seed, 6u, max(u.frequency, 1e-5), 2.3, 0.6);
            let ridge = cpu_ridged_custom(world * 1.7, seed, 6u, max(u.frequency, 1e-5), 2.3, 0.6);
            out_h = h + (fbm * 0.45 + (ridge * 2.0 - 1.0) * 0.55) * u.amount;
        }
        case 51u: { // BorderBlend with an authored absolute target
            let margin = select(
                max(clamp(u.amount, 0.02, 0.45) * min(u.world_x, u.world_z), u.dx),
                u.amount,
                u.amount > 1.0,
            );
            let distance = min(
                min(f32(i) * u.dx, f32(i32(u.width) - 1 - i) * u.dx),
                min(f32(j) * u.dz, f32(i32(u.height) - 1 - j) * u.dz),
            );
            let blend = 1.0 - smoothstep(0.0, margin, distance);
            out_h = mix(h, u.sea_level, blend);
        }
        case 52u: { // FlattenFilter with an authored absolute target
            let pull = select(
                clamp(u.amount, 0.15, 0.95),
                clamp(u.amount / (u.amount + 8.0), 0.15, 0.95),
                u.amount > 1.0,
            );
            out_h = mix(h, u.sea_level, pull);
        }
        case 53u: { // Hexagons — axial cell snap and nearest field resample
            let world = cpu_world(i, j);
            let cell = max(1.0 / max(u.frequency, 1e-5), 2.0);
            let q = world.x * (2.0 / 3.0) / cell;
            let axial_r = (-world.x / 3.0 + sqrt(3.0) / 3.0 * world.y) / cell;
            let qi = round(q);
            let ri = round(axial_r);
            let center = vec2<f32>(
                cell * 1.5 * qi,
                cell * (sqrt(3.0) * 0.5 * qi + sqrt(3.0) * ri),
            );
            let uv = clamp(center / vec2<f32>(u.world_x, u.world_z), vec2<f32>(0.0), vec2<f32>(1.0));
            let si = i32(round(uv.x * f32(u.width - 1u)));
            let sj = i32(round(uv.y * f32(u.height - 1u)));
            let cell_height = sample_h(si, sj);
            let blend = clamp(u.amount / (u.amount + 4.0), 0.2, 0.9);
            out_h = mix(h, cell_height, blend);
        }
        case 54u: { // TerraceSteep — slope/orientation gated geological terrace
            let min_h = ordered_to_float(range_state.min_ordered);
            let max_h = ordered_to_float(range_state.max_ordered);
            let height_range = max(max_h - min_h, 1e-5);
            let levels = clamp(round(u.amount), 2.0, 24.0);
            let interval = select(
                max(height_range / levels, 1e-4),
                max(u.terrace_height, 1e-4),
                u.terrace_height > 1e-4,
            );
            let world = cpu_world(i, j);
            let frequency = max(u.frequency, 1e-5);
            var phase_perturb = cpu_perlin(world * frequency * 0.5, u.seed ^ 0x000057eeu) * 0.15;
            let center = sample_h(i, j);
            let raw_gradient = vec2<f32>(
                (sample_h(i + 1, j) - center) / max(u.dx, 1e-5),
                (sample_h(i, j + 1) - center) / max(u.dz, 1e-5),
            );
            let magnitude = length(raw_gradient);
            if (magnitude > 1e-6) {
                phase_perturb += dot(world, raw_gradient / magnitude) * frequency * 0.08;
            }
            let lo = min(u.slope_min, u.slope_max);
            let hi = max(max(u.slope_min, u.slope_max), lo + 1e-3);
            let weight = smootherstep(clamp((atan(magnitude) * 57.2957795 - lo) / (hi - lo), 0.0, 1.0));
            let phase = fract(u.terrace_offset + phase_perturb);
            let t = (h - min_h) / interval + phase;
            let band = floor(t);
            let frac = t - band;
            let sharpness = clamp(u.riser_sharpness, 0.0, 1.0);
            let riser_width = clamp(1.0 - sharpness, 0.04, 0.92);
            let tread_end = 1.0 - riser_width;
            var stepped = band;
            if (frac > tread_end) {
                stepped = band + smootherstep((frac - tread_end) / max(riser_width, 1e-4));
            }
            var terrace_h = min_h + (stepped - phase) * interval;
            let top = clamp(u.top_smoothness, 0.0, 1.0);
            if (top > 1e-4) {
                let edge = select(0.0, clamp((tread_end - frac) / max(tread_end, 1e-4), 0.0, 1.0), frac <= tread_end);
                let soften = (1.0 - smootherstep(edge)) * top;
                terrace_h = terrace_h * (1.0 - soften * 0.65) + h * (soften * 0.65);
            }
            terrace_h = clamp(terrace_h, min_h, min_h + height_range);
            out_h = mix(h, terrace_h, weight);
        }
        default: {
            out_h = box_blur(i, j, r);
        }
    }

    if (u.invert > 0.5) {
        out_h = h - (out_h - h);
    }
    let s = clamp(u.strength, 0.0, 1.0);
    return mix(h, out_h, s);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let has_region = u.region_w > 0u && u.region_h > 0u;
    let i = select(gid.x, u.region_x + gid.x, has_region);
    let j = select(gid.y, u.region_y + gid.y, has_region);
    if (has_region) {
        if (gid.x >= u.region_w || gid.y >= u.region_h) { return; }
    } else if (gid.x >= u.width || gid.y >= u.height) {
        return;
    }
    if (i >= u.width || j >= u.height) { return; }
    let h = sample_h(i32(i), i32(j));
    let out_h = apply_filter(i32(i), i32(j), h);
    textureStore(dst, vec2<i32>(i32(i), i32(j)), vec4<f32>(out_h, 0.0, 0.0, 0.0));
}
