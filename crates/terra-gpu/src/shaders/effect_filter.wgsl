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
    // Dirty-region dispatch (full field when region_w/h == 0).
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<r32float, write>;

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
        case 6u: { // Deflate
            out_h = h - u.amount * u.strength;
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
