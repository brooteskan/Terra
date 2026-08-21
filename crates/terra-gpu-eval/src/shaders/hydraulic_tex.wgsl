// Pass B: gather neighbor outflows, erode/deposit, write height/water/sediment.
// Deliberate Mei simplifications: no pipe cross-section, no velocity field, 4-neigh only.
// Softness = 1 - K from hardness texture (R32Float), with optional layered bedrock/sediment K.
// Phase G: terrain-slope capacity, fan boost at slope breaks, floodplain bias, stagnant settle.
// Phase 4: rainfall map + transport-model incision / layered erodibility.
struct Uniforms {
    width: u32,
    height: u32,
    timestep: f32,
    rainfall: f32,
    evaporation: f32,
    erosion: f32,
    deposition: f32,
    capacity: f32,
    fan_boost: f32,
    floodplain_bias: f32,
    dx: f32,
    incision_bias: f32,
    bedrock_k: f32,
    sediment_k: f32,
    layered: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var height_src: texture_2d<f32>;
@group(0) @binding(2) var water_src: texture_2d<f32>;
@group(0) @binding(3) var sed_src: texture_2d<f32>;
@group(0) @binding(4) var outflow: texture_2d<f32>;
@group(0) @binding(5) var height_dst: texture_storage_2d<r32float, write>;
@group(0) @binding(6) var water_dst: texture_storage_2d<r32float, write>;
@group(0) @binding(7) var sed_dst: texture_storage_2d<r32float, write>;
@group(0) @binding(8) var hardness: texture_2d<f32>;
@group(0) @binding(9) var rainfall_map: texture_2d<f32>;
@group(0) @binding(10) var loose_sediment: texture_2d<f32>;

fn terrain_slope(p: vec2<i32>, h0: f32) -> f32 {
    var max_drop = 0.0;
    let n0 = vec2<i32>(p.x - 1, p.y);
    if (n0.x >= 0) {
        max_drop = max(max_drop, h0 - textureLoad(height_src, n0, 0).r);
    }
    let n1 = vec2<i32>(p.x + 1, p.y);
    if (n1.x < i32(u.width)) {
        max_drop = max(max_drop, h0 - textureLoad(height_src, n1, 0).r);
    }
    let n2 = vec2<i32>(p.x, p.y - 1);
    if (n2.y >= 0) {
        max_drop = max(max_drop, h0 - textureLoad(height_src, n2, 0).r);
    }
    let n3 = vec2<i32>(p.x, p.y + 1);
    if (n3.y < i32(u.height)) {
        max_drop = max(max_drop, h0 - textureLoad(height_src, n3, 0).r);
    }
    return max(max_drop / max(u.dx, 1e-6), 0.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    if (i >= u.width || j >= u.height) { return; }
    let p = vec2<i32>(i32(i), i32(j));
    let rain = max(textureLoad(rainfall_map, p, 0).r, 0.0);
    let h0 = textureLoad(height_src, p, 0).r;
    let w0 = textureLoad(water_src, p, 0).r + u.rainfall * rain;
    let s0 = textureLoad(sed_src, p, 0).r;
    let f = textureLoad(outflow, p, 0);
    let out_sum = f.x + f.y + f.z + f.w;
    let loose = max(textureLoad(loose_sediment, p, 0).r, 0.0);
    var k = clamp(textureLoad(hardness, p, 0).r, 0.0, 1.0);
    if (u.layered > 0.5) {
        // Soft sediment first; expose harder bedrock once the mantle is gone.
        k = select(clamp(u.bedrock_k, 0.0, 1.0), clamp(u.sediment_k, 0.0, 1.0), loose > 1e-4);
    }
    let soft = 1.0 - k;
    let erode_scale = max(u.incision_bias, 0.05);

    // Gather inflows from neighbors' opposing fluxes (L←R, R←L, D←U, U←D).
    var inflow = 0.0;
    var sed_in = 0.0;
    let ii = i32(i);
    let jj = i32(j);
    if (ii > 0) {
        let np = vec2<i32>(ii - 1, jj);
        let nf = textureLoad(outflow, np, 0);
        let nw = textureLoad(water_src, np, 0).r + u.rainfall * max(textureLoad(rainfall_map, np, 0).r, 0.0);
        let fl = nf.y; // neighbor's Right → us
        inflow += fl;
        if (nw > 1e-8) {
            sed_in += textureLoad(sed_src, np, 0).r * (fl / nw);
        }
    }
    if (ii + 1 < i32(u.width)) {
        let np = vec2<i32>(ii + 1, jj);
        let nf = textureLoad(outflow, np, 0);
        let nw = textureLoad(water_src, np, 0).r + u.rainfall * max(textureLoad(rainfall_map, np, 0).r, 0.0);
        let fl = nf.x; // neighbor's Left → us
        inflow += fl;
        if (nw > 1e-8) {
            sed_in += textureLoad(sed_src, np, 0).r * (fl / nw);
        }
    }
    if (jj + 1 < i32(u.height)) {
        let np = vec2<i32>(ii, jj + 1);
        let nf = textureLoad(outflow, np, 0);
        let nw = textureLoad(water_src, np, 0).r + u.rainfall * max(textureLoad(rainfall_map, np, 0).r, 0.0);
        let fl = nf.w; // neighbor's Up → us
        inflow += fl;
        if (nw > 1e-8) {
            sed_in += textureLoad(sed_src, np, 0).r * (fl / nw);
        }
    }
    if (jj > 0) {
        let np = vec2<i32>(ii, jj - 1);
        let nf = textureLoad(outflow, np, 0);
        let nw = textureLoad(water_src, np, 0).r + u.rainfall * max(textureLoad(rainfall_map, np, 0).r, 0.0);
        let fl = nf.z; // neighbor's Down → us
        inflow += fl;
        if (nw > 1e-8) {
            sed_in += textureLoad(sed_src, np, 0).r * (fl / nw);
        }
    }

    let sed_leave = select(0.0, s0 * min(out_sum / max(w0, 1e-8), 1.0), w0 > 1e-8);
    var w = max(w0 - out_sum + inflow, 0.0);
    var s = max(s0 - sed_leave + sed_in, 0.0);
    var h = h0;

    let flow = out_sum;
    let t_slope = terrain_slope(p, h0);
    // Approximate upslope as max of neighbor slopes (same helper).
    var upslope = 0.0;
    if (ii > 0) { upslope = max(upslope, terrain_slope(vec2<i32>(ii - 1, jj), textureLoad(height_src, vec2<i32>(ii - 1, jj), 0).r)); }
    if (ii + 1 < i32(u.width)) { upslope = max(upslope, terrain_slope(vec2<i32>(ii + 1, jj), textureLoad(height_src, vec2<i32>(ii + 1, jj), 0).r)); }
    if (jj > 0) { upslope = max(upslope, terrain_slope(vec2<i32>(ii, jj - 1), textureLoad(height_src, vec2<i32>(ii, jj - 1), 0).r)); }
    if (jj + 1 < i32(u.height)) { upslope = max(upslope, terrain_slope(vec2<i32>(ii, jj + 1), textureLoad(height_src, vec2<i32>(ii, jj + 1), 0).r)); }

    let flat_t = clamp(1.0 - clamp(t_slope / (t_slope + 0.08), 0.0, 1.0), 0.0, 1.0);
    var break_t = 0.0;
    if (flat_t > 0.4 && upslope > t_slope + 0.05) {
        break_t = clamp((upslope - t_slope) / (upslope + 1e-4), 0.0, 1.0) * flat_t;
    }
    let flow_proxy = flow + w * 0.35;
    let corridor = clamp(flow_proxy / (flow_proxy + 0.04), 0.0, 1.0);
    let fan = 1.0 + max(u.fan_boost, 0.0) * break_t;
    let flood = 1.0 + max(u.floodplain_bias, 0.0) * flat_t * corridor;
    let dep_rate = clamp(u.deposition * fan * flood, 0.0, 2.5);

    let slope_term = max(flow * 0.25, t_slope * 0.5);
    let transport = max(flow, w * 0.08);
    let cap_scale = max(1.0 + max(u.fan_boost, 0.0) * break_t + max(u.floodplain_bias, 0.0) * flat_t * corridor * 0.5, 1.0);
    let cap = u.capacity * slope_term * transport / cap_scale;

    if (transport > 1e-8 || s > 1e-8) {
        if (s < cap && flow > 1e-8) {
            let erode_amt = min((cap - s) * u.erosion * erode_scale * soft, max(h, 0.0) * 0.1);
            h -= erode_amt;
            s += erode_amt;
        } else if (s > cap) {
            let dep = min((s - cap) * dep_rate, s);
            h += dep;
            s = max(s - dep, 0.0);
        } else if (flow < 1e-6 && s > 1e-8 && flat_t > 0.35) {
            let settle = s * u.deposition * (0.2 + 0.8 * flat_t) * (0.5 + 0.5 * flood);
            let dep = min(settle, s);
            h += dep;
            s = max(s - dep, 0.0);
        }
    }

    w *= (1.0 - u.evaporation);
    // Finite / non-negative clamps (interactive stability).
    h = select(h, 0.0, !((h == h)));
    w = max(select(w, 0.0, !((w == w))), 0.0);
    s = max(select(s, 0.0, !((s == s))), 0.0);
    textureStore(height_dst, p, vec4<f32>(h, 0.0, 0.0, 0.0));
    textureStore(water_dst, p, vec4<f32>(w, 0.0, 0.0, 0.0));
    textureStore(sed_dst, p, vec4<f32>(s, 0.0, 0.0, 0.0));
}
