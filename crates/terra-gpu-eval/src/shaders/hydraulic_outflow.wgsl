// Pass A: Mei-lite pipe outflows (L,R,D,U) into RGBA. Atomic-free; scaled so Σflux ≤ water.
// Phase 4: rainfall map multiplies the uniform rainfall rate (CFL-safe clamp stays on CPU uniforms).
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
@group(0) @binding(3) var outflow_dst: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var rainfall_map: texture_2d<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    if (i >= u.width || j >= u.height) { return; }
    let p = vec2<i32>(i32(i), i32(j));
    let h0 = textureLoad(height_src, p, 0).r;
    let rain = max(textureLoad(rainfall_map, p, 0).r, 0.0);
    let w = textureLoad(water_src, p, 0).r + u.rainfall * rain;
    let total = h0 + w;

    var raw = vec4<f32>(0.0);
    // L R D U
    let n0 = vec2<i32>(p.x - 1, p.y);
    if (n0.x >= 0) {
        let nr = max(textureLoad(rainfall_map, n0, 0).r, 0.0);
        let d = total - (textureLoad(height_src, n0, 0).r + textureLoad(water_src, n0, 0).r + u.rainfall * nr);
        if (d > 0.0) { raw.x = d; }
    }
    let n1 = vec2<i32>(p.x + 1, p.y);
    if (n1.x < i32(u.width)) {
        let nr = max(textureLoad(rainfall_map, n1, 0).r, 0.0);
        let d = total - (textureLoad(height_src, n1, 0).r + textureLoad(water_src, n1, 0).r + u.rainfall * nr);
        if (d > 0.0) { raw.y = d; }
    }
    let n2 = vec2<i32>(p.x, p.y + 1);
    if (n2.y < i32(u.height)) {
        let nr = max(textureLoad(rainfall_map, n2, 0).r, 0.0);
        let d = total - (textureLoad(height_src, n2, 0).r + textureLoad(water_src, n2, 0).r + u.rainfall * nr);
        if (d > 0.0) { raw.z = d; }
    }
    let n3 = vec2<i32>(p.x, p.y - 1);
    if (n3.y >= 0) {
        let nr = max(textureLoad(rainfall_map, n3, 0).r, 0.0);
        let d = total - (textureLoad(height_src, n3, 0).r + textureLoad(water_src, n3, 0).r + u.rainfall * nr);
        if (d > 0.0) { raw.w = d; }
    }

    let sum = raw.x + raw.y + raw.z + raw.w;
    var flux = vec4<f32>(0.0);
    if (sum > 0.0 && w > 0.0) {
        let scale = min(sum * u.timestep, w) / sum;
        flux = raw * scale;
    }
    textureStore(outflow_dst, p, flux);
}
