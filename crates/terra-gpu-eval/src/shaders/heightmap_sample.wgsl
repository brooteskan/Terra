struct Uniforms {
    width: u32,
    height: u32,
    source_width: u32,
    source_height: u32,
    mode: u32,
    _pad0: u32,
    height_scale: f32,
    height_offset: f32,
    world_x: f32,
    world_z: f32,
    offset_x: f32,
    offset_z: f32,
    inv_scale: f32,
    sin_t: f32,
    cos_t: f32,
    blend_size: f32,
    blend_roundness: f32,
    _pad1a: f32,
    _pad1b: f32,
    _pad1c: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var output_height: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var output_mask: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) { return; }
    let p = vec2<i32>(i32(gid.x), i32(gid.y));
    var uv = vec2<f32>(f32(gid.x) / f32(u.width), f32(gid.y) / f32(u.height));
    var weight = 1.0;
    if (u.mode == 1u) {
        let x = (f32(gid.x) + 0.5) / f32(u.width) * u.world_x;
        let z = (f32(gid.y) + 0.5) / f32(u.height) * u.world_z;
        let dx = x - u.world_x * 0.5 - u.offset_x;
        let dz = z - u.world_z * 0.5 - u.offset_z;
        let rx = (dx * u.cos_t - dz * u.sin_t) * u.inv_scale;
        let rz = (dx * u.sin_t + dz * u.cos_t) * u.inv_scale;
        uv = vec2<f32>(rx / u.world_x + 0.5, rz / u.world_z + 0.5);
        let axis = abs(uv - vec2<f32>(0.5)) * 2.0;
        let edge = max(axis.x, axis.y);
        let radial = length(axis);
        let roundness = clamp(u.blend_roundness, 0.0, 1.0);
        let shape_edge = edge * (1.0 - roundness) + radial * roundness;
        let inner = max(1.0 - clamp(u.blend_size, 0.0, 1.0), 0.0);
        if (shape_edge <= inner) {
            weight = 1.0;
        } else if (shape_edge >= 1.0) {
            weight = 0.0;
        } else {
            weight = clamp((1.0 - shape_edge) / max(1.0 - inner, 1e-6), 0.0, 1.0);
        }
    }
    let clamped_uv = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let sx = min(u32(clamped_uv.x * f32(u.source_width)), u.source_width - 1u);
    let sy = min(u32(clamped_uv.y * f32(u.source_height)), u.source_height - 1u);
    let value = textureLoad(source, vec2<i32>(i32(sx), i32(sy)), 0).r;
    textureStore(output_height, p, vec4<f32>(value * u.height_scale + u.height_offset, 0.0, 0.0, 0.0));
    textureStore(output_mask, p, vec4<f32>(weight, 0.0, 0.0, 0.0));
}
