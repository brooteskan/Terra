struct Uniforms {
    width: u32,
    height: u32,
    world_x: f32,
    world_z: f32,
    height_min: f32,
    height_max: f32,
    direction: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) { return; }
    let uv = vec2<f32>((f32(gid.x) + 0.5) / f32(u.width), (f32(gid.y) + 0.5) / f32(u.height));
    let dir = vec2<f32>(cos(u.direction), sin(u.direction));
    let t = clamp(dot(uv - vec2<f32>(0.5, 0.5), dir) + 0.5, 0.0, 1.0);
    let h = mix(u.height_min, u.height_max, t);
    textureStore(dst, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(h, 0.0, 0.0, 0.0));
}
