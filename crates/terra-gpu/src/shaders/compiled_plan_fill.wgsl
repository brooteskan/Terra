struct Uniforms {
    width: u32,
    height: u32,
    value: f32,
    _pad: f32,
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.region_w || gid.y >= u.region_h) { return; }
    let p = vec2<u32>(u.region_x + gid.x, u.region_y + gid.y);
    if (p.x >= u.width || p.y >= u.height) { return; }
    textureStore(dst, vec2<i32>(p), vec4<f32>(u.value, 0.0, 0.0, 0.0));
}
