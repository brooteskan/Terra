struct Uniforms {
    width: u32,
    height: u32,
    value: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) { return; }
    textureStore(dst, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(u.value, 0.0, 0.0, 0.0));
}
