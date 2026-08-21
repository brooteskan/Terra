struct Uniforms {
    width: u32,
    height: u32,
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.region_w || gid.y >= u.region_h) { return; }
    let x = u.region_x + gid.x;
    let y = u.region_y + gid.y;
    if (x >= u.width || y >= u.height) { return; }
    let p = vec2<i32>(i32(x), i32(y));
    let v = textureLoad(src, p, 0).r;
    textureStore(dst, p, vec4<f32>(v, 0.0, 0.0, 0.0));
}
