struct Uniforms {
    source_width: u32,
    source_height: u32,
    origin_x: u32,
    origin_z: u32,
    interior_width: u32,
    interior_height: u32,
    halo: u32,
    page_extent: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var destination: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.page_extent || gid.y >= u.page_extent) { return; }
    let valid_width = u.interior_width + 2u * u.halo;
    let valid_height = u.interior_height + 2u * u.halo;
    var value = 0.0;
    if (gid.x < valid_width && gid.y < valid_height) {
        let sx = clamp(
            i32(u.origin_x) + i32(gid.x) - i32(u.halo),
            0,
            i32(u.source_width) - 1,
        );
        let sy = clamp(
            i32(u.origin_z) + i32(gid.y) - i32(u.halo),
            0,
            i32(u.source_height) - 1,
        );
        value = textureLoad(source, vec2<i32>(sx, sy), 0).r;
    }
    textureStore(destination, vec2<i32>(gid.xy), vec4<f32>(value, 0.0, 0.0, 0.0));
}
