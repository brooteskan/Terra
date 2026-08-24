struct Uniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    world_height: vec4<f32>,
    cursor_radius: vec4<f32>,
    color: vec4<f32>,
};

struct SurfacePick {
    hit_uv_height: vec4<f32>,
    world_pos_request: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var<storage, read_write> result: SurfacePick;

@compute @workgroup_size(1)
fn main() {
    result.hit_uv_height = vec4<f32>(0.0);
    result.world_pos_request = vec4<f32>(0.0);
    if (u.cursor_radius.w < 0.5) {
        return;
    }

    let dim = textureDimensions(depth_tex);
    if (dim.x == 0u || dim.y == 0u) {
        return;
    }
    let uv = vec2<f32>(
        u.cursor_radius.x * 0.5 + 0.5,
        0.5 - u.cursor_radius.y * 0.5,
    );
    let pixel = clamp(
        vec2<i32>(floor(uv * vec2<f32>(dim))),
        vec2<i32>(0),
        vec2<i32>(dim) - vec2<i32>(1),
    );
    let depth = textureLoad(depth_tex, pixel, 0);
    if (depth >= 1.0) {
        return;
    }

    let homogeneous = u.inv_view_proj
        * vec4<f32>(u.cursor_radius.xy, depth, 1.0);
    if (abs(homogeneous.w) < 1.0e-8) {
        return;
    }
    let local = homogeneous.xyz / homogeneous.w;
    result.hit_uv_height = vec4<f32>(1.0, 0.0, 0.0, local.y);
    result.world_pos_request = vec4<f32>(local, 0.0);
}
