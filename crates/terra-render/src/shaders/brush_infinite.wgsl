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
@group(0) @binding(1) var height_tex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> pick: SurfacePick;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VsOut {
    var out: VsOut;
    out.color = u.color;
    if (u.cursor_radius.w < 0.5 || pick.hit_uv_height.x < 0.5) {
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }

    let angle = (f32(vertex_index) / 64.0) * 6.28318530718;
    let radius_m = max(u.cursor_radius.z, 0.001);
    let local = pick.world_pos_request.xyz + vec3<f32>(
        cos(angle) * radius_m,
        1.5,
        sin(angle) * radius_m,
    );
    out.clip = u.view_proj * vec4<f32>(local, 1.0);
    return out;
}

@fragment
fn fs_main(v: VsOut) -> @location(0) vec4<f32> {
    return v.color;
}
