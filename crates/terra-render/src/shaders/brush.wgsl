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

fn sample_height_bilinear(uv: vec2<f32>) -> f32 {
    let dim = textureDimensions(height_tex);
    let p = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * vec2<f32>(dim - vec2<u32>(1u));
    let p0 = vec2<i32>(floor(p));
    let p1 = min(p0 + vec2<i32>(1), vec2<i32>(dim) - vec2<i32>(1));
    let f = fract(p);
    let h00 = textureLoad(height_tex, p0, 0).r;
    let h10 = textureLoad(height_tex, vec2<i32>(p1.x, p0.y), 0).r;
    let h01 = textureLoad(height_tex, vec2<i32>(p0.x, p1.y), 0).r;
    let h11 = textureLoad(height_tex, p1, 0).r;
    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
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
    let uv = pick.hit_uv_height.yz
        + vec2<f32>(cos(angle), sin(angle)) * max(u.cursor_radius.z, 0.002);
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }

    let y = sample_height_bilinear(uv) + 1.5;
    let world = vec3<f32>(uv.x * u.world_height.x, y, uv.y * u.world_height.y);
    out.clip = u.view_proj * vec4<f32>(world, 1.0);
    return out;
}

@fragment
fn fs_main(v: VsOut) -> @location(0) vec4<f32> {
    return v.color;
}
