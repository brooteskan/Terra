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
@group(0) @binding(2) var<storage, read_write> result: SurfacePick;

fn sample_height_uv(uv: vec2<f32>) -> f32 {
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

fn world_to_uv(p: vec3<f32>) -> vec2<f32> {
    return vec2<f32>(
        p.x / max(u.world_height.x, 1.0e-3),
        p.z / max(u.world_height.y, 1.0e-3),
    );
}

fn project_point(inv: mat4x4<f32>, p: vec3<f32>) -> vec3<f32> {
    let h = inv * vec4<f32>(p, 1.0);
    return h.xyz / max(abs(h.w), 1.0e-8);
}

@compute @workgroup_size(1)
fn main() {
    result.hit_uv_height = vec4<f32>(0.0);
    result.world_pos_request = vec4<f32>(0.0);
    if (u.cursor_radius.w < 0.5) {
        return;
    }

    let near = project_point(u.inv_view_proj, vec3<f32>(u.cursor_radius.xy, 0.0));
    let far = project_point(u.inv_view_proj, vec3<f32>(u.cursor_radius.xy, 1.0));
    let rd = normalize(far - near);
    let ro = near;
    let wx = max(u.world_height.x, 1.0e-3);
    let wz = max(u.world_height.y, 1.0e-3);
    let min_h = min(u.world_height.z, u.world_height.w);
    let max_h = max(u.world_height.z, u.world_height.w);
    let h_pad = max((max_h - min_h) * 0.05, 1.0);

    var t_min = 0.0;
    var t_max = distance(near, far);
    if (abs(rd.x) > 1.0e-8) {
        let t0 = (0.0 - ro.x) / rd.x;
        let t1 = (wx - ro.x) / rd.x;
        t_min = max(t_min, min(t0, t1));
        t_max = min(t_max, max(t0, t1));
    } else if (ro.x < 0.0 || ro.x > wx) {
        return;
    }
    if (abs(rd.z) > 1.0e-8) {
        let t0 = (0.0 - ro.z) / rd.z;
        let t1 = (wz - ro.z) / rd.z;
        t_min = max(t_min, min(t0, t1));
        t_max = min(t_max, max(t0, t1));
    } else if (ro.z < 0.0 || ro.z > wz) {
        return;
    }
    if (abs(rd.y) > 1.0e-8) {
        let t0 = ((min_h - h_pad) - ro.y) / rd.y;
        let t1 = ((max_h + h_pad) - ro.y) / rd.y;
        t_min = max(t_min, min(t0, t1));
        t_max = min(t_max, max(t0, t1));
    }
    if (t_min > t_max) {
        return;
    }

    // Walk at least once per height texel along the longest axis. This is a
    // single cursor ray, so preserving narrow peaks is cheap compared with the
    // full-frame terrain pass and avoids the old fixed-iteration displacement.
    let dim = textureDimensions(height_tex);
    let steps = min(max(max(dim.x, dim.y), 128u), 8192u);
    var t_prev = t_min;
    var p_prev = ro + rd * t_prev;
    var y_prev = p_prev.y - sample_height_uv(world_to_uv(p_prev));
    if (y_prev <= 0.0) {
        let uv = clamp(world_to_uv(p_prev), vec2<f32>(0.0), vec2<f32>(1.0));
        result.hit_uv_height = vec4<f32>(1.0, uv, p_prev.y);
        result.world_pos_request = vec4<f32>(p_prev, 0.0);
        return;
    }

    var found = false;
    var t_hit = t_max;
    for (var i = 1u; i <= steps; i = i + 1u) {
        let t = mix(t_min, t_max, f32(i) / f32(steps));
        let p = ro + rd * t;
        let y = p.y - sample_height_uv(world_to_uv(p));
        if (y <= 0.0 && y_prev > 0.0) {
            t_hit = t;
            found = true;
            break;
        }
        y_prev = y;
        t_prev = t;
    }
    if (!found) {
        return;
    }

    var lo = t_prev;
    var hi = t_hit;
    for (var j = 0; j < 12; j = j + 1) {
        let tm = 0.5 * (lo + hi);
        let p = ro + rd * tm;
        if (p.y - sample_height_uv(world_to_uv(p)) > 0.0) {
            lo = tm;
        } else {
            hi = tm;
        }
    }
    let pos = ro + rd * hi;
    let uv = clamp(world_to_uv(pos), vec2<f32>(0.0), vec2<f32>(1.0));
    let height = sample_height_uv(uv);
    result.hit_uv_height = vec4<f32>(1.0, uv, height);
    result.world_pos_request = vec4<f32>(pos.x, height, pos.z, 0.0);
}
