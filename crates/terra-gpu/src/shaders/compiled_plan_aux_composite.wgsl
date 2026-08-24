struct Uniforms {
    width: u32,
    height: u32,
    opacity: f32,
    has_parent: u32,
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
    channel_class: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var parent_field: texture_2d<f32>;
@group(0) @binding(2) var child_field: texture_2d<f32>;
@group(0) @binding(3) var mask_field: texture_2d<f32>;
@group(0) @binding(4) var output_field: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.region_w || gid.y >= u.region_h) { return; }
    let absolute = vec2<u32>(u.region_x + gid.x, u.region_y + gid.y);
    if (absolute.x >= u.width || absolute.y >= u.height) { return; }
    let p = vec2<i32>(absolute);
    var parent = 0.0;
    if (u.has_parent != 0u) {
        parent = textureLoad(parent_field, p, 0).r;
    }
    let child = textureLoad(child_field, p, 0).r;
    let weight = clamp(textureLoad(mask_field, p, 0).r * u.opacity, 0.0, 1.0);
    var result = parent * (1.0 - weight) + child * weight;
    if (u.channel_class == 0u) {
        result = clamp(result, 0.0, 1.0);
    } else if (u.channel_class == 2u) {
        result = select(parent, child, weight >= 0.5);
    }
    textureStore(output_field, p, vec4<f32>(result, 0.0, 0.0, 0.0));
}
