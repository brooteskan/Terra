struct Uniforms {
    width: u32,
    height: u32,
    opacity: f32,
    has_parent: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var parent_field: texture_2d<f32>;
@group(0) @binding(2) var child_field: texture_2d<f32>;
@group(0) @binding(3) var mask_field: texture_2d<f32>;
@group(0) @binding(4) var output_field: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) { return; }
    let p = vec2<i32>(gid.xy);
    var parent = 0.0;
    if (u.has_parent != 0u) {
        parent = textureLoad(parent_field, p, 0).r;
    }
    let child = textureLoad(child_field, p, 0).r;
    let weight = clamp(textureLoad(mask_field, p, 0).r * u.opacity, 0.0, 1.0);
    let result = parent * (1.0 - weight) + child * weight;
    textureStore(output_field, p, vec4<f32>(result, 0.0, 0.0, 0.0));
}
