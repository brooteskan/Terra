struct Uniforms {
    child_width: u32,
    child_height: u32,
    parent_width: u32,
    parent_height: u32,
    tile_size: u32,
    metadata_offset: u32,
    tiles_x: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var child: texture_2d<f32>;
@group(0) @binding(2) var parent: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> errors: array<atomic<u32>>;

fn parent_approximation(x: u32, y: u32) -> f32 {
    let uv = (vec2<f32>(f32(x), f32(y)) + vec2<f32>(0.5))
        / vec2<f32>(f32(u.child_width), f32(u.child_height));
    let p = uv * vec2<f32>(f32(u.parent_width), f32(u.parent_height)) - vec2<f32>(0.5);
    let p0 = vec2<i32>(floor(p));
    let t = fract(p);
    let limit = vec2<i32>(i32(u.parent_width) - 1, i32(u.parent_height) - 1);
    let a = clamp(p0, vec2<i32>(0), limit);
    let b = clamp(p0 + vec2<i32>(1), vec2<i32>(0), limit);
    let h00 = textureLoad(parent, a, 0).r;
    let h10 = textureLoad(parent, vec2<i32>(b.x, a.y), 0).r;
    let h01 = textureLoad(parent, vec2<i32>(a.x, b.y), 0).r;
    let h11 = textureLoad(parent, b, 0).r;
    return mix(mix(h00, h10, t.x), mix(h01, h11, t.x), t.y);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.child_width || gid.y >= u.child_height) { return; }
    let actual = textureLoad(child, vec2<i32>(gid.xy), 0).r;
    let reconstructed = parent_approximation(gid.x, gid.y);
    var error = abs(actual - reconstructed);
    if (!(error >= 0.0) || error > 3.402823466e+38) {
        error = 3.402823466e+38;
    }
    let tile_x = gid.x / u.tile_size;
    let tile_z = gid.y / u.tile_size;
    let index = u.metadata_offset + tile_z * u.tiles_x + tile_x;
    atomicMax(&errors[index], bitcast<u32>(error));
}
