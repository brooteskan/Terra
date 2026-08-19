// Executes one unary mask operation or combines an entry with the accumulator.
struct Uniforms {
    width: u32,
    height: u32,
    mode: u32,
    radius: u32,
    a: f32,
    b: f32,
    c: f32,
    pad0: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var src_a: texture_2d<f32>;
@group(0) @binding(2) var src_b: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<r32float, write>;

fn load_a(i: i32, j: i32) -> f32 {
    return textureLoad(src_a, vec2<i32>(clamp(i, 0, i32(u.width) - 1), clamp(j, 0, i32(u.height) - 1)), 0).r;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) { return; }
    let p = vec2<i32>(gid.xy);
    let x = textureLoad(src_a, p, 0).r;
    let y = textureLoad(src_b, p, 0).r;
    var out = x;
    switch u.mode {
        case 0u: { out = clamp(x + x * u.a, 0.0, 1.0); } // Add
        case 1u: { out = clamp(x - x * u.a, 0.0, 1.0); } // Subtract
        case 2u: { out = clamp(x * (x * u.a + (1.0 - u.a)), 0.0, 1.0); } // Multiply
        case 3u: { out = min(x, x); } // Min (unary CPU semantics)
        case 4u: { out = max(x, x); } // Max (unary CPU semantics)
        case 5u: { out = 1.0 - x; }
        case 6u: { out = clamp(x, u.a, u.b); }
        case 7u: {
            let t = clamp((x - u.a) / max(u.b - u.a, 1e-6), 0.0, 1.0);
            out = pow(t, 1.0 / max(u.c, 1e-6));
        }
        case 8u: {
            let t = clamp((x - u.a) / max(u.b - u.a, 1e-6), 0.0, 1.0);
            out = t * t * (3.0 - 2.0 * t);
        }
        case 9u: {
            var sum = 0.0;
            var count = 0.0;
            let r = i32(min(u.radius, 16u));
            for (var dj = -r; dj <= r; dj++) {
                for (var di = -r; di <= r; di++) {
                    let ii = i32(gid.x) + di;
                    let jj = i32(gid.y) + dj;
                    if (ii >= 0 && jj >= 0 && ii < i32(u.width) && jj < i32(u.height)) {
                        sum += load_a(ii, jj);
                        count += 1.0;
                    }
                }
            }
            out = sum / max(count, 1.0);
        }
        case 10u: { out = u.a + x * (u.b - u.a); }
        case 20u: { out = clamp(x * y, 0.0, 1.0); }
        case 21u: { out = clamp(x + y, 0.0, 1.0); }
        case 22u: { out = clamp(x - y, 0.0, 1.0); }
        case 23u: { out = min(x, y); }
        case 24u: { out = max(x, y); }
        case 25u: { out = clamp(y, 0.0, 1.0); }
        case 26u: { out = clamp(1.0 - x, 0.0, 1.0); }
        case 27u: { out = select(clamp(x, 0.0, 1.0), clamp(y, 0.0, 1.0), y > 1e-4); }
        default: {}
    }
    textureStore(dst, p, vec4<f32>(clamp(out, 0.0, 1.0), 0.0, 0.0, 0.0));
}
