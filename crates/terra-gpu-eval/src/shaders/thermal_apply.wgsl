// Pass B: apply -outflow + neighbor deposits (CPU-equivalent redistribution).
struct Uniforms {
    width: u32,
    height: u32,
    dx: f32,
    talus: f32,
    strength: f32,
    _p2: f32,
    _p3: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var delta: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<r32float, write>;

fn outflow_share(h0: f32, src_h: f32, move_amt: f32, sum: f32) -> f32 {
    let diff = h0 - src_h - u.talus;
    if (diff <= 0.0 || sum <= 0.0 || move_amt <= 0.0) {
        return 0.0;
    }
    return move_amt * (diff / sum);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    if (i >= u.width || j >= u.height) { return; }
    let p = vec2<i32>(i32(i), i32(j));
    let h0 = textureLoad(src, p, 0).r;
    let my_out = textureLoad(delta, p, 0).r;
    var h = h0 - my_out;

    // Gather deposits from neighbors that flow toward this cell.
    let ii = i32(i);
    let jj = i32(j);
    // Neighbor at left: may deposit to the right (us)
    if (ii > 0) {
        let np = vec2<i32>(ii - 1, jj);
        let nh = textureLoad(src, np, 0).r;
        let nout = textureLoad(delta, np, 0).r;
        var sum = 0.0;
        let n0 = vec2<i32>(ii - 2, jj);
        if (n0.x >= 0) {
            let d = nh - textureLoad(src, n0, 0).r - u.talus;
            if (d > 0.0) { sum += d; }
        }
        let n1 = vec2<i32>(ii, jj); // us
        {
            let d = nh - textureLoad(src, n1, 0).r - u.talus;
            if (d > 0.0) { sum += d; }
        }
        let n2 = vec2<i32>(ii - 1, jj - 1);
        if (n2.y >= 0) {
            let d = nh - textureLoad(src, n2, 0).r - u.talus;
            if (d > 0.0) { sum += d; }
        }
        let n3 = vec2<i32>(ii - 1, jj + 1);
        if (n3.y < i32(u.height)) {
            let d = nh - textureLoad(src, n3, 0).r - u.talus;
            if (d > 0.0) { sum += d; }
        }
        h += outflow_share(nh, h0, nout, sum);
    }
    if (ii + 1 < i32(u.width)) {
        let np = vec2<i32>(ii + 1, jj);
        let nh = textureLoad(src, np, 0).r;
        let nout = textureLoad(delta, np, 0).r;
        var sum = 0.0;
        let n0 = vec2<i32>(ii, jj);
        {
            let d = nh - textureLoad(src, n0, 0).r - u.talus;
            if (d > 0.0) { sum += d; }
        }
        let n1 = vec2<i32>(ii + 2, jj);
        if (n1.x < i32(u.width)) {
            let d = nh - textureLoad(src, n1, 0).r - u.talus;
            if (d > 0.0) { sum += d; }
        }
        let n2 = vec2<i32>(ii + 1, jj - 1);
        if (n2.y >= 0) {
            let d = nh - textureLoad(src, n2, 0).r - u.talus;
            if (d > 0.0) { sum += d; }
        }
        let n3 = vec2<i32>(ii + 1, jj + 1);
        if (n3.y < i32(u.height)) {
            let d = nh - textureLoad(src, n3, 0).r - u.talus;
            if (d > 0.0) { sum += d; }
        }
        h += outflow_share(nh, h0, nout, sum);
    }
    if (jj > 0) {
        let np = vec2<i32>(ii, jj - 1);
        let nh = textureLoad(src, np, 0).r;
        let nout = textureLoad(delta, np, 0).r;
        var sum = 0.0;
        let n0 = vec2<i32>(ii - 1, jj - 1);
        if (n0.x >= 0) {
            let d = nh - textureLoad(src, n0, 0).r - u.talus;
            if (d > 0.0) { sum += d; }
        }
        let n1 = vec2<i32>(ii + 1, jj - 1);
        if (n1.x < i32(u.width)) {
            let d = nh - textureLoad(src, n1, 0).r - u.talus;
            if (d > 0.0) { sum += d; }
        }
        let n2 = vec2<i32>(ii, jj - 2);
        if (n2.y >= 0) {
            let d = nh - textureLoad(src, n2, 0).r - u.talus;
            if (d > 0.0) { sum += d; }
        }
        let n3 = vec2<i32>(ii, jj);
        {
            let d = nh - textureLoad(src, n3, 0).r - u.talus;
            if (d > 0.0) { sum += d; }
        }
        h += outflow_share(nh, h0, nout, sum);
    }
    if (jj + 1 < i32(u.height)) {
        let np = vec2<i32>(ii, jj + 1);
        let nh = textureLoad(src, np, 0).r;
        let nout = textureLoad(delta, np, 0).r;
        var sum = 0.0;
        let n0 = vec2<i32>(ii - 1, jj + 1);
        if (n0.x >= 0) {
            let d = nh - textureLoad(src, n0, 0).r - u.talus;
            if (d > 0.0) { sum += d; }
        }
        let n1 = vec2<i32>(ii + 1, jj + 1);
        if (n1.x < i32(u.width)) {
            let d = nh - textureLoad(src, n1, 0).r - u.talus;
            if (d > 0.0) { sum += d; }
        }
        let n2 = vec2<i32>(ii, jj);
        {
            let d = nh - textureLoad(src, n2, 0).r - u.talus;
            if (d > 0.0) { sum += d; }
        }
        let n3 = vec2<i32>(ii, jj + 2);
        if (n3.y < i32(u.height)) {
            let d = nh - textureLoad(src, n3, 0).r - u.talus;
            if (d > 0.0) { sum += d; }
        }
        h += outflow_share(nh, h0, nout, sum);
    }

    textureStore(dst, p, vec4<f32>(h, 0.0, 0.0, 0.0));
}
