enable f16;

// Fast vectorized f16-weight product for k % 4 == 0:
// y[m][n] = sum_k x[m][k] * f32(w[n][k]).
// Workgroup (16, 8, 1) = 128 threads (4 warps / 4 Apple Silicon simdgroups)
// computes a 32x64 output tile (4x4 per thread). Uses 128-bit vec4<f32> loads
// for x, 64-bit vec4<f16> loads for w, f16 shared memory for w (9,216 bytes
// total -> 5 WGs/SM on RTX 4090 and 3 WGs/core on Apple Metal), and pure fma().

const BM: u32 = 32u;
const BN: u32 = 64u;
const BK: u32 = 32u;
const PAD: u32 = 9u;

struct Params {
    m: u32,
    n: u32,
    k: u32,
    flags: u32,
}

@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> w: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

var<workgroup> xs: array<vec4<f32>, 288>;
var<workgroup> ws: array<vec4<f16>, 576>;

fn mad4(acc: f32, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return fma(a.w, b.w, fma(a.z, b.z, fma(a.y, b.y, fma(a.x, b.x, acc))));
}

fn finish(value: f32, b: f32, has_bias: bool, has_relu: bool) -> f32 {
    var v = value;
    if (has_bias) {
        v = v + b;
    }
    if (has_relu) {
        v = max(v, 0.0);
    }
    return v;
}

@compute @workgroup_size(16, 8, 1)
fn gemm(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) linear: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tx = lid.x;
    let ty = lid.y;
    let row_base = wid.y * BM;
    let col_base = wid.x * BN;
    let r0 = ty;
    let r1 = ty + 8u;
    let r2 = ty + 16u;
    let r3 = ty + 24u;
    let c0 = tx;
    let c1 = tx + 16u;
    let c2 = tx + 32u;
    let c3 = tx + 48u;

    let xr0 = r0 * PAD;
    let xr1 = r1 * PAD;
    let xr2 = r2 * PAD;
    let xr3 = r3 * PAD;
    let wc0 = c0 * PAD;
    let wc1 = c1 * PAD;
    let wc2 = c2 * PAD;
    let wc3 = c3 * PAD;

    var a00 = 0.0; var a01 = 0.0; var a02 = 0.0; var a03 = 0.0;
    var a10 = 0.0; var a11 = 0.0; var a12 = 0.0; var a13 = 0.0;
    var a20 = 0.0; var a21 = 0.0; var a22 = 0.0; var a23 = 0.0;
    var a30 = 0.0; var a31 = 0.0; var a32 = 0.0; var a33 = 0.0;

    let k4_stride = p.k >> 2u;
    let local = linear >> 3u;
    let k4 = linear & 7u;
    let at0 = local * PAD + k4;
    let at1 = (local + 16u) * PAD + k4;
    let at2 = (local + 32u) * PAD + k4;
    let at3 = (local + 48u) * PAD + k4;

    let g_x0 = row_base + local;
    let g_x1 = row_base + local + 16u;
    let g_w0 = col_base + local;
    let g_w1 = col_base + local + 16u;
    let g_w2 = col_base + local + 32u;
    let g_w3 = col_base + local + 48u;

    let tiles = (k4_stride + 7u) / 8u;
    for (var tile = 0u; tile < tiles; tile = tile + 1u) {
        let gk4 = tile * 8u + k4;
        let in_k = gk4 < k4_stride;
        var xv0 = vec4<f32>(0.0);
        var xv1 = vec4<f32>(0.0);
        var wv0 = vec4<f16>(0.0h);
        var wv1 = vec4<f16>(0.0h);
        var wv2 = vec4<f16>(0.0h);
        var wv3 = vec4<f16>(0.0h);
        if (in_k) {
            if (g_x0 < p.m) { xv0 = x[g_x0 * k4_stride + gk4]; }
            if (g_x1 < p.m) { xv1 = x[g_x1 * k4_stride + gk4]; }
            if (g_w0 < p.n) { wv0 = w[g_w0 * k4_stride + gk4]; }
            if (g_w1 < p.n) { wv1 = w[g_w1 * k4_stride + gk4]; }
            if (g_w2 < p.n) { wv2 = w[g_w2 * k4_stride + gk4]; }
            if (g_w3 < p.n) { wv3 = w[g_w3 * k4_stride + gk4]; }
        }
        xs[at0] = xv0;
        xs[at1] = xv1;
        ws[at0] = wv0;
        ws[at1] = wv1;
        ws[at2] = wv2;
        ws[at3] = wv3;
        workgroupBarrier();

        for (var step = 0u; step < 8u; step = step + 1u) {
            let x0 = xs[xr0 + step];
            let x1 = xs[xr1 + step];
            let x2 = xs[xr2 + step];
            let x3 = xs[xr3 + step];
            let w0 = vec4<f32>(ws[wc0 + step]);
            let w1 = vec4<f32>(ws[wc1 + step]);
            let w2 = vec4<f32>(ws[wc2 + step]);
            let w3 = vec4<f32>(ws[wc3 + step]);
            a00 = mad4(a00, x0, w0); a01 = mad4(a01, x0, w1); a02 = mad4(a02, x0, w2); a03 = mad4(a03, x0, w3);
            a10 = mad4(a10, x1, w0); a11 = mad4(a11, x1, w1); a12 = mad4(a12, x1, w2); a13 = mad4(a13, x1, w3);
            a20 = mad4(a20, x2, w0); a21 = mad4(a21, x2, w1); a22 = mad4(a22, x2, w2); a23 = mad4(a23, x2, w3);
            a30 = mad4(a30, x3, w0); a31 = mad4(a31, x3, w1); a32 = mad4(a32, x3, w2); a33 = mad4(a33, x3, w3);
        }
        workgroupBarrier();
    }

    let has_bias = (p.flags & FLAG_BIAS) != 0u;
    let has_relu = (p.flags & FLAG_RELU) != 0u;
    let g_c0 = col_base + c0;
    let g_c1 = col_base + c1;
    let g_c2 = col_base + c2;
    let g_c3 = col_base + c3;
    var b0 = 0.0;
    var b1 = 0.0;
    var b2 = 0.0;
    var b3 = 0.0;
    if (has_bias) {
        if (g_c0 < p.n) { b0 = bias[g_c0]; }
        if (g_c1 < p.n) { b1 = bias[g_c1]; }
        if (g_c2 < p.n) { b2 = bias[g_c2]; }
        if (g_c3 < p.n) { b3 = bias[g_c3]; }
    }

    let g_r0 = row_base + r0;
    let g_r1 = row_base + r1;
    let g_r2 = row_base + r2;
    let g_r3 = row_base + r3;

    if (g_r0 < p.m) {
        let base = g_r0 * p.n;
        if (g_c0 < p.n) { y[base + g_c0] = finish(a00, b0, has_bias, has_relu); }
        if (g_c1 < p.n) { y[base + g_c1] = finish(a01, b1, has_bias, has_relu); }
        if (g_c2 < p.n) { y[base + g_c2] = finish(a02, b2, has_bias, has_relu); }
        if (g_c3 < p.n) { y[base + g_c3] = finish(a03, b3, has_bias, has_relu); }
    }
    if (g_r1 < p.m) {
        let base = g_r1 * p.n;
        if (g_c0 < p.n) { y[base + g_c0] = finish(a10, b0, has_bias, has_relu); }
        if (g_c1 < p.n) { y[base + g_c1] = finish(a11, b1, has_bias, has_relu); }
        if (g_c2 < p.n) { y[base + g_c2] = finish(a12, b2, has_bias, has_relu); }
        if (g_c3 < p.n) { y[base + g_c3] = finish(a13, b3, has_bias, has_relu); }
    }
    if (g_r2 < p.m) {
        let base = g_r2 * p.n;
        if (g_c0 < p.n) { y[base + g_c0] = finish(a20, b0, has_bias, has_relu); }
        if (g_c1 < p.n) { y[base + g_c1] = finish(a21, b1, has_bias, has_relu); }
        if (g_c2 < p.n) { y[base + g_c2] = finish(a22, b2, has_bias, has_relu); }
        if (g_c3 < p.n) { y[base + g_c3] = finish(a23, b3, has_bias, has_relu); }
    }
    if (g_r3 < p.m) {
        let base = g_r3 * p.n;
        if (g_c0 < p.n) { y[base + g_c0] = finish(a30, b0, has_bias, has_relu); }
        if (g_c1 < p.n) { y[base + g_c1] = finish(a31, b1, has_bias, has_relu); }
        if (g_c2 < p.n) { y[base + g_c2] = finish(a32, b2, has_bias, has_relu); }
        if (g_c3 < p.n) { y[base + g_c3] = finish(a33, b3, has_bias, has_relu); }
    }
}
