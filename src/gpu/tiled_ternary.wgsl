// Tiled ternary product over pre-quantized activations xq[m][k]
// (produced once per activation tensor by quantize_group128.wgsl).
// y[m][n] = sum_k xq[m][k] * scale[n][k / 128] * (code[n][k] - 1).
// Sixteen 2-bit codes sit in one u32, low bits first; k is a multiple of 128.
//
// Workgroup (16, 8, 1) = 128 threads computes a 32x64 output tile (4x4 per thread):
// - 128-bit coalesced vec4<f32> loads for x
// - Exactly 1 u32 code word + 1 f32 scale load per thread (zero redundant decode)
// - Pure fma() inner loop and 13,824 bytes workgroup storage (high occupancy on
//   RTX 4090 Vulkan/DX12 and Apple Silicon Metal)
// - 2x more M-tiles than 64x64, cutting wave quantization on short and medium clips.

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
@group(0) @binding(1) var<storage, read> codes: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<uniform> p: Params;

var<workgroup> xs: array<vec4<f32>, 288>;
var<workgroup> ws: array<vec4<f32>, 576>;

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
    let words = p.k >> 4u;
    let groups = p.k >> 7u;

    let x_row0 = linear >> 3u;
    let x_row1 = x_row0 + 16u;
    let k4 = linear & 7u;
    let x_at0 = x_row0 * PAD + k4;
    let x_at1 = x_row1 * PAD + k4;
    let g_x0 = row_base + x_row0;
    let g_x1 = row_base + x_row1;

    let w_col = linear >> 1u;
    let w_half = linear & 1u;
    let g_w = col_base + w_col;
    let w_at = w_col * PAD + w_half * 4u;

    let tiles = p.k / BK;
    for (var tile = 0u; tile < tiles; tile = tile + 1u) {
        let gk4 = tile * 8u + k4;
        var xv0 = vec4<f32>(0.0);
        if (g_x0 < p.m) { xv0 = x[g_x0 * k4_stride + gk4]; }
        xs[x_at0] = xv0;
        var xv1 = vec4<f32>(0.0);
        if (g_x1 < p.m) { xv1 = x[g_x1 * k4_stride + gk4]; }
        xs[x_at1] = xv1;

        var wv0 = vec4<f32>(0.0);
        var wv1 = vec4<f32>(0.0);
        var wv2 = vec4<f32>(0.0);
        var wv3 = vec4<f32>(0.0);
        if (g_w < p.n) {
            let word_idx = tile * 2u + w_half;
            let word = codes[g_w * words + word_idx];
            let scale = scales[g_w * groups + (word_idx >> 3u)];
            wv0 = vec4<f32>(
                f32((word >> 0u) & 3u) - 1.0,
                f32((word >> 2u) & 3u) - 1.0,
                f32((word >> 4u) & 3u) - 1.0,
                f32((word >> 6u) & 3u) - 1.0,
            ) * scale;
            wv1 = vec4<f32>(
                f32((word >> 8u) & 3u) - 1.0,
                f32((word >> 10u) & 3u) - 1.0,
                f32((word >> 12u) & 3u) - 1.0,
                f32((word >> 14u) & 3u) - 1.0,
            ) * scale;
            wv2 = vec4<f32>(
                f32((word >> 16u) & 3u) - 1.0,
                f32((word >> 18u) & 3u) - 1.0,
                f32((word >> 20u) & 3u) - 1.0,
                f32((word >> 22u) & 3u) - 1.0,
            ) * scale;
            wv3 = vec4<f32>(
                f32((word >> 24u) & 3u) - 1.0,
                f32((word >> 26u) & 3u) - 1.0,
                f32((word >> 28u) & 3u) - 1.0,
                f32((word >> 30u) & 3u) - 1.0,
            ) * scale;
        }
        ws[w_at + 0u] = wv0;
        ws[w_at + 1u] = wv1;
        ws[w_at + 2u] = wv2;
        ws[w_at + 3u] = wv3;
        workgroupBarrier();

        for (var step = 0u; step < 8u; step = step + 1u) {
            let x0 = xs[xr0 + step];
            let x1 = xs[xr1 + step];
            let x2 = xs[xr2 + step];
            let x3 = xs[xr3 + step];
            let w0 = ws[wc0 + step];
            let w1 = ws[wc1 + step];
            let w2 = ws[wc2 + step];
            let w3 = ws[wc3 + step];
            a00 = a00 + dot(x0, w0); a01 = a01 + dot(x0, w1); a02 = a02 + dot(x0, w2); a03 = a03 + dot(x0, w3);
            a10 = a10 + dot(x1, w0); a11 = a11 + dot(x1, w1); a12 = a12 + dot(x1, w2); a13 = a13 + dot(x1, w3);
            a20 = a20 + dot(x2, w0); a21 = a21 + dot(x2, w1); a22 = a22 + dot(x2, w2); a23 = a23 + dot(x2, w3);
            a30 = a30 + dot(x3, w0); a31 = a31 + dot(x3, w1); a32 = a32 + dot(x3, w2); a33 = a33 + dot(x3, w3);
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
