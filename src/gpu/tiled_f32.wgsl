// Tiled f32 product: y[m][n] = sum_k x[m][k] * w[n][k].
// One 16x16 workgroup computes a 64x64 output tile (4x4 outputs per thread,
// strided by 16 for coalesced stores and bank-conflict-free shared reads).
// Workgroup storage: 2 * 64 * 9 * 16 = 18,432 bytes (fits 32 KB Metal limit).
// Bindings match gemm_f32.wgsl.
// flags bit 0 adds bias[n]. flags bit 1 applies max(0, y).

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 32u;
const PAD: u32 = 9u;

struct Params {
    m: u32,
    n: u32,
    k: u32,
    flags: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

// [row or column in the 64-wide tile][k / 4], with one spare vec4 per row.
var<workgroup> xs: array<vec4<f32>, 576>;
var<workgroup> ws: array<vec4<f32>, 576>;

fn load_x4(row: u32, k: u32) -> vec4<f32> {
    if (row >= p.m || k >= p.k) {
        return vec4<f32>(0.0);
    }
    let base = row * p.k + k;
    if (k + 3u < p.k) {
        return vec4<f32>(x[base], x[base + 1u], x[base + 2u], x[base + 3u]);
    }
    var v = vec4<f32>(0.0);
    v.x = x[base];
    if (k + 1u < p.k) {
        v.y = x[base + 1u];
    }
    if (k + 2u < p.k) {
        v.z = x[base + 2u];
    }
    return v;
}

fn load_w4(col: u32, k: u32) -> vec4<f32> {
    if (col >= p.n || k >= p.k) {
        return vec4<f32>(0.0);
    }
    let base = col * p.k + k;
    if (k + 3u < p.k) {
        return vec4<f32>(w[base], w[base + 1u], w[base + 2u], w[base + 3u]);
    }
    var v = vec4<f32>(0.0);
    v.x = w[base];
    if (k + 1u < p.k) {
        v.y = w[base + 1u];
    }
    if (k + 2u < p.k) {
        v.z = w[base + 2u];
    }
    return v;
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

@compute @workgroup_size(TILE, TILE, 1)
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
    let r1 = ty + 16u;
    let r2 = ty + 32u;
    let r3 = ty + 48u;
    let c0 = tx;
    let c1 = tx + 16u;
    let c2 = tx + 32u;
    let c3 = tx + 48u;

    var a00 = 0.0; var a01 = 0.0; var a02 = 0.0; var a03 = 0.0;
    var a10 = 0.0; var a11 = 0.0; var a12 = 0.0; var a13 = 0.0;
    var a20 = 0.0; var a21 = 0.0; var a22 = 0.0; var a23 = 0.0;
    var a30 = 0.0; var a31 = 0.0; var a32 = 0.0; var a33 = 0.0;

    let local0 = linear >> 3u;
    let local1 = local0 + 32u;
    let k4 = linear & 7u;
    let at0 = local0 * PAD + k4;
    let at1 = local1 * PAD + k4;

    let tiles = (p.k + BK - 1u) / BK;
    for (var tile = 0u; tile < tiles; tile = tile + 1u) {
        let gk = tile * BK + k4 * 4u;
        xs[at0] = load_x4(row_base + local0, gk);
        xs[at1] = load_x4(row_base + local1, gk);
        ws[at0] = load_w4(col_base + local0, gk);
        ws[at1] = load_w4(col_base + local1, gk);
        workgroupBarrier();

        for (var step = 0u; step < 8u; step = step + 1u) {
            let x0 = xs[r0 * PAD + step];
            let x1 = xs[r1 * PAD + step];
            let x2 = xs[r2 * PAD + step];
            let x3 = xs[r3 * PAD + step];
            let w0 = ws[c0 * PAD + step];
            let w1 = ws[c1 * PAD + step];
            let w2 = ws[c2 * PAD + step];
            let w3 = ws[c3 * PAD + step];
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
