// Tiled f32 product: y[m][n] = sum_k x[m][k] * w[n][k].
// One workgroup writes a 32 by 32 tile. 16 by 16 threads, two by two
// outputs each. The K tile is 32. The inner loop dots four K values.
// Bindings match gemm_f32.wgsl.
// flags bit 0 adds bias[n]. flags bit 1 applies max(0, y).

const BM: u32 = 32u;
const BN: u32 = 32u;
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

// [row or column in the tile][k / 4], with one spare vec4 per row.
var<workgroup> xs: array<vec4<f32>, 288>;
var<workgroup> ws: array<vec4<f32>, 288>;

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

fn store_one(row: u32, col: u32, value: f32) {
    if (row >= p.m || col >= p.n) {
        return;
    }
    var v = value;
    if ((p.flags & FLAG_BIAS) != 0u) {
        v = v + bias[col];
    }
    if ((p.flags & FLAG_RELU) != 0u) {
        v = max(v, 0.0);
    }
    y[row * p.n + col] = v;
}

@compute @workgroup_size(TILE, TILE, 1)
fn gemm(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) linear: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tx = lid.x;
    let ty = lid.y;
    let row0 = wid.y * BM + ty * 2u;
    let col0 = wid.x * BN + tx * 2u;
    let row_base = wid.y * BM;
    let col_base = wid.x * BN;
    var a00 = 0.0;
    var a01 = 0.0;
    var a10 = 0.0;
    var a11 = 0.0;
    let r = ty * 2u;
    let c = tx * 2u;

    let tiles = (p.k + BK - 1u) / BK;
    for (var tile = 0u; tile < tiles; tile = tile + 1u) {
        let k0 = tile * BK;
        let local = linear >> 3u;
        let k4 = linear & 7u;
        let gk = k0 + k4 * 4u;
        let at = local * PAD + k4;
        xs[at] = load_x4(row_base + local, gk);
        ws[at] = load_w4(col_base + local, gk);
        workgroupBarrier();

        for (var step = 0u; step < 8u; step = step + 1u) {
            let x0 = xs[r * PAD + step];
            let x1 = xs[(r + 1u) * PAD + step];
            let w0 = ws[c * PAD + step];
            let w1 = ws[(c + 1u) * PAD + step];
            a00 = a00 + dot(x0, w0);
            a01 = a01 + dot(x1, w0);
            a10 = a10 + dot(x0, w1);
            a11 = a11 + dot(x1, w1);
        }
        workgroupBarrier();
    }

    store_one(row0, col0, a00);
    store_one(row0 + 1u, col0, a01);
    store_one(row0, col0 + 1u, a10);
    store_one(row0 + 1u, col0 + 1u, a11);
}
