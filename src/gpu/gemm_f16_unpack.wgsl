// The f32 product with f16 weights, without SHADER_F16.
// Each u32 holds two f16 values. unpack2x16float widens them.

const TILE: u32 = 16u;

struct Params {
    m: u32,
    n: u32,
    k: u32,
    flags: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

var<workgroup> xs: array<f32, 256>;
var<workgroup> ws: array<f32, 256>;

fn load_w(row: u32, col: u32) -> f32 {
    let i = row * p.k + col;
    let pair = unpack2x16float(w[i >> 1u]);
    return select(pair.x, pair.y, (i & 1u) == 1u);
}

@compute @workgroup_size(16, 16, 1)
fn gemm(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.y * TILE + lid.y;
    let col = wid.x * TILE + lid.x;
    var acc = 0.0;
    let tiles = (p.k + TILE - 1u) / TILE;
    for (var t = 0u; t < tiles; t = t + 1u) {
        let k0 = t * TILE;
        let xrow = wid.y * TILE + lid.y;
        let xcol = k0 + lid.x;
        var xv = 0.0;
        if (xrow < p.m && xcol < p.k) {
            xv = x[xrow * p.k + xcol];
        }
        xs[lid.y * TILE + lid.x] = xv;

        let wrow = wid.x * TILE + lid.x;
        let wcol = k0 + lid.y;
        var wv = 0.0;
        if (wrow < p.n && wcol < p.k) {
            wv = load_w(wrow, wcol);
        }
        ws[lid.x * TILE + lid.y] = wv;
        workgroupBarrier();

        for (var i = 0u; i < TILE; i = i + 1u) {
            acc = acc + xs[lid.y * TILE + i] * ws[lid.x * TILE + i];
        }
        workgroupBarrier();
    }
    if (row < p.m && col < p.n) {
        if ((p.flags & 1u) != 0u) {
            acc = acc + bias[col];
        }
        if ((p.flags & 2u) != 0u) {
            acc = max(acc, 0.0);
        }
        y[row * p.n + col] = acc;
    }
}
