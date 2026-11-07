// Tiled ternary product.
// Each group of 128 activations is quantized, then multiplied.
// scale_x = max(abs(x)) / 127, or 1 when the max is 0.
// q = round(x / scale_x), ties to even, clamped to [-127, 127].
// The product uses q * scale_x. This matches the CPU group quantizer.
// y[m][n] = sum_k xq[m][k] * scale[n][k / 128] * (code[n][k] - 1).
// Sixteen codes sit in one u32, low bits first.
// One workgroup writes a 32 by 32 tile.
// The inner loop dots four K values.
// Bindings match ternary.wgsl.
// flags bit 0 adds bias[n]. flags bit 1 applies max(0, y).

const BM: u32 = 32u;
const BN: u32 = 32u;
const BK: u32 = 32u;
const PAD: u32 = 9u;
const GROUP: u32 = 128u;

struct Params {
    m: u32,
    n: u32,
    k: u32,
    flags: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> codes: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<uniform> p: Params;

// One row holds 128 floats as 32 vec4 values. 32 rows.
var<workgroup> raw: array<vec4<f32>, 1024>;
// [column in the tile][k / 4], with one spare vec4 per column.
var<workgroup> ws: array<vec4<f32>, 288>;
var<workgroup> part: array<f32, TILE * TILE>;
var<workgroup> row_max: array<f32, 32>;

fn decode(word: u32, lane: u32, scale: f32) -> f32 {
    let code = (word >> (lane * 2u)) & 3u;
    return scale * (f32(code) - 1.0);
}

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
    let words = p.k >> 4u;
    let groups = p.k >> 7u;
    if (k + 3u < p.k) {
        let word = codes[col * words + (k >> 4u)];
        let scale = scales[col * groups + (k >> 7u)];
        let lane = k & 15u;
        return vec4<f32>(
            decode(word, lane, scale),
            decode(word, lane + 1u, scale),
            decode(word, lane + 2u, scale),
            decode(word, lane + 3u, scale),
        );
    }
    var v = vec4<f32>(0.0);
    v.x = decode(codes[col * words + (k >> 4u)], k & 15u, scales[col * groups + (k >> 7u)]);
    if (k + 1u < p.k) {
        v.y = decode(
            codes[col * words + ((k + 1u) >> 4u)],
            (k + 1u) & 15u,
            scales[col * groups + ((k + 1u) >> 7u)],
        );
    }
    if (k + 2u < p.k) {
        v.z = decode(
            codes[col * words + ((k + 2u) >> 4u)],
            (k + 2u) & 15u,
            scales[col * groups + ((k + 2u) >> 7u)],
        );
    }
    return v;
}

fn quant_one(v: f32, inv: f32, scale: f32) -> f32 {
    return clamp(round(v * inv), -127.0, 127.0) * scale;
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
    let local = linear >> 3u;
    let lane = linear & 7u;

    let groups = (p.k + GROUP - 1u) / GROUP;
    for (var group = 0u; group < groups; group = group + 1u) {
        let group_k = group * GROUP;
        for (var sub = 0u; sub < 4u; sub = sub + 1u) {
            let gk = group_k + sub * BK + lane * 4u;
            raw[local * 32u + sub * 8u + lane] = load_x4(row_base + local, gk);
        }
        workgroupBarrier();

        var max_abs = 0.0;
        let basev = local * 32u + lane * 4u;
        for (var i = 0u; i < 4u; i = i + 1u) {
            let v = raw[basev + i];
            max_abs = max(max_abs, abs(v.x));
            max_abs = max(max_abs, abs(v.y));
            max_abs = max(max_abs, abs(v.z));
            max_abs = max(max_abs, abs(v.w));
        }
        part[linear] = max_abs;
        workgroupBarrier();
        if (lane == 0u) {
            var m = 0.0;
            let part_base = local * 8u;
            for (var i = 0u; i < 8u; i = i + 1u) {
                m = max(m, part[part_base + i]);
            }
            row_max[local] = m;
        }
        workgroupBarrier();

        let peak = row_max[local];
        var scale = 1.0;
        if (peak > 0.0) {
            scale = peak / 127.0;
        }
        let inv = 1.0 / scale;
        for (var i = 0u; i < 4u; i = i + 1u) {
            let v = raw[basev + i];
            raw[basev + i] = vec4<f32>(
                quant_one(v.x, inv, scale),
                quant_one(v.y, inv, scale),
                quant_one(v.z, inv, scale),
                quant_one(v.w, inv, scale),
            );
        }
        workgroupBarrier();

        for (var sub = 0u; sub < 4u; sub = sub + 1u) {
            let k0 = group_k + sub * BK;
            let gk = k0 + lane * 4u;
            ws[local * PAD + lane] = load_w4(col_base + local, gk);
            workgroupBarrier();
            let x_base = sub * 8u;
            for (var step = 0u; step < 8u; step = step + 1u) {
                let x0 = raw[r * 32u + x_base + step];
                let x1 = raw[(r + 1u) * 32u + x_base + step];
                let w0 = ws[c * PAD + step];
                let w1 = ws[(c + 1u) * PAD + step];
                a00 = a00 + dot(x0, w0);
                a01 = a01 + dot(x1, w0);
                a10 = a10 + dot(x0, w1);
                a11 = a11 + dot(x1, w1);
            }
            workgroupBarrier();
        }
    }

    store_one(row0, col0, a00);
    store_one(row0 + 1u, col0, a01);
    store_one(row0, col0 + 1u, a10);
    store_one(row0 + 1u, col0 + 1u, a11);
}
