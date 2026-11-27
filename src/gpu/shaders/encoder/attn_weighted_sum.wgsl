// Tiled attention-weighted sum of values.
// out[i, h, e] = sum_j scores[h, i, j] * v[j, h, e]
// scores is [heads, t, t], after softmax_rows.
// v and out are [t, heads * hd], head h in columns h * hd.
//
// One 16x16 workgroup computes a 32-query x 64-dim tile for one head,
// tiling the key dimension in chunks of BK = 32 in shared memory so each
// scores tile is reused across 64 dims and each v tile across 32 queries.
// Workgroup storage: (32 * 9 + 32 * 17) * 16 = 13,312 bytes.
//
// Entry point: attn_weighted_sum
// Workgroup: (16, 16, 1)
// Dispatch: (ceil(hd / 64), ceil(t / 32), heads)

const BM: u32 = 32u;
const BN: u32 = 64u;
const BK: u32 = 32u;
const SPAD: u32 = 9u;
const VPAD: u32 = 17u;

struct Params {
    t: u32,
    hd: u32,
    heads: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(1) var<storage, read> v: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

// ss[local_q * 9 + k4] = 4 consecutive attention weights for one query.
var<workgroup> ss: array<vec4<f32>, 288>;
// vs[local_k * 17 + d4] = 4 consecutive value dims for one key.
var<workgroup> vs: array<vec4<f32>, 544>;

fn load_s4(head: u32, query: u32, k: u32) -> vec4<f32> {
    if (head >= p.heads || query >= p.t || k >= p.t) {
        return vec4<f32>(0.0);
    }
    let base = (head * p.t + query) * p.t + k;
    if (k + 3u < p.t) {
        return vec4<f32>(scores[base], scores[base + 1u], scores[base + 2u], scores[base + 3u]);
    }
    var r = vec4<f32>(0.0);
    r.x = scores[base];
    if (k + 1u < p.t) { r.y = scores[base + 1u]; }
    if (k + 2u < p.t) { r.z = scores[base + 2u]; }
    return r;
}

fn load_v4(head: u32, key: u32, e: u32, d: u32) -> vec4<f32> {
    if (head >= p.heads || key >= p.t || e >= p.hd) {
        return vec4<f32>(0.0);
    }
    let base = key * d + head * p.hd + e;
    if (e + 3u < p.hd) {
        return vec4<f32>(v[base], v[base + 1u], v[base + 2u], v[base + 3u]);
    }
    var r = vec4<f32>(0.0);
    r.x = v[base];
    if (e + 1u < p.hd) { r.y = v[base + 1u]; }
    if (e + 2u < p.hd) { r.z = v[base + 2u]; }
    return r;
}

@compute @workgroup_size(16, 16, 1)
fn attn_weighted_sum(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) linear: u32,
) {
    let d_base = wid.x * BN;
    let q_base = wid.y * BM;
    let head = wid.z;
    let tx = lid.x;
    let ty = lid.y;
    let d = p.heads * p.hd;

    let r0 = ty;
    let r1 = ty + 16u;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);

    let s_q = linear >> 3u;
    let s_k4 = linear & 7u;
    let v_k0 = linear >> 4u;
    let v_k1 = v_k0 + 16u;
    let v_d4 = linear & 15u;

    let tiles = (p.t + BK - 1u) / BK;
    for (var tile = 0u; tile < tiles; tile = tile + 1u) {
        let k0 = tile * BK;
        ss[s_q * SPAD + s_k4] = load_s4(head, q_base + s_q, k0 + s_k4 * 4u);
        vs[v_k0 * VPAD + v_d4] = load_v4(head, k0 + v_k0, d_base + v_d4 * 4u, d);
        vs[v_k1 * VPAD + v_d4] = load_v4(head, k0 + v_k1, d_base + v_d4 * 4u, d);
        workgroupBarrier();

        for (var step = 0u; step < 8u; step = step + 1u) {
            let w0 = ss[r0 * SPAD + step];
            let w1 = ss[r1 * SPAD + step];
            let kb = step * 4u;
            let vv0 = vs[(kb + 0u) * VPAD + tx];
            let vv1 = vs[(kb + 1u) * VPAD + tx];
            let vv2 = vs[(kb + 2u) * VPAD + tx];
            let vv3 = vs[(kb + 3u) * VPAD + tx];
            acc0 = acc0 + w0.x * vv0 + w0.y * vv1 + w0.z * vv2 + w0.w * vv3;
            acc1 = acc1 + w1.x * vv0 + w1.y * vv1 + w1.z * vv2 + w1.w * vv3;
        }
        workgroupBarrier();
    }

    let e0 = d_base + tx * 4u;
    if (head < p.heads && e0 < p.hd) {
        let g_r0 = q_base + r0;
        if (g_r0 < p.t) {
            let base = g_r0 * d + head * p.hd + e0;
            out[base] = acc0.x;
            if (e0 + 1u < p.hd) { out[base + 1u] = acc0.y; }
            if (e0 + 2u < p.hd) { out[base + 2u] = acc0.z; }
            if (e0 + 3u < p.hd) { out[base + 3u] = acc0.w; }
        }
        let g_r1 = q_base + r1;
        if (g_r1 < p.t) {
            let base = g_r1 * d + head * p.hd + e0;
            out[base] = acc1.x;
            if (e0 + 1u < p.hd) { out[base + 1u] = acc1.y; }
            if (e0 + 2u < p.hd) { out[base + 2u] = acc1.z; }
            if (e0 + 3u < p.hd) { out[base + 3u] = acc1.w; }
        }
    }
}
