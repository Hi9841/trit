// Relative-position attention scores, one value per (head, query, key).
// score[h, i, j] = (dot(q[i, h] + u[h], k[j, h]) + dot(q[i, h] + v[h], R[h, i - j])) * scale
// scale is 1 / sqrt(hd). u is bias_u. v is bias_v. Both have length d = heads * hd.
// R is the projected position table [heads, 2 * reach + 1, hd].
// Row 0 of a head is relative position +reach. Row reach is position 0.
// The row for offset (i - j) is reach - (i - j).
//
// One 16x16 workgroup computes a 32-query x 64-key tile (2x4 outputs per thread,
// keys strided by 16 for coalesced stores), tiling hd in chunks of BK = 16.
// Workgroup storage: (32 + 32 + 64 + 95) * 5 * 16 = 17,840 bytes (fits 32 KB Metal limit).
//
// Entry point: rel_pos_scores
// Workgroup: (16, 16, 1)
// Dispatch: (ceil(t / 64), ceil(t / 32), heads)

const BQ: u32 = 32u;
const BN: u32 = 64u;
const BK: u32 = 16u;
// Extra vec4 per row so adjacent keys/rows do not share a bank.
const PAD: u32 = 5u;

struct Params {
    t: u32,
    hd: u32,
    heads: u32,
    reach: u32,
    scale: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> rel: array<f32>;
@group(0) @binding(3) var<storage, read> bias_u: array<f32>;
@group(0) @binding(4) var<storage, read> bias_v: array<f32>;
@group(0) @binding(5) var<storage, read_write> scores: array<f32>;
@group(0) @binding(6) var<uniform> p: Params;

var<workgroup> qu: array<vec4<f32>, 160>;
var<workgroup> qv: array<vec4<f32>, 160>;
var<workgroup> ks: array<vec4<f32>, 320>;
var<workgroup> rs: array<vec4<f32>, 475>;

@compute @workgroup_size(16, 16, 1)
fn rel_pos_scores(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) linear: u32,
) {
    let q_base = wid.y * BQ;
    let k_base = wid.x * BN;
    let head = wid.z;
    let tx = lid.x;
    let ty = lid.y;
    let d = p.heads * p.hd;
    let positions = p.reach + p.reach + 1u;
    let hbase = head * p.hd;
    let rel_min = i32(p.reach) + i32(k_base) - i32(q_base) - 31;

    let r0 = ty;
    let r1 = ty + 16u;
    let c0 = tx;
    let c1 = tx + 16u;
    let c2 = tx + 32u;
    let c3 = tx + 48u;

    var c00 = 0.0; var c01 = 0.0; var c02 = 0.0; var c03 = 0.0;
    var c10 = 0.0; var c11 = 0.0; var c12 = 0.0; var c13 = 0.0;
    var p00 = 0.0; var p01 = 0.0; var p02 = 0.0; var p03 = 0.0;
    var p10 = 0.0; var p11 = 0.0; var p12 = 0.0; var p13 = 0.0;

    let lr0_base = 31u - r0;
    let lr1_base = 31u - r1;

    let chunks = (p.hd + BK - 1u) / BK;
    for (var chunk = 0u; chunk < chunks; chunk = chunk + 1u) {
        let e0 = chunk * BK;
        if (linear < BQ * 4u) {
            let lq = linear >> 2u;
            let e4 = linear & 3u;
            let e = e0 + e4 * 4u;
            let query = q_base + lq;
            var qq = vec4<f32>(0.0);
            var vv = vec4<f32>(0.0);
            if (query < p.t && head < p.heads && e < p.hd) {
                let base = query * d + hbase + e;
                let ub = hbase + e;
                if (e + 3u < p.hd) {
                    let qval = vec4<f32>(q[base], q[base + 1u], q[base + 2u], q[base + 3u]);
                    qq = qval + vec4<f32>(bias_u[ub], bias_u[ub + 1u], bias_u[ub + 2u], bias_u[ub + 3u]);
                    vv = qval + vec4<f32>(bias_v[ub], bias_v[ub + 1u], bias_v[ub + 2u], bias_v[ub + 3u]);
                } else {
                    let q0 = q[base];
                    qq.x = q0 + bias_u[ub];
                    vv.x = q0 + bias_v[ub];
                    if (e + 1u < p.hd) {
                        let q1 = q[base + 1u];
                        qq.y = q1 + bias_u[ub + 1u];
                        vv.y = q1 + bias_v[ub + 1u];
                    }
                    if (e + 2u < p.hd) {
                        let q2 = q[base + 2u];
                        qq.z = q2 + bias_u[ub + 2u];
                        vv.z = q2 + bias_v[ub + 2u];
                    }
                }
            }
            qu[lq * PAD + e4] = qq;
            qv[lq * PAD + e4] = vv;
        }

        {
            let lk = linear >> 2u;
            let e4 = linear & 3u;
            let e = e0 + e4 * 4u;
            let key = k_base + lk;
            var kv = vec4<f32>(0.0);
            if (key < p.t && head < p.heads && e < p.hd) {
                let base = key * d + hbase + e;
                if (e + 3u < p.hd) {
                    kv = vec4<f32>(k[base], k[base + 1u], k[base + 2u], k[base + 3u]);
                } else {
                    kv.x = k[base];
                    if (e + 1u < p.hd) { kv.y = k[base + 1u]; }
                    if (e + 2u < p.hd) { kv.z = k[base + 2u]; }
                }
            }
            ks[lk * PAD + e4] = kv;
        }

        for (var step = 0u; step < 2u; step = step + 1u) {
            let id = step * 256u + linear;
            if (id < 95u * 4u) {
                let lr = id >> 2u;
                let e4 = id & 3u;
                let e = e0 + e4 * 4u;
                let rel_i = rel_min + i32(lr);
                var rv = vec4<f32>(0.0);
                if (head < p.heads && rel_i >= 0 && u32(rel_i) < positions && e < p.hd) {
                    let base = (head * positions + u32(rel_i)) * p.hd + e;
                    if (e + 3u < p.hd) {
                        rv = vec4<f32>(rel[base], rel[base + 1u], rel[base + 2u], rel[base + 3u]);
                    } else {
                        rv.x = rel[base];
                        if (e + 1u < p.hd) { rv.y = rel[base + 1u]; }
                        if (e + 2u < p.hd) { rv.z = rel[base + 2u]; }
                    }
                }
                rs[lr * PAD + e4] = rv;
            }
        }
        workgroupBarrier();

        for (var e4 = 0u; e4 < 4u; e4 = e4 + 1u) {
            let qq0 = qu[r0 * PAD + e4];
            let qv0 = qv[r0 * PAD + e4];
            let qq1 = qu[r1 * PAD + e4];
            let qv1 = qv[r1 * PAD + e4];

            let kv0 = ks[c0 * PAD + e4];
            let kv1 = ks[c1 * PAD + e4];
            let kv2 = ks[c2 * PAD + e4];
            let kv3 = ks[c3 * PAD + e4];

            c00 = c00 + dot(qq0, kv0); p00 = p00 + dot(qv0, rs[(c0 + lr0_base) * PAD + e4]);
            c01 = c01 + dot(qq0, kv1); p01 = p01 + dot(qv0, rs[(c1 + lr0_base) * PAD + e4]);
            c02 = c02 + dot(qq0, kv2); p02 = p02 + dot(qv0, rs[(c2 + lr0_base) * PAD + e4]);
            c03 = c03 + dot(qq0, kv3); p03 = p03 + dot(qv0, rs[(c3 + lr0_base) * PAD + e4]);

            c10 = c10 + dot(qq1, kv0); p10 = p10 + dot(qv1, rs[(c0 + lr1_base) * PAD + e4]);
            c11 = c11 + dot(qq1, kv1); p11 = p11 + dot(qv1, rs[(c1 + lr1_base) * PAD + e4]);
            c12 = c12 + dot(qq1, kv2); p12 = p12 + dot(qv1, rs[(c2 + lr1_base) * PAD + e4]);
            c13 = c13 + dot(qq1, kv3); p13 = p13 + dot(qv1, rs[(c3 + lr1_base) * PAD + e4]);
        }
        workgroupBarrier();
    }

    if (head < p.heads) {
        let k0 = k_base + c0;
        let k1 = k_base + c1;
        let k2 = k_base + c2;
        let k3 = k_base + c3;
        let g_r0 = q_base + r0;
        if (g_r0 < p.t) {
            let row = (head * p.t + g_r0) * p.t;
            if (k0 < p.t) { scores[row + k0] = (c00 + p00) * p.scale; }
            if (k1 < p.t) { scores[row + k1] = (c01 + p01) * p.scale; }
            if (k2 < p.t) { scores[row + k2] = (c02 + p02) * p.scale; }
            if (k3 < p.t) { scores[row + k3] = (c03 + p03) * p.scale; }
        }
        let g_r1 = q_base + r1;
        if (g_r1 < p.t) {
            let row = (head * p.t + g_r1) * p.t;
            if (k0 < p.t) { scores[row + k0] = (c10 + p10) * p.scale; }
            if (k1 < p.t) { scores[row + k1] = (c11 + p11) * p.scale; }
            if (k2 < p.t) { scores[row + k2] = (c12 + p12) * p.scale; }
            if (k3 < p.t) { scores[row + k3] = (c13 + p13) * p.scale; }
        }
    }
}
