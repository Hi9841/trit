// Relative-position attention scores, one value per (head, query, key).
// score[h, i, j] = (dot(q[i, h] + u[h], k[j, h]) + dot(q[i, h] + v[h], R[h, i - j])) * scale
// scale is 1 / sqrt(hd). u is bias_u. v is bias_v. Both have length d = heads * hd.
// R is the projected position table [heads, 2 * reach + 1, hd].
// Row 0 of a head is relative position +reach. Row reach is position 0.
// The row for offset (i - j) is reach - (i - j). This is the rel-shift.
// Dispatch only when reach + 1 >= t, so every i - j is inside the table.
// There is no causal mask and no padding mask.
// One workgroup writes a 16 by 64 block of scores for one head.
// The K block and the relative rows are loaded once and reused for every query in the block.
//
// Group 0:
//   binding 0  storage read        q       [t, heads * hd]
//   binding 1  storage read        k       [t, heads * hd]
//   binding 2  storage read        rel     [heads, 2 * reach + 1, hd]
//   binding 3  storage read        bias_u  [heads * hd]
//   binding 4  storage read        bias_v  [heads * hd]
//   binding 5  storage read_write  scores  [heads, t, t]
//   binding 6  uniform             Params  { t, hd, heads, reach, scale, pad, pad, pad }
//
// Entry point: rel_pos_scores
// Workgroup: (16, 16, 1)
// Dispatch: (ceil(t / 64), ceil(t / 16), heads)
//   workgroup x: a tile of 64 keys
//   workgroup y: a tile of 16 queries
//   workgroup z: head index

const BQ: u32 = 16u;
const BN: u32 = 64u;
const BK: u32 = 32u;
// Extra vec4 so adjacent keys do not share a bank.
const PAD: u32 = 9u;

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

var<workgroup> qu: array<vec4<f32>, 144>;
var<workgroup> qv: array<vec4<f32>, 144>;
var<workgroup> ks: array<vec4<f32>, 576>;
var<workgroup> rs: array<vec4<f32>, 711>;

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
    let rel_min = i32(p.reach) + i32(k_base) - i32(q_base) - 15;
    var c0 = 0.0;
    var c1 = 0.0;
    var c2 = 0.0;
    var c3 = 0.0;
    var p0 = 0.0;
    var p1 = 0.0;
    var p2 = 0.0;
    var p3 = 0.0;

    let chunks = (p.hd + BK - 1u) / BK;
    for (var chunk = 0u; chunk < chunks; chunk = chunk + 1u) {
        let e0 = chunk * BK;
        if (linear < BQ * 8u) {
            let lq = linear >> 3u;
            let e4 = linear & 7u;
            let e = e0 + e4 * 4u;
            let query = q_base + lq;
            let ok = query < p.t && head < p.heads && e + 3u < p.hd;
            let base = query * d + hbase + e;
            var qq = vec4<f32>(0.0);
            var vv = vec4<f32>(0.0);
            if (ok) {
                let ub = hbase + e;
                qq = vec4<f32>(q[base], q[base + 1u], q[base + 2u], q[base + 3u]);
                vv = qq;
                qq = qq + vec4<f32>(bias_u[ub], bias_u[ub + 1u], bias_u[ub + 2u], bias_u[ub + 3u]);
                vv = vv + vec4<f32>(bias_v[ub], bias_v[ub + 1u], bias_v[ub + 2u], bias_v[ub + 3u]);
            }
            qu[lq * PAD + e4] = qq;
            qv[lq * PAD + e4] = vv;
        }

        for (var step = 0u; step < 2u; step = step + 1u) {
            let id = step * 256u + linear;
            let lk = id >> 3u;
            let e4 = id & 7u;
            let e = e0 + e4 * 4u;
            let key = k_base + lk;
            var kv = vec4<f32>(0.0);
            if (key < p.t && head < p.heads && e + 3u < p.hd) {
                let base = key * d + hbase + e;
                kv = vec4<f32>(k[base], k[base + 1u], k[base + 2u], k[base + 3u]);
            }
            ks[lk * PAD + e4] = kv;
        }

        for (var step = 0u; step < 3u; step = step + 1u) {
            let id = step * 256u + linear;
            if (id < 79u * 8u) {
                let lr = id >> 3u;
                let e4 = id & 7u;
                let e = e0 + e4 * 4u;
                let rel_i = rel_min + i32(lr);
                var rv = vec4<f32>(0.0);
                if (head < p.heads && rel_i >= 0 && u32(rel_i) < positions && e + 3u < p.hd) {
                    let base = (head * positions + u32(rel_i)) * p.hd + e;
                    rv = vec4<f32>(rel[base], rel[base + 1u], rel[base + 2u], rel[base + 3u]);
                }
                rs[lr * PAD + e4] = rv;
            }
        }
        workgroupBarrier();

        let query = q_base + ty;
        let lk0 = tx * 4u;
        if (query < p.t && head < p.heads) {
            for (var e4 = 0u; e4 < 8u; e4 = e4 + 1u) {
                if (e0 + e4 * 4u < p.hd) {
                    let qq = qu[ty * PAD + e4];
                    let vv = qv[ty * PAD + e4];
                    let lr0 = lk0 + 15u - ty;
                    c0 = c0 + dot(qq, ks[lk0 * PAD + e4]);
                    p0 = p0 + dot(vv, rs[lr0 * PAD + e4]);
                    c1 = c1 + dot(qq, ks[(lk0 + 1u) * PAD + e4]);
                    p1 = p1 + dot(vv, rs[(lr0 + 1u) * PAD + e4]);
                    c2 = c2 + dot(qq, ks[(lk0 + 2u) * PAD + e4]);
                    p2 = p2 + dot(vv, rs[(lr0 + 2u) * PAD + e4]);
                    c3 = c3 + dot(qq, ks[(lk0 + 3u) * PAD + e4]);
                    p3 = p3 + dot(vv, rs[(lr0 + 3u) * PAD + e4]);
                }
            }
        }
        workgroupBarrier();
    }

    let query = q_base + ty;
    let key0 = k_base + tx * 4u;
    if (query < p.t && head < p.heads) {
        let row = (head * p.t + query) * p.t;
        if (key0 < p.t) {
            scores[row + key0] = (c0 + p0) * p.scale;
        }
        if (key0 + 1u < p.t) {
            scores[row + key0 + 1u] = (c1 + p1) * p.scale;
        }
        if (key0 + 2u < p.t) {
            scores[row + key0 + 2u] = (c2 + p2) * p.scale;
        }
        if (key0 + 3u < p.t) {
            scores[row + key0 + 3u] = (c3 + p3) * p.scale;
        }
    }
}
