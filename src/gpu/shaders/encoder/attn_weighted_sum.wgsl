// Attention-weighted sum of values.
// out[i, h, e] = sum_j scores[h, i, j] * v[j, h, e]
// scores is [heads, t, t], after softmax_rows.
// v and out are [t, heads * hd], head h in columns h * hd.
// Trit stores V transposed per head and uses matmul_into. The values are the same.
// The output projection o_proj is the next Phase 2 GEMM, not this shader.
//
// Group 0:
//   binding 0  storage read        scores  [heads, t, t]
//   binding 1  storage read        v       [t, heads * hd]
//   binding 2  storage read_write  out     [t, heads * hd]
//   binding 3  uniform             Params  { t, hd, heads, pad }
//
// Entry point: attn_weighted_sum
// Workgroup: (256, 1, 1)
// Dispatch: (ceil(t * heads * hd / 256), 1, 1)
// One thread writes one output lane and loops over the t keys.

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

@compute @workgroup_size(256)
fn attn_weighted_sum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = p.heads * p.hd;
    let n = p.t * d;
    let i = gid.x;
    if (i >= n) {
        return;
    }
    let query = i / d;
    let col = i - query * d;
    let head = col / p.hd;
    let e = col - head * p.hd;
    let sbase = (head * p.t + query) * p.t;
    var acc = 0.0;
    for (var j = 0u; j < p.t; j = j + 1u) {
        let weight = scores[sbase + j];
        let value = v[j * d + head * p.hd + e];
        acc = acc + weight * value;
    }
    out[i] = acc;
}
