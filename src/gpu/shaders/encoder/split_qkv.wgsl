// Split one stacked QKV row into Q, K, and V.
// qkv[frame] = [q (d) | k (d) | v (d)]. Head h of each part is d-major at h * hd.
// This pass does not add bias_u or bias_v. rel_pos_scores adds those.
// Trit: the qkv Linear is q_proj, k_proj, v_proj stacked on the output axis.
//
// Group 0:
//   binding 0  storage read        qkv  [t, 3 * d]
//   binding 1  storage read_write  q    [t, d]
//   binding 2  storage read_write  k    [t, d]
//   binding 3  storage read_write  v    [t, d]
//   binding 4  uniform             Params  { t, d, pad, pad }
//
// Entry point: split_qkv
// Workgroup: (256, 1, 1)
// Dispatch: (ceil(t * d / 256), 1, 1)

struct Params {
    t: u32,
    d: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> q: array<f32>;
@group(0) @binding(2) var<storage, read_write> k: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

@compute @workgroup_size(256)
fn split_qkv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = p.t * p.d;
    if (i >= n) {
        return;
    }
    let frame = i / p.d;
    let col = i - frame * p.d;
    let base = frame * (p.d + p.d + p.d);
    q[i] = qkv[base + col];
    k[i] = qkv[base + p.d + col];
    v[i] = qkv[base + p.d + p.d + col];
}
