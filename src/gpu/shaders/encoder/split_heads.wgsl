// Pack head h's lanes into a contiguous block.
// src is [rows, heads * hd], head h in columns h * hd .. (h + 1) * hd.
// dst is [heads, rows, hd].
// Trit does this once for the projected relative-position table (RelRows).
// Q, K, and V stay [t, d] and do not need this pass.
//
// Group 0:
//   binding 0  storage read        src  [rows, heads * hd]
//   binding 1  storage read_write  dst  [heads, rows, hd]
//   binding 2  uniform             Params  { rows, heads, hd, pad }
//
// Entry point: split_heads
// Workgroup: (256, 1, 1)
// Dispatch: (ceil(rows * heads * hd / 256), 1, 1)

struct Params {
    rows: u32,
    heads: u32,
    hd: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(256)
fn split_heads(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = p.heads * p.hd;
    let n = p.rows * d;
    let i = gid.x;
    if (i >= n) {
        return;
    }
    let row = i / d;
    let col = i - row * d;
    let head = col / p.hd;
    let e = col - head * p.hd;
    dst[(head * p.rows + row) * p.hd + e] = src[i];
}
