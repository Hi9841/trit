// GLU over the last dimension.
// wide is [rows, 2d]. The first d lanes are the values. The second d lanes are the gate.
// out[r, c] = wide[r, c] * sigmoid(wide[r, d + c])
// sigmoid(v) = 1 / (1 + exp(-v))
// Trit: Block::convolution, after pointwise_conv1. The CPU sigmoid is activation::sigmoid.
// This shader uses the WGSL exp built-in.
//
// Group 0:
//   binding 0  storage read        wide  [rows, 2 * d]
//   binding 1  storage read_write  out   [rows, d]
//   binding 2  uniform             Params  { rows, d, pad, pad }
//
// Entry point: glu
// Workgroup: (256, 1, 1)
// Dispatch: (ceil(rows * d / 256), 1, 1)

struct Params {
    rows: u32,
    d: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> wide: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(256)
fn glu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = p.rows * p.d;
    if (i >= n) {
        return;
    }
    let row = i / p.d;
    let col = i - row * p.d;
    let base = row * (p.d + p.d);
    let value = wide[base + col];
    let gate = wide[base + p.d + col];
    let sigmoid = 1.0 / (1.0 + exp(-gate));
    out[i] = value * sigmoid;
}
