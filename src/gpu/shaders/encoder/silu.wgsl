// Swish, which Trit calls SiLU: x = x / (1 + exp(-x)).
// The feed-forward modules and the depthwise conv output use this function.
// Trit: cpu/activation.rs silu. This shader uses the WGSL exp built-in.
// The CPU exp is the Cephes polynomial in that file, clamped to [-87.3, 88].
//
// Group 0:
//   binding 0  storage read_write  x  [n]
//   binding 1  uniform             Params  { n, pad, pad, pad }
//
// Entry point: silu
// Workgroup: (256, 1, 1)
// Dispatch: (ceil(n / 256), 1, 1)

struct Params {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform> p: Params;

@compute @workgroup_size(256)
fn silu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < p.n) {
        let v = x[i];
        x[i] = v / (1.0 + exp(-v));
    }
}
