// x = x * scale, element by element.
// Trit multiplies the subsampler output by sqrt(hidden) when scale_input is set.
// Skip this dispatch when the stored input_scale is 1.
//
// Group 0:
//   binding 0  storage read_write  x  [n]
//   binding 1  uniform             Params  { n, scale, pad, pad }
//
// Entry point: apply_scale
// Workgroup: (256, 1, 1)
// Dispatch: (ceil(n / 256), 1, 1)

struct Params {
    n: u32,
    scale: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform> p: Params;

@compute @workgroup_size(256)
fn apply_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < p.n) {
        x[i] = x[i] * p.scale;
    }
}
