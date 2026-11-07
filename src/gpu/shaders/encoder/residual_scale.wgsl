// x = x + scale * y, element by element.
// Trit: src/models/layers.rs add_scaled, called from Norm::residual.
// The scale is the FastConformer residual weight for this step.
//
// x and y must be different buffers.
//
// Group 0:
//   binding 0  storage read_write  x  [n]
//   binding 1  storage read        y  [n]
//   binding 2  uniform             Params  { n, scale, pad, pad }
//
// Entry point: residual_scale
// Workgroup: (256, 1, 1)
// Dispatch: (ceil(n / 256), 1, 1)

struct Params {
    n: u32,
    scale: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(256)
fn residual_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < p.n) {
        x[i] = x[i] + p.scale * y[i];
    }
}
