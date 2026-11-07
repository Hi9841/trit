// Row layer norm for the Parakeet encoder.
// y = (x - mean) / sqrt(var + eps) * weight + bias.
// mean = sum(x) / d. var = sum((x - mean)^2) / d. The divisor is d.
// Trit: src/models/layers.rs Norm::apply and normalize_row.
// eps is the FastConformer norm epsilon.
//
// Do not bind y to the same buffer as x.
//
// Group 0:
//   binding 0  storage read        x       [rows, d]
//   binding 1  storage read        weight  [d]
//   binding 2  storage read        bias    [d]
//   binding 3  storage read_write  y       [rows, d]
//   binding 4  uniform             Params  { rows, d, eps, pad }
//
// Entry point: layer_norm
// Workgroup: (256, 1, 1)
// Dispatch: (rows, 1, 1)
// One workgroup reads one row and writes one row.

const WG: u32 = 256u;

struct Params {
    rows: u32,
    d: u32,
    eps: f32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

var<workgroup> scratch: array<f32, 256>;

fn reduce_sum(lane: u32) -> f32 {
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            scratch[lane] = scratch[lane] + scratch[lane + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    return scratch[0];
}

@compute @workgroup_size(256)
fn layer_norm(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.x;
    let lane = lid.x;
    let live = row < p.rows;
    var partial = 0.0;
    if (live) {
        let base = row * p.d;
        for (var i = lane; i < p.d; i = i + WG) {
            partial = partial + x[base + i];
        }
    }
    scratch[lane] = partial;
    let mean = reduce_sum(lane) / f32(p.d);

    partial = 0.0;
    if (live) {
        let base = row * p.d;
        for (var i = lane; i < p.d; i = i + WG) {
            let diff = x[base + i] - mean;
            partial = partial + diff * diff;
        }
    }
    scratch[lane] = partial;
    let inv = 1.0 / sqrt(reduce_sum(lane) / f32(p.d) + p.eps);

    if (live) {
        let base = row * p.d;
        for (var i = lane; i < p.d; i = i + WG) {
            let v = x[base + i];
            y[base + i] = (v - mean) * inv * weight[i] + bias[i];
        }
    }
}
