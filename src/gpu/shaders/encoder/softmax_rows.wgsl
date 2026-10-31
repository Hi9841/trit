// Softmax over each row, in place.
// row = exp(row - max(row)) / sum(exp(row - max(row))).
// There is no mask. Every column of the row is visible.
// Trit attention does this after the score scale. The CPU exp is Cephes.
// This shader uses the WGSL exp built-in.
// For attention, rows = heads * t and cols = t. Layout is [heads, t, t].
//
// Group 0:
//   binding 0  storage read_write  x  [rows, cols]
//   binding 1  uniform             Params  { rows, cols, pad, pad }
//
// Entry point: softmax_rows
// Workgroup: (256, 1, 1)
// Dispatch: (rows, 1, 1)
// One workgroup handles one row. cols must be at least 1.

const WG: u32 = 256u;

struct Params {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform> p: Params;

var<workgroup> scratch: array<f32, 256>;

fn neg_inf() -> f32 {
    return bitcast<f32>(0xff800000u);
}

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

fn reduce_max(lane: u32) -> f32 {
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            scratch[lane] = max(scratch[lane], scratch[lane + stride]);
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    return scratch[0];
}

@compute @workgroup_size(256)
fn softmax_rows(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.x;
    let lane = lid.x;
    let live = row < p.rows;
    var partial = neg_inf();
    if (live) {
        let base = row * p.cols;
        for (var i = lane; i < p.cols; i = i + WG) {
            partial = max(partial, x[base + i]);
        }
    }
    scratch[lane] = partial;
    let row_max = reduce_max(lane);

    partial = 0.0;
    if (live) {
        let base = row * p.cols;
        for (var i = lane; i < p.cols; i = i + WG) {
            let e = exp(x[base + i] - row_max);
            x[base + i] = e;
            partial = partial + e;
        }
    }
    scratch[lane] = partial;
    let inv = 1.0 / reduce_sum(lane);
    if (live) {
        let base = row * p.cols;
        for (var i = lane; i < p.cols; i = i + WG) {
            x[base + i] = x[base + i] * inv;
        }
    }
}
