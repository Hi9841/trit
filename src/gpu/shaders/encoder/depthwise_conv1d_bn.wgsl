// Depthwise conv1d over time, then the folded batch-norm affine.
// One output lane per (frame, channel). Channels do not mix.
// pad = (kernel - 1) / 2, integer division, same as time_conv_row.
// For tap k, the source frame is frame + k - pad.
// A source frame outside 0 .. t - 1 adds nothing. The accumulator starts at 0.
// out = acc * bn_scale + bn_shift.
// bn_scale and bn_shift already include gamma, running var, beta, running mean,
// and the depthwise bias. Do not scale the taps by bn_scale again.
// Weight layout is [kernel, d], tap-major, one tap then the next.
// Trit: Block::convolution and time_conv_row. SiLU is the next dispatch, not this one.
//
// Group 0:
//   binding 0  storage read        x         [t, d]
//   binding 1  storage read        weight    [kernel, d]
//   binding 2  storage read        bn_scale  [d]
//   binding 3  storage read        bn_shift  [d]
//   binding 4  storage read_write  y         [t, d]
//   binding 5  uniform             Params  { t, d, kernel, pad }
//
// Entry point: depthwise_conv1d_bn
// Workgroup: (256, 1, 1)
// Dispatch: (ceil(t * d / 256), 1, 1)

struct Params {
    t: u32,
    d: u32,
    kernel: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bn_scale: array<f32>;
@group(0) @binding(3) var<storage, read> bn_shift: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<uniform> p: Params;

@compute @workgroup_size(256)
fn depthwise_conv1d_bn(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = p.t * p.d;
    if (i >= n) {
        return;
    }
    let frame = i / p.d;
    let channel = i - frame * p.d;
    let pad = i32(p.kernel - 1u) / 2;
    var acc = 0.0;
    for (var k = 0u; k < p.kernel; k = k + 1u) {
        let src = i32(frame) + i32(k) - pad;
        if (src >= 0 && src < i32(p.t)) {
            let xv = x[u32(src) * p.d + channel];
            let wv = weight[k * p.d + channel];
            acc = acc + wv * xv;
        }
    }
    y[i] = acc * bn_scale[channel] + bn_shift[channel];
}
