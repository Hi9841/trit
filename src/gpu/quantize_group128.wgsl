// Group-128 activation quantizer matching the CPU ternary quantizer and HEAD tiled_ternary.wgsl:
// scale = max(|x|) / 127 (or 1 when max == 0)
// xq = clamp(round(x / scale), -127, 127) * scale
//
// Uses 32 threads (one NVIDIA warp / Apple Silicon simdgroup), each handling
// one vec4<f32> (128-bit coalesced load/store) so 32 * 4 = 128 floats per group.
// Dispatch: (k / 128, m, 1), workgroup_size(32, 1, 1).

struct Params {
    m: u32,
    n: u32,
    k: u32,
    flags: u32,
}

@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> xq: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> p: Params;

var<workgroup> smax: array<f32, 32>;

@compute @workgroup_size(32, 1, 1)
fn quantize(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let group = wid.x;
    let row = wid.y;
    let lane = lid.x;
    let k4 = p.k >> 2u;
    let idx = row * k4 + group * 32u + lane;
    let live = row < p.m && (group * 32u + lane) < k4;
    var v = vec4<f32>(0.0);
    if (live) {
        v = x[idx];
    }
    // Filter NaN lanes before the max reduction so NaN never poisons the group peak.
    let av = select(vec4<f32>(0.0), abs(v), v == v);
    smax[lane] = max(max(av.x, av.y), max(av.z, av.w));
    workgroupBarrier();

    if (lane < 16u) { smax[lane] = max(smax[lane], smax[lane + 16u]); }
    workgroupBarrier();
    if (lane < 8u) { smax[lane] = max(smax[lane], smax[lane + 8u]); }
    workgroupBarrier();
    if (lane < 4u) { smax[lane] = max(smax[lane], smax[lane + 4u]); }
    workgroupBarrier();
    if (lane < 2u) { smax[lane] = max(smax[lane], smax[lane + 2u]); }
    workgroupBarrier();
    if (lane == 0u) { smax[0] = max(smax[0], smax[1]); }
    workgroupBarrier();

    let peak = smax[0];
    let scale = select(1.0, peak / 127.0, peak > 0.0);
    // Use fast multiplication when `1.0 / scale` is finite; fall back to division on
    // Inf or deep subnormals so `0.0 * Inf` / `Inf * 0.0` never poison valid lanes with NaN.
    let use_inv = scale > 1.0e-35 && scale < 3.4028234e38;
    let inv = select(1.0, 1.0 / scale, use_inv);
    if (live) {
        let scaled = select(v / scale, v * inv, use_inv);
        let q = select(
            vec4<f32>(0.0),
            clamp(round(scaled), vec4<f32>(-127.0), vec4<f32>(127.0)),
            scaled == scaled,
        );
        xq[idx] = q * scale;
    }
}
