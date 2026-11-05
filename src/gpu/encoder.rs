//! The Parakeet conformer on the GPU.
//! Subsampling stays on the CPU. One call uploads the frames, runs every
//! layer, and reads the encoder output back. Weights stay on the device.

use super::{pipeline, readback, Gpu, Layer, Weights};
use crate::cpu::buf::Buf;
use parking_lot::Mutex;
use std::num::NonZeroU64;

const STRIDE: usize = 256;
/// Output tile of the tiled product. The workgroup is 16 by 16 threads.
const TILE: u32 = 32;
/// Output tile of the f16 product from Phase 2.
const SMALL: u32 = 16;
const LANE: u32 = 256;
/// Timestamp slots for one profiled encode. 24 layers use about 630.
const STAMPS: u32 = 1024;

const S_LN: usize = 0;
const S_SILU_F: usize = 1;
const S_SILU_D: usize = 2;
const S_HALF: usize = 3;
const S_ONE: usize = 4;
const S_SPLIT: usize = 5;
const S_REL: usize = 6;
const S_SOFTMAX: usize = 7;
const S_ATTN: usize = 8;
const S_GLU: usize = 9;
const S_DW: usize = 10;
const S_SCALE: usize = 11;
const S_HEADS: usize = 12;
const S_BASE: usize = 13;
const MATS: usize = 8;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Quad {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ScaleParams {
    n: u32,
    scale: f32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RelParams {
    t: u32,
    hd: u32,
    heads: u32,
    reach: u32,
    scale: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[derive(Clone, Copy, Debug)]
enum SlotKind {
    Read,
    Write,
    Uniform,
}

/// Widths of one Parakeet encoder.
pub(crate) struct Cfg {
    pub d: u32,
    pub f: u32,
    pub heads: u32,
    pub kernel: u32,
    /// `sqrt(hidden)` or 1. The value 1 skips the scale dispatch.
    pub input_scale: f32,
    /// Smallest reach the growth rule keeps. This is `POSITION_REACH`.
    pub min_reach: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MatKind {
    F32,
    F16,
    Ternary,
}

/// One uploaded product. The buffers are the `gpu::Layer` buffers.
pub(crate) struct Mat {
    kind: MatKind,
    n: u32,
    k: u32,
    flags: u32,
    weight: wgpu::Buffer,
    scales: Option<wgpu::Buffer>,
    bias: Option<wgpu::Buffer>,
}

impl Mat {
    pub(crate) fn from_layer(layer: &Layer) -> Self {
        let flags = u32::from(layer.bias().is_some());
        let (kind, weight, scales) = match layer.weights() {
            Weights::F32(weight) => (MatKind::F32, weight.clone(), None),
            Weights::F16(weight) => (MatKind::F16, weight.clone(), None),
            Weights::Ternary { codes, scales } => {
                (MatKind::Ternary, codes.clone(), Some(scales.clone()))
            }
        };
        Self {
            kind,
            n: layer.rows(),
            k: layer.cols(),
            flags,
            weight,
            scales,
            bias: layer.bias().cloned(),
        }
    }
}

/// Host weights for one conformer layer, plus the products already on the device.
pub(crate) struct LayerUpload {
    /// Layer norms in order: ff1, attention, conv, ff2, out. Each pair is weight, bias.
    pub norms: Vec<(Vec<f32>, Vec<f32>)>,
    /// Products in order: ff1 up, ff1 down, qkv, o, pointwise 1, pointwise 2, ff2 up, ff2 down.
    pub mats: Vec<Mat>,
    pub rel_k: Mat,
    pub bias_u: Vec<f32>,
    pub bias_v: Vec<f32>,
    pub dw: Vec<f32>,
    pub bn_scale: Vec<f32>,
    pub bn_shift: Vec<f32>,
}

struct Pipes {
    ln: wgpu::ComputePipeline,
    residual: wgpu::ComputePipeline,
    scale: wgpu::ComputePipeline,
    silu: wgpu::ComputePipeline,
    glu: wgpu::ComputePipeline,
    dw: wgpu::ComputePipeline,
    softmax: wgpu::ComputePipeline,
    rel: wgpu::ComputePipeline,
    attn: wgpu::ComputePipeline,
    split_qkv: wgpu::ComputePipeline,
    split_heads: wgpu::ComputePipeline,
    gemm_f32: wgpu::ComputePipeline,
    gemm_f16: wgpu::ComputePipeline,
    ternary: wgpu::ComputePipeline,
    f32_tile: u32,
    ternary_tile: u32,
    ln_layout: wgpu::BindGroupLayout,
    residual_layout: wgpu::BindGroupLayout,
    scale_layout: wgpu::BindGroupLayout,
    silu_layout: wgpu::BindGroupLayout,
    glu_layout: wgpu::BindGroupLayout,
    dw_layout: wgpu::BindGroupLayout,
    softmax_layout: wgpu::BindGroupLayout,
    rel_layout: wgpu::BindGroupLayout,
    attn_layout: wgpu::BindGroupLayout,
    split_layout: wgpu::BindGroupLayout,
    heads_layout: wgpu::BindGroupLayout,
    gemm_layout: wgpu::BindGroupLayout,
    ternary_layout: wgpu::BindGroupLayout,
}

struct Act {
    cap: u32,
    a: wgpu::Buffer,
    b: wgpu::Buffer,
    branch: wgpu::Buffer,
    hidden: wgpu::Buffer,
    qkv: wgpu::Buffer,
    q: wgpu::Buffer,
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    scores: wgpu::Buffer,
    ctx: wgpu::Buffer,
    wide: wgpu::Buffer,
    gated: wgpu::Buffer,
    sine: wgpu::Buffer,
    projected: wgpu::Buffer,
    table_rows: u32,
    staging: wgpu::Buffer,
    staging_bytes: u64,
}

struct LayerW {
    norms: Vec<(wgpu::Buffer, wgpu::Buffer)>,
    mats: Vec<Mat>,
    rel_k: Mat,
    bias_u: wgpu::Buffer,
    bias_v: wgpu::Buffer,
    dw: wgpu::Buffer,
    bn_scale: wgpu::Buffer,
    bn_shift: wgpu::Buffer,
    rel: wgpu::Buffer,
}

struct GemmBind {
    /// 0 f32, 1 f16, 2 ternary.
    pipe: u8,
    group: wgpu::BindGroup,
    n: u32,
    tile: u32,
}

struct LayerBind {
    lns: Vec<wgpu::BindGroup>,
    res_half: wgpu::BindGroup,
    res_one: wgpu::BindGroup,
    /// `stream += pointwise 2`. That product writes the norm buffer, not `branch`.
    res_conv: wgpu::BindGroup,
    gemms: Vec<GemmBind>,
    dw: wgpu::BindGroup,
    rel: wgpu::BindGroup,
}

struct Bound {
    layers: Vec<LayerBind>,
    silu_f: wgpu::BindGroup,
    silu_d: wgpu::BindGroup,
    split: wgpu::BindGroup,
    softmax: wgpu::BindGroup,
    attn: wgpu::BindGroup,
    glu: wgpu::BindGroup,
    scale: Option<wgpu::BindGroup>,
    cap: u32,
    reach: u32,
}

struct Inner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    instance: wgpu::Instance,
    dummy: wgpu::Buffer,
    params: wgpu::Buffer,
    params_cpu: Vec<u8>,
    pipes: Pipes,
    cfg: Cfg,
    layers: Vec<LayerW>,
    act: Act,
    bound: Option<Bound>,
    reach: u32,
    limit: u32,
    profile: bool,
}

/// The resident encoder. Calls share the activation buffers.
pub(crate) struct Encoder {
    inner: Mutex<Inner>,
}

impl Encoder {
    /// Upload the norms and the position table. `sine` is the sinusoid for `reach`.
    pub(crate) fn build(
        cfg: Cfg,
        uploads: Vec<LayerUpload>,
        sine: &[f32],
        reach: u32,
    ) -> Result<Self, String> {
        if cfg.heads == 0 || !cfg.d.is_multiple_of(cfg.heads) {
            return Err("encoder heads do not divide the hidden size".into());
        }
        let gpu = Gpu::get()?;
        let align = gpu.device().limits().min_uniform_buffer_offset_alignment as usize;
        if align > STRIDE {
            return Err(format!("GPU uniform alignment is {align}, above {STRIDE}"));
        }
        let pipes = build_pipes(gpu)?;
        let mut layers = Vec::with_capacity(uploads.len());
        for upload in uploads {
            if upload.norms.len() != 5 || upload.mats.len() != MATS {
                return Err("encoder layer is missing a weight".into());
            }
            let expect = cfg.kernel as usize * cfg.d as usize;
            if upload.dw.len() != expect
                || upload.bn_scale.len() != cfg.d as usize
                || upload.bn_shift.len() != cfg.d as usize
                || upload.bias_u.len() != cfg.d as usize
                || upload.bias_v.len() != cfg.d as usize
            {
                return Err("encoder conv weights do not match the config".into());
            }
            let mut norms = Vec::with_capacity(5);
            for (index, (weight, bias)) in upload.norms.into_iter().enumerate() {
                if weight.len() != cfg.d as usize || bias.len() != cfg.d as usize {
                    return Err("encoder norm does not match the hidden size".into());
                }
                norms.push((
                    gpu.upload(
                        &format!("trit-norm-w-{index}"),
                        bytemuck::cast_slice(&weight),
                    )?,
                    gpu.upload(&format!("trit-norm-b-{index}"), bytemuck::cast_slice(&bias))?,
                ));
            }
            layers.push(LayerW {
                norms,
                mats: upload.mats,
                rel_k: upload.rel_k,
                bias_u: gpu.upload("trit-bias-u", bytemuck::cast_slice(&upload.bias_u))?,
                bias_v: gpu.upload("trit-bias-v", bytemuck::cast_slice(&upload.bias_v))?,
                dw: gpu.upload("trit-dw", bytemuck::cast_slice(&upload.dw))?,
                bn_scale: gpu.upload("trit-bn-scale", bytemuck::cast_slice(&upload.bn_scale))?,
                bn_shift: gpu.upload("trit-bn-shift", bytemuck::cast_slice(&upload.bn_shift))?,
                rel: gpu.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("trit-rel-pending"),
                    size: 4,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            });
        }
        let slots = S_BASE + layers.len() * 9;
        let params = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("trit-encoder-params"),
            size: (slots * STRIDE) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let limit = gpu.device().limits().max_compute_workgroups_per_dimension;
        let mut inner = Inner {
            device: gpu.device().clone(),
            queue: gpu.queue().clone(),
            instance: gpu.instance().clone(),
            dummy: gpu.dummy().clone(),
            params,
            params_cpu: vec![0u8; slots * STRIDE],
            pipes,
            cfg,
            layers,
            act: Act::empty(gpu.device()),
            bound: None,
            reach: 0,
            limit,
            profile: gpu.timestamps(),
        };
        // 512 frames is about 41 s. The takes in the speed check fit.
        inner.ensure_cap(512)?;
        inner.project(reach, sine)?;
        inner.rebuild_binds()?;
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    /// Run the conformer. `sine(reach)` is the sinusoid table for that reach.
    pub(crate) fn encode(
        &self,
        x: &[f32],
        frames: usize,
        sine: impl Fn(usize) -> Vec<f32>,
    ) -> Result<Buf, String> {
        let mut inner = self.inner.lock();
        let t = u32::try_from(frames).map_err(|_| format!("encoder has {frames} frames"))?;
        if t == 0 {
            return Ok(Buf::scratch(0));
        }
        let d = inner.cfg.d as usize;
        let cells = frames
            .checked_mul(d)
            .ok_or_else(|| format!("encoder input {frames}x{d} overflows"))?;
        if x.len() < cells {
            return Err(format!(
                "encoder input has {} values, needs {cells}",
                x.len()
            ));
        }
        if frames > inner.reach as usize + 1 {
            let reach = (frames - 1)
                .max(inner.cfg.min_reach as usize)
                .max((inner.reach as usize).saturating_mul(2));
            let reach = u32::try_from(reach).map_err(|_| format!("position reach {reach}"))?;
            let table = sine(reach as usize);
            inner.project(reach, &table)?;
            inner.rebuild_binds()?;
        }
        inner.ensure_cap(t)?;
        inner.rebuild_binds()?;
        inner.run(&x[..cells], t)
    }
}

impl Act {
    fn empty(device: &wgpu::Device) -> Self {
        let tiny = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trit-act-empty"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trit-encoder-staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            cap: 0,
            a: tiny.clone(),
            b: tiny.clone(),
            branch: tiny.clone(),
            hidden: tiny.clone(),
            qkv: tiny.clone(),
            q: tiny.clone(),
            k: tiny.clone(),
            v: tiny.clone(),
            scores: tiny.clone(),
            ctx: tiny.clone(),
            wide: tiny.clone(),
            gated: tiny,
            sine: staging.clone(),
            projected: staging.clone(),
            table_rows: 0,
            staging,
            staging_bytes: 4,
        }
    }
}

impl Inner {
    fn alloc(&self, label: &str, elems: u32) -> Result<wgpu::Buffer, String> {
        let bytes = u64::from(elems) * 4;
        let limits = self.device.limits();
        if bytes == 0
            || bytes > limits.max_buffer_size
            || bytes > limits.max_storage_buffer_binding_size
        {
            return Err(format!(
                "{label} is {bytes} bytes, above the GPU buffer limit"
            ));
        }
        Ok(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    }

    fn ensure_cap(&mut self, t: u32) -> Result<(), String> {
        if t <= self.act.cap {
            return Ok(());
        }
        let mut cap = self.act.cap.max(64);
        while cap < t {
            cap = cap.saturating_mul(2);
        }
        let (d, f, h) = (self.cfg.d, self.cfg.f, self.cfg.heads);
        let act = Act {
            cap,
            a: self.alloc("trit-x", mul(cap, d)?)?,
            b: self.alloc("trit-norm", mul(cap, d)?)?,
            branch: self.alloc("trit-branch", mul(cap, d)?)?,
            hidden: self.alloc("trit-hidden", mul(cap, f)?)?,
            qkv: self.alloc("trit-qkv", mul(cap, mul(3, d)?)?)?,
            q: self.alloc("trit-q", mul(cap, d)?)?,
            k: self.alloc("trit-k", mul(cap, d)?)?,
            v: self.alloc("trit-v", mul(cap, d)?)?,
            scores: self.alloc("trit-scores", mul(h, mul(cap, cap)?)?)?,
            ctx: self.alloc("trit-ctx", mul(cap, d)?)?,
            wide: self.alloc("trit-wide", mul(cap, mul(2, d)?)?)?,
            gated: self.alloc("trit-gated", mul(cap, d)?)?,
            sine: self.act.sine.clone(),
            projected: self.act.projected.clone(),
            table_rows: self.act.table_rows,
            staging: self.act.staging.clone(),
            staging_bytes: self.act.staging_bytes,
        };
        self.act = act;
        self.bound = None;
        Ok(())
    }

    fn ensure_table(&mut self, rows: u32) -> Result<(), String> {
        if rows <= self.act.table_rows {
            return Ok(());
        }
        let elems = mul(rows, self.cfg.d)?;
        let sine = self.alloc("trit-sine", elems)?;
        let projected = self.alloc("trit-proj", elems)?;
        let mut rels = Vec::with_capacity(self.layers.len());
        for _ in 0..self.layers.len() {
            rels.push(self.alloc("trit-rel", elems)?);
        }
        self.act.sine = sine;
        self.act.projected = projected;
        for (layer, rel) in self.layers.iter_mut().zip(rels) {
            layer.rel = rel;
        }
        self.act.table_rows = rows;
        self.bound = None;
        Ok(())
    }

    fn ensure_staging(&mut self, bytes: u64) -> Result<(), String> {
        if self.act.staging_bytes >= bytes {
            return Ok(());
        }
        let cap = bytes.next_power_of_two().max(256);
        self.act.staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trit-encoder-staging"),
            size: cap,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.act.staging_bytes = cap;
        Ok(())
    }

    /// Project the sinusoid through `relative_k` and split the heads.
    fn project(&mut self, reach: u32, sine: &[f32]) -> Result<(), String> {
        let rows = reach
            .checked_mul(2)
            .and_then(|rows| rows.checked_add(1))
            .ok_or_else(|| format!("position reach {reach} overflows"))?;
        let elems = mul(rows, self.cfg.d)? as usize;
        if sine.len() < elems {
            return Err(format!(
                "position table has {} values, needs {elems}",
                sine.len()
            ));
        }
        self.ensure_table(rows)?;
        self.queue
            .write_buffer(&self.act.sine, 0, bytemuck::cast_slice(&sine[..elems]));
        let hd = self.cfg.d / self.cfg.heads;
        put(
            &mut self.params_cpu,
            S_HEADS,
            &Quad {
                a: rows,
                b: self.cfg.heads,
                c: hd,
                d: 0,
            },
        );
        for (index, layer) in self.layers.iter().enumerate() {
            put(
                &mut self.params_cpu,
                mat_slot(index, MATS),
                &Quad {
                    a: rows,
                    b: layer.rel_k.n,
                    c: layer.rel_k.k,
                    d: layer.rel_k.flags,
                },
            );
        }
        self.queue.write_buffer(&self.params, 0, &self.params_cpu);
        let scopes = super::push_scopes(&self.device);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("trit-positions"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("trit-positions"),
                timestamp_writes: None,
            });
            let count = self.layers.len();
            for index in 0..count {
                let slot = mat_slot(index, MATS);
                let group = bind_product(
                    &ProductGpu {
                        device: &self.device,
                        dummy: &self.dummy,
                        params: &self.params,
                        pipes: &self.pipes,
                    },
                    &self.layers[index].rel_k,
                    &self.act.sine,
                    &self.act.projected,
                    slot,
                );
                dispatch_product(
                    &mut pass,
                    &self.pipes,
                    &self.layers[index].rel_k,
                    &group,
                    rows,
                );
                let split = make_group(
                    &self.device,
                    &self.pipes.heads_layout,
                    "trit-split-heads",
                    &[
                        storage(0, &self.act.projected),
                        storage(1, &self.layers[index].rel),
                        uniform(2, &self.params, S_HEADS),
                    ],
                );
                pass.set_pipeline(&self.pipes.split_heads);
                pass.set_bind_group(0, &split, &[]);
                let n = mul(rows, self.cfg.d)?;
                pass.dispatch_workgroups(n.div_ceil(LANE), 1, 1);
            }
        }
        self.queue.submit([encoder.finish()]);
        // The queue runs this before the next encode. Poll so a bad shader fails here.
        self.instance.poll_all(true);
        super::pop_scopes(scopes)?;
        self.reach = reach;
        self.bound = None;
        Ok(())
    }

    fn fill_params(&mut self, t: u32) {
        let d = self.cfg.d;
        let f = self.cfg.f;
        let heads = self.cfg.heads;
        let hd = d / heads;
        let buf = &mut self.params_cpu;
        put(
            buf,
            S_LN,
            &Quad {
                a: t,
                b: d,
                c: 0,
                d: 0,
            },
        );
        put(
            buf,
            S_SILU_F,
            &Quad {
                a: t.saturating_mul(f),
                b: 0,
                c: 0,
                d: 0,
            },
        );
        put(
            buf,
            S_SILU_D,
            &Quad {
                a: t.saturating_mul(d),
                b: 0,
                c: 0,
                d: 0,
            },
        );
        put(
            buf,
            S_HALF,
            &ScaleParams {
                n: t.saturating_mul(d),
                scale: 0.5,
                pad0: 0,
                pad1: 0,
            },
        );
        put(
            buf,
            S_ONE,
            &ScaleParams {
                n: t.saturating_mul(d),
                scale: 1.0,
                pad0: 0,
                pad1: 0,
            },
        );
        put(
            buf,
            S_SPLIT,
            &Quad {
                a: t,
                b: d,
                c: 0,
                d: 0,
            },
        );
        put(
            buf,
            S_REL,
            &RelParams {
                t,
                hd,
                heads,
                reach: self.reach,
                scale: 1.0 / (hd as f32).sqrt(),
                pad0: 0,
                pad1: 0,
                pad2: 0,
            },
        );
        put(
            buf,
            S_SOFTMAX,
            &Quad {
                a: heads.saturating_mul(t),
                b: t,
                c: 0,
                d: 0,
            },
        );
        put(
            buf,
            S_ATTN,
            &Quad {
                a: t,
                b: hd,
                c: heads,
                d: 0,
            },
        );
        put(
            buf,
            S_GLU,
            &Quad {
                a: t,
                b: d,
                c: 0,
                d: 0,
            },
        );
        put(
            buf,
            S_DW,
            &Quad {
                a: t,
                b: d,
                c: self.cfg.kernel,
                d: 0,
            },
        );
        put(
            buf,
            S_SCALE,
            &ScaleParams {
                n: t.saturating_mul(d),
                scale: self.cfg.input_scale,
                pad0: 0,
                pad1: 0,
            },
        );
        for (index, layer) in self.layers.iter().enumerate() {
            for (which, mat) in layer.mats.iter().enumerate() {
                put(
                    buf,
                    mat_slot(index, which),
                    &Quad {
                        a: t,
                        b: mat.n,
                        c: mat.k,
                        d: mat.flags,
                    },
                );
            }
        }
    }

    fn rebuild_binds(&mut self) -> Result<(), String> {
        if self
            .bound
            .as_ref()
            .is_some_and(|bound| bound.cap == self.act.cap && bound.reach == self.reach)
        {
            return Ok(());
        }
        let mut layers = Vec::with_capacity(self.layers.len());
        for index in 0..self.layers.len() {
            layers.push(self.bind_layer(index));
        }
        let device = &self.device;
        let params = &self.params;
        let pipes = &self.pipes;
        let act = &self.act;
        let scale_input = self.cfg.input_scale != 1.0;
        let cap = act.cap;
        let reach = self.reach;
        let bound = Bound {
            silu_f: make_group(
                device,
                &pipes.silu_layout,
                "trit-silu-f",
                &[storage(0, &act.hidden), uniform(1, params, S_SILU_F)],
            ),
            silu_d: make_group(
                device,
                &pipes.silu_layout,
                "trit-silu-d",
                &[storage(0, &act.branch), uniform(1, params, S_SILU_D)],
            ),
            split: make_group(
                device,
                &pipes.split_layout,
                "trit-split-qkv",
                &[
                    storage(0, &act.qkv),
                    storage(1, &act.q),
                    storage(2, &act.k),
                    storage(3, &act.v),
                    uniform(4, params, S_SPLIT),
                ],
            ),
            softmax: make_group(
                device,
                &pipes.softmax_layout,
                "trit-softmax",
                &[storage(0, &act.scores), uniform(1, params, S_SOFTMAX)],
            ),
            attn: make_group(
                device,
                &pipes.attn_layout,
                "trit-attn",
                &[
                    storage(0, &act.scores),
                    storage(1, &act.v),
                    storage(2, &act.ctx),
                    uniform(3, params, S_ATTN),
                ],
            ),
            glu: make_group(
                device,
                &pipes.glu_layout,
                "trit-glu",
                &[
                    storage(0, &act.wide),
                    storage(1, &act.gated),
                    uniform(2, params, S_GLU),
                ],
            ),
            scale: scale_input.then(|| {
                make_group(
                    device,
                    &pipes.scale_layout,
                    "trit-scale",
                    &[storage(0, &act.a), uniform(1, params, S_SCALE)],
                )
            }),
            layers,
            cap,
            reach,
        };
        self.bound = Some(bound);
        Ok(())
    }

    fn bind_layer(&self, index: usize) -> LayerBind {
        let device = &self.device;
        let params = &self.params;
        let pipes = &self.pipes;
        let act = &self.act;
        let layer = &self.layers[index];
        let (stream, norm) = if index.is_multiple_of(2) {
            (&act.a, &act.b)
        } else {
            (&act.b, &act.a)
        };
        let mut lns = Vec::with_capacity(5);
        for (weight, bias) in &layer.norms {
            lns.push(make_group(
                device,
                &pipes.ln_layout,
                "trit-ln",
                &[
                    storage(0, stream),
                    storage(1, weight),
                    storage(2, bias),
                    storage(3, norm),
                    uniform(4, params, S_LN),
                ],
            ));
        }
        let add = |y: &wgpu::Buffer, slot: usize| {
            make_group(
                device,
                &pipes.residual_layout,
                "trit-res",
                &[storage(0, stream), storage(1, y), uniform(2, params, slot)],
            )
        };
        let pairs = [
            (norm, &act.hidden),
            (&act.hidden, &act.branch),
            (norm, &act.qkv),
            (&act.ctx, &act.branch),
            (norm, &act.wide),
            (&act.branch, norm),
            (norm, &act.hidden),
            (&act.hidden, &act.branch),
        ];
        let mut gemms = Vec::with_capacity(MATS);
        for (which, (x, y)) in pairs.into_iter().enumerate() {
            let mat = &layer.mats[which];
            gemms.push(GemmBind {
                pipe: pipe_id(mat.kind),
                group: bind_product(
                    &ProductGpu {
                        device,
                        dummy: &self.dummy,
                        params,
                        pipes,
                    },
                    mat,
                    x,
                    y,
                    mat_slot(index, which),
                ),
                n: mat.n,
                tile: tile_of(pipes, mat.kind),
            });
        }
        LayerBind {
            lns,
            res_half: add(&act.branch, S_HALF),
            res_one: add(&act.branch, S_ONE),
            res_conv: add(norm, S_ONE),
            gemms,
            dw: make_group(
                device,
                &pipes.dw_layout,
                "trit-dw",
                &[
                    storage(0, &act.gated),
                    storage(1, &layer.dw),
                    storage(2, &layer.bn_scale),
                    storage(3, &layer.bn_shift),
                    storage(4, &act.branch),
                    uniform(5, params, S_DW),
                ],
            ),
            rel: make_group(
                device,
                &pipes.rel_layout,
                "trit-rel-scores",
                &[
                    storage(0, &act.q),
                    storage(1, &act.k),
                    storage(2, &layer.rel),
                    storage(3, &layer.bias_u),
                    storage(4, &layer.bias_v),
                    storage(5, &act.scores),
                    uniform(6, params, S_REL),
                ],
            ),
        }
    }

    fn run(&mut self, x: &[f32], t: u32) -> Result<Buf, String> {
        self.check_limits(t)?;
        self.fill_params(t);
        self.queue.write_buffer(&self.params, 0, &self.params_cpu);
        self.queue
            .write_buffer(&self.act.a, 0, bytemuck::cast_slice(x));
        let bytes = u64::from(t) * u64::from(self.cfg.d) * 4;
        self.ensure_staging(bytes)?;
        let scopes = super::push_scopes(&self.device);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("trit-encoder"),
            });
        let queries = self.profile.then(|| {
            self.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("trit-encoder-ts"),
                ty: wgpu::QueryType::Timestamp,
                count: STAMPS,
            })
        });
        let mut marks = Marks {
            queries: queries.as_ref(),
            slot: 0,
            names: Vec::new(),
        };
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("trit-encoder"),
                timestamp_writes: None,
            });
            self.record(&mut pass, t, &mut marks)?;
        }
        let stamp_bytes = u64::from(marks.slot) * 8;
        let (resolved, stamp_read) = if stamp_bytes > 0 {
            let resolved = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("trit-encoder-ts-resolve"),
                size: stamp_bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let stamp_read = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("trit-encoder-ts-read"),
                size: stamp_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            if let Some(queries) = &queries {
                encoder.resolve_query_set(queries, 0..marks.slot, &resolved, 0);
            }
            encoder.copy_buffer_to_buffer(&resolved, 0, &stamp_read, 0, stamp_bytes);
            (Some(resolved), Some(stamp_read))
        } else {
            (None, None)
        };
        let out = self.output().clone();
        encoder.copy_buffer_to_buffer(&out, 0, &self.act.staging, 0, bytes);
        self.queue.submit([encoder.finish()]);
        let values = readback(&self.instance, &self.act.staging, bytes);
        let stamped = stamp_read
            .as_ref()
            .map(|buffer| read_stamps(&self.instance, buffer, stamp_bytes));
        super::pop_scopes(scopes)?;
        if let Some(stamped) = stamped {
            match stamped {
                Ok(ticks) => report_times(self.queue.get_timestamp_period(), &marks.names, &ticks),
                Err(error) => eprintln!("trit: gpu profile read failed ({error})"),
            }
        }
        drop(resolved);
        let values = values?;
        if values.len() != x.len() {
            return Err(format!(
                "encoder read {} values, expected {}",
                values.len(),
                x.len()
            ));
        }
        let mut out = Buf::scratch(values.len());
        out.copy_from_slice(&values);
        Ok(out)
    }

    fn output(&self) -> &wgpu::Buffer {
        if self.layers.len().is_multiple_of(2) {
            &self.act.a
        } else {
            &self.act.b
        }
    }

    fn check_limits(&self, t: u32) -> Result<(), String> {
        let limit = self.limit;
        let wide = self.cfg.f.max(self.cfg.d.saturating_mul(3));
        let groups = [
            t,
            self.cfg.heads.saturating_mul(t),
            t.div_ceil(SMALL),
            wide.div_ceil(SMALL),
            (t.saturating_mul(self.cfg.f)).div_ceil(LANE),
        ];
        if groups.iter().any(|group| *group > limit) {
            return Err(format!(
                "encoder dispatch for {t} frames exceeds the GPU workgroup limit"
            ));
        }
        Ok(())
    }

    fn record(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        t: u32,
        marks: &mut Marks<'_>,
    ) -> Result<(), String> {
        let pipes = &self.pipes;
        let d = self.cfg.d;
        let f = self.cfg.f;
        let heads = self.cfg.heads;
        let bound = self
            .bound
            .as_ref()
            .ok_or_else(|| "encoder bindings are not ready".to_string())?;
        if let Some(scale) = &bound.scale {
            marks.at(pass, "scale");
            dispatch_elem(pass, &pipes.scale, scale, t.saturating_mul(d));
        }
        for layer in &bound.layers {
            marks.at(pass, "ln");
            dispatch_ln(pass, &pipes.ln, &layer.lns[0], t);
            marks.at(pass, "gemm");
            dispatch_gemm(pass, pipes, &layer.gemms[0], t);
            marks.at(pass, "silu");
            dispatch_elem(pass, &pipes.silu, &bound.silu_f, t.saturating_mul(f));
            marks.at(pass, "gemm");
            dispatch_gemm(pass, pipes, &layer.gemms[1], t);
            marks.at(pass, "res");
            dispatch_elem(pass, &pipes.residual, &layer.res_half, t.saturating_mul(d));
            marks.at(pass, "ln");
            dispatch_ln(pass, &pipes.ln, &layer.lns[1], t);
            marks.at(pass, "gemm");
            dispatch_gemm(pass, pipes, &layer.gemms[2], t);
            marks.at(pass, "split");
            dispatch_elem(pass, &pipes.split_qkv, &bound.split, t.saturating_mul(d));
            marks.at(pass, "rel");
            pass.set_pipeline(&pipes.rel);
            pass.set_bind_group(0, &layer.rel, &[]);
            pass.dispatch_workgroups(t.div_ceil(64), t.div_ceil(16), heads);
            marks.at(pass, "softmax");
            pass.set_pipeline(&pipes.softmax);
            pass.set_bind_group(0, &bound.softmax, &[]);
            pass.dispatch_workgroups(heads.saturating_mul(t), 1, 1);
            marks.at(pass, "attn");
            dispatch_elem(pass, &pipes.attn, &bound.attn, t.saturating_mul(d));
            marks.at(pass, "gemm");
            dispatch_gemm(pass, pipes, &layer.gemms[3], t);
            marks.at(pass, "res");
            dispatch_elem(pass, &pipes.residual, &layer.res_one, t.saturating_mul(d));
            marks.at(pass, "ln");
            dispatch_ln(pass, &pipes.ln, &layer.lns[2], t);
            marks.at(pass, "gemm");
            dispatch_gemm(pass, pipes, &layer.gemms[4], t);
            marks.at(pass, "glu");
            dispatch_elem(pass, &pipes.glu, &bound.glu, t.saturating_mul(d));
            marks.at(pass, "dw");
            dispatch_elem(pass, &pipes.dw, &layer.dw, t.saturating_mul(d));
            marks.at(pass, "silu");
            dispatch_elem(pass, &pipes.silu, &bound.silu_d, t.saturating_mul(d));
            marks.at(pass, "gemm");
            dispatch_gemm(pass, pipes, &layer.gemms[5], t);
            // Pointwise 2 wrote the norm buffer. Add that, not the depthwise row.
            marks.at(pass, "res");
            dispatch_elem(pass, &pipes.residual, &layer.res_conv, t.saturating_mul(d));
            marks.at(pass, "ln");
            dispatch_ln(pass, &pipes.ln, &layer.lns[3], t);
            marks.at(pass, "gemm");
            dispatch_gemm(pass, pipes, &layer.gemms[6], t);
            marks.at(pass, "silu");
            dispatch_elem(pass, &pipes.silu, &bound.silu_f, t.saturating_mul(f));
            marks.at(pass, "gemm");
            dispatch_gemm(pass, pipes, &layer.gemms[7], t);
            marks.at(pass, "res");
            dispatch_elem(pass, &pipes.residual, &layer.res_half, t.saturating_mul(d));
            marks.at(pass, "ln");
            dispatch_ln(pass, &pipes.ln, &layer.lns[4], t);
        }
        marks.at(pass, "end");
        Ok(())
    }
}

/// One compute-pass timestamp per dispatch, used only when profiling.
struct Marks<'a> {
    queries: Option<&'a wgpu::QuerySet>,
    slot: u32,
    names: Vec<&'static str>,
}

impl Marks<'_> {
    fn at(&mut self, pass: &mut wgpu::ComputePass<'_>, name: &'static str) {
        let Some(queries) = self.queries else {
            return;
        };
        if self.slot >= STAMPS {
            return;
        }
        pass.write_timestamp(queries, self.slot);
        self.slot += 1;
        self.names.push(name);
    }
}

fn make_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    label: &str,
    entries: &[wgpu::BindGroupEntry<'_>],
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries,
    })
}

struct ProductGpu<'a> {
    device: &'a wgpu::Device,
    dummy: &'a wgpu::Buffer,
    params: &'a wgpu::Buffer,
    pipes: &'a Pipes,
}

fn bind_product(
    gpu: &ProductGpu<'_>,
    mat: &Mat,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    slot: usize,
) -> wgpu::BindGroup {
    let bias = mat.bias.as_ref().unwrap_or(gpu.dummy);
    match mat.kind {
        MatKind::Ternary => {
            let scales = mat.scales.as_ref().unwrap_or(gpu.dummy);
            make_group(
                gpu.device,
                &gpu.pipes.ternary_layout,
                "trit-ternary",
                &[
                    storage(0, x),
                    storage(1, &mat.weight),
                    storage(2, scales),
                    storage(3, y),
                    storage(4, bias),
                    uniform(5, gpu.params, slot),
                ],
            )
        }
        MatKind::F32 | MatKind::F16 => make_group(
            gpu.device,
            &gpu.pipes.gemm_layout,
            "trit-gemm",
            &[
                storage(0, x),
                storage(1, &mat.weight),
                storage(2, y),
                storage(3, bias),
                uniform(4, gpu.params, slot),
            ],
        ),
    }
}

fn tile_of(pipes: &Pipes, kind: MatKind) -> u32 {
    match kind {
        MatKind::F32 => pipes.f32_tile,
        MatKind::F16 => SMALL,
        MatKind::Ternary => pipes.ternary_tile,
    }
}

fn dispatch_elem(
    pass: &mut wgpu::ComputePass<'_>,
    pipe: &wgpu::ComputePipeline,
    group: &wgpu::BindGroup,
    n: u32,
) {
    pass.set_pipeline(pipe);
    pass.set_bind_group(0, group, &[]);
    pass.dispatch_workgroups(n.div_ceil(LANE), 1, 1);
}

fn dispatch_ln(
    pass: &mut wgpu::ComputePass<'_>,
    pipe: &wgpu::ComputePipeline,
    group: &wgpu::BindGroup,
    t: u32,
) {
    pass.set_pipeline(pipe);
    pass.set_bind_group(0, group, &[]);
    pass.dispatch_workgroups(t, 1, 1);
}

fn dispatch_gemm(pass: &mut wgpu::ComputePass<'_>, pipes: &Pipes, gemm: &GemmBind, t: u32) {
    let (pipe, tile) = match gemm.pipe {
        0 => (&pipes.gemm_f32, gemm.tile),
        1 => (&pipes.gemm_f16, SMALL),
        _ => (&pipes.ternary, gemm.tile),
    };
    pass.set_pipeline(pipe);
    pass.set_bind_group(0, &gemm.group, &[]);
    pass.dispatch_workgroups(gemm.n.div_ceil(tile), t.div_ceil(tile), 1);
}

fn dispatch_product(
    pass: &mut wgpu::ComputePass<'_>,
    pipes: &Pipes,
    mat: &Mat,
    group: &wgpu::BindGroup,
    rows: u32,
) {
    let pipe = match mat.kind {
        MatKind::F32 => &pipes.gemm_f32,
        MatKind::F16 => &pipes.gemm_f16,
        MatKind::Ternary => &pipes.ternary,
    };
    let tile = tile_of(pipes, mat.kind);
    pass.set_pipeline(pipe);
    pass.set_bind_group(0, group, &[]);
    pass.dispatch_workgroups(mat.n.div_ceil(tile), rows.div_ceil(tile), 1);
}

fn pipe_id(kind: MatKind) -> u8 {
    match kind {
        MatKind::F32 => 0,
        MatKind::F16 => 1,
        MatKind::Ternary => 2,
    }
}

fn mat_slot(layer: usize, which: usize) -> usize {
    S_BASE + layer * 9 + which
}

fn mul(a: u32, b: u32) -> Result<u32, String> {
    a.checked_mul(b)
        .ok_or_else(|| format!("encoder shape {a} by {b} overflows"))
}

fn put<T: bytemuck::Pod>(buf: &mut [u8], slot: usize, value: &T) {
    let start = slot * STRIDE;
    let bytes = bytemuck::bytes_of(value);
    buf[start..start + bytes.len()].copy_from_slice(bytes);
}

fn storage<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn uniform<'a>(binding: u32, buffer: &'a wgpu::Buffer, slot: usize) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset: (slot * STRIDE) as u64,
            size: NonZeroU64::new(STRIDE as u64),
        }),
    }
}

fn build_pipes(gpu: &Gpu) -> Result<Pipes, String> {
    let device = gpu.device();
    let ln_layout = layout(
        device,
        &[
            SlotKind::Read,
            SlotKind::Read,
            SlotKind::Read,
            SlotKind::Write,
            SlotKind::Uniform,
        ],
        "trit-ln",
    );
    let residual_layout = layout(
        device,
        &[SlotKind::Write, SlotKind::Read, SlotKind::Uniform],
        "trit-res",
    );
    let scale_layout = layout(device, &[SlotKind::Write, SlotKind::Uniform], "trit-scale");
    let silu_layout = layout(device, &[SlotKind::Write, SlotKind::Uniform], "trit-silu");
    let glu_layout = layout(
        device,
        &[SlotKind::Read, SlotKind::Write, SlotKind::Uniform],
        "trit-glu",
    );
    let dw_layout = layout(
        device,
        &[
            SlotKind::Read,
            SlotKind::Read,
            SlotKind::Read,
            SlotKind::Read,
            SlotKind::Write,
            SlotKind::Uniform,
        ],
        "trit-dw",
    );
    let softmax_layout = layout(
        device,
        &[SlotKind::Write, SlotKind::Uniform],
        "trit-softmax",
    );
    let rel_layout = layout(
        device,
        &[
            SlotKind::Read,
            SlotKind::Read,
            SlotKind::Read,
            SlotKind::Read,
            SlotKind::Read,
            SlotKind::Write,
            SlotKind::Uniform,
        ],
        "trit-rel",
    );
    let attn_layout = layout(
        device,
        &[
            SlotKind::Read,
            SlotKind::Read,
            SlotKind::Write,
            SlotKind::Uniform,
        ],
        "trit-attn",
    );
    let split_layout = layout(
        device,
        &[
            SlotKind::Read,
            SlotKind::Write,
            SlotKind::Write,
            SlotKind::Write,
            SlotKind::Uniform,
        ],
        "trit-split",
    );
    let heads_layout = layout(
        device,
        &[SlotKind::Read, SlotKind::Write, SlotKind::Uniform],
        "trit-heads",
    );
    let shader = |layout: &wgpu::BindGroupLayout, source: &str, entry: &str, label: &str| {
        pipeline(device, layout, source, entry, label)
    };
    let gemm_f32_small = gpu.gemm_f32_pipeline().clone();
    let ternary_small = gpu.ternary_pipeline().clone();
    let (gemm_f32, f32_tile) = match shader(
        gpu.gemm_layout(),
        include_str!("tiled_f32.wgsl"),
        "gemm",
        "tiled-f32",
    ) {
        Ok(pipe) => (pipe, TILE),
        Err(error) => {
            eprintln!("trit: tiled f32 shader failed ({error}). The 16-wide product is used.");
            (gemm_f32_small.clone(), SMALL)
        }
    };
    let (ternary, ternary_tile) = match shader(
        gpu.ternary_layout(),
        include_str!("tiled_ternary.wgsl"),
        "gemm",
        "tiled-ternary",
    ) {
        Ok(pipe) => (pipe, TILE),
        Err(error) => {
            eprintln!("trit: tiled ternary shader failed ({error}). The 16-wide product is used.");
            (ternary_small.clone(), SMALL)
        }
    };
    Ok(Pipes {
        ln: shader(
            &ln_layout,
            include_str!("shaders/encoder/layer_norm.wgsl"),
            "layer_norm",
            "ln",
        )?,
        residual: shader(
            &residual_layout,
            include_str!("shaders/encoder/residual_scale.wgsl"),
            "residual_scale",
            "res",
        )?,
        scale: shader(
            &scale_layout,
            include_str!("shaders/encoder/scale.wgsl"),
            "apply_scale",
            "scale",
        )?,
        silu: shader(
            &silu_layout,
            include_str!("shaders/encoder/silu.wgsl"),
            "silu",
            "silu",
        )?,
        glu: shader(
            &glu_layout,
            include_str!("shaders/encoder/glu.wgsl"),
            "glu",
            "glu",
        )?,
        dw: shader(
            &dw_layout,
            include_str!("shaders/encoder/depthwise_conv1d_bn.wgsl"),
            "depthwise_conv1d_bn",
            "dw",
        )?,
        softmax: shader(
            &softmax_layout,
            include_str!("shaders/encoder/softmax_rows.wgsl"),
            "softmax_rows",
            "softmax",
        )?,
        rel: shader(
            &rel_layout,
            include_str!("shaders/encoder/rel_pos_scores.wgsl"),
            "rel_pos_scores",
            "rel",
        )?,
        attn: shader(
            &attn_layout,
            include_str!("shaders/encoder/attn_weighted_sum.wgsl"),
            "attn_weighted_sum",
            "attn",
        )?,
        split_qkv: shader(
            &split_layout,
            include_str!("shaders/encoder/split_qkv.wgsl"),
            "split_qkv",
            "split-qkv",
        )?,
        split_heads: shader(
            &heads_layout,
            include_str!("shaders/encoder/split_heads.wgsl"),
            "split_heads",
            "split-heads",
        )?,
        gemm_f32,
        gemm_f16: gpu.gemm_f16_pipeline().clone(),
        ternary,
        f32_tile,
        ternary_tile,
        ln_layout,
        residual_layout,
        scale_layout,
        silu_layout,
        glu_layout,
        dw_layout,
        softmax_layout,
        rel_layout,
        attn_layout,
        split_layout,
        heads_layout,
        gemm_layout: gpu.gemm_layout().clone(),
        ternary_layout: gpu.ternary_layout().clone(),
    })
}

fn layout(device: &wgpu::Device, kinds: &[SlotKind], label: &str) -> wgpu::BindGroupLayout {
    let entries: Vec<wgpu::BindGroupLayoutEntry> = kinds
        .iter()
        .enumerate()
        .map(|(index, kind)| {
            let ty = match kind {
                SlotKind::Uniform => wgpu::BufferBindingType::Uniform,
                SlotKind::Read => wgpu::BufferBindingType::Storage { read_only: true },
                SlotKind::Write => wgpu::BufferBindingType::Storage { read_only: false },
            };
            wgpu::BindGroupLayoutEntry {
                binding: index as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

fn read_stamps(
    instance: &wgpu::Instance,
    staging: &wgpu::Buffer,
    bytes: u64,
) -> Result<Vec<u64>, String> {
    let slice = staging.slice(..bytes);
    let (send, recv) = std::sync::mpsc::sync_channel(1);
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = send.send(result);
    });
    let _ = instance;
    super::wait_for_device("gpu timestamp read")?;
    recv.try_recv()
        .map_err(|_| "gpu timestamp read did not complete".to_string())?
        .map_err(|error| format!("gpu timestamp read: {error}"))?;
    let ticks = match slice.get_mapped_range() {
        Ok(view) => {
            let mut ticks = Vec::with_capacity(view.len() / 8);
            for chunk in view.as_chunks::<8>().0 {
                ticks.push(u64::from_le_bytes(*chunk));
            }
            drop(view);
            ticks
        }
        Err(error) => {
            staging.unmap();
            return Err(format!("gpu timestamp read: {error}"));
        }
    };
    staging.unmap();
    Ok(ticks)
}

fn report_times(period_ns: f32, names: &[&str], ticks: &[u64]) {
    if names.len() != ticks.len() || ticks.len() < 2 {
        eprintln!("trit: gpu profile has {} stamps", ticks.len());
        return;
    }
    let mut totals: Vec<(&str, f64)> = Vec::new();
    for index in 0..ticks.len() - 1 {
        let ms =
            ticks[index + 1].saturating_sub(ticks[index]) as f64 * f64::from(period_ns) / 1.0e6;
        if let Some(slot) = totals.iter_mut().find(|item| item.0 == names[index]) {
            slot.1 += ms;
        } else {
            totals.push((names[index], ms));
        }
    }
    totals.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sum: f64 = totals.iter().map(|item| item.1).sum();
    eprintln!("trit: gpu profile {sum:.1} ms");
    for (name, ms) in totals {
        eprintln!("trit: gpu profile {name} {ms:.1} ms");
    }
}
