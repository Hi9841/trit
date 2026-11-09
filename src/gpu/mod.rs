//! Matrix products on a GPU, through wgpu (DX12, Vulkan, or Metal).
//! Weights are uploaded once at load. A single product uploads the
//! activation, runs one tiled product, and reads the result back.
//! The Parakeet encoder keeps the whole conformer on the device.
//!
//! `Gpu::get` keeps one device for the process. `TRIT_DEVICE=gpu` fails
//! the load when the machine has no adapter. `auto` uses the CPU then.
//! A device error does not panic. The caller runs that call on the CPU.

use crate::cpu::buf::Buf;
use parking_lot::Mutex;
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, OnceLock};

pub(crate) mod encoder;

const TILE: u32 = 16;
const FLAG_BIAS: u32 = 1;
const FLAG_RELU: u32 = 2;
/// Kept activation buffers of one size. A forward pass uses two.
const POOL_PER_SIZE: usize = 4;
/// Bytes of activation buffers the pool keeps. Above it the pool starts again.
const POOL_BYTES: u64 = 256 << 20;
/// Bytes of queued weight writes that wgpu may hold as staging buffers.
/// Every `write_buffer` makes a staging buffer, and wgpu keeps all of them
/// until the next submit: a load without a flush holds the whole model twice.
const STAGED_FLUSH: u64 = 32 << 20;

/// A dense product with fewer rows stays on the CPU: a decoder step has 1
/// to 16 rows, and the trip to the GPU costs more than the CPU product.
pub(crate) const MIN_ROWS: usize = 48;

static GPU: OnceLock<Result<Gpu, String>> = OnceLock::new();
static LOGGED_FALLBACK: AtomicBool = AtomicBool::new(false);
/// GPU calls that failed one after the other.
static FAILURES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// After this many failures in sequence the GPU stays off: each more try
/// would cost its time and then the time of the CPU run.
const MAX_FAILURES: u32 = 3;
/// A readback takes milliseconds. A longer wait means that the driver hangs,
/// and the call must go to the CPU.
const READBACK_LIMIT: std::time::Duration = std::time::Duration::from_secs(10);

thread_local! {
    static HOST_PRODUCTS: Cell<bool> = const { Cell::new(false) };
}

pub(crate) fn use_host_products(host: bool) {
    HOST_PRODUCTS.with(|flag| flag.set(host));
}

pub(crate) fn host_products() -> bool {
    HOST_PRODUCTS.with(|flag| flag.get())
}

/// Whether calls still try the GPU.
pub(crate) fn usable() -> bool {
    FAILURES.load(Ordering::Relaxed) < MAX_FAILURES
}

pub(crate) fn failed(error: &str) {
    log_fallback(error);
    if FAILURES.fetch_add(1, Ordering::Relaxed) + 1 == MAX_FAILURES {
        crate::log(&format!(
            "the GPU failed {MAX_FAILURES} times. All later calls use the CPU."
        ));
    }
}

pub(crate) fn worked() {
    if FAILURES.load(Ordering::Relaxed) != 0 && usable() {
        FAILURES.store(0, Ordering::Relaxed);
    }
}

/// Waits for the last submission and its callbacks, at most `READBACK_LIMIT`.
pub(crate) fn wait_for_device(what: &str) -> Result<(), String> {
    Gpu::get()?
        .device()
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(READBACK_LIMIT),
        })
        .map(drop)
        .map_err(|error| format!("{what}: {error}"))
}

/// Log the first GPU failure. Later failures stay quiet.
pub(crate) fn log_fallback(error: &str) {
    if !LOGGED_FALLBACK.swap(true, Ordering::Relaxed) {
        crate::log(&format!("GPU failed ({error}). This call uses the CPU."));
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    m: u32,
    n: u32,
    k: u32,
    flags: u32,
}

/// One process-wide device, queue, and the product pipelines.
pub struct Gpu {
    instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
    path: String,
    shader_f16: bool,
    /// `TRIT_GPU_PROFILE` asked for per-dispatch timestamps, and the device has them.
    timestamps: bool,
    gemm_layout: wgpu::BindGroupLayout,
    ternary_layout: wgpu::BindGroupLayout,
    gemm_f32: wgpu::ComputePipeline,
    gemm_f16: wgpu::ComputePipeline,
    ternary: wgpu::ComputePipeline,
    params: wgpu::Buffer,
    /// Bound when a layer has no bias. The shader does not read it.
    dummy: wgpu::Buffer,
    gate: Mutex<Pool>,
    weight_bytes: AtomicU64,
    /// Bytes written since the last flush of the queued writes.
    staged: AtomicU64,
}

struct Pool {
    storage: HashMap<u64, Vec<wgpu::Buffer>>,
    /// Bytes of the buffers in `storage`.
    held: u64,
    staging: Option<wgpu::Buffer>,
    staging_cap: u64,
}

/// Weights of one dense or ternary layer, resident on the device.
pub struct Layer {
    kind: Kind,
    n: u32,
    k: u32,
    bias: Option<wgpu::Buffer>,
}

enum Kind {
    F32(wgpu::Buffer),
    /// f16 bits, two per word, low address first.
    F16(wgpu::Buffer),
    Ternary {
        codes: wgpu::Buffer,
        scales: wgpu::Buffer,
    },
}

impl Gpu {
    pub fn get() -> Result<&'static Gpu, String> {
        match GPU.get_or_init(Self::open) {
            Ok(gpu) => Ok(gpu),
            Err(error) => Err(error.clone()),
        }
    }

    /// Adapter name, backend, and whether `SHADER_F16` is present.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// The adapter accepts `enable f16` in WGSL.
    pub fn shader_f16(&self) -> bool {
        self.shader_f16
    }

    pub(crate) fn timestamps(&self) -> bool {
        self.timestamps
    }

    fn open() -> Result<Self, String> {
        let wanted = wgpu::Backends::DX12 | wgpu::Backends::VULKAN | wgpu::Backends::METAL;
        let backends = wgpu::Backends::from_env().unwrap_or(wanted)
            & wgpu::Instance::enabled_backend_features();
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = backends;
        let instance = wgpu::Instance::new(descriptor);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .map_err(|error| format!("no GPU adapter ({error})"))?;
        let info = adapter.get_info();
        let shader_f16 = adapter.features().contains(wgpu::Features::SHADER_F16);
        let mut required = wgpu::Features::empty();
        if shader_f16 {
            required |= wgpu::Features::SHADER_F16;
        }
        let timestamps = profile_requested()
            && adapter.features().contains(
                wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
            );
        if timestamps {
            required |=
                wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
            crate::log(&format!(
                "gpu profile on, shared memory {} bytes",
                adapter.limits().max_compute_workgroup_storage_size
            ));
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("trit"),
            required_features: required,
            required_limits: adapter.limits(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|error| format!("GPU device: {error}"))?;
        device.on_uncaptured_error(Arc::new(|error| {
            log_fallback(&error.to_string());
        }));

        let gemm_layout = bind_layout(&device, 5, "trit-gemm");
        let ternary_layout = bind_layout(&device, 6, "trit-ternary");
        let gemm_f32 = pipeline(
            &device,
            &gemm_layout,
            include_str!("gemm_f32.wgsl"),
            "gemm",
            "gemm-f32",
        )?;
        let f16_source = if shader_f16 {
            include_str!("gemm_f16.wgsl")
        } else {
            include_str!("gemm_f16_unpack.wgsl")
        };
        let gemm_f16 = match pipeline(&device, &gemm_layout, f16_source, "gemm", "gemm-f16") {
            Ok(pipeline) => pipeline,
            Err(error) if shader_f16 => {
                // The feature bit is set, but this shader failed. The unpack
                // path uses the same f16 bit layout and needs no feature.
                let _ = error;
                pipeline(
                    &device,
                    &gemm_layout,
                    include_str!("gemm_f16_unpack.wgsl"),
                    "gemm",
                    "gemm-f16-unpack",
                )?
            }
            Err(error) => return Err(error),
        };
        let ternary = pipeline(
            &device,
            &ternary_layout,
            include_str!("ternary.wgsl"),
            "gemm",
            "ternary",
        )?;
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trit-params"),
            size: 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trit-dummy-bias"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&dummy, 0, &[0u8; 16]);
        let float = if shader_f16 { "f16" } else { "f32" };
        Ok(Self {
            instance,
            device,
            queue,
            path: format!("{}, {}, {float}", info.name, backend_name(info.backend)),
            shader_f16,
            timestamps,
            gemm_layout,
            ternary_layout,
            gemm_f32,
            gemm_f16,
            ternary,
            params,
            dummy,
            gate: Mutex::new(Pool {
                storage: HashMap::new(),
                held: 0,
                staging: None,
                staging_cap: 0,
            }),
            weight_bytes: AtomicU64::new(0),
            staged: AtomicU64::new(0),
        })
    }

    pub(crate) fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub(crate) fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub(crate) fn instance(&self) -> &wgpu::Instance {
        &self.instance
    }

    pub(crate) fn dummy(&self) -> &wgpu::Buffer {
        &self.dummy
    }

    pub(crate) fn gemm_layout(&self) -> &wgpu::BindGroupLayout {
        &self.gemm_layout
    }

    pub(crate) fn ternary_layout(&self) -> &wgpu::BindGroupLayout {
        &self.ternary_layout
    }

    pub(crate) fn gemm_f16_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.gemm_f16
    }

    pub(crate) fn gemm_f32_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.gemm_f32
    }

    pub(crate) fn ternary_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.ternary
    }

    pub(crate) fn upload(&self, label: &str, bytes: &[u8]) -> Result<wgpu::Buffer, String> {
        if bytes.is_empty() {
            return Err(format!("{label} is empty"));
        }
        let padded;
        let data = if bytes.len().is_multiple_of(4) {
            bytes
        } else {
            let mut copy = bytes.to_vec();
            copy.resize(copy.len().next_multiple_of(4), 0);
            padded = copy;
            &padded
        };
        let size = u64::try_from(data.len()).map_err(|_| format!("{label} is too large"))?;
        let limits = self.device.limits();
        if size > limits.max_buffer_size || size > limits.max_storage_buffer_binding_size {
            return Err(format!(
                "{label} is {size} bytes, above the GPU buffer limit"
            ));
        }
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, data);
        self.weight_bytes.fetch_add(size, Ordering::Relaxed);
        if self.staged.fetch_add(size, Ordering::Relaxed) + size >= STAGED_FLUSH {
            self.flush_writes();
        }
        Ok(buffer)
    }

    fn flush_writes(&self) {
        self.staged.store(0, Ordering::Relaxed);
        self.queue.submit([]);
        self.instance.poll_all(true);
    }

    /// Frees the pooled activation and staging buffers, then lets the device
    /// finish the destruction of every buffer that was dropped.
    fn trim(&self) {
        {
            let mut pool = self.gate.lock();
            pool.storage.clear();
            pool.held = 0;
            pool.staging = None;
            pool.staging_cap = 0;
        }
        self.flush_writes();
    }

    fn bias_buffer(&self, bias: Option<&[f32]>) -> Result<Option<wgpu::Buffer>, String> {
        match bias {
            Some(bias) if !bias.is_empty() => {
                Ok(Some(self.upload("trit-bias", bytemuck::cast_slice(bias))?))
            }
            _ => Ok(None),
        }
    }

    /// `x` is `rows x k`. The result is `rows x n`.
    fn forward(&self, layer: &Layer, x: &[f32], rows: usize, relu: bool) -> Result<Buf, String> {
        let m = u32::try_from(rows).map_err(|_| format!("gemm has {rows} rows"))?;
        let (n, k) = (layer.n, layer.k);
        if m == 0 || n == 0 {
            return Ok(Buf::scratch(0));
        }
        let cells = rows
            .checked_mul(k as usize)
            .ok_or_else(|| format!("gemm input {rows}x{k} overflows"))?;
        if x.len() < cells {
            return Err(format!("gemm input has {} values, needs {cells}", x.len()));
        }
        let x_bytes = (cells * 4) as u64;
        let y_cells = rows
            .checked_mul(n as usize)
            .ok_or_else(|| format!("gemm output {rows}x{n} overflows"))?;
        let y_bytes = (y_cells * 4) as u64;
        let limit = self.device.limits().max_compute_workgroups_per_dimension;
        let groups_n = n.div_ceil(TILE);
        let groups_m = m.div_ceil(TILE);
        if groups_n > limit || groups_m > limit {
            return Err(format!("gemm {m}x{n} exceeds the GPU workgroup limit"));
        }

        let mut pool = self.gate.lock();
        let x_buf = pool.take(&self.device, x_bytes);
        let y_buf = pool.take(&self.device, y_bytes);
        let staging = pool.staging(&self.device, y_bytes);
        self.queue
            .write_buffer(&x_buf, 0, bytemuck::cast_slice(&x[..cells]));
        let flags = u32::from(layer.bias.is_some()) * FLAG_BIAS + u32::from(relu) * FLAG_RELU;
        self.queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&Params { m, n, k, flags }),
        );
        let bias = layer.bias.as_ref().unwrap_or(&self.dummy);
        let (pipeline, bind) = match &layer.kind {
            Kind::F32(weight) => (
                &self.gemm_f32,
                bind_group(
                    &self.device,
                    &self.gemm_layout,
                    "trit-gemm",
                    &[&x_buf, weight, &y_buf, bias, &self.params],
                ),
            ),
            Kind::F16(weight) => (
                &self.gemm_f16,
                bind_group(
                    &self.device,
                    &self.gemm_layout,
                    "trit-gemm",
                    &[&x_buf, weight, &y_buf, bias, &self.params],
                ),
            ),
            Kind::Ternary { codes, scales } => (
                &self.ternary,
                bind_group(
                    &self.device,
                    &self.ternary_layout,
                    "trit-ternary",
                    &[&x_buf, codes, scales, &y_buf, bias, &self.params],
                ),
            ),
        };
        let scopes = push_scopes(&self.device);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("trit-gemm"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("trit-gemm"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups_n, groups_m, 1);
        }
        encoder.copy_buffer_to_buffer(&y_buf, 0, &staging, 0, y_bytes);
        self.queue.submit([encoder.finish()]);
        let values = readback(&self.instance, &staging, y_bytes);
        let scope_error = pop_scopes(scopes);
        if values.is_err() {
            // The map of this buffer can still be pending, and a second map
            // fails: one timeout then failed every later call, and three
            // failures turn the GPU off. The next call makes a new buffer.
            pool.staging = None;
            pool.staging_cap = 0;
        }
        pool.give(x_bytes, x_buf);
        pool.give(y_bytes, y_buf);
        drop(pool);
        scope_error?;
        let values = values?;

        let mut out = Buf::scratch(y_cells);
        out.copy_from_slice(&values);
        Ok(out)
    }
}

impl Pool {
    fn take(&mut self, device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        let size = size.max(4);
        if let Some(buffer) = self.storage.get_mut(&size).and_then(Vec::pop) {
            self.held -= size;
            return buffer;
        }
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trit-activation"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn give(&mut self, size: u64, buffer: wgpu::Buffer) {
        let size = size.max(4);
        // Each take length gives buffers of its own sizes. Without a limit a
        // program that never calls `release_memory` kept them all.
        if self.held + size > POOL_BYTES {
            self.storage.clear();
            self.held = 0;
        }
        let slot = self.storage.entry(size).or_default();
        if slot.len() < POOL_PER_SIZE && size <= POOL_BYTES {
            slot.push(buffer);
            self.held += size;
        }
    }

    fn staging(&mut self, device: &wgpu::Device, need: u64) -> wgpu::Buffer {
        if self.staging_cap < need {
            let cap = need.next_power_of_two().max(256);
            self.staging = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("trit-staging"),
                size: cap,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.staging_cap = cap;
        }
        self.staging.as_ref().unwrap().clone()
    }
}

impl Layer {
    /// f32 weights, row-major `[n][k]`.
    pub fn f32(n: usize, k: usize, weight: &[f32], bias: Option<&[f32]>) -> Result<Self, String> {
        let gpu = Gpu::get()?;
        check_len("f32", n, k, weight.len())?;
        Ok(Self {
            kind: Kind::F32(gpu.upload("trit-f32", bytemuck::cast_slice(weight))?),
            n: dim(n)?,
            k: dim(k)?,
            bias: gpu.bias_buffer(bias)?,
        })
    }

    /// f16 weights as bits, row-major `[n][k]`.
    pub fn f16_bits(
        n: usize,
        k: usize,
        bits: &[u16],
        bias: Option<&[f32]>,
    ) -> Result<Self, String> {
        let gpu = Gpu::get()?;
        check_len("f16", n, k, bits.len())?;
        // `upload` pads an odd count of halves to a whole word.
        Ok(Self {
            kind: Kind::F16(gpu.upload("trit-f16", bytemuck::cast_slice(bits))?),
            n: dim(n)?,
            k: dim(k)?,
            bias: gpu.bias_buffer(bias)?,
        })
    }

    /// Ternary codes packed 16 per `u32`, and f32 scales `[n][k / 128]`.
    pub fn ternary(
        n: usize,
        k: usize,
        codes: &[u32],
        scales: &[f32],
        bias: Option<&[f32]>,
    ) -> Result<Self, String> {
        let gpu = Gpu::get()?;
        let words = k / 16;
        let groups = k / 128;
        if !k.is_multiple_of(128) || codes.len() != n * words || scales.len() != n * groups {
            return Err(format!("ternary gpu layer {n}x{k} does not match its pack"));
        }
        Ok(Self {
            kind: Kind::Ternary {
                codes: gpu.upload("trit-codes", bytemuck::cast_slice(codes))?,
                scales: gpu.upload("trit-scales", bytemuck::cast_slice(scales))?,
            },
            n: dim(n)?,
            k: dim(k)?,
            bias: gpu.bias_buffer(bias)?,
        })
    }

    pub fn forward(&self, x: &[f32], rows: usize, relu: bool) -> Result<Buf, String> {
        Gpu::get()?.forward(self, x, rows, relu)
    }

    pub(crate) fn rows(&self) -> u32 {
        self.n
    }

    pub(crate) fn cols(&self) -> u32 {
        self.k
    }

    pub(crate) fn bias(&self) -> Option<&wgpu::Buffer> {
        self.bias.as_ref()
    }

    pub(crate) fn weights(&self) -> Weights<'_> {
        match &self.kind {
            Kind::F32(weight) => Weights::F32(weight),
            Kind::F16(weight) => Weights::F16(weight),
            Kind::Ternary { codes, scales } => Weights::Ternary { codes, scales },
        }
    }
}

/// Borrowed weight buffers of one [`Layer`].
pub(crate) enum Weights<'a> {
    F32(&'a wgpu::Buffer),
    F16(&'a wgpu::Buffer),
    Ternary {
        codes: &'a wgpu::Buffer,
        scales: &'a wgpu::Buffer,
    },
}

/// Does nothing before the first GPU load.
pub fn trim() {
    if let Some(Ok(gpu)) = GPU.get() {
        gpu.trim();
    }
}

pub fn weight_bytes() -> u64 {
    GPU.get()
        .and_then(|gpu| gpu.as_ref().ok())
        .map(|gpu| gpu.weight_bytes.load(Ordering::Relaxed))
        .unwrap_or(0)
}

struct Scopes {
    internal: wgpu::ErrorScopeGuard,
    memory: wgpu::ErrorScopeGuard,
    validation: wgpu::ErrorScopeGuard,
}

fn push_scopes(device: &wgpu::Device) -> Scopes {
    // Push order is the reverse of pop order.
    Scopes {
        internal: device.push_error_scope(wgpu::ErrorFilter::Internal),
        memory: device.push_error_scope(wgpu::ErrorFilter::OutOfMemory),
        validation: device.push_error_scope(wgpu::ErrorFilter::Validation),
    }
}

fn pop_scopes(scopes: Scopes) -> Result<(), String> {
    let mut message = None;
    for error in [
        pollster::block_on(scopes.validation.pop()),
        pollster::block_on(scopes.memory.pop()),
        pollster::block_on(scopes.internal.pop()),
    ]
    .into_iter()
    .flatten()
    {
        message.get_or_insert(error.to_string());
    }
    match message {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

fn dim(value: usize) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("layer dimension {value} does not fit in u32"))
}

fn check_len(kind: &str, n: usize, k: usize, len: usize) -> Result<(), String> {
    if n.checked_mul(k) == Some(len) {
        Ok(())
    } else {
        Err(format!("{kind} gpu layer {n}x{k} has {len} weights"))
    }
}

fn profile_requested() -> bool {
    matches!(std::env::var("TRIT_GPU_PROFILE").as_deref(), Ok("1"))
}

fn backend_name(backend: wgpu::Backend) -> &'static str {
    match backend {
        wgpu::Backend::Dx12 => "Dx12",
        wgpu::Backend::Vulkan => "Vulkan",
        wgpu::Backend::Metal => "Metal",
        wgpu::Backend::Gl => "Gl",
        wgpu::Backend::BrowserWebGpu => "WebGpu",
        _ => "Gpu",
    }
}

fn bind_layout(device: &wgpu::Device, count: u32, label: &str) -> wgpu::BindGroupLayout {
    let mut entries = Vec::with_capacity(count as usize);
    for binding in 0..count {
        let uniform = binding == count - 1;
        entries.push(wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if uniform {
                    wgpu::BufferBindingType::Uniform
                } else if binding == count - 3 {
                    // The output binding: gemm uses 2, ternary uses 3.
                    wgpu::BufferBindingType::Storage { read_only: false }
                } else {
                    wgpu::BufferBindingType::Storage { read_only: true }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

pub(crate) fn pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    source: &str,
    entry: &str,
    label: &str,
) -> Result<wgpu::ComputePipeline, String> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let source = shader_source(source);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    });
    match pollster::block_on(scope.pop()) {
        Some(error) => Err(format!("{label} shader: {error}")),
        None => Ok(pipeline),
    }
}

/// `TILE`, `FLAG_BIAS`, and `FLAG_RELU` for every product shader.
/// An `enable` line stays first: WGSL rejects a declaration before it.
fn shader_source(body: &str) -> String {
    let prelude = format!(
        "const TILE: u32 = {TILE}u;\nconst FLAG_BIAS: u32 = {FLAG_BIAS}u;\nconst FLAG_RELU: u32 = {FLAG_RELU}u;\n"
    );
    if let Some(rest) = body.strip_prefix("enable ") {
        let end = rest
            .find('\n')
            .map(|index| index + "enable ".len() + 1)
            .unwrap_or(body.len());
        let (head, tail) = body.split_at(end);
        format!("{head}{prelude}{tail}")
    } else {
        format!("{prelude}{body}")
    }
}

fn bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    label: &str,
    buffers: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry<'_>> = buffers
        .iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: u32::try_from(binding).expect("bind group"),
            resource: buffer.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &entries,
    })
}

struct Unmap<'a>(&'a wgpu::Buffer);

impl Drop for Unmap<'_> {
    fn drop(&mut self) {
        self.0.unmap();
    }
}

pub(crate) fn readback(
    instance: &wgpu::Instance,
    staging: &wgpu::Buffer,
    bytes: u64,
) -> Result<Vec<f32>, String> {
    let slice = staging.slice(..bytes);
    let (send, recv) = mpsc::sync_channel(1);
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = send.send(result);
    });
    let _ = instance;
    wait_for_device("gpu readback")?;
    recv.try_recv()
        .map_err(|_| "gpu readback did not complete".to_string())?
        .map_err(|error| format!("gpu readback: {error}"))?;
    let _unmap = Unmap(staging);
    let view = slice
        .get_mapped_range()
        .map_err(|error| format!("gpu readback: {error}"))?;
    let values = bytemuck::try_cast_slice(&view)
        .map_err(|error| format!("gpu readback: {error}"))?
        .to_vec();
    drop(view);
    Ok(values)
}
