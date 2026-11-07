//! Where the products run, and the one place that knows if a GPU exists.

/// Where the dense and ternary products run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Device {
    /// Every product runs on the CPU.
    #[default]
    Cpu,
    /// Dense and ternary products run on the GPU. The load fails when the
    /// machine has no adapter, or when Trit was built without `gpu`.
    Gpu,
    /// The GPU when it is available, otherwise the CPU.
    Auto,
}

impl Device {
    pub const NAMES: [&str; 3] = ["cpu", "gpu", "auto"];

    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            "auto" => Ok(Self::Auto),
            other => Err(format!(
                "device `{other}`: use one of {}",
                Self::NAMES.join(", ")
            )),
        }
    }

    pub fn name(self) -> &'static str {
        Self::NAMES[self as usize]
    }
}

/// `TRIT_DEVICE=cpu|gpu|auto` overrides `requested`. `Gpu` fails when there
/// is no adapter. `Auto` is then the CPU. The result is `Cpu` or `Gpu`.
pub(crate) fn active_device(requested: Device) -> Result<Device, String> {
    let requested = match std::env::var("TRIT_DEVICE") {
        Ok(name) if !name.is_empty() => Device::parse(&name.to_ascii_lowercase())?,
        _ => requested,
    };
    match requested {
        Device::Cpu => Ok(Device::Cpu),
        Device::Gpu => gpu_adapter().map(|_| Device::Gpu),
        Device::Auto => Ok(if gpu_adapter().is_ok() {
            Device::Gpu
        } else {
            Device::Cpu
        }),
    }
}

/// The GPU adapter, for example `NVIDIA GeForce RTX 4090, Dx12, f16`, or the
/// reason there is none.
#[cfg(feature = "gpu")]
pub fn gpu_adapter() -> Result<String, String> {
    crate::gpu::Gpu::get().map(|gpu| gpu.path().to_string())
}

/// The GPU adapter, or the reason there is none.
#[cfg(not(feature = "gpu"))]
pub fn gpu_adapter() -> Result<String, String> {
    Err("trit was built without the gpu feature".into())
}

/// `gpu_adapter` as text: the adapter or the reason.
pub fn gpu_path() -> String {
    gpu_adapter().unwrap_or_else(|reason| reason)
}

/// Bytes of weight buffers resident on the GPU. 0 before a GPU load.
pub fn gpu_weight_bytes() -> u64 {
    #[cfg(feature = "gpu")]
    {
        crate::gpu::weight_bytes()
    }
    #[cfg(not(feature = "gpu"))]
    {
        0
    }
}
