//! Kokoro's decoder and its iSTFTNet generator. The decoder joins the text
//! frames, the pitch and the energy, and doubles the frame rate. The
//! generator makes a harmonic source from the pitch (nine sines, as
//! `SineGen` does), takes its STFT, upsamples the decoder output twice,
//! adds the source at each stage, and turns the last layer into a
//! magnitude and a phase for the inverse STFT.

use super::blocks::{AdaInResBlock, SnakeBlock};
use super::Noise;
use crate::audio::stft::Stft;
use crate::cpu::{activation, buf::Buf, team, tensors::Tensors};
use crate::models::config::Json;
use crate::models::parts::conv::{Conv1d, Upsample};

/// The source: the harmonics of F0 (the fundamental and 8 overtones).
const HARMONICS: usize = 9;
const SINE_AMP: f32 = 0.1;
const NOISE_STD: f32 = 0.003;
/// F0 above this (Hz) is voiced.
const VOICED: f32 = 10.0;
/// The slope of the leaky ReLU before each upsampling.
const SLOPE: f32 = 0.1;
/// The slope of the leaky ReLU before the last convolution (PyTorch's default).
const FINAL_SLOPE: f32 = 0.01;

pub struct Decoder {
    f0_conv: Conv1d,
    n_conv: Conv1d,
    encode: AdaInResBlock,
    decode: Vec<AdaInResBlock>,
    asr_res: Conv1d,
    generator: Generator,
    hidden: usize,
}

impl Decoder {
    pub fn load(t: &Tensors, c: &Json, (hidden, style): (usize, usize)) -> Result<Self, String> {
        let wide = 1024;
        let residual = 64;
        let joined = wide + residual + 2;
        let block = |name: &str, (i, o): (usize, usize), up: bool| {
            AdaInResBlock::load(t, name, (i, o, style), up)
        };
        Ok(Self {
            f0_conv: Conv1d::load_strided(t, "decoder.F0_conv", (1, 1), (3, 2, 1))?,
            n_conv: Conv1d::load_strided(t, "decoder.N_conv", (1, 1), (3, 2, 1))?,
            encode: block("decoder.encode", (hidden + 2, wide), false)?,
            decode: vec![
                block("decoder.decode.0", (joined, wide), false)?,
                block("decoder.decode.1", (joined, wide), false)?,
                block("decoder.decode.2", (joined, wide), false)?,
                block("decoder.decode.3", (joined, hidden), true)?,
            ],
            asr_res: Conv1d::load(t, "decoder.asr_res.0", (hidden, residual), (1, 1))?,
            generator: Generator::load(t, c, style)?,
            hidden,
        })
    }

    /// The waveform of `asr` (`frames x hidden`), with the pitch and energy
    /// curves of `2 * frames` values.
    pub fn forward(
        &self,
        asr: &[f32],
        frames: usize,
        (f0, energy): (&[f32], &[f32]),
        style: &[f32],
        noise: f32,
        rng: &mut Noise,
    ) -> Vec<f32> {
        let f0_down = self.f0_conv.forward(f0, 2 * frames, None);
        let n_down = self.n_conv.forward(energy, 2 * frames, None);
        let joined = |x: &[f32], width: usize, extra: &[&[f32]]| -> Buf {
            let wide = width + extra.iter().map(|e| e.len() / frames).sum::<usize>();
            let mut out = Buf::scratch(frames * wide);
            for (r, row) in out.chunks_exact_mut(wide).enumerate() {
                row[..width].copy_from_slice(&x[r * width..(r + 1) * width]);
                let mut at = width;
                for part in extra {
                    let w = part.len() / frames;
                    row[at..at + w].copy_from_slice(&part[r * w..(r + 1) * w]);
                    at += w;
                }
            }
            out
        };
        let x = joined(asr, self.hidden, &[&f0_down, &n_down]);
        let (mut x, _) = self.encode.forward(&x, frames, style);
        let residual = self.asr_res.forward(asr, frames, None);
        let mut length = frames;
        let mut add = true;
        for block in &self.decode {
            if add {
                x = joined(
                    &x,
                    block_inputs(&x, frames),
                    &[&residual, &f0_down, &n_down],
                );
            }
            (x, length) = block.forward(&x, length, style);
            if length != frames {
                add = false;
            }
        }
        self.generator.forward(&x, length, f0, style, noise, rng)
    }
}

/// The width of `x` with `frames` rows.
fn block_inputs(x: &[f32], frames: usize) -> usize {
    x.len() / frames
}

struct Stage {
    up: Upsample,
    noise_conv: Conv1d,
    noise_res: SnakeBlock,
    blocks: Vec<SnakeBlock>,
}

struct Generator {
    stages: Vec<Stage>,
    post: Conv1d,
    /// `l_linear` of the source: 9 weights and a bias.
    merge: (Vec<f32>, f32),
    stft: Stft,
    /// Samples per F0 value: the upsampling rates times the STFT hop.
    scale: usize,
    sample_rate: f32,
}

impl Generator {
    fn load(t: &Tensors, c: &Json, style: usize) -> Result<Self, String> {
        let list = |key: &str| -> Result<Vec<usize>, String> {
            c.get(key)
                .as_array()
                .and_then(|v| v.iter().map(|x| x.as_u64().map(|x| x as usize)).collect())
                .ok_or_else(|| format!("config.json has no {key}"))
        };
        let rates = list("istftnet.upsample_rates")?;
        let kernels = list("istftnet.upsample_kernel_sizes")?;
        let block_kernels = list("istftnet.resblock_kernel_sizes")?;
        let dilations: Vec<Vec<usize>> = c
            .get("istftnet.resblock_dilation_sizes")
            .as_array()
            .and_then(|rows| {
                rows.iter()
                    .map(|r| {
                        r.as_array()?
                            .iter()
                            .map(|x| x.as_u64().map(|x| x as usize))
                            .collect()
                    })
                    .collect()
            })
            .ok_or("config.json has no istftnet.resblock_dilation_sizes")?;
        let initial = c.num("istftnet.upsample_initial_channel")?;
        let n_fft = c.num("istftnet.gen_istft_n_fft")?;
        let hop = c.num("istftnet.gen_istft_hop_size")?;
        let source = n_fft + 2;
        let name = "decoder.generator";
        let mut stages = Vec::new();
        for (i, (rate, kernel)) in rates.iter().zip(&kernels).enumerate() {
            let (inputs, outputs) = (initial >> i, initial >> (i + 1));
            let (noise_conv, noise_res) = if i + 1 < rates.len() {
                let stride: usize = rates[i + 1..].iter().product();
                (
                    Conv1d::load_strided(
                        t,
                        &format!("{name}.noise_convs.{i}"),
                        (source, outputs),
                        (2 * stride, stride, stride.div_ceil(2)),
                    )?,
                    SnakeBlock::load(
                        t,
                        &format!("{name}.noise_res.{i}"),
                        (outputs, style),
                        (7, &[1, 3, 5]),
                    )?,
                )
            } else {
                (
                    Conv1d::load(
                        t,
                        &format!("{name}.noise_convs.{i}"),
                        (source, outputs),
                        (1, 1),
                    )?,
                    SnakeBlock::load(
                        t,
                        &format!("{name}.noise_res.{i}"),
                        (outputs, style),
                        (11, &[1, 3, 5]),
                    )?,
                )
            };
            let blocks = block_kernels
                .iter()
                .zip(&dilations)
                .enumerate()
                .map(|(j, (k, d))| {
                    SnakeBlock::load(
                        t,
                        &format!("{name}.resblocks.{}", i * block_kernels.len() + j),
                        (outputs, style),
                        (*k, d),
                    )
                })
                .collect::<Result<Vec<_>, String>>()?;
            stages.push(Stage {
                up: Upsample::load(
                    t,
                    &format!("{name}.ups.{i}"),
                    (inputs, outputs),
                    (*kernel, *rate),
                )?,
                noise_conv,
                noise_res,
                blocks,
            });
        }
        let last = initial >> rates.len();
        let weights = t.f32_len(&format!("{name}.m_source.l_linear.weight"), HARMONICS)?;
        let bias = t.f32_len(&format!("{name}.m_source.l_linear.bias"), 1)?[0];
        Ok(Self {
            stages,
            post: Conv1d::load(t, &format!("{name}.conv_post"), (last, source), (7, 1))?,
            merge: (weights, bias),
            stft: Stft::new(n_fft, hop),
            scale: rates.iter().product::<usize>() * hop,
            sample_rate: 24_000.0,
        })
    }

    /// The harmonic source of `f0` (one value per `scale` samples), as
    /// `SourceModuleHnNSF` makes it: nine sines whose phase is the summed
    /// F0 at the frame rate, interpolated to the sample rate, then mixed by
    /// `l_linear` and tanh. The random start phase of `SineGen` does not
    /// reach the output (the frame-rate interpolation skips sample 0); the
    /// noise does, scaled by `noise`.
    fn source(&self, f0: &[f32], noise: f32, rng: &mut Noise) -> Vec<f32> {
        let scale = self.scale;
        let samples = f0.len() * scale;
        // PyTorch's interpolation scales: 1 / scale_factor, in f64, then as f32.
        let step = (1.0f64 / scale as f64) as f32;
        let ratio = (1.0f64 / (1.0f64 / scale as f64)) as f32;
        // F0 per sample: nearest neighbor, with PyTorch's float index.
        let at = |i: usize| -> f32 { f0[((i as f32 * step).floor() as usize).min(f0.len() - 1)] };
        // The per-sample phase increment of each harmonic, (f / rate) mod 1,
        // averaged over the two middle samples of each frame (linear
        // interpolation down by `scale`).
        let increment = |i: usize, h: usize| -> f32 {
            let x = at(i) * (h + 1) as f32 / self.sample_rate;
            x - x.floor()
        };
        let frames = f0.len();
        let mut phase = vec![0.0f32; frames * HARMONICS];
        let mut sum = [0.0f64; HARMONICS];
        for d in 0..frames {
            let real = ratio * (d as f32 + 0.5) - 0.5;
            let i0 = real.floor() as usize;
            let lambda = real - i0 as f32;
            let i1 = (i0 + 1).min(samples - 1);
            for (h, s) in sum.iter_mut().enumerate() {
                let value = (1.0 - lambda) * increment(i0, h) + lambda * increment(i1, h);
                *s += f64::from(value);
                phase[d * HARMONICS + h] =
                    ((*s as f32) * 2.0 * std::f32::consts::PI) * scale as f32;
            }
        }
        // Back to the sample rate (linear interpolation up by `scale`), sines,
        // voiced or noise, mixed.
        let (weights, bias) = (&self.merge.0, self.merge.1);
        let mut out = vec![0.0f32; samples];
        let mut draws = vec![0.0f32; samples * HARMONICS];
        if noise > 0.0 {
            draws.iter_mut().for_each(|v| *v = rng.normal() * noise);
        }
        let draws = &draws;
        team::chunks_mut(&mut out, 4096, |chunk, block| {
            for (j, o) in block.iter_mut().enumerate() {
                let i = chunk * 4096 + j;
                let real = (step * (i as f32 + 0.5) - 0.5).max(0.0);
                let i0 = real as usize;
                let i1 = if i0 < frames - 1 { i0 + 1 } else { i0 };
                let l1 = real - i0 as f32;
                let l0 = 1.0 - l1;
                let voiced = at(i) > VOICED;
                let amp = if voiced { NOISE_STD } else { SINE_AMP / 3.0 };
                let mut mixed = bias;
                for h in 0..HARMONICS {
                    let p = l0 * phase[i0 * HARMONICS + h] + l1 * phase[i1 * HARMONICS + h];
                    let sine = p.sin() * SINE_AMP;
                    let value = if voiced { sine } else { 0.0 } + amp * draws[i * HARMONICS + h];
                    mixed += weights[h] * value;
                }
                *o = mixed;
            }
        });
        activation::tanh(&mut out);
        out
    }

    fn forward(
        &self,
        x: &[f32],
        frames: usize,
        f0: &[f32],
        style: &[f32],
        noise: f32,
        rng: &mut Noise,
    ) -> Vec<f32> {
        let source = self.source(f0, noise, rng);
        let spectrum = self.stft.forward(&source);
        let spectrum_frames = self.stft.frames(source.len());
        let mut x = {
            let mut copy = Buf::scratch(x.len());
            copy.copy_from_slice(x);
            copy
        };
        let mut length = frames;
        let last = self.stages.len() - 1;
        for (i, stage) in self.stages.iter().enumerate() {
            let harmonic = stage.noise_conv.forward(&spectrum, spectrum_frames, None);
            let harmonic_frames = stage.noise_conv.out_frames(spectrum_frames);
            let harmonic = stage.noise_res.forward(&harmonic, harmonic_frames, style);
            let mut up = stage.up.forward(&x, length, SLOPE);
            length *= stage.up.stride;
            let channels = stage.up.outputs;
            if i == last {
                // Reflection padding of one frame at the start.
                let mut padded = Buf::scratch((length + 1) * channels);
                padded[..channels].copy_from_slice(&up[channels..2 * channels]);
                padded[channels..].copy_from_slice(&up);
                up = padded;
                length += 1;
            }
            debug_assert_eq!(length, harmonic_frames);
            team::chunks_mut(&mut up, 8192, |index, part| {
                for (v, h) in part.iter_mut().zip(&harmonic[index * 8192..]) {
                    *v += h;
                }
            });
            let mut sum = Buf::zeroed(up.len());
            for block in &stage.blocks {
                let y = block.forward(&up, length, style);
                sum.iter_mut().zip(y.iter()).for_each(|(s, v)| *s += v);
            }
            let inv = 1.0 / stage.blocks.len() as f32;
            sum.iter_mut().for_each(|v| *v *= inv);
            x = sum;
        }
        let post = self.post.forward(&x, length, Some(FINAL_SLOPE));
        let bins = self.stft.bins();
        self.stft.inverse(length, |t, k| {
            let row = &post[t * 2 * bins..(t + 1) * 2 * bins];
            (row[k].exp(), row[bins + k].sin())
        })
    }
}
