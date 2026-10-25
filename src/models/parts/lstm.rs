//! A bidirectional LSTM layer (PyTorch `nn.LSTM(bidirectional=True)`, one
//! layer, gates in the order input, forget, cell, output). The input of all
//! steps is projected in one product; the steps of each direction then run
//! in order, the two directions at the same time.

use crate::cpu::{activation, buf::Buf, fgemm, gemm::Dense, team, tensors::Tensors};

/// One direction: the input projection (with both biases) and the
/// recurrent weights, transposed to `[hidden][4 * hidden]`.
struct Direction {
    input: Dense,
    recurrent: Vec<f32>,
}

pub struct BiLstm {
    forward: Direction,
    backward: Direction,
    pub hidden: usize,
}

impl BiLstm {
    /// `name.weight_ih_l0`, `name.weight_hh_l0`, `name.bias_ih_l0`,
    /// `name.bias_hh_l0`, and the same with `_reverse`.
    pub fn load(t: &Tensors, name: &str, inputs: usize, hidden: usize) -> Result<Self, String> {
        let direction = |suffix: &str| -> Result<Direction, String> {
            let gates = 4 * hidden;
            let w_ih = t.f32_len(&format!("{name}.weight_ih_l0{suffix}"), gates * inputs)?;
            let w_hh = t.f32_len(&format!("{name}.weight_hh_l0{suffix}"), gates * hidden)?;
            let b_ih = t.f32_len(&format!("{name}.bias_ih_l0{suffix}"), gates)?;
            let b_hh = t.f32_len(&format!("{name}.bias_hh_l0{suffix}"), gates)?;
            let bias = b_ih.iter().zip(&b_hh).map(|(a, b)| a + b).collect();
            let mut recurrent = vec![0.0f32; hidden * gates];
            for g in 0..gates {
                for h in 0..hidden {
                    recurrent[h * gates + g] = w_hh[g * hidden + h];
                }
            }
            Ok(Direction {
                input: Dense::new(w_ih, Some(bias), gates, inputs)?,
                recurrent,
            })
        };
        Ok(Self {
            forward: direction("")?,
            backward: direction("_reverse")?,
            hidden,
        })
    }

    /// `x` (`steps x inputs`) to `steps x 2 * hidden`: the forward state,
    /// then the backward state, of each step.
    pub fn forward(&self, x: &[f32], steps: usize) -> Buf {
        let hidden = self.hidden;
        let projected = [
            self.forward.input.forward(x, steps),
            self.backward.input.forward(x, steps),
        ];
        let gates = 4 * hidden;
        let mut out = Buf::zeroed(steps * 2 * hidden);
        let out_ptr = team::SendPtr::new(&mut out);
        // The state of each direction (h, c, gates), from the caller: the
        // workers allocate nothing.
        let per = 2 * hidden + gates;
        let mut state = Buf::zeroed(2 * per);
        let state_ptr = team::SendPtr::new(&mut state);
        team::for_each(2, |direction| {
            let weights = if direction == 0 {
                &self.forward
            } else {
                &self.backward
            };
            let pre = &projected[direction];
            // SAFETY: each direction owns its part of `state`, which outlives the step.
            let own = unsafe { state_ptr.slice_mut(direction * per, per) };
            let (h, rest) = own.split_at_mut(hidden);
            let (c, g) = rest.split_at_mut(hidden);
            for step in 0..steps {
                let t = if direction == 0 {
                    step
                } else {
                    steps - 1 - step
                };
                fgemm::matmul_nn_into(h, 1, hidden, &weights.recurrent, gates, g);
                for (v, p) in g.iter_mut().zip(&pre[t * gates..(t + 1) * gates]) {
                    *v += p;
                }
                let (ifo, cell) = g.split_at_mut(2 * hidden);
                let (cell, output) = cell.split_at_mut(hidden);
                activation::sigmoid(ifo);
                activation::tanh(cell);
                activation::sigmoid(output);
                let (input, forget) = ifo.split_at(hidden);
                for j in 0..hidden {
                    c[j] = forget[j] * c[j] + input[j] * cell[j];
                }
                h.copy_from_slice(c);
                activation::tanh(h);
                for (v, o) in h.iter_mut().zip(output.iter()) {
                    *v *= o;
                }
                // SAFETY: each direction writes its own half of each row; `out` outlives the step.
                let row = unsafe { out_ptr.slice_mut(t * 2 * hidden + direction * hidden, hidden) };
                row.copy_from_slice(h);
            }
        });
        out
    }
}
