use crate::tensor::{Tensor, TensorShape};
use crate::layer::{Layer, Params};
use crate::backend::{Backend, BackendGemm, BackendBias, BackendScale, BackendAxpy};
use crate::optimizer::{Optimizable, Optimizer};
use crate::backends::Native;

pub struct LinearConfig {
    pub outputs: u32,
}

impl Default for LinearConfig {
    fn default() -> Self {
        Self {
            outputs: 1
        }
    }
}

pub struct Linear<N, B, O> 
    where B: Backend<N>,
          O: Optimizer<N, B>
{
    outputs: u32,
    input_shape: TensorShape,
    weights: Params<N, B, O>,
    // biases: Params<N, B, O>,
}

impl <N, B, O> Layer<N, B> for Linear<N, B, O> 
    where B: Backend<N> + BackendGemm<N> + BackendBias<N>,
          O: Optimizer<N, B>
{
    type Config = LinearConfig;
    
    fn create(input_shape: TensorShape, cfg: Self::Config) -> Self {
        let dim0 = input_shape.get(0);

        Linear {
            outputs: cfg.outputs,
            input_shape,
            weights: Params::new((dim0, cfg.outputs)),
            // biases: Params::new((cfg.outputs, )),
        }
    }

    fn init(&mut self, backend: &B) {
        self.weights.init_random(backend, self.outputs + self.input_shape.size() as u32);
        // self.biases.init_zero(backend);
    }

    fn output_shape(&self) -> TensorShape {
        TensorShape::new1d(self.outputs)
    }
    
    fn forward(&self, backend: &B, dst: &mut B::Tensor, inputs: &B::Tensor) {
        backend.matmul(dst, inputs, &self.weights.params);
        // backend.bias_add(dst, &self.biases.params);
    }

    fn backward(&mut self, backend: &B, dst: &mut B::Tensor, deltas: &B::Tensor, _: &B::Tensor) {
        backend.matmul_nt(dst, deltas, &self.weights.params);
    }
}

impl <N, B, O> Optimizable<N, B, O> for Linear<N, B, O>
    where B: Backend<N> + BackendGemm<N> + BackendBias<N> + BackendScale<N>  + BackendAxpy<N>,
          O: Optimizer<N, B>
{
    fn calc_gradients(&mut self, backend: &B, inputs: &B::Tensor, deltas: &B::Tensor) {
        backend.matmul_tn(&mut self.weights.grads, inputs, deltas);
        backend.scale(&mut self.weights.grads, backend.scalar_f32(1.0 / inputs.shape().get(0) as f32));

        // backend.print_tensor(&self.weights.grads);
        // backend.bias_grad(&mut self.biases.grads, inputs);
    }

    fn optimize(&mut self, backend: &B, optimizer: &O) {
        backend.axpy(&mut self.weights.params, backend.scalar_f32(-0.01), &self.weights.grads);
        // backend.print_tensor(&self.weights.params);
        // backend.print_tensor(&self.weights.params);
        // optimizer.update_gradients(backend, &mut self.weights.ctx, &mut self.weights.params, &self.weights.grads);
        // optimizer.update_gradients(backend, &mut self.biases.ctx, &mut self.biases.params, &self.biases.grads);
    }
}