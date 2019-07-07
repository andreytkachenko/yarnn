use crate::tensor::{Tensor, TensorShape};
use crate::layer::Layer;
use crate::params::Params;
use crate::backend::{Backend, BackendGemm, BackendBias, BackendScale};
use crate::optimizer::{Optimizable, Optimizer};

pub struct LinearConfig {
    pub units: u32,
    pub biases: bool,
}

impl Default for LinearConfig {
    fn default() -> Self {
        Self {
            units: 1,
            biases: false,
        }
    }
}

pub struct Linear<N, B, O> 
    where B: Backend<N>,
          O: Optimizer<N, B>
{
    inputs: u32,
    outputs: u32,
    use_biases: bool,
    weights: Params<N, B, O>,
    biases: Params<N, B, O>,
}

impl <N, B, O> Layer<N, B> for Linear<N, B, O> 
    where B: Backend<N> + BackendGemm<N> + BackendBias<N>,
          O: Optimizer<N, B>
{
    type Config = LinearConfig;

    fn name(&self) -> &str {
        "Linear"
    }
    
    fn create(input_shape: TensorShape, cfg: Self::Config) -> Self {
        assert!(input_shape.dims == 1);

        let inputs = input_shape.get(0);

        Linear {
            inputs,
            outputs: cfg.units,
            use_biases: cfg.biases,
            weights: Params::new((inputs, cfg.units)),
            biases: Params::new((cfg.units, )),
        }
    }

    fn init(&mut self, backend: &B) {
        self.weights.init_random(backend, self.inputs + self.outputs);
        if self.use_biases {
            self.biases.init_zero(backend);
        }
    }

    #[inline]
    fn input_shape(&self) -> TensorShape {
        TensorShape::new1d(self.inputs)
    }

    #[inline]
    fn output_shape(&self) -> TensorShape {
        TensorShape::new1d(self.outputs)
    }
    
    #[inline]
    fn forward(&self, backend: &B, y: &mut B::Tensor, x: &B::Tensor) {
        backend.matmul(y, x, &self.weights.params);

        if self.use_biases {
            backend.bias_add(y, &self.biases.params);
        }
    }

    #[inline]
    fn backward(&self, backend: &B, dx: &mut B::Tensor, dy: &B::Tensor, _: &B::Tensor, _: &B::Tensor) {
        backend.matmul_nt(dx, dy, &self.weights.params);
    }
}

impl <N, B, O> Optimizable<N, B, O> for Linear<N, B, O>
    where B: Backend<N> + BackendGemm<N> + BackendBias<N> + BackendScale<N>,
          O: Optimizer<N, B>
{
    fn calc_gradients(&mut self, backend: &B, inputs: &B::Tensor, deltas: &B::Tensor) {
        let prescaler = 1.0 / inputs.shape().get(0) as f32;

        backend.matmul_tn(&mut self.weights.grads, inputs, deltas);
        backend.scale(&mut self.weights.grads, backend.scalar_f32(prescaler));

        if self.use_biases {
            backend.scale(&mut self.biases.grads, backend.scalar_f32(prescaler));
            backend.bias_grad(&mut self.biases.grads, deltas);
        }
    }

    #[inline]
    fn optimize(&mut self, backend: &B, optimizer: &O) {
        optimizer.update_params(backend, &mut self.weights.ctx, &mut self.weights.params, &self.weights.grads);

        if self.use_biases {
            optimizer.update_params(backend, &mut self.biases.ctx, &mut self.biases.params, &self.biases.grads);
        }
    }
}