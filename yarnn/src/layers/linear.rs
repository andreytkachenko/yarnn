use crate::backend::{Backend, BackendBias, BackendGemm, BackendScale};
use crate::layer::{DefaultLayerContext, Layer, LayerExt};
use crate::optimizer::Optimizer;
use crate::params::Params;
use crate::tensor::{Tensor, TensorShape};

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
where
    B: Backend<N>,
    O: Optimizer<N, B>,
{
    inputs: u32,
    outputs: u32,
    use_biases: bool,
    weights: Params<N, B, O>,
    biases: Params<N, B, O>,
}

impl<N, B, O> Layer<N, B, O> for Linear<N, B, O>
where
    B: Backend<N> + BackendGemm<N> + BackendBias<N> + BackendScale<N>,
    O: Optimizer<N, B>,
{
    type Context = DefaultLayerContext<N, B>;

    fn name(&self) -> &str {
        "Linear"
    }

    #[inline]
    fn param_count(&self) -> usize {
        if self.use_biases {
            self.weights.params.shape().size() + self.biases.params.shape().size()
        } else {
            self.weights.params.shape().size()
        }
    }

    fn init(&mut self, backend: &B) {
        self.weights
            .init_random(backend, self.inputs + self.outputs);
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
    fn forward(&self, backend: &B, x: &B::Tensor, ctx: &mut Self::Context) {
        ctx.update_outputs_shape(x.shape().get(0), &self.output_shape());

        backend.matmul(&mut ctx.outputs, x, &self.weights.params);

        if self.use_biases {
            backend.bias_add(&mut ctx.outputs, &self.biases.params);
        }
    }

    #[inline]
    fn backward(&mut self, backend: &B, dy: &B::Tensor, x: &B::Tensor, ctx: &mut Self::Context) {
        ctx.update_deltas_shape(x.shape().get(0), &self.input_shape());

        backend.matmul_nt(&mut ctx.deltas, dy, &self.weights.params);
    }

    fn calc_gradients(
        &mut self,
        backend: &B,
        dy: &B::Tensor,
        x: &B::Tensor,
        _ctx: &mut Self::Context,
    ) {
        let prescaler = 1.0 / x.shape().get(0) as f32;

        backend.matmul_tn(&mut self.weights.grads, x, dy);
        backend.scale(&mut self.weights.grads, backend.scalar_f32(prescaler));

        if self.use_biases {
            backend.scale(&mut self.biases.grads, backend.scalar_f32(prescaler));
            backend.bias_grad(&mut self.biases.grads, &dy);
        }
    }

    #[inline]
    fn optimize(&mut self, backend: &B, optimizer: &O) {
        optimizer.update_params(
            backend,
            &mut self.weights.ctx,
            &mut self.weights.params,
            &mut self.weights.grads,
        );

        if self.use_biases {
            optimizer.update_params(
                backend,
                &mut self.biases.ctx,
                &mut self.biases.params,
                &mut self.biases.grads,
            );
        }
    }
}

impl<N, B, O> LayerExt<N, B, O> for Linear<N, B, O>
where
    B: Backend<N> + BackendGemm<N> + BackendBias<N> + BackendScale<N>,
    O: Optimizer<N, B>,
{
    type Config = LinearConfig;

    fn create(input_shape: TensorShape, cfg: Self::Config) -> Self {
        assert!(input_shape.dims == 1);

        let inputs = input_shape.get(0);

        Linear {
            inputs,
            outputs: cfg.units,
            use_biases: cfg.biases,
            weights: Params::new((inputs, cfg.units)),
            biases: Params::new((cfg.units,)),
        }
    }
}
