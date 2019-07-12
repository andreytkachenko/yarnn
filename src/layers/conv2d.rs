use crate::tensor::{Tensor, TensorShape};
use crate::layer::Layer;
use crate::params::Params;
use crate::backend::{Backend, Conv2dInfo, PaddingKind, BackendBias, BackendConv2d, BackendScale};
use crate::optimizer::{Optimizable, Optimizer};

pub struct Conv2dConfig {
    pub filters: u32,
    pub kernel: (u32, u32),
    pub strides: (u32, u32),
    pub padding: PaddingKind,
    pub biases: bool,
}

impl Default for Conv2dConfig {
    fn default() -> Self {
        Self {
            filters: 1,
            kernel: (3, 3),
            strides: (1, 1),
            padding: PaddingKind::Valid,
            biases: false,
        }
    }
}

pub struct Conv2d<N, B, O> 
    where B: Backend<N>,
          O: Optimizer<N, B>
{
    input_shape: TensorShape,
    units: u32,
    conv_info: Conv2dInfo,
    use_biases: bool,
    filters: Params<N, B, O>,
    biases: Params<N, B, O>,
}

impl <N, B, O> Layer<N, B> for Conv2d<N, B, O> 
    where B: Backend<N> + BackendConv2d<N> + BackendBias<N>,
          O: Optimizer<N, B>
{
    type Config = Conv2dConfig;

    fn name(&self) -> &str {
        "Conv2d"
    }
    
    fn create(input_shape: TensorShape, cfg: Self::Config) -> Self {
        assert!(input_shape.dims == 3);

        Conv2d {
            input_shape,
            units: cfg.filters,
            conv_info: Conv2dInfo {
                kernel: cfg.kernel,
                padding: cfg.padding, 
                strides: cfg.strides,
            },
            use_biases: cfg.biases,
            filters: Params::new((cfg.filters, cfg.kernel.0, cfg.kernel.1)),
            biases: Params::new((cfg.filters, )),
        }
    }
    
    fn init(&mut self, backend: &B) {
        self.filters.init_random(backend, self.conv_info.kernel.0 * self.conv_info.kernel.1 + self.filters.params.shape().get(0));

        if self.use_biases {
            self.biases.init_zero(backend);
        }
    }

    #[inline]
    fn input_shape(&self) -> TensorShape {
        self.input_shape.clone()
    }

    #[inline]
    fn output_shape(&self) -> TensorShape {
        let is = self.input_shape.as_slice();

        // O = (W - K + 2P) / S + 1

        let rows = (is[1] - self.conv_info.kernel.0) / self.conv_info.strides.0 + 1;
        let cols = (is[2] - self.conv_info.kernel.1) / self.conv_info.strides.1 + 1;

        TensorShape::new3d(
            self.units,
            rows,
            cols,
        )
    }
    
    #[inline]
    fn forward(&self, backend: &B, y: &mut B::Tensor, x: &B::Tensor) {
        assert_eq!(y.shape().dims, 4);
        assert_eq!(x.shape().dims, 4);

        backend.conv2d_forward(y, x, &self.filters.params, &self.conv_info);

        if self.use_biases {
            unimplemented!();
            // backend.bias_add(y, &self.biases.params);
        }
    }

    #[inline]
    fn backward(&self, backend: &B, dx: &mut B::Tensor, dy: &B::Tensor, _: &B::Tensor, _: &B::Tensor) {
        assert_eq!(dy.shape().dims, 4);
        assert_eq!(dx.shape().dims, 4);

        backend.conv2d_backward_input(dx, dy, &self.filters.params, &self.conv_info);
    }
}

impl <N, B, O> Optimizable<N, B, O> for Conv2d<N, B, O>
    where B: Backend<N> + BackendConv2d<N> + BackendBias<N> + BackendScale<N>,
          O: Optimizer<N, B>
{
    #[inline]
    fn calc_gradients(&mut self, backend: &B, x: &B::Tensor, dy: &B::Tensor) {
        assert_eq!(dy.shape().dims, 4);
        assert_eq!(x.shape().dims, 4);

        backend.conv2d_backward_filter(&mut self.filters.grads, x, dy, &self.conv_info);
        let prescaler = 1.0 / x.shape().get(0) as f32;

        backend.scale(&mut self.filters.grads, backend.scalar_f32(prescaler));
    }

    #[inline]
    fn optimize(&mut self, backend: &B, optimizer: &O) {
        optimizer.update_params(backend, &mut self.filters.ctx, &mut self.filters.params, &mut self.filters.grads);

        if self.use_biases {
            unimplemented!()
        //     optimizer.update_params(backend, &mut self.biases.ctx, &mut self.biases.params, &self.biases.grads);
        }
    }
}