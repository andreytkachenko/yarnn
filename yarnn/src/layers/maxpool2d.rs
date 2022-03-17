use crate::backend::{Backend, BackendMaxPool2d, Conv2dInfo, PaddingKind};
use crate::layer::{DefaultLayerContext, Layer, LayerExt};
use crate::optimizer::Optimizer;
use crate::tensor::{Tensor, TensorShape};
use core::marker::PhantomData;

pub struct MaxPool2dConfig {
    pub pool: (u32, u32),
    pub strides: Option<(u32, u32)>,
}

impl Default for MaxPool2dConfig {
    fn default() -> Self {
        Self {
            pool: (2, 2),
            strides: None,
        }
    }
}

pub struct MaxPool2d<N, B>
where
    B: Backend<N>,
{
    input_shape: TensorShape,
    conv_info: Conv2dInfo,
    _m: PhantomData<fn(N, B)>,
}

impl<N, B, O> Layer<N, B, O> for MaxPool2d<N, B>
where
    B: Backend<N> + BackendMaxPool2d<N>,
    O: Optimizer<N, B>,
{
    type Context = DefaultLayerContext<N, B>;

    fn name(&self) -> &str {
        "MaxPool2d"
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

        TensorShape::new3d(is[0], rows, cols)
    }

    #[inline]
    fn forward(&self, backend: &B, x: &B::Tensor, ctx: &mut Self::Context) {
        ctx.update_outputs_shape(x.shape().get(0), &Layer::<N, B, O>::output_shape(self));

        backend.max_pool2d(&mut ctx.outputs, x, &self.conv_info)
    }

    #[inline]
    fn backward(&mut self, backend: &B, dy: &B::Tensor, x: &B::Tensor, ctx: &mut Self::Context) {
        ctx.update_deltas_shape(x.shape().get(0), &self.input_shape);

        backend.max_pool2d_backprop(&mut ctx.deltas, dy, x, &self.conv_info);
    }
}

impl<N, B, O> LayerExt<N, B, O> for MaxPool2d<N, B>
where
    B: Backend<N> + BackendMaxPool2d<N>,
    O: Optimizer<N, B>,
{
    type Config = MaxPool2dConfig;

    fn create(input_shape: TensorShape, config: Self::Config) -> Self {
        assert!(input_shape.dims == 3);

        MaxPool2d {
            input_shape,
            conv_info: Conv2dInfo {
                kernel: config.pool,
                strides: config.strides.unwrap_or(config.pool),
                padding: PaddingKind::Valid,
            },
            _m: Default::default(),
        }
    }
}
