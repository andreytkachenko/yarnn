use crate::tensor::{Tensor, TensorShape};
use crate::layer::{Layer, LayerExt, DefaultLayerContext};
use crate::backend::{Backend, PaddingKind, BackendAvgPool2d, Conv2dInfo};
use crate::optimizer::Optimizer;

use core::marker::PhantomData;

pub struct AvgPool2dConfig {
    pub pool: (u32, u32),
    pub strides: Option<(u32, u32)>,
}

impl Default for AvgPool2dConfig {
    fn default() -> Self {
        Self {
            pool: (2, 2),
            strides: None,
        }
    }
}

pub struct AvgPool2d<N, B> 
    where B: Backend<N>
{
    input_shape: TensorShape,
    conv_info: Conv2dInfo,
    _m: PhantomData<fn(N, B)>
}

impl <N, B, O> Layer<N, B, O> for AvgPool2d<N, B> 
    where B: Backend<N> + BackendAvgPool2d<N>,
          O: Optimizer<N, B>
{
    type Context = DefaultLayerContext<N, B>;

    fn name(&self) -> &str {
        "AvgPool2d"
    }
    
    #[inline]
    fn input_shape(&self) -> TensorShape {
        self.input_shape.clone()
    }

    #[inline]
    fn output_shape(&self) -> TensorShape {
        let is = self.input_shape.as_slice();

        // O = (W - K + 2P) / S + 1

        let rows = (is[0] - self.conv_info.kernel.0) / self.conv_info.strides.0 + 1;
        let cols = (is[1] - self.conv_info.kernel.1) / self.conv_info.strides.1 + 1;

        TensorShape::new3d(
            is[0],
            rows,
            cols,
        )
    }
    
    #[inline]
    fn forward(&self, backend: &B, x: &B::Tensor, ctx: &mut Self::Context) {
        ctx.update_outputs_shape(x.shape().get(0), &Layer::<N, B, O>::output_shape(self));

        let y = &mut ctx.outputs;

        assert_eq!(y.shape().dims, 4);
        assert_eq!(x.shape().dims, 4);

        backend.avg_pool2d(y, x, &self.conv_info)
    }

    #[inline]
    fn backward(&mut self, backend: &B, dy: &B::Tensor, x: &B::Tensor, ctx: &mut Self::Context) {
        ctx.update_deltas_shape(x.shape().get(0), &self.input_shape);

        let dx = &mut ctx.deltas;

        assert_eq!(dy.shape().dims, 4);
        assert_eq!(dx.shape().dims, 4);

        backend.avg_pool2d_backprop(dx, dy, x, &self.conv_info);
    }
}

impl <N, B, O> LayerExt<N, B, O> for AvgPool2d<N, B> 
    where B: Backend<N> + BackendAvgPool2d<N>,
          O: Optimizer<N, B>
{
    type Config = AvgPool2dConfig;

    fn create(input_shape: TensorShape, config: Self::Config) -> Self {
        assert!(input_shape.dims == 3);

        AvgPool2d {
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
