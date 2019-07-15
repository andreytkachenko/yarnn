use crate::tensor::{Tensor, TensorShape};
use crate::backend::{Backend, BackendCopy};
use crate::layer::{Layer, LayerExt, DefaultLayerContext};
use crate::optimizer::Optimizer;
use core::marker::PhantomData;

#[derive(Default)]
pub struct ZeroPadding2dConfig {
    pub paddings: (u32, u32),
}

pub struct ZeroPadding2d<N, B> 
    where B: Backend<N>,
{
    input_shape: TensorShape,
    config: ZeroPadding2dConfig,
    _x: PhantomData<fn(N, B)>,
}

impl <N, B, O> Layer<N, B, O> for ZeroPadding2d<N, B> 
    where B: Backend<N> + BackendCopy<N>,
          O: Optimizer<N, B>
{
    type Context = DefaultLayerContext<N, B>;

    fn name(&self) -> &str {
        "ZeroPadding2d"
    }
    
    #[inline]
    fn input_shape(&self) -> TensorShape {
        self.input_shape.clone()
    }

    #[inline]
    fn output_shape(&self) -> TensorShape {
        let is = self.input_shape.as_slice();

        TensorShape::new3d(
            is[0],
            is[1] + self.config.paddings.0 * 2,
            is[2] + self.config.paddings.1 * 2
        )
    }
    
    #[inline]
    fn forward(&self, _backend: &B, x: &B::Tensor, ctx: &mut Self::Context) {
        ctx.update_outputs_shape(x.shape().get(0), &Layer::<N, B, O>::output_shape(self));
        // backend.copy_with_padding(y, x, 0.0, (self.config.0, self.config.1, self.config.0, self.config.1));
    }

    #[inline]
    fn backward(&mut self, _backend: &B, dy: &B::Tensor, _x: &B::Tensor, ctx: &mut Self::Context) {
        ctx.update_deltas_shape(dy.shape().get(0), &self.input_shape);
        // backend.copy(dx, dy);
    }
}

impl <N, B, O> LayerExt<N, B, O> for ZeroPadding2d<N, B> 
    where B: Backend<N> + BackendCopy<N>,
          O: Optimizer<N, B>
{
    type Config = ZeroPadding2dConfig;

    fn create(input_shape: TensorShape, config: Self::Config) -> Self {
        ZeroPadding2d {
            input_shape,
            config,
            _x: Default::default()
        }
    }

}