use crate::tensor::{Tensor, TensorShape};
use crate::backend::{Backend, BackendReLu};
use crate::layer::{Layer, LayerExt, DefaultLayerContext};
use crate::optimizer::Optimizer;
use core::marker::PhantomData;

#[derive(Default)]
pub struct ReLuConfig;

pub struct ReLu<N, B> 
    where B: Backend<N>,
{
    input_shape: TensorShape,
    _x: PhantomData<fn(N, B)>,
}

impl <N, B, O> Layer<N, B, O> for ReLu<N, B> 
    where B: Backend<N> + BackendReLu<N>,
          O: Optimizer<N, B>
{
    type Context = DefaultLayerContext<N, B>;

    fn name(&self) -> &str {
        "ReLU"
    }
    
    #[inline]
    fn input_shape(&self) -> TensorShape {
        self.input_shape.clone()
    }
    
    #[inline]
    fn forward(&self, backend: &B, x: &B::Tensor, ctx: &mut Self::Context) {
        ctx.update_outputs_shape(x.shape().get(0), &self.input_shape);
        
        backend.relu(&mut ctx.outputs, x);
    }

    #[inline]
    fn backward(&mut self, backend: &B, dy: &B::Tensor, _: &B::Tensor, ctx: &mut Self::Context) {
        ctx.update_deltas_shape(dy.shape().get(0), &self.input_shape);
        
        backend.relu_grad(&mut ctx.deltas, &ctx.outputs, dy);
    }
}

impl <N, B, O> LayerExt<N, B, O> for ReLu<N, B> 
    where B: Backend<N> + BackendReLu<N>,
          O: Optimizer<N, B>
{
    type Config = ReLuConfig;

    fn create(input_shape: TensorShape, _cfg: Self::Config) -> Self {
        ReLu {
            input_shape,
            _x: Default::default()
        }
    }
}