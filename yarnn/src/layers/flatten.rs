use crate::tensor::{TensorShape, Tensor};
use crate::backend::{Backend, BackendCopy};
use crate::layer::{Layer, LayerExt, DefaultLayerContext};
use crate::optimizer::Optimizer;
use core::marker::PhantomData;


#[derive(Default)]
pub struct FlattenConfig;

pub struct Flatten<N, B> 
    where B: Backend<N>,
{
    input_shape: TensorShape,
    _x: PhantomData<fn(N, B)>,
}

impl <N, B, O> Layer<N, B, O> for Flatten<N, B> 
    where B: Backend<N> + BackendCopy<N>,
          O: Optimizer<N, B>
{
    type Context = DefaultLayerContext<N, B>;

    fn name(&self) -> &str {
        "Flatten"
    }
    
    #[inline]
    fn input_shape(&self) -> TensorShape {
        self.input_shape.clone()
    }

    #[inline]
    fn output_shape(&self) -> TensorShape {
        TensorShape::new1d(self.input_shape.size() as u32)
    }
    
    #[inline]
    fn forward(&self, backend: &B, x: &B::Tensor, ctx: &mut Self::Context) {
        ctx.update_outputs_shape(x.shape().get(0), &Layer::<N, B, O>::output_shape(self));
        
        backend.copy(&mut ctx.outputs, x);
    }

    #[inline]
    fn backward(&mut self, backend: &B, dy: &B::Tensor, x: &B::Tensor, ctx: &mut Self::Context) {
        ctx.update_deltas_shape(x.shape().get(0), &self.input_shape);

        backend.copy(&mut ctx.deltas, dy);
    }
}

impl <N, B, O> LayerExt<N, B, O> for Flatten<N, B> 
    where B: Backend<N> + BackendCopy<N>,
          O: Optimizer<N, B>
{
    type Config = FlattenConfig;

    fn create(input_shape: TensorShape, _cfg: Self::Config) -> Self {
        Flatten {
            input_shape,
            _x: Default::default()
        }
    }
}