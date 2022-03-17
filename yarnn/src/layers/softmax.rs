use crate::backend::{Backend, BackendSoftmax};
use crate::layer::{DefaultLayerContext, Layer, LayerExt};
use crate::optimizer::Optimizer;
use crate::tensor::{Tensor, TensorShape};
use core::marker::PhantomData;

#[derive(Default)]
pub struct SoftmaxConfig;

pub struct Softmax<N, B>
where
    B: Backend<N>,
{
    input_shape: TensorShape,
    _x: PhantomData<fn(N, B)>,
}

impl<N, B, O> Layer<N, B, O> for Softmax<N, B>
where
    B: Backend<N> + BackendSoftmax<N>,
    O: Optimizer<N, B>,
{
    type Context = DefaultLayerContext<N, B>;

    fn name(&self) -> &str {
        "Softmax"
    }

    #[inline]
    fn input_shape(&self) -> TensorShape {
        self.input_shape.clone()
    }

    #[inline]
    fn forward(&self, backend: &B, x: &B::Tensor, ctx: &mut Self::Context) {
        ctx.update_outputs_shape(x.shape().get(0), &self.input_shape);

        backend.softmax(&mut ctx.outputs, x);
    }

    #[inline]
    fn backward(&mut self, backend: &B, dy: &B::Tensor, _: &B::Tensor, ctx: &mut Self::Context) {
        ctx.update_deltas_shape(dy.shape().get(0), &self.input_shape);

        backend.copy(&mut ctx.deltas, dy);
    }
}

impl<N, B, O> LayerExt<N, B, O> for Softmax<N, B>
where
    B: Backend<N> + BackendSoftmax<N>,
    O: Optimizer<N, B>,
{
    type Config = SoftmaxConfig;

    fn create(input_shape: TensorShape, _cfg: Self::Config) -> Self {
        Softmax {
            input_shape,
            _x: Default::default(),
        }
    }
}
