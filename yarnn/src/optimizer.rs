use crate::backend::Backend;
use crate::tensor::TensorShape;

pub trait OptimizerContext {
    fn new<S: Into<TensorShape>>(shape: S) -> Self;
}

pub trait Optimizer<N, B: Backend<N>> {
    type Context: OptimizerContext;

    fn update_params(
        &self,
        backend: &B,
        ctx: &mut Self::Context,
        params: &mut B::Tensor,
        grads: &mut B::Tensor,
    );
}

impl<'a, N, B: Backend<N>, O: Optimizer<N, B>> Optimizer<N, B> for &'a O {
    type Context = O::Context;

    #[inline]
    fn update_params(
        &self,
        backend: &B,
        ctx: &mut Self::Context,
        params: &mut B::Tensor,
        grads: &mut B::Tensor,
    ) {
        (**self).update_params(backend, ctx, params, grads)
    }
}
