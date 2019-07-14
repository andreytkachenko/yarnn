mod sgd;
mod adam;
mod rmsprop;

pub use self::sgd::*;
pub use self::adam::*;
pub use self::rmsprop::*;

use crate::backend::{Backend, BackendAxpys};
use crate::optimizer::Optimizer;
use core::marker::PhantomData;

pub struct WeightDecay<N, B, O>
    where B: Backend<N>,
          O: Optimizer<N, B>
{
    lamda: f32,
    optimizer: O,
    _m: PhantomData<fn(N, B, O)>,
}

impl<N, B, O> WeightDecay<N, B, O>
    where B: Backend<N>,
          O: Optimizer<N, B> 
{
    pub fn new(lamda: f32, optimizer: O) -> Self {
        Self {
            lamda,
            optimizer,
            _m: Default::default(),
        }
    }
}

impl<N, B, O> Optimizer<N, B> for WeightDecay<N, B, O> 
    where B: Backend<N> + BackendAxpys<N>,
          O: Optimizer<N, B>
{
    type Context = O::Context;

    #[inline]
    fn update_params(&self, backend: &B, ctx: &mut Self::Context, params: &mut B::Tensor, grads: &mut B::Tensor) {
        backend.axpys(grads, backend.scalar_f32(self.lamda), params);

        self.optimizer.update_params(backend, ctx, params, grads);
    }
}
