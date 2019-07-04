use std::marker::PhantomData;
use crate::backend::{Backend, BackendScale, BackendAxpy, BackendAdd};
use crate::optimizer::{Optimizer, OptimizerContext};
use crate::tensor::{Tensor, TensorShape};


pub struct SgdContext<N, B> 
    where B: Backend<N>
{
    moments: B::Tensor,
    _m: PhantomData<fn(N, B)>,
}

impl<N, B: Backend<N>> OptimizerContext for SgdContext<N, B> {
    fn new<S: Into<TensorShape>>(shape: S) -> Self {
        Self {
            moments: B::Tensor::new(shape),
            _m: Default::default(),
        }
    }
}

pub struct Sgd<N, B: Backend<N>> {
    learning_rate: f32,
    momentum: f32,
    nesterov: bool,
    _m: PhantomData<fn(N, B)>,   
}

impl<N, B: Backend<N>> Sgd<N, B> {
    pub fn new(learning_rate: f32, momentum: f32, nesterov: bool) -> Self {
        Self {
            learning_rate,
            momentum,
            nesterov,
            _m: Default::default(),
        }
    }
}

impl<N, B: Backend<N> + BackendScale<N> + BackendAxpy<N> + BackendAdd<N>> Optimizer<N, B> for Sgd<N, B> {
    type Context = SgdContext<N, B>;

    fn update_gradients(&self, backend: &B, ctx: &mut Self::Context, params: &mut B::Tensor, grads: &B::Tensor) {
        backend.axpy(params, backend.scalar_f32(-self.learning_rate), grads);

        /*// m = momentum * m - lr * grads
        backend.scale(&mut ctx.moments, backend.scalar_f32(self.momentum));
        backend.axpy(&mut ctx.moments, backend.scalar_f32(-self.learning_rate), grads);

        if self.nesterov {
            // p += momentum * m - lr * grads
            backend.axpy(params, backend.scalar_f32(self.momentum), &ctx.moments);
            backend.axpy(params, backend.scalar_f32(-self.learning_rate), grads);
        } else {
            // p += m
            backend.add(params, &ctx.moments);
        }*/
    }
}
