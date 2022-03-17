use crate::backend::{Backend, BackendAdam};
use crate::optimizer::{Optimizer, OptimizerContext};
use crate::tensor::{Tensor, TensorShape};
use core::marker::PhantomData;

pub struct RMSPropContext<N, B>
where
    B: Backend<N>,
{
    accum: B::Tensor,
    _m: PhantomData<fn(N, B)>,
}

impl<N, B: Backend<N>> OptimizerContext for RMSPropContext<N, B> {
    fn new<S: Into<TensorShape>>(shape: S) -> Self {
        let shape = shape.into();

        Self {
            accum: B::Tensor::new(shape),
            _m: Default::default(),
        }
    }
}

pub struct RMSProp<N, B: Backend<N>> {
    learning_rate: f32,
    rho: f32,
    epsilon: Option<f32>,
    _m: PhantomData<fn(N, B)>,
}

impl<N, B> Default for RMSProp<N, B>
where
    B: Backend<N>,
{
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            rho: 0.9,
            epsilon: None,
            _m: Default::default(),
        }
    }
}

impl<N, B: Backend<N>> RMSProp<N, B> {
    pub fn new(learning_rate: f32, rho: f32) -> Self {
        Self {
            learning_rate,
            rho,
            epsilon: None,
            _m: Default::default(),
        }
    }
}

impl<N, B: Backend<N> + BackendAdam<N>> Optimizer<N, B> for RMSProp<N, B> {
    type Context = RMSPropContext<N, B>;

    fn update_params(
        &self,
        backend: &B,
        ctx: &mut Self::Context,
        params: &mut B::Tensor,
        grads: &mut B::Tensor,
    ) {
        // new_a = self.rho * a + (1. - self.rho) * K.square(g)
        backend.scale(&mut ctx.accum, backend.scalar_f32(self.rho));
        backend.axpys(&mut ctx.accum, backend.scalar_f32(1.0 - self.rho), grads);

        // new_p = p - lr * g / (K.sqrt(new_a) + self.epsilon)
        backend.adam_p(
            params,
            backend.scalar_f32(-self.learning_rate),
            &grads,
            &ctx.accum,
            backend.scalar_f32(self.epsilon.unwrap_or(core::f32::EPSILON)),
        );
    }
}
