use std::marker::PhantomData;
use crate::backend::{Backend, BackendAdam};
use crate::optimizer::{Optimizer, OptimizerContext};
use crate::tensor::{Tensor, TensorShape};
use std::cell::Cell;


pub struct AdamContext<N, B> 
    where B: Backend<N>
{
    moms: B::Tensor,
    vels: B::Tensor,
    vhats: B::Tensor,
    _m: PhantomData<fn(N, B)>,
}

impl<N, B: Backend<N>> OptimizerContext for AdamContext<N, B> {
    fn new<S: Into<TensorShape>>(shape: S) -> Self {
        let shape = shape.into();

        Self {
            moms: B::Tensor::new(shape.clone()),
            vels: B::Tensor::new(shape.clone()),
            vhats: B::Tensor::new(shape),
            _m: Default::default(),
        }
    }
}

pub struct Adam<N, B: Backend<N>> {
    learning_rate: f32,
    beta_1: f32,
    beta_2: f32,
    epsilon: Option<f32>,
    amsgrad: bool,
    iteration: Cell<f32>,
    _m: PhantomData<fn(N, B)>,   
}

impl<N, B> Default for Adam<N, B> 
    where B: Backend<N>
{
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: None,
            amsgrad: false,
            iteration: Cell::new(0.0),
            _m: Default::default(),
        }
    }
}

impl<N, B: Backend<N>> Adam<N, B> {
    pub fn new(learning_rate: f32, beta_1: f32, beta_2: f32, amsgrad: bool) -> Self {
        Self {
            learning_rate,
            beta_1,
            beta_2,
            epsilon: None,
            amsgrad,
            iteration: Cell::new(0.0),
            _m: Default::default(),
        }
    }
}

impl<N, B: Backend<N> + BackendAdam<N>> Optimizer<N, B> for Adam<N, B> {
    type Context = AdamContext<N, B>;

    fn update_params(&self, backend: &B, ctx: &mut Self::Context, params: &mut B::Tensor, grads: &mut B::Tensor) {
        let iter = self.iteration.get();
        let t = iter + 1.0;
        self.iteration.set(iter + 0.25);

        let lr_t = self.learning_rate * ((1.0 - self.beta_2.powf(t)).sqrt() / (1.0 - self.beta_1.powf(t)));

        // m_t = (self.beta_1 * m) + (1. - self.beta_1) * g;
        backend.scale(&mut ctx.moms, backend.scalar_f32(self.beta_1));
        backend.axpy(&mut ctx.moms, backend.scalar_f32(1.0 - self.beta_1), grads);

        // v_t = (self.beta_2 * v) + (1. - self.beta_2) * square(grads);
        backend.scale(&mut ctx.vels, backend.scalar_f32(self.beta_2));
        backend.axpys(&mut ctx.vels, backend.scalar_f32(1.0 - self.beta_2), grads);

        if self.amsgrad {
            backend.maximum(&mut ctx.vhats, &ctx.vels);
            backend.adam_p(params, backend.scalar_f32(-lr_t), &ctx.moms, &ctx.vhats, backend.scalar_f32(self.epsilon.unwrap_or(std::f32::EPSILON)));
        } else {
            // p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)
            backend.adam_p(params, backend.scalar_f32(-lr_t), &ctx.moms, &ctx.vels, backend.scalar_f32(self.epsilon.unwrap_or(std::f32::EPSILON)));
        }
    }
}
