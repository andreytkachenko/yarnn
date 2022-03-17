use crate::backend::Backend;
use crate::optimizer::{Optimizer, OptimizerContext};
use crate::tensor::{Tensor, TensorShape};

pub struct Params<N, B: Backend<N>, O: Optimizer<N, B>> {
    pub params: B::Tensor,
    pub grads: B::Tensor,
    pub ctx: O::Context,
}

impl<N, B: Backend<N>, O: Optimizer<N, B>> Params<N, B, O> {
    pub fn new<S: Into<TensorShape>>(shape: S) -> Self {
        let shape = shape.into();

        Self {
            params: B::Tensor::new(shape.clone()),
            grads: B::Tensor::new(shape.clone()),
            ctx: O::Context::new(shape),
        }
    }

    pub fn init_random(&mut self, backend: &B, count: u32) {
        let to = backend.scalar_f32((1.0 / (count as f32)).sqrt());

        backend.fill_random(&mut self.params, backend.scalar_f32(0.0), to);
    }

    pub fn init_zero(&mut self, backend: &B) {
        backend.fill_scalar(&mut self.params, backend.scalar_f32(0.0));
    }
}
