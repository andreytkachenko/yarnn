use crate::tensor::TensorShape;
use crate::backend::{Backend, BackendSigmoid};
use crate::layer::Layer;
use std::marker::PhantomData;

#[derive(Default)]
pub struct SigmoidConfig;

pub struct Sigmoid<N, B> 
    where B: Backend<N>,
{
    input_shape: TensorShape,
    _x: PhantomData<fn(N, B)>,
}

impl <N, B> Layer<N, B> for Sigmoid<N, B> 
    where B: Backend<N> + BackendSigmoid<N>
{
    type Config = SigmoidConfig;
    
    fn name(&self) -> &str {
        "Sigmoid"
    }
    
    fn create(input_shape: TensorShape, _cfg: Self::Config) -> Self {
        Sigmoid {
            input_shape,
            _x: Default::default()
        }
    }

    #[inline]
    fn input_shape(&self) -> TensorShape {
        self.input_shape.clone()
    }

    #[inline]
    fn forward(&self, backend: &B, y: &mut B::Tensor, x: &B::Tensor) {
        backend.sigmoid(y, x);
    }

    #[inline]
    fn backward(&self, backend: &B, dx: &mut B::Tensor, dy: &B::Tensor, _x: &B::Tensor, y: &B::Tensor) {
        backend.sigmoid_grad(dx, y, dy);
    }
}