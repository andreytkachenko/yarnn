use crate::tensor::TensorShape;
use crate::backend::{Backend, BackendSoftmax};
use crate::layer::Layer;
use core::marker::PhantomData;

#[derive(Default)]
pub struct SoftmaxConfig;

pub struct Softmax<N, B> 
    where B: Backend<N>,
{
    input_shape: TensorShape,
    _x: PhantomData<fn(N, B)>,
}

impl <N, B> Layer<N, B> for Softmax<N, B> 
    where B: Backend<N> + BackendSoftmax<N>
{
    type Config = SoftmaxConfig;
    
    fn name(&self) -> &str {
        "Softmax"
    }
    
    fn create(input_shape: TensorShape, _cfg: Self::Config) -> Self {
        Softmax {
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
        backend.softmax(y, x);
    }

    #[inline]
    fn backward(&self, backend: &B, dx: &mut B::Tensor, dy: &B::Tensor, _: &B::Tensor, _: &B::Tensor) {
        backend.copy(dx, dy);
    }
}