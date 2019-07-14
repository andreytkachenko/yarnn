use crate::tensor::TensorShape;
use crate::backend::{Backend, BackendReLu};
use crate::layer::Layer;
use core::marker::PhantomData;

#[derive(Default)]
pub struct ReLuConfig;

pub struct ReLu<N, B> 
    where B: Backend<N>,
{
    input_shape: TensorShape,
    _x: PhantomData<fn(N, B)>,
}

impl <N, B> Layer<N, B> for ReLu<N, B> 
    where B: Backend<N> + BackendReLu<N>
{
    type Config = ReLuConfig;
    
    fn name(&self) -> &str {
        "ReLU"
    }
    
    fn create(input_shape: TensorShape, _cfg: Self::Config) -> Self {
        ReLu {
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
        backend.relu(y, x);
    }

    #[inline]
    fn backward(&self, backend: &B, dx: &mut B::Tensor, dy: &B::Tensor, _x: &B::Tensor, y: &B::Tensor) {
        backend.relu_grad(dx, y, dy);
    }
}