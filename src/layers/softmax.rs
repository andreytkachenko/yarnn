use crate::tensor::TensorShape;
use crate::backend::{Backend, BackendSoftmax};
use crate::layer::Layer;
use std::marker::PhantomData;

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
    
    fn create(input_shape: TensorShape, _cfg: Self::Config) -> Self {
        Softmax {
            input_shape,
            _x: Default::default()
        }
    }

    fn output_shape(&self) -> TensorShape {
        self.input_shape.clone()
    }
    
    fn forward(&self, backend: &B, dst: &mut B::Tensor, inputs: &B::Tensor) {
        backend.softmax(dst, inputs);
    }

    fn backward(&self, backend: &B, dy: &mut B::Tensor, dx: &B::Tensor, _: &B::Tensor) {
        backend.copy(dy, dx);
    }
}