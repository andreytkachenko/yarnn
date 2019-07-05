use crate::tensor::TensorShape;
use crate::backend::{Backend, BackendReLu};
use crate::layer::Layer;
use std::marker::PhantomData;

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
    
    fn create(input_shape: TensorShape, _cfg: Self::Config) -> Self {
        ReLu {
            input_shape,
            _x: Default::default()
        }
    }

    fn output_shape(&self) -> TensorShape {
        self.input_shape.clone()
    }
    
    fn forward(&self, backend: &B, dst: &mut B::Tensor, inputs: &B::Tensor) {
        backend.relu(dst, inputs);
    }

    fn backward(&mut self, backend: &B, dst: &mut B::Tensor, deltas: &B::Tensor, outputs: &B::Tensor) {
        backend.relu_grad(dst, outputs, deltas);
    }
}