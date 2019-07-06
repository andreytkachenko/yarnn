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
    
    fn create(input_shape: TensorShape, _cfg: Self::Config) -> Self {
        Sigmoid {
            input_shape,
            _x: Default::default()
        }
    }

    fn output_shape(&self) -> TensorShape {
        self.input_shape.clone()
    }
    
    fn forward(&self, backend: &B, dst: &mut B::Tensor, inputs: &B::Tensor) {
        backend.sigmoid(dst, inputs);
    }

    fn backward(&self, backend: &B, dst: &mut B::Tensor, deltas: &B::Tensor, outputs: &B::Tensor) {
        backend.sigmoid_grad(dst, outputs, deltas);
    }
}