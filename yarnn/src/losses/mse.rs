use crate::loss::Loss;
use crate::backend::{Backend, BackendMse};
use crate::tensor::Tensor;

use core::marker::PhantomData;


pub struct MeanSquareErrorLoss<N, B> {
    _m: PhantomData<fn(N, B)>   
}

impl<N, B> MeanSquareErrorLoss<N, B> {
    pub fn new() -> Self {
        Self {
            _m: Default::default()
        }
    }
}

impl<N, B> Loss<N, B> for MeanSquareErrorLoss<N, B> 
    where B: Backend<N> + BackendMse<N>
{
    fn compute(&self, backend: &B, dst: &mut B::Tensor, pred: &B::Tensor, target: &B::Tensor) {
        let batch_size = pred.shape().get(0) as f32;

        backend.scaled_square_diff(dst, target, pred, backend.scalar_f32(0.5 * batch_size));
    }

    fn derivative(&self, backend: &B, dst: &mut B::Tensor, pred: &B::Tensor, target: &B::Tensor) {
        let batch_size = pred.shape().get(0) as f32;

        backend.scaled_diff(dst, pred, target, backend.scalar_f32(1.0 / batch_size));
    }
}