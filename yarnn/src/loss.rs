use crate::backend::Backend;

pub trait Loss<N, B: Backend<N>> {
    fn compute(&self, backend: &B, dst: &mut B::Tensor, pred: &B::Tensor, target: &B::Tensor);
    fn derivative(&self, backend: &B, dst: &mut B::Tensor, pred: &B::Tensor, target: &B::Tensor);
}