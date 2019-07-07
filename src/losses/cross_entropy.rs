use crate::loss::Loss;
use crate::backend::{Backend, BackendSub};

use std::marker::PhantomData;

pub struct CrossEntropyLoss<N, B> {
    _m: PhantomData<fn(N, B)>   
}

impl<N, B> CrossEntropyLoss<N, B> {
    pub fn new() -> Self {
        Self {
            _m: Default::default()
        }
    }
}

impl<N, B: Backend<N> + BackendSub<N>> Loss<N, B> for CrossEntropyLoss<N, B> {
    fn compute(&self, _backend: &B, _dst: &mut B::Tensor, _pred: &B::Tensor, _target: &B::Tensor) {
        // TODO
    }

    fn derivative(&self, backend: &B, dst: &mut B::Tensor, pred: &B::Tensor, target: &B::Tensor) {      
        backend.sub(dst, pred, target);
    }
}