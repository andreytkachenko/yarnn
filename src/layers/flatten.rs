use crate::tensor::{Tensor, TensorShape};
use crate::backend::{Backend, BackendCopy};
use crate::layer::Layer;
use std::marker::PhantomData;

#[derive(Default)]
pub struct FlattenConfig;

pub struct Flatten<N, B> 
    where B: Backend<N>,
{
    input_shape: TensorShape,
    _x: PhantomData<fn(N, B)>,
}

impl <N, B> Layer<N, B> for Flatten<N, B> 
    where B: Backend<N> + BackendCopy<N>
{
    type Config = FlattenConfig;

    fn name(&self) -> &str {
        "Flatten"
    }
    
    fn create(input_shape: TensorShape, _cfg: Self::Config) -> Self {
        Flatten {
            input_shape,
            _x: Default::default()
        }
    }

    #[inline]
    fn input_shape(&self) -> TensorShape {
        self.input_shape.clone()
    }

    #[inline]
    fn output_shape(&self) -> TensorShape {
        TensorShape::new1d(self.input_shape.size() as u32)
    }
    
    #[inline]
    fn forward(&self, backend: &B, y: &mut B::Tensor, x: &B::Tensor) {
        backend.copy(y, x);
    }

    #[inline]
    fn backward(&self, backend: &B, dx: &mut B::Tensor, dy: &B::Tensor, _: &B::Tensor, _: &B::Tensor) {
        backend.copy(dx, dy);
    }
}