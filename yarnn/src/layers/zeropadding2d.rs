use crate::tensor::TensorShape;
use crate::backend::{Backend, BackendCopy};
use crate::layer::Layer;
use core::marker::PhantomData;

#[derive(Default)]
pub struct ZeroPadding2dConfig {
    pub paddings: (u32, u32),
}

pub struct ZeroPadding2d<N, B> 
    where B: Backend<N>,
{
    input_shape: TensorShape,
    config: ZeroPadding2dConfig,
    _x: PhantomData<fn(N, B)>,
}

impl <N, B> Layer<N, B> for ZeroPadding2d<N, B> 
    where B: Backend<N> + BackendCopy<N>
{
    type Config = ZeroPadding2dConfig;

    fn name(&self) -> &str {
        "ZeroPadding2d"
    }
    
    fn create(input_shape: TensorShape, config: Self::Config) -> Self {
        ZeroPadding2d {
            input_shape,
            config,
            _x: Default::default()
        }
    }

    #[inline]
    fn input_shape(&self) -> TensorShape {
        self.input_shape.clone()
    }

    #[inline]
    fn output_shape(&self) -> TensorShape {
        let is = self.input_shape.as_slice();

        TensorShape::new3d(
            is[0],
            is[1] + self.config.paddings.0 * 2,
            is[2] + self.config.paddings.1 * 2
        )
    }
    
    #[inline]
    fn forward(&self, _backend: &B, _y: &mut B::Tensor, _x: &B::Tensor) {
        // backend.copy_with_padding(y, x, 0.0, (self.config.0, self.config.1, self.config.0, self.config.1));
    }

    #[inline]
    fn backward(&self, _backend: &B, _dx: &mut B::Tensor, _dy: &B::Tensor, _: &B::Tensor, _: &B::Tensor) {
        // backend.copy(dx, dy);
    }
}