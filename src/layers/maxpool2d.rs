use crate::tensor::{Tensor, TensorShape};
use crate::layer::Layer;
use crate::backend::{Backend, PaddingKind, BackendMaxPool2d, Conv2dInfo};
use core::marker::PhantomData;

pub struct MaxPool2dConfig {
    pub pool: (u32, u32),
    pub strides: Option<(u32, u32)>,
}

impl Default for MaxPool2dConfig {
    fn default() -> Self {
        Self {
            pool: (2, 2),
            strides: None,
        }
    }
}

pub struct MaxPool2d<N, B> 
    where B: Backend<N>
{
    input_shape: TensorShape,
    conv_info: Conv2dInfo,
    _m: PhantomData<fn(N, B)>
}

impl <N, B> Layer<N, B> for MaxPool2d<N, B> 
    where B: Backend<N> + BackendMaxPool2d<N>,
{
    type Config = MaxPool2dConfig;
    
    fn name(&self) -> &str {
        "MaxPool2d"
    }
    
    fn create(input_shape: TensorShape, config: Self::Config) -> Self {
        assert!(input_shape.dims == 3);

        MaxPool2d {
            input_shape,
            conv_info: Conv2dInfo {
                kernel: config.pool,
                strides: config.strides.unwrap_or(config.pool),
                padding: PaddingKind::Valid,
            },
            _m: Default::default(),
        }
    }

    #[inline]
    fn input_shape(&self) -> TensorShape {
        self.input_shape.clone()
    }

    #[inline]
    fn output_shape(&self) -> TensorShape {
        let is = self.input_shape.as_slice();
        
        // O = (W - K + 2P) / S + 1

        let rows = (is[1] - self.conv_info.kernel.0) / self.conv_info.strides.0 + 1;
        let cols = (is[2] - self.conv_info.kernel.1) / self.conv_info.strides.1 + 1;

        TensorShape::new3d(
            is[0],
            rows,
            cols,
        )
    }
    
    #[inline]
    fn forward(&self, backend: &B, y: &mut B::Tensor, x: &B::Tensor) {
        backend.max_pool2d(y, x, &self.conv_info)
    }
    
    #[inline]
    fn backward(&self, backend: &B, dx: &mut B::Tensor, dy: &B::Tensor, x: &B::Tensor, _: &B::Tensor) {
        backend.max_pool2d_backprop(dx, dy, x, &self.conv_info);
    }
}