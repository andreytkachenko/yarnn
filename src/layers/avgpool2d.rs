use crate::tensor::{Tensor, TensorShape};
use crate::layer::Layer;
use crate::backend::{Backend, PaddingKind, BackendAvgPool2d, Conv2dInfo};
use core::marker::PhantomData;

pub struct AvgPool2dConfig {
    pub pool: (u32, u32),
    pub strides: (u32, u32),
    pub padding: PaddingKind,
}

impl Default for AvgPool2dConfig {
    fn default() -> Self {
        Self {
            pool: (2, 2),
            strides: (2, 2),
            padding: PaddingKind::Valid,
        }
    }
}

pub struct AvgPool2d<N, B> 
    where B: Backend<N>
{
    input_shape: TensorShape,
    conv_info: Conv2dInfo,
    _m: PhantomData<fn(N, B)>
}

impl <N, B> Layer<N, B> for AvgPool2d<N, B> 
    where B: Backend<N> + BackendAvgPool2d<N>
{
    type Config = AvgPool2dConfig;

    fn name(&self) -> &str {
        "AvgPool2d"
    }
    
    fn create(input_shape: TensorShape, config: Self::Config) -> Self {
        assert!(input_shape.dims == 3);

        AvgPool2d {
            input_shape,
            conv_info: Conv2dInfo {
                kernel: config.pool,
                strides: config.strides,
                padding: config.padding,
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

        let rows = (is[0] - self.conv_info.kernel.0) / self.conv_info.strides.0 + 1;
        let cols = (is[1] - self.conv_info.kernel.1) / self.conv_info.strides.1 + 1;

        TensorShape::new3d(
            is[0],
            rows,
            cols,
        )
    }
    
    #[inline]
    fn forward(&self, backend: &B, y: &mut B::Tensor, x: &B::Tensor) {
        assert_eq!(y.shape().dims, 4);
        assert_eq!(x.shape().dims, 4);

        backend.avg_pool2d(y, x, &self.conv_info)
    }

    #[inline]
    fn backward(&self, backend: &B, dx: &mut B::Tensor, dy: &B::Tensor, x: &B::Tensor, _: &B::Tensor) {
        assert_eq!(dy.shape().dims, 4);
        assert_eq!(dx.shape().dims, 4);

        backend.avg_pool2d_backprop(dx, dy, x, &self.conv_info);
    }
}