use crate::layers::*;
use crate::layer::*;
use crate::model;

model! {
    DenseModel (h: u32, w: u32, c: u32) {
        input_shape: (h * w),
        layers: {
            Flatten<N, B>,
            Linear<N, B, O> {
                units: 16
            },
            ReLu<N, B>,
            Linear<N, B, O> {
                units: 10
            },
            Softmax<N, B>
        }
    }
}