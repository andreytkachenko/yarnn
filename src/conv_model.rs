use crate::layers::*;
use crate::layer::*;
use crate::model;

model! {
    ConvModel (h: u32, w: u32, c: u32) {
        input_shape: (c, h, w),
        layers: {
            Conv2d<N, B, O> {
                filters: 8
            },
            ReLu<N, B>,
            MaxPool2d<N, B> {
                pool: (2, 2)
            },

            Conv2d<N, B, O> {
                filters: 8
            },
            ReLu<N, B>,
            MaxPool2d<N, B> {
                pool: (2, 2)
            },

            Flatten<N, B>,
            Linear<N, B, O> {
                units: 10
            },

            Sigmoid<N, B>
        }
    }
}
