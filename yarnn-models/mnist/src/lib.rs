#![feature(trait_alias)]

pub use self::dense::MnistDenseModel;
pub use self::conv::MnistConvModel;

mod dense {
    use yarnn::model;
    use yarnn::layers::*;

    model! {
        MnistDenseModel (h: u32, w: u32, _c: u32) {
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
}


mod conv {
    use yarnn::model;
    use yarnn::layers::*;
    
    model! {
        MnistConvModel (h: u32, w: u32, c: u32) {
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
}
