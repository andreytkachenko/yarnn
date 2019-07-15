## Yet Another Rust Neural Network framework aka YARNN
Inspired by `darknet` and `leaf`

## What it can right now:
 * not requires `std` (only `alloc` for tensor allocations, bump allocator is ok, so it can be compiled to stm32f4 board)
 * available layers: `Linear`, `ReLu`, `Sigmoid`, `Softmax`(no backward), `Conv2d`, `ZeroPadding2d`, `MaxPool2d`, `AvgPool2d`(no backward), `Flatten`
 * available optimizers: `Sgd`, `Adam`, `RMSProp`
 * available losses: `CrossEntropy`(no forward), `MeanSquareError`
 * available backends: `Native`, `NativeBlas`(no convolution yet)

## What it will can (I hope):
### 1st stage:
 * example of running `yarnn` in browser using `WASM`
 * example of running `yarnn` on `stm32f4` board
 * finish `AvgPool2d` backpropogation
 * add `Dropout` layer
 * add `BatchNorm` layer
 * convolution with BLAS support 
### 2nd stage:
 * `CUDA` support
 * `OpenCL` support
### 3rd stage:
 * `DepthwiseConv2d` layer
 * `Conv3d` layer
 * `Deconv2d` layer
 * `k210` backend

## Model definition example
```rust
use yarnn::model;
use yarnn::layer::*;
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
```

## Contributors are welcome