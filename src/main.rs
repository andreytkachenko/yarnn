#![feature(specialization)]
#![feature(core_intrinsics)]

pub mod layer;
pub mod layers;

pub mod optimizer;
pub mod optimizers;

pub mod backend;
pub mod backends;

pub mod loss;
pub mod losses;

pub mod tensor;
pub mod params;

use self::backends::{Native, NativeTensorF32};
use self::optimizers::*;
use self::layers::*;
use self::layer::*;
use crate::backend::{Backend, BackendScale};
use self::tensor::Tensor;
use self::loss::Loss;
use self::losses::CrossEntropyLoss;
use mnist::{Mnist, MnistBuilder};

// model! {
//     Conv2dNormalized {
//         Sequential {
//             Conv2d {
//                 kernel_shape: , 
//                 filters: planes,
//                 stride: stride,
//                 bias: false,
//             },

//             BatchNorm2d {

//             }
//         }
//     }
// }

// model! {
//     Conv2dNormalized (height, width, channels) {
//         input: (height, width, channels),

//         name: "",
//         description: "",

//         layers: {
//             conv1: Conv2d {
//                 kernel_shape: , 
//                 filters: planes,
//                 stride: stride,
//                 bias: false,
//             };

//             bn1: BatchNorm2d {

//             };
//         };

//         connect: input -> conv1 -> bn1
//     }
// }



// model! {
//     Bottleneck(inplanes, planes, stride=1, downsample=None) {
//         Sequential  {
//             Conv2d {
//                 kernel_shape: , 
//                 filters: planes,
//                 stride: stride,
//                 bias: false,
//             },

//             BatchNorm2d {

//             },

//             Conv2d {

//             },

//             BatchNorm2d {

//             },

//             Conv2d {

//             },

//             BatchNorm2d {

//             }
//         }
//     }
// }

// model! {
//     Sequential {
//         input_shape: (3, 224, 224),

//         Parallel {
//             final: Sequential {
//                 Addition,
//                 ReLu,
//             },

//             Stream {
//                 Conv2d {
//                     kernel: (3, 3),
//                     strides: (1, 1),
//                 },
//                 ReLu,
//                 MaxPool2d {
//                     kernel: (2, 2),
//                     strides: (2, 2), 
//                 },

//                 Conv2d {
//                     kernel: (3, 3),
//                     strides: (1, 1),
//                 },

//                 MaxPool2d {
//                     kernel: (2, 2),
//                     strides: (2, 2), 
//                 },
//             },

//             Stream {
//                 Conv2d {
//                     kernel: (3, 3),
//                     strides: (1, 1),
//                 },

//                 AvgPool2d {
//                     kernel: (2, 2),
//                     strides: (2, 2), 
//                 },

//                 Conv2d {
//                     kernel: (3, 3),
//                     strides: (1, 1),
//                 },

//                 AvgPool2d {
//                     kernel: (2, 2),
//                     strides: (2, 2), 
//                 },
//             }
//         },

//         Flatten,

//         Linear {
//             units: 4096
//         }
//     }
// }

fn calc_accuracy<N, B: Backend<N>>(back: &B, pred: &B::Tensor, targets: &[u8]) -> f32 {
    let mut vec = vec![0.0; pred.shape().size()];
    back.store_tensor_f32(&pred, vec.as_mut_slice());

    let mut positives = 0;
    let mut total = 0;

    for (x, &y) in vec.chunks(10).zip(targets.iter()) {
        let x = &x[0 .. 10];

        let mut max = 0;
        let mut max_value = 0.0;

        for (idx, &i) in x.iter().enumerate() {
            if i > max_value {
                max_value = i;
                max = idx as u8;
            }
        }

        if max == y {
            positives += 1;
        }

        total += 1;
    }

    (positives as f32) / (total as f32) 
}

fn main() {
    const BATCH_SIZE: usize = 64;

    let backend = Native;
    // let optimizer = Sgd::new(0.1, 0.01, true);
    let optimizer = RMSProp::default();
    let hidden_count = 16;
    
    let mut model = Chain::new(
        LayerImpl::new(&backend, &optimizer, Linear::create(
            (784, ).into(),
            LinearConfig {
                units: hidden_count,
                ..Default::default()
            }
        )),
        Chain::new(
            LayerImpl::new(&backend, &optimizer, Sigmoid::create((hidden_count, ).into(), Default::default())),
            Chain::new(
                LayerImpl::new(&backend, &optimizer, Linear::create(
                    (hidden_count, ).into(),
                    LinearConfig {
                        units: 10,
                        ..Default::default()
                    })
                ),
                LayerImpl::new(&backend, &optimizer, Sigmoid::create((10, ).into(), Default::default()))
            )
        )
    );
    
    let mut train_ctx = ChainContext::new(
        CommonLayerContext::new(),
        ChainContext::new(
            CommonLayerContext::new(),
            ChainContext::new(
                CommonLayerContext::new(),
                CommonLayerContext::new()
            )
        )
    );

    let mut test_ctx = ChainContext::new(
        CommonLayerContext::new(),
        ChainContext::new(
            CommonLayerContext::new(),
            ChainContext::new(
                CommonLayerContext::new(),
                CommonLayerContext::new()
            )
        )
    );

    let loss = CrossEntropyLoss::new();

    let Mnist { trn_img, trn_lbl, tst_img, tst_lbl, .. } = MnistBuilder::new()
        .base_path("./dataset/mnist")
        .label_format_digit()
        .finalize();

    let mut inputs = NativeTensorF32::new((BATCH_SIZE as u32, 784));
    let mut targets = NativeTensorF32::new((BATCH_SIZE as u32, 10));
    let mut deltas = NativeTensorF32::new((BATCH_SIZE as u32, 10));

    let test_count = 1000;

    let mut inputs0 = NativeTensorF32::new((test_count as u32, 784));
    let mut targets0 = NativeTensorF32::new((test_count as u32, 10));

    let mut tmp = vec![0u8; 10 * test_count];

    let inputs0_slice = &tst_img[0..test_count * 784];
    let targets0_slice = &tst_lbl[0..test_count];

    backend.load_tensor_u8(&mut inputs0, inputs0_slice);
    backend.scale(&mut inputs0, 1.0 / 255.0);

    for (idx, &t) in targets0_slice.iter().enumerate() {
        tmp[idx * 10 + t as usize] = 1;
    }

    backend.load_tensor_u8(&mut targets0, &tmp[..]);

    for epoch in 1 ..= 25 {
        println!("epoch {}", epoch);

        for step in 0 .. (60000 / BATCH_SIZE) {
            let offset = step * BATCH_SIZE;
            let mut tmp = [0u8; 10 * BATCH_SIZE];

            let inputs_slice = &trn_img[offset * 784 .. (offset + BATCH_SIZE) * 784 ];
            let targets_slice = &trn_lbl[offset..offset + BATCH_SIZE];

            backend.load_tensor_u8(&mut inputs, inputs_slice);
            backend.scale(&mut inputs, 1.0 / 255.0);

            for (idx, &t) in targets_slice.iter().enumerate() {
                tmp[idx * 10 + t as usize] = 1;
            }

            backend.load_tensor_u8(&mut targets, &tmp[..]);

            model.forward(&inputs, &mut train_ctx);
            loss.derivative(&backend, &mut deltas, train_ctx.outputs(), &targets);
            model.backward(&deltas, &mut train_ctx);            
            model.update(&inputs, &deltas, &mut train_ctx);
        }

        model.forward(&inputs0, &mut test_ctx);

        println!("Accuracy {}", calc_accuracy(&backend, test_ctx.outputs(), targets0_slice));
    }
}



