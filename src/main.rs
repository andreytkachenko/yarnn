#![feature(specialization)]

pub mod layer;
pub mod layers;

pub mod optimizer;
pub mod optimizers;

pub mod backend;
pub mod backends;

pub mod loss;
pub mod losses;

pub mod tensor;

use self::backends::{Native, NativeTensorF32};
use self::optimizers::Sgd;
use self::layers::*;
use self::layer::*;
use crate::backend::{Backend, BackendScale};
use self::tensor::Tensor;
use self::loss::Loss;
use self::losses::CrossEntropyLoss;
use mnist::{Mnist, MnistBuilder};

fn calc_accuracy<N, B: Backend<N>>(back: &B, pred: &B::Tensor, targets: &[u8]) -> f32 {
    let mut vec = vec![0.0; pred.shape().size()];
    back.store_tensor_f32(&pred, vec.as_mut_slice());

    let mut positives = 0;
    let mut total = 0;

    for (x, &y) in vec.chunks(10).zip(targets.iter()) {
        let mut max = 0;
        let mut max_value = 0.0;
        let x = &x[0 .. 10];

        for idx in 0 .. 10 {
            let i = x[idx];
            if i > max_value {
                max_value = i;
                max = (idx + 1) as u8;
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
    const BATCH_SIZE: usize = 10;
    let backend = Native;
    let optimizer = Sgd::new(0.01, 0.0, false);
    
    let mut linear_1: LayerImpl<_, _, _, Linear<_, _, &Sgd<_, _>>> = LayerImpl::new((784, ).into(), &backend, &optimizer, LinearConfig {
        outputs: 16
    });

    let mut sigmoid_1: LayerImpl<_, _, _, Sigmoid<_, _>> = LayerImpl::new((16, ).into(), &backend, &optimizer, SigmoidConfig);

    let mut linear_2: LayerImpl<_, _, _, Linear<_, _, &Sgd<_, _>>> = LayerImpl::new((16, ).into(), &backend, &optimizer, LinearConfig {
        outputs: 10
    });

    let mut sigmoid_2: LayerImpl<_, _, _, Sigmoid<_, _>> = LayerImpl::new((10, ).into(), &backend, &optimizer, SigmoidConfig);

    let loss = CrossEntropyLoss::new();

    let Mnist { trn_img, trn_lbl, tst_img, tst_lbl, .. } = MnistBuilder::new()
        .base_path("./dataset/mnist")
        .label_format_digit()
        .finalize();

    let mut targets = NativeTensorF32::new((BATCH_SIZE as u32, 10));
    let mut deltas = NativeTensorF32::new((BATCH_SIZE as u32, 10));
    let mut inputs = NativeTensorF32::new((BATCH_SIZE as u32, 784));

    for epoch in 1 ..= 10 {
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
            
            // forward
            linear_1.forward(&inputs);
            sigmoid_1.forward(linear_1.outputs());
            linear_2.forward(sigmoid_1.outputs());
            sigmoid_2.forward(linear_2.outputs());

            println!("Accuracy {}", calc_accuracy(&backend, sigmoid_2.outputs(), targets_slice));

            // loss
            loss.derivative(&backend, &mut deltas, sigmoid_2.outputs(), &targets);

            // backward
            sigmoid_2.backward(&deltas);
            linear_2.backward(sigmoid_2.deltas());
            sigmoid_1.backward(linear_2.deltas());
            // linear_1.backward(sigmoid_1.deltas());

            // update
            
            linear_1.update(&inputs, sigmoid_1.deltas());
            // sigmoid_1.update(linear_1.outputs(), linear_2.deltas());
            // backend.print_tensor(sigmoid_2.deltas());
            // std::thread::sleep(std::time::Duration::from_secs(5));
            linear_2.update(sigmoid_1.outputs(), sigmoid_2.deltas());
            // sigmoid_2.update(linear_2.outputs(), &deltas);

            // std::thread::sleep(std::time::Duration::from_secs(10));
        }
    }
}



