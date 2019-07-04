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
use self::optimizers::Adam;
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
    const BATCH_SIZE: usize = 128;

    let backend = Native;
    // let optimizer = Sgd::new(0.01, 0.1, false);
    let optimizer = Adam::default();
    let hidden_count = 64;
    
    let mut linear_1: LayerImpl<_, _, _, Linear<_, _, &Adam<_, _>>> = LayerImpl::new((784, ).into(), &backend, &optimizer, LinearConfig {
        outputs: hidden_count
    });

    let mut sigmoid_1: LayerImpl<_, _, _, Sigmoid<_, _>> = LayerImpl::new((hidden_count, ).into(), &backend, &optimizer, SigmoidConfig);

    let mut linear_2: LayerImpl<_, _, _, Linear<_, _, &Adam<_, _>>> = LayerImpl::new((hidden_count, ).into(), &backend, &optimizer, LinearConfig {
        outputs: 10
    });

    let mut sigmoid_2: LayerImpl<_, _, _, Sigmoid<_, _>> = LayerImpl::new((10, ).into(), &backend, &optimizer, SigmoidConfig);

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

    let mut train_linear_1 = LayerContext::new();
    let mut train_sigmoid_1 = LayerContext::new();
    let mut train_linear_2 = LayerContext::new();
    let mut train_sigmoid_2 = LayerContext::new();
    
    let mut test_linear_1 = LayerContext::new();
    let mut test_sigmoid_1 = LayerContext::new();
    let mut test_linear_2 = LayerContext::new();
    let mut test_sigmoid_2 = LayerContext::new();

    for epoch in 1 ..= 80 {
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
            linear_1.forward(&inputs, &mut train_linear_1);
            sigmoid_1.forward(&train_linear_1.outputs, &mut train_sigmoid_1);
            linear_2.forward(&train_sigmoid_1.outputs, &mut train_linear_2);
            sigmoid_2.forward(&train_linear_2.outputs, &mut train_sigmoid_2);

            // loss
            loss.derivative(&backend, &mut deltas, &train_sigmoid_2.outputs, &targets);

            // backward
            sigmoid_2.backward(&deltas, &mut train_sigmoid_2);
            linear_2.backward(&train_sigmoid_2.deltas, &mut train_linear_2);
            sigmoid_1.backward(&train_linear_2.deltas, &mut train_sigmoid_1);
            // linear_1.backward(&train_sigmoid_1.deltas, &mut train_linear_1);

            // update
            
            linear_1.update(&inputs, &train_sigmoid_1.deltas, &mut train_linear_1);
            sigmoid_1.update(train_linear_1.outputs(), train_linear_2.deltas(), &mut train_sigmoid_1);
            linear_2.update(&train_sigmoid_1.outputs, &train_sigmoid_2.deltas, &mut train_linear_2);
            sigmoid_2.update(train_linear_2.outputs(), &deltas, &mut train_sigmoid_2);
        }

        linear_1.forward(&inputs0, &mut test_linear_1);
        sigmoid_1.forward(test_linear_1.outputs(), &mut test_sigmoid_1);
        linear_2.forward(test_sigmoid_1.outputs(), &mut test_linear_2);
        sigmoid_2.forward(test_linear_2.outputs(), &mut test_sigmoid_2);

        println!("Accuracy {}", calc_accuracy(&backend, &test_sigmoid_2.outputs, targets0_slice));
    }
}



