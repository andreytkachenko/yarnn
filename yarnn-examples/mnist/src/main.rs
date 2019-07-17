use yarnn::prelude::*;
use yarnn::native::{Native, NativeTensor};
use yarnn_model_mnist::*;
use yarnn::losses::CrossEntropyLoss;
use yarnn::optimizers::Adam;
use yarnn_native_blas::NativeBlas;
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
    const BATCH_SIZE: usize = 64;

    let backend: NativeBlas<f32, Native<_>> = Default::default();
    let optimizer = Adam::default();
    
    // let mut model = MnistDenseModel::new(28, 28, 1);
    let mut model = MnistConvModel::new(28, 28, 1);
    model.init(&backend);

    println!("{}", &model);

    let mut train_ctx = Default::default();
    let mut test_ctx = Default::default();

    let loss = CrossEntropyLoss::new();

    let Mnist { trn_img, trn_lbl, tst_img, tst_lbl, .. } = MnistBuilder::new()
        .base_path("./datasets/mnist")
        .label_format_digit()
        .finalize();

    let mut inputs = NativeTensor::new((BATCH_SIZE as u32, 1, 28, 28));
    let mut targets = NativeTensor::new((BATCH_SIZE as u32, 10));
    let mut deltas = NativeTensor::new((BATCH_SIZE as u32, 10));

    let test_count = 1000;

    let mut inputs0 = NativeTensor::new((test_count as u32, 1, 28, 28));
    let mut targets0 = NativeTensor::new((test_count as u32, 10));

    let mut tmp = vec![0u8; 10 * test_count];

    let inputs0_slice = &tst_img[0..test_count * 784];
    let targets0_slice = &tst_lbl[0..test_count];

    backend.load_tensor_u8(&mut inputs0, inputs0_slice);
    backend.scale(&mut inputs0, 1.0 / 255.0);

    for (idx, &t) in targets0_slice.iter().enumerate() {
        tmp[idx * 10 + t as usize] = 1;
    }

    backend.load_tensor_u8(&mut targets0, &tmp[..]);

    for epoch in 1 ..= 4 {
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

            model.forward(&backend, &inputs, &mut train_ctx);
            loss.derivative(&backend, &mut deltas, train_ctx.outputs(), &targets);
            model.backward(&backend, &deltas, &inputs, &mut train_ctx);            
            model.calc_gradients(&backend, &deltas, &inputs, &mut train_ctx);
            model.optimize(&backend, &optimizer);
        }

        model.forward(&backend, &inputs0, &mut test_ctx);

        println!("Accuracy {}", calc_accuracy(&backend, test_ctx.outputs(), targets0_slice));
    }
}
