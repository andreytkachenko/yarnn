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
use crate::backend::*;
use self::tensor::Tensor;
use self::loss::Loss;
use self::losses::CrossEntropyLoss;
use mnist::{Mnist, MnistBuilder};
use crate::optimizer::Optimizer;


pub type DenseModelContext<N, B> = ChainContext<N, B, 
    CommonLayerContext<N, B>,
    ChainContext<N, B,
        CommonLayerContext<N, B>,
        ChainContext<N, B,
            CommonLayerContext<N, B>,
            CommonLayerContext<N, B>,
        >
    >
>;

pub struct DenseModel<N, B, O> 
    where B: Backend<N> + BackendReLu<N> + BackendGemm<N> + BackendBias<N> + BackendSoftmax<N>,
          O: Optimizer<N, B>,
{
    inner: Chain<N, B, O,
        LayerImpl<N, B, O, Linear<N, B, O>>,
        Chain<N, B, O,
            LayerImpl<N, B, O, ReLu<N, B>>,
            Chain<N, B, O,
                LayerImpl<N, B, O, Linear<N, B, O>>,
                LayerImpl<N, B, O, Softmax<N, B>>,
            >
        >
    >
}

impl<N, B, O> DenseModel<N, B, O> 
    where B: Backend<N> + BackendReLu<N> + BackendGemm<N> + BackendBias<N> + BackendSoftmax<N>,
          O: Optimizer<N, B>,
{
    pub fn new(input_size: u32, output_size: u32, hidden_count: u32) -> Self {
        Self {
            inner: Chain::new(
                LayerImpl::new(Linear::create(
                    (input_size, ).into(),
                    LinearConfig {
                        units: hidden_count,
                        ..Default::default()
                    }
                )),
                Chain::new(
                    LayerImpl::new(ReLu::create((hidden_count, ).into(), Default::default())),
                    Chain::new(
                        LayerImpl::new(Linear::create(
                            (hidden_count, ).into(),
                            LinearConfig {
                                units: output_size,
                                ..Default::default()
                            })),
                        LayerImpl::new(Softmax::create((output_size, ).into(), Default::default()))
                    )
                )
            )
        }
    }
}

impl<N, B, O> AbstractLayer<N, B, O> for DenseModel<N, B, O> 
    where B: Backend<N> + BackendReLu<N> + BackendGemm<N> + BackendBias<N> + BackendSoftmax<N>,
          O: Optimizer<N, B>
{
    type Context = DenseModelContext<N, B>;

    #[inline]
    fn forward(&mut self, backend: &B,  inputs: &B::Tensor, ctx: &mut Self::Context) {
        self.inner.forward(backend, inputs, ctx);
    }

    #[inline]
    fn backward(&mut self, backend: &B, deltas: &B::Tensor, ctx: &mut Self::Context) {
        self.inner.backward(backend, deltas, ctx);
    }

    #[inline]
    fn update(&mut self, backend: &B, optimizer: &O, inputs: &B::Tensor, deltas: &B::Tensor, ctx: &mut Self::Context) {
        self.inner.update(backend, optimizer, inputs, deltas, ctx);
    }
}

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
    let optimizer = Adam::default();
    
    let mut model = DenseModel::new(784, 10, 16);
    let mut train_ctx = Default::default();
    let mut test_ctx = Default::default();

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

    for epoch in 1 ..= 20 {
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
            model.backward(&backend, &deltas, &mut train_ctx);            
            model.update(&backend, &optimizer, &inputs, &deltas, &mut train_ctx);
        }

        model.forward(&backend, &inputs0, &mut test_ctx);

        println!("Accuracy {}", calc_accuracy(&backend, test_ctx.outputs(), targets0_slice));
    }
}



