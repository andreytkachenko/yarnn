use crate::backend::Backend;
use crate::optimizer::{Optimizer, Optimizable};
use crate::tensor::{Tensor, TensorShape};

use core::marker::PhantomData;

pub trait Layer<N, B: Backend<N>> {
    type Config: Default;
    fn create(input_shape: TensorShape, cfg: Self::Config) -> Self;
    
    #[inline]
    fn init(&mut self, _backend: &B) {}
    
    #[inline]
    fn input_shape(&self) -> TensorShape {
        self.output_shape()
    }
    
    fn output_shape(&self) -> TensorShape;
    
    fn forward(&self, backend: &B, dst: &mut B::Tensor, inputs: &B::Tensor);
    
    fn backward(&self, backend: &B, dst: &mut B::Tensor, deltas: &B::Tensor, outputs: &B::Tensor);
}

/// Temporary solution until I find a solution with problem of inference with specializations
impl <T, N, B, O> Optimizable<N, B, O> for T
    where T: Layer<N, B>,
          B: Backend<N>,
          O: Optimizer<N, B>
{
    default fn calc_gradients(&mut self, _backend: &B, _inputs: &B::Tensor, _deltas: &B::Tensor) {}
    default fn optimize(&mut self, _backend: &B, _optimizer: &O) {}
}

pub trait AbstractLayer<N, B: Backend<N>> {
    type Context: LayerContext<N, B>;

    fn forward(&mut self, inputs: &B::Tensor, ctx: &mut Self::Context);
    fn backward(&mut self, deltas: &B::Tensor, ctx: &mut Self::Context);
    fn update(&mut self, inputs: &B::Tensor, deltas: &B::Tensor, ctx: &mut Self::Context);
}

pub trait LayerContext<N, B: Backend<N>> {
    fn outputs(&self) -> &B::Tensor;
    fn deltas(&self) -> &B::Tensor;
}

pub struct CommonLayerContext<N, B> 
    where B: Backend<N>,
{
    pub outputs: B::Tensor,
    pub deltas: B::Tensor,
}

impl <N, B> CommonLayerContext<N, B> 
    where B: Backend<N>,
{
    pub fn new() -> Self {
        Self {
            outputs: B::Tensor::new(()),
            deltas: B::Tensor::new(()),
        }
    }

    pub fn update_deltas_bs(&mut self, bs: u32, input_shape: &TensorShape) {
        let mut new_deltas_shape = TensorShape::new1d(bs);
        new_deltas_shape.append(input_shape.clone());

        if self.deltas.shape() != &new_deltas_shape {
            self.deltas.resize(new_deltas_shape.clone());
        }
    }

    pub fn update_outputs_bs(&mut self, bs: u32, output_shape: &TensorShape) {
        let mut new_output_shape = TensorShape::new1d(bs);

        new_output_shape.append(output_shape.clone());

        if self.outputs.shape() != &new_output_shape {
            self.outputs.resize(new_output_shape);
        }
    }
}

impl <N, B> LayerContext<N, B> for CommonLayerContext<N, B>
    where B: Backend<N>,
{
    #[inline]
    fn outputs(&self) -> &B::Tensor {
        &self.outputs
    }

    #[inline]
    fn deltas(&self) -> &B::Tensor {
        &self.deltas
    }
}

pub struct LayerImpl <N, B, O, L> 
    where B: Backend<N>,
          O: Optimizer<N, B>
{
    layer: L,
    backend: B,
    optimizer: O,
    _m: PhantomData<fn(N)>,
}

impl <N, B, O, L> LayerImpl<N, B, O, L> 
    where B: Backend<N>,
          O: Optimizer<N, B>,
          L: Layer<N, B> + Optimizable<N, B, O>
{
    pub fn new(b: B, o: O, mut layer: L) -> Self {
        layer.init(&b);

        Self {
            backend: b,
            optimizer: o,
            layer,
            _m: Default::default(),
        }
    }
}

impl <N, B, O, L> AbstractLayer<N, B> for LayerImpl<N, B, O, L> 
    where B: Backend<N>,
          O: Optimizer<N, B>,
          L: Layer<N, B> + Optimizable<N, B, O>
{
    type Context = CommonLayerContext<N, B>;

    #[inline]
    fn forward(&mut self, inputs: &B::Tensor, ctx: &mut Self::Context) {
        ctx.update_outputs_bs(inputs.shape().get(0), &self.layer.output_shape());
        self.layer.forward(&self.backend, &mut ctx.outputs, inputs);
    }

    #[inline]
    fn backward(&mut self, deltas: &B::Tensor, ctx: &mut Self::Context) {
        ctx.update_deltas_bs(deltas.shape().get(0), &self.layer.input_shape());
        self.layer.backward(&self.backend, &mut ctx.deltas, deltas, &ctx.outputs);
    }

    #[inline]
    fn update(&mut self, inputs: &B::Tensor, deltas: &B::Tensor, _ctx: &mut Self::Context) {
        self.layer.calc_gradients(&self.backend, inputs, deltas);
        self.layer.optimize(&self.backend, &self.optimizer);
    }
}
