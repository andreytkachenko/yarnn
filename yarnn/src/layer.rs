use crate::backend::Backend;
use crate::optimizer::{Optimizer, Optimizable};
use crate::tensor::{Tensor, TensorShape};

use core::marker::PhantomData;


pub trait Layer<N, B: Backend<N>> {
    type Config: Default;
    fn name(&self) -> &str;
    fn create(input_shape: TensorShape, cfg: Self::Config) -> Self;
    
    #[inline]
    fn init(&mut self, _backend: &B) {}

    fn input_shape(&self) -> TensorShape;

    #[inline]
    fn output_shape(&self) -> TensorShape {
        self.input_shape()
    }
    
    fn forward(&self, backend: &B, y: &mut B::Tensor, x: &B::Tensor);
    fn backward(&self, backend: &B, dx: &mut B::Tensor, dy: &B::Tensor, x: &B::Tensor, y: &B::Tensor);
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

pub trait AbstractLayer<N, B: Backend<N>, O: Optimizer<N, B>>: core::fmt::Display {
    type Context: LayerContext<N, B>;

    fn forward(&mut self, backend: &B, inputs: &B::Tensor, ctx: &mut Self::Context);
    fn backward(&mut self, backend: &B, deltas: &B::Tensor, inputs: &B::Tensor, ctx: &mut Self::Context);
    fn update(&mut self, backend: &B, optimizer: &O, inputs: &B::Tensor, deltas: &B::Tensor, ctx: &mut Self::Context);
    
    #[inline]
    fn add_layer<L: Layer<N, B>>(self, cfg: L::Config) -> crate::layers::Chain<N, B, O, Self, LayerImpl<N, B, O, L>> 
        where Self: Sized
    {
        crate::layers::Chain::new(
            self,
            LayerImpl::new(L::create(().into(), cfg)),
        )
    }
}

pub trait LayerContext<N, B: Backend<N>>: Default {
    fn outputs(&self) -> &B::Tensor;
    fn deltas(&self) -> &B::Tensor;
}

pub struct CommonLayerContext<N, B> 
    where B: Backend<N>,
{
    pub outputs: B::Tensor,
    pub deltas: B::Tensor,
}

impl <N, B> Default for CommonLayerContext<N, B> 
    where B: Backend<N>,
{
    fn default() -> Self {
        Self {
            outputs: B::Tensor::new(()),
            deltas: B::Tensor::new(()),
        }
    }
}

impl <N, B> CommonLayerContext<N, B> 
    where B: Backend<N>,
{
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
    pub layer: L,
    initialized: bool,
    _m: PhantomData<fn(N, B, O)>,
}

impl <N, B, O, L> LayerImpl<N, B, O, L> 
    where B: Backend<N>,
          O: Optimizer<N, B>,
          L: Layer<N, B> + Optimizable<N, B, O>
{
    pub fn new(layer: L) -> Self {
        Self {
            layer,
            initialized: false,
            _m: Default::default(),
        }
    }
}

impl <N, B, O, L> core::fmt::Display for LayerImpl<N, B, O, L> 
    where B: Backend<N>,
          O: Optimizer<N, B>,
          L: Layer<N, B>
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "{} -> ", self.layer.input_shape())?;
        write!(f, "{}", self.layer.name())?;
        writeln!(f, " -> {}", self.layer.output_shape())?;

        Ok(())
    }
}

impl <N, B, O, L> AbstractLayer<N, B, O> for LayerImpl<N, B, O, L> 
    where B: Backend<N>,
          O: Optimizer<N, B>,
          L: Layer<N, B> + Optimizable<N, B, O>
{
    type Context = CommonLayerContext<N, B>;

    #[inline]
    fn forward(&mut self, backend: &B, inputs: &B::Tensor, ctx: &mut Self::Context) {
        if !self.initialized {
            self.initialized = true;
            self.layer.init(&backend);
        }

        ctx.update_outputs_bs(inputs.shape().get(0), &self.layer.output_shape());
        self.layer.forward(&backend, &mut ctx.outputs, inputs);
    }

    #[inline]
    fn backward(&mut self, backend: &B, deltas: &B::Tensor, inputs: &B::Tensor, ctx: &mut Self::Context) {
        ctx.update_deltas_bs(deltas.shape().get(0), &self.layer.input_shape());
        self.layer.backward(&backend, &mut ctx.deltas, deltas, inputs, &ctx.outputs);
    }

    #[inline]
    fn update(&mut self, backend: &B, optimizer: &O, inputs: &B::Tensor, deltas: &B::Tensor, _ctx: &mut Self::Context) {
        self.layer.calc_gradients(&backend, inputs, deltas);
        self.layer.optimize(&backend, &optimizer);
    }
}
