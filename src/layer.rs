use crate::backend::Backend;
use crate::optimizer::{Optimizer, OptimizerContext, Optimizable};
use crate::tensor::{Tensor, TensorShape};
use std::marker::PhantomData;
use std::borrow::Cow;

pub trait Layer<N, B: Backend<N>> {
    type Config: Default;
    fn init(&mut self, _backend: &B) {}
    fn create(input_shape: TensorShape, cfg: Self::Config) -> Self;
    fn output_shape(&self) -> TensorShape;
    fn forward(&self, backend: &B, dst: &mut B::Tensor, inputs: &B::Tensor);
    fn backward(&mut self, backend: &B, dst: &mut B::Tensor, deltas: &B::Tensor, outputs: &B::Tensor);
}

pub trait AbstractLayer<N, B: Backend<N>> {
    type Context;

    fn forward(&mut self, inputs: &B::Tensor, ctx: &mut Self::Context);
    fn backward(&mut self, deltas: &B::Tensor, ctx: &mut Self::Context);
    fn update(&mut self, inputs: &B::Tensor, deltas: &B::Tensor, ctx: &mut Self::Context);
}

pub struct LayerContext<N, B> 
    where B: Backend<N>,
{
    pub outputs: B::Tensor,
    pub deltas: B::Tensor,
}

impl <N, B> LayerContext<N, B> 
    where B: Backend<N>,
{
    pub fn new() -> Self {
        Self {
            outputs: B::Tensor::new(()),
            deltas: B::Tensor::new(()),
        }
    }

    pub fn set_shape(&mut self, bs: u32, input_shape: &TensorShape, output_shape: &TensorShape) {
        let mut new_output_shape = TensorShape::new1d(bs);
        let mut new_deltas_shape = TensorShape::new1d(bs);

        new_output_shape.append(output_shape.clone());
        new_deltas_shape.append(input_shape.clone());

        if self.outputs.shape() != &new_output_shape {
            self.outputs.resize(new_output_shape);
        }

        if self.deltas.shape() != &new_deltas_shape {
            self.deltas.resize(new_deltas_shape.clone());
        }
    }

    #[inline]
    pub fn outputs(&self) -> &B::Tensor {
        &self.outputs
    }

    #[inline]
    pub fn deltas(&self) -> &B::Tensor {
        &self.deltas
    }
}

pub struct LayerImpl <N, B, O, L> 
    where B: Backend<N>,
          O: Optimizer<N, B>
{
    id: u64,
    name: Cow<'static, str>,
    layer: L,
    backend: B,
    optimizer: O,
    input_shape: TensorShape,
    _m: PhantomData<fn(N)>,
}

impl <N, B, O, L> LayerImpl<N, B, O, L> 
    where B: Backend<N>,
          O: Optimizer<N, B>,
          L: Layer<N, B>
{
    pub fn new(input_shape: TensorShape, b: B, o: O, cfg: L::Config) -> Self {
        let mut layer = L::create(input_shape.slice(1..), cfg);
        layer.init(&b);

        Self {
            id: 0,
            name: "x".into(),
            backend: b,
            optimizer: o,
            layer,
            input_shape,
            _m: Default::default(),
        }
    }

}

impl <N, B, O, L> AbstractLayer<N, B> for LayerImpl<N, B, O, L> 
    where B: Backend<N>,
          O: Optimizer<N, B>,
          L: Layer<N, B>
{
    type Context = LayerContext<N, B>;

    fn forward(&mut self, inputs: &B::Tensor, ctx: &mut Self::Context) {
        ctx.set_shape(inputs.shape().get(0), &self.input_shape, &self.layer.output_shape());
        self.layer.forward(&self.backend, &mut ctx.outputs, inputs);
    }

    fn backward(&mut self, deltas: &B::Tensor, ctx: &mut Self::Context) {
        ctx.set_shape(deltas.shape().get(0), &self.input_shape, &self.layer.output_shape());
        self.layer.backward(&self.backend, &mut ctx.deltas, deltas, &ctx.outputs);
    }

    #[inline]
    default fn update(&mut self, _inputs: &B::Tensor, _deltas: &B::Tensor, _ctx: &mut Self::Context) {}
}

impl <N, B, O, L> AbstractLayer<N, B> for LayerImpl<N, B, O, L> 
    where B: Backend<N>,
          O: Optimizer<N, B>,
          L: Layer<N, B> + Optimizable<N, B, O>
{
    fn update(&mut self, inputs: &B::Tensor, deltas: &B::Tensor, _ctx: &mut Self::Context) {
        self.layer.calc_gradients(&self.backend, inputs, deltas);
        self.layer.optimize(&self.backend, &self.optimizer);
    }
}

pub struct Params<N, B: Backend<N>, O: Optimizer<N, B>> {
    pub params: B::Tensor,
    pub grads: B::Tensor,
    pub ctx: O::Context,
}

impl<N, B: Backend<N>, O: Optimizer<N, B>> Params<N, B, O> {
    pub fn new<S: Into<TensorShape>>(shape: S) -> Self {
        let shape = shape.into();

        Self {
            params: B::Tensor::new(shape.clone()),
            grads: B::Tensor::new(shape.clone()),
            ctx: O::Context::new(shape),
        }
    }

    pub fn init_random(&mut self, backend: &B, count: u32) {
        let to = backend.scalar_f32((1.0 / (count as f32)).sqrt());

        backend.fill_random(&mut self.params, backend.scalar_f32(0.0), to);
    }

    pub fn init_zero(&mut self, backend: &B) {
        backend.fill_scalar(&mut self.params, backend.scalar_f32(0.0));
    }
}
