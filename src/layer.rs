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
    fn outputs(&self) -> &B::Tensor;
    fn deltas(&self) -> &B::Tensor;
    fn forward(&mut self, inputs: &B::Tensor);
    fn backward(&mut self, deltas: &B::Tensor);
    fn update(&mut self, inputs: &B::Tensor, deltas: &B::Tensor);
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
    outputs: B::Tensor,
    deltas: B::Tensor,
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

        let mut output_shape = TensorShape::new1d(32);
        let mut deltas_shape = TensorShape::new1d(32);

        output_shape.append(layer.output_shape());
        deltas_shape.append(input_shape);

        Self {
            id: 0,
            name: "x".into(),
            backend: b,
            optimizer: o,
            layer,
            outputs: B::Tensor::new(output_shape),
            deltas: B::Tensor::new(deltas_shape),
            _m: Default::default(),
        }
    }

}

impl <N, B, O, L> AbstractLayer<N, B> for LayerImpl<N, B, O, L> 
    where B: Backend<N>,
          O: Optimizer<N, B>,
          L: Layer<N, B>
{
    #[inline]
    fn outputs(&self) -> &B::Tensor {
        &self.outputs
    }

    #[inline]
    fn deltas(&self) -> &B::Tensor {
        &self.deltas
    }

    fn forward(&mut self, inputs: &B::Tensor) {
        let current_bs = self.outputs.shape().get(0);
        let new_bs = inputs.shape().get(0);

        if new_bs != current_bs {
            let mut new_shape = self.outputs.shape().clone();
            new_shape.set(0, new_bs);
            self.outputs.resize(new_shape);
        }

        self.layer.forward(&self.backend, &mut self.outputs, inputs);
    }

    fn backward(&mut self, deltas: &B::Tensor) {
        let current_bs = self.deltas.shape().get(0);
        let new_bs = deltas.shape().get(0);

        if new_bs != current_bs {
            let mut new_shape = self.deltas.shape().clone();
            new_shape.set(0, new_bs);
            self.deltas.resize(new_shape);
        }

        self.layer.backward(&self.backend, &mut self.deltas, deltas, &self.outputs);
    }

    #[inline]
    default fn update(&mut self, _inputs: &B::Tensor, _deltas: &B::Tensor) {}
}

impl <N, B, O, L> AbstractLayer<N, B> for LayerImpl<N, B, O, L> 
    where B: Backend<N>,
          O: Optimizer<N, B>,
          L: Layer<N, B> + Optimizable<N, B, O>
{
    fn update(&mut self, inputs: &B::Tensor, deltas: &B::Tensor) {
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
