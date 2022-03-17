use crate::backend::Backend;
use crate::optimizer::Optimizer;
use crate::tensor::{Tensor, TensorShape};

// use core::marker::PhantomData;

pub trait Layer<N, B, O>
where
    B: Backend<N>,
    O: Optimizer<N, B>,
{
    type Context: LayerContext<N, B>;

    fn name(&self) -> &str;
    fn param_count(&self) -> usize {
        0
    }

    #[inline]
    fn init(&mut self, _backend: &B) {}

    fn input_shape(&self) -> TensorShape;

    #[inline]
    fn output_shape(&self) -> TensorShape {
        self.input_shape()
    }

    fn forward(&self, backend: &B, x: &B::Tensor, ctx: &mut Self::Context);
    fn backward(&mut self, backend: &B, dy: &B::Tensor, x: &B::Tensor, ctx: &mut Self::Context);

    #[inline]
    fn calc_gradients(
        &mut self,
        _backend: &B,
        _dy: &B::Tensor,
        _x: &B::Tensor,
        _ctx: &mut Self::Context,
    ) {
    }

    #[inline]
    fn optimize(&mut self, _backend: &B, _optimizer: &O) {}

    fn fmt(&self, f: &mut core::fmt::Formatter, padding: usize) -> core::fmt::Result {
        writeln!(
            f,
            "{}{} -> {}[{}] -> {}",
            "".repeat(padding),
            self.input_shape(),
            self.name(),
            self.param_count(),
            self.output_shape()
        )?;

        Ok(())
    }
}

pub trait LayerExt<N, B, O>: Layer<N, B, O>
where
    B: Backend<N>,
    O: Optimizer<N, B>,
{
    type Config: Default;

    fn create(input_shape: TensorShape, cfg: Self::Config) -> Self;

    #[inline]
    fn add_layer<L: LayerExt<N, B, O>>(
        self,
        cfg: L::Config,
    ) -> crate::layers::Chain<N, B, O, Self, L>
    where
        Self: Sized,
    {
        let shape = self.output_shape();

        crate::layers::Chain::new(self, L::create(shape, cfg))
    }
}

pub trait LayerContext<N, B: Backend<N>>: Default {
    fn outputs(&self) -> &B::Tensor;
    fn deltas(&self) -> &B::Tensor;
}

pub struct DefaultLayerContext<N, B>
where
    B: Backend<N>,
{
    pub outputs: B::Tensor,
    pub deltas: B::Tensor,
}

impl<N, B> Default for DefaultLayerContext<N, B>
where
    B: Backend<N>,
{
    fn default() -> Self {
        Self {
            outputs: B::Tensor::new(()),
            deltas: B::Tensor::new(()),
        }
    }
}

impl<N, B> DefaultLayerContext<N, B>
where
    B: Backend<N>,
{
    pub fn update_deltas_shape(&mut self, bs: u32, input_shape: &TensorShape) {
        let mut new_deltas_shape = TensorShape::new1d(bs);
        new_deltas_shape.append(input_shape.clone());

        if self.deltas.shape() != &new_deltas_shape {
            self.deltas.resize(new_deltas_shape.clone());
        }
    }

    pub fn update_outputs_shape(&mut self, bs: u32, output_shape: &TensorShape) {
        let mut new_output_shape = TensorShape::new1d(bs);

        new_output_shape.append(output_shape.clone());

        if self.outputs.shape() != &new_output_shape {
            self.outputs.resize(new_output_shape);
        }
    }
}

impl<N, B> LayerContext<N, B> for DefaultLayerContext<N, B>
where
    B: Backend<N>,
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
