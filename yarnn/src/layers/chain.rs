use crate::backend::Backend;
use crate::layer::{Layer, LayerContext};
use crate::optimizer::Optimizer;
use crate::tensor::TensorShape;

use core::marker::PhantomData;

pub struct ChainContext<N, B, L, R> 
    where B: Backend<N>,
          L: LayerContext<N, B>,
          R: LayerContext<N, B>,
{
    left: L, 
    right: R,
    _m: PhantomData<fn(N, B)>
}

impl<N, B, L, R> Default for ChainContext<N, B, L, R> 
    where B: Backend<N>,
          L: LayerContext<N, B>,
          R: LayerContext<N, B>,
{
    fn default() -> Self {
        Self {
            left: Default::default(),
            right: Default::default(),
            _m: Default::default(),
        }
    }
}

impl<N, B, L, R> LayerContext<N, B> for ChainContext<N, B, L, R>
    where B: Backend<N>,
          L: LayerContext<N, B>,
          R: LayerContext<N, B>,
{
    #[inline]
    fn outputs(&self) -> &B::Tensor {
        self.right.outputs()
    }

    #[inline]
    fn deltas(&self) -> &B::Tensor {
        self.left.deltas()
    }
} 

pub struct Chain<N, B, O, L, R> 
    where B: Backend<N>,
          O: Optimizer<N, B>,
          L: Layer<N, B, O>,
          R: Layer<N, B, O>,
{
    left: L, 
    right: R,
    _m: PhantomData<fn(N, B, O)>
}

impl<N, B, O, L, R> Chain<N, B, O, L, R> 
    where B: Backend<N>,
          O: Optimizer<N, B>,
          L: Layer<N, B, O>,
          R: Layer<N, B, O>,
{
    pub fn new(left: L, right: R) -> Self {
        Self {
            left,
            right,
            _m: Default::default(),
        }
    }
}

// impl<N, B, O, L, R> core::fmt::Display for Chain<N, B, O, L, R> 
//     where B: Backend<N>,
//           O: Optimizer<N, B>,
//           L: Layer<N, B, O>,
//           R: Layer<N, B, O>,
// {
//     fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
//         self.left.fmt(f)?;
//         self.right.fmt(f)?;

//         Ok(())
//     }
// }

impl<N, B, O, L, R> Layer<N, B, O> for Chain<N, B, O, L, R> 
    where B: Backend<N>,
          O: Optimizer<N, B>,
          L: Layer<N, B, O>,
          R: Layer<N, B, O>,
{
    type Context = ChainContext<N, B, L::Context, R::Context>;

    #[inline]
    fn name(&self) -> &str {
        "Chain Layer"
    }

    #[inline]
    fn param_count(&self) -> usize {
        self.left.param_count() + self.right.param_count()
    } 

    #[inline]
    fn init(&mut self, backend: &B) {
        self.left.init(backend);
        self.right.init(backend);
    }

    #[inline]
    fn input_shape(&self) -> TensorShape {
        self.left.input_shape()
    }

    #[inline]
    fn output_shape(&self) -> TensorShape {
        self.right.output_shape()
    }

    #[inline]
    fn forward(&self, backend: &B, inputs: &B::Tensor, ctx: &mut Self::Context) {
        self.left.forward(backend, inputs, &mut ctx.left);
        self.right.forward(backend, ctx.left.outputs(), &mut ctx.right);
    }

    #[inline]
    fn backward(&mut self, backend: &B, deltas: &B::Tensor, inputs: &B::Tensor, ctx: &mut Self::Context) {
        self.right.backward(backend, deltas, ctx.left.outputs(), &mut ctx.right);
        self.left.backward(backend, ctx.right.deltas(), inputs, &mut ctx.left);
    }

    #[inline]
    fn calc_gradients(&mut self, backend: &B, deltas: &B::Tensor, inputs: &B::Tensor, ctx: &mut Self::Context) {
        self.left.calc_gradients(backend, ctx.right.deltas(), inputs, &mut ctx.left);
        self.right.calc_gradients(backend, deltas, ctx.left.outputs(), &mut ctx.right);
    }

    #[inline]
    fn optimize(&mut self, backend: &B, optimizer: &O) {
        self.left.optimize(backend, optimizer);
        self.right.optimize(backend, optimizer);
    }

    fn fmt(&self, f: &mut core::fmt::Formatter, padding: usize) -> core::fmt::Result {
        self.left.fmt(f, padding)?;
        self.right.fmt(f, padding)?;
        
        Ok(())
    }
}