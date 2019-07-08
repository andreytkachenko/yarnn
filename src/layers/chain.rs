use crate::backend::Backend;
use crate::layer::{AbstractLayer, LayerContext};
use crate::optimizer::Optimizer;

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
          L: AbstractLayer<N, B, O>,
          R: AbstractLayer<N, B, O>,
{
    left: L, 
    right: R,
    _m: PhantomData<fn(N, B, O)>
}

impl<N, B, O, L, R> Chain<N, B, O, L, R> 
    where B: Backend<N>,
          O: Optimizer<N, B>,
          L: AbstractLayer<N, B, O>,
          R: AbstractLayer<N, B, O>,
{
    pub fn new(left: L, right: R) -> Self {
        Self {
            left,
            right,
            _m: Default::default(),
        }
    }
}

impl<N, B, O, L, R> AbstractLayer<N, B, O> for Chain<N, B, O, L, R> 
    where B: Backend<N>,
          O: Optimizer<N, B>,
          L: AbstractLayer<N, B, O>,
          R: AbstractLayer<N, B, O>,
{
    type Context = ChainContext<N, B, L::Context, R::Context>;

    #[inline]
    fn forward(&mut self, backend: &B, inputs: &B::Tensor, ctx: &mut Self::Context) {
        self.left.forward(backend, inputs, &mut ctx.left);
        self.right.forward(backend, ctx.left.outputs(), &mut ctx.right);
    }

    #[inline]
    fn backward(&mut self, backend: &B, deltas: &B::Tensor, ctx: &mut Self::Context) {
        self.right.backward(backend, deltas, &mut ctx.right);
        self.left.backward(backend, ctx.right.deltas(), &mut ctx.left);
    }

    #[inline]
    fn update(&mut self, backend: &B, optimizer: &O, inputs: &B::Tensor, deltas: &B::Tensor, ctx: &mut Self::Context) {
        self.left.update(backend, optimizer, inputs, ctx.right.deltas(), &mut ctx.left);
        self.right.update(backend, optimizer, ctx.left.outputs(), deltas, &mut ctx.right);
    }
}