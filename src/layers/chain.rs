use crate::backend::Backend;
use crate::layer::{AbstractLayer, LayerContext};

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

impl<N, B, L, R> ChainContext<N, B, L, R> 
    where B: Backend<N>,
          L: LayerContext<N, B>,
          R: LayerContext<N, B>,
{
    pub fn new(left: L, right: R) -> Self {
        Self {
            left,
            right,
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

pub struct Chain<N, B, L, R> 
    where B: Backend<N>,
          L: AbstractLayer<N, B>,
          R: AbstractLayer<N, B>,
{
    left: L, 
    right: R,
    _m: PhantomData<fn(N, B)>
}

impl<N, B, L, R> Chain<N, B, L, R> 
    where B: Backend<N>,
          L: AbstractLayer<N, B>,
          R: AbstractLayer<N, B>,
{
    pub fn new(left: L, right: R) -> Self {
        Self {
            left,
            right,
            _m: Default::default(),
        }
    }
}

impl<N, B, L, R> AbstractLayer<N, B> for Chain<N, B, L, R> 
    where B: Backend<N>,
          L: AbstractLayer<N, B>,
          R: AbstractLayer<N, B>,
{
    type Context = ChainContext<N, B, L::Context, R::Context>;

    #[inline]
    fn forward(&mut self, inputs: &B::Tensor, ctx: &mut Self::Context) {
        self.left.forward(inputs, &mut ctx.left);
        self.right.forward(ctx.left.outputs(), &mut ctx.right);
    }

    #[inline]
    fn backward(&mut self, deltas: &B::Tensor, ctx: &mut Self::Context) {
        self.right.backward(deltas, &mut ctx.right);
        self.left.backward(ctx.right.deltas(), &mut ctx.left);
    }

    #[inline]
    fn update(&mut self, inputs: &B::Tensor, deltas: &B::Tensor, ctx: &mut Self::Context) {
        self.left.update(inputs, ctx.right.deltas(), &mut ctx.left);
        self.right.update(ctx.left.outputs(), deltas, &mut ctx.right);
    }
}