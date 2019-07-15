
pub trait Model<N, B, O> 
    where B: Backend<N>,
          O: Optimizer<N, B>
{
    fn init(&mut self, backend: &B);
    fn predict(&mut self, backend: &B, x: &B::Tensor);
    fn evaluate(&mut self, backend: &B, x: &B::Tensor, y: &B::Tensor) -> ConfusionMatrix;
    fn train(&mut self, backend: &B, optimizer: &O, x: &B::Tensor, y: &B::Tensor);
}

pub struct DefaultModel<N, B, O, L> 
    where B: Backend<N>,
          O: Optimizer<N, B>,
          L: AbstractLayer<N, B, O>
{
    inner: L,
    train_ctx: L::Context,
    evaluate_ctx: L::Context,
}

impl<N, B, O, L> Model<N, B, O> for DefaultModel<N, B, O, L> 

{
    fn init(&mut self, backend: &B) {

    }

    fn predict(&mut self, backend: &B, x: &B::Tensor) -> &B::Tensor {
        self.inner.forward(backend, x, self.evaluate_ctx);

        self.evaluate_ctx.outputs()
    }

    fn evaluate(&mut self, backend: &B, x: &B::Tensor, y: &B::Tensor) -> ConfusionMatrix {
        
    }

    fn train(&mut self, backend: &B, optimizer: &O, x: &B::Tensor, y: &B::Tensor) {
        
    }
}