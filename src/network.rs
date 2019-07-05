pub struct Network<N, B, O, L> 
    where B: Backend<N>,
          O: Optimizer<N, B>,
          LS: Loss<N, B>,
          L: AbstractLayer<N, B>
{
    backend: B,
    optimizer: O,
    loss: LS,
    train_ctx: L::Context,
    predict_ctx: RefCell<L::Context>,
    layer: L,
}

impl<N, B, O, L>  Network<N, B, O, L> 
    where B: Backend<N>,
          O: Optimizer<N, B>,
          L: AbstractLayer<N, B>
{
    pub fn new(backend: B, optimizer: O, loss: LS, layer: L) -> Self {
        Self {
            backend,
            optimizer,
            loss,
            layer,
            train_ctx: L::Context::default(),
            predict_ctx: RefCell::new(L::Context::default()),
        }
    }

    fn evaluate(&self, X: &[f32], y: &[f32]) {

    }

    fn predict(&self, X: &[f32], pred: &mut [f32]) {

    }

    fn train(&mut self, X: &[f32], y: &[f32], batch: usize, epochs: usize) {

    }
}