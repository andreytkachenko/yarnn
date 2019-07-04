// use crate::layer::AbstractLayer;
// use crate::backend::Backend;
// use crate::optimizer::Optimizer;
// use std::marker::PhantomData;


// pub struct SequentialLayer<N, B, O, L> 
//     where B: Backend<N>,
//           O: Optimizer<N, B>,
//           L: for<'a> IntoChainIter<'a, N, B> 
// {
//     layers: L,
//     backend: B,
//     optimizer: O,
//     _m: PhantomData<N>
// }

// impl<N, B, O, L>  SequentialLayer<N, B, O, L> 
//     where B: Backend<N>,
//           O: Optimizer<N, B>,
//           L: for<'a> IntoChainIter<'a, N, B> 
// {
//     pub fn new(backend: B, optimizer: O, layers: L) -> Self {
//         Self {
//             backend,
//             optimizer,
//             layers,
//             _m: Default::default()
//         }
//     }
// }


// #[derive(Debug)]
// pub struct Chain<L, R>(pub L, pub R);

// #[derive(Debug)]
// pub struct Leaf<L>(pub L);

// impl<'a, N, B, T> IntoChainIter<'a, N, B> for Leaf<T>
//     where   N: 'a, 
//             B: Backend<N> + 'a,
//             T: AbstractLayer<N, B> + 'a
// {
//     type Iter = std::iter::Once<&'a dyn AbstractLayer<N, B>>;
    
//     fn iter(&'a mut self) -> Self::Iter {
//         std::iter::once(&self.0)
//     }
// }

// pub trait IntoChainIter<'a, N: 'a, B: Backend<N> + 'a> {
//     type Iter: DoubleEndedIterator<Item = &'a dyn AbstractLayer<N, B>> + 'a;
    
//     fn iter(&'a mut self) -> Self::Iter;
// }

// impl<'a, N, B, L, R> IntoChainIter<'a, N, B> for Chain<L, R>
//     where N: 'a,
//           B: Backend<N> + 'a,
//           L: IntoChainIter<'a, N, B>,
//           R: IntoChainIter<'a, N, B>
// {
//     type Iter = std::iter::Chain<L::Iter, R::Iter>;
    
//     fn iter(&'a mut self) -> Self::Iter {
//         self.0.iter().chain(self.1.iter())
//     }
// }

// #[macro_export]
// macro_rules! sequential_impl {
//     ($b:expr, $o:expr, $t:ty { $($name:ident: $e:expr),* }) => {{
//         let layer: LayerImpl<_, _, _, $t> = LayerImpl::new((123,).into(), $b.clone(), $o.clone(), {
//             let mut cfg = <$t as $crate::layer::Layer<_, _>>::Config::default();
//             $(
//                 cfg.$name = ($e).into();
//             )*
//             cfg
//         });

//         Leaf(layer)
//     }};
//     ($b:expr, $o:expr, $t:ty { $($tt:tt),* },) => {
//         sequential_impl!{ $b, $o, $t {$($tt)*} }
//     };
//     ($b:expr, $o:expr, $t:ty,) => {
//         sequential_impl!{ $b, $o, $t { } }
//     };
//     ($b:expr, $o:expr, $t:ty) => {
//         sequential_impl!{ $b, $o, $t { } }
//     };
//     ($b:expr, $o:expr, $t:ty { $($name:ident: $e:expr),*}, $($tail:tt)*) => {{
//         let layer: LayerImpl<_, _, _, $t> = LayerImpl::new((123,).into(), $b.clone(), $o.clone(), {
//             let mut cfg = <$t as $crate::layer::Layer<_, _>>::Config::default();
//             $(
//                 cfg.$name = $e;
//             )*
//             cfg
//         });

//         Chain(Leaf(layer), sequential_impl!{$b, $o, $($tail)*})
//     }};
// }

// #[macro_export]
// macro_rules! sequential {
//     ($b:expr, $o:expr, $($t:tt)*) => (SequentialLayer::new($b, $o, sequential_impl!{$b, $o, $($t)*}))
// }
