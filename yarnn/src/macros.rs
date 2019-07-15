
#[macro_export]
macro_rules! sequential_type {
    (input_shape: ($($shape:tt)*), layers: { $($layers:tt)* }) => {
        $crate::sequential_type_impl!( $($layers)* )
    };
}

#[macro_export]
macro_rules! sequential_type_impl {
    ($t:ty {$($tt:tt)*}) => ($t);

    ($t:ty {$($xx:tt)*}, $($tt:tt)*) => {
        $crate::layers::Chain<N, B, O, 
            $t, $crate::sequential_type_impl!($($tt)*)
        >
    };
    ($t:ty) => ($t);
    ($t:ty, $($tt:tt)*) => {
        $crate::layers::Chain<N, B, O, 
            $t, $crate::sequential_type_impl!($($tt)*)
        >
    };
}

#[macro_export]
macro_rules! sequential {
    (input_shape: ($($shape:tt)*), layers: { $($layers:tt)* }) => {{
        let initial_shape = $crate::tensor::TensorShape::from(($($shape)*,));

        $crate::sequential_impl!( initial_shape, $($layers)* )
    }};
}

#[macro_export]
macro_rules! sequential_impl {   
    ($p:expr, $t:ty { $($name:ident : $val:expr),* }) => {{
        #[allow(unused_mut)]
        let mut params = <$t as $crate::layer::LayerExt<N, B, O>>::Config::default();
        $(
            params.$name = core::convert::TryInto::try_into($val).unwrap_or($val);
        )*

        <$t as $crate::layer::LayerExt<N, B, O>>::create(
            $p, params
        )
    }};

    ($p:expr, $t:ty { $($name:ident : $val:expr),* }, $($tt:tt)*) => {{
        #[allow(unused_mut)]
        let mut params = <$t as $crate::layer::LayerExt<N, B, O>>::Config::default();
        $(
            params.$name = core::convert::TryInto::try_into($val).unwrap_or($val);
        )*

        let layer = <$t as $crate::layer::LayerExt<N, B, O>>::create(
            $p, params
        );

        let prev_shape = $crate::layer::Layer::<N, B, O>::output_shape(&layer);
        
        $crate::layers::Chain::new(
            layer, $crate::sequential_impl! { prev_shape, $($tt)* },
        )
    }};

    ($p:expr, $t:ty) => {
        $crate::sequential_impl!{ $p, $t {}}
    };

    ($p:expr, $t:ty, $($tt:tt)*) => {
        $crate::sequential_impl!{ $p, $t {}, $($tt)* }
    };
}

#[macro_export]
macro_rules! sequential_type_ctx {
    (input_shape: ($($shape:tt)*), layers: { $($layers:tt)* }) => {
        $crate::sequential_type_ctx_impl!( $($layers)* )
    };
}

#[macro_export]
macro_rules! sequential_type_ctx_impl {
    ($t:ty {$($xx:tt)*}) => {
        $crate::layer::DefaultLayerContext<N, B>
    };

    ($t:ty {$($xx:tt)*}, $($tt:tt)*) => {
        $crate::layers::ChainContext<N, B,
            $crate::layer::DefaultLayerContext<N, B>,
            $crate::sequential_type_ctx_impl!($($tt)*)
        >
    };

    ($t:ty) => {
        $crate::sequential_type_ctx_impl!( $t {} )
    };

    ($t:ty, $($tt:tt)*) => {
        $crate::sequential_type_ctx_impl!( $t {}, $($tt)* )
    };
}

#[macro_export]
macro_rules! model_impl {
    ($name:ident <$trait:path> ($($init:tt)*) { $($tt:tt)* }) => {
        mod ctx {
            #[allow(unused_imports)]
            use super::*;
            pub type $name<N, B> = $crate::sequential_type_ctx!($($tt)*);
        }

        pub struct $name <N, B, O>
            where B: $crate::backend::Backend<N> + $trait,
                  O: $crate::optimizer::Optimizer<N, B>
        {
            inner: $crate::sequential_type!($($tt)*),
            _m: core::marker::PhantomData<fn(N, B, O)>,
        }

        impl<N, B, O> $name<N, B, O> 
            where B: $crate::backend::Backend<N> + $trait,
                  O: $crate::optimizer::Optimizer<N, B>
        {
            #[allow(dead_code)]
            pub fn new($($init)*) -> Self {
                #[allow(unused_imports)]
                use $crate::backend::PaddingKind::*;

                #[allow(unused_imports)]
                use $crate::backend::PoolingKind::*;
                
                Self {
                    inner: $crate::sequential!($($tt)*),
                    _m: Default::default(),
                }
            }
        }

        // impl<N, B, O> core::fmt::Display for $name<N, B, O> 
        //     where B: $crate::backend::Backend<N> + $trait,
        //           O: $crate::optimizer::Optimizer<N, B>
        // {
        //     fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        //         writeln!(f, "{} {{", stringify!($name))?;
        //         write!(f, "{}", self.inner)?;
        //         writeln!(f, "}}")?;

        //         Ok(())
        //     }
        // }

        impl<N, B, O> $crate::layer::Layer<N, B, O> for $name<N, B, O> 
            where B: $crate::backend::Backend<N> + $trait,
                  O: $crate::optimizer::Optimizer<N, B>
        {
            type Context = ctx::$name<N, B>;

            #[inline]
            fn name(&self) -> &str {
                stringify!($name)
            }

            #[inline]
            fn init(&mut self, backend: &B) {
                self.inner.init(backend);
            }

            #[inline]
            fn param_count(&self) -> usize {
                self.inner.param_count()
            } 

            #[inline]
            fn input_shape(&self) -> $crate::tensor::TensorShape {
                self.inner.input_shape()
            }

            #[inline]
            fn output_shape(&self) -> $crate::tensor::TensorShape {
                self.inner.output_shape()
            }

            #[inline]
            fn forward(&self, backend: &B, inputs: &B::Tensor, ctx: &mut Self::Context) {
                self.inner.forward(backend, inputs, ctx);
            }

            #[inline]
            fn backward(&mut self, backend: &B, deltas: &B::Tensor, inputs: &B::Tensor, ctx: &mut Self::Context) {
                self.inner.backward(backend, deltas, inputs, ctx);
            }

            #[inline]
            fn calc_gradients(&mut self, backend: &B, deltas: &B::Tensor, inputs: &B::Tensor, ctx: &mut Self::Context) {
                self.inner.calc_gradients(backend, deltas, inputs, ctx);
            }

            #[inline]
            fn optimize(&mut self, backend: &B, optimizer: &O) {
                self.inner.optimize(backend, optimizer);
            }

            fn fmt(&self, f: &mut core::fmt::Formatter, padding: usize) -> core::fmt::Result {
                writeln!(f, "{}{}[{}] {{",  "", self.name(), self.param_count())?;
                self.inner.fmt(f, padding + 2)?;
                write!(f, "}}")?;
                
                Ok(())
            }
        }

        impl<N, B, O> core::fmt::Display for $name<N, B, O> 
            where B: $crate::backend::Backend<N> + $trait,
                  O: $crate::optimizer::Optimizer<N, B>
        {
            #[inline]
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                <Self as $crate::layer::Layer<_, _, _>>::fmt(self, f, 0)
            }
        }
    };
}

#[macro_export]
macro_rules! model {
    ($name:ident ($($init:tt)*) { $($tt:tt)* }) => {
        mod tmp {
            pub trait BackendDefault<N> = $crate::backend::BackendReLu<N> 
                  + $crate::backend::BackendBias<N>
                  + $crate::backend::BackendScale<N>
                  + $crate::backend::BackendSigmoid<N>
                  + $crate::backend::BackendSoftmax<N>
                  + $crate::backend::BackendGemm<N>
                  + $crate::backend::BackendConv2d<N>
                  + $crate::backend::BackendMaxPool2d<N>
                  + $crate::backend::BackendAvgPool2d<N>;
        }
        $crate::model_impl!($name <tmp::BackendDefault<N>> ($($init)*) { $($tt)* });
    };
    ($name:ident <$trait:path> ($($init:tt)*) { $($tt:tt)* }) => {
        $crate::model_impl!($name <$trait> ($($init)*) { $($tt)* });
    };
}