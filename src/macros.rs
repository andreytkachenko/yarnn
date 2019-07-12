
#[macro_export]
macro_rules! sequential_type {
    (input_shape: ($($shape:tt)*), layers: { $($layers:tt)* }) => {
        $crate::sequential_type_impl!( $($layers)* )
    };
}

#[macro_export]
macro_rules! sequential_type_impl {
    ($t:ty {$($tt:tt)*}) => {
        $crate::layer::LayerImpl<N, B, O, $t>
    };
    ($t:ty {$($xx:tt)*}, $($tt:tt)*) => {
        $crate::layers::Chain<N, B, O, 
            $crate::layer::LayerImpl<N, B, O, $t>,
            $crate::sequential_type_impl!($($tt)*)
        >
    };
    ($t:ty) => {
        $crate::layer::LayerImpl<N, B, O, $t>
    };
    ($t:ty, $($tt:tt)*) => {
        $crate::layers::Chain<N, B, O, 
            $crate::layer::LayerImpl<N, B, O, $t>,
            $crate::sequential_type_impl!($($tt)*)
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
        #[allow(unused_imports)]
        use std::convert::TryInto;

        #[allow(unused_mut)]
        let mut params = <$t as $crate::layer::Layer<_, _>>::Config::default();
        $(
            params.$name = ($val).try_into().unwrap_or($val);
        )*

        $crate::layer::LayerImpl::new(<$t as $crate::layer::Layer<_, _>>::create(
            $p, params
        ))
    }};

    ($p:expr, $t:ty { $($name:ident : $val:expr),* }, $($tt:tt)*) => {{
        #[allow(unused_imports)]
        use std::convert::TryInto;

        #[allow(unused_mut)]
        let mut params = <$t as $crate::layer::Layer<_, _>>::Config::default();
        $(
            params.$name = ($val).try_into().unwrap_or($val);;
        )*

        let layer = $crate::layer::LayerImpl::new(<$t as $crate::layer::Layer<_, _>>::create(
            $p, params
        ));

        let prev_shape = layer.layer.output_shape();
        
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
        $crate::layer::CommonLayerContext<N, B>
    };

    ($t:ty {$($xx:tt)*}, $($tt:tt)*) => {
        $crate::layers::ChainContext<N, B,
            $crate::layer::CommonLayerContext<N, B>,
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
            _m: std::marker::PhantomData<fn(N, B, O)>,
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

        impl<N, B, O> std::fmt::Display for $name<N, B, O> 
            where B: $crate::backend::Backend<N> + $trait,
                  O: $crate::optimizer::Optimizer<N, B>
        {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                writeln!(f, "{} {{", stringify!($name))?;
                write!(f, "{}", self.inner)?;
                writeln!(f, "}}")?;

                Ok(())
            }
        }

        impl<N, B, O> $crate::layer::AbstractLayer<N, B, O> for $name<N, B, O> 
            where B: $crate::backend::Backend<N> + $trait,
                  O: $crate::optimizer::Optimizer<N, B>
        {
            type Context = ctx::$name<N, B>;

            #[inline]
            fn forward(&mut self, backend: &B, inputs: &B::Tensor, ctx: &mut Self::Context) {
                $crate::layer::AbstractLayer::forward(&mut self.inner, backend, inputs, ctx)
            }

            #[inline]
            fn backward(&mut self, backend: &B, deltas: &B::Tensor, inputs: &B::Tensor, ctx: &mut Self::Context) {
                $crate::layer::AbstractLayer::backward(&mut self.inner, backend, deltas, inputs, ctx);
            }

            #[inline]
            fn update(&mut self, backend: &B, optimizer: &O, inputs: &B::Tensor, deltas: &B::Tensor, ctx: &mut Self::Context) {
                $crate::layer::AbstractLayer::update(&mut self.inner, backend, optimizer, inputs, deltas, ctx);
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