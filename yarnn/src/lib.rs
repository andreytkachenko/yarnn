#![feature(specialization, trait_alias)]
#![recursion_limit="128"]

pub mod layer;
pub mod layers;

pub mod optimizer;
pub mod optimizers;

pub mod backend;
pub mod native;

pub mod loss;
pub mod losses;

pub mod tensor;
pub mod params;

#[macro_use]
mod macros;

pub mod prelude {
    pub use super::backend::*;
    pub use super::loss::*;
    pub use super::tensor::*;
    pub use super::layer::*;
}