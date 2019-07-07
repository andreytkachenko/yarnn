use crate::tensor::{Tensor};


pub trait Backend<N> {
    type Tensor: Tensor<N>;

    fn store_tensor_f32(&self, t: &Self::Tensor, data: &mut [f32]);
    fn load_tensor_u8(&self, t: &mut Self::Tensor, data: &[u8]);
    fn load_tensor_f32(&self, t: &mut Self::Tensor, data: &[f32]);
    fn scalar_f32(&self, val: f32) -> N;
    fn fill_scalar(&self, t: &mut Self::Tensor, scalar: N);
    fn fill_random(&self, t: &mut Self::Tensor, from: N, to: N);
    fn print_tensor(&self, t: &Self::Tensor);
}

impl <'a, N, T: Backend<N>> Backend<N> for &'a T {
    type Tensor = T::Tensor;

    #[inline]
    fn store_tensor_f32(&self, dst: &Self::Tensor, slice: &mut [f32]) {
         (**self).store_tensor_f32(dst, slice)
    }

    #[inline]
    fn load_tensor_u8(&self, dst: &mut Self::Tensor, slice: &[u8]) {
         (**self).load_tensor_u8(dst, slice)
    }

    #[inline]
    fn load_tensor_f32(&self, dst: &mut Self::Tensor, slice: &[f32]) {
         (**self).load_tensor_f32(dst, slice)
    }

    #[inline]
    fn scalar_f32(&self, val: f32) -> N {
        (**self).scalar_f32(val)
    }

    #[inline]
    fn fill_scalar(&self, t: &mut Self::Tensor, scalar: N) {
        (**self).fill_scalar(t, scalar)
    }

    #[inline]
    fn fill_random(&self, t: &mut Self::Tensor, from: N, to: N) {
        (**self).fill_random(t, from, to)
    }

    #[inline]
    fn print_tensor(&self, t: &Self::Tensor) {
        (**self).print_tensor(t)
    }
}

pub trait BackendGemm<N>: Backend<N> {
    fn matmul(&self, dst: &mut Self::Tensor, a: &Self::Tensor, b: &Self::Tensor);
    fn matmul_nt(&self, dst: &mut Self::Tensor, a: &Self::Tensor, b: &Self::Tensor);
    fn matmul_tn(&self, dst: &mut Self::Tensor, a: &Self::Tensor, b: &Self::Tensor);
    fn matmul_tt(&self, dst: &mut Self::Tensor, a: &Self::Tensor, b: &Self::Tensor);
}

impl <'a, N, T: BackendGemm<N>> BackendGemm<N> for &'a T {
    #[inline]
    fn matmul(&self, dst: &mut Self::Tensor, a: &Self::Tensor, b: &Self::Tensor) {
        (**self).matmul(dst, a, b)
    }

    #[inline]
    fn matmul_nt(&self, dst: &mut Self::Tensor, a: &Self::Tensor, b: &Self::Tensor) {
        (**self).matmul_nt(dst, a, b)
    }

    #[inline]
    fn matmul_tn(&self, dst: &mut Self::Tensor, a: &Self::Tensor, b: &Self::Tensor) {
        (**self).matmul_tn(dst, a, b)
    }

    #[inline]
    fn matmul_tt(&self, dst: &mut Self::Tensor, a: &Self::Tensor, b: &Self::Tensor) {
        (**self).matmul_tt(dst, a, b)
    }
}

pub trait BackendBias<N>: Backend<N> {
    fn bias_add(&self, dst: &mut Self::Tensor, bias: &Self::Tensor);
    fn bias_grad(&self, bias: &mut Self::Tensor, inputs: &Self::Tensor);
}

impl <'a, N, T: BackendBias<N>> BackendBias<N> for &'a T {
    #[inline]
    fn bias_add(&self, dst: &mut Self::Tensor, bias: &Self::Tensor) {
        (**self).bias_add(dst, bias)
    }

    #[inline]
    fn bias_grad(&self, bias: &mut Self::Tensor, inputs: &Self::Tensor) {
        (**self).bias_grad(bias, inputs)
    }
}

pub trait BackendSigmoid<N>: Backend<N> {
    fn sigmoid(&self, dst: &mut Self::Tensor, data: &Self::Tensor);
    fn sigmoid_grad(&self, dst: &mut Self::Tensor, z: &Self::Tensor, d: &Self::Tensor);
}

impl <'a, N, T: BackendSigmoid<N>> BackendSigmoid<N> for &'a T {
    #[inline]
    fn sigmoid(&self, dst: &mut Self::Tensor, data: &Self::Tensor) {
        (**self).sigmoid(dst, data)
    }

    #[inline]
    fn sigmoid_grad(&self, dst: &mut Self::Tensor, z: &Self::Tensor, d: &Self::Tensor) {
        (**self).sigmoid_grad(dst, z, d)
    }
}


pub trait BackendReLu<N>: Backend<N> {
    fn relu(&self, dst: &mut Self::Tensor, data: &Self::Tensor);
    fn relu_grad(&self, dst: &mut Self::Tensor, z: &Self::Tensor, d: &Self::Tensor);
}

impl <'a, N, T: BackendReLu<N>> BackendReLu<N> for &'a T {
    #[inline]
    fn relu(&self, dst: &mut Self::Tensor, data: &Self::Tensor) {
        (**self).relu(dst, data)
    }

    #[inline]
    fn relu_grad(&self, dst: &mut Self::Tensor, z: &Self::Tensor, d: &Self::Tensor) {
        (**self).relu_grad(dst, z, d)
    }
}

pub trait BackendScale<N>: Backend<N> {
    fn scale(&self, dst: &mut Self::Tensor, scale: N);
}

impl <'a, N, T: BackendScale<N>> BackendScale<N> for &'a T {
    #[inline]
    fn scale(&self, dst: &mut Self::Tensor, scale: N) {
        (**self).scale(dst, scale)
    }
}

pub trait BackendMse<N>: Backend<N> {
    fn scaled_square_diff(&self, dst: &mut Self::Tensor, a: &Self::Tensor, b: &Self::Tensor, scale: N);
    fn scaled_diff(&self, dst: &mut Self::Tensor, a: &Self::Tensor, b: &Self::Tensor, scale: N);
}

impl <'a, N, T: BackendMse<N>> BackendMse<N> for &'a T {
    #[inline]
    fn scaled_square_diff(&self, dst: &mut Self::Tensor, a: &Self::Tensor, b: &Self::Tensor, scale: N) {
        (**self).scaled_square_diff(dst, a, b, scale)
    }

    #[inline]
    fn scaled_diff(&self, dst: &mut Self::Tensor, a: &Self::Tensor, b: &Self::Tensor, scale: N) {
        (**self).scaled_diff(dst, a, b, scale)
    }
}

pub trait BackendAxpy<N>: Backend<N> {
    fn axpy(&self, dst: &mut Self::Tensor, scale: N, a: &Self::Tensor);
}

impl <'a, N, T: BackendAxpy<N>> BackendAxpy<N> for &'a T {
    #[inline]
    fn axpy(&self, dst: &mut Self::Tensor, scale: N, a: &Self::Tensor) {
        (**self).axpy(dst, scale, a)
    }
}

pub trait BackendAxpys<N>: Backend<N> {
    fn axpys(&self, dst: &mut Self::Tensor, scale: N, a: &Self::Tensor);
}

impl <'a, N, T: BackendAxpys<N>> BackendAxpys<N> for &'a T {
    #[inline]
    fn axpys(&self, dst: &mut Self::Tensor, scale: N, a: &Self::Tensor) {
        (**self).axpys(dst, scale, a)
    }
}

pub trait BackendAdd<N>: Backend<N> {
    fn add(&self, dst: &mut Self::Tensor, a: &Self::Tensor);
}

impl <'a, N, T: BackendAdd<N>> BackendAdd<N> for &'a T {
    #[inline]
    fn add(&self, dst: &mut Self::Tensor, a: &Self::Tensor) {
        (**self).add(dst, a)
    }
}

pub trait BackendSub<N>: Backend<N> {
    fn sub(&self, dst: &mut Self::Tensor, a: &Self::Tensor, b: &Self::Tensor);
}

impl <'a, N, T: BackendSub<N>> BackendSub<N> for &'a T {
    #[inline]
    fn sub(&self, dst: &mut Self::Tensor, a: &Self::Tensor, b: &Self::Tensor) {
        (**self).sub(dst, a, b)
    }
}

pub trait BackendMul<N>: Backend<N> {
    fn mul(&self, dst: &mut Self::Tensor, a: &Self::Tensor);
}

impl <'a, N, T: BackendMul<N>> BackendMul<N> for &'a T {
    #[inline]
    fn mul(&self, dst: &mut Self::Tensor, a: &Self::Tensor) {
        (**self).mul(dst, a)
    }
}

pub trait BackendCopy<N>: Backend<N> {
    fn copy(&self, dst: &mut Self::Tensor, a: &Self::Tensor);
}

impl <'a, N, T: BackendCopy<N>> BackendCopy<N> for &'a T {
    #[inline]
    fn copy(&self, dst: &mut Self::Tensor, a: &Self::Tensor) {
        (**self).copy(dst, a)
    }
}

pub trait BackendMaximum<N>: Backend<N> {
    fn maximum(&self, dst: &mut Self::Tensor, a: &Self::Tensor);
}

impl <'a, N, T: BackendMaximum<N>> BackendMaximum<N> for &'a T {
    #[inline]
    fn maximum(&self, dst: &mut Self::Tensor, a: &Self::Tensor) {
        (**self).maximum(dst, a)
    }
}

pub trait BackendAdam<N>: BackendScale<N> + BackendAxpy<N> + BackendAxpys<N> + BackendMaximum<N> {
    fn adam_p(&self, dst: &mut Self::Tensor, lr: N, moms: &Self::Tensor, vels: &Self::Tensor, eps: N);
}

impl <'a, N, T: BackendAdam<N>> BackendAdam<N> for &'a T {
    #[inline]
    fn adam_p(&self, dst: &mut Self::Tensor, lr: N, moms: &Self::Tensor, vels: &Self::Tensor, eps: N) {
        (**self).adam_p(dst, lr, moms, vels, eps)
    }
}

pub trait BackendSoftmax<N>: BackendCopy<N> {
    fn softmax(&self, y: &mut Self::Tensor, x: &Self::Tensor);
}

impl <'a, N, T: BackendSoftmax<N>> BackendSoftmax<N> for &'a T {
    #[inline]
    fn softmax(&self, y: &mut Self::Tensor, x: &Self::Tensor) {
        (**self).softmax(y, x)
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum PaddingKind {
    Valid,
    Same,
    Full,
}

#[derive(Clone, PartialEq, Debug)]
pub struct Conv2dInfo {
    pub padding: PaddingKind, 
    pub strides: (u32, u32),
    pub kernel: (u32, u32),
}

pub trait BackendConv2d<N>: Backend<N> {
    type Context;

    fn conv2d_forward(&self, y: &mut Self::Tensor, x: &Self::Tensor, filter: &Self::Tensor, conv_info: &Conv2dInfo);
    fn conv2d_backward_input(&self, dx: &mut Self::Tensor, dy: &Self::Tensor, filter: &Self::Tensor, conv_info: &Conv2dInfo);
    fn conv2d_backward_filter(&self, dw: &mut Self::Tensor, x: &Self::Tensor, dy: &Self::Tensor, conv_info: &Conv2dInfo);
}

impl <'a, N, T: BackendConv2d<N>> BackendConv2d<N> for &'a T {
    type Context = ();
    
    #[inline]
    fn conv2d_forward(&self, y: &mut Self::Tensor, x: &Self::Tensor, filters: &Self::Tensor, conv_info: &Conv2dInfo) {
        (**self).conv2d_forward(y, x, filters, conv_info)
    }
    
    #[inline]
    fn conv2d_backward_input(&self, dx: &mut Self::Tensor, dy: &Self::Tensor, filters: &Self::Tensor, conv_info: &Conv2dInfo) {
        (**self).conv2d_backward_input(dx, dy, filters, conv_info)
    }
    
    #[inline]
    fn conv2d_backward_filter(&self, dw: &mut Self::Tensor, x: &Self::Tensor, dy: &Self::Tensor, conv_info: &Conv2dInfo) {
        (**self).conv2d_forward(dw, x, dy, conv_info)
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum PoolingKind {
    Max,
    Avg,
}

pub trait BackendMaxPool2d<N>: Backend<N> {
    fn max_pool2d(&self, y: &mut Self::Tensor, x: &Self::Tensor, conv_info: &Conv2dInfo);
    fn max_pool2d_backprop(&self, dx: &mut Self::Tensor, dy: &Self::Tensor, x: &Self::Tensor, conv_info: &Conv2dInfo);
}

impl <'a, N, T: BackendMaxPool2d<N>> BackendMaxPool2d<N> for &'a T {
    #[inline]
    fn max_pool2d(&self, y: &mut Self::Tensor, x: &Self::Tensor, conv_info: &Conv2dInfo) {
        (**self).max_pool2d(y, x, conv_info)
    }
    
    #[inline]
    fn max_pool2d_backprop(&self, dx: &mut Self::Tensor, dy: &Self::Tensor, x: &Self::Tensor, conv_info: &Conv2dInfo) {
        (**self).max_pool2d_backprop(dx, dy, x, conv_info)
    }
}

pub trait BackendAvgPool2d<N>: Backend<N> {
    fn avg_pool2d(&self, y: &mut Self::Tensor, x: &Self::Tensor, conv_info: &Conv2dInfo);
    fn avg_pool2d_backprop(&self, dx: &mut Self::Tensor, dy: &Self::Tensor, x: &Self::Tensor, conv_info: &Conv2dInfo);
}

impl <'a, N, T: BackendAvgPool2d<N>> BackendAvgPool2d<N> for &'a T {
    #[inline]
    fn avg_pool2d(&self, y: &mut Self::Tensor, x: &Self::Tensor, conv_info: &Conv2dInfo) {
        (**self).avg_pool2d(y, x, conv_info)
    }
    
    #[inline]
    fn avg_pool2d_backprop(&self, dx: &mut Self::Tensor, dy: &Self::Tensor, x: &Self::Tensor, conv_info: &Conv2dInfo) {
        (**self).avg_pool2d_backprop(dx, dy, x, conv_info)
    }
}
