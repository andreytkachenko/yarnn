use crate::tensor::Tensor;


pub trait Backend<N> {
    type Tensor: Tensor<N>;

    fn store_tensor_f32(&self, t: &Self::Tensor, data: &mut [f32]);
    fn load_tensor_u8(&self, t: &mut Self::Tensor, data: &[u8]);
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