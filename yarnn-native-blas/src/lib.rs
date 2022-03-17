mod img2col;

use std::marker::PhantomData;
use yarnn::backend::*;
use yarnn::native::*;
use yarnn::tensor::*;

extern crate openblas_src;

pub struct NativeBlas<N, B>
where
    N: NativeNumber,
    B: NativeBackend<N>,
{
    inner: B,
    _m: PhantomData<fn(N)>,
}

impl<N, B> Default for NativeBlas<N, B>
where
    N: NativeNumber,
    B: NativeBackend<N>,
{
    fn default() -> Self {
        Self {
            inner: Default::default(),
            _m: Default::default(),
        }
    }
}

impl<N, B> NativeBackend<N> for NativeBlas<N, B>
where
    N: NativeNumber,
    B: NativeBackend<N>,
{
    #[inline]
    fn read_tensor<'a>(&self, t: &'a Self::Tensor) -> &'a [N] {
        self.inner.read_tensor(t)
    }

    #[inline]
    fn write_tensor<'a>(&self, t: &'a mut Self::Tensor) -> &'a mut [N] {
        self.inner.write_tensor(t)
    }
}

impl<N, B> NativeBlas<N, B>
where
    N: NativeNumber,
    B: NativeBackend<N>,
{
    pub fn new(inner: B) -> Self {
        Self {
            inner,
            _m: Default::default(),
        }
    }
}

impl<N, B> Backend<N> for NativeBlas<N, B>
where
    N: NativeNumber,
    B: NativeBackend<N>,
{
    type Tensor = B::Tensor;

    #[inline]
    fn store_tensor_f32(&self, t: &Self::Tensor, data: &mut [f32]) {
        self.inner.store_tensor_f32(t, data)
    }

    #[inline]
    fn load_tensor_u8(&self, t: &mut Self::Tensor, data: &[u8]) {
        self.inner.load_tensor_u8(t, data)
    }

    #[inline]
    fn load_tensor_f32(&self, t: &mut Self::Tensor, data: &[f32]) {
        self.inner.load_tensor_f32(t, data)
    }

    #[inline]
    fn scalar_f32(&self, val: f32) -> N {
        N::from_f32(val)
    }

    #[inline]
    fn fill_scalar(&self, t: &mut Self::Tensor, scalar: N) {
        self.inner.fill_scalar(t, scalar)
    }

    #[inline]
    fn fill_random(&self, t: &mut Self::Tensor, from: N, to: N) {
        self.inner.fill_random(t, from, to)
    }

    #[inline]
    fn print_tensor(&self, t: &Self::Tensor) {
        self.inner.print_tensor(t)
    }
}

impl<B> BackendGemm<f32> for NativeBlas<f32, B>
where
    B: NativeBackend<f32>,
{
    #[inline]
    fn matmul(&self, dst: &mut Self::Tensor, a: &Self::Tensor, b: &Self::Tensor) {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let c_shape = dst.shape().clone();

        assert_eq!(a_shape.get(0), c_shape.get(0));
        assert_eq!(b_shape.get(1), c_shape.get(1));

        assert_eq!(a_shape.dims, 2);
        assert_eq!(b_shape.dims, 2);

        let m = a_shape.get(0) as i32;
        let n = b_shape.get(1) as i32;
        let k = b_shape.get(0) as i32;

        unsafe {
            blas::sgemm(
                'N' as u8,
                'N' as u8,
                n,
                m,
                k,
                1.0,
                self.read_tensor(b),
                n,
                self.read_tensor(a),
                k,
                0.0,
                self.write_tensor(dst),
                n,
            );
        }
    }

    #[inline]
    fn matmul_nt(&self, dst: &mut Self::Tensor, a: &Self::Tensor, b: &Self::Tensor) {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let c_shape = dst.shape().clone();

        assert_eq!(a_shape.get(0), c_shape.get(0));
        assert_eq!(b_shape.get(0), c_shape.get(1));

        assert_eq!(a_shape.dims, 2);
        assert_eq!(b_shape.dims, 2);

        let m = a_shape.get(0) as i32;
        let n = b_shape.get(0) as i32;
        let k = b_shape.get(1) as i32;

        unsafe {
            blas::sgemm(
                'T' as u8,
                'N' as u8,
                n,
                m,
                k,
                1.0,
                self.read_tensor(b),
                k,
                self.read_tensor(a),
                k,
                0.0,
                self.write_tensor(dst),
                n,
            );
        }
    }

    #[inline]
    fn matmul_tn(&self, dst: &mut Self::Tensor, a: &Self::Tensor, b: &Self::Tensor) {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let c_shape = dst.shape().clone();

        assert_eq!(a_shape.get(1), c_shape.get(0));
        assert_eq!(b_shape.get(1), c_shape.get(1));

        assert_eq!(a_shape.dims, 2);
        assert_eq!(b_shape.dims, 2);

        let m = a_shape.get(1) as i32;
        let n = b_shape.get(1) as i32;
        let k = b_shape.get(0) as i32;

        unsafe {
            blas::sgemm(
                'N' as u8,
                'T' as u8,
                n,
                m,
                k,
                1.0,
                self.read_tensor(b),
                n,
                self.read_tensor(a),
                m,
                0.0,
                self.write_tensor(dst),
                n,
            );
        }
    }

    #[inline]
    fn matmul_tt(&self, _dst: &mut Self::Tensor, _a: &Self::Tensor, _b: &Self::Tensor) {
        unimplemented!();
    }
}

impl<B> BackendAxpy<f32> for NativeBlas<f32, B>
where
    B: NativeBackend<f32>,
{
    #[inline]
    fn axpy(&self, dst: &mut Self::Tensor, scale: f32, x: &Self::Tensor) {
        let dst_size = dst.shape().size();

        assert!(x.shape() == dst.shape());

        unsafe {
            blas::saxpy(
                dst_size as i32,
                scale,
                self.read_tensor(x),
                1,
                self.write_tensor(dst),
                1,
            );
        }
    }
}

impl<B> BackendScale<f32> for NativeBlas<f32, B>
where
    B: NativeBackend<f32>,
{
    #[inline]
    fn scale(&self, dst: &mut Self::Tensor, scale: f32) {
        let dst_size = dst.shape().size();

        unsafe {
            blas::sscal(dst_size as i32, scale, self.write_tensor(dst), 1);
        }
    }
}

impl<B: NativeBackend<f32> + BackendSigmoid<f32>> BackendSigmoid<f32> for NativeBlas<f32, B> {
    #[inline]
    fn sigmoid(&self, dst: &mut Self::Tensor, data: &Self::Tensor) {
        self.inner.sigmoid(dst, data)
    }

    #[inline]
    fn sigmoid_grad(&self, dst: &mut Self::Tensor, z: &Self::Tensor, d: &Self::Tensor) {
        self.inner.sigmoid_grad(dst, z, d)
    }
}

impl<B: NativeBackend<f32> + BackendReLu<f32>> BackendReLu<f32> for NativeBlas<f32, B> {
    #[inline]
    fn relu(&self, dst: &mut Self::Tensor, data: &Self::Tensor) {
        self.inner.relu(dst, data)
    }

    #[inline]
    fn relu_grad(&self, dst: &mut Self::Tensor, z: &Self::Tensor, d: &Self::Tensor) {
        self.inner.relu_grad(dst, z, d)
    }
}

impl<B: NativeBackend<f32> + BackendBias<f32>> BackendBias<f32> for NativeBlas<f32, B> {
    #[inline]
    fn bias_add(&self, dst: &mut Self::Tensor, biases: &Self::Tensor) {
        self.inner.bias_add(dst, biases)
    }

    #[inline]
    fn bias_grad(&self, dbiases: &mut Self::Tensor, deltas: &Self::Tensor) {
        self.inner.bias_grad(dbiases, deltas)
    }
}

impl<B: NativeBackend<f32> + BackendMse<f32>> BackendMse<f32> for NativeBlas<f32, B> {
    #[inline]
    fn scaled_square_diff(
        &self,
        dst: &mut Self::Tensor,
        a: &Self::Tensor,
        b: &Self::Tensor,
        scale: f32,
    ) {
        self.inner.scaled_square_diff(dst, a, b, scale)
    }

    #[inline]
    fn scaled_diff(&self, dst: &mut Self::Tensor, a: &Self::Tensor, b: &Self::Tensor, scale: f32) {
        self.inner.scaled_diff(dst, a, b, scale)
    }
}

impl<B: NativeBackend<f32> + BackendAxpys<f32>> BackendAxpys<f32> for NativeBlas<f32, B> {
    #[inline]
    fn axpys(&self, dst: &mut Self::Tensor, scale: f32, a: &Self::Tensor) {
        self.inner.axpys(dst, scale, a)
    }
}

impl<B: NativeBackend<f32> + BackendAdd<f32>> BackendAdd<f32> for NativeBlas<f32, B> {
    #[inline]
    fn add(&self, dst: &mut Self::Tensor, a: &Self::Tensor) {
        self.inner.add(dst, a)
    }
}

impl<B: NativeBackend<f32> + BackendSub<f32>> BackendSub<f32> for NativeBlas<f32, B> {
    #[inline]
    fn sub(&self, dst: &mut Self::Tensor, a: &Self::Tensor, b: &Self::Tensor) {
        self.inner.sub(dst, a, b)
    }
}

impl<B: NativeBackend<f32> + BackendMul<f32>> BackendMul<f32> for NativeBlas<f32, B> {
    #[inline]
    fn mul(&self, dst: &mut Self::Tensor, a: &Self::Tensor) {
        self.inner.mul(dst, a)
    }
}

impl<B: NativeBackend<f32> + BackendCopy<f32>> BackendCopy<f32> for NativeBlas<f32, B> {
    #[inline]
    fn copy(&self, dst: &mut Self::Tensor, a: &Self::Tensor) {
        self.inner.copy(dst, a)
    }
}

impl<B: NativeBackend<f32> + BackendMaximum<f32>> BackendMaximum<f32> for NativeBlas<f32, B> {
    #[inline]
    fn maximum(&self, dst: &mut Self::Tensor, a: &Self::Tensor) {
        self.inner.maximum(dst, a)
    }
}

impl<B: NativeBackend<f32> + BackendAdam<f32>> BackendAdam<f32> for NativeBlas<f32, B> {
    #[inline]
    fn adam_p(
        &self,
        dst: &mut Self::Tensor,
        lr: f32,
        moms: &Self::Tensor,
        vels: &Self::Tensor,
        eps: f32,
    ) {
        self.inner.adam_p(dst, lr, moms, vels, eps)
    }
}

impl<B: NativeBackend<f32> + BackendSoftmax<f32>> BackendSoftmax<f32> for NativeBlas<f32, B> {
    #[inline]
    fn softmax(&self, y: &mut Self::Tensor, x: &Self::Tensor) {
        self.inner.softmax(y, x)
    }
}

impl<B: NativeBackend<f32> + BackendConv2d<f32>> BackendConv2d<f32> for NativeBlas<f32, B> {
    type Context = ();

    #[inline]
    fn conv2d_forward(
        &self,
        y: &mut Self::Tensor,
        x: &Self::Tensor,
        w: &Self::Tensor,
        conv_info: &Conv2dInfo,
    ) {
        self.inner.conv2d_forward(y, x, w, conv_info)
    }

    #[inline]
    fn conv2d_backward_input(
        &self,
        dx: &mut Self::Tensor,
        dy: &Self::Tensor,
        w: &Self::Tensor,
        conv_info: &Conv2dInfo,
    ) {
        self.inner.conv2d_backward_input(dx, dy, w, conv_info)
    }

    #[inline]
    fn conv2d_backward_filter(
        &self,
        dw: &mut Self::Tensor,
        x: &Self::Tensor,
        dy: &Self::Tensor,
        conv_info: &Conv2dInfo,
    ) {
        self.inner.conv2d_backward_filter(dw, x, dy, conv_info)
    }
}

impl<B: NativeBackend<f32> + BackendMaxPool2d<f32>> BackendMaxPool2d<f32> for NativeBlas<f32, B> {
    #[inline]
    fn max_pool2d(&self, y: &mut Self::Tensor, x: &Self::Tensor, conv_info: &Conv2dInfo) {
        self.inner.max_pool2d(y, x, conv_info)
    }

    #[inline]
    fn max_pool2d_backprop(
        &self,
        dx: &mut Self::Tensor,
        dy: &Self::Tensor,
        x: &Self::Tensor,
        conv_info: &Conv2dInfo,
    ) {
        self.inner.max_pool2d_backprop(dx, dy, x, conv_info)
    }
}

impl<B: NativeBackend<f32> + BackendAvgPool2d<f32>> BackendAvgPool2d<f32> for NativeBlas<f32, B> {
    #[inline]
    fn avg_pool2d(&self, y: &mut Self::Tensor, x: &Self::Tensor, conv_info: &Conv2dInfo) {
        self.inner.avg_pool2d(y, x, conv_info)
    }

    #[inline]
    fn avg_pool2d_backprop(
        &self,
        dx: &mut Self::Tensor,
        dy: &Self::Tensor,
        x: &Self::Tensor,
        conv_info: &Conv2dInfo,
    ) {
        self.inner.avg_pool2d_backprop(dx, dy, x, conv_info)
    }
}

impl<B: NativeBackend<f32> + BackendPaddingCopy2d<f32>> BackendPaddingCopy2d<f32>
    for NativeBlas<f32, B>
{
    #[inline]
    fn copy_with_padding2d(
        &self,
        y: &mut Self::Tensor,
        x: &Self::Tensor,
        y_paddings: (u32, u32),
        x_paddings: (u32, u32),
    ) {
        self.inner.copy_with_padding2d(y, x, y_paddings, x_paddings)
    }
}
