mod conv2d;
mod gemm;
mod pool2d;

use crate::backend::*;
use crate::tensor::*;

use self::conv2d::*;
use self::gemm::*;
use self::pool2d::*;

use core::fmt;
use core::fmt::Write;
use rand_distr::{Distribution, Normal};

pub trait NativeNumber: Copy {
    fn from_f32(val: f32) -> Self;
    fn from_f64(val: f64) -> Self;
    fn from_i64(val: i64) -> Self;
    fn from_i32(val: i32) -> Self;
    fn from_i16(val: i16) -> Self;
    fn from_i8(val: i8) -> Self;
    fn from_u64(val: u64) -> Self;
    fn from_u32(val: u32) -> Self;
    fn from_u16(val: u16) -> Self;
    fn from_u8(val: u8) -> Self;
}

impl NativeNumber for f32 {
    fn from_f32(val: f32) -> Self {
        val as f32
    }
    fn from_f64(val: f64) -> Self {
        val as f32
    }
    fn from_i64(val: i64) -> Self {
        val as f32
    }
    fn from_i32(val: i32) -> Self {
        val as f32
    }
    fn from_i16(val: i16) -> Self {
        val as f32
    }
    fn from_i8(val: i8) -> Self {
        val as f32
    }
    fn from_u64(val: u64) -> Self {
        val as f32
    }
    fn from_u32(val: u32) -> Self {
        val as f32
    }
    fn from_u16(val: u16) -> Self {
        val as f32
    }
    fn from_u8(val: u8) -> Self {
        val as f32
    }
}

impl NativeNumber for f64 {
    fn from_f32(val: f32) -> Self {
        val as f64
    }
    fn from_f64(val: f64) -> Self {
        val as f64
    }
    fn from_i64(val: i64) -> Self {
        val as f64
    }
    fn from_i32(val: i32) -> Self {
        val as f64
    }
    fn from_i16(val: i16) -> Self {
        val as f64
    }
    fn from_i8(val: i8) -> Self {
        val as f64
    }
    fn from_u64(val: u64) -> Self {
        val as f64
    }
    fn from_u32(val: u32) -> Self {
        val as f64
    }
    fn from_u16(val: u16) -> Self {
        val as f64
    }
    fn from_u8(val: u8) -> Self {
        val as f64
    }
}

pub trait NativeBackend<N: NativeNumber>: Backend<N> + Default {
    fn read_tensor<'a>(&self, t: &'a Self::Tensor) -> &'a [N];
    fn write_tensor<'a>(&self, t: &'a mut Self::Tensor) -> &'a mut [N];
}

pub struct NativeTensor<N: NativeNumber> {
    shape: TensorShape,
    ptr: Option<Box<[N]>>,
}

impl<N: NativeNumber> NativeTensor<N> {
    pub fn read(&self) -> &[N] {
        self.ptr.as_ref().unwrap()
    }

    pub fn write(&mut self) -> &mut [N] {
        if self.ptr.is_none() {
            self.ptr = Some(vec![N::from_f32(0.0); self.shape.size()].into_boxed_slice());
        }

        return self.ptr.as_mut().unwrap();
    }
}

impl<N: NativeNumber> Tensor<N> for NativeTensor<N> {
    fn new<S: Into<TensorShape>>(shape: S) -> Self {
        NativeTensor {
            shape: shape.into(),
            ptr: None,
        }
    }

    fn shape(&self) -> &TensorShape {
        &self.shape
    }

    fn resize(&mut self, shape: TensorShape) {
        self.ptr = None;
        self.shape = shape;
    }
}

#[derive(Default)]
pub struct Native<N: NativeNumber>(core::marker::PhantomData<N>);

impl<N: NativeNumber + core::fmt::Display> Native<N> {
    fn fmt_tensor(&self, t: &NativeTensor<N>, f: &mut String) -> fmt::Result {
        let strides = t.shape.default_strides();
        let last_idx = strides.dims - 1;
        writeln!(
            f,
            "default stridses {} {}",
            t.shape.default_strides(),
            last_idx
        )?;
        write!(f, "Tensor(shape={}, data=[", t.shape)?;

        for (idx, val) in t.read().iter().enumerate() {
            let is_first = idx == 0;
            let mut need_nl = false;
            let padding = 2;

            for (sidx, s) in strides.iter().enumerate() {
                if sidx != last_idx && idx % s as usize == 0 {
                    need_nl = true;
                }
            }

            if !is_first {
                write!(f, ", ")?;
            }

            if need_nl {
                write!(f, "\n{}", " ".repeat(padding))?;
            }

            write!(f, "{}", val)?;
        }

        writeln!(f, "\n])")?;

        Ok(())
    }
}

impl Backend<f32> for Native<f32> {
    type Tensor = NativeTensor<f32>;

    fn store_tensor_f32(&self, t: &Self::Tensor, data: &mut [f32]) {
        let size = t.shape().size();
        assert!(data.len() >= size);

        let dst = t.read();

        for i in 0..size {
            data[i] = dst[i] as f32;
        }
    }

    fn load_tensor_u8(&self, t: &mut Self::Tensor, data: &[u8]) {
        let size = t.shape().size();
        assert!(data.len() >= size);

        let dst = &mut t.write()[0..size];

        for i in 0..size {
            dst[i] = data[i] as f32;
        }
    }

    fn load_tensor_f32(&self, t: &mut Self::Tensor, data: &[f32]) {
        let size = t.shape().size();
        assert!(data.len() >= size);

        let dst = &mut t.write()[0..size];

        for i in 0..size {
            dst[i] = data[i];
        }
    }

    #[inline]
    fn scalar_f32(&self, val: f32) -> f32 {
        val
    }

    #[inline]
    fn fill_scalar(&self, t: &mut Self::Tensor, scalar: f32) {
        let size = t.shape().size();
        let dst = t.write();

        for i in 0..size {
            dst[i] = scalar;
        }
    }

    #[inline]
    fn fill_random(&self, t: &mut Self::Tensor, from: f32, to: f32) {
        let seed = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16,
        ];

        let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed(seed);
        let normal = Normal::new(from, to).unwrap();
        let size = t.shape().size();
        let dst = t.write();

        for i in 0..size {
            dst[i] = normal.sample(&mut rng);
        }
    }

    fn print_tensor(&self, t: &Self::Tensor) {
        let mut s = String::new();
        self.fmt_tensor(t, &mut s).unwrap();
        println!("{}", s);
    }
}

impl NativeBackend<f32> for Native<f32> {
    #[inline]
    fn read_tensor<'a>(&self, t: &'a Self::Tensor) -> &'a [f32] {
        t.read()
    }

    #[inline]
    fn write_tensor<'a>(&self, t: &'a mut Self::Tensor) -> &'a mut [f32] {
        t.write()
    }
}

impl BackendGemm<f32> for Native<f32> {
    fn matmul(&self, dst: &mut Self::Tensor, a: &Self::Tensor, b: &Self::Tensor) {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let c_shape = dst.shape().clone();

        assert_eq!(a_shape.get(0), c_shape.get(0));
        assert_eq!(b_shape.get(1), c_shape.get(1));

        assert_eq!(a_shape.dims, 2);
        assert_eq!(b_shape.dims, 2);

        let m = a_shape.get(0) as usize;
        let n = b_shape.get(1) as usize;
        let k = b_shape.get(0) as usize;

        gemm(
            false,
            false,
            m,
            n,
            k,
            1.0,
            a.read(),
            k,
            b.read(),
            n,
            0.0,
            &mut dst.write(),
            n,
        );
    }

    fn matmul_nt(&self, dst: &mut Self::Tensor, a: &Self::Tensor, b: &Self::Tensor) {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let c_shape = dst.shape().clone();

        assert_eq!(a_shape.get(0), c_shape.get(0));
        assert_eq!(b_shape.get(0), c_shape.get(1));

        assert_eq!(a_shape.dims, 2);
        assert_eq!(b_shape.dims, 2);

        let m = a_shape.get(0) as usize;
        let n = b_shape.get(0) as usize;
        let k = b_shape.get(1) as usize;

        gemm(
            false,
            true,
            m,
            n,
            k,
            1.0,
            a.read(),
            k,
            b.read(),
            k,
            0.0,
            &mut dst.write(),
            n,
        );
    }

    fn matmul_tn(&self, dst: &mut Self::Tensor, a: &Self::Tensor, b: &Self::Tensor) {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let c_shape = dst.shape().clone();

        assert_eq!(a_shape.get(1), c_shape.get(0));
        assert_eq!(b_shape.get(1), c_shape.get(1));

        assert_eq!(a_shape.dims, 2);
        assert_eq!(b_shape.dims, 2);

        let m = a_shape.get(1) as usize;
        let n = b_shape.get(1) as usize;
        let k = b_shape.get(0) as usize;

        gemm(
            true,
            false,
            m,
            n,
            k,
            1.0,
            a.read(),
            m,
            b.read(),
            n,
            0.0,
            &mut dst.write(),
            n,
        );
    }

    fn matmul_tt(&self, _dst: &mut Self::Tensor, _a: &Self::Tensor, _b: &Self::Tensor) {
        unimplemented!();
    }
}

impl BackendSigmoid<f32> for Native<f32> {
    fn sigmoid(&self, dst: &mut Self::Tensor, data: &Self::Tensor) {
        let dst_size = dst.shape().size();

        assert!(dst.shape() == data.shape());

        let data_s = &data.read()[0..dst_size];
        let dst_s = &mut dst.write()[0..dst_size];

        for i in 0..dst_size {
            dst_s[i] = 1.0 / (1.0 + (-data_s[i]).exp());
        }
    }

    fn sigmoid_grad(&self, dst: &mut Self::Tensor, z: &Self::Tensor, d: &Self::Tensor) {
        let dst_size = dst.shape().size();

        assert!(dst.shape() == z.shape());
        assert!(dst.shape() == d.shape());

        let z_s = &z.read()[0..dst_size];
        let d_s = &d.read()[0..dst_size];
        let dst_s = &mut dst.write()[0..dst_size];

        for i in 0..dst_size {
            dst_s[i] = (z_s[i] * (1.0 - z_s[i])) * d_s[i];
        }
    }
}

impl BackendReLu<f32> for Native<f32> {
    fn relu(&self, dst: &mut Self::Tensor, data: &Self::Tensor) {
        let dst_size = dst.shape().size();

        assert!(dst.shape() == data.shape());

        let data_s = &data.read()[0..dst_size];
        let dst_s = &mut dst.write()[0..dst_size];

        for i in 0..dst_size {
            let val = if data_s[i] > 0.0 { data_s[i] } else { 0.0 };

            dst_s[i] = val;
        }
    }

    fn relu_grad(&self, dst: &mut Self::Tensor, z: &Self::Tensor, d: &Self::Tensor) {
        let dst_size = dst.shape().size();

        assert!(dst.shape() == z.shape());
        assert!(dst.shape() == d.shape());

        let z_s = &z.read()[0..dst_size];
        let d_s = &d.read()[0..dst_size];
        let dst_s = &mut dst.write()[0..dst_size];

        for i in 0..dst_size {
            dst_s[i] = if z_s[i] > 0.0 { d_s[i] } else { 0.0 };
        }
    }
}

impl BackendBias<f32> for Native<f32> {
    fn bias_add(&self, dst: &mut Self::Tensor, biases: &Self::Tensor) {
        let biases_shape = biases.shape();
        let dst_shape = dst.shape().clone();
        let biases_size = biases_shape.get(0) as usize;
        let dst_size = dst_shape.size();

        assert!(dst_shape.get(dst_shape.dims - 1) as usize == biases_size);

        let batch_size = dst_shape.get(0) as usize;
        let biases_s = &biases.read()[0..biases_size];
        let dst_s = &mut dst.write()[0..dst_size];

        let mut inner = 1usize;

        for (idx, i) in dst_shape.as_slice().iter().enumerate() {
            if idx == 0 || idx == dst_shape.dims - 1 {
                continue;
            }

            inner *= *i as usize;
        }

        for b in 0..batch_size {
            for i in 0..inner {
                for l in 0..biases_size {
                    let offset = b * (inner * biases_size) + i * biases_size + l;

                    dst_s[offset] += biases_s[l];
                }
            }
        }
    }

    fn bias_grad(&self, dbiases: &mut Self::Tensor, deltas: &Self::Tensor) {
        let dbiases_shape = dbiases.shape();
        let deltas_shape = deltas.shape();
        let dbiases_size = dbiases_shape.get(0) as usize;
        let deltas_size = deltas_shape.size();

        assert!(deltas_shape.get(deltas_shape.dims - 1) as usize == dbiases_size);

        let batch_size = deltas_shape.get(0) as usize;
        let dbiases_s = &mut dbiases.write()[0..dbiases_size];
        let deltas_s = &deltas.read()[0..deltas_size];

        let mut inner = 1usize;

        for (idx, i) in deltas_shape.as_slice().iter().enumerate() {
            if idx == 0 || idx == deltas_shape.dims - 1 {
                continue;
            }

            inner *= *i as usize;
        }

        for b in 0..batch_size {
            for l in 0..dbiases_size {
                let mut bias_grad = 0.0;
                for i in 0..inner {
                    let offset = b * (inner * dbiases_size) + i * dbiases_size + l;
                    bias_grad += deltas_s[offset];
                }

                dbiases_s[l] = bias_grad;
            }
        }
    }
}

impl BackendScale<f32> for Native<f32> {
    fn scale(&self, dst: &mut Self::Tensor, scale: f32) {
        let dst_size = dst.shape().size();
        let dst_s = &mut dst.write()[0..dst_size];

        for i in 0..dst_size {
            dst_s[i] *= scale;
        }
    }
}

impl BackendMse<f32> for Native<f32> {
    fn scaled_square_diff(
        &self,
        dst: &mut Self::Tensor,
        a: &Self::Tensor,
        b: &Self::Tensor,
        scale: f32,
    ) {
        let a_size = a.shape().size();
        let b_size = b.shape().size();
        let dst_size = dst.shape().size();

        assert_eq!(a_size, dst_size);
        assert_eq!(b_size, dst_size);

        let a_s = &a.read()[0..dst_size];
        let b_s = &b.read()[0..dst_size];
        let dst_s = &mut dst.write()[0..dst_size];

        for i in 0..dst_size {
            let diff = a_s[i] - b_s[i];

            dst_s[i] = scale * diff * diff;
        }
    }

    fn scaled_diff(&self, dst: &mut Self::Tensor, a: &Self::Tensor, b: &Self::Tensor, scale: f32) {
        let a_size = a.shape().size();
        let b_size = b.shape().size();
        let dst_size = dst.shape().size();

        assert_eq!(a_size, dst_size);
        assert_eq!(b_size, dst_size);

        let a_s = &a.read()[0..dst_size];
        let b_s = &b.read()[0..dst_size];
        let dst_s = &mut dst.write()[0..dst_size];

        for i in 0..dst_size {
            dst_s[i] = scale * (a_s[i] - b_s[i]);
        }
    }
}

impl BackendAxpy<f32> for Native<f32> {
    default fn axpy(&self, dst: &mut Self::Tensor, scale: f32, a: &Self::Tensor) {
        let dst_size = dst.shape().size();

        assert!(a.shape() == dst.shape());

        let a_s = &a.read()[0..dst_size];
        let dst_s = &mut dst.write()[0..dst_size];

        for i in 0..dst_size {
            dst_s[i] += scale * a_s[i];
        }
    }
}

impl BackendAxpys<f32> for Native<f32> {
    fn axpys(&self, dst: &mut Self::Tensor, scale: f32, a: &Self::Tensor) {
        let dst_size = dst.shape().size();

        assert!(a.shape() == dst.shape());

        let a_s = &a.read()[0..dst_size];
        let dst_s = &mut dst.write()[0..dst_size];

        for i in 0..dst_size {
            dst_s[i] += scale * a_s[i] * a_s[i];
        }
    }
}

impl BackendAdd<f32> for Native<f32> {
    fn add(&self, dst: &mut Self::Tensor, a: &Self::Tensor) {
        let dst_size = dst.shape().size();

        assert!(a.shape() == dst.shape());

        let a_s = &a.read()[0..dst_size];
        let dst_s = &mut dst.write()[0..dst_size];

        for i in 0..dst_size {
            dst_s[i] += a_s[i];
        }
    }
}

impl BackendSub<f32> for Native<f32> {
    fn sub(&self, dst: &mut Self::Tensor, a: &Self::Tensor, b: &Self::Tensor) {
        let a_size = a.shape().size();
        let b_size = b.shape().size();
        let dst_size = dst.shape().size();

        assert_eq!(dst_size, a_size);
        assert_eq!(dst_size, b_size);

        let a_s = &a.read()[0..dst_size];
        let b_s = &b.read()[0..dst_size];
        let dst_s = &mut dst.write()[0..dst_size];

        for i in 0..dst_size {
            dst_s[i] = a_s[i] - b_s[i];
        }
    }
}

impl BackendMul<f32> for Native<f32> {
    fn mul(&self, dst: &mut Self::Tensor, a: &Self::Tensor) {
        let dst_size = dst.shape().size();

        assert!(a.shape() == dst.shape());

        let a_s = &a.read()[0..dst_size];
        let dst_s = &mut dst.write()[0..dst_size];

        for i in 0..dst_size {
            dst_s[i] *= a_s[i];
        }
    }
}

impl BackendCopy<f32> for Native<f32> {
    fn copy(&self, dst: &mut Self::Tensor, a: &Self::Tensor) {
        let size = dst.shape().size();

        assert!(a.shape().size() == size);

        let a_s = &a.read()[0..size];
        let dst_s = &mut dst.write()[0..size];

        for i in 0..size {
            dst_s[i] = a_s[i];
        }
    }
}

impl BackendMaximum<f32> for Native<f32> {
    fn maximum(&self, dst: &mut Self::Tensor, a: &Self::Tensor) {
        let dst_size = dst.shape().size();

        assert!(a.shape() == dst.shape());

        let a_s = &a.read()[0..dst_size];
        let dst_s = &mut dst.write()[0..dst_size];

        for i in 0..dst_size {
            dst_s[i] = f32::max(a_s[i], dst_s[i]);
        }
    }
}

impl BackendAdam<f32> for Native<f32> {
    fn adam_p(
        &self,
        dst: &mut Self::Tensor,
        lr: f32,
        moms: &Self::Tensor,
        vels: &Self::Tensor,
        eps: f32,
    ) {
        let dst_size = dst.shape().size();

        assert!(moms.shape() == dst.shape());
        assert!(vels.shape() == dst.shape());

        let moms_s = &moms.read()[0..dst_size];
        let vels_s = &vels.read()[0..dst_size];
        let dst_s = &mut dst.write()[0..dst_size];

        for i in 0..dst_size {
            dst_s[i] += lr * moms_s[i] / (vels_s[i].sqrt() + eps)
        }
    }
}

impl BackendSoftmax<f32> for Native<f32> {
    fn softmax(&self, y: &mut Self::Tensor, x: &Self::Tensor) {
        let y_shape = y.shape();
        let x_shape = x.shape();
        let size = y_shape.size();
        let axis = y_shape.last_axis() as usize;

        assert!(y_shape == x_shape);

        let x_s = &x.read()[0..size];
        let y_s = &mut y.write()[0..size];

        // copy x to y
        for i in 0..size {
            y_s[i] = x_s[i];
        }

        for i in (0..size).step_by(axis as usize) {
            assert!(i + (axis - 1) < size);

            // max(x)
            let mut max_x = core::f32::NEG_INFINITY;
            for j in 0..axis {
                let val = x_s[i + j];
                if val > max_x {
                    max_x = val;
                }
            }

            // exp(x - max(x))
            for j in 0..axis {
                let offset = i + j;
                y_s[offset] = (y_s[offset] - max_x).exp();
            }

            // 1/sum(ex)
            let mut sum = 0.0;
            for j in 0..axis {
                sum += y_s[i + j];
            }
            let rsum = 1.0 / sum;

            // ex * (1/sum(ex))
            for j in 0..axis {
                y_s[i + j] *= rsum;
            }
        }
    }
}

impl BackendConv2d<f32> for Native<f32> {
    type Context = ();

    fn conv2d_forward(
        &self,
        y: &mut Self::Tensor,
        x: &Self::Tensor,
        w: &Self::Tensor,
        conv_info: &Conv2dInfo,
    ) {
        let x_shape = &x.shape().as_slice()[0..4];
        let y_shape = &y.shape().as_slice()[0..4];
        let w_shape = &w.shape().as_slice()[0..3];

        assert_eq!(x_shape[0], y_shape[0]);

        let batch_size = x_shape[0] as isize;
        let y_channels = y_shape[1] as isize;

        let x_channels = x_shape[1] as isize;
        let x_height = x_shape[2] as isize;
        let x_width = x_shape[3] as isize;

        let filter_height = w_shape[1] as isize;
        let filter_width = w_shape[2] as isize;

        let (stride_y, stride_x) = conv_info.strides;
        let _padding = conv_info.padding;

        self.fill_scalar(y, 0.0);

        if filter_height == 3 && filter_width == 3 {
            conv2d_forward_3x3(
                y.write(),
                x.read(),
                w.read(),
                batch_size,
                x_channels,
                y_channels,
                x_height,
                x_width,
                stride_y as isize,
                stride_x as isize,
            )
        } else if filter_height == 5 && filter_width == 5 {
            conv2d_forward_5x5(
                y.write(),
                x.read(),
                w.read(),
                batch_size,
                x_channels,
                y_channels,
                x_height,
                x_width,
                stride_y as isize,
                stride_x as isize,
            )
        } else {
            conv2d_forward(
                y.write(),
                x.read(),
                w.read(),
                batch_size,
                x_channels,
                y_channels,
                x_height,
                x_width,
                filter_height,
                filter_width,
                stride_y as isize,
                stride_x as isize,
            )
        }
    }

    fn conv2d_backward_input(
        &self,
        dx: &mut Self::Tensor,
        dy: &Self::Tensor,
        w: &Self::Tensor,
        conv_info: &Conv2dInfo,
    ) {
        let dx_shape = &dx.shape().as_slice()[0..4];
        let dy_shape = &dy.shape().as_slice()[0..4];
        let w_shape = &w.shape().as_slice()[0..3];

        assert_eq!(dx_shape[0], dy_shape[0]);

        let batch_size = dx_shape[0] as isize;
        let dy_channels = dy_shape[1] as isize;
        let dy_height = dy_shape[2] as isize;
        let dy_width = dy_shape[3] as isize;

        let dx_channels = dx_shape[1] as isize;

        let filter_height = w_shape[1] as isize;
        let filter_width = w_shape[2] as isize;

        let _padding = conv_info.padding;
        let (stride_y, stride_x) = conv_info.strides;

        self.fill_scalar(dx, 0.0);

        if filter_height == 3 && filter_width == 3 {
            conv2d_backward_3x3(
                dx.write(),
                dy.read(),
                w.read(),
                batch_size,
                dx_channels,
                dy_channels,
                dy_height,
                dy_width,
                stride_y as isize,
                stride_x as isize,
            )
        } else if filter_height == 5 && filter_width == 5 {
            conv2d_backward_5x5(
                dx.write(),
                dy.read(),
                w.read(),
                batch_size,
                dx_channels,
                dy_channels,
                dy_height,
                dy_width,
                stride_y as isize,
                stride_x as isize,
            )
        } else {
            conv2d_backward(
                dx.write(),
                dy.read(),
                w.read(),
                batch_size,
                dx_channels,
                dy_channels,
                dy_height,
                dy_width,
                filter_height,
                filter_width,
                stride_y as isize,
                stride_x as isize,
            )
        }
    }

    fn conv2d_backward_filter(
        &self,
        dw: &mut Self::Tensor,
        x: &Self::Tensor,
        dy: &Self::Tensor,
        conv_info: &Conv2dInfo,
    ) {
        let x_shape = &x.shape().as_slice()[0..4];
        let dy_shape = &dy.shape().as_slice()[0..4];

        assert_eq!(x_shape[0], dy_shape[0]);

        let batch_size = x_shape[0] as isize;
        let dy_channels = dy_shape[1] as isize;
        let dy_height = dy_shape[2] as isize;
        let dy_width = dy_shape[3] as isize;

        let x_channels = x_shape[1] as isize;
        let x_height = x_shape[2] as isize;
        let x_width = x_shape[3] as isize;

        let _padding = conv_info.padding;
        let (stride_y, stride_x) = conv_info.strides;

        self.fill_scalar(dw, 0.0);

        conv2d_grads(
            dw.write(),
            x.read(),
            dy.read(),
            batch_size,
            x_channels,
            dy_channels,
            x_height,
            x_width,
            dy_height,
            dy_width,
            stride_y as isize,
            stride_x as isize,
        )
    }
}

impl BackendMaxPool2d<f32> for Native<f32> {
    fn max_pool2d(&self, y: &mut Self::Tensor, x: &Self::Tensor, conv_info: &Conv2dInfo) {
        let x_shape = &x.shape().as_slice()[0..4];
        let y_shape = &y.shape().as_slice()[0..4];

        assert_eq!(x_shape[0], y_shape[0]);
        assert_eq!(x_shape[1], y_shape[1]);

        let (stride_y, stride_x) = conv_info.strides;
        let (stride_y, stride_x) = (stride_y as isize, stride_x as isize);

        let (pool_y, pool_x) = conv_info.kernel;
        let (pool_y, pool_x) = (pool_y as isize, pool_x as isize);

        let batch_size = x_shape[0] as isize;
        let channels = x_shape[1] as isize;

        let x_rows = x_shape[2] as isize;
        let x_cols = x_shape[3] as isize;

        let y_rows = (x_rows - pool_y) / stride_y + 1;
        let y_cols = (x_cols - pool_x) / stride_x + 1;

        assert_eq!(y_rows, y_shape[2] as isize);
        assert_eq!(y_cols, y_shape[3] as isize);

        let x_img_size = x_rows * x_cols;
        let x_batch_size = x_img_size * channels;

        let y_img_size = y_rows * y_cols;
        let y_batch_size = y_img_size * channels;

        let x_vals = &x.read()[0..(batch_size * channels * x_img_size) as usize];
        let y_vals = &mut y.write()[0..(batch_size * channels * y_img_size) as usize];

        for bi in 0..batch_size {
            for ch in 0..channels {
                let x_offset = (bi * x_batch_size + ch * x_img_size) as usize;
                let x_img = &x_vals[x_offset..x_offset + x_img_size as usize];

                let y_offset = (bi * y_batch_size + ch * y_img_size) as usize;
                let y_img = &mut y_vals[y_offset..y_offset + y_img_size as usize];

                maxpool2d(
                    y_img, x_img, y_rows, y_cols, x_rows, x_cols, pool_y, pool_x, stride_y,
                    stride_x,
                );
            }
        }
    }

    fn max_pool2d_backprop(
        &self,
        dx: &mut Self::Tensor,
        dy: &Self::Tensor,
        x: &Self::Tensor,
        conv_info: &Conv2dInfo,
    ) {
        let x_shape = &x.shape().as_slice()[0..4];
        let dy_shape = &dy.shape().as_slice()[0..4];
        let dx_shape = &dx.shape().as_slice()[0..4];

        assert_eq!(x_shape, dx_shape);
        assert_eq!(x_shape[0], dy_shape[0]);
        assert_eq!(x_shape[1], dy_shape[1]);

        let batch_size = x_shape[0] as isize;
        let channels = x_shape[1] as isize;

        let x_rows = x_shape[2] as isize;
        let x_cols = x_shape[3] as isize;

        let (stride_y, stride_x) = conv_info.strides;
        let (stride_y, stride_x) = (stride_y as isize, stride_x as isize);

        let (pool_y, pool_x) = conv_info.kernel;
        let (pool_y, pool_x) = (pool_y as isize, pool_x as isize);

        let x_img_size = x_rows * x_cols;
        let x_batch_size = x_img_size * channels;

        let y_rows = (x_rows - pool_y) / stride_y + 1;
        let y_cols = (x_cols - pool_x) / stride_x + 1;

        let y_img_size = y_rows * y_cols;
        let y_batch_size = y_img_size * channels;

        let x_size = (batch_size * channels * x_img_size) as usize;
        let y_size = (batch_size * channels * y_img_size) as usize;

        let x_vals = &x.read()[0..x_size];
        let dy_vals = &dy.read()[0..y_size];
        let dx_vals = &mut dx.write()[0..x_size];

        for bi in 0..batch_size {
            for ch in 0..channels {
                let x_offset = (bi * x_batch_size + ch * x_img_size) as usize;
                let x_img = &x_vals[x_offset..x_offset + x_img_size as usize];
                let dx_img = &mut dx_vals[x_offset..x_offset + x_img_size as usize];

                let dy_offset = (bi * y_batch_size + ch * y_img_size) as usize;
                let dy_img = &dy_vals[dy_offset..dy_offset + y_img_size as usize];

                maxpool2d_backward(
                    dx_img, x_img, dy_img, x_rows, x_cols, y_rows, y_cols, pool_y, pool_x,
                    stride_y, stride_x,
                );
            }
        }
    }
}

impl BackendAvgPool2d<f32> for Native<f32> {
    fn avg_pool2d(&self, y: &mut Self::Tensor, x: &Self::Tensor, conv_info: &Conv2dInfo) {
        let x_shape = &x.shape().as_slice()[0..4];
        let y_shape = &y.shape().as_slice()[0..4];

        assert_eq!(x_shape[0], y_shape[0]);
        assert_eq!(x_shape[1], y_shape[1]);

        let (stride_y, stride_x) = conv_info.strides;
        let (stride_y, stride_x) = (stride_y as isize, stride_x as isize);

        let (pool_y, pool_x) = conv_info.kernel;
        let (pool_y, pool_x) = (pool_y as isize, pool_x as isize);

        let batch_size = x_shape[0] as isize;
        let channels = x_shape[1] as isize;

        let x_rows = x_shape[2] as isize;
        let x_cols = x_shape[3] as isize;

        let y_rows = (x_rows - pool_y) / stride_y + 1;
        let y_cols = (x_cols - pool_x) / stride_x + 1;

        assert_eq!(y_rows, y_shape[2] as isize);
        assert_eq!(y_cols, y_shape[3] as isize);

        let x_img_size = x_rows * x_cols;
        let x_batch_size = x_img_size * channels;

        let y_img_size = y_rows * y_cols;
        let y_batch_size = y_img_size * channels;

        let x_vals = &x.read()[0..(batch_size * channels * x_img_size) as usize];
        let y_vals = &mut y.write()[0..(batch_size * channels * y_img_size) as usize];

        for bi in 0..batch_size {
            for ch in 0..channels {
                let x_offset = (bi * x_batch_size + ch * x_img_size) as usize;
                let x_img = &x_vals[x_offset..x_offset + x_img_size as usize];

                let y_offset = (bi * y_batch_size + ch * y_img_size) as usize;
                let y_img = &mut y_vals[y_offset..y_offset + y_img_size as usize];

                avgpool2d(
                    y_img, x_img, y_rows, y_cols, x_rows, x_cols, pool_y, pool_x, stride_y,
                    stride_x,
                );
            }
        }
    }

    fn avg_pool2d_backprop(
        &self,
        _dx: &mut Self::Tensor,
        _dy: &Self::Tensor,
        _x: &Self::Tensor,
        _conv_info: &Conv2dInfo,
    ) {
        unimplemented!()
    }
}

impl BackendPaddingCopy2d<f32> for Native<f32> {
    fn copy_with_padding2d(
        &self,
        y: &mut Self::Tensor,
        x: &Self::Tensor,
        y_paddings: (u32, u32),
        x_paddings: (u32, u32),
    ) {
        let y_shape = &y.shape().as_slice()[0..4];
        let x_shape = &x.shape().as_slice()[0..4];

        let y_batch_size = y_shape[0] as usize;
        let y_filters = y_shape[1] as usize;
        let y_rows = y_shape[2] as usize;
        let y_cols = y_shape[3] as usize;

        let x_batch_size = x_shape[0] as usize;
        let x_filters = x_shape[1] as usize;
        let x_rows = x_shape[2] as usize;
        let x_cols = x_shape[3] as usize;

        assert_eq!(y_batch_size, x_batch_size);
        assert_eq!(y_filters, x_filters);

        let y_filter_stride = y_rows * y_cols;
        let y_batch_stride = y_filters * y_filter_stride;

        let x_filter_stride = x_rows * x_cols;
        let x_batch_stride = x_filters * x_filter_stride;

        let y_size = y_batch_size * y_filters * y_rows * y_cols;
        let x_size = x_batch_size * x_filters * x_rows * x_cols;

        let y_s = &mut y.write()[0..y_size];
        let x_s = &x.read()[0..x_size];

        for batch in 0..y_batch_size {
            for filter in 0..y_filters {
                for y_row in 0..y_rows {
                    for y_col in 0..y_cols {
                        if y_row < y_paddings.0 as usize || y_col < y_paddings.1 as usize {
                            continue;
                        }

                        if y_row - y_paddings.0 as usize >= x_rows
                            || y_col - y_paddings.1 as usize >= x_cols
                        {
                            continue;
                        }

                        let x_row = y_row - y_paddings.0 as usize + x_paddings.0 as usize;
                        let x_col = y_col - y_paddings.1 as usize + x_paddings.1 as usize;

                        println!("{} {}, {} {}", y_row, y_col, x_row, x_col);

                        let y_idx = batch * y_batch_stride
                            + filter * y_filter_stride
                            + y_row * y_cols
                            + y_col;
                        let x_idx = batch * x_batch_stride
                            + filter * x_filter_stride
                            + x_row * x_cols
                            + x_col;

                        y_s[y_idx] = x_s[x_idx];
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Native, NativeTensor};
    use crate::backend::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_copy_with_padding2d() {
        let bac: Native<f32> = Default::default();
        let mut a1 = NativeTensor::new((1, 1, 3, 3));
        let mut b1 = NativeTensor::new((1, 1, 5, 5));
        let mut a2 = NativeTensor::new((1, 1, 5, 5));
        let mut b2 = NativeTensor::new((1, 1, 3, 3));

        bac.load_tensor_u8(&mut a1, &[1, 2, 3, 4, 5, 6, 7, 8, 9]);

        bac.load_tensor_u8(
            &mut a2,
            &[
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25,
            ],
        );

        bac.copy_with_padding2d(&mut b1, &a1, (1, 1), (0, 0));
        bac.copy_with_padding2d(&mut b2, &a2, (0, 0), (1, 1));

        assert!(
            b1.read()
                == &[
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.0,
                    7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ]
        );

        assert!(b2.read() == &[7.0, 8.0, 9.0, 12.0, 13.0, 14.0, 17.0, 18.0, 19.0,]);
    }

    #[test]
    fn test_softmax() {
        let bac: Native<f32> = Default::default();
        let mut a = NativeTensor::new((3, 3));
        let mut b = NativeTensor::new((3, 3));

        bac.load_tensor_u8(&mut a, &[1, 2, 3, 4, 5, 6, 7, 8, 9]);

        bac.softmax(&mut b, &a);

        assert!(
            b.read()
                == &[
                    0.09003057, 0.24472847, 0.66524096, 0.09003057, 0.24472847, 0.66524096,
                    0.09003057, 0.24472847, 0.66524096,
                ]
        );
    }

    #[test]
    fn test_matmul() {
        let bac: Native<f32> = Default::default();
        let mut a = NativeTensor::new((2, 3));
        let mut b = NativeTensor::new((3, 4));
        let mut c = NativeTensor::new((2, 4));

        bac.load_tensor_u8(&mut a, &[1, 2, 3, 4, 5, 6]);

        bac.load_tensor_u8(&mut b, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

        bac.matmul(&mut c, &a, &b);

        assert!(c.read() == &[38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0,]);
    }

    #[test]
    fn test_matmul_nt() {
        let bac: Native<f32> = Default::default();
        let mut a = NativeTensor::new((2, 3));
        let mut b = NativeTensor::new((4, 3));
        let mut c = NativeTensor::new((2, 4));

        bac.load_tensor_u8(&mut a, &[1, 2, 3, 4, 5, 6]);

        bac.load_tensor_u8(&mut b, &[1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]);

        bac.matmul_nt(&mut c, &a, &b);

        assert!(c.read() == &[38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0,]);
    }

    #[test]
    fn test_matmul_tn() {
        let bac: Native<f32> = Default::default();
        let mut a = NativeTensor::new((8, 5));
        let mut b = NativeTensor::new((8, 3));
        let mut c = NativeTensor::new((5, 3));

        bac.load_tensor_u8(
            &mut a,
            &[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            ],
        );

        bac.load_tensor_u8(
            &mut b,
            &[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23,
            ],
        );

        bac.matmul_tn(&mut c, &a, &b);

        assert!(
            c.read()
                == &[
                    2100.0, 2240.0, 2380.0, 2184.0, 2332.0, 2480.0, 2268.0, 2424.0, 2580.0, 2352.0,
                    2516.0, 2680.0, 2436.0, 2608.0, 2780.0
                ]
        );
    }

    #[test]
    fn test_axpy() {
        let bac: Native<f32> = Default::default();

        let mut a = NativeTensor::new((2, 2));
        let mut b = NativeTensor::new((2, 2));

        bac.load_tensor_u8(&mut a, &[1, 2, 3, 4]);
        bac.load_tensor_u8(&mut b, &[1, 2, 3, 4]);

        bac.axpy(&mut a, 2.0f32, &b);

        assert!(a.read() == &[3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_add() {
        let bac: Native<f32> = Default::default();

        let mut a = NativeTensor::new((2, 2));
        let mut b = NativeTensor::new((2, 2));

        bac.load_tensor_u8(&mut a, &[1, 2, 3, 4]);
        bac.load_tensor_u8(&mut b, &[1, 2, 3, 4]);

        bac.add(&mut a, &b);

        assert!(a.read() == &[2.0, 4.0, 6.0, 8.0]);
    }
}
