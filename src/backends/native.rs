use crate::tensor::*;
use crate::backend::*;
use std::fmt;
use std::fmt::Write;
use rand_distr::{Normal, Distribution};
use super::conv2d::*;
use super::pool2d::*;
use super::gemm::*;


pub struct NativeTensorF32 {
    shape: TensorShape,
    ptr: Option<Box<[f32]>>
}

impl NativeTensorF32 {
    pub fn read(&self) -> &[f32] {
        self.ptr.as_ref().unwrap()
    } 

    pub fn write(&mut self) -> &mut [f32] {
        if self.ptr.is_none() {
            self.ptr = Some(vec![0.0; self.shape.size()].into_boxed_slice());
        }

        return self.ptr.as_mut().unwrap()
    }
}

impl Tensor<f32> for NativeTensorF32 {
    fn new<S: Into<TensorShape>>(shape: S) -> Self {
        NativeTensorF32 {
            shape: shape.into(),
            ptr: None,
        }
    }

    fn shape(&self) -> &TensorShape {
        &self.shape
    }

    fn resize(&mut self, shape: TensorShape) {
        self.ptr = if let Some(ptr) = self.ptr.take() {
            let size = self.shape.size();
            let raw = Box::into_raw(ptr) as *mut f32;
            let mut data = unsafe {Vec::from_raw_parts(raw, size, size)};
            data.resize(shape.size(), 0.0);

            Some(data.into_boxed_slice())
        } else {
            None
        };
        self.shape = shape;
    }
}

pub struct Native;

impl Native {
    fn fmt_tensor(&self, t: &NativeTensorF32, f: &mut String) -> fmt::Result {
        let strides = t.shape.default_strides();
        let last_idx = strides.dims - 1;
        writeln!(f, "default stridses {} {}", t.shape.default_strides(), last_idx)?;
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

impl Backend<f32> for Native {
    type Tensor = NativeTensorF32;

    fn store_tensor_f32(&self, t: &Self::Tensor, data: &mut [f32]) {
        let size = t.shape().size();
        assert!(data.len() >= size);

        let dst = t.read();

        for i in 0 .. size {
            data[i] = dst[i] as f32;
        }
    }

    fn load_tensor_u8(&self, t: &mut Self::Tensor, data: &[u8]) {
        let size = t.shape().size();
        assert!(data.len() >= size);

        let dst = &mut t.write()[0..size];

        for i in 0 .. size {
            dst[i] = data[i] as f32;
        }
    }

    fn load_tensor_f32(&self, t: &mut Self::Tensor, data: &[f32]) {
        let size = t.shape().size();
        assert!(data.len() >= size);

        let dst = &mut t.write()[0..size];

        for i in 0 .. size {
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

        for i in 0 .. size {
            dst[i] = scalar;
        }
    }

    #[inline]
    fn fill_random(&self, t: &mut Self::Tensor, from: f32, to: f32) {
        let seed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

        let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed(seed);
        let normal = Normal::new(from, to).unwrap();
        let size = t.shape().size();
        let dst = t.write();

        for i in 0 .. size {
            dst[i] = normal.sample(&mut rng);
        }
    }

    fn print_tensor(&self, t: &Self::Tensor) {
        let mut s = String::new();
        self.fmt_tensor(t, &mut s).unwrap();
        println!("{}", s);
    } 
}


impl BackendGemm<f32> for Native {
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
            false, false,
            m, n, k, 
            1.0, 
            a.read(), k, 
            b.read(), n, 
            0.0, 
            &mut dst.write(), n
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

        gemm(false, true,
             m, n, k, 
             1.0, 
             a.read(), k, 
             b.read(), k, 
             0.0, 
             &mut dst.write(), n);
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

        gemm(true, false,
            m, n, k, 
            1.0, 
            a.read(), m, 
            b.read(), n, 
            0.0, 
            &mut dst.write(), n);
    }

    fn matmul_tt(&self, _dst: &mut Self::Tensor, _a: &Self::Tensor, _b: &Self::Tensor) {
        unimplemented!();
    }
}

impl BackendSigmoid<f32> for Native {
    fn sigmoid(&self, dst: &mut Self::Tensor, data: &Self::Tensor) {
        let dst_size = dst.shape().size();

        assert!(dst.shape() == data.shape());

        let data_s = &data.read()[0 .. dst_size];
        let dst_s = &mut dst.write()[0 .. dst_size];

        for i in 0 .. dst_size {
            dst_s[i] = 1.0 / (1.0 + (-data_s[i]).exp());
        }
    }

    fn sigmoid_grad(&self, dst: &mut Self::Tensor, z: &Self::Tensor, d: &Self::Tensor) {
        let dst_size = dst.shape().size();

        assert!(dst.shape() == z.shape());
        assert!(dst.shape() == d.shape());

        let z_s = &z.read()[0 .. dst_size];
        let d_s = &d.read()[0 .. dst_size];
        let dst_s = &mut dst.write()[0 .. dst_size];

        for i in 0 .. dst_size {
            dst_s[i] = (z_s[i] * (1.0 - z_s[i])) * d_s[i];
        }
    }
}

impl BackendReLu<f32> for Native {
    fn relu(&self, dst: &mut Self::Tensor, data: &Self::Tensor) {
        let dst_size = dst.shape().size();

        assert!(dst.shape() == data.shape());

        let data_s = &data.read()[0 .. dst_size];
        let dst_s = &mut dst.write()[0 .. dst_size];

        for i in 0 .. dst_size {
            let val = if data_s[i] > 0.0 {
                data_s[i]
            } else {
                0.0
            };

            dst_s[i] = val;
        }
    }

    fn relu_grad(&self, dst: &mut Self::Tensor, z: &Self::Tensor, d: &Self::Tensor) {
        let dst_size = dst.shape().size();

        assert!(dst.shape() == z.shape());
        assert!(dst.shape() == d.shape());

        let z_s = &z.read()[0 .. dst_size];
        let d_s = &d.read()[0 .. dst_size];
        let dst_s = &mut dst.write()[0 .. dst_size];

        for i in 0 .. dst_size {
            dst_s[i] = if z_s[i] > 0.0 {
                d_s[i]
            } else {
                0.0
            };
        }
    }
}

impl BackendBias<f32> for Native {
    fn bias_add(&self, dst: &mut Self::Tensor, biases: &Self::Tensor) {
        let biases_shape = biases.shape();
        let dst_shape = dst.shape().clone();
        let biases_size = biases_shape.get(0) as usize;
        let dst_size = dst_shape.size();
        
        assert!(dst_shape.get(dst_shape.dims - 1) as usize == biases_size);
        
        let batch_size = dst_shape.get(0) as usize;
        let biases_s = &biases.read()[0 .. biases_size];
        let dst_s = &mut dst.write()[0 .. dst_size];

        let mut inner = 1usize;

        for (idx, i) in dst_shape.as_slice().iter().enumerate() {
            if idx == 0 || idx == dst_shape.dims - 1 {
                continue;
            }

            inner *= *i as usize;
        }

        for b in 0 .. batch_size {
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
        let dbiases_s = &mut dbiases.write()[0 .. dbiases_size];
        let deltas_s = &deltas.read()[0 .. deltas_size];

        let mut inner = 1usize;

        for (idx, i) in deltas_shape.as_slice().iter().enumerate() {
            if idx == 0 || idx == deltas_shape.dims - 1 {
                continue;
            }

            inner *= *i as usize;
        }

        for b in 0 .. batch_size {
            for l in 0 .. dbiases_size {
                let mut bias_grad = 0.0;
                for i in 0 .. inner {
                    let offset = b * (inner * dbiases_size) + i * dbiases_size + l;
                    bias_grad += deltas_s[offset];
                }

                dbiases_s[l] = bias_grad;
            }
        }
    }
}

impl BackendScale<f32> for Native {
    fn scale(&self, dst: &mut Self::Tensor, scale: f32) {
        let dst_size = dst.shape().size();
        let dst_s = &mut dst.write()[0 .. dst_size];

        for i in 0 .. dst_size {
            dst_s[i] *= scale;
        }
    }
}

impl BackendMse<f32> for Native {
    fn scaled_square_diff(&self, dst: &mut Self::Tensor, a: &Self::Tensor, b: &Self::Tensor, scale: f32) {
        let a_size = a.shape().size();
        let b_size = b.shape().size();
        let dst_size = dst.shape().size();

        assert_eq!(a_size, dst_size);
        assert_eq!(b_size, dst_size);

        let a_s = &a.read()[0 .. dst_size];
        let b_s = &b.read()[0 .. dst_size];
        let dst_s = &mut dst.write()[0 .. dst_size];

        for i in 0 .. dst_size {
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

        let a_s = &a.read()[0 .. dst_size];
        let b_s = &b.read()[0 .. dst_size];
        let dst_s = &mut dst.write()[0 .. dst_size];

        for i in 0 .. dst_size {
            dst_s[i] = scale * (a_s[i] - b_s[i]);
        }
    }
}

impl BackendAxpy<f32> for Native {
    default fn axpy(&self, dst: &mut Self::Tensor, scale: f32, a: &Self::Tensor) {
        let dst_size = dst.shape().size();

        assert!(a.shape() == dst.shape());

        let a_s = &a.read()[0 .. dst_size];
        let dst_s = &mut dst.write()[0 .. dst_size];

        for i in 0 .. dst_size {
            dst_s[i] += scale * a_s[i];
        }
    }
}

impl BackendAxpys<f32> for Native {
    fn axpys(&self, dst: &mut Self::Tensor, scale: f32, a: &Self::Tensor) {
        let dst_size = dst.shape().size();

        assert!(a.shape() == dst.shape());

        let a_s = &a.read()[0 .. dst_size];
        let dst_s = &mut dst.write()[0 .. dst_size];

        for i in 0 .. dst_size {
            dst_s[i] += scale * a_s[i] * a_s[i];
        }
    }
}

impl BackendAdd<f32> for Native {
    fn add(&self, dst: &mut Self::Tensor, a: &Self::Tensor) {
        let dst_size = dst.shape().size();

        assert!(a.shape() == dst.shape());

        let a_s = &a.read()[0 .. dst_size];
        let dst_s = &mut dst.write()[0 .. dst_size];

        for i in 0 .. dst_size {
            dst_s[i] += a_s[i];
        }
    }
}

impl BackendSub<f32> for Native {
    fn sub(&self, dst: &mut Self::Tensor, a: &Self::Tensor, b: &Self::Tensor) {
        let a_size = a.shape().size();
        let b_size = b.shape().size();
        let dst_size = dst.shape().size();

        assert_eq!(dst_size, a_size);
        assert_eq!(dst_size, b_size);

        let a_s = &a.read()[0 .. dst_size];
        let b_s = &b.read()[0 .. dst_size];
        let dst_s = &mut dst.write()[0 .. dst_size];

        for i in 0 .. dst_size {
            dst_s[i] = a_s[i] - b_s[i];
        }
    }
    
}

impl BackendMul<f32> for Native {
    fn mul(&self, dst: &mut Self::Tensor, a: &Self::Tensor) {
        let dst_size = dst.shape().size();

        assert!(a.shape() == dst.shape());

        let a_s = &a.read()[0 .. dst_size];
        let dst_s = &mut dst.write()[0 .. dst_size];

        for i in 0 .. dst_size {
            dst_s[i] *= a_s[i];
        }
    }
}


impl BackendCopy<f32> for Native {
    fn copy(&self, dst: &mut Self::Tensor, a: &Self::Tensor) {
        let size = dst.shape().size();

        assert!(a.shape().size() == size);

        let a_s = &a.read()[0 .. size];
        let dst_s = &mut dst.write()[0 .. size];

        for i in 0 .. size {
            dst_s[i] = a_s[i];
        }
    }
}

impl BackendMaximum<f32> for Native {
    fn maximum(&self, dst: &mut Self::Tensor, a: &Self::Tensor) {
        let dst_size = dst.shape().size();

        assert!(a.shape() == dst.shape());

        let a_s = &a.read()[0 .. dst_size];
        let dst_s = &mut dst.write()[0 .. dst_size];

        for i in 0 .. dst_size {
            dst_s[i] = f32::max(a_s[i], dst_s[i]);
        }
    }
}


impl BackendAdam<f32> for Native {
    fn adam_p(&self, dst: &mut Self::Tensor, lr: f32, moms: &Self::Tensor, vels: &Self::Tensor, eps: f32) {
        let dst_size = dst.shape().size();

        assert!(moms.shape() == dst.shape());
        assert!(vels.shape() == dst.shape());

        let moms_s = &moms.read()[0 .. dst_size];
        let vels_s = &vels.read()[0 .. dst_size];
        let dst_s = &mut dst.write()[0 .. dst_size];

        for i in 0 .. dst_size {
            dst_s[i] += lr * moms_s[i] / (vels_s[i].sqrt() + eps)
        }
    }
}

impl BackendSoftmax<f32> for Native {
    fn softmax(&self, y: &mut Self::Tensor, x: &Self::Tensor) {
        let y_shape = y.shape();
        let x_shape = x.shape();
        let size = y_shape.size();
        let axis = y_shape.last_axis() as usize;

        assert!(y_shape == x_shape);

        let x_s = &x.read()[0 .. size];
        let y_s = &mut y.write()[0 .. size];

        // copy x to y
        for i in 0..size {
            y_s[i] = x_s[i];
        }

        for i in (0..size).step_by(axis as usize) {
            assert!(i + (axis - 1) < size);

            // max(x)
            let mut max_x = std::f32::NEG_INFINITY;
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

impl BackendConv2d<f32> for Native {
    type Context = ();

    fn conv2d_forward(&self, y: &mut Self::Tensor, x: &Self::Tensor, w: &Self::Tensor, conv_info: &Conv2dInfo) {
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
        conv2d_forward(
            y.write(), x.read(), w.read(),
            batch_size, x_channels, y_channels,
            x_height, x_width, filter_height, filter_width,
            stride_y as isize, stride_x as isize
        )
    }

    fn conv2d_backward_input(&self, dx: &mut Self::Tensor, dy: &Self::Tensor, w: &Self::Tensor, conv_info: &Conv2dInfo) {
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

        conv2d_backward(
            dx.write(), dy.read(), w.read(),
            batch_size, dx_channels, dy_channels,
            dy_height, dy_width,
            filter_height, filter_width,
            stride_y as isize, stride_x as isize
        )
    }

    fn conv2d_backward_filter(&self, dw: &mut Self::Tensor, x: &Self::Tensor, dy: &Self::Tensor, conv_info: &Conv2dInfo) {
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
            dw.write(), x.read(), dy.read(), 
            batch_size, x_channels, dy_channels,
            x_height, x_width, dy_height, dy_width,
            stride_y as isize, stride_x as isize
        )
    }
}

impl BackendMaxPool2d<f32> for Native {
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

                maxpool2d(y_img, x_img,  y_rows, y_cols, x_rows, x_cols,
                          pool_y, pool_x, stride_y, stride_x);
            }   
        }
    }

    fn max_pool2d_backprop(&self, dx: &mut Self::Tensor, dy: &Self::Tensor, x: &Self::Tensor, conv_info: &Conv2dInfo) {
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

                maxpool2d_backward(dx_img, x_img, dy_img,
                                   x_rows, x_cols, y_rows, y_cols,
                                   pool_y, pool_x, stride_y, stride_x);
            }   
        }
    }
}

impl BackendAvgPool2d<f32> for Native {
    fn avg_pool2d(&self, _y: &mut Self::Tensor, _x: &Self::Tensor, _conv_info: &Conv2dInfo) {
        unimplemented!()
    }

    fn avg_pool2d_backprop(&self, _dx: &mut Self::Tensor, _dy: &Self::Tensor, _x: &Self::Tensor, _conv_info: &Conv2dInfo) {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::*;
    use super::{Native, NativeTensorF32};
    use crate::tensor::{Tensor, TensorShape};

    #[test]
    fn test_softmax() {
        let bac = Native;
        let mut a = NativeTensorF32::new((3, 3));
        let mut b = NativeTensorF32::new((3, 3));

        bac.load_tensor_u8(&mut a, &[
            1,2,3,
            4,5,6,
            7,8,9,
        ]);

        bac.softmax(&mut b, &a);

        assert!(
            b.read() == &[
                0.09003057, 0.24472847, 0.66524096,  
                0.09003057, 0.24472847, 0.66524096, 
                0.09003057, 0.24472847, 0.66524096,
            ]
        );
    }

    #[test]
    fn test_matmul() {
        let bac = Native;
        let mut a = NativeTensorF32::new((2, 3));
        let mut b = NativeTensorF32::new((3, 4));
        let mut c = NativeTensorF32::new((2, 4));

        bac.load_tensor_u8(&mut a, &[
            1,2,3,
            4,5,6
        ]);

        bac.load_tensor_u8(&mut b, &[
            1,2,3,4,
            5,6,7,8,
            9,10,11,12
        ]);

        bac.matmul(&mut c, &a, &b);

        assert!(
            c.read() == &[
                38.0,  44.0,  50.0,  56.0,
                83.0,  98.0, 113.0, 128.0,
            ]
        );
    }

    #[test]
    fn test_matmul_nt() {
        let bac = Native;
        let mut a = NativeTensorF32::new((2, 3));
        let mut b = NativeTensorF32::new((4, 3));
        let mut c = NativeTensorF32::new((2, 4));

        bac.load_tensor_u8(&mut a, &[
            1,2,3,
            4,5,6
        ]);

        bac.load_tensor_u8(&mut b, &[
            1,5,9,
            2,6,10,
            3,7,11,
            4,8,12
        ]);

        bac.matmul_nt(&mut c, &a, &b);

        assert!(
            c.read() == &[
                38.0,  44.0,  50.0,  56.0,
                83.0,  98.0, 113.0, 128.0,
            ]
        );
    }


    #[test]
    fn test_matmul_tn() {
        let bac = Native;
        let mut a = NativeTensorF32::new((8, 5));
        let mut b = NativeTensorF32::new((8, 3));
        let mut c = NativeTensorF32::new((5, 3));

        bac.load_tensor_u8(&mut a, &[
            0,  1,  2,  3,  4,  
            5,  6,  7,  8,  9, 
            10, 11, 12, 13, 14, 
            15, 16, 17, 18, 19, 
            20, 21, 22, 23, 24, 
            25, 26, 27, 28, 29, 
            30, 31, 32, 33, 34, 
            35, 36, 37, 38, 39
        ]);

        bac.load_tensor_u8(&mut b, &[
            0,  1,  2,  
            3,  4,  5,  
            6,  7,  8,  
            9, 10, 11,
            12, 13, 14, 
            15, 16, 17, 
            18, 19, 20, 
            21, 22, 23
        ]);

        bac.matmul_tn(&mut c, &a, &b);

        assert!(
            c.read() == &[
                2100.0, 2240.0, 2380.0,
                2184.0, 2332.0, 2480.0,
                2268.0, 2424.0, 2580.0,
                2352.0, 2516.0, 2680.0,
                2436.0, 2608.0, 2780.0
            ]
        );
    }


    #[test]
    fn test_axpy() {
        let bac = Native;

        let mut a = NativeTensorF32::new((2, 2));
        let mut b = NativeTensorF32::new((2, 2));

        bac.load_tensor_u8(&mut a, &[1, 2, 3, 4]);
        bac.load_tensor_u8(&mut b, &[1, 2, 3, 4]);

        bac.axpy(&mut a, 2.0f32, &b);

        assert!(
            a.read() == &[3.0, 6.0, 9.0, 12.0]
        );
    } 

    #[test]
    fn test_add() {
        let bac = Native;

        let mut a = NativeTensorF32::new((2, 2));
        let mut b = NativeTensorF32::new((2, 2));

        bac.load_tensor_u8(&mut a, &[1, 2, 3, 4]);
        bac.load_tensor_u8(&mut b, &[1, 2, 3, 4]);

        bac.add(&mut a, &b);
        
        assert!(
            a.read() == &[2.0, 4.0, 6.0, 8.0]
        );
    } 


    // #[test]
    // fn test_conv2d() {
    //     let bac = Native;

    //     let mut x = NativeTensorF32::new((1, 12, 12, 1));
    //     let mut y = NativeTensorF32::new((1, 12, 12, 2));
    //     let mut f = NativeTensorF32::new((3, 3, 2));

    //     bac.load_tensor_u8(&mut x, &[
    //         0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  
    //         12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,
    //         24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,
    //         36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
    //         48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
    //         60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,
    //         72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
    //         84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
    //         96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107,
    //         108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    //         120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
    //         132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
    //     ]);


    //     bac.load_tensor_f32(&mut f, &[
    //         0.5, 0.1,  0.6, 0.3,  0.5, 0.8,
    //         0.6, 0.4,  0.5, 0.9,  0.6, 0.4,
    //         0.5, 0.8,  0.6, 0.3,  0.5, 0.1,
    //     ]);

    //     let info = Conv2dInfo {
    //         kernel: (3, 3),
    //         padding: PaddingKind::Valid,
    //         strides: (1, 1),
    //     };

    //     bac.conv2d_forward(&mut y, &x, &f, &info);

    //     assert!(
    //         slice_eq(y.read(), &[
    //             14.3,   5.3,  22.5,  16.6,  25.8,  19.5,  29.1,  22.4,  32.4,  25.3,  35.7,  28.2,  
    //             39.0,  31.1,  42.3,  34.0,  45.6,  36.9,  48.9,  39.8,  52.2,  42.7,  36.3,  38.4,
    //             41.2,  26.5,  63.7,  53.3,  68.6,  57.4,  73.5,  61.5,  78.4,  65.6,  83.3,  69.7,  
    //             88.2,  73.8,  93.1,  77.9,  98.0,  82.0, 102.9,  86.1, 107.8,  90.2,  74.3,  71.5,
    //             80.8,  60.1, 122.5, 102.5, 127.4, 106.6, 132.3, 110.7, 137.2, 114.8, 142.1, 118.9, 
    //             147.0, 123.0, 151.9, 127.1, 156.8, 131.2, 161.7, 135.3, 166.6, 139.4, 113.9, 105.1,
    //             120.4,  93.7, 181.3, 151.7, 186.2, 155.8, 191.1, 159.9, 196.0, 164.0, 200.9, 168.1, 
    //             205.8, 172.2, 210.7, 176.3, 215.6, 180.4, 220.5, 184.5, 225.4, 188.6, 153.5, 138.7,
    //             160.0, 127.3, 240.1, 200.9, 245.0, 205.0, 249.9, 209.1, 254.8, 213.2, 259.7, 217.3, 
    //             264.6, 221.4, 269.5, 225.5, 274.4, 229.6, 279.3, 233.7, 284.2, 237.8, 193.1, 172.3,
    //             199.6, 160.9, 298.9, 250.1, 303.8, 254.2, 308.7, 258.3, 313.6, 262.4, 318.5, 266.5, 
    //             323.4, 270.6, 328.3, 274.7, 333.2, 278.8, 338.1, 282.9, 343.0, 287.0, 232.7, 205.9,
    //             239.2, 194.5, 357.7, 299.3, 362.6, 303.4, 367.5, 307.5, 372.4, 311.6, 377.3, 315.7, 
    //             382.2, 319.8, 387.1, 323.9, 392.0, 328.0, 396.9, 332.1, 401.8, 336.2, 272.3, 239.5,
    //             278.8, 228.1, 416.5, 348.5, 421.4, 352.6, 426.3, 356.7, 431.2, 360.8, 436.1, 364.9, 
    //             441.0, 369.0, 445.9, 373.1, 450.8, 377.2, 455.7, 381.3, 460.6, 385.4, 311.9, 273.1,
    //             318.4, 261.7, 475.3, 397.7, 480.2, 401.8, 485.1, 405.9, 490.0, 410.0, 494.9, 414.1, 
    //             499.8, 418.2, 504.7, 422.3, 509.6, 426.4, 514.5, 430.5, 519.4, 434.6, 351.5, 306.7,
    //             358.0, 295.3, 534.1, 446.9, 539.0, 451.0, 543.9, 455.1, 548.8, 459.2, 553.7, 463.3, 
    //             558.6, 467.4, 563.5, 471.5, 568.4, 475.6, 573.3, 479.7, 578.2, 483.8, 391.1, 340.3,
    //             397.6, 328.9, 592.9, 496.1, 597.8, 500.2, 602.7, 504.3, 607.6, 508.4, 612.5, 512.5, 
    //             617.4, 516.6, 622.3, 520.7, 627.2, 524.8, 632.1, 528.9, 637.0, 533.0, 430.7, 373.9,
    //             278.3, 304.8, 419.7, 372.0, 423.0, 374.9, 426.3, 377.8, 429.6, 380.7, 432.9, 383.6, 
    //             436.2, 386.5, 439.5, 389.4, 442.8, 392.3, 446.1, 395.2, 449.4, 398.1, 300.3, 237.8
    //         ])
    //     );
    // } 


    // #[test]
    // fn test_conv2d_backward_input() {
    //     let bac = Native;

    //     let mut dx = NativeTensorF32::new((1, 12, 12, 1));
    //     let mut dy = NativeTensorF32::new((1, 12, 12, 2));
    //     let mut f = NativeTensorF32::new((3, 3, 2));

    //     bac.load_tensor_f32(&mut dy, &[
    //         0.0,  0.0,   1.0,  0.1,   2.0,  0.2,   3.0,  0.3,   4.0,  0.4,   5.0,  0.5,   6.0,  0.6,   7.0,  0.7,   8.0,  0.8,   9.0,  0.9,  10.0,  1.0,  11.0,  1.1,
    //         12.0,  1.2,  13.0,  1.3,  14.0,  1.4,  15.0,  1.5,  16.0,  1.6,  17.0,  1.7,  18.0,  1.8,  19.0,  1.9,  20.0,  2.0,  21.0,  2.1,  22.0,  2.2,  23.0,  2.3,
    //         24.0,  2.4,  25.0,  2.5,  26.0,  2.6,  27.0,  2.7,  28.0,  2.8,  29.0,  2.9,  30.0,  3.0,  31.0,  3.1,  32.0,  3.2,  33.0,  3.3,  34.0,  3.4,  35.0,  3.5,
    //         36.0,  3.6,  37.0,  3.7,  38.0,  3.8,  39.0,  3.9,  40.0,  4.0,  41.0,  4.1,  42.0,  4.2,  43.0,  4.3,  44.0,  4.4,  45.0,  4.5,  46.0,  4.6,  47.0,  4.7,
    //         48.0,  4.8,  49.0,  4.9,  50.0,  5.0,  51.0,  5.1,  52.0,  5.2,  53.0,  5.3,  54.0,  5.4,  55.0,  5.5,  56.0,  5.6,  57.0,  5.7,  58.0,  5.8,  59.0,  5.9,
    //         60.0,  6.0,  61.0,  6.1,  62.0,  6.2,  63.0,  6.3,  64.0,  6.4,  65.0,  6.5,  66.0,  6.6,  67.0,  6.7,  68.0,  6.8,  69.0,  6.9,  70.0,  7.0,  71.0,  7.1,
    //         72.0,  7.2,  73.0,  7.3,  74.0,  7.4,  75.0,  7.5,  76.0,  7.6,  77.0,  7.7,  78.0,  7.8,  79.0,  7.9,  80.0,  8.0,  81.0,  8.1,  82.0,  8.2,  83.0,  8.3,
    //         84.0,  8.4,  85.0,  8.5,  86.0,  8.6,  87.0,  8.7,  88.0,  8.8,  89.0,  8.9,  90.0,  9.0,  91.0,  9.1,  92.0,  9.2,  93.0,  9.3,  94.0,  9.4,  95.0,  9.5,
    //         96.0,  9.6,  97.0,  9.7,  98.0,  9.8,  99.0,  9.9, 100.0, 10.0, 101.0, 10.1, 102.0, 10.2, 103.0, 10.3, 104.0, 10.4, 105.0, 10.5, 106.0, 10.6, 107.0, 10.7,
    //         108.0, 10.8, 109.0, 10.9, 110.0, 11.0, 111.0, 11.1, 112.0, 11.2, 113.0, 11.3, 114.0, 11.4, 115.0, 11.5, 116.0, 11.6, 117.0, 11.7, 118.0, 11.8, 119.0, 11.9,
    //         120.0, 12.0, 121.0, 12.1, 122.0, 12.2, 123.0, 12.3, 124.0, 12.4, 125.0, 12.5, 126.0, 12.6, 127.0, 12.7, 128.0, 12.8, 129.0, 12.9, 130.0, 13.0, 131.0, 13.1,
    //         132.0, 13.2, 133.0, 13.3, 134.0, 13.4, 135.0, 13.5, 136.0, 13.6, 137.0, 13.7, 138.0, 13.8, 139.0, 13.9, 140.0, 14.0, 141.0, 14.1, 142.0, 14.2, 143.0, 14.3,
    //     ]);

    //     bac.load_tensor_f32(&mut f, &[
    //         0.5, 0.1,  0.6, 0.3,  0.5, 0.8,
    //         0.6, 0.4,  0.5, 0.9,  0.6, 0.4,
    //         0.5, 0.8,  0.6, 0.3,  0.5, 0.1,
    //     ]);

    //     let info = Conv2dInfo {
    //         padding: PaddingKind::Valid,
    //         strides: (1, 1),
    //         kernel: (2, 2)
    //     };

    //     bac.conv2d_backward_input(&mut dx, &dy, &f, &info);

    //     assert!(
    //         slice_eq(dx.read(), &[
    //             14.83,  24.16,  27.75,  31.34,  34.93,  38.52,   42.11,  45.70,  49.29,  52.88,  56.47, 40.14, 
    //             43.85,  69.03,  74.34,  79.65,  84.96,  90.27,   95.58, 100.89, 106.20, 111.51, 116.82, 81.45, 
    //             86.81,  132.75, 138.06, 143.37, 148.68, 153.99, 159.30, 164.61, 169.92, 175.23, 180.54, 124.41, 
    //             129.77, 196.47, 201.78, 207.09, 212.4,  217.71, 223.02, 228.33, 233.64, 238.95, 244.26, 167.37, 
    //             172.73, 260.19, 265.5,  270.81, 276.12, 281.43, 286.74, 292.05, 297.36, 302.67, 307.98, 210.33, 
    //             215.69, 323.91, 329.22, 334.53, 339.84, 345.15, 350.46, 355.77, 361.08, 366.39, 371.7,  253.29, 
    //             258.65, 387.63, 392.94, 398.25, 403.56, 408.87, 414.18, 419.49, 424.80, 430.11, 435.42, 296.25, 
    //             301.61, 451.35, 456.66, 461.97, 467.28, 472.59, 477.90, 483.21, 488.52, 493.83, 499.14, 339.21, 
    //             344.57, 515.07, 520.38, 525.69, 531.00, 536.31, 541.62, 546.93, 552.24, 557.55, 562.86, 382.17, 
    //             387.53, 578.79, 584.1,  589.41, 594.72, 600.03, 605.3401, 610.65, 615.96, 621.27, 626.58, 425.13, 
    //             430.49, 642.51, 647.82, 653.13, 658.44, 663.75, 669.06, 674.37, 679.68, 684.99, 690.30, 468.09, 
    //             308.78, 456.9,  460.49, 464.08, 467.67, 471.26, 474.85, 478.44, 482.03, 485.62, 489.21, 324.08
    //         ])
    //     );
    // } 



    // #[test]
    // fn test_maxpool2d() {
    //     let bac = Native;

    //     let mut x = NativeTensorF32::new((1, 12, 12, 1));
    //     let mut y = NativeTensorF32::new((1, 6, 6, 1));

    //     bac.load_tensor_u8(&mut x, &[
    //         0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  
    //         12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,
    //         24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,
    //         36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
    //         48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
    //         60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,
    //         72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
    //         84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
    //         96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107,
    //         108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    //         120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
    //         132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
    //     ]);

    //     let info = Conv2dInfo {
    //         padding: PaddingKind::Valid,
    //         strides: (2, 2),
    //         kernel: (2, 2)
    //     };

    //     bac.max_pool2d(&mut y, &x, &info);

    //     assert!(
    //         slice_eq(y.read(), &[
    //             13.0,  15.0,  17.0,  19.0,  21.0,  23.0,
    //             37.0,  39.0,  41.0,  43.0,  45.0,  47.0,
    //             61.0,  63.0,  65.0,  67.0,  69.0,  71.0,
    //             85.0,  87.0,  89.0,  91.0,  93.0,  95.0,
    //             109.0, 111.0, 113.0, 115.0, 117.0, 119.0,
    //             133.0, 135.0, 137.0, 139.0, 141.0, 143.0,
    //         ])
    //     );
    // } 

    fn slice_eq(a: &[f32], b: &[f32]) -> bool {
        if a.len() != b.len() {
            return false;
        }

        let size = a.len();

        for i in 0..size {
            if (a[i] - b[i]).abs() > 0.0001 {
                return false;
            }
        }

        true
    }

    fn print_tensor(t: &NativeTensorF32, override_strides: Option<&[u32]>) {
        let tmp = t.shape.default_strides();
        let strides = if let Some(strides) = override_strides {
            strides
        } else {
            tmp.as_slice()
        };

        let last_idx = strides.len() - 1;

        println!("default stridses {} {}", t.shape.default_strides(), last_idx);
        print!("Tensor(shape={}, data=[", t.shape);

        for (idx, val) in t.read().iter().enumerate() {
            let is_first = idx == 0;
            let mut need_nl = false;
            let padding = 2;

            for (sidx, &s) in strides.iter().enumerate() {
                if sidx != last_idx && idx % s as usize == 0 {
                    need_nl = true;
                }
            }

            if !is_first {
                print!(", ");
            }

            if need_nl {
                print!("\n{}", " ".repeat(padding));
            }

            print!("{}", val);
        }

        print!("\n])");
    }
}