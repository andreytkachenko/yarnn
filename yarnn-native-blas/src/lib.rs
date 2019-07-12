pub struct NativeBlas {
    inner: Native,
}

impl NativeBlas {

}

impl Backend<f32> for NativeBlas {
    type Tensor = NativeTensorF32;

    #[inline]
    fn store_tensor_f32(&self, t: &Self::Tensor, data: &mut [f32]) {
        self.inner.store_tensor_f32(t, data);
    }

    #[inline]
    fn load_tensor_u8(&self, t: &mut Self::Tensor, data: &[u8]) {
        self.inner.load_tensor_u8(t, data);
    }

    #[inline]
    fn load_tensor_f32(&self, t: &mut Self::Tensor, data: &[f32]) {
        self.inner.load_tensor_f32(t, data);
    }

    #[inline]
    fn scalar_f32(&self, val: f32) -> N {
        self.inner.scalar_f32(val);
    }

    #[inline]
    fn fill_scalar(&self, t: &mut Self::Tensor, scalar: N) {
        self.inner.scalar_f32(t, scalar);
    }

    #[inline]
    fn fill_random(&self, t: &mut Self::Tensor, from: N, to: N) {
        self.inner.scalar_f32(t, from, to);
    }

    #[inline]
    fn print_tensor(&self, t: &Self::Tensor) {
        self.inner.scalar_f32(t);
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
        
        unsafe {
            sgemm('N' as u8, 'N' as u8,
                n, m, k, 
                1.0, 
                b.read(), n, 
                a.read(), k, 
                0.0, 
                &mut dst.write(), n);
        }
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
        
        unsafe {
            sgemm('T' as u8, 'N' as u8,
                n, m, k, 
                1.0, 
                b.read(), k, 
                a.read(), k, 
                0.0, 
                &mut dst.write(), n);
        }
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
        
        unsafe {
            sgemm('N' as u8, 'T' as u8,
                n, m, k, 
                1.0, 
                b.read(), n, 
                a.read(), m, 
                0.0, 
                &mut dst.write(), n);
        }
    }

    fn matmul_tt(&self, _dst: &mut Self::Tensor, _a: &Self::Tensor, _b: &Self::Tensor) {
        unimplemented!();
    }
}

impl BackendAxpy<f32> for Native {
    fn axpy(&self, dst: &mut Self::Tensor, scale: f32, x: &Self::Tensor) {
        let dst_size = dst.shape().size();

        assert!(x.shape() == dst.shape());

        unsafe {
            blas::saxpy(
                dst_size as i32,
                scale,
                x.read(),
                1,
                dst.write(),
                1
            );
        }
    }
}

impl BackendScale<f32> for Native {
    fn scale(&self, dst: &mut Self::Tensor, scale: f32) {
        let dst_size = dst.shape().size();

        unsafe {
            blas::sscal(
                dst_size as i32,
                scale,
                dst.write(),
                1
            );
        }
    }
}