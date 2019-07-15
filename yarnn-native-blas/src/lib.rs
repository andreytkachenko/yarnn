use yarnn::backend::*;

extern crate openblas_src;

pub struct NativeBlas<N, B: Native> {
    inner: B,
}

impl<N, B: Native> Native for NativeBlas<N, B> {}

impl<N, B> NativeBlas<N, B> 
    where N: NativeTensor,
          B: NativeBackend<N>
{
    pub fn new(native: B) -> Self {
        Self {
            inner: native
        }
    }
}

impl<N, B> Backend<N> for NativeBlas<N, B> 
    where N: NativeTensor,
          B: NativeBackend<N>
{
    type Tensor = B::Tensor;

    fn store_tensor_f32(&self, t: &Self::Tensor, data: &mut [f32]) {

    }
    fn load_tensor_u8(&self, t: &mut Self::Tensor, data: &[u8]) {

    }
    fn load_tensor_f32(&self, t: &mut Self::Tensor, data: &[f32]) {

    }
    fn scalar_f32(&self, val: f32) -> N {

    }
    fn fill_scalar(&self, t: &mut Self::Tensor, scalar: N) {

    }
    fn fill_random(&self, t: &mut Self::Tensor, from: N, to: N) {

    }
    fn print_tensor(&self, t: &Self::Tensor) {

    }
}

impl<B> BackendGemm<f32> for NativeBlas<f32, B> 
    where B: NativeBackend<f32>
{
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
            blas::sgemm('N' as u8, 'N' as u8,
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
            blas::sgemm('T' as u8, 'N' as u8,
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
            blas::sgemm('N' as u8, 'T' as u8,
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

impl<B> BackendAxpy<f32> for NativeBlas<f32, B> 
    where B: NativeBackend<f32>
{
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

impl<B> BackendScale<f32> for NativeBlas<f32, B> 
    where B: NativeBackend<f32>
{
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