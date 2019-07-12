
fn gemm_nn(m: usize, n: usize, k: usize, alpha: f32, 
           a: &[f32], lda: usize, 
           b: &[f32], ldb: usize,
           c: &mut [f32], ldc: usize)
{
    let a = &a[0..m * k];
    let b = &b[0..n * k];
    let c = &mut c[0..m * n];

    for i_m in 0..m {
        for i_k in 0..k {
            let a_part = alpha * a[i_m * lda + i_k];
            for i_n in 0..n {
                c[i_m * ldc + i_n] += a_part * b[i_k * ldb + i_n];
            }
        }
    }
}

fn gemm_nt(m: usize, n: usize, k: usize, alpha: f32, 
           a: &[f32], lda: usize, 
           b: &[f32], ldb: usize,
           c: &mut [f32], ldc: usize)
{
    let a = &a[0..m * k];
    let b = &b[0..n * k];
    let c = &mut c[0..m * n];

    for i_m in 0..m {
        for i_n in 0..n {
            let mut sum = 0.0;

            for i_k in 0..k {
                sum += alpha * a[i_m * lda + i_k] * b[i_n * ldb + i_k];
            }

            c[i_m * ldc + i_n] += sum;
        }
    }
}

fn gemm_tn(m: usize, n: usize, k: usize, alpha: f32, 
           a: &[f32], lda: usize, 
           b: &[f32], ldb: usize,
           c: &mut [f32], ldc: usize)
{    
    let a = &a[0..m * k];
    let b = &b[0..n * k];
    let c = &mut c[0..m * n];

    for i_m in 0..m {
        for i_k in 0..k {
            let a_part = alpha * a[i_k * lda + i_m];

            for i_n in 0..n {
                c[i_m * ldc + i_n] += a_part * b[i_k * ldb + i_n];
            }
        }
    }
}

fn gemm_tt(m: usize, n: usize, k: usize, alpha: f32, 
           a: &[f32], lda: usize, 
           b: &[f32], ldb: usize,
           c: &mut [f32], ldc: usize)
{
    let a = &a[0..m * k];
    let b = &b[0..n * k];
    let c = &mut c[0..m * n];
    
    for i_m in 0..m {
        for i_n in 0..n {
            let mut sum = 0.0;
            
            for i_k in 0..k {
                sum += alpha * a[i_k * lda + i_m] * b[i_n * ldb + i_k];
            }

            c[i_m * ldc + i_n] += sum;
        }
    }
}

pub fn gemm(ta: bool, tb: bool, m: usize, n: usize, k: usize, alpha: f32, 
        a: &[f32], lda: usize, 
        b: &[f32], ldb: usize, beta: f32,
        c: &mut [f32], ldc: usize)
{
    for i in 0..m {
        for j in 0..n {
            c[i * ldc + j] *= beta;
        }
    }

    if !ta && !tb {
        gemm_nn(m, n, k, alpha, a, lda, b, ldb, c, ldc);
    } else if ta && !tb {
        gemm_tn(m, n, k, alpha, a, lda, b, ldb, c, ldc);
    } else if !ta && tb {
        gemm_nt(m, n, k, alpha, a, lda, b, ldb, c, ldc);
    } else {
        gemm_tt(m, n, k, alpha, a, lda, b, ldb, c, ldc);
    }
}