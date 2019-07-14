#[allow(dead_code)]
pub fn valid_conv2d(y: &mut [f32], x: &[f32], w: &[f32], alpha: f32,
                    x_rows: isize, x_cols: isize, 
                    w_rows: isize, w_cols: isize, 
                    s_row: isize, s_col: isize) {
    
    let y_rows = (x_rows - w_rows) / s_row + 1;
    let y_cols = (x_cols - w_cols) / s_col + 1;

    let y = &mut y[0..(y_rows * y_cols) as usize];
    let x = &x[0..(x_rows * x_cols) as usize];
    let w = &w[0..(w_rows * w_cols) as usize];
    
    for y_y in 0..y_rows {
        for y_x in 0..y_cols {
            
            let mut xi = s_row * y_y * x_cols + s_col * y_x;
            let mut wi = 0;
            
            let mut sum = 0.0;
            for _ in 0..w_rows {
                for w_x in 0..w_cols {
                    sum += x[(xi + w_x) as usize] * w[(wi + w_x) as usize];
                }
                
                xi += x_cols;
                wi += w_cols;
            }

            y[(y_y * y_cols + y_x) as usize] += alpha * sum;
        }
    }
}

#[allow(dead_code)]
pub fn valid_xcorr2d(y: &mut [f32], x: &[f32], w: &[f32], alpha: f32,
                     x_rows: isize, x_cols: isize, 
                     w_rows: isize, w_cols: isize, 
                     s_row: isize, s_col: isize) {
    
    let y_rows = (x_rows - w_rows) / s_row + 1;
    let y_cols = (x_cols - w_cols) / s_col + 1;

    let y = &mut y[0..(y_rows * y_cols) as usize];
    let x = &x[0..(x_rows * x_cols) as usize];
    let w = &w[0..(w_rows * w_cols) as usize];
    
    for y_y in 0..y_rows {
        for y_x in 0..y_cols {
            
            let mut xi = s_row * y_y * x_cols + s_col * y_x;
            let mut wi = w_rows * w_cols - 1;
            
            let mut sum = 0.0;
            for _ in 0..w_rows {
                for w_x in 0..w_cols {
                    sum += x[(xi + w_x) as usize] * w[(wi - w_x) as usize];
                }
                
                xi += x_cols;
                wi -= w_cols;
            }

            y[(y_y * y_cols + y_x) as usize] += alpha * sum;
        }
    }
}

#[allow(dead_code)]
pub fn full_conv2d(y: &mut [f32], x: &[f32], w: &[f32], alpha: f32,
                   x_rows: isize, x_cols: isize, 
                   w_rows: isize, w_cols: isize, 
                   s_row: isize, s_col: isize) {
    
    let y_cols = (x_cols - 1) * s_col + w_cols;
    let y_rows = (x_rows - 1) * s_row + w_rows;

    let y = &mut y[0..(y_rows * y_cols) as usize];
    let x = &x[0..(x_rows * x_cols) as usize];
    let w = &w[0..(w_rows * w_cols) as usize];
    
    for x_y in 0..x_rows {
        for x_x in 0..x_cols {
            let mut yi = s_row * x_y * y_cols + s_col * x_x;
            let mut wi = 0;
            let z = alpha * x[(x_y * x_cols + x_x) as usize];
            
            for _ in 0..w_rows {
                for w_x in 0..w_cols {
                    y[(yi + w_x) as usize] += z * w[(wi + w_x) as usize];
                }
                
                yi += y_cols;
                wi += w_cols;
            }
        }
    }
}

#[allow(dead_code)]
pub fn full_xcorr2d(y: &mut [f32], x: &[f32], w: &[f32], alpha: f32,
                    x_rows: isize, x_cols: isize, 
                    w_rows: isize, w_cols: isize, 
                    s_row: isize, s_col: isize) {
    
    let y_cols = (x_cols - 1) * s_col + w_cols;
    let y_rows = (x_rows - 1) * s_row + w_rows;

    let y = &mut y[0..(y_rows * y_cols) as usize];
    let x = &x[0..(x_rows * x_cols) as usize];
    let w = &w[0..(w_rows * w_cols) as usize];
    
    for x_y in 0..x_rows {
        for x_x in 0..x_cols {
            let mut yi = s_row * x_y * y_cols + s_col * x_x;
            let mut wi = w_rows * w_cols - 1;
            let z = alpha * x[(x_y * x_cols + x_x) as usize];
            
            for _ in 0..w_rows {
                for w_x in 0..w_cols {
                    y[(yi + w_x) as usize] += z * w[(wi - w_x) as usize];
                }
                
                yi += y_cols;
                wi -= w_cols;
            }
        }
    }
}

pub fn conv2d_forward(y: &mut [f32], x: &[f32], w: &[f32],
                  bs: isize, x_channels: isize, y_channels: isize,
                  x_rows: isize, x_cols: isize,
                  w_rows: isize, w_cols: isize,
                  s_row: isize, s_col: isize) {
    
    let y_rows = (x_rows - w_rows) / s_row + 1;
    let y_cols = (x_cols - w_cols) / s_col + 1;
    
    let x_img_size = x_rows * x_cols;
    let y_img_size = y_rows * y_cols;
    let w_img_size = w_rows * w_cols;
    
    let x_batch_size = x_channels * x_img_size;
    let y_batch_size = y_channels * y_img_size;
    
    let y = &mut y[0..(bs * y_batch_size) as usize];
    let x = &x[0..(bs * x_batch_size) as usize];
    let w = &w[0..(y_channels * w_img_size) as usize];
    
    for bi in 0..bs {
        for x_ch in 0..x_channels {
            let x_offset = (bi * x_batch_size + x_ch * x_img_size) as usize;
            let x_img = &x[x_offset..x_offset + x_img_size as usize];

            for y_ch in 0..y_channels {
                let y_offset = (bi * y_batch_size + y_ch * y_img_size) as usize;
                let y_img = &mut y[y_offset..y_offset + y_img_size as usize];
                
                let w_offset = (y_ch * w_img_size) as usize;
                let w = &w[w_offset..w_offset + w_img_size as usize];
                
                valid_conv2d(y_img, x_img, w, 1.0, x_rows, x_cols, w_rows, w_cols, s_row, s_col);
            }
        }   
    }
} 

pub fn conv2d_backward(dx: &mut [f32], dy: &[f32], w: &[f32],
                  bs: isize, x_channels: isize, y_channels: isize,
                  y_rows: isize, y_cols: isize,
                  w_rows: isize, w_cols: isize,
                  s_row: isize, s_col: isize) {

    let x_cols = (y_cols - 1) * s_col + w_cols;
    let x_rows = (y_rows - 1) * s_row + w_rows;
    
    let dx_img_size = x_rows * x_cols;
    let dy_img_size = y_rows * y_cols;
    let w_img_size = w_rows * w_cols;
    
    let dx_batch_size = x_channels * dx_img_size;
    let dy_batch_size = y_channels * dy_img_size;
    
    let dx = &mut dx[0..(bs * dx_batch_size) as usize];
    let dy = &dy[0..(bs * dy_batch_size) as usize];
    let w = &w[0..(y_channels * w_img_size) as usize];
    
    for bi in 0..bs {
        for y_ch in 0..y_channels {
            let dy_offset = (bi * dy_batch_size + y_ch * dy_img_size) as usize;
            let dy_img = &dy[dy_offset..dy_offset + dy_img_size as usize];
            
            for x_ch in 0..x_channels {
                let dx_offset = (bi * dx_batch_size + x_ch * dx_img_size) as usize;
                let dx_img = &mut dx[dx_offset..dx_offset + dx_img_size as usize];
                
                let w_offset = (y_ch * w_img_size) as usize;
                let w = &w[w_offset..w_offset + w_img_size as usize];
                
                full_xcorr2d(dx_img, dy_img, w, 1.0, y_rows, y_cols, w_rows, w_cols, s_row, s_col);
            }
        }   
    }
}

pub fn conv2d_grads(dw: &mut [f32], x: &[f32], dy: &[f32], 
                  bs: isize, x_channels: isize, y_channels: isize,
                  x_rows: isize, x_cols: isize,
                  y_rows: isize, y_cols: isize,
                  s_row: isize, s_col: isize) {

    let w_cols = x_cols - y_cols + 1;
    let w_rows = x_rows - y_rows + 1;
    
    let x_img_size = x_rows * x_cols;
    let dy_img_size = y_rows * y_cols;
    let dw_img_size = w_rows * w_cols;
    
    let x_batch_size = x_channels * x_img_size;
    let dy_batch_size = y_channels * dy_img_size;
    
    let dw = &mut dw[0..(y_channels * dw_img_size) as usize];
    let dy = &dy[0..(bs * dy_batch_size) as usize];
    let x = &x[0..(bs * x_batch_size) as usize];
    
    for bi in 0..bs {
        for x_ch in 0..x_channels {
            let x_offset = (bi * x_batch_size + x_ch * x_img_size) as usize;
            let x_img = &x[x_offset..x_offset + x_img_size as usize];
            
            for y_ch in 0..y_channels {
                let dy_offset = (bi * dy_batch_size + y_ch * dy_img_size) as usize;
                let dy_img = &dy[dy_offset..dy_offset + dy_img_size as usize];
                
                let dw_offset = (y_ch * dw_img_size) as usize;
                let dw = &mut dw[dw_offset..dw_offset + dw_img_size as usize];
                
                valid_conv2d(dw, x_img, dy_img, 1.0, x_rows, x_cols, y_rows, y_cols, s_row, s_col);
            }
        }   
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_valid_conv2d() {
        let x: &[f32] = &[
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
            1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
            2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
            3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7,
            4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7,
            5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7,
            6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7,
        ];
        
        let w: &[f32] = &[
            0.1, 0.2, 0.3,
            0.4, 0.5, 0.6,
            0.7, 0.8, 0.9,
        ];
        
        let y: &mut [f32] = &mut [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]; 
        
        valid_conv2d(y, x, w, 1.0, 7, 8, 3, 3, 1, 1);

        assert_eq!(y, &[
            6.81, 7.26, 7.71, 8.16, 8.610001, 9.06, 
            11.31, 11.76, 12.210001, 12.66, 13.11, 13.56, 
            15.809999, 16.26, 16.710001, 17.160002, 17.61, 18.06, 
            20.31, 20.76, 21.210001, 21.66, 22.11, 22.559998, 
            24.81, 25.26, 25.710001, 26.160002, 26.61, 27.060001
        ])
    }

    #[test]
    fn test_valid_xcorr2d() {
        let x: &[f32] = &[
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
            1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
            2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
            3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7,
            4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7,
            5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7,
            6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7,
        ];
        
        let w: &[f32] = &[
            0.1, 0.2, 0.3,
            0.4, 0.5, 0.6,
            0.7, 0.8, 0.9,
        ];
        
        let y: &mut [f32] = &mut [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]; 
        
        valid_xcorr2d(y, x, w, 1.0, 7, 8, 3, 3, 1, 1);

        assert_eq!(y, &[
            3.0900004, 3.54, 3.9900002, 4.44, 4.8900003, 5.34, 
            7.59, 8.04, 8.490001, 8.940001, 9.389999, 9.84, 
            12.089999, 12.540001, 12.989999, 13.44, 13.889998, 14.340001, 
            16.59, 17.039999, 17.490002, 17.939999, 18.39, 18.84, 
            21.09, 21.539999, 21.99, 22.44, 22.89, 23.34
        ])
    }

    #[test]
    fn test_full_conv2d() {
        let x: &[f32] = &[
            1.0, 2.0,
            3.0, 4.0,
        ];
        
        let w: &[f32] = &[
            1.0, 2.0, 1.0,
            3.0, 2.0, 1.0,
            1.0, 4.0, 3.0,
        ];
        
        let y: &mut [f32] = &mut [
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ]; 
        
        full_conv2d(y, x, w, 1.0, 2, 2, 3, 3, 1, 1);

        assert_eq!(y, &[
             1.0,  4.0,  5.0,  2.0, 
             6.0, 18.0, 16.0,  6.0, 
            10.0, 24.0, 22.0, 10.0, 
             3.0, 16.0, 25.0, 12.0,
        ])
    }

    #[test]
    fn test_full_xcorr2d() {
        let x: &[f32] = &[
            1.0, 2.0,
            3.0, 4.0,
        ];
        
        let w: &[f32] = &[
            1.0, 2.0, 1.0,
            3.0, 2.0, 1.0,
            1.0, 4.0, 3.0,
        ];
        
        let y: &mut [f32] = &mut [
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ]; 
        
        full_xcorr2d(y, x, w, 1.0, 2, 2, 3, 3, 1, 1);

        assert_eq!(y, &[
             3.0, 10.0,  9.0,  2.0, 
            10.0, 28.0, 26.0, 10.0, 
             4.0, 14.0, 22.0, 14.0, 
             3.0, 10.0, 11.0,  4.0,
        ])
    }

    #[test]
    fn test_conv2d_grads() {
        let x: &[f32] = &[
            0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0,
            0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0,
        ];
        
        let true_y: &[f32] = &[
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            
            0.0, 0.0, 0.0,
            1.0, 1.0, 1.0,
            0.0, 0.0, 0.0,
        ];
        
        let w: &mut [f32] = &mut [
            0.02, -0.01, 0.01,
            -0.0012, 0.001, -0.005,
            0.021, -0.0001, 0.008,
            
            0.021, -0.0001, 0.008,
            -0.0012, 0.001, -0.005,
            0.02, -0.01, 0.01,
        ];
        
        for _ in 0..10 {
            let y: &mut [f32] = &mut [
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            ];
            
            let dw: &mut [f32] = &mut [
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            ];
        
            let dy: &mut [f32] = &mut [
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            ];
            
            conv2d_forward(y, x, w, 1, 1, 2, 5, 5, 3, 3, 1, 1);

            for i in 0..y.len() {
                dy[i] = y[i] - true_y[i];
            }

            conv2d_grads(dw, x, dy, 1, 1, 2, 5, 5, 3, 3, 1, 1);
            
            for i in 0..w.len() {
                w[i] -= dw[i] * 0.01;
            }
        }
        
        assert_eq!(w, &[
            0.018572275, 0.16292913, 0.011668432, 
            0.0014115912, 0.17488956, 0.00011491659, 
            0.018765995, 0.17117187, 0.009149003, 
            
            0.018765992, 0.0035881882, 0.009149002, 
            0.16899528, 0.17488956, 0.16769859,
            0.018572278, -0.004654532, 0.011668428
        ]);
    }
}