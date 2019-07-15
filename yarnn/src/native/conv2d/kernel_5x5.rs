#[allow(dead_code)]
pub fn valid_conv2d_5x5(y: &mut [f32], x: &[f32], w: &[f32], alpha: f32,
                        x_rows: isize, x_cols: isize, s_row: isize, s_col: isize) {
    
    let y_rows = (x_rows - 5) / s_row + 1;
    let y_cols = (x_cols - 5) / s_col + 1;

    let y = &mut y[0..(y_rows * y_cols) as usize];
    let x = &x[0..(x_rows * x_cols) as usize];
    let w = &w[0..25];
    
    for y_y in 0..y_rows {
        for y_x in 0..y_cols {
            let mut xi = s_row * y_y * x_cols + s_col * y_x;            
            let mut sum = 0.0;

            sum += x[(xi + 0) as usize] * w[0];
            sum += x[(xi + 1) as usize] * w[1];
            sum += x[(xi + 2) as usize] * w[2];
            sum += x[(xi + 3) as usize] * w[3];
            sum += x[(xi + 4) as usize] * w[4];
            xi += x_cols;
            
            sum += x[(xi + 0) as usize] * w[5];
            sum += x[(xi + 1) as usize] * w[6];
            sum += x[(xi + 2) as usize] * w[7];
            sum += x[(xi + 3) as usize] * w[8];
            sum += x[(xi + 4) as usize] * w[9];
            xi += x_cols;
            
            sum += x[(xi + 0) as usize] * w[10];
            sum += x[(xi + 1) as usize] * w[11];
            sum += x[(xi + 2) as usize] * w[12];
            sum += x[(xi + 3) as usize] * w[13];
            sum += x[(xi + 4) as usize] * w[14];
            xi += x_cols;
            
            sum += x[(xi + 0) as usize] * w[15];
            sum += x[(xi + 1) as usize] * w[16];
            sum += x[(xi + 2) as usize] * w[17];
            sum += x[(xi + 3) as usize] * w[18];
            sum += x[(xi + 4) as usize] * w[19];
            xi += x_cols;
            
            sum += x[(xi + 0) as usize] * w[20];
            sum += x[(xi + 1) as usize] * w[21];
            sum += x[(xi + 2) as usize] * w[22];
            sum += x[(xi + 3) as usize] * w[23];
            sum += x[(xi + 4) as usize] * w[24];

            y[(y_y * y_cols + y_x) as usize] += alpha * sum;
        }
    }
}


#[allow(dead_code)]
pub fn full_xcorr2d_5x5(y: &mut [f32], x: &[f32], w: &[f32], alpha: f32,
                        x_rows: isize, x_cols: isize, 
                        s_row: isize, s_col: isize) {
    
    let y_cols = (x_cols - 1) * s_col + 5;
    let y_rows = (x_rows - 1) * s_row + 5;

    let y = &mut y[0..(y_rows * y_cols) as usize];
    let x = &x[0..(x_rows * x_cols) as usize];
    let w = &w[0..25];
    
    for x_y in 0..x_rows {
        for x_x in 0..x_cols {
            let mut yi = s_row * x_y * y_cols + s_col * x_x;
            let z = alpha * x[(x_y * x_cols + x_x) as usize];

            y[(yi + 0) as usize] += z * w[24];
            y[(yi + 1) as usize] += z * w[23];
            y[(yi + 2) as usize] += z * w[22];
            y[(yi + 3) as usize] += z * w[21];
            y[(yi + 4) as usize] += z * w[20];
            yi += y_cols;
            
            y[(yi + 0) as usize] += z * w[19];
            y[(yi + 1) as usize] += z * w[18];
            y[(yi + 2) as usize] += z * w[17];
            y[(yi + 3) as usize] += z * w[16];
            y[(yi + 4) as usize] += z * w[15];
            yi += y_cols;
            
            y[(yi + 0) as usize] += z * w[14];
            y[(yi + 1) as usize] += z * w[13];
            y[(yi + 2) as usize] += z * w[12];
            y[(yi + 3) as usize] += z * w[11];
            y[(yi + 4) as usize] += z * w[10];
            yi += y_cols;
            
            y[(yi + 0) as usize] += z * w[9];
            y[(yi + 1) as usize] += z * w[8];
            y[(yi + 2) as usize] += z * w[7];
            y[(yi + 3) as usize] += z * w[6];
            y[(yi + 4) as usize] += z * w[5];
            yi += y_cols;
            
            y[(yi + 0) as usize] += z * w[4];
            y[(yi + 1) as usize] += z * w[3];
            y[(yi + 2) as usize] += z * w[2];
            y[(yi + 3) as usize] += z * w[1];
            y[(yi + 4) as usize] += z * w[0];
        }
    }
}

pub fn conv2d_forward_5x5(y: &mut [f32], x: &[f32], w: &[f32],
                          bs: isize, x_channels: isize, y_channels: isize,
                          x_rows: isize, x_cols: isize, s_row: isize, s_col: isize) {
    
    let y_rows = (x_rows - 5) / s_row + 1;
    let y_cols = (x_cols - 5) / s_col + 1;
    
    let x_img_size = x_rows * x_cols;
    let y_img_size = y_rows * y_cols;
    let w_img_size = 25;
    
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
                
                valid_conv2d_5x5(y_img, x_img, w, 1.0, x_rows, x_cols, s_row, s_col);
            }
        }   
    }
} 


pub fn conv2d_backward_5x5(dx: &mut [f32], dy: &[f32], w: &[f32],
                       bs: isize, x_channels: isize, y_channels: isize,
                       y_rows: isize, y_cols: isize,
                       s_row: isize, s_col: isize) {

    let x_cols = (y_cols - 1) * s_col + 5;
    let x_rows = (y_rows - 1) * s_row + 5;
    
    let dx_img_size = x_rows * x_cols;
    let dy_img_size = y_rows * y_cols;
    let w_img_size = 25;
    
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
                
                full_xcorr2d_5x5(dx_img, dy_img, w, 1.0, y_rows, y_cols, s_row, s_col);
            }
        }   
    }
}