pub fn maxpool2d(y: &mut [f32], x: &[f32],
                 y_rows: isize, y_cols: isize, 
                 x_rows: isize, x_cols: isize, 
                 w_rows: isize, w_cols: isize, 
                 s_row: isize, s_col: isize) {
    
    // let y_rows = (x_rows - w_rows) / s_row + 1;
    // let y_cols = (x_cols - w_cols) / s_col + 1;

    let y = &mut y[0..(y_rows * y_cols) as usize];
    let x = &x[0..(x_rows * x_cols) as usize];
    
    for y_y in 0..y_rows {
        for y_x in 0..y_cols {
            let mut xi = s_row * y_y * x_cols + s_col * y_x;
            
            let mut max = std::f32::NEG_INFINITY;
            for _ in 0..w_rows {
                for w_x in 0..w_cols {
                    let val = x[(xi + w_x) as usize];
                    if val > max {
                        max = val;
                    }
                }
                
                xi += x_cols;
            }
            
            y[(y_y * y_cols + y_x) as usize] = max;
        }
    }
}

pub fn maxpool2d_backward(dx: &mut [f32], x: &[f32], dy: &[f32],
                       x_rows: isize, x_cols: isize,
                       y_rows: isize, y_cols: isize,
                       w_rows: isize, w_cols: isize, 
                       s_row: isize, s_col: isize) 
{
    // let y_cols = (x_cols - 1) * s_col + w_cols;
    // let y_rows = (x_rows - 1) * s_row + w_rows;

    let dx = &mut dx[0..(x_rows * x_cols) as usize];
    let x = &x[0..(x_rows * x_cols) as usize];
    let dy = &dy[0..(y_rows * y_cols) as usize];
    
    for dy_y in 0..y_rows {
        for dy_x in 0..y_cols {
            let mut xi = s_row * dy_y * x_cols + s_col * dy_x;

            let mut max = std::f32::NEG_INFINITY;
            let mut max_idx = -1;
            for _ in 0..w_rows {
                for w_x in 0..w_cols {
                    let idx = xi + w_x;
                    let val = x[idx as usize];
                    if val > max {
                        max = val;
                        max_idx = idx;
                    }
                }
                xi += x_cols;
            }
            
            dx[max_idx as usize] = dy[(dy_y * y_cols + dy_x) as usize];
        }
    }
}

pub fn avgpool2d(y: &mut [f32], x: &[f32],
                 x_rows: isize, x_cols: isize, 
                 w_rows: isize, w_cols: isize, 
                 s_row: isize, s_col: isize) {
    
    let w_size = w_rows * w_cols;
    let y_rows = (x_rows - w_rows) / s_row + 1;
    let y_cols = (x_cols - w_cols) / s_col + 1;

    let y = &mut y[0..(y_rows * y_cols) as usize];
    let x = &x[0..(x_rows * x_cols) as usize];
    
    for y_y in 0..y_rows {
        for y_x in 0..y_cols {
            let mut xi = s_row * y_y * x_cols + s_col * y_x;
            
            let mut sum = 0.0;

            for _ in 0..w_rows {
                for w_x in 0..w_cols {
                    sum += x[(xi + w_x) as usize];
                }
                
                xi += x_cols;
            }
            
            y[(y_y * y_cols + y_x) as usize] = sum / w_size as f32;
        }
    }
}


pub fn avgpool2d_backward(dx: &mut [f32], x: &[f32], dy: &[f32],
                          x_rows: isize, x_cols: isize,
                          y_rows: isize, y_cols: isize,
                          w_rows: isize, w_cols: isize, 
                          s_row: isize, s_col: isize) 
{
    // let y_cols = (x_cols - 1) * s_col + w_cols;
    // let y_rows = (x_rows - 1) * s_row + w_rows;

    let dx = &mut dx[0..(x_rows * x_cols) as usize];
    let x = &x[0..(x_rows * x_cols) as usize];
    let dy = &dy[0..(y_rows * y_cols) as usize];
    
    for dy_y in 0..y_rows {
        for dy_x in 0..y_cols {
            let mut xi = s_row * dy_y * x_cols + s_col * dy_x;

            let mut max = std::f32::NEG_INFINITY;
            let mut max_idx = -1;
            for _ in 0..w_rows {
                for w_x in 0..w_cols {
                    let idx = xi + w_x;
                    let val = x[idx as usize];
                    if val > max {
                        max = val;
                        max_idx = idx;
                    }
                }
                xi += x_cols;
            }
            
            dx[max_idx as usize] = dy[(dy_y * y_cols + dy_x) as usize];
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxpool2d_test() {
        let x: &[f32] = &[
            1.0,  2.0,  3.0,  4.0,  5.0,  6.0,
            7.0,  8.0,  9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 
            25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 
            31.0, 32.0, 33.0, 34.0, 35.0, 36.0,
        ];
        
        let y: &mut [f32] = &mut [
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ];
        
        maxpool2d(y, x, 6, 6, 2, 2, 2, 2);

        assert_eq!(y, &[
             8.0, 10.0, 12.0, 
            20.0, 22.0, 24.0, 
            32.0, 34.0, 36.0,
        ])
    }

    #[test]
    fn test_maxpool2d_backward() {
        let x: &[f32] = &[
            1.0,  2.0,  3.0,  4.0,  5.0,  6.0,
            7.0,  8.0,  9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 
            25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 
            31.0, 32.0, 33.0, 34.0, 35.0, 36.0,
        ];
        
        let dy: &[f32] = &[
            9.0, 8.0, 7.0, 
            6.0, 5.0, 4.0, 
            3.0, 2.0, 1.0
        ];
        
        let dx: &mut [f32] = &mut [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        
        maxpool2d_backward(dx, x, dy, 6, 6, 3, 3, 2, 2, 2, 2);

        let tt: &[f32] = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 9.0, 0.0, 8.0, 0.0, 7.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 6.0, 0.0, 5.0, 0.0, 4.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 3.0, 0.0, 2.0, 0.0, 1.0
        ]; 

        assert_eq!(dx, tt);
    }
}