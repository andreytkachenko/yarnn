#[allow(dead_code)]
fn img_to_col_get_pixel(img: &[f32], img_rows: usize, img_cols: usize, 
                        mut row: isize, mut col: isize, channel: usize, 
                        pad_row: usize, pad_col: usize) -> f32 
{
    row -= pad_row as isize;
    col -= pad_col as isize;
    
    if row < 0 || col < 0 ||
       row >= img_rows as isize || 
       col >= img_cols as isize { return 0.0 }
       
    img[(channel * img_rows + row as usize) * img_cols + col as usize]
}

#[allow(dead_code)]
pub fn img_to_col(col: &mut [f32], img: &[f32], channels: usize,
              k_rows: usize, k_cols: usize,
              img_rows: usize, img_cols: usize, 
              s_row: usize, s_col: usize, 
              pad_row: usize, pad_col: usize) 
{
    let col_rows = (img_rows + 2 * pad_row - k_rows) / s_row + 1;
    let col_cols = (img_cols + 2 * pad_col - k_cols) / s_col + 1;
    
    let k_size = k_rows * k_cols;
    let channels_col = channels * k_size;
    
    let out_size = col_rows * col_cols;
    let col_size = channels_col * out_size;
    let col_s = &mut col[0..col_size];
    
    for ch in 0..channels_col {
        let offset_ch = ch / k_rows / k_cols;
        let offset_row = (ch / k_rows) % k_cols;
        let offset_col = ch % k_rows;
        
        for row in 0..col_rows {
            for col in 0..col_cols {
                let img_row = row * s_row + offset_row;
                let img_col = col * s_col + offset_col;
                
                let index_row = row * col_cols + col;
                let index_col = offset_row * k_rows + offset_col;
                let index = offset_ch * (k_size * out_size) + index_row * k_size + index_col;
                
                col_s[index] = img_to_col_get_pixel(
                    img, img_rows, img_cols, 
                    img_row as isize, img_col as isize,
                    offset_ch, pad_row, pad_col
                );
            }
        }
    }
}

fn col_to_img_add_pixel(img: &mut [f32], img_rows: usize, img_cols: usize, 
                        mut row: isize, mut col: isize, channel: usize, 
                        pad_row: usize, pad_col: usize, val: f32) {
                        
    row -= pad_row as isize;
    col -= pad_col as isize;
    
    if row < 0 || col < 0 ||
      row >= img_rows as isize || 
      col >= img_cols as isize { return; }
       
    img[(channel * img_rows + row as usize) * img_cols + col as usize] += val;
}

#[allow(dead_code)]
pub fn col_to_img(img: &mut [f32], col: &[f32], channels: usize,
              k_rows: usize, k_cols: usize,
              img_rows: usize, img_cols: usize, 
              s_row: usize, s_col: usize, 
              pad_row: usize, pad_col: usize) {

    let col_rows = (img_rows + 2 * pad_row - k_rows) / s_row + 1;
    let col_cols = (img_cols + 2 * pad_col - k_cols) / s_col + 1;
    
    let k_size = k_rows * k_cols;
    let channels_col = channels * k_size;
    
    let out_size = col_rows * col_cols;
    let col_size = channels_col * out_size;
    
    let col_s = &col[0..col_size];
    
    for ch in 0..channels_col {
        let offset_ch = ch / k_rows / k_cols;
        let offset_row = (ch / k_rows) % k_cols;
        let offset_col = ch % k_rows;
        
        for row in 0..col_rows {
            for col in 0..col_cols {
                let img_row = row * s_row + offset_row;
                let img_col = col * s_col + offset_col;
                
                let index_row = row * col_cols + col;
                let index_col = offset_row * k_rows + offset_col;
                let index = offset_ch * (k_size * out_size) + index_row * k_size + index_col;
                
                col_to_img_add_pixel(
                    img, img_rows, img_cols, 
                    img_row as isize, img_col as isize,
                    offset_ch, pad_row, pad_col, 
                    col_s[index]
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_img_to_col() {
        let img: &[f32] = &[
            1.0,  2.0,  3.0,  4.0,  
            5.0,  6.0,  7.0,  8.0,  
            9.0, 10.0, 11.0, 12.0, 
            13.0, 14.0, 15.0, 16.0,
            
            1.5,  2.5,  3.5,  4.5,  
            5.5,  6.5,  7.5,  8.5,  
            9.5, 10.5, 11.5, 12.5, 
            13.5, 14.5, 15.5, 16.5,
        ];
        
        let mut col = vec![0.0; 72];

        img_to_col(&mut col, img, 2, 3, 3, 4, 4, 1, 1, 0, 0);
        
        let tmp: &[f32] = &[
            1.0,  2.0,  3.0,  5.0,  6.0,  7.0,  9.0, 10.0, 11.0,
            2.0,  3.0,  4.0,  6.0,  7.0,  8.0, 10.0, 11.0, 12.0,
            5.0,  6.0,  7.0,  9.0, 10.0, 11.0, 13.0, 14.0, 15.0,
            6.0,  7.0,  8.0, 10.0, 11.0, 12.0, 14.0, 15.0, 16.0,
            1.5,  2.5,  3.5,  5.5,  6.5,  7.5,  9.5, 10.5, 11.5,
            2.5,  3.5,  4.5,  6.5,  7.5,  8.5, 10.5, 11.5, 12.5,
            5.5,  6.5,  7.5,  9.5, 10.5, 11.5, 13.5, 14.5, 15.5,
            6.5,  7.5,  8.5, 10.5, 11.5, 12.5, 14.5, 15.5, 16.5,
        ];
        
        assert_eq!(
            col.as_slice(),
            tmp
        );
    }

    #[test]
    fn test_col_to_img() {
        let col: &[f32] = &[
            1.0,  2.0,  3.0,  5.0,  6.0,  7.0,  9.0, 10.0, 11.0,
            2.0,  3.0,  4.0,  6.0,  7.0,  8.0, 10.0, 11.0, 12.0,
            5.0,  6.0,  7.0,  9.0, 10.0, 11.0, 13.0, 14.0, 15.0,
            6.0,  7.0,  8.0, 10.0, 11.0, 12.0, 14.0, 15.0, 16.0,
            1.5,  2.5,  3.5,  5.5,  6.5,  7.5,  9.5, 10.5, 11.5,
            2.5,  3.5,  4.5,  6.5,  7.5,  8.5, 10.5, 11.5, 12.5,
            5.5,  6.5,  7.5,  9.5, 10.5, 11.5, 13.5, 14.5, 15.5,
            6.5,  7.5,  8.5, 10.5, 11.5, 12.5, 14.5, 15.5, 16.5,
        ];

        let y: &mut [f32] = &mut [
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        
        col_to_img(y, col, 2, 3, 3, 4, 4, 1, 1, 0, 0);

        let tmp: &[f32] = &[
             1.0,  4.0,  6.0,  4.0, 
            10.0, 24.0, 28.0, 16.0, 
            18.0, 40.0, 44.0, 24.0, 
            13.0, 28.0, 30.0, 16.0, 
            
             1.5,  5.0,  7.0,  4.5, 
            11.0, 26.0, 30.0, 17.0, 
            19.0, 42.0, 46.0, 25.0, 
            13.5, 29.0, 31.0, 16.5,
        ];
        
        assert_eq!(y, tmp);
    }
}