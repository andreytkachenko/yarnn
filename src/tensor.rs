
use std::fmt;

pub struct TensorShapeIter<'a> {
    shape: &'a TensorShape,
    left: usize,
    right: usize,
} 

impl<'a> Iterator for TensorShapeIter<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.left > self.right {
            None
        } else {
            let idx = self.left;
            self.left += 1;

            Some(self.shape.shape[idx])
        }
    }
}

impl<'a> DoubleEndedIterator for TensorShapeIter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.right == 0 {
            None
        } else {
            let idx = self.right;
            self.right -= 1;

            Some(self.shape.shape[idx])
        }
    }
}

impl<'a> ExactSizeIterator for TensorShapeIter<'a> {
    fn len(&self) -> usize {
        (self.right + 1) - self.left
    }
}

#[derive(Clone, PartialEq)]
pub struct TensorShape {
    shape: [u32; 4],
    pub dims: usize,
}

impl fmt::Display for TensorShape {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(")?;
        for i in 0 .. self.dims {
            if i != 0 {
                write!(f, ", ")?;
            }

            write!(f, "{}", self.shape[i])?;
        }
        write!(f, ")")?;

        Ok(())
    }
}

impl TensorShape {
    pub fn zero() -> Self {
        TensorShape {
            shape: [0, 0, 0, 0],
            dims: 0,
        }
    }

    pub fn new0d() -> Self {
        TensorShape {
            shape: [1, 0, 0, 0],
            dims: 0,
        }
    }
    
    pub fn new1d(w: u32) -> Self {
        TensorShape {
            shape: [w, 0, 0, 0],
            dims: 1,
        }
    }

    pub fn new2d(h: u32, w: u32) -> Self {
        TensorShape {
            shape: [h, w, 0, 0],
            dims: 2,
        }
    }
    
    pub fn new3d(b: u32, h: u32, w: u32) -> Self {
        TensorShape {
            shape: [b, h, w, 0],
            dims: 3,
        }
    }
    
    pub fn new4d(b: u32, c: u32, h: u32, w: u32) -> Self {
        TensorShape {
            shape: [b, c, h, w],
            dims: 4,
        }
    }

    pub fn iter(&self) -> TensorShapeIter<'_> {
        TensorShapeIter {
            shape: self,
            left: 0,
            right: self.dims - 1,
        }
    }

    pub fn append<S: Into<TensorShape>>(&mut self, s: S) -> &mut Self {
        let s = s.into();
        let sd = self.dims;

        for i in 0 .. s.dims {
            self.shape[i + sd] = s.shape[i];
        }

        self.dims += s.dims;

        self
    }

    pub fn get(&self, index: usize) -> u32 {
        self.shape[index]
    }

    pub fn set(&mut self, index: usize, val: u32) {
        self.shape[index] = val;
    }

    pub fn slice<R: std::ops::RangeBounds<u32>>(&self, range: R) -> TensorShape {
        let mut shape = [0u32; 4];
        let mut dims = 0;

        for s in self.shape.iter() {
            if range.contains(s) {
                shape[dims] = *s;
                dims += 1;
            }
        }

        TensorShape {
            shape, 
            dims
        }
    }

    pub fn size(&self) -> usize {
        let mut product = 1;
        
        for i in 0 .. self.dims {
            product *= self.shape[i] as usize;
        }

        product
    }

    pub fn default_strides(&self) -> TensorShape {
        let mut strides = [0; 4];
        let mut product = 1;

        for i in  0..self.dims {
            let si = self.dims - i - 1;

            strides[si] = product;
            product *= self.shape[si]; 
        }

        TensorShape { shape: strides, dims: self.dims }
    }

    pub fn as_slice(&self) -> &[u32] {
        &self.shape[0..self.dims]
    }
}

impl From<()> for TensorShape {
    fn from(_: ()) -> Self {
        TensorShape {
            shape: [0, 0, 0, 0],
            dims: 0,
        }
    }
}

impl From<(u32, )> for TensorShape {
    fn from(x: (u32, )) -> Self {
        TensorShape {
            shape: [x.0, 0, 0, 0],
            dims: 1,
        }
    }
}

impl From<(u32, u32)> for TensorShape {
    fn from(x: (u32, u32)) -> Self {
        TensorShape {
            shape: [x.0, x.1, 0, 0],
            dims: 2,
        }
    }
}

impl From<(u32, u32, u32)> for TensorShape {
    fn from(x: (u32, u32, u32)) -> Self {
        TensorShape {
            shape: [x.0 , x.1, x.2, 0],
            dims: 3,
        }
    }
}

pub trait Tensor<N> {
    fn new<S: Into<TensorShape>>(shape: S) -> Self;
    fn shape(&self) -> &TensorShape;
    fn resize(&mut self, shape: TensorShape);
}
