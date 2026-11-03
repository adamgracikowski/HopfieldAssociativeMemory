//! Sets of bipolar patterns (pixel values `1` and `-1`) shown as rectangular
//! bitmaps.

pub mod csv;
pub mod npy;
pub mod registry;

use ndarray::{Array2, ArrayView1, ArrayView2};

use crate::{Error, Result};

/// A set of `p` bipolar patterns of `N = width * height` pixels each.
///
/// Patterns are the rows of a `p x N` matrix. Pixels of a bitmap are stored
/// row by row (row-major): pixel `(x, y)` has index `y * width + x`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatternSet {
    width: usize,
    height: usize,
    data: Array2<i8>,
}

impl PatternSet {
    /// Creates a pattern set after checking that it has at least one pattern,
    /// that every pattern has `width * height` pixels and that every pixel is
    /// `1` or `-1`.
    pub fn new(width: usize, height: usize, data: Array2<i8>) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(shape(format!("bitmap size {width}x{height} is empty")));
        }
        if data.nrows() == 0 {
            return Err(shape("the set has no patterns".to_owned()));
        }
        let neurons = width * height;
        if data.ncols() != neurons {
            return Err(shape(format!(
                "patterns have {} pixels, expected {width}x{height} = {neurons}",
                data.ncols()
            )));
        }
        if let Some(((mu, i), value)) = data.indexed_iter().find(|(_, v)| !matches!(**v, 1 | -1)) {
            return Err(shape(format!(
                "pattern {mu}, pixel {i} has value {value}, expected 1 or -1"
            )));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Bitmap width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Bitmap height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Number of patterns `p`.
    pub fn len(&self) -> usize {
        self.data.nrows()
    }

    /// Always `false`: a pattern set has at least one pattern.
    pub fn is_empty(&self) -> bool {
        self.data.nrows() == 0
    }

    /// Number of pixels of a pattern, equal to the number of neurons `N`.
    pub fn neurons(&self) -> usize {
        self.data.ncols()
    }

    /// All patterns as a `p x N` matrix.
    pub fn patterns(&self) -> ArrayView2<'_, i8> {
        self.data.view()
    }

    /// Pattern `mu` as a vector of `N` pixels.
    ///
    /// # Panics
    ///
    /// Panics if `mu >= self.len()`.
    pub fn pattern(&self, mu: usize) -> ArrayView1<'_, i8> {
        self.data.row(mu)
    }

    /// Consumes the set and returns the `p x N` matrix.
    pub fn into_inner(self) -> Array2<i8> {
        self.data
    }
}

fn shape(message: String) -> Error {
    Error::Shape { message }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn accepts_a_valid_set() {
        let set = PatternSet::new(2, 1, array![[1, -1], [-1, -1], [1, 1]]).unwrap();
        assert_eq!((set.width(), set.height()), (2, 1));
        assert_eq!((set.len(), set.neurons()), (3, 2));
        assert!(!set.is_empty());
        assert_eq!(set.pattern(1), array![-1, -1]);
    }

    #[test]
    fn rejects_a_size_mismatch() {
        let error = PatternSet::new(2, 2, array![[1, -1, 1]]).unwrap_err();
        assert!(error.to_string().contains("expected 2x2 = 4"), "{error}");
    }

    #[test]
    fn rejects_values_other_than_plus_minus_one() {
        let error = PatternSet::new(3, 1, array![[1, -1, 1], [1, 0, 1]]).unwrap_err();
        assert!(
            error.to_string().contains("pattern 1, pixel 1 has value 0"),
            "{error}"
        );
    }

    #[test]
    fn rejects_an_empty_set() {
        assert!(PatternSet::new(2, 1, Array2::zeros((0, 2))).is_err());
        assert!(PatternSet::new(0, 1, Array2::zeros((1, 0))).is_err());
    }
}
