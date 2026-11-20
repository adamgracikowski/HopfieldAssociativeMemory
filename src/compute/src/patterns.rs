//! Sets of bipolar patterns (pixel values `1` and `-1`) shown as rectangular
//! bitmaps.

pub mod bipolar;
pub mod bitmap;
pub mod csv;
pub mod npy;
pub mod registry;
pub mod summary;

use ndarray::{Array2, ArrayView2, Axis};

pub use bipolar::{BipolarVector, Pattern, PatternView};
pub use bitmap::Bitmap;

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
        let neurons = bitmap_size(width, height)?;
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

    /// Creates a pattern set from separate patterns, in order, with the same
    /// checks as [`PatternSet::new`].
    pub fn from_patterns(width: usize, height: usize, patterns: &[Pattern]) -> Result<Self> {
        if patterns.is_empty() {
            return Err(shape("the set has no patterns".to_owned()));
        }
        let rows: Vec<_> = patterns.iter().map(|pattern| pattern.values()).collect();
        let data = ndarray::stack(Axis(0), &rows).map_err(|_| {
            shape(format!(
                "patterns have different lengths, expected {width}x{height} pixels each"
            ))
        })?;
        Self::new(width, height, data)
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

    /// Pattern `mu`, borrowed from the set without copying.
    ///
    /// # Panics
    ///
    /// Panics if `mu >= self.len()`.
    pub fn pattern(&self, mu: usize) -> PatternView<'_> {
        BipolarVector::new_unchecked(self.data.row(mu))
    }

    /// Pattern `mu` shown as a `width x height` bitmap.
    ///
    /// # Panics
    ///
    /// Panics if `mu >= self.len()`.
    pub fn bitmap(&self, mu: usize) -> Bitmap<'_> {
        Bitmap::new(self.pattern(mu), self.width, self.height)
            .expect("patterns of a set have `width * height` pixels")
    }

    /// All patterns in order, borrowed from the set.
    pub fn iter(&self) -> impl Iterator<Item = PatternView<'_>> {
        self.data
            .rows()
            .into_iter()
            .map(BipolarVector::new_unchecked)
    }

    /// Symmetric `p x p` matrix of overlaps `m^{mu nu}` (see
    /// [`BipolarVector::overlap`]), with ones on the diagonal.
    pub fn overlap_matrix(&self) -> Array2<f64> {
        let p = self.len();
        let mut matrix = Array2::eye(p);
        for mu in 0..p {
            for nu in mu + 1..p {
                let m = self.pattern(mu).overlap(&self.pattern(nu));
                matrix[[mu, nu]] = m;
                matrix[[nu, mu]] = m;
            }
        }
        matrix
    }

    /// The set of the first `patterns` patterns, or an error unless `1 <=
    /// patterns <= p`.
    pub fn prefix(&self, patterns: usize) -> Result<Self> {
        if patterns == 0 || patterns > self.len() {
            return Err(Error::InvalidArgument {
                message: format!(
                    "cannot take the first {patterns} of {} patterns",
                    self.len()
                ),
            });
        }
        let rows: Vec<usize> = (0..patterns).collect();
        Ok(Self {
            width: self.width,
            height: self.height,
            data: self.data.select(Axis(0), &rows),
        })
    }

    /// Consumes the set and returns the `p x N` matrix.
    pub fn into_inner(self) -> Array2<i8> {
        self.data
    }
}

/// Number of pixels `width * height`, or an error on overflow.
pub(crate) fn bitmap_size(width: usize, height: usize) -> Result<usize> {
    width
        .checked_mul(height)
        .ok_or_else(|| shape(format!("bitmap size {width}x{height} is too large")))
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
        assert_eq!(set.pattern(1).values(), array![-1, -1]);
        assert_eq!(set.iter().count(), 3);
    }

    #[test]
    fn takes_a_prefix() {
        let set = PatternSet::new(2, 1, array![[1, -1], [-1, -1], [1, 1]]).unwrap();
        let prefix = set.prefix(2).unwrap();
        assert_eq!(prefix.patterns(), array![[1, -1], [-1, -1]]);
        assert_eq!((prefix.width(), prefix.height()), (2, 1));
        assert_eq!(set.prefix(3).unwrap(), set);
        for patterns in [0, 4] {
            let error = set.prefix(patterns).unwrap_err();
            assert!(error.to_string().contains("of 3 patterns"), "{error}");
        }
    }

    /// 5x5 set: `a` all ones, `b` -1 at pixels 0-9, `c` 1 in the first
    /// column (pixels 0, 5, 10, 15, 20):
    /// - `a . b = 15 - 10 = 5`, overlap 0.2,
    /// - `a . c = 5 - 20 = -15`, overlap -0.6,
    /// - `b . c = (8 - 2) + (3 - 12) = -3` (pixels 0-9, 10-24), overlap -0.12.
    #[test]
    fn overlap_matrix_of_known_patterns() {
        let data = Array2::from_shape_fn((3, 25), |(mu, i)| match mu {
            0 => 1,
            1 if i < 10 => -1,
            1 => 1,
            _ if i % 5 == 0 => 1,
            _ => -1,
        });
        let m = PatternSet::new(5, 5, data).unwrap().overlap_matrix();
        let expected = array![[1.0, 0.2, -0.6], [0.2, 1.0, -0.12], [-0.6, -0.12, 1.0]];
        for (x, y) in m.iter().zip(&expected) {
            assert!((x - y).abs() < 1e-12, "{m}");
        }
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
    fn rejects_a_bitmap_size_that_overflows() {
        let error = PatternSet::new(usize::MAX, 2, Array2::zeros((1, 0))).unwrap_err();
        assert!(error.to_string().contains("is too large"), "{error}");
    }

    #[test]
    fn builds_a_set_from_separate_patterns() {
        let a = Pattern::new(array![1, -1, 1, 1]).unwrap();
        let b = Pattern::new(array![-1, -1, 1, -1]).unwrap();
        let set = PatternSet::from_patterns(2, 2, &[a.clone(), b]).unwrap();
        assert_eq!(set.len(), 2);
        assert_eq!(set.pattern(0), a);

        let short = Pattern::new(array![1, 1]).unwrap();
        assert!(PatternSet::from_patterns(2, 2, &[a.clone(), short]).is_err());
        assert!(PatternSet::from_patterns(3, 1, &[a]).is_err());
        assert!(PatternSet::from_patterns(2, 2, &[]).is_err());
    }

    #[test]
    fn rejects_an_empty_set() {
        assert!(PatternSet::new(2, 1, Array2::zeros((0, 2))).is_err());
        assert!(PatternSet::new(0, 1, Array2::zeros((1, 0))).is_err());
    }
}
