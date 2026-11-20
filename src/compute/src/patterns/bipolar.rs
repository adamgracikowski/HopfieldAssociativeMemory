//! Bipolar vectors: patterns and network states with pixels `1` and `-1`.

use std::fmt;

use ndarray::{Array1, ArrayBase, ArrayView1, Data, Ix1, OwnedRepr, ViewRepr};
use rand::seq::index::sample;
use rand::{Rng, RngExt};

use crate::{Error, Result};

/// A non-empty vector whose every element is `1` or `-1`.
///
/// Like `ndarray`'s `ArrayBase`, the type is generic over the storage `S`:
/// [`Pattern`] owns its pixels, [`PatternView`] borrows them (e.g. a row of a
/// [`PatternSet`](super::PatternSet)) without copying. Methods taking another
/// vector accept both kinds.
pub struct BipolarVector<S: Data<Elem = i8>>(ArrayBase<S, Ix1>);

/// Bipolar vector owning its pixels.
pub type Pattern = BipolarVector<OwnedRepr<i8>>;

/// Bipolar vector borrowing its pixels.
pub type PatternView<'a> = BipolarVector<ViewRepr<&'a i8>>;

impl<S: Data<Elem = i8>> BipolarVector<S> {
    /// Wraps `values` after checking that it is not empty and that every
    /// element is `1` or `-1`.
    pub fn new(values: ArrayBase<S, Ix1>) -> Result<Self> {
        if values.is_empty() {
            return Err(Error::InvalidArgument {
                message: "a bipolar vector must not be empty".to_owned(),
            });
        }
        if let Some((i, value)) = values.indexed_iter().find(|(_, v)| !matches!(**v, 1 | -1)) {
            return Err(Error::InvalidArgument {
                message: format!("element {i} has value {value}, expected 1 or -1"),
            });
        }
        Ok(Self(values))
    }

    /// Wraps values already known to be a non-empty bipolar vector.
    pub(crate) fn new_unchecked(values: ArrayBase<S, Ix1>) -> Self {
        debug_assert!(!values.is_empty() && values.iter().all(|v| matches!(*v, 1 | -1)));
        Self(values)
    }

    /// Number of pixels `N`.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Always `false`: a bipolar vector has at least one pixel.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Pixels as an `ndarray` view.
    pub fn values(&self) -> ArrayView1<'_, i8> {
        self.0.view()
    }

    /// Borrowed view of this vector.
    pub fn view(&self) -> PatternView<'_> {
        BipolarVector(self.0.view())
    }

    /// Owned copy of this vector.
    pub fn to_pattern(&self) -> Pattern {
        BipolarVector(self.0.to_owned())
    }

    /// Number of pixels in which `self` and `other` differ.
    ///
    /// # Panics
    ///
    /// Panics if the vectors have different lengths.
    pub fn hamming<T: Data<Elem = i8>>(&self, other: &BipolarVector<T>) -> usize {
        self.assert_same_len(other);
        self.0.iter().zip(&other.0).filter(|(x, y)| x != y).count()
    }

    /// Overlap `m = (1/N) sum_i a_i b_i` with `other`: `1` for equal vectors,
    /// `-1` for opposite ones and close to `0` for unrelated ones.
    ///
    /// # Panics
    ///
    /// Panics if the vectors have different lengths.
    pub fn overlap<T: Data<Elem = i8>>(&self, other: &BipolarVector<T>) -> f64 {
        self.assert_same_len(other);
        let sum: i64 = self
            .0
            .iter()
            .zip(&other.0)
            .map(|(&x, &y)| i64::from(x * y))
            .sum();
        sum as f64 / self.len() as f64
    }

    /// Fraction of pixels equal to `1`.
    pub fn fraction_on(&self) -> f64 {
        self.0.iter().filter(|&&x| x == 1).count() as f64 / self.len() as f64
    }

    /// Copy with exactly `k` distinct pixels flipped, chosen uniformly at
    /// random.
    pub fn flipped<R: Rng + ?Sized>(&self, k: usize, rng: &mut R) -> Result<Pattern> {
        let n = self.len();
        if k > n {
            return Err(Error::InvalidArgument {
                message: format!("cannot flip {k} pixels of a pattern with {n} pixels"),
            });
        }
        let mut flipped = self.0.to_owned();
        for i in sample(rng, n, k) {
            flipped[i] = -flipped[i];
        }
        Ok(BipolarVector(flipped))
    }

    fn assert_same_len<T: Data<Elem = i8>>(&self, other: &BipolarVector<T>) {
        assert_eq!(
            self.len(),
            other.len(),
            "bipolar vectors must have the same length"
        );
    }
}

impl Pattern {
    /// Random pattern of `n` independent pixels, each `1` or `-1` with
    /// probability 1/2.
    pub fn random<R: Rng + ?Sized>(n: usize, rng: &mut R) -> Result<Self> {
        Self::new(Array1::from_shape_fn(n, |_| {
            if rng.random::<bool>() { 1 } else { -1 }
        }))
    }

    /// Consumes the pattern and returns its pixels.
    pub fn into_inner(self) -> Array1<i8> {
        self.0
    }
}

impl TryFrom<Array1<i8>> for Pattern {
    type Error = Error;

    fn try_from(values: Array1<i8>) -> Result<Self> {
        Self::new(values)
    }
}

impl<S: Data<Elem = i8>> std::ops::Neg for &BipolarVector<S> {
    type Output = Pattern;

    /// Opposite vector: every pixel flipped.
    fn neg(self) -> Pattern {
        BipolarVector(-&self.0)
    }
}

impl<S, T> PartialEq<BipolarVector<T>> for BipolarVector<S>
where
    S: Data<Elem = i8>,
    T: Data<Elem = i8>,
{
    fn eq(&self, other: &BipolarVector<T>) -> bool {
        self.0 == other.0
    }
}

impl<S: Data<Elem = i8>> Eq for BipolarVector<S> {}

impl Clone for Pattern {
    fn clone(&self) -> Self {
        BipolarVector(self.0.clone())
    }
}

impl Clone for PatternView<'_> {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for PatternView<'_> {}

impl<S: Data<Elem = i8>> fmt::Debug for BipolarVector<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("BipolarVector").field(&self.0).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// 5x5 pattern of ones and a copy with the first 10 pixels set to -1.
    fn pair() -> (Pattern, Pattern) {
        let a = Pattern::new(Array1::from_elem(25, 1)).unwrap();
        let b = Pattern::new(Array1::from_shape_fn(25, |i| if i < 10 { -1 } else { 1 })).unwrap();
        (a, b)
    }

    #[test]
    fn new_checks_the_values() {
        assert!(Pattern::new(array![1, -1, 1]).is_ok());
        assert!(Pattern::new(array![1, 0, 1]).is_err());
        assert!(Pattern::new(Array1::zeros(0)).is_err());
        assert!(Pattern::try_from(array![2]).is_err());
    }

    #[test]
    fn hamming_counts_differing_pixels() {
        let (a, b) = pair();
        assert_eq!(a.hamming(&b), 10);
        assert_eq!(a.hamming(&a.view()), 0);
    }

    /// `a` and `b` have 15 equal and 10 different pixels: (15 - 10) / 25.
    #[test]
    fn overlap_of_known_patterns() {
        let (a, b) = pair();
        assert_eq!(a.overlap(&b), 0.2);
        assert_eq!(a.overlap(&a), 1.0);
        assert_eq!(a.overlap(&-&a), -1.0);
    }

    #[test]
    fn fraction_on_counts_ones() {
        let (a, b) = pair();
        assert_eq!(a.fraction_on(), 1.0);
        assert_eq!(b.fraction_on(), 0.6);
    }

    #[test]
    fn flipped_flips_exactly_k_pixels() {
        let (a, _) = pair();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        for k in [0, 1, 12, 25] {
            assert_eq!(a.hamming(&a.flipped(k, &mut rng).unwrap()), k);
        }
        assert!(matches!(
            a.flipped(26, &mut rng),
            Err(Error::InvalidArgument { .. })
        ));
    }

    #[test]
    fn same_seed_gives_the_same_result() {
        let (a, _) = pair();
        let run = |seed| {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            (
                a.flipped(7, &mut rng).unwrap(),
                Pattern::random(100, &mut rng).unwrap(),
            )
        };
        assert_eq!(run(5), run(5));
        assert_ne!(run(5), run(6));
    }

    #[test]
    fn random_pattern_is_bipolar_and_balanced() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let pattern = Pattern::random(10_000, &mut rng).unwrap();
        assert!(pattern.values().iter().all(|&x| x == 1 || x == -1));
        let on = pattern.fraction_on();
        assert!((0.48..0.52).contains(&on), "{on}");
        assert!(Pattern::random(0, &mut rng).is_err());
    }

    #[test]
    fn views_and_owned_vectors_compare_and_convert() {
        let (a, _) = pair();
        let view = a.view();
        assert_eq!(view, a);
        assert_eq!(view.to_pattern(), a);
        assert_eq!(a.clone().into_inner(), Array1::from_elem(25, 1));
    }
}
