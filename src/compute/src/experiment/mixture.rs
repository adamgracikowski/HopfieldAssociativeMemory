//! Mixtures of stored patterns, a kind of spurious state.

use crate::PatternSet;
use crate::patterns::PatternView;

/// Numbers of patterns in the checked mixtures; odd, so the sign of the sum
/// is always defined.
pub const MIXTURE_SIZES: [usize; 3] = [3, 5, 7];

/// A mixture state `sgn(sum_mu e_mu xi^mu)` of `size` stored patterns with
/// signs `e_mu`, and its Hamming distance to a state; odd symmetric mixtures
/// are stable in a Hebbian network with few patterns (Amit, Gutfreund and
/// Sompolinsky 1985).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mixture {
    /// Number of patterns, one of [`MIXTURE_SIZES`].
    pub size: usize,
    /// Indices of the patterns, in increasing order.
    pub patterns: Vec<usize>,
    /// Sign `e_mu` of every pattern, in the order of `patterns`.
    pub signs: Vec<i8>,
    /// Hamming distance of the state to the mixture.
    pub distance: usize,
}

impl Mixture {
    /// The closest mixture of size `k` in [`MIXTURE_SIZES`] (`k <= p`), each
    /// built from the `k` patterns with the largest `|m|` and the signs of
    /// their overlaps; the smaller mixture wins ties. The distance is an upper
    /// bound for correlated patterns. `None` for `p < 3`.
    ///
    /// # Panics
    ///
    /// Panics if `overlaps` does not have one value per pattern or `state`
    /// does not have `N` pixels.
    pub fn closest(state: PatternView<'_>, overlaps: &[f64], set: &PatternSet) -> Option<Self> {
        assert_eq!(overlaps.len(), set.len(), "one overlap per pattern");
        let mut order: Vec<usize> = (0..set.len()).collect();
        order.sort_by(|&a, &b| overlaps[b].abs().total_cmp(&overlaps[a].abs()));
        MIXTURE_SIZES
            .iter()
            .filter(|&&size| size <= set.len())
            .map(|&size| {
                let mut patterns = order[..size].to_vec();
                patterns.sort_unstable();
                let signs: Vec<i8> = patterns
                    .iter()
                    .map(|&mu| if overlaps[mu] < 0.0 { -1 } else { 1 })
                    .collect();
                let distance = Self::distance(state, set, &patterns, &signs);
                Self {
                    size,
                    patterns,
                    signs,
                    distance,
                }
            })
            .reduce(|best, next| {
                if next.distance < best.distance {
                    next
                } else {
                    best
                }
            })
    }

    /// Whether the state equals the mixture.
    pub fn is_exact(&self) -> bool {
        self.distance == 0
    }

    /// Hamming distance of `state` to `sgn(sum e_mu xi^mu)`.
    fn distance(
        state: PatternView<'_>,
        set: &PatternSet,
        patterns: &[usize],
        signs: &[i8],
    ) -> usize {
        let rows = set.patterns();
        state
            .values()
            .iter()
            .enumerate()
            .filter(|&(i, &x)| {
                let sum: i32 = patterns
                    .iter()
                    .zip(signs)
                    .map(|(&mu, &sign)| i32::from(sign) * i32::from(rows[[mu, i]]))
                    .sum();
                (sum > 0) != (x > 0)
            })
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patterns::Pattern;
    use ndarray::{Array1, Array2, array};

    /// Rows 0, 1 and 3 of the 8 x 8 Hadamard matrix, pairwise orthogonal.
    fn hadamard() -> PatternSet {
        PatternSet::new(
            8,
            1,
            array![
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, -1, 1, -1, 1, -1, 1, -1],
                [1, 1, -1, -1, 1, 1, -1, -1]
            ],
        )
        .unwrap()
    }

    fn closest(state: Array1<i8>, set: &PatternSet) -> Option<Mixture> {
        let state = Pattern::new(state).unwrap();
        let overlaps: Vec<f64> = set.iter().map(|xi| state.overlap(&xi)).collect();
        Mixture::closest(state.view(), &overlaps, set)
    }

    /// `xi^1 - xi^2 + xi^3 = (1, 3, -1, 1, 1, 3, -1, 1)`, so the mixture is
    /// `s = (1, 1, -1, 1, 1, 1, -1, 1)` with overlaps `4/8, -4/8, 4/8`.
    /// Flipping pixel 0 gives overlaps `2/8, -6/8, 2/8`: the same three
    /// patterns with the same signs, at distance 1.
    #[test]
    fn finds_a_mixture_of_three_patterns() {
        let set = hadamard();
        let exact = closest(array![1, 1, -1, 1, 1, 1, -1, 1], &set).unwrap();
        assert_eq!(
            exact,
            Mixture {
                size: 3,
                patterns: vec![0, 1, 2],
                signs: vec![1, -1, 1],
                distance: 0,
            }
        );
        assert!(exact.is_exact());
        let near = closest(array![-1, 1, -1, 1, 1, 1, -1, 1], &set).unwrap();
        assert_eq!((near.patterns, near.signs), (vec![0, 1, 2], vec![1, -1, 1]));
        assert_eq!(near.distance, 1);
        assert!(
            !Mixture {
                distance: 1,
                ..exact
            }
            .is_exact()
        );
    }

    /// With five orthogonal rows both sizes 3 and 5 are candidates. The sign
    /// of the sum of all five, `(1, 1, 1, 1, 1, -1, -1, -1)`, has overlaps
    /// `1/4` with rows 0-3 and `3/4` with row 4; the 3-mixture of rows 0, 1
    /// and 4 gives `(1, 1, 1, 1, 1, -1, 1, -1)`, distance 1, so the exact
    /// 5-mixture wins. The sign of the sum of rows 0-2, `(1, 1, 1, -1, 1,
    /// 1, 1, -1)`, has overlaps `1/2, 1/2, 1/2, -1/2, 0`: its 3-mixture is
    /// exact and wins the tie with any 5-mixture.
    #[test]
    fn prefers_the_closer_and_then_the_smaller_mixture() {
        let rows: Array2<i8> = array![
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, -1, 1, -1, 1, -1, 1, -1],
            [1, 1, -1, -1, 1, 1, -1, -1],
            [1, -1, -1, 1, 1, -1, -1, 1],
            [1, 1, 1, 1, -1, -1, -1, -1]
        ];
        let set = PatternSet::new(8, 1, rows.clone()).unwrap();
        let sum = |mus: &[usize]| -> Array1<i8> {
            (0..8)
                .map(|i| {
                    let s: i32 = mus.iter().map(|&mu| i32::from(rows[[mu, i]])).sum();
                    if s > 0 { 1 } else { -1 }
                })
                .collect()
        };
        let five = closest(sum(&[0, 1, 2, 3, 4]), &set).unwrap();
        assert_eq!((five.size, five.distance), (5, 0));
        let three = closest(sum(&[0, 1, 2]), &set).unwrap();
        assert_eq!((three.size, three.distance), (3, 0));
    }

    #[test]
    fn needs_at_least_three_patterns() {
        let set = PatternSet::new(2, 1, array![[1, 1], [1, -1]]).unwrap();
        assert_eq!(closest(array![1, 1], &set), None);
    }
}
