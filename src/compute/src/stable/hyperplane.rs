//! Bipolar states on a hyperplane through the origin.

use rayon::prelude::*;

use crate::patterns::Pattern;
use crate::{Error, Result};

/// Largest number of neurons whose states are enumerated (`2^N` states).
pub const MAX_NEURONS: usize = 32;

/// Number of low bits of a state index enumerated by one task, so that a
/// fold splits `2^N` states into the same chunks on every machine.
const CHUNK_BITS: usize = 16;

/// The hyperplane `u . x = 0` of a vector `u` of positive integers, and
/// its bipolar points `S_u = {xi in {-1, 1}^N : u . xi = 0}`.
///
/// States are enumerated by their index `0..2^N`: pixel `j` of state `k`
/// is `1` if bit `j` of `k` is set and `-1` otherwise. Changing the sign of
/// `u_j` maps `S_u` to itself with pixel `j` flipped, so positive weights
/// lose no set up to such a flip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hyperplane {
    weights: Vec<u32>,
    total: u64,
}

impl Hyperplane {
    /// Hyperplane of `weights`, which must be positive and at most
    /// [`MAX_NEURONS`].
    pub fn new(weights: Vec<u32>) -> Result<Self> {
        if weights.is_empty() || weights.len() > MAX_NEURONS {
            return Err(Error::InvalidArgument {
                message: format!(
                    "a hyperplane needs 1 to {MAX_NEURONS} weights, not {}",
                    weights.len()
                ),
            });
        }
        if weights.contains(&0) {
            return Err(Error::InvalidArgument {
                message: "the weights of a hyperplane must be positive".to_owned(),
            });
        }
        let total: u64 = weights.iter().map(|&w| u64::from(w)).sum();
        Ok(Self { weights, total })
    }

    /// Every non-decreasing vector of `neurons` weights from 1 to
    /// `max_weight`, in lexicographic order: one representative of every
    /// set of vectors that differ by a permutation, which permutes the
    /// pixels of `S_u` and changes neither its size nor its stability.
    pub fn candidates(neurons: usize, max_weight: u32) -> Result<Vec<Self>> {
        let mut current = Vec::with_capacity(neurons);
        fn extend(
            current: &mut Vec<u32>,
            neurons: usize,
            low: u32,
            max_weight: u32,
            out: &mut Vec<Vec<u32>>,
        ) {
            if current.len() == neurons {
                out.push(current.clone());
                return;
            }
            for weight in low..=max_weight {
                current.push(weight);
                extend(current, neurons, weight, max_weight, out);
                current.pop();
            }
        }
        let mut vectors = Vec::new();
        extend(&mut current, neurons, 1, max_weight, &mut vectors);
        vectors.into_iter().map(Self::new).collect()
    }

    /// The weights `u`.
    pub fn weights(&self) -> &[u32] {
        &self.weights
    }

    /// Number of neurons `N`.
    pub fn neurons(&self) -> usize {
        self.weights.len()
    }

    /// Size of `S_u`, counted without enumerating it: the number of sign
    /// vectors with `sum_j u_j xi_j = 0`, by dynamic programming over the
    /// partial sums of the weights with sign `+1`, which must equal half of
    /// `sum_j u_j` (none if that sum is odd).
    pub fn count(&self) -> u64 {
        if self.total % 2 == 1 {
            return 0;
        }
        let target = (self.total / 2) as usize;
        let mut ways = vec![0u64; target + 1];
        ways[0] = 1;
        for &weight in &self.weights {
            let weight = weight as usize;
            for sum in (weight..=target).rev() {
                ways[sum] += ways[sum - weight];
            }
        }
        ways[target]
    }

    /// Whether the state with index `index` lies on the hyperplane: the
    /// weights of its `+1` pixels add up to half of `sum_j u_j`.
    pub fn contains_index(&self, index: u64) -> bool {
        let mut sum = 0u64;
        let mut bits = index;
        while bits != 0 {
            let j = bits.trailing_zeros() as usize;
            sum += u64::from(self.weights[j]);
            bits &= bits - 1;
        }
        2 * sum == self.total
    }

    /// Writes the pixels of the state with index `index` into `state`.
    ///
    /// # Panics
    ///
    /// Panics if `state` does not have `N` elements.
    pub fn fill_state(index: u64, state: &mut [i8]) {
        for (j, pixel) in state.iter_mut().enumerate() {
            *pixel = if index >> j & 1 == 1 { 1 } else { -1 };
        }
    }

    /// Folds the members of `S_u` in index order within fixed chunks of
    /// `2^16` indices, the chunks in parallel, and returns one accumulator
    /// per chunk in chunk order, so that the caller can reduce them in an
    /// order that does not depend on the number of threads.
    pub fn fold_members<T: Send>(
        &self,
        init: impl Fn() -> T + Sync,
        fold: impl Fn(&mut T, &[i8]) + Sync,
    ) -> Vec<T> {
        let n = self.neurons();
        let chunk_bits = CHUNK_BITS.min(n);
        let chunks = 1u64 << (n - chunk_bits);
        (0..chunks)
            .into_par_iter()
            .map(|chunk| {
                let mut accumulator = init();
                let mut state = vec![0i8; n];
                let start = chunk << chunk_bits;
                for index in start..start + (1u64 << chunk_bits) {
                    if self.contains_index(index) {
                        Self::fill_state(index, &mut state);
                        fold(&mut accumulator, &state);
                    }
                }
                accumulator
            })
            .collect()
    }

    /// The members of `S_u` in index order.
    pub fn members(&self) -> impl Iterator<Item = Pattern> + '_ {
        let n = self.neurons();
        (0..1u64 << n)
            .filter(|&index| self.contains_index(index))
            .map(move |index| {
                let mut state = vec![0i8; n];
                Self::fill_state(index, &mut state);
                Pattern::try_from(ndarray::Array1::from(state)).expect("states are bipolar")
            })
    }

    /// A well-conditioned basis of the span of `S_u`, at most `N - 1`
    /// members (the dimension of the hyperplane). Gram-Schmidt with
    /// pivoting picks, from [`sample`](Self::sample) of `8 N` members, the
    /// one farthest from the span of those already picked (the first one
    /// on ties), so the basis is close to orthogonal; if these members do
    /// not span the hyperplane, the remaining members in index order fill
    /// it. A residual counts as independent when its squared norm exceeds
    /// `1e-6` (independent bipolar vectors of at most 32 pixels leave far
    /// larger ones). A nearly orthogonal basis matters for Oja's rule,
    /// which converges slowly along the small eigenvalues of the Gram
    /// matrix of its training set.
    pub fn basis(&self) -> Vec<Pattern> {
        let n = self.neurons();
        let target = n - 1;
        let pool = self.sample(8 * n);
        let mut residuals: Vec<Vec<f64>> = pool.iter().map(as_f64).collect();
        let mut used = vec![false; pool.len()];
        let mut orthonormal: Vec<Vec<f64>> = Vec::new();
        let mut basis = Vec::new();
        while basis.len() < target {
            let mut best: Option<(usize, f64)> = None;
            for (k, residual) in residuals.iter().enumerate() {
                let norm = squared_norm(residual);
                if !used[k] && norm > 1e-6 && best.is_none_or(|(_, b)| norm > b) {
                    best = Some((k, norm));
                }
            }
            let Some((k, norm)) = best else { break };
            used[k] = true;
            let q: Vec<f64> = residuals[k].iter().map(|r| r / norm.sqrt()).collect();
            for residual in &mut residuals {
                subtract_projection(residual, &q);
            }
            orthonormal.push(q);
            basis.push(pool[k].clone());
        }
        if basis.len() < target {
            for member in self.members() {
                if basis.len() == target {
                    break;
                }
                let mut residual = as_f64(&member);
                for q in &orthonormal {
                    subtract_projection(&mut residual, q);
                }
                let norm = squared_norm(&residual);
                if norm > 1e-6 {
                    orthonormal.push(residual.iter().map(|r| r / norm.sqrt()).collect());
                    basis.push(member);
                }
            }
        }
        basis
    }

    /// `count` members spread evenly over `S_u`: member number
    /// `floor(k |S_u| / count)` in index order for `k = 0, ..., count - 1`
    /// (all of them if `S_u` has at most `count` members).
    pub fn sample(&self, count: usize) -> Vec<Pattern> {
        let size = self.count();
        let wanted: Vec<u64> = if size <= count as u64 {
            (0..size).collect()
        } else {
            (0..count as u64).map(|k| k * size / count as u64).collect()
        };
        let mut next = wanted.iter().peekable();
        let mut out = Vec::with_capacity(wanted.len());
        for (rank, member) in self.members().enumerate() {
            match next.peek() {
                Some(&&target) if target == rank as u64 => {
                    out.push(member);
                    next.next();
                }
                Some(_) => {}
                None => break,
            }
        }
        out
    }
}

/// Pixels of `pattern` as `f64`.
fn as_f64(pattern: &Pattern) -> Vec<f64> {
    pattern.values().iter().map(|&x| f64::from(x)).collect()
}

/// Squared Euclidean norm of `v`.
fn squared_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum()
}

/// Removes from `v` its component along the unit vector `q`.
fn subtract_projection(v: &mut [f64], q: &[f64]) {
    let dot: f64 = v.iter().zip(q).map(|(a, b)| a * b).sum();
    v.iter_mut().zip(q).for_each(|(a, b)| *a -= dot * b);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute force: the states whose weighted sum is zero.
    fn brute_count(weights: &[u32]) -> u64 {
        let n = weights.len();
        (0..1u64 << n)
            .filter(|&index| {
                let sum: i64 = (0..n)
                    .map(|j| {
                        let x = if index >> j & 1 == 1 { 1 } else { -1 };
                        x * i64::from(weights[j])
                    })
                    .sum();
                sum == 0
            })
            .count() as u64
    }

    #[test]
    fn counts_the_hyperplane_points() {
        for weights in [
            vec![1, 1],
            vec![1, 1, 2],
            vec![1, 1, 1],
            vec![1, 2, 3, 3, 1],
            vec![1, 1, 1, 1, 1, 1, 1, 2],
            vec![2, 2, 2, 3, 3, 1, 1, 1, 3, 2],
        ] {
            let plane = Hyperplane::new(weights.clone()).unwrap();
            assert_eq!(plane.count(), brute_count(&weights), "{weights:?}");
            assert_eq!(plane.members().count() as u64, plane.count());
            let chunks = plane.fold_members(|| 0u64, |n, _| *n += 1);
            assert_eq!(chunks.iter().sum::<u64>(), plane.count());
        }
    }

    /// `u = (1, ..., 1, 2)` with 25 weights: the 24 ones must sum to `-2
    /// xi_25`, so 11 or 13 of them are `+1`: `2 * C(24, 11) = 2 * 2496144`.
    #[test]
    fn counts_the_large_hyperplane_without_enumerating_it() {
        let mut weights = vec![1; 24];
        weights.push(2);
        assert_eq!(Hyperplane::new(weights).unwrap().count(), 4_992_288);
    }

    /// Non-decreasing vectors of 3 weights from 1 to 2: 111, 112, 122, 222.
    #[test]
    fn lists_sorted_candidates() {
        let candidates = Hyperplane::candidates(3, 2).unwrap();
        let weights: Vec<&[u32]> = candidates.iter().map(Hyperplane::weights).collect();
        assert_eq!(
            weights,
            [&[1, 1, 1][..], &[1, 1, 2], &[1, 2, 2], &[2, 2, 2]]
        );
        assert_eq!(Hyperplane::candidates(25, 3).unwrap().len(), 351);
    }

    #[test]
    fn rejects_zero_and_too_many_weights() {
        assert!(Hyperplane::new(vec![1, 0]).is_err());
        assert!(Hyperplane::new(vec![1; 33]).is_err());
        assert!(Hyperplane::new(vec![]).is_err());
    }

    /// The basis has `N - 1` independent members of `S_u`, and a sample is
    /// spread over the members in index order.
    #[test]
    fn finds_a_basis_and_a_sample_of_members() {
        let plane = Hyperplane::new(vec![1, 1, 1, 1, 1, 1, 1, 1, 2]).unwrap();
        let basis = plane.basis();
        assert_eq!(basis.len(), 8);
        for pattern in &basis {
            let dot: i64 = pattern
                .values()
                .iter()
                .zip(plane.weights())
                .map(|(&x, &w)| i64::from(x) * i64::from(w))
                .sum();
            assert_eq!(dot, 0);
        }
        let members: Vec<Pattern> = plane.members().collect();
        let sample = plane.sample(4);
        let size = members.len();
        let expected: Vec<&Pattern> = (0..4).map(|k| &members[k * size / 4]).collect();
        assert_eq!(sample.iter().collect::<Vec<_>>(), expected);
        assert_eq!(plane.sample(1000).len(), size);
    }
}
