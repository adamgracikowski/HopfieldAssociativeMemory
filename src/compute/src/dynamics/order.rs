//! Order of neuron updates in the asynchronous mode.

use rand::seq::SliceRandom;
use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};

/// Order in which the asynchronous mode updates the neurons within one
/// sweep of `N` updates. Random orders draw from the generator passed to the
/// simulation, so a seeded generator makes every run reproducible.
///
/// [`Display`](std::fmt::Display) and `&'static str` conversion give the
/// name used in configuration files (`permutation`, `random`,
/// `sequential`).
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    Deserialize,
    Serialize,
    strum::Display,
    strum::IntoStaticStr,
    strum::EnumIter,
)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
pub enum UpdateOrder {
    /// A new random permutation of all neurons in every sweep, so every
    /// neuron is updated exactly once per sweep.
    #[default]
    Permutation,
    /// `N` neurons drawn uniformly with repetition, as in Hopfield's (1982)
    /// model with random update times. A sweep may skip some neurons.
    Random,
    /// Neurons `0, 1, ..., N - 1` in every sweep.
    Sequential,
}

impl UpdateOrder {
    /// Replaces the contents of `sweep` with the `n` neurons to update in
    /// the next sweep.
    pub fn fill_sweep<R: Rng + ?Sized>(self, sweep: &mut Vec<usize>, n: usize, rng: &mut R) {
        sweep.clear();
        match self {
            Self::Permutation => {
                sweep.extend(0..n);
                sweep.shuffle(rng);
            }
            Self::Random => sweep.extend((0..n).map(|_| rng.random_range(0..n))),
            Self::Sequential => sweep.extend(0..n),
        }
    }

    /// Whether every sweep updates every neuron, so that a sweep without
    /// changes proves a fixed point.
    pub fn covers_all_neurons(self) -> bool {
        !matches!(self, Self::Random)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use strum::IntoEnumIterator;

    fn sweeps(order: UpdateOrder, seed: u64) -> Vec<Vec<usize>> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..3)
            .map(|_| {
                let mut sweep = vec![99];
                order.fill_sweep(&mut sweep, 20, &mut rng);
                sweep
            })
            .collect()
    }

    #[test]
    fn permutation_updates_every_neuron_once_per_sweep() {
        let all = sweeps(UpdateOrder::Permutation, 1);
        for sweep in &all {
            let mut sorted = sweep.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..20).collect::<Vec<_>>());
        }
        assert_ne!(all[0], all[1]);
        assert!(UpdateOrder::Permutation.covers_all_neurons());
    }

    #[test]
    fn random_draws_n_neurons_in_range() {
        for sweep in sweeps(UpdateOrder::Random, 2) {
            assert_eq!(sweep.len(), 20);
            assert!(sweep.iter().all(|&i| i < 20));
        }
        assert!(!UpdateOrder::Random.covers_all_neurons());
    }

    #[test]
    fn sequential_is_fixed() {
        for sweep in sweeps(UpdateOrder::Sequential, 3) {
            assert_eq!(sweep, (0..20).collect::<Vec<_>>());
        }
    }

    #[test]
    fn same_seed_gives_the_same_sweeps() {
        for order in UpdateOrder::iter() {
            assert_eq!(sweeps(order, 5), sweeps(order, 5));
        }
        assert_ne!(
            sweeps(UpdateOrder::Random, 5),
            sweeps(UpdateOrder::Random, 6)
        );
    }

    #[test]
    fn names_round_trip_through_configuration() {
        #[derive(Deserialize)]
        struct Wrapper {
            order: UpdateOrder,
        }
        for order in UpdateOrder::iter() {
            let text = format!("order = \"{order}\"");
            assert_eq!(toml::from_str::<Wrapper>(&text).unwrap().order, order);
        }
        assert_eq!(UpdateOrder::default(), UpdateOrder::Permutation);
    }
}
