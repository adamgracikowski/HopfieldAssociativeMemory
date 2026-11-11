//! Pattern sets checked by a capacity run.

use std::fmt;

use crate::patterns::Pattern;
use crate::seed::Seed;
use crate::{PatternSet, Result};

/// Where the patterns of a capacity task come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternSource {
    /// The first patterns of dataset `index` of the configuration.
    Prefix {
        /// Index into the configured datasets.
        index: usize,
    },
    /// Independent random patterns of `neurons` neurons.
    Random {
        /// Number of neurons `N`.
        neurons: usize,
    },
}

/// One pattern set of a capacity run: `patterns` patterns from `source`,
/// with the trial number and, for random patterns, the seed that draws
/// them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Task {
    /// Origin of the patterns.
    pub source: PatternSource,
    /// Number of patterns `p`.
    pub patterns: usize,
    /// Trial number; always 0 for a prefix, which is deterministic.
    pub trial: usize,
    /// Seed of the random patterns; `None` for a prefix.
    pub seed: Option<Seed>,
}

impl Task {
    /// Task with the first `patterns` patterns of dataset `index`.
    pub fn prefix(index: usize, patterns: usize) -> Self {
        Self {
            source: PatternSource::Prefix { index },
            patterns,
            trial: 0,
            seed: None,
        }
    }

    /// Task with `patterns` random patterns of `neurons` neurons, drawn from
    /// the seed derived from `master` and the text
    /// `random/<neurons>/<patterns>/<trial>`. The seed does not depend on
    /// the learning rule, so all rules see the same patterns.
    pub fn random(master: u64, neurons: usize, patterns: usize, trial: usize) -> Self {
        Self {
            source: PatternSource::Random { neurons },
            patterns,
            trial,
            seed: Some(Seed::derive(
                master,
                &[&"random", &neurons, &patterns, &trial],
            )),
        }
    }

    /// The pattern set of the task: the first `patterns` rows of
    /// `datasets[index]`, or random patterns drawn one after another from
    /// the input stream of the seed, as a `N x 1` bitmap.
    ///
    /// # Panics
    ///
    /// Panics if a prefix is longer than its dataset or refers to a missing
    /// dataset.
    pub fn pattern_set(&self, datasets: &[PatternSet]) -> Result<PatternSet> {
        match self.source {
            PatternSource::Prefix { index } => {
                let set = &datasets[index];
                assert!(self.patterns <= set.len(), "prefix longer than the dataset");
                set.prefix(self.patterns)
            }
            PatternSource::Random { neurons } => {
                let seed = self.seed.expect("a random task has a seed");
                let mut rng = seed.input_rng();
                let patterns = (0..self.patterns)
                    .map(|_| Pattern::random(neurons, &mut rng))
                    .collect::<Result<Vec<_>>>()?;
                PatternSet::from_patterns(neurons, 1, &patterns)
            }
        }
    }
}

impl fmt::Display for PatternSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Prefix { index } => write!(f, "dataset {index}"),
            Self::Random { neurons } => write!(f, "random (N = {neurons})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn prefix_takes_the_first_patterns() {
        let set = PatternSet::new(2, 1, array![[1, 1], [1, -1], [-1, -1]]).unwrap();
        let prefix = Task::prefix(0, 2).pattern_set(&[set]).unwrap();
        assert_eq!(prefix.patterns(), array![[1, 1], [1, -1]]);
        assert_eq!((prefix.width(), prefix.height()), (2, 1));
    }

    #[test]
    fn random_sets_repeat_with_the_seed_and_change_with_the_trial() {
        let set = |trial| Task::random(3, 40, 5, trial).pattern_set(&[]).unwrap();
        assert_eq!(set(0), set(0));
        assert_ne!(set(0), set(1));
        assert_eq!(set(0).len(), 5);
        assert_eq!(set(0).neurons(), 40);
        assert_ne!(
            Task::random(3, 40, 5, 0).seed,
            Task::random(3, 40, 6, 0).seed
        );
    }
}
