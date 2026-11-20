//! Execution of a search for stable hyperplane sets.

use std::cmp::Reverse;

use serde::Serialize;

use super::{HebbSum, Hyperplane, StableConfig, StableMetadata, StableProvenance, StrictStability};
use crate::learning::LearningRule;
use crate::run::OutputDir;
use crate::run::train::METADATA_FILE;
use crate::weights::Backend;
use crate::{PatternSet, Result};

/// Name of the file with the candidates and the checked sets.
pub const SUMMARY_FILE: &str = "summary.json";

/// A hyperplane vector `u` and the size of its set `S_u`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Candidate {
    /// Weights `u`, non-decreasing.
    pub weights: Vec<u32>,
    /// Size of `S_u`.
    pub size: u64,
}

/// A candidate checked for strict stability with both rules.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Verified {
    /// Weights `u`, non-decreasing.
    pub weights: Vec<u32>,
    /// Size of `S_u`.
    pub size: u64,
    /// The Hebbian rule trained on the whole set (see [`HebbSum`]).
    pub hebb: StrictStability,
    /// Oja's rule trained on a basis of the set.
    pub oja: StrictStability,
    /// Number of patterns of that basis.
    pub oja_patterns: usize,
    /// Training epochs of Oja's rule.
    pub oja_epochs: usize,
    /// Whether Oja's rule converged.
    pub oja_converged: bool,
}

impl Verified {
    /// Whether every member is strictly stable with both rules.
    pub fn all_stable(&self) -> bool {
        self.hebb.all_stable() && self.oja.all_stable()
    }
}

/// Contents of [`SUMMARY_FILE`].
#[derive(Serialize)]
struct Summary<'a> {
    neurons: usize,
    best: Option<&'a Verified>,
    verified: &'a [Verified],
    candidates: &'a [Candidate],
}

/// An executed search.
#[derive(Debug, Clone, PartialEq)]
pub struct StableSearch {
    /// Every candidate `u`, largest `S_u` first, ties in lexicographic order
    /// of `u`.
    pub candidates: Vec<Candidate>,
    /// The largest non-empty candidates, checked, in the same order.
    pub verified: Vec<Verified>,
    /// Index into `verified` of the largest set strictly stable with both
    /// rules, if any.
    pub best: Option<usize>,
    /// Basis of the best set (the training set of Oja's rule), as bitmaps.
    pub basis: Option<PatternSet>,
    /// Evenly spread members of the best set, as bitmaps.
    pub sample: Option<PatternSet>,
}

impl StableSearch {
    /// Counts `|S_u|` for every candidate `u` and checks the `verify` largest
    /// non-empty sets: Hebb exactly on the whole set ([`HebbSum`]) and Oja
    /// (dense) on its [`basis`](Hyperplane::basis). The best set is the
    /// first one strictly stable with both rules.
    pub fn execute(config: &StableConfig) -> Result<Self> {
        let planes = Hyperplane::candidates(config.neurons(), config.max_weight)?;
        let mut ranked: Vec<(u64, Hyperplane)> = planes
            .into_iter()
            .map(|plane| (plane.count(), plane))
            .collect();
        ranked
            .sort_by(|(a, x), (b, y)| (Reverse(*a), x.weights()).cmp(&(Reverse(*b), y.weights())));
        let candidates = ranked
            .iter()
            .map(|(size, plane)| Candidate {
                weights: plane.weights().to_vec(),
                size: *size,
            })
            .collect();

        let mut verified = Vec::new();
        let mut bases = Vec::new();
        for (size, plane) in ranked
            .iter()
            .filter(|(size, _)| *size > 0)
            .take(config.verify)
        {
            let hebb = HebbSum::of(plane).strict_stability(plane);
            let basis = plane.basis();
            let basis = PatternSet::from_patterns(config.width, config.height, &basis)?;
            let trained = config.oja.train(&basis, Backend::Dense)?;
            let oja = StrictStability::of(&trained.weights, plane);
            verified.push(Verified {
                weights: plane.weights().to_vec(),
                size: *size,
                hebb,
                oja,
                oja_patterns: basis.len(),
                oja_epochs: trained.epochs,
                oja_converged: trained.converged,
            });
            bases.push((plane, basis));
        }
        let best = verified.iter().position(Verified::all_stable);
        let (basis, sample) = match best {
            Some(index) => {
                let (plane, basis) = &bases[index];
                let sample = plane.sample(config.samples);
                let sample = PatternSet::from_patterns(config.width, config.height, &sample)?;
                (Some(basis.clone()), Some(sample))
            }
            None => (None, None),
        };
        Ok(Self {
            candidates,
            verified,
            best,
            basis,
            sample,
        })
    }

    /// The best checked set, if any.
    pub fn best(&self) -> Option<&Verified> {
        self.best.map(|index| &self.verified[index])
    }

    /// Writes `metadata.json` and [`SUMMARY_FILE`] to `out`, and the basis
    /// and the sample of the best set to the configured `.npy` files with
    /// their [`StableProvenance`].
    pub fn write(&self, config: &StableConfig, out: &OutputDir) -> Result<()> {
        out.write_json(METADATA_FILE, &StableMetadata::new(config))?;
        out.write_json(
            SUMMARY_FILE,
            &Summary {
                neurons: config.neurons(),
                best: self.best(),
                verified: &self.verified,
                candidates: &self.candidates,
            },
        )?;
        let Some(best) = self.best() else {
            return Ok(());
        };
        for (file, set, content) in [
            (&config.basis, &self.basis, "basis"),
            (&config.sample, &self.sample, "sample"),
        ] {
            if let (Some(file), Some(set)) = (file, set) {
                let file = config.resolve(file);
                StableProvenance::new(config, &file, &best.weights, best.size, content, set)
                    .write(set, &file)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::WeightMatrix;
    use std::path::Path;

    fn config(text: &str) -> StableConfig {
        StableConfig::parse(text, Path::new("../../data/experiments/x/config.toml")).unwrap()
    }

    /// 5 x 3 bitmaps (`N = 15`) with weights 1 and 2: the largest set is
    /// `u = (1, ..., 1, 2)` with `2 C(14, 6) = 6006` members (the fourteen
    /// ones sum to `-2 xi_15`), strictly stable with both rules. The next
    /// one, `u = (1, ..., 1, 2, 2, 2)`, has `220 + 2376 + 2376 + 220 = 5192`
    /// members (`a` of the twelve ones and `b` of the three twos with sign
    /// `+1`, `a + 2b = 9`).
    #[test]
    fn finds_the_largest_stable_hyperplane_set() {
        let search = StableSearch::execute(&config(
            "width = 5\nheight = 3\nmax_weight = 2\nverify = 3\n",
        ))
        .unwrap();
        assert_eq!(search.candidates.len(), 15);
        assert_eq!(search.candidates[1].size, 5192);
        let sizes: Vec<u64> = search.candidates.iter().map(|c| c.size).collect();
        assert!(sizes.windows(2).all(|w| w[0] >= w[1]));
        assert_eq!(search.best, Some(0));
        let best = search.best().unwrap();
        assert_eq!(best.weights, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]);
        assert_eq!(best.size, 6006);
        assert!(best.all_stable());
        assert_eq!((best.hebb.patterns, best.oja.patterns), (6006, 6006));
        assert_eq!(best.oja_patterns, 14);
        assert!(best.oja_converged);
        assert_eq!(search.verified.len(), 3);
        assert_eq!(search.sample.as_ref().unwrap().len(), 24);
        assert_eq!(search.basis.as_ref().unwrap().width(), 5);
    }

    /// The Hebbian rule fails on the same kind of set when it is small: for
    /// `N = 9`, `u = (1, ..., 1, 2)`, no member is strictly stable, so the
    /// search finds no set stable with both rules.
    #[test]
    fn reports_no_best_set_when_the_hebbian_rule_fails() {
        let search = StableSearch::execute(&config(
            "width = 3\nheight = 3\nmax_weight = 2\nverify = 1\n",
        ))
        .unwrap();
        assert_eq!(search.verified[0].size, 112);
        assert_eq!(search.verified[0].hebb.stable, 0);
        assert!(search.verified[0].oja.all_stable());
        assert_eq!(search.best, None);
        assert!(search.basis.is_none() && search.sample.is_none());
    }

    /// Oja's rule on the basis of `S_u` tends to the projection onto
    /// `u^perp`: `P = I - u u^T / |u|^2`, here with the diagonal set to
    /// zero, so the aligned field of every member is `1 - P_ii = u_i^2 /
    /// |u|^2`, smallest `1 / 12` for `u = (1, ..., 1, 2)` with `|u|^2 = 8 +
    /// 4 = 12`.
    #[test]
    fn oja_learns_the_projection_onto_the_hyperplane() {
        let plane = Hyperplane::new(vec![1, 1, 1, 1, 1, 1, 1, 1, 2]).unwrap();
        let basis = PatternSet::from_patterns(9, 1, &plane.basis()).unwrap();
        let trained = crate::learning::Oja::default()
            .train(&basis, Backend::Dense)
            .unwrap();
        let weights = trained.weights.to_dense();
        let u = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0];
        for i in 0..9 {
            for j in 0..9 {
                let expected = if i == j { 0.0 } else { -u[i] * u[j] / 12.0 };
                assert!((weights[[i, j]] - expected).abs() < 1e-3, "{i} {j}");
            }
        }
        let check = StrictStability::of(&trained.weights, &plane);
        assert!(check.all_stable());
        assert!((check.min_aligned - 1.0 / 12.0).abs() < 1e-3);
    }
}
