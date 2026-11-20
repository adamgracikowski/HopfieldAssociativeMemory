//! The `recall` command: run the network from an initial state and save
//! every visited state.

use serde::Serialize;

use super::train::METADATA_FILE;
use super::{DatasetInfo, Initial, Metadata, OutputDir, RunConfig, TrainingInfo};
use crate::dynamics::{History, Outcome, Termination};
use crate::learning::LearningRule;
use crate::patterns::PatternView;
use crate::seed::Seed;
use crate::{PatternSet, Result};

/// Name of the state history file (`int8`, `frames x height x width`).
pub const HISTORY_FILE: &str = "history.npy";
/// Name of the per-frame statistics file.
pub const FRAMES_FILE: &str = "frames.json";

/// The stored pattern closest to a state.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Nearest {
    /// Index of the pattern with the largest `|m|`.
    pub pattern: usize,
    /// Its overlap `m` with the state; negative for a state close to the
    /// negated pattern.
    pub overlap: f64,
}

impl Nearest {
    /// The pattern with the largest absolute overlap among `overlaps`, the
    /// first one on ties.
    pub fn of(overlaps: &[f64]) -> Self {
        let (pattern, &overlap) = overlaps
            .iter()
            .enumerate()
            .reduce(|best, next| {
                if next.1.abs() > best.1.abs() {
                    next
                } else {
                    best
                }
            })
            .expect("a pattern set is not empty");
        Self { pattern, overlap }
    }
}

/// Summary of a recall run as recorded in the metadata.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RecallSummary {
    /// Applied steps (synchronous updates or asynchronous sweeps).
    pub steps: usize,
    /// Total number of neuron changes.
    pub changes: usize,
    /// Why the run stopped.
    pub termination: Termination,
    /// Energy of the final state.
    pub energy: f64,
    /// Number of recorded frames.
    pub frames: usize,
    /// Stored pattern the initial state was made from, if any.
    pub target: Option<usize>,
    /// Hamming distance of the initial state to the target.
    pub initial_distance: Option<usize>,
    /// Hamming distance of the final state to the target.
    pub final_distance: Option<usize>,
    /// Stored pattern closest to the final state.
    pub nearest: Nearest,
}

/// Energy, changes and overlaps with every stored pattern of each frame.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Frames<'a> {
    /// Energy of each frame.
    pub energy: &'a [f64],
    /// Neurons changed since the previous frame.
    pub changes: &'a [usize],
    /// Overlaps `m^mu` of each frame with every pattern (`frames x p`).
    pub overlaps: Vec<Vec<f64>>,
}

/// A recall run: the trained network, its initial state and every frame on
/// the way to the final state.
#[derive(Debug, Clone)]
pub struct RecallRun {
    /// The dataset.
    pub dataset: DatasetInfo,
    /// Its patterns.
    pub set: PatternSet,
    /// Result of training.
    pub training: TrainingInfo,
    /// The initial state.
    pub initial: Initial,
    /// Result of the dynamics.
    pub outcome: Outcome,
    /// Every recorded frame.
    pub history: History,
}

#[derive(Serialize)]
struct RecallResult<'a> {
    outcome: &'a RecallSummary,
}

impl RecallRun {
    /// Loads the dataset, trains the network, builds the initial state and
    /// runs the dynamics, all as configured.
    pub fn execute(config: &RunConfig) -> Result<Self> {
        let seed = Seed(config.require_seed()?);
        let input = config.require_input()?;
        let (dataset, set) = config.load_dataset()?;
        let trained = config.learning.train(&set, config.backend)?;

        let initial = input.initial(&set, &mut seed.input_rng())?;
        let mut history = History::new();
        let outcome = config.dynamics.run(
            &trained.weights,
            initial.state.view(),
            &mut seed.dynamics_rng(),
            &mut history,
        )?;
        Ok(Self {
            dataset,
            set,
            training: TrainingInfo::from(&trained),
            initial,
            outcome,
            history,
        })
    }

    /// Overlaps of `state` with every stored pattern.
    pub fn overlaps(&self, state: PatternView<'_>) -> Vec<f64> {
        self.set
            .iter()
            .map(|pattern| state.overlap(&pattern))
            .collect()
    }

    /// Summary of the run for the metadata.
    pub fn summary(&self) -> RecallSummary {
        let target = self.initial.target;
        let distance =
            |state: PatternView<'_>| target.map(|mu| state.hamming(&self.set.pattern(mu)));
        RecallSummary {
            steps: self.outcome.steps,
            changes: self.outcome.changes,
            termination: self.outcome.termination,
            energy: self.outcome.energy,
            frames: self.history.len(),
            target,
            initial_distance: distance(self.initial.state.view()),
            final_distance: distance(self.outcome.state.view()),
            nearest: Nearest::of(&self.overlaps(self.outcome.state.view())),
        }
    }

    /// Statistics of every frame.
    pub fn frames(&self) -> Frames<'_> {
        Frames {
            energy: self.history.energies(),
            changes: self.history.changes(),
            overlaps: (0..self.history.len())
                .map(|k| self.overlaps(self.history.state(k)))
                .collect(),
        }
    }

    /// Writes `metadata.json`, [`HISTORY_FILE`] and [`FRAMES_FILE`].
    pub fn write(&self, config: &RunConfig, out: &OutputDir) -> Result<()> {
        let summary = self.summary();
        let metadata = Metadata::new(
            "recall",
            config,
            &self.dataset,
            self.training,
            RecallResult { outcome: &summary },
        );
        out.write_json(METADATA_FILE, &metadata)?;
        let shape = [self.history.len(), self.set.height(), self.set.width()];
        out.write_npy(HISTORY_FILE, &shape, self.history.states().iter().copied())?;
        out.write_json(FRAMES_FILE, &self.frames())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::Mode;
    use crate::patterns::npy::write_npy;
    use ndarray::{Array2, array};
    use std::path::Path;

    /// Writes a registry with the 2x2 patterns `data` and a run
    /// configuration with Hebbian learning and `extra` lines into `dir`.
    fn config(dir: &Path, data: Array2<i8>, extra: &str) -> RunConfig {
        let set = PatternSet::new(2, 2, data).unwrap();
        write_npy(&set, &dir.join("set.npy")).unwrap();
        std::fs::write(
            dir.join("registry.toml"),
            "[datasets.set]\nfile = \"set.npy\"\nwidth = 2\nheight = 2\ndescription = \"Test.\"\n",
        )
        .unwrap();
        let text = format!(
            "dataset = \"set\"\nregistry = \"registry.toml\"\nseed = 1\n\
             [learning]\nrule = \"hebb\"\n{extra}"
        );
        RunConfig::parse(&text, &dir.join("config.toml")).unwrap()
    }

    /// The orthogonal patterns `(1, 1, 1, 1)` and `(1, -1, 1, -1)`.
    fn orthogonal_config(dir: &Path, extra: &str) -> RunConfig {
        config(dir, array![[1, 1, 1, 1], [1, -1, 1, -1]], extra)
    }

    /// One frame, overlaps 1 with itself and 0 with the other pattern.
    #[test]
    fn stored_pattern_stays() {
        let dir = tempfile::tempdir().unwrap();
        let config = orthogonal_config(dir.path(), "[input]\nsource = \"pattern\"\nindex = 1");
        let run = RecallRun::execute(&config).unwrap();
        let summary = run.summary();
        assert_eq!(summary.termination, Termination::FixedPoint);
        assert_eq!((summary.steps, summary.frames), (0, 1));
        assert_eq!(summary.target, Some(1));
        assert_eq!(
            (summary.initial_distance, summary.final_distance),
            (Some(0), Some(0))
        );
        assert_eq!(
            summary.nearest,
            Nearest {
                pattern: 1,
                overlap: 1.0
            }
        );
        assert_eq!(run.frames().overlaps, [[0.0, 1.0]]);
    }

    /// `(1, 1, 1, 1)` gives `w_ij = 1/4`. With one pixel flipped, that
    /// neuron has `h = 3/4` and the others `(2 - 1)/4 = 1/4`, so one
    /// synchronous step restores the pattern: overlaps `(4 - 2)/4 = 0.5`
    /// and 1, one change.
    #[test]
    fn damaged_pattern_is_recovered() {
        let dir = tempfile::tempdir().unwrap();
        let config = config(
            dir.path(),
            array![[1, 1, 1, 1]],
            "[dynamics]\nmode = \"synchronous\"\n[input]\nsource = \"pattern\"\nindex = 0\nflips = 1",
        );
        assert_eq!(config.dynamics.mode, Mode::Synchronous);
        let run = RecallRun::execute(&config).unwrap();
        let summary = run.summary();
        assert_eq!(summary.termination, Termination::FixedPoint);
        assert_eq!((summary.steps, summary.changes, summary.frames), (1, 1, 2));
        assert_eq!(
            (summary.initial_distance, summary.final_distance),
            (Some(1), Some(0))
        );
        let frames = run.frames();
        assert_eq!(frames.overlaps, [[0.5], [1.0]]);
        assert_eq!(frames.changes, [0, 1]);
    }

    #[test]
    fn same_seed_gives_the_same_run() {
        let dir = tempfile::tempdir().unwrap();
        let config = orthogonal_config(dir.path(), "[input]\nsource = \"random\"");
        let first = RecallRun::execute(&config).unwrap();
        let second = RecallRun::execute(&config).unwrap();
        assert_eq!(first.initial, second.initial);
        assert_eq!(first.history, second.history);
    }

    #[test]
    fn damaged_input_does_not_depend_on_the_dynamics() {
        let dir = tempfile::tempdir().unwrap();
        let input = "[input]\nsource = \"pattern\"\nindex = 0\nflips = 2";
        let initial = |extra: &str| {
            let config = orthogonal_config(dir.path(), &format!("{extra}\n{input}"));
            RecallRun::execute(&config).unwrap().initial
        };
        let asynchronous = initial("[dynamics]\norder = \"random\"");
        assert_eq!(initial("[dynamics]\nmode = \"synchronous\""), asynchronous);
        assert_eq!(initial("[dynamics]\norder = \"sequential\""), asynchronous);

        let config = orthogonal_config(dir.path(), input);
        let reseeded = (1..20)
            .map(|seed| {
                let mut config = config.clone();
                config.seed = Some(seed + 1);
                RecallRun::execute(&config).unwrap().initial
            })
            .any(|other| other != asynchronous);
        assert!(reseeded);
    }

    #[test]
    fn requires_a_seed_and_an_input() {
        let dir = tempfile::tempdir().unwrap();
        let config = orthogonal_config(dir.path(), "");
        let error = RecallRun::execute(&config).unwrap_err();
        assert!(error.to_string().contains("`[input]` table"), "{error}");
    }

    #[test]
    fn nearest_prefers_the_largest_absolute_overlap() {
        assert_eq!(
            Nearest::of(&[0.2, -0.8, 0.8]),
            Nearest {
                pattern: 1,
                overlap: -0.8
            }
        );
    }
}
