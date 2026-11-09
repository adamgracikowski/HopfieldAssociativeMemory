//! Execution of an experiment.

use ndarray::Array2;
use rayon::prelude::*;
use serde::Serialize;

use super::{
    Cell, CellSummary, DatasetStatistics, ExperimentConfig, ExperimentInput, ExperimentMetadata,
    FieldTable, Trial, TrialResult, TrialTable,
};
use crate::dynamics::Stability;
use crate::learning::LearningRule;
use crate::patterns::summary::Summary as PatternSummary;
use crate::run::train::METADATA_FILE;
use crate::run::{DatasetInfo, OutputDir, TrainingInfo};
use crate::seed::Seed;
use crate::{PatternSet, Result};

/// Name of the trial table.
pub const TRIALS_FILE: &str = "trials.parquet";
/// Name of the file with the aggregates of every cell.
pub const SUMMARY_FILE: &str = "summary.json";
/// Name of the file with the statistics of every dataset.
pub const DATASETS_FILE: &str = "datasets.json";
/// Name of the table of aligned local fields (see [`FieldTable`]).
pub const FIELDS_FILE: &str = "stability.parquet";

/// Contents of [`SUMMARY_FILE`].
#[derive(Serialize)]
struct Summary {
    cells: Vec<CellSummary>,
}

/// Contents of [`DATASETS_FILE`].
#[derive(Serialize)]
struct Datasets<'a> {
    datasets: &'a [DatasetStatistics],
}

/// Training of one rule on one dataset, shared by all cells of the pair.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Training {
    /// Dataset name.
    pub dataset: String,
    /// Learning rule name.
    pub learning: String,
    /// Result of training.
    #[serde(flatten)]
    pub info: TrainingInfo,
    /// Stability of the stored patterns, with `sgn(0)` from the dynamics
    /// settings.
    pub stability: Stability,
}

/// An executed experiment: the datasets, one training per (dataset, rule)
/// pair and the result of every trial.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentRun {
    /// The datasets in configuration order.
    pub datasets: Vec<DatasetInfo>,
    /// Statistics of the datasets in configuration order.
    pub statistics: Vec<DatasetStatistics>,
    /// Names of the learning rules in configuration order.
    pub learning: Vec<String>,
    /// Trainings ordered by dataset, then rule.
    pub trainings: Vec<Training>,
    /// The [aligned fields](Stability::aligned_fields) (`p x N`) of every
    /// training, in the order of `trainings`, if the configuration asks for
    /// `stability_fields`; otherwise empty.
    pub fields: Vec<Array2<f64>>,
    /// Cells of the grid.
    pub cells: Vec<Cell>,
    /// Results ordered by cell, then trial, then target.
    pub results: Vec<TrialResult>,
}

impl ExperimentRun {
    /// Loads the datasets, checks that their noise levels give different
    /// numbers of flipped pixels and, for every (dataset, rule) pair in turn,
    /// trains the network, checks pattern stability, passes the training to
    /// `on_training` and runs the trials of the pair's cells in parallel.
    ///
    /// Every trial draws only from its own [`Seed`], derived from the
    /// experiment seed, the dataset name, the input source, the number of
    /// flipped pixels, the target and the trial number, and the results are
    /// collected in task order, so they do not depend on the number of
    /// threads. The seed does not depend on the rule or the mode: all of
    /// them start from the same initial states.
    pub fn execute(
        config: &ExperimentConfig,
        mut on_training: impl FnMut(&Training),
    ) -> Result<Self> {
        let registry = config.open_registry()?;
        let loaded = config
            .datasets
            .iter()
            .map(|name| DatasetInfo::load(&registry, name, config.directory()))
            .collect::<Result<Vec<_>>>()?;
        for (info, _) in &loaded {
            config.check_noise_levels(&info.name, info.neurons)?;
        }
        let statistics = loaded
            .iter()
            .map(|(info, set)| {
                DatasetStatistics::from(&PatternSummary::new(&info.name, &info.description, set))
            })
            .collect();
        let learning = config.learning_labels();
        let cells = config.cells();
        let mut trainings = Vec::new();
        let mut fields = Vec::new();
        let mut results = Vec::new();
        for (d, (info, set)) in loaded.iter().enumerate() {
            for (l, rule) in config.learning.iter().enumerate() {
                let trained = rule.train(set, config.backend)?;
                let aligned = Stability::aligned_fields(&trained.weights, set);
                let training = Training {
                    dataset: info.name.clone(),
                    learning: learning[l].clone(),
                    info: TrainingInfo::from(&trained),
                    stability: Stability::from_aligned_fields(
                        aligned.view(),
                        set,
                        config.dynamics.zero_field,
                    ),
                };
                on_training(&training);
                trainings.push(training);
                if config.stability_fields {
                    fields.push(aligned);
                }

                let tasks: Vec<(Trial, Cell)> = cells
                    .iter()
                    .enumerate()
                    .filter(|(_, cell)| (cell.dataset, cell.learning) == (d, l))
                    .flat_map(|(index, cell)| {
                        trials(config, index, cell, &info.name, set)
                            .into_iter()
                            .map(|trial| (trial, *cell))
                    })
                    .collect();
                let pair = tasks
                    .par_iter()
                    .map(|(trial, cell)| {
                        trial.run(&trained.weights, set, &config.dynamics.for_mode(cell.mode))
                    })
                    .collect::<Result<Vec<_>>>()?;
                results.extend(pair);
            }
        }
        Ok(Self {
            datasets: loaded.into_iter().map(|(info, _)| info).collect(),
            statistics,
            learning,
            trainings,
            fields,
            cells,
            results,
        })
    }

    /// Aggregates of every cell, in cell order.
    pub fn summaries(&self) -> Vec<CellSummary> {
        self.results
            .chunk_by(|a, b| a.trial.cell == b.trial.cell)
            .map(|results| {
                let cell = self.cells[results[0].trial.cell];
                CellSummary::new(
                    self.datasets[cell.dataset].name.clone(),
                    self.learning[cell.learning].clone(),
                    cell,
                    results,
                )
            })
            .collect()
    }

    /// Writes `metadata.json`, [`SUMMARY_FILE`] (`{"cells": [...]}` with the
    /// [`CellSummary`] of every cell), [`DATASETS_FILE`]
    /// (`{"datasets": [...]}` with the [`DatasetStatistics`] of every
    /// dataset), [`TRIALS_FILE`] (the [`TrialTable`]) and, if the run kept
    /// them, [`FIELDS_FILE`] (the [`FieldTable`]); otherwise an old
    /// [`FIELDS_FILE`] is removed, so a run without fields in a reused
    /// directory leaves none. All files but the metadata depend only on the
    /// configuration and the code, not on the time or the number of threads.
    pub fn write(&self, config: &ExperimentConfig, out: &OutputDir) -> Result<()> {
        out.write_json(METADATA_FILE, &ExperimentMetadata::new(config, self))?;
        out.write_json(
            SUMMARY_FILE,
            &Summary {
                cells: self.summaries(),
            },
        )?;
        out.write_json(
            DATASETS_FILE,
            &Datasets {
                datasets: &self.statistics,
            },
        )?;
        out.write_parquet(TRIALS_FILE, TrialTable::new(self).batch())?;
        if config.stability_fields {
            out.write_parquet(FIELDS_FILE, FieldTable::new(self).batch())
        } else {
            out.remove(FIELDS_FILE)
        }
    }
}

/// The trials of cell `index`: for every trial number every target (all
/// stored patterns, or one random state). A
/// [deterministic](Cell::is_deterministic) cell has only trial 0.
fn trials(
    config: &ExperimentConfig,
    index: usize,
    cell: &Cell,
    dataset: &str,
    set: &PatternSet,
) -> Vec<Trial> {
    let flips = cell
        .noise
        .map_or(0, |noise| ExperimentInput::flips(noise, set.neurons()));
    let targets: Vec<Option<usize>> = match config.input {
        ExperimentInput::Pattern { .. } => (0..set.len()).map(Some).collect(),
        ExperimentInput::Random {} => vec![None],
    };
    let source = config.input.source();
    let count = if cell.is_deterministic(flips) {
        1
    } else {
        config.trials
    };
    let mut trials = Vec::new();
    for trial in 0..count {
        for &target in &targets {
            let (flips_part, target_part) = match target {
                Some(mu) => (flips.to_string(), mu.to_string()),
                None => ("-".to_owned(), "-".to_owned()),
            };
            let seed = Seed::derive(
                config.seed,
                &[&dataset, &source, &flips_part, &target_part, &trial],
            );
            trials.push(Trial {
                cell: index,
                trial,
                target,
                flips,
                seed,
            });
        }
    }
    trials
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::{Mode, Termination};
    use crate::run::{RecallRun, RunConfig};
    use std::collections::HashMap;
    use std::path::Path;

    const CONFIG: &str = r#"
        seed = 11
        datasets = ["small-7x7"]
        trials = 3

        [[learning]]
        rule = "hebb"
        [[learning]]
        rule = "oja"

        [dynamics]
        modes = ["synchronous", "asynchronous"]

        [input]
        source = "pattern"
        noise = [0.0, 0.2]
    "#;

    fn config(text: &str) -> ExperimentConfig {
        ExperimentConfig::parse(
            text,
            Path::new("../../data/experiments/00-example/config.toml"),
        )
        .unwrap()
    }

    fn execute(text: &str) -> ExperimentRun {
        ExperimentRun::execute(&config(text), |_| {}).unwrap()
    }

    /// `small-7x7` has `p = 5` patterns of `N = 49` neurons: 2 rules x 2
    /// modes x 2 noise levels = 8 cells of 3 trials x 5 targets, with
    /// `round(0.2 * 49) = 10` flipped pixels at the higher noise level,
    /// except the 2 deterministic cells (synchronous, no noise) with one
    /// trial: `6 * 15 + 2 * 5 = 100` runs.
    #[test]
    fn runs_every_trial_of_the_grid() {
        let mut trained = Vec::new();
        let run = ExperimentRun::execute(&config(CONFIG), |training| {
            trained.push((training.dataset.clone(), training.learning.clone()));
        })
        .unwrap();
        assert_eq!(
            trained,
            [
                ("small-7x7".to_owned(), "hebb".to_owned()),
                ("small-7x7".to_owned(), "oja (subspace)".to_owned())
            ]
        );
        assert_eq!(run.trainings.len(), 2);
        assert_eq!(run.results.len(), 100);
        for result in &run.results {
            let cell = run.cells[result.trial.cell];
            let flips = if cell.noise == Some(0.2) { 10 } else { 0 };
            assert_eq!(result.trial.flips, flips);
            assert_eq!(result.initial_distance, Some(flips));
            let fixed = result.termination == Termination::FixedPoint;
            assert_eq!(
                result.recovered,
                Some(fixed && result.final_distance == Some(0))
            );
        }
        let first: Vec<_> = run.results[..11]
            .iter()
            .map(|r| (r.trial.cell, r.trial.trial, r.trial.target))
            .collect();
        assert_eq!(
            first,
            [
                (0, 0, Some(0)),
                (0, 0, Some(1)),
                (0, 0, Some(2)),
                (0, 0, Some(3)),
                (0, 0, Some(4)),
                (1, 0, Some(0)),
                (1, 0, Some(1)),
                (1, 0, Some(2)),
                (1, 0, Some(3)),
                (1, 0, Some(4)),
                (1, 1, Some(0))
            ]
        );

        let summaries = run.summaries();
        assert_eq!(summaries.len(), 8);
        for summary in &summaries {
            let deterministic = summary.mode == Mode::Synchronous && summary.flips == Some(0);
            assert_eq!(summary.deterministic, deterministic);
            assert_eq!(summary.runs, if deterministic { 5 } else { 15 });
            let classes = summary.classes;
            let total = classes.pattern.count
                + classes.negation.count
                + classes.spurious.count
                + classes.cycle.count
                + classes.unconverged.count;
            assert_eq!(total, summary.runs);
            let recovered = summary.recovered.unwrap();
            assert_eq!(recovered.lower == recovered.upper, deterministic);
            assert!(summary.final_overlap.is_some());
        }
        assert_eq!(summaries[1].flips, Some(10));
        assert_eq!(summaries[2].mode, Mode::Asynchronous);
        assert_eq!(summaries[4].learning, "oja (subspace)");
    }

    #[test]
    fn results_do_not_depend_on_the_number_of_threads() {
        let in_pool = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| execute(CONFIG))
        };
        assert_eq!(in_pool(1), in_pool(4));
    }

    /// Trials with the same dataset, flips, target and number share their
    /// seed, so every rule and mode starts from the same damaged input.
    #[test]
    fn rules_and_modes_start_from_the_same_inputs() {
        let run = execute(CONFIG);
        let mut seeds = HashMap::new();
        for result in &run.results {
            let trial = result.trial;
            let key = (trial.flips, trial.target, trial.trial);
            let seed = *seeds.entry(key).or_insert(trial.seed);
            assert_eq!(trial.seed, seed, "{key:?}");
        }
        assert_eq!(seeds.len(), 2 * 5 * 3);
    }

    /// A `recall` run with the seed, rule, mode, target and flips of a trial
    /// repeats it.
    #[test]
    fn recall_repeats_a_trial() {
        let run = execute(CONFIG);
        let checked = run
            .results
            .iter()
            .filter(|result| result.trial.flips > 0)
            .step_by(7)
            .map(|result| {
                let cell = run.cells[result.trial.cell];
                let text = format!(
                    "dataset = \"small-7x7\"\nseed = {}\n[learning]\nrule = \"{}\"\n\
                     [dynamics]\nmode = \"{}\"\n[input]\nsource = \"pattern\"\n\
                     index = {}\nflips = {}\n",
                    result.trial.seed,
                    ["hebb", "oja"][cell.learning],
                    cell.mode,
                    result.trial.target.unwrap(),
                    result.trial.flips,
                );
                let config = RunConfig::parse(
                    &text,
                    Path::new("../../data/experiments/00-example/recall.toml"),
                )
                .unwrap();
                let summary = RecallRun::execute(&config).unwrap().summary();
                assert_eq!(summary.steps, result.steps);
                assert_eq!(summary.changes, result.changes);
                assert_eq!(summary.termination, result.termination);
                assert_eq!(summary.energy, result.energy);
                assert_eq!(summary.final_distance, result.final_distance);
                assert_eq!(summary.nearest, result.nearest);
            })
            .count();
        assert!(checked >= 8);
    }

    /// Random initial states give one run per trial, without a target.
    #[test]
    fn random_inputs_have_no_target() {
        let run = execute(&CONFIG.replace(
            "source = \"pattern\"\n        noise = [0.0, 0.2]",
            "source = \"random\"",
        ));
        assert_eq!(run.results.len(), 4 * 3);
        let result = &run.results[0];
        assert_eq!((result.trial.target, result.recovered), (None, None));
        let summary = &run.summaries()[0];
        assert_eq!(
            (summary.noise, summary.flips, summary.recovered),
            (None, None, None)
        );
        assert_eq!(summary.final_overlap, None);
    }
}
