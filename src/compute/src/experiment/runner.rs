//! Execution of an experiment.

use ndarray::Array2;
use rayon::prelude::*;
use serde::Serialize;

use super::{
    Attractor, AttractorTable, Cell, CellSummary, DatasetStatistics, ExperimentConfig,
    ExperimentInput, ExperimentMetadata, FieldTable, Trial, TrialResult, TrialTable,
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
/// Name of the table of distinct final states (see [`AttractorTable`]).
pub const ATTRACTORS_FILE: &str = "attractors.parquet";

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
    /// [Aligned fields](Stability::aligned_fields) (`p x N`) of every
    /// training with `stability_fields`; otherwise empty.
    pub fields: Vec<Array2<f64>>,
    /// Cells of the grid.
    pub cells: Vec<Cell>,
    /// Results ordered by cell, then trial, then target.
    pub results: Vec<TrialResult>,
    /// Distinct final states ordered by cell, then
    /// [rank](Attractor::rank), with `attractors`; otherwise empty.
    pub attractors: Vec<Attractor>,
}

impl ExperimentRun {
    /// Loads the datasets, checks that their damage levels give distinct
    /// damaged-pixel counts, trains every (dataset, rule) pair, reports it to
    /// `on_training` and runs its trials in parallel (with `attractors`, cell
    /// by cell). Trial seeds depend on the dataset registry name, source,
    /// damaged pixels, target and trial, not on the prefix, rule or mode.
    pub fn execute(
        config: &ExperimentConfig,
        mut on_training: impl FnMut(&Training),
    ) -> Result<Self> {
        let registry = config.open_registry()?;
        let loaded = config
            .datasets
            .iter()
            .map(|entry| {
                DatasetInfo::load(&registry, entry.name(), entry.prefix(), config.directory())
            })
            .collect::<Result<Vec<_>>>()?;
        for (info, _) in &loaded {
            config.check_damage_levels(&info.label(), info.width, info.height)?;
        }
        let statistics = loaded
            .iter()
            .map(|(info, set)| {
                DatasetStatistics::from(&PatternSummary::new(&info.label(), &info.description, set))
            })
            .collect();
        let learning = config.learning_labels();
        let cells = config.cells();
        let mut trainings = Vec::new();
        let mut fields = Vec::new();
        let mut results = Vec::new();
        let mut attractors = Vec::new();
        for (d, (info, set)) in loaded.iter().enumerate() {
            for (l, rule) in config.learning.iter().enumerate() {
                let trained = rule.train(set, config.backend)?;
                let aligned = Stability::aligned_fields(&trained.weights, set);
                let training = Training {
                    dataset: info.label(),
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
                if config.attractors {
                    for cell in tasks.chunk_by(|a, b| a.0.cell == b.0.cell) {
                        let runs = cell
                            .par_iter()
                            .map(|(trial, cell)| {
                                let dynamics = config.dynamics.for_mode(cell.mode);
                                trial.run_with_state(&trained.weights, set, &dynamics)
                            })
                            .collect::<Result<Vec<_>>>()?;
                        attractors.extend(Attractor::collect(
                            &runs,
                            set,
                            &trained.weights,
                            config.dynamics.zero_field,
                        ));
                        results.extend(runs.into_iter().map(|(result, _)| result));
                    }
                } else {
                    let pair = tasks
                        .par_iter()
                        .map(|(trial, cell)| {
                            let dynamics = config.dynamics.for_mode(cell.mode);
                            trial.run(&trained.weights, set, &dynamics)
                        })
                        .collect::<Result<Vec<_>>>()?;
                    results.extend(pair);
                }
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
            attractors,
        })
    }

    /// Aggregates of every cell, in cell order.
    pub fn summaries(&self) -> Vec<CellSummary> {
        self.results
            .chunk_by(|a, b| a.trial.cell == b.trial.cell)
            .map(|results| {
                let cell = self.cells[results[0].trial.cell];
                CellSummary::new(
                    self.datasets[cell.dataset].label(),
                    self.learning[cell.learning].clone(),
                    cell,
                    results,
                )
            })
            .collect()
    }

    /// Writes metadata, summary, datasets and trial table, plus the field and
    /// attractor tables if configured (removing stale ones otherwise).
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
            out.write_parquet(FIELDS_FILE, FieldTable::new(self).batch())?;
        } else {
            out.remove(FIELDS_FILE)?;
        }
        if config.attractors {
            out.write_parquet(ATTRACTORS_FILE, AttractorTable::new(self).batch())
        } else {
            out.remove(ATTRACTORS_FILE)
        }
    }
}

/// The trials of cell `index`; a deterministic cell has only trial 0.
fn trials(
    config: &ExperimentConfig,
    index: usize,
    cell: &Cell,
    dataset: &str,
    set: &PatternSet,
) -> Vec<Trial> {
    let damage = cell
        .noise
        .and_then(|level| config.input.damage(level, set.width(), set.height()));
    let pixels = damage.map_or(0, |damage| damage.pixels());
    let targets: Vec<Option<usize>> = match config.input {
        ExperimentInput::Random {} => vec![None],
        _ => (0..set.len()).map(Some).collect(),
    };
    let source = config.input.source();
    let count = if cell.is_deterministic(pixels) {
        1
    } else {
        config.trials
    };
    let mut trials = Vec::new();
    for trial in 0..count {
        for &target in &targets {
            let (pixels_part, target_part) = match target {
                Some(mu) => (pixels.to_string(), mu.to_string()),
                None => ("-".to_owned(), "-".to_owned()),
            };
            let seed = Seed::derive(
                config.seed,
                &[&dataset, &source, &pixels_part, &target_part, &trial],
            );
            trials.push(Trial {
                cell: index,
                trial,
                target,
                damage,
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
    use crate::experiment::Damage;
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
            Path::new("../../data/experiments/example/config.toml"),
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
            assert_eq!(result.trial.damaged_pixels(), flips);
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
            assert!(recovered.count <= summary.nearest_target.unwrap().count);
            assert!(summary.final_overlap.is_some());
            assert!(summary.final_distance.is_some());
        }
        for result in &run.results {
            if result.recovered == Some(true) {
                assert_eq!(result.nearest_target(), Some(true), "{result:?}");
            }
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
            let key = (trial.damage, trial.target, trial.trial);
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
            .filter(|result| result.trial.damaged_pixels() > 0)
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
                    result.trial.damaged_pixels(),
                );
                let config = RunConfig::parse(
                    &text,
                    Path::new("../../data/experiments/example/recall.toml"),
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

    /// Occlusion levels 0 and 0.3 of the 7-row `small-7x7` cover 0 and
    /// `round(2.1) = 2` rows, 14 pixels; only covered pixels equal to `1`
    /// change, so the initial distance is at most 14. The grid has the same
    /// shape as with noise (100 runs), but the seeds differ from the noise
    /// trials, and `recall` with `source = "occlusion"` repeats a trial.
    #[test]
    fn covers_rows_of_the_targets() {
        let text = CONFIG.replace(
            "source = \"pattern\"\n        noise = [0.0, 0.2]",
            "source = \"occlusion\"\n        occlusion = [0.0, 0.3]",
        );
        let run = execute(&text);
        assert_eq!(run.results.len(), 100);
        let noise = execute(CONFIG);
        let mut checked = 0;
        for (result, flipped) in run.results.iter().zip(&noise.results) {
            let cell = run.cells[result.trial.cell];
            let distance = result.initial_distance.unwrap();
            if cell.noise == Some(0.3) {
                let damage = Damage::Occlusion {
                    rows: 2,
                    pixels: 14,
                };
                assert_eq!(result.trial.damage, Some(damage));
                assert!(distance <= 14, "{result:?}");
                assert_ne!(result.trial.seed, flipped.trial.seed);
                if checked < 4 && distance > 0 {
                    checked += 1;
                    let text = format!(
                        "dataset = \"small-7x7\"\nseed = {}\n[learning]\nrule = \"{}\"\n\
                         [dynamics]\nmode = \"{}\"\n[input]\nsource = \"occlusion\"\n\
                         index = {}\nrows = 2\n",
                        result.trial.seed,
                        ["hebb", "oja"][cell.learning],
                        cell.mode,
                        result.trial.target.unwrap(),
                    );
                    let config = RunConfig::parse(
                        &text,
                        Path::new("../../data/experiments/example/recall.toml"),
                    )
                    .unwrap();
                    let summary = RecallRun::execute(&config).unwrap().summary();
                    assert_eq!(summary.steps, result.steps);
                    assert_eq!(summary.energy, result.energy);
                    assert_eq!(summary.final_distance, result.final_distance);
                }
            } else {
                assert_eq!(result.trial.damaged_pixels(), 0);
                assert_eq!(distance, 0);
            }
        }
        assert_eq!(checked, 4);
        let summaries = run.summaries();
        assert_eq!(summaries[1].flips, Some(14));
        assert!(summaries[0].deterministic && !summaries[1].deterministic);
    }

    /// `small-7x7` has 5 patterns, so its prefix of 5 is the whole set: the
    /// same 100 runs under the label `small-7x7:5`. The prefix of 3 has
    /// `6 * 3 * 3 + 2 * 3 = 60` runs whose seeds, and so damaged inputs,
    /// equal those of the same targets in the whole set; `recall` with
    /// `patterns = 3` repeats them.
    #[test]
    fn prefixes_store_the_first_patterns() {
        let text = CONFIG.replace(
            "datasets = [\"small-7x7\"]",
            "datasets = [\"small-7x7\", { name = \"small-7x7\", patterns = 5 }, \
             { name = \"small-7x7\", patterns = 3 }]",
        );
        let run = execute(&text);
        let labels: Vec<String> = run.datasets.iter().map(DatasetInfo::label).collect();
        assert_eq!(labels, ["small-7x7", "small-7x7:5", "small-7x7:3"]);
        assert_eq!(run.datasets[2].patterns, 3);
        assert_eq!(run.statistics[2].name, "small-7x7:3");
        assert_eq!(run.trainings[4].dataset, "small-7x7:3");
        assert_eq!(run.trainings[4].stability.unstable_neurons.len(), 3);
        assert_eq!(run.results.len(), 260);

        let (full, rest) = run.results.split_at(100);
        let (whole, prefix) = rest.split_at(100);
        for (a, b) in full.iter().zip(whole) {
            let mut b = b.clone();
            b.trial.cell -= 8;
            assert_eq!(a, &b);
        }
        let mut seeds = HashMap::new();
        for result in full {
            let key = (result.trial.cell, result.trial.trial, result.trial.target);
            seeds.insert(key, result);
        }
        let mut checked = 0;
        for result in prefix {
            let cell = run.cells[result.trial.cell];
            let key = (
                result.trial.cell - 16,
                result.trial.trial,
                result.trial.target,
            );
            let same = seeds[&key];
            assert_eq!(result.trial.seed, same.trial.seed);
            assert_eq!(result.initial_distance, same.initial_distance);
            if checked < 3 && result.trial.damaged_pixels() > 0 {
                checked += 1;
                let text = format!(
                    "dataset = \"small-7x7\"\npatterns = 3\nseed = {}\n[learning]\n\
                     rule = \"{}\"\n[dynamics]\nmode = \"{}\"\n[input]\n\
                     source = \"pattern\"\nindex = {}\nflips = {}\n",
                    result.trial.seed,
                    ["hebb", "oja"][cell.learning],
                    cell.mode,
                    result.trial.target.unwrap(),
                    result.trial.damaged_pixels(),
                );
                let config = RunConfig::parse(
                    &text,
                    Path::new("../../data/experiments/example/recall.toml"),
                )
                .unwrap();
                let summary = RecallRun::execute(&config).unwrap().summary();
                assert_eq!(summary.energy, result.energy);
                assert_eq!(summary.nearest, result.nearest);
            }
        }
        assert_eq!(checked, 3);

        let error = ExperimentRun::execute(
            &config(&CONFIG.replace(
                "datasets = [\"small-7x7\"]",
                "datasets = [{ name = \"small-7x7\", patterns = 6 }]",
            )),
            |_| {},
        )
        .unwrap_err();
        assert!(error.to_string().contains("which has 5"), "{error}");
    }

    /// [`CONFIG`] with random initial states, 30 trials and attractors.
    fn random_attractors() -> String {
        CONFIG
            .replace(
                "source = \"pattern\"
        noise = [0.0, 0.2]",
                "source = \"random\"",
            )
            .replace(
                "trials = 3",
                "trials = 30
        attractors = true",
            )
    }

    /// The attractors are the same with one and with four threads, and a
    /// `recall` run with the seed, rule and mode of an attractor's first run
    /// ends in its state (or, for a cycle, in one of its two states).
    #[test]
    fn recall_repeats_the_first_run_of_an_attractor() {
        let text = random_attractors();
        let in_pool = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| execute(&text))
        };
        let run = in_pool(1);
        assert_eq!(run, in_pool(4));
        assert!(run.attractors.len() >= 8);
        for attractor in &run.attractors {
            let cell = run.cells[attractor.cell];
            let text = format!(
                "dataset = \"small-7x7\"
seed = {}
[learning]
rule = \"{}\"
                 [dynamics]
mode = \"{}\"
[input]
source = \"random\"
",
                attractor.first.seed,
                ["hebb", "oja"][cell.learning],
                cell.mode,
            );
            let config = RunConfig::parse(
                &text,
                Path::new("../../data/experiments/example/recall.toml"),
            )
            .unwrap();
            let recall = RecallRun::execute(&config).unwrap();
            assert_eq!(recall.outcome.termination, attractor.termination);
            let state = recall.outcome.state.values();
            let expected = attractor.state.values();
            if attractor.termination == Termination::TwoCycle {
                let frames = recall.history.states();
                let previous = frames.row(frames.nrows() - 2);
                assert!(state == expected || previous == expected);
            } else {
                assert_eq!(state, expected);
            }
        }
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
        assert_eq!(
            (summary.nearest_target, summary.final_distance),
            (None, None)
        );
    }
}
