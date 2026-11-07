//! Execution of a capacity run.

use rayon::prelude::*;
use serde::Serialize;

use super::{
    CapacityConfig, CapacityMetadata, CapacitySummary, CapacityTable, PatternSource, Task,
};
use crate::dynamics::Stability;
use crate::learning::LearningRule;
use crate::run::train::METADATA_FILE;
use crate::run::{DatasetInfo, OutputDir, TrainingInfo};
use crate::{PatternSet, Result};

/// Name of the table with one row per pattern set and rule.
pub const CAPACITY_FILE: &str = "capacity.parquet";
/// Name of the file with the aggregates of every (source, `p`, rule).
pub const SUMMARY_FILE: &str = "summary.json";
/// Name of the Hebbian rule (its [`Learning`](crate::run::Learning)
/// display).
const HEBB_LABEL: &str = "hebb";

/// Contents of [`SUMMARY_FILE`].
#[derive(Serialize)]
struct Summary {
    rows: Vec<CapacitySummary>,
}

/// Stability of one pattern set trained with one rule.
#[derive(Debug, Clone, PartialEq)]
pub struct CapacityRow {
    /// The pattern set.
    pub task: Task,
    /// Index of the learning rule in the configuration.
    pub learning: usize,
    /// Result of training.
    pub training: TrainingInfo,
    /// Stability of the patterns.
    pub stability: Stability,
}

impl CapacityRow {
    /// Number of neurons `N` of the pattern set.
    pub fn neurons(&self) -> usize {
        self.training.neurons
    }

    /// Total number of unstable neurons over the patterns.
    pub fn unstable_neurons(&self) -> usize {
        self.stability.unstable_neurons.iter().sum()
    }
}

/// An executed capacity run: the stability of every pattern set with every
/// learning rule.
#[derive(Debug, Clone, PartialEq)]
pub struct CapacityRun {
    /// The datasets whose prefixes were checked, in configuration order.
    pub datasets: Vec<DatasetInfo>,
    /// Names of the learning rules in configuration order.
    pub learning: Vec<String>,
    /// Rows ordered by task (see [`CapacityRun::tasks`]), then rule.
    pub rows: Vec<CapacityRow>,
}

impl CapacityRun {
    /// Loads the datasets and, in parallel over the pattern sets, trains
    /// every rule on every set and checks the stability of its patterns
    /// (with the configured `zero_field`). Every set depends only on its
    /// own [`Task`] and the results are collected in task order, so they do
    /// not depend on the number of threads.
    pub fn execute(config: &CapacityConfig) -> Result<Self> {
        let mut datasets = Vec::new();
        let mut sets = Vec::new();
        if !config.datasets.is_empty() {
            let registry = config.open_registry()?;
            for name in &config.datasets {
                let (info, set) = DatasetInfo::load(&registry, name, config.directory())?;
                datasets.push(info);
                sets.push(set);
            }
        }
        let tasks = Self::tasks(config, &sets);
        let rows = tasks
            .par_iter()
            .map(|task| {
                let set = task.pattern_set(&sets)?;
                config
                    .learning
                    .iter()
                    .enumerate()
                    .map(|(learning, rule)| row(config, task, &set, learning, rule))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        Ok(Self {
            datasets,
            learning: config.learning_labels(),
            rows,
        })
    }

    /// The pattern sets of `config`: for every dataset (`sets` in
    /// configuration order) its prefixes of `1, 2, ..., len` patterns, then
    /// for every random entry and every number of patterns in its range
    /// `trials` random sets.
    pub fn tasks(config: &CapacityConfig, sets: &[PatternSet]) -> Vec<Task> {
        let mut tasks = Vec::new();
        for (index, set) in sets.iter().enumerate() {
            tasks.extend((1..=set.len()).map(|p| Task::prefix(index, p)));
        }
        for random in &config.random {
            for p in random.patterns.values() {
                tasks.extend(
                    (0..config.trials)
                        .map(|trial| Task::random(config.seed, random.neurons, p, trial)),
                );
            }
        }
        tasks
    }

    /// Name of the source of `task`: the dataset name or `random`.
    pub fn source_name(&self, task: &Task) -> &str {
        match task.source {
            PatternSource::Prefix { index } => &self.datasets[index].name,
            PatternSource::Random { .. } => "random",
        }
    }

    /// Aggregates of every (source, `p`, rule), ordered by source (as the
    /// tasks), `p` and rule. Random sets trained with the Hebbian rule get
    /// the [`Theory`](super::Theory) of random patterns.
    pub fn summaries(&self) -> Vec<CapacitySummary> {
        let mut groups: Vec<(PatternSource, usize, usize, Vec<&CapacityRow>)> = Vec::new();
        for row in &self.rows {
            let key = (row.task.source, row.task.patterns);
            match groups
                .iter_mut()
                .find(|(source, p, learning, _)| (*source, *p) == key && *learning == row.learning)
            {
                Some(group) => group.3.push(row),
                None => groups.push((key.0, key.1, row.learning, vec![row])),
            }
        }
        groups
            .into_iter()
            .map(|(source, _, learning, rows)| {
                let label = &self.learning[learning];
                let hebbian_random =
                    matches!(source, PatternSource::Random { .. }) && label == HEBB_LABEL;
                CapacitySummary::new(
                    self.source_name(&rows[0].task).to_owned(),
                    label.clone(),
                    &rows,
                    hebbian_random,
                )
            })
            .collect()
    }

    /// Writes `metadata.json`, [`SUMMARY_FILE`] (`{"rows": [...]}` with the
    /// [`CapacitySummary`] of every (source, `p`, rule)) and
    /// [`CAPACITY_FILE`] (the [`CapacityTable`]). The last two depend only
    /// on the configuration and the code, not on the time or the number of
    /// threads.
    pub fn write(&self, config: &CapacityConfig, out: &OutputDir) -> Result<()> {
        out.write_json(METADATA_FILE, &CapacityMetadata::new(config, self))?;
        out.write_json(
            SUMMARY_FILE,
            &Summary {
                rows: self.summaries(),
            },
        )?;
        out.write_parquet(CAPACITY_FILE, CapacityTable::new(self).batch())
    }
}

/// Trains rule `learning` on `set` of `task` and checks its stability.
fn row(
    config: &CapacityConfig,
    task: &Task,
    set: &PatternSet,
    learning: usize,
    rule: &impl LearningRule,
) -> Result<CapacityRow> {
    let trained = rule.train(set, config.backend)?;
    Ok(CapacityRow {
        task: *task,
        learning,
        training: TrainingInfo::from(&trained),
        stability: Stability::of(&trained.weights, set, config.zero_field),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::ZeroField;
    use crate::learning::Hebb;
    use crate::weights::Backend;
    use std::path::Path;

    const CONFIG: &str = r#"
        seed = 11
        datasets = ["small-7x7"]
        trials = 3

        [[random]]
        neurons = 100
        patterns = { start = 5, end = 15, step = 5 }

        [[learning]]
        rule = "hebb"
        [[learning]]
        rule = "oja"
    "#;

    fn config(text: &str) -> CapacityConfig {
        CapacityConfig::parse(text, Path::new("../../data/experiments/x/config.toml")).unwrap()
    }

    /// `small-7x7` gives prefixes of 1 to 5 patterns, the random entry 3
    /// values of `p` with 3 trials: `5 + 9 = 14` sets, 2 rules each.
    #[test]
    fn checks_every_set_with_every_rule_in_task_order() {
        let run = CapacityRun::execute(&config(CONFIG)).unwrap();
        assert_eq!(run.rows.len(), 28);
        let first: Vec<_> = run.rows[..4]
            .iter()
            .map(|r| (r.task.patterns, r.learning))
            .collect();
        assert_eq!(first, [(1, 0), (1, 1), (2, 0), (2, 1)]);
        let last = run.rows.last().unwrap();
        assert_eq!(
            (last.task.source, last.task.patterns, last.task.trial),
            (PatternSource::Random { neurons: 100 }, 15, 2)
        );
        assert_eq!(last.neurons(), 100);
        assert_eq!(run.source_name(&run.rows[0].task), "small-7x7");
        assert_eq!(run.source_name(&last.task), "random");
    }

    /// The full prefix of a dataset is the dataset: its stability equals a
    /// direct check of the whole set.
    #[test]
    fn full_prefix_matches_the_whole_dataset() {
        let config = config(CONFIG);
        let run = CapacityRun::execute(&config).unwrap();
        let set = config.open_registry().unwrap().load("small-7x7").unwrap();
        let weights = Hebb.train(&set, Backend::Auto).unwrap().weights;
        let expected = Stability::of(&weights, &set, ZeroField::Keep);
        let full = run
            .rows
            .iter()
            .find(|r| r.task.patterns == 5 && r.task.seed.is_none() && r.learning == 0)
            .unwrap();
        assert_eq!(full.stability, expected);
    }

    #[test]
    fn results_do_not_depend_on_the_number_of_threads() {
        let in_pool = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| CapacityRun::execute(&config(CONFIG)).unwrap())
        };
        assert_eq!(in_pool(1), in_pool(4));
    }

    /// Both rules of a random set see the same patterns (same seed), and
    /// Oja's subspace rule, which converges to the projection rule, keeps
    /// every one of `p <= 15` random patterns of `N = 100` stable.
    #[test]
    fn rules_share_random_sets_and_oja_keeps_them_stable() {
        let run = CapacityRun::execute(&config(CONFIG)).unwrap();
        for pair in run.rows.chunks(2) {
            assert_eq!(pair[0].task, pair[1].task);
            assert_eq!((pair[0].learning, pair[1].learning), (0, 1));
            if pair[1].task.seed.is_some() {
                assert!(pair[1].training.converged);
                assert_eq!(pair[1].stability.stable, pair[1].task.patterns);
            }
        }
    }

    #[test]
    fn summaries_group_trials_by_source_size_and_rule() {
        let run = CapacityRun::execute(&config(CONFIG)).unwrap();
        let summaries = run.summaries();
        assert_eq!(summaries.len(), (5 + 3) * 2);
        assert_eq!(summaries[0].source, "small-7x7");
        assert_eq!(summaries[0].trials, 1);
        assert!(summaries[0].deterministic);
        let random = &summaries[10];
        assert_eq!(
            (
                random.source.as_str(),
                random.patterns,
                random.learning.as_str()
            ),
            ("random", 5, "hebb")
        );
        assert_eq!(random.trials, 3);
        assert!(random.theory.is_some());
        assert!(summaries[11].theory.is_none());
    }
}
