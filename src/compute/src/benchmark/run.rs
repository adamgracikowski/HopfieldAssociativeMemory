//! Execution of a benchmark.

use std::time::Instant;

use ndarray::Array1;
use rand::Rng;

use super::{BenchmarkConfig, BenchmarkMetadata, BenchmarkTable, HeapProbe};
use crate::capacity::Task;
use crate::dynamics::{Dynamics, Mode, NoHistory, Outcome, UpdateOrder, ZeroField};
use crate::experiment::ExperimentInput;
use crate::learning::LearningRule;
use crate::patterns::{Pattern, PatternView};
use crate::run::train::METADATA_FILE;
use crate::run::{DatasetInfo, Learning, OutputDir};
use crate::seed::Seed;
use crate::weights::{Backend, FieldTracker, WeightMatrix, Weights};
use crate::{PatternSet, Result};

/// Name of the table with one row per measurement.
pub const BENCHMARK_FILE: &str = "benchmark.parquet";

/// A pattern set of a benchmark.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkSet {
    /// Dataset name, or `random`.
    pub source: String,
    /// Number of neurons `N`.
    pub neurons: usize,
    /// Number of patterns `p`.
    pub patterns: usize,
}

/// Time and memory of one training and recall of one set with one backend
/// and rule. Times are wall-clock seconds; heap sizes are `None` when the
/// heap is not measured.
#[derive(Debug, Clone, PartialEq)]
pub struct Measurement {
    /// Index into [`BenchmarkRun::sets`].
    pub set: usize,
    /// Backend of the weights.
    pub backend: Backend,
    /// Index of the learning rule in the configuration.
    pub learning: usize,
    /// Number of the repeat, from 0.
    pub repeat: usize,
    /// Epochs of training, 0 for the Hebbian rule.
    pub epochs: usize,
    /// Time of training.
    pub train_seconds: f64,
    /// Bytes of the stored weights (see [`Weights::bytes`](crate::weights::Weights::bytes)).
    pub weights_bytes: usize,
    /// Largest growth of the heap during training.
    pub train_peak_bytes: Option<usize>,
    /// Time of the local fields of one random state (after an untimed
    /// warm-up call).
    pub fields_seconds: f64,
    /// Time of one asynchronous sweep from that state.
    pub sweep_seconds: f64,
    /// Time of a synchronous recall run from the damaged input.
    pub sync_seconds: f64,
    /// Steps of that run.
    pub sync_steps: usize,
    /// Time of an asynchronous recall run from the same input.
    pub async_seconds: f64,
    /// Sweeps of that run.
    pub async_sweeps: usize,
    /// Largest growth of the heap during both recall runs.
    pub recall_peak_bytes: Option<usize>,
}

/// An executed benchmark.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkRun {
    /// The registered datasets, in configuration order.
    pub datasets: Vec<DatasetInfo>,
    /// The measured sets: the datasets, then the random sets by number of
    /// neurons as configured.
    pub sets: Vec<BenchmarkSet>,
    /// Names of the learning rules in configuration order.
    pub learning: Vec<String>,
    /// Number of threads of the parallel local fields.
    pub threads: usize,
    /// Whether the heap was measured.
    pub heap_measured: bool,
    /// Measurements ordered by set, backend (as configured), rule and
    /// repeat.
    pub measurements: Vec<Measurement>,
}

impl BenchmarkRun {
    /// Measures every set x backend x rule `repeats` times, passing each
    /// measurement to `on_measurement`: training, local fields of a random
    /// state, one asynchronous sweep, and synchronous and asynchronous recall
    /// from pattern 0 with `round(noise N)` flips. Inputs depend only on
    /// seed, source, `N` and repeat.
    pub fn execute(
        config: &BenchmarkConfig,
        probe: &impl HeapProbe,
        mut on_measurement: impl FnMut(&BenchmarkSet, &Measurement),
    ) -> Result<Self> {
        let mut datasets = Vec::new();
        let mut loaded = Vec::new();
        if !config.datasets.is_empty() {
            let registry = config.open_registry()?;
            for name in &config.datasets {
                let (info, set) = DatasetInfo::load(&registry, name, None, config.directory())?;
                loaded.push((name.clone(), set));
                datasets.push(info);
            }
        }
        let mut sources: Vec<(String, Option<PatternSet>, usize, usize)> = loaded
            .into_iter()
            .map(|(name, set)| {
                let (n, p) = (set.neurons(), set.len());
                (name, Some(set), n, p)
            })
            .collect();
        if let Some(random) = &config.random {
            sources.extend(
                random
                    .neurons
                    .iter()
                    .map(|&n| ("random".to_owned(), None, n, random.patterns)),
            );
        }
        let neurons: Vec<usize> = sources.iter().map(|source| source.2).collect();
        config.check_measured(&neurons)?;

        let heap_measured = probe.start().is_some();
        let mut sets = Vec::new();
        let mut measurements = Vec::new();
        for (index, (source, set, neurons, patterns)) in sources.into_iter().enumerate() {
            let set = match set {
                Some(set) => set,
                None => Task::random(config.seed, neurons, patterns, 0).pattern_set(&[])?,
            };
            let info = BenchmarkSet {
                source,
                neurons,
                patterns,
            };
            for &backend in &config.backends {
                if !config.measures(backend, neurons) {
                    continue;
                }
                for (learning, rule) in config.learning.iter().enumerate() {
                    for repeat in 0..config.repeats {
                        let measurement =
                            measure(config, probe, &info, &set, backend, rule, repeat)?;
                        let measurement = Measurement {
                            set: index,
                            learning,
                            ..measurement
                        };
                        on_measurement(&info, &measurement);
                        measurements.push(measurement);
                    }
                }
            }
            sets.push(info);
        }
        Ok(Self {
            datasets,
            sets,
            learning: config.learning_labels(),
            threads: rayon::current_num_threads(),
            heap_measured,
            measurements,
        })
    }

    /// Writes `metadata.json` and [`BENCHMARK_FILE`]; unlike other results,
    /// the table is machine-dependent.
    pub fn write(&self, config: &BenchmarkConfig, out: &OutputDir) -> Result<()> {
        out.write_json(METADATA_FILE, &BenchmarkMetadata::new(config, self))?;
        out.write_parquet(BENCHMARK_FILE, BenchmarkTable::new(self).batch())
    }
}

/// One measurement of `set` with `backend` and `rule`; `set` and
/// `learning` of the result are left 0 for the caller.
fn measure(
    config: &BenchmarkConfig,
    probe: &impl HeapProbe,
    info: &BenchmarkSet,
    set: &PatternSet,
    backend: Backend,
    rule: &Learning,
    repeat: usize,
) -> Result<Measurement> {
    let ((trained, train_seconds), train_peak_bytes) =
        probe.measure(|| timed(|| rule.train(set, backend)));
    let trained = trained?;
    let weights = &trained.weights;
    let n = info.neurons;

    let seed = Seed::derive(config.seed, &[&info.source, &n, &repeat]);
    let mut rng = seed.input_rng();
    let state = Pattern::random(n, &mut rng)?;
    let mut fields = Array1::zeros(n);
    weights.local_fields_into(state.values(), fields.view_mut());
    let ((), fields_seconds) =
        timed(|| weights.local_fields_into(state.values(), fields.view_mut()));
    let sweep_seconds = sweep(weights, state.view(), &mut rng);
    let flips = ExperimentInput::flips(config.noise, n);
    let input = set.pattern(0).flipped(flips, &mut rng)?;

    let recall = |mode: Mode| -> (Result<Outcome>, f64) {
        let dynamics = Dynamics {
            mode,
            ..Dynamics::default()
        };
        timed(|| {
            dynamics.run(
                weights,
                input.view(),
                &mut seed.dynamics_rng(),
                &mut NoHistory,
            )
        })
    };
    let ((sync, asynchronous), recall_peak_bytes) =
        probe.measure(|| (recall(Mode::Synchronous), recall(Mode::Asynchronous)));
    let (sync, sync_seconds) = (sync.0?, sync.1);
    let (asynchronous, async_seconds) = (asynchronous.0?, asynchronous.1);

    Ok(Measurement {
        set: 0,
        backend,
        learning: 0,
        repeat,
        epochs: trained.epochs,
        train_seconds,
        weights_bytes: weights.bytes(),
        train_peak_bytes,
        fields_seconds,
        sweep_seconds,
        sync_seconds,
        sync_steps: sync.steps,
        async_seconds,
        async_sweeps: asynchronous.steps,
        recall_peak_bytes,
    })
}

/// Seconds of one asynchronous sweep from `state`, excluding the setup.
fn sweep(weights: &Weights, state: PatternView<'_>, rng: &mut impl Rng) -> f64 {
    let n = state.len();
    let mut tracker = FieldTracker::new(weights, state);
    let mut order = Vec::with_capacity(n);
    UpdateOrder::Permutation.fill_sweep(&mut order, n, rng);
    let ((), seconds) = timed(|| {
        for &i in &order {
            let x = tracker.state().values()[i];
            if ZeroField::Keep.activate(tracker.field(i), x) != x {
                tracker.flip(i);
            }
        }
    });
    seconds
}

/// Result of `work` and its wall-clock time in seconds.
fn timed<T>(work: impl FnOnce() -> T) -> (T, f64) {
    let start = Instant::now();
    let result = work();
    (result, start.elapsed().as_secs_f64())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::NoProbe;
    use std::cell::Cell;
    use std::path::Path;

    const CONFIG: &str = r#"
        seed = 7
        datasets = ["small-7x7"]
        repeats = 2
        backends = ["dense", "factored"]
        dense_limit = 64
        noise = 0.1

        [random]
        neurons = [64, 128]
        patterns = 3

        [[learning]]
        rule = "hebb"
        [[learning]]
        rule = "oja"
    "#;

    fn config() -> BenchmarkConfig {
        BenchmarkConfig::parse(CONFIG, Path::new("../../data/experiments/x/config.toml")).unwrap()
    }

    /// `small-7x7` (`N = 49`) and the random set of `N = 64` are measured
    /// with both backends, the one of `N = 128 > dense_limit` only with
    /// factored weights: `(2 + 2 + 1) * 2 rules * 2 repeats = 20`
    /// measurements. Dense weights of `N = 64` take `8 * 64^2 = 32768`
    /// bytes, factored ones of `p = 3` patterns `3 * 64 + 8 * 64 * 3 + 8 *
    /// 64 = 192 + 1536 + 512 = 2240` bytes.
    #[test]
    fn measures_every_set_backend_rule_and_repeat() {
        let run = BenchmarkRun::execute(&config(), &NoProbe, |_, _| {}).unwrap();
        assert_eq!(run.measurements.len(), 20);
        assert_eq!(
            run.sets
                .iter()
                .map(|s| (s.source.as_str(), s.neurons, s.patterns))
                .collect::<Vec<_>>(),
            [("small-7x7", 49, 5), ("random", 64, 3), ("random", 128, 3)]
        );
        let keys: Vec<_> = run.measurements[..5]
            .iter()
            .map(|m| (m.set, m.backend, m.learning, m.repeat))
            .collect();
        assert_eq!(
            keys,
            [
                (0, Backend::Dense, 0, 0),
                (0, Backend::Dense, 0, 1),
                (0, Backend::Dense, 1, 0),
                (0, Backend::Dense, 1, 1),
                (0, Backend::Factored, 0, 0),
            ]
        );
        let last = &run.measurements[16..];
        assert!(
            last.iter()
                .all(|m| m.set == 2 && m.backend == Backend::Factored)
        );
        let bytes = |set, backend| {
            run.measurements
                .iter()
                .find(|m| m.set == set && m.backend == backend)
                .unwrap()
                .weights_bytes
        };
        assert_eq!(bytes(1, Backend::Dense), 32768);
        assert_eq!(bytes(1, Backend::Factored), 2240);
        assert!(
            run.measurements
                .iter()
                .all(|m| m.train_peak_bytes.is_none())
        );
        assert!(!run.heap_measured);
        assert!(run.measurements.iter().map(|m| m.sync_steps).sum::<usize>() > 0);
        assert!(
            run.measurements
                .iter()
                .all(|m| m.fields_seconds > 0.0 && m.sweep_seconds > 0.0)
        );
        let oja = run.measurements.iter().find(|m| m.learning == 1).unwrap();
        assert!(oja.epochs > 0);
    }

    /// Both backends of the same set, rule and repeat start from the same
    /// input and reach the same final state, so they take the same number
    /// of steps.
    #[test]
    fn backends_run_the_same_dynamics() {
        let run = BenchmarkRun::execute(&config(), &NoProbe, |_, _| {}).unwrap();
        for dense in run
            .measurements
            .iter()
            .filter(|m| m.backend == Backend::Dense)
        {
            let factored = run
                .measurements
                .iter()
                .find(|m| {
                    (m.set, m.learning, m.repeat, m.backend)
                        == (dense.set, dense.learning, dense.repeat, Backend::Factored)
                })
                .unwrap();
            assert_eq!(
                (dense.sync_steps, dense.async_sweeps),
                (factored.sync_steps, factored.async_sweeps)
            );
        }
    }

    /// A probe counting its starts: one when the run checks it, then two
    /// per measurement (training and recall).
    struct Counting {
        starts: Cell<usize>,
    }

    impl HeapProbe for Counting {
        fn start(&self) -> Option<usize> {
            self.starts.set(self.starts.get() + 1);
            Some(1000)
        }

        fn peak(&self) -> Option<usize> {
            Some(1500)
        }
    }

    #[test]
    fn measures_the_heap_around_training_and_recall() {
        let probe = Counting {
            starts: Cell::new(0),
        };
        let mut seen = 0;
        let run = BenchmarkRun::execute(&config(), &probe, |_, _| seen += 1).unwrap();
        assert_eq!(seen, 20);
        assert_eq!(probe.starts.get(), 1 + 2 * 20);
        assert!(run.heap_measured);
        assert!(
            run.measurements
                .iter()
                .all(|m| { m.train_peak_bytes == Some(500) && m.recall_peak_bytes == Some(500) })
        );
    }
}
