//! The table of distinct final states of an experiment.

use std::sync::Arc;

use arrow_array::types::{Float64Type, Int8Type, UInt32Type};
use arrow_array::{
    ArrayRef, Float64Array, ListArray, RecordBatch, StringArray, UInt32Array, UInt64Array,
};

use super::{Attractor, Cell, ExperimentRun};

/// The [attractors](Attractor) of every cell as an Arrow record batch, one
/// row per (cell, distinct final state); mixture columns are null for fewer
/// than three stored patterns:
///
/// | Column | Type | Content |
/// |---|---|---|
/// | `dataset`, `learning`, `mode` | string | grid cell |
/// | `noise` | float64, nullable | damage level; null for random initial states |
/// | `rank` | uint32 | position in the cell by `count`, from 0 |
/// | `count` | uint32 | runs that ended in the state |
/// | `fraction` | float64 | `count` over the runs of the cell |
/// | `class` | string | final state class |
/// | `termination` | string | `fixed-point`, `two-cycle`, `step-limit` |
/// | `energy` | float64 | energy of the state |
/// | `nearest` | uint32 | pattern with the largest `abs(m)` |
/// | `nearest_overlap` | float64 | its overlap, negative near a negation |
/// | `overlaps` | list of float64 | overlap with every stored pattern |
/// | `mixture` | uint32, nullable | number of patterns of the mixture equal to the state; null if none is |
/// | `mixture_size` | uint32, nullable | number of patterns of the closest mixture |
/// | `mixture_distance` | uint32, nullable | Hamming distance to the closest mixture |
/// | `mixture_patterns` | list of uint32, nullable | its patterns, in increasing order |
/// | `mixture_signs` | list of int8, nullable | their signs |
/// | `target` | uint32, nullable | target of the first run; null for random initial states |
/// | `trial` | uint32 | trial number of the first run |
/// | `seed` | uint64 | seed of the first run, for `recall` |
/// | `state` | list of int8 | the state, row-major |
#[derive(Debug, Clone, PartialEq)]
pub struct AttractorTable {
    batch: RecordBatch,
}

impl AttractorTable {
    /// The table of `run`; empty if the run kept no attractors.
    pub fn new<'a>(run: &'a ExperimentRun) -> Self {
        let rows: &'a [Attractor] = &run.attractors;
        let labels: Vec<String> = run.datasets.iter().map(|info| info.label()).collect();
        let cell = |row: &Attractor| -> Cell { run.cells[row.cell] };
        let strings = |value: &dyn Fn(&'a Attractor) -> &'a str| -> ArrayRef {
            Arc::new(StringArray::from_iter_values(rows.iter().map(value)))
        };
        let counts = |value: &dyn Fn(&Attractor) -> usize| -> ArrayRef {
            Arc::new(UInt32Array::from_iter_values(
                rows.iter().map(|row| to_u32(value(row))),
            ))
        };
        let optional_counts = |value: &dyn Fn(&Attractor) -> Option<usize>| -> ArrayRef {
            Arc::new(UInt32Array::from_iter(
                rows.iter().map(|row| value(row).map(to_u32)),
            ))
        };
        let floats = |value: &dyn Fn(&Attractor) -> Option<f64>| -> ArrayRef {
            Arc::new(Float64Array::from_iter(rows.iter().map(value)))
        };

        let columns: Vec<(&str, ArrayRef, bool)> = vec![
            (
                "dataset",
                Arc::new(StringArray::from_iter_values(
                    rows.iter().map(|row| labels[cell(row).dataset].as_str()),
                )),
                false,
            ),
            (
                "learning",
                strings(&|row| &run.learning[cell(row).learning]),
                false,
            ),
            ("mode", strings(&|row| cell(row).mode.into()), false),
            ("noise", floats(&|row| cell(row).noise), true),
            ("rank", counts(&|row| row.rank), false),
            ("count", counts(&|row| row.count), false),
            ("fraction", floats(&|row| Some(row.fraction)), false),
            ("class", strings(&|row| row.class.into()), false),
            ("termination", strings(&|row| row.termination.into()), false),
            ("energy", floats(&|row| Some(row.energy)), false),
            ("nearest", counts(&|row| row.nearest.pattern), false),
            (
                "nearest_overlap",
                floats(&|row| Some(row.nearest.overlap)),
                false,
            ),
            (
                "overlaps",
                Arc::new(ListArray::from_iter_primitive::<Float64Type, _, _>(
                    rows.iter()
                        .map(|row| Some(row.overlaps.iter().copied().map(Some))),
                )),
                false,
            ),
            (
                "mixture",
                optional_counts(&|row| {
                    row.mixture
                        .as_ref()
                        .filter(|mixture| mixture.is_exact())
                        .map(|mixture| mixture.size)
                }),
                true,
            ),
            (
                "mixture_size",
                optional_counts(&|row| row.mixture.as_ref().map(|mixture| mixture.size)),
                true,
            ),
            (
                "mixture_distance",
                optional_counts(&|row| row.mixture.as_ref().map(|mixture| mixture.distance)),
                true,
            ),
            (
                "mixture_patterns",
                Arc::new(ListArray::from_iter_primitive::<UInt32Type, _, _>(
                    rows.iter().map(|row| {
                        row.mixture
                            .as_ref()
                            .map(|mixture| mixture.patterns.iter().map(|&mu| Some(to_u32(mu))))
                    }),
                )),
                true,
            ),
            (
                "mixture_signs",
                Arc::new(ListArray::from_iter_primitive::<Int8Type, _, _>(
                    rows.iter().map(|row| {
                        row.mixture
                            .as_ref()
                            .map(|mixture| mixture.signs.iter().copied().map(Some))
                    }),
                )),
                true,
            ),
            ("target", optional_counts(&|row| row.first.target), true),
            ("trial", counts(&|row| row.first.trial), false),
            (
                "seed",
                Arc::new(UInt64Array::from_iter_values(
                    rows.iter().map(|row| row.first.seed.0),
                )),
                false,
            ),
            (
                "state",
                Arc::new(ListArray::from_iter_primitive::<Int8Type, _, _>(
                    rows.iter()
                        .map(|row| Some(row.state.values().to_vec().into_iter().map(Some))),
                )),
                false,
            ),
        ];
        let batch = RecordBatch::try_from_iter_with_nullable(columns)
            .expect("columns have one value per row and match their types");
        Self { batch }
    }

    /// The record batch.
    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }
}

fn to_u32(value: usize) -> u32 {
    u32::try_from(value).expect("count fits in u32")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment::{ExperimentConfig, FinalClass};
    use arrow_array::{Array, Float64Array, Int8Array, UInt32Array};
    use arrow_schema::DataType;
    use std::path::Path;

    fn run(text: &str) -> ExperimentRun {
        let config =
            ExperimentConfig::parse(text, Path::new("../../data/experiments/x/config.toml"))
                .unwrap();
        ExperimentRun::execute(&config, |_| {}).unwrap()
    }

    const CONFIG: &str = "seed = 1\ndatasets = [\"small-7x7\"]\ntrials = 40\nattractors = true\n\
                          [[learning]]\nrule = \"hebb\"\n[[learning]]\nrule = \"oja\"\n\
                          [dynamics]\nmodes = [\"synchronous\", \"asynchronous\"]\n\
                          [input]\nsource = \"random\"\n";

    /// `small-7x7`: `N = 49`, `p = 5`, 40 runs per cell.
    #[test]
    fn has_a_row_per_distinct_final_state() {
        let run = run(CONFIG);
        let table = AttractorTable::new(&run);
        let batch = table.batch();
        assert_eq!(batch.num_rows(), run.attractors.len());
        assert_eq!(batch.num_columns(), 22);
        let schema = batch.schema();
        let field = |name: &str| schema.field_with_name(name).unwrap().clone();
        assert!(matches!(field("state").data_type(), DataType::List(_)));
        assert!(field("mixture").is_nullable() && !field("count").is_nullable());

        for (cell, summary) in run.summaries().iter().enumerate() {
            let rows: Vec<&Attractor> = run.attractors.iter().filter(|a| a.cell == cell).collect();
            assert_eq!(rows.iter().map(|a| a.count).sum::<usize>(), 40);
            assert!(rows.windows(2).all(|w| w[0].count >= w[1].count));
            assert!(rows.iter().enumerate().all(|(k, a)| a.rank == k));
            let count = |class: FinalClass| {
                rows.iter()
                    .filter(|a| a.class == class)
                    .map(|a| a.count)
                    .sum::<usize>()
            };
            let classes = summary.classes;
            assert_eq!(count(FinalClass::Pattern), classes.pattern.count);
            assert_eq!(count(FinalClass::Negation), classes.negation.count);
            assert_eq!(count(FinalClass::Spurious), classes.spurious.count);
            assert_eq!(count(FinalClass::Cycle), classes.cycle.count);
            assert_eq!(count(FinalClass::Unconverged), classes.unconverged.count);
            for a in &rows {
                assert_eq!(a.state.len(), 49);
                assert_eq!(a.overlaps.len(), 5);
                assert!(a.mixture.is_some());
                assert_eq!(a.first.target, None);
            }
        }

        let column = |name: &str| batch.column_by_name(name).unwrap().clone();
        let states = column("state");
        let states = states.as_any().downcast_ref::<ListArray>().unwrap();
        let first = states.value(0);
        let first = first.as_any().downcast_ref::<Int8Array>().unwrap();
        assert_eq!(
            first.values().to_vec(),
            run.attractors[0].state.values().to_vec()
        );
        assert_eq!(column("noise").null_count(), batch.num_rows());
        assert_eq!(column("target").null_count(), batch.num_rows());
        let mixture = column("mixture");
        let exact = run
            .attractors
            .iter()
            .filter(|a| a.mixture.as_ref().is_some_and(|m| m.is_exact()))
            .count();
        assert_eq!(mixture.len() - mixture.null_count(), exact);
    }

    /// 4 trials x 5 targets = 20 runs per cell, 5 (one trial) in the two
    /// synchronous cells without noise.
    #[test]
    fn keeps_the_noise_level_and_target_of_damaged_inputs() {
        let run = run(&CONFIG.replace("trials = 40", "trials = 4").replace(
            "source = \"random\"\n",
            "source = \"pattern\"\nnoise = [0.0, 0.2]\n",
        ));
        assert_eq!(run.cells.len(), 2 * 2 * 2);
        for (cell, summary) in run.summaries().iter().enumerate() {
            let runs = if run.cells[cell].is_deterministic(0) && summary.noise == Some(0.0) {
                5
            } else {
                20
            };
            assert_eq!(summary.runs, runs);
            let rows: Vec<&Attractor> = run.attractors.iter().filter(|a| a.cell == cell).collect();
            assert_eq!(rows.iter().map(|a| a.count).sum::<usize>(), runs);
            for a in rows {
                assert!(a.first.target.is_some());
                assert_eq!(a.first.cell, cell);
                assert!(run.results.iter().any(|r| r.trial == a.first));
            }
        }

        let batch = AttractorTable::new(&run).batch().clone();
        let column = |name: &str| batch.column_by_name(name).unwrap().clone();
        let noise = column("noise");
        let noise = noise.as_any().downcast_ref::<Float64Array>().unwrap();
        let target = column("target");
        let target = target.as_any().downcast_ref::<UInt32Array>().unwrap();
        assert_eq!((noise.null_count(), target.null_count()), (0, 0));
        for (row, a) in run.attractors.iter().enumerate() {
            assert_eq!(Some(noise.value(row)), run.cells[a.cell].noise);
            assert_eq!(Some(target.value(row) as usize), a.first.target);
        }
    }

    #[test]
    fn is_empty_without_the_option() {
        let run = run(&CONFIG.replace("attractors = true\n", ""));
        assert!(run.attractors.is_empty());
        assert_eq!(AttractorTable::new(&run).batch().num_rows(), 0);
    }
}
