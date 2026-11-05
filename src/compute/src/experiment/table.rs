//! The trial table of an experiment.

use std::sync::Arc;

use arrow_array::{
    ArrayRef, BooleanArray, Float64Array, RecordBatch, StringArray, UInt32Array, UInt64Array,
};

use super::{Cell, ExperimentRun, TrialResult};

/// Every trial result as an Arrow record batch, one row per run, in the
/// order of [`ExperimentRun::results`]. Columns (nullable ones are null for
/// random initial states):
///
/// | Column | Type | Content |
/// |---|---|---|
/// | `dataset`, `learning`, `mode` | string | grid cell |
/// | `noise` | float64, nullable | fraction of flipped pixels |
/// | `flips` | uint32 | flipped pixels |
/// | `target` | uint32, nullable | pattern the input is made from |
/// | `trial` | uint32 | trial number in the cell |
/// | `seed` | uint64 | seed of the run, for `recall` |
/// | `steps` | uint32 | applied steps |
/// | `changes` | uint64 | neuron changes |
/// | `termination` | string | `fixed-point`, `two-cycle`, `step-limit` |
/// | `class` | string | final state class |
/// | `energy` | float64 | energy of the final state |
/// | `recovered` | bool, nullable | final state equals the target |
/// | `initial_distance`, `final_distance` | uint32, nullable | Hamming distance to the target |
/// | `final_overlap` | float64, nullable | overlap of the final state with the target |
/// | `nearest` | uint32 | pattern with the largest `abs(m)` |
/// | `nearest_overlap` | float64 | its overlap, negative near a negation |
#[derive(Debug, Clone, PartialEq)]
pub struct TrialTable {
    batch: RecordBatch,
}

impl TrialTable {
    /// The table of `run`.
    pub fn new<'a>(run: &'a ExperimentRun) -> Self {
        let results: &'a [TrialResult] = &run.results;
        let cell = |result: &TrialResult| -> Cell { run.cells[result.trial.cell] };
        let strings = |value: &dyn Fn(&'a TrialResult) -> &'a str| -> ArrayRef {
            Arc::new(StringArray::from_iter_values(results.iter().map(value)))
        };
        let counts = |value: &dyn Fn(&TrialResult) -> usize| -> ArrayRef {
            Arc::new(UInt32Array::from_iter_values(
                results.iter().map(|r| to_u32(value(r))),
            ))
        };
        let optional_counts = |value: &dyn Fn(&TrialResult) -> Option<usize>| -> ArrayRef {
            Arc::new(UInt32Array::from_iter(
                results.iter().map(|r| value(r).map(to_u32)),
            ))
        };
        let floats = |value: &dyn Fn(&TrialResult) -> Option<f64>| -> ArrayRef {
            Arc::new(Float64Array::from_iter(results.iter().map(value)))
        };

        let columns: Vec<(&str, ArrayRef, bool)> = vec![
            (
                "dataset",
                strings(&|r| &run.datasets[cell(r).dataset].name),
                false,
            ),
            (
                "learning",
                strings(&|r| &run.learning[cell(r).learning]),
                false,
            ),
            ("mode", strings(&|r| cell(r).mode.into()), false),
            ("noise", floats(&|r| cell(r).noise), true),
            ("flips", counts(&|r| r.trial.flips), false),
            ("target", optional_counts(&|r| r.trial.target), true),
            ("trial", counts(&|r| r.trial.trial), false),
            (
                "seed",
                Arc::new(UInt64Array::from_iter_values(
                    results.iter().map(|r| r.trial.seed.0),
                )),
                false,
            ),
            ("steps", counts(&|r| r.steps), false),
            (
                "changes",
                Arc::new(UInt64Array::from_iter_values(
                    results.iter().map(|r| r.changes as u64),
                )),
                false,
            ),
            ("termination", strings(&|r| r.termination.into()), false),
            ("class", strings(&|r| r.class.into()), false),
            ("energy", floats(&|r| Some(r.energy)), false),
            (
                "recovered",
                Arc::new(BooleanArray::from_iter(results.iter().map(|r| r.recovered))),
                true,
            ),
            (
                "initial_distance",
                optional_counts(&|r| r.initial_distance),
                true,
            ),
            (
                "final_distance",
                optional_counts(&|r| r.final_distance),
                true,
            ),
            ("final_overlap", floats(&|r| r.final_overlap), true),
            ("nearest", counts(&|r| r.nearest.pattern), false),
            (
                "nearest_overlap",
                floats(&|r| Some(r.nearest.overlap)),
                false,
            ),
        ];
        let batch = RecordBatch::try_from_iter_with_nullable(columns)
            .expect("columns have one value per result and match their types");
        Self { batch }
    }

    /// The record batch.
    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }
}

/// `value` as `u32`; counts, distances and indices of a network with at
/// most `u32::MAX` neurons and trials fit.
fn to_u32(value: usize) -> u32 {
    u32::try_from(value).expect("count fits in u32")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment::ExperimentConfig;
    use arrow_array::Array;
    use arrow_schema::DataType;
    use std::path::Path;

    fn run(input: &str) -> ExperimentRun {
        let text = format!(
            "seed = 1\ndatasets = [\"small-7x7\"]\ntrials = 2\n\
             [[learning]]\nrule = \"hebb\"\n[dynamics]\nmodes = [\"synchronous\"]\n{input}"
        );
        let config =
            ExperimentConfig::parse(&text, Path::new("../../data/experiments/x/config.toml"))
                .unwrap();
        ExperimentRun::execute(&config, |_| {}).unwrap()
    }

    #[test]
    fn has_a_row_per_run_and_typed_columns() {
        let run = run("[input]\nsource = \"pattern\"\nnoise = [0.1]");
        let table = TrialTable::new(&run);
        let batch = table.batch();
        assert_eq!(batch.num_rows(), 2 * 5);
        assert_eq!(batch.num_columns(), 19);
        let schema = batch.schema();
        let field = |name: &str| schema.field_with_name(name).unwrap().clone();
        assert_eq!(field("dataset").data_type(), &DataType::Utf8);
        assert_eq!(field("seed").data_type(), &DataType::UInt64);
        assert_eq!(field("recovered").data_type(), &DataType::Boolean);
        assert!(!field("steps").is_nullable() && field("target").is_nullable());

        let column = |name: &str| batch.column_by_name(name).unwrap().clone();
        let flips = column("flips");
        let flips = flips.as_any().downcast_ref::<UInt32Array>().unwrap();
        assert!(flips.values().iter().all(|&k| k == 5));
        let seeds = column("seed");
        let seeds = seeds.as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(seeds.value(3), run.results[3].trial.seed.0);
        let mode = column("mode");
        let mode = mode.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(mode.value(0), "synchronous");
    }

    #[test]
    fn random_inputs_leave_target_columns_null() {
        let table = TrialTable::new(&run("[input]\nsource = \"random\""));
        let batch = table.batch();
        assert_eq!(batch.num_rows(), 2);
        for name in [
            "noise",
            "target",
            "recovered",
            "final_overlap",
            "final_distance",
        ] {
            let column = batch.column_by_name(name).unwrap();
            assert_eq!(column.null_count(), 2, "{name}");
        }
    }
}
