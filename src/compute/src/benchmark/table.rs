//! The table of a benchmark.

use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, RecordBatch, StringArray, UInt32Array, UInt64Array};

use super::{BenchmarkRun, Measurement};

/// Every measurement of a benchmark as an Arrow record batch, in the order
/// of [`BenchmarkRun::measurements`]. Columns:
///
/// | Column | Type | Content |
/// |---|---|---|
/// | `source` | string | dataset name, or `random` |
/// | `neurons` | uint32 | `N` |
/// | `patterns` | uint32 | `p` |
/// | `backend` | string | `dense` or `factored` |
/// | `learning` | string | learning rule |
/// | `repeat` | uint32 | number of the repeat, from 0 |
/// | `epochs` | uint32 | training epochs, 0 for the Hebbian rule |
/// | `train_seconds` | float64 | time of training |
/// | `weights_bytes` | uint64 | bytes of the stored weights |
/// | `train_peak_bytes` | uint64, nullable | largest growth of the heap during training |
/// | `fields_seconds` | float64 | time of the local fields of one state |
/// | `sync_seconds` | float64 | time of the synchronous recall run |
/// | `sync_steps` | uint32 | its steps |
/// | `async_seconds` | float64 | time of the asynchronous recall run |
/// | `async_sweeps` | uint32 | its sweeps |
/// | `recall_peak_bytes` | uint64, nullable | largest growth of the heap during both recall runs |
#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkTable {
    batch: RecordBatch,
}

impl BenchmarkTable {
    /// The table of `run`.
    pub fn new(run: &BenchmarkRun) -> Self {
        let rows = &run.measurements;
        let counts = |value: &dyn Fn(&Measurement) -> usize| -> ArrayRef {
            Arc::new(UInt32Array::from_iter_values(
                rows.iter().map(|row| to_u32(value(row))),
            ))
        };
        let bytes = |value: &dyn Fn(&Measurement) -> Option<usize>| -> ArrayRef {
            Arc::new(UInt64Array::from_iter(
                rows.iter().map(|row| value(row).map(|b| b as u64)),
            ))
        };
        let seconds = |value: &dyn Fn(&Measurement) -> f64| -> ArrayRef {
            Arc::new(Float64Array::from_iter_values(rows.iter().map(value)))
        };
        let columns: Vec<(&str, ArrayRef, bool)> = vec![
            (
                "source",
                strings(rows.iter().map(|row| run.sets[row.set].source.as_str())),
                false,
            ),
            ("neurons", counts(&|row| run.sets[row.set].neurons), false),
            ("patterns", counts(&|row| run.sets[row.set].patterns), false),
            (
                "backend",
                strings(rows.iter().map(|row| row.backend.into())),
                false,
            ),
            (
                "learning",
                strings(rows.iter().map(|row| run.learning[row.learning].as_str())),
                false,
            ),
            ("repeat", counts(&|row| row.repeat), false),
            ("epochs", counts(&|row| row.epochs), false),
            ("train_seconds", seconds(&|row| row.train_seconds), false),
            (
                "weights_bytes",
                bytes(&|row| Some(row.weights_bytes)),
                false,
            ),
            ("train_peak_bytes", bytes(&|row| row.train_peak_bytes), true),
            ("fields_seconds", seconds(&|row| row.fields_seconds), false),
            ("sync_seconds", seconds(&|row| row.sync_seconds), false),
            ("sync_steps", counts(&|row| row.sync_steps), false),
            ("async_seconds", seconds(&|row| row.async_seconds), false),
            ("async_sweeps", counts(&|row| row.async_sweeps), false),
            (
                "recall_peak_bytes",
                bytes(&|row| row.recall_peak_bytes),
                true,
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

/// A string column of `values`.
fn strings<'a>(values: impl Iterator<Item = &'a str>) -> ArrayRef {
    Arc::new(StringArray::from_iter_values(values))
}

/// `value` as `u32`; numbers of neurons, patterns, repeats, epochs and
/// steps fit.
fn to_u32(value: usize) -> u32 {
    u32::try_from(value).expect("count fits in u32")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{BenchmarkConfig, NoProbe};
    use arrow_array::Array;
    use std::path::Path;

    #[test]
    fn has_a_row_per_measurement() {
        let config = BenchmarkConfig::parse(
            "seed = 1\nrepeats = 2\nbackends = [\"factored\"]\ndense_limit = 0\nnoise = 0.2\n\
             [random]\nneurons = [30]\npatterns = 2\n[[learning]]\nrule = \"hebb\"\n",
            Path::new("../../data/experiments/x/config.toml"),
        )
        .unwrap();
        let run = BenchmarkRun::execute(&config, &NoProbe, |_, _| {}).unwrap();
        let table = BenchmarkTable::new(&run);
        let batch = table.batch();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 16);
        let column = |name: &str| batch.column_by_name(name).unwrap().clone();
        assert_eq!(column("train_peak_bytes").null_count(), 2);
        let backend = column("backend");
        let backend = backend.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(backend.value(0), "factored");
        let bytes = column("weights_bytes");
        let bytes = bytes.as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(bytes.value(0), 2 * 30 + 8 * 30 * 2 + 8 * 30);
        let neurons = column("neurons");
        let neurons = neurons.as_any().downcast_ref::<UInt32Array>().unwrap();
        assert_eq!(neurons.value(1), 30);
    }
}
