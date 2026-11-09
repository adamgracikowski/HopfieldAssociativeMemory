//! The table of aligned local fields of the stored patterns.

use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, RecordBatch, StringArray, UInt32Array};

use super::ExperimentRun;

/// The [aligned fields](crate::dynamics::Stability::aligned_fields)
/// `xi_i^mu h_i` of every training of an experiment as an Arrow record
/// batch, one row per (training, pattern, neuron), ordered by training (as
/// [`ExperimentRun::trainings`]), then pattern, then neuron. Columns:
///
/// | Column | Type | Content |
/// |---|---|---|
/// | `dataset`, `learning` | string | training |
/// | `pattern` | uint32 | stored pattern `mu` |
/// | `neuron` | uint32 | neuron `i`, row-major pixel index |
/// | `field` | float64 | `xi_i^mu h_i` for `h = W xi^mu` |
///
/// A neuron with a negative field would change; a zero field is decided by
/// the `sgn(0)` convention of the dynamics.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldTable {
    batch: RecordBatch,
}

impl FieldTable {
    /// The table of `run`, which must hold the fields of every training.
    ///
    /// # Panics
    ///
    /// Panics if `run` has no fields for some training (the experiment ran
    /// without `stability_fields`).
    pub fn new(run: &ExperimentRun) -> Self {
        assert_eq!(
            run.fields.len(),
            run.trainings.len(),
            "the run keeps the aligned fields of every training"
        );
        let mut dataset = Vec::new();
        let mut learning = Vec::new();
        let mut pattern = Vec::new();
        let mut neuron = Vec::new();
        let mut field = Vec::new();
        for (training, fields) in run.trainings.iter().zip(&run.fields) {
            for ((mu, i), &value) in fields.indexed_iter() {
                dataset.push(training.dataset.as_str());
                learning.push(training.learning.as_str());
                pattern.push(to_u32(mu));
                neuron.push(to_u32(i));
                field.push(value);
            }
        }
        let columns: Vec<(&str, ArrayRef)> = vec![
            ("dataset", Arc::new(StringArray::from(dataset))),
            ("learning", Arc::new(StringArray::from(learning))),
            ("pattern", Arc::new(UInt32Array::from(pattern))),
            ("neuron", Arc::new(UInt32Array::from(neuron))),
            ("field", Arc::new(Float64Array::from(field))),
        ];
        let batch = RecordBatch::try_from_iter_with_nullable(
            columns
                .into_iter()
                .map(|(name, array)| (name, array, false)),
        )
        .expect("columns have one value per row");
        Self { batch }
    }

    /// The record batch.
    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }
}

/// `value` as `u32`; pattern and neuron indices fit.
fn to_u32(value: usize) -> u32 {
    u32::try_from(value).expect("index fits in u32")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::{Stability, ZeroField};
    use crate::experiment::ExperimentConfig;
    use arrow_array::Array;
    use std::path::Path;

    /// `small-7x7` has `p = 5` patterns of `N = 49` neurons: 245 rows per
    /// rule. Row `mu * 49 + i` of a training holds neuron `i` of pattern
    /// `mu`, and the negative fields of a pattern are its unstable neurons.
    #[test]
    fn has_a_row_per_pattern_and_neuron_of_every_training() {
        let text = "seed = 1\ndatasets = [\"small-7x7\"]\ntrials = 1\nstability_fields = true\n\
                    [[learning]]\nrule = \"hebb\"\n[[learning]]\nrule = \"oja\"\n\
                    [dynamics]\nmodes = [\"synchronous\"]\n\
                    [input]\nsource = \"pattern\"\nnoise = [0.0]\n";
        let config =
            ExperimentConfig::parse(text, Path::new("../../data/experiments/x/config.toml"))
                .unwrap();
        let run = ExperimentRun::execute(&config, |_| {}).unwrap();
        let table = FieldTable::new(&run);
        let batch = table.batch();
        assert_eq!(batch.num_rows(), 2 * 245);
        let column = |name: &str| batch.column_by_name(name).unwrap().clone();
        let learning = column("learning");
        let learning = learning.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(
            (learning.value(244), learning.value(245)),
            ("hebb", "oja (subspace)")
        );
        let pattern = column("pattern");
        let pattern = pattern.as_any().downcast_ref::<UInt32Array>().unwrap();
        let neuron = column("neuron");
        let neuron = neuron.as_any().downcast_ref::<UInt32Array>().unwrap();
        assert_eq!((pattern.value(52), neuron.value(52)), (1, 3));
        let field = column("field");
        let field = field.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(field.value(52), run.fields[0][[1, 3]]);
        assert_eq!(field.null_count(), 0);

        let set = config.open_registry().unwrap().load("small-7x7").unwrap();
        for (training, fields) in run.trainings.iter().zip(&run.fields) {
            let stability = Stability::from_aligned_fields(fields.view(), &set, ZeroField::Keep);
            assert_eq!(stability, training.stability);
        }
    }
}
