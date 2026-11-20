//! Statistics of the datasets of an experiment.

use serde::Serialize;

use crate::patterns::summary::Summary;

/// Size, pixel balance and overlaps of one dataset, as written to
/// `datasets.json`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DatasetStatistics {
    /// Dataset name.
    pub name: String,
    /// Bitmap width.
    pub width: usize,
    /// Bitmap height.
    pub height: usize,
    /// Number of patterns `p`.
    pub patterns: usize,
    /// Number of neurons `N`.
    pub neurons: usize,
    /// Load ratio `p / N`.
    pub load: f64,
    /// Fraction of pixels equal to `1` in every pattern.
    pub fraction_on: Vec<f64>,
    /// Overlap matrix `m^{mu nu}`, row by row.
    pub overlaps: Vec<Vec<f64>>,
    /// For every pattern, the sum of `|m^{mu nu}|` over the other patterns.
    pub overlap_sums: Vec<f64>,
    /// Largest `|m^{mu nu}|` over pairs of different patterns; `None` for a
    /// single pattern.
    pub max_abs_overlap: Option<MaxOverlap>,
}

/// The largest absolute overlap of two different patterns.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct MaxOverlap {
    /// `|m^{mu nu}|`.
    pub value: f64,
    /// The first pair `(mu, nu)`, `mu < nu`, that reaches it.
    pub patterns: [usize; 2],
}

impl From<&Summary> for DatasetStatistics {
    fn from(summary: &Summary) -> Self {
        Self {
            name: summary.name.clone(),
            width: summary.width,
            height: summary.height,
            patterns: summary.patterns,
            neurons: summary.neurons,
            load: summary.load(),
            fraction_on: summary.fraction_on.clone(),
            overlaps: summary
                .overlaps
                .rows()
                .into_iter()
                .map(|row| row.to_vec())
                .collect(),
            overlap_sums: summary.overlap_sums(),
            max_abs_overlap: summary.max_abs_overlap().map(|(value, mu, nu)| MaxOverlap {
                value,
                patterns: [mu, nu],
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PatternSet;
    use ndarray::array;

    /// Patterns `(1, 1, 1, 1)` and `(1, 1, -1, 1)` of a 2x2 bitmap: overlap
    /// `(1 + 1 - 1 + 1) / 4 = 0.5`, fractions of `1` equal to 1 and 0.75.
    #[test]
    fn serializes_the_summary() {
        let set = PatternSet::new(2, 2, array![[1, 1, 1, 1], [1, 1, -1, 1]]).unwrap();
        let statistics = DatasetStatistics::from(&Summary::new("s", "Two.", &set));
        let json = serde_json::to_value(&statistics).unwrap();
        assert_eq!(json["name"], "s");
        assert_eq!(json["load"], 0.5);
        assert_eq!(json["fraction_on"], serde_json::json!([1.0, 0.75]));
        assert_eq!(
            json["overlaps"],
            serde_json::json!([[1.0, 0.5], [0.5, 1.0]])
        );
        assert_eq!(json["overlap_sums"], serde_json::json!([0.5, 0.5]));
        assert_eq!(
            json["max_abs_overlap"],
            serde_json::json!({"value": 0.5, "patterns": [0, 1]})
        );
    }
}
