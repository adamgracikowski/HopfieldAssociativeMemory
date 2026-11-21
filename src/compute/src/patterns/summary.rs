//! Statistics of a pattern set.

use std::fmt;

use ndarray::Array2;

use super::PatternSet;

/// Size, pixel balance and mutual overlaps of a pattern set.
///
/// [`Display`](fmt::Display) prints a plain-text report with three decimal
/// places.
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    /// Dataset name.
    pub name: String,
    /// What the patterns show.
    pub description: String,
    /// Bitmap width in pixels.
    pub width: usize,
    /// Bitmap height in pixels.
    pub height: usize,
    /// Number of patterns `p`.
    pub patterns: usize,
    /// Number of neurons `N = width * height`.
    pub neurons: usize,
    /// Fraction of pixels equal to `1` in every pattern.
    pub fraction_on: Vec<f64>,
    /// Overlap matrix `m^{mu nu}` (see [`PatternSet::overlap_matrix`]).
    pub overlaps: Array2<f64>,
}

impl Summary {
    /// Computes the statistics of `set`.
    pub fn new(name: &str, description: &str, set: &PatternSet) -> Self {
        Self {
            name: name.to_owned(),
            description: description.to_owned(),
            width: set.width(),
            height: set.height(),
            patterns: set.len(),
            neurons: set.neurons(),
            fraction_on: set.iter().map(|pattern| pattern.fraction_on()).collect(),
            overlaps: set.overlap_matrix(),
        }
    }

    /// Load ratio `p/N`.
    pub fn load(&self) -> f64 {
        self.patterns as f64 / self.neurons as f64
    }

    /// For every pattern `mu`, the sum of `|m^{mu nu}|` over the other
    /// patterns `nu`. With Hebbian weights pattern `mu` is stable when this
    /// sum is below the signal `1 - p/N` (a sufficient condition: the
    /// crosstalk at every neuron is at most the sum).
    pub fn overlap_sums(&self) -> Vec<f64> {
        self.overlaps
            .rows()
            .into_iter()
            .enumerate()
            .map(|(mu, row)| {
                row.iter()
                    .enumerate()
                    .filter(|&(nu, _)| nu != mu)
                    .map(|(_, m)| m.abs())
                    .sum()
            })
            .collect()
    }

    /// Largest `|m^{mu nu}|` over pairs of different patterns, with the first
    /// pair `(mu, nu)`, `mu < nu`, that reaches it. `None` for one pattern.
    pub fn max_abs_overlap(&self) -> Option<(f64, usize, usize)> {
        let p = self.patterns;
        (0..p)
            .flat_map(|mu| (mu + 1..p).map(move |nu| (mu, nu)))
            .map(|(mu, nu)| (self.overlaps[[mu, nu]].abs(), mu, nu))
            .reduce(|best, next| if next.0 > best.0 { next } else { best })
    }
}

/// Three decimals, or three significant digits below 0.001.
fn format_load(load: f64) -> String {
    if load >= 1e-3 {
        format!("{load:.3}")
    } else {
        format!("{load:.2e}")
    }
}

impl fmt::Display for Summary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "dataset      {}", self.name)?;
        writeln!(f, "description  {}", self.description)?;
        writeln!(
            f,
            "bitmap       {}x{} (N = {})",
            self.width, self.height, self.neurons
        )?;
        writeln!(
            f,
            "patterns     p = {}, p/N = {}",
            self.patterns,
            format_load(self.load())
        )?;
        match self.max_abs_overlap() {
            Some((m, mu, nu)) => writeln!(f, "max |m|      {m:.3} (patterns {mu} and {nu})")?,
            None => writeln!(f, "max |m|      -")?,
        }

        writeln!(f, "\nfraction of 1 per pattern")?;
        for (mu, fraction) in self.fraction_on.iter().enumerate() {
            writeln!(f, "{mu:>5}  {fraction:.3}")?;
        }

        writeln!(f, "\noverlaps m (row mu, column nu)")?;
        write!(f, "     ")?;
        for nu in 0..self.patterns {
            write!(f, " {nu:>6}")?;
        }
        writeln!(f)?;
        for (mu, row) in self.overlaps.rows().into_iter().enumerate() {
            write!(f, "{mu:>5}")?;
            for m in row {
                write!(f, " {m:>6.3}")?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn summary() -> Summary {
        let set =
            PatternSet::new(2, 2, array![[1, 1, 1, 1], [1, 1, -1, -1], [-1, 1, 1, 1]]).unwrap();
        Summary::new("s", "Three patterns.", &set)
    }

    #[test]
    fn computes_balance_load_and_largest_overlap() {
        let summary = summary();
        assert_eq!(summary.fraction_on, [1.0, 0.5, 0.75]);
        assert_eq!(summary.load(), 0.75);
        assert_eq!(summary.max_abs_overlap(), Some((0.5, 0, 2)));
    }

    /// `m^{01} = 0`, `m^{02} = 0.5`, `m^{12} = -0.5`: sums `0 + 0.5`,
    /// `0 + 0.5`, `0.5 + 0.5`.
    #[test]
    fn sums_absolute_overlaps_with_the_other_patterns() {
        assert_eq!(summary().overlap_sums(), [0.5, 0.5, 1.0]);
    }

    #[test]
    fn report_lists_size_and_overlaps() {
        let text = summary().to_string();
        assert!(text.contains("bitmap       2x2 (N = 4)"), "{text}");
        assert!(text.contains("patterns     p = 3, p/N = 0.750"), "{text}");
        assert!(
            text.contains("max |m|      0.500 (patterns 0 and 2)"),
            "{text}"
        );
        assert!(text.contains("    1  0.000  1.000 -0.500"), "{text}");
    }

    #[test]
    fn small_loads_use_scientific_notation() {
        assert_eq!(format_load(0.75), "0.750");
        assert_eq!(format_load(0.001), "0.001");
        assert_eq!(format_load(12.0 / 60000.0), "2.00e-4");
    }

    #[test]
    fn single_pattern_has_no_largest_overlap() {
        let set = PatternSet::new(1, 1, array![[1]]).unwrap();
        assert_eq!(Summary::new("s", "", &set).max_abs_overlap(), None);
    }
}
