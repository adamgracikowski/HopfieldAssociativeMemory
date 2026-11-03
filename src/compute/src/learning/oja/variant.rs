//! Forms of Oja's rule for the whole weight matrix.

use serde::Deserialize;

/// Form of Oja's rule used to update the whole weight matrix.
///
/// Both forms use the outputs `y = W x` of the network for the normalized
/// patterns `x = xi / sqrt(N)`. [`Display`](std::fmt::Display) and
/// `&'static str` conversion give the name used in configuration files
/// (`subspace`, `single-neuron`).
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    Deserialize,
    strum::Display,
    strum::IntoStaticStr,
    strum::EnumIter,
)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
pub enum OjaVariant {
    /// Oja's subspace rule `dW = eta (y x^T - y y^T W)` (Oja 1989).
    ///
    /// Started from the (scaled) Hebbian weights it converges to the
    /// projection onto the span of the patterns, `P = Xi^T (Xi Xi^T)^-1 Xi`,
    /// the projection (pseudo-inverse) rule of Personnaz, Guyon and Dreyfus
    /// (1985), for which every linearly independent pattern is mapped to
    /// itself.
    #[default]
    Subspace,
    /// Oja's single-neuron rule (Oja 1982) applied to every row:
    /// `dW = eta (y x^T - diag(y * y) W)`.
    ///
    /// Every row converges to the principal eigenvector of the pattern
    /// correlation matrix (Sanger 1989, p. 465), so the weights tend to a
    /// rank-one matrix that stores at most one pattern direction.
    SingleNeuron,
}

#[cfg(test)]
mod tests {
    use super::*;
    use strum::IntoEnumIterator;

    #[test]
    fn names_round_trip_through_configuration() {
        #[derive(Deserialize)]
        struct Wrapper {
            variant: OjaVariant,
        }
        for variant in OjaVariant::iter() {
            let text = format!("variant = \"{variant}\"");
            assert_eq!(toml::from_str::<Wrapper>(&text).unwrap().variant, variant);
        }
        assert_eq!(OjaVariant::SingleNeuron.to_string(), "single-neuron");
        assert_eq!(OjaVariant::default(), OjaVariant::Subspace);
    }
}
