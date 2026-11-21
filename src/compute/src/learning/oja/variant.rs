//! Forms of Oja's rule for the whole weight matrix.

use serde::{Deserialize, Serialize};

/// Form of Oja's rule, with outputs `y = W x` for `x = xi / sqrt(N)`.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    Deserialize,
    Serialize,
    strum::Display,
    strum::IntoStaticStr,
    strum::EnumIter,
)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
pub enum OjaVariant {
    /// Subspace rule `dW = eta (y x^T - y y^T W)` (Oja 1989); from the
    /// scaled Hebbian start it converges to the projection (pseudo-inverse)
    /// rule (Personnaz, Guyon and Dreyfus 1985).
    #[default]
    Subspace,
    /// Single-neuron rule (Oja 1982) on every row,
    /// `dW = eta (y x^T - diag(y * y) W)`; rows converge to the principal
    /// eigenvector, so it stores about one pattern direction (Sanger 1989,
    /// p. 465).
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
