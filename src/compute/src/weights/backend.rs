//! Choice of the weight storage backend.

use serde::{Deserialize, Serialize};

/// Largest dense matrix the `auto` backend allows: 256 MiB of `f64`, that is
/// `N <= 5792` neurons.
pub const DENSE_LIMIT_BYTES: usize = 256 * 1024 * 1024;

/// Weight storage backend.
///
/// [`Display`](std::fmt::Display) and `&'static str` conversion give the name
/// used in configuration files (`auto`, `dense`, `factored`).
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
pub enum Backend {
    /// Dense if the `N x N` matrix of `f64` fits in [`DENSE_LIMIT_BYTES`],
    /// factored otherwise.
    #[default]
    Auto,
    /// Full `N x N` matrix ([`DenseWeights`](super::DenseWeights)).
    Dense,
    /// Factored matrix built from the patterns
    /// ([`FactoredWeights`](super::FactoredWeights)).
    Factored,
}

impl Backend {
    /// The concrete backend for a network of `neurons` neurons: `Auto`
    /// becomes `Dense` or `Factored` by the memory limit, the others stay
    /// unchanged.
    pub fn resolve(self, neurons: usize) -> Self {
        match self {
            Self::Auto => {
                let bytes = neurons
                    .checked_mul(neurons)
                    .and_then(|n| n.checked_mul(size_of::<f64>()));
                match bytes {
                    Some(bytes) if bytes <= DENSE_LIMIT_BYTES => Self::Dense,
                    _ => Self::Factored,
                }
            }
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use strum::IntoEnumIterator;

    #[test]
    fn auto_switches_to_factored_above_the_memory_limit() {
        assert_eq!(Backend::Auto.resolve(25), Backend::Dense);
        assert_eq!(Backend::Auto.resolve(5792), Backend::Dense);
        assert_eq!(Backend::Auto.resolve(5793), Backend::Factored);
        assert_eq!(Backend::Auto.resolve(60_000), Backend::Factored);
        assert_eq!(Backend::Auto.resolve(usize::MAX), Backend::Factored);
    }

    #[test]
    fn explicit_backends_stay_unchanged() {
        assert_eq!(Backend::Dense.resolve(60_000), Backend::Dense);
        assert_eq!(Backend::Factored.resolve(25), Backend::Factored);
    }

    #[test]
    fn parses_from_kebab_case() {
        #[derive(Deserialize)]
        struct Wrapper {
            backend: Backend,
        }
        let parse = |s: &str| toml::from_str::<Wrapper>(&format!("backend = \"{s}\""));
        for backend in Backend::iter() {
            assert_eq!(parse(&backend.to_string()).unwrap().backend, backend);
        }
        let name: &str = Backend::Factored.into();
        assert_eq!(name, "factored");
        assert!(parse("sparse").is_err());
        assert_eq!(Backend::default(), Backend::Auto);
    }
}
