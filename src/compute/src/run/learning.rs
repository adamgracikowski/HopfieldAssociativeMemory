//! Choice of the learning rule in a run configuration.

use serde::{Deserialize, Serialize};

use crate::learning::{Hebb, LearningRule, Oja, Trained};
use crate::weights::Backend;
use crate::{PatternSet, Result};

/// Learning rule with its parameters, written as a `[learning]` table whose
/// `rule` key names the rule:
///
/// ```toml
/// [learning]
/// rule = "oja"         # or "hebb", which takes no parameters
/// variant = "subspace" # parameters of `Oja`, defaults when left out
/// eta = 0.5
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
#[serde(tag = "rule", rename_all = "kebab-case", deny_unknown_fields)]
pub enum Learning {
    /// Hebbian rule ([`Hebb`]).
    Hebb {},
    /// Oja's rule ([`Oja`]).
    Oja(Oja),
}

impl LearningRule for Learning {
    fn train(&self, set: &PatternSet, backend: Backend) -> Result<Trained> {
        match self {
            Self::Hebb {} => Hebb.train(set, backend),
            Self::Oja(oja) => oja.train(set, backend),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning::OjaVariant;
    use crate::weights::WeightMatrix;
    use ndarray::array;

    #[derive(Debug, Deserialize)]
    struct Wrapper {
        learning: Learning,
    }

    fn parse(text: &str) -> std::result::Result<Learning, toml::de::Error> {
        toml::from_str::<Wrapper>(text).map(|wrapper| wrapper.learning)
    }

    #[test]
    fn parses_both_rules() {
        assert_eq!(
            parse("[learning]\nrule = \"hebb\"").unwrap(),
            Learning::Hebb {}
        );
        let oja =
            parse("[learning]\nrule = \"oja\"\nvariant = \"single-neuron\"\neta = 0.1").unwrap();
        assert_eq!(
            oja,
            Learning::Oja(Oja {
                variant: OjaVariant::SingleNeuron,
                eta: 0.1,
                ..Oja::default()
            })
        );
        assert_eq!(
            parse("[learning]\nrule = \"oja\"").unwrap(),
            Learning::Oja(Oja::default())
        );
    }

    #[test]
    fn rejects_unknown_rules_and_parameters() {
        assert!(parse("[learning]\nrule = \"storkey\"").is_err());
        assert!(parse("[learning]\nrule = \"hebb\"\neta = 0.1").is_err());
        assert!(parse("[learning]\nrule = \"oja\"\nrate = 0.1").is_err());
        assert!(parse("[learning]\neta = 0.1").is_err());
    }

    #[test]
    fn serializes_with_the_rule_name() {
        let json = serde_json::to_value(Learning::Oja(Oja::default())).unwrap();
        assert_eq!(json["rule"], "oja");
        assert_eq!(json["variant"], "subspace");
        assert_eq!(json["eta"], 0.5);
        let hebb = serde_json::to_value(Learning::Hebb {}).unwrap();
        assert_eq!(hebb, serde_json::json!({ "rule": "hebb" }));
    }

    #[test]
    fn trains_with_the_chosen_rule() {
        let set = PatternSet::new(2, 2, array![[1, -1, 1, 1], [1, 1, -1, 1]]).unwrap();
        let hebb = Learning::Hebb {}.train(&set, Backend::Dense).unwrap();
        let direct = Hebb.train(&set, Backend::Dense).unwrap();
        assert_eq!(hebb.weights.to_dense(), direct.weights.to_dense());
        let oja = Learning::Oja(Oja::default())
            .train(&set, Backend::Factored)
            .unwrap();
        let direct = Oja::default().train(&set, Backend::Factored).unwrap();
        assert_eq!(oja.weights.to_dense(), direct.weights.to_dense());
        assert_eq!(oja.epochs, direct.epochs);
    }
}
