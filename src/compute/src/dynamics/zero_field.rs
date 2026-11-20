//! Activation of a neuron and the convention for a zero local field.

use serde::{Deserialize, Serialize};

/// Local fields with `|h| <= ZERO_FIELD_TOLERANCE` count as zero (rounding
/// noise); the smallest real Hebbian field, `1/N`, is far above it.
pub const ZERO_FIELD_TOLERANCE: f64 = 1e-10;

/// Value of `sgn(0)`: keep the state (Rojas 1996, Personnaz et al. 1985) or
/// `+1` (Bruck 1990, McEliece et al. 1987, MacKay 2003); both keep the
/// convergence guarantees.
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
pub enum ZeroField {
    /// The neuron keeps its previous state.
    #[default]
    Keep,
    /// The neuron becomes `+1`.
    PlusOne,
}

impl ZeroField {
    /// New state `sgn(field)` of a neuron in state `previous`, with this
    /// convention within [`ZERO_FIELD_TOLERANCE`] of zero.
    pub fn activate(self, field: f64, previous: i8) -> i8 {
        if field > ZERO_FIELD_TOLERANCE {
            1
        } else if field < -ZERO_FIELD_TOLERANCE {
            -1
        } else {
            match self {
                Self::Keep => previous,
                Self::PlusOne => 1,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use strum::IntoEnumIterator;

    #[test]
    fn non_zero_fields_give_their_sign() {
        for rule in ZeroField::iter() {
            for previous in [1, -1] {
                assert_eq!(rule.activate(0.3, previous), 1);
                assert_eq!(rule.activate(-1e-5, previous), -1);
            }
        }
    }

    #[test]
    fn zero_fields_follow_the_convention() {
        for field in [0.0, 1e-12, -1e-12, ZERO_FIELD_TOLERANCE] {
            assert_eq!(ZeroField::Keep.activate(field, -1), -1);
            assert_eq!(ZeroField::Keep.activate(field, 1), 1);
            assert_eq!(ZeroField::PlusOne.activate(field, -1), 1);
        }
    }

    #[test]
    fn names_round_trip_through_configuration() {
        #[derive(Deserialize)]
        struct Wrapper {
            zero_field: ZeroField,
        }
        for rule in ZeroField::iter() {
            let text = format!("zero_field = \"{rule}\"");
            assert_eq!(toml::from_str::<Wrapper>(&text).unwrap().zero_field, rule);
        }
        assert_eq!(ZeroField::PlusOne.to_string(), "plus-one");
        assert_eq!(ZeroField::default(), ZeroField::Keep);
    }
}
