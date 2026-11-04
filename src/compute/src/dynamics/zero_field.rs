//! Activation of a neuron and the convention for a zero local field.

use serde::Deserialize;

/// Local fields with `|h| <= ZERO_FIELD_TOLERANCE` count as zero.
///
/// Fields that are exactly zero in exact arithmetic come out as tiny values
/// of either sign after rounding, differently in the dense and the factored
/// backend. The smallest non-zero Hebbian field is `1/N`, at least
/// `1.7e-5` for `N <= 60000`, far above this tolerance.
pub const ZERO_FIELD_TOLERANCE: f64 = 1e-10;

/// New state of a neuron whose local field is zero, i.e. the value of
/// `sgn(0)`.
///
/// The literature uses both conventions: `sgn(0) = +1` (Bruck 1990,
/// McEliece et al. 1987, MacKay 2003) and keeping the previous state (Rojas
/// 1996, Personnaz et al. 1985). Keeping the state acts like a positive
/// self-coupling `epsilon`, i.e. weights `W + epsilon I`, which stay
/// symmetric with a non-negative diagonal, so the convergence results for
/// both update modes still apply. [`Display`](std::fmt::Display) and
/// `&'static str` conversion give the name used in configuration files
/// (`keep`, `plus-one`).
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
pub enum ZeroField {
    /// The neuron keeps its previous state.
    #[default]
    Keep,
    /// The neuron becomes `+1`.
    PlusOne,
}

impl ZeroField {
    /// New state `sgn(h)` of a neuron with local field `field` and state
    /// `previous`: `1` above [`ZERO_FIELD_TOLERANCE`], `-1` below
    /// `-ZERO_FIELD_TOLERANCE` and this convention in between.
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
