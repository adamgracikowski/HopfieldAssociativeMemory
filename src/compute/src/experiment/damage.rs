//! Damage applied to a stored pattern to make the initial state of a trial.

use crate::run::Input;

/// How a trial damages its target, fixed by the cell's damage level and the
/// bitmap size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Damage {
    /// `pixels` pixels flipped, chosen at random.
    Flip {
        /// Number of flipped pixels.
        pixels: usize,
    },
    /// A band of `rows` consecutive full-width rows at a random position set
    /// to `-1`.
    Occlusion {
        /// Number of covered rows.
        rows: usize,
        /// Number of covered pixels, `rows` times the bitmap width.
        pixels: usize,
    },
}

impl Damage {
    /// Number of damaged pixels: flipped, or covered (only those equal to
    /// `1` change).
    pub fn pixels(&self) -> usize {
        match *self {
            Self::Flip { pixels } | Self::Occlusion { pixels, .. } => pixels,
        }
    }

    /// The `recall` input that applies this damage to stored pattern
    /// `index`.
    pub fn input(&self, index: usize) -> Input {
        match *self {
            Self::Flip { pixels } => Input::Pattern {
                index,
                flips: pixels,
            },
            Self::Occlusion { rows, .. } => Input::Occlusion { index, rows },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_pixels_and_builds_the_input() {
        let flip = Damage::Flip { pixels: 5 };
        assert_eq!(flip.pixels(), 5);
        assert_eq!(flip.input(2), Input::Pattern { index: 2, flips: 5 });
        let occlusion = Damage::Occlusion {
            rows: 3,
            pixels: 21,
        };
        assert_eq!(occlusion.pixels(), 21);
        assert_eq!(occlusion.input(1), Input::Occlusion { index: 1, rows: 3 });
    }
}
