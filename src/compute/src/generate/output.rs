//! Output datasets of a generation.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// One generated dataset: the fraction of flipped pixels of its variants
/// and the `.npy` file it is written to.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GroupsOutput {
    /// Fraction of the pixels of a prototype flipped in each variant, from 0
    /// to 1.
    pub flip: f64,
    /// `.npy` file to write.
    pub file: PathBuf,
}

impl GroupsOutput {
    /// Number of flipped pixels of a variant of `neurons` pixels: `flip *
    /// N` rounded to the nearest integer, halves away from zero.
    pub fn flips(&self, neurons: usize) -> usize {
        (self.flip * neurons as f64).round() as usize
    }

    /// File name of the output without its extension, e.g.
    /// `groups-25x25-d05`.
    pub fn stem(&self) -> String {
        self.file
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `0.05 * 625 = 31.25` rounds to 31 flipped pixels, `0.02 * 625 =
    /// 12.5` to 13.
    #[test]
    fn parses_and_rounds_the_flips() {
        let output: GroupsOutput =
            toml::from_str("flip = 0.05\nfile = \"../d/groups-d05.npy\"").unwrap();
        assert_eq!(output.flips(625), 31);
        assert_eq!(output.stem(), "groups-d05");
        let output = GroupsOutput {
            flip: 0.02,
            ..output
        };
        assert_eq!(output.flips(625), 13);
        assert!(toml::from_str::<GroupsOutput>("flip = 0.1").is_err());
    }
}
