//! Seeds of the random generators of a run.

use std::fmt;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// Seed of one run: it starts two independent streams of `ChaCha8Rng`, one
/// for the initial state and one for the dynamics, so changing the update
/// order does not change a damaged input.
///
/// A `recall` run uses the seed of its configuration; every trial of an
/// experiment uses a seed derived from the experiment seed with
/// [`Seed::derive`], so `recall` with that seed repeats the trial.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(transparent)]
pub struct Seed(pub u64);

impl Seed {
    /// Stream of the initial state.
    const INPUT_STREAM: u64 = 0;
    /// Stream of the dynamics (update order).
    const DYNAMICS_STREAM: u64 = 1;

    /// First 8 bytes (big-endian) of the SHA-1 of `master/part/part/...`,
    /// independent of the platform. Parts must not contain `/`.
    pub fn derive(master: u64, parts: &[&dyn fmt::Display]) -> Self {
        let mut text = master.to_string();
        for part in parts {
            text.push('/');
            text.push_str(&part.to_string());
        }
        let digest = sha1_smol::Sha1::from(text).digest().bytes();
        let mut first = [0; 8];
        first.copy_from_slice(&digest[..8]);
        Self(u64::from_be_bytes(first))
    }

    /// Generator of the initial state.
    pub fn input_rng(self) -> ChaCha8Rng {
        self.stream(Self::INPUT_STREAM)
    }

    /// Generator of the dynamics.
    pub fn dynamics_rng(self) -> ChaCha8Rng {
        self.stream(Self::DYNAMICS_STREAM)
    }

    fn stream(self, stream: u64) -> ChaCha8Rng {
        let mut rng = ChaCha8Rng::seed_from_u64(self.0);
        rng.set_stream(stream);
        rng
    }
}

impl fmt::Display for Seed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt;

    /// SHA-1 of `7/small-7x7/3/0/12` is `20d0d6d7e169fcc1...`, computed with
    /// `sha1sum`; its first 8 bytes give the seed.
    #[test]
    fn derives_a_fixed_seed_from_the_parts() {
        let seed = Seed::derive(7, &[&"small-7x7", &3, &0, &12]);
        assert_eq!(seed, Seed(0x20d0_d6d7_e169_fcc1));
    }

    #[test]
    fn every_part_changes_the_seed() {
        let base = Seed::derive(7, &[&"a", &1, &2]);
        for other in [
            Seed::derive(8, &[&"a", &1, &2]),
            Seed::derive(7, &[&"b", &1, &2]),
            Seed::derive(7, &[&"a", &2, &2]),
            Seed::derive(7, &[&"a", &1, &3]),
            Seed::derive(7, &[&"a", &1]),
        ] {
            assert_ne!(other, base);
        }
    }

    /// The two streams of a seed differ, and each repeats for the same seed.
    #[test]
    fn streams_are_independent_and_repeatable() {
        let seed = Seed(42);
        let draw = |mut rng: ChaCha8Rng| -> Vec<u64> { (0..4).map(|_| rng.random()).collect() };
        assert_eq!(draw(seed.input_rng()), draw(seed.input_rng()));
        assert_ne!(draw(seed.input_rng()), draw(seed.dynamics_rng()));
        assert_ne!(draw(seed.input_rng()), draw(Seed(43).input_rng()));
    }
}
