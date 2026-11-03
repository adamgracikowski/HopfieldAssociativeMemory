//! Version and source revision of the build, recorded with every result so
//! that a run can be traced back to its code.

/// Version of the crate from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Git commit of the sources the binary was built from (`unknown` without git).
pub const GIT_COMMIT: &str = env!("COMPUTE_GIT_COMMIT");

/// Whether files of this crate differed from [`GIT_COMMIT`] at build time:
/// `true`, `false` or `unknown` (without git).
pub const GIT_DIRTY: &str = env!("COMPUTE_GIT_DIRTY");

/// Version line printed by `compute --version`: crate version, commit and
/// dirty flag.
pub const LONG_VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    " (commit ",
    env!("COMPUTE_GIT_COMMIT"),
    ", dirty: ",
    env!("COMPUTE_GIT_DIRTY"),
    ")"
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn git_commit_is_a_full_hash_or_unknown() {
        let is_hash = GIT_COMMIT.len() == 40 && GIT_COMMIT.chars().all(|c| c.is_ascii_hexdigit());
        assert!(is_hash || GIT_COMMIT == "unknown", "{GIT_COMMIT}");
    }

    #[test]
    fn git_dirty_has_a_known_value() {
        assert!(["true", "false", "unknown"].contains(&GIT_DIRTY));
    }

    #[test]
    fn long_version_contains_all_parts() {
        assert!(LONG_VERSION.starts_with(VERSION));
        assert!(LONG_VERSION.contains(GIT_COMMIT));
        assert!(LONG_VERSION.ends_with(&format!("dirty: {GIT_DIRTY})")));
    }
}
