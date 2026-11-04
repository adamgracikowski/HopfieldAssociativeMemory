//! Version and source revision of the build, recorded with every result so
//! that a run can be traced back to its code.

use serde::Serialize;

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

/// Version, commit and dirty flag of this build, as recorded in result
/// metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuildInfo {
    /// Crate version ([`VERSION`]).
    pub version: &'static str,
    /// Git commit ([`GIT_COMMIT`]).
    pub commit: &'static str,
    /// Whether the crate files differed from the commit ([`GIT_DIRTY`]);
    /// `None` (`null` in JSON) when it is unknown.
    pub dirty: Option<bool>,
}

impl BuildInfo {
    /// The build of this binary.
    pub const CURRENT: Self = Self {
        version: VERSION,
        commit: GIT_COMMIT,
        dirty: parse_dirty(GIT_DIRTY),
    };
}

/// `Some(true)` for `true`, `Some(false)` for `false`, `None` otherwise.
const fn parse_dirty(flag: &str) -> Option<bool> {
    match flag.as_bytes() {
        b"true" => Some(true),
        b"false" => Some(false),
        _ => None,
    }
}

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

    #[test]
    fn current_build_serializes_all_parts() {
        let json = serde_json::to_value(BuildInfo::CURRENT).unwrap();
        assert_eq!(json["version"], VERSION);
        assert_eq!(json["commit"], GIT_COMMIT);
        assert_eq!(
            json["dirty"].to_string(),
            GIT_DIRTY.replace("unknown", "null")
        );
    }

    #[test]
    fn dirty_flag_is_parsed_to_a_boolean() {
        assert_eq!(parse_dirty("true"), Some(true));
        assert_eq!(parse_dirty("false"), Some(false));
        assert_eq!(parse_dirty("unknown"), None);
    }
}
