//! Paths written in configuration files.

use std::path::{Component, Path, PathBuf};

/// Joins `relative` to `base` and removes `.` components and `..` components
/// that follow a directory name, without touching the file system (so a `..`
/// after a symbolic link is resolved lexically).
pub fn resolve(base: &Path, relative: &Path) -> PathBuf {
    let mut path = PathBuf::new();
    for component in base.join(relative).components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir
                if matches!(path.components().next_back(), Some(Component::Normal(_))) =>
            {
                path.pop();
            }
            other => path.push(other),
        }
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_parent_directories_lexically() {
        let base = Path::new("data/images/cats");
        assert_eq!(
            resolve(base, Path::new("../../datasets/./a.npy")),
            Path::new("data/datasets/a.npy")
        );
        assert_eq!(
            resolve(Path::new("a"), Path::new("../../b.npy")),
            Path::new("../b.npy")
        );
        assert_eq!(resolve(Path::new(""), Path::new("raw")), Path::new("raw"));
    }
}
