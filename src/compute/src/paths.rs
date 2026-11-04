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

/// Directory of the file `path`: its parent, or `.` for a bare file name
/// (whose parent is empty).
pub fn directory_of(path: &Path) -> &Path {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    }
}

/// Path of `target` relative to the directory `base`, with `/` separators,
/// e.g. `../images/cats/config.toml`. Both are made absolute against the
/// current directory and normalized lexically first. Returns `None` when
/// they have no common root (e.g. different drives).
pub fn relative_to(target: &Path, base: &Path) -> Option<String> {
    let normalize = |path: &Path| {
        let path = if path.as_os_str().is_empty() {
            Path::new(".")
        } else {
            path
        };
        std::path::absolute(path)
            .ok()
            .map(|path| resolve(Path::new(""), &path))
    };
    let (target, base) = (normalize(target)?, normalize(base)?);
    let target: Vec<Component<'_>> = target.components().collect();
    let base: Vec<Component<'_>> = base.components().collect();
    if target.first() != base.first() {
        return None;
    }
    let common = target.iter().zip(&base).take_while(|(a, b)| a == b).count();
    let parts: Vec<String> = std::iter::repeat_n("..".to_owned(), base.len() - common)
        .chain(
            target[common..]
                .iter()
                .map(|part| part.as_os_str().to_string_lossy().into_owned()),
        )
        .collect();
    Some(if parts.is_empty() {
        ".".to_owned()
    } else {
        parts.join("/")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_paths_use_parent_steps_and_slashes() {
        let relative = |target: &str, base: &str| relative_to(Path::new(target), Path::new(base));
        assert_eq!(
            relative("data/images/cats/config.toml", "data/datasets").as_deref(),
            Some("../images/cats/config.toml")
        );
        assert_eq!(
            relative("data/datasets/a.npy", "data/datasets").as_deref(),
            Some("a.npy")
        );
        assert_eq!(
            relative("data/./x/../datasets", "data/datasets").as_deref(),
            Some(".")
        );
        assert_eq!(relative("a/b.toml", "").as_deref(), Some("a/b.toml"));
    }

    #[test]
    fn directory_of_a_bare_file_name_is_the_current_one() {
        assert_eq!(directory_of(Path::new("c.toml")), Path::new("."));
        assert_eq!(directory_of(Path::new("a/c.toml")), Path::new("a"));
    }

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
