//! Integration tests of the `compute` command-line interface.

use std::path::{Path, PathBuf};

use assert_cmd::Command;
use compute::build_info;
use compute::images::config::Source;
use compute::patterns::npy::read_npy;
use image::{GrayImage, Luma};
use npyz::NpyFile;
use predicates::prelude::*;
use predicates::str::contains;

fn compute() -> Command {
    Command::cargo_bin("compute").expect("binary `compute` is built")
}

#[test]
fn version_shows_crate_version_and_commit() {
    compute()
        .arg("--version")
        .assert()
        .success()
        .stdout(contains(build_info::VERSION))
        .stdout(contains(build_info::GIT_COMMIT));
}

#[test]
fn no_arguments_prints_usage_and_fails() {
    compute()
        .assert()
        .failure()
        .stderr(contains("Usage: compute"));
}

#[test]
fn inspect_prints_statistics_and_bitmaps() {
    compute()
        .args(["inspect", "small-7x7"])
        .assert()
        .success()
        .stdout(contains("bitmap       7x7 (N = 49)"))
        .stdout(contains("patterns     p = 5, p/N = 0.102"))
        .stdout(contains("pattern 4\n###.###\n"))
        .stdout(contains("pattern 0\n#.....#\n.#...#.\n"));
}

#[test]
fn inspect_shows_only_selected_patterns() {
    compute()
        .args(["inspect", "small-7x7", "--pattern", "1"])
        .assert()
        .success()
        .stdout(contains("pattern 1\n#######\n"))
        .stdout(contains("pattern 0").not());
}

#[test]
fn inspect_without_render_prints_no_bitmaps() {
    compute()
        .args(["inspect", "letters-abc-8x12", "--no-render"])
        .assert()
        .success()
        .stdout(contains("p = 3"))
        .stdout(contains("\npattern ").not());
}

#[test]
fn inspect_accepts_an_explicit_registry() {
    let registry = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../data/datasets/datasets.toml"
    );
    compute()
        .current_dir(std::env::temp_dir())
        .args([
            "inspect",
            "animals-14x9",
            "--no-render",
            "--registry",
            registry,
        ])
        .assert()
        .success()
        .stdout(contains("bitmap       14x9 (N = 126)"));
}

#[test]
fn inspect_rejects_an_unknown_dataset() {
    compute()
        .args(["inspect", "nope"])
        .assert()
        .failure()
        .stderr(contains("unknown dataset `nope`"));
}

#[test]
fn inspect_rejects_a_missing_pattern() {
    compute()
        .args(["inspect", "small-7x7", "--pattern", "5"])
        .assert()
        .failure()
        .stderr(contains("pattern 5 does not exist"));
}

/// Conversion configuration used by the `convert-images` tests.
const CONVERSION_CONFIG: &str = r#"
width = 4
height = 2
sources = "sources.tsv"
images = "raw"

[[outputs]]
method = "median"
file = "out/median.npy"

[[outputs]]
method = "floyd-steinberg"
file = "out/dither.npy"
"#;

/// Writes the conversion configuration, two 8x4 gradient images (the second
/// mirrored) and their source list, and returns the configuration path.
fn conversion_setup(dir: &Path) -> PathBuf {
    let config = dir.join("config.toml");
    std::fs::write(&config, CONVERSION_CONFIG).unwrap();
    std::fs::create_dir(dir.join("raw")).unwrap();
    let mut sources = String::from("file\tsha1\turl\tpage\tauthor\tlicense\tlicense_url\n");
    for (name, mirror) in [("a.png", false), ("b.png", true)] {
        let path = dir.join("raw").join(name);
        GrayImage::from_fn(8, 4, |x, _| {
            let x = if mirror { 7 - x } else { x };
            Luma([(x * 32) as u8])
        })
        .save(&path)
        .unwrap();
        let sha1 = Source::checksum(&path).unwrap();
        sources.push_str(&format!("{name}\t{sha1}\turl\tpage\tA\tCC0\tlicense\n"));
    }
    std::fs::write(dir.join("sources.tsv"), sources).unwrap();
    config
}

#[test]
fn convert_images_writes_datasets_and_previews() {
    let dir = tempfile::tempdir().unwrap();
    let config = conversion_setup(dir.path());
    let preview = dir.path().join("preview");
    compute()
        .arg("convert-images")
        .arg("--config")
        .arg(&config)
        .arg("--preview")
        .arg(&preview)
        .assert()
        .success()
        .stdout(contains("dataset      median"))
        .stdout(contains(
            "description  2 images, floyd-steinberg binarization",
        ))
        .stdout(contains("bitmap       4x2 (N = 8)"));

    let median = read_npy(&dir.path().join("out").join("median.npy")).unwrap();
    assert_eq!((median.len(), median.width(), median.height()), (2, 4, 2));
    assert_eq!(
        median.pattern(0).values().to_vec(),
        [1, 1, -1, -1, 1, 1, -1, -1]
    );
    assert!(dir.path().join("out").join("dither.npy").is_file());
    assert!(preview.join("dither").join("01-b.png").is_file());

    let provenance = read_json(&dir.path().join("out").join("dither.json"));
    assert_eq!(provenance["command"], "convert-images");
    assert_eq!(provenance["build"]["commit"], build_info::GIT_COMMIT);
    assert_eq!(provenance["config"], "../config.toml");
    assert_eq!(provenance["method"], "floyd-steinberg");
    assert_eq!(provenance["sources"][0]["file"], "a.png");
}

#[test]
fn convert_images_reports_a_missing_image() {
    let dir = tempfile::tempdir().unwrap();
    let config = conversion_setup(dir.path());
    std::fs::remove_file(dir.path().join("raw").join("a.png")).unwrap();
    compute()
        .arg("convert-images")
        .arg("--config")
        .arg(&config)
        .assert()
        .failure()
        .stderr(contains("a.png").and(contains("not found")))
        .stderr(contains("download.ps1"));
}

#[test]
fn convert_images_rejects_an_unknown_method() {
    let dir = tempfile::tempdir().unwrap();
    let config = conversion_setup(dir.path());
    let text = std::fs::read_to_string(&config).unwrap();
    std::fs::write(&config, text.replace("\"median\"", "\"otsu\"")).unwrap();
    compute()
        .arg("convert-images")
        .arg("--config")
        .arg(&config)
        .assert()
        .failure()
        .stderr(contains("invalid configuration"));
}

/// Writes a run configuration for `small-7x7` into `dir`, with the
/// repository registry and `extra` lines appended.
fn run_config(dir: &Path, extra: &str) -> PathBuf {
    let registry = std::path::absolute("../../data/datasets/datasets.toml").unwrap();
    let text = format!(
        "dataset = \"small-7x7\"\nregistry = '{}'\n{extra}\n",
        registry.display()
    );
    let path = dir.join("config.toml");
    std::fs::write(&path, text).unwrap();
    path
}

fn read_json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn npy_shape(path: &Path) -> Vec<u64> {
    let bytes = std::fs::read(path).unwrap();
    NpyFile::new(&bytes[..]).unwrap().shape().to_vec()
}

#[test]
fn train_writes_weights_and_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let config = run_config(dir.path(), "[learning]\nrule = \"hebb\"");
    let out = dir.path().join("run");
    compute()
        .arg("train")
        .arg("--config")
        .arg(&config)
        .arg("--out")
        .arg(&out)
        .assert()
        .success()
        .stdout(contains("learning     hebb, dense backend"))
        .stdout(contains("of 5 patterns"));

    assert_eq!(npy_shape(&out.join("weights.npy")), [49, 49]);
    let metadata = read_json(&out.join("metadata.json"));
    assert_eq!(metadata["command"], "train");
    assert_eq!(metadata["build"]["commit"], build_info::GIT_COMMIT);
    assert_eq!(metadata["config"]["learning"]["rule"], "hebb");
    assert_eq!(metadata["config"]["dynamics"]["mode"], "asynchronous");
    assert_eq!(metadata["dataset"]["neurons"], 49);
    assert_eq!(metadata["training"]["backend"], "dense");
    assert_eq!(
        metadata["stability"]["unstable_neurons"]
            .as_array()
            .unwrap()
            .len(),
        5
    );
}

#[test]
fn train_saves_coefficients_of_factored_weights() {
    let dir = tempfile::tempdir().unwrap();
    let config = run_config(
        dir.path(),
        "backend = \"factored\"\n[learning]\nrule = \"oja\"",
    );
    let out = dir.path().join("run");
    compute()
        .arg("train")
        .arg("--config")
        .arg(&config)
        .arg("--out")
        .arg(&out)
        .assert()
        .success()
        .stdout(contains("learning     oja (subspace), factored backend"))
        .stdout(contains("stable       5 of 5 patterns"));
    assert_eq!(npy_shape(&out.join("coefficients.npy")), [49, 5]);
    assert!(!out.join("weights.npy").exists());
    let metadata = read_json(&out.join("metadata.json"));
    assert_eq!(metadata["training"]["converged"], true);
    assert_eq!(metadata["config"]["learning"]["eta"], 0.5);
}

#[test]
fn train_refuses_a_non_empty_output_without_force() {
    let dir = tempfile::tempdir().unwrap();
    let config = run_config(dir.path(), "[learning]\nrule = \"hebb\"");
    let run = |force: bool| {
        let mut command = compute();
        command
            .arg("train")
            .arg("--config")
            .arg(&config)
            .arg("--out")
            .arg(dir.path());
        if force {
            command.arg("--force");
        }
        command.assert()
    };
    run(false)
        .failure()
        .stderr(contains("is not empty"))
        .stderr(contains("--force"));
    run(true).success();
    assert!(dir.path().join("weights.npy").is_file());
    assert!(dir.path().join("config.toml").is_file());
}

const RECALL: &str = "seed = 3\n[learning]\nrule = \"hebb\"\n[dynamics]\nmode = \"synchronous\"\n\
                      [input]\nsource = \"pattern\"\nindex = 1\nflips = 6";

fn recall(config: &Path, out: &Path) -> assert_cmd::assert::Assert {
    compute()
        .arg("recall")
        .arg("--config")
        .arg(config)
        .arg("--out")
        .arg(out)
        .assert()
}

#[test]
fn recall_writes_history_frames_and_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let config = run_config(dir.path(), RECALL);
    let out = dir.path().join("run");
    recall(&config, &out)
        .success()
        .stdout(contains("input        pattern 1 with 6 flipped pixels"))
        .stdout(contains("target       pattern 1, distance 6 -> "));

    let metadata = read_json(&out.join("metadata.json"));
    assert_eq!(metadata["command"], "recall");
    assert_eq!(metadata["config"]["seed"], 3);
    assert_eq!(metadata["outcome"]["target"], 1);
    assert_eq!(metadata["outcome"]["initial_distance"], 6);
    let frames = metadata["outcome"]["frames"].as_u64().unwrap();
    assert!(frames >= 1);
    assert_eq!(npy_shape(&out.join("history.npy")), [frames, 7, 7]);
    let stats = read_json(&out.join("frames.json"));
    assert_eq!(stats["energy"].as_array().unwrap().len() as u64, frames);
    assert_eq!(stats["overlaps"][0].as_array().unwrap().len(), 5);
    assert_eq!(stats["changes"][0], 0);
}

/// Running again with the same seed reproduces every file; the metadata
/// differs only in `created`.
#[test]
fn recall_is_reproducible() {
    let dir = tempfile::tempdir().unwrap();
    let config = run_config(
        dir.path(),
        &RECALL.replace(
            "\"synchronous\"",
            "\"asynchronous\"\norder = \"random\"\nrecord_every = 7",
        ),
    );
    let (first, second) = (dir.path().join("a"), dir.path().join("b"));
    recall(&config, &first).success();
    recall(&config, &second).success();
    for name in ["history.npy", "frames.json"] {
        assert_eq!(
            std::fs::read(first.join(name)).unwrap(),
            std::fs::read(second.join(name)).unwrap(),
            "{name}"
        );
    }
    let without_time = |dir: &Path| {
        let mut metadata = read_json(&dir.join("metadata.json"));
        metadata.as_object_mut().unwrap().remove("created");
        metadata
    };
    assert_eq!(without_time(&first), without_time(&second));
}

#[test]
fn recall_renders_the_initial_and_final_state() {
    let dir = tempfile::tempdir().unwrap();
    let config = run_config(dir.path(), RECALL);
    compute()
        .arg("recall")
        .arg("--config")
        .arg(&config)
        .arg("--out")
        .arg(dir.path().join("run"))
        .arg("--render")
        .assert()
        .success()
        .stdout(contains("\ninitial state\n"))
        .stdout(contains("\nfinal state\n"));
}

#[test]
fn recall_requires_a_seed() {
    let dir = tempfile::tempdir().unwrap();
    let config = run_config(dir.path(), &RECALL.replace("seed = 3", ""));
    recall(&config, &dir.path().join("run"))
        .failure()
        .stderr(contains("compute::invalid_config"))
        .stderr(contains("`seed`"));
}

/// A repository-like tree in `root`: `data/datasets/` with `small-7x7` and
/// its registry, and `exp/config.toml` without a `registry` key, so the
/// registry is found from the configuration directory upwards.
fn discovery_setup(root: &Path) -> PathBuf {
    let datasets = root.join("data").join("datasets");
    std::fs::create_dir_all(&datasets).unwrap();
    std::fs::copy(
        "../../data/datasets/small-7x7.csv",
        datasets.join("small-7x7.csv"),
    )
    .unwrap();
    std::fs::write(
        datasets.join("datasets.toml"),
        "[datasets.small-7x7]\nfile = \"small-7x7.csv\"\nwidth = 7\nheight = 7\n\
         description = \"Shapes.\"\n",
    )
    .unwrap();
    let exp = root.join("exp");
    std::fs::create_dir(&exp).unwrap();
    std::fs::write(
        exp.join("config.toml"),
        format!("dataset = \"small-7x7\"\n{RECALL}\n"),
    )
    .unwrap();
    exp
}

/// The configuration given by its bare file name finds the registry, and
/// runs from two working directories give the same metadata, whose paths
/// are relative to the configuration.
#[test]
fn recall_does_not_depend_on_the_working_directory() {
    let dir = tempfile::tempdir().unwrap();
    let exp = discovery_setup(dir.path());
    compute()
        .current_dir(&exp)
        .args(["recall", "--config", "config.toml", "--out", "a"])
        .assert()
        .success();
    compute()
        .current_dir(dir.path())
        .args(["recall", "--config", "exp/config.toml", "--out", "exp/b"])
        .assert()
        .success();

    let without_time = |run: &str| {
        let mut metadata = read_json(&exp.join(run).join("metadata.json"));
        metadata.as_object_mut().unwrap().remove("created");
        metadata
    };
    let metadata = without_time("a");
    assert_eq!(metadata, without_time("b"));
    assert_eq!(
        metadata["dataset"]["registry"],
        "../data/datasets/datasets.toml"
    );
    assert_eq!(metadata["dataset"]["file"], "small-7x7.csv");
    assert!(metadata["config"].get("registry").is_none());
    assert_eq!(
        std::fs::read(exp.join("a").join("history.npy")).unwrap(),
        std::fs::read(exp.join("b").join("history.npy")).unwrap()
    );
}

#[test]
fn failed_recall_leaves_no_output_directory() {
    let dir = tempfile::tempdir().unwrap();
    let config = run_config(dir.path(), &RECALL.replace("flips = 6", "flips = 50"));
    let out = dir.path().join("run");
    recall(&config, &out)
        .failure()
        .stderr(contains("cannot flip 50 pixels"));
    assert!(!out.exists());
}
