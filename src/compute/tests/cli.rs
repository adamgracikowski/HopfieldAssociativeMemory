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

/// Two prototypes of `small-7x7` with three variants each,
/// `round(0.1 * 49) = 5` flipped pixels; generating again writes identical
/// files.
#[test]
fn generate_writes_datasets_and_provenance() {
    let dir = tempfile::tempdir().unwrap();
    let registry = std::path::absolute("../../data/datasets/datasets.toml").unwrap();
    let config = dir.path().join("generated").join("config.toml");
    std::fs::create_dir_all(config.parent().unwrap()).unwrap();
    std::fs::write(
        &config,
        format!(
            "seed = 3\ndataset = \"small-7x7\"\nregistry = '{}'\nprototypes = [0, 2]\n\
             variants = 3\n[[outputs]]\nflip = 0.1\nfile = \"../out/groups-d10.npy\"\n",
            registry.display()
        ),
    )
    .unwrap();
    let generate = || {
        compute()
            .arg("generate")
            .arg("--config")
            .arg(&config)
            .assert()
            .success()
            .stdout(contains("dataset      groups-d10"))
            .stdout(contains(
                "description  2 prototypes of small-7x7 x 3 variants, 5 flipped pixels each",
            ))
    };
    generate();
    let out = dir.path().join("out");
    let set = read_npy(&out.join("groups-d10.npy")).unwrap();
    assert_eq!((set.len(), set.width(), set.height()), (6, 7, 7));
    let provenance = read_json(&out.join("groups-d10.json"));
    assert_eq!(provenance["command"], "generate");
    assert_eq!(provenance["build"]["commit"], build_info::GIT_COMMIT);
    assert_eq!(provenance["config"], "../generated/config.toml");
    assert_eq!(provenance["dataset"], "small-7x7");
    assert_eq!(provenance["prototypes"][1], 2);
    assert_eq!(provenance["flips"], 5);

    let files = |name: &str| std::fs::read(out.join(name)).unwrap();
    let (npy, json) = (files("groups-d10.npy"), files("groups-d10.json"));
    generate();
    assert_eq!(files("groups-d10.npy"), npy);
    assert_eq!(files("groups-d10.json"), json);
}

/// Only the `.npy` files are compared: the provenance records the build.
#[test]
fn generated_datasets_match_the_committed_files() {
    let source = Path::new("../../data/generated/groups-25x25/config.toml");
    let datasets = Path::new("../../data/datasets");
    let registry = std::path::absolute(datasets.join("datasets.toml")).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let text = std::fs::read_to_string(source)
        .unwrap()
        .replace("file = \"../../datasets/", "file = \"");
    let config = dir.path().join("config.toml");
    std::fs::write(
        &config,
        format!("registry = '{}'\n{text}", registry.display()),
    )
    .unwrap();
    compute()
        .arg("generate")
        .arg("--config")
        .arg(&config)
        .assert()
        .success();
    for d in ["02", "05", "10", "20"] {
        let name = format!("groups-25x25-d{d}.npy");
        assert_eq!(
            std::fs::read(dir.path().join(&name)).unwrap(),
            std::fs::read(datasets.join(&name)).unwrap(),
            "{name} differs from the committed file; regenerate it"
        );
    }
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

/// Metadata differs only in `created`.
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

/// A bare configuration file name finds the registry; metadata paths are
/// relative to the configuration.
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

#[test]
fn train_with_another_backend_removes_the_old_weights() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("run");
    let train = |backend: &str| {
        let config = run_config(
            dir.path(),
            &format!("backend = \"{backend}\"\n[learning]\nrule = \"hebb\""),
        );
        compute()
            .arg("train")
            .arg("--config")
            .arg(&config)
            .arg("--out")
            .arg(&out)
            .arg("--force")
            .assert()
            .success();
    };
    train("dense");
    assert!(out.join("weights.npy").is_file());
    train("factored");
    assert!(out.join("coefficients.npy").is_file());
    assert!(!out.join("weights.npy").exists());
    train("dense");
    assert!(!out.join("coefficients.npy").exists());
}

/// Hebbian learning on `small-7x7` (`p = 5`), both modes, two noise levels
/// and 4 trials: 3 cells of 4 x 5 = 20 runs and one deterministic cell
/// (synchronous, no noise) of 5 runs.
const EXPERIMENT: &str = "seed = 5\ndatasets = [\"small-7x7\"]\ntrials = 4\n\
    [[learning]]\nrule = \"hebb\"\n\
    [dynamics]\nmodes = [\"synchronous\", \"asynchronous\"]\n\
    [input]\nsource = \"pattern\"\nnoise = [0.0, 0.2]\n";

/// Writes the experiment configuration `text` with an absolute registry into
/// `dir` and returns its path.
fn experiment_config(dir: &Path, text: &str) -> PathBuf {
    let registry = std::path::absolute("../../data/datasets/datasets.toml").unwrap();
    let path = dir.join("config.toml");
    std::fs::write(
        &path,
        format!("registry = '{}'\n{text}", registry.display()),
    )
    .unwrap();
    path
}

fn experiment(config: &Path) -> Command {
    let mut command = compute();
    command.arg("experiment").arg("--config").arg(config);
    command
}

/// The whole table of a Parquet file as one record batch.
fn read_parquet(path: &Path) -> arrow_array::RecordBatch {
    let file = std::fs::File::open(path).unwrap();
    let builder =
        parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let rows = builder.metadata().file_metadata().num_rows() as usize;
    let mut batches: Vec<_> = builder
        .with_batch_size(rows)
        .build()
        .unwrap()
        .map(Result::unwrap)
        .collect();
    assert_eq!(batches.len(), 1, "the table is read in one batch");
    batches.remove(0)
}

#[test]
fn experiment_writes_tables_and_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let config = experiment_config(dir.path(), EXPERIMENT);
    let out = dir.path().join("run");
    experiment(&config)
        .arg("--out")
        .arg(&out)
        .assert()
        .success()
        .stderr(contains("trained      small-7x7, hebb: 0 epochs"))
        .stdout(contains("grid         4 cells, 65 runs"));

    let trials = read_parquet(&out.join("trials.parquet"));
    assert_eq!(trials.num_rows(), 65);
    let names: Vec<String> = trials
        .schema()
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect();
    assert_eq!(names[..4], ["dataset", "learning", "mode", "noise"]);
    assert!(names.contains(&"seed".to_owned()) && names.contains(&"class".to_owned()));

    let summary = read_json(&out.join("summary.json"));
    let cells = summary["cells"].as_array().unwrap();
    assert_eq!(cells.len(), 4);
    assert_eq!(cells[1]["noise"], 0.2);
    assert_eq!(cells[1]["flips"], 10);
    assert_eq!(cells[1]["runs"], 20);
    assert!(cells[1]["recovered"]["lower"].is_f64());
    assert_eq!(cells[0]["runs"], 5);
    assert_eq!(cells[0]["deterministic"], true);
    assert_eq!(cells[1]["deterministic"], false);

    let metadata = read_json(&out.join("metadata.json"));
    assert_eq!(metadata["command"], "experiment");
    assert_eq!(metadata["build"]["commit"], build_info::GIT_COMMIT);
    assert_eq!(metadata["config"]["trials"], 4);
    assert_eq!(metadata["datasets"][0]["neurons"], 49);
    assert_eq!(metadata["trainings"][0]["learning"], "hebb");
    assert_eq!(metadata["trainings"][0]["backend"], "dense");
    assert_eq!(
        metadata["trainings"][0]["stability"]["unstable_neurons"]
            .as_array()
            .unwrap()
            .len(),
        5
    );

    let datasets = read_json(&out.join("datasets.json"));
    let small = &datasets["datasets"][0];
    assert_eq!(small["name"], "small-7x7");
    assert_eq!(small["patterns"], 5);
    assert_eq!(small["overlaps"].as_array().unwrap().len(), 5);
    assert_eq!(small["overlap_sums"].as_array().unwrap().len(), 5);
    assert!(small["max_abs_overlap"]["value"].is_f64());
    assert!(!out.join("stability.parquet").exists());
}

/// Covering `round(0.3 * 7) = 2` rows of the 7 x 7 patterns damages 14
/// pixels; levels covering the same rows are rejected.
#[test]
fn experiment_and_recall_cover_rows() {
    let dir = tempfile::tempdir().unwrap();
    let text = EXPERIMENT.replace(
        "source = \"pattern\"\nnoise = [0.0, 0.2]",
        "source = \"occlusion\"\nocclusion = [0.0, 0.3]",
    );
    let config = experiment_config(dir.path(), &text);
    let out = dir.path().join("run");
    experiment(&config)
        .arg("--out")
        .arg(&out)
        .assert()
        .success()
        .stdout(contains("grid         4 cells, 65 runs"));
    let cells = read_json(&out.join("summary.json"))["cells"].clone();
    assert_eq!(cells[1]["noise"], 0.3);
    assert_eq!(cells[1]["flips"], 14);
    let metadata = read_json(&out.join("metadata.json"));
    assert_eq!(metadata["config"]["input"]["source"], "occlusion");
    assert_eq!(metadata["config"]["input"]["occlusion"][1], 0.3);

    let recall_config = run_config(
        dir.path(),
        &RECALL.replace(
            "source = \"pattern\"\nindex = 1\nflips = 6",
            "source = \"occlusion\"\nindex = 1\nrows = 2",
        ),
    );
    recall(&recall_config, &dir.path().join("recall"))
        .success()
        .stdout(contains("input        pattern 1 with 2 covered rows"));

    let clash = experiment_config(dir.path(), &text.replace("[0.0, 0.3]", "[0.1, 0.2]"));
    experiment(&clash)
        .arg("--out")
        .arg(dir.path().join("clash"))
        .assert()
        .failure()
        .stderr(contains("both cover 1 of the 7 rows"));
}

/// One row per pattern and neuron, 5 x 49 for `small-7x7` and one rule,
/// with negative fields at the unstable neurons of the metadata; a run
/// without the option removes the old table.
#[test]
fn experiment_writes_stability_fields_on_request() {
    let dir = tempfile::tempdir().unwrap();
    let config = experiment_config(
        dir.path(),
        &format!("stability_fields = true\n{EXPERIMENT}"),
    );
    let out = dir.path().join("run");
    experiment(&config)
        .arg("--out")
        .arg(&out)
        .assert()
        .success();

    let fields = read_parquet(&out.join("stability.parquet"));
    assert_eq!(fields.num_rows(), 245);
    let column = |name: &str| fields.column_by_name(name).unwrap().clone();
    let pattern = column("pattern");
    let pattern = pattern
        .as_any()
        .downcast_ref::<arrow_array::UInt32Array>()
        .unwrap();
    let field = column("field");
    let field = field
        .as_any()
        .downcast_ref::<arrow_array::Float64Array>()
        .unwrap();
    let mut negative = [0_u64; 5];
    for row in 0..fields.num_rows() {
        if field.value(row) < 0.0 {
            negative[pattern.value(row) as usize] += 1;
        }
    }
    let metadata = read_json(&out.join("metadata.json"));
    assert_eq!(metadata["config"]["stability_fields"], true);
    let unstable: Vec<u64> = metadata["trainings"][0]["stability"]["unstable_neurons"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_u64().unwrap())
        .collect();
    assert_eq!(unstable, negative);

    let config = experiment_config(dir.path(), EXPERIMENT);
    experiment(&config)
        .arg("--out")
        .arg(&out)
        .arg("--force")
        .assert()
        .success();
    assert!(
        !out.join("stability.parquet").exists(),
        "a run without fields leaves an old stability.parquet"
    );
}

/// One row per distinct final state of every cell, with counts adding up
/// to the runs of the cell; a run without the option removes the old table.
#[test]
fn experiment_writes_attractors_on_request() {
    let dir = tempfile::tempdir().unwrap();
    let random = EXPERIMENT.replace(
        "source = \"pattern\"\nnoise = [0.0, 0.2]",
        "source = \"random\"",
    );
    let config = experiment_config(dir.path(), &format!("attractors = true\n{random}"));
    let out = dir.path().join("run");
    experiment(&config)
        .arg("--out")
        .arg(&out)
        .assert()
        .success();

    let table = read_parquet(&out.join("attractors.parquet"));
    let column = |name: &str| table.column_by_name(name).unwrap().clone();
    let mode = column("mode");
    let mode = mode
        .as_any()
        .downcast_ref::<arrow_array::StringArray>()
        .unwrap();
    let count = column("count");
    let count = count
        .as_any()
        .downcast_ref::<arrow_array::UInt32Array>()
        .unwrap();
    let summary = read_json(&out.join("summary.json"));
    for cell in summary["cells"].as_array().unwrap() {
        let runs: u32 = (0..table.num_rows())
            .filter(|&row| mode.value(row) == cell["mode"].as_str().unwrap())
            .map(|row| count.value(row))
            .sum();
        assert_eq!(u64::from(runs), cell["runs"].as_u64().unwrap());
    }
    let metadata = read_json(&out.join("metadata.json"));
    assert_eq!(metadata["config"]["attractors"], true);

    let config = experiment_config(dir.path(), &random);
    experiment(&config)
        .arg("--out")
        .arg(&out)
        .arg("--force")
        .assert()
        .success();
    assert!(
        !out.join("attractors.parquet").exists(),
        "a run without attractors leaves an old attractors.parquet"
    );
}

/// Metadata differs only in `created`.
#[test]
fn experiment_tables_do_not_depend_on_the_number_of_threads() {
    let dir = tempfile::tempdir().unwrap();
    let config = experiment_config(
        dir.path(),
        &format!("stability_fields = true\nattractors = true\n{EXPERIMENT}").replace(
            "[[learning]]\nrule = \"hebb\"\n",
            "[[learning]]\nrule = \"hebb\"\n[[learning]]\nrule = \"oja\"\n",
        ),
    );
    let run = |threads: &str| {
        let out = dir.path().join(format!("threads-{threads}"));
        experiment(&config)
            .env("RAYON_NUM_THREADS", threads)
            .arg("--out")
            .arg(&out)
            .assert()
            .success();
        out
    };
    let (one, four) = (run("1"), run("4"));
    for file in [
        "trials.parquet",
        "summary.json",
        "datasets.json",
        "stability.parquet",
        "attractors.parquet",
    ] {
        assert_eq!(
            std::fs::read(one.join(file)).unwrap(),
            std::fs::read(four.join(file)).unwrap(),
            "{file}"
        );
    }
    assert_eq!(read_parquet(&one.join("trials.parquet")).num_rows(), 130);
    let without_time = |out: &Path| {
        let mut metadata = read_json(&out.join("metadata.json"));
        metadata.as_object_mut().unwrap().remove("created");
        metadata
    };
    assert_eq!(without_time(&one), without_time(&four));
}

#[test]
fn experiment_writes_results_next_to_the_config_by_default() {
    let dir = tempfile::tempdir().unwrap();
    let config = experiment_config(dir.path(), EXPERIMENT);
    experiment(&config)
        .assert()
        .success()
        .stdout(contains("results"));
    let results = dir.path().join("results");
    assert!(results.join("trials.parquet").is_file());

    experiment(&config)
        .assert()
        .failure()
        .stderr(contains("compute::output_exists"));
    experiment(&config).arg("--force").assert().success();
}

#[test]
fn experiment_requires_a_seed() {
    let dir = tempfile::tempdir().unwrap();
    let config = experiment_config(dir.path(), &EXPERIMENT.replace("seed = 5\n", ""));
    experiment(&config)
        .assert()
        .failure()
        .stderr(contains("compute::config"))
        .stderr(contains("`seed`"));
    assert!(!dir.path().join("results").exists());
}

/// Committed `example` results equal the current output with 1 and 4
/// threads.
#[test]
fn example_experiment_matches_the_committed_results() {
    let example = Path::new("../../data/experiments/example");
    let committed = example.join("results");
    let dir = tempfile::tempdir().unwrap();
    for threads in ["1", "4"] {
        let out = dir.path().join(threads);
        experiment(&example.join("config.toml"))
            .env("RAYON_NUM_THREADS", threads)
            .arg("--out")
            .arg(&out)
            .assert()
            .success();
        assert!(
            read_parquet(&out.join("trials.parquet"))
                == read_parquet(&committed.join("trials.parquet")),
            "trials.parquet with {threads} threads differs from the committed one"
        );
        for file in ["summary.json", "datasets.json"] {
            assert!(
                std::fs::read(out.join(file)).unwrap()
                    == std::fs::read(committed.join(file)).unwrap(),
                "{file} with {threads} threads differs from the committed one"
            );
        }
    }
}

const CAPACITY: &str = "seed = 3\ndatasets = [\"small-7x7\"]\ntrials = 4\n\
    [[random]]\nneurons = 60\npatterns = { start = 2, end = 8, step = 3 }\n\
    [[learning]]\nrule = \"hebb\"\n[[learning]]\nrule = \"oja\"\n";

fn capacity(config: &Path) -> Command {
    let mut command = compute();
    command.arg("capacity").arg("--config").arg(config);
    command
}

/// 5 prefixes of `small-7x7` and random `p = 2, 5, 8` with 4 trials each:
/// 17 sets x 2 rules = 34 rows, `(5 + 3) * 2 = 16` summary rows.
#[test]
fn capacity_writes_tables_and_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let config = experiment_config(dir.path(), CAPACITY);
    let out = dir.path().join("run");
    capacity(&config)
        .arg("--out")
        .arg(&out)
        .assert()
        .success()
        .stdout(contains("sets         17 pattern sets x 2 rules"));

    let table = read_parquet(&out.join("capacity.parquet"));
    assert_eq!(table.num_rows(), 34);
    let summary = read_json(&out.join("summary.json"));
    let rows = summary["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 16);
    assert_eq!(rows[0]["source"], "small-7x7");
    assert_eq!(rows[0]["deterministic"], true);
    assert!(rows[0]["theory"].is_null());
    assert_eq!(rows[10]["source"], "random");
    assert_eq!(rows[10]["learning"], "hebb");
    assert_eq!(rows[10]["trials"], 4);
    assert!(rows[10]["theory"]["bit_error"].is_f64());
    assert!(rows[11]["theory"].is_null());

    let metadata = read_json(&out.join("metadata.json"));
    assert_eq!(metadata["command"], "capacity");
    assert_eq!(metadata["config"]["zero_field"], "keep");
    assert_eq!(metadata["datasets"][0]["neurons"], 49);
}

/// Without `--out` the results go next to the configuration.
#[test]
fn capacity_tables_do_not_depend_on_the_number_of_threads() {
    let dir = tempfile::tempdir().unwrap();
    let config = experiment_config(dir.path(), CAPACITY);
    let run = |threads: &str| {
        let out = dir.path().join(format!("threads-{threads}"));
        capacity(&config)
            .env("RAYON_NUM_THREADS", threads)
            .arg("--out")
            .arg(&out)
            .assert()
            .success();
        out
    };
    let (one, four) = (run("1"), run("4"));
    for file in ["capacity.parquet", "summary.json"] {
        assert_eq!(
            std::fs::read(one.join(file)).unwrap(),
            std::fs::read(four.join(file)).unwrap(),
            "{file}"
        );
    }
    capacity(&config).assert().success();
    assert!(
        dir.path()
            .join("results")
            .join("capacity.parquet")
            .is_file()
    );
}

#[test]
fn capacity_rejects_too_many_random_patterns() {
    let dir = tempfile::tempdir().unwrap();
    let config = experiment_config(dir.path(), &CAPACITY.replace("end = 8", "end = 61"));
    capacity(&config)
        .assert()
        .failure()
        .stderr(contains("compute::invalid_config"))
        .stderr(contains("at most N"));
}

const BENCHMARK: &str = "seed = 5\ndatasets = [\"small-7x7\"]\nrepeats = 2\n\
                         backends = [\"dense\", \"factored\"]\ndense_limit = 49\nnoise = 0.1\n\
                         [random]\nneurons = [40, 80]\npatterns = 3\n\
                         [[learning]]\nrule = \"hebb\"\n[[learning]]\nrule = \"oja\"\n";

/// 20 measurements with the heap counted by the binary's allocator; dense
/// `small-7x7` weights take `8 * 49^2 = 19208` bytes, and training them
/// allocates at least as much.
#[test]
fn benchmark_writes_the_table_and_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let config = experiment_config(dir.path(), BENCHMARK);
    let out = dir.path().join("run");
    let benchmark = || {
        let mut command = compute();
        command
            .arg("benchmark")
            .arg("--config")
            .arg(&config)
            .arg("--out")
            .arg(&out);
        command
    };
    benchmark()
        .assert()
        .success()
        .stdout(contains("measurements 20"));

    let table = read_parquet(&out.join("benchmark.parquet"));
    assert_eq!(table.num_rows(), 20);
    let column = |name: &str| table.column_by_name(name).unwrap().clone();
    let peak = column("train_peak_bytes");
    let peak = peak
        .as_any()
        .downcast_ref::<arrow_array::UInt64Array>()
        .unwrap();
    assert_eq!(arrow_array::Array::null_count(peak), 0);
    let bytes = column("weights_bytes");
    let bytes = bytes
        .as_any()
        .downcast_ref::<arrow_array::UInt64Array>()
        .unwrap();
    assert_eq!(bytes.value(0), 19208);
    assert!(peak.value(0) >= 19208);
    let metadata = read_json(&out.join("metadata.json"));
    assert_eq!(metadata["command"], "benchmark");
    assert_eq!(metadata["heap_measured"], true);
    assert!(metadata["threads"].as_u64().unwrap() >= 1);

    benchmark()
        .assert()
        .failure()
        .stderr(contains("compute::output_exists"));
    benchmark().arg("--force").assert().success();
}

/// The committed `recall-cycle.toml`: a 2-cycle after one synchronous step
/// to the negated checkerboard (25 changes) with the same energy.
#[test]
fn checkerboard_oscillates_in_the_network_of_the_cycle_shapes() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("run");
    recall(
        Path::new("../../data/experiments/cycle/recall-cycle.toml"),
        &out,
    )
    .success();
    let metadata = read_json(&out.join("metadata.json"));
    assert_eq!(metadata["outcome"]["termination"], "two-cycle");
    assert_eq!(metadata["outcome"]["steps"], 1);
    assert_eq!(metadata["outcome"]["changes"], 25);
    let history = read_npy(&out.join("history.npy")).unwrap();
    assert_eq!(history.len(), 2);
    assert_eq!(history.pattern(1), -&history.pattern(0));
    let frames = read_json(&out.join("frames.json"));
    assert_eq!(frames["energy"][0], frames["energy"][1]);
}

/// 5 x 3 bitmaps: the basis (14 patterns) and a sample (24) of the best set
/// are written as `(p, 3, 5)` datasets with provenance.
#[test]
fn search_stable_writes_the_summary_and_the_datasets() {
    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("config.toml");
    std::fs::write(
        &config,
        "width = 5\nheight = 3\nmax_weight = 2\nverify = 2\n\
         basis = \"data/basis.npy\"\nsample = \"data/sample.npy\"\n",
    )
    .unwrap();
    let out = dir.path().join("run");
    let search = || {
        let mut command = compute();
        command
            .arg("search-stable")
            .arg("--config")
            .arg(&config)
            .arg("--out")
            .arg(&out);
        command
    };
    search()
        .assert()
        .success()
        .stdout(contains("best         |S_u| = 6006"));

    let summary = read_json(&out.join("summary.json"));
    assert_eq!(summary["best"]["size"], 6006);
    assert_eq!(summary["best"]["hebb"]["stable"], 6006);
    assert_eq!(summary["best"]["oja"]["stable"], 6006);
    assert_eq!(summary["verified"].as_array().unwrap().len(), 2);
    assert_eq!(summary["candidates"].as_array().unwrap().len(), 15);
    let data = dir.path().join("data");
    assert_eq!(npy_shape(&data.join("basis.npy")), [14, 3, 5]);
    assert_eq!(npy_shape(&data.join("sample.npy")), [24, 3, 5]);
    let provenance = read_json(&data.join("basis.json"));
    assert_eq!(provenance["command"], "search-stable");
    assert_eq!(provenance["content"], "basis");
    assert_eq!(provenance["set_size"], 6006);
    assert_eq!(provenance["config"], "../config.toml");
    assert_eq!(
        read_json(&out.join("metadata.json"))["command"],
        "search-stable"
    );

    search()
        .assert()
        .failure()
        .stderr(contains("compute::output_exists"));
    search().arg("--force").assert().success();
}

/// Whether the written file `name` in `out` equals the one in `expected`: a
/// Parquet table as decoded data, because the file also records the version
/// of the `parquet` crate; any other file byte for byte.
fn same_file(out: &Path, expected: &Path, name: &str) -> bool {
    if name.ends_with(".parquet") {
        read_parquet(&out.join(name)) == read_parquet(&expected.join(name))
    } else {
        std::fs::read(out.join(name)).unwrap() == std::fs::read(expected.join(name)).unwrap()
    }
}

/// Committed `stability-single-neuron` results equal the current output.
#[test]
fn single_neuron_stability_matches_the_committed_results() {
    let experiment_dir = Path::new("../../data/experiments/stability-single-neuron");
    let committed = experiment_dir.join("results");
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("run");
    experiment(&experiment_dir.join("config.toml"))
        .arg("--out")
        .arg(&out)
        .assert()
        .success();
    for file in [
        "trials.parquet",
        "summary.json",
        "datasets.json",
        "stability.parquet",
    ] {
        assert!(
            same_file(&out, &committed, file),
            "{file} differs from the committed one"
        );
    }
}

/// Fixtures in `src/analysis/tests/fixtures` equal the current output
/// (regenerate per their README).
#[test]
fn analysis_fixtures_match_the_current_output() {
    let fixtures = Path::new("../analysis/tests/fixtures");
    let dir = tempfile::tempdir().unwrap();
    let tables = &["trials.parquet", "summary.json", "datasets.json"][..];
    for (fixture, command, files) in [
        ("recall", "recall", &["history.npy", "frames.json"][..]),
        ("train", "train", &["weights.npy"][..]),
        (
            "experiment",
            "experiment",
            &[
                "trials.parquet",
                "summary.json",
                "datasets.json",
                "stability.parquet",
            ][..],
        ),
        ("occlusion", "experiment", tables),
        (
            "random",
            "experiment",
            &[
                "trials.parquet",
                "summary.json",
                "datasets.json",
                "attractors.parquet",
            ][..],
        ),
        (
            "capacity",
            "capacity",
            &["capacity.parquet", "summary.json"][..],
        ),
    ] {
        let out = dir.path().join(fixture);
        compute()
            .arg(command)
            .arg("--config")
            .arg(fixtures.join(format!("{fixture}.toml")))
            .arg("--out")
            .arg(&out)
            .assert()
            .success();
        for file in files {
            assert!(
                same_file(&out, &fixtures.join(fixture), file),
                "{fixture}/{file} differs from the fixture; regenerate it"
            );
        }
    }
}
