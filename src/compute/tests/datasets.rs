//! Every dataset of the repository registry loads with the expected size.

use std::path::Path;

use compute::patterns::registry::Registry;

/// Name, number of patterns `p`, width and height.
const EXPECTED: [(&str, usize, usize, usize); 8] = [
    ("small-7x7", 5, 7, 7),
    ("letters-abc-8x12", 3, 8, 12),
    ("animals-14x9", 6, 14, 9),
    ("letters-14x20", 26, 14, 20),
    ("OCRA-12x30-cut", 7, 12, 30),
    ("large-25x25", 6, 25, 25),
    ("large-25x25.plus", 15, 25, 25),
    ("large-25x50", 6, 25, 50),
];

fn registry() -> Registry {
    Registry::discover(Path::new(env!("CARGO_MANIFEST_DIR"))).expect("repository registry")
}

#[test]
fn registry_lists_exactly_the_expected_datasets() {
    let mut expected: Vec<_> = EXPECTED.iter().map(|(name, ..)| *name).collect();
    expected.sort_unstable();
    assert_eq!(registry().names().collect::<Vec<_>>(), expected);
}

#[test]
fn every_dataset_loads_with_the_expected_size() {
    let registry = registry();
    for (name, p, width, height) in EXPECTED {
        let set = registry
            .load(name)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            (set.len(), set.width(), set.height(), set.neurons()),
            (p, width, height, width * height),
            "{name}"
        );
    }
}
