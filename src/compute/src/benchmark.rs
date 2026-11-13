//! Benchmark: time and memory of training and recall as the number of
//! neurons grows, for random pattern sets and registered datasets, with
//! every weight storage backend.

pub mod config;
pub mod metadata;
pub mod probe;
pub mod run;
pub mod table;

pub use config::{BenchmarkConfig, RandomSizes};
pub use metadata::BenchmarkMetadata;
pub use probe::{HeapProbe, NoProbe};
pub use run::{BenchmarkRun, BenchmarkSet, Measurement};
pub use table::BenchmarkTable;
