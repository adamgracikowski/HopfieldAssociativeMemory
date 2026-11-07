//! Capacity: how many stored patterns stay stable as the number of patterns
//! `p` grows, for prefixes of registered datasets and for random patterns.

pub mod config;
pub mod metadata;
pub mod run;
pub mod source;
pub mod summary;
pub mod table;

pub use config::{CapacityConfig, PatternRange, RandomSets};
pub use metadata::CapacityMetadata;
pub use run::{CapacityRow, CapacityRun};
pub use source::{PatternSource, Task};
pub use summary::{CapacitySummary, Theory};
pub use table::CapacityTable;
