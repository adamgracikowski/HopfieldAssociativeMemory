//! Generated pattern sets: groups of similar patterns made from stored
//! pictures.
//!
//! Every group holds `variants` copies of one prototype (a pattern of a
//! registered dataset) with `k = round(flip * N)` random pixels flipped, so
//! each variant has overlap `1 - 2k/N` with its prototype.

pub mod config;
pub mod output;
pub mod provenance;
pub mod set;

pub use config::GroupsConfig;
pub use output::GroupsOutput;
pub use provenance::GroupsProvenance;
pub use set::GroupSet;
