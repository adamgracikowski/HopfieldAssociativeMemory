//! Generated pattern sets: groups of similar patterns made from stored
//! pictures.
//!
//! Every group starts from one prototype, a pattern of a registered
//! dataset, and holds `variants` copies of it with a fraction `flip` of the
//! pixels flipped at random. A variant has overlap exactly `1 - 2k/N` with
//! its prototype (`k = round(flip * N)` flipped pixels) and two variants of
//! one prototype about `(1 - 2k/N)^2` with each other, so the similarity
//! inside a group is set by `flip`, while the groups are as different as
//! their prototypes.

pub mod config;
pub mod output;
pub mod provenance;
pub mod set;

pub use config::GroupsConfig;
pub use output::GroupsOutput;
pub use provenance::GroupsProvenance;
pub use set::GroupSet;
