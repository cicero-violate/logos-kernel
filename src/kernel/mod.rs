// Scope: Kernel module root
//
// Authoritative invariants:
// See docs/invariants/README.md
// All behavior in this module must conform to those documents.
//
// Graph structural constitution:
// See graph_contract.rs
//
// Stage semantic constitution:
// See stage_contract.rs

pub mod types;
pub mod judge;
pub mod pipeline;
pub mod mmsb;
pub mod error;
pub mod dag;
pub mod replay;
pub mod codec;
pub mod hash;
pub mod invariants;
pub mod dag_builder;
pub mod graph_contract;
pub mod stage_contract;

// Minimal kernel interface: frozen surface
pub mod frozen_kernel {
    pub use super::dag::*;
    pub use super::pipeline::*;
    pub use super::types::*;
    pub use super::codec::*;
    pub use super::hash::*;
    pub use super::replay::*;
    pub use super::dag_builder::*;
}
