// Scope: Core shared types for the kernel (small and frozen)

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ComputeMode {
    Logic,
    Differential,
}

#[derive(Clone, Copy)]
pub struct TypeSig {
    pub allow_logic: bool,
    pub allow_differential: bool,
}

pub type Proof = Vec<u8>;

pub struct Delta<T> {
    pub before: T,
    pub after: T,
}

// Node-level execution result
pub enum NodeResult<T> {
    Value {
        delta: Delta<T>,
        proof: Option<Proof>,
    },
    Error(PipelineError),
}

pub const DEFAULT_SIG: TypeSig = TypeSig {
    allow_logic: true,
    allow_differential: true,
};
use crate::kernel::error::PipelineError;
