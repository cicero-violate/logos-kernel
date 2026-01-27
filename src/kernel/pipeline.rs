// Scope: Stage and Pipeline traits (execution model)
//
// Semantic obligations for all Stage implementations:
// See src/kernel/stage_contract.rs
// All implementations of Stage MUST conform to that contract.

use crate::kernel::types::*;
use crate::kernel::error::PipelineError;

pub struct ComputeOut<T> {
    pub delta: Delta<T>,
    pub proof: Option<Proof>,
}

pub trait Stage: Send + Sync {
    type T;

    fn logic(&self, input: Self::T, sig: &TypeSig)
        -> Result<(Delta<Self::T>, Proof), PipelineError>;

    fn differential(&self, input: Self::T, sig: &TypeSig)
        -> Result<(Delta<Self::T>, f32), PipelineError>;

    fn run_with_mode(
        &self,
        input: Self::T,
        sig: &TypeSig,
        mode: ComputeMode
    ) -> Result<ComputeOut<Self::T>, PipelineError> {
        match mode {
            ComputeMode::Logic => {
                let (d, p) = self.logic(input, sig)?;
                Ok(ComputeOut { delta: d, proof: Some(p) })
            }
            ComputeMode::Differential => {
                let (d, _) = self.differential(input, sig)?;
                Ok(ComputeOut { delta: d, proof: None })
            }
        }
    }
}
