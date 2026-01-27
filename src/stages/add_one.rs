// Scope: Example concrete stage

use crate::kernel::pipeline::*;
use crate::kernel::types::*;
use crate::kernel::error::PipelineError;

#[derive(Clone)]
pub struct AddOne;

impl Stage for AddOne {
    type T = i64;

    fn logic(&self, x: i64, _: &TypeSig)
        -> Result<(Delta<i64>, Proof), PipelineError>
    {
        Ok((Delta { before: x, after: x + 1 }, vec![]))
    }

    fn differential(&self, x: i64, _: &TypeSig)
        -> Result<(Delta<i64>, f32), PipelineError>
    {
        Ok((Delta { before: x, after: x + 1 }, 1.0))
    }
}
