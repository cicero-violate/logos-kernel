// Scope: Example concrete stage

use crate::kernel::pipeline::*;
use crate::kernel::types::*;
use crate::kernel::error::PipelineError;

#[derive(Clone)]
pub struct MulTwo;

impl Stage for MulTwo {
    type T = i64;

    fn logic(&self, x: i64, _: &TypeSig)
        -> Result<(Delta<i64>, Proof), PipelineError>
    {
        Ok((Delta { before: x, after: x * 2 }, vec![]))
    }

    fn differential(&self, x: i64, _: &TypeSig)
        -> Result<(Delta<i64>, f32), PipelineError>
    {
        Ok((Delta { before: x, after: x * 2 }, 1.0))
    }
}
