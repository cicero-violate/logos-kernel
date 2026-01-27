// Scope: Decide which compute path a stage may use
//
// NOTE:
// Policy (admission, budgeting, capability gating) is applied BEFORE this layer.
// Judge assumes execution is already admitted and only selects compute mode.

use crate::kernel::types::*;

pub struct DefaultJudge;

impl DefaultJudge {
    pub fn choose(sig: &TypeSig) -> ComputeMode {
        if sig.allow_logic {
            ComputeMode::Logic
        } else if sig.allow_differential {
            ComputeMode::Differential
        } else {
            ComputeMode::Differential
        }
    }
}
