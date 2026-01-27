# Kernel Public Surface (Frozen)

This document enumerates the **intentionally exposed kernel surface**.
Anything not listed here is considered internal and may change.

Once stabilized, this surface must remain unchanged for ≥ 90 days.

---

## Modules

- kernel::types  
- kernel::pipeline  
- kernel::dag  
- kernel::judge  
- kernel::codec  
- kernel::hash  
- kernel::replay  
- kernel::error  

---

## Core Traits

### Stage

```
trait Stage {
    type T;
    fn logic(...)
    fn differential(...)
    fn run_with_mode(...)
}
```

### DagPipeline

```
trait DagPipeline {
    type T;
    fn nodes(&self) -> HashMap<NodeId, DagNode<T>>;
    fn output_node(&self) -> NodeId;
    fn run(...)
}
```

### Encode / Decode

```
trait Encode { fn encode(&self) -> Vec<u8>; }
trait Decode { fn decode(bytes: &[u8]) -> Option<Self>; }
```

---

## Core Types

- NodeId = u64  
- Delta<T>  
- NodeResult<T>  
- TypeSig  
- ComputeMode  
- Proof  
- PipelineError  

---

## Invariants

1. All hashes originate from canonical `Encode` bytes.
2. NodeId = H(name || stage_type || input NodeIds)
3. GraphHash includes node ids, names, stage identity, and edges.
4. Same graph + same input → identical output + identical hashes.
5. Kernel values have canonical byte encoding.
6. Delta encoding is length-framed.
7. Hash algorithm is FNV-1a 64-bit and frozen.
8. Topological traversal is deterministic.
9. Judgment selection is global (per-graph).
10. Node input arity is restricted to zero or one input (see node_input_arity_invariant.md).
11. Identity semantics are defined in docs/invariants/identity_invariant.md.
12. Judgment semantics are defined in docs/invariants/judgment_invariant.md.
13. Replay semantics are defined in docs/invariants/replay_invariant.md.
14. Encoding semantics are defined in docs/invariants/encoding_invariant.md.
15. Hash semantics are defined in docs/invariants/hash_invariant.md.
16. Determinism semantics are defined in docs/invariants/determinism_invariant.md.
17. Kernel surface freeze semantics are defined in docs/invariants/kernel_surface_freeze_invariant.md.
15. Hash semantics are defined in docs/invariants/hash_invariant.md.

---

## Non-Goals (For Now)

- Distributed execution
- Per-node judgment
- MMSB integration

---

Status: Frozen
Owner: Kernel
