# Node Input Arity Invariant

This document defines the permitted number of inputs per node.

---

## Current Invariant

Each node may have **zero or one input**.

Formally:

```
inputs.len() ∈ {0, 1}
```

---

## Rationale

1. Simplifies identity
2. Simplifies hashing
3. Simplifies execution model
4. Enables early determinism
5. Avoids premature tuple semantics

---

## Code Implications

- DAG execution reads only `inputs[0]`
- NodeId hashing incorporates ordered input list
- Structural hash assumes ordered inputs

---

## Non-Goals (For Now)

- Fan-in nodes
- Variadic arity
- Tuple packing
- Argument labeling

These may be introduced later as **explicit structural types** rather than expanding node arity.

---

## Change Policy

Any relaxation of this invariant requires:

1. New invariant document
2. NodeId version bump
3. GraphHash version bump
4. Replay format version bump

---

Status: Frozen
Owner: Kernel
