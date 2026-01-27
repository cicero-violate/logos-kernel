# Judgment Invariant

This document defines how Judgment constrains execution.

---

## Current Invariant

Judgment is applied **once per graph** before execution.

Formally:

```
ComputeMode = Judge(TypeSig)
For all nodes N in graph:
  N executes using ComputeMode
```

---

## Properties

1. Judgment does not affect identity.
2. Judgment does not affect hashing.
3. Judgment does not affect structural ordering.
4. Judgment only selects execution mode.

---

## Relationship to Policy

Judgment operates only within capabilities already permitted by Policy.

Policy decides:
- Whether execution is admitted
- What capabilities are allowed
- What budgets or caps apply

Judgment decides:
- Which compute mode to select (e.g. Logic vs Differential)

Judgment must NOT:
- Deny execution
- Enforce budgets
- Apply security or admission rules

Those responsibilities belong exclusively to the Policy layer.

---

## Non-Goals (For Now)

- Per-node judgment
- Dynamic judgment during execution
- Learning-based judgment

---

## Change Policy

Any move to per-node judgment requires:

1. New invariant version
2. Kernel public surface bump
3. Replay format version bump

---

Status: Frozen
Owner: Kernel
