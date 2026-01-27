# Determinism Invariant

This document defines determinism requirements for kernel execution.

---

## Current Invariant

Given:

```
Graph
Input
TypeSig
```

The kernel must produce:

```
Identical Output
Identical Replay Record
```

across runs.

---

## Sources of Determinism

1. Canonical structural hashing
2. Deterministic topological sort
3. Deterministic layer ordering
4. Deterministic Encode bytes
5. Deterministic Stage behavior

---

## Properties

1. No dependence on thread scheduling.
2. No dependence on memory addresses.
3. No dependence on wall-clock time.
4. No randomness.

---

## Non-Goals (For Now)

- Bitwise identical floating point across architectures
- GPU determinism

---

## Change Policy

Any change requires:

1. New invariant version
2. Replay invariant update
3. Kernel public surface bump

---

Status: Frozen
Owner: Kernel

