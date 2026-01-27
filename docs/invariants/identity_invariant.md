# Identity Invariant

This document defines how identity is assigned to nodes, graphs, and values.

---

## Node Identity

NodeId = H(name || stage_type || ordered input NodeIds)

Where:
- name is a static string
- stage_type is the concrete Rust type name of the Stage
- inputs are ordered

---

## Graph Identity

GraphHash = H(
  for each NodeId in sorted order:
    NodeId ||
    node.name ||
    stage_type ||
    ordered input NodeIds
)

---

## Value Identity

ValueHash = H(Encode(value))

---

## Invariants

1. Identity is content-addressed.
2. Identity is deterministic.
3. Identity does not depend on execution order.
4. Identity does not depend on ComputeMode.
5. Identity does not depend on Judgment.

---

## Change Policy

Any change requires:

1. New identity invariant version
2. NodeId version bump
3. GraphHash version bump
4. Replay format version bump

---

Status: Frozen
Owner: Kernel

