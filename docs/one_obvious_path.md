# One Obvious Path

This document defines what "one obvious path" means for this project.

---

## Principle

For any core operation, there should exist exactly **one** endorsed way to perform it.

Multiple equivalent mechanisms are considered accidental complexity.

---

## Kernel Layer

Execution:

```
DagPipeline::run(input, sig)
```

No alternative execution entrypoints are permitted.

---

## Identity

Only:

```
DagNode::id()
structural_graph_hash(...)
Encode
```

may participate in identity.

---

## Replay

Only:

```
replay::record(...)
replay::load_from_disk(...)
replay::verify(...)
```

---

## Interfaces

CLI, Server, and Library must all call the same pipeline path.

No interface-specific execution logic.

---

## Disallowed

- Parallel alternative kernels
- Shadow execution paths
- Hidden fast paths
- Feature-flagged behavior inside kernel

---

## Change Policy

Any additional path requires:

1. Written justification
2. Invariant update
3. Goal document update

---

Status: Frozen  
Owner: Kernel

