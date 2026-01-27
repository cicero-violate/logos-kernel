# Replay Invariant

This document defines the semantics of execution replay logging.

---

## Current Invariant

Each successful pipeline execution records:

```
(GraphHash, InputHash, OutputHash)
```

in an append-only log.

---

## Properties

1. Replay records are append-only.
2. Ordering of records is chronological.
3. Replay does not affect execution results.
4. Replay does not affect identity or hashing.
5. Same (GraphHash, InputHash) must always correspond to the same OutputHash.

---

## Non-Goals (For Now)

- Distributed replay
- Compaction
- Merkle indexing
- Cryptographic signatures

---

## Change Policy

Any change requires:

1. New invariant version
2. Replay format version bump
3. Kernel public surface bump

---

Status: Frozen
Owner: Kernel

