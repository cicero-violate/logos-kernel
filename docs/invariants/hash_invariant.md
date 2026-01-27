# Hash Invariant

This document defines requirements for hashing inside the kernel.

---

## Current Invariant

All hashes use **FNV-1a 64-bit**.

No other hash algorithm is permitted.

---

## Hash Domains

1. NodeId
2. GraphHash
3. ValueHash
4. Replay InputHash / OutputHash

---

## Properties

1. Deterministic
2. Platform-independent
3. Pure function of bytes
4. No salting
5. No runtime configuration

---

## Non-Goals (For Now)

- Cryptographic security
- Pluggable hash algorithms
- Hash agility

---

## Change Policy

Any change requires:

1. New invariant version
2. Identity invariant update
3. Replay invariant update
4. Kernel public surface bump

---

Status: Frozen
Owner: Kernel

