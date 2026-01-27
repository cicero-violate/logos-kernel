# Encoding Invariant

This document defines canonical byte encoding requirements.

---

## Current Invariant

All kernel values that participate in hashing or replay MUST implement:

```
trait Encode { fn encode(&self) -> Vec<u8>; }
```

Encoding must be:

1. Deterministic
2. Platform-independent
3. Total (no panics)

---

## Properties

1. Same value → identical bytes.
2. Bytes uniquely determine value.
3. No implicit padding.
4. No pointer addresses.
5. No randomness.

---

## Non-Goals (For Now)

- Compression
- Variable endianness
- Schema evolution

---

## Change Policy

Any change requires:

1. New invariant version
2. Identity invariant update
3. Kernel public surface bump

---

Status: Frozen
Owner: Kernel

