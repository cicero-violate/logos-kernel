# Kernel Surface Freeze Invariant

This document defines the freeze semantics for the kernel public surface.

---

## Current Invariant

The contents of:

```
docs/kernel_public_surface.md
```

define the entire kernel public surface.

Once marked **Frozen**, this surface must not change for ≥ 90 days.

---

## Properties

1. Additions are prohibited.
2. Removals are prohibited.
3. Signature changes are prohibited.
4. Reordering is permitted.
5. Formatting changes are permitted.

---

## Non-Goals (For Now)

- Multiple kernel surface tiers
- Experimental namespaces

---

## Change Policy

Any change requires:

1. Written justification
2. Surface freeze invariant update
3. Kernel public surface bump

---

Status: Frozen
Owner: Kernel

