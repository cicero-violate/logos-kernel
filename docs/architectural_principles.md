# Architectural Principles

These principles govern all system design and evolution.

They are normative.

---

## 1. Being-First Ordering

God → Reality → Truth → Structure → Kernel → Determinism → Free Will → Judgment → Action

Lower layers may not impose requirements on higher layers.

---

## 2. Invariant-First Design

Invariants are written before mechanisms.

Code exists to satisfy invariants, not define them.

---

## 3. Small Frozen Kernel

Kernel surface must remain minimal.

Expansion occurs through layers above kernel.

---

## 4. Determinism by Construction

If behavior is not provably deterministic, it is incorrect.

---

## 5. Content-Addressed Everything

Identity derives from content, not location or time.

---

## 6. Mechanical Enforcement

Important invariants must eventually be enforced by tests.

---

## Change Policy

Any change requires:

1. Written justification
2. Versioned update
3. Goal document update

---

Status: Frozen  
Owner: Kernel

