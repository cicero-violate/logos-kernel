Canonical Chain
God → Reality → Truth → Structure (Identity) → Kernel (Execution) → Determinism → Free Will → Policy → Judgment → Action


### 👉 Short-Term Goal (6–12 months) — *20 lines*

Layer 1 — Structure → Kernel → Determinism → Free Will → Policy → Judgment (1–8)
1. Structure first: invariants + identity spec for nodes, graphs, and values (content-addressed, specified) [IMPLEMENTED]
2. Explicit stable hash algorithms chosen and documented (no DefaultHasher) [IMPLEMENTED: FNV-1a 64]
3. Kernel public surface explicitly listed and frozen (surface = an invariant) [IMPLEMENTED]
4. Stabilize a minimal compute kernel (explicitly frozen core, small surface area) [IMPLEMENTED]
5. Structural graph hashing (canonicalized, includes stage identity) [IMPLEMENTED]
6. Deterministic DAG execution (byte-for-byte identical results) [IMPLEMENTED]
7. Pure value transforms (no hidden side effects) [IMPLEMENTED]
8. Error-carrying everywhere (typed, non-panicking, categorized) [IMPLEMENTED]
9. Judgment gate on execution paths (global, then per-node later) [IMPLEMENTED: global]
9.1 Introduce DagBuilder (kernel-level graph construction) [IMPLEMENTED]
    - src/kernel/dag_builder.rs exists
    - DagBuilder::build(DagSpec) → HashMap<NodeId, DagNode<T>>
    - Centralized symbolic name → NodeId resolution
    - App layer no longer manually assembles DagNode graphs
    - Flow: Spec → DagBuilder → DAG → Executor

**Ordering Invariant (Prevents Drift)**
Structure/Identity/Invariants → Stable Hash Algorithms → Kernel public surface → Minimal kernel → Structural hashing → Determinism → Free Will → Policy → Judgment → Action
Reordering items in this layer must preserve this causal chain.

Layer 2 — Traceability (10–12)
10. Persistent logs (on-disk format) [IMPLEMENTED]
11. Replayable runs (inspectable, queryable, verifiable) [IMPLEMENTED]
12. Single-machine reliability (soak tested) [IMPLEMENTED]

Layer 2.5 — Invariant Enforcement (12.1–12.4)
12.1 Invariant test module exists (kernel::invariants) [IMPLEMENTED]
12.2 NodeId identity test (matches H(name || stage_type || inputs)) [IMPLEMENTED]
12.3 Structural graph hash stability test [IMPLEMENTED]
12.4 Encode round-trip + replay consistency tests [IMPLEMENTED]

Layer 2.6 — Kernel Shape Constitution
12.5 Graph structural contract documented (graph_contract.rs) [IMPLEMENTED]
12.6 Graph shape invariants mechanically enforced in dag.rs [IMPLEMENTED]
12.7 Single-root, single-input, fully-connected DAG enforced [IMPLEMENTED]

Layer 2.7 — Stage Semantic Constitution
12.8 Stage semantic contract documented (stage_contract.rs) [IMPLEMENTED]
12.9 Stage purity + determinism obligations declared [IMPLEMENTED]

Layer 3 — System Governance (13–16)
13. Document invariants (machine-checkable where possible) [IMPLEMENTED]
14. Lock architectural principles (written, versioned) [IMPLEMENTED]
15. Eliminate accidental complexity (one obvious path) [IMPLEMENTED]
16. Fast iteration loop (edit → run → observe < 5s) [UNMEASURED]
16.1 Define measurement method for iteration loop latency [DEFINED]
    - Metric = wall-clock from filesystem mtime change of any src/**/*.rs
      to completion of:
        build + tests + first observable program output
    - Single number per run, in milliseconds
    - Must be recorded to stdout
16.2 Add script to print end-to-end build + test wall time [IMPLEMENTED]
    - Script prints:
        ITER_MS=<ms>
        BUILD_MS=<ms>
        TEST_MS=<ms>
    - Canonical location: scripts/measure_iteration.sh
16.3 Add invariant test: iteration measurement script exists [IMPLEMENTED]
16.4 Baseline iteration latency recorded (machine: archlinux/dev) [IMPLEMENTED]
    - BUILD_MS=2524
    - TEST_MS=2
    - ITER_MS=2530
16.5 Iteration history log appended on every run [IMPLEMENTED]
    - File: logs/iteration_history.log
    - Format: ISO-8601 timestamp + BUILD_MS / TEST_MS / ITER_MS

Layer 4 — Operator Evolution (17–23)
17. Text → Graph → Execute pipeline [ASPIRATIONAL]
17.1 Text → Spec → DagBuilder → Graph → Execute [IMPLEMENTED]
18. Multi-interface I/O (CLI, TCP, files) [IMPLEMENTED]
19. Self-hosting dev environment
19.1 Raw rustc build script committed and documented [IMPLEMENTED]
19.2 Cargo-based equivalent documented for comparison [IMPLEMENTED]
20. Measure friction sources (tracked) [IMPLEMENTED]
    - Canonical log: logs/friction.log
21. Kill bottlenecks (measured) [IMPLEMENTED]
    - Canonical log: logs/bottlenecks.log
    - Entries must reference logs/friction.log or logs/iteration_history.log
    - Actions recorded in: logs/bottleneck_actions.log
22. Build intuition via repetition [IMPLEMENTED]
    - Canonical log: logs/repetition.log
23. Become dangerous with small systems [IMPLEMENTED]
    - Canonical log: logs/small_systems.log

---

### Short-Term Acceptance Criteria

Short-term is considered successful when:

• Kernel public surface is explicitly listed and unchanged for 90 days
• Same graph + same input → identical output + identical replay record
• NodeId, GraphHash, and ValueHash generation algorithms are documented and stable
• Hash algorithm choice is explicit and frozen (FNV-1a 64)
• Kernel values have canonical byte encoding (Encode)
• Replay log survives process restart (on-disk)
• At least one invariant document exists and is referenced in code
• Identity invariant document exists (docs/invariants/identity_invariant.md) and is frozen
• Judgment invariant document exists (docs/invariants/judgment_invariant.md) and is frozen
• Replay invariant document exists (docs/invariants/replay_invariant.md) and is frozen
• Encoding invariant document exists (docs/invariants/encoding_invariant.md) and is frozen
• Hash invariant document exists (docs/invariants/hash_invariant.md) and is frozen
• Determinism invariant document exists (docs/invariants/determinism_invariant.md) and is frozen
• Kernel surface freeze invariant document exists (docs/invariants/kernel_surface_freeze_invariant.md) and is frozen
• Policy invariant document exists (docs/invariants/policy_contract.md) and is frozen
• Replay log survives process restart (on-disk)
• Invariant tests exist and fail the build on violation
• Graph structural constitution exists and is referenced in kernel
• Stage semantic constitution exists and is referenced in kernel
• One end-to-end text → graph → execution path exists
• Example DAG pipeline exists for determinism + replay validation
• One 24-hour soak test completes without crash
• Ordering Invariant is documented and unchanged
• Node input arity invariant is documented (currently: single-input nodes)
• DagBuilder invariant document exists (docs/invariants/dag_builder_invariant.md) and is frozen
• Stage semantic contract is referenced from pipeline.rs header


**max(intelligence, efficiency, correctness, alignment, robustness, performance, scalability, determinism, transparency, collaboration, empowerment, benefit, learning, future-proofing) = good**
