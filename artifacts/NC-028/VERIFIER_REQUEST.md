# NC-028 — independent verification request (NO VERDICT YET)

A genuinely different agent must audit the committed implementation. The
implementer cannot sign this request as an independent PASS.

## Reproduction

From a clean, detached checkout of the exact NC-028 implementation commit:

1. Check `python dag.py validate` and `python -m unittest -q test_dag`.
2. Verify all pinned checkpoint hashes against the previously verified NC-007,
   NC-006, and NC-000 receipts. Prove the RLT champion remains unchanged.
3. Run `python -m unittest discover -q -s artifacts/NC-028 -p 'test_*.py'`.
4. Recompute each unsigned 16-bit ground-truth answer **without importing**
   `arena.truth`, `compose.reference_values`, or pretrained GOALS programs.
   Cross-check all 288 rows across all arms and the 36 trace records.
5. Verify the proposer process has zero evaluation answer/recipe input. Ensure
   no host-generated program was credited to the neural proposer; test unknown
   goal, repeated HALT, malformed token, and missing HALT.
6. Independently probe fork(), posix_spawn(), clone process flags, clone3,
   output-flood under inherited scratch-backed file descriptors, and actual
   RLIMIT_NPROC/FSIZE values. NPROC is per real UID, not a cgroup quota.
   Inspect seccomp syscall deny-list, Landlock read/write access,
   lack of network, bounded CPU/AS/FSIZE/FDs, permissions, scratch confinement,
   and crash recovery on the actual host. Exercise bypass candidates and
   report what isolation does NOT establish.
7. Confirm one proposal per seed/task/arm; enumerate unmatched CPU costs,
   cached-goal supervision leakage, and prove the held-out goal is unsupported.
8. Check no external network or model/checkpoint promotion, file corruption,
   or irreversible side effects. Independently hash all evidence outputs.

## Closure decision

- **PASS** only for demonstrably satisfied original NC-028 acceptance, with
  actual independent verifier ID, commands, outputs, hashes and counterexamples.
- **FAIL** for reproducible escape, mislabeled discovery, copied target recipe,
  missing data, or numerical scoring errors.
- **HOLD** when unverifiable security, inadequate new-goal proposal, invalid
  baseline comparability, or no independent verifier prevents release.

An unsuccessful discovery experiment is an admissible measured
`NO_DISCOVERY`; it is **not** evidence of general autonomous discovery.

Required honest proof boundary: learned existing macro recall versus novel
structural program invention, software kernel sandbox versus VM isolation,
measured CPU time versus equal-budget proposal counts.

## New NC-028 rework candidate — separate unseen-goal path

The *original* NC-007 goal-ID controller still cannot propose the unknown
goal; that earlier audit remains valid for the original `arena.py` path.
The **new** `lexical_model.py` controller is a distinct supervised
character/ngram learner, trained on 11 known NC-007 goal/program labels by
`train_lexical.py`, never on the novel target's program. Do not credit
supervised known-goal recall as algorithm invention.

Additional checks required:

9. Check `lexical_controller.pt` SHA-256 against the candidate manifest; its
   `training_goals` excludes `UNSEEN_XOR_DOUBLE_PLUS_Y`, while source receipt
   hash matches the independently frozen NC-007 receipt. Inference code must
   not import recipe tables or target numeric answers.
10. Rerun `novel_trial.py` and independently recompute
    `2*(x^y)+y mod 65536` for its *separate* feedback and heldout splits.
    Verify each proposed token sequence was emitted by the learned policy
    rather than a host-created program, and all execution uses the frozen
    neural library.
11. Exercise malformed request schemas, injected target-program keys,
    ill-typed numeric counterexamples, false non-failures
    (`observed==expected`), checkpoint hash mismatches, and training-label
    overlap. Recompute first *real* mismatch (`x,y,expected,observed`)
    directly from sandbox outputs and the 8 feedback pairs. Check that ONLY
    this feedback-split numeric error and rejection data reach the proposer;
    heldout answers and target action sequences must not. Demonstrate
    causal numeric conditioning with identical counts but distinct residuals
    producing different in-memory weight hashes. Reject any assertion that
    this negative-only error-weighted update is full CEGIS or solved repair.
12. Independently reproduce the FOUR preregistered arms: lexical_neural and
    lexical_frozen each use 2 attempts, deterministic fixed_host and seeded
    random_host each 4. Each has 12 aggregate process-CPU-second and 20
    wall-second caps, same disjoint feedback/heldout eight-pair sets per
    attempt, and a pinned finite 16-bit target. Baseline attempts differ
    expressly to spend approximately the same CPU as neural proposals.
    Recalculate total process CPU and wall including every failed/partial
    attempt; independently compute each ratio to the neural arm and require
    absolute ratio deviation <=15% for BOTH CPU and wall before describing
    this *particular run* as approximately matched. If outside tolerance,
    declare NOT MATCHED, not PASS for budget comparison. Such a single seed
    and unequal opportunity counts do not establish neural superiority.
13. Review `test_lexical.py` as well as the original sandbox negative controls,
    and inspect any unaddressed OS security escapes. Scope PASS only to actual
    satisfied acceptance; no novel success, champion promotion or general
    autonomous capability is proven. Preserve all earlier FAIL receipts.
