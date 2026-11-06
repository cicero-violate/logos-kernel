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
