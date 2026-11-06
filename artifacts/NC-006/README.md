# NC-006 — Verified finite neural circuit/operator library

This task freezes and measures the existing NC-005 nine-primitive expanded MLP.
It does NOT retrain, extract algorithms from RLT weights, learn program control,
replace the unified RLT champion, or discover new operators.

## Files / exact dependencies

- verify_circuits.py: Python 3 stdlib, PyTorch, NC-005/curriculum.py
  (ExpandedISA architecture only), NC-005 stage06 frozen rehearsal and
  new_only checkpoints and checkpoint_manifest.json, NC-000/inventory.json.
- circuits.jsonl: 60 versioned neural circuit candidates (6 models x 9 local
  primitives + 1 ZERO16) plus ONE separately marked symbolic NC-001 reference.
  Candidate status is NOT independent verification or champion promotion.
- proof_receipts.jsonl: exact frozen seed/arm/op measurements and failed inputs;
  observed finite-exhaustive scope or explicit FAIL. No symbolic fallback.
- library_manifest.json: source/model integrity hashes, counts, scientific
  boundaries, negative arms, rollback/no-promotion policy.
- verify_branch.py and branch_conformance.json: 65,536 JZ and 65,536 JNZ
  predicate-guided host branch decisions PER seed for 3 rehearsal checkpoints.
  Neural part: a learned 16-bit is-zero predicate. Host part: opcode selection,
  target selection, PC arithmetic, inversion for JNZ, HALT. The same zero
  predicate underlies JZ/JNZ, not three independent learned abilities.
- test_circuits.py: reference truth tests, checkpoint hashes, exact registry,
  full-neural vs symbolic mode, failed new_only retention cases and branch
  boundary tests.
- independent_receipt.json: to be issued only after a different real agent
  independently audits committed frozen models and exact test measurements.
- accepted_library.json: optional release artifact after independent verdict;
  only qualified, proof-scoped neural candidates may be listed here, never the
  disqualified new_only entries, and this does NOT promote a default champion.

## Measured evidence before independent verification

For each rehearsal seed (11,19,29): 9 primitive operators x 64 distinct
binary local feature patterns = 576/576 correct, all 9 learned opcode meaning
labels correct, no source model/weights edited. Three frozen ZERO heads each
perform 65,536/65,536 on exhaustive raw unsigned 16-bit values.

For each new_only seed: 438/576, 428/576, 375/576 on all local cases;
the failing operations and actual counterexamples remain preserved and are
DISQUALIFIED. All three new_only zero heads separately meet the 65536-case
predicate test, but this does not excuse arithmetic forgetting.

JZ/JNZ host integration for every rehearsal seed: 65,536/65,536 JZ and
65,536/65,536 JNZ correct. Inverting the frozen neural is-zero decision
reduces JZ/JNZ to 0/65,536. No learned target/PC/HALT or full control
flow is implied.

All 64 local primitive feature combinations were used during NC-005 training,
so exhaustive local accuracy is NOT evidence of unseen input-feature
generalization. A complete local feature table is NOT proof for arbitrary
register- or program-level executions. Operator bit sequencing and carrying
remain Python host routines. The frozen models are 5,981-param MLPs, not the
selected RLT Transformer. All weights were preserved.

## Reproduction

From /workspace/experiments/logos-kernel:

    PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-006/verify_circuits.py
    PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-006/verify_branch.py
    PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -q -s artifacts/NC-006 -p test_circuits.py
    python dag.py validate
    python -m unittest -q test_dag

No training and no source model checkpoint modifications occur. New source
hashes and checkpoint SHA256 are retained in every candidate and receipt.
No symbolic/lookup answer is substituted for neural logits at evaluation.

Formal neural weight proofs and full-program correctness are NOT claimed;
proof strength is finite-exhaustive observed inference on the exact declared
domain and frozen software environment. An external independent agent must
verify and promote only scoped entries after tests and source hashes pass.

## Independent FAIL and remedy

The FIRST distinct Codex audit of ecdb2dc independently reproduced all
neural results but returned FAIL: failure_examples was truncated to 16 per
failing new_only operator. That omitted 185 of 487 observed failures across
13 affected receipts. No PASS or library release was recorded. The repaired
verifier writes the FULL input/observed/expected trace for every failed
finite local configuration and adds a hard completeness assertion in tests.
The original failed verifier output and trace are retained as evidence, NOT
accepted as a PASS. A second distinct real agent must verify the corrected
commit before any bounded library release.

## Independent bounded release — implemented after verifier PASS

The first independent Codex verifier returned FAIL against the original
implementation because 185 of 487 new-only counterexamples were omitted.
This FAIL is permanently retained in codex_verifier_final.json and
codex_verifier_trace.jsonl; it has not been converted into a PASS.

The implementer corrected all failure trace omissions: 487/487 preserved.
A second distinct real Codex verifier independently reloaded all six frozen
models, checked each failure set and returned PASS against commit 611a18f.
Its identity comes from the actual codex_verifier_v2_trace.jsonl thread.started
runtime event, not a fabricated worker ID.

- codex_verifier_v2_final.json / trace.jsonl: raw independent PASS
- independent_receipt.json: implementer-assembled hash-bound provenance
- library_manifest.json: independently verified bounded release, not champion
- accepted_library.json: only 30 rehearsal neural candidates (27 one-bit
  primitives + 3 16-bit is-zero predicates)
- 30 new-only controls remain excluded; 487 failures remain visible
- The NC-001 symbolic reference is separately marked and never called neural.

Proof scope: each one-bit primitive passes all 64 locally training-seen
six-bit feature combinations. Each accepted is-zero head passes all 65,536
raw 16-bit values. Host-controlled JZ/JNZ based on that zero head also pass
65,536 inputs per seed, but opcode selection, targets, PC, HALT, bit indexing,
register wiring and the fixed multiplication ROM are NOT learned.

Only a separate Codex closure verifier may approve the later committed
closure state. Then the implementer records DAG PASS and pushes origin/main
with a normal non-forced fast-forward push.

The implementation now refuses to overwrite an independently verified
library_manifest.json: the verify_circuits.py generator raises
NC006_VERIFIED_RELEASE_IMMUTABLE. Reproduce raw generation in an isolated
copy, not on top of the accepted release.

Dependencies: Python 3, Git, PyTorch, frozen NC005 ExpandedISA checkpoint
packages (5,981-parameter MLP), NC001 canonical contract and NC000 champion
inventory. No original RLT Transformer retraining or promotion is involved.
