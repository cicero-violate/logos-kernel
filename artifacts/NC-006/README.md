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
