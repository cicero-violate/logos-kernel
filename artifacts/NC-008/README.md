# NC-008 — bounded Transformer proposals, independent review pending

A **2-layer Transformer encoder** reads 20 task I/O examples and the previous
program token, then proposes an action sequence. The target operation is
specified by input/output behavior, **not a provided target ISA program**.
A local symbolic oracle ranks each proposed program; policy updates imitate
only those proposals. The final candidate is the model's *greedy decoded*
program, **not** an externally selected ranked sample. The program is then
executed by the preexisting hash-verified **NC-005/006 frozen neural ISA**.

Acceptance scope: INC, DOUBLE, Boolean composite, shift composite each
256/256 on disjoint held-out 16-bit operand pairs (1,024/1,024 total).
The 12 zero-shot transfer evaluations on three other goal contracts had
0/12 successes: **no unseen-task generalization**. Uniform random search
found solutions to all four original goals within the same budget; no
statistical evidence the neural proposer is superior to simple random search.
This is verifier-reward-trained neural-guided program search, **not new
primitive invention, original mathematical discovery, or AGI**.

## Reproduction

    cd /workspace/experiments/logos-kernel
    PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -q -s artifacts/NC-008 -p test_invent.py
    python dag.py validate
    python -m unittest -q test_dag
    python -m unittest -q test_pull

The experiment training source can reproduce outputs with:

    PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-008/invent.py
    PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-008/unseen_probe.py

**Do not rerun training after a candidate is pinned for review**: it
rewrites task-local checkpoints and evidence and invalidates hashes.

## Files and dependencies

- invent.py: Python 3, PyTorch, NumPy, artifacts/NC-007/compose.py;
  transitively artifacts/NC-005/curriculum.py + frozen stage06 checkpoint,
  artifacts/NC-006/accepted_library.json + manifest,
  artifacts/NC-001/canonical_isa.py, and original RLT champion hash gate.
- test_invent.py: invent.py and the four saved policy checkpoints,
  proposals.jsonl, invention_receipt.json, the same frozen ISA assets.
- unseen_probe.py: invent.py, four policy checkpoints, frozen ISA imports.
- Code, negative controls, policy checkpoints and manifests are committed
  under artifacts/NC-008 only; no upstream champion changed.

All scoring/pretraining use actual finite example oracles: no proof for
unbounded inputs. Local model identity is not cryptographic attestation.
