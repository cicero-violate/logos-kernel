# NC-004 — Causal neural ISA-to-math semantics, bounded five-primitive experiment

## Exact claim

A new matched experiment trained 3 seeds × 2 arms for 1,500 updates each.
The **coupled** arm predicts a five-class math meaning distribution and sends
that distribution to its neural bit-transition network. The **detached** arm
predicts identical meaning labels but sends the raw instruction one-hot to
arithmetic instead. Both arms have identical tensors, architecture size,
random initial weights per seed, training examples, and update count.

Changing only the predicted math distribution changes numeric outputs **only
in the coupled model**. The numeric loss has a nonzero gradient to its
meaning classifier. This is a causal neural semantic channel—not a detached
label head and not a claim that its canonical AST interpreter executes neural
mathematics.

## Implementation boundaries

- Learned: five primitive names/meaning classes `ADD, SHL, SHR, AND1, OR`;
  16-bit numerical bit transitions from neural logits, with 16-bit carry.
- Deterministic host: render the **predicted meaning class** into the
  `logos.isa/1` typed AST using NC-001's contract, register wiring,
  flag updates, PC and HALT, and pre-specified two-instruction composition.
- Oracle: NC-001 `canonical_isa.reference` runs **after** the predicted
  result and never replaces it. Compositions are provided as fixed
  programs, **not** invented by the model.
- Neither arm learned 10 other NC-001 opcodes (SUB, XOR, AND, SAR, MOV,
  LOADI, JZ, JNZ, JMP, HALT). The original trained RLT champion is not
  updated, fused or promoted.

## Evidence

- `checkpoint.json`: six training checkpoint hashes, seed/config/update
  provenance, training phase outcomes, parameter count, source dependencies.
- `coupled_seed{11,19,29}.pt` and `detached_seed{11,19,29}.pt`:
  full neural tensors, optimizer moments, RNG state and step/provenance.
- `semantic_receipt.json`: 320/320 trained local truth patterns and
  320/320 math labels for **each seed and arm**, numeric loss gradient,
  controlled semantic intervention, canonical AST classes, three fixed
  two-instruction compositions (256 exhaustive four-bit operand pairs +
  512 sampled eight-bit operand pairs **per composition**).
- `state_conformance.json`: 5 operations × 256 operand pairs × 6 models
  = 7,680 full host-assisted (neural R + explicit host PC/flags/HALT) state
  checks against the NC-001 reference.
- `test_semantics.py`: ten tests covering provenance, causal gradients,
  semantic intervention, typed AST, actual checkpoints and limits.

The 320 local truth patterns **all appear in training**. Composition tests
use new sequence arrangements, but they do not prove generalization to
unseen primitive local bit patterns. The 512 sampled eight-bit pairs are
not exhaustive 65,536-pair coverage. Neither arm outperforms the other on
tested composed-answer accuracy. Semantic causality is established on the
bounded set; semantic optimality or general intelligence is not.

## Reproduction

From `/workspace/experiments/logos-kernel` (with Python 3, PyTorch, NumPy,
and the frozen source repos present):

```bash
PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-004/causal_semantics.py train
PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-004/causal_semantics.py evaluate
PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-004/state_conformance.py
PYTHONDONTWRITEBYTECODE=1 python -m unittest -q artifacts/NC-004/test_semantics.py
python dag.py validate
python -m unittest -q test_dag
```

Training **replaces only** the six local experimental files, checkpoint.json
and never source/champion checkpoints. Do not rerun after independent
verification until preserving the old evidence hashes.

## Exact code dependencies

- `causal_semantics.py`: standard-library Python, PyTorch, NumPy,
  `/workspace/experiments/latticeforge_neural_isa_test/train_causal.py`,
  `artifacts/NC-001/canonical_isa.py`, `artifacts/NC-000/inventory.json`,
  frozen `/workspace/experiments/rlt_split_algorithms_ablation_v1/snapshots/current_unified_v3.pt`.
- `state_conformance.py`: `causal_semantics.py` plus six saved .pt model
  files and the same NC-001 interpreter.
- `test_semantics.py`: `causal_semantics.py`, the six model checkpoints,
  `checkpoint.json`, `semantic_receipt.json`, Python unittest.
- DAG: `dag.py`, `instructions.jsonl`, `test_dag.py`, standard library
  and Git; not automatically executed by this training script.

## Verification and promotion

The implementer's self-tests are not an independent signature. A different
real agent must independently re-run and inspect the gradients/intervention,
model tensor equality at matched seeds, semantics correctness,
two-instruction composition, frozen champion SHA and failure attribution.
Only after writing `independent_receipt.json` and confirming distinct
runtime identity can the current real Event Lab owner append NC-004 PASS.
No default champion promotion is authorized.
