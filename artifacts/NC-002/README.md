# NC-002 — Complete neural state persistence

## Intent and boundaries
Preserve **full original model weights and topology**, plus new reproducible,
resumable *continuation* state without editing source repos or changing the selected
champion. The source files were inference snapshots: their historical Adam/AdamW
optimizer moments, exact RNG state, scheduler, data cursor and some training progress
were **not saved**. They are unrecoverable from weights. A new optimizer at a clearly
labeled continuation boundary is not a reconstruction of the old training run.

Original components:
- `general_rlt`: current 7-operation RLT general core.
- `carry_gru`: active trained addition specialist.
- `shared_rlt_isa`: experimental single shared attention ISA multiplier,
  not promoted; fixed program ROM and Python host routing.
- `standalone_isa`: standalone trained arithmetic MLP, separate from Transformer.
- `standalone_controller_v2`: separate trained Transformer controller, not a
  jointly trained general core.

## Artifacts

`checkpoint_manifest.json`: frozen inputs, exact source architecture/dependency/
task-ISA/verification receipt hashes, original source checkpoint SHA-256, tensor
counts, vocabulary, and exact fresh-process continuation proofs for five models.

`*_initial.pt`: frozen original learned weights with full model keys, architecture
metadata, vocabulary, no external learned normalization statistics, known source
training progress, newly initialized AdamW optimizer (moment state empty), and
Python/NumPy/PyTorch RNG states. These are non-promotable continuation branches.

`*_continuation_step1.pt`: after one **synthetic continuation proof step**
(non-task training). AdamW moments, exact full model tensors, RNG and progress
are saved; independently verified step two must match uninterrupted execution.
Do **not** treat these as improved trained models.

`reload_receipt.json`: implementer clean-process replay. Separate independent
verifier must author `independent_receipt.json`, validating every claim and negative
test before the Event Lab owner can record PASS.

## Reproduce

From `/workspace/experiments/logos-kernel`:

```bash
PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-002/state_package.py create
PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-002/state_package.py verify
PYTHONDONTWRITEBYTECODE=1 python -m unittest -q artifacts/NC-002/test_state_package.py
python dag.py validate
python -m unittest -q test_dag
```

`create` overwrites **only NC-002 fork packages and its manifest**, so do not rerun
after a verifier signs its hashes. `verify` is read-only for packages, but writes
the implementer's `reload_receipt.json`. The verifier must use its own commands and
independent identity; no self-certification.

## Failure modes
- Refuses missing/mutated model architecture or source weights; verifies all
  60 NC-000 inventory files and hashes additional imported code.
- Strict model tensor schema and vocabulary checks on restore.
- Fails closed if checkpoint optimizer definition or package format changes.
- `torch.load(weights_only=True)` is used for all persisted state; do not accept
  arbitrary pickled classes or untrusted weights.
- GPU RNG included if initialized, but bitwise continuity is qualified only on
  the reproduced CPU/PyTorch/NumPy/Python environment.
- A correct answer from the verifier or fixed ROM does not prove the neural
  kernel independently learned the algorithm.

## File dependencies
`state_package.py` + Python 3 stdlib + PyTorch + NumPy, frozen
`/workspace/experiments/rlt_split_algorithms_ablation_v1` and
`/workspace/experiments/latticeforge_neural_isa_test` model definitions,
`artifacts/NC-000/inventory.json`, `artifacts/NC-000/baseline_receipt.json`,
`artifacts/NC-001/isa_spec.json`, `artifacts/NC-001/independent_verify.json`.
The canonical ISA spec is provenance-only here; models are NOT retroactively
trained on all 15 new opcodes.
