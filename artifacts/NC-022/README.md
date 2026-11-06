# NC-022 — Matched-exposure rehearsal on the real 25,089-parameter RLT

State: experimental candidates, independent scientific verification required
before any DAG PASS, promotion or Git push.

## Model and protocol

The model is the actual shared TaskConditionedRLT, with causal encoder,
recurrent decoder, cross-attention and seven learned operation embeddings.
Frozen champion snapshot and original code are SHA-256-pinned. Each seed
starts from the corresponding pretrained model and separately trains
three task-local copies with 144 gradient updates each.

The preregistered protocol is PROTOCOL.json, committed before training:
- new_only: new curriculum-stage tasks exclusively.
- rehearsal: interleaved old and new tasks.
- blocked_replay: EXACTLY the same per-stage, per-width, per-operand-pair,
  per-operation examples as rehearsal, presented in a different order.

All arms share the original weights, data, number of examples and
updates, optimizer, stage order, learning rate and source semantics.
Stage order: original baseline, add/xor/sub, mul/lt, parity_and/compound,
old-skill restoration. These tasks already existed in the pretrained
RLT. Novel means reintroduced in THIS curriculum, not newly invented.

## Files and dependencies

- rlt_rehearsal.py: Python 3 stdlib, PyTorch, NumPy, the immutable
  /workspace/experiments/rlt_split_algorithms_ablation_v1/
  task_conditioned_rlt.py, rlt_core.py, snapshots/current_unified_v3.pt,
  and artifacts/NC-000/inventory.json.
- PROTOCOL.json: reproducible static training/evaluation specification.
- train_and_holdout_ledger.json: deterministic training pairs, action labels,
  disjoint heldouts, and exposed source dataset hashes.
- checkpoints/: full trained 25,089-param model, AdamW optimizer moments,
  torch/python/numpy RNG states, stage/seed/arm and source hashes.
- eval/: raw paired operands, neural predictions and independent arithmetic
  target outputs for all seven tasks and three host-wired composition probes.
- rlt_retention_matrix.json: exact heldout measures, baseline and previous-stage
  regression counts, task-specific learning and host-supplied composition probes.
- exposure_matched_ablation.json: per-task exposure and exact rehearsal-vs-blocked
  equality, sample budgets, paired gains or NO_GAIN.
- checkpoint_manifest.json: all stage/source/checkpoint and evaluation hashes.
- test_rlt_retention.py: loaded checkpoints, hashes, raw correctness, data
  disjointness, exposure equality and result recomputation.
- DAG operations: dag.py, test_dag.py, instructions.jsonl and Git.

## Reproduction (not after verified release)

    cd /workspace/experiments/logos-kernel
    PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-022/rlt_rehearsal.py
    PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -q -s artifacts/NC-022 -p test_rlt_retention.py
    python dag.py validate
    python -m unittest -q test_dag

The training command replaces only task-local candidate checkpoint and
evidence outputs. Never rerun over independently verified releases.

## Hard limits

The original model already trained on all seven task IDs; historical
pretraining independence is NOT established. Heldout cases are a new
generated split and width32 examples are extrapolation. Host-programmed
composition probes have fixed action routing and do not prove learned
program invention, branching, loops, language, AGI or general-purpose
reasoning. Neural outputs are scored against an external exact oracle;
the reference never replaces model predictions. If no consistent matched
three-seed gain is found, report NO_GAIN, preserving regressions.
