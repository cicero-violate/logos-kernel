# NC-007 — Bounded supervised neural action composition

Status: implementer-tested; independent scientific audit and closure required.

## Method

A new autoregressive MacroController receives a known formal macro goal ID,
position and previous predicted action token, then predicts each next action.
It is trained separately at seeds 11, 19 and 29 using 11 explicitly supplied
target recipes. Training does not include numeric operand values.

During evaluation, the model's generated tokens route execution through
the accepted frozen NC-005/NC-006 neural MLP primitive logits. A separate
NC-001 symbolic AST evaluates expected answers only, never substitutes
for neural predictions. Host code supplies register routing, constant K=1,
bit iteration, carry, HALT decoding, and control flow.

The supervised recipes are NOT invented operations. There is no ability
demonstrated on unseen macro IDs, unsupervised composition, arbitrary tasks,
or a unified RLT Transformer. The local neural primitives were already
trained on all 64 local feature combinations.

## Outputs

- composition_traces.jsonl: each seed/macro; predicted sequence, exact
  source circuit IDs, bounded correctness, counterexamples, run timing,
  program length and acyclic data-dependency DAG.
- composition_receipt.json: full input/checkpoint hashes, training update
  counts and correct/test denominators, reuse and hard limits.
- checkpoints/controller_seed*.pt: full separate controller weights,
  topology (pinned source), optimizer and Python/NumPy/Torch RNG states.
- HOST_LOOP_TRIPLE: separate explicitly host-controlled cyclic runtime
  CFG; neural ADD executes, but program counter, branch and repeat are host.

## Reproduce (before independent evidence is frozen)

    cd /workspace/experiments/logos-kernel
    PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-007/compose.py
    PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -q -s artifacts/NC-007 -p test_composition.py
    python dag.py validate
    python -m unittest -q test_dag

Running compose.py retrains task-local controllers and replaces task-local
checkpoint/trace output. Do NOT rerun after independent verification.

## Code dependencies

- compose.py: Python 3, NumPy, PyTorch, NC-005/curriculum.py,
  three frozen rehearsal stage06 checkpoints, NC-005/checkpoint_manifest.json,
  NC-006/accepted_library.json and library_manifest.json, NC-001/canonical_isa.py,
  original frozen RLT champion used for hash validation only.
- test_composition.py: compose.py and all its dependencies, plus task-local
  controller checkpoints and the two generated JSON/JSONL evidence files.
- DAG mutation: dag.py, instructions.jsonl, test_dag.py, Git.

## Limits

Bounded 16-bit operand samples are not exhaustive 2^32 operand-pair proof.
This does not demonstrate learned PC/HALT, runtime loops, discovered programs,
general reasoning or new-language competence. Do not promote the champion.
