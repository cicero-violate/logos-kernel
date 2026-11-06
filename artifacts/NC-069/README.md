# NC-069: predicting interventions on the ISA machine

Tests whether a small CPU model trained on interventions can predict the outcome of interventions it has not seen,
using `artifacts/NC-001/canonical_isa.py` as an exact oracle. This is implementer evidence, not independent verification.

- `causal_benchmark.py` defines four interventions (change an input, set a register, skip a step, replace an instruction)
  by rewriting the program or inputs and running the canonical interpreter.
- `train_eval.py` trains the arms and writes `results.json` and one checkpoint per arm and seed.
- `make_manifest.py` writes `results_summary.json`, `intervention_benchmark.json` and `CANDIDATE_MANIFEST.json`.
- `test_causal_benchmark.py` checks the oracle against a separately written evaluator and checks the splits.

## Arms

All arms use one architecture with equal samples and steps.

| Arm | Trained on | Tested on |
|---|---|---|
| `interventional` | Intervened and plain runs | Intervention tokens |
| `passive_tokens` | Plain runs only | Intervention tokens it never trained on |
| `passive_rewrite` | Plain runs only | The program already rewritten, so it is given the rule |
| `shuffled_control` | Intervention tokens shuffled across each batch | True intervention tokens |
| `leaked_control` | The answer as an input | The answer as an input |

`heldout_combo` uses unseen programs and unseen (intervention, opcode) combinations, `heldout_program` uses unseen programs
with combinations seen in training, and `factual` uses unseen programs with no intervention. Effect accuracy counts only cases
where the intervention changed the outcome, which is where the evidence is, since copying the plain run answers the rest.

## Commands

    PYTHONDONTWRITEBYTECODE=1 python -m unittest -q artifacts/NC-069/test_causal_benchmark.py
    PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-069/train_eval.py --steps 2500
    PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-069/make_manifest.py "$AGENT_ID"

Dependencies: Python 3, PyTorch (CPU) and `canonical_isa.py`. The stated limits are in `CANDIDATE_MANIFEST.json`.
