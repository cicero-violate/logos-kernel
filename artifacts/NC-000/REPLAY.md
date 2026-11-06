# NC-000 — reproducible baseline replay and independent-verifier handoff

## Source of truth
- `AGENTS.md`, `instructions.jsonl`, `dag.py`
- Implementer inventory: `artifacts/NC-000/inventory.json`
- Implementer first-pass observations: `artifacts/NC-000/implementer_smoke.json`
- Distinct-agent verifier must author: `artifacts/NC-000/baseline_receipt.json` (**not yet created**)

## Read-only replay commands

```bash
cd /workspace/experiments/logos-kernel
python dag.py validate
python -m unittest -q test_dag
PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-000/replay_baseline.py --seeds 11,19,29 --random8 128

# Deeper deterministic coverage: 4096 random 8-bit pairs + 121 edge-case pairs
PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-000/replay_baseline.py --seeds 11,19,29 --random8 4096 --output artifacts/NC-000/replay_full4217.json
```

The 128-pair replay measures 249 eight-bit cases per seed (128 deterministic random + 121 fixed edge cases), **not** 4,217 or 65,536. Exhaustive four-bit coverage is all 256 operand pairs per seed. The deeper option measures the historical 4,217-case dataset per seed; because the random stream is identical between seeds, it is not three independent data samples. No 65,536-pair claim is made.

## Code dependencies

| Component | Exact code and artifact dependencies |
|---|---|
| DAG | `dag.py`, `instructions.jsonl`, `test_dag.py`, Python 3 standard library |
| Replay orchestration | `replay_baseline.py`, `inventory.json`, Python 3, Git, PyTorch |
| Selected champion | `/workspace/experiments/rlt_split_algorithms_ablation_v1/run.py`, `system.py`, `task_conditioned_rlt.py`, `config.json`, source checkpoints |
| Standalone ISA | `/workspace/experiments/latticeforge_neural_isa_test/train_causal.py`, `transformer_controller/train_controller.py`, `causal/weights.pt`, v1/v2 controller checkpoints |
| Shared-attention ISA | `neural_isa_transfer_v1/unified_shared_v1/verify_shared.py`, `model.py`, `outputs/seed*/jointly_tuned.pt`, `neural_isa_transfer_v1/verify_hybrid_independent.py`, `rlt_micro_checkpoints/`, original RLT checkpoint and copied vendor code |

The replay writes only its selected JSON output under `artifacts/NC-000/`. It does not retrain models or modify source checkpoints.

## Verification boundary

The replay is **implementer-driven evidence**, not independent verification. A **different, real agent** must inspect the implementation, recompute and compare hashes, run its own fresh-process tests and write `baseline_receipt.json` with exact commands, inputs, denominators, checkpoint hashes, failures, runtime boundaries and limitations. Do not treat a PASS from this script as independent verification. The DAG's PASS transition also requires the current claim owner to supply the distinct verifier's actual identity.

The original RLT seven-task general core has weak multiplication; the separate ISA specialist uses a **predefined** eight-op program and Python register routing. The shared ISA model uses one attention core but is experimental. Old standalone controller v1 is expected to fail some halted rollouts; corrected v2 is the qualified checkpoint. A numerically correct but unhalted rollout is a **failure**.
