# NC-046 — CPU-local trainer worker (isolated canary; production blocked)

## What exists

- `trainer_worker.py`: fixed 25,089-parameter RLT model loaded only from the
  hash-pinned immutable champion; 3-seed/3-arm deterministic short CPU run.
  Typed, pinned synthetic NC-022 data with train/heldout unordered-pair exclusion,
  one-thread CPU execution, process CPU RLIMIT, and no network or model API.
- Durable model + AdamW optimizer + Python/NumPy/PyTorch RNG + step cursor
  checkpoints, immutable SHA-addressed filenames, parent checksum lineage and
  nonblocking local `flock`. Crash before a complete link leaves no acknowledged
  step; retry may recompute that *unacknowledged* update.
- `test_trainer_worker.py`: two interrupted/restarted trials, exact model
  state/loss/prediction comparison, idempotency, tamper, leakage, and lock controls.
- `run_pilot.py`: 3 seeds (11/19/29), new-only/rehearsal/blocked-replay,
  four actual gradient updates and 192 examples per seed/arm. Read-only
  NC-045 trainer status checked; no Trainer DAG claims/ledger writes.
- `audit_pilot.py`: replay all 36 candidate checkpoints, two restart pairs,
  source and champion SHA checks; implementer evidence only.
- `PROTOCOL.json`, `trainer_pipeline.json`, `restart_replay_receipt.json`:
  preregistered pilot, quantitative NO_GAIN results, bounded recovery receipt.
- `MANIFEST.json`: source/data/weights/run hashes, restrictions and review scope.
  Independent reviewer must author `independent_verification.json` separately.

## Dependencies (no implicit files)

- Python 3.10+ (stdlib `fcntl`, `resource`, `hashlib`, `json`,
  `tempfile`, `subprocess`, `unittest`), Linux, Git, PyTorch CPU, NumPy.
- `/workspace/experiments/logos-kernel/artifacts/NC-022/`:
  `rlt_rehearsal.py`, `PROTOCOL.json`, `train_and_holdout_ledger.json`,
  `independent_receipt.json`; these data are read-only.
- `/workspace/experiments/logos-kernel/artifacts/NC-000/inventory.json`.
- `/workspace/experiments/rlt_split_algorithms_ablation_v1/`:
  `task_conditioned_rlt.py`, `rlt_core.py`,
  `snapshots/current_unified_v3.pt` (SHA-256 `4c42d13f...`).
- `/workspace/experiments/logos-trainer/trainer_dag.py`,
  `training.jsonl` (read-only `status`, not episode ownership).

## Reproduce

```bash
cd /workspace/experiments/logos-kernel
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s artifacts/NC-046 -p test_trainer_worker.py -q
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python artifacts/NC-046/audit_pilot.py
python dag.py validate
python -m unittest -q test_dag
python -m unittest -q test_pull
```

Rerunning full tests on an existing test prefix refuses intentional overwrites:
run in a fresh isolated clone or use unique run IDs. The evidence files are
replayable without overwriting the RLT champion.

## Admission boundary and unmet acceptance

**The real trainer task TR-002 is unapproved, has placeholder source/data pins
and depends on the unsealed canary TR-001.** Also NC-045's ledger has no atomic
running-job liveness state, so an expired lease can be reassigned even with a
suspended live worker. Consequently `--mode production` must fail closed.
This ticket DOES NOT install or start a resident worker, claim TR-002, register
an actual trainer checkpoint, publish a learned model, or promote the champion.
The NC-046 isolated canary is *engineering evidence*, not an approved
continuous-learning deployment.

Multi-seed matched-resource pilot result: `NO_GAIN`; all widths/operations
remain exactly within the frozen NC-022 task families. Unseen mathematical
discovery, causal transfer, energy efficiency, and generalized reasoning are
**not** shown. Absolute CPU timings are host-load dependent. Training
outputs here have no independently signed scientific verdict.

Before final review/closure: obtain explicitly authorized trainer tasks with
nonplaceholder source/data hashes, make worker liveness and job ownership
atomic with trainer ledger transitions, integrate guarded candidate checkpoint
registration/recovery, and independently audit the expanded real episode.
Never silently waive these gates or mark training active because a tab exists.
