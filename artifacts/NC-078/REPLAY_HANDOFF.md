# NC-078 — checkpoint-count replay, incomplete

This is an IMPLEMENTER rework aid, not independent PASS evidence. The original candidate and independent HOLD receipt remain unchanged.

The acceptance gate requires checkpoint-derived solve-count replay for neural, shuffled, and untrained arms over 3 seeds × 56 goals × 3 arms = 504 sequential goal-arm evaluations.

Run from the repository root, with Python 3, NumPy, PyTorch CPU, and Git:

    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 python -m unittest -q artifacts/NC-078/test_replay_checkpoints.py
    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 python artifacts/NC-078/replay_checkpoints.py --seed 11 --out /tmp/nc078-check-11
    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 python artifacts/NC-078/replay_checkpoints.py --seed 19 --out /tmp/nc078-check-19
    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 python artifacts/NC-078/replay_checkpoints.py --seed 29 --out /tmp/nc078-check-29

Each invocation resumes after its last durably recorded goal, checking source, model, result, canonical interpreter, goal, and record hashes. The optional --max-goals N bounds progress to N total processed goal-arm cases of a seed's 168. Resume was tested on seed 11 across two processes: first 4 goals, then 2 goals. All six matched the frozen archived results exactly, including positive programs and candidate counts.

The model sampling stream is recreated by Torch manual seed and two deterministic model initializations, loading the pinned trained weights. The original training consumes NumPy RNG for examples and minibatches and uses zero Transformer dropout. Accepted programs are rechecked by the canonical interpreter over all 256 four-bit inputs. Checkpoint training is not repeated.

**Limit: 6/504 goal-arm cases reproduced.** No completed 56-goal arm or 3-seed result; no verified full count; no candidate republished. A distinct reviewer must independently complete and audit all 504 cases before acceptance. Original HOLD remains appropriate. Host CPU contention may raise runtime substantially.

Source dependencies: replay_checkpoints.py, synth.py, train_synth.py, NC-001/canonical_isa.py. Data dependencies: model_seed{11,19,29}.pt, results.json, proposals.jsonl, CANDIDATE_MANIFEST.json. No champion change.
