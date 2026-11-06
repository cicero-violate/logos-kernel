# NC-029 — CPU-only compute and correctness baseline

**Boundary:** implementer measurement only, pending independent verifier. Not a new trained model, no architectural superiority claim, no champion promotion.

## Files and dependencies

- `bench_cpu.py`: Python >=3.10, Linux `resource`, PyTorch, Git; reads the original **unchanged** `/workspace/experiments/rlt_split_algorithms_ablation_v1/{config.json,rlt_core.py,task_conditioned_rlt.py,algorithms/carry_recurrence/module.py,snapshots/current_unified_v3.pt}` and exact two seed11 checkpoints selected by `config.json`.
- `measurement_protocol.json`: frozen 128-pair 8-bit addition task, CPU measurements, 3 separate-process trials, 5 timed repeats/trial and no network/API use.
- `test_bench_cpu.py`: independent-of-reported-results semantic check of 3 exact carry implementations, deterministic tasks, SHA-256 freeze and corrupted-answer negative control.
- `cpu_baseline.json`: measured per-run timings, exact accuracy denominators, example failures, source/protocol hashes and within-class Pareto frontiers.
- `CANDIDATE_MANIFEST.json`: pinned evidence, source hashes, reviewer reproduction instructions. `independent_verification.json` is **not** provided by the implementer.

## Reproduce without modifying upstream

From `/workspace/experiments/logos-kernel`:

```bash
python dag.py validate
python -m unittest -q test_dag
python -m unittest -q test_pull
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest -q artifacts/NC-029/test_bench_cpu.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python artifacts/NC-029/bench_cpu.py --run --output artifacts/NC-029/replay_local.json
```

The last command makes a **new local timing observation**, not byte-identical timing results. The generated `replay_local.json` should be compared with `cpu_baseline.json` for the dataset SHA, provenance SHA, denominators, exact control accuracy, neural predictions/counterexamples and plausible timing variation. Do not call different thermal/scheduler conditions a deterministic timing mismatch.

## Scope and failure modes

- The three **analytic controls** all implement known addition. Dense: 16-state GEMM transition table. Sparse: local two-state recurrence. Scalar: bitwise carry recurrence. Their correctness does **not** show neural discovery. The dense GEMM toy is **not** a Transformer.
- The trained RLT is a 25,089-parameter multitask Transformer; the deployed addition champion is a separate GRU; neither was trained during this benchmark. Show neural performance separately from hardcoded controls.
- Reference truth is used only after prediction. Model forward sees operands and operation ID, never the target answer. All three benchmark trials reuse the **same 128 pairs**; 3 trials are timing replication, not independently sampled evaluation data.
- Report full CPU seconds, not GPU speed, MAC estimates or unmeasured wattage. Energy entries remain **null**; `ru_maxrss` includes interpreter and PyTorch, not just tensors. No external CPU isolation; measured rankings can change under load.
- Original champion `snapshots/current_unified_v3.pt` is pinned SHA-256 `4c42d13fd46f57ac0db6eccbaef35e99c4cefcec393a3a4faf146645e8f683`; no upstream file writes, model updates or service changes.

**Reviewer:** independently inspect source/inputs, rerun a fresh local sample and negative controls, report any hash deviation/failure and precise scope, author your own `artifacts/NC-029/independent_verification.json` receipt if eligible, and record `review-result` with actual separate reviewer ID. Do not let implementer sign for reviewer.
