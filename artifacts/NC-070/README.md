# NC-070 — sequential causal graph discovery from ISA interventions

This is an IMPLEMENTER candidate awaiting independent review. No champion or upstream model is modified. The scorer has access to true graphs only after each episode; run_episode(query, seed, episode_nonce, arm, rounds) receives an opaque oracle callback and no graph/program/graph index.

## Exactly what is tested

- 64 labeled rooted trees on four canonical registers X,Y,R,T, with known copy structural equations. Each graph is compiled to a valid 4-bit, forward-only MOV ISA program.
- Observationally all graphs generate 0000 or 1111 with probability 0.5 each. Interventions do(register := 1-U) can distinguish descendants and orientations. The explicit observational-equivalence witness is in discovery_results.json.
- Heldout evaluation uses 14 SHA-256-split graph/programs; every policy has all 64 possible graph hypotheses but does not get the actual heldout target. Finite version-space identification is NOT generalization to unseen graph families or learned causal discovery.
- At each step, the active learner chooses an unused register that minimizes expected surviving version-space size, observes the canonical ISA oracle's response, and eliminates inconsistent models.
- The random arm chooses unused registers uniformly; the passive arm repeats factual observations. All arms have the same exogenous bits, oracle-call count, and the same full 64 × 4 candidate-prediction workload per step. Actual CPU is measured, not assumed equal.
- A separately written graph simulator is tested against the independently passed NC-069 canonical ISA oracle over all 64 graphs × 4 actions × 2 exogenous values = 512 interventions.

## Frozen result — five seeds (11, 19, 29, 31, 43)

Three interventions per case; 14 heldout graphs per seed; 70 cases and 210 oracle calls per arm.

| Arm | Exact graph recoveries | Uniquely identified |
| --- | ---: | ---: |
| Active | 52/70 (74.3%) | 45/70 |
| Random | 42/70 (60.0%) | 32/70 |
| Passive | 1/70 (1.4%) | 0/70 |

Active minus random = +14.29 percentage points; graph-clustered paired 95% bootstrap CI [-14.29, +42.86] points. The interval includes zero: NO_CONFIRMED_GAIN over random. It would be incorrect to claim a reliably superior discovery policy from these 14 graphs.

Aggregate measured CPU: active 0.38758 s, random 0.38577 s, passive 0.36584 s (exact figures in frozen JSON). Active/random CPU ratio 1.0047, active/passive 1.0594, within the prespecified ±20% budget-parity gate. Per-seed CPU figures and exact denominators are included in the JSON. CPU excludes generating the candidate universe and compiling the ISA programs; both are common preparation costs.

Experiment-count curve at budgets 0/1/2/3 is saved to budget_sweep.json; exact recoveries are:
- Active: 1 / 8 / 26 / 52 of 70
- Random: 1 / 5 / 20 / 42 of 70
- Passive: 1 / 1 / 1 / 1 of 70

## Reproduce

From repository root, no network or GPU required:

    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest -q artifacts/NC-070/test_discovery.py artifacts/NC-070/test_evidence.py
    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python artifacts/NC-070/discovery.py --seeds 11 19 29 31 43 --rounds 3 --out /tmp/nc070-review
    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python artifacts/NC-070/recompute.py artifacts/NC-070
    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python artifacts/NC-070/run_budget_sweep.py --out /tmp/nc070-budget-review.json

The independent verifier should inspect the opaque-query boundary and inspect the oracle's compiled true program separately. recompute.py regenerates every outcome, action, graph guess and graph-bootstrap CI, excluding variable CPU timings. It fails closed on mutated truth labels or observations. Mechanical replay is NOT an independent agent's scientific review.

## Dependencies, assumptions and limits

- Python 3 standard library, Git, artifacts/NC-069/causal_benchmark.py (intervention mapping and exact machine oracle), and artifacts/NC-001/canonical_isa.py. No NumPy or PyTorch is directly required by this experiment.
- The true family is a finite class of deterministic identity-copy trees. This benchmark is constructed so passive observations cannot distinguish models; it does not model hidden confounders, noisy mechanisms, arbitrary programs, unknown operators or general DAGs.
- All 64 hypotheses are supplied, including the heldout true structures, though the true identity is hidden. No new out-of-distribution structure learning is shown.
- Graph-clustered CI resamples 14 graphs, retaining five seed replicates. Positive active-over-random point estimate is inconclusive.
- Controls receive the same counterfactual preparation work even if they do not USE active information. This is explicitly budget-matched information control, not a claim random/passive methods normally need this computation.
- CPU times are tiny local process-CPU measurements under shared host load; 20% tolerance is an acceptance diagnostic, not portable performance.
- Local Git provenance does not authenticate agent identity. No claims of AGI, unrestricted world-model learning, new mathematics or champion promotion.
