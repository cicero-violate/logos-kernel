# NC-056: bounded minimum-order and recurrence search

`discover.py` is a standalone, prefix-only symbolic participant. It proposes
time-aware Newton polynomials of degree 0–4, exact AR(1–4), periodic recurrences,
quadratic modular maps with inferred modulus/offset, and a fitted logistic map.
It tests lower description costs first, validates free rollouts on withheld
prefix points, and retains the smallest supported candidate within its budget.
No training, model weights, neural proposal, daemon, or champion modification.

The contract pins code before the evaluator draws three new seeds. `evaluate.py`
generates unseen parameters, 16/30-point prefixes and 8/24-point horizons across
13 families, including aliasing, noise, switching, and chaotic failure cases.
Bubblewrap excludes the workspace, oracle, host environment and network from
the participant. A batch has hard 30-second CPU, 45-second wall, 512-MiB address
space, and 16-MiB output limits. Missing isolation fails closed.

Each arm has the same 128 candidate-check ceiling and 0.20 CPU-second stopping
threshold; actual consumed CPU and verification counts differ and are reported.
The latter is a cooperative check between candidates, not a precise per-arm OS
limit. Fixed interpolation, AR(2), persistence, randomized candidate search, and
zero/no-invariant forecasts are compared with MDL search. Polynomial/AR fitting
work is bounded separately by at most 64 observations, degree/order four and
4096-bit rational inputs; fitting attempts that produce no candidate are not
counted as candidate verification calls. No exact equal-used-compute claim.

Forecast accuracy and structural signature matching are separate. Polynomial
identity against a private known polynomial can be checked exactly; this is not
a proof that an unknown physical process follows that polynomial forever.
Periods found through another equivalent AR representation may forecast well
without matching the deliberately conservative structural signature gate.
Floating forecasts are scored at absolute 1e-6 plus relative 1e-8 tolerance;
exact rational equality is also reported, never inferred from a tolerance pass.

Run tests from the repository root:

```
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s artifacts/NC-056 -p 'test_*.py' -v
PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-056/evaluate.py --output artifacts/NC-056/reviewer_fresh.json
PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-056/evaluate.py --replay-seeds artifacts/NC-056/heldout_law_recovery.json --output artifacts/NC-056/reviewer_replay.json
```

The evaluator refuses changed/uncommitted frozen sources and existing output
paths. IDs vary across runs; numeric fixtures reproduce from disclosed seeds.
The first run is an implementer result, not independent verification. A reviewer
must author the separate declared `independent_verification.json` after code,
oracle, negative-control, fresh-seed, and replay checks. Do not put that receipt
in the implementer manifest's immutable input hash map.
