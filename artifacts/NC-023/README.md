# NC-023 — Typed symbolic rewrite and verified equivalence-class baseline

**Scope:** This is a deterministic, bounded, reference-only symbolic mathematical optimizer. It is **not** a trained model, learned program invention, full e-graph implementation, proof of globally optimal expressions, general intelligence, hardware speedup, or upstream model promotion.

## Files and dependencies

- `symbolic_search.py` — Python 3 stdlib; imports local `../NC-020/benchmark.py`, transitively `../NC-001/canonical_isa.py`. The two existing upstream files remain unmodified.
- `run_baseline.py` — Python 3 stdlib, `symbolic_search.py`, existing NC-020 dataset `equivalence_benchmark.jsonl` and `splits_and_semantics.json`.
- `test_symbolic_search.py` — Python 3 stdlib, `symbolic_search.py`, existing NC-020 benchmark and dataset.
- `rewrite_rules.json` — handwritten rule list, proof admission rule and pinned sources.
- `equality_saturation_baseline.json` — per-split results, operation-count comparators and documented limitations.
- `equivalence_receipts.jsonl` — one row per case with certified whole-expression proof, accepted step provenance, counterexamples on invalid cases, exploration counts, and Python interpreter latency.
- `MANIFEST.json` — proposed candidate source/data/output hashes and reproducibility instructions. This is an **implementer** artifact, not independent review.

## Reproduce

Run from `/workspace/experiments/logos-kernel`:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s artifacts/NC-023 -p test_symbolic_search.py -v
PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-023/run_baseline.py
python dag.py validate
python -m unittest -q test_dag test_pull
```

## Mechanism

- Enumerate explicitly scoped integer and bitvector identities.
- Generate bounded rooted and nested rewrites, canonicalize AST identity, and retain only variants accepted by an exact whole-expression equivalence check.
- For integers: coefficient-map normalization over `Z[x,y,z]`; applies only to finite `+,-,*` polynomials with unbounded exact integers.
- For bitvectors: NC-001 strict type/width/sign validation and **all** X,Y raw bitpatterns for fixed 3/4/5-bit widths; output bits only, no flags, PC, register mutation or HALT.
- No unsafe floating-point reassociation: float negative controls remain explicitly rejected.
- Saturation: bounded verified equivalence-class graph search (32 states), **not** congruence-closure e-graph. Greedy first-improvement is the matched deterministic control.
- Static cost: number of unique operator subexpressions as ideal DAG/CSE estimate, separate from unshared tree operator count and timed Python interpreter path. No direct hardware performance inference.
- All case IDs, source hashes, data hashes, and family splits preserved from NC-020. External benchmark files are never regenerated or edited by this ticket.

## Proof and scientific boundaries

Proof receipts are candidate-side and reuse NC-020's canonical oracle. A separate independent reviewer must reimplement/recheck sample proof steps and negative controls before PASS. Evaluation case families are structurally disjoint **within NC-020**, but historical pretraining overlap is unproven. A successful handwritten rule is a **symbolic control**, not neural discovery. If greedy matches or outperforms bounded saturation, that result is reported without relabeling.

The original RLT champion and candidate checkpoints are never loaded or changed.
