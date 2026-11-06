# NC-020 — Typed mathematical semantic-equivalence benchmark

**Status:** Implementation, pending independent Codex audit before DAG PASS.

The three generated outputs are:

- equivalence_benchmark.jsonl: typed AST pairs, split, family, case ID, true label, proof scope, counterexamples, record hash.
- splits_and_semantics.json: deterministic seed, versioned family registry, source SHA-256, case ID sets and leakage caveats.
- proof_audit.json: per-scope coverage, negative controls, challenge complexity and receipt status.

## Proof-scope boundaries

**Exact integers**: polynomial-only Z[x,y,z] with +, -, *. Canonical
coefficient forms establish identities over *all* integer assignments.
An explicit integer assignment is sufficient to refute an identity. No claims
for general division, exponentiation, inequalities or transcendental functions.

**Bitvectors**: NC-001 strict AST validator/evaluator; all X,Y operand
assignments for each fixed 3/4/5-bit width and signedness. Results certify
only output-bit equivalence; R,T are not referenced. Flags, state transitions,
PC, HALT, full program execution and larger widths are out of scope.

**IEEE-754 binary64**: concrete counterexamples for reassociation and
rounding. Counterexamples disprove universal identities; no full theorem of
floating-point equivalence is claimed.

**Invalid types**: mixed widths/signedness are rejected and excluded from
binary same-type equivalence scoring.

**Held-out**: entire template families and exact pair-structure fingerprints
are disjoint across training, validation and test. No historical pretraining
disjointness or cross-dataset independence is claimed.

## Reproduce

From /workspace/experiments/logos-kernel:

    PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-020/benchmark.py
    PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -q -s artifacts/NC-020 -p test_benchmark.py
    python dag.py validate
    python -m unittest -q test_dag

No neural network is trained, queried, or promoted by this benchmark.

## Code dependencies

- benchmark.py: Python 3 standard library; NC-001 canonical_isa.py for strict
  typed bitvector evaluation; NC-019 semantic_objective.json for integrity metadata.
- test_benchmark.py: benchmark.py, Python 3 standard library and all three
  generated dataset/evidence outputs.
- generated dataset: benchmark.py plus exact hashed NC-001 and NC-019 inputs.
- DAG operations: dag.py, test_dag.py, instructions.jsonl and Git.
- Original RLT champion, checkpoints, and NC-006 bounded circuit library:
  not loaded, modified or promoted.

## Release gate

A different real Codex verifier must independently recompute mathematical
cases without relying only on the producer's oracle, independently detect
wrong rewrites and type errors, inspect exact source hashes, and audit
family/structure split disjointness. Only then may the task owner append PASS.
