# NC-019 — Semantic-first evaluation contract

Source of truth: semantic_objective.json, benchmark_contract.json,
precedence_and_audit.md.

Supplementary evidence: reference_contract.py, score_contract.py,
reference_receipt.json, test_semantic_objective.py.

Task NC-019 is an objective/benchmark preregistration. It does not run
neural training, promote a champion or rewrite NC-018. A real separate
independent verifier must produce independent_receipt.json before DAG PASS.

Run from /workspace/experiments/logos-kernel:

    PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-019/reference_contract.py
    PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -q -s artifacts/NC-019 -p test_semantic_objective.py
    python dag.py validate
    python -m unittest -q test_dag

Dependencies: Python3 stdlib, NC-001 canonical_isa.py plus frozen NC-000/
NC-004/NC-005 source/receipt artifacts and original RLT champion weights
for integrity hashes only; PyTorch and trained models are NOT loaded.
