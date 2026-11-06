# NC-079 — partial independent control audit; not a candidate

This is an IMPLEMENTER-owned supplementary audit. The previous candidate remains at independent FAIL. No revised candidate is published.

The current working tree already contained uncommitted changes to stack.py, run_compounding.py, run_curriculum.py and test_stack.py before this turn. They belong to another worker, and this turn did not change or stage them. The source hashes in matched_control_smoke_d93446.json are a snapshot of that *uncommitted* code, not a committed or independently frozen candidate. A future owner must reconcile and commit those changes before seeking independent review.

Scope of new evidence:
- Small real-search replay: 3 seeds (11/19/29), 3 worlds plus curriculum, 3 generations, budget 1500 per goal, 3 training goals and 2 sealed goals, library and random-library arms only. All 12 pairs matched exact count and expression-size multiset per generation. Some small-budget libraries remained empty; these checks do not demonstrate a gain or compounding.
- A new mock-based contract suite verifies exact distribution matching across all 3 worlds, curriculum and 3 seeds, and independently rejects equal counts with the wrong expression-size multiset.
- A separate scalar checker exhaustively checks 65,536 input pairs instead of the incumbent verifier's sampled 1,024. A negative test injects a wrong scalar result at an input outside the sample; the incumbent verifier incorrectly returns True, while the new exhaustive helper returns False. The helper is NOT connected to existing run_generation/run_arm acceptance, so the scientific gate remains unmet.

Reproduce the fast new tests from repository root (Python 3, NumPy, NC-061 scoreboard dependency):

    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest -q artifacts/NC-079/test_matched_library_controls_d93446.py artifacts/NC-079/test_exhaustive_reference_d93446.py

Reproduce the bounded real-search audit (output path can be changed):

    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python artifacts/NC-079/audit_matched_controls_d93446.py --out /tmp/nc079_matched_again.json

What remains to satisfy NC-079:
1. Reconcile the pre-existing modified runner and verifier files without overwriting another worker's work. Persist and pin the fixed implementation. The experiment-level assertion must cover exact sizes, not only total counts.
2. Connect exhaustive independent reference verification to *every accepted solution*.
3. Rerun all experimental arms in structured/arithmetic/unstructured worlds and curriculum, all 3 seeds, original budgets/generations; capture data, CPU budgets and scoreboard conclusions. The current small smoke audit is not that campaign.
4. Publish a revised CANDIDATE_MANIFEST.json with correct new owner, source/evidence hashes and reproduction commands, and request genuinely independent review. NO_COMPOUNDING remains the default until supported otherwise.

No verifier PASS, no champion changes, no claim of AGI or self-improvement. Dependencies: Python 3, NumPy, Git, stack.py, run_compounding.py, run_curriculum.py, NC-061/scoreboard.py, NC-001/canonical_isa.py, new audit and test files.
