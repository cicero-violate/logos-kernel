# NC-083: can a learned proposer find shorter verified programs than deterministic methods?

The exam is 40 verified LA-015 solutions per seed, unique by function, with held-out functions chosen by hash. Every arm offers smaller replacements for subexpressions through one frame. A replacement is accepted only if it computes the same function on all 65,536 pairs, and each final program is confirmed by two separately written evaluators. This is implementer evidence, not independent verification.

NO_GAIN: from 1570 operations, the exact table saved 108, search at equal total CPU 48, the learned proposer 3, another program's tokens 1, random proposals 0, NC-023 rewrites 0. The learned arm's CPU includes training; search was calibrated to it within a fixed 10% tolerance.

Programs shortened by the exact table and small enough for NC-082's machine ran on its Transformer and NC-080's operators with no teacher forcing: 1696 of 1696 rollouts correct, 10464 executed steps before and 8864 after.

    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest -q artifacts/NC-083/test_shorten.py
    OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-083/shorten.py --out /tmp/nc083-replay

Dependencies: Python 3, NumPy, PyTorch CPU, `artifacts/NC-023`, `artifacts/NC-020`, `artifacts/NC-001`, and `artifacts/NC-082` and `artifacts/NC-080` for execution.
