# NC-079: does a recursive verified concept library compound?

Bottom-up search over ISA arithmetic keeps subexpressions of verified solutions as a library, and later searches start from it. This is implementer evidence, not independent verification.

- `stack.py`: search, worlds, verification and library harvest.
- `run_compounding.py`: three worlds, three arms, 8 generations, 3 seeds.
- `run_curriculum.py`: the hard arithmetic exam with a training ladder.

    PYTHONDONTWRITEBYTECODE=1 python -m unittest -q artifacts/NC-079/test_stack.py
    PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-079/run_compounding.py
    PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-079/run_curriculum.py

Dependencies: Python 3, NumPy, `artifacts/NC-061/scoreboard.py` and `artifacts/NC-001/canonical_isa.py` through the scoreboard.
