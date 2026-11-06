# NC-079: does a recursive verified concept library compound?

Bottom-up search over ISA arithmetic keeps subexpressions of verified solutions as a library, and later searches start from it. This is implementer evidence, not independent verification.

- `stack.py`: search, worlds, verification and library harvest.
- `run_compounding.py`: three worlds, three arms, 8 generations, 3 seeds.
- `run_curriculum.py`: the hard arithmetic exam with a training ladder.

    PYTHONDONTWRITEBYTECODE=1 python -m unittest -q artifacts/NC-079/test_stack.py
    PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-079/run_compounding.py
    PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-079/run_curriculum.py

Dependencies: Python 3, NumPy, `artifacts/NC-061/scoreboard.py` and `artifacts/NC-001/canonical_isa.py` through the scoreboard.

## Results on shared exams with a matched random control

Each random library copies the real library arm's concept count and sizes at the same generation, world and seed, and the run stops if they differ.

- Structured, final generation: library 0.544, no library 0.311, random library 0.178. The scoreboard returns NO_IMPROVEMENT, since retention of 0.933 is below its 0.95 floor.
- Unstructured: 0.089, 0.089 and 0.067. No gain, as a control should show.
- Arithmetic without a ladder: 0 in all arms.
- Curriculum on the hard arithmetic exam: library 0.211, against 0 for no library, random library and library without a ladder. All three comparisons return IMPROVEMENT.
- Every world and the curriculum: NO_COMPOUNDING.
