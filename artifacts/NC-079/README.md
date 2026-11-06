# NC-079: does a recursive verified concept library compound?

Bottom-up search over ISA arithmetic keeps subexpressions of verified solutions as a library, and later searches start from it. This is implementer evidence, not independent verification.

- `stack.py`: search, worlds, verification and library harvest.
- `run_compounding.py`: three worlds, three arms, 8 generations, 3 seeds.
- `run_curriculum.py`: the hard arithmetic exam with a training ladder.

    PYTHONDONTWRITEBYTECODE=1 python -m unittest -q artifacts/NC-079/test_stack.py
    PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-079/run_compounding.py
    PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-079/run_curriculum.py

Dependencies: Python 3, NumPy, `artifacts/NC-061/scoreboard.py` and `artifacts/NC-001/canonical_isa.py` through the scoreboard.

## Results on shared exams

These replace the first run, which a review failed because arms faced different exams and random libraries had the wrong sizes. Structured world, final generation: library 0.544, no library 0.311, random library 0.156. The scoreboard still returns NO_IMPROVEMENT, since retention of 0.933 is below its 0.95 floor. Unstructured: no gain, as a control should show. Arithmetic without a ladder: 0 in all arms. Curriculum: 0.211 against 0 for no library, random library and library without a ladder. It passes against no library and no ladder, and is refused against the random library at 2.19 CPU seconds against a limit of 2.00. All worlds: NO_COMPOUNDING.
