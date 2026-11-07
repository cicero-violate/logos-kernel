# NC-010: a Transformer writes programs with branches and loops from examples

NO_GENERALIZATION on multiplication: 0 of 3 seeds within 4,096 candidates. Partial recombination: 3 of 12 held-out goal runs solved, against 0 of 12 without skeletons, 0 of 12 on another goal's examples, and 0 of 12 for random programs. Goals from trained combinations: 18 of 18, against 0 of 18 without skeletons.

Training shows bit-loop skeletons over most operator combinations plus random programs. Four combinations, including every ADD and SHL program, stay out, by construction and by behaviour on the examples. Candidates must fit 16 examples and then compute the goal on all 65,536 pairs. Solved held-out programs ran on NC-082's Transformer and NC-080's operators on 4,217 pairs with no teacher forcing. This is implementer evidence, not independent verification.

This is recombination of taught parts, not invention from nothing: the arm with no skeletons solved nothing, and token structure is given by a type mask. Multiplication was not invented.

    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest -q artifacts/NC-010/test_invent.py
    OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-010/invent.py --seeds 11 --out /tmp/nc010-replay

Dependencies: Python 3, PyTorch CPU, `artifacts/NC-082` and `artifacts/NC-080`.
