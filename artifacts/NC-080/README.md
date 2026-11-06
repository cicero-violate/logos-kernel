# NC-080: learned operators execute verified concepts

A small network learns each operation's exhaustive local bit rule, the same recipe that taught neural ISA multiplication. The host applies it across bits to run verified concepts from a pinned LA-015 library snapshot. Concepts are first merged into exact behavioral classes on all 65,536 pairs. This is implementer evidence, not independent verification.

    PYTHONDONTWRITEBYTECODE=1 python -m unittest -q artifacts/NC-080/test_neural_quotient.py
    PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-080/neural_quotient.py --library artifacts/NC-080/library_snapshot.jsonl

Dependencies: Python 3, NumPy, PyTorch CPU.
