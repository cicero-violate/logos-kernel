# NC-009: verify, deduplicate and promote operators

`promote.py` canonicalizes NC-008's proposals by removing dead steps, checks them with a separately written verifier, and appends accepted operators to the hash-chained registry `operator_promotions.jsonl`. `promotion_audit.json` records every decision, including duplicates and rejections.

    PYTHONDONTWRITEBYTECODE=1 python -m unittest -q artifacts/NC-009/test_promote.py
    PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-009/promote.py

A rerun appends nothing. Dependencies: Python 3 standard library for `promote.py`; `test_promote.py` also imports NC-007's `compose.py`, which needs NumPy and PyTorch.
