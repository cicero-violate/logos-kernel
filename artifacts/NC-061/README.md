# NC-061: self-improvement scoreboard

Measures whether a generation really improved, on sealed goals it cannot have trained on. Implementer evidence, not independent verification.

- `scoreboard.py`: commit and reveal, sealed goal generation, exhaustive scoring, generation and compounding verdicts.
- `scoreboard_spec.json`: thresholds, verdict definitions, procedure and limits.
- `controls.py` and `control_results.json`: control candidates that must get the right verdict. Their seeds are public, unlike a real run.
- `test_scoreboard.py`: tests.

    PYTHONDONTWRITEBYTECODE=1 python -m unittest -q artifacts/NC-061/test_scoreboard.py
    PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-061/controls.py

Dependencies: Python 3 standard library and `artifacts/NC-001/canonical_isa.py`.
