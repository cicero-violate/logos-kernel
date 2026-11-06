"""NC-061 revised candidate independent CPU-budget counterexample.
Dependencies: Python 3 standard library; sibling scoreboard.py; ../NC-001/canonical_isa.py.
No writes to candidate code or artifacts.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scoreboard as sb


def score(solve_rate, cpu_seconds):
    return {"solve_rate": solve_rate, "candidates_used": 3,
            "cpu_seconds": cpu_seconds, "retention": 1.0,
            "training_overlap": 0.0, "over_budget_goals": 0}


def run():
    previous = [score(0.0, 0.001) for _ in range(3)]
    current = [score(1.0, 0.9) for _ in range(3)]
    verdict = sb.compare(previous, current)
    assert verdict["verdict"] == "IMPROVEMENT", verdict
    assert current[0]["cpu_seconds"] / previous[0]["cpu_seconds"] == 900.0
    assert sb.THRESHOLDS["cpu_tolerance"] == 0.25
    assert sb.THRESHOLDS["cpu_floor_seconds"] == 1.0
    control = sb.compare([score(0.0, 10.0)] * 3, [score(1.0, 20.0)] * 3)
    assert control["verdict"] == "NO_IMPROVEMENT"
    # Previously reported flaws were fixed in the revised implementation.
    assert sb.compounding([0.05] * 3, ["IMPROVEMENT"] * 3)["verdict"] == "NO_COMPOUNDING"
    return {
        "test": "unmatched CPU for very cheap seeded baselines",
        "seeds": 3, "previous_cpu_seconds_each": 0.001,
        "candidate_cpu_seconds_each": 0.9, "cpu_increase_factor": 900,
        "same_candidates_used": 3, "solve_rates": [0.0, 1.0],
        "threshold_relative_tolerance": sb.THRESHOLDS["cpu_tolerance"],
        "absolute_floor_seconds": sb.THRESHOLDS["cpu_floor_seconds"],
        "observed_verdict": verdict["verdict"], "expected": "NO_IMPROVEMENT",
        "ordinary_2x_budget_control": control["verdict"],
        "reason": "1s additive floor allows materially higher CPU budget on cheap baselines"
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True, indent=2))
