"""Independent NC-061 equal-CPU negative control.
Dependencies: Python 3 standard library, sibling scoreboard.py, ../NC-001/canonical_isa.py.
No candidate source or manifest writes.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scoreboard as sb

def row(solve, cpu):
    return {"solve_rate": solve, "candidates_used": 40, "cpu_seconds": cpu,
            "retention": 1.0, "training_overlap": 0.0, "over_budget_goals": 0}

def run():
    prior = [row(0.0, 0.001) for _ in range(3)]
    improvement = [row(1.0, 0.049) for _ in range(3)]
    result = sb.compare(prior, improvement)
    assert result["verdict"] == "IMPROVEMENT", result
    assert improvement[0]["cpu_seconds"] / prior[0]["cpu_seconds"] == 49.0
    over_floor = sb.compare(prior, [row(1.0, 0.051) for _ in range(3)])
    assert over_floor["verdict"] == "NO_IMPROVEMENT", over_floor
    ordinary = sb.compare([row(0.0, 0.1)] * 3, [row(1.0, 0.126)] * 3)
    assert ordinary["verdict"] == "NO_IMPROVEMENT", ordinary
    return {"task": "NC-061", "case": "unmatched_cpu_below_absolute_noise_floor",
            "previous_cpu_seconds_per_seed": 0.001, "current_cpu_seconds_per_seed": 0.049,
            "cpu_ratio": 49.0, "seed_count": 3, "candidate_counts_equal": True,
            "retention": 1.0, "training_overlap": 0.0, "score_rates": [0.0, 1.0],
            "observed": result["verdict"], "expected": "NOT_ESTABLISHED or NO_IMPROVEMENT",
            "limit_case_above_0_05": over_floor["verdict"],
            "limit_case_above_ratio": ordinary["verdict"],
            "reason": "A 0.05 s absolute floor exempts a 49x CPU increase from the declared matched-CPU gate",
            "scope": "Synthetic per-seed score rows; proves policy false positive, not an actual external model experiment"}

if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True, indent=2))
