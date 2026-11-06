"""Independent NC-061 falsification probes. Does not modify candidate code.
Dependencies: Python 3 stdlib; sibling scoreboard.py and controls.py;
../NC-001/canonical_isa.py. Run from any directory.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scoreboard as sb
import controls


def row(rate, used, cpu):
    return {"solve_rate": rate, "candidates_used": used, "cpu_seconds": cpu,
            "training_overlap": 0., "over_budget_goals": 0, "retention": 1.0}


def run():
    prev = [row(0.0, 3, 0.01) for _ in range(3)]
    cur = [row(1.0, 3, 3600.0) for _ in range(3)]
    cpu = sb.compare(prev, cur)
    assert cpu["verdict"] == "IMPROVEMENT", cpu
    reveal = controls.control_reveal(123, 3)
    goals = sb.sealed_goals(reveal)
    perfect = controls.known_good(goals)
    count_tests = {}
    for spent in (-1, 0):
        report = sb.score_candidate(lambda examples, budget, n=spent:
                                    (perfect(examples, budget)[0], n),
                                    goals, budget=1, retention=1.0)
        assert report["solved"] == 3 and report["over_budget_goals"] == 0, report
        count_tests[str(spent)] = {"solved": report["solved"],
                                  "over_budget_goals": report["over_budget_goals"],
                                  "candidates_used": report["candidates_used"]}
    steady = sb.compounding([0.05, 0.05, 0.05], ["IMPROVEMENT"] * 3)
    assert steady["verdict"] == "COMPOUNDING", steady
    return {
        "counterexample_cpu": {
            "previous_cpu_seconds_each": 0.01, "current_cpu_seconds_each": 3600.0,
            "cpu_factor": 360000, "verdict": cpu["verdict"],
            "conclusion": "matched candidate counts are not matched CPU budgets"},
        "counterexample_invalid_candidate_counts": count_tests,
        "counterexample_non_accelerating_compounding": {
            "gains": [0.05, 0.05, 0.05], "verdict": steady["verdict"],
            "conclusion": "constant rate is classified as compounding growth"},
        "probe_scope": "Pure calls on frozen source; record values are intentionally constructed adversarial inputs",
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True, indent=2))
