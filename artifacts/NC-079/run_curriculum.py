"""NC-079 curriculum test: the sealed exam stays hard, while training climbs a ladder of a*X + b*Y coefficients.

The hard-only run scored zero because nothing could be solved, so a library never started. This asks whether easy
goals let the library bootstrap. Sealed goals come from the hard family only and share no function with training.
"""
import argparse
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stack as s  # noqa: E402

ARMS = ("library", "none", "random_library", "library_no_ladder")


class Ladder:
    """Coefficients grow with the level; level 0 is nearly trivial."""
    name = "ladder"

    def __init__(self, level):
        self.top = min(63, 3 + 4 * level * level)

    def goal(self, rng):
        return ("LINEAR", rng.randrange(1, self.top + 1), rng.randrange(0, self.top + 1))


def run_arm(arm, seed, a, x, y, real_libraries=None):
    """Returns rows and the library grown after each generation. The random arm copies the sizes of the real arm's libraries."""
    hard = s.ArithmeticWorld(seed)
    other = random.Random(f"curriculum|{arm}|{seed}|library")
    solutions, library, seen, retention, rows, grown = [], [], set(), [], [], []
    for generation in range(a.generations):
        sealed = s.draw_goals(hard, s.exam_rng("curriculum", seed, generation, "sealed"), a.sealed_goals, "sealed")
        solved, used, cpu = 0, 0, 0.0
        best = 0
        for goal in sealed:
            expr, spent, took = s.search(s.goal_vector(goal, x, y), library, a.budget, x, y)
            used += spent
            cpu += took
            if expr is not None and s.verified(expr, goal):
                solved += 1
                best = max(best, s.size(expr))
        kept = sum(s.search(s.goal_vector(g, x, y), library, a.budget, x, y)[0] is not None for g in retention)
        rows.append({"generation": generation, "solved": solved, "total": len(sealed), "solve_rate": solved / max(1, len(sealed)),
                     "candidates_used": used, "cpu_seconds": round(cpu, 3), "over_budget_goals": 0, "training_overlap": 0.0,
                     "retention": kept / len(retention) if retention else 1.0, "library_size": len(library), "largest_solution": best})
        teacher = hard if arm == "library_no_ladder" else Ladder(generation)
        train = s.draw_goals(teacher, s.exam_rng(f"curriculum|{teacher.name}", seed, generation, "train"), a.train_goals, "train", excluded=seen)
        seen |= {s.goal_key(g) for g in train}
        for goal in train:
            expr = s.search(s.goal_vector(goal, x, y), library, a.budget, x, y)[0]
            if expr is not None and s.verified(expr, goal):
                solutions.append(expr)
                if len(retention) < a.retention_goals:
                    retention.append(goal)
        real = s.harvest(solutions)
        grown.append(real)
        library = [] if arm == "none" else s.random_library(other, real_libraries[generation]) if arm == "random_library" else real
    return rows, grown


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generations", type=int, default=10)
    p.add_argument("--budget", type=int, default=20000)
    p.add_argument("--train-goals", type=int, default=30)
    p.add_argument("--sealed-goals", type=int, default=30)
    p.add_argument("--retention-goals", type=int, default=15)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 19, 29])
    p.add_argument("--out", type=Path, default=HERE / "curriculum_results.json")
    a = p.parse_args()
    x, y = s.probes()
    runs, real_libraries = {}, {}
    for arm in ARMS:
        for seed in a.seeds:
            rows, grown = run_arm(arm, seed, a, x, y, real_libraries.get(seed))
            if arm == "library":
                real_libraries[seed] = grown
            elif arm == "random_library":
                s.assert_matched_sizes(runs["library"][str(seed)], rows)
            runs.setdefault(arm, {})[str(seed)] = rows
            print("CURRICULUM", arm, seed, [round(r["solve_rate"], 2) for r in rows], "lib", rows[-1]["library_size"], "big", max(r["largest_solution"] for r in rows), "cpu", round(sum(r["cpu_seconds"] for r in rows), 1), flush=True)
    def at(arm, g):
        return [runs[arm][str(seed)][g] for seed in a.seeds]
    last = a.generations - 1
    gens = [s.sb.compare(at("library", g - 1), at("library", g)) for g in range(1, a.generations)]
    gains = [v.get("gain", 0.0) for v in gens]
    analysis = {"mean_solve_rate": {arm: [round(statistics.mean(r["solve_rate"] for r in at(arm, g)), 3) for g in range(a.generations)] for arm in ARMS},
                "final_vs_none": s.sb.compare(at("none", last), at("library", last)),
                "final_vs_random_library": s.sb.compare(at("random_library", last), at("library", last)),
                "final_vs_no_ladder": s.sb.compare(at("library_no_ladder", last), at("library", last)),
                "generation_verdicts": [v["verdict"] for v in gens], "generation_gains": gains,
                "compounding": s.sb.compounding(gains, [v["verdict"] for v in gens])}
    a.out.write_text(json.dumps({"schema": "logos.nc079.curriculum/1", "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, "analysis": analysis, "runs": runs}, indent=1))
    print("ANALYSIS", json.dumps(analysis), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
