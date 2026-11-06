"""NC-079 experiment: library, no-library and random-library arms across generations, on sealed goals in three worlds."""
import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stack as s  # noqa: E402

WORLDS = (s.StructuredWorld, s.ArithmeticWorld, s.UnstructuredWorld)
ARMS = ("library", "none", "random_library")
THRESHOLDS = dict(s.sb.THRESHOLDS)


def run_arm(world_cls, arm, seed, a, x, y, real_libraries=None):
    """Returns rows and the library grown after each generation. The random arm copies the sizes of the real arm's libraries."""
    world = world_cls(seed)
    other = random.Random(f"{arm}|{seed}|library")
    solutions, library, seen, rows = [], [], set(), []
    retention_goals, grown = [], []
    for generation in range(a.generations):
        row, learned = s.run_generation(world, library, a.budget, s.exam_rng(world.name, seed, generation, "sealed"),
                                        s.exam_rng(world.name, seed, generation, "train"), x, y, a.train_goals, a.sealed_goals, seen)
        kept = sum(s.search(s.goal_vector(g, x, y), library, a.budget, x, y)[0] is not None for g in retention_goals)
        row["retention"] = kept / len(retention_goals) if retention_goals else 1.0
        row["generation"] = generation
        rows.append(row)
        solutions += learned
        real = s.harvest(solutions)
        grown.append(real)
        library = real if arm == "library" else [] if arm == "none" else s.random_library(other, real_libraries[generation])
        if arm == "library":
            retention_goals = solutions[:a.retention_goals]
    return rows, grown


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generations", type=int, default=8)
    p.add_argument("--budget", type=int, default=20000)
    p.add_argument("--train-goals", type=int, default=30)
    p.add_argument("--sealed-goals", type=int, default=30)
    p.add_argument("--retention-goals", type=int, default=15)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 19, 29])
    p.add_argument("--out", type=Path, default=HERE / "compounding_results.json")
    a = p.parse_args()
    x, y = s.probes()
    runs = {}
    for world_cls in WORLDS:
        real_libraries = {}
        for arm in ARMS:
            for seed in a.seeds:
                began = time.process_time()
                rows, grown = run_arm(world_cls, arm, seed, a, x, y, real_libraries.get(seed))
                if arm == "library":
                    real_libraries[seed] = grown
                elif arm == "random_library":
                    s.assert_matched_sizes(runs[world_cls.name]["library"][str(seed)], rows)
                runs.setdefault(world_cls.name, {}).setdefault(arm, {})[str(seed)] = rows
                print(world_cls.name, arm, seed, "rates", [round(r["solve_rate"], 2) for r in rows], "lib", rows[-1]["library_size"], "big", rows[-1]["largest_solution"], "cpu", round(time.process_time() - began, 1), flush=True)
    analysis = {}
    for name, arms in runs.items():
        analysis[name] = {}
        def at(arm, generation):
            return [arms[arm][str(seed)][generation] for seed in a.seeds]
        rows = {arm: [statistics.mean(r["solve_rate"] for r in at(arm, g)) for g in range(a.generations)] for arm in ARMS}
        gens = [s.sb.compare(at("library", g - 1), at("library", g), THRESHOLDS) for g in range(1, a.generations)]
        gains = [v.get("gain", 0.0) for v in gens]
        analysis[name] = {"mean_solve_rate_by_generation": {k: [round(v, 3) for v in r] for k, r in rows.items()},
                          "final_vs_none": s.sb.compare(at("none", a.generations - 1), at("library", a.generations - 1), THRESHOLDS),
                          "final_vs_random_library": s.sb.compare(at("random_library", a.generations - 1), at("library", a.generations - 1), THRESHOLDS),
                          "generation_verdicts": [v["verdict"] for v in gens], "generation_gains": gains,
                          "compounding": s.sb.compounding(gains, [v["verdict"] for v in gens], THRESHOLDS)}
    a.out.write_text(json.dumps({"schema": "logos.nc079.results/1", "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, "analysis": analysis, "runs": runs}, indent=1))
    print(json.dumps(analysis, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
