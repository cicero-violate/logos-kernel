"""NC-078 follow-up analysis from pinned checkpoints: a fairer ablation, spec sensitivity and an equal-CPU comparison.

The shuffled-example arm in results.json keeps each goal's mix of output values. Here each goal instead receives another
task's outputs, so no real information about that goal reaches the model.
"""
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spec_sensitivity as sens  # noqa: E402
import synth  # noqa: E402
import train_synth as ts  # noqa: E402


def other_task_search(model, goal, decoy, seed, name):
    rng = np.random.default_rng([seed, int(hashlib.sha256(("other|" + name).encode()).hexdigest()[:8], 16)])
    inputs = rng.choice(synth.DOMAIN, ts.EXAMPLES, replace=False)
    used = 0
    ex = ts.example_tokens(np.array([inputs]), np.array([decoy[inputs]]))
    for _ in range(ts.ROUNDS):
        for tokens in ts.sample(model, ex, ts.PER_ROUND):
            used += 1
            body = ts.decode_tokens(tokens)
            if body is not None and np.array_equal(synth.evaluate(body), goal):
                return {"solved": synth.verify_with_reference(body, goal), "candidates": used}
    return {"solved": False, "candidates": used}


def main():
    torch.set_num_threads(2)
    results = json.loads((HERE / "results.json").read_text())
    named = synth.named_tables()
    unseen = synth.unseen_goals(results["config"]["unseen"])
    goals = {**named, **unseen}
    names = list(goals)
    report = {"schema": "logos.nc078.analysis/1", "seeds": {}}
    for result in results["seeds"]:
        seed = result["info"]["seed"]
        model = ts.Model()
        model.load_state_dict(torch.load(HERE / result["info"]["checkpoint"], weights_only=True)["state"])
        model.eval()
        started = time.process_time()
        rows = {n: other_task_search(model, goals[n], goals[names[(i + 7) % len(names)]], seed, n) for i, n in enumerate(names)}
        neural = result["arms"]["neural"]
        other = {"named": ts.tally({n: rows[n] for n in named}), "unseen": ts.tally({n: rows[n] for n in unseen}), "cpu_seconds": round(time.process_time() - started, 1)}
        rng = np.random.default_rng(7)
        train = synth.training_programs(seed, 40000)
        seen = [train[i] for i in rng.choice(len(train), 400, replace=False)]
        report["seeds"][str(seed)] = {"neural_at_512": {"named": neural["named"]["512"], "unseen": neural["unseen"]["512"]},
                                      "shuffled_at_512": {"named": result["arms"]["shuffled_examples"]["named"]["512"], "unseen": result["arms"]["shuffled_examples"]["unseen"]["512"]},
                                      "other_task_outputs": other, "neural_cpu_seconds": neural["cpu_seconds"],
                                      "random_at_512": {"named": result["arms"]["random"]["named"]["512"], "unseen": result["arms"]["random"]["unseen"]["512"]},
                                      "train_spec_sensitivity": sens.evaluate(model, seen, rng)}
        print(seed, report["seeds"][str(seed)]["neural_at_512"], "other", other["named"]["512"], other["unseen"]["512"], "cpu", other["cpu_seconds"], flush=True)
    # Equal CPU: how far does enumeration reach in the CPU the neural arm spent per goal?
    started = time.process_time()
    count = 0
    for body in synth.enumerate_shortest_first():
        synth.evaluate(body)
        count += 1
        if count >= 20000:
            break
    per_second = count / (time.process_time() - started)
    needed = results["enumeration"]["needed"]
    neural_cpu = statistics.mean(r["arms"]["neural"]["cpu_seconds"] for r in results["seeds"])
    allowance = per_second * neural_cpu / len(goals)
    report["equal_cpu"] = {"enumeration_candidates_per_cpu_second": round(per_second), "neural_mean_cpu_seconds_per_goal": round(neural_cpu / len(goals), 2),
                           "enumeration_candidates_in_that_cpu": round(allowance),
                           "enumeration_solved": {"named": sum(n in named and v <= allowance for n, v in needed.items()), "unseen": sum(n in unseen and v <= allowance for n, v in needed.items())},
                           "neural_mean_solved": {"named": round(statistics.mean(r["arms"]["neural"]["named"]["512"] for r in results["seeds"]), 1), "unseen": round(statistics.mean(r["arms"]["neural"]["unseen"]["512"] for r in results["seeds"]), 1)},
                           "goals": {"named": len(named), "unseen": len(unseen)},
                           "caveats": ["Training CPU is extra and is not counted for the neural arm.", "Both arms ran under shared load, so CPU figures are approximate.", "Enumeration capped at its control limit, so it may undercount."]}
    (HERE / "analysis.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["equal_cpu"], indent=0))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
