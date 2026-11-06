"""NC-082: which held-out programs, if any, also appeared in a seed's real training stream. Run after extended_eval.py."""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import controller as c  # noqa: E402
import extended_eval as e  # noqa: E402


def groups():
    """The same held-out programs as extended_eval.main, drawn in the same random order."""
    concepts, _ = c.test_programs(HERE / "library_snapshot.jsonl", json.loads((HERE / "controller_results.json").read_text())["config"]["concepts"])
    mul = c.multiply_program()
    rng = random.Random("NC-082/extended")
    [(rng.randrange(256), rng.randrange(256)) for _ in range(4096 + 256)]
    variants = [e.renamed(mul, rng) for _ in range(10)] + [e.relocated(mul, n) for n in (1, 3, 7, 14)]
    return {"concept": concepts, "multiply": [mul], "other_loops": [p for p, _ in c.branching_programs().values()], "multiply_variant": variants, "fresh_random": e.fresh_programs(random.Random("NC-082/extended/fresh"), 200)}


def main():
    held = groups()
    lookup = {}
    for name, programs in held.items():
        for index, program in enumerate(programs):
            lookup.setdefault(c.encode_program(program).tobytes(), []).append((name, index, len(program)))
    base = json.loads((HERE / "controller_results.json").read_text())["config"]
    out = {"schema": "logos.nc082.audit/1", "held_out_per_group": {k: len(v) for k, v in held.items()}, "seeds": {}}
    for seed in (11, 19, 29):
        rng = random.Random(f"NC-082/train/{seed}")
        occurrences, programs = {}, {}
        for _ in range(base["steps"]):
            for row in c.batch_rows(rng, base["batch"])[0].numpy():
                for name, index, length in lookup.get(row.tobytes(), ()):
                    occurrences[name] = occurrences.get(name, 0) + 1
                    programs.setdefault(name, {})[index] = length
        out["seeds"][str(seed)] = {"occurrences_by_group": occurrences, "distinct_held_out_seen_in_training": {k: len(v) for k, v in programs.items()}, "lengths_seen": {k: sorted(set(v.values())) for k, v in programs.items()}, "training_programs_checked": base["steps"] * base["batch"]}
        print(seed, out["seeds"][str(seed)], flush=True)
    (HERE / "training_audit_by_group.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
