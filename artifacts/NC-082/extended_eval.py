"""NC-082 stress tests on saved checkpoints: small operands, wider input sets, new programs, a full training audit and controls.

The training audit regenerates every program from each seed's real training stream and compares it with all held-out
programs. It replaces the sampled collision figure in controller_results.json, which did not follow the branch curriculum.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import controller as c  # noqa: E402

CHUNK = 512


def load(seed):
    model = c.Controller()
    model.load_state_dict(torch.load(HERE / f"controller_seed{seed}.pt", weights_only=True)["state"])
    return model.eval()


def run(model, alu, program, inputs):
    """Rollouts in chunks. A row counts only if it halted with exactly the right result."""
    correct, halted, steps = 0, 0, 0
    for start in range(0, len(inputs), CHUNK):
        part = inputs[start:start + CHUNK]
        got, done, used = c.rollout(model, alu, program, part)
        for (x, y), g, d in zip(part, got, done):
            correct += d and g == c.exact_run(program, x, y)
        halted += sum(done)
        steps = max(steps, max(used))
    return {"correct": correct, "total": len(inputs), "halted": halted, "max_steps": steps}


def edge_pairs():
    """121 pairs from 11 edge values, as in the original 8-bit multiplication test."""
    edges = (0, 1, 2, 3, 15, 16, 127, 128, 170, 254, 255)
    return [(x, y) for x in edges for y in edges]


def renamed(program, rng):
    """The same algorithm in other work registers. r0, r1 and r2 keep their roles so inputs and result stay comparable."""
    order = list(range(3, c.REGS))
    rng.shuffle(order)
    mapping = {0: 0, 1: 1, 2: 2, **dict(zip(range(3, c.REGS), order))}
    return [(n, mapping[a], mapping[b], mapping[d], i) for n, a, b, d, i in program]


def relocated(program, padding):
    """Move a program later by leading no-ops on a spare register, retargeting its jumps. Tests that nothing depends on position."""
    pad = [("OR", 7, 7, 7, 0)] * padding
    out = [(n, a, b, d, i + padding if n in ("JZ", "JMP") else i) for n, a, b, d, i in program]
    return pad + out


def fresh_programs(rng, count):
    """Random programs that halt within the limit on the test inputs, with loops and branches kept separately countable."""
    out = []
    while len(out) < count:
        program = c.random_program(rng)
        if all(c.exact_run(program, x, y) is not None for x, y in ((0, 0), (1, 255), (37, 91), (255, 255))):
            out.append(program)
    return out


def has_backward_jump(program):
    return any(n in ("JZ", "JMP") and i <= pc for pc, (n, _, _, _, i) in enumerate(program))


def audit_training(seed, steps, batch, held_out):
    """Every program the seed trained on, from its real stream, against every held-out program."""
    rng = random.Random(f"NC-082/train/{seed}")
    wanted = {c.encode_program(q).tobytes() for q in held_out}
    seen, collisions = 0, 0
    for _ in range(steps):
        programs = c.batch_rows(rng, batch)[0].numpy()
        for row in programs:
            seen += 1
            collisions += row.tobytes() in wanted
    return {"training_programs_checked": seen, "collisions_with_held_out": collisions, "held_out_programs": len(wanted)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 19, 29])
    p.add_argument("--fresh", type=int, default=200)
    p.add_argument("--out", type=Path, default=HERE / "extended_results.json")
    a = p.parse_args()
    torch.set_num_threads(2)
    base = json.loads((HERE / "controller_results.json").read_text())
    concepts, _ = c.test_programs(HERE / "library_snapshot.jsonl", base["config"]["concepts"])
    mul = c.multiply_program()
    loops = {name: program for name, (program, _) in c.branching_programs().items()}
    rng = random.Random("NC-082/extended")
    small = [(x, y) for x in range(16) for y in range(16)]
    sampled = [(rng.randrange(256), rng.randrange(256)) for _ in range(4096)] + edge_pairs()
    wide = [(rng.randrange(256), rng.randrange(256)) for _ in range(256)]
    variants = {f"renamed_{i}": renamed(mul, rng) for i in range(10)} | {f"relocated_{n}": relocated(mul, n) for n in (1, 3, 7, 14)}
    assert all(c.exact_run(q, x, y) == (x * y) & 255 for q in variants.values() for x, y in wide[:32])
    fresh = fresh_programs(random.Random("NC-082/extended/fresh"), a.fresh)
    held_out = [*concepts, mul, *loops.values(), *variants.values(), *fresh]
    report = {"schema": "logos.nc082.extended/1", "controller_results_sha256": hashlib.sha256((HERE / "controller_results.json").read_bytes()).hexdigest(),
              "fresh_programs": len(fresh), "fresh_with_backward_jump": sum(map(has_backward_jump, fresh)), "seeds": {}}
    for seed in a.seeds:
        model, alu = load(seed), c.NeuralALU(seed)
        row = {"training_audit": audit_training(seed, base["config"]["steps"], base["config"]["batch"], held_out)}
        row["multiply_all_4bit_pairs"] = run(model, alu, mul, small)
        row["multiply_4217_pairs"] = run(model, alu, mul, sampled)
        row["loops_all_4bit_pairs"] = {name: run(model, alu, q, small) for name, q in loops.items()}
        row["multiply_variants"] = {name: run(model, alu, q, wide) for name, q in variants.items()}
        rows = [run(model, alu, q, wide[:32]) for q in concepts]
        row["concepts_32_inputs"] = {"fully_correct": sum(r["correct"] == r["total"] for r in rows), "programs": len(rows), "rollouts_correct": sum(r["correct"] for r in rows), "rollouts": sum(r["total"] for r in rows)}
        rows = [run(model, alu, q, wide[:32]) for q in fresh]
        failed = [i for i, r in enumerate(rows) if r["correct"] != r["total"]]
        row["fresh_random_programs"] = {"fully_correct": len(fresh) - len(failed), "programs": len(fresh), "rollouts_correct": sum(r["correct"] for r in rows), "rollouts": sum(r["total"] for r in rows),
                                        "fully_correct_with_backward_jump": sum(has_backward_jump(q) and r["correct"] == r["total"] for q, r in zip(fresh, rows)), "failed_program_lengths": [len(fresh[i]) for i in failed][:30]}
        model.blank = True
        controls = {name: run(model, alu, q, wide[:64]) for name, q in {**loops, "multiply": mul}.items()}
        blanks = [run(model, alu, q, wide[:16]) for q in concepts]
        fresh_blanks = [run(model, alu, q, wide[:16]) for q in fresh]
        model.blank = False
        row["blank_program_control"] = {"programs": controls, "concepts_fully_correct": sum(r["correct"] == r["total"] for r in blanks), "fresh_fully_correct": sum(r["correct"] == r["total"] for r in fresh_blanks)}
        report["seeds"][str(seed)] = row
        a.out.write_text(json.dumps(report, indent=1))
        print(seed, json.dumps({k: row[k] for k in ("training_audit", "multiply_all_4bit_pairs", "multiply_4217_pairs", "concepts_32_inputs", "fresh_random_programs")}), "variants", sum(v["correct"] == v["total"] for v in row["multiply_variants"].values()), "of", len(variants), "blank", row["blank_program_control"]["concepts_fully_correct"], row["blank_program_control"]["fresh_fully_correct"], flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
