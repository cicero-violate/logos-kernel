"""NC-081: a small network chooses which verified concepts search starts from. It never executes or accepts anything.

Training is supervised on generated compositions ("dreams") of the concepts in a pinned LA-015 library snapshot, so each
example has known labels. Every arm faces the same sealed exam, and answers are verified exhaustively by library.verify.
"""
import argparse
import hashlib
import importlib.util
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
LIB = HERE / "la015_library_pinned.py"
LIB_SHA256 = "0c876a9e8d90e9401323b0068a66acc9b6d1e9f3051f22a771fe9d2a2d098b8a"
LIB_SOURCE = "logos-autonomy@b4a8d1b21e034f05372e73e28cfa34ea4ca380f6:discovery/library.py"
TOLERANCE = 0.10  # Fixed before the run: the baseline's measured CPU must land within 10% of the model's total.
CALIBRATION_TRIES = 5


def load_pinned(path, sha256):
    """Refuse to run on anything but the exact pinned library, before any training starts."""
    found = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if found != sha256:
        raise SystemExit(f"pinned library mismatch: expected {sha256}, found {found}")
    spec = importlib.util.spec_from_file_location("la015_library", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lib = load_pinned(LIB, LIB_SHA256)

EXAMPLES = 16
BITS = 8
TOP_K = 12
BUDGET = 8000
KEY_XS, KEY_YS = np.meshgrid(np.arange(0, 256, 3), np.arange(0, 256, 5), indexing="ij")
KEY_XS, KEY_YS = KEY_XS.ravel(), KEY_YS.ravel()
XS16 = lib.probes()[0][:EXAMPLES], lib.probes()[1][:EXAMPLES]


def vector_eval(expr, x, y, memo):
    if expr in memo:
        return memo[expr]
    head = expr[0]
    if head == "X":
        out = x
    elif head == "Y":
        out = y
    elif head == "K":
        out = np.full(x.shape, expr[1] & 255)
    elif head == "SHL":
        out = (vector_eval(expr[1], x, y, memo) << 1) & 255
    elif head == "SHR":
        out = vector_eval(expr[1], x, y, memo) >> 1
    else:
        a, b = vector_eval(expr[1], x, y, memo), vector_eval(expr[2], x, y, memo)
        out = {"ADD": (a + b) & 255, "SUB": (a - b) & 255, "AND": a & b, "OR": a | b, "XOR": a ^ b}[head]
    memo[expr] = out
    return out


def function_key(expr):
    """The same key as library.goal_table_key, so dreams on the sealed side can be excluded."""
    return hashlib.sha256(np.asarray(vector_eval(expr, KEY_XS, KEY_YS, {}), dtype=np.uint8).tobytes()).hexdigest()


def is_sealed_key(key):
    return int(key[:8], 16) % 2 == 1


def features(outputs):
    """Bits of X, Y and R for each fixed probe, so the pairing of inputs and outputs is available to the model."""
    xs, ys = np.array(XS16[0]), np.array(XS16[1])
    rows = np.stack([xs, ys, np.asarray(outputs)], 1)
    return np.concatenate([(rows >> i) & 1 for i in range(BITS)], 1).astype(np.float32)


class Selector(nn.Module):
    """Permutation-invariant over examples, because their order carries no meaning."""
    def __init__(self, concepts, width=128):
        super().__init__()
        self.each = nn.Sequential(nn.Linear(3 * BITS, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(width * 2, width), nn.ReLU(), nn.Linear(width, concepts))

    def forward(self, x):
        h = self.each(x)
        return self.head(torch.cat((h.mean(1), h.max(1).values), 1))


def training_side(concepts):
    """Indices of concepts whose own function is not on the sealed side. Only these may appear in a dream or be a label."""
    return [i for i, concept in enumerate(concepts) if not is_sealed_key(function_key(concept))]


def dream(rng, concepts, allowed):
    """A function built from 1 to 3 training-side concepts, with those concepts as labels."""
    count = rng.choice((1, 2, 2, 3))
    picks = rng.sample(allowed, min(count, len(allowed)))
    expr = concepts[picks[0]]
    for index in picks[1:]:
        expr = (rng.choice(lib.BINARY), expr, concepts[index])
    if rng.random() < 0.5:
        expr = (rng.choice(lib.BINARY), expr, rng.choice(lib.LEAVES[:2]))
    if rng.random() < 0.2:
        expr = (rng.choice(lib.UNARY), expr)
    return expr, picks


def make_dreams(seed, concepts, count):
    rng = random.Random(f"NC-081/dreams/{seed}")
    x, y = np.array(XS16[0]), np.array(XS16[1])
    allowed = training_side(concepts)
    inputs, labels, skipped = [], [], 0
    while len(inputs) < count:
        expr, picks = dream(rng, concepts, allowed)
        if is_sealed_key(function_key(expr)):
            skipped += 1  # Sealed-side functions never enter training.
            continue
        inputs.append(features(vector_eval(expr, x, y, {})))
        row = np.zeros(len(concepts), dtype=np.float32)
        row[picks] = 1
        labels.append(row)
    return torch.tensor(np.stack(inputs)), torch.tensor(np.stack(labels)), skipped


def train(seed, concepts, count, steps):
    torch.manual_seed(seed)
    started = time.process_time()
    x, y, skipped = make_dreams(seed, concepts, count)
    model = Selector(len(concepts))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(x), (256,), generator=generator)
        loss = nn.functional.binary_cross_entropy_with_logits(model(x[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model, {"dreams": count, "sealed_side_skipped": skipped, "final_loss": round(loss.item(), 4), "train_cpu_seconds": round(time.process_time() - started, 1)}


def load_concepts(path):
    log = lib.Library(path)
    return log.concepts(), log


def exam(seed, count):
    """Hard goals from the sealed side, chosen before any arm runs and shared by all arms."""
    rng = random.Random(f"NC-081/exam/{seed}")
    out, keys = [], set()
    while len(out) < count:
        goal = lib.hard_goal(rng, rng.choice(lib.FAMILIES))
        if lib.is_sealed_side(goal):
            key = lib.goal_table_key(goal)
            if key not in keys:
                keys.add(key)
                out.append(goal)
    return out


def solve(goal, seeds, budget, xs, ys):
    target = tuple(lib.goal_value(goal, x, y) for x, y in zip(xs, ys))
    expression, spent = lib.search(target, seeds, budget, xs, ys)
    return (expression is not None and lib.verify(expression, goal)), spent


def calibrate(run, target_cpu, start_budget, tolerance=TOLERANCE, tries=CALIBRATION_TRIES):
    """Find a budget whose measured CPU is within tolerance of the target. Every attempt is kept, and matched says if one landed."""
    budget, attempts = start_budget, []
    for _ in range(tries):
        result = run(budget)
        attempts.append({"budget_per_goal": budget, "solved": result["solved"], "cpu_seconds": result["cpu_seconds"]})
        if abs(result["cpu_seconds"] - target_cpu) <= tolerance * target_cpu:
            return {**result, "budget_per_goal": budget, "target_cpu_seconds": round(target_cpu, 2), "tolerance": tolerance, "matched": True, "attempts": attempts}
        budget = max(1, int(budget * target_cpu / max(result["cpu_seconds"], 1e-6)))
    return {**result, "budget_per_goal": attempts[-1]["budget_per_goal"], "target_cpu_seconds": round(target_cpu, 2), "tolerance": tolerance, "matched": False, "attempts": attempts}


def goal_features(goal):
    return features([lib.goal_value(goal, int(x), int(y)) for x, y in zip(*XS16)])


def run_seed(seed, path, a):
    concepts, log = load_concepts(path)
    model, info = train(seed, concepts, a.dreams, a.steps)
    goals = exam(seed, a.exam_goals)
    xs, ys = lib.probes()
    counts = {}
    for expression in log.solutions:
        for part in lib.subexpressions(expression):
            counts[part] = counts.get(part, 0) + 1
    frequent = sorted(concepts, key=lambda c: (-counts.get(c, 0), repr(c)))[:a.top_k]
    rng = random.Random(f"NC-081/random/{seed}")
    started = time.process_time()
    with torch.no_grad():
        scores = model(torch.tensor(np.stack([goal_features(g) for g in goals]))).numpy()
    inference = time.process_time() - started
    picks = lambda row: [concepts[i] for i in np.argsort(-row, kind="stable")[:a.top_k]]
    arms = {"all_concepts": lambda i: (concepts, a.budget), "model": lambda i: (picks(scores[i]), a.budget),
            "random_pick": lambda i: (rng.sample(concepts, min(a.top_k, len(concepts))), a.budget),
            "frequent": lambda i: (frequent, a.budget), "other_goal_examples": lambda i: (picks(scores[(i + 1) % len(goals)]), a.budget),
            "no_library": lambda i: ([], a.budget)}
    out = {}
    for arm, choose in arms.items():
        began = time.process_time()
        solved, used = [], 0
        for i, goal in enumerate(goals):
            seeds, budget = choose(i)
            ok, spent = solve(goal, seeds, budget, xs, ys)
            used += spent
            if ok:
                solved.append(i)
        cpu = time.process_time() - began
        out[arm] = {"solved": len(solved), "solved_goals": solved, "candidates": used, "cpu_seconds": round(cpu, 2)}
        print(seed, arm, len(solved), "of", len(goals), "cpu", round(cpu, 1), flush=True)
    # Equal total CPU: all concepts get the model's training, inference and search time, confirmed by measuring the arm itself.
    target = info["train_cpu_seconds"] + inference + out["model"]["cpu_seconds"]
    def baseline(budget):
        began = time.process_time()
        solved, used = [], 0
        for i, goal in enumerate(goals):
            ok, spent = solve(goal, concepts, budget, xs, ys)
            used += spent
            if ok:
                solved.append(i)
        return {"solved": len(solved), "solved_goals": solved, "candidates": used, "cpu_seconds": round(time.process_time() - began, 2)}
    start = max(a.budget, int(a.budget * target / max(out["all_concepts"]["cpu_seconds"], 1e-6)))
    out["all_concepts_equal_total_cpu"] = calibrate(baseline, target, start)
    print(seed, "all_concepts_equal_total_cpu", out["all_concepts_equal_total_cpu"]["solved"], "of", len(goals), "matched", out["all_concepts_equal_total_cpu"]["matched"], out["all_concepts_equal_total_cpu"]["attempts"], flush=True)
    torch.save({"state": model.state_dict(), "seed": seed}, a.out / f"selector_seed{seed}.pt")
    return {"library_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "library_head": log.head, "concepts": len(concepts), "exam_goals": len(goals),
            "exam_sha256": hashlib.sha256(json.dumps(goals).encode()).hexdigest(), "training": info, "inference_cpu_seconds": round(inference, 4),
            "checkpoint_sha256": hashlib.sha256((a.out / f"selector_seed{seed}.pt").read_bytes()).hexdigest(), "arms": out}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--libraries", nargs="+", required=True, help="seed=path to a LA-015 library.jsonl, copied here")
    p.add_argument("--dreams", type=int, default=30000)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--exam-goals", type=int, default=60)
    p.add_argument("--budget", type=int, default=BUDGET)
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--out", type=Path, default=HERE)
    a = p.parse_args()
    torch.set_num_threads(1)
    a.out.mkdir(parents=True, exist_ok=True)
    report = {"schema": "logos.nc081.results/1", "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items() if k != "libraries"},
              "la015_library_py_sha256": hashlib.sha256(LIB.read_bytes()).hexdigest(), "la015_library_source": LIB_SOURCE, "seeds": {}}
    for item in a.libraries:
        seed, path = item.split("=", 1)
        snapshot = a.out / f"library_seed{seed}.jsonl"
        if Path(path).resolve() != snapshot.resolve():
            shutil.copyfile(path, snapshot)
        report["seeds"][seed] = run_seed(int(seed), snapshot, a)
        (a.out / "selector_results.json").write_text(json.dumps(report, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
