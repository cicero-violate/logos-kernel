"""NC-083: can a learned proposer shorten real verified programs better than deterministic methods at equal CPU?

The exam is verified solutions from pinned LA-015 libraries, with held-out functions chosen by hash before any arm runs.
Every arm repeatedly offers smaller replacements for subexpressions through one frame, which accepts one only if it computes
the same function on all 65,536 pairs. The final program is confirmed by two separately written evaluators.
"""
import argparse
import hashlib
import importlib.util
import json
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
LIB_SHA256 = "0c876a9e8d90e9401323b0068a66acc9b6d1e9f3051f22a771fe9d2a2d098b8a"
TOLERANCE = 0.10  # Fixed before the run: an equal-CPU baseline must land within 10% of the learned arm's total.
CALIBRATION_TRIES = 4
EXAM_SIZE = 40
MAX_SUBTREE = 24  # Largest subexpression, in tree nodes, offered to a proposer.
MAX_TARGET = 16  # Longest learned proposal, in tokens.
PASSES = 2
SAMPLES = 4
NC023_LIMIT_SECONDS = 10
K_VALUES = (0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255)


def load_module(path, name, sha256=None):
    if sha256 is not None and hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha256:
        raise SystemExit(f"pinned library mismatch: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lib = load_module(HERE / "la015_library_pinned.py", "la015_library", LIB_SHA256)
ALL_X, ALL_Y = np.meshgrid(np.arange(256, dtype=np.uint8), np.arange(256, dtype=np.uint8), indexing="ij")
ALL_X, ALL_Y = ALL_X.ravel(), ALL_Y.ravel()
_rng = np.random.default_rng(83)
SAMPLE_INDEX = np.sort(_rng.choice(65536, 512, replace=False))
BINARY = ("ADD", "SUB", "AND", "OR", "XOR")
UNARY = ("SHL", "SHR")


def apply_op(name, a, b=None):
    if name == "ADD":
        return a + b
    if name == "SUB":
        return a - b
    if name == "AND":
        return a & b
    if name == "OR":
        return a | b
    if name == "XOR":
        return a ^ b
    if name == "SHL":
        return a << 1
    if name == "SHR":
        return a >> 1
    raise ValueError(name)


def values(expr, x=ALL_X, y=ALL_Y, memo=None):
    """All outputs as uint8, which wraps exactly as the 8-bit machine does. Shared parts are computed once."""
    memo = {} if memo is None else memo
    if expr in memo:
        return memo[expr]
    head = expr[0]
    if head == "X":
        out = x
    elif head == "Y":
        out = y
    elif head == "K":
        out = np.full_like(x, expr[1] & 255)
    elif head in UNARY:
        out = apply_op(head, values(expr[1], x, y, memo))
    else:
        out = apply_op(head, values(expr[1], x, y, memo), values(expr[2], x, y, memo))
    memo[expr] = out
    return out


def function_key(expr, memo=None):
    return hashlib.sha256(values(expr, memo=memo).tobytes()).hexdigest()


def is_held_out(key):
    """One function in four is reserved for the exam, by hash, before anything runs."""
    return int(key[:8], 16) % 4 == 0


def tree_size(expr):
    return 1 + sum(tree_size(e) for e in expr[1:] if isinstance(e, tuple))


def parts(expr, out=None):
    """Distinct subexpressions, children before parents."""
    out = {} if out is None else out
    if expr not in out:
        for child in expr[1:]:
            if isinstance(child, tuple):
                parts(child, out)
        out[expr] = None
    return out


def dag_ops(expr):
    """Operations with shared subexpressions counted once, as NC-023 counts them. This is the score."""
    return sum(e[0] not in ("X", "Y", "K") for e in parts(expr))


def cost(expr):
    return dag_ops(expr), tree_size(expr)


def confirmed_equal(a, b):
    """Two separately written evaluators on all 65,536 pairs: the NumPy one here and LA-015's stack machine."""
    if not np.array_equal(values(a), values(b)):
        return False
    code_a, code_b = lib.compile_stack(a), lib.compile_stack(b)
    return all(lib.run_stack(code_a, x, y) == lib.run_stack(code_b, x, y) for x, y in zip(ALL_X.tolist(), ALL_Y.tolist()))


def replace(expr, swaps):
    if expr in swaps:
        return swaps[expr]
    if not any(isinstance(e, tuple) for e in expr[1:]):
        return expr
    return (expr[0], *[replace(e, swaps) if isinstance(e, tuple) else e for e in expr[1:]])


def shrink(source, propose):
    """The shared frame. A proposer sees subexpressions and may offer anything: only exact, smaller replacements are accepted."""
    current = source
    offered = accepted = 0
    for _ in range(PASSES):
        subtrees = [e for e in parts(current) if e[0] not in ("X", "Y", "K") and tree_size(e) <= MAX_SUBTREE]
        if not subtrees:
            break
        swaps = {}
        for subtree, candidates in zip(subtrees, propose(subtrees)):
            want = values(subtree)
            best = None
            for candidate in candidates:
                offered += 1
                if cost(candidate) < cost(best or subtree) and np.array_equal(values(candidate), want):
                    best = candidate
            if best is not None:
                swaps[subtree] = best
        # Larger subexpressions first, so a replacement of a whole part wins over changes inside it.
        ordered = dict(sorted(swaps.items(), key=lambda kv: -tree_size(kv[0])))
        new = replace(current, ordered)
        if cost(new) < cost(current) and np.array_equal(values(new), values(source)):
            accepted += len(swaps)
            current = new
        else:
            break
    if current != source and not confirmed_equal(source, current):
        raise AssertionError("a shortened program failed exhaustive confirmation")
    return current, {"offered": offered, "accepted": accepted}


class Table:
    """Exact shortest forms by bottom-up enumeration, keyed on sampled outputs and always confirmed before use."""

    def __init__(self, max_tried=1_500_000, max_entries=200_000):
        started = time.process_time()
        x, y = ALL_X[SAMPLE_INDEX], ALL_Y[SAMPLE_INDEX]
        self.best, by_size = {}, {}
        def add(expr, vector, size):
            key = vector.tobytes()
            if key not in self.best:
                self.best[key] = expr
                by_size.setdefault(size, []).append((expr, vector))
        for leaf in (("X",), ("Y",), *[("K", k) for k in K_VALUES]):
            add(leaf, values(leaf, x, y), 1)
        tried, size = 0, 1
        while tried < max_tried and len(self.best) < max_entries:
            size += 1
            for name in UNARY:
                for expr, vector in by_size.get(size - 1, []):
                    tried += 1
                    add((name, expr), apply_op(name, vector), size)
            for left in range(1, size - 1):
                for a, va in by_size.get(left, []):
                    for b, vb in by_size.get(size - 1 - left, []):
                        for name in BINARY:
                            tried += 1
                            add((name, a, b), apply_op(name, va, vb), size)
                        if tried >= max_tried or len(self.best) >= max_entries:
                            break
                    if tried >= max_tried or len(self.best) >= max_entries:
                        break
                if tried >= max_tried or len(self.best) >= max_entries:
                    break
        self.x, self.y = x, y
        self.entries = list(self.best.values())
        self.info = {"entries": len(self.entries), "tried": tried, "largest_size": size, "cpu_seconds": round(time.process_time() - started, 2)}

    def lookup(self, expr):
        return self.best.get(values(expr, self.x, self.y).tobytes())

    def propose(self, subtrees):
        return [[found] if (found := self.lookup(e)) is not None else [] for e in subtrees]


def load_exam(path, count=EXAM_SIZE):
    """Real verified solutions, one per function, with held-out functions only, in hash order."""
    programs = {}
    for expr in lib.Library(path).solutions:
        key = function_key(expr)
        if is_held_out(key) and dag_ops(expr) >= 3 and (key not in programs or cost(expr) < cost(programs[key])):
            programs[key] = expr
    return [programs[k] for k in sorted(programs)][:count], len(programs)


# Tokens: PAD BOS EOS SEP X Y, seven operators, then one per byte value for constants and example outputs.
NAMES = ("PAD", "BOS", "EOS", "SEP", "X", "Y", "K", *BINARY, *UNARY)
TOKEN = {name: i for i, name in enumerate(NAMES)}
BYTE = len(NAMES)
VOCAB = BYTE + 256
MAX_SOURCE = 48
EXAMPLES = 12
EXAMPLE_INDEX = SAMPLE_INDEX[:: len(SAMPLE_INDEX) // EXAMPLES][:EXAMPLES]


def encode(expr, out=None):
    out = [] if out is None else out
    if expr[0] == "K":
        out += [TOKEN["K"], BYTE + (expr[1] & 255)]
    else:
        out.append(TOKEN[expr[0]])
        for child in expr[1:]:
            encode(child, out)
    return out


def decode(tokens):
    """Prefix tokens back to an expression, or None if malformed or not fully consumed."""
    position = 0
    def read():
        nonlocal position
        if position >= len(tokens):
            raise ValueError
        token = tokens[position]
        position += 1
        if token == TOKEN["X"]:
            return ("X",)
        if token == TOKEN["Y"]:
            return ("Y",)
        if token == TOKEN["K"]:
            if position >= len(tokens) or tokens[position] < BYTE:
                raise ValueError
            position += 1
            return ("K", tokens[position - 1] - BYTE)
        if token < BYTE and NAMES[token] in UNARY:
            return (NAMES[token], read())
        if token < BYTE and NAMES[token] in BINARY:
            a = read()
            return (NAMES[token], a, read())
        raise ValueError
    try:
        expr = read()
    except (ValueError, RecursionError):
        return None
    return expr if position == len(tokens) else None


def source_tokens(expr):
    """The program's tokens, then its outputs on fixed examples, so a proposer sees both form and behaviour."""
    examples = values(expr)[EXAMPLE_INDEX].tolist()
    tokens = encode(expr)[:MAX_SOURCE - EXAMPLES - 1] + [TOKEN["SEP"]] + [BYTE + v for v in examples]
    return tokens + [0] * (MAX_SOURCE - len(tokens))


class Proposer(nn.Module):
    def __init__(self, dim=96, layers=2, heads=4):
        super().__init__()
        self.token = nn.Embedding(VOCAB, dim, padding_idx=0)
        self.source_position = nn.Embedding(MAX_SOURCE, dim)
        self.target_position = nn.Embedding(MAX_TARGET + 2, dim)
        self.net = nn.Transformer(dim, heads, layers, layers, dim * 2, dropout=0.0, batch_first=True, activation="gelu")
        self.head = nn.Linear(dim, VOCAB)

    def memory(self, source):
        h = self.token(source) + self.source_position(torch.arange(source.shape[1]))
        return self.net.encoder(h, src_key_padding_mask=source == 0), source == 0

    def step(self, target, memory, padding):
        h = self.token(target) + self.target_position(torch.arange(target.shape[1]))
        mask = nn.Transformer.generate_square_subsequent_mask(target.shape[1])
        return self.head(self.net.decoder(h, memory, tgt_mask=mask, memory_key_padding_mask=padding, tgt_is_causal=True))


def make_pairs(table, count, seed):
    """Larger programs with a known shortest form on the training side. Held-out functions never enter, as source or target."""
    rng = random.Random(f"NC-083/pairs/{seed}")
    small = [e for e in table.entries if tree_size(e) <= 5]
    sources, targets, skipped = [], [], 0
    def compose():
        expr = rng.choice(small)
        for _ in range(rng.randint(1, 3)):
            expr = (rng.choice(BINARY), expr, rng.choice(small)) if rng.random() < .85 else (rng.choice(UNARY), expr)
        return expr
    while len(sources) < count:
        source = compose()
        if not 3 <= tree_size(source) <= MAX_SUBTREE:
            continue
        target = table.lookup(source)
        if target is None or cost(target) >= cost(source) or len(encode(target)) > MAX_TARGET:
            continue
        key = function_key(source)
        if is_held_out(key):
            skipped += 1
            continue
        if not np.array_equal(values(source), values(target)):  # The table keys on samples, so confirm before teaching.
            continue
        sources.append(source_tokens(source))
        tokens = [TOKEN["BOS"], *encode(target), TOKEN["EOS"]]
        targets.append(tokens + [0] * (MAX_TARGET + 2 - len(tokens)))
    return torch.tensor(sources), torch.tensor(targets), skipped


def train(seed, table, pairs, steps, batch):
    started = time.process_time()
    torch.manual_seed(seed)
    source, target, skipped = make_pairs(table, pairs, seed)
    model = Proposer()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        pick = torch.randint(len(source), (batch,), generator=gen)
        s, t = source[pick], target[pick]
        memory, padding = model.memory(s)
        logits = model.step(t[:, :-1], memory, padding)
        loss = nn.functional.cross_entropy(logits.reshape(-1, VOCAB), t[:, 1:].reshape(-1), ignore_index=0)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    model.eval()
    return model, {"pairs": pairs, "held_out_skipped": skipped, "steps": steps, "final_loss": round(loss.item(), 4), "parameters": sum(p.numel() for p in model.parameters()), "train_cpu_seconds": round(time.process_time() - started, 1)}


@torch.no_grad()
def model_proposals(model, subtrees, seed, shift=0):
    """Greedy plus sampled decodes for each subexpression. With shift, the model sees another subexpression: the control."""
    if not subtrees:
        return []
    gen = torch.Generator().manual_seed(seed)
    shown = [subtrees[(i + shift) % len(subtrees)] for i in range(len(subtrees))]
    source = torch.tensor([source_tokens(e) for e in shown]).repeat_interleave(SAMPLES, 0)
    memory, padding = model.memory(source)
    target = torch.full((len(source), 1), TOKEN["BOS"], dtype=torch.long)
    for _ in range(MAX_TARGET + 1):
        logits = model.step(target, memory, padding)[:, -1]
        logits[:, 0] = -1e9
        greedy = logits.argmax(1)
        sampled = torch.multinomial(torch.softmax(logits, 1), 1, generator=gen).flatten()
        target = torch.cat((target, torch.where(torch.arange(len(source)) % SAMPLES == 0, greedy, sampled)[:, None]), 1)
    out = []
    for i in range(len(subtrees)):
        offers = []
        for row in target[i * SAMPLES:(i + 1) * SAMPLES, 1:].tolist():
            if TOKEN["EOS"] in row:
                row = row[:row.index(TOKEN["EOS"])]
            found = decode(row)
            if found is not None:
                offers.append(found)
        out.append(offers)
    return out


def search_proposals(budget):
    xs, ys = lib.probes()
    def propose(subtrees):
        out = []
        for e in subtrees:
            target = tuple(lib.evaluate(e, xs, ys))
            found, _ = lib.search(target, [], budget, xs, ys)
            out.append([found] if found is not None else [])
        return out
    return propose


def random_proposals(table, seed):
    rng = random.Random(f"NC-083/random/{seed}")
    def propose(subtrees):
        return [[rng.choice(table.entries) for _ in range(SAMPLES)] for _ in subtrees]
    return propose


def run_arm(exam, propose_for):
    started = time.process_time()
    rows, offered, accepted = [], 0, 0
    for index, program in enumerate(exam):
        result, info = shrink(program, propose_for(index))
        rows.append(result)
        offered += info["offered"]
        accepted += info["accepted"]
    return {"total_ops": sum(map(dag_ops, rows)), "total_tree_nodes": sum(map(tree_size, rows)), "improved": sum(cost(a) < cost(b) for a, b in zip(rows, exam)),
            "offered": offered, "accepted": accepted, "cpu_seconds": round(time.process_time() - started, 2), "per_program_ops": [dag_ops(r) for r in rows]}, rows


def calibrate(run, target_cpu, start_budget, tolerance=TOLERANCE, tries=CALIBRATION_TRIES):
    """Find a budget whose measured CPU lands within tolerance of the target. All attempts are kept, and matched says if one did."""
    budget, attempts = start_budget, []
    for _ in range(tries):
        result = run(budget)
        attempts.append({"budget_per_subtree": budget, "total_ops": result["total_ops"], "cpu_seconds": result["cpu_seconds"]})
        if abs(result["cpu_seconds"] - target_cpu) <= tolerance * target_cpu:
            return {**result, "budget_per_subtree": budget, "target_cpu_seconds": round(target_cpu, 2), "matched": True, "attempts": attempts}
        budget = max(10, int(budget * target_cpu / max(result["cpu_seconds"], 1e-6)))
    return {**result, "budget_per_subtree": attempts[-1]["budget_per_subtree"], "target_cpu_seconds": round(target_cpu, 2), "matched": False, "attempts": attempts}


def to_nc023(expr, isa):
    """An expression in NC-023's typed form, at width 8, unsigned."""
    if expr[0] in ("X", "Y"):
        return {"kind": "bv.read", "name": expr[0], "width": 8, "signedness": "unsigned"}
    if expr[0] == "K":
        return isa.lit(expr[1] & 255, 8, "unsigned")
    return {"kind": "bv." + expr[0].lower(), "width": 8, "signedness": "unsigned", "args": [to_nc023(e, isa) for e in expr[1:]]}


class OutOfTime(Exception):
    pass


def run_nc023(exam):
    """NC-023's saturation search on each whole program under a fixed CPU limit. A timeout keeps the source and is counted."""
    sys.path.insert(0, str(HERE.parent / "NC-023"))
    import symbolic_search as ss
    isa = ss.ref.isa
    def on_limit(*_):
        raise OutOfTime
    previous = signal.signal(signal.SIGPROF, on_limit)
    started = time.process_time()
    per, timeouts, errors = [], 0, 0
    try:
        for program in exam:
            case = {"domain": "bitvector", "width": 8, "signedness": "unsigned", "lhs": to_nc023(program, isa)}
            before = dag_ops(program)
            signal.setitimer(signal.ITIMER_PROF, NC023_LIMIT_SECONDS)
            try:
                per.append(min(before, ss.search(case)["operations"]["dag_ops"]))
            except OutOfTime:
                timeouts += 1
                per.append(before)
            except (ValueError, TypeError, KeyError, ss.ref.isa.ISAError):
                errors += 1
                per.append(before)
            finally:
                signal.setitimer(signal.ITIMER_PROF, 0)
    finally:
        signal.signal(signal.SIGPROF, previous)
    return {"total_ops": sum(per), "improved": sum(a < dag_ops(b) for a, b in zip(per, exam)), "timeouts": timeouts, "errors": errors, "limit_seconds_per_program": NC023_LIMIT_SECONDS,
            "cpu_seconds": round(time.process_time() - started, 2), "per_program_ops": per}


def executed_steps(before, after, seed):
    """Run programs on NC-082's Transformer and NC-080's operators where both versions fit its machine, with no teacher forcing."""
    controller = load_module(HERE.parent / "NC-082" / "controller.py", "nc082_controller")
    model = controller.Controller()
    model.load_state_dict(torch.load(HERE.parent / "NC-082" / f"controller_seed{seed}.pt", weights_only=True)["state"])
    model.eval()
    alu = controller.NeuralALU(seed)
    rng = random.Random(f"NC-083/inputs/{seed}")
    inputs = [(rng.randrange(256), rng.randrange(256)) for _ in range(16)]
    out = {"programs": 0, "steps_before": 0, "steps_after": 0, "rollouts_correct": 0, "rollouts": 0, "does_not_fit": 0}
    for a, b in zip(before, after):
        if a == b:
            continue
        pa, pb = controller.compile_concept(a), controller.compile_concept(b) if b[0] not in ("X", "Y", "K") else None
        if pa is None or pb is None:
            out["does_not_fit"] += 1
            continue
        for program, field in ((pa, "steps_before"), (pb, "steps_after")):
            got, done, steps = controller.rollout(model, alu, program, inputs)
            out[field] += sum(steps)
            want = [int(values(a, np.array([x], dtype=np.uint8), np.array([y], dtype=np.uint8))[0]) for x, y in inputs]
            out["rollouts_correct"] += sum(d and g == w for g, w, d in zip(got, want, done))
            out["rollouts"] += len(inputs)
        out["programs"] += 1
    return out


def run_seed(seed, path, a):
    exam, available = load_exam(path, a.exam)
    table = Table()
    print(seed, "exam", len(exam), "of", available, "source ops", sum(map(dag_ops, exam)), "table", table.info, flush=True)
    model, info = train(seed, table, a.pairs, a.steps, a.batch)
    print(seed, "trained", info, flush=True)
    arms = {}
    arms["model"], model_rows = run_arm(exam, lambda i: lambda subtrees: model_proposals(model, subtrees, seed * 1000 + i))
    arms["other_program_control"], _ = run_arm(exam, lambda i: lambda subtrees: model_proposals(model, subtrees, seed * 1000 + i, shift=max(1, len(subtrees) // 2)))
    arms["random_proposals"], _ = run_arm(exam, lambda i: random_proposals(table, seed))
    arms["exact_table"], table_rows = run_arm(exam, lambda i: table.propose)
    arms["exact_table"]["cpu_seconds_including_table"] = round(arms["exact_table"]["cpu_seconds"] + table.info["cpu_seconds"], 2)
    arms["nc023_saturation"] = run_nc023(exam)
    target = info["train_cpu_seconds"] + table.info["cpu_seconds"] + arms["model"]["cpu_seconds"]
    def baseline(budget):
        return run_arm(exam, lambda i: search_proposals(budget))[0]
    first = baseline(a.search_start)
    start = max(10, int(a.search_start * target / max(first["cpu_seconds"], 1e-6)))
    arms["search_equal_total_cpu"] = calibrate(baseline, target, start)
    arms["search_equal_total_cpu"]["start_probe"] = {"budget_per_subtree": a.search_start, "total_ops": first["total_ops"], "cpu_seconds": first["cpu_seconds"]}
    for arm, row in arms.items():
        print(seed, arm, "ops", row["total_ops"], "improved", row["improved"], "cpu", row["cpu_seconds"], flush=True)
    torch.save({"state": model.state_dict(), "seed": seed}, a.out / f"proposer_seed{seed}.pt")
    return {"library_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "exam_available": available, "exam_size": len(exam), "exam_sha256": hashlib.sha256(json.dumps(exam).encode()).hexdigest(),
            "source_total_ops": sum(map(dag_ops, exam)), "source_total_tree_nodes": sum(map(tree_size, exam)), "source_per_program_ops": [dag_ops(e) for e in exam], "table": table.info, "training": info, "learned_total_cpu_seconds": round(target, 2), "arms": arms,
            "execution_on_nc082_model_arm": executed_steps(exam, model_rows, seed), "execution_on_nc082_exact_table_arm": executed_steps(exam, table_rows, seed),
            "checkpoint_sha256": hashlib.sha256((a.out / f"proposer_seed{seed}.pt").read_bytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 19, 29])
    p.add_argument("--exam", type=int, default=EXAM_SIZE)
    p.add_argument("--pairs", type=int, default=30000)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--search-start", type=int, default=2000)
    p.add_argument("--out", type=Path, default=HERE)
    a = p.parse_args()
    torch.set_num_threads(2)
    a.out.mkdir(parents=True, exist_ok=True)
    report = {"schema": "logos.nc083.results/1", "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, "tolerance": TOLERANCE, "library_py_sha256": LIB_SHA256, "seeds": {}}
    for seed in a.seeds:
        report["seeds"][str(seed)] = run_seed(seed, HERE / f"library_seed{seed}.jsonl", a)
        (a.out / "shorten_results.json").write_text(json.dumps(report, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
