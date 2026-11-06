"""NC-078 experiment: meta-learned synthesis against ablations and symbolic controls on equal candidate budgets."""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import synth  # noqa: E402

REGS = list(synth.isa.REGS)
TOKENS = ["<pad>", "<bos>", "<eos>", "<none>", *synth.OPS, *REGS, *[f"v{i}" for i in range(synth.VALUES)]]
ID = {t: i for i, t in enumerate(TOKENS)}
PAD, BOS, EOS, NONE = (ID[t] for t in ("<pad>", "<bos>", "<eos>", "<none>"))
VAL = [ID[f"v{i}"] for i in range(synth.VALUES)]
EXAMPLES = 8
PROGRAM_TOKENS = synth.MAX_LENGTH * 4 + 1
PREFIX = EXAMPLES * 3
LENGTH = PREFIX + 1 + PROGRAM_TOKENS
ALLOWED = [[ID[o] for o in synth.OPS] + [EOS], [ID[r] for r in REGS], [ID[r] for r in REGS] + VAL, [ID[r] for r in REGS] + [NONE]]
BUDGETS = (64, 128, 256, 512)
ROUNDS = 8
PER_ROUND = 64
CONTROL_LIMIT = 250_000


def encode_body(body):
    out = []
    for item in body:
        op = item["op"]
        if op == "LOADI":
            out += [ID[op], ID[item["dst"]], VAL[item["imm"]], NONE]
        elif op in synth.isa.BINARY:
            out += [ID[op], ID[item["dst"]], ID[item["a"]], ID[item["b"]]]
        else:
            out += [ID[op], ID[item["dst"]], ID[item["src"]], NONE]
    out.append(EOS)
    return out + [PAD] * (PROGRAM_TOKENS - len(out))


def decode_tokens(tokens):
    """Strict decoding; a malformed candidate returns None and still uses budget."""
    body = []
    names = {i: t for t, i in ID.items()}
    for start in range(0, len(tokens), 4):
        if tokens[start] == EOS:
            return body or None
        chunk = tokens[start:start + 4]
        if len(chunk) < 4:
            return None
        op, dst, a, b = (names[t] for t in chunk)
        if op == "LOADI":
            if not a.startswith("v") or b != "<none>":
                return None
            body.append({"op": op, "dst": dst, "imm": int(a[1:])})
        elif op in synth.isa.BINARY:
            if a not in REGS or b not in REGS:
                return None
            body.append({"op": op, "dst": dst, "a": a, "b": b})
        else:
            if a not in REGS or b != "<none>":
                return None
            body.append({"op": op, "dst": dst, "src": a})
    return None


class Model(nn.Module):
    def __init__(self, width=96, layers=3, heads=4):
        super().__init__()
        self.tokens = nn.Embedding(len(TOKENS), width, padding_idx=PAD)
        self.positions = nn.Embedding(LENGTH, width)
        layer = nn.TransformerEncoderLayer(width, heads, width * 2, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(width, len(TOKENS))
        mask = torch.triu(torch.ones(LENGTH, LENGTH, dtype=torch.bool), 1)
        mask[:PREFIX + 1, :PREFIX + 1] = False  # Examples see each other; a program prefix never sees ahead.
        self.register_buffer("mask", mask)

    def forward(self, x):
        n = x.shape[1]
        h = self.tokens(x) + self.positions(torch.arange(n))
        return self.out(self.encoder(h, mask=self.mask[:n, :n]))


def example_tokens(inputs, outputs):
    """inputs are domain indices, one row per task."""
    x = np.stack([np.array(VAL)[synth.XS[inputs]], np.array(VAL)[synth.YS[inputs]], np.array(VAL)[outputs]], axis=2)
    return x.reshape(len(inputs), -1)


def train(seed, args, programs, out):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tables = np.stack([t for _, t in programs])
    bodies = np.array([encode_body(b) for b, _ in programs])
    model = Model()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    start = time.process_time()
    model.train()
    for _ in range(args.steps):
        idx = rng.integers(0, len(bodies), args.batch)
        inputs = np.stack([rng.choice(synth.DOMAIN, EXAMPLES, replace=False) for _ in idx])
        ex = example_tokens(inputs, np.take_along_axis(tables[idx], inputs, axis=1))
        target = torch.tensor(bodies[idx])
        x = torch.cat([torch.tensor(ex), torch.full((len(idx), 1), BOS), target[:, :-1]], dim=1)
        logits = model(x)[:, PREFIX:]
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=PAD)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    path = out / f"model_seed{seed}.pt"
    torch.save({"state": model.state_dict(), "seed": seed}, path)
    return model, {"seed": seed, "cpu_seconds": round(time.process_time() - start, 1), "final_loss": round(loss.item(), 4),
                   "parameters": sum(p.numel() for p in model.parameters()), "checkpoint": path.name,
                   "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def sample(model, ex_tokens, count):
    model.eval()
    x = torch.cat([torch.tensor(ex_tokens).repeat(count, 1), torch.full((count, 1), BOS)], dim=1)
    done = torch.zeros(count, dtype=torch.bool)
    with torch.no_grad():
        for position in range(PROGRAM_TOKENS - 1):
            logits = model(x)[:, -1]
            allow = torch.full_like(logits, float("-inf"))
            allow[:, ALLOWED[position % 4]] = 0
            next_token = torch.multinomial(torch.softmax(logits + allow, -1), 1)
            next_token[done] = PAD
            done |= next_token[:, 0] == EOS
            x = torch.cat([x, next_token], dim=1)
            if done.all():
                break
    return [row[row != PAD].tolist() for row in x[:, PREFIX + 1:]]


def neural_search(model, goal, seed, name, shuffled=False):
    """Propose, test, and revise with a counterexample chosen by a verifier the model cannot read."""
    rng = np.random.default_rng([seed, int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)])
    inputs = list(rng.choice(synth.DOMAIN, EXAMPLES, replace=False))
    used = 0
    revisions = 0
    for _ in range(ROUNDS):
        outputs = goal[inputs]
        if shuffled:
            outputs = rng.permutation(goal)[:EXAMPLES]
        ex = example_tokens(np.array([inputs]), np.array([outputs]))
        counterexample = None
        for tokens in sample(model, ex, PER_ROUND):
            used += 1
            body = decode_tokens(tokens)
            if body is None:
                continue
            table = synth.evaluate(body)
            if np.array_equal(table, goal):
                return {"solved": synth.verify_with_reference(body, goal), "candidates": used, "program": body, "revisions": revisions}
            if counterexample is None and np.array_equal(table[inputs], goal[inputs]):
                counterexample = int(rng.choice(np.flatnonzero(table != goal)))
        if counterexample is not None and not shuffled:
            inputs = [*inputs[1:], counterexample]
            revisions += 1
    return {"solved": False, "candidates": used, "program": None, "revisions": revisions}


def random_search(goal, seed, name):
    rng = random.Random(f"{seed}|{name}|random")
    for used in range(1, ROUNDS * PER_ROUND + 1):
        body = synth.random_body(rng)
        if np.array_equal(synth.evaluate(body), goal):
            return {"solved": synth.verify_with_reference(body, goal), "candidates": used, "program": body}
    return {"solved": False, "candidates": ROUNDS * PER_ROUND, "program": None}


def enumeration_index(goals):
    """Candidates shortest-first enumeration needs to reach each goal, up to CONTROL_LIMIT."""
    wanted = {synth.table_key(t): n for n, t in goals.items()}
    found = {}
    for used, body in enumerate(synth.enumerate_shortest_first(), 1):
        if used > CONTROL_LIMIT or len(found) == len(wanted):
            break
        name = wanted.get(synth.table_key(synth.evaluate(body)))
        if name and name not in found:
            found[name] = {"candidates": used, "program": body}
    return found


def tally(rows):
    return {str(k): sum(1 for r in rows.values() if r["solved"] and r["candidates"] <= k) for k in BUDGETS}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--train-programs", type=int, default=40000)
    p.add_argument("--unseen", type=int, default=40)
    p.add_argument("--named-limit", type=int, default=None, help="Quick checks only; full runs use every named goal")
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 19, 29])
    p.add_argument("--out", type=Path, default=HERE)
    a = p.parse_args()
    torch.set_num_threads(2)
    a.out.mkdir(parents=True, exist_ok=True)
    named = dict(list(synth.named_tables().items())[:a.named_limit])
    unseen = synth.unseen_goals(a.unseen)
    goals = {**named, **unseen}
    control = enumeration_index(goals)
    print("enumeration solved", len(control), "of", len(goals), "within", CONTROL_LIMIT, flush=True)
    report = {"schema": "logos.nc078.results/1", "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
              "budgets": BUDGETS, "goals": {"named": list(named), "unseen": len(unseen)},
              "enumeration": {"limit": CONTROL_LIMIT, "needed": {n: control[n]["candidates"] for n in control}}, "seeds": []}
    def split(rows, names):
        return {n: rows[n] for n in names}
    proposals = (a.out / "proposals.jsonl").open("w")
    enum_rows = {n: {"solved": n in control, "candidates": control[n]["candidates"] if n in control else CONTROL_LIMIT} for n in goals}
    report["enumeration"]["solved_by_budget"] = {"named": tally(split(enum_rows, named)), "unseen": tally(split(enum_rows, unseen))}
    for seed in a.seeds:
        programs = synth.training_programs(seed, a.train_programs)
        model, info = train(seed, a, programs, a.out)
        info.update(train_programs=len(programs), train_functions=len({synth.table_key(t) for _, t in programs}))
        untrained_state = Model()
        arms = {"neural": lambda n, g: neural_search(model, g, seed, n),
                "shuffled_examples": lambda n, g: neural_search(model, g, seed, n, shuffled=True),
                "untrained": lambda n, g: neural_search(untrained_state, g, seed, n),
                "random": lambda n, g: random_search(g, seed, n)}
        result = {"info": info, "arms": {}}
        for arm, fn in arms.items():
            started = time.process_time()
            rows = {n: fn(n, g) for n, g in goals.items()}
            for n, r in rows.items():
                if r["solved"]:
                    proposals.write(json.dumps({"seed": seed, "arm": arm, "goal": n, "candidates": r["candidates"], "program": r["program"], "revisions": r.get("revisions")}) + "\n")
            proposals.flush()
            result["arms"][arm] = {"cpu_seconds": round(time.process_time() - started, 1),
                                   "named": tally(split(rows, named)), "unseen": tally(split(rows, unseen)),
                                   "named_solved": sorted(n for n in named if rows[n]["solved"]),
                                   "revisions": sum(r.get("revisions", 0) for r in rows.values())}
            print(seed, arm, "named", result["arms"][arm]["named"], "unseen", result["arms"][arm]["unseen"], "cpu", result["arms"][arm]["cpu_seconds"], flush=True)
        report["seeds"].append(result)
        (a.out / "results.json").write_text(json.dumps(report, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
