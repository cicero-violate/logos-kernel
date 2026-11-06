"""NC-080: learned bit-local operators execute verified recursive concepts, after an exact behavioral quotient.

The recipe follows latticeforge_neural_isa_test: one small network learns an exhaustive local truth table through a causal
meaning layer, and the host applies it at each bit. The host iterates bits, routes carries and wires concepts; the network
computes each bit. Concepts come from a read-only snapshot of a verified library. No concept is invented here.
"""
import argparse
import hashlib
import itertools
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
OPS = ("ADD", "SUB", "AND", "OR", "XOR", "SHL", "SHR")
WIDTH = 8
GRID = np.arange(256, dtype=np.int64)
XS, YS = (v.ravel() for v in np.meshgrid(GRID, GRID, indexing="ij"))


def local_rule(op, a, b, carry, previous, following):
    if op == "ADD":
        total = a + b + carry
        return total % 2, int(total >= 2)
    if op == "SUB":
        total = a - b - carry
        return total % 2, int(total < 0)
    if op == "AND":
        return a & b, 0
    if op == "OR":
        return a | b, 0
    if op == "XOR":
        return a ^ b, 0
    return (previous, 0) if op == "SHL" else (following, 0)


def truth_table():
    x, y, ids = [], [], []
    for j, op in enumerate(OPS):
        for a, b, c, p, f in itertools.product((0, 1), repeat=5):
            x.append([int(k == j) for k in range(len(OPS))] + [a, b, c, p, f])
            y.append(list(local_rule(op, a, b, c, p, f)))
            ids.append(j)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(ids)


class NeuralOperators(nn.Module):
    def __init__(self):
        super().__init__()
        n = len(OPS)
        self.meaning = nn.Linear(n, n)
        self.bit = nn.Sequential(nn.Linear(n + 5, 96), nn.Tanh(), nn.Linear(96, 96), nn.Tanh(), nn.Linear(96, 2))
        self.permute = None

    def forward(self, features):
        # Numeric execution sees only the learned meaning, never the raw opcode.
        meaning = self.meaning(features[:, :len(OPS)]).softmax(1)
        if self.permute is not None:
            meaning = meaning[:, self.permute]
        return self.bit(torch.cat((meaning, features[:, len(OPS):]), 1))


def train(seed, stage_epochs=(500, 600, 900)):
    """Progressive stages with replay, as in the original recipe."""
    torch.manual_seed(seed)
    x, y, ids = truth_table()
    model = NeuralOperators()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    stages = [("ADD",), ("ADD", "SUB", "SHL", "SHR"), OPS]
    report = []
    for choices, epochs in zip(stages, stage_epochs):
        idx = torch.tensor([i for i, v in enumerate(ids.tolist()) if OPS[v] in choices])
        for _ in range(epochs):
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(x[idx]), y[idx]) + 0.2 * F.cross_entropy(model.meaning(x[idx, :len(OPS)]), ids[idx])
            loss.backward()
            opt.step()
        with torch.no_grad():
            report.append({"operations": list(choices), "local_accuracy": (model(x[idx]).sigmoid().ge(.5).float() == y[idx]).all(1).float().mean().item()})
    with torch.no_grad():
        overall = (model(x).sigmoid().ge(.5).float() == y).all(1).float().mean().item()
    return model, {"stages": report, "local_truth_rows": len(x), "final_local_accuracy": overall}


def bits(values):
    return torch.tensor(np.stack([(values >> i) & 1 for i in range(WIDTH)], 1), dtype=torch.float32)


def network_table(model):
    """The network's own decision for all 224 possible inputs. Lookups from this equal running it, since inputs are binary."""
    x, _, _ = truth_table()
    with torch.no_grad():
        model._table = model(x).sigmoid().ge(.5).float().reshape(len(OPS), 32, 2)


def neural_op(model, name, a, b):
    """Apply the learned rule at each bit. The host only moves bits and the carry between positions."""
    n = a.shape[0]
    table = model._table[OPS.index(name)]
    out = []
    carry = torch.zeros(n)
    for i in range(WIDTH):
        previous = a[:, i - 1] if i > 0 else torch.zeros(n)
        following = a[:, i + 1] if i < WIDTH - 1 else torch.zeros(n)
        index = (a[:, i] * 16 + b[:, i] * 8 + carry * 4 + previous * 2 + following).long()
        decided = table[index]
        out.append(decided[:, 0])
        carry = decided[:, 1]
    return torch.stack(out, 1)


def neural_eval(model, expr, memo):
    if expr in memo:
        return memo[expr]
    head = expr[0]
    if head == "X":
        out = bits(XS)
    elif head == "Y":
        out = bits(YS)
    elif head == "K":
        out = bits(np.full(XS.shape, expr[1] & 255))
    elif head in ("SHL", "SHR"):
        a = neural_eval(model, expr[1], memo)
        out = neural_op(model, head, a, torch.zeros_like(a))
    else:
        out = neural_op(model, head, neural_eval(model, expr[1], memo), neural_eval(model, expr[2], memo))
    memo[expr] = out
    return out


def exact_eval(expr, memo):
    """Exact integers on all 65,536 pairs, used to judge the network and define the quotient."""
    if expr in memo:
        return memo[expr]
    head = expr[0]
    if head == "X":
        out = XS
    elif head == "Y":
        out = YS
    elif head == "K":
        out = np.full(XS.shape, expr[1] & 255)
    elif head == "SHL":
        out = (exact_eval(expr[1], memo) << 1) & 255
    elif head == "SHR":
        out = exact_eval(expr[1], memo) >> 1
    else:
        a, b = exact_eval(expr[1], memo), exact_eval(expr[2], memo)
        out = {"ADD": (a + b) & 255, "SUB": (a - b) & 255, "AND": a & b, "OR": a | b, "XOR": a ^ b}[head]
    memo[expr] = out
    return out


def to_tuple(node):
    return (node[0], *[to_tuple(e) if isinstance(e, list) else e for e in node[1:]])


def size(expr):
    return 1 + sum(size(e) for e in expr[1:] if isinstance(e, tuple))


def parts(expr, seen):
    if expr in seen:
        return
    seen.add(expr)
    for e in expr[1:]:
        if isinstance(e, tuple):
            parts(e, seen)


def quotient(expressions):
    """Exact classes over all pairs. The canonical form is smallest, then lexically first."""
    memo, classes = {}, {}
    for expr in expressions:
        key = hashlib.sha256(exact_eval(expr, memo).astype(np.uint8).tobytes()).hexdigest()
        classes.setdefault(key, []).append(expr)
    return {k: sorted(v, key=lambda e: (size(e), repr(e)))[0] for k, v in classes.items()}, memo


def approximate_merge_count(expressions, memo):
    """A 64-probe signature, reported only to show how many distinct functions it would wrongly merge."""
    rng = np.random.default_rng(5)
    picks = rng.integers(0, len(XS), 64)
    seen, exact = set(), set()
    for expr in expressions:
        vector = exact_eval(expr, memo)
        seen.add(vector[picks].tobytes())
        exact.add(vector.tobytes())
    return {"exact_classes": len(exact), "probe_signature_classes": len(seen)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", type=Path, required=True, help="A LA-015 library.jsonl, copied here read-only")
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 19, 29])
    p.add_argument("--out", type=Path, default=HERE)
    a = p.parse_args()
    torch.set_num_threads(2)
    a.out.mkdir(parents=True, exist_ok=True)
    snapshot = a.out / "library_snapshot.jsonl"
    if a.library.resolve() != snapshot.resolve():
        shutil.copyfile(a.library, snapshot)
    rows = [json.loads(line) for line in snapshot.read_text().splitlines()]
    solutions = [to_tuple(r["expression"]) for r in rows]
    everything = set()
    for expr in solutions:
        parts(expr, everything)
    everything = sorted(e for e in everything if e[0] not in ("X", "Y", "K"))
    classes, memo = quotient(everything)
    canonical = sorted(classes.values(), key=lambda e: (size(e), repr(e)))
    report = {"schema": "logos.nc080.results/1", "library_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(), "library_solutions": len(solutions),
              "distinct_subexpressions": len(everything), "exact_quotient_classes": len(classes), "largest_canonical_size": max(size(e) for e in canonical),
              "approximate_signature": approximate_merge_count(everything, memo), "neural_scope": "The network computes each bit from local inputs. The host iterates bits, routes carries, and wires concepts.", "seeds": {}}
    print("solutions", len(solutions), "subexpressions", len(everything), "classes", len(classes), report["approximate_signature"], flush=True)
    sample = canonical[:: max(1, len(canonical) // 400)]
    for seed in a.seeds:
        model, info = train(seed)
        model.eval()
        with torch.no_grad():
            network_table(model)
            nm = {}
            correct_ops = full = bit_errors = 0
            worst = []
            for expr in sample:
                got = neural_eval(model, expr, nm)
                want = bits(exact_eval(expr, memo))
                wrong = (got != want).any(1).sum().item()
                full += wrong == 0
                bit_errors += wrong
                if wrong:
                    worst.append({"expression_size": size(expr), "wrong_pairs": wrong})
            solved = sum(bool((neural_eval(model, e, nm) == bits(exact_eval(e, memo))).all()) for e in solutions[:: max(1, len(solutions) // 150)])
            checked = len(solutions[:: max(1, len(solutions) // 150)])
            # Controls.
            model.permute = torch.roll(torch.arange(len(OPS)), 1)
            network_table(model)
            pm = {}
            permuted = sum(bool((neural_eval(model, e, pm) == bits(exact_eval(e, memo))).all()) for e in sample[:40])
            model.permute = None
            network_table(model)
            torch.manual_seed(seed + 1)
            blank = NeuralOperators().eval()
            network_table(blank)
            bm = {}
            untrained = sum(bool((neural_eval(blank, e, bm) == bits(exact_eval(e, memo))).all()) for e in sample[:40])
        del model._table
        torch.save({"state": model.state_dict(), "seed": seed}, a.out / f"operators_seed{seed}.pt")
        report["seeds"][str(seed)] = {**info, "concepts_tested": len(sample), "concepts_exact_on_all_pairs": full, "wrong_pairs_total": bit_errors, "failures": worst[:10],
                                      "solutions_checked": checked, "solutions_exact_on_all_pairs": solved, "permuted_meaning_exact": permuted, "untrained_exact": untrained, "controls_tested": min(40, len(sample)),
                                      "checkpoint_sha256": hashlib.sha256((a.out / f"operators_seed{seed}.pt").read_bytes()).hexdigest()}
        print(seed, "local", info["final_local_accuracy"], "concepts", full, "of", len(sample), "solutions", solved, "of", checked, "controls", permuted, untrained, flush=True)
    (a.out / "neural_quotient_results.json").write_text(json.dumps(report, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
