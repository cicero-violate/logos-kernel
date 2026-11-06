"""LA-015: search, verified concept library and self-paced worlds for 8-bit functions of (X, Y). Standard library only.

Expressions are nested tuples evaluated by this module, never executed as code. Search builds output vectors bottom-up
and keeps one expression per distinct vector. Solutions become reusable concepts, so later concepts can contain earlier ones.
"""
import hashlib
import json
import os
import random
from pathlib import Path

MASK = 255
BINARY = ("ADD", "SUB", "AND", "OR", "XOR")
UNARY = ("SHL", "SHR")
LEAVES = (("X",), ("Y",), ("K", 1))
PROBES = 48
CAP = 64
FAMILIES = ("linear", "linear_xor", "masked", "shift_mix", "nested")
MAX_LEVEL = 12
ZERO = "0" * 64


def probes(seed=0):
    rng = random.Random(f"LA-015/probes/{seed}")
    return tuple(rng.randrange(256) for _ in range(PROBES)), tuple(rng.randrange(256) for _ in range(PROBES))


def _apply(name, a, b=None):
    if name == "ADD":
        return tuple((p + q) & MASK for p, q in zip(a, b))
    if name == "SUB":
        return tuple((p - q) & MASK for p, q in zip(a, b))
    if name == "AND":
        return tuple(p & q for p, q in zip(a, b))
    if name == "OR":
        return tuple(p | q for p, q in zip(a, b))
    if name == "XOR":
        return tuple(p ^ q for p, q in zip(a, b))
    if name == "SHL":
        return tuple((p << 1) & MASK for p in a)
    if name == "SHR":
        return tuple(p >> 1 for p in a)
    raise ValueError(name)


def evaluate(expr, xs, ys, memo=None):
    """Recursive vector evaluator. Memoized, because a concept repeated in an expression is computed once."""
    memo = {} if memo is None else memo
    if expr in memo:
        return memo[expr]
    head = expr[0]
    if head == "X":
        out = tuple(xs)
    elif head == "Y":
        out = tuple(ys)
    elif head == "K":
        out = (expr[1] & MASK,) * len(xs)
    elif head in UNARY:
        out = _apply(head, evaluate(expr[1], xs, ys, memo))
    elif head in BINARY:
        out = _apply(head, evaluate(expr[1], xs, ys, memo), evaluate(expr[2], xs, ys, memo))
    else:
        raise ValueError(f"unknown expression head {head!r}")
    memo[expr] = out
    return out


def compile_stack(expr, seen=None, code=None):
    """Second, separately structured evaluator: compile to a stack program and interpret it on integers."""
    code = [] if code is None else code
    head = expr[0]
    if head in ("X", "Y"):
        code.append(("PUSH", head))
    elif head == "K":
        code.append(("CONST", expr[1] & MASK))
    else:
        for part in expr[1:]:
            compile_stack(part, seen, code)
        code.append(("OP", head))
    return code


def run_stack(code, x, y):
    stack = []
    for kind, value in code:
        if kind == "PUSH":
            stack.append(x if value == "X" else y)
        elif kind == "CONST":
            stack.append(value)
        elif value == "SHL":
            stack.append((stack.pop() << 1) & MASK)
        elif value == "SHR":
            stack.append(stack.pop() >> 1)
        else:
            b, a = stack.pop(), stack.pop()
            stack.append({"ADD": (a + b) & MASK, "SUB": (a - b) & MASK, "AND": a & b, "OR": a | b, "XOR": a ^ b}[value])
    return stack.pop()


def size(expr):
    return 1 + sum(size(e) for e in expr[1:] if isinstance(e, tuple))


def distinct_size(expr, seen=None):
    seen = set() if seen is None else seen
    if expr in seen:
        return 0
    seen.add(expr)
    return 1 + sum(distinct_size(e, seen) for e in expr[1:] if isinstance(e, tuple))


def subexpressions(expr, seen=None):
    seen = set() if seen is None else seen
    if expr in seen:
        return
    seen.add(expr)
    yield expr
    for e in expr[1:]:
        if isinstance(e, tuple):
            yield from subexpressions(e, seen)


def search(target, library, budget, xs, ys):
    """Bottom-up search from leaves and library. Returns (expression or None, candidates used)."""
    pool, seen, used = [], {}, 0
    def add(expr, vector):
        if vector not in seen:
            seen[vector] = expr
            pool.append((expr, vector))
    for expr in (*LEAVES, *library):
        memo = {}
        add(expr, evaluate(expr, xs, ys, memo))
    if target in seen:
        return seen[target], 1
    start = 0
    while used < budget and start < len(pool):
        limit = len(pool)
        for i in range(limit):
            ei, vi = pool[i]
            if i >= start:
                for name in UNARY:
                    if used >= budget:
                        return None, used
                    used += 1
                    v = _apply(name, vi)
                    add((name, ei), v)
                    if v == target:
                        return seen[v], used
            for j in range(limit):
                if max(i, j) < start:
                    continue
                ej, vj = pool[j]
                for name in BINARY:
                    if used >= budget:
                        return None, used
                    used += 1
                    v = _apply(name, vi, vj)
                    add((name, ei, ej), v)
                    if v == target:
                        return seen[v], used
        start = limit
    return None, max(1, used)


# Goals are parameters of a family, evaluated by a plain function. No library expression is used to define a goal.
def goal_value(goal, x, y):
    family, a, b, c = goal
    if family == "linear":
        return (a * x + b * y) & MASK
    if family == "linear_xor":
        return ((a * x) ^ (b * y)) & MASK
    if family == "masked":
        return ((a * x + b * y) & c) & MASK
    if family == "shift_mix":
        return ((a * x + b * (y >> 1)) ^ (c * (x >> 2))) & MASK
    if family == "nested":
        return (a * ((b * x + y) & MASK) + c * y) & MASK
    raise ValueError(family)


def draw_goal(rng, family, level):
    top = min(255, 2 + 2 ** (level // 2 + 1) + level)
    return (family, rng.randrange(1, top + 1), rng.randrange(0, top + 1), rng.randrange(1, top + 1))


def hard_goal(rng, family):
    """The sealed exam: large odd coefficients, regardless of the current level."""
    return (family, rng.randrange(3, 128, 2), rng.randrange(3, 128, 2), rng.randrange(3, 128, 2))


def goal_table_key(goal):
    return hashlib.sha256(bytes(goal_value(goal, x, y) for x in range(0, 256, 3) for y in range(0, 256, 5))).hexdigest()


def is_sealed_side(goal):
    """A hash split: sealed exams and training can never share a function."""
    return int(goal_table_key(goal)[:8], 16) % 2 == 1


def verify(expr, goal):
    """Exhaustive over all 65,536 pairs by each of the two separately written evaluators."""
    xs, ys = zip(*((x, y) for x in range(256) for y in range(256)))
    values = evaluate(expr, xs, ys)
    if any(v != goal_value(goal, x, y) for v, x, y in zip(values, xs, ys)):
        return False
    code = compile_stack(expr)
    if len(code) > 4096:
        return False
    return all(run_stack(code, x, y) == goal_value(goal, x, y) for x, y in zip(xs, ys))


def _to_json(expr):
    return [expr[0], *[_to_json(e) if isinstance(e, tuple) else e for e in expr[1:]]]


def _from_json(node):
    return (node[0], *[_from_json(e) if isinstance(e, list) else e for e in node[1:]])


class Library:
    """Append-only concept log with a hash chain, recomputed on every load so tampering is refused.

    A chain alone cannot notice a deleted tail, so each append also records its count and head in a separate anchor file. The
    log is written first, so it may run ahead of the anchor after a crash, never behind it. A snapshot with no anchor loads on
    its chain alone unless require_anchor is set. Anyone able to rewrite both files can forge both: there is no secret key here.
    """

    def __init__(self, path, require_anchor=False):
        self.path = Path(path)
        self.anchor_path = self.path.with_name(self.path.name + ".anchor")
        self.solutions, self.head = [], ZERO
        self.heads, self.offsets = [ZERO], [0]
        data = self.path.read_bytes() if self.path.exists() else b""
        complete = data[:data.rfind(b"\n") + 1]  # A partial final line is a torn write and was never acknowledged.
        for line in complete.splitlines(keepends=True):
            row = json.loads(line)
            body = {k: v for k, v in row.items() if k != "chain_sha256"}
            self.head = hashlib.sha256((self.head + json.dumps(body, sort_keys=True, separators=(",", ":"))).encode()).hexdigest()
            if row["chain_sha256"] != self.head:
                raise ValueError(f"library chain broken at generation {row.get('generation')}")
            self.solutions.append(_from_json(row["expression"]))
            self.heads.append(self.head)
            self.offsets.append(self.offsets[-1] + len(line))
        if self.anchor_path.exists():
            anchor = json.loads(self.anchor_path.read_text())
            if anchor["entries"] > len(self.solutions) or self.heads[anchor["entries"]] != anchor["head"]:
                raise ValueError(f"library anchor mismatch: anchor has {anchor['entries']} entries, log has {len(self.solutions)}")
        elif require_anchor and data:
            raise ValueError("library anchor missing")

    def _write_anchor(self, entries, head):
        temp = self.anchor_path.with_name(self.anchor_path.name + f".tmp-{os.getpid()}")
        with temp.open("w") as out:
            json.dump({"entries": entries, "head": head}, out, sort_keys=True)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, self.anchor_path)
        descriptor = os.open(self.anchor_path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _cut(self, size):
        if self.path.exists() and self.path.stat().st_size > size:
            os.truncate(self.path, size)

    def append(self, expression, goal, generation):
        self._cut(self.offsets[-1])
        body = {"expression": _to_json(expression), "goal": list(goal), "generation": generation}
        self.head = hashlib.sha256((self.head + json.dumps(body, sort_keys=True, separators=(",", ":"))).encode()).hexdigest()
        line = json.dumps({**body, "chain_sha256": self.head}, sort_keys=True) + "\n"
        with self.path.open("a") as out:
            out.write(line)
            out.flush()
            os.fsync(out.fileno())
        self.solutions.append(expression)
        self.heads.append(self.head)
        self.offsets.append(self.offsets[-1] + len(line.encode()))
        self._write_anchor(len(self.solutions), self.head)

    def rollback(self, entries, head):
        """Discard work after a committed point. The anchor drops first, so the log is never shorter than the anchor."""
        if entries > len(self.solutions) or self.heads[entries] != head:
            raise ValueError(f"library anchor mismatch: committed {entries} entries, log has {len(self.solutions)}")
        self._write_anchor(entries, head)
        self._cut(self.offsets[entries])
        del self.solutions[entries:], self.heads[entries + 1:], self.offsets[entries + 1:]
        self.head = head

    def concepts(self, cap=CAP):
        """Recurring verified parts first, favouring larger ones. Leaves are always free."""
        counts = {}
        for expression in self.solutions[-2000:]:
            for part in subexpressions(expression):
                if size(part) > 1 and part[0] != "K":
                    counts[part] = counts.get(part, 0) + 1
        return sorted(counts, key=lambda e: (-counts[e] * distinct_size(e), distinct_size(e), repr(e)))[:cap]
