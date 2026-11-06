"""NC-079: does a recursive library of verified concepts compound? Bottom-up search over ISA arithmetic.

An expression is a nested tuple. Goals are functions of (X, Y) on 8-bit values given by example vectors. Search builds
values bottom-up and keeps one expression per distinct output vector, so a concept used twice is searched once.
"""
import hashlib
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "NC-061"))
import scoreboard as sb  # noqa: E402

MASK = 255
BINARY = {"ADD": lambda a, b: (a + b) & MASK, "SUB": lambda a, b: (a - b) & MASK, "AND": lambda a, b: a & b,
          "OR": lambda a, b: a | b, "XOR": lambda a, b: a ^ b}
UNARY = {"SHL": lambda a: (a << 1) & MASK, "SHR": lambda a: a >> 1}
LEAVES = (("X",), ("Y",), ("K", 1))
EXAMPLES = 64
LIBRARY_CAP = 48


def probes(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, EXAMPLES, dtype=np.int64), rng.integers(0, 256, EXAMPLES, dtype=np.int64)


def evaluate(expr, x, y):
    """Value of an expression on arrays or integers."""
    head = expr[0]
    if head == "X":
        return x
    if head == "Y":
        return y
    if head == "K":
        return (x * 0 + expr[1]) & MASK
    if head in UNARY:
        return UNARY[head](evaluate(expr[1], x, y))
    return BINARY[head](evaluate(expr[1], x, y), evaluate(expr[2], x, y))


def size(expr):
    return 1 + sum(size(e) for e in expr[1:] if isinstance(e, tuple))


def subexpressions(expr):
    yield expr
    for e in expr[1:]:
        if isinstance(e, tuple):
            yield from subexpressions(e)


def search(goal_vector, library, budget, x, y):
    """Bottom-up search from leaves and library. Returns (expression or None, candidates, cpu seconds)."""
    started = time.process_time()
    pool, seen = [], {}
    used = 0
    def add(expr, vector):
        key = vector.tobytes()
        if key not in seen:
            seen[key] = expr
            pool.append((expr, vector))
            return True
        return False
    target = goal_vector.tobytes()
    for expr in [*LEAVES, *library]:
        add(expr, np.asarray(evaluate(expr, x, y), dtype=np.int64))
    if target in seen:
        return seen[target], max(1, used), time.process_time() - started
    start = 0
    while used < budget and start < len(pool):
        limit = len(pool)
        for i in range(limit):
            for name, fn in UNARY.items():
                if i < start or used >= budget:
                    continue
                used += 1
                vector = fn(pool[i][1])
                expr = (name, pool[i][0])
                if add(expr, vector) and vector.tobytes() == target:
                    return expr, used, time.process_time() - started
            for j in range(limit):
                if max(i, j) < start:
                    continue
                for name, fn in BINARY.items():
                    if used >= budget:
                        return None, used, time.process_time() - started
                    used += 1
                    vector = fn(pool[i][1], pool[j][1])
                    expr = (name, pool[i][0], pool[j][0])
                    if add(expr, vector) and vector.tobytes() == target:
                        return expr, used, time.process_time() - started
        start = limit
    return None, max(1, used), time.process_time() - started


class VerifyDomain:
    """Every 8-bit pair, evaluated by a second, integer-only interpreter on a sample for cross-checking."""
    def __init__(self):
        grid = np.arange(256, dtype=np.int64)
        self.x, self.y = np.meshgrid(grid, grid, indexing="ij")
        self.x, self.y = self.x.ravel(), self.y.ravel()


def reference_value(expr, x, y):
    """Separate evaluator on Python integers, matching the canonical ISA's semantics."""
    head = expr[0]
    if head == "X":
        return x
    if head == "Y":
        return y
    if head == "K":
        return expr[1]
    a = reference_value(expr[1], x, y)
    if head == "SHL":
        return (a << 1) & MASK
    if head == "SHR":
        return a >> 1
    b = reference_value(expr[2], x, y)
    return {"ADD": (a + b) & MASK, "SUB": (a - b) & MASK, "AND": a & b, "OR": a | b, "XOR": a ^ b}[head]


# --- Worlds. Each makes (name, table function) goals; train and sealed goals never share a function. ---
def _random_op(rng, a, b):
    if rng.random() < 0.2:
        return (rng.choice(tuple(UNARY)), a)
    return (rng.choice(tuple(BINARY)), a, b)


class StructuredWorld:
    """Hidden concepts stacked in levels; goals combine two concepts into a function no training goal used."""
    name = "structured"

    def __init__(self, seed):
        rng = random.Random(f"structured|{seed}")
        levels = [list(LEAVES)]
        for _ in range(4):
            levels.append([_random_op(rng, rng.choice(sum(levels, [])), rng.choice(sum(levels, []))) for _ in range(6)])
        self.concepts = sum(levels[1:], [])
        self.deep = levels[2:]

    def goal(self, rng):
        a, b = rng.choice(sum(self.deep, [])), rng.choice(sum(self.deep, []))
        return (rng.choice(tuple(BINARY)), a, b)


class ArithmeticWorld:
    """Natural family a*X + b*Y with odd coefficients, designed without reference to any library."""
    name = "arithmetic"

    def __init__(self, seed):
        self.seed = seed

    def goal(self, rng):
        return ("LINEAR", rng.randrange(3, 64, 2), rng.randrange(3, 64, 2))


class UnstructuredWorld:
    """Fresh random expressions with no shared concepts; a library should not help."""
    name = "unstructured"

    def __init__(self, seed):
        pass

    def goal(self, rng):
        def build(depth):
            if depth == 0:
                return rng.choice(LEAVES[:2]) if rng.random() < .8 else ("K", rng.randrange(1, 256))
            return _random_op(rng, build(depth - 1), build(depth - 1))
        return build(rng.choice((3, 4)))


def goal_vector(goal, x, y):
    if goal[0] == "LINEAR":
        return (goal[1] * x + goal[2] * y) & MASK
    return np.asarray(evaluate(goal, x, y), dtype=np.int64)


def goal_key(goal):
    domain = VERIFY
    return hashlib.sha256(np.asarray(goal_vector(goal, domain.x, domain.y), dtype=np.uint8).tobytes()).hexdigest()


VERIFY = VerifyDomain()


def draw_goals(world, rng, count, split, excluded=()):
    """Distinct non-trivial functions in a hash-assigned split, so sealed goals can never reach training."""
    wanted = {"train": lambda h: h % 2 == 0, "sealed": lambda h: h % 2 == 1}[split]
    out, keys = [], set(excluded)
    tries = 0
    while len(out) < count and tries < count * 400:
        tries += 1
        goal = world.goal(rng)
        table = goal_vector(goal, VERIFY.x, VERIFY.y)
        if len(np.unique(table)) < 2:
            continue
        key = goal_key(goal)
        if key in keys or not wanted(int(key[:8], 16)):
            continue
        keys.add(key)
        out.append(goal)
    return out


def verified(expr, goal, sample_seed=1):
    """Exhaustive check on all 65,536 pairs, then a cross-check with the separate integer evaluator."""
    if not np.array_equal(np.asarray(evaluate(expr, VERIFY.x, VERIFY.y), dtype=np.int64) & MASK, goal_vector(goal, VERIFY.x, VERIFY.y)):
        return False
    rng = random.Random(sample_seed)
    for _ in range(64):
        a, b = rng.randrange(256), rng.randrange(256)
        want = goal_vector(goal, np.int64(a), np.int64(b))
        if reference_value(expr, a, b) != int(want):
            return False
    return True


def harvest(solutions, cap=LIBRARY_CAP):
    """Keep the subexpressions of verified solutions that recur most, preferring larger ones. Leaves are free."""
    counts = {}
    for expr in solutions:
        for part in set(subexpressions(expr)):
            if size(part) > 1 and not (part[0] == "K"):
                counts[part] = counts.get(part, 0) + 1
    ranked = sorted(counts, key=lambda e: (-counts[e] * size(e), size(e), repr(e)))
    return ranked[:cap]


def random_library(rng, like):
    """Same count and sizes as the real library, with content unrelated to any solution."""
    def build(n):
        if n <= 1:
            return rng.choice(LEAVES)
        if rng.random() < .2:
            return (rng.choice(tuple(UNARY)), build(n - 1))
        left = rng.randrange(1, n)
        return (rng.choice(tuple(BINARY)), build(left), build(max(1, n - 1 - left)))
    return [build(size(e)) for e in like]


def run_generation(world, library, budget, rng, x, y, train_count, sealed_count, seen):
    """One generation: score the frozen library on sealed goals, then grow it from new training goals only."""
    sealed = draw_goals(world, rng, sealed_count, "sealed")
    def attempt(goals):
        solved, used, cpu = [], 0, 0.0
        for goal in goals:
            expr, spent, took = search(goal_vector(goal, x, y), library, budget, x, y)
            used += spent
            cpu += took
            if expr is not None and verified(expr, goal):
                solved.append(expr)
        return solved, used, cpu
    solved, used, cpu = attempt(sealed)
    row = {"solved": len(solved), "total": len(sealed), "solve_rate": len(solved) / max(1, len(sealed)), "candidates_used": used,
           "cpu_seconds": round(cpu, 3), "over_budget_goals": 0, "training_overlap": 0.0, "library_size": len(library),
           "largest_solution": max((size(e) for e in solved), default=0), "largest_concept": max((size(e) for e in library), default=0)}
    train = draw_goals(world, rng, train_count, "train", excluded=seen)
    seen |= {goal_key(g) for g in train}
    learned, _, _ = attempt(train)
    return row, learned
