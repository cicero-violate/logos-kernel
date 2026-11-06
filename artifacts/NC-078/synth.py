"""NC-078: program synthesis from input-output examples over the canonical ISA.

The goal is a function (X, Y) -> R on 4-bit values, with R and T starting at zero. A fast array evaluator
ranks candidates; any accepted solution is confirmed by canonical_isa.reference on all 256 inputs.
"""
import hashlib
import itertools
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "NC-001"))
import canonical_isa as isa  # noqa: E402

WIDTH = 4
VALUES = 1 << WIDTH
MASK = VALUES - 1
SIGN = "twos_complement"
MAX_LENGTH = 4
OPS = ("LOADI", "MOV", *isa.BINARY, *isa.UNARY)
SALT = "NC-078/split/v1"
XS, YS = np.meshgrid(np.arange(VALUES), np.arange(VALUES), indexing="ij")
XS, YS = XS.ravel(), YS.ravel()
DOMAIN = len(XS)


def evaluate(body):
    """Truth table of R over every (X, Y), as a uint8 array."""
    regs = {"X": XS.copy(), "Y": YS.copy(), "R": np.zeros(DOMAIN, dtype=np.int64), "T": np.zeros(DOMAIN, dtype=np.int64)}
    for item in body:
        op = item["op"]
        if op == "LOADI":
            value = np.full(DOMAIN, item["imm"], dtype=np.int64)
        elif op == "MOV":
            value = regs[item["src"]]
        elif op in isa.BINARY:
            a, b = regs[item["a"]], regs[item["b"]]
            value = {"ADD": a + b, "SUB": a - b, "AND": a & b, "OR": a | b, "XOR": a ^ b}[op] & MASK
        else:
            s = regs[item["src"]]
            value = {"AND1": s & 1, "SHL": (s << 1) & MASK, "SHR": s >> 1,
                     "SAR": (s >> 1) | (s & (1 << (WIDTH - 1)))}[op]
        regs[item["dst"]] = value
    return regs["R"].astype(np.uint8)


def table_key(table):
    return table.tobytes()


def full_program(body):
    return isa.validate({"version": isa.VERSION, "width": WIDTH, "signedness": SIGN, "program": [*body, {"op": "HALT"}]})


def verify_with_reference(body, table):
    """Exhaustive confirmation by the canonical interpreter, independent of evaluate()."""
    program = full_program(body)
    for x, y, want in zip(XS.tolist(), YS.tolist(), table.tolist()):
        got = isa.reference(program, {"X": x, "Y": y, "R": 0, "T": 0}, max_steps=16)
        if got["status"] != "halted" or got["registers"]["R"] != want:
            return False
    return True


NAMED_GOALS = {
    "INC": lambda x, y: x + 1, "DEC": lambda x, y: x - 1, "NEG": lambda x, y: -x, "NOT": lambda x, y: ~x,
    "DOUBLE": lambda x, y: x * 2, "TRIPLE": lambda x, y: x * 3, "QUADRUPLE": lambda x, y: x * 4,
    "ADD_PLUS_1": lambda x, y: x + y + 1, "SUB_MINUS_1": lambda x, y: x - y - 1, "ADD_DOUBLE_Y": lambda x, y: x + 2 * y,
    "AVERAGE_FLOOR": lambda x, y: (x & y) + ((x ^ y) >> 1), "XOR_THEN_DOUBLE": lambda x, y: (x ^ y) * 2,
    "NAND": lambda x, y: ~(x & y), "NOR": lambda x, y: ~(x | y), "XNOR": lambda x, y: ~(x ^ y), "ANDNOT": lambda x, y: x & ~y,
}


def named_tables():
    return {name: (fn(XS, YS) & MASK).astype(np.uint8) for name, fn in NAMED_GOALS.items()}


def is_heldout_function(table):
    digest = hashlib.sha256(SALT.encode() + table.tobytes()).digest()
    return digest[0] % 10 == 0


def random_instruction(rng):
    op = rng.choice(OPS)
    dst = rng.choice(isa.REGS)
    if op == "LOADI":
        return {"op": op, "dst": dst, "imm": rng.randrange(VALUES)}
    if op in isa.BINARY:
        return {"op": op, "dst": dst, "a": rng.choice(isa.REGS), "b": rng.choice(isa.REGS)}
    return {"op": op, "dst": dst, "src": rng.choice(isa.REGS)}


def random_body(rng, max_length=MAX_LENGTH):
    return [random_instruction(rng) for _ in range(rng.randint(1, max_length))]


def is_interesting(table):
    """Skip constants and bare copies of an input, which carry little about composition."""
    return (len(np.unique(table)) > 1 and not np.array_equal(table, XS.astype(np.uint8))
            and not np.array_equal(table, YS.astype(np.uint8)))


def training_programs(seed, count, max_per_function=24):
    """Up to count programs, capped per function so common behaviour cannot dominate training."""
    named = {table_key(t) for t in named_tables().values()}
    rng = random.Random(f"{SALT}|train|{seed}")
    out = []
    seen = {}
    for _ in range(count * 40):
        if len(out) >= count:
            break
        body = random_body(rng)
        table = evaluate(body)
        key = table_key(table)
        if (not is_interesting(table) or key in named or is_heldout_function(table)
                or seen.get(key, 0) >= max_per_function):
            continue
        seen[key] = seen.get(key, 0) + 1
        out.append((body, table))
    return out


def unseen_goals(count, seed=1000):
    """Distinct held-out functions. Named goals are kept apart from this set."""
    named = {table_key(t) for t in named_tables().values()}
    rng = random.Random(f"{SALT}|unseen|{seed}")
    out = {}
    while len(out) < count:
        body = random_body(rng, 3)
        table = evaluate(body)
        key = table_key(table)
        if is_interesting(table) and is_heldout_function(table) and key not in named:
            out.setdefault(key, table)
    return {f"unseen_{i:03d}": t for i, t in enumerate(out.values())}


def all_instructions():
    items = []
    for dst in isa.REGS:
        items += [{"op": "LOADI", "dst": dst, "imm": v} for v in range(VALUES)]
        items += [{"op": op, "dst": dst, "src": s} for op in ("MOV", *isa.UNARY) for s in isa.REGS]
        items += [{"op": op, "dst": dst, "a": a, "b": b} for op in isa.BINARY for a in isa.REGS for b in isa.REGS]
    return items


def enumerate_shortest_first():
    """Fixed-order enumeration, the deterministic symbolic control."""
    instructions = all_instructions()
    for length in range(1, MAX_LENGTH + 1):
        for combo in itertools.product(instructions, repeat=length):
            yield list(combo)
