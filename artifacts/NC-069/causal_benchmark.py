"""NC-069: interventions on the canonical Logos ISA, with ground truth from its reference interpreter.

Every intervention is applied by rewriting the program or its inputs and then running
canonical_isa.reference. This module holds no model code and no second interpreter.
Standard library only.
"""
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "NC-001"))
import canonical_isa as isa  # noqa: E402

WIDTH = 4
SIGN = "twos_complement"
VALUES = 1 << WIDTH
MIN_BODY, MAX_BODY = 3, 6
MAX_PROGRAM = MAX_BODY + 1  # Body plus HALT.
SPLIT_SALT = "NC-069/split/v1"
KINDS = ("none", "change_input", "set_register", "skip_step", "replace_instruction")
PLAIN_OPS = ("LOADI", "MOV", *isa.BINARY, *isa.UNARY)
BRANCH_OPS = ("JZ", "JNZ", "JMP")
BRANCH_RATE = 0.15


def _random_plain(rng, op=None):
    op = op or rng.choice(PLAIN_OPS)
    item = {"op": op}
    for field in isa.FIELDS[op]:
        item[field] = rng.randrange(VALUES) if field == "imm" else rng.choice(isa.REGS)
    return item


def random_program(rng):
    """Forward-only jumps: each instruction runs at most once, so every program halts."""
    length = rng.randint(MIN_BODY, MAX_BODY)
    body = []
    for index in range(length):
        if rng.random() < BRANCH_RATE:
            op = rng.choice(BRANCH_OPS)
            item = {"op": op, "target": rng.randint(index + 1, length)}
            if op != "JMP":
                item["src"] = rng.choice(isa.REGS)
            body.append(item)
        else:
            body.append(_random_plain(rng))
    return isa.validate({"version": isa.VERSION, "width": WIDTH, "signedness": SIGN,
                         "program": body + [{"op": "HALT"}]})


def apply_intervention(program, inputs, intervention):
    """Return the rewritten (program, inputs) that defines an intervention's meaning."""
    kind = intervention["kind"]
    body = [dict(item) for item in program["program"]]
    inputs = dict(inputs)
    if kind == "change_input":
        inputs[intervention["register"]] = intervention["value"]
    elif kind == "skip_step":
        # MOV X,X changes no register or flag and keeps every branch target valid.
        body[intervention["step"]] = {"op": "MOV", "dst": "X", "src": "X"}
    elif kind == "replace_instruction":
        body[intervention["step"]] = dict(intervention["instruction"])
    elif kind == "set_register":
        step = intervention["step"]
        for item in body:
            # A jump landing on the step still passes through the inserted write.
            if "target" in item and item["target"] > step:
                item["target"] += 1
        body.insert(step, {"op": "LOADI", "dst": intervention["register"], "imm": intervention["value"]})
    elif kind != "none":
        raise ValueError(f"unknown intervention {kind!r}")
    return isa.validate({**program, "program": body}), inputs


def outcome(program, inputs, intervention=None):
    """Exact final registers from the canonical interpreter."""
    rewritten, values = apply_intervention(program, inputs, intervention or {"kind": "none"})
    result = isa.reference(rewritten, values, max_steps=64)
    if result["status"] != "halted":
        raise ValueError(f"program did not halt: {result['status']}")
    return [result["registers"][name] for name in isa.REGS]


def random_intervention(rng, program):
    kind = rng.choice(KINDS[1:])
    last = len(program["program"]) - 2
    if kind == "change_input":
        return {"kind": kind, "register": rng.choice(isa.REGS), "value": rng.randrange(VALUES)}
    step = rng.randint(0, last)
    if kind == "skip_step":
        return {"kind": kind, "step": step}
    if kind == "set_register":
        return {"kind": kind, "step": step, "register": rng.choice(isa.REGS), "value": rng.randrange(VALUES)}
    current = program["program"][step]["op"]
    choices = [op for op in PLAIN_OPS if op != current]
    return {"kind": kind, "step": step, "instruction": _random_plain(rng, rng.choice(choices))}


def combo_key(program, intervention):
    """The (intervention, target opcode) pairing that held-out combinations are drawn from."""
    kind = intervention["kind"]
    if kind == "none":
        return "none"
    if kind == "change_input":
        return f"{kind}|{intervention['register']}"
    target = program["program"][intervention["step"]]["op"]
    if kind == "replace_instruction":
        return f"{kind}|{target}|{intervention['instruction']['op']}"
    return f"{kind}|{target}"


def _bucket(text):
    return int.from_bytes(hashlib.sha256(f"{SPLIT_SALT}|{text}".encode()).digest()[:4], "big") % 5


def is_test_program(program):
    return _bucket("program|" + isa.encode(program)) == 0


def is_heldout_combo(key):
    return key != "none" and _bucket("combo|" + key) == 0


def make_sample(program, inputs, intervention):
    return {"program": program, "inputs": inputs, "intervention": intervention,
            "target": outcome(program, inputs, intervention), "factual": outcome(program, inputs),
            "combo": combo_key(program, intervention)}


def generate(seed, count, split):
    """Draw samples for one split. Splits share no program and held-out combos are disjoint.

    train          training programs, seen combinations, mostly intervened
    passive_train  training programs, no intervention
    heldout_combo  unseen programs and unseen (intervention, opcode) combinations
    heldout_program unseen programs and combinations seen in training
    factual        unseen programs, no intervention
    """
    if split not in ("train", "passive_train", "heldout_combo", "heldout_program", "factual"):
        raise ValueError(split)
    rng = random.Random(f"{SPLIT_SALT}|{split}|{seed}")
    samples = []
    while len(samples) < count:
        program = random_program(rng)
        if is_test_program(program) != (split not in ("train", "passive_train")):
            continue
        inputs = {name: rng.randrange(VALUES) for name in isa.REGS}
        if split in ("passive_train", "factual"):
            intervention = {"kind": "none"}
        elif split == "train" and rng.random() < 0.2:
            intervention = {"kind": "none"}
        else:
            intervention = random_intervention(rng, program)
            if is_heldout_combo(combo_key(program, intervention)) != (split == "heldout_combo"):
                continue
        samples.append(make_sample(program, inputs, intervention))
    return samples
