"""NC-061: a tamper-evident scoreboard for generational self-improvement.

A candidate receives input-output examples for sealed goals and returns one program per goal. Answers are checked
exhaustively by canonical_isa.reference. Goals come from a generator, seed and thresholds committed by hash before a
run and revealed only after the candidate is frozen. Standard library only.
"""
import hashlib
import json
import random
import secrets
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "NC-001"))
import canonical_isa as isa  # noqa: E402

GENERATOR = "logos.nc061.goals/1"
WIDTH = 4
VALUES = 1 << WIDTH
DOMAIN = [(x, y) for x in range(VALUES) for y in range(VALUES)]
OPS = ("LOADI", "MOV", *isa.BINARY, *isa.UNARY)
THRESHOLDS = {"min_seeds": 3, "improvement_margin": 0.05, "retention_floor": 0.95, "budget_tolerance": 0.10,
              "cpu_tolerance": 0.25, "cpu_floor_seconds": 1.0, "contamination_limit": 0.0, "compounding_generations": 3, "compounding_min_growth": 0.01, "examples": 8, "goals": 40, "max_program": 5}
UNTRUSTED = "Candidate programs are data: they run only through canonical_isa.reference with a step limit."


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def new_commitment(thresholds=None):
    """Return (public commitment, private reveal). Publish the first before any candidate is frozen."""
    reveal = {"generator": GENERATOR, "seed": secrets.token_hex(16), "thresholds": dict(thresholds or THRESHOLDS),
              "nonce": secrets.token_hex(16)}
    return {"schema": "logos.nc061.commitment/1", "sha256": digest(reveal)}, reveal


def verify_reveal(commitment, reveal):
    if digest(reveal) != commitment["sha256"] or reveal.get("generator") != GENERATOR:
        raise ValueError("reveal does not match the published commitment")
    return reveal


def _instruction(rng):
    op = rng.choice(OPS)
    dst = rng.choice(isa.REGS)
    if op == "LOADI":
        return {"op": op, "dst": dst, "imm": rng.randrange(VALUES)}
    if op in isa.BINARY:
        return {"op": op, "dst": dst, "a": rng.choice(isa.REGS), "b": rng.choice(isa.REGS)}
    return {"op": op, "dst": dst, "src": rng.choice(isa.REGS)}


def full_program(body):
    return isa.validate({"version": isa.VERSION, "width": WIDTH, "signedness": "twos_complement",
                         "program": [*body, {"op": "HALT"}]})


def truth_table(body):
    """Function of (X, Y) to R, by the canonical interpreter. Invalid programs raise isa.ISAError."""
    program = full_program(body)
    out = []
    for x, y in DOMAIN:
        result = isa.reference(program, {"X": x, "Y": y, "R": 0, "T": 0}, max_steps=len(body) + 2)
        if result["status"] != "halted":
            raise isa.ISAError("program did not halt")
        out.append(result["registers"]["R"])
    return tuple(out)


def sealed_goals(reveal):
    """Distinct non-constant functions that are not bare copies of an input."""
    t = reveal["thresholds"]
    rng = random.Random(f"{reveal['generator']}|{reveal['seed']}")
    copies = {tuple(x for x, _ in DOMAIN), tuple(y for _, y in DOMAIN)}
    goals, seen = [], set()
    while len(goals) < t["goals"]:
        body = [_instruction(rng) for _ in range(rng.randint(2, t["max_program"]))]
        table = truth_table(body)
        if len(set(table)) > 1 and table not in copies and table not in seen:
            seen.add(table)
            inputs = rng.sample(range(len(DOMAIN)), t["examples"])
            goals.append({"table": table, "answer": body, "examples": [{"X": DOMAIN[i][0], "Y": DOMAIN[i][1], "R": table[i]} for i in inputs]})
    return goals


def table_key(table):
    return hashlib.sha256(bytes(table)).hexdigest()


def score_candidate(candidate, goals, budget, training_table_keys=(), retention=None):
    """Score one frozen candidate: candidate(examples, budget) -> (program body or None, candidates used)."""
    training = set(training_table_keys)
    solved = used = unseen_solved = overlap = invalid = over_budget = 0
    cpu = 0.0
    for goal in goals:
        key = table_key(goal["table"])
        overlap += key in training
        # CPU is measured here and covers only the candidate; its own usage report is not trusted alone.
        started = time.process_time()
        body, spent = candidate(goal["examples"], budget)
        cpu += time.process_time() - started
        if type(spent) is not int or not 1 <= spent <= budget:
            over_budget += 1
            continue
        used += spent
        if body is None:
            continue
        try:
            correct = truth_table(body) == goal["table"]
        except (isa.ISAError, TypeError, KeyError):
            invalid += 1
            continue
        solved += correct
        unseen_solved += correct and key not in training
    report = {"solved": solved, "total": len(goals), "solve_rate": solved / len(goals), "unseen_solved": unseen_solved,
              "candidates_used": used, "budget_per_goal": budget, "cpu_seconds": round(cpu, 3),
              "cpu_seconds_per_solve": round(cpu / solved, 4) if solved else None,
              "invalid_programs": invalid, "over_budget_goals": over_budget, "training_overlap": overlap / len(goals)}
    if retention is not None:
        report["retention"] = retention
    return report


def compare(previous, current, thresholds=None):
    """Decide one generation from per-seed scores. Only sealed-goal scores are ever passed in."""
    t = thresholds or THRESHOLDS
    reasons = []
    if len(current) < t["min_seeds"] or len(previous) < t["min_seeds"]:
        return {"verdict": "NOT_ESTABLISHED", "reasons": [f"needs at least {t['min_seeds']} seeds"]}
    if any(r["training_overlap"] > t["contamination_limit"] for r in current):
        return {"verdict": "CONTAMINATED", "reasons": ["sealed goals appear in declared training data"]}
    if any(r["over_budget_goals"] for r in current):
        return {"verdict": "INVALID", "reasons": ["candidate exceeded its budget"]}
    mean = lambda rows, k: sum(r[k] for r in rows) / len(rows)
    budget_ratio = mean(current, "candidates_used") / max(1, mean(previous, "candidates_used"))
    if budget_ratio > 1 + t["budget_tolerance"]:
        reasons.append(f"used {budget_ratio:.2f} times the previous generation's candidates")
    if any("cpu_seconds" not in r for r in [*previous, *current]):
        reasons.append("CPU was not measured on every seed")
    else:
        # The floor stops timer noise deciding very cheap runs; it never excuses a real increase.
        limit = mean(previous, "cpu_seconds") * (1 + t["cpu_tolerance"]) + t["cpu_floor_seconds"]
        if mean(current, "cpu_seconds") > limit:
            reasons.append(f"used {mean(current, 'cpu_seconds'):.2f} CPU seconds against a limit of {limit:.2f}")
    gain = mean(current, "solve_rate") - mean(previous, "solve_rate")
    if gain < t["improvement_margin"]:
        reasons.append(f"gain {gain:.3f} is below the margin {t['improvement_margin']}")
    if sum(r["solve_rate"] > p["solve_rate"] for r, p in zip(current, previous)) <= len(current) / 2:
        reasons.append("did not beat the previous generation on most seeds")
    retention = [r["retention"] for r in current if "retention" in r]
    if len(retention) != len(current):
        reasons.append("retention was not measured on every seed")
    elif min(retention) < t["retention_floor"]:
        reasons.append(f"retention {min(retention):.3f} is below the floor {t['retention_floor']}")
    return {"verdict": "NO_IMPROVEMENT" if reasons else "IMPROVEMENT", "gain": round(gain, 4), "reasons": reasons}


def compounding(generation_gains, verdicts, thresholds=None):
    """Compounding means the rate of improvement itself grows across consecutive verified generations."""
    t = thresholds or THRESHOLDS
    need = t["compounding_generations"]
    if len(generation_gains) < need or len(verdicts) != len(generation_gains):
        return {"verdict": "NOT_ESTABLISHED", "reason": f"needs {need} consecutive generations"}
    gains, labels = generation_gains[-need:], verdicts[-need:]
    if any(v != "IMPROVEMENT" for v in labels):
        return {"verdict": "NO_COMPOUNDING", "reason": "a generation in the window did not improve"}
    if any(round(b - a, 9) < t["compounding_min_growth"] for a, b in zip(gains, gains[1:])):
        return {"verdict": "NO_COMPOUNDING", "reason": "gains did not grow by the required margin, so this is improvement without compounding"}
    return {"verdict": "COMPOUNDING", "reason": f"gains {gains} grew across {need} generations"}
