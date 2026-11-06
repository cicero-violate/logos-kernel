"""NC-009: verify, deduplicate and promote operators proposed by NC-008 into an append-only registry.

The verifier here is written separately from NC-008's scorer and receives no model output beyond the program tokens.
Standard library only. A promoted operator is a checked program over known primitives, not a new primitive.
"""
import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MASK = 0xFFFF
VERSION = "logos.nc009.operator/1"
SEMANTICS = {  # token: (operation, first source, second source), all writing R.
    "ADD_X1": ("ADD", "X", "K"), "ADD_XX": ("ADD", "X", "X"), "XOR_XY": ("XOR", "X", "Y"), "AND_XY": ("AND", "X", "Y"),
    "OR_XY": ("OR", "X", "Y"), "AND1_X": ("AND1", "X", None), "ADD_RX": ("ADD", "R", "X"), "ADD_RY": ("ADD", "R", "Y"),
    "XOR_RX": ("XOR", "R", "X"), "XOR_RY": ("XOR", "R", "Y"), "SHL_R": ("SHL", "R", None), "SHR_R": ("SHR", "R", None)}
CONTRACTS = {"INC": lambda x, y: (x + 1) & MASK, "DOUBLE": lambda x, y: (2 * x) & MASK,
             "BOOLEAN_COMPOSITE": lambda x, y: x & ~y & MASK, "SHIFT_COMPOSITE": lambda x, y: ((x ^ y) << 1) & MASK}
CORNERS = [(0, 0), (MASK, 0), (0, MASK), (MASK, MASK), (1, 1), (0xAAAA, 0x5555), (0x5555, 0xAAAA), (0x8000, 0x7FFF), (0x8000, 0x8000), (0xFF, 0xFF00)]
SAMPLED = 65536
SEED = "NC-009/verifier/v1"
PROOF_SCOPE = ("All 65,536 pairs of 8-bit operands, 10 corner cases and 65,536 seeded 16-bit pairs, evaluated symbolically. "
               "This is finite testing, not a symbolic equivalence proof over all 2^32 pairs, and not a test of neural execution.")


def run(body, x, y):
    r = 0
    for token in body:
        op, a, b = SEMANTICS[token]
        state = {"X": x, "Y": y, "K": 1, "R": r}
        v, w = state[a], state[b] if b else 0
        r = {"ADD": (v + w) & MASK, "XOR": v ^ w, "AND": v & w, "OR": v | w, "AND1": v & 1, "SHL": (v << 1) & MASK, "SHR": v >> 1}[op]
    return r


def body_of(program):
    """Exactly one HALT, in last place, after known string tokens. Anything else is rejected."""
    if type(program) is not list or not program or any(type(token) is not str for token in program):
        raise ValueError("program must be a nonempty list of tokens")
    if program[-1] != "HALT" or program.count("HALT") != 1:
        raise ValueError("program must end with its only HALT")
    for token in program[:-1]:
        if token not in SEMANTICS:
            raise ValueError(f"unknown token {token!r}")
    return program[:-1]


def canonicalize(body):
    """Drop every step whose R is overwritten without being read. Steps that read R are kept."""
    kept = []
    for token in body:
        if "R" not in SEMANTICS[token][1:]:
            kept = []
        kept.append(token)
    return kept


def test_pairs():
    rng = random.Random(SEED)
    pairs = [(x, y) for x in range(256) for y in range(256)] + list(CORNERS)
    pairs += [(rng.randrange(1 << 16), rng.randrange(1 << 16)) for _ in range(SAMPLED)]
    return pairs


def verify(body, contract, pairs):
    """Return the first counterexample against the contract, or None."""
    for x, y in pairs:
        got, want = run(body, x, y), contract(x, y)
        if got != want:
            return {"x": x, "y": y, "got": got, "want": want}
    return None


def behaviour_key(body, pairs):
    h = hashlib.sha256()
    for x, y in pairs:
        h.update(run(body, x, y).to_bytes(2, "big"))
    return h.hexdigest()


def canonical_key(body):
    return hashlib.sha256(json.dumps(body, separators=(",", ":")).encode()).hexdigest()


def chain(previous, record):
    return hashlib.sha256((previous + json.dumps(record, sort_keys=True, separators=(",", ":"))).encode()).hexdigest()


def verify_chain(rows):
    """Recompute the whole chain, so editing, removing or reordering an earlier row is detected."""
    previous = "0" * 64
    for row in rows:
        body = {k: v for k, v in row.items() if k != "chain_sha256"}
        previous = chain(previous, body)
        if row.get("chain_sha256") != previous:
            raise ValueError(f"registry chain broken at {row.get('operator_id')!r}")


def process(candidates, pairs, existing=()):
    """Decide each candidate in order. Returns registry rows to append and the full audit trail."""
    verify_chain(existing)
    registry = list(existing)
    by_key = {row["canonical_key"]: row for row in registry}
    by_behaviour = {(row["task"], row["behaviour_key"]): row for row in registry}
    audit = []
    new = []
    previous = registry[-1]["chain_sha256"] if registry else "0" * 64
    for c in candidates:
        entry = {"task": c["task"], "source": c["source"], "program": c["program"]}
        audit.append(entry)
        try:
            original = body_of(c["program"])
        except ValueError as exc:
            entry.update(decision="REJECTED", reason=str(exc))
            continue
        contract = CONTRACTS.get(c["task"]) if type(c["task"]) is str else None
        if contract is None:
            entry.update(decision="REJECTED", reason="no independent contract for this task")
            continue
        body = canonicalize(original)
        entry["canonical_program"] = body
        problem = verify(body, contract, pairs)
        if problem:
            entry.update(decision="REJECTED", reason="counterexample", counterexample=problem)
            continue
        if verify(original, contract, pairs):
            entry.update(decision="REJECTED", reason="canonical form disagrees with the original program")
            continue
        key = canonical_key(body)
        entry["canonical_key"] = key
        if key in by_key:
            entry.update(decision="DUPLICATE", duplicate_of=by_key[key]["operator_id"])
            continue
        behaviour = behaviour_key(body, pairs)
        twin = by_behaviour.get((c["task"], behaviour))
        if twin and len(body) >= twin["reuse_cost_steps"]:
            entry.update(decision="SUBSUMED", subsumed_by=twin["operator_id"])
            continue
        row = {"version": VERSION, "operator_id": f"{c['task']}:v{sum(r['task'] == c['task'] for r in registry) + 1}",
               "task": c["task"], "program": body, "source_program": c["program"], "source": c["source"],
               "canonical_key": key, "behaviour_key": behaviour, "reuse_cost_steps": len(body), "proof_scope": PROOF_SCOPE,
               "pairs_tested": len(pairs), "proposal_checkpoint_sha256": c.get("checkpoint_sha256"),
               "frozen_primitive_ids": c.get("frozen_primitive_ids"), "implementation_mode": "symbolic check of a neural proposal"}
        row["chain_sha256"] = previous = chain(previous, row)
        registry.append(row)
        new.append(row)
        by_key[key] = row
        by_behaviour[(c["task"], behaviour)] = row
        entry.update(decision="PROMOTED", operator_id=row["operator_id"])
    return new, audit


def nc008_candidates():
    out = []
    for line in (ROOT / "artifacts/NC-008/proposals.jsonl").read_text().splitlines():
        d = json.loads(line)
        for label in ("greedy_program", "best_sampled_program"):
            out.append({"task": d["task"], "source": f"NC-008/{label}", "program": d[label],
                        "checkpoint_sha256": d.get("proposal_checkpoint_sha256"), "frozen_primitive_ids": d.get("frozen_primitive_ids")})
    return out


def main():
    pairs = test_pairs()
    registry_path = HERE / "operator_promotions.jsonl"
    existing = [json.loads(l) for l in registry_path.read_text().splitlines()] if registry_path.exists() else []
    new, audit = process(nc008_candidates(), pairs, existing)
    with registry_path.open("a") as out:
        for row in new:
            out.write(json.dumps(row, sort_keys=True) + "\n")
    report = {"schema": "logos.nc009.audit/1", "proof_scope": PROOF_SCOPE, "pairs_tested": len(pairs),
              "promoted_now": [r["operator_id"] for r in new], "registry_size": len(existing) + len(new), "decisions": audit,
              "nonclaims": ["Promotion is not invention of a new primitive.", "NC-008's random baseline also found these tasks.",
                            "Neural execution of these programs is NC-008 evidence, not rechecked here."]}
    (HERE / "promotion_audit.json").write_text(json.dumps(report, indent=1))
    for e in audit:
        print(e["task"], e["source"].split("/")[1][:6], e["program"], "->", e.get("canonical_program"), e["decision"], e.get("reason", ""), e.get("counterexample", ""))


if __name__ == "__main__":
    main()
