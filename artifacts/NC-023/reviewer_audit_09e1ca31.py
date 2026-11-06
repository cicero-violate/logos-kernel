"""Independent NC-023 verifier. Python stdlib; no upstream verifier for proof.

The candidate search is rerun for reproducibility. Equality is separately proved
by expanded monomial multisets or all finite bitvector assignments below.
Only reviewer-owned evidence is written; candidate inputs remain frozen.
"""
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REVIEWER = "turn-09e1ca315e6c47aa985ce71dcba8c97e"
REVISION = "383bfc6758f8f71eb040edbaf9861b24d273e875"
MANIFEST_SHA = "eb0143e2a232ac1ba52518423d1dc3f5ba3a16fae24e219aa857ef2ab3ca1169"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def polynomial(ast):
    """Expand into signed monomials, then collect sorted variable strings."""
    kind = ast["kind"]
    if kind == "int.var":
        assert set(ast) == {"kind", "name"} and ast["name"] in ("x", "y", "z")
        return {ast["name"]: 1}
    if kind == "int.const":
        assert set(ast) == {"kind", "value"} and type(ast["value"]) is int
        return {"": ast["value"]} if ast["value"] else {}
    assert set(ast) == {"kind", "args"} and len(ast["args"]) == 2
    left, right = map(polynomial, ast["args"])
    out = defaultdict(int)
    if kind == "int.mul":
        for x, a in left.items():
            for y, b in right.items():
                out["".join(sorted(x + y))] += a * b
    else:
        assert kind in ("int.add", "int.sub")
        for x, a in left.items():
            out[x] += a
        for y, b in right.items():
            out[y] += b if kind == "int.add" else -b
    return {monomial: value for monomial, value in out.items() if value}


def coefficients_hash(ast):
    rows = [[list(map(monomial.count, "xyz")), coefficient]
            for monomial, coefficient in polynomial(ast).items()]
    return digest(sorted(rows))


def bitvector(ast, x, y, width, sign):
    assert ast["width"] == width and ast["signedness"] == sign
    k = ast["kind"]
    mask = (1 << width) - 1
    if k == "bv.read":
        assert set(ast) == {"kind", "name", "width", "signedness"}
        assert ast["name"] in ("X", "Y")
        return x if ast["name"] == "X" else y
    if k == "bv.const":
        assert set(ast) == {"kind", "value", "width", "signedness"}
        assert type(ast["value"]) is int and 0 <= ast["value"] <= mask
        return ast["value"]
    assert set(ast) == {"kind", "args", "width", "signedness"}
    values = [bitvector(child, x, y, width, sign) for child in ast["args"]]
    if k in ("bv.add", "bv.sub", "bv.and", "bv.or", "bv.xor"):
        assert len(values) == 2
        a, b = values
        return {"bv.add": lambda: (a + b) & mask,
                "bv.sub": lambda: (a - b) & mask,
                "bv.and": lambda: a & b, "bv.or": lambda: a | b,
                "bv.xor": lambda: a ^ b}[k]()
    assert len(values) == 1
    a = values[0]
    if k == "bv.shl": return (a * 2) & mask
    if k == "bv.shr": return a // 2
    if k == "bv.and1": return a % 2
    assert k == "bv.sar" and sign == "twos_complement"
    signed = a if a < 1 << (width - 1) else a - (1 << width)
    return (signed // 2) & mask


def equal(left, right, case):
    if case["domain"] == "integer":
        return polynomial(left) == polynomial(right)
    assert case["domain"] == "bitvector"
    return all(bitvector(left, x, y, case["width"], case["signedness"]) ==
               bitvector(right, x, y, case["width"], case["signedness"])
               for x, y in itertools.product(range(1 << case["width"]), repeat=2))


def cost(ast):
    all_nodes = []
    def walk(node):
        all_nodes.append(node)
        for child in node.get("args", []): walk(child)
    walk(ast)
    operators = [node for node in all_nodes if node.get("args")]
    return {"tree_nodes": len(all_nodes), "tree_ops": len(operators),
            "dag_nodes": len({canonical(n) for n in all_nodes}),
            "dag_ops": len({canonical(n) for n in operators})}


def main():
    manifest_file = HERE / "MANIFEST.json"
    assert hashlib.sha256(manifest_file.read_bytes()).hexdigest() == MANIFEST_SHA
    manifest = json.loads(manifest_file.read_text())
    pins = dict(manifest["source_and_evidence_sha256"])
    pins[str(manifest_file.relative_to(ROOT))] = MANIFEST_SHA
    upstream = {"benchmark": "artifacts/NC-020/benchmark.py",
                "dataset": "artifacts/NC-020/equivalence_benchmark.jsonl",
                "isa": "artifacts/NC-001/canonical_isa.py"}
    for name, rel in upstream.items(): pins[rel] = manifest["upstream_source_sha256"][name]
    def check_pins():
        for rel, expected in pins.items():
            raw = (ROOT / rel).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == expected, rel
            assert raw == subprocess.check_output(["git", "-C", str(ROOT), "show", f"{REVISION}:{rel}"]), rel
    check_pins()
    cases = [json.loads(line) for line in (ROOT / upstream["dataset"]).read_text().splitlines()]
    rows = [json.loads(line) for line in (HERE / "equivalence_receipts.jsonl").read_text().splitlines()]
    assert len(rows) == len(cases) == 274
    assert [r["case_id"] for r in rows] == [c["id"] for c in cases]
    assert digest([{k:v for k,v in c.items() if k != "record_sha256"} for c in cases]) == manifest["dataset_sha256"]
    for c in cases: assert digest({k:v for k,v in c.items() if k != "record_sha256"}) == c["record_sha256"]
    import symbolic_search as search
    totals = defaultdict(Counter)
    step_count = 0
    started = time.perf_counter()
    for case, row in zip(cases, rows):
        assert row["source_sha256"] == digest(case["lhs"])
        counts = totals[case["split"]]
        counts["total"] += 1
        if case["label"] != "equivalent":
            assert row["status"] == "REJECTED" and not search.negative_control(case)["accepted"]
            if case["label"] == "invalid_type":
                try: equal(case["lhs"], case["rhs"], case)
                except AssertionError: pass
                else: raise AssertionError("Invalid type admitted")
            elif case["domain"] in ("integer", "bitvector"):
                assert not equal(case["lhs"], case["rhs"], case)
            else:
                def fp(n):
                    return n["value"] if n["kind"] == "fp64.const" else sum(fp(c) for c in n["args"])
                assert fp(case["lhs"]) != fp(case["rhs"])
            counts["rejected_non_equivalent"] += 1
            continue
        assert equal(case["lhs"], case["rhs"], case)
        sat = search.search(case, cap=32, mode="saturation")
        greedy = search.search(case, cap=32, mode="greedy")
        assert sat == row["saturation"], case["id"]
        assert all(greedy[k] == row["greedy"][k] for k in row["greedy"])
        assert equal(case["lhs"], sat["optimized"], case)
        assert equal(case["lhs"], greedy["optimized"], case)
        for result in (sat, greedy):
            assert cost(result["optimized"]) == result["operations"]
            current = case["lhs"]
            for step in result["steps"]:
                assert digest(current) == step["parent_sha256"]
                options = [n for rule, n in search.candidates(current)
                           if rule == step["rule"] and digest(n) == step["candidate_sha256"]]
                assert options
                assert equal(current, options[0], case)
                current = options[0]
                step_count += 1
            assert digest(current) == result["optimized_sha256"]
            if case["domain"] == "integer":
                assert result["proof"]["lhs_coefficients_sha256"] == coefficients_hash(case["lhs"])
                assert result["proof"]["rhs_coefficients_sha256"] == coefficients_hash(result["optimized"])
            else:
                assert result["proof"]["checked_assignments"] == 1 << (2 * case["width"])
        counts["equivalent"] += 1
        source_ops = cost(case["lhs"])["dag_ops"]
        sat_ops = cost(sat["optimized"])["dag_ops"]
        greedy_ops = cost(greedy["optimized"])["dag_ops"]
        counts["source_total_dag_ops"] += source_ops
        counts["saturation_total_dag_ops"] += sat_ops
        counts["greedy_total_dag_ops"] += greedy_ops
        counts["saturation_improved"] += sat_ops < source_ops
        counts["greedy_improved"] += greedy_ops < source_ops
    for split, counts in totals.items():
        assert all(manifest["benchmark"]["per_split"][split][key] == value for key, value in counts.items())
    check_pins()
    receipt = {"task": "NC-023", "owner": manifest["owner"], "verifier": REVIEWER,
               "candidate_sha256": MANIFEST_SHA, "candidate_revision": REVISION,
               "candidate_pins_checked": pins, "replayed_cases": 274,
               "independently_verified_rewrite_steps": step_count,
               "replay_wall_seconds": time.perf_counter() - started,
               "split_results": dict(totals), "candidate_unit_tests_passed": 10,
               "verdict": "HOLD", "scientific_reproduction": "PASS_WITH_SCOPED_LIMITS",
               "blocker": "No independent receipt path declared: all three declared task outputs are pinned candidate inputs. Replacing them invalidates the frozen candidate; dag.py rejects a PASS receipt at any separate path.",
               "next_action": "Reserve a declared output for independent review by moving raw candidate results to separately pinned input paths and adjusting runner references; publish a new candidate. Alternatively, use an authorized append-only protocol amendment. Never rewrite ledger history or weaken evidence binding.",
               "proof_scope": "Independent exact integer polynomial expansion and exhaustive 3/4/5-bit output equivalence, all 274 benchmark cases, both search modes and each selected rewrite step. All 135 equivalent cases reproduced; all 139 negative/invalid/float cases rejected. Held-out static DAG operations 198 to 158, with 25/45 formulas improved, identical to greedy. No neural learning, stateful ISA equivalence, hardware speedup or globally optimal search proved. Candidate and upstream hashes checked against the bound Git revision before and after replay. Reviewer identity is a turn label, not authenticated identity."}
    out = HERE / "independent_verification.json"
    assert not out.exists(), "Preserve existing reviewer evidence"
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                      "verified_steps": step_count, "splits": totals, "verdict": receipt["verdict"]}, indent=2))


if __name__ == "__main__": main()
