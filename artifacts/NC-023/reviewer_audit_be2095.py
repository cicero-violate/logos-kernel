"""Independent NC-023 verifier: frozen hashes, polynomial algebra, exhaustive bitvectors,
all benchmark cases and accepted rewrite steps.
Dependencies: Python 3 standard library, current NC-023 pinned files,
NC-020 benchmark dataset, NC-001 ISA source; symbolic_search used only to
reconstruct candidate rewrite ancestry, never as proof oracle.
"""
import collections
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "artifacts" / "NC-023"
sys.path.insert(0, str(HERE))
from symbolic_search import candidates

AGENT = "turn-be2095c4c8154d17a0350a51452d4379"
EXPECTED_MANIFEST = "3b5ac010a19a2144f1677566e785c385ec1abfdadd0f93dd6d16aac3f125c186"
EXPECTED_REVISION = "0ac9d869cfe112865e3fdff3a52df81bb79a462f"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()

def loadlines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]

def verify_candidate():
    manifest = json.loads((HERE / "MANIFEST.json").read_text())
    assert sha(HERE / "MANIFEST.json") == EXPECTED_MANIFEST, "candidate manifest drift"
    assert manifest["owner"] == "claude-code-turn-6fab0d6a43294c558ed9b1c62f2f0d92"
    assert manifest["external_network_used"] is False
    assert manifest["immutable_champion_modified"] is False
    assert manifest["task"] == "NC-023"
    paths = {}
    for rel, h in manifest["source_and_evidence_sha256"].items():
        assert sha(ROOT / rel) == h, "candidate pin mismatch: " + rel
        paths[rel] = h
    sources = {
        "benchmark": "artifacts/NC-020/benchmark.py",
        "dataset": "artifacts/NC-020/equivalence_benchmark.jsonl",
        "isa": "artifacts/NC-001/canonical_isa.py",
        "optimizer": "artifacts/NC-023/symbolic_search.py",
        "runner": "artifacts/NC-023/run_baseline.py",
    }
    for name, rel in sources.items():
        assert sha(ROOT / rel) == manifest["upstream_source_sha256"][name], name
    assert subprocess.check_output(["git","cat-file","-t", EXPECTED_REVISION],text=True).strip() == "commit"
    frozen = subprocess.check_output(["git","show",EXPECTED_REVISION + ":artifacts/NC-023/MANIFEST.json"])
    assert hashlib.sha256(frozen).hexdigest() == EXPECTED_MANIFEST
    return manifest, paths

def polyn(node):
    names = {"x":0, "y":1, "z":2}
    k = node["kind"]
    if k == "int.var":
        assert set(node) == {"kind", "name"} and node["name"] in names
        m = [0,0,0]; m[names[node["name"]]] = 1
        return {tuple(m):1}
    if k == "int.const":
        assert set(node) == {"kind","value"} and type(node["value"]) is int
        return {(0,0,0):node["value"]} if node["value"] else {}
    assert set(node) == {"kind", "args"} and k in ("int.add","int.sub","int.mul")
    assert type(node["args"]) is list and len(node["args"]) == 2
    a,b = [polyn(v) for v in node["args"]]
    c = collections.defaultdict(int)
    if k == "int.mul":
        for m,v in a.items():
            for n,w in b.items():
                c[tuple(x+y for x,y in zip(m,n))] += v*w
    else:
        for m,v in a.items(): c[m] += v
        for m,v in b.items(): c[m] += (v if k == "int.add" else -v)
    return {m:v for m,v in c.items() if v}

def bval(node, x, y, w, sign, depth=0):
    assert depth < 64
    k = node["kind"]
    assert node["width"] == w and type(node["width"]) is int
    assert node["signedness"] == sign
    assert sign in ("unsigned","twos_complement")
    mask = (1<<w)-1
    if k == "bv.read":
        assert set(node) == {"kind","name","width","signedness"}
        assert node["name"] in ("X","Y")
        return x if node["name"] == "X" else y
    if k == "bv.const":
        assert set(node) == {"kind","value","width","signedness"}
        assert type(node["value"]) is int and 0 <= node["value"] <= mask
        return node["value"]
    assert set(node) == {"kind","args","width","signedness"}
    ops = ("bv.add","bv.sub","bv.and","bv.or","bv.xor","bv.shl","bv.shr","bv.sar","bv.and1")
    assert k in ops
    arity = 1 if k in ("bv.shl","bv.shr","bv.sar","bv.and1") else 2
    assert len(node["args"]) == arity
    z = [bval(t,x,y,w,sign,depth+1) for t in node["args"]]
    a = z[0]; b = z[1] if arity == 2 else 0
    if k == "bv.add": return (a+b)&mask
    if k == "bv.sub": return (a-b)&mask
    if k == "bv.and": return a&b
    if k == "bv.or": return a|b
    if k == "bv.xor": return a^b
    if k == "bv.and1": return a&1
    if k == "bv.shl": return (a<<1)&mask
    if k == "bv.shr": return a>>1
    if k == "bv.sar":
        assert sign == "twos_complement"
        return ((a>>1)|(a&(1<<(w-1))))&mask
    raise AssertionError(k)

def equivalent(a,b,case):
    d=case["domain"]
    if d=="integer":
        return polyn(a)==polyn(b), "polynomial over all Z[x,y,z]"
    if d=="bitvector":
        w=case["width"]; sign=case["signedness"]
        assert w in (3,4,5)
        for x in range(1<<w):
            for y in range(1<<w):
                if bval(a,x,y,w,sign)!=bval(b,x,y,w,sign):
                    return False, "finite exhaustive " + str((1<<w)**2)
        return True, "finite exhaustive " + str((1<<w)**2)
    assert d=="IEEE754_binary64_python_float"
    def fp(node):
        if node["kind"]=="fp64.const": return node["value"]
        assert node["kind"]=="fp64.add"
        return fp(node["args"][0])+fp(node["args"][1])
    return fp(a)==fp(b), "concrete IEEE binary64 counterexample"

def dag_cost(ast):
    unique_ops=set()
    seen=set()
    def walk(n):
        s=digest(n)
        if s in seen: return
        seen.add(s)
        if n.get("args"): unique_ops.add(s)
        for child in n.get("args",[]): walk(child)
    walk(ast)
    return len(unique_ops)

def replay():
    manifest,pins=verify_candidate()
    candidates_data = loadlines(ROOT / "artifacts/NC-020/equivalence_benchmark.jsonl")
    original = loadlines(HERE / "equivalence_receipts.jsonl")
    replayed = loadlines(HERE / "replay_be2095/equivalence_receipts.jsonl")
    assert len(candidates_data)==len(original)==len(replayed)==274
    assert digest([{k:v for k,v in c.items() if k!="record_sha256"} for c in candidates_data]) == manifest["dataset_sha256"]
    assert len(set(c["id"] for c in candidates_data))==274
    count=collections.Counter()
    steps=0
    split=collections.defaultdict(lambda:collections.Counter())
    proof_scopes=collections.Counter()
    for c,record,replayed_row in zip(candidates_data,original,replayed):
        assert c["record_sha256"]==digest({k:v for k,v in c.items() if k!="record_sha256"})
        assert c["id"]==record["case_id"]==replayed_row["case_id"]
        assert c["split"]==record["split"]==replayed_row["split"]
        assert c["family"]==record["family"]==replayed_row["family"]
        # All semantic outputs identical; only time measurements may vary.
        for key in ("status","source_sha256","source_cost","saturation","greedy","negative"):
            if key in record: assert record[key]==replayed_row[key], (c["id"],key)
        assert digest(c["lhs"])==record["source_sha256"]
        if c["label"]=="equivalent":
            ok,scope = equivalent(c["lhs"],c["rhs"],c)
            assert ok, c["id"]+" dataset truth false"
            optimized=record["saturation"]["optimized"]
            ok, _=equivalent(c["lhs"],optimized,c)
            assert ok, c["id"]+" unsound rewrite"
            assert digest(optimized)==record["saturation"]["optimized_sha256"]
            assert dag_cost(c["lhs"])==record["source_cost"]["dag_ops"]
            assert dag_cost(optimized)==record["saturation"]["operations"]["dag_ops"]
            # Trace path from original source. Every claimed local rewrite
            # must exist and independently preserve whole-expression semantics.
            current=c["lhs"]
            for step in record["saturation"]["steps"]:
                assert digest(current)==step["parent_sha256"], c["id"]
                possible=[ast for name,ast in candidates(current)
                          if name==step["rule"] and digest(ast)==step["candidate_sha256"]]
                assert possible, (c["id"],step["rule"])
                next_expr=possible[0]
                ok,_=equivalent(c["lhs"],next_expr,c)
                assert ok, (c["id"],"unsound accepted step")
                current=next_expr
                steps+=1
            assert digest(current)==record["saturation"]["optimized_sha256"], c["id"]
            if record["source_cost"]["dag_ops"]>record["saturation"]["operations"]["dag_ops"]:
                split[c["split"]]["saturation_improved"]+=1
            count["equivalent"]+=1
            split[c["split"]]["equivalent"]+=1
            split[c["split"]]["source_total_dag_ops"]+=record["source_cost"]["dag_ops"]
            split[c["split"]]["saturation_total_dag_ops"]+=record["saturation"]["operations"]["dag_ops"]
            split[c["split"]]["greedy_total_dag_ops"]+=record["greedy"]["operations"]["dag_ops"]
            if record["source_cost"]["dag_ops"]>record["greedy"]["operations"]["dag_ops"]:
                split[c["split"]]["greedy_improved"]+=1
            proof_scopes[scope]+=1
        else:
            assert record["status"]=="REJECTED" and record["negative"]["accepted"] is False
            if c["label"]=="invalid_type":
                try: equivalent(c["lhs"],c["rhs"],c)
                except (AssertionError,KeyError,ValueError,TypeError): pass
                else: raise AssertionError("invalid type accepted")
            else:
                ok,_=equivalent(c["lhs"],c["rhs"],c)
                assert not ok, c["id"]+" negative incorrectly accepted"
            count["rejected"]+=1
            split[c["split"]]["rejected_non_equivalent"]+=1
        split[c["split"]]["total"]+=1
    stats=json.loads((HERE/"equality_saturation_baseline.json").read_text())
    replay_stats=json.loads((HERE/"replay_be2095/equality_saturation_baseline.json").read_text())
    assert stats["total_cases"]==replay_stats["total_cases"]==274
    for name in ("train","validation","test"):
        for k in ("total","equivalent","rejected_non_equivalent",
                  "saturation_improved","greedy_improved","source_total_dag_ops",
                  "saturation_total_dag_ops","greedy_total_dag_ops"):
            assert split[name][k]==stats["splits"][name][k]==replay_stats["splits"][name][k],(name,k)
    assert count["equivalent"]==135 and count["rejected"]==139
    assert split["test"]["source_total_dag_ops"]==198
    assert split["test"]["saturation_total_dag_ops"]==158
    assert split["test"]["greedy_total_dag_ops"]==158
    # Fresh wrong rewrite, mixed width, float reassociation negative controls.
    var=lambda name: {"kind":"int.var","name":name}
    op=lambda name,a,b: {"kind":"int."+name,"args":[a,b]}
    bad_int=op("add",var("x"),var("y"))
    bad_int_rhs=op("mul",var("x"),var("y"))
    assert polyn(bad_int)!=polyn(bad_int_rhs)
    invalid={"kind":"bv.read","name":"X","width":5,"signedness":"unsigned"}
    typed={"domain":"bitvector","width":4,"signedness":"unsigned"}
    try: equivalent(invalid,invalid,typed)
    except AssertionError: pass
    else: raise AssertionError("mixed width negative control missed")
    float_controls=[c for c in candidates_data if c["domain"]=="IEEE754_binary64_python_float"]
    assert len(float_controls)==2 and all(not equivalent(c["lhs"],c["rhs"],c)[0] for c in float_controls)
    result={
        "task":"NC-023","verifier":AGENT,
        "candidate_sha256":EXPECTED_MANIFEST,
        "candidate_revision":EXPECTED_REVISION,
        "source_hashes":pins,
        "upstream_hashes":manifest["upstream_source_sha256"],
        "data_sha256":manifest["dataset_sha256"],
        "original_results_sha256":sha(HERE/"equality_saturation_baseline.json"),
        "original_receipts_sha256":sha(HERE/"equivalence_receipts.jsonl"),
        "fresh_results_sha256":sha(HERE/"replay_be2095/equality_saturation_baseline.json"),
        "fresh_receipts_sha256":sha(HERE/"replay_be2095/equivalence_receipts.jsonl"),
        "cases":dict(count),"split":{k:dict(v) for k,v in split.items()},
        "independently_verified_rewrite_steps":steps,
        "proof_scopes":dict(proof_scopes),
        "negative_controls":{
            "non_equivalent_or_invalid_rejected":count["rejected"],
            "wrong_polynomial_identity":"PASS",
            "mixed_width_ast":"PASS",
            "fp64_nonassociativity_and_rounding":"PASS",
        },
        "test_command":"PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s artifacts/NC-023 -p test_symbolic_search.py -q",
        "replay_command":"python -c 'import run_baseline; run_baseline.HERE=isolated_replay_dir; run_baseline.run()'",
        "limitations":[
            "Exact polynomial proof restricted to Z[x,y,z] with +,-,*.",
            "Finite bitvector proof exhausts X,Y only for 3/4/5 bits and outputs only, not flags/state.",
            "No floating-point rewrites certified and no neural training performed.",
            "Static DAG operation counts are not hardware runtime or global optimality.",
            "Bounded equivalence class search, not a full e-graph or machine intelligence.",
            "Timing is environment-sensitive and the dataset families are authored, not proven historically unseen."
        ],
    }
    out=HERE/"reviewer_audit_be2095.json"
    assert not out.exists(), "avoid overwriting prior reviewer audit"
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"audit":str(out),"audit_sha256":sha(out),"cases":result["cases"],
                      "steps":steps,"heldout":dict(split["test"]),
                      "negative_controls":"PASS","hashes":"PASS"},sort_keys=True))

if __name__=="__main__":
    replay()
