"""NC-023 deterministic runner: Python 3 stdlib + symbolic_search.py + NC-020."""
from __future__ import annotations
import collections
import hashlib
import json
import sys
import time
from pathlib import Path
from symbolic_search import ROOT,HERE,VERSION,RULES,canonical,cost,digest,negative_control,search,score,ref

def timed_eval(ast,case,iterations=20):
    # This is Python AST interpreter latency, not machine/hardware FLOP counts.
    if case["domain"]=="integer":
        env={"x":3,"y":-5,"z":7}
        f=lambda: ref.eval_int(ast,env)
    else:
        f=lambda:ref.eval_bv(ast,3,5,case["width"],case["signedness"])
    for _ in range(4):f()
    begin=time.perf_counter_ns()
    for _ in range(iterations):f()
    return (time.perf_counter_ns()-begin)/iterations

def run():
    cases=[json.loads(l) for l in (ROOT/"artifacts/NC-020/equivalence_benchmark.jsonl").read_text().splitlines()]
    metadata=json.loads((ROOT/"artifacts/NC-020/splits_and_semantics.json").read_text())
    assert len(cases)==sum(metadata["splits"]["case_counts"].values()),"dataset has missing records"
    assert digest([{k:v for k,v in row.items() if k!="record_sha256"} for row in cases])==metadata["dataset_sha256"]
    assert all(digest({k:v for k,v in c.items() if k!="record_sha256"})==c["record_sha256"] for c in cases)
    # Every split and family is retained. Full 274-case data, including 4 invalid/FP controls.
    receipts=[];cases_by_split=collections.defaultdict(list)
    overall_start_ns=time.perf_counter_ns()
    for c in cases:
        cases_by_split[c["split"]].append(c)
    for c in cases:
        item={"case_id":c["id"],"family":c["family"],"split":c["split"],"domain":c["domain"],
              "label":c["label"],"source_sha256":digest(c["lhs"])}
        if c["label"]=="equivalent":
            sat_start_ns=time.perf_counter_ns()
            sat=search(c,cap=32,mode="saturation")
            sat_cpu_wall_ns=time.perf_counter_ns()-sat_start_ns
            greedy_start_ns=time.perf_counter_ns()
            greedy=search(c,cap=32,mode="greedy")
            greedy_cpu_wall_ns=time.perf_counter_ns()-greedy_start_ns
            item.update({"status":"CERTIFIED","source_cost":cost(c["lhs"]),"saturation":sat,
                         "search_wall_ns":{"saturation":sat_cpu_wall_ns,"greedy":greedy_cpu_wall_ns},
                         "greedy":{"optimized_sha256":greedy["optimized_sha256"],
                                   "operations":greedy["operations"],"explored":greedy["explored"]},
                         "latency_python_ns_per_eval":{"source":timed_eval(c["lhs"],c),
                                                        "selected":timed_eval(sat["optimized"],c)},
                         "source_rhs_sha256":digest(c["rhs"])})
        else:
            item["negative"]=negative_control(c)
            item["status"]="REJECTED" if not item["negative"]["accepted"] else "UNEXPECTED_ACCEPT"
            if item["status"]!="REJECTED":raise AssertionError("non-equivalent accepted "+c["id"])
        receipts.append(item)
    counters=collections.defaultdict(collections.Counter)
    samples=[]
    for item in receipts:
        split=item["split"];cnt=counters[split]
        cnt["total"]+=1
        if item["status"]=="CERTIFIED":
            cnt["equivalent"]+=1
            sc=item["source_cost"];opt=item["saturation"]["operations"];gr=item["greedy"]["operations"]
            if opt["dag_ops"]<sc["dag_ops"]:
                cnt["saturation_improved"]+=1
                if len(samples)<20:samples.append({"family":item["family"],"split":split,"id":item["case_id"],
                     "before":sc["dag_ops"],"after":opt["dag_ops"],"greedy":gr["dag_ops"]})
            if gr["dag_ops"]<sc["dag_ops"]:cnt["greedy_improved"]+=1
            if opt["dag_ops"]<gr["dag_ops"]:cnt["saturation_better_than_greedy"]+=1
            cnt["saturation_search_wall_ns"]+=item["search_wall_ns"]["saturation"]
            cnt["greedy_search_wall_ns"]+=item["search_wall_ns"]["greedy"]
            cnt["saturation_total_dag_ops"]+=opt["dag_ops"]
            cnt["source_total_dag_ops"]+=sc["dag_ops"]
            cnt["greedy_total_dag_ops"]+=gr["dag_ops"]
        else:cnt["rejected_non_equivalent"]+=1
    total_runner_wall_ns=time.perf_counter_ns()-overall_start_ns
    sources={
        "optimizer":HERE/"symbolic_search.py","runner":HERE/"run_baseline.py",
        "benchmark":ROOT/"artifacts/NC-020/benchmark.py",
        "dataset":ROOT/"artifacts/NC-020/equivalence_benchmark.jsonl",
        "isa":ROOT/"artifacts/NC-001/canonical_isa.py"
    }
    source_hashes={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in sources.items()}
    rules={"version":VERSION,"claim":"handwritten deterministic identities, not learned algorithms",
           "semantics":{"integer":"exact unbounded Z[x,y,z]; +,-,* only",
                        "bitvector":"NC-001 strict fixed width; output bits; all X,Y assignments",
                        "floating":"not optimized; supplied counterexamples rejected",
                        "state":"no side effects, PC, flags, full program, R/T",
                        "integer_overflow":"not applicable; not floating"},
           "identities":[dict(id=id,domain=domain,law=law,precondition=condition)
                         for id,domain,law,condition in RULES],
           "proof_admission":"Every candidate independently checked by exact polynomial canonical form or complete finite bitvector enumeration BEFORE entry; no probabilistic test substituted for proof.",
           "source_sha256":source_hashes}
    summary={"version":VERSION,"total_cases":len(receipts),"total_runner_wall_ns":total_runner_wall_ns,"splits":{k:dict(v) for k,v in sorted(counters.items())},
             "nontrivial_improvements":samples,
             "analysis":{"heldout_saturation_vs_greedy":"No held-out incremental static DAG operation win if saturation_better_than_greedy=0; equal or worse is not a discovery advantage",
                "runtime_cost_limit":"Python wall timings include interpreter, proof checks and search; do not identify full CPU share under host contention."},"search_policy":{"saturation":"bounded BFS of verified equal forms",
             "greedy":"bounded first-improving local rewrite","max_equivalent_states":32,
             "max_tree_nodes":76,"max_subtree_rewrite_depth":2},
             "cost_model":{"dag_ops":"unique operator AST subexpressions (ideal CSE graph)",
                "tree_ops":"operator occurrences without sharing",
                "python_ns_per_eval":"sequential AST interpreter; not a CPU architecture-independent latency model",
                "proposer_cpu":"not directly measured; benchmark runner uses Python full CPU"},
             "input_dataset_sha256":metadata["dataset_sha256"],
             "source_sha256":source_hashes,
             "status":"CANDIDATE_ONLY_NEEDS_INDEPENDENT_REVIEW",
             "limitations":["No neural proposal, learning or stochastic algorithm invention",
               "Saturation is a bounded equivalence-class graph, not a congruence-closure e-graph implementation",
               "Equality proof uses NC-020 exact oracle; independent reviewer must use a separately authored verifier",
               "Finite 3/4/5-bit proof scope; no IEEE float rewrite; no stateful ISA effects",
               "No proof of global minimum or independent measured hardware speedup",
               "Benchmark families authored with deterministic source grammar; historical pretraining independence unknown",
               "A lower static operator count is not sufficient evidence of end-to-end advantage"]}
    (HERE/"rewrite_rules.json").write_text(json.dumps(rules,indent=2,sort_keys=True)+"\n")
    (HERE/"equivalence_receipts.jsonl").write_text("".join(canonical(r)+"\n" for r in receipts))
    (HERE/"equality_saturation_baseline.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"total_cases":len(receipts),"splits":summary["splits"],
          "nontrivial_improvements":samples[:5]},sort_keys=True))
    return summary

if __name__=="__main__":
    run()
