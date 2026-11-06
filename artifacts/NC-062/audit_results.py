"""Implementer reproducibility audit for NC-062 (not an independent reviewer).

Dependencies: Python3 stdlib, run_loop.py, NC-028 isolation, NC-009 library,
NC-061 scoreboard. For full per-goal model replay (--full), CPU PyTorch sandbox.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import run_loop as r

def shape_and_bytes(name,x):
    if name=="net.0.weight":shape=(24,24)
    elif name=="net.0.bias":shape=(24,)
    elif name=="net.2.weight":shape=(4,24)
    elif name=="net.2.bias":shape=(4,)
    else:raise RuntimeError("UNKNOWN_TENSOR:"+name)
    def flatten(v,shape):
        if not shape:
            if type(v) not in (float,int):raise RuntimeError("BAD_TENSOR_VALUE")
            return struct.pack("<f",v)
        if type(v) is not list or len(v)!=shape[0]:raise RuntimeError("BAD_TENSOR_SHAPE")
        return b"".join(flatten(i,shape[1:]) for i in v)
    return flatten(x,shape)

def calc_model_hash(state):
    h=hashlib.sha256()
    for name in sorted(state):
        h.update(name.encode())
        h.update(shape_and_bytes(name,state[name]))
    return h.hexdigest()

def verify(full=False):
    root=r.HERE
    protocol=json.loads((root/"PROTOCOL.json").read_text())
    results=json.loads((root/"scoreboard_results.json").read_text())
    seals=json.loads((root/"commitments.json").read_text())["public"]
    reveals=json.loads((root/"sealed_reveals.json").read_text())["releases"]
    library_versions=json.loads((root/"library_versions.json").read_text())["versions"]
    original=r.verified_initial_library()
    assert protocol["sandbox_worker_source_sha256"]==r.WORKER_SHA
    assert len(seals)==len(reveals)==9
    assert len(library_versions)==10
    assert library_versions[0]["entries"]==original
    assert results["completed_generations"]==3 and len(results["per_seed_generation_scores"])==9
    assert results["champion_sha256_unchanged"]==r.sha(r.CHAMPION)
    assert results["status"]=="NO_DISCOVERY" and results["independent_discovery_count"]==0
    for rel,h in protocol["reference_dependencies"].items():
        p=Path(rel) if rel.startswith("/") else r.ROOT/rel
        assert r.sha(p)==h,rel
    last=library_versions[0]["chain_sha256"]
    assert last==r.digest(original)
    for i,ver in enumerate(library_versions[1:],1):
        assert ver["version"]==i and ver["parent_chain_sha256"]==last
        assert ver["entries"]==original and ver["added_verified_programs"]==0
        gen=r.GENS[(i-1)//3];seed=r.SEEDS[(i-1)%3]
        assert ver["chain_sha256"]==r.digest({"previous":last,"entries":original,
                                "new":0,"generation":gen,"seed":seed})
        last=ver["chain_sha256"]
    statehashes={}
    previous={}
    for seed in r.SEEDS:
        for gen in (0,*r.GENS):
            path=root/f"checkpoint_g{gen}_seed{seed}.json"
            ck=json.loads(path.read_text())
            assert ck["seed"]==seed and ck["generation"]==gen
            assert ck["worker_source_sha256"]==r.WORKER_SHA
            state_hash=calc_model_hash(ck["model_state"])
            assert state_hash==ck["model_state_sha256"],(seed,gen,"model hash mismatch")
            if gen:
                assert ck["parent_state_sha256"]==previous[seed]
                train=r.train_rows(original,seed,gen)
                assert r.digest(train)==ck["train_input_sha256"],(seed,gen)
                assert ck["gradient_updates"]==8
            else:
                assert ck["parent_state_sha256"] is None and ck["gradient_updates"]==0
            previous[seed]=state_hash
            statehashes[f"{gen}:{seed}"]={"sha256":state_hash,"file_sha256":r.sha(path)}
    for item in seals:
        corresponding=next(q for q in reveals if (q["generation"],q["seed"])==(item["generation"],item["seed"]))
        rev=corresponding["reveal"]
        assert r.sb.verify_reveal(item["commitment"],rev)==rev
    score_map={(x["generation"],x["seed"]):x for x in results["per_seed_generation_scores"]}
    assert len(score_map)==9
    seal_map={(x["generation"],x["seed"]):x["reveal"] for x in reveals}
    table_overlap=[]
    training_discovery_overlap=[]
    for (generation,seed),entry in score_map.items():
        goals=r.sb.sealed_goals(seal_map[(generation,seed)])
        assert len(goals)==r.GOALS
        keys=[r.sb.table_key(g["table"]) for g in goals]
        assert keys==entry["evaluation_goal_keys"]
        training_discovery_overlap.append(len(set(keys)&set(entry["training_goal_keys"])))
        known=set(x["training_table_key"] for x in original)
        overlap=sum(k in known for k in keys)/len(keys)
        assert entry["train_score"]["training_overlap"]==overlap
        assert entry["no_library_training_ablation"]["training_overlap"]==overlap
        assert entry["train_score"]["solved"]<=r.GOALS
        assert entry["train_score"]["total"]==r.GOALS
        assert entry["train_score"]["candidates_used"]==r.GOALS
        table_overlap.append(overlap)
        ck=json.loads((root/f"checkpoint_g{generation}_seed{seed}.json").read_text())
        assert ck["model_state_sha256"]==entry["candidate_model_sha256"]
        if full:
            model,_=r.spawn("infer",seed,ck["model_state"],queries=[g["examples"] for g in goals])
            init=json.loads((root/f"checkpoint_g0_seed{seed}.json").read_text())
            no_lib,_=r.spawn("infer",seed,init["model_state"],queries=[g["examples"] for g in goals])
            # CPU from fresh sandbox differs; compare mathematical outcome,
            # not wall clock, CPU ratio, or rounding of very small timers.
            current=r.score_cached(goals,model["choices"],original,[x["training_table_key"] for x in original],1.)
            base=r.score_cached(goals,no_lib["choices"],original,[x["training_table_key"] for x in original],1.)
            for key in ("solved","total","unseen_solved","candidates_used",
                        "training_overlap","over_budget_goals","invalid_programs"):
                assert current[key]==entry["train_score"][key],(generation,seed,key)
                assert base[key]==entry["no_library_training_ablation"][key],(generation,seed,key)
    trace=[json.loads(x) for x in (root/"generation_trace.jsonl").read_text().splitlines()]
    assert len(trace)==len(r.GENS)*len(r.SEEDS)*r.DISCOVERY_GOALS==54
    discovery_cache={}
    for item in trace:
        gen,seed,index=item["generation"],item["seed"],item["goal_index"]
        if (gen,seed) not in discovery_cache:
            reveal={"generator":r.sb.GENERATOR,
                    "seed":f"NC062:discovery:{gen}:{seed}",
                    "nonce":"public-training-only",
                    "thresholds":{**r.sb.THRESHOLDS,"goals":r.DISCOVERY_GOALS}}
            discovery_cache[(gen,seed)]=r.sb.sealed_goals(reveal)
        goal=discovery_cache[(gen,seed)][index]
        assert r.sb.table_key(goal["table"])==item["goal_fingerprint"]
        assert item["received_full_goal_table"] is False
        assert item["new_discovery"] is False
        for attempt in item["attempts"]:
            index=attempt["library_index"]
            code=r.candidate_from_index(original,index)
            if attempt["verdict"]!="REJECTED_BY_FEEDBACK":
                proof=r.proof(goal,code)
                for field,val in proof.items():
                    assert attempt[field]==val,(gen,seed,item["goal_index"],field)
    assert all(x["scoreboard"]["verdict"]=="CONTAMINATED" for x in results["scoreboard_comparisons"]), "v1 flagged unclean, as it must"
    assert results["scoreboard_compounding"]["verdict"]!="COMPOUNDING"
    # Negative control: mutate a committed private reveal after the fact.
    altered=json.loads(json.dumps(reveals[0]["reveal"]))
    altered["thresholds"]["improvement_margin"]=0
    try:r.sb.verify_reveal(seals[0]["commitment"],altered)
    except ValueError:pass
    else:raise RuntimeError("SEALED_THRESHOLD_TAMPER_ACCEPTED")
    # Negative control: a training-only fit cannot count as a new proof.
    target=[{"op":"ADD","dst":"R","a":"X","b":"Y"}]
    false=[{"op":"LOADI","dst":"R","imm":0}]
    assert r.proof({"table":r.sb.truth_table(target)},false)["verdict"]=="COUNTEREXAMPLE"
    return {
        "status":"IMPLEMENTER_AUDIT_PASS_NOT_INDEPENDENT",
        "checkpoint_count":len(statehashes),
        "model_state_and_lineage_sha256":statehashes,
        "generation_seed_score_rows":len(score_map),
        "discovery_trace_rows":len(trace),
        "finite_exhaustive_translation_cases":4*256,
        "sealed_commitments_verified":len(seals),
        "max_training_overlap_fraction":max(table_overlap),
        "original_v1_contamination_detected":any(x>0 for x in table_overlap),
        "training_discovery_function_collisions":training_discovery_overlap,
        "scoreboard_comparisons":results["scoreboard_comparisons"],
        "compounding":results["scoreboard_compounding"],
        "no_discovery":True,
        "independently_verified_by_distinct_agent":False,
        "full_model_prediction_replay":full,
        "champion_unchanged":True,
        "global_registry_unchanged":True,
        "limitations":["Verifier and scoreboard are in the parent, model is in sandbox.",
                       "Training success is scoped to initial known templates.",
                       "No actual new program was registered; no hidden-goal feedback was used for training.",
                       "Process CPU varies between replays; don't assert bitwise CPU equivalence."]
    }

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--full",action="store_true")
    args=p.parse_args()
    report=verify(full=args.full)
    out=r.HERE/"replay_audit.json"
    if out.exists() and args.full:
        report["previous_audit_sha256"]=r.sha(out)
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":report["status"],"checkpoints":report["checkpoint_count"],
                      "traces":report["discovery_trace_rows"],
                      "score_rows":report["generation_seed_score_rows"],
                      "audit_sha256":r.sha(out),"full":args.full},sort_keys=True))
