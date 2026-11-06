"""NC-062 implementer evidence check for fresh uncontaminated sealed evaluations.

Dependencies: Python 3 standard library, run_loop.py, NC-028/isolation.py,
NC-001/canonical_isa.py, NC-009/promote.py+registry and NC-061/scoreboard.py.
--full independently reruns frozen neural checkpoint inference through NC-028.
Not an independent reviewer identity or process-isolation attestation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import run_loop as r

def verify(full=False):
    base=r.HERE
    protocol=json.loads((base/"eval_v2_protocol.json").read_text())
    commits=json.loads((base/"eval_v2_commitments.json").read_text())
    freeze=json.loads((base/"eval_v2_candidate_freeze.json").read_text())
    assert commits["protocol_sha256"]==freeze["protocol_sha256"]==r.sha(base/"eval_v2_protocol.json")
    assert freeze["public_commitments_sha256"]==r.sha(base/"eval_v2_commitments.json")
    assert freeze["worker_source_sha256"]==r.WORKER_SHA
    assert freeze["champion_sha256"]==r.sha(r.CHAMPION)
    results=json.loads((base/"eval_v2_results.json").read_text())
    reveals=json.loads((base/"eval_v2_reveals.json").read_text())["released"]
    assert len(reveals)==9 and len(commits["commitments"])==9
    assert len(results["score_rows"])==9 and results["heldout_goals_total"]==90
    assert results["all_training_overlap_zero"] and results["new_discoveries"]==0
    assert results["original_contaminated_v1_preserved_sha256"]==r.sha(base/"scoreboard_results.json")
    lib=r.verified_initial_library()
    known={x["training_table_key"] for x in lib}
    originals=json.loads((base/"scoreboard_results.json").read_text())
    prior={(e["generation"],e["seed"]):e for e in originals["per_seed_generation_scores"]}
    checks=[]
    for row in results["score_rows"]:
        gen,seed=row["generation"],row["seed"]
        ident=(gen,seed)
        release=next(z["reveal"] for z in reveals if (z["generation"],z["seed"])==ident)
        pub=next(z["commitment"] for z in commits["commitments"] if (z["generation"],z["seed"])==ident)
        assert r.sb.verify_reveal(pub,release)==release
        pool=r.sb.sealed_goals(release)
        excluded=known|set(prior[ident]["training_goal_keys"])
        selection=[(i,g) for i,g in enumerate(pool) if r.sb.table_key(g["table"]) not in excluded][:10]
        assert len(pool)==40 and len(selection)==10
        assert row["selected_pool_indices"]==[i for i,g in selection]
        keys=[r.sb.table_key(g["table"]) for i,g in selection]
        assert keys==row["sealed_goal_function_keys"]
        assert set(keys).isdisjoint(excluded)
        got_excl=[{"pool_index":i,"function_key":r.sb.table_key(g["table"]),
                  "why":"already in training library or prior discovery query"} for i,g in enumerate(pool)
                 if r.sb.table_key(g["table"]) in excluded]
        assert got_excl==row["excluded_pool_items"]
        ck=json.loads((base/f"checkpoint_g{gen}_seed{seed}.json").read_text())
        start=json.loads((base/f"checkpoint_g0_seed{seed}.json").read_text())
        assert row["model_sha256"]==ck["model_state_sha256"]
        assert row["baseline_model_sha256"]==start["model_state_sha256"]
        assert r.sha(base/f"checkpoint_g{gen}_seed{seed}.json")==row["candidate_checkpoint_sha256"]
        assert freeze["candidate_checkpoints"][f"{gen}:{seed}"]["checkpoint_sha256"]==row["candidate_checkpoint_sha256"]
        goals=[g for i,g in selection]
        if full:
            current,_=r.spawn("infer",seed,ck["model_state"],queries=[g["examples"] for g in goals])
            baseline,_=r.spawn("infer",seed,start["model_state"],queries=[g["examples"] for g in goals])
            assert current["choices"]==row["candidate_ranked_model_choices"]
            assert baseline["choices"]==row["baseline_ranked_model_choices"]
        # Exact score oracle recomputed in this parent; hardware timing is
        # intentionally replay-excluded because of host load.
        candidate=r.score_cached(goals,row["candidate_ranked_model_choices"],lib,known,row["candidate"]["cpu_seconds"])
        baseline=r.score_cached(goals,row["baseline_ranked_model_choices"],lib,known,row["baseline"]["cpu_seconds"])
        for key in ("solved","total","solve_rate","unseen_solved","candidates_used","cpu_seconds",
                    "cpu_seconds_per_solve","invalid_programs","over_budget_goals","training_overlap"):
            assert candidate[key]==row["candidate"][key],(ident,key)
            assert baseline[key]==row["baseline"][key],(ident,key)
        assert candidate["training_overlap"]==baseline["training_overlap"]==0
        checks.append({"generation":gen,"seed":seed,"selected_goals":len(goals),
                       "candidate_verified":candidate["solved"],"ablation_verified":baseline["solved"],
                       "candidate_cpu_s":candidate["cpu_seconds"],
                       "ablation_cpu_s":baseline["cpu_seconds"],
                       "candidate_model_sha256":ck["model_state_sha256"]})
    assert len({(x["generation"],x["seed"]) for x in checks})==9
    compare=[]
    for gen in r.GENS:
        per=[z for z in results["score_rows"] if z["generation"]==gen]
        cmp=r.sb.compare([x["baseline"] for x in per],
                         [x["candidate"] for x in per],{**r.sb.THRESHOLDS,"goals":10})
        expected=next(c for c in results["comparisons"] if c["generation"]==gen)
        assert cmp=={k:v for k,v in expected.items() if k!="generation"}
        compare.append({"generation":gen,**cmp})
    comp=r.sb.compounding([x.get("gain",0) for x in compare],
                          [x["verdict"] for x in compare])
    assert comp==results["compounding"]
    assert comp["verdict"]!="COMPOUNDING"
    # Negative controls: any changed seed, generator, thresholds or nonce
    # must break its pinned SHA before new goal generation.
    first=reveals[0]["reveal"]
    commit=commits["commitments"][0]["commitment"]
    for key in ("seed","generator","nonce"):
        bad=json.loads(json.dumps(first));bad[key]="tampered"
        try:r.sb.verify_reveal(commit,bad)
        except ValueError:pass
        else:raise AssertionError("TAMPER_ACCEPTED:"+key)
    bad=json.loads(json.dumps(first));bad["thresholds"]["improvement_margin"]=0
    try:r.sb.verify_reveal(commit,bad)
    except ValueError:pass
    else:raise AssertionError("THRESHOLD_TAMPER_ACCEPTED")
    return {
        "schema":"nc062-implementer-clean-score-audit/1",
        "status":"PASS_IMPLEMENTER_NOT_INDEPENDENT",
        "checked_scores":checks,
        "matched_candidate_checkpoint_hashes":9,
        "sealed_reveal_and_commitment_match":9,
        "all_90_goals_functionally_disjoint_from_declared_training_and_discovery":True,
        "negative_controls":"PASS: seed/generator/nonce/threshold tamper",
        "scoreboard_comparisons":compare,"compounding":comp,
        "independent_model_inference_recomputed":full,
        "global_champion_untouched":True,
        "independent_scientific_verdict":"PENDING_INDEPENDENT_REVIEW",
        "limits":[
         "Even uncontaminated heldout may be too small to establish a small gain.",
         "CPU is measured by Linux child process accounting, not equality of wall clock or model complexity.",
         "Scoring code is pinned but remains in same-user trusted parent; no third-party secret attestation.",
         "The 4-class model is a classifier of known operators, not general program synthesis.",
         "Functional truth-table disjointness does not establish unseen program structure or algorithmic novelty."
        ]}
if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--full",action="store_true")
    args=parser.parse_args()
    result=verify(full=args.full)
    out=r.HERE/"eval_v2_audit.json"
    out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":result["status"],"score_rows":len(result["checked_scores"]),
                     "verdicts":result["scoreboard_comparisons"],
                     "compounding":result["compounding"]["verdict"],
                     "full_inference_replay":args.full,
                     "sha256":r.sha(out)},sort_keys=True))
