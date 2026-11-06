#!/usr/bin/env python3
"""NC-062 second, prospectively committed clean evaluation.

V1 is preserved as CONTAMINATED. V2 fixes the *selection protocol* before
generating any new secret seed or inspecting new sealed results:
  - NC-061 generator samples a 40-goal pool from a fresh committed seed.
  - keep the first 10 distinct function tables NOT in frozen verified
    training library or in the prior train-only discovery-goal set.
  - no score-based selection, rejection sampling by outcome, re-training,
    model editing or threshold tuning is permitted.
  - candidate checkpoints are SHA-frozen and unchanged from V1.
Dependencies: Python stdlib, run_loop.py, NC-028 isolation, NC-061 scoreboard,
NC-009 registry (no writes), 12 immutable NC-062 checkpoints.
"""
import json
from pathlib import Path
import sys
import time
import run_loop as r

PROTOCOL = {
 "schema":"NC062/fresh-clean-evaluation-v2",
 "task":"NC-062",
 "reason":"First sealed eval was correctly CONTAMINATED due to direct training-library function overlaps; preserved, never used as improvement claim.",
 "selection":"Independent NC-061 sealed generator with committed generator/seed/nonce/thresholds, produce 40 distinct goals, filter by *function hash only* against fixed four training functions and six per-generation training-only discovery functions, select FIRST ten survivors before seeing candidate scores. Never filter by candidate success or policy output.",
 "pool_size":40,"selected_goals":10,
 "seeds":[11,19,29],"generations":[1,2,3],
 "no_retraining":True,
 "thresholds":{**r.sb.THRESHOLDS,"goals":40},
 "excluded_data":"frozen initial NC-009 library keys plus previously exposed six discovery-goal keys, not prior heldout results",
 "budget_per_worker":{"cpu_s":24,"wall_s":35},
 "scope":"Exact 4-bit symbolic program verification; 3 seeds, 3 candidate generations; no novel operator or fresh training from sealed goals",
 "nonclaims":["Not external identity/secret attestation","Not globally hidden from parent/same UID",
              "Not evidence of AGI/ASI or algorithm invention",
              "Not a fully independent verifier",
              "No physical equal-CPU guarantee; report actual measured process CPU."]
}

def run():
    base=r.HERE
    v1=json.loads((base/"scoreboard_results.json").read_text())
    require=r.require
    require(v1["completed_generations"]==3 and v1["independent_discovery_count"]==0,"V1_NOT_FINAL")
    p=base/"eval_v2_protocol.json"
    r.write_once(p,PROTOCOL)
    # No reveal is written, even to a temp file, before scoring. The parent
    # keeps them in process-local memory and the contained worker never reads.
    secrets={}
    commits=[]
    for gen in r.GENS:
        for seed in r.SEEDS:
            c,reveal=r.sb.new_commitment(PROTOCOL["thresholds"])
            secrets[(gen,seed)]=reveal
            commits.append({"generation":gen,"seed":seed,"commitment":c})
    r.write_once(base/"eval_v2_commitments.json",{"schema":"nc062-eval-v2-precommit/1",
             "protocol_sha256":r.sha(p),"commitments":commits})
    freezes={}
    for gen in r.GENS:
        for seed in r.SEEDS:
            path=base/f"checkpoint_g{gen}_seed{seed}.json"
            ck=json.loads(path.read_text())
            require(ck["generation"]==gen and ck["seed"]==seed,"WRONG_CHECKPOINT")
            freezes[f"{gen}:{seed}"]={"checkpoint_path":path.name,
                  "checkpoint_sha256":r.sha(path),
                  "frozen_model_sha256":ck["model_state_sha256"]}
    frozen={"schema":"nc062-eval-v2-candidate-freeze/1",
            "protocol_sha256":r.sha(p),
            "public_commitments_sha256":r.sha(base/"eval_v2_commitments.json"),
            "candidate_checkpoints":freezes,
            "frozen_before_reveal":True,
            "no_new_training":True,
            "champion_sha256":r.sha(r.CHAMPION),
            "worker_source_sha256":r.WORKER_SHA}
    r.write_once(base/"eval_v2_candidate_freeze.json",frozen)
    first=r.verified_initial_library()
    keys={q["training_table_key"] for q in first}
    original={(e["generation"],e["seed"]):e for e in v1["per_seed_generation_scores"]}
    results=[];releases=[]
    t0=time.monotonic()
    for gen in r.GENS:
        for seed in r.SEEDS:
            reveal=secrets[(gen,seed)]
            pub=next(x["commitment"] for x in commits if
                     x["generation"]==gen and x["seed"]==seed)
            r.sb.verify_reveal(pub,reveal)
            pool=r.sb.sealed_goals(reveal)
            excluded=keys|set(original[(gen,seed)]["training_goal_keys"])
            selected=[]
            skipped=[]
            for i,goal in enumerate(pool):
                key=r.sb.table_key(goal["table"])
                if key in excluded:
                    skipped.append({"pool_index":i,"function_key":key,
                                    "why":"already in training library or prior discovery query"})
                elif len(selected)<PROTOCOL["selected_goals"]:
                    selected.append((i,goal))
            require(len(selected)==10,"INSUFFICIENT_UNSEEN_HELDOUT_POOL")
            require(len({r.sb.table_key(g["table"]) for i,g in selected})==10,"DUPLICATE_SELECTED")
            queries=[goal["examples"] for _,goal in selected]
            ck=json.loads((base/f"checkpoint_g{gen}_seed{seed}.json").read_text())
            start=json.loads((base/f"checkpoint_g0_seed{seed}.json").read_text())
            require(r.sha(base/f"checkpoint_g{gen}_seed{seed}.json")==freezes[f"{gen}:{seed}"]["checkpoint_sha256"],"CANDIDATE_MUTATION")
            pred,pcost=r.spawn("infer",seed,ck["model_state"],queries=queries)
            control,ccost=r.spawn("infer",seed,start["model_state"],queries=queries)
            require(pred["model_sha256_before"]==ck["model_state_sha256"],"CANDIDATE_MODEL_MISMATCH")
            require(control["model_sha256_before"]==start["model_state_sha256"],"CONTROL_MODEL_MISMATCH")
            goals=[g for i,g in selected]
            candidate=r.score_cached(goals,pred["choices"],first,keys,pcost["cpu_seconds"])
            baseline=r.score_cached(goals,control["choices"],first,keys,ccost["cpu_seconds"])
            require(candidate["training_overlap"]==baseline["training_overlap"]==0.0,"CONTAMINATED_V2")
            results.append({"generation":gen,"seed":seed,
                            "model_sha256":ck["model_state_sha256"],
                            "baseline_model_sha256":start["model_state_sha256"],
                            "sealed_goal_function_keys":[r.sb.table_key(g["table"]) for g in goals],
                            "pool_size":len(pool),"selected_pool_indices":[i for i,g in selected],
                            "excluded_pool_items":skipped,
                            "sealed_goal_count":len(goals),
                            "candidate":candidate,"baseline":baseline,
                            "candidate_ranked_model_choices":pred["choices"],
                            "baseline_ranked_model_choices":control["choices"],
                            "candidate_cost":pcost,"baseline_cost":ccost,
                            "commitment_sha256":pub["sha256"],
                            "candidate_checkpoint_sha256":freezes[f"{gen}:{seed}"]["checkpoint_sha256"],
                            "training_overlap":0.0})
            # Release after program proposals, exact oracle judgments, and
            # CPU charges have been durably accumulated in parent memory.
            releases.append({"generation":gen,"seed":seed,"reveal":reveal})
    comparisons=[]
    for gen in r.GENS:
        rows=[z for z in results if z["generation"]==gen]
        cmp=r.sb.compare([x["baseline"] for x in rows],
                         [x["candidate"] for x in rows],
                         {**r.sb.THRESHOLDS,"goals":10})
        comparisons.append({"generation":gen,**cmp})
    compound=r.sb.compounding([x.get("gain",0) for x in comparisons],
                              [x["verdict"] for x in comparisons])
    # Always use the official NC-061 decision for any positive claim.
    assert all(x["verdict"]!="CONTAMINATED" for x in comparisons)
    no_new=v1["independent_discovery_count"]==0
    provenance={"schema":"nc062-fresh-clean-evaluation-v2",
                "status":"NO_DISCOVERY" if no_new else "REQUIRES_INDEPENDENT_NOVELTY_AUDIT",
                "new_discoveries":v1["independent_discovery_count"],
                "original_contaminated_v1_preserved_sha256":r.sha(base/"scoreboard_results.json"),
                "protocol_sha256":r.sha(p),
                "public_commitments_sha256":r.sha(base/"eval_v2_commitments.json"),
                "candidate_freeze_sha256":r.sha(base/"eval_v2_candidate_freeze.json"),
                "candidate_scored_without_retraining":True,
                "comparisons":comparisons,"compounding":compound,
                "score_rows":results,
                "heldout_goals_total":len(results)*10,
                "total_worker_inferences":len(results)*2,
                "all_retention_values_are_caller_reported":True,
                "all_training_overlap_zero":True,
                "full_reveals_published_after_scoring":True,
                "fresh_run_wall_seconds":time.monotonic()-t0,
                "limit":["Initial contaminated v1 remains in evidence; do not score it as gain.",
                         "CPU is RUSAGE_CHILDREN per process; host-load variation may cause unmatched cost.",
                         "No newly discovered operator; class choices are four known verified templates.",
                         "No independent reviewer audit or global champion promotion.",
                         "Commitments are local file integrity, not third-party timestamp or trusted secret storage."]}
    r.write_once(base/"eval_v2_results.json",provenance)
    r.write_once(base/"eval_v2_reveals.json",{"schema":"nc062-eval-v2-reveal-after-freeze/1",
                                        "released":releases})
    require(r.sha(r.CHAMPION)==frozen["champion_sha256"],"CHAMPION_DRIFT")
    require(r.sha(p)==frozen["protocol_sha256"],"PROTOCOL_TAMPER")
    print(json.dumps({"status":provenance["status"],
        "verdicts":[x["verdict"] for x in comparisons],
        "compounding":compound["verdict"],
        "rejected_pool_counts":[len(x["excluded_pool_items"]) for x in results],
        "sealed_goal_count":provenance["heldout_goals_total"],
        "score_count":len(results),
        "wall_seconds":provenance["fresh_run_wall_seconds"]},sort_keys=True))

if __name__=="__main__":run()
