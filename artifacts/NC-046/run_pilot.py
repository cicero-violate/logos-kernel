#!/usr/bin/env python3
"""NC-046 bounded 3-seed canary pilot.
Dependencies: Python stdlib, trainer_worker.py, PROTOCOL.json, validated NC-022
RLT checkpoints/dataset and read-only NC-045 training DAG.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from trainer_worker import (
    HERE, N22, ARMS, SEEDS, MAX_STEPS, count_exposure,
    digest, existing, make_steps, reference, run_case,
    sha_file, trainer_status,
)

def write_once(path,obj):
    if path.exists(): raise RuntimeError("EVIDENCE_FILE_EXISTS:"+str(path))
    path.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")

def main():
    spec=json.loads((HERE/"PROTOCOL.json").read_text())
    assert spec["updates_per_seed_arm"]==4
    before=trainer_status()
    ref,data=reference()
    champion_sha=sha_file(ref.SOURCE)
    results=[]
    exposure=[]
    t0=time.monotonic()
    process0=time.process_time()
    for seed in SEEDS:
        controls={arm:count_exposure(make_steps(data,seed,arm,ref)) for arm in ARMS}
        assert controls["rehearsal"]==controls["blocked_replay"]
        assert sum(v[-1] for v in controls["new_only"])==sum(v[-1] for v in controls["rehearsal"])==192
        exposure.append({"seed":seed,"rehearsal_blocked_equal":True,
                         "rehearsal_multiset_sha256":digest(controls["rehearsal"]),
                         "new_only_multiset_sha256":digest(controls["new_only"]),
                         "new_only_total_examples":192,
                         "rehearsal_total_examples":192})
        for arm in ARMS:
            result=run_case(seed,arm,HERE/"runs"/f"pilot-seed{seed}-{arm}",stop_at=MAX_STEPS)
            assert result["steps_committed"]==MAX_STEPS
            assert result["total_examples"]==192
            old=[z for z in result["heldout"] if z["task"] in ("add","xor","sub")]
            current=[z for z in result["heldout"] if z["task"] in ("mul","lt")]
            old_mean=sum(x["correct"] for x in old)/sum(x["tested"] for x in old)
            current_mean=sum(x["correct"] for x in current)/sum(x["tested"] for x in current)
            result["old_skill_accuracy"]=old_mean
            result["current_skill_accuracy"]=current_mean
            result["heldout_correct"]=sum(x["correct"] for x in result["heldout"])
            result["heldout_total"]=sum(x["tested"] for x in result["heldout"])
            result["all_step_checkpoint_hashes"]=[h for _,_,h in existing(HERE/"runs"/f"pilot-seed{seed}-{arm}")]
            results.append(result)
            print(json.dumps({"seed":seed,"arm":arm,"old_accuracy":old_mean,
                              "current_accuracy":current_mean,"steps":4}),flush=True)
    evidence=[]
    strict=True
    for seed in SEEDS:
        sub={r["arm"]:r for r in results if r["seed"]==seed}
        delta_new=sub["rehearsal"]["old_skill_accuracy"]-sub["new_only"]["old_skill_accuracy"]
        delta_blocked=sub["rehearsal"]["old_skill_accuracy"]-sub["blocked_replay"]["old_skill_accuracy"]
        if not (delta_new>=.01 and delta_blocked>=.01):strict=False
        evidence.append({"seed":seed,"rehearsal_minus_new_only":delta_new,
                         "rehearsal_minus_blocked_replay":delta_blocked})
    after=trainer_status()
    assert before["seq"]==after["seq"] and before["head_sha256"]==after["head_sha256"],"TRAINER_LEDGER_CHANGED"
    assert sha_file(ref.SOURCE)==champion_sha,"CHAMPION_CHANGED"
    summary={"task":"NC-046","schema":"nc046-pilot-1",
             "status":"ISOLATED_CANARY_NOT_PRODUCTION",
             "protocol_sha256":sha_file(HERE/"PROTOCOL.json"),
             "source_sha256":{"model":sha_file(N22/"rlt_rehearsal.py"),
                              "data":sha_file(N22/"train_and_holdout_ledger.json"),
                              "champion":champion_sha},
             "baseline_nc022_scientific_verdict":"NO_GAIN",
             "hypothesis":"strict superiority over both controls under fixed small subset",
             "claim":"PILOT_REHEARSAL_GAIN" if strict else "NO_GAIN",
             "strict_success":strict,"three_seed_comparison":evidence,
             "exposure_matching":exposure,
             "results":results,
             "total_checkpoint_count":sum(len(r["all_step_checkpoint_hashes"]) for r in results),
             "total_training_examples":sum(r["total_examples"] for r in results),
             "total_optimizer_updates":sum(r["steps_committed"] for r in results),
             "process_cpu_seconds_including_orchestration":time.process_time()-process0,
             "wall_seconds_including_orchestration":time.monotonic()-t0,
             "trainer_ledger_seq_unchanged":before["seq"],
             "trainer_ledger_head_sha256":before["head_sha256"],
             "candidate_only":True,"promoted":False,
             "limitations":spec["limits"]+[
                 "A small 3-seed 4-update canary cannot validate statistical generalization or replace the NC-022 NO_GAIN conclusion.",
                 "Production gate remains closed until TR-002 approval and atomic worker-liveness support."
             ]}
    write_once(HERE/"trainer_pipeline.json",summary)
    print(json.dumps({"status":summary["status"],"claim":summary["claim"],"strict":strict,
                       "checkpoints":summary["total_checkpoint_count"],
                       "cpu_s":summary["process_cpu_seconds_including_orchestration"],
                       "evidence_sha256":sha_file(HERE/"trainer_pipeline.json")}))
if __name__=="__main__":main()
