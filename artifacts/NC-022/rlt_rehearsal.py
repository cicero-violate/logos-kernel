#!/usr/bin/env python3
"""NC-022: preregistered matched-exposure RLT rehearsal experiment.

Code dependencies:
Python 3, PyTorch, NumPy, /workspace/experiments/rlt_split_algorithms_ablation_v1/
task_conditioned_rlt.py and rlt_core.py; original immutable
snapshots/current_unified_v3.pt; ../NC-000/inventory.json;
local PROTOCOL.json; Python stdlib json/hashlib/random/time/collections/pathlib.
NO external weights/source modified, no GPU or model promotion.
"""
from __future__ import annotations
import collections
import hashlib
import io
import itertools
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RLT=Path("/workspace/experiments/rlt_split_algorithms_ablation_v1")
sys.path.insert(0,str(RLT))
from task_conditioned_rlt import TaskConditionedRLT, OPERATIONS, SCALAR_OPERATIONS
from rlt_core import decode_logits

P=json.loads((HERE/"PROTOCOL.json").read_text())
SCHEMA=P["schema"]
SEEDS=tuple(P["seeds"])
ARMS=tuple(P["arms"])
STAGES=tuple(P["stage_order"])
BATCH=int(P["batch_size"])
SOURCE=RLT/"snapshots/current_unified_v3.pt"
MODEL_SOURCE=RLT/"task_conditioned_rlt.py"
CORE_SOURCE=RLT/"rlt_core.py"
INVENTORY=ROOT/"artifacts/NC-000/inventory.json"
CHECKPOINTS=HERE/"checkpoints"
EVAL=HERE/"eval"
MASK=lambda w:(1<<w)-1

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()

def canon(obj):
    return json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)

def digest(obj):
    return hashlib.sha256(canon(obj).encode()).hexdigest()

def stable_seed(*values):
    return int.from_bytes(hashlib.sha256("|".join(map(str,values)).encode()).digest()[:8],"little")

def source_hashes():
    return {k:sha_file(v) for k,v in {
        "snapshot":SOURCE,"architecture":MODEL_SOURCE,"backbone":CORE_SOURCE,
        "inventory":INVENTORY,"protocol":HERE/"PROTOCOL.json",
        "experiment":HERE/"rlt_rehearsal.py"}.items()}

def validate_contract():
    assert tuple(OPERATIONS)==tuple(P["baseline_known_tasks"])
    assert SEEDS==(11,19,29)
    assert ARMS==("new_only","rehearsal","blocked_replay")
    assert len(STAGES)==5 and sum(x["updates"] for x in STAGES)==144
    assert STAGES[0]["updates"]==0
    assert all(s["updates"]%4==0 for s in STAGES[1:])
    assert BATCH==48 and BATCH%2==0
    assert sha_file(SOURCE)==P["source_snapshot_sha256"]
    assert json.loads(INVENTORY.read_text())["champion"]["selected_bundle_sha256"]==sha_file(SOURCE)
    for p in (SOURCE,MODEL_SOURCE,CORE_SOURCE,INVENTORY):
        if not p.is_file():raise FileNotFoundError(p)

def truth(x,y,w,op):
    x,y=int(x),int(y);m=MASK(w)
    if op=="add":return (x+y)&m
    if op=="xor":return x^y
    if op=="sub":return (x-y)&m
    if op=="mul":return (x*y)&m
    if op=="lt":return int(x<y)
    if op=="parity_and":return ((x&y).bit_count()&1)
    if op=="compound":return ((x+y)^(x&y))&m
    raise ValueError(op)

def pairkey(x,y):
    return tuple(sorted((int(x),int(y))))

def samples(seed,group,count,excluded):
    if group.startswith("iid"):w=int(group[3:])
    elif group.startswith("carry"):w=int(group[5:])
    else:raise ValueError(group)
    rng=random.Random(stable_seed("holdout",seed,group,SCHEMA))
    seen=set(excluded)
    points=[]
    corners=[(0,0),(1,1),(MASK(w),1),(1,MASK(w)),
             (MASK(w),MASK(w)),(1<<(w-1),1<<(w-1)),
             (0,MASK(w)),(MASK(w),0),(0x55&MASK(w),0xAA&MASK(w))]
    j=0
    while len(points)<count:
        if group.startswith("iid"):
            pair=corners[j] if j<len(corners) else (rng.randrange(1<<w),rng.randrange(1<<w))
        else:
            k=1+(j%(w-1))
            lower=(1<<k)-1
            upper=rng.randrange(1<<(w-k))
            pair=((upper<<k)|lower,1)
            # Avoid degenerate duplicates in 8-bit carry families.
            if pairkey(*pair) in seen and j>=count*4:
                pair=(rng.randrange(1<<w),rng.randrange(1<<w))
        j+=1
        key=pairkey(*pair)
        if key in seen:continue
        seen.add(key)
        points.append([int(pair[0]),int(pair[1])])
    return points,seen

def heldouts(seed):
    groups={}
    seen=set()
    for name,count in P["evaluation_groups"].items():
        pairs,seen=samples(seed,name,count,seen)
        groups[name]=pairs
    assert sum(len(v) for v in groups.values())==sum(P["evaluation_groups"].values())
    return groups,seen

def training_ledger(seed,heldoutkeys):
    used=set()
    output={}
    for stage in STAGES[1:]:
        idx=stage["index"]
        blocks=[]
        for block in range(stage["updates"]//2):
            w=8 if block%2==0 else 16
            rng=random.Random(stable_seed("train",seed,idx,block,w))
            pairs=[]
            while len(pairs)<BATCH:
                a=rng.randrange(1<<w);b=rng.randrange(1<<w)
                k=pairkey(a,b)
                if k in heldoutkeys or k in used:continue
                used.add(k)
                pairs.append([a,b])
            current=list(stage["current"])
            prior=list(stage["prior"])
            # Rotate across blocks to avoid fixed position-to-op correlation.
            shift=(block*BATCH+idx+seed)%len(current)
            cur=[current[(i+shift)%len(current)] for i in range(BATCH)]
            if prior:
                shift2=(block*BATCH+idx+seed)%len(prior)
                old=[prior[(i+shift2)%len(prior)] for i in range(BATCH)]
            else:
                old=cur[:]
            blocks.append({"block":block,"width":w,"pairs":pairs,
                           "current":cur,"prior":old})
        output[str(idx)]=blocks
    assert len(used)==sum(stage["updates"]//2*BATCH for stage in STAGES[1:])
    return output

def task_for_block(stage,block,arm,phase):
    cur,old=block["current"],block["prior"]
    if arm=="new_only" or not stage["prior"]:return list(cur)
    if arm=="blocked_replay":return list(cur if phase==0 else old)
    if arm=="rehearsal":
        half=BATCH//2
        return list(cur[:half]+old[half:]) if phase==0 else list(old[:half]+cur[half:])
    raise ValueError(arm)

def exposures(ledger,arm):
    result=collections.Counter()
    op_counts=collections.Counter()
    for stage in STAGES[1:]:
        for block in ledger[str(stage["index"])]:
            for phase in (0,1):
                ops=task_for_block(stage,block,arm,phase)
                assert len(ops)==BATCH
                for pair,op in zip(block["pairs"],ops):
                    result[(stage["index"],block["width"],pair[0],pair[1],op)]+=1
                    op_counts[(stage["index"],block["width"],op)]+=1
    return result,op_counts

def matched_exposure_check(ledger):
    reh,reht=exposures(ledger,"rehearsal")
    block,blockt=exposures(ledger,"blocked_replay")
    new,newt=exposures(ledger,"new_only")
    assert reh==block and reht==blockt,"REHEARSAL_BLOCK_REPLAY_NOT_EXPOSURE_MATCHED"
    assert sum(reh.values())==sum(block.values())==sum(new.values())==P["total_model_train_examples_per_seed_arm"]
    # Equality is at EACH (stage,width,operand pair,operation); not just counts.
    def format_counts(counter):
        return [{"stage":k[0],"width":k[1],"task":k[2],"count":v}
                for k,v in sorted(counter.items())]
    return {"exact_operand_task_width_matched":True,
            "rehearsal_vs_block_replay_l1_difference":sum((reh-block).values()),
            "per_seed_arm_total_examples":sum(reh.values()),
            "rehearsal_task_width_counts":format_counts(reht),
            "blocked_task_width_counts":format_counts(blockt),
            "new_only_task_width_counts":format_counts(newt),
            "matched_operand_task_multiset_sha256":digest(sorted([[*k,v] for k,v in reh.items()])),
            "new_only_total_updates_examples_matched":True,
            "new_only_per_task_exposure_matched":False}

def tensors(pairs,ops,width):
    a=torch.tensor([p[0] for p in pairs],dtype=torch.int64)
    b=torch.tensor([p[1] for p in pairs],dtype=torch.int64)
    y=torch.zeros((len(pairs),width),dtype=torch.float32)
    weight=torch.ones((len(pairs),width),dtype=torch.float32)/width
    for i,(p,op) in enumerate(zip(pairs,ops)):
        t=truth(p[0],p[1],width,op)
        if op in SCALAR_OPERATIONS:
            y[i,-1]=float(t)
            weight[i,:]=0
            weight[i,-1]=1.
        else:
            for bit in range(width):y[i,bit]=float((t>>bit)&1)
    return a,b,y,weight

def model_start(seed,bundle):
    m=TaskConditionedRLT()
    assert m.counts["total"]==25089
    m.load_state_dict(bundle["general_core_weights"][str(seed)],strict=True)
    return m

def model_state_sha(model):
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def eval_infer(model,pairs,op,w):
    a=torch.tensor([p[0] for p in pairs],dtype=torch.int64)
    b=torch.tensor([p[1] for p in pairs],dtype=torch.int64)
    with torch.no_grad():
        logits=model(a,b,op,width=w)
        return (logits[:,-1]>=0).long().tolist() if op in SCALAR_OPERATIONS else decode_logits(logits).tolist()

def eval_compose(model,pairs,program,w):
    # The program is host-supplied as a diagnostic only; not learned control.
    op1,op2,arg2=program
    first=eval_infer(model,pairs,op1,w)
    second_pairs=[[z,p[1] if arg2=="Y" else p[0]] for z,p in zip(first,pairs)]
    return eval_infer(model,second_pairs,op2,w)

COMPOSITIONS={
    "xor_then_add":("xor","add","Y"),
    "sub_then_xor":("sub","xor","X"),
    "add_then_sub":("add","sub","Y"),
}

def compose_truth(pairs,program,w):
    x,y=map(int,pairs)
    a=truth(x,y,w,program[0])
    return truth(a,y if program[2]=="Y" else x,w,program[1])

def evaluate(model,groups,seed,arm,stage):
    model.eval()
    rows=[]
    trace_rows=[]
    for group,pairs in groups.items():
        w=int(group[3:]) if group.startswith("iid") else int(group[5:])
        heldout_hash=digest(pairs)
        for task in OPERATIONS:
            predictions=eval_infer(model,pairs,task,w)
            expected=[truth(x,y,w,task) for x,y in pairs]
            correct=sum(a==b for a,b in zip(predictions,expected))
            failures=[{"a":x,"b":y,"expected":e,"observed":o} for (x,y),e,o in
                      zip(pairs,expected,predictions) if e!=o]
            rec={"seed":seed,"arm":arm,"stage":stage,"kind":"direct",
                 "group":group,"width":w,"task":task,"tested":len(pairs),
                 "correct":correct,"accuracy":correct/len(pairs),
                 "pair_set_sha256":heldout_hash,"failure_count":len(failures),
                 "failure_examples":failures[:8]}
            rows.append(rec)
            trace_rows.append({"seed":seed,"arm":arm,"stage":stage,"kind":"direct",
                               "group":group,"task":task,"width":w,"pairs":pairs,
                               "prediction":predictions,"truth":expected})
    for group in ("iid8","iid16"):
        w=int(group[3:]);pairs=groups[group][:32]
        for name,program in COMPOSITIONS.items():
            predicted=eval_compose(model,pairs,program,w)
            expected=[compose_truth(pair,program,w) for pair in pairs]
            correct=sum(a==b for a,b in zip(predicted,expected))
            failures=[{"a":x,"b":y,"expected":e,"observed":o} for (x,y),e,o in zip(pairs,expected,predicted) if e!=o]
            rec={"seed":seed,"arm":arm,"stage":stage,"kind":"host_composition",
                 "group":group,"width":w,"task":name,"tested":len(pairs),
                 "correct":correct,"accuracy":correct/len(pairs),
                 "pair_set_sha256":digest(pairs),"failure_count":len(failures),
                 "failure_examples":failures[:8],"controller":"host_not_learned"}
            rows.append(rec)
            trace_rows.append({"seed":seed,"arm":arm,"stage":stage,
                               "kind":"host_composition","group":group,
                               "task":name,"width":w,"pairs":pairs,
                               "prediction":predicted,"truth":expected})
    EVAL.mkdir(parents=True,exist_ok=True)
    path=EVAL/f"seed{seed}_{arm}_stage{stage:02d}_predictions.jsonl"
    path.write_text("".join(canon(r)+"\n" for r in trace_rows))
    return rows,{"path":str(path.relative_to(ROOT)),"sha256":sha_file(path),
                 "rows":len(trace_rows),"predictions":sum(len(r["pairs"]) for r in trace_rows)}

def checkpoint(model,opt,seed,arm,stage,cumulative_updates,exposure_counts,source_hash,
               trace_info,last_loss,stage_seconds):
    CHECKPOINTS.mkdir(parents=True,exist_ok=True)
    package={
        "schema":SCHEMA,"task":"NC-022","source":"copied immutable trained RLT baseline",
        "seed":seed,"arm":arm,"stage":stage,
        "stage_name":STAGES[stage]["name"],
        "updates_total":cumulative_updates,"updates_this_stage":STAGES[stage]["updates"],
        "model_architecture":"TaskConditionedRLT","model_parameters":model.counts["total"],
        "model_state_sha256":model_state_sha(model),
        "model":model.state_dict(),"optimizer":opt.state_dict(),
        "optimizer_name":"AdamW","learning_rate":P["optimizer"]["lr"],
        "random_torch":torch.get_rng_state(),"random_python":random.getstate(),
        "random_numpy":np.random.get_state(),
        "source_sha256":source_hash,"training_exposure_counts":dict(exposure_counts),
        "evaluation_trace":trace_info,"last_loss":last_loss,
        "stage_train_seconds":stage_seconds,
        "optimizer_carried_forward":True,
        "original_champion_modified":False,"promoted":False}
    path=CHECKPOINTS/f"seed{seed}_{arm}_stage{stage:02d}.pt"
    torch.save(package,path)
    return {"path":str(path.relative_to(ROOT)),"sha256":sha_file(path),
            "bytes":path.stat().st_size,"seed":seed,"arm":arm,"stage":stage,
            "updates_total":cumulative_updates,"model_state_sha256":package["model_state_sha256"],
            "evaluation_trace":trace_info,
            "last_loss":last_loss,"stage_train_seconds":stage_seconds,
            "task_exposure_counts":dict(exposure_counts)}

def score_summary(records):
    maps={(r["seed"],r["arm"],r["stage"],r["kind"],r["group"],r["task"]):r
          for r in records}
    out=[]
    for seed in SEEDS:
        for stage in STAGES:
            idx=stage["index"]
            for arm in ARMS:
                direct=[r for r in records if r["seed"]==seed and r["stage"]==idx and
                        r["arm"]==arm and r["kind"]=="direct"]
                assert len(direct)==len(P["evaluation_groups"])*len(OPERATIONS)
                compose=[r for r in records if r["seed"]==seed and r["stage"]==idx and
                         r["arm"]==arm and r["kind"]=="host_composition"]
                assert len(compose)==6
                old=stage["prior"] if idx else []
                old_s=[r for r in direct if r["task"] in old]
                base=[maps[(seed,arm,0,"direct",r["group"],r["task"])] for r in old_s]
                prior=[maps[(seed,arm,idx-1,"direct",r["group"],r["task"])] for r in old_s] if idx else []
                counts={
                  "regressions_vs_baseline_5pp":sum(r["accuracy"]<b["accuracy"]-0.05
                                                       for r,b in zip(old_s,base)),
                  "regressions_vs_previous_5pp":sum(r["accuracy"]<b["accuracy"]-0.05
                                                       for r,b in zip(old_s,prior)),
                  "old_skill_cells":len(old_s),
                  "old_skill_mean_accuracy":sum(r["accuracy"] for r in old_s)/len(old_s) if old_s else None,
                  "old_skill_baseline_mean":sum(r["accuracy"] for r in base)/len(base) if base else None,
                  "old_skill_prev_stage_mean":sum(r["accuracy"] for r in prior)/len(prior) if prior else None
                }
                by_task={op:sum(r["accuracy"] for r in direct if r["task"]==op)/len(P["evaluation_groups"])
                         for op in OPERATIONS}
                out.append({"seed":seed,"arm":arm,"stage":idx,"stage_name":stage["name"],
                            "task_means":by_task,
                            "task_balanced_mean":sum(by_task.values())/len(by_task),
                            "host_composition_mean":sum(r["accuracy"] for r in compose)/len(compose),
                            **counts})
    compare=[]
    for seed in SEEDS:
        for stage in STAGES[2:]:
            idx=stage["index"]
            r={arm:next(q for q in out if q["seed"]==seed and q["stage"]==idx and q["arm"]==arm)
               for arm in ARMS}
            compare.append({
                "seed":seed,"stage":idx,"previous_tasks":stage["prior"],
                "rehearsal_minus_new_only":r["rehearsal"]["old_skill_mean_accuracy"]-
                                            r["new_only"]["old_skill_mean_accuracy"],
                "rehearsal_minus_blocked_replay":r["rehearsal"]["old_skill_mean_accuracy"]-
                                                  r["blocked_replay"]["old_skill_mean_accuracy"],
                "prior_task_scores":{a:r[a]["old_skill_mean_accuracy"] for a in ARMS},
                "regressed_cells_5pp":{a:r[a]["regressions_vs_baseline_5pp"] for a in ARMS}})
    gains=[c["rehearsal_minus_new_only"] for c in compare if c["stage"] in (2,3)]
    blocked=[c["rehearsal_minus_blocked_replay"] for c in compare if c["stage"] in (2,3)]
    matched_effect=bool(gains) and all(v>0 for v in gains) and all(v>0 for v in blocked)
    restoration=[]
    for seed in SEEDS:
        for arm in ARMS:
            r3=next(r for r in out if r["seed"]==seed and r["stage"]==3 and r["arm"]==arm)
            r4=next(r for r in out if r["seed"]==seed and r["stage"]==4 and r["arm"]==arm)
            skills=("add","xor","sub")
            before=sum(r3["task_means"][x] for x in skills)/len(skills)
            after=sum(r4["task_means"][x] for x in skills)/len(skills)
            restoration.append({"seed":seed,"arm":arm,"before":before,"after":after,
                                "delta":after-before,"restored":after>before})
    return {"by_seed_arm_stage":out,"rehearsal_comparisons":compare,
            "strict_reproduced_gain":matched_effect,
            "claim":"GAIN_MATCHED" if matched_effect else "NO_GAIN",
            "restoration":restoration}

def run():
    validate_contract()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    before=source_hashes()
    assert before["snapshot"]==P["source_snapshot_sha256"]
    allrecords=[];allck=[];ledger_out={};exposure_out={}
    bundle=torch.load(SOURCE,map_location="cpu",weights_only=False)
    assert bundle["general_core_selected"]=="task_conditioned_rlt"
    assert set(bundle["general_core_weights"])==set(map(str,SEEDS))
    begin=time.perf_counter()
    for seed in SEEDS:
        groups,heldkeys=heldouts(seed)
        ledger=training_ledger(seed,heldkeys)
        match=matched_exposure_check(ledger)
        ledger_out[str(seed)]={"groups":groups,"blocks":ledger,
                               "train_pair_set_sha256":digest(ledger),
                               "heldout_pair_set_sha256":digest(groups)}
        exposure_out[str(seed)]=match
        for arm in ARMS:
            torch.manual_seed(seed*31+7)
            random.seed(seed*31+7)
            np.random.seed(seed*31+7)
            model=model_start(seed,bundle).cpu().train()
            opt=torch.optim.AdamW(model.parameters(),
                  lr=P["optimizer"]["lr"],weight_decay=P["optimizer"]["weight_decay"])
            total_updates=0;last_loss=None
            exposure=collections.Counter()
            for stage in STAGES:
                t0=time.perf_counter()
                idx=stage["index"]
                if idx>0:
                    for block in ledger[str(idx)]:
                        for phase in (0,1):
                            ops=task_for_block(stage,block,arm,phase)
                            width=block["width"]
                            a,b,target,weight=tensors(block["pairs"],ops,width)
                            model.train()
                            logits=model(a,b,ops,width=width)
                            elem=F.binary_cross_entropy_with_logits(logits,target,reduction="none")
                            loss=(elem*weight).sum(dim=1).mean()
                            opt.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(),P["optimizer"]["clip_grad_norm"])
                            opt.step()
                            total_updates+=1
                            last_loss=float(loss.item())
                            for op in ops:exposure[f"w{width}:{op}"]+=1
                seconds=time.perf_counter()-t0
                rows,trace=evaluate(model,groups,seed,arm,idx)
                allrecords+=rows
                item=checkpoint(model,opt,seed,arm,idx,total_updates,exposure,before,trace,last_loss,seconds)
                allck.append(item)
                print(canon({"seed":seed,"arm":arm,"stage":idx,
                  "updates_total":total_updates,"records":len(rows),
                  "direct_mean":round(sum(r["accuracy"] for r in rows if r["kind"]=="direct")/
                                       sum(r["kind"]=="direct" for r in rows),5),
                  "model_state":item["model_state_sha256"][:12],
                  "duration_seconds":round(seconds,3)}),flush=True)
                model.train()
            assert total_updates==P["update_budget_per_seed_arm"]
            assert sum(exposure.values())==P["total_model_train_examples_per_seed_arm"]
    if source_hashes()!=before:
        raise RuntimeError("NC022_IMMUTABLE_SOURCE_OR_CHAMPION_CHANGED")
    ledger_path=HERE/"train_and_holdout_ledger.json"
    ledger_path.write_text(json.dumps(ledger_out,indent=2,sort_keys=True)+"\n")
    score=score_summary(allrecords)
    matrix={
     "schema":SCHEMA,"task":"NC-022","status":"IMPLEMENTER_TESTED_PENDING_INDEPENDENT",
     "records":allrecords,"summary":score,
     "source_sha256":before,
     "exposure_matching":exposure_out,
     "seeds":list(SEEDS),"arms":list(ARMS),
     "stages":list(STAGES),
     "baseline_model":"frozen pretrained three-seed 25,089-parameter RLT, copied and then separately optimized",
     "neural_inference_without_symbolic_fallback":True,
     "evaluation_pair_sets_sha256":{str(k):v["heldout_pair_set_sha256"]
                                    for k,v in ledger_out.items()},
     "train_and_holdout_ledger_file":"artifacts/NC-022/train_and_holdout_ledger.json",
     "train_and_holdout_ledger_sha256":sha_file(ledger_path),
     "candidate_checkpoint_count":len(allck),
     "old_skill_threshold_pp":5,
     "archived_original_champion_unchanged":True,"promoted":False,
     "limitations":P["scientific_caveats"]+
       ["Exact accuracy is measured on bounded heldout pairs and is not proof for all 16/32-bit operand combinations",
        "Host-built multi-step composition probes use two neural calls and fixed wiring; no learned program counter or control flow"]
    }
    (HERE/"rlt_retention_matrix.json").write_text(json.dumps(matrix,indent=2,sort_keys=True)+"\n")
    exposures_report={
     "schema":SCHEMA,"task":"NC-022","claim":score["claim"],
     "strict_reproduced_gain":score["strict_reproduced_gain"],
     "three_seed_stagewise_comparison":score["rehearsal_comparisons"],
     "restoration":score["restoration"],
     "matched_exposure_proof":exposure_out,
     "training_updates_per_arm_seed":P["update_budget_per_seed_arm"],
     "examples_per_arm_seed":P["total_model_train_examples_per_seed_arm"],
     "stage_order":list(STAGES),"source_sha256":before,
     "comparison_limits":P["scientific_caveats"],
     "compute_accounting":{"seed_count":len(SEEDS),"arm_count":len(ARMS),
       "total_optimizer_updates":len(SEEDS)*len(ARMS)*P["update_budget_per_seed_arm"],
       "total_training_examples":len(SEEDS)*len(ARMS)*P["total_model_train_examples_per_seed_arm"],
       "total_elapsed_wall_seconds":time.perf_counter()-begin,
       "sum_stage_training_seconds":sum(c["stage_train_seconds"] for c in allck)}
    }
    (HERE/"exposure_matched_ablation.json").write_text(json.dumps(exposures_report,indent=2,sort_keys=True)+"\n")
    manifest={
     "schema":SCHEMA,"task":"NC-022","status":"IMPLEMENTER_TESTED_PENDING_INDEPENDENT",
     "source_sha256":before,"protocol_commit":"01a1eb2fdf143e9344026518949b8d8c379af3d9",
     "original_champion_sha256":before["snapshot"],
     "checkpoints":allck,"checkpoint_count":len(allck),
     "evaluation_trace_count":len(allck),
     "evaluations":len(allrecords),
     "train_ledger_sha256":sha_file(ledger_path),
     "retention_matrix_sha256":sha_file(HERE/"rlt_retention_matrix.json"),
     "exposure_matched_ablation_sha256":sha_file(HERE/"exposure_matched_ablation.json"),
     "seed_count":len(SEEDS),"stage_count":len(STAGES),"arm_count":len(ARMS),
     "checkpoint_format":"full model, full optimizer, torch/python/numpy RNG states, stage, source/checkpoint hashes",
     "no_champion_modification":True,"promoted":False}
    (HERE/"checkpoint_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(canon({"status":"DONE_AWAITING_INDEPENDENT",
        "checkpoint_count":len(allck),"evaluation_records":len(allrecords),
        "gain_verdict":score["claim"],
        "three_seed_rehearsal_minus_new_only":
         [[r["seed"],r["stage"],round(r["rehearsal_minus_new_only"],5)] for r in score["rehearsal_comparisons"]],
        "matched_exposure_all":all(r["exact_operand_task_width_matched"] for r in exposure_out.values()),
        "checkpoint_manifest_sha256":sha_file(HERE/"checkpoint_manifest.json")}),flush=True)

if __name__=="__main__":
    run()
