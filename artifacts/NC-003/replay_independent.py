#!/usr/bin/env python3
"""NC-003 frozen-model baseline verifier, separate inference host implementation.

Dependencies:
 Python 3 stdlib + PyTorch;
 /workspace/experiments/rlt_split_algorithms_ablation_v1/neural_isa_transfer_v1/unified_shared_v1/model.py
 and its rlt_micro_v2/rlt_micro/rlt_core/task_conditioned_rlt module dependencies;
 immutable seed11/19/29 native_v2 and jointly_tuned/frozen_shared .pt checkpoints,
 source seven-task pretrained .pt; NC-000 inventory.json.
Does NOT import train.py, verify_shared.py, verify_hybrid_independent.py,
runtime.py, original program teacher, reference multiply during rollout, or MLP.
Writes ONLY artifacts/NC-003/verification.json.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
import platform
import random
import sys
from collections import Counter
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("/workspace/experiments/rlt_split_algorithms_ablation_v1")
TRANSFER = SOURCE/"neural_isa_transfer_v1"
SHARED = TRANSFER/"unified_shared_v1"
PRETRAIN = SOURCE/"task_conditioned_rlt_v1/multitask_train_v1/outputs/seed11/seed11_conditioned_step640.pt"
for p in (SOURCE,TRANSFER,SHARED,TRANSFER/"vendor",TRANSFER/"vendor/transformer_controller"):
    sys.path.insert(0,str(p))
from model import SharedRLTISA,ROM
WIDTH=16
ISA={"ADD":0,"SHL":1,"SHR":2,"AND1":3,"OR":4}
EDGES=(0,1,2,3,7,15,31,63,127,128,255)
SEED_BASE=11052026

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()

def frozen_load(seed,arm):
    training=json.loads((SHARED/"outputs"/f"seed{seed}"/f"{arm}_train.json").read_text())
    assert training["status"]=="TRAINING_COMPLETE" and training["seed"]==seed
    for path,digest in training["source_hashes"].items():
        assert sha(path)==digest, ("SOURCE_MODIFIED",path)
    weights=SHARED/"outputs"/f"seed{seed}"/f"{arm}.pt"
    assert sha(weights)==training["checkpoint_sha256"]
    source_weights=TRANSFER/"rlt_micro_checkpoints"/f"seed{seed}_v2.pt"
    source=torch.load(source_weights,weights_only=True,map_location="cpu")
    model=SharedRLTISA(PRETRAIN,source["state_dict"])
    saved=torch.load(weights,weights_only=True,map_location="cpu")
    assert saved["seed"]==seed and saved["arm"]==arm and saved["shared_attention"]
    model.load_state_dict(saved["model"],strict=True)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    structural=model.structural_proof()
    assert structural["shared_causal_attention_modules"]==1
    return model,{"seed":seed,"arm":arm,"weights":str(weights),
                  "sha256":sha(weights),"parameter_count":structural["parameters"],
                  "source_pretrained_sha256":sha(PRETRAIN)}

def bits(values):
    return torch.tensor([[(int(v)>>bit)&1 for bit in range(WIDTH)] for v in values],dtype=torch.float32)

def words(bitrows):
    return [sum((1<<bit) for bit,val in enumerate(row) if val>0.5) for row in bitrows.tolist()]

def local_primitive(model,name,left,right=None):
    """A second, independently structured host ISA implementation, predictions only."""
    n=len(left)
    if right is None:
        right=torch.zeros_like(left)
    opid=ISA[name]
    if name=="ADD":
        carry=torch.zeros(n)
        output=torch.empty_like(left)
        for bit in range(WIDTH):
            features=torch.stack((left[:,bit],right[:,bit],carry,
                                  torch.zeros(n),torch.zeros(n),torch.zeros(n)),dim=1)
            logits,_=model.local_logits(torch.full((n,),opid,dtype=torch.long),features)
            result=(logits>=0)
            output[:,bit]=result[:,0].float()
            carry=result[:,1].float()
        return output
    prev=torch.nn.functional.pad(left[:,:-1],(1,0))
    after=torch.nn.functional.pad(left[:,1:],(0,1))
    col=torch.arange(WIDTH).repeat(n)
    zero=torch.zeros(n*WIDTH)
    features=torch.stack((left.reshape(-1),right.reshape(-1),zero,
                          prev.reshape(-1),after.reshape(-1), (col==0).float()),dim=1)
    logits,_=model.local_logits(torch.full((n*WIDTH,),opid,dtype=torch.long),features)
    return (logits[:,0]>=0).float().view(n,WIDTH)

def independent_primitive_table(model):
    op=[];signal=[];answer=[]
    for code in range(5):
        for a,b,c,p,q,f in itertools.product((0,1),repeat=6):
            result=(
                ((a+b+c)&1, int(a+b+c>=2)) if code==0 else
                (p,0) if code==1 else (q,0) if code==2 else
                (a&f,0) if code==3 else (a|b,0)
            )
            op.append(code);signal.append((a,b,c,p,q,f));answer.append(result)
    ids=torch.tensor(op,dtype=torch.long)
    raw=torch.tensor(signal,dtype=torch.float32)
    expectation=torch.tensor(answer,dtype=torch.float32)
    logits,language=model.local_logits(ids,raw)
    bits_out=(logits>=0).float()
    good=((bits_out==expectation).all(dim=1))
    syntax=(language.argmax(dim=1)==ids)
    first=[int(i) for i in torch.nonzero(~good).reshape(-1)[:8].tolist()]
    return {"local_isa":{"correct":int(good.sum()),"tested":len(ids),"fail_rows":first},
            "opcode_semantics":{"correct":int(syntax.sum()),"tested":len(ids)},
            "all_320_truth_patterns_supervised":True}

def reference_next(pc,breg,treg):
    """Teacher/reference logic used ONLY for scoring predicted controller transitions."""
    if pc==0:return 7 if breg==0 else 1
    if pc==2:return 4 if treg==0 else 3
    if pc==6:return 0
    if pc==7:return 7
    return pc+1

def reproduce(model,pairs, *, poison_pc=False,zero_local=False):
    """No multiplication truth or oracle enters controller inputs/host updates."""
    n=len(pairs)
    a=bits([x for x,y in pairs])
    b=bits([y for x,y in pairs])
    result=torch.zeros_like(a)
    scratch=torch.zeros_like(a)
    pc=torch.zeros(n,dtype=torch.long)
    halted=torch.zeros(n,dtype=torch.bool)
    op_matches=0;pc_matches=0;states=0
    cases_branch=Counter()
    for cycle in range(75):
        if bool(halted.all()):break
        active=~halted
        regs=torch.stack((a,b,result,scratch),dim=1)
        action_logits,next_logits,stop_logits=model(regs,pc)
        op=action_logits.argmax(dim=-1)
        next_pc=next_logits.argmax(dim=-1)
        stop=(stop_logits.argmax(dim=-1)==1)|(op==7)
        if poison_pc: next_pc=(next_pc+1)%8
        active_idx=active.nonzero().reshape(-1)
        current_pc=pc[active_idx].tolist()
        curr_b=words(b[active_idx]);curr_t=words(scratch[active_idx])
        for j,oldpc in enumerate(current_pc):
            if not 0<=oldpc<len(ROM):continue
            idx=int(active_idx[j])
            expected=reference_next(oldpc,curr_b[j],curr_t[j])
            op_matches+=int(op[idx].item()==oldpc)
            pc_matches+=int(next_pc[idx].item()==expected)
            states+=1
            if oldpc in (0,2):
                cases_branch[f"PC{oldpc}_{'zero' if (curr_b[j] if oldpc==0 else curr_t[j])==0 else 'nonzero'}"]+=1
        for opcode,name,dest,x,y in ((1,"AND1","scratch",b,None),
                                      (3,"ADD","result",result,a),
                                      (4,"SHL","a",a,None),
                                      (5,"SHR","b",b,None)):
            selected=active&(op==opcode)
            if not selected.any():continue
            computed=local_primitive(model,name,x[selected],y[selected] if y is not None else None)
            if zero_local:computed=torch.zeros_like(computed)
            if dest=="scratch":scratch[selected]=computed
            elif dest=="result":result[selected]=computed
            elif dest=="a":a[selected]=computed
            else:b[selected]=computed
        halted=halted|(active&stop)
        pc=torch.where(active&(~stop),next_pc,pc)
    predictions=words(result)
    hits=[int(pred==x*y and bool(h)) for pred,(x,y),h in zip(predictions,pairs,halted.tolist())]
    failures=[{"a":int(x),"b":int(y),"actual":predictions[i],"oracle":int(x*y),
               "halted":bool(halted[i])}
              for i,(x,y) in enumerate(pairs) if not hits[i]]
    return {"tested":n,"correct":sum(hits),"failures":failures[:16],
            "failure_count":len(failures),
            "halted":int(halted.sum()),"unhalted":int((~halted).sum()),
            "controller_decision_steps":states,
            "op_rom_agreement":{"correct":op_matches,"tested":states},
            "pc_teacher_agreement":{"correct":pc_matches,"tested":states},
            "observed_branch_state_counts":dict(cases_branch),
            "neural_forward_and_logits_only":True,"oracle_is_post_prediction_only":True}

def run():
    arg=argparse.ArgumentParser()
    arg.add_argument("--random8",type=int,default=4096)
    arg.add_argument("--batch",type=int,default=128)
    args=arg.parse_args()
    assert args.random8==4096, "NC003 registered historical 4217-case scope"
    assert 1<=args.batch<=256
    torch.set_num_threads(1)
    all4=[(a,b) for a in range(16) for b in range(16)]
    randomizer=random.Random(SEED_BASE+991122)
    all8=[(randomizer.randrange(256),randomizer.randrange(256)) for _ in range(args.random8)]
    all8 +=[(a,b) for a in EDGES for b in EDGES]
    assert len(all8)==4217
    probe=[(13,11),(255,255),(37,63),(128,128),(16,16),(7,9),(127,128),(3,2)]
    baseline=ROOT/"artifacts/NC-000/inventory.json"
    inventory=json.loads(baseline.read_text())
    champ=SOURCE/"snapshots/current_unified_v3.pt"
    assert sha(champ)==inventory["champion"]["selected_bundle_sha256"]
    result={"task":"NC-003","status":"RUNNING","role":"IMPLEMENTER_FROZEN_REPLAY",
            "not_independent_agent_verification":True,
            "evaluation":{"fourbit":{"tested":256,"exhaustive":True},
               "eightbit":{"tested":4217,"random":4096,"edge_cross_product":121,
                           "all_65536_pairs":False,"shared_pairs_across_seeds":True,
                           "sampling_seed":SEED_BASE+991122}},
            "runtime":{"python":platform.python_version(),"torch":torch.__version__,
                       "cuda":torch.cuda.is_available(),"device":"CPU","threads":1},
            "source_repos":{"rlt":str(SOURCE)},
            "original_champion_sha256":sha(champ),
            "source_checkpoint_pretrain_sha256":sha(PRETRAIN),
            "arm_results":{}}
    with torch.no_grad():
        for arm in ("jointly_tuned","frozen_shared"):
            rows=[]
            for seed in (11,19,29):
                model,meta=frozen_load(seed,arm)
                primitives=independent_primitive_table(model)
                checks={}
                for name,data in (("fourbit",all4),("eightbit",all8)):
                    segments=[reproduce(model,data[i:i+args.batch]) for i in range(0,len(data),args.batch)]
                    checks[name]={
                       "correct":sum(z["correct"] for z in segments),
                       "tested":sum(z["tested"] for z in segments),
                       "failure_count":sum(z["failure_count"] for z in segments),
                       "failures":[a for z in segments for a in z["failures"]][:16],
                       "halted":sum(z["halted"] for z in segments),
                       "unhalted":sum(z["unhalted"] for z in segments),
                       "controller_decision_steps":sum(z["controller_decision_steps"] for z in segments),
                       "op_rom_agreement":{"correct":sum(z["op_rom_agreement"]["correct"] for z in segments),
                                           "tested":sum(z["op_rom_agreement"]["tested"] for z in segments)},
                       "pc_teacher_agreement":{"correct":sum(z["pc_teacher_agreement"]["correct"] for z in segments),
                                               "tested":sum(z["pc_teacher_agreement"]["tested"] for z in segments)},
                       "branch_counts":dict(sum((Counter(z["observed_branch_state_counts"]) for z in segments),Counter()))
                    }
                pc_corrupt=reproduce(model,probe,poison_pc=True)
                arithmetic_corrupt=reproduce(model,probe,zero_local=True)
                normal=reproduce(model,probe)
                status=("PASS" if primitives["local_isa"]["correct"]==320 and
                        primitives["opcode_semantics"]["correct"]==320 and
                        checks["fourbit"]["correct"]==256 and
                        checks["eightbit"]["correct"]==4217 else "FAIL")
                row={"seed":seed,**meta,**primitives,
                     **checks,"eightcase_control":normal,
                     "pc_corruption_control":pc_corrupt,
                     "zero_arithmetic_control":arithmetic_corrupt,
                     "fixed_rom":list(ROM),"controller_teacher_contains_program_states":True,
                     "no_teacher_PC_or_next_action_in_forward":True,
                     "no_input_mul_answer_labels_during_this_integration":True,
                     "original_pretrained_RLT_included_mul_training":True,
                     "status":status}
                rows.append(row)
                print(json.dumps({"arm":arm,"seed":seed,
                                  "primitive":primitives["local_isa"]["correct"],
                                  "4bit":checks["fourbit"]["correct"],
                                  "8bit":checks["eightbit"]["correct"],
                                  "status":status},sort_keys=True),flush=True)
            result["arm_results"][arm]=rows
    result["selected_arm"]="jointly_tuned"
    result["excluded_arms"]=["frozen_shared"]
    result["status"]=("IMPLEMENTER_PASS_QUALIFIED_ARM" if all(
            row["status"]=="PASS" for row in result["arm_results"]["jointly_tuned"])
            else "IMPLEMENTER_FAIL_QUALIFIED_ARM")
    result["limitations"]=[
        "Only 256 four-bit pairs exhaustive; 4217 eight-bit pairs sampled/structured, not 65536 exhaustive",
        "All 320 local binary patterns supervised; this cannot establish Boolean rule transfer",
        "Controller training supervised fixed ROM states; teacher-forcing/train-state leakage not ruled out by testing",
        "Predicted controller instructions and local arithmetic logits executed by Python host fixed ROM and wiring",
        "Original seven-task pretrained RLT already trained on multiplication; no pristine pretraining-disjoint proof",
        "Frozen_shared seed19 historical 4216/4217 is an actual failure; cannot promote that arm",
        "No model weights trained, checkpoint promoted, or inferred new multiplication instruction",
        "Fresh independently implemented host replay is authored by current task owner; separate real agent must verify."
    ]
    out=ROOT/"artifacts/NC-003/verification.json"
    out.write_text(json.dumps(result,indent=2,sort_keys=True,default=str)+"\n")
    print(json.dumps({"result":result["status"],"output":str(out),
                      "qualified":{str(r["seed"]):r["eightbit"]["correct"]
                                   for r in result["arm_results"]["jointly_tuned"]},
                      "frozen":{str(r["seed"]):r["eightbit"]["correct"]
                                for r in result["arm_results"]["frozen_shared"]}},
                     indent=2),flush=True)
    if result["status"]!="IMPLEMENTER_PASS_QUALIFIED_ARM":raise SystemExit(2)
if __name__=="__main__":
    run()
