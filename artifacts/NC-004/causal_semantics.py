#!/usr/bin/env python3
"""NC-004: matched five-primitive neural meaning-channel causal ablation.

Dependencies: Python3, torch, frozen standalone-ISA train_causal.py, NC-001
canonical_isa.py, NC-000 inventory/champion hash, test_dag.py. The instruction
and math AST are rendered via canonical_isa after neural opcode meaning prediction;
NUMERIC forward consumes meaning softmax, NOT the host AST interpreter.
Control arm predicts math as a detached auxiliary classifier but feeds raw
opcode one-hot into the arithmetic bit network. This is not full 15-opcode
training, neural PC/HALT learning, or autonomous program synthesis.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
SOURCE=Path("/workspace/experiments/latticeforge_neural_isa_test")
RLT=Path("/workspace/experiments/rlt_split_algorithms_ablation_v1")
sys.path.insert(0,str(SOURCE))
sys.path.insert(0,str(ROOT/"artifacts/NC-001"))
from train_causal import NeuralISA,OPS,truth_table,encode,decode,W
import canonical_isa as isa

SEEDS=(11,19,29)
ARMS=("coupled","detached")
OPS=tuple(OPS)
STAGES=(("ADD",("ADD",),300),
        ("SHIFTS",("ADD","SHL","SHR"),350),
        ("ALL",OPS,850))
STEPS=sum(z[2] for z in STAGES)
MAP=tuple("bv."+s.lower() for s in OPS)
CHAMPION=RLT/"snapshots/current_unified_v3.pt"
BASELINE_SOURCE=SOURCE/"train_causal.py"
ISA_SOURCE=ROOT/"artifacts/NC-001/canonical_isa.py"
INV=ROOT/"artifacts/NC-000/inventory.json"

def sha(path:Path):
 h=hashlib.sha256()
 with Path(path).open("rb") as fd:
  for c in iter(lambda:fd.read(1<<20),b""):h.update(c)
 return h.hexdigest()

class DetachedMeaningISA(NeuralISA):
 """Same topology/parameter count, numeric forward bypasses meaning head."""
 def bit_from_features(self,features):
  return self.bit(features)

def build(arm):
 if arm=="coupled":return NeuralISA()
 if arm=="detached":return DetachedMeaningISA()
 raise ValueError(arm)

def metrics(model,x,y,ids):
 model.eval()
 with torch.no_grad():
  logits=model.bit_from_features(x)
  good=(logits>=0).eq(y.bool()).all(1)
  labels=model.meaning(x[:,:5]).argmax(1).eq(ids)
 return {"numeric_correct":int(good.sum()),"numeric_tested":len(x),
         "math_labels_correct":int(labels.sum()),"math_labels_tested":len(ids)}

def semantic_intervention(model,arm,x,y):
 """Rotate only predicted meaning distribution; never change raw input opcode."""
 model.eval()
 oh=x[:,:5]
 signal=x[:,5:]
 with torch.no_grad():
  baseline=model.bit_from_features(x)
  predicted=torch.softmax(model.meaning(oh),dim=1)
  intervention=torch.cat((predicted.roll(1,dims=1),signal),dim=1)
  if arm=="detached":
   intervened=model.bit(torch.cat((oh,signal),dim=1))
  else:
   intervened=model.bit(intervention)
  original=(baseline>=0).eq(y.bool()).all(1)
  corrupted=(intervened>=0).eq(y.bool()).all(1)
  changes=int((baseline.ge(0)!=intervened.ge(0)).any(1).sum())
 return {"original_correct":int(original.sum()),"intervened_correct":int(corrupted.sum()),
         "tested":len(y),"outputs_changed":changes,
         "numerical_correctness_degradation":int(original.sum())-int(corrupted.sum())}

def numeric_gradient(model,x,y):
 model.train()
 loss=F.binary_cross_entropy_with_logits(model.bit_from_features(x),y)
 grad=torch.autograd.grad(loss,model.meaning.weight,allow_unused=True)[0]
 model.eval()
 return {"numeric_loss":float(loss.item()),
         "meaning_weight_gradient_l1":float(grad.abs().sum()) if grad is not None else 0.,
         "meaning_head_in_numeric_graph":grad is not None}

def save_model(path,model,opt,seed,arm,loss):
 checkpoint={
  "model":model.state_dict(), "optimizer":opt.state_dict(),
  "seed":seed,"arm":arm,"trained_updates":STEPS,
  "seed_rng_python":random.getstate(),"seed_rng_torch":torch.get_rng_state(),
  "seed_rng_numpy":{"bit_generator":np.random.get_state()[0],
     "keys":np.random.get_state()[1].tolist(),"position":np.random.get_state()[2],
     "has_gaussian_cache":np.random.get_state()[3],
     "cached_gaussian":float(np.random.get_state()[4])}, "last_loss":loss,
  "operations":OPS, "canonical_math_AST_kinds":MAP,
  "architecture":"train_causal.NeuralISA" if arm=="coupled" else "DetachedMeaningISA",
  "checkpoint_kind":"new_experimental_five_primitive_training_no_promotion",
  "source_sha256":sha(BASELINE_SOURCE),"canonical_isa_sha256":sha(ISA_SOURCE),
  "original_champion_sha256":sha(CHAMPION),
 }
 torch.save(checkpoint,path)

def train(seed,arm):
 torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
 model=build(arm)
 x,y,ids=truth_table()
 assert len(x)==320 and len(OPS)==5
 opt=torch.optim.Adam(model.parameters(),lr=.015)
 history=[]
 lossval=None
 for stage,allowed,count in STAGES:
  keep=torch.tensor([i for i,z in enumerate(ids.tolist()) if OPS[z] in allowed],dtype=torch.long)
  for step in range(count):
   opt.zero_grad(set_to_none=True)
   pred=model.bit_from_features(x[keep])
   loss=F.binary_cross_entropy_with_logits(pred,y[keep])
   semantic_loss=F.cross_entropy(model.meaning(x[keep,:5]),ids[keep])
   total=loss+.2*semantic_loss
   total.backward()
   opt.step()
   lossval=float(total.item())
  observed=metrics(model,x,y,ids)
  history.append({"phase":stage,"updates":count,"cumulative_updates":sum(z[2] for z in STAGES[:len(history)+1]),
                  "whole_320_numeric_correct":observed["numeric_correct"],
                  "whole_320_math_labels_correct":observed["math_labels_correct"],
                  "last_total_loss":lossval})
 out=HERE/f"{arm}_seed{seed}.pt"
 save_model(out,model,opt,seed,arm,lossval)
 return {"seed":seed,"arm":arm,"updates":STEPS,"history":history,
         "checkpoint":out.name,"checkpoint_sha256":sha(out),
         "parameter_count":sum(p.numel() for p in model.parameters()),
         "local":metrics(model,x,y,ids),
         "numeric_gradient":numeric_gradient(model,x,y),
         "intervention":semantic_intervention(model,arm,x,y)}

def load(seed,arm):
 path=HERE/f"{arm}_seed{seed}.pt"
 checkpoint=torch.load(path,weights_only=True,map_location="cpu")
 assert checkpoint["seed"]==seed and checkpoint["arm"]==arm
 assert checkpoint["operations"]==OPS and checkpoint["canonical_math_AST_kinds"]==MAP
 assert checkpoint["source_sha256"]==sha(BASELINE_SOURCE)
 assert checkpoint["canonical_isa_sha256"]==sha(ISA_SOURCE)
 assert checkpoint["original_champion_sha256"]==sha(CHAMPION)
 torch.manual_seed(seed)
 model=build(arm)
 model.load_state_dict(checkpoint["model"],strict=True)
 model.eval()
 return model

@torch.no_grad()
def step_neural(model,op,a,b=None):
 """Learned 16bit bit transition. Host encodes features and updates registers."""
 if b is None:b=torch.zeros_like(a)
 assert op in OPS and a.shape==b.shape and a.ndim==2 and a.shape[-1]==W
 n=len(a)
 if op=="ADD":
  carry=torch.zeros(n);zeros=torch.zeros(n)
  result=torch.zeros_like(a)
  for i in range(W):
   code=F.one_hot(torch.full((n,),OPS.index(op)),len(OPS)).float()
   signal=torch.stack((a[:,i],b[:,i],carry,zeros,zeros,zeros),dim=1)
   logits=model.bit_from_features(torch.cat((code,signal),dim=1))
   prediction=logits.ge(0).float()
   result[:,i]=prediction[:,0]
   carry=prediction[:,1]
  return result
 prev=torch.cat((torch.zeros_like(a[:,:1]),a[:,:-1]),dim=1).flatten()
 following=torch.cat((a[:,1:],torch.zeros_like(a[:,:1])),dim=1).flatten()
 j=torch.arange(W).repeat(n)
 z=torch.zeros(n*W)
 signal=torch.stack((a.flatten(),b.flatten(),z,prev,following,(j==0).float()),dim=1)
 code=F.one_hot(torch.full((n*W,),OPS.index(op)),len(OPS)).float()
 logits=model.bit_from_features(torch.cat((code,signal),dim=1))
 return logits.ge(0).float()[:,0].reshape(n,W)

def instruction(op):
 if op in ("ADD","OR"):
  return {"op":op,"dst":"R","a":"X","b":"Y"}
 return {"op":op,"dst":"R","src":"X"}

def predicted_ast(model,op):
 # A template class is learned; canonical AST rendering is deterministic host code.
 vec=F.one_hot(torch.tensor([OPS.index(op)]),len(OPS)).float()
 with torch.no_grad():
  predicted=OPS[int(model.meaning(vec).argmax(1).item())]
 prog={"version":isa.VERSION,"width":16,"signedness":"unsigned",
       "program":[instruction(predicted),{"op":"HALT"}]}
 ast=isa.lower(prog,0)["write"]["value"]
 isa.validate_ast(ast,16,"unsigned")
 return predicted,ast

def execute_program(model,ops,a,b):
 """Neural value transitions, fixed supplied program sequence (no neural planner)."""
 x=encode(a);y=encode(b)
 r=torch.zeros_like(x)
 for op,src_a,src_b in ops:
  operands={"X":x,"Y":y,"R":r}
  r=step_neural(model,op,operands[src_a],operands[src_b] if src_b else None)
 return decode(r)

def reference_program(ops,a,b):
 instructions=[]
 for op,src_a,src_b in ops:
  if op in ("ADD","OR"):
   instructions.append({"op":op,"dst":"R","a":src_a,"b":src_b})
  else:
   assert src_b is None
   instructions.append({"op":op,"dst":"R","src":src_a})
 instructions.append({"op":"HALT"})
 p={"version":isa.VERSION,"width":16,"signedness":"unsigned","program":instructions}
 return [isa.reference(p,{"X":x,"Y":y,"R":0,"T":0},max_steps=len(instructions)+1)["registers"]["R"]
         for x,y in zip(a,b)]

def calculate_composition(model,seed,arm):
 # The values/whole-program ordering are untrained even though all primitive
 # local bit patterns appear in training. No multiplication program is invented.
 rng=random.Random(17011)
 pairs4=[(x,y) for x in range(16) for y in range(16)]
 pairs8=[(rng.randrange(256),rng.randrange(256)) for _ in range(512)]
 programs={
  "double_plus":(("SHL","X",None),("ADD","R","Y")),
  "lowbit_or":(("AND1","X",None),("OR","R","Y")),
  "half_plus":(("SHR","X",None),("ADD","R","Y")),
 }
 results={}
 for pname,ops in programs.items():
  for split,pairs in (("fourbit_all",pairs4),("eightbit_sample",pairs8)):
   aa=[x for x,y in pairs];bb=[y for x,y in pairs]
   pred=execute_program(model,ops,aa,bb)
   ref=reference_program(ops,aa,bb)
   failed=[{"a":x,"b":y,"predicted":v,"truth":t} for (x,y),v,t in zip(pairs,pred,ref) if v!=t]
   results[pname+"_"+split]={"correct":len(pairs)-len(failed),"tested":len(pairs),
                                "failures":failed[:6]}
 return results

def eval_all():
 inv=json.loads(INV.read_text())
 assert sha(CHAMPION)==inv["champion"]["selected_bundle_sha256"]
 if sha(BASELINE_SOURCE)!={z["path"]:z["sha256"] for z in inv["files"]
                           if z["repo"]=="standalone_isa"}["train_causal.py"]:
  raise RuntimeError("ORIGINAL_STANDALONE_SOURCE_CHANGED")
 torch.set_num_threads(1)
 x,y,ids=truth_table()
 report={"task":"NC-004","scope":"five_of_fifteen_canonical_ISA_primitives",
         "experimental":True,"promoted":False,
         "canonical_isa_version":isa.VERSION,
         "learned_ops":list(OPS),"all_canonical_ops":sorted(isa.FIELDS),
         "training_local_patterns":320,"training_local_patterns_seen":320,
         "three_seeds":list(SEEDS),"training_updates_per_arm_seed":STEPS,
         "models":{},"baseline_source_sha256":sha(BASELINE_SOURCE),
         "canonical_isa_source_sha256":sha(ISA_SOURCE),
         "original_champion_sha256":sha(CHAMPION),
         "original_champion_unchanged":True,
         "semantics_boundary":{
          "learned":"ISA five-class meaning logits and 16-bit bit-transition arithmetic",
          "host":"deterministic canonical-typed-AST rendering, operand/register routing, flags and PC/HALT",
          "oracle":"canonical_isa.reference called only after neural answer",
          "composition":"user-supplied fixed 2-instruction programs, not model invented",
          "unqualified":"remaining 10 canonical ISA opcodes, learned flags/PC/HALT, exact 64-bit semantics"
         }}
 for arm in ARMS:
  for seed in SEEDS:
   model=load(seed,arm)
   local=metrics(model,x,y,ids)
   grad=numeric_gradient(model,x,y)
   intervention=semantic_intervention(model,arm,x,y)
   labels={}
   for op in OPS:
    pred,ast=predicted_ast(model,op)
    expected={"version":isa.VERSION,"width":16,"signedness":"unsigned",
              "program":[instruction(op),{"op":"HALT"}]}
    labels[op]={"predicted_meaning":pred,"expected_typed_ast":isa.lower(expected,0)["write"]["value"],
                "predicted_typed_ast":ast,
                "matches_reference":pred==op and ast==isa.lower(expected,0)["write"]["value"]}
   # Complete two-instruction interpreter comparison: host routes registers, flags and PC
   # while only the arithmetic result is a learned neural prediction. Flags and PC unlearned.
   full_state={}
   aa=[i for i in range(16) for j in range(16)]
   bb=[j for i in range(16) for j in range(16)]
   for op in OPS:
    pred=execute_program(model,[(op,"X","Y" if op in ("ADD","OR") else None)],aa,bb)
    truth=reference_program([(op,"X","Y" if op in ("ADD","OR") else None)],aa,bb)
    failures=[{"a":a,"b":b,"value":p,"expected":v} for a,b,p,v in zip(aa,bb,pred,truth) if p!=v]
    full_state[op]={"correct_value":len(aa)-len(failures),"tested":len(aa),"failures":failures[:6],
                    "host_pc_halt_flags_unlearned":True}
   comps=calculate_composition(model,seed,arm)
   report["models"][arm+"_seed"+str(seed)]={"seed":seed,"arm":arm,
       "local":local,"gradient":grad,"intervention":intervention,
       "canonical_ast_labels":labels,"single_op_full16_state_value":full_state,
       "compositions":comps}
   print(json.dumps({"seed":seed,"arm":arm,"numeric":local["numeric_correct"],"math":local["math_labels_correct"],
      "gradient_l1":grad["meaning_weight_gradient_l1"],"intervened_correct":intervention["intervened_correct"],
      "compositions":{k: f"{v['correct']}/{v['tested']}" for k,v in comps.items()}}),flush=True)
 (HERE/"semantic_receipt.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print("RECEIPT_SHA256",sha(HERE/"semantic_receipt.json"))

def run_training():
 assert sha(CHAMPION)==json.loads(INV.read_text())["champion"]["selected_bundle_sha256"]
 torch.set_num_threads(1)
 results=[]
 for seed in SEEDS:
  for arm in ARMS:
   result=train(seed,arm)
   results.append(result)
   print(json.dumps({"seed":seed,"arm":arm,"local":result["local"],
                     "gradient":result["numeric_gradient"],"intervention":result["intervention"]}),flush=True)
 manifest={"task":"NC-004","kind":"experimental_new_training_not_champion",
   "seed_list":list(SEEDS),"arm_list":list(ARMS),"steps_each":STEPS,"training_curriculum":STAGES,
   "models":results,
   "canonical_isa_version":isa.VERSION,
   "source_files":{str(p):sha(p) for p in (BASELINE_SOURCE,ISA_SOURCE,INV)},
   "original_champion_checkpoint_sha256":sha(CHAMPION),
   "original_weights_modified":False,"promoted":False,
   "claims_not_permitted":["all 15 ISA opcodes learned","learned PC/HALT/flags",
                            "unseen local primitive combinations","program synthesis",
                            "general reasoning"] }
 (HERE/"checkpoint.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
 print("MANIFEST_SHA256",sha(HERE/"checkpoint.json"))

if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("mode",choices=("train","evaluate"))
 opt=p.parse_args()
 if opt.mode=="train":run_training()
 else:eval_all()
