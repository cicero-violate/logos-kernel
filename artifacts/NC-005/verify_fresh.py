#!/usr/bin/env python3
"""NC005 fresh-process frozen-stage model reload, independent numerical datasets.

Author/implementer checks; distinct external agent verification still required.
Dependencies: Python3 stdlib, PyTorch, curriculum.py, NC004 coupled seed
checkpoints, NC000 champion inventory and stage .pt packages.
Only writes artifacts/NC-005/fresh_reload.json.
"""
import hashlib
import json
import random
import sys
from pathlib import Path
import torch
from torch.nn import functional as F
import curriculum as c

HERE=c.HERE
def source_model(seed,arm):
 path=c.CHECKPOINTS/f"{arm}_seed{seed}_stage06.pt"
 record=torch.load(path,map_location="cpu",weights_only=True)
 assert record["task"]=="NC-005" and record["seed"]==seed and record["arm"]==arm
 assert record["stage_index"]==6 and record["total_updates"]==c.STAGE_UPDATES
 assert record["original_champion_sha256"]==c.sha(c.base.CHAMPION)
 assert record["nc004_source_checkpoint_sha256"]==c.sha(c.base.HERE/f"coupled_seed{seed}.pt")
 assert len(record["optimizer"]["state"])>0
 state=record["model"]
 m=c.ExpandedISA()
 m.load_state_dict(state,strict=True)
 m.eval()
 return m,path,record

def full16_pairs(seed):
 rng=random.Random(seed+53509)
 edges=(0,1,2,3,255,256,32767,32768,32769,65534,65535)
 result=[(a,b) for a in edges for b in edges]
 result += [(rng.randrange(65536),rng.randrange(65536)) for _ in range(256)]
 return result

def exhaustive_zero(model):
 failures=[];correct=0
 with torch.no_grad():
  for offset in range(0,65536,4096):
   values=list(range(offset,min(offset+4096,65536)))
   bit=c.bits(values)
   output=(model.zero_logits(bit)>=0).tolist()
   for number,pred in zip(values,output):
    expected=number==0
    correct+=int(pred==expected)
    if pred!=expected and len(failures)<8:
     failures.append({"value":number,"predicted_zero":pred,"expected_zero":expected})
 return {"correct":correct,"tested":65536,"failures":failures,
    "scope":"complete 16bit register-zero predicate, not all ISA machine states"}

def full16_ops(model,seed):
 pairs=full16_pairs(seed)
 x=c.bits([a for a,b in pairs]);y=c.bits([b for a,b in pairs])
 scores={}
 with torch.no_grad():
  for op in c.OPS:
   out,carry=c.step_numeric(model,op,x,y)
   words=c.words(out)
   if op=="ADD":target=[(a+b)&c.MASK for a,b in pairs]
   elif op=="SUB":target=[(a-b)&c.MASK for a,b in pairs]
   elif op=="AND":target=[a&b for a,b in pairs]
   elif op=="OR":target=[a|b for a,b in pairs]
   elif op=="XOR":target=[a^b for a,b in pairs]
   elif op=="AND1":target=[a&1 for a,b in pairs]
   elif op=="SHL":target=[(a<<1)&c.MASK for a,b in pairs]
   elif op=="SHR":target=[a>>1 for a,b in pairs]
   else:target=[(a>>1)|(a&32768) for a,b in pairs]
   fail=[{"a":a,"b":b,"actual":p,"reference":truth}
        for (a,b),p,truth in zip(pairs,words,target) if p!=truth]
   carry_score=None
   if op=="SUB":
    assert carry is not None
    expected=[int(a>=b) for a,b in pairs]
    correct=sum(int(v>=.5)==truth for v,truth in zip(carry.tolist(),expected))
    carry_score={"correct":correct,"tested":len(pairs),
                 "mode":"SUB final no-borrow bit; host collects carry"}
   scores[op]={"correct":len(pairs)-len(fail),"tested":len(pairs),
               "failures":fail[:6],"carry":carry_score}
 return scores

def intervention(model):
 x,y,ids=c.local_table()
 with torch.no_grad():
  normal=model.bit_from_features(x)
  z=model.meaning(x[:,:len(c.OPS)]).softmax(-1)
  rotated=model.bit(torch.cat((z.roll(1,dims=1),x[:,len(c.OPS):]),dim=1))
  base=(normal>=0).eq(y.bool()).all(-1)
  intervened=(rotated>=0).eq(y.bool()).all(-1)
  delta=int((normal.ge(0)!=rotated.ge(0)).any(-1).sum())
 return {"normal_correct":int(base.sum()),"intervened_correct":int(intervened.sum()),
         "tested":len(x),"bits_changed":delta,
         "mechanism":"rotation of ONLY predicted semantic distribution into numeric MLP"}

def main():
 torch.set_num_threads(1)
 inventory=json.loads(c.CHAMPION.read_text())
 assert c.sha(c.base.CHAMPION)==inventory["champion"]["selected_bundle_sha256"]
 report=json.loads((HERE/"training_report.json").read_text())
 assert len(report["results"])==6
 rows={}
 for seed in c.SEEDS:
  for arm in c.ARMS:
   model,path,ckpt=source_model(seed,arm)
   tx,ty,ids=c.local_table()
   trained=c.bit_accuracy(model,tx,ty,ids)
   result={"seed":seed,"arm":arm,"checkpoint":str(path.relative_to(c.ROOT)),
       "checkpoint_sha256":c.sha(path),
       "exact_local":trained,
       "exhaustive_zero_16bit":exhaustive_zero(model),
       "fresh_16bit_arithmetic":full16_ops(model,seed),
       "meaning_intervention":intervention(model),
       "same_training_budget":ckpt["total_updates"],
       "original_champion_sha256":c.sha(c.base.CHAMPION)}
   rows[f"{arm}_seed{seed}"]=result
   print(json.dumps({"name":f"{arm}_seed{seed}",
     "local":f"{trained['correct']}/{trained['tested']}",
     "zero":f"{result['exhaustive_zero_16bit']['correct']}/65536",
     "sixteen_bit_ops":{name:score["correct"] for name,score in result["fresh_16bit_arithmetic"].items()},
     "meaning_intervened_correct":result["meaning_intervention"]["intervened_correct"]},
     sort_keys=True),flush=True)
 doc={"task":"NC-005","status":"IMPLEMENTER_FROZEN_RELOAD",
  "environment":{"python":sys.version.split()[0],"torch":torch.__version__,"cuda":torch.cuda.is_available()},
  "source":"full persisted NC005 stage06 neural and optimizer checkpoints, no retraining",
  "program_control_boundary":"16bit operations neural bit/carry predictions; register code and zero predicate checks separate, no learned general program planner",
  "heldout_scope":"121 edge cross-products + 256 pseudorandom pairs per model, non-guaranteed disjoint from original NC004 pretraining; full 65536 zero predicate enumerated",
  "models":rows,"champion_sha256":c.sha(c.base.CHAMPION),
  "no_promotion":True}
 (HERE/"fresh_reload.json").write_text(json.dumps(doc,sort_keys=True,indent=2,default=str)+"\n")
 print("FROZEN_RELOAD_SHA256",c.sha(HERE/"fresh_reload.json"),flush=True)
if __name__=="__main__":main()
