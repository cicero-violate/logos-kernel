#!/usr/bin/env python3
"""NC-005: warm-start progressive neural ISA curricula and explicit forgetting.

Dependencies: Python3, PyTorch, NumPy, NC-004/causal_semantics.py +
coupled_seed{11,19,29}.pt, NC-001/canonical_isa.py, NC-000/inventory.json,
the original standalone train_causal.py and its frozen source hash.
Only files below artifacts/NC-005/ are written; no original weights changed.

Scope: a new extended MLP bit/meaning model and a learned zero detector.
Instruction selection, PC arithmetic, flags, ROM, program composition and
register routing are HOST-provided, not learned autonomy or general AGI.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(ROOT/"artifacts/NC-004"))
import causal_semantics as base
ISA=base.isa
OLD=tuple(base.OPS)
OPS=(*OLD,"AND","XOR","SUB","SAR")
assert len(OPS)==9 and len(set(OPS))==9
INDEX={op:i for i,op in enumerate(OPS)}
SEEDS=(11,19,29)
ARMS=("rehearsal","new_only")
WIDTH=16
MASK=(1<<WIDTH)-1
STAGES=(
 ("BOOLEAN", ("AND1","OR","AND","XOR"), 300),
 ("CARRY_ADD", ("ADD",), 300),
 ("SHIFTS", ("SHL","SHR","SAR"), 300),
 ("COMPARE_SUB", ("SUB",), 350),
 ("PC_BRANCH", (), 400),
 ("SEQUENCES", ("AND1","OR","ADD","SHL","SHR","AND","XOR","SUB","SAR"), 250),
 ("LOOPS", ("AND1","ADD","SHL","SHR"), 350),
)
STAGE_UPDATES=sum(x[2] for x in STAGES)
BATCH=256
CHECKPOINTS=HERE/"checkpoints"
CHAMPION=ROOT/"artifacts/NC-000/inventory.json"
PREFIX_SOURCE=ROOT/"artifacts/NC-004/causal_semantics.py"

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()

class ExpandedISA(nn.Module):
 """One trainable bit/meaning network and learned zero head, no symbolic fallback."""
 def __init__(self):
  super().__init__()
  self.meaning=nn.Linear(len(OPS),len(OPS))
  self.bit=nn.Sequential(nn.Linear(len(OPS)+6,64),nn.Tanh(),
                         nn.Linear(64,64),nn.Tanh(),nn.Linear(64,2))
  self.is_zero=nn.Sequential(nn.Linear(WIDTH,32),nn.Tanh(),nn.Linear(32,1))
 def bit_from_features(self,x):
  z=self.meaning(x[:,:len(OPS)]).softmax(-1)
  return self.bit(torch.cat((z,x[:,len(OPS):]),-1))
 def local(self,op,signal):
  n=len(signal)
  ids=torch.full((n,),INDEX[op],dtype=torch.long)
  x=torch.cat((F.one_hot(ids,len(OPS)).float(),signal),1)
  return self.bit_from_features(x)
 def zero_logits(self,bits):
  return self.is_zero(bits).flatten()

def warm_start(seed):
 """Transfer all NC-004 tensors to an expanded model; no overwrite of source."""
 torch.manual_seed(seed)
 src=base.load(seed,"coupled")
 m=ExpandedISA()
 with torch.no_grad():
  # Existing five opcode columns retained; new target classes strongly suppressed
  # on pre-existing opcode inputs. New classes activate for their own opcode.
  m.meaning.weight.zero_()
  m.meaning.bias[:len(OLD)].copy_(src.meaning.bias)
  m.meaning.weight[:len(OLD),:len(OLD)].copy_(src.meaning.weight)
  m.meaning.bias[len(OLD):].fill_(-15.)
  for i in range(len(OLD),len(OPS)):
   m.meaning.weight[i,i]=20.
   m.meaning.weight[:len(OLD),i].fill_(-5.)
  m.bit[0].weight.zero_()
  m.bit[0].weight[:,:len(OLD)].copy_(src.bit[0].weight[:,:len(OLD)])
  m.bit[0].weight[:,len(OPS):].copy_(src.bit[0].weight[:,len(OLD):])
  # New columns start as trainable small deterministic weights.
  m.bit[0].weight[:,len(OLD):len(OPS)].uniform_(-.03,.03)
  m.bit[0].bias.copy_(src.bit[0].bias)
  for i in (2,4):
   m.bit[i].load_state_dict(src.bit[i].state_dict())
  m.is_zero.load_state_dict(src.is_zero.state_dict())
 return m

def feature(op,a,b,c,p,n,f):
 return [float(i==INDEX[op]) for i in range(len(OPS))]+[a,b,c,p,n,f]

def primitive_target(op,a,b,c,p,n,f):
 if op=="ADD":
  s=a+b+c; return (s&1,int(s>=2))
 if op=="SUB":
  s=a+(1-b)+c; return (s&1,int(s>=2)) # c=1 for the least-significant bit
 if op=="AND":return (a&b,0)
 if op=="OR":return (a|b,0)
 if op=="XOR":return (a^b,0)
 if op=="AND1":return (a&f,0)
 if op=="SHL":return (p,0)
 if op=="SHR":return (n,0)
 if op=="SAR":return (a if f else n,0) # f marks sign/top bit in SAR
 raise ValueError(op)

def local_table():
 rows=[];target=[];which=[]
 for op in OPS:
  for a,b,c,p,n,f in itertools.product((0.,1.),repeat=6):
   rows.append(feature(op,a,b,c,p,n,f))
   target.append(primitive_target(op,int(a),int(b),int(c),int(p),int(n),int(f)))
   which.append(INDEX[op])
 return (torch.tensor(rows,dtype=torch.float32),
         torch.tensor(target,dtype=torch.float32),
         torch.tensor(which,dtype=torch.long))

def zeros_dataset(seed,split):
 """Balanced zero/nonzero, 16-bit held-out values separated by hash partition.
 Split separation is for this new NC-005 training, not old NC-004 pretraining.
 """
 rng=random.Random(7919+seed)
 train=[0]*192
 train += [1<<i for i in range(WIDTH)]
 train += [MASK^(1<<i) for i in range(WIDTH)]
 train += [i for i in range(1,256)]
 train += [rng.randrange(1,1<<WIDTH) for _ in range(1024)]
 hold=[0]+[1<<i for i in range(WIDTH)]
 hold += [rng.randrange(1,1<<WIDTH) for _ in range(512)]
 if split=="train": values=train
 elif split=="heldout":values=hold
 else:raise ValueError(split)
 x=torch.tensor([[(v>>i)&1 for i in range(WIDTH)] for v in values],dtype=torch.float32)
 y=torch.tensor([int(v==0) for v in values],dtype=torch.float32)
 return x,y,values

def bit_accuracy(model,x,y,which,subset=None):
 with torch.no_grad():
  model.eval()
  selected=torch.ones(len(x),dtype=torch.bool) if subset is None else torch.isin(which,torch.tensor([INDEX[o] for o in subset]))
  if not bool(selected.any()):
   return {"correct":0,"tested":0,"meaning_correct":0,"meaning_tested":0}
  pred=model.bit_from_features(x[selected])
  exact=(pred.ge(0)==y[selected].bool()).all(-1)
  ids=which[selected]
  labels=model.meaning(x[selected,:len(OPS)]).argmax(-1).eq(ids)
  return {"correct":int(exact.sum()),"tested":int(len(exact)),
          "meaning_correct":int(labels.sum()),"meaning_tested":len(ids)}

def zero_accuracy(model,x,y):
 with torch.no_grad():
  model.eval()
  logits=model.zero_logits(x)
  outputs=logits>=0
  truth=y.bool()
  failures=(outputs!=truth).nonzero().flatten().tolist()
  return {"correct":int((outputs==truth).sum()),"tested":len(y),
          "false_positive":int((outputs&(~truth)).sum()),
          "false_negative":int(((~outputs)&truth).sum()),"failed_indices":failures[:8]}

@torch.no_grad()
def bits(values):
 return torch.tensor([[(int(v)>>k)&1 for k in range(WIDTH)] for v in values],dtype=torch.float32)

@torch.no_grad()
def words(encoded):
 return [sum((1<<k) for k,z in enumerate(row) if z>=0.5) for row in encoded.tolist()]

@torch.no_grad()
def step_numeric(model,op,a,b=None):
 """Actual neural bit logits; no mathematical reference inside inference."""
 if b is None:b=torch.zeros_like(a)
 assert a.shape==b.shape and a.shape[-1]==WIDTH
 n=len(a)
 if op in ("ADD","SUB"):
  carry=torch.ones(n) if op=="SUB" else torch.zeros(n)
  zero=torch.zeros(n)
  out=torch.empty_like(a)
  for k in range(WIDTH):
   sig=torch.stack((a[:,k],b[:,k],carry,zero,zero,zero),dim=1)
   result=(model.local(op,sig)>=0).float()
   out[:,k]=result[:,0]
   carry=result[:,1]
  return out,carry
 prev=F.pad(a[:,:-1],(1,0))
 following=F.pad(a[:,1:],(0,1))
 pos=torch.arange(WIDTH).repeat(n)
 z=torch.zeros(n*WIDTH)
 f=(pos==WIDTH-1).float() if op=="SAR" else (pos==0).float()
 sig=torch.stack((a.reshape(-1),b.reshape(-1),z,prev.reshape(-1),
                  following.reshape(-1),f),dim=1)
 output=model.local(op,sig).ge(0).float()[:,0].reshape(n,WIDTH)
 return output,None

@torch.no_grad()
def take_branch(model,op,registers):
 """Only zero/nonzero decision learned. Opcode/target/PC wiring is host-given."""
 if op not in ("JZ","JNZ","JMP","HALT"):
  raise ValueError(op)
 if op=="JMP":return [True]*len(registers)
 if op=="HALT":return [False]*len(registers)
 zero=(model.zero_logits(registers)>=0).tolist()
 return zero if op=="JZ" else [not bool(b) for b in zero]

def eval_heldout(model,seed, pairs_n=96):
 rng=random.Random(8800+seed)
 cases=[(rng.randrange(256),rng.randrange(256)) for _ in range(pairs_n)]
 xs=bits([a for a,b in cases]);ys=bits([b for a,b in cases])
 results={}
 with torch.no_grad():
  for op in OPS:
   pred,carry=step_numeric(model,op,xs,ys)
   values=words(pred)
   expected=[]
   for a,b in cases:
    if op=="ADD":t=(a+b)&MASK
    elif op=="SUB":t=(a-b)&MASK
    elif op=="AND":t=a&b
    elif op=="OR":t=a|b
    elif op=="XOR":t=a^b
    elif op=="AND1":t=a&1
    elif op=="SHL":t=(a<<1)&MASK
    elif op=="SHR":t=a>>1
    else:t=(a>>1)&MASK
    expected.append(t)
   misses=[{"a":a,"b":b,"pred":got,"expected":want}
           for (a,b),got,want in zip(cases,values,expected) if got!=want]
   results[op]={"correct":len(cases)-len(misses),"tested":len(cases),
                "failures":misses[:4]}
 return results

def sequence_programs():
 return {
  "double_plus":[("SHL","X",None),("ADD","R","Y")],
  "subtract_then_xor":[("SUB","X","Y"),("XOR","R","X")],
  "or_then_shift":[("OR","X","Y"),("SHR","R",None)],
  "and_then_add":[("AND","X","Y"),("ADD","R","X")],
  "xor_then_double":[("XOR","X","Y"),("SHL","R",None)],
 }

def run_sequence(model,program,pairs):
 """Supplied program, host routing; unary+binary neural primitive only."""
 state={"X":bits([a for a,b in pairs]),"Y":bits([b for a,b in pairs])}
 state["R"]=torch.zeros_like(state["X"])
 with torch.no_grad():
  for op,lhs,rhs in program:
   value,_=step_numeric(model,op,state[lhs],state[rhs] if rhs else None)
   state["R"]=value
 return words(state["R"])

def reference_sequence(program,pairs):
 def fn(op,a,b):
  if op=="ADD":return (a+b)&MASK
  if op=="SUB":return (a-b)&MASK
  if op=="OR":return a|b
  if op=="AND":return a&b
  if op=="XOR":return a^b
  if op=="AND1":return a&1
  if op=="SHL":return (a<<1)&MASK
  if op=="SHR":return a>>1
  if op=="SAR":return (a>>1)|(a&0x8000)
  raise ValueError(op)
 res=[]
 for a,b in pairs:
  r=0
  for op,left,right in program:
   regs={"X":a,"Y":b,"R":r}
   r=fn(op,regs[left],regs[right] if right else 0)
  res.append(r)
 return res

def eval_sequences(model,seed,n=64):
 rng=random.Random(1111+seed)
 test=[(rng.randrange(256),rng.randrange(256)) for _ in range(n)]
 rows={}
 for name,program in sequence_programs().items():
  pred=run_sequence(model,program,test)
  ref=reference_sequence(program,test)
  bad=[{"a":a,"b":b,"pred":x,"reference":y}
       for (a,b),x,y in zip(test,pred,ref) if x!=y]
  rows[name]={"correct":len(test)-len(bad),"tested":len(test),"failures":bad[:4]}
 return rows

def run_fixed_mul(model,pairs,max_cycles=18):
 """Fixed human-supplied shift/add ROM and host PC; neural JZ and primitives."""
 a=bits([x for x,y in pairs]);b=bits([y for x,y in pairs])
 r=torch.zeros_like(a)
 halt=torch.zeros(len(pairs),dtype=torch.bool)
 cycles=torch.zeros(len(pairs),dtype=torch.long)
 with torch.no_grad():
  for i in range(max_cycles):
   active=~halt
   if not active.any(): break
   zero=model.zero_logits(b)>=0
   halt=halt|(active&zero)
   work=active&(~zero)
   if not work.any():continue
   odd,_=step_numeric(model,"AND1",b[work])
   updated,_=step_numeric(model,"ADD",r[work],a[work])
   select=odd[:,0]>=.5
   r[work]=torch.where(select[:,None],updated,r[work])
   shift_a,_=step_numeric(model,"SHL",a[work])
   shift_b,_=step_numeric(model,"SHR",b[work])
   a[work]=shift_a;b[work]=shift_b
   cycles[work]+=1
 result=words(r)
 valid=halt.tolist()
 score=[int(t==x*y and h) for t,(x,y),h in zip(result,pairs,valid)]
 fails=[{"a":x,"b":y,"predicted":result[i],"expected":x*y,"halted":valid[i]}
        for i,(x,y) in enumerate(pairs) if not score[i]]
 return {"correct":sum(score),"tested":len(pairs),"halted":sum(valid),
         "unhalted":len(pairs)-sum(valid),"failures":fails[:8],
         "host_rom":True,"neural_zero_branch":True,"neural_arithmetic":True,
         "model_generated_program":False}

def eval_loops(model,seed):
 rng=random.Random(9007+seed)
 cases4=[(a,b) for a in range(16) for b in range(16)]
 cases8=[(rng.randrange(256),rng.randrange(256)) for _ in range(128)]
 return {"fourbit_exhaustive":run_fixed_mul(model,cases4),
         "eightbit_sample":run_fixed_mul(model,cases8)}

def branch_heldout(model,seed):
 zx,zy,values=zeros_dataset(seed,"heldout")
 zero=zero_accuracy(model,zx,zy)
 with torch.no_grad():
  for op in ("JZ","JNZ"):
   predicted=take_branch(model,op,zx)
   truth=[int(v==0) if op=="JZ" else int(v!=0) for v in values]
   zero[op+"_correct"]=sum(int(p)==z for p,z in zip(predicted,truth))
 return zero

def checkpoint(model,opt,arm,seed,stage_no,total_updates,loss):
 CHECKPOINTS.mkdir(parents=True,exist_ok=True)
 dest=CHECKPOINTS/f"{arm}_seed{seed}_stage{stage_no:02d}.pt"
 payload={"schema":1,"task":"NC-005","seed":seed,"arm":arm,
          "stage_index":stage_no,"stage_name":STAGES[stage_no][0],
          "total_updates":total_updates,"model":model.state_dict(),
          "optimizer":opt.state_dict(),"rng_torch":torch.get_rng_state(),
          "rng_python":random.getstate(),
          "numpy_rng":np.random.get_state()[1].tolist(),
          "ops":OPS,"width":WIDTH,"stages":STAGES,
          "loss":loss,
          "nc004_source_checkpoint_sha256":sha(base.HERE/f"coupled_seed{seed}.pt"),
          "original_champion_sha256":sha(base.CHAMPION),
          "programs_fixed_and_host_wired":True,
          "optimizer_initialized_new_not_historical_exact_resume":True}
 torch.save(payload,dest)
 return {"path":str(dest.relative_to(ROOT)),"sha256":sha(dest),"bytes":dest.stat().st_size,
         "stage":STAGES[stage_no][0],"total_updates":total_updates}

def stage_input(stage_no,arm,x,y,ids):
 name,current,_=STAGES[stage_no]
 if name=="PC_BRANCH":
  return x[:0],y[:0],ids[:0]
 if name=="SEQUENCES":
  # Current-task distributions of supplied operations, not generated programs.
  current=tuple(OPS)
 if name=="LOOPS":
  current=("AND1","ADD","SHL","SHR")
 if arm=="rehearsal":
  # preserve the five qualified warm-start skills and all introduced operations
  allowed=set(OLD)
  for i in range(stage_no+1):
   allowed.update(STAGES[i][1])
  if name in ("SEQUENCES","LOOPS"):allowed.update(current)
 else:
  allowed=set(current)
 mask=torch.isin(ids,torch.tensor([INDEX[o] for o in allowed],dtype=torch.long))
 return x[mask],y[mask],ids[mask]

def curriculum_step(model,opt,arm,stage_no,tx,ty,tids,zx,zy,step):
 name,current,_=STAGES[stage_no]
 use_zero=(name in ("PC_BRANCH","LOOPS"))
 a,b,ix=stage_input(stage_no,arm,tx,ty,tids)
 if name=="PC_BRANCH" and arm=="rehearsal":
  a,b,ix=stage_input(stage_no,"rehearsal",tx,ty,tids) # no bit data in this special stage
 if name=="PC_BRANCH":
  a,b,ix=(tx[:0],ty[:0],tids[:0]) if arm=="new_only" else (tx,ty,tids)
 # same BATCH and optimizer steps, mixed tasks only differ in replay policy
 opt.zero_grad(set_to_none=True)
 loss=None
 if len(a):
  # deterministic seed-local minibatch sequence; separate arm changes set only
  index=torch.randint(len(a),(BATCH,))
  logits=model.bit_from_features(a[index])
  bit=F.binary_cross_entropy_with_logits(logits,b[index])
  meaning=F.cross_entropy(model.meaning(a[index,:len(OPS)]),ix[index])
  loss=bit+.2*meaning
 if use_zero:
  idx=torch.randint(len(zx),(BATCH,))
  zl=F.binary_cross_entropy_with_logits(model.zero_logits(zx[idx]),zy[idx])
  loss=zl if loss is None else loss+zl
 if loss is None:raise RuntimeError("stage has no neural gradients")
 loss.backward()
 opt.step()
 return float(loss.item())

def train_seed(seed,arm,updates_scale=1):
 torch.set_num_threads(1)
 torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
 m=warm_start(seed);m.train()
 opt=torch.optim.Adam(m.parameters(),lr=.008)
 tx,ty,tids=local_table()
 zx,zy,_=zeros_dataset(seed,"train")
 report={"seed":seed,"arm":arm,"architecture":"warm_started_NC004_NeuralISA_expanded_9op_MLP_plus_neural_is_zero",
  "source_seed":seed,"parameter_count":sum(z.numel() for z in m.parameters()),
  "source_checkpoint_sha256":sha(base.HERE/f"coupled_seed{seed}.pt"),
  "initial_old_five":bit_accuracy(m,tx,ty,tids,OLD),
  "stages":[]}
 updates=0
 for stage_no,(name,current,count) in enumerate(STAGES):
  loss=0.
  for step in range(count*updates_scale):
   loss=curriculum_step(m,opt,arm,stage_no,tx,ty,tids,zx,zy,step)
  updates+=count*updates_scale
  m.eval()
  allrows={op:bit_accuracy(m,tx,ty,tids,(op,)) for op in OPS}
  old=bit_accuracy(m,tx,ty,tids,OLD)
  prior=list(OLD)
  for i in range(stage_no+1):prior.extend(STAGES[i][1])
  prior=list(dict.fromkeys(prior))
  prior_scores={op:allrows[op] for op in prior}
  branch=branch_heldout(m,seed)
  # Staged evaluation: do not test fixed-loop multiplication before loop stage.
  sequences=eval_sequences(m,seed) if stage_no>=5 else None
  loops=eval_loops(m,seed) if stage_no>=6 else None
  full=eval_heldout(m,seed) if stage_no in (3,4,5,6) else None
  info=checkpoint(m,opt,arm,seed,stage_no,updates,loss)
  row={"stage":name,"stage_index":stage_no,"current_ops":list(current),
       "updates_this_stage":count*updates_scale,"updates_cumulative":updates,
       "last_loss":loss,"qualified_prior_ops":prior,
       "local_by_op":allrows,"prior_retention":prior_scores,
       "old_five_retention":old,"heldout_branch":branch,
       "heldout_8bit_ops":full,"heldout_sequences":sequences,
       "heldout_loops":loops,"checkpoint":info}
  report["stages"].append(row)
  print(json.dumps({"seed":seed,"arm":arm,"stage":name,"old_retention":f"{old['correct']}/{old['tested']}",
   "all_prior":f"{sum(v['correct'] for v in prior_scores.values())}/{sum(v['tested'] for v in prior_scores.values())}",
   "branch":f"{branch['correct']}/{branch['tested']}",
   "loop4":loops["fourbit_exhaustive"]["correct"] if loops else None,
   "loss":round(loss,6)},sort_keys=True),flush=True)
  m.train()
 report["last_checkpoint"]=report["stages"][-1]["checkpoint"]
 return report

def run_training():
 src_inv=json.loads(CHAMPION.read_text())
 assert sha(base.CHAMPION)==src_inv["champion"]["selected_bundle_sha256"]
 assert sha(PREFIX_SOURCE)==sha(ROOT/"artifacts/NC-004/causal_semantics.py")
 torch.set_num_threads(1)
 summary={"task":"NC-005","status":"IMPLEMENTER_TRAINING_COMPLETE",
  "role":"new experiment; no champion promotion",
  "seeds":list(SEEDS),"arms":list(ARMS),"stages":[list(x[:2])+[x[2]] for x in STAGES],
  "stage_updates":STAGE_UPDATES,"batch":BATCH,
  "matched_controls":"same seed, NC004 warm-start weights, stage count, optimizer type/lr, batch, step budget, shared stage datasets; rehearsal changes exposure to prior stages deliberately",
  "not_data_matched":"The two arms necessarily see different training-example frequencies due to rehearsal; report exposure as intervention, do not claim identical example streams",
  "training_data":"all 64 bit patterns per trained primitive in source pool; Boolean local generalization NOT evaluated",
  "prior_model":"NC004 coupled semantic MLP; expanded opcode vocabulary and newly initialized optimizer branch",
  "model_semantics":"predicted meaning softmax enters neural arithmetic; zero predicate is learned, all program choice/register routing/PC/HALT host",
  "source_shas":{str(p):sha(p) for p in [PREFIX_SOURCE,ROOT/"artifacts/NC-001/canonical_isa.py",CHAMPION,base.CHAMPION]},
  "original_champion_sha256":sha(base.CHAMPION),
  "results":{},
  "run_environment":{"torch":str(torch.__version__),"cpu":True,"threads":1},
  "limitations":["Existing five primitive skills pre-trained on full local truth tables; transfer is not learning from zero",
    "No neural source general RLT champion training or promotion",
    "New expanded optimizer cannot recover NC004 historical optimizer moments",
    "JZ/JNZ learned zero test only; opcode, target selection, HALT, PC arithmetic host-provided",
    "Sequences/loops supplied by host and evaluated on unseen operand values, not invented programs",
    "Matched update budget not identical training samples after rehearsal alters sampling",
    "Some stages may exhibit forgetting; any regression must be recorded and cannot be silently passed"]}
 out=HERE/"curriculum_receipts.jsonl"
 with out.open("w") as stream:
  for seed in SEEDS:
   for arm in ARMS:
    r=train_seed(seed,arm)
    summary["results"][f"{arm}_seed{seed}"]=r
    for row in r["stages"]:
     stream.write(json.dumps({"task":"NC-005","seed":seed,"arm":arm,
       "stage":row["stage"],"checkpoint":row["checkpoint"],
       "local_by_op":row["local_by_op"],"old_five_retention":row["old_five_retention"],
       "heldout_branch":row["heldout_branch"],
       "heldout_8bit_ops":row["heldout_8bit_ops"],
       "heldout_sequences":row["heldout_sequences"],
       "heldout_loops":row["heldout_loops"],
       "updates":row["updates_cumulative"],
       "previous_qualified_ops":row["qualified_prior_ops"],
       "prior_retention":row["prior_retention"],
       "implementation":"trained neural primitive bit and zero logits; host-defined programs/control"},sort_keys=True)+"\n")
    stream.flush()
 (HERE/"training_report.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
 print("TRAINING_REPORT_SHA256",sha(HERE/"training_report.json"),flush=True)
if __name__=="__main__":
 parser=argparse.ArgumentParser()
 parser.add_argument("mode",choices=("train",))
 opt=parser.parse_args()
 if opt.mode=="train":run_training()
