"""Generate supervised neural primitive examples from supplied program *traces*.

Dependencies: Python 3 stdlib, torch, NC-001 canonical_isa.py through the
caller-provided curriculum module. Teacher/oracle used only for TRAINING data.
No teacher next PC / arithmetic result is fed into inference.
"""
import random
import torch

def fn(op,a,b,mask=65535):
 if op=="ADD":return (a+b)&mask
 if op=="SUB":return (a-b)&mask
 if op=="AND":return a&b
 if op=="OR":return a|b
 if op=="XOR":return a^b
 if op=="AND1":return a&1
 if op=="SHL":return (a<<1)&mask
 if op=="SHR":return a>>1
 if op=="SAR":return ((a>>1)|(a&32768))&mask
 raise ValueError(op)

def emit(c,op,a,b):
 rows=[]
 carry=1 if op=="SUB" else 0
 for bit in range(16):
  ai=(a>>bit)&1;bi=(b>>bit)&1
  prev=(a>>(bit-1))&1 if bit else 0
  nxt=(a>>(bit+1))&1 if bit<15 else 0
  flag=int(bit==15) if op=="SAR" else int(bit==0)
  y=c.primitive_target(op,ai,bi,carry,prev,nxt,flag)
  rows.append((c.feature(op,float(ai),float(bi),float(carry),float(prev),
                         float(nxt),float(flag)),y,c.INDEX[op]))
  if op in ("ADD","SUB"):carry=y[1]
 expected=fn(op,a,b)
 bits=sum(rows[bit][1][0]<<bit for bit in range(16))
 assert bits==expected,(op,a,b,bits,expected)
 return rows

def create(seed,c):
 """Disjoint train/eval generators within this NC005 run, not older pretraining."""
 seq=[
  ("double_plus",c.sequence_programs()["double_plus"]),
  ("or_then_shift",c.sequence_programs()["or_then_shift"]),
  ("and_then_add",c.sequence_programs()["and_then_add"]),
 ]
 rng=random.Random(81009+seed)
 sequence_pairs=[(rng.randrange(256),rng.randrange(256)) for _ in range(48)]
 rows=[]
 for _,instructions in seq:
  for a,b in sequence_pairs:
   regs={"X":a,"Y":b,"R":0}
   for op,lhs,rhs in instructions:
    left=regs[lhs];right=regs[rhs] if rhs else 0
    rows.extend(emit(c,op,left,right))
    regs["R"]=fn(op,left,right)
   assert regs["R"]==c.reference_sequence(instructions,[(a,b)])[0]
 loop_rng=random.Random(92008+seed)
 loop_pairs=[(loop_rng.randrange(32),loop_rng.randrange(32)) for _ in range(96)]
 loop_rows=[];branch_values=[]
 for a,b in loop_pairs:
  x,y,r=a,b,0
  for cycle in range(9):
   branch_values.append(y)
   if y==0:break
   loop_rows.extend(emit(c,"AND1",y,0))
   if y&1:
    loop_rows.extend(emit(c,"ADD",r,x))
    r=(r+x)&65535
   loop_rows.extend(emit(c,"SHL",x,0))
   loop_rows.extend(emit(c,"SHR",y,0))
   x=(x<<1)&65535;y=y>>1
  assert y==0 and r==a*b,(a,b,r)
 def tensors(rows):
  return (torch.tensor([r[0] for r in rows],dtype=torch.float32),
          torch.tensor([r[1] for r in rows],dtype=torch.float32),
          torch.tensor([r[2] for r in rows],dtype=torch.long))
 return {"SEQUENCES":tensors(rows),"LOOPS":tensors(loop_rows),
         "sources":{"sequence_programs":[x[0] for x in seq],
            "sequence_training_operand_pairs":len(sequence_pairs),
            "sequence_local_transition_examples":len(rows),
            "loop_ROM":"fixed human-provided shift-and-add loop",
            "loop_training_operand_pairs":len(loop_pairs),
            "loop_local_transition_examples":len(loop_rows),
            "loop_branch_state_examples":len(branch_values),
            "no_new_operation_invention":True,
            "original_NC004_full_local_truth_table_seen":True,
            "test_sequences":list(c.sequence_programs()),
            "unseen_sequence_structure_test":["subtract_then_xor","xor_then_double"]}}
