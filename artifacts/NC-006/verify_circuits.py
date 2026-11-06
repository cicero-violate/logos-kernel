#!/usr/bin/env python3
"""NC-006 frozen neural bit-circuit/operator library, scoped finite verification.

Dependencies: Python 3 stdlib, PyTorch, NC-005/curriculum.py (ExpandedISA only),
NC-005/checkpoint_manifest.json and saved stage06 .pt, NC-000 inventory.json.
Original RLT/champion and standalone ISA checkpoints read for hashes ONLY.
No training, controller learning, symbolic fallback, code synthesis or promotion.
"""
import hashlib
import itertools
import json
import platform
import sys
from pathlib import Path
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"artifacts/NC-005"))
import curriculum as c
VERSION="logos.neural-circuit/1"
BINARY=("ADD","SUB","AND","OR","XOR")
UNARY=("AND1","SHL","SHR","SAR")
OPS=tuple(c.OPS)
assert set(BINARY+UNARY)==set(OPS)
SEEDS=(11,19,29)
WIDTH=16

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()

def expected(op,a,b,carry,prev,nxt,first):
 """Independently authored exact 1-bit feature truth targets, NOT trained outputs."""
 if op=="ADD":
  total=a+b+carry
  return int(total&1),int(total>=2)
 if op=="SUB":
  total=a+(1-b)+carry
  return int(total&1),int(total>=2)
 if op=="AND":return a&b,0
 if op=="OR":return a|b,0
 if op=="XOR":return a^b,0
 if op=="AND1":return a&first,0
 if op=="SHL":return prev,0
 if op=="SHR":return nxt,0
 if op=="SAR":return a if first else nxt,0
 raise ValueError("unknown neural opcode")

def inputs(op):
 """All 2^6 binary local feature tuples, no sampled gap."""
 patterns=list(itertools.product((0,1),repeat=6))
 features=torch.tensor(patterns,dtype=torch.float32)
 oracle=torch.tensor([expected(op,*pattern) for pattern in patterns],dtype=torch.int64)
 return patterns,features,oracle

def frozen_model(seed,arm,manifest):
 assert arm in ("rehearsal","new_only")
 path=c.CHECKPOINTS/f"{arm}_seed{seed}_stage06.pt"
 listed=[x for x in manifest["checkpoints"] if x["path"]==str(path.relative_to(ROOT))]
 assert len(listed)==1
 assert sha(path)==listed[0]["sha256"] and path.stat().st_size==listed[0]["size"]
 payload=torch.load(path,map_location="cpu",weights_only=True)
 assert payload["task"]=="NC-005" and payload["stage_index"]==6
 assert payload["seed"]==seed and payload["arm"]==arm
 assert payload["total_updates"]==c.STAGE_UPDATES
 assert payload["original_champion_sha256"]==sha(c.base.CHAMPION)
 assert payload["nc004_source_checkpoint_sha256"]==sha(c.base.HERE/f"coupled_seed{seed}.pt")
 model=c.ExpandedISA()
 model.load_state_dict(payload["model"],strict=True)
 model.eval()
 for p in model.parameters():p.requires_grad_(False)
 return model,path,payload

def local_neural(model,op):
 patterns,signals,expected_bits=inputs(op)
 with torch.no_grad():
  actual_logits=model.local(op,signals)
  predicted=(actual_logits>=0).int()
  correct=(predicted==expected_bits).all(dim=1)
  predicted_op=int(model.meaning(F.one_hot(torch.tensor([c.INDEX[op]]),len(OPS)).float()).argmax(dim=1)[0])
 failures=[{"inputs":list(patterns[i]),"observed":predicted[i].tolist(),
            "expected":expected_bits[i].tolist()}
           for i in range(len(patterns)) if not bool(correct[i])]
 return {"correct":int(correct.sum()),"tested":64,
   "output_bit_accuracy":int((predicted==expected_bits).sum()),"output_bit_tested":128,
   "meaning_label_correct":predicted_op==c.INDEX[op],
   "meaning_predicted":OPS[predicted_op],
   "failure_count":len(failures),"failure_examples":failures,
   "float_logit_finiteness":bool(torch.isfinite(actual_logits).all()),
   "all_64_local_features_supervised_in_source_training":True}

def exhaustive_zero(model):
 count=0;wrong=[];positive=0;negative=0
 with torch.no_grad():
  for start in range(0,1<<WIDTH,4096):
   vals=list(range(start,min(start+4096,1<<WIDTH)))
   # Encode feature bitpattern only, no zero-answer attached to inference.
   features=torch.tensor([[(v>>b)&1 for b in range(WIDTH)] for v in vals],dtype=torch.float32)
   logits=model.zero_logits(features)
   decision=(logits>=0).tolist()
   assert bool(torch.isfinite(logits).all())
   for val,pred in zip(vals,decision):
    truth=val==0
    count+=int(pred==truth)
    positive+=int(truth and pred)
    negative+=int((not truth) and (not pred))
    if pred!=truth:wrong.append({"value":val,"predicted_zero":pred,"expected_zero":truth})
 return {"correct":count,"tested":65536,
   "true_positive":positive,"positive_tested":1,
   "true_negative":negative,"negative_tested":65535,
   "false_positive":sum(not v["expected_zero"] for v in wrong),
   "false_negative":sum(v["expected_zero"] for v in wrong),
   "failed_cases":wrong,"failure_count":len(wrong),
   "scope":"exhaustive all raw 16-bit inputs to frozen learned ZERO head",
   "not_learned":"JZ/JNZ opcode selection, PC, target and HALT host-provided"}

def entry(model_sha,seed,arm,op,proof):
 qualifies=proof["correct"]==proof["tested"] and proof.get("failure_count",0)==0
 if op!="ZERO16":
  qualifies &= proof["meaning_label_correct"] and proof["float_logit_finiteness"]
 return {
  "contract_version":VERSION,
  "id":f"{arm}:seed{seed}:{op}:v1",
  "operation":op,"seed":seed,"arm":arm,"checkpoint_sha256":model_sha,
  "implementation_mode":"neural_inference",
  "input_contract":{"features":["a","b","carry","previous","following","first"],
      "widths":{k:1 for k in ("a","b","carry","previous","following","first")},
      "opcode_is_fixed_input":True} if op!="ZERO16" else
      {"features":["16 raw register bits, bit0 first"],"width":16},
  "output_contract":{"bit":1,"carry":1} if op!="ZERO16" else {"zero_bit":1},
  "model_architecture":"NC005 ExpandedISA causal semantic MLP, 5981 params",
  "proof_scope":"EXHAUSTIVE_FINITE_LOCAL_6BIT_FEATURE_SET" if op!="ZERO16"
                else "EXHAUSTIVE_FINITE_16BIT_REGISTER_ZERO_SET",
  "strength":"OBSERVED_FROZEN_NEURAL_FINITE_EXHAUSTIVE_RUNTIME_NOT_FORMAL_WEIGHT_PROOF",
  "all_local_bit_truth_patterns_were_training_seen":op!="ZERO16",
  "qualified_bounded":bool(qualifies),
  "library_status":"CANDIDATE_AWAITING_DISTINCT_INDEPENDENT_VERIFICATION" if qualifies else "REJECTED_COUNTEREXAMPLE",
  "no_default_champion_promotion":True,
  "host_boundary":"neural primitive only; bit indexing, carry propagation, register updates, PC, HALT and program choice are NOT learned",
 }

def run():
 torch.set_num_threads(1)
 manifest=json.loads((ROOT/"artifacts/NC-005/checkpoint_manifest.json").read_text())
 original=json.loads((ROOT/"artifacts/NC-000/inventory.json").read_text())
 champ=Path(original["roots"]["rlt_champion"])/"snapshots/current_unified_v3.pt"
 assert sha(champ)==original["champion"]["selected_bundle_sha256"]
 assert manifest["checkpoint_count"]==42
 source_paths={
  "NC005_curriculum":ROOT/"artifacts/NC-005/curriculum.py",
  "NC005_checkpoint_manifest":ROOT/"artifacts/NC-005/checkpoint_manifest.json",
  "NC005_independent_receipt":ROOT/"artifacts/NC-005/independent_receipt.json",
  "NC001_canonical_isa":ROOT/"artifacts/NC-001/canonical_isa.py",
  "NC000_inventory":ROOT/"artifacts/NC-000/inventory.json",
  "original_RLT_champion":champ}
 records=[];receipts=[];stats={}
 for arm in ("rehearsal","new_only"):
  for seed in SEEDS:
   model,path,payload=frozen_model(seed,arm,manifest)
   digest=sha(path)
   results={}
   for op in OPS:
    result=local_neural(model,op)
    records.append(entry(digest,seed,arm,op,result))
    results[op]=result
    receipts.append({"contract_version":VERSION,"circuit_id":records[-1]["id"],
      "checkpoint_sha256":digest,"operation":op,
      "mode":"frozen_neural_logits_only","scope":records[-1]["proof_scope"],
      "proof_status":"FINITE_EXHAUSTIVE_OBSERVED" if records[-1]["qualified_bounded"]
                     else "FAIL_WITH_EXPLICIT_COUNTEREXAMPLE",
      "measurement":result,
      "no_fallback_or_synthetic_result":True,"reference_implemented_independently":True})
   zero=exhaustive_zero(model)
   records.append(entry(digest,seed,arm,"ZERO16",zero))
   receipts.append({"contract_version":VERSION,"circuit_id":records[-1]["id"],
     "checkpoint_sha256":digest,"operation":"ZERO16",
     "mode":"frozen_neural_logits_only","scope":records[-1]["proof_scope"],
     "proof_status":"FINITE_EXHAUSTIVE_OBSERVED" if records[-1]["qualified_bounded"]
                    else "FAIL_WITH_EXPLICIT_COUNTEREXAMPLE",
     "measurement":zero,"no_fallback_or_synthetic_result":True,
     "reference_implemented_independently":True})
   stats[f"{arm}_seed{seed}"]={
     "checkpoint_sha256":digest,"local_correct":sum(x["correct"] for x in results.values()),
     "local_tested":len(OPS)*64,
     "local_meaning_correct":sum(x["meaning_label_correct"] for x in results.values()),
     "local_meaning_tested":len(OPS),
     "zero_correct":zero["correct"],"zero_tested":zero["tested"],
     "qualified_local_ops":[op for op,result in results.items() if result["correct"]==64 and result["meaning_label_correct"]],
     "failed_local_ops":[op for op,result in results.items() if result["correct"]<64 or not result["meaning_label_correct"]],
     "all_model_parameters_frozen":not any(p.requires_grad for p in model.parameters())}
   print(json.dumps({"model":f"{arm}_seed{seed}",**{k:stats[f'{arm}_seed{seed}'][k] for k in
      ("local_correct","local_tested","zero_correct","failed_local_ops")}},sort_keys=True),flush=True)
 # Register symbolic NC001 as an intentionally DIFFERENT execution mode, not neural.
 references=[
  {"contract_version":VERSION,"id":"symbolic:NC001:canonical_isa:v1",
   "implementation_mode":"symbolic_reference","operation":"NC001_bitvector_executor",
   "proof_scope":"NC001_reference_semantics_not_neural_weights",
   "qualified_bounded":False,"library_status":"SEPARATE_SYMBOLIC_REFERENCE_NOT_NEURAL_PROMOTABLE",
   "checkpoint_sha256":None,"no_default_champion_promotion":True}
 ]
 assert len(records)==2*3*(len(OPS)+1)
 assert all(stats[f"rehearsal_seed{seed}"]["local_correct"]==576 and
        stats[f"rehearsal_seed{seed}"]["zero_correct"]==65536 for seed in SEEDS)
 assert any(stats[f"new_only_seed{seed}"]["failed_local_ops"] for seed in SEEDS)
 for name,data in (("circuits.jsonl",[json.dumps(z,sort_keys=True) for z in records+references]),
                   ("proof_receipts.jsonl",[json.dumps(z,sort_keys=True) for z in receipts])):
  (HERE/name).write_text("\n".join(data)+"\n")
 summary={"task":"NC-006","status":"IMPLEMENTER_PROPOSED_NOT_INDEPENDENTLY_VERIFIED",
  "version":VERSION,"source_sha256":{k:sha(p) for k,p in source_paths.items()},
  "source_files":{k:str(p) for k,p in source_paths.items()},
  "seeds":list(SEEDS),"arms":["rehearsal","new_only"],
  "checkpoint_manifest_sha256":sha(source_paths["NC005_checkpoint_manifest"]),
  "stats":stats,"neural_circuits_count":len(records),
  "neural_local_qualified_rehearsal":sum(1 for r in records if r["arm"]=="rehearsal" and r["operation"]!="ZERO16" and r["qualified_bounded"]),
  "zero_qualified_rehearsal":sum(1 for r in records if r["arm"]=="rehearsal" and r["operation"]=="ZERO16" and r["qualified_bounded"]),
  "symbolic_references_count":len(references),
  "checkpoint_weights_modified":False,"original_champion_sha256":sha(champ),
  "independent_verifier_status":"PENDING",
  "scope_boundaries":["64 feature configurations for each local neural primitive, all locally training-seen",
    "16bit learned is_zero predicate over 65536 exact register inputs",
    "JZ/JNZ target, PC changes, flags, HALT and register routing are host, NOT learned",
    "full 16bit arithmetic, composite programs, different widths, model invention NOT proven by finite local exhaustive checks",
    "sampled correctness after local coverage cannot establish all reachable program states",
    "no silent symbolic fallback, no architecture promotion, no general intelligence claim"],
  "rollback":"No weights written; original champion selection unchanged; registry remains unpromoted without distinct verifier"}
 (HERE/"library_manifest.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"result":summary["status"],
   "registered_neural_circuits":len(records),
   "qualified_rehearsal_local":summary["neural_local_qualified_rehearsal"],
   "qualified_rehearsal_zero":summary["zero_qualified_rehearsal"],
   "no_default_promotion":True,
   "checkpoint_source_hash_count":len(source_paths)},indent=2),flush=True)
if __name__=="__main__":run()
