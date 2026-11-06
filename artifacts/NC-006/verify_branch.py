#!/usr/bin/env python3
"""NC006 exhaustive JZ/JNZ HOST integration with frozen neural ZERO predicate.

Dependencies: Python3 stdlib, PyTorch, NC006/verify_circuits.py,
NC005/curriculum.py, NC005 frozen stage06 rehearsal checkpoint packages,
NC005 checkpoint_manifest.json and NC000 champion manifest.
No training or learned PC/branch target; only ZERO predicate learned.
"""
import json
import platform
from pathlib import Path
import torch
import verify_circuits as v
ROOT=v.ROOT;HERE=v.HERE
WIDTH=v.WIDTH
def verify():
 torch.set_num_threads(1)
 manifest=json.loads((ROOT/"artifacts/NC-005/checkpoint_manifest.json").read_text())
 inv=json.loads((ROOT/"artifacts/NC-000/inventory.json").read_text())
 champ=Path(inv["roots"]["rlt_champion"])/"snapshots/current_unified_v3.pt"
 assert v.sha(champ)==inv["champion"]["selected_bundle_sha256"]
 result={"task":"NC-006","scope":"exhaustive JZ and JNZ decisions for EVERY raw 16bit register input",
  "execution_mode":"neural_is_zero_predicate_plus_host_predicate_to_PC_update",
  "PC_target_selection":"HOST fixed target, not neural","HALT":"HOST not learned",
  "branch_type":"JZ zero -> target, JNZ nonzero -> target; else PC+1",
  "width":16,"seeds":list(v.SEEDS),"arms":["rehearsal"],
  "source_paths":{"NC006_verifier":str(HERE/"verify_circuits.py"),
       "NC005_architecture":str(ROOT/"artifacts/NC-005/curriculum.py")},
  "source_sha256":{"NC006_verifier":v.sha(HERE/"verify_circuits.py"),
       "NC005_architecture":v.sha(ROOT/"artifacts/NC-005/curriculum.py")},
  "champion_sha256":v.sha(champ),
  "result":"IMPLEMENTER_BOUNDARY_VERIFIED_NOT_INDEPENDENT",
  "models":{},"model_invention":False,"learned_PC":False,
  "limitations":["Register zero predicate is trained neural inference, target/PC/HALT host logic",
     "All 65536 register values enumerated, but arbitrary program state/branch instruction choice not learned or tested",
     "This check is correlated with ZERO16 and must not be counted as separate learned capability",
     "Independent verifier must rerun before bounded library status is promoted"]}
 for seed in v.SEEDS:
  model,path,_=v.frozen_model(seed,"rehearsal",manifest)
  correct_jz=0;correct_jnz=0;flip_jz=0;flip_jnz=0
  failed_jz=[];failed_jnz=[]
  with torch.no_grad():
   for start in range(0,65536,4096):
    values=list(range(start,start+4096))
    encoded=torch.tensor([[(value>>bit)&1 for bit in range(WIDTH)] for value in values],dtype=torch.float32)
    neural_zero=(model.zero_logits(encoded)>=0).tolist()
    for value,learned in zip(values,neural_zero):
     oracle_zero=(value==0)
     # These outputs are host-applied branch directions. No neural PC or target is predicted.
     branch_jz=bool(learned)
     branch_jnz=not branch_jz
     good_jz=branch_jz==oracle_zero
     good_jnz=branch_jnz==(not oracle_zero)
     correct_jz+=int(good_jz);correct_jnz+=int(good_jnz)
     flip_jz+=int((not branch_jz)==oracle_zero)
     flip_jnz+=int((not branch_jnz)==(not oracle_zero))
     if not good_jz and len(failed_jz)<16:failed_jz.append({"value":value,"predicted_take":branch_jz})
     if not good_jnz and len(failed_jnz)<16:failed_jnz.append({"value":value,"predicted_take":branch_jnz})
  result["models"][f"seed{seed}"]={
    "checkpoint_sha256":v.sha(path),
    "JZ":{"correct":correct_jz,"tested":65536,"failures":failed_jz,
          "head":"neural_is_zero","branch_target":"host"},
    "JNZ":{"correct":correct_jnz,"tested":65536,"failures":failed_jnz,
          "head":"neural_is_zero_plus_host_inversion","branch_target":"host"},
    "inverted_neural_zero_control":{"JZ_correct":flip_jz,"JNZ_correct":flip_jnz,
       "tested_each":65536,"scope":"direct perturbation to neural ZERO decision, not a trained ablation"},
    "frozen_source":True
  }
  print(json.dumps({"seed":seed,"JZ":f"{correct_jz}/65536",
    "JNZ":f"{correct_jnz}/65536","inverted_zero_control_JZ":flip_jz},sort_keys=True),flush=True)
 out=HERE/"branch_conformance.json"
 out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 print("BRANCH_CONFORMANCE_SHA256",v.sha(out),flush=True)
 assert all(z["JZ"]["correct"]==65536 and z["JNZ"]["correct"]==65536
      and z["inverted_neural_zero_control"]["JZ_correct"]==0
      for z in result["models"].values())
if __name__=="__main__":verify()
