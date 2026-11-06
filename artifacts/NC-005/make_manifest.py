#!/usr/bin/env python3
"""Audit checkpoint state, provenance and immutable original inputs.

Dependencies: Python 3 stdlib, torch, NC005 training_report.json,
retention_matrix.json, fresh_reload.json, NC004 original .pt/checkpoint.json,
NC000 inventory and original frozen RLT checkpoint. Writes checkpoint_manifest.json.
"""
import hashlib
import json
import subprocess
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()
def git(root,*args):
 return subprocess.check_output(["git","-C",str(root),*args],text=True).strip()
def file_record(p):
 return {"path":str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
         "size":p.stat().st_size,"sha256":sha(p)}
def run():
 d=json.loads((HERE/"training_report.json").read_text())
 matrix=json.loads((HERE/"retention_matrix.json").read_text())
 fresh=json.loads((HERE/"fresh_reload.json").read_text())
 assert matrix["status"]=="IMPLEMENTER_STAGE_GATES_PASS"
 assert len(d["results"])==len(fresh["models"])==6
 assert len(d["results"])==6
 inv=json.loads((ROOT/"artifacts/NC-000/inventory.json").read_text())
 rlt=Path(inv["roots"]["rlt_champion"])
 isa=Path(inv["roots"]["standalone_isa"])
 champ=rlt/"snapshots/current_unified_v3.pt"
 assert sha(champ)==inv["champion"]["selected_bundle_sha256"]
 srcmanifest=json.loads((ROOT/"artifacts/NC-004/checkpoint.json").read_text())
 source={}
 for seed in (11,19,29):
  file=ROOT/f"artifacts/NC-004/coupled_seed{seed}.pt"
  line=next(x for x in srcmanifest["models"] if x["arm"]=="coupled" and x["seed"]==seed)
  assert sha(file)==line["checkpoint_sha256"]
  source[str(seed)]=file_record(file)
 files=[]
 optimizer_verified=0;total_params=0
 for key,r in sorted(d["results"].items()):
  assert len(r["stages"])==7
  total_params=r["parameter_count"]
  for row in r["stages"]:
   p=ROOT/row["checkpoint"]["path"]
   assert sha(p)==row["checkpoint"]["sha256"]
   payload=torch.load(p,weights_only=True,map_location="cpu")
   assert payload["task"]=="NC-005" and payload["seed"]==r["seed"]
   assert payload["arm"]==r["arm"] and payload["total_updates"]==row["updates_cumulative"]
   assert "model" in payload and "optimizer" in payload
   assert "rng_torch" in payload and "rng_python" in payload
   assert "numpy_rng" in payload
   assert isinstance(payload["optimizer"]["state"],dict)
   assert len(payload["optimizer"]["state"])>0
   assert payload["original_champion_sha256"]==sha(champ)
   optimizer_verified+=1
   modeltensors=sum(t.numel() for t in payload["model"].values()
                    if hasattr(t,"numel"))
   assert modeltensors>=total_params
   files.append(file_record(p))
 assert len(files)==optimizer_verified==42
 pilots=[file_record(p) for p in sorted((HERE/"pilot/checkpoints").glob("*.pt"))]
 assert len(pilots)==42
 source_files=[HERE/name for name in ("curriculum.py","trace_training.py","analyze_retention.py",
 "verify_fresh.py","test_curriculum.py","README.md")]
 for p in source_files:assert p.is_file()
 original_hashes=[]
 for record in inv["files"]:
  path=Path(inv["roots"][record["repo"]])/record["path"]
  good=path.is_file() and path.stat().st_size==record["bytes"] and sha(path)==record["sha256"]
  assert good,str(path)
  original_hashes.append(str(path))
 assert len(original_hashes)==60
 for folder in (rlt,isa):
  assert not git(folder,"diff","--name-only")
  assert not git(folder,"diff","--cached","--name-only")
 manifest={
 "task":"NC-005","result":"IMPLEMENTER_CHECKPOINT_INTEGRITY_PASS_NOT_INDEPENDENT",
 "checkpoint_count":42,"pilot_checkpoint_count":42,"model_count":6,
 "torch":str(torch.__version__),"python":"3.14.6","device":"CPU",
 "model_parameter_count":total_params,
 "source_nc004_coupled_weights":source,
 "source_nc004_checkpoint_manifest_sha256":sha(ROOT/"artifacts/NC-004/checkpoint.json"),
 "source_nc001_isa_sha256":sha(ROOT/"artifacts/NC-001/canonical_isa.py"),
 "source_nc000_inventory_sha256":sha(ROOT/"artifacts/NC-000/inventory.json"),
 "source_original_repositories":{
    "rlt":{"path":str(rlt),"commit":git(rlt,"rev-parse","HEAD")},
    "isa":{"path":str(isa),"commit":git(isa,"rev-parse","HEAD")}
 },
 "original_checkpoint_sha256":sha(champ),
 "original_inventory_files_checked":len(original_hashes),
 "saved_optimizer_and_rng_checked":optimizer_verified,
 "code_files":[file_record(p) for p in source_files],
 "receipts":[file_record(HERE/name) for name in ("training_report.json",
   "curriculum_receipts.jsonl","retention_matrix.json","fresh_reload.json")],
 "checkpoints":files,"pilot_checkpoints_excluded_from_scientific_claim":pilots,
 "explanation":"Each saved stage contains model weights, original source SHA, newly initialized Adam state, torch/python/numpy RNG, stage/seed/updatestep and all topology from pinned Python module; does not reconstruct NC004 historical optimizer.",
 "scientific_boundaries":["The source NC004 5-op MLP was already trained on all five local binary truth tables",
    "All 9 * 64 local patterns available during NC005 training",
    "Training sequence/loop data generated from supplied host program traces, not independently generated programs",
    "Program PC, opcode dispatch, flags, register routing and HALT still host-provided",
    "Rehearsal vs fresh-only matches seeds, initialization and step budgets but changes exposure by design",
    "No general reasoning, learned program invention, generalization beyond declared scope or AGI demonstrated",
    "Independent verifier must inspect actual training and freshly replay final models before PASS"],
 "promotion":False,"distinct_verifier_completed":False}
 path=HERE/"checkpoint_manifest.json"
 path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"result":manifest["result"],"models":manifest["model_count"],
  "checkpoints":manifest["checkpoint_count"],"verified_optimizer_state":optimizer_verified,
  "pilot_archived":len(pilots),"source_inventory_60":len(original_hashes),
  "champion_sha":sha(champ),"manifest_sha":sha(path)},indent=2))
if __name__=="__main__":run()
