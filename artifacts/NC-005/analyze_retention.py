#!/usr/bin/env python3
"""Audit NC005 stage receipts and generate a nonpromotional retention matrix.

Dependencies: Python 3 stdlib, artifacts/NC-005/training_report.json,
curriculum_receipts.jsonl, frozen champion SHA and 42 saved stage checkpoints.
No training and no weight edits.
"""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
STAGES=("BOOLEAN","CARRY_ADD","SHIFTS","COMPARE_SUB","PC_BRANCH","SEQUENCES","LOOPS")
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as fd:
  for chunk in iter(lambda:fd.read(1<<20),b""):h.update(chunk)
 return h.hexdigest()

def analyze():
 report=json.loads((HERE/"training_report.json").read_text())
 receipts=[json.loads(z) for z in (HERE/"curriculum_receipts.jsonl").read_text().splitlines() if z.strip()]
 assert len(receipts)==42,(len(receipts),"missing stage receipts")
 assert len(report["results"])==6
 seeds=report["seeds"];arms=report["arms"]
 assert seeds==[11,19,29] and arms==["rehearsal","new_only"]
 fail=[];matrix={};models={}
 seen=set()
 for row in receipts:
  key=(row["arm"],row["seed"],row["stage"])
  assert key not in seen,("duplicate receipt",key)
  seen.add(key)
 assert len(seen)==42
 for arm in arms:
  matrix[arm]={}
  for seed in seeds:
   key=f"{arm}_seed{seed}"
   model=report["results"][key]
   assert [x["stage"] for x in model["stages"]]==list(STAGES)
   assert model["initial_old_five"]["correct"]==320
   assert model["parameter_count"]==5981
   prior={};branch_ready=False
   rows=[]
   for idx,stage in enumerate(model["stages"]):
    source=ROOT/stage["checkpoint"]["path"]
    assert source.is_file() and sha(source)==stage["checkpoint"]["sha256"],str(source)
    assert source.stat().st_size==stage["checkpoint"]["bytes"]
    current_old=stage["old_five_retention"]
    denominator=sum(x["tested"] for x in stage["prior_retention"].values())
    numerator=sum(x["correct"] for x in stage["prior_retention"].values())
    assert denominator==64*len(stage["qualified_prior_ops"])
    failures=[name for name,z in stage["prior_retention"].items() if z["correct"]!=z["tested"]
              or z["meaning_correct"]!=z["meaning_tested"]]
    branch=stage["heldout_branch"]
    stage_results={"stage":stage["stage"],"index":idx,"updates":stage["updates_cumulative"],
       "qualified_prior_truth":{"correct":numerator,"tested":denominator},
       "failed_prior_ops":failures,
       "NC004_old_five":{"correct":current_old["correct"],"tested":320},
       "branch_zero_test":{"correct":branch["correct"],"tested":branch["tested"],
                           "false_positive":branch["false_positive"],
                           "false_negative":branch["false_negative"]},
       "source_checkpoint":stage["checkpoint"],
       "new_8bit_test":None,"sequence_test":None,"loop_test":None}
    if idx>=3:
     op=stage["heldout_8bit_ops"]
     assert set(op)==set(report["results"][key]["stages"][-1]["local_by_op"])
     count=sum(z["correct"] for z in op.values())
     total=sum(z["tested"] for z in op.values())
     stage_results["new_8bit_test"]={"correct":count,"tested":total,
            "failed_ops":[o for o,z in op.items() if z["correct"]!=z["tested"]]}
    if idx>=5:
     seq=stage["heldout_sequences"]
     stage_results["sequence_test"]={"correct":sum(z["correct"] for z in seq.values()),
       "tested":sum(z["tested"] for z in seq.values()),
       "failed_programs":[p for p,z in seq.items() if z["correct"]!=z["tested"]],
       "unseen_training_programs":["subtract_then_xor","xor_then_double"]}
    if idx>=6:
     loop=stage["heldout_loops"]
     stage_results["loop_test"]={k:{"correct":z["correct"],"tested":z["tested"],
               "unhalted":z["unhalted"]} for k,z in loop.items()}
    if arm=="rehearsal":
     if failures:fail.append({"seed":seed,"stage":stage["stage"],"reason":"prior local retention", "failed_ops":failures})
     if idx>=4 and branch["correct"]!=branch["tested"]:
      fail.append({"seed":seed,"stage":stage["stage"],"reason":"trained branch regression"})
     if idx>=3 and stage_results["new_8bit_test"]["correct"]!=stage_results["new_8bit_test"]["tested"]:
      fail.append({"seed":seed,"stage":stage["stage"],"reason":"8bit heldout arithmetic regression"})
     if idx>=5 and stage_results["sequence_test"]["correct"]!=stage_results["sequence_test"]["tested"]:
      fail.append({"seed":seed,"stage":stage["stage"],"reason":"supplied-sequence regression"})
     if idx>=6 and any(z["correct"]!=z["tested"] or z["unhalted"] for z in stage_results["loop_test"].values()):
      fail.append({"seed":seed,"stage":stage["stage"],"reason":"loop regression"})
    rows.append(stage_results)
   models[key]={"total_updates":model["stages"][-1]["updates_cumulative"],
                "checkpoints_verified":7,
                "program_trace_sources":model["program_trace_sources"],
                "last_qualified_prior_truth":rows[-1]["qualified_prior_truth"],
                "last_branch":rows[-1]["branch_zero_test"],
                "last_loops":rows[-1]["loop_test"]}
   matrix[arm][str(seed)]=rows
 for arm in arms:
  for seed in seeds:
   assert models[f"{arm}_seed{seed}"]["total_updates"]==report["stage_updates"]
 base=matrix["rehearsal"];control=matrix["new_only"]
 contrasts=[]
 for seed in seeds:
  for index,name in enumerate(STAGES):
   a=base[str(seed)][index];b=control[str(seed)][index]
   contrasts.append({"seed":seed,"stage":name,
       "rehearsal_prior_correct":a["qualified_prior_truth"]["correct"],
       "new_only_prior_correct":b["qualified_prior_truth"]["correct"],
       "same_denominator":a["qualified_prior_truth"]["tested"]==b["qualified_prior_truth"]["tested"],
       "prior_examples":a["qualified_prior_truth"]["tested"],
       "rehearsal_advantage":a["qualified_prior_truth"]["correct"]-b["qualified_prior_truth"]["correct"]})
 result={"task":"NC-005",
  "status":"IMPLEMENTER_STAGE_GATES_PASS" if not fail else "IMPLEMENTER_STAGE_GATES_FAIL",
  "validation":"data from actual trained stage models, not proven independent until distinct agent audits",
  "architecture":"NC004 warm-start expanded 5981-param coupled semantic MLP and learned zero detector, NOT original RLT champion",
  "model_count":6,"checkpoint_count":42,
  "checkpoints_per_model":7,
  "stage_order":list(STAGES),
  "seeds":seeds,"arms":arms,
  "data_matching":"matched initial weights/seed/optimizer/updates/minibatch size and stage source pools; intentionally NOT identical exposure because replay policy is intervention",
  "semantic_scopes":{"learned":"nine local bit operations and a neural is-zero predicate",
     "host":"supplied sequences/loops, instruction choice, branching target/PC/HALT arithmetic, flags and register routing",
     "oracle":"heldout expected numeric result checked post prediction",
     "pretrained":"five NC004 operations had all local truth patterns already seen"},
  "trained_trajectory_sources":{str(seed):models[f"rehearsal_seed{seed}"]["program_trace_sources"] for seed in seeds},
  "matrix":matrix,"models":models,
  "comparisons":contrasts,
  "regression_failures_in_rehearsal":fail,
  "new_only_failed_stage_count":sum(z["rehearsal_advantage"]>0 for z in contrasts),
  "limits":["Old five operations are pretrained, not acquired from scratch in NC005",
     "All 64 local patterns per new primitive appear in training; final local truth accuracy is in-domain retention",
     "Sequence/loop examples come from host-supplied program traces and known primitive labels; no learned operation selection or program synthesis",
     "Branch head learns only zero test, host supplies JZ/JNZ inversion, target, PC update and HALT",
     "Some heldout test operand pairs may overlap old NC004/source pretraining; pretraining-disjoint proof absent",
     "Matched optimization step counts do not equal matched example exposure; rehearsal intentionally changes example distribution",
     "Pilot run without real sequence traces archived separately and excluded from acceptance",
     "Expanded module is not seven-operation RLT champion; no promotion.",
     "Fresh separate agent verification required before ledger PASS"]}
 (HERE/"retention_matrix.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"result":result["status"],"seeds":seeds,
   "checkpoints":42,"rehearsal_failures":fail,
   "new_only_stages_with_forgetting":result["new_only_failed_stage_count"],
   "prior_truth_final":{arm:{str(seed):matrix[arm][str(seed)][-1]["qualified_prior_truth"]
     for seed in seeds} for arm in arms},
   "matrix_sha256":sha(HERE/"retention_matrix.json")},indent=2))
 if fail:raise SystemExit(2)
if __name__=="__main__":analyze()
