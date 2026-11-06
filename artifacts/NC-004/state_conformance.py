#!/usr/bin/env python3
"""NC-004: actual neural R-value + explicit HOST flags/PC/HALT vs NC-001.

Dependencies: causal_semantics.py, torch, NC-001 canonical_isa.py, frozen
standalone train_causal.py, saved six NC-004 checkpoints. This does NOT
claim flags, PC, HALT or program synthesis are learned.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from causal_semantics import (
    HERE, CHAMPION, INV, OPS, ARMS, SEEDS, isa,
    load, execute_program, instruction, predicted_ast, sha,
)
def verify_all():
    torch.set_num_threads(1)
    inv=json.loads(INV.read_text())
    assert sha(CHAMPION)==inv["champion"]["selected_bundle_sha256"]
    cases=[(x,y) for x in range(16) for y in range(16)]
    aa=[x for x,y in cases]; bb=[y for x,y in cases]
    rows={}
    for arm in ARMS:
        for seed in SEEDS:
            name=arm+"_seed"+str(seed)
            model=load(seed,arm)
            op_rows={}
            for op in OPS:
                item=instruction(op)
                program={"version":isa.VERSION,"width":16,
                         "signedness":"unsigned","program":[item,{"op":"HALT"}]}
                predicted_math,predicted_typed_ast=predicted_ast(model,op)
                required_ast=isa.lower(program,0)["write"]["value"]
                assert predicted_math==op and predicted_typed_ast==required_ast
                predicted_values=execute_program(model,[(op,"X","Y" if op in ("ADD","OR") else None)],aa,bb)
                correct=0
                failed=[]
                for (a,b),value in zip(cases,predicted_values):
                    original={"X":a,"Y":b,"R":0,"T":0}
                    predicted={"version":isa.VERSION,"status":"halted","halted":True,
                         "pc":1,"steps":2,
                         "registers":{**original,"R":value},
                         "flags":isa.flags_alu(op,item,original,value,16)}
                    # This oracle is invoked only AFTER neural prediction.
                    reference=isa.reference(program,original,max_steps=3)
                    reference={k:reference[k] for k in predicted}
                    if predicted==reference:
                        correct+=1
                    elif len(failed)<5:
                        failed.append({"a":a,"b":b,"neural_value":value,
                                      "expected_value":reference["registers"]["R"],
                                      "predicted_host_transition":predicted,
                                      "reference_transition":reference})
                op_rows[op]={"correct_full_transition":correct,"tested":len(cases),
                             "math_AST_match":True,"failed_examples":failed}
            rows[name]=op_rows
            print(json.dumps({"model":name,"full_state":{k:f"{v['correct_full_transition']}/{v['tested']}" for k,v in op_rows.items()}}),flush=True)
    total=sum(r["tested"] for v in rows.values() for r in v.values())
    correct=sum(r["correct_full_transition"] for v in rows.values() for r in v.values())
    report={"task":"NC-004","result":"IMPLEMENTER_HOST_ASSISTED_FULL_TRANSITION_CONFORMANCE",
      "cases_per_op_model":256,"learned_neural_components":["math class logits","arithmetic result"],
      "deterministic_host_components":["AST rendering","flag formulas","PC increment","HALT","register wiring"],
      "symbolic_reference":"NC-001 canonical_isa.reference independent post-prediction",
      "all_training_primitives_trained_on_all_local_truth_patterns":True,
      "full_transition_correct":correct,"full_transition_tested":total,
      "frozen_champion_sha256":sha(CHAMPION),"rows":rows}
    out=HERE/"state_conformance.json"
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"file":out.name,"sha256":sha(out),"correct":correct,"tested":total}))
    return correct==total

if __name__=="__main__":
    raise SystemExit(0 if verify_all() else 1)
