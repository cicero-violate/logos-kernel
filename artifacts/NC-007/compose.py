#!/usr/bin/env python3
"""NC-007: bounded neural macro selector over frozen NC-006 local primitives.

Dependencies: Python 3 stdlib, PyTorch, NumPy, artifacts/NC-005/curriculum.py,
frozen NC-005 rehearsal stage06 checkpoints, NC-005 checkpoint_manifest.json,
NC-006 accepted_library.json + library_manifest.json, NC-001/canonical_isa.py,
and NC-000 original RLT checkpoint (hash only). No GPU, neural checkpoint
retraining, symbolic inference fallback, loop-control learning or promotion.
"""
import collections
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"artifacts/NC-005"))
import curriculum as frozen
sys.path.insert(0,str(ROOT/"artifacts/NC-001"))
import canonical_isa as isa

SCHEMA="logos.nc007.composition/1"
WIDTH=16
MASK=(1<<WIDTH)-1
SEEDS=(11,19,29)
UPDATES=350
MAX_STEPS=5
CONTROLLER_DIM=32
CHECKPOINTS=HERE/"checkpoints"

# Action tokens bind fully typed (op, sources, destination) calls. HOST registers,
# token decoding, constants and control flow are not learned primitives.
ACTIONS={
 "ADD_XX":("ADD","X","X"),
 "ADD_X1":("ADD","X","K"),
 "AND1_X":("AND1","X",None),
 "XOR_XY":("XOR","X","Y"),
 "SHL_R":("SHL","R",None),
 "AND_XY":("AND","X","Y"),
 "ADD_RX":("ADD","R","X"),
 "SUB_XY":("SUB","X","Y"),
 "XOR_RX":("XOR","R","X"),
 "OR_XY":("OR","X","Y"),
 "SHR_R":("SHR","R",None),
 "ADD_XY":("ADD","X","Y"),
 "XOR_RY":("XOR","R","Y"),
 "ADD_RY":("ADD","R","Y"),
 "HALT":(None,None,None),
}
ACTION_NAMES=tuple(ACTIONS)
ACTION_ID={a:i for i,a in enumerate(ACTION_NAMES)}
START_ID=len(ACTION_NAMES)

# These programs are SUPERVISED teaching labels, never discoveries by the model.
# The trained controller receives only the goal token and previous predicted
# token, NOT the action list or a symbolic implementation at evaluation.
GOALS={
 "DOUBLE":("ADD_XX","HALT"),
 "INC":("ADD_X1","HALT"),
 "PARITY":("AND1_X","HALT"),
 "XOR_DOUBLE":("XOR_XY","SHL_R","HALT"),
 "AND_PLUS_X":("AND_XY","ADD_RX","HALT"),
 "SUB_XOR_X":("SUB_XY","XOR_RX","HALT"),
 "OR_SHIFT_RIGHT":("OR_XY","SHR_R","HALT"),
 "ADD_XOR_Y":("ADD_XY","XOR_RY","HALT"),
 "XOR_ADD_X":("XOR_XY","ADD_RX","HALT"),
 "DOUBLE_PLUS_Y":("ADD_XX","ADD_RY","HALT"),
 "TRIPLE":("ADD_XX","ADD_RX","HALT"),
}
GOAL_NAMES=tuple(GOALS)
GOAL_ID={x:i for i,x in enumerate(GOAL_NAMES)}
LIB_PATH=ROOT/"artifacts/NC-006/accepted_library.json"
LIB_MANIFEST=ROOT/"artifacts/NC-006/library_manifest.json"
CHECKPOINT_MANIFEST=ROOT/"artifacts/NC-005/checkpoint_manifest.json"
NEURAL_SRC=ROOT/"artifacts/NC-005/curriculum.py"
ISA_SRC=ROOT/"artifacts/NC-001/canonical_isa.py"

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)

def sha_json(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()

class MacroController(nn.Module):
    """Autoregressive learned next-action distribution, fixed typed token space."""
    def __init__(self):
        super().__init__()
        self.goal=nn.Embedding(len(GOALS),CONTROLLER_DIM)
        self.previous=nn.Embedding(len(ACTIONS)+1,CONTROLLER_DIM)
        self.position=nn.Embedding(MAX_STEPS,CONTROLLER_DIM)
        self.decoder=nn.Sequential(
            nn.Linear(3*CONTROLLER_DIM,96),nn.Tanh(),
            nn.Linear(96,96),nn.Tanh(),nn.Linear(96,len(ACTIONS))
        )
    def forward(self,goals,positions,previous):
        e=torch.cat((self.goal(goals),self.position(positions),self.previous(previous)),dim=-1)
        return self.decoder(e)

def teaching_rows():
    rows=[]
    for name,actions in GOALS.items():
        prev=START_ID
        for step,action in enumerate(actions):
            rows.append((GOAL_ID[name],step,prev,ACTION_ID[action]))
            prev=ACTION_ID[action]
    return rows

def predict(model,name):
    """No target macro access at evaluation: neural next-token autoregression."""
    if name not in GOAL_ID:raise ValueError("unknown goal; not independently generalized")
    model.eval()
    pred=[]
    prev=START_ID
    with torch.no_grad():
        for step in range(MAX_STEPS):
            logits=model(torch.tensor([GOAL_ID[name]]),
                         torch.tensor([step]),torch.tensor([prev]))
            token=ACTION_NAMES[int(logits.argmax(dim=-1)[0])]
            pred.append(token)
            if token=="HALT":break
            prev=ACTION_ID[token]
    if not pred or pred[-1]!="HALT":
        return {"status":"NO_HALT","actions":pred}
    if len(pred)<=1:return {"status":"EMPTY_PROGRAM","actions":pred}
    return {"status":"HALTED","actions":pred}

def load_frozen(seed):
    lib=json.loads(LIB_PATH.read_text())
    libmeta=json.loads(LIB_MANIFEST.read_text())
    ckmeta=json.loads(CHECKPOINT_MANIFEST.read_text())
    assert lib["accepted_neural_count"]==30 and lib["excluded_neural_count"]==30
    assert lib["champion_promoted"] is False
    assert libmeta["independent_verifier_status"]=="PASS"
    assert file_sha(NEURAL_SRC)==libmeta["source_sha256"]["NC005_curriculum"]
    assert file_sha(ISA_SRC)==libmeta["source_sha256"]["NC001_canonical_isa"]
    assert file_sha(CHECKPOINT_MANIFEST)==libmeta["checkpoint_manifest_sha256"]
    assert file_sha(LIB_PATH)==libmeta["independent_verification"]["accepted_library_sha256"]
    assert file_sha(Path(libmeta["source_files"]["original_RLT_champion"]))==libmeta["original_champion_sha256"]
    path=ROOT/f"artifacts/NC-005/checkpoints/rehearsal_seed{seed}_stage06.pt"
    ck=next(x for x in ckmeta["checkpoints"] if x["path"]==str(path.relative_to(ROOT)))
    assert file_sha(path)==ck["sha256"]
    registry={}
    for a,(op,_,_) in ACTIONS.items():
        if op is None:continue
        candidate=[q for q in lib["accepted_neural"] if
                   q["seed"]==seed and q["operation"]==op and
                   q["arm"]=="rehearsal" and q["implementation_mode"]=="neural_inference"]
        assert len(candidate)==1 and candidate[0]["checkpoint_sha256"]==ck["sha256"]
        registry[a]=candidate[0]
    # The payload is a pre-verified LOCAL PyTorch checkpoint; pickle loading is
    # not safe for arbitrary untrusted files and must not be generalized.
    payload=torch.load(path,map_location="cpu",weights_only=False)
    assert tuple(payload["ops"])==tuple(frozen.OPS)
    model=frozen.ExpandedISA().cpu().eval()
    model.load_state_dict(payload["model"],strict=True)
    model.requires_grad_(False)
    # Bind the exact accepted source records to the verified loaded model,
    # independently of the mutable mapping handed to the executor.
    # This is a provenance check, not authentication against malicious Python
    # code with write access to model internals.
    model._nc007_provenance={
        "seed":seed,"checkpoint_sha256":ck["sha256"],
        "accepted_library_sha256":file_sha(LIB_PATH),
        "registered_actions":json.loads(canonical(registry))}
    return model,registry,{"checkpoint_path":str(path.relative_to(ROOT)),
                            "checkpoint_sha256":ck["sha256"],
                            "library_sha256":file_sha(LIB_PATH),
                            "isa_sha256":file_sha(ISA_SRC),
                            "source_sha256":file_sha(NEURAL_SRC)}

def train_controller(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(1)
    model=MacroController().cpu()
    opt=torch.optim.AdamW(model.parameters(),lr=0.018,weight_decay=0)
    rows=teaching_rows()
    goals=torch.tensor([x[0] for x in rows],dtype=torch.long)
    pos=torch.tensor([x[1] for x in rows],dtype=torch.long)
    prev=torch.tensor([x[2] for x in rows],dtype=torch.long)
    target=torch.tensor([x[3] for x in rows],dtype=torch.long)
    untrained={g:predict(model,g) for g in GOALS}
    t0=time.perf_counter()
    for _ in range(UPDATES):
        model.train()
        loss=F.cross_entropy(model(goals,pos,prev),target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    seconds=time.perf_counter()-t0
    trained={g:predict(model,g) for g in GOALS}
    correct=sum(trained[g]["status"]=="HALTED" and
                tuple(trained[g]["actions"])==GOALS[g] for g in GOALS)
    before=sum(untrained[g]["status"]=="HALTED" and
               tuple(untrained[g]["actions"])==GOALS[g] for g in GOALS)
    payload={
        "schema":SCHEMA,"task":"NC-007","seed":seed,"architecture":"MacroController",
        "model":model.state_dict(),"optimizer":opt.state_dict(),
        "rng_python":random.getstate(),"rng_numpy":np.random.get_state(),
        "rng_torch":torch.get_rng_state(),
        "action_vocabulary":ACTION_NAMES,"goals":GOAL_NAMES,
        "teaching_rows_sha256":sha_json(rows),
        "updates":UPDATES,"learning_rate":0.018,
        "last_loss":float(loss.item()),
        "source_sha256":file_sha(HERE/"compose.py"),
        "library_sha256":file_sha(LIB_PATH),
        "frozen_source_sha256":file_sha(NEURAL_SRC),
        "checkpoint_kind":"separate trained controller, NOT frozen ISA checkpoint",
        "no_champion_promotion":True
    }
    CHECKPOINTS.mkdir(parents=True,exist_ok=True)
    dest=CHECKPOINTS/f"controller_seed{seed}.pt"
    torch.save(payload,dest)
    return model,{"seed":seed,"checkpoint_path":str(dest.relative_to(ROOT)),
                  "checkpoint_sha256":file_sha(dest),
                  "model_parameters":sum(p.numel() for p in model.parameters()),
                  "training_updates":UPDATES,"supervision_rows":len(rows),
                  "teacher_forced_final_loss":float(loss.item()),"training_seconds":seconds,
                  "before_training_exact_macros":before,
                  "after_training_exact_macros":correct,"total_macros":len(GOALS)}

def reload_controller(seed,meta):
    path=ROOT/meta["checkpoint_path"]
    assert file_sha(path)==meta["checkpoint_sha256"]
    data=torch.load(path,map_location="cpu",weights_only=False)
    assert data["schema"]==SCHEMA and data["seed"]==seed
    assert data["source_sha256"]==file_sha(HERE/"compose.py")
    assert tuple(data["action_vocabulary"])==ACTION_NAMES
    assert tuple(data["goals"])==GOAL_NAMES
    assert data["teaching_rows_sha256"]==sha_json(teaching_rows())
    m=MacroController().eval()
    m.load_state_dict(data["model"],strict=True)
    return m

def bit_constants(pairs):
    x=frozen.bits([x for x,y in pairs]);y=frozen.bits([y for x,y in pairs])
    z=torch.zeros_like(x);one=frozen.bits([1]*len(pairs))
    return {"X":x,"Y":y,"R":z,"K":one}

def execute_neural(frozen_model,program,pairs,registry):
    """Model-generated choices affect computation, never oracle replacement."""
    if not program or len(program)<2 or program[-1]!="HALT" or "HALT" in program[:-1]:
        raise ValueError("unterminated/malformed controller-generated program")
    trust=getattr(frozen_model,"_nc007_provenance",None)
    if type(trust) is not dict or type(trust.get("registered_actions")) is not dict:
        raise ValueError("NC007_MODEL_WITHOUT_VERIFIED_PROVENANCE")
    if file_sha(LIB_PATH)!=trust["accepted_library_sha256"]:
        raise ValueError("NC007_ACCEPTED_LIBRARY_CHANGED")
    # Fail closed before any neural execution: one corrupted entry must reject
    # the entire candidate program, never execute and report forged identity.
    for token in program[:-1]:
        if token not in ACTIONS or token not in registry:
            raise ValueError("unregistered, non-neural or unknown token")
        op,a,b=ACTIONS[token]
        entry=registry[token]
        expected=trust["registered_actions"].get(token)
        if type(entry) is not dict or entry!=expected or (
                entry.get("operation")!=op or
                entry.get("seed")!=trust["seed"] or
                entry.get("arm")!="rehearsal" or
                entry.get("implementation_mode")!="neural_inference" or
                entry.get("checkpoint_sha256")!=trust["checkpoint_sha256"]):
            raise ValueError("NC007_UNVERIFIED_PRIMITIVE_PROVENANCE")
    state=bit_constants(pairs)
    source_refs=[]
    t0=time.perf_counter_ns()
    with torch.no_grad():
        for token in program[:-1]:
            op,a,b=ACTIONS[token]
            x=state[a];y=state[b] if b is not None else None
            result,_=frozen.step_numeric(frozen_model,op,x,y)
            state["R"]=result
            source_refs.append(registry[token]["id"])
    elapsed_ns=time.perf_counter_ns()-t0
    return frozen.words(state["R"]),source_refs,elapsed_ns

def reference_ast(program):
    """The separate symbolic NC-001 oracle is ONLY for expected results."""
    if not program or program[-1]!="HALT":raise ValueError("unterminated")
    state={"X":isa.read("X",WIDTH,"unsigned"),
           "Y":isa.read("Y",WIDTH,"unsigned"),
           "K":isa.lit(1,WIDTH,"unsigned"),
           "R":isa.lit(0,WIDTH,"unsigned")}
    for token in program[:-1]:
        if token not in ACTIONS:raise ValueError("bad reference instruction")
        op,a,b=ACTIONS[token]
        vals=[state[a]] if b is None else [state[a],state[b]]
        state["R"]={"kind":"bv."+op.lower(),"width":WIDTH,
                    "signedness":"unsigned","args":vals}
    isa.validate_ast(state["R"],WIDTH,"unsigned")
    return state["R"]

def reference_values(program,pairs):
    expr=reference_ast(program)
    return [isa.eval_ast(expr,{"X":x,"Y":y,"R":0,"T":0},WIDTH,"unsigned")
            for x,y in pairs]

def heldout_pairs(seed,n=256):
    # The controller is trained on token sequences, never on operand values.
    rng=random.Random(0xA17C7+seed)
    corners=[(0,0),(0,MASK),(MASK,0),(MASK,MASK),
             (1,1),(0,1),(1,0),(MASK,1),(1,MASK),
             (32768,32768),(32767,32768),(32768,32767),
             (0x5555,0xAAAA),(0xAAAA,0x5555)]
    pairs=list(dict.fromkeys(corners))
    while len(pairs)<n:
        v=(rng.randrange(1<<WIDTH),rng.randrange(1<<WIDTH))
        if v not in pairs:pairs.append(v)
    assert len(set(pairs))==n
    return pairs

def data_dependency_dag(program,registry):
    """Acyclic data dependencies; no CFG loops encoded as DAG edges."""
    nodes=[]
    prior=None
    for k,token in enumerate(program[:-1]):
        op,a,b=ACTIONS[token]
        deps=[prior] if prior is not None and "R" in (a,b) else []
        nodes.append({"id":f"n{k}","action":token,"operation":op,
                      "inputs":[a]+([b] if b is not None else []),
                      "output":"R","depends_on":deps,
                      "neural_circuit":registry[token]["id"]})
        prior=f"n{k}"
    return {"kind":"data_dependency_DAG","nodes":nodes,
            "edges":[[d,n["id"]] for n in nodes for d in n["depends_on"]],
            "acyclic":True}

def runtime_loop_cfg():
    """Explicit host-controlled runtime CFG, deliberately NOT a task/DAG loop."""
    return {"kind":"runtime_CFG","name":"HOST_LOOP_TRIPLE","repeat":3,
            "nodes":[{"id":"entry","mechanism":"host R=0, counter=3"},
                     {"id":"add","mechanism":"frozen neural ADD(R,X)"},
                     {"id":"decrement","mechanism":"host counter--"},
                     {"id":"branch","mechanism":"host conditional PC"},
                     {"id":"halt","mechanism":"host HALT"}],
            "edges":[["entry","add"],["add","decrement"],
                     ["decrement","branch"],["branch","add"],["branch","halt"]],
            "cycle_present":True,"host_controlled":True,
            "model_generated":False}

def eval_loop(frozen_model,seed,registry,pairs):
    state=bit_constants(pairs)
    t0=time.perf_counter_ns()
    with torch.no_grad():
        for iteration in range(3):
            value,_=frozen.step_numeric(frozen_model,"ADD",state["R"],state["X"])
            state["R"]=value
    predictions=frozen.words(state["R"])
    elapsed=time.perf_counter_ns()-t0
    reference=[(3*x)&MASK for x,y in pairs]
    errors=[{"X":x,"Y":y,"expected":e,"observed":p}
            for (x,y),e,p in zip(pairs,reference,predictions) if e!=p]
    return {"schema":SCHEMA,"seed":seed,"goal":"HOST_LOOP_TRIPLE",
            "kind":"runtime_CFG","cfg":runtime_loop_cfg(),
            "data_dependency_DAG":None,
            "tested":len(pairs),"correct":len(pairs)-len(errors),
            "failure_examples":errors,"neural_primitive_calls":3,
            "source_proof_circuit_id":registry["ADD_RX"]["id"],
            "duration_ns":elapsed,"not_learned_control":True}

def run(seed,controller,controller_meta,neural_model,registry,frozen_info):
    rows=[]
    pairs=heldout_pairs(seed)
    case_sha=sha_json(pairs)
    for name,expected_program in GOALS.items():
        pred=predict(controller,name)
        if pred["status"]!="HALTED":
            row={"schema":SCHEMA,"seed":seed,"goal":name,"status":pred["status"],
                 "generated_program":pred["actions"],"reference_program":expected_program,
                 "correct":0,"tested":len(pairs),"failure_examples":[
                 {"reason":"no executable model-generated program"}],
                 "controller_generated":True,"no_symbolic_fallback":True,
                 "controller_checkpoint_sha256":controller_meta["checkpoint_sha256"],
                 "heldout_pairs_sha256":case_sha}
            rows.append(row);continue
        actions=pred["actions"]
        try:
            results,proofs,elapsed=execute_neural(neural_model,actions,pairs,registry)
            reference=reference_values(expected_program,pairs)
            failures=[{"X":x,"Y":y,"expected":t,"observed":o}
                      for (x,y),t,o in zip(pairs,reference,results) if t!=o]
            correct=len(pairs)-len(failures)
            status="EXECUTED"
        except (ValueError,AssertionError,KeyError) as e:
            failures=[{"reason":type(e).__name__+":"+str(e)}]
            correct=0;status="INVALID_PROGRAM"
            proofs=[];elapsed=0
        row={"schema":SCHEMA,"seed":seed,"goal":name,"status":status,
             "generated_program":actions,"reference_program":expected_program,
             "exact_target_program":tuple(actions)==expected_program,
             "correct":correct,"tested":len(pairs),"failure_examples":failures,
             "controller_generated":True,"no_symbolic_fallback":True,
             "controller_checkpoint_sha256":controller_meta["checkpoint_sha256"],
             "frozen_checkpoint_sha256":frozen_info["checkpoint_sha256"],
             "heldout_pairs_sha256":case_sha,"program_length":len(actions)-1,
             "reference_program_length":len(expected_program)-1,
             "neural_primitive_calls":len(proofs),"source_proof_circuit_ids":proofs,
             "duration_ns":elapsed,
             "data_dependency_DAG":data_dependency_dag(actions,registry)}
        rows.append(row)
    rows.append(eval_loop(neural_model,seed,registry,pairs))
    return rows

def run_all():
    assert all(v[-1]=="HALT" for v in GOALS.values())
    assert len(GOALS)==len(set(GOALS))
    assert tuple(frozen.SEEDS)==SEEDS and frozen.WIDTH==WIDTH
    entries=[];trained=[];controller_predictions={}
    t0=time.perf_counter()
    for seed in SEEDS:
        model,meta=train_controller(seed)
        assert meta["after_training_exact_macros"]==len(GOALS),(
            f"controller failure at seed {seed}: {meta}")
        reloaded=reload_controller(seed,meta)
        assert all(predict(model,g)==predict(reloaded,g) for g in GOALS)
        neural,reg,info=load_frozen(seed)
        rows=run(seed,reloaded,meta,neural,reg,info)
        entries.extend(rows);trained.append({**meta,"frozen_neural":info})
        controller_predictions[str(seed)]={g:predict(reloaded,g) for g in GOALS}
    HERE.mkdir(parents=True,exist_ok=True)
    trace=HERE/"composition_traces.jsonl"
    trace.write_text("".join(canonical(r)+"\n" for r in entries))
    numeric=[r for r in entries if r["goal"] in GOALS]
    loop_rows=[r for r in entries if r["goal"]=="HOST_LOOP_TRIPLE"]
    tested=sum(x["tested"] for x in numeric)
    correct=sum(x["correct"] for x in numeric)
    assert all(x["correct"]==x["tested"] for x in numeric),(
        "NC007_NUMERIC_GATE_FAIL",[(x["seed"],x["goal"],x["correct"],x["tested"])
                                   for x in numeric if x["correct"]!=x["tested"]])
    assert all(x["correct"]==x["tested"] for x in loop_rows)
    report={
        "schema":SCHEMA,"task":"NC-007","status":"IMPLEMENTER_TESTED_PENDING_INDEPENDENT",
        "versioned_macro_programs":{k:list(v) for k,v in GOALS.items()},
        "neural_action_contracts":{k:list(v) for k,v in ACTIONS.items()},
        "training":[x for x in trained],
        "controller_architecture":"goal+position+previous action embeddings -> MLP next token",
        "controller_input_scope":"known formal macro goal ID; no operand inputs; no unseen goal families",
        "total_train_examples_per_seed":len(teaching_rows()),
        "gradient_updates_per_seed":UPDATES,
        "new_controller_parameters":sum(p.numel() for p in MacroController().parameters()),
        "precision":"uint16 output bits via frozen local neural primitives",
        "numeric_compositions":{"correct":correct,"tested":tested,
             "macro_count":len(GOALS),"seeds":list(SEEDS),
             "per_seed_tested":256*len(GOALS),
             "generated_programs":len(numeric)},
        "host_loop":{"correct":sum(x["correct"] for x in loop_rows),
            "tested":sum(x["tested"] for x in loop_rows),
            "controller_generated":False,"cfg":runtime_loop_cfg()},
        "neural_controller_predictions":controller_predictions,
        "data_dependency_dags_separate_from_runtime_cfg":True,
        "source_sha256":{"compose.py":file_sha(HERE/"compose.py"),
                        "nc005_curriculum":file_sha(NEURAL_SRC),
                        "nc006_accepted_library":file_sha(LIB_PATH),
                        "nc005_checkpoint_manifest":file_sha(CHECKPOINT_MANIFEST),
                        "nc001_canonical_isa":file_sha(ISA_SRC)},
        "trace_file_sha256":file_sha(trace),
        "trace_entries":len(entries),
        "code_length_ops":dict(collections.Counter(
            x["program_length"] for x in numeric)),
        "source_reuse":dict(collections.Counter(c for x in numeric for c in
                                                 x["source_proof_circuit_ids"])),
        "elapsed_training_and_eval_seconds":time.perf_counter()-t0,
        "no_model_promotion":True,"frozen_original_neural_weights_unchanged":True,
        "limit":["The controller learns a mapping from 11 known supervised macro IDs to action-token sequences; it does not invent algorithms or generalize to new macro IDs.",
                 "Operand values are not seen by the controller during training, but 64 local neural primitive patterns were seen in earlier NC-005 training.",
                 "Register wiring, constants, decoding, bit iteration, carry, symbolic truth oracle and HALT execution remain host mechanisms.",
                 "Loop CFG is host-controlled, not generated by the controller.",
                 "Only tested finite 16-bit operand samples; no exhaustive 16-bit operand-pair proof.",
                 "Not a Transformer and not the unified RLT champion; no program invention, AGI, language or autonomous control proven."]
    }
    out=HERE/"composition_receipt.json"
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(canonical({"status":report["status"],"macro_programs":len(numeric),
                     "numeric_correct":correct,"numeric_tested":tested,
                     "loop_correct":report["host_loop"]["correct"],
                     "loop_tested":report["host_loop"]["tested"],
                     "training_macros":[z["after_training_exact_macros"] for z in trained],
                     "trace_sha256":report["trace_file_sha256"]}))
    return report

if __name__=="__main__":
    run_all()
