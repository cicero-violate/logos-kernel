#!/usr/bin/env python3
"""NC-046 fail-closed, bounded CPU replay training worker.

Dependencies: Python>=3.10, PyTorch CPU, NumPy, Git, Linux flock,
artifacts/NC-022/{rlt_rehearsal.py,PROTOCOL.json,train_and_holdout_ledger.json,
independent_receipt.json}, artifacts/NC-000/inventory.json, and the *pinned*
original RLT task_conditioned_rlt.py / rlt_core.py / snapshot.
Runs ONLY explicit isolated canary experiments; never modifies the champion or
trainer ledger and never represents a canary as an approved TR-002 episode.
"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import re
import resource
import subprocess
import sys
import tempfile
import time

# Set before PyTorch import to isolate CPU workers from GPU availability.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
N22 = ROOT / "artifacts/NC-022"
TRAINER = Path("/workspace/experiments/logos-trainer")
REQUIRED_CHAMPION = "4c42d13fd46f57ac0db6eccbaef35e99c4cefcec393a3a3a4faf146645e8f683"
SOURCE_PINS = {
    "experiment": "2d07d82b1659320356addd8a0f54e130ad200891b6a1990fbed073ce79f96ded",
    "protocol": "152a2435601083c663c8c4ec3c25caf0d3bcb5f3a977cb19f9cf4dddaf690a9f",
    "dataset": "1c60767d692e09587293cdb964e0b40919905ac78b1c47aaf7eafaa042da94a3",
    "architecture": "99f70b03e2e864d1635ffdbd3ce56598d915bcc1673e1492964d3322714335fb",
    "backbone": "3c4abc18752236d3118487213fcfa1cf4a18e52e2ceb17ad52a0cdec141013c7",
}
ARMS = ("new_only", "rehearsal", "blocked_replay")
SEEDS = (11,19,29)
MAX_STEPS = 4
ZERO = "0"*64

class Refused(RuntimeError):
    pass

def require(ok, message):
    if not ok:
        raise Refused(message)

def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for b in iter(lambda: fh.read(1<<20), b""):
            h.update(b)
    return h.hexdigest()

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def validate_disjoint(blocks, heldkeys, ref):
    seen=set()
    for items in blocks.values():
        for block in items:
            for pair in block["pairs"]:
                key=ref.pairkey(*pair)
                require(key not in heldkeys, "TRAIN_EVAL_LEAK")
                require(key not in seen, "DUPLICATE_TRAIN_PAIR")
                seen.add(key)
    require(len(seen)==sum(len(x["pairs"]) for b in blocks.values() for x in b),"DUPLICATE_TRAIN")
    return len(seen)

def reference():
    paths = {
        "experiment": N22/"rlt_rehearsal.py",
        "protocol": N22/"PROTOCOL.json",
        "dataset": N22/"train_and_holdout_ledger.json",
        "architecture": Path("/workspace/experiments/rlt_split_algorithms_ablation_v1/task_conditioned_rlt.py"),
        "backbone": Path("/workspace/experiments/rlt_split_algorithms_ablation_v1/rlt_core.py"),
    }
    for name,path in paths.items():
        require(path.is_file() and sha_file(path)==SOURCE_PINS[name], "PIN_DRIFT:"+name)
    spec=importlib.util.spec_from_file_location("nc022_frozen_reference",paths["experiment"])
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    require(sha_file(module.SOURCE)==REQUIRED_CHAMPION,"CHAMPION_DRIFT")
    module.validate_contract()
    data=json.loads(paths["dataset"].read_text())
    require(set(data)==set(map(str,SEEDS)), "UNEXPECTED_SEEDS")
    for seed in SEEDS:
        groups,held=module.heldouts(seed)
        assert groups == data[str(seed)]["groups"]
        train=data[str(seed)]["blocks"]
        assert module.digest(groups)==data[str(seed)]["heldout_pair_set_sha256"]
        assert module.digest(train)==data[str(seed)]["train_pair_set_sha256"]
        validate_disjoint(train, held, module)
    return module,data

def trainer_status():
    p=subprocess.run([sys.executable,str(TRAINER/"trainer_dag.py"),"status"],
                     capture_output=True,text=True,timeout=20,cwd=TRAINER)
    require(p.returncode==0,"TRAINER_LEDGER_UNAVAILABLE")
    v=json.loads(p.stdout)
    require(v["git_synced"] is True and v["execution"]=="not_enabled","TRAINER_UNSAFE")
    return v

def production_guard():
    status=trainer_status()
    eligible=[t for t in status["tasks"] if t["approved"] and
              t["id"]!="TR-001" and t["source_sha256"]!=ZERO and
              t["data_sha256"]!=ZERO and t["status"]=="planned"]
    # No ledger job_state='running' or proc-bound authorization protocol exists
    # in NC-045. A same-UID untrusted tab can otherwise steal an active lease.
    require(False, "PRODUCTION_DISABLED:TR-002_UNAPPROVED_OR_NO_ATOMIC_LIVENESS_GATE")
    return eligible

def make_steps(data,seed,arm,ref):
    blocks=data[str(seed)]["blocks"]
    steps=[]
    for stage in (1,2):
        block=blocks[str(stage)][0]
        for phase in (0,1):
            ops=ref.task_for_block(ref.STAGES[stage],block,arm,phase)
            steps.append({"stage":stage,"block":0,"phase":phase,
                          "width":block["width"],"pairs":block["pairs"],"ops":ops})
    require(len(steps)==MAX_STEPS and all(len(x["ops"])==48 for x in steps),"BAD_STEP_PLAN")
    return steps

def count_exposure(steps):
    from collections import Counter
    out=Counter()
    for st in steps:
        for pair,op in zip(st["pairs"],st["ops"]):
            out[(st["stage"],st["width"],*pair,op)]+=1
    return sorted([[*key,v] for key,v in out.items()])

def evaluation(model,data,seed,ref):
    groups=data[str(seed)]["groups"]
    rows=[]
    model.eval()
    for group in ("carry8","iid16"):
        width=8 if group=="carry8" else 16
        pairs=groups[group][:12]
        for operation in ("add","xor","sub","mul","lt"):
            pred=[int(x) for x in ref.eval_infer(model,pairs,operation,width)]
            target=[ref.truth(*pair,width,operation) for pair in pairs]
            rows.append({"group":group,"width":width,"task":operation,
                         "pair_set_sha256":ref.digest(pairs),
                         "tested":len(pairs),"correct":sum(x==y for x,y in zip(pred,target)),
                         "predictions":pred,"truth":target})
    model.train()
    return rows

def tensor_digest(model):
    h=hashlib.sha256()
    for name,v in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def fsync_directory(path):
    fd=os.open(path,os.O_DIRECTORY|os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)

def save_atomic(directory,step,package):
    tmp=tempfile.NamedTemporaryFile(prefix=".pending-",suffix=".pt",dir=directory,delete=False)
    path=Path(tmp.name)
    try:
        with tmp:
            torch.save(package,tmp)
            tmp.flush()
            os.fsync(tmp.fileno())
        h=sha_file(path)
        final=directory/f"step{step:04d}-{h}.pt"
        # Link is atomic and fails when the target exists. No replacement of
        # a previously accepted checkpoint; the temp file is then unlinked.
        os.link(path,final)
        fsync_directory(directory)
        return final,h
    finally:
        path.unlink(missing_ok=True)

def existing(directory):
    found=[]
    for path in directory.glob("step*.pt"):
        m=re.fullmatch(r"step(\d{4})-([0-9a-f]{64})\.pt",path.name)
        require(bool(m) and path.is_file() and not path.is_symlink(),"UNSAFE_CHECKPOINT_FILE")
        step=int(m.group(1))
        require(step==len(found) or step>=0,"BAD_CHECKPOINT_STEP")
        require(sha_file(path)==m.group(2),"CORRUPTED_CHECKPOINT")
        found.append((step,path,m.group(2)))
    found.sort()
    require([k for k,_,_ in found]==list(range(len(found))),"GAPPED_OR_DUPLICATED_STEPS")
    return found

@contextlib.contextmanager
def exclusive(directory):
    directory.mkdir(parents=True,exist_ok=True)
    require(not directory.is_symlink(),"SYMLINK_RUN_ROOT")
    guard=directory/"worker.lock"
    require(not guard.is_symlink(),"SYMLINK_LOCK")
    fd=os.open(guard,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e: raise Refused("WORKER_ALREADY_RUNNING") from e
        yield
    finally:
        fcntl.flock(fd,fcntl.LOCK_UN)
        os.close(fd)

def run_case(seed,arm,run_dir,stop_at=MAX_STEPS):
    require(seed in SEEDS and arm in ARMS,"BAD_SEED_OR_ARM")
    require(0<=stop_at<=MAX_STEPS,"INVALID_BUDGET")
    require(run_dir.resolve().is_relative_to((HERE/"runs").resolve()),"CROSS_PROJECT_WRITE")
    ref,data=reference()
    status=trainer_status()  # read-only gate; never claim the live ledger.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    # Kernel-enforced process CPU ceiling; bounded step count is the primary
    # deterministic budget. No unbounded background loop or scheduler.
    soft,hard=resource.getrlimit(resource.RLIMIT_CPU)
    if hard==resource.RLIM_INFINITY or hard>=120:
        resource.setrlimit(resource.RLIMIT_CPU,(120,hard))
    run_dir.mkdir(parents=True,exist_ok=True)
    with exclusive(run_dir):
        steps=make_steps(data,seed,arm,ref)
        source_hashes={"champion":REQUIRED_CHAMPION,**SOURCE_PINS}
        case={"schema":"NC046-canary-1","seed":seed,"arm":arm,"steps":MAX_STEPS,
              "source_hashes":source_hashes,
              "source_data_sha256":SOURCE_PINS["dataset"],
              "step_plan_sha256":digest(steps),"trainer_ledger_id":status["ledger_id"],
              "scope":"isolated unapproved CPU canary, not trainer-ledger episode"}
        existing_case=run_dir/"case.json"
        if existing_case.exists():
            require(json.loads(existing_case.read_text())==case,"CASE_PIN_DRIFT")
        else:
            existing_case.write_text(json.dumps(case,sort_keys=True,indent=2)+"\n")
        snapshots=existing(run_dir)
        require(len(snapshots)<=MAX_STEPS,"OVER_BUDGET_CHECKPOINTS")
        torch.manual_seed(seed*31+7)
        random.seed(seed*31+7)
        np.random.seed(seed*31+7)
        bundle=torch.load(ref.SOURCE,map_location="cpu",weights_only=False)
        model=ref.model_start(seed,bundle).cpu().train()
        opt=torch.optim.AdamW(model.parameters(),lr=ref.P["optimizer"]["lr"],
                              weight_decay=ref.P["optimizer"]["weight_decay"])
        assert model.counts["total"]==25089
        prev=ZERO
        exposure=[]
        records=[]
        cumulative_cpu=0.0
        for i,path,h in snapshots:
            package=torch.load(path,map_location="cpu",weights_only=False)
            require(package["schema"]=="NC046-checkpoint-1","CHECKPOINT_SCHEMA")
            require(package["seed"]==seed and package["arm"]==arm and package["step"]==i,"CHECKPOINT_ID")
            require(package["case_sha256"]==digest(case),"CHECKPOINT_CASE_DRIFT")
            require(package["parent_sha256"]==prev,"BROKEN_CHECKPOINT_LINEAGE")
            require(package["model_parameters"]==25089,"WRONG_MODEL_ARCH")
            model.load_state_dict(package["model"],strict=True)
            opt.load_state_dict(package["optimizer"])
            require(tensor_digest(model)==package["model_state_sha256"],"CORRUPTED_MODEL_STATE")
            torch.set_rng_state(package["torch_rng"])
            random.setstate(package["python_rng"])
            np.random.set_state(package["numpy_rng"])
            prev=h
            exposure=package["exposure"]
            records=package["evaluations"]
            cumulative_cpu=package["cumulative_cpu_s"]
        for i in range(len(snapshots),stop_at):
            st=steps[i]
            require(time.process_time()<110,"CPU_BUDGET_EXHAUSTED")
            a,b,truth,weight=ref.tensors(st["pairs"],st["ops"],st["width"])
            t0=time.process_time()
            logits=model(a,b,st["ops"],width=st["width"])
            loss=(F.binary_cross_entropy_with_logits(logits,truth,reduction="none")*weight).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),ref.P["optimizer"]["clip_grad_norm"])
            opt.step()
            cpu=time.process_time()-t0
            cumulative_cpu+=cpu
            exposure=exposure+[{"stage":st["stage"],"width":st["width"],
                                 "op_counts":dict(__import__("collections").Counter(st["ops"])),
                                 "examples":len(st["pairs"]),"step":i,"cpu_seconds":cpu}]
            records=records+[{"step":i,"loss":float(loss.item()),
                              "heldout":evaluation(model,data,seed,ref)}]
            package={"schema":"NC046-checkpoint-1","seed":seed,"arm":arm,
                     "step":i,"parent_sha256":prev,"case_sha256":digest(case),
                     "model":model.state_dict(),"model_state_sha256":tensor_digest(model),
                     "optimizer":opt.state_dict(),"torch_rng":torch.get_rng_state(),
                     "python_rng":random.getstate(),"numpy_rng":np.random.get_state(),
                     "model_parameters":25089,"exposure":exposure,
                     "evaluations":records,"cumulative_cpu_s":cumulative_cpu,
                     "source_hashes":source_hashes}
            _,prev=save_atomic(run_dir,i,package)
        found=existing(run_dir)
        last_path=found[-1][1] if found else None
        return {"seed":seed,"arm":arm,"steps_committed":len(found),
                "checkpoint_sha256":found[-1][2] if found else None,
                "model_sha256":tensor_digest(model),
                "losses":[e["loss"] for e in records],
                "heldout":records[-1]["heldout"] if records else evaluation(model,data,seed,ref),
                "cpu_seconds":cumulative_cpu,"total_examples":sum(x["examples"] for x in exposure),
                "step_plan_sha256":case["step_plan_sha256"],
                "candidate_path":str(last_path) if last_path else None,
                "trainer_ledger_unchanged":True,"promoted":False,
                "raw_checkpoint_recovery":True,"case_path":str(existing_case)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("canary","production"),required=True)
    p.add_argument("--seed",type=int,default=11)
    p.add_argument("--arm",choices=ARMS,default="rehearsal")
    p.add_argument("--run-id",default="manual")
    p.add_argument("--stop-at",type=int,default=MAX_STEPS)
    a=p.parse_args()
    try:
        if a.mode=="production": production_guard()
        require(bool(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}",a.run_id)),"BAD_RUN_ID")
        path=HERE/"runs"/a.run_id
        result=run_case(a.seed,a.arm,path,a.stop_at)
        print(json.dumps(result,sort_keys=True))
        return 0
    except (Refused,ValueError,KeyError,AssertionError) as e:
        print(json.dumps({"status":"REFUSED","error":str(e)}))
        return 2

if __name__=="__main__":
    raise SystemExit(main())
