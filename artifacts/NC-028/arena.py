"""NC-028 bounded NO_DISCOVERY pilot, not a release authorization.

Deps: isolation.py, propose_worker.py, execute_worker.py; NC-007/NC-005/
NC-006/NC-001, frozen controller/primitive checkpoints; Python stdlib, Torch.
"""
import hashlib, json, random, resource, time
from pathlib import Path
import isolation
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SEEDS=(11,19,29)
MASK=65535
GOALS=("DOUBLE","INC","XOR_ADD_X","UNSEEN_XOR_DOUBLE_PLUS_Y")
BUDGET={"wall_seconds_per_subprocess":35,"cpu_seconds":24,"address_space_bytes":8*1024**3,
        "file_size_bytes":4*1024**2,"per_real_uid_tasks_limit":2048,
        "process_creation":"seccomp fork/vfork/clone process denied; clone3 ENOSYS","max_action_tokens":5,"max_proposals_per_goal":1,
        "heldout_pairs_per_task":8,"gpu":False,"network":False}
def hash_file(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for part in iter(lambda:f.read(1<<20),b""):h.update(part)
    return h.hexdigest()
def truth(goal,x,y):
    if goal=="DOUBLE":return (2*x)&MASK
    if goal=="INC":return (x+1)&MASK
    if goal=="XOR_ADD_X":return ((x^y)+x)&MASK
    if goal=="UNSEEN_XOR_DOUBLE_PLUS_Y":return (2*(x^y)+y)&MASK
    raise ValueError("unknown task")
def pairs(seed,goal):
    rng=random.Random(f"NC-028:{seed}:{goal}:20261106")
    return [[rng.randrange(65536),rng.randrange(65536)] for _ in range(8)]
def isolated(path,data):
    t=time.perf_counter()
    usage_before=resource.getrusage(resource.RUSAGE_CHILDREN)
    cp=isolation.run([str(HERE/path)],json.dumps(data),timeout=35)
    elapsed=time.perf_counter()-t
    usage_after=resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_seconds=((usage_after.ru_utime+usage_after.ru_stime)
                 -(usage_before.ru_utime+usage_before.ru_stime))
    if cp.returncode:
        return None,{"returncode":cp.returncode,"stderr_tail":cp.stderr[-1000:],
                     "wall_seconds":elapsed,"cpu_seconds":cpu_seconds}
    try:
        return json.loads(cp.stdout.strip().splitlines()[-1]),{"returncode":0,"wall_seconds":elapsed,"cpu_seconds":cpu_seconds,"nonprotocol_stdout_lines":max(0,len(cp.stdout.strip().splitlines())-1)}
    except (ValueError,TypeError):
        return None,{"returncode":0,"stderr_tail":"malformed worker output",
                     "wall_seconds":elapsed,"cpu_seconds":cpu_seconds}
def main():
    # Trust gate: compare actual bytes to previously independently audited,
    # Git-committed NC-007/NC-006 receipts BEFORE any pickle checkpoint load.
    prev=json.loads((HERE/"../NC-007/composition_receipt.json").read_text())
    library=json.loads((HERE/"../NC-006/library_manifest.json").read_text())
    inventory=json.loads((HERE/"../NC-000/inventory.json").read_text())
    audit=json.loads((HERE/"../NC-007/independent_receipt.json").read_text())
    if audit["status"]!="PASS" or library["independent_verifier_status"]!="PASS":
        raise RuntimeError("upstream verification not PASS")
    for tag,rel in {"compose.py":"../NC-007/compose.py",
                    "nc005_curriculum":"../NC-005/curriculum.py",
                    "nc006_accepted_library":"../NC-006/accepted_library.json",
                    "nc001_canonical_isa":"../NC-001/canonical_isa.py",
                    "nc005_checkpoint_manifest":"../NC-005/checkpoint_manifest.json"}.items():
        if hash_file(HERE/rel)!=prev["source_sha256"][tag]:
            raise RuntimeError(f"upstream source hash mismatch: {tag}")
    for rec in prev["training"]:
        checkpoint=ROOT/rec["checkpoint_path"]
        if hash_file(checkpoint)!=rec["checkpoint_sha256"]:
            raise RuntimeError("controller checkpoint hash mismatch")
    hashes={}
    for f in ("arena.py","isolation.py","propose_worker.py","execute_worker.py",
              "../NC-007/compose.py","../NC-006/accepted_library.json",
              "../NC-006/library_manifest.json","../NC-005/curriculum.py",
              "../NC-001/canonical_isa.py"):
        hashes[f]=hash_file(HERE/f)
    for seed in SEEDS:
        hashes[f"controller_seed{seed}.pt"]=hash_file(HERE/f"../NC-007/checkpoints/controller_seed{seed}.pt")
        hashes[f"primitives_seed{seed}.pt"]=hash_file(HERE/f"../NC-005/checkpoints/rehearsal_seed{seed}_stage06.pt")
    champion=ROOT.parent/"rlt_split_algorithms_ablation_v1/snapshots/current_unified_v3.pt"
    old=hash_file(champion)
    if old != inventory["champion"]["selected_bundle_sha256"] or old != library["original_champion_sha256"]:
        raise RuntimeError("unified champion changed since verified freeze")
    hashes["unified_RLT_champion"]=old
    manifest={"schema":"logos.nc028.sandbox/1","task":"NC-028",
              "status":"IMPLEMENTER_UNVERIFIED","isolation":"Landlock ABI >=5 + seccomp socket/escape deny",
              "sandbox_mode":"read-only allowlisted filesystem, denied socket syscalls; no namespace/VM",
              "limits":BUDGET,"source_sha256":hashes,
              "candidate_execution":"typed whitelisted NC-007 actions over frozen NC-006 neural primitives",
              "proposer":"NC-007 supervised controller only; no unseen goal token support",
              "permission_gates":{"external_network":False,"filesystem_mutation":False,
                                  "champion_promotion":False,"automatic_deployment":False}}
    (HERE/"sandbox_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    contract={"schema":"logos.nc028.contract/1","task":"NC-028",
              "heldout_specifications":list(GOALS),"evaluation_seed":20261106,
              "task_input":"known goal name only; never target sequence, target output or oracle label",
              "novel_family":"UNSEEN_XOR_DOUBLE_PLUS_Y: model vocabulary does not support it",
              "oracle_execution":"host computes independent arithmetic reference after proposal emission",
              "frozen_primitive_contract":"NC-006 accepted_library.json",
              "baseline_controls":["fixed_ADD_XY","uniform_random_one_step","supervised_neural_controller"],
              "budget":BUDGET,"no_new_skill_proof":True,"requires_independent_verifier":True,
              "promotion":"disabled; future NC-009 and human approval"}
    (HERE/"discovery_release_contract.json").write_text(json.dumps(contract,indent=2,sort_keys=True)+"\n")
    records=[]
    for seed in SEEDS:
        for goal in GOALS:
            pp=pairs(seed,goal)
            for arm in ("supervised_neural_controller","fixed_ADD_XY","uniform_random_one_step"):
                if arm=="supervised_neural_controller":
                    proposal,pcost=isolated("propose_worker.py",{"seed":seed,"goal":goal})
                    actions=proposal["actions"] if proposal and proposal["status"]=="PROPOSED" else []
                    pstatus=proposal["status"] if proposal else "PROCESS_FAILED"
                elif arm=="fixed_ADD_XY":
                    actions=["ADD_XY","HALT"];pstatus="HOST_BASELINE"
                    pcost={"wall_seconds":0,"cpu_seconds":0,"returncode":0}
                else:
                    rr=random.Random(f"NC028:rand:{seed}:{goal}")
                    vocabulary=["ADD_XX","ADD_X1","XOR_XY","AND_XY","SUB_XY","OR_XY","ADD_XY"]
                    actions=[rr.choice(vocabulary),"HALT"];pstatus="HOST_BASELINE"
                    pcost={"wall_seconds":0,"cpu_seconds":0,"returncode":0}
                expected=[truth(goal,x,y) for x,y in pp]
                if actions:
                    result, ecost=isolated("execute_worker.py",{"seed":seed,"actions":actions,"pairs":pp})
                else:result,ecost=None,{"returncode":None,"wall_seconds":0,"cpu_seconds":0}
                observed=result.get("outputs",[]) if result else []
                errors=[{"X":x,"Y":y,"expected":t,"observed":observed[i] if i<len(observed) else None}
                        for i,((x,y),t) in enumerate(zip(pp,expected))
                        if i>=len(observed) or observed[i]!=t]
                records.append({"seed":seed,"goal":goal,"arm":arm,"proposal_status":pstatus,
                                "model_generated":arm=="supervised_neural_controller" and bool(actions),
                                "actions":actions,"correct":len(pp)-len(errors),"tested":len(pp),
                                "failures":errors,"heldout_pairs_sha256":hashlib.sha256(json.dumps(pp).encode()).hexdigest(),
                                "proposal_cost":pcost,"execution_cost":ecost,
                                "frozen_checkpoint_sha256":result.get("frozen_sha256") if result else None,
                                "proof_circuit_ids":result.get("proof_circuit_ids") if result else [],
                                "invention":False})
    (HERE/"proposal_trace.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in records))
    totals={}
    for arm in dict.fromkeys(r["arm"] for r in records):
        rr=[r for r in records if r["arm"]==arm]
        totals[arm]={"correct":sum(r["correct"] for r in rr),
                     "tested":sum(r["tested"] for r in rr),
                     "proposals":len(rr),
                     "proposal_wall_seconds":sum(r["proposal_cost"]["wall_seconds"] for r in rr),
                     "exec_wall_seconds":sum(r["execution_cost"]["wall_seconds"] for r in rr),
                     "proposal_cpu_seconds":sum(r["proposal_cost"]["cpu_seconds"] for r in rr),
                     "exec_cpu_seconds":sum(r["execution_cost"]["cpu_seconds"] for r in rr)}
    summary={"schema":"logos.nc028.results/1","status":"NO_DISCOVERY",
             "scope":"novel goal unsupported; learned recipes are supervised known macros",
             "counts":totals,"source_sha256":hashes,
             "heldout":{g:sum(r["tested"] for r in records if r["goal"]==g and r["arm"]=="supervised_neural_controller") for g in GOALS},
             "proof_scope":"finite 16-bit numeric examples; no structural transfer, no independent verification",
             "new_discoveries":0,
             "baseline_comparability":"one proposal and same 8 numeric cases per seed/task; unequal wall CPU runtime measured; host baselines not neural",
             "independent_verdict":"PENDING"}
    (HERE/"pilot_results.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    if hash_file(champion)!=old:
        raise RuntimeError("original champion mutated")
    print(json.dumps({"status":"NO_DISCOVERY","counts":totals,"champion_unchanged":True}))
if __name__=="__main__":main()
