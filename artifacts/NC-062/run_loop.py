#!/usr/bin/env python3
"""NC-062: bounded 3-generation neural program proposal and verified library.

The model trains/infer in NC-028 Landlock+seccomp, never sees canonical
truth tables, target programs for sealed goals, or reveal seeds. The
referee/verifier + NC-061 scoreboard operate in a separate trusted parent.

Dependencies: Python3.14 stdlib, Git, NC-028/isolation.py (Landlock/libseccomp),
PyTorch CPU in sandbox, NC-001/canonical_isa.py, NC-009/operator_promotions.jsonl
and promote.py, NC-061/scoreboard.py; original champion hashes read-only.
NC-062/worker_source.py is passed as Python -c inside sandbox since NC-028
Landlock intentionally denies filesystem reads of NC-062.
"""
from __future__ import annotations
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import resource
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
NC28=ROOT/"artifacts/NC-028"
NC9=ROOT/"artifacts/NC-009"
NC61=ROOT/"artifacts/NC-061"
sys.path.insert(0,str(NC61))
import scoreboard as sb

def loadfile(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m; spec.loader.exec_module(m)
    return m

iso=loadfile("nc062_NC028_isolation",NC28/"isolation.py")
promote=loadfile("nc062_NC009_promote",NC9/"promote.py")

SEEDS=(11,19,29)
GENS=(1,2,3)
GOALS=10
DISCOVERY_GOALS=6
TRAINING_SAMPLES_PER_OPERATOR=6
CANDIDATES_PER_GOAL=4
CPU_CAP_PER_WORKER=24
WALL_CAP_PER_WORKER=35
SEED_TRAIN=923
ROOT_REPO=str(ROOT)
BUDGET={"seeds":list(SEEDS),"generations":list(GENS),
        "discovery_goals_per_seed_generation":DISCOVERY_GOALS,
        "sealed_goals_per_seed_generation":GOALS,
        "max_proposals_per_goal":CANDIDATES_PER_GOAL,
        "max_train_rows_per_worker":TRAINING_SAMPLES_PER_OPERATOR*4,
        "training_gradient_updates_per_seed_generation":8,
        "cpu_seconds_per_worker":CPU_CAP_PER_WORKER,
        "wall_seconds_per_worker":WALL_CAP_PER_WORKER,
        "source_checkpoint":"NC-062 own frozen CPU MLP; not an NC-028 pretrained model",
        "promotion":"LOCAL_VERSIONED_LIBRARY_ONLY; no NC-009 mutation"}

SOURCE=HERE/"worker_source.py"
WORKER=SOURCE.read_text()
WORKER_SHA=hashlib.sha256(WORKER.encode()).hexdigest()
CHAMPION=Path("/workspace/experiments/rlt_split_algorithms_ablation_v1/snapshots/current_unified_v3.pt")
FROZEN_FILES=(
 NC28/"isolation.py",NC9/"promote.py",NC9/"operator_promotions.jsonl",
 NC61/"scoreboard.py",ROOT/"artifacts/NC-001/canonical_isa.py",CHAMPION
)
MASK=15
# Each token is translated mechanically to the exact canonical ISA operation.
# This is compiler plumbing, not a task-specific target recipe or a claimed
# discovered algorithm. Unknown tokens fail closed.
ISA_TOKEN={
 "ADD_X1":[{"op":"LOADI","dst":"T","imm":1},
           {"op":"ADD","dst":"R","a":"X","b":"T"}],
 "ADD_XX":[{"op":"ADD","dst":"R","a":"X","b":"X"}],
 "AND1_X":[{"op":"AND1","dst":"R","src":"X"}],
 "XOR_XY":[{"op":"XOR","dst":"R","a":"X","b":"Y"}],
 "AND_XY":[{"op":"AND","dst":"R","a":"X","b":"Y"}],
 "OR_XY":[{"op":"OR","dst":"R","a":"X","b":"Y"}],
 "XOR_RX":[{"op":"XOR","dst":"R","a":"R","b":"X"}],
 "XOR_RY":[{"op":"XOR","dst":"R","a":"R","b":"Y"}],
 "ADD_RX":[{"op":"ADD","dst":"R","a":"R","b":"X"}],
 "ADD_RY":[{"op":"ADD","dst":"R","a":"R","b":"Y"}],
 "SHL_R":[{"op":"SHL","dst":"R","src":"R"}],
 "SHR_R":[{"op":"SHR","dst":"R","src":"R"}]
}
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda:fh.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()
def canonical(obj):
    return json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False)
def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()
def require(cond,why):
    if not cond:raise RuntimeError(why)
def write_once(path,obj):
    require(not path.exists(),"OUTPUT_EXISTS:"+str(path))
    path.write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n")
def compile_program(tokens):
    require(isinstance(tokens,list) and 1<=len(tokens)<=4,"ACTION_LENGTH")
    body=[]
    for token in tokens:
        require(token in ISA_TOKEN,"UNMAPPED_ACTION")
        body.extend(ISA_TOKEN[token])
    require(len(body)<=8,"EXPANSION_BUDGET")
    sb.full_program(body)
    return body

def verified_initial_library():
    rows=[json.loads(s) for s in (NC9/"operator_promotions.jsonl").read_text().splitlines()]
    promote.verify_chain(rows)
    require(len(rows)==4,"INITIAL_REGISTRY_COUNT_DRIFT")
    require(set(r["task"] for r in rows)==
            {"INC","DOUBLE","BOOLEAN_COMPOSITE","SHIFT_COMPOSITE"},"REGISTRY_SCOPE_DRIFT")
    results=[]
    for i,row in enumerate(rows):
        tokens=row["program"]
        require(tokens==promote.canonicalize(tokens),"NONCANONICAL_INITIAL")
        require(promote.canonical_key(tokens)==row["canonical_key"],"KEY_DRIFT")
        body=compile_program(tokens)
        table=sb.truth_table(body)
        for x,y in sb.DOMAIN:
            lower=promote.run(tokens,x,y)&MASK
            require(table[sb.DOMAIN.index((x,y))]==lower,"ISA_COMPILER_NOT_EQUIVALENT")
        results.append({
         "library_version":"NC062/local/0","parent_operator_id":row["operator_id"],
         "source":"independently verified NC-009 operator, not NC-062 discovery",
         "program":body,"program_sha256":digest(body),
         "training_table_key":sb.table_key(table),"verified_cases":256,
         "template_label":i,"nc009_chain_sha256":row["chain_sha256"]
        })
    require(len({r["training_table_key"] for r in results})==4,"INITIAL_DUPLICATE_FUNCTION")
    return results

def spawn(mode,seed,state,training=None,queries=None):
    payload={"mode":mode,"seed":seed,"state":state,
             "training":training or [],"queries":queries or []}
    t=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    cp=iso.run(["-c",WORKER],payload=canonical(payload),timeout=WALL_CAP_PER_WORKER,
               cpu_seconds=CPU_CAP_PER_WORKER)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
    wall=time.monotonic()-t
    require(cp.returncode==0,"SANDBOX_FAILED:"+str(cp.returncode)+":"+cp.stderr[-350:])
    lines=cp.stdout.splitlines()
    require(lines,"SANDBOX_NO_RESULT")
    result=json.loads(lines[-1])
    require(result["mode"]==mode and result["seed"]==seed,"SANDBOX_ID_MISMATCH")
    require(cpu<=CPU_CAP_PER_WORKER+1 and wall<=WALL_CAP_PER_WORKER+1,"BOUND_EXCEEDED")
    return result,{"cpu_seconds":cpu,"wall_seconds":wall,"returncode":cp.returncode,
                   "total_stdout_lines":len(lines)}

def example_set(table,rng):
    ids=rng.sample(range(256),8)
    return [{"X":sb.DOMAIN[i][0],"Y":sb.DOMAIN[i][1],"R":table[i]} for i in ids]

def train_rows(library,seed,generation):
    rng=random.Random(f"NC062/train:{seed}:{generation}")
    records=[]
    for item in library:
        table=sb.truth_table(item["program"])
        for _ in range(TRAINING_SAMPLES_PER_OPERATOR):
            records.append({"examples":example_set(table,rng),"label":item["template_label"]})
    rng.shuffle(records)
    require(len(records)==24,"TRAIN_BUDGET_DRIFT")
    return records

def candidate_from_index(library,index):
    require(type(index) is int and 0<=index<len(library),"BAD_MODEL_INDEX")
    return library[index]["program"]

def proof(goal,program):
    got=sb.truth_table(program)
    expected=goal["table"]
    if got==expected:return {"verdict":"VERIFIED","tested":256,"proof_sha256":digest(got)}
    for idx,(x,y) in enumerate(sb.DOMAIN):
        if got[idx]!=expected[idx]:
            return {"verdict":"COUNTEREXAMPLE","tested":256,
                    "x":x,"y":y,"want":expected[idx],"got":got[idx],
                    "counterexample_index":idx}
    raise RuntimeError("IMPOSSIBLE_MISMATCH")

def decide_discovery(goals,choices,library,generation,seed):
    out=[];accepted=0;new=0
    known={r["training_table_key"] for r in library}
    for j,(goal,rankings) in enumerate(zip(goals,choices)):
        require(len(rankings)==4 and sorted(rankings)==list(range(4)),"MODEL_RANKING_INVALID")
        attempts=[]
        conflict=[]
        for rank,idx in enumerate(rankings):
            program=candidate_from_index(library,idx)
            # A model-ranked alternative is rejected if it cannot satisfy an
            # actual prior counterexample. This is rejection feedback, NOT
            # gradient-based CEGIS or invented program synthesis.
            table=sb.truth_table(program)
            if any(table[i]!=v for i,v in conflict):
                attempts.append({"rank":rank,"library_index":idx,"verdict":"REJECTED_BY_FEEDBACK"})
                continue
            result=proof(goal,program)
            attempts.append({"rank":rank,"library_index":idx,
                             "program_sha256":digest(program),**result})
            if result["verdict"]=="VERIFIED":
                accepted+=1
                key=sb.table_key(goal["table"])
                if key not in known:
                    # A genuinely new independently verified model-emitted
                    # program could be registered here. Our four-class model
                    # emits only the initial templates; never fabricate one.
                    raise RuntimeError("UNEXPECTED_NEW_PROGRAM_IN_FIXED_LIBRARY")
                break
            conflict.append((result["counterexample_index"],result["want"]))
        out.append({"generation":generation,"seed":seed,"goal_index":j,
                    "goal_fingerprint":sb.table_key(goal["table"]),
                    "model_emitted_ranked_class_ids":rankings,
                    "attempts":attempts,"train_only":True,
                    "received_full_goal_table":False,
                    "accepted":any(a["verdict"]=="VERIFIED" for a in attempts),
                    "new_discovery":False})
    return out,accepted,new

def score_cached(goals,choices,library,training_keys,proposal_cpu):
    require(len(choices)==len(goals),"SCORE_COUNT")
    # Candidate only sees 8 public examples, never answers or sealed table.
    cache={canonical(goal["examples"]):candidate_from_index(library,order[0])
           for goal,order in zip(goals,choices)}
    def frozen_candidate(examples,budget):
        return cache[canonical(examples)],1
    report=sb.score_candidate(frozen_candidate,goals,1,
                              training_table_keys=training_keys,retention=1.0)
    # NC-061 measures in-process candidate calls; here model inference was
    # sandboxed in another process. Replace CPU score with externally measured
    # RUSAGE_CHILDREN process CPU, conservatively and transparently.
    report["score_candidate_cache_cpu_seconds"]=report["cpu_seconds"]
    report["cpu_seconds"]=round(proposal_cpu,6)
    report["cpu_seconds_per_solve"]=round(proposal_cpu/report["solved"],6) if report["solved"] else None
    report["cpu_provenance"]="RUSAGE_CHILDREN around NC-028 sandboxed inference worker"
    report["actual_inference_worker_calls"]=1
    report["retention_provenance"]="controlled fixed original library; not independent old-skill neural retention"
    return report

def run():
    require(not (HERE/"generation_trace.jsonl").exists(),"TRACE_ALREADY_EXISTS")
    require(not (HERE/"scoreboard_results.json").exists(),"SCORE_ALREADY_EXISTS")
    source_hashes={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha(p) for p in FROZEN_FILES}
    initial_library=verified_initial_library()
    library_keys=[r["training_table_key"] for r in initial_library]
    thresholds={**sb.THRESHOLDS,"goals":GOALS}
    # Precommit all independent sealed evaluations before creating/freeze
    # successor checkpoints. Do not write private seeds to any candidate IPC.
    private={}
    public=[]
    for gen in GENS:
        for seed in SEEDS:
            commit,reveal=sb.new_commitment(thresholds)
            private[(gen,seed)]=reveal
            public.append({"generation":gen,"seed":seed,"commitment":commit})
    protocol={"schema":"nc062-experiment-preregister/1","agent":"turn-cb84af5cc3054e79af7776e1ac0af3c8",
              "limits":BUDGET,"evaluation_thresholds":thresholds,
              "scoreboard_source_sha256":sha(NC61/"scoreboard.py"),
              "reference_dependencies":source_hashes,
              "sandbox_worker_source_sha256":WORKER_SHA,
              "scope":"4bit finite known-program library; model-generated class selection, not new algorithm invention",
              "heldout_reveal_protocol":"public hashes written before candidate checkpoint freeze; secrets remain parent memory, never sent to model worker; revealed to evidence only after scoring",
              "risk":"no separate uid; parent is trusted; no leakage if parent violates control; no internet/service"}
    write_once(HERE/"PROTOCOL.json",protocol)
    write_once(HERE/"commitments.json",{"schema":"NC062/sealed-precommit/1","public":public})
    lib_history=[{"version":0,"entries":initial_library,"chain_sha256":digest(initial_library)}]
    (HERE/"generation_trace.jsonl").touch(exist_ok=False)
    snapshots={}
    initial_states={}
    before_hash=sha(CHAMPION)
    created=time.monotonic()
    scores=[]
    comparison=[]
    independent_discoveries=0
    all_reveals=[]
    total_calls=0
    for seed in SEEDS:
        init,charge=spawn("init",seed,None)
        total_calls+=1
        init_rec={"schema":"nc062-checkpoint/1","generation":0,"seed":seed,
                  "parent_state_sha256":None,
                  "model_state":init["model_state"],"model_state_sha256":init["model_sha256_after"],
                  "worker_source_sha256":WORKER_SHA,"gradient_updates":0}
        loc=HERE/f"checkpoint_g0_seed{seed}.json"
        write_once(loc,init_rec)
        initial_states[seed]=init_rec
        snapshots[(0,seed)]=init_rec
    for generation in GENS:
        for seed in SEEDS:
            parent=snapshots[(generation-1,seed)]
            # Train *only* on existing verified pre-seal library; never
            # on sealed evaluation goals or their verifier decisions.
            train=train_rows(initial_library,seed,generation)
            train_sha=digest(train)
            trained,charge=spawn("train",seed,parent["model_state"],training=train)
            total_calls+=1
            require(trained["gradient_updates"]==8,"TRAINING_MISSING")
            require(trained["model_sha256_before"]==parent["model_state_sha256"],"STATE_LINEAGE_DRIFT")
            checkpoint={"schema":"nc062-checkpoint/1","generation":generation,"seed":seed,
                        "parent_state_sha256":parent["model_state_sha256"],
                        "model_state":trained["model_state"],
                        "model_state_sha256":trained["model_sha256_after"],
                        "gradient_updates":8,"train_input_sha256":train_sha,
                        "library_version":lib_history[-1]["version"],
                        "training_cpu_seconds":charge["cpu_seconds"],
                        "worker_source_sha256":WORKER_SHA}
            path=HERE/f"checkpoint_g{generation}_seed{seed}.json"
            write_once(path,checkpoint)
            snapshots[(generation,seed)]=checkpoint
            # Training/discovery goals are disjoint from sealed evaluation
            # by a separately seeded PRNG, never reused as sealed goals.
            discovery_reveal={"generator":sb.GENERATOR,
                              "seed":f"NC062:discovery:{generation}:{seed}",
                              "nonce":"public-training-only",
                              "thresholds":{**thresholds,"goals":DISCOVERY_GOALS}}
            discovery=sb.sealed_goals(discovery_reveal)
            public_examples=[g["examples"] for g in discovery]
            discovered,dcharge=spawn("infer",seed,checkpoint["model_state"],queries=public_examples)
            total_calls+=1
            require(discovered["model_sha256_before"]==checkpoint["model_state_sha256"],"INFERENCE_WEIGHTS_DRIFT")
            trace,accepted,new=decide_discovery(discovery,discovered["choices"],
                                                initial_library,generation,seed)
            independent_discoveries+=new
            with (HERE/"generation_trace.jsonl").open("a") as out:
                for row in trace:out.write(canonical(row)+"\n")
            # Materialize versioned local library with hash of exact preceding
            # entries; no new rows when all discoveries are duplicates/failures.
            prior=lib_history[-1]
            next_ver={"version":len(lib_history),"entries":list(initial_library),
                      "parent_chain_sha256":prior["chain_sha256"],
                      "added_verified_programs":new,
                      "chain_sha256":digest({"previous":prior["chain_sha256"],"entries":initial_library,
                                             "new":new,"generation":generation,"seed":seed})}
            lib_history.append(next_ver)
            # Candidate checkpoint frozen on disk before reveal.
            reveal=private[(generation,seed)]
            pub=next(x["commitment"] for x in public if x["generation"]==generation and x["seed"]==seed)
            sb.verify_reveal(pub,reveal)
            sealed=sb.sealed_goals(reveal)
            queries=[g["examples"] for g in sealed]
            choice,ec=spawn("infer",seed,checkpoint["model_state"],queries=queries)
            ref,bc=spawn("infer",seed,initial_states[seed]["model_state"],queries=queries)
            total_calls+=2
            current=score_cached(sealed,choice["choices"],initial_library,library_keys,ec["cpu_seconds"])
            baseline=score_cached(sealed,ref["choices"],initial_library,library_keys,bc["cpu_seconds"])
            if any(sb.table_key(x["table"]) in library_keys for x in sealed):
                require(current["training_overlap"]>0,"MISSED_CONTAMINATION")
            scores.append({"generation":generation,"seed":seed,
                          "candidate_model_sha256":checkpoint["model_state_sha256"],
                          "initial_model_sha256":initial_states[seed]["model_state_sha256"],
                          "sealed_commitment_sha256":pub["sha256"],
                          "private_reveal_sha256":sb.digest(reveal),
                          "sealed_goal_count":GOALS,
                          "training_goal_keys":[sb.table_key(g["table"]) for g in discovery],
                          "evaluation_goal_keys":[sb.table_key(g["table"]) for g in sealed],
                          "train_score":current,"no_library_training_ablation":baseline,
                          "discovery_accepted_duplicates":accepted,
                          "new_discoveries":new,
                          "train_worker_cost":charge,"discovery_worker_cost":dcharge,
                          "score_worker_cost":ec,"ablation_worker_cost":bc})
            # Reveals published to reviewer only AFTER evaluation and after
            # freezing candidate. This parent never sends them to workers.
            all_reveals.append({"generation":generation,"seed":seed,"reveal":reveal})
    for generation in GENS:
        per=[x for x in scores if x["generation"]==generation]
        baseline=[x["no_library_training_ablation"] for x in per]
        candidate=[x["train_score"] for x in per]
        result=sb.compare(baseline,candidate,thresholds)
        comparison.append({"generation":generation,"scoreboard":result})
    comp=sb.compounding([x["scoreboard"].get("gain",0) for x in comparison],
                        [x["scoreboard"]["verdict"] for x in comparison],thresholds)
    require(sha(CHAMPION)==before_hash,"CHAMPION_MODIFIED")
    for rel,h in source_hashes.items():
        pp=ROOT/rel if not rel.startswith("/") else Path(rel)
        require(sha(pp)==h,"SOURCE_PIN_DRIFT:"+rel)
    require(all(x["train_score"]["total"]==GOALS for x in scores),"SCORE_DENOMINATOR")
    require(len(scores)==9 and len(lib_history)==10,"GENERATION_COUNT")
    write_once(HERE/"library_versions.json",{"schema":"NC062/local-verified-library/1",
                                          "versions":lib_history,"mutated_global_registry":False})
    write_once(HERE/"sealed_reveals.json",{"schema":"NC062/private-reveal-post-evaluation/1",
                                        "releases":all_reveals})
    result={"schema":"NC062/three-generation-evidence/1","task":"NC-062",
            "status":"NO_DISCOVERY" if independent_discoveries==0 else "REVIEW_NOVEL_CANDIDATES",
            "independent_discovery_count":independent_discoveries,
            "neural_model":"24-feature small CPU MLP 24->24->4, trained on four previously verified operators",
            "one_frozen_start_per_seed":True,
            "run_scientific_verdict":"NO_IMPROVEMENT",
            "scoreboard_comparisons":comparison,
            "scoreboard_compounding":comp,
            "per_seed_generation_scores":scores,
            "completed_generations":3,
            "evaluated_seeds":list(SEEDS),
            "discovery_goals_total":len(GENS)*len(SEEDS)*DISCOVERY_GOALS,
            "sealed_eval_goals_total":len(scores)*GOALS,
            "sealed_precommit_sha256":sha(HERE/"commitments.json"),
            "reveal_release_sha256":sha(HERE/"sealed_reveals.json"),
            "model_checkpoint_count":len(snapshots),
            "model_subprocess_calls":total_calls,
            "champion_sha256_unchanged":before_hash,
            "source_hashes_unchanged":True,
            "global_NC009_registry_unchanged":True,
            "wall_seconds":time.monotonic()-created,
            "failures_or_nonclaims":[
                "The model emits only four known NC-009 algorithms; no genuinely new operator learned.",
                "NC-009 canonicalization/dedup carried into local 4-bit library; no writes to global NC-009.",
                "Independent verifier must audit, not self-PASS.",
                "Local input-output examples can collide with known functions; scoreboard reports contamination when declared.",
                "Coded decoder/adapter is host infrastructure; proposals are model-selected existing-program classes.",
                "The model never infers unseen general-purpose instructions, and this result is not AGI.",
                "Retention is fixed library preservation, not independently tested neural old-skill retention.",
                "NC-028 sandbox protects subprocess but parent/referee shares UID; no separate cryptographic secrecy.",
                "Subprocess measured CPU, budget cap per worker, no aggregate budget equivalence guarantee.",
                "Generation isolation is an honest bounded canary, not a production worker/service."
            ]}
    write_once(HERE/"scoreboard_results.json",result)
    print(json.dumps({"status":result["status"],"verdicts":comparison,
                      "compounding":comp["verdict"],"checkpoints":len(snapshots),
                      "sealed_goals":result["sealed_eval_goals_total"],
                      "discovery_goals":result["discovery_goals_total"],
                      "new_discoveries":independent_discoveries,"worker_calls":total_calls,
                      "wall_seconds":result["wall_seconds"]},sort_keys=True))
if __name__=="__main__":run()
