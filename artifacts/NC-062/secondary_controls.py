"""NC-062 fixed, seeded random and bounded symbolic controls on the *same*
uncontaminated v2 sealed goals.

Honest limitation: physical CPU not matched to neural PyTorch subprocess.
No control is reported as demonstrating compute-matched superiority.
Dependencies: Python 3 stdlib, run_loop.py, NC-061 scoreboard, NC-009 and
NC-001; reads eval_v2_reveals.json only after candidate freeze/scoring.
"""
import json
import random
import time
import run_loop as r

def run():
    path=r.HERE/"secondary_controls.json"
    r.require(not path.exists(),"NO_OVERWRITE")
    eval_rows=json.loads((r.HERE/"eval_v2_results.json").read_text())["score_rows"]
    reveals=json.loads((r.HERE/"eval_v2_reveals.json").read_text())["released"]
    lib=r.verified_initial_library()
    training={x["training_table_key"] for x in lib}
    reports=[]
    for row in eval_rows:
        gen,seed=row["generation"],row["seed"]
        release=next(x["reveal"] for x in reveals if x["generation"]==gen and x["seed"]==seed)
        pool=r.sb.sealed_goals(release)
        goals=[pool[i] for i in row["selected_pool_indices"]]
        assert all(r.sb.table_key(g["table"]) not in training for g in goals)
        rng=random.Random(f"nc062/seeded-random-control:{gen}:{seed}")
        def constant(examples,budget):
            return [{"op":"LOADI","dst":"R","imm":0}],1
        def fixed(examples,budget):
            return lib[0]["program"],1
        def random_control(examples,budget):
            return lib[rng.randrange(4)]["program"],1
        def symbolic(examples,budget):
            for i,reg in enumerate(lib):
                candidate=reg["program"]
                # Only disclosed examples, not withheld goal function, decide.
                if all(r.sb.isa.reference(r.sb.full_program(candidate),
                        {"X":e["X"],"Y":e["Y"],"R":0,"T":0},
                        max_steps=len(candidate)+2)["registers"]["R"]==e["R"]
                        for e in examples):
                    return candidate,i+1
            return None,4
        for name,fn,budget in [
            ("constant_output",constant,1),
            ("fixed_verified_library",fixed,1),
            ("seeded_random_verified_library",random_control,1),
            ("symbolic_fit_public_examples_max4",symbolic,4)]:
            start=time.process_time()
            out=r.sb.score_candidate(fn,goals,budget,
                  training_table_keys=training,retention=1.0)
            elapsed=time.process_time()-start
            reports.append({"generation":gen,"seed":seed,"control":name,
                            "score":out,
                            "elapsed_process_cpu_seconds":elapsed,
                            "candidate_count_budget_per_goal":budget,
                            "cpu_matched_to_neural":False,
                            "comparability":"DIAGNOSTIC_ONLY_NOT_CPU_MATCHED"})
    output={"schema":"NC062/secondary-controls-v1","task":"NC-062",
            "rows":reports,
            "all_same_90_clean_goals":True,
            "seeded_random_has_fixed_seed":True,
            "source":"finite, deterministic host baselines, NOT neural generation",
            "limitations":["Fixed/random/symbolic host controls have unlike physical CPU costs.",
                           "Symbolic control has max four proposal candidates per goal, not one.",
                           "Do not report performance superiority from these controls.",
                           "No labels from sealed programs are passed to candidate model."]}
    r.write_once(path,output)
    print(json.dumps({"control_rows":len(reports),"fixed_solved":sum(x["score"]["solved"] for x in reports if x["control"]=="fixed_verified_library"),
       "symbolic_solved":sum(x["score"]["solved"] for x in reports if x["control"]=="symbolic_fit_public_examples_max4"),
       "sha256":r.sha(path),"cpu_matched":False},sort_keys=True))
if __name__=="__main__":run()
