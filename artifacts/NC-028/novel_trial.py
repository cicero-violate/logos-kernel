"""Model-only unseen-goal proposal and negative-feedback benchmark (NO_DISCOVERY).

Dependencies: local isolation.py, arena.py, lexical_model.py and pinned
lexical_controller.pt, execute_worker.py; downstream frozen NC-006/NC-007
neural primitives, PyTorch, NumPy, Linux Landlock/seccomp.
"""
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import time

import arena
import isolation

HERE = Path(__file__).resolve().parent
GOAL = "UNSEEN_XOR_DOUBLE_PLUS_Y"
# Preregistered CPU-matching schedule: two neural proposal+execution
# calls cost approximately four execution-only host baseline calls.
# Neither test labels nor observed success affect this schedule.
ATTEMPTS_PER_ARM = {"lexical_neural": 2, "lexical_frozen": 2,
                    "fixed_host": 4, "random_host": 4}
MAX_ATTEMPTS = max(ATTEMPTS_PER_ARM.values())
MATCH_TOLERANCE = 0.15
ARM_CPU_SECONDS = 12.0
ARM_WALL_SECONDS = 20.0
ARMS = tuple(ATTEMPTS_PER_ARM)


def sandbox(name, payload, remaining_cpu, remaining_wall):
    cpu_seconds = min(24, math.floor(remaining_cpu))
    if cpu_seconds < 1 or remaining_wall <= 1.0:
        return None, {"status": "BUDGET_EXHAUSTED", "cpu_seconds": 0, "wall_seconds": 0}
    start = time.perf_counter()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        cp = isolation.run([str(HERE/name)], json.dumps(payload),
                           cpu_seconds=cpu_seconds,
                           timeout=min(35, remaining_wall))
    except Exception as exc:
        cp = None
        error = f"{type(exc).__name__}: {exc}"
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cost = {"cpu_seconds": (after.ru_utime+after.ru_stime) -
                           (before.ru_utime+before.ru_stime),
            "wall_seconds": time.perf_counter()-start}
    if cp is None:
        return None, {**cost, "status": "FAILED", "error": error}
    cost["returncode"] = cp.returncode
    if cp.returncode:
        return None, {**cost, "status": "FAILED", "stderr_tail": cp.stderr[-500:]}
    try:
        result = json.loads(cp.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None, {**cost, "status": "BAD_JSON"}
    return result, {**cost, "status": "OK"}


def evaluate(actions, seed, probe, heldout, cpu_remaining, wall_remaining):
    result, cost = sandbox("execute_worker.py", {
        "seed": seed, "actions": actions, "pairs": probe + heldout
    }, cpu_remaining, wall_remaining)
    values = result["outputs"] if result else []
    expected = [arena.truth(GOAL, x, y) for x,y in probe+heldout]
    counts = {
        "feedback_correct": sum(v == t for v,t in zip(values[:8], expected[:8])),
        "heldout_correct": sum(v == t for v,t in zip(values[8:16], expected[8:16])),
        "feedback_tested": 8, "heldout_tested": 8
    }
    if len(values) != 16:
        counts["feedback_correct"] = counts["heldout_correct"] = 0
    counterexamples = [
        {"pair_index": i, "x": x, "y": y, "expected": expected[i],
         "observed": values[i]}
        for i, (x,y) in enumerate(probe)
        if len(values) == 16 and values[i] != expected[i]
    ]
    return counts, cost, counterexamples


def trial(seed=11):
    probe = arena.pairs(seed, GOAL)
    heldout = arena.pairs(29, GOAL)
    assert probe != heldout and not {tuple(p) for p in probe}.intersection(map(tuple, heldout))
    checkpoint = HERE/"lexical_controller.pt"
    digest = arena.hash_file(checkpoint)
    meta = __import__("torch").load(checkpoint, map_location="cpu", weights_only=True)
    if GOAL in meta["training_goals"] or len(meta["training_goals"]) != 11:
        raise RuntimeError("unseen goal leaked into training")
    source_receipt = HERE.parent/"NC-007"/"composition_receipt.json"
    if arena.hash_file(source_receipt) != meta["source_receipt_sha256"]:
        raise RuntimeError("training source changed")
    champion = arena.ROOT.parent/"rlt_split_algorithms_ablation_v1/snapshots/current_unified_v3.pt"
    original = arena.hash_file(champion)
    rows = []
    consumed = {}
    for arm in ARMS:
        cpu, wall = 0.0, 0.0
        rejected = []
        for index in range(ATTEMPTS_PER_ARM[arm]):
            if ARM_CPU_SECONDS-cpu < 3 or ARM_WALL_SECONDS-wall < 4:
                rows.append({"arm":arm,"attempt":index+1,"status":"BUDGET_EXHAUSTED"})
                break
            if arm in ("lexical_neural", "lexical_frozen"):
                proposal, pcost = sandbox(
                    "lexical_model.py",
                    {"goal":GOAL, "checkpoint_sha256":digest,
                     "feedback":rejected if arm == "lexical_neural" else []},
                    ARM_CPU_SECONDS-cpu, ARM_WALL_SECONDS-wall)
                cpu += pcost["cpu_seconds"]
                wall += pcost["wall_seconds"]
                actions = proposal.get("actions",[]) if proposal else []
                model_generated = bool(proposal and proposal.get("model_generated"))
                proposal_status = proposal.get("status","FAILED") if proposal else "FAILED"
            elif arm == "fixed_host":
                actions = ([["ADD_XY","HALT"], ["ADD_XX","HALT"],
                            ["XOR_XY","HALT"], ["AND_XY","HALT"]][index])
                pcost={"cpu_seconds":0.0,"wall_seconds":0.0,"status":"HOST"}
                model_generated=False
                proposal_status="HOST_BASELINE"
            else:
                rnd=random.Random(f"novel:rand:{seed}:{index}")
                actions=[rnd.choice(["ADD_XX","ADD_X1","XOR_XY","AND_XY","ADD_XY","SUB_XY"]),
                         "HALT"]
                pcost={"cpu_seconds":0.0,"wall_seconds":0.0,"status":"HOST"}
                model_generated=False
                proposal_status="HOST_BASELINE"
            if not actions:
                rows.append({"arm":arm,"attempt":index+1,
                             "status":"CANNOT_PROPOSE","proposal_cost":pcost})
                break
            counts, ecost, counterexamples = evaluate(actions,seed,probe,heldout,
                                     ARM_CPU_SECONDS-cpu,ARM_WALL_SECONDS-wall)
            cpu += ecost["cpu_seconds"]
            wall += ecost["wall_seconds"]
            rows.append({"arm":arm,"attempt":index+1,
                         "status":"TESTED" if ecost["status"] == "OK" else "EXECUTION_FAILED",
                         "actions":actions,"model_generated":model_generated,
                         "proposal_status":proposal_status,
                         "feedback_method": ("numeric-counterexample-rejection-gradient"
                                              if arm=="lexical_neural" and rejected else "none"),
                         "feedback_counterexamples":counterexamples,
                         "proposal_cost":pcost,"execution_cost":ecost,
                         "aggregate_cpu_seconds":cpu,"aggregate_wall_seconds":wall,
                         **counts,
                         "policy_weights_before": proposal.get("weights_before") if arm in ("lexical_neural","lexical_frozen") else None,
                         "policy_weights_after": proposal.get("weights_after") if arm in ("lexical_neural","lexical_frozen") else None})
            if (arm == "lexical_neural" and ecost["status"] == "OK"
                    and counterexamples):
                # Only the FIRST real feedback-split numeric failure is sent.
                # Heldout labels and target recipes never reach the proposer.
                rejected.append({"actions":actions,"correct":counts["feedback_correct"],
                                 "tested":counts["feedback_tested"],
                                 "counterexample":counterexamples[0]})
        # Charge failures and interrupted attempts too, not only tested rows.
        consumed[arm] = {"cpu_seconds":cpu, "wall_seconds":wall}
        if cpu > ARM_CPU_SECONDS+0.05 or wall > ARM_WALL_SECONDS+0.05:
            raise RuntimeError(f"allocated resource budget exceeded by {arm}: {cpu}/{wall}")
    if arena.hash_file(champion) != original:
        raise RuntimeError("champion modified")
    checks = {}
    for arm in ARMS:
        rr=[row for row in rows if row["arm"]==arm and row["status"]=="TESTED"]
        checks[arm]={"heldout_correct":sum(r["heldout_correct"] for r in rr),
                     "heldout_tested":sum(r["heldout_tested"] for r in rr),
                     "feedback_correct":sum(r["feedback_correct"] for r in rr),
                     "feedback_tested":sum(r["feedback_tested"] for r in rr),
                     "cpu_seconds":consumed[arm]["cpu_seconds"],
                     "wall_seconds":consumed[arm]["wall_seconds"],
                     "attempts":len(rr),
                     "best_by_feedback": next((
                         {"attempt":r["attempt"],"feedback_correct":r["feedback_correct"],
                          "heldout_correct":r["heldout_correct"],"heldout_tested":r["heldout_tested"]}
                         for r in sorted(rr, key=lambda r:(-r["feedback_correct"],r["attempt"]))
                     ), None)}
    reference_cpu=checks["lexical_neural"]["cpu_seconds"]
    reference_wall=checks["lexical_neural"]["wall_seconds"]
    if reference_cpu <= 0 or reference_wall <= 0:
        raise RuntimeError("no measured neural budget")
    comparability={
        arm: {"cpu_ratio_to_neural": checks[arm]["cpu_seconds"]/reference_cpu,
              "wall_ratio_to_neural": checks[arm]["wall_seconds"]/reference_wall}
        for arm in ARMS
    }
    cpu_matched=all(abs(v["cpu_ratio_to_neural"]-1) <= MATCH_TOLERANCE
                    for v in comparability.values())
    wall_matched=all(abs(v["wall_ratio_to_neural"]-1) <= MATCH_TOLERANCE
                     for v in comparability.values())
    receipt={"schema":"logos.nc028.unseen-feedback-benchmark/2",
             "status":"NO_DISCOVERY",
             "independent_verdict":"PENDING",
             "goal":GOAL, "train_label_leakage":False,
             "training_style":"supervised known macro labels, then OOD goal-name proposal",
             "novel_program_success":False,
             "feedback_method":"real numeric feedback mismatch scales rejection-only gradient; not full CEGIS synthesis",
             "scoring":"separate 8-example feedback and untouched 8-example heldout sets",
             "cpu_budget_per_arm":ARM_CPU_SECONDS,
             "wall_budget_per_arm":ARM_WALL_SECONDS,
             "max_attempts_per_arm":ATTEMPTS_PER_ARM,
             "cpu_matching_tolerance_fraction":MATCH_TOLERANCE,
             "cpu_matched":cpu_matched,"wall_matched":wall_matched,
             "compute_ratios":comparability,
             "comparability":("MEASURED CPU/WALL MATCHED WITHIN 15 PERCENT; "
                              "BASELINE ATTEMPT COUNTS DIFFER; NO SUPERIORITY CLAIM"
                              if cpu_matched and wall_matched else
                              "NOT COMPUTE MATCHED; NO PERFORMANCE COMPARISON"),
             "seed":seed,
             "feedback_pairs_sha256":hashlib.sha256(json.dumps(probe).encode()).hexdigest(),
             "heldout_pairs_sha256":hashlib.sha256(json.dumps(heldout).encode()).hexdigest(),
             "checkpoint_sha256":digest,
             "champion_sha256_unchanged":original,
             "source_sha256":{name:arena.hash_file(HERE/name) for name in
                    ("lexical_model.py","train_lexical.py","novel_trial.py","isolation.py","execute_worker.py")},
             "arm_totals":checks,"attempts":rows}
    receipt["novel_program_success"] = any(
        r["arm"]=="lexical_neural" and r["status"]=="TESTED"
        and r["heldout_correct"]==r["heldout_tested"]
        for r in rows)
    # A successful observed sample is not a proof of novel operator discovery.
    receipt["new_discoveries"]=0
    dest=HERE/"novel_feedback_results.json"
    dest.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    return receipt


if __name__=="__main__":
    result=trial()
    print(json.dumps({"status":result["status"],"counts":result["arm_totals"],
                      "heldout_program_proposed":any(row.get("model_generated") for row in result["attempts"] if row["arm"]=="lexical_neural"),
                      "feedback_weight_changed":any(row.get("policy_weights_before") != row.get("policy_weights_after")
                                                   for row in result["attempts"] if row["arm"]=="lexical_neural" and row.get("feedback_method")=="numeric-counterexample-rejection-gradient")},
                     sort_keys=True))
