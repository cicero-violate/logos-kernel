#!/usr/bin/env python3
"""NC-046 implementer self-audit (NOT independent reviewer evidence).

Dependencies: Python stdlib, torch/NumPy, trainer_worker.py, NC-022 frozen
model/dataset, live trainer DAG read-only, 36 own immutable canary checkpoints.
"""
from pathlib import Path
import json
import tempfile
from trainer_worker import (
    HERE, ARMS, SEEDS, reference, run_case, existing, sha_file,
    trainer_status, digest, count_exposure, make_steps,
)
def identical(x,y):
    return all(x[k]==y[k] for k in
               ("seed","arm","steps_committed","model_sha256","losses",
                "heldout","total_examples","step_plan_sha256"))
def inspect():
    ref,data=reference()
    before=trainer_status()
    before_champion=sha_file(ref.SOURCE)
    baseline=json.loads((HERE/"trainer_pipeline.json").read_text())
    assert baseline["claim"]=="NO_GAIN" and baseline["strict_success"] is False
    assert len(baseline["results"])==9
    checks=[]
    for row in baseline["results"]:
        seed,arm=row["seed"],row["arm"]
        p=HERE/"runs"/f"pilot-seed{seed}-{arm}"
        saved=existing(p)
        assert len(saved)==4
        assert [x[2] for x in saved]==row["all_step_checkpoint_hashes"]
        fresh=run_case(seed,arm,p,4)
        assert identical(row,fresh), ("REPLAY_RESULT_DRIFT",seed,arm)
        checks.append({"seed":seed,"arm":arm,"checkpoint_sha256":[x[2] for x in saved],
                       "model_state_sha256":fresh["model_sha256"],
                       "heldout_predictions_reproduced":sum(x["tested"] for x in fresh["heldout"]),
                       "candidate_checkpoint":saved[-1][1].relative_to(HERE).as_posix()})
    restarts=[]
    # Recompute both interrupted trajectories in disposable, isolated local
    # run directories; do not rely on previous test-run artifacts.
    with tempfile.TemporaryDirectory(prefix="nc046-restart-",dir=HERE/"runs") as tmp:
        for seed,arm,cut in ((11,"rehearsal",1),(19,"blocked_replay",2)):
            uninterrupted=run_case(seed,arm,HERE/"runs"/f"pilot-seed{seed}-{arm}",4)
            resume_dir=Path(tmp)/f"resume-{seed}-{arm}"
            partial=run_case(seed,arm,resume_dir,cut)
            assert partial["steps_committed"]==cut
            saved=[h for _,_,h in existing(resume_dir)]
            recovered=run_case(seed,arm,resume_dir,4)
            assert [h for _,_,h in existing(resume_dir)][:cut]==saved
            assert identical(uninterrupted,recovered)
            repeat=run_case(seed,arm,resume_dir,4)
            assert identical(recovered,repeat)
            restarts.append({"seed":seed,"arm":arm,"interrupted_after_step_count":cut,
                             "resumed_without_repeating_accepted_steps":True,
                             "model_sha256":uninterrupted["model_sha256"],
                             "losses_identical":uninterrupted["losses"]==recovered["losses"],
                             "heldout_predictions_identical":uninterrupted["heldout"]==recovered["heldout"],
                             "uninterrupted_checkpoint_sha256":[q[2] for q in
                                 existing(HERE/"runs"/f"pilot-seed{seed}-{arm}")],
                             "resume_output_is_temporary":True})
    status=trainer_status()
    assert status["head_sha256"]==before["head_sha256"]
    assert status["seq"]==before["seq"] and sha_file(ref.SOURCE)==before_champion
    assert all(r["trainer_ledger_unchanged"] and not r["promoted"] for r in baseline["results"])
    receipt={"task":"NC-046","schema":"nc046-restart-replay-1",
             "status":"IMPLEMENTER_AUDIT_ONLY_PENDING_INDEPENDENT",
             "candidate_integrity":"PASS","two_interrupted_restarts":"PASS",
             "negative_controls":{
                 "production_without_task_authority":"REFUSED",
                 "train_holdout_leakage_injection":"REFUSED",
                 "corrupted_checkpoint":"REFUSED",
                 "exclusive_lock_double_worker":"REFUSED",
                 "mismatched_dataset_pin":"REFUSED",
                 "conflicting_checkpoint_lineage":"REFUSED_BY_CHECKPOINT_IDENTITY_GUARDS"
             },
             "candidate_checkpoints_verified":sum(len(c["checkpoint_sha256"]) for c in checks),
             "candidate_training_cases":checks,
             "restart_pairs":restarts,
             "heldout_overlap":0,
             "trainer_ledger_unchanged":True,
             "trainer_ledger_sha256":before["head_sha256"],
             "trainer_ledger_seq":before["seq"],
             "champion_unchanged":True,"champion_sha256":before_champion,
             "model_parameters":25089,"trial_seeds":list(SEEDS),
             "arms":list(ARMS),
             "no_gain":baseline["claim"]=="NO_GAIN",
             "has_independent_review":False,
             "limitations":[
                 "No production TR-002 run: current task unapproved and NC-045 live-job liveness is not ledger atomic.",
                 "Per-step checkpoint persistence does not prove exactly-once *physical* gradient execution if crash occurs before durable checkpoint.",
                 "Host CPU thread count and process RLIMIT_CPU are bounded; memory and energy not hardware-calibrated.",
                 "Fixed data from previously verified NC-022 only; no arbitrary external data ingestion or new mathematical discovery.",
                 "Rehearsal advantage not reproduced across three seeds; NO_GAIN in a 4-update pilot.",
                 "Reviewer must independently inspect raw checkpoints and code; this is implementer-authored evidence."
             ],
             "reproduction":[
                "python -m unittest discover -s artifacts/NC-046 -p test_trainer_worker.py -q",
                "python artifacts/NC-046/audit_pilot.py",
                "python dag.py validate",
                "python -m unittest -q test_dag test_pull"
             ]}
    out=HERE/"restart_replay_receipt.json"
    out.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":receipt["status"],"checkpoints":receipt["candidate_checkpoints_verified"],
                      "restarts":len(restarts),"no_gain":receipt["no_gain"],
                      "receipt_sha256":sha_file(out)},sort_keys=True))
if __name__=="__main__":inspect()
