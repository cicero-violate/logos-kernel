"""Freeze NC-062 implementation source, CPU training artifacts and science claims.

Dependencies: Python stdlib, Git, all named NC-062 files, pinned NC-001/NC-009/
NC-028/NC-061 sources, immutable champion. Never overwrites a frozen candidate.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import run_loop as r
import audit_results as audit

SOURCE_NAMES=[
 "worker_source.py","run_loop.py","evaluate_uncontaminated.py",
 "audit_results.py","audit_clean_eval.py","training_replay.py",
 "secondary_controls.py","test_loop.py","make_manifest.py","README.md"
]
EVIDENCE_NAMES=[
 "PROTOCOL.json","commitments.json","generation_trace.jsonl",
 "scoreboard_results.json","sealed_reveals.json","library_versions.json",
 "replay_audit.json","eval_v2_protocol.json","eval_v2_commitments.json",
 "eval_v2_candidate_freeze.json","eval_v2_results.json",
 "eval_v2_reveals.json","eval_v2_audit.json",
 "secondary_controls.json","training_replay.json"
]
def make():
    p=r.HERE
    all_names=SOURCE_NAMES+EVIDENCE_NAMES
    for f in all_names:
        r.require((p/f).is_file(),"MISSING_ARTIFACT:"+f)
    r.require(not (p/"MANIFEST.json").exists(),"CANDIDATE_MANIFEST_ALREADY_FROZEN")
    v1=json.loads((p/"scoreboard_results.json").read_text())
    v2=json.loads((p/"eval_v2_results.json").read_text())
    selfaudit=json.loads((p/"replay_audit.json").read_text())
    validaudit=json.loads((p/"eval_v2_audit.json").read_text())
    trains=json.loads((p/"training_replay.json").read_text())
    controls=json.loads((p/"secondary_controls.json").read_text())
    r.require(v1["completed_generations"]==v2["comparisons"][-1]["generation"]==3,
              "BAD_GENERATION_COUNT")
    r.require(v1["independent_discovery_count"]==v2["new_discoveries"]==0,
              "UNEXPECTED_NOVEL_DISCOVERY_CLAIM")
    r.require(all(x["scoreboard"]["verdict"]=="CONTAMINATED" for x in v1["scoreboard_comparisons"]),
              "V1_CONTAMINATION_UNDISCLOSED")
    r.require(all(x["verdict"]=="NO_IMPROVEMENT" for x in v2["comparisons"]),
              "UNJUSTIFIED_NO_IMPROVEMENT_CLAIM")
    r.require(v2["compounding"]["verdict"]=="NO_COMPOUNDING","COMPOUNDING_FABRICATION")
    r.require(all(x["training_overlap"]==0 for x in v2["score_rows"]),"V2_CONTAMINATION")
    r.require(selfaudit["checkpoint_count"]==12 and selfaudit["status"].startswith("IMPLEMENTER_AUDIT_PASS"),
              "V1_SELF_AUDIT_FAILED")
    r.require(validaudit["status"]=="PASS_IMPLEMENTER_NOT_INDEPENDENT" and
              len(validaudit["checked_scores"])==9,"CLEAN_AUDIT_FAILED")
    r.require(trains["all_training_state_sha_reproduced"] and
              trains["checkpoint_count"]==12,"TRAINING_REPLAY_FAILED")
    r.require(len(controls["rows"])==36 and
              all(x["cpu_matched_to_neural"] is False for x in controls["rows"]),
              "UNSUPPORTED_MATCHED_CPU_CLAIM")
    r.require(v2["heldout_goals_total"]==90 and len(v1["per_seed_generation_scores"])==9,
              "WRONG_DENOMINATOR")
    r.require(r.sha(r.CHAMPION)==v1["champion_sha256_unchanged"],"CHAMPION_DRIFT")
    source={f:{"sha256":r.sha(p/f),"bytes":(p/f).stat().st_size} for f in SOURCE_NAMES}
    evidence={f:{"sha256":r.sha(p/f),"bytes":(p/f).stat().st_size} for f in EVIDENCE_NAMES}
    checkpoints={}
    for gen in (0,*r.GENS):
        for seed in r.SEEDS:
            name=f"checkpoint_g{gen}_seed{seed}.json"
            q=p/name
            obj=json.loads(q.read_text())
            r.require(audit.calc_model_hash(obj["model_state"])==obj["model_state_sha256"],
                      "CHECKPOINT_SOURCE_INVALID")
            checkpoints[name]={"sha256":r.sha(q),"bytes":q.stat().st_size,
                               "model_state_sha256":obj["model_state_sha256"],
                               "generation":gen,"seed":seed}
    deps={str(q.relative_to(r.ROOT)) if q.is_relative_to(r.ROOT) else str(q):
          r.sha(q) for q in r.FROZEN_FILES}
    manifest={
      "schema":"logos.nc062.candidate/1",
      "task":"NC-062","owner":"turn-cb84af5cc3054e79af7776e1ac0af3c8",
      "status":"IMPLEMENTER_TESTED_AWAIT_INDEPENDENT_REVIEW",
      "candidate_type":"bounded supervised four-class CPU neural discovery-loop canary",
      "source_sha256":{k:v["sha256"] for k,v in source.items()},
      "source_files":source,"evidence_sha256":{k:v["sha256"] for k,v in evidence.items()},
      "evidence_files":evidence,
      "checkpoint_sha256":{k:v["sha256"] for k,v in checkpoints.items()},
      "checkpoint_files":checkpoints,"dependencies_sha256":deps,
      "scoreboard_source_sha256":r.sha(r.NC61/"scoreboard.py"),
      "sandbox_source_sha256":r.sha(r.NC28/"isolation.py"),
      "verified_initial_NC009_registry_sha256":r.sha(r.NC9/"operator_promotions.jsonl"),
      "source_champion_sha256":r.sha(r.CHAMPION),
      "initial_models":3,"candidate_training_generations":3,
      "seed_count":3,"neural_training_gradient_updates":72,
      "candidate_checkpoint_count":12,
      "training_discovery_goals":54,
      "first_evaluation_sealed_goals":90,
      "first_evaluation_verdict":"CONTAMINATED in all three generations; preserved, excluded from positive inference",
      "clean_v2_sealed_goals":90,
      "clean_v2_solve_rate":{"candidate":0,"baseline":0,"denominator":90},
      "science_result":"NO_DISCOVERY; NO_IMPROVEMENT x3; NO_COMPOUNDING",
      "generation_verdicts":[x["verdict"] for x in v2["comparisons"]],
      "independent_verdict":"NOT_YET_REVIEWED",
      "review_output_path":"artifacts/NC-062/independent_receipt.json",
      "positive_claims":[
         "NC-028 OS-contained Python neural inference and CPU SGD ran from frozen known-template starting state.",
         "All 12 checkpoint model SHA/state transitions reproduced by implementer.",
         "Exact finite 4-bit verification and counterexamples for 54 discovery goals.",
         "NC-061 score arithmetic on 90 fresh uncontaminated finite goals independently recomputed in parent.",
         "Candidate model always sees only public examples; all sealed private reveals stayed in trusted parent until after candidate scoring.",
         "NC-009 initial four programs verified and bound to hash-linked local library with no new discoveries.",
         "Original contaminated evaluation was preserved rather than silently excluded."
      ],
      "negative_controls":[
         "Unknown ISA tokens and action lengths refused",
         "Sample-fitting incorrect 4-bit program rejected by exhaustive 256-case verifier",
         "Fake model ranking refused",
         "Registry tampering and truncated chain rejected",
         "Seed, nonce, threshold, generator sealed-reveal tampering rejected",
         "All nine clean scoreboard evaluations exclude original-library and prior discovery-goal function hashes",
         "Trained successors compared with the exact no-library-training initial model on same sealed goal sets",
         "Secondary fixed/random/symbolic baselines included but labeled physically CPU UNMATCHED",
         "Model checkpoint SHA/lineage and all gradient updates reproduced"
      ],
      "limits":[
         "The model can emit only four previously accepted NC-009 operator templates, so no unseen program invention is possible with this architecture.",
         "There were zero newly discovered programs; all local library versions contain only four original entries.",
         "The V1 experiment had a real train/evaluation function collision and is CONTAMINATED; V2 is a separate prospective re-evaluation on fresh committed seeds and a preregistered disjoint selection algorithm.",
         "The clean V2 evaluation found 0/90 exact solves for trained and untrained candidate, not superiority or transfer.",
         "No improved model was promoted; original champion and NC-009 global registry were never written.",
         "NC-061 fixed catalog retention=1.0 is not independent measurement of neural old-skill retention.",
         "NC-028 child sandbox seccomp/Landlock is OS isolation for this bounded subprocess, not a separate UID/VM, proof of all absence of leakage or full host confinement.",
         "CPU process time is measured around sandboxed child; between-worker timing comparisons are subject to host load.",
         "Fixed/random/symbolic baselines are diagnostic and NOT matched on consumed CPU. No claim of compute superiority is supported.",
         "NC-028 sandbox worker class policy was supervised on known NC-009 programs, not program synthesis; counterexample filtering is not differentiable CEGIS.",
         "No program, weight or verifier has been independently reviewed for this candidate yet; implementer self-audit is not independent approval.",
         "No external network, ChatGPT API, autonomous service, champion promotion, AGI or unrestricted mathematical generalization occurred."
      ],
      "dependencies":[
        "Python 3.10+ standard library, Git, Linux Landlock ABI>=5 and libseccomp.so.2",
        "PyTorch CPU and NumPy in NC-028 sandbox; OMP/MKL/OPENBLAS threads fixed to one",
        "artifacts/NC-028/isolation.py; artifacts/NC-062/{worker_source.py,run_loop.py,evaluate_uncontaminated.py}",
        "artifacts/NC-001/canonical_isa.py, artifacts/NC-009/{promote.py,operator_promotions.jsonl}, artifacts/NC-061/scoreboard.py",
        "Read-only RLT champion /workspace/experiments/rlt_split_algorithms_ablation_v1/snapshots/current_unified_v3.pt"
      ],
      "reproduction":[
       "PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s artifacts/NC-062 -p test_loop.py -q",
       "PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-062/audit_results.py",
       "PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-062/audit_clean_eval.py",
       "PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-062/training_replay.py on clean duplicate checkout (creates new training_replay.json)",
       "PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-062/audit_clean_eval.py --full to reproduce all 18 sandboxed inference outputs",
       "python dag.py validate; python -m unittest -q test_dag test_pull",
       "In fresh checkout only: PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-062/run_loop.py (generates *new* secrets, not byte-identical evidence)"
      ]
    }
    out=p/"MANIFEST.json"
    r.write_once(out,manifest)
    alias=p/"CANDIDATE_MANIFEST.json"
    r.write_once(alias,manifest)
    r.require(r.sha(out)==r.sha(alias),"MANIFEST_ALIAS_DRIFT")
    print(json.dumps({"manifest_sha256":r.sha(out),"candidate_alias_sha256":r.sha(alias),
                      "checkpoint_count":len(checkpoints),"evidence_files":len(evidence),
                      "source_files":len(source),"clean_goals":90,
                      "new_discoveries":0},sort_keys=True))
if __name__=="__main__":make()
