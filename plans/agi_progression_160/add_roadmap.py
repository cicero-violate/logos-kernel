"""User-authorized 160-stage roadmap. Only dag.py mutates the canonical ledger.

69 existing milestones are reused; 91 missing capabilities are added. Running
again checks already-present task definitions instead of duplicating events.
"""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

# title | dependencies | concrete experiment and acceptance gate
SPEC="""
Operationally define broad general intelligence|NC-019,NC-020|Preregister domains, human comparison groups, permitted experience, autonomy levels and disqualifying failures; publish an AGI claim rubric before measuring candidates.
Audit contamination across the complete task lifecycle|AGI-001,NC-061|Trace every training, search, retrieval and evaluation input; planted leaked tasks must be detected and excluded from generalization claims.
Establish calibrated human reference distributions|AGI-001|Time and score appropriately skilled participants on consented or licensed tasks; publish sample sizes, uncertainty and experience matching without unsupported human parity claims.
Measure one checkpoint across a capability matrix|AGI-001,NC-015|Evaluate the same pinned model and allowed tools across every declared domain; prohibit selecting a different hidden specialist per evaluation cell.
Build an independent blind evaluation interface|AGI-002,NC-047|Evaluator owns fresh tasks and answers; attack tests must prevent candidate access to labels, scoring code and promotion authority.
Preregister statistical power and stopping rules|AGI-003,NC-029|Specify effect sizes, confidence intervals, multiple-comparison correction and stopping before runs; simulate null experiments to measure false discoveries.
Account for total experience and compute|AGI-006|Count pretraining, search, retries, retrieval, teacher cost and evaluator calls separately; compare equal allocated budgets and publish actual consumed budgets.
Connect kernel evidence to the local autonomy runtime|AGI-002,NC-063|Read version-pinned Logos Autonomy contracts and receipts; implement a kernel-side adapter and prove round-trip task/result identity without changing the other repository.
Recover integrated experiments after interruption|AGI-008|Kill bounded runs at every durable transition; resume without duplicate promotions, lost receipts or unaccounted compute and retain an executable rollback.
Demonstrate a bounded running research campaign|AGI-009|Run the integration on real local processes through multiple proposal/evaluate/reject-or-accept cycles; include live process evidence, stop control and final durable accounting.
Learn explicit relational role binding|NC-030,NC-026|Train subject-relation-object and variable-role binding; test unseen role permutations against bag-of-features and memorization controls.
Generalize under alpha-renaming and symbol replacement|AGI-011|Replace every symbol and register name at evaluation; preserve semantics and report failures on novel alphabets and swapped roles.
Track object identity under transformations|AGI-011|Use controlled synthetic scenes with occlusion, permutation and distractors; score persistent identity against nearest-position baselines.
Segment temporal events into reusable transitions|AGI-013,NC-076|Learn boundaries without future labels; score event segmentation and downstream prediction on unseen event compositions.
Reason across spatial reference frames|AGI-013|Transform viewpoint and coordinate frames while preserving relational answers; compare invariant and coordinate-memorizing controls.
Preserve units and dimensional types|AGI-012,NC-030|Reject dimensionally invalid operations and transfer valid formulas across unit systems; include adversarial scale and offset conversions.
Ground language into typed task contracts|AGI-011,NC-078|Map varied instructions into executable typed goals; test unseen paraphrases, negation and argument order with exact execution checks.
Execute grounded instructions with explicit semantics|AGI-017,NC-010|Translate contracts into programs and verify state effects; score execution separately from fluent explanation and reject underspecified effects.
Align symbolic and perceptual representations|AGI-013,AGI-017|Match descriptions and controlled observations under novel combinations; ablate each modality and require causal use rather than label correlation.
Audit compositional representation transfer|AGI-012,AGI-014,AGI-015,AGI-016,AGI-018,AGI-019|Hold out combinations and entire relation families; report transfer per family and compare identical training experience.
Build bounded task-relevant working memory|AGI-020,NC-027|Vary distractors, memory length and reasoning depth; measure accuracy per byte against context-only and random-memory controls.
Store replayable episodic experience|AGI-021,NC-028|Persist state/action/outcome traces with provenance; reconstruct episodes after restart and reject corrupted or mismatched identities.
Consolidate episodes into reusable semantic knowledge|AGI-022,NC-026|Extract abstractions that improve new tasks; compare raw episode retrieval, shuffled episodes and frozen-memory baselines.
Retrieve evidence with provenance and uncertainty|AGI-023,AGI-002|Return supporting sources and confidence with each retrieved claim; inject stale, conflicting and fabricated records and score detection.
Revise memory when evidence contradicts it|AGI-024|Update beliefs without erasing history; test reversed rules and conflicting observations with explicit posterior or version changes.
Measure amortized benefit from memory reuse|AGI-025,NC-065|Charge storage, retrieval and consolidation costs; show held-out benefit or NO_GAIN versus cold memory across multiple task batches.
Learn a stream of changing task distributions|AGI-020,NC-022|Run preregistered sequential tasks without resetting the learner; track forward transfer, backward transfer and forgetting after each stage.
Establish a persistent forgetting regression suite|AGI-027|Retest all earlier skills after every update with fresh instances; expose worst-family loss instead of hiding it in mean accuracy.
Reduce destructive interference between skills|AGI-028,NC-027|Compare shared, modular and regularized updates at matched parameter and compute budgets; require retained competence on earlier tasks.
Compress replay without losing rare capabilities|AGI-026,AGI-029|Compare compressed replay with full and random replay; preserve rare-case retention under a fixed memory budget.
Calibrate uncertainty across tasks and shifts|NC-061,AGI-006|Measure proper scoring rules and reliability curves on fresh tasks; include confident-wrong examples and distribution shifts.
Learn when to abstain or ask for assistance|AGI-031|Preregister error/coverage tradeoffs and assistance costs; require lower costly-error rates at matched coverage or report NO_GAIN.
Detect unfamiliar task distributions|AGI-020,AGI-031|Test unseen mechanisms and observation formats; measure false-alarm and miss rates without access to test-family labels.
Combine learned reasoning with checkable proof obligations|AGI-032,NC-023|Emit executable witnesses or explicit empirical scope; reject invalid proofs and distinguish verifier success from plausible prose.
Allocate search using expected information value|AGI-031,NC-064|Compare stopping and expansion policies at matched budgets; report verified solutions per total CPU and missed-solutions costs.
Separate causal heuristics from spurious shortcuts|AGI-033,NC-077|Reverse training correlations while retaining mechanisms; require intervention prediction to survive shortcut destruction.
Transfer abstractions through structural analogy|AGI-020,NC-059|Map novel relational structures between domains; ablate surface similarity and measure transfer against random analogies.
Revise reasoning from independent counterexamples|AGI-034,NC-051|Use withheld counterexamples to repair hypotheses; measure new-case improvement and retention without exposing final test labels.
Generalize reasoning to greater depth and length|AGI-037,NC-010|Evaluate longer dependency chains and programs than training; report failure probability by depth, not only aggregate success.
Control reasoning effort with calibrated metacognition|AGI-035,AGI-038,AGI-039|Allocate verification, recall and search according to uncertainty; beat fixed allocation on a preregistered cost-quality frontier or record NO_GAIN.
Plan with explicit state and goal models|AGI-018,NC-076|Solve held-out deterministic planning tasks with exact simulators; distinguish valid plans from plans that actually reach goals.
Detect divergence between plans and execution|AGI-041|Perturb actions and observations during execution; localize failure and replan without silently changing the goal.
Maintain beliefs under partial observability|AGI-042,NC-075|Hide state in controlled environments; evaluate belief calibration and planning against fully observed and memoryless controls.
Choose actions that resolve uncertainty|AGI-043,NC-070|Compare active experiments with passive and random observation at equal interaction cost; measure downstream goal completion.
Learn reusable hierarchical skills and options|AGI-044,NC-074|Compose independently verified skills on unseen goals; charge option discovery and demonstrate gains over flat planning.
Use tools through reversible capability boundaries|AGI-018,NC-047|Separate proposal, validation and execution; exercise rollback and denied operations in a local sandbox before any external action.
Generalize to unfamiliar tool schemas|AGI-046|Present unseen but documented local APIs; score correct arguments and effects, including malformed responses and version changes.
Recover from stale observations and tool failures|AGI-047,AGI-042|Inject timeouts, partial results and changed state; require bounded retries, idempotency and truthful failure reports.
Extend reliable multi-step task completion|AGI-045,AGI-048|Measure task completion versus human-duration and action count; include interruptions and repeated independent trials.
Stop and reallocate bounded campaigns reliably|AGI-049,AGI-010|Interrupt during planning, tool use and checkpoint writing; verify cancellation, resource release and consistent resumable state.
Acquire new primitives from few demonstrations|AGI-020,AGI-007,NC-078|Hold out primitive functions and vary demonstration counts; compare learning curves with nearest-neighbor and fresh-search baselines.
Compose newly acquired skills without retraining|AGI-051,NC-071|Combine new primitives with old operators on unseen tasks; measure compositional transfer and retained primitive correctness.
Generate procedural curricula with coverage guarantees|AGI-051,NC-053|Enumerate controlled families and track coverage; prevent duplicate samples from masquerading as new experience.
Hold out entire generative task families|AGI-053,AGI-002|Evaluator generates new families after learner freeze; report family-level results and contamination checks separately from instance holdouts.
Optimize prerequisite-aware skill acquisition|AGI-054,NC-050|Compare curriculum orderings at equal total teaching cost; retain negative transfer and failed prerequisites in the report.
Distill transferable skills across domains|AGI-055,AGI-030|Distill from verified multi-domain experience; compare frozen teachers, independent students and shared students without hidden extra data.
Explore tasks with sparse or delayed feedback|AGI-054,AGI-044|Vary reward delay and feedback availability; compare exploration policies under identical interaction and reset budgets.
Balance new learning with old capability retention|AGI-056,AGI-028|Require the declared retention floor at every curriculum step; publish accuracy-cost tradeoffs when acquisition conflicts with retention.
Transfer causal mechanisms between environments|AGI-052,NC-077|Change observation mapping and distractors while preserving mechanisms; evaluate intervention transfer and distinguish it from passive fitting.
Audit skill-acquisition efficiency across domains|AGI-058,AGI-059,AGI-007|Compare performance as a function of new experience and total compute on multiple unseen families; require gains across seeds rather than one showcase.
Recognize ambiguous or underspecified user goals|AGI-017,AGI-032|Generate tasks with missing constraints and multiple valid interpretations; score ambiguity detection without inventing user preferences.
Ask clarification questions with measurable value|AGI-061,AGI-044|Charge each question and compare completion with no-question and fixed-question policies; ask only when answers change decisions.
Maintain versioned user preferences and corrections|AGI-025,AGI-062|Update explicit preferences after corrections; preserve provenance and separate current preferences from inferred assumptions.
Satisfy interacting goals and hard constraints|AGI-018,AGI-063|Solve unseen multi-constraint tasks with executable validators; never count a high average reward as satisfying violated hard constraints.
Track roles, authority and information access|AGI-046,AGI-064|Test local simulated roles with conflicting instructions and private data; enforce explicit authority boundaries and deny unauthorized actions.
Coordinate bounded cooperative problem solving|AGI-065,AGI-045|Use simulated partners with complementary information; compare cooperation with single-policy baselines including communication cost.
Report epistemic and execution status truthfully|AGI-031,AGI-025|Compare claims to actual logs, artifacts and outcomes; penalize fabricated completion, unsupported certainty and concealed failures.
Produce explanations faithful to causal computation|AGI-067,NC-073|Intervene on claimed reasons and internal evidence; measure explanation faithfulness instead of relying on subjective fluency.
Expose and negotiate conflicting preferences|AGI-064,AGI-067|Use bounded preference-conflict scenarios; require transparent tradeoffs and user-specified resolution rather than hidden objective substitution.
Preserve effective human oversight and interruption|AGI-050,AGI-065,AGI-069|Test corrections, cancellation and denied actions throughout long tasks; require acceptance of authorized intervention and accurate recovery.
Attribute improvement to the proposed self-update|AGI-060,NC-062,NC-077|Use randomized paired update/no-update controls and ablations; separate gains from extra search, new data and evaluator changes.
Keep updater and evaluator authority disjoint|AGI-071,AGI-005,NC-047|Attempt changes to tests, budgets and receipts from candidate code; require rejection while legitimate bounded updates remain possible.
Meta-learn optimization on unseen task families|AGI-072,NC-067|Hold out entire optimization problems and compare fixed optimizers at equal total development and evaluation cost.
Propose small reversible source or architecture changes|AGI-073,NC-068|Confine mutations to declared interfaces, pin parent/child artifacts and verify rollback; no uncontrolled self-modification.
Reject corrupt or harmful candidate updates|AGI-074|Seed regressions, receipt tampering and resource violations; measure rejection sensitivity and false rejection of valid candidates.
Retain capabilities across successive self-updates|AGI-075,AGI-058|Run multiple generations against sealed retention tasks; stop promotion on any preregistered capability-floor breach.
Preserve task objectives through self-modification|AGI-076,AGI-064|Evaluate unchanged objectives and hard constraints after updates; detect proxy gaming and reward changes instead of treating them as gains.
Measure net improvement after all learning costs|AGI-077,AGI-007|Amortize training, failed proposals and verification across useful tasks; report break-even workload and NO_GAIN when costs dominate.
Detect plateaus and falsify unproductive hypotheses|AGI-078,AGI-006|Preregister stopping rules and alternative explanations; retain null results and redirect bounded budgets using evidence.
Demonstrate compounding transfer across generations|AGI-079,NC-065|Compare inherited, reset and shuffled libraries on new task families; require cross-generation benefit beyond accumulating test-specific patches.
Solve unfamiliar abstraction puzzles under controlled priors|AGI-060,AGI-039|Use evaluator-created rule families and restricted demonstrations; score exact solutions, learning curves and failures without benchmark memorization claims.
Repair unfamiliar software from executable specifications|AGI-081,AGI-048|Use isolated repositories with hidden tests and seeded regressions; verify functional repair, minimal collateral change and realistic human baselines.
Conduct bounded quantitative scientific investigations|AGI-081,AGI-059|Choose experiments, infer models and predict withheld outcomes in simulators; distinguish correct predictions, mechanism identification and formal proof.
Synthesize evidence from long conflicting documents|AGI-024,AGI-068|Use licensed or synthetic documents with contradictions and updates; require sourced answers and calibrated unresolved uncertainty.
Solve unfamiliar resource-allocation and scheduling problems|AGI-081,AGI-045|Generate constraints and costs after freeze; compare feasible utility and runtime with strong optimization baselines, including infeasible instances.
Compare breadth and depth against human references|AGI-082,AGI-083,AGI-084,AGI-085,AGI-003|Apply the preregistered multi-domain rubric to one pinned system; report every domain and uncertainty without cherry-picking.
Stress generalization under adversarial distribution change|AGI-086,AGI-033|Vary representations, goals, causal structure and incentives independently; quantify worst-group degradation and calibrated recovery.
Measure robust long-horizon general task performance|AGI-087,AGI-049|Estimate 50-percent and 90-percent completion horizons with confidence intervals across domains; distinguish reliability from occasional long successes.
Obtain independent replication on fresh task generators|AGI-088,AGI-005|Independent reviewers select fresh seeds and task families, reproduce costs and failures, and publish signed or otherwise attributed receipts with limitations.
Produce an auditable general-capability readiness dossier|AGI-089,AGI-070,AGI-080|Assemble source/checkpoint lineage, all domain scores, controls, risks, retention and rollback; preserve failed gates and prohibit automatic deployment approval.
Assess AGI evidence against the preregistered definition|AGI-090,AGI-001|An independent panel assesses breadth, depth, efficient new-skill acquisition, reliability and oversight against the original rubric; report NOT_ESTABLISHED whenever any required evidence is missing or weak. Completing this roadmap does not itself certify AGI.
""".strip()

SOURCES=[
 {"title":"On the Measure of Intelligence","url":"https://arxiv.org/abs/1911.01547v2","use":"Skill acquisition, priors and experience should be measured separately from trained task scores."},
 {"title":"Levels of AGI","url":"https://arxiv.org/abs/2311.02462v5","use":"Separate breadth, depth and autonomy in the operational rubric."},
 {"title":"Measuring AI Ability to Complete Long Software Tasks","url":"https://arxiv.org/abs/2503.14499v4","use":"Measure completion horizons and their uncertainty; software results alone do not establish generality."},
 {"title":"DreamCoder","url":"https://arxiv.org/abs/2006.08381v1","use":"Treat reusable program libraries and learned proposal policies as testable mechanisms."},
 {"title":"Towards Causal Representation Learning","url":"https://arxiv.org/abs/2102.11107v1","use":"Test causal representations through interventions and transfer, not association alone."},
]

def ledger():
    return [json.loads(line)["event"] for line in (ROOT/"instructions.jsonl").read_text().splitlines()]

def main():
    events=ledger();existing={e["id"]:e for e in events if e["kind"]=="task"}
    core=[f"NC-{i:03}" for i in list(range(16))+list(range(18,32))+list(range(40,79))]
    assert len(core)==69 and all(i in existing for i in core)
    specifications=SPEC.splitlines();assert len(specifications)==91
    new=[]
    for number,line in enumerate(specifications,1):
        title,deps,experiment=line.split("|")
        task_id=f"AGI-{number:03}"
        priority="P1" if number<=10 else "P2" if number<=60 else "P3"
        task={"kind":"task","id":task_id,"title":title,"status":"planned","priority":priority,
              "execution":"agent_requested","depends_on":deps.split(","),
              "instruction":[experiment,
                  "Before execution, pin the task-specific protocol, sources, budget, comparison baselines and stopping rules; use bounded local experiments and preserve existing agents' claims.",
                  "Reuse qualified predecessors. Record exact inputs, outputs, failure cases, resource costs and proof scope; do not equate a planned roadmap milestone with an implemented capability."],
              "acceptance":[experiment,
                  "A distinct independent reviewer reproduces the declared result from pinned artifacts, including negative controls. Empirical gain claims require at least three seeds or independently sampled task sets, per-family denominators, uncertainty and matched allocated budgets; report NO_GAIN or NOT_ESTABLISHED when appropriate.",
                  "Previous qualified capabilities meet their preregistered retention floors; raw candidate evidence and the independent receipt occupy separate declared paths."],
              "outputs":[f"artifacts/{task_id}/protocol.json",f"artifacts/{task_id}/results.json",f"artifacts/{task_id}/CANDIDATE_MANIFEST.json",f"artifacts/{task_id}/independent_verification.json"],
              "avoid":["No claim that this proposed ordering is a globally optimal or guaranteed route to AGI; rank reflects current hypotheses and measured blockers.",
                  "No self-approval, rewriting ledger history, implicit external permissions, unbounded search, hidden evaluator access, or champion promotion without independent evidence and required operator authorization."],
              "roadmap":{"id":"agi-progression-160-v1","branch_index":number,"basis":"User-requested proposed research progression; source-informed synthesis, not a source-established theorem."}}
        new.append(task)
    combined={**{i:existing[i] for i in core},**{t["id"]:t for t in new}}
    # Real dependency order, preserving original tasks and their exact contracts.
    order=[];remaining=list(combined)
    while remaining:
        eligible=next((i for i in remaining if all(d in order for d in combined[i]["depends_on"])),None)
        if eligible is None:raise RuntimeError("missing dependency or cycle in roadmap")
        order.append(eligible);remaining.remove(eligible)
    assert len(order)==160
    ranks={task_id:n for n,task_id in enumerate(order,1)}
    for task in new:task["roadmap"]["progression_stage"]=ranks[task["id"]]
    (HERE/"new_tasks.jsonl").write_text("".join(json.dumps(t,sort_keys=True)+"\n" for t in new))
    roadmap=[{"stage":ranks[i],"task_id":i,"title":combined[i]["title"],"depends_on":combined[i]["depends_on"],
              "reused_existing_task":i in core,"acceptance":combined[i]["acceptance"],"outputs":combined[i]["outputs"]} for i in order]
    (HERE/"roadmap.jsonl").write_text("".join(json.dumps(t,sort_keys=True)+"\n" for t in roadmap))
    (HERE/"sources.json").write_text(json.dumps({"accessed":"2026-11-06","sources":SOURCES,"scope":"References support design choices; the 160-stage roadmap and its priority are the assistant's synthesis, not a proven AGI recipe."},indent=2)+"\n")
    text=["# Proposed 160-stage progression toward AGI","",
          "69 existing kernel tasks are reused; 91 additional milestones extend them. This is a dependency-ordered research roadmap, not a guarantee, a global optimum, or an assertion that AGI has been achieved.","",
          "The immediate spine remains: exact baselines → sealed scoreboard → verified invention and reusable operators → causal intervention/transfer → measured generation-over-generation gain → bounded unattended campaigns. Later branches develop generality, memory, uncertainty, planning, new-skill learning, human oversight and independent evaluation.","",
          "Canonical task definitions live in `../../instructions.jsonl`. This readable index does not assign workers or mark tasks complete. Runtime priority keeps existing tasks in their current relative order; new milestones follow until evidence justifies a future priority snapshot.","",
          "Optional existing lexicon and speculative hardware branches remain in the kernel but are not prerequisites for this core 160-stage progression.","",
          "| Stage | Task | Milestone | Prerequisites |","|---:|---|---|---|"]
    for row in roadmap:text.append(f"| {row['stage']} | {row['task_id']} | {row['title']} | {', '.join(row['depends_on']) or '—'} |")
    text += ["","Research basis: "+"; ".join(f"[{s['title']}]({s['url']})" for s in SOURCES)+".","",
             "Every new task declares a separate independent review receipt. A scientific NO_GAIN result may close an experiment, but dependent capability claims still require the stated evidence; a passed task is not automatic proof of AGI."]
    (HERE/"README.md").write_text("\n".join(text)+"\n")
    # One dag.py invocation per event; its own append lock and ledger-only commit
    # serialize concurrent writers. Never write instructions.jsonl directly.
    for index,task in enumerate(new,1):
        current={e["id"]:e for e in ledger() if e["kind"]=="task"}
        if task["id"] in current:
            if current[task["id"]]!=task:raise RuntimeError("task ID collision: "+task["id"])
            continue
        path=HERE/"next_task.json";path.write_text(json.dumps(task)+"\n")
        result=subprocess.run(["rtk","proxy","python","dag.py","add-task","--file",str(path)],cwd=ROOT,capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stdout+result.stderr)
        if index%10==0 or index==len(new):print(f"Appended {index}/91 new roadmap tasks",flush=True)
    # Read the live state after all appends so claims/status changes by other
    # agents are preserved. Existing task order stays unchanged.
    for attempt in range(3):
        state=json.loads(subprocess.check_output(["rtk","proxy","python","dag.py","list"],cwd=ROOT,text=True))
        opened=[t for t in state if t["status"] in ("planned","working")]
        old=[t for t in opened if t["id"] not in {n["id"] for n in new}]
        added=sorted((t for t in opened if t["id"] in {n["id"] for n in new}),key=lambda t:ranks[t["id"]])
        priority={"kind":"priorities","reason":"User requested TOP 150+ AGI progression in JSONL. Add the 160-stage proposed roadmap (69 existing + 91 new); preserve existing relative priorities and all ownership. New stages are hypotheses with independent gates, not guaranteed AGI. Index: plans/agi_progression_160/roadmap.jsonl.",
                  "assignments":[{"id":t["id"],"priority":t["priority"],"rank":i} for i,t in enumerate(old+added,1)]}
        path=HERE/"priority_snapshot.json";path.write_text(json.dumps(priority,indent=2)+"\n")
        result=subprocess.run(["rtk","proxy","python","dag.py","priority-plan","--file",str(path)],cwd=ROOT,capture_output=True,text=True)
        if result.returncode==0:break
        if "priority plan must cover every open task exactly once" not in result.stdout+result.stderr:raise RuntimeError(result.stdout+result.stderr)
    else:raise RuntimeError("concurrent state changes prevented priority snapshot; tasks are safely appended")
    (HERE/"next_task.json").unlink(missing_ok=True)
    print(json.dumps({"roadmap_stages":160,"existing_reused":69,"new_appended":91,"canonical_ledger":str(ROOT/"instructions.jsonl"),"index":str(HERE/"README.md")}),flush=True)

if __name__=="__main__":main()
