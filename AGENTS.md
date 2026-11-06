# AGENTS.md — Logos Kernel Append-Only DAG Protocol

## Identity, scope, and source of truth

This repository is a task DAG for progressive neural learning, operation composition, verification and promotion—not an agent runner.




- Repository: `/workspace/experiments/logos-kernel` (actual path: `/workspace/experiments/logos-kernel`)
- Instructions: `instructions.jsonl`, a hash-chained, **append-only** JSONL event log
- Only ledger writer: `python dag.py`; do not edit, truncate, reorder or reconstruct earlier records
- Task specification: full `kind=task` records in `instructions.jsonl` including `depends_on`, `instruction`, `acceptance`, `outputs`, `avoid`
- There is **no automatic worker**, implicit owner, heartbeat, lease, timer or abandoned-claim cleanup.
- A task being *claimed* does **not** prove its agent is alive or working.

## Pull-based tickets: mandatory dispatch

**Deployment topology:** 3 stateless ChatGPT tabs dedicated to this repository (6 across the two isolated repositories). All tabs run the same pull command; no permanent verifier/implementer roles. Each requires a distinct actual Event Lab session identity. This DAG does not create browser tabs or authenticate session strings.



**Never decide the role manually.** The single assignment entry point for both repositories is:

```bash
# AGENT_ID must come from the actual Event Lab runtime, never from a prompt.
python dag.py validate
python -m unittest -q test_dag
python -m unittest -q test_pull
python dag.py pull --agent "$AGENT_ID"
```

The JSONL writer selects inside one exclusive lock and commits the resulting event. Outcome is **VERIFIER** (pending independent audit first), **IMPLEMENTER** (genuinely owned or dependency-ready implementation), or **HOLD** (no eligible ticket). A repeat pull returns the existing assignment, without adding a duplicate event. It will never automatically pretend that a new browser tab is a new agent.

### IMPLEMENTER

Implement only the task returned by pull. Commit exact code, tests and a complete candidate manifest with source, config, checkpoint hashes, reproduction commands, negative controls, proof limitations and dependencies. Publish the review request through the same append-only JSONL writer:

```bash
python dag.py request-review TASK_ID --owner "$AGENT_ID" --candidate-path artifacts/TASK_ID/CANDIDATE_MANIFEST.json
```

The owner can request review of a claimed planned task, or of a blocked/failed task for which it was the last owner. An existing ad hoc file is NOT pulled until a review request event exists. The manifest's SHA-256 and current Git HEAD enter the ledger. The manifest alone does not freeze untracked or external dependencies. Do not mark your own task PASS.

### VERIFIER

Read the pulled candidate path, bound manifest hash and Git revision. Independently inspect original implementation and all referenced inputs, rerun claims, test counterexamples and record precise proof limits. Do not change the implementation and then sign it. Use an actual reviewer identity distinct from the implementer's; the known `chatgpt-session-<id>` alias equals `<id>` and is NOT independent. Write the receipt at the verification output path declared in that task's JSONL record. Then record the result:

```bash
python dag.py review-result TASK_ID --reviewer "$AGENT_ID" --verdict HOLD --note "missing evidence"
python dag.py review-result TASK_ID --reviewer "$AGENT_ID" --verdict FAIL --note "reproducible counterexample"
# For real independently supported PASS with the completed declared receipt:
python dag.py review-result TASK_ID --reviewer "$AGENT_ID" --verdict PASS --evidence artifacts/TASK_ID/DECLARED_RECEIPT.json
```

A verifier does not claim/release/reopen the implementation or append its final PASS. The legitimate implementer can PASS only after matching completed review (and an authorized reopen/claim if blocked). Reviewer IDs remain unauthenticated strings at this software layer, not identity attestation. Incomplete provenance or a changed candidate requires HOLD.

### HOLD and boundaries

If no pending review is independently eligible and no task is claimable, report HOLD with the exact blocker. Review tickets have exclusive claims; a second agent cannot pull the same review. This is a synchronous ticket queue, **not** an automatic scheduler or a running verifier. Preserve original RLT champion, task dependencies, bounded scientific proof scope and other agents' staged/untracked files. Never fabricate a reviewer, force-release another claim, rewrite instructions.jsonl, promote a champion without independent evidence or assume Git hash-chain integrity authenticates reviewers.

Output: `Assigned role | Actual session | Ticket | Candidate/hash | Changed files | Tests | Independent verdict | Blocker/next action`.

## At the beginning of each actual agent session

```bash
cd /workspace/experiments/logos-kernel
python dag.py validate
python -m unittest -q test_dag
python dag.py ready
python dag.py list
```

Stop and report a ledger-integrity failure; never "fix" it by rewriting historical JSONL. Read the complete task specification before execution.

## Exclusive task ownership — claims, not imaginary workers

Only a task with `status=planned`, `owner=null` and **all dependencies passed** appears in `ready`.

Set the shell variable `AGENT_ID` to the **actual identity assigned by your agent runtime**. No placeholder, fabricated worker name or guessed identity is permitted.

```bash
test -n "$AGENT_ID" || { echo 'Real AGENT_ID required'; exit 2; }
python dag.py claim NC-000 --owner "$AGENT_ID"
python dag.py list
```

Replace `NC-000` with a task **currently** listed by `ready`. An exclusive claim adds a `kind=claim` event and sets the current owner; its task status stays `planned`. A second claim on the same task is rejected atomically, even if two agents saw it ready.

If already claimed by your exact agent identity, resume the task rather than claiming twice. If claimed by another owner, do not take it. If you stop or hand off before completion, **release your claim**:

```bash
python dag.py release NC-000 --by "$AGENT_ID" --reason "handoff"
python dag.py ready
```

The owning agent should release a claim it no longer needs; then a different agent can claim it. A human/operator can clear a provably invalid, abandoned or mistaken claim:

```bash
test -n "$OPERATOR_ID" || { echo 'Real OPERATOR_ID required'; exit 2; }
python dag.py release NC-000 --by "$OPERATOR_ID" --force --reason "confirmed stale claim"
```

Forced releases append an audit record with the old owner and reason. **`--force` is an administrative protocol flag, not an authenticated access control.** Before release, check that the actual owner isn't working. There is deliberately no heartbeat, auto-expiration, auto-reassignment, synthetic worker or background execution.

## Execute and verify a claimed task

1. Inventory existing work, lock task contracts and preserve the original RLT champion/checkpoints.
2. Implement only the claimed task's instructions in the appropriate external project, with exact source/commit/config/checkpoint hashes.
3. Save full trained model topology, weights, architecture, vocabulary, optimizer and random state when training must resume; isolated neuron values are insufficient.
4. Run bounded acceptance tests, clean-process inference, held-out/counterexample tests and prior-skill retention gates.
5. Obtain **independent verification by a different, real agent**. The verifier produces a receipt at the task's declared repository-relative evidence path.
6. The owning agent records PASS only after actual independent verification. Keep every result traceable, then inspect the next `ready` task.

Verification receipts must record task ID, implementer, distinct verifier, exact repo/commit and checkpoint SHA-256, commands, environment, splits, seed/run information, measured inputs/denominators, failed cases, proof scope, neural-versus-symbolic execution boundary, result and limits.

```bash
test -n "$AGENT_ID" || exit 2
test -n "$VERIFIER_ID" || exit 2

# After a real independent verifier writes this file:
python dag.py pass NC-000 --owner "$AGENT_ID" --verifier "$VERIFIER_ID" --evidence artifacts/NC-000/baseline_receipt.json

# Alternative outcomes (do NOT run after PASS):
python dag.py fail NC-000 --owner "$AGENT_ID" --note "acceptance test failure"
python dag.py block NC-000 --owner "$AGENT_ID" --note "specific blocker"

# Operator reopens a previously failed/blocked task:
python dag.py reopen NC-000 --by "$OPERATOR_ID" --reason "blocker addressed"
```

**A file existing and a verifier ID being supplied do not constitute proof of independent verification.** The script checks event invariants and evidence-file existence, but not identity authentication or scientific validity.

New claims leave a task `planned` until one of `passed`, `failed`, or `blocked` is recorded. Each of these statuses clears ownership. Failed/blocked tasks require explicit `reopen` before a new claim. The historical `working` status is replayed for ledger compatibility only; new tasks use `claim`, not `start`.

## New instructions and concurrency

To add a new task, write a separate JSON task event (matching existing task fields; unique ID, already-existing dependencies) and append via:

```bash
python dag.py add-task --file /path/to/new_task_event.json
python dag.py validate
```

To reprioritize, prepare one full priority snapshot: kind=priorities, a reason, and assignments covering every open task exactly once. Each assignment has id, priority (P0 through P5), and a unique contiguous rank (1 is first). Append with: python dag.py priority-plan --file /path/to/plan.json. The original task event and dependencies stay immutable. The ready/list output is sorted by rank; dependency gates still control readiness. New tasks receive effective rank only after the next priority snapshot.

Never edit earlier JSONL, even to correct mistakes; use a new task and clearly document supersession. The task DAG is acyclic; **the programs being learned may contain loops in their runtime CFG**.

- The JSONL writer locks the local file and fsyncs each append; this does not serialize Git operations or coordinate across incompatible filesystems.
- `dag.py` automatically commits each append-only JSONL event in a **ledger-only local Git commit**; do not separately stage or commit `instructions.jsonl`.
- If an append reports Git failure after fsync, use `python dag.py sync` before adding another event; do not replay, rewrite, or discard the recorded event.
- Inspect Git status before edits; stage only your own **non-ledger** files. Coordinate Git operations with other agents.
- Do not hard-reset, restore, rebase away, force-push or hand-merge diverged ledger histories.
- Local Git+hash chain detect accidental editing, not malicious full-history rewrite. Use protected offsite storage for stronger integrity.

## Scientific boundaries — hard gates

- Local ISA 320/320 is not all reachable machine states; 4,217 sampled 8-bit MUL tests are not all 65,536 operand pairs.
- Distinguish the external arithmetic MLP, RLT attention arithmetic, Transformer instruction controller, hardcoded ROM, Python register routing, verification oracle and one jointly trained shared-weight Transformer.
- Executing an existing multiplication ROM is **not** inventing multiplication. For operation invention, the neural model must generate the program without receiving the complete program at evaluation.
- A math-template prediction must causally affect numeric results to support an ISA→math→execution claim.
- A verified symbolic operator is not equivalent to a verified neural implementation; preserve exact semantics and implementation mode.
- Checkpoint reuse is not continual learning; require training, transfer and retention evidence.
- Use matched-data/compute multi-seed ablations for architecture claims; never claim universal optimality or AGI from bounded benchmarks.
- No champion/default promotion before independent verification, preservation of old capabilities and demonstrated rollback.

## Source dependencies and output

- This instruction repository depends only on **Python 3 standard library and Git**.
- Existing specialist: `/workspace/experiments/latticeforge_neural_isa_test/`.
- Existing unified RLT interface: `/workspace/experiments/rlt_split_algorithms_ablation_v1/`. Validate actual code/weight paths under `NC-000`.
- Downstream neural projects depend on their own Python modules, model weights, datasets and PyTorch; there is no implied GPU.
- Report exact task ID, claim owner, changed paths, hashes, acceptance receipts, quantitative results, independent verifier, limits and next ready task.

**Start with `python dag.py validate` and `python dag.py pull --agent "$AGENT_ID"`; let the writer atomically assign VERIFIER, IMPLEMENTER or HOLD.**
