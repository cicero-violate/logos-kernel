# AGENTS.md — Logos Kernel Append-Only DAG Protocol

## Identity, scope, and source of truth

This repository is a task DAG for progressive neural learning, operation composition, verification and promotion—not an agent runner.




- Repository: `/workspace/experiments/logos-kernel` (actual path: `/workspace/experiments/logos-kernel`)
- Instructions: `instructions.jsonl`, a hash-chained, **append-only** JSONL event log
- Only ledger writer: `python dag.py`; do not edit, truncate, reorder or reconstruct earlier records
- Task specification: full `kind=task` records in `instructions.jsonl` including `depends_on`, `instruction`, `acceptance`, `outputs`, `avoid`
- There is **no automatic worker** or heartbeat. Abandoned claims are only reassigned when a later `pull` finds their 2-hour lease expired.
- A task being *claimed* does **not** prove its agent is alive or working.

## Pull-based tickets: mandatory dispatch

**Deployment topology:** 3 stateless ChatGPT tabs dedicated to this repository (6 across the two isolated repositories). All tabs run the same pull command; no permanent verifier/implementer roles. All tabs use the same generic prompt. Every stateless turn creates its own agent ID; no Event Lab session, registration or per-tab setup is required. This DAG does not create browser tabs or authenticate session strings.



**Never decide the role manually.** The single assignment entry point for both repositories is:

```bash
# Stateless: create AGENT_ID once per turn with: python3 -c 'import uuid; print("turn-" + uuid.uuid4().hex)'
# Reuse that exact value for the rest of the turn. Never reuse an ID from the ledger or an earlier turn.
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

A verifier does not claim/release/reopen the implementation or append its final PASS. Ownership belongs to the ledger, not to a turn: after a closed review, `pull` gives the next turn a `takeover` event, once per verdict, as FINALIZER (PASS, then run `dag.py pass` with the returned verifier and evidence) or IMPLEMENTER (FAIL or HOLD, returned with the review note to fix). A turn cannot take over a task it just reviewed. A claim with no closed review is taken over only after its 2-hour lease expires, because it may still be in progress. A reviewer that disappears after claiming a review still blocks that task; nothing reassigns claimed reviews. Reviewer IDs remain unauthenticated strings at this software layer, not identity attestation. Incomplete provenance or a changed candidate requires HOLD.

### HOLD and boundaries

If no pending review is independently eligible and no task is claimable, report HOLD with the exact blocker, then stop working in this repository for this turn, however many times HOLD repeats: do not reopen or claim tasks, repair or harden code, add tests or tasks, or pull here again. A turn told to check another repository moves on to it; otherwise it ends. If an IMPLEMENTER pull returns a task whose `python dag.py list` entry shows `review_state` pending or claimed, the candidate is frozen: treat the turn as HOLD. Rework comes from `pull` after a closed FAIL or HOLD review; do not reopen tasks manually. A bug found while waiting is reported, not grounds to reopen. Review tickets have exclusive claims; a second agent cannot pull the same review. This is a synchronous ticket queue, **not** an automatic scheduler or a running verifier. Preserve original RLT champion, task dependencies, bounded scientific proof scope and other agents' staged/untracked files. Never fabricate a reviewer, force-release another claim, rewrite instructions.jsonl, promote a champion without independent evidence or assume Git hash-chain integrity authenticates reviewers.

Output: `Assigned role | Agent ID | Ticket | Candidate/hash | Changed files | Tests | Independent verdict | Blocker/next action`.

## Operational cycle (all three stateless tabs)

Each tab recovers Git and JSONL state every turn, creates its agent ID for the turn, validates ledger/tests, then runs the **only normal dispatch command**:

```bash
cd /workspace/experiments/logos-kernel
python dag.py validate
python -m unittest -q test_dag
python -m unittest -q test_pull
test -n "$AGENT_ID" || exit 2
python dag.py pull --agent "$AGENT_ID"
```

- VERIFIER: independently reproduce the returned candidate; record `review-result` with the task's declared receipt and exact verdict.
- IMPLEMENTER: implement the pulled task; commit owned evidence; record `request-review` with a complete pinned candidate.
- HOLD: report the exact blocker. Never bypass role assignment, fabricate a second identity or self-approve.
- The current owner records task PASS only after genuine independent reviewer PASS. `pull` assigns that final step, and any rework, to a later turn.
- Admin recovery (`sync`, `reopen`, `release`) is exceptional, not part of normal tab assignment. `--force` does not authenticate an operator. Claims are not renewed: the 2-hour lease counts from the claim or takeover.

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
