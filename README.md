# Logos Kernel — append-only task DAG

Canonical path: `/workspace/experiments/logos-kernel/`

## Source of truth and ownership

- `instructions.jsonl` is a SHA-256 hash-chained, append-only event log. Existing events are never rewritten.
- `dag.py` validates the task DAG and appends atomic task, claim, review-request, review-claim, review-result, status and recovery events.
- **Every successfully appended event is immediately committed to local Git** as a ledger-only commit, while the append lock is held. Other staged and unstaged files are excluded.
- Before appending, the CLI verifies that committed history is an exact byte-prefix of the current ledger; the ledger must match Git HEAD or the append stops and requests `python dag.py sync`.
- If Git commit fails after an event is fsynced, the event **remains appended** and the CLI exits nonzero. Run `python dag.py sync`; never re-add the same event or rewrite the ledger.
- This is a **local commit**, not a Git push, cloud backup, or signed/remote tamper-proof record.
- `AGENTS.md` is the mandatory entry point and complete agent protocol.
- A claim records an **exclusive current owner**, not an alive worker or a job execution state.
- No scheduler, heartbeat, lease, auto-timeout, fake workers or automatic reassignment.
- The task DAG is acyclic; generated runtime instruction programs may contain CFG cycles.

## Pull-based ticket selection

**Target topology: three stateless ChatGPT tabs per repository; two isolated repositories, six tabs total.** The DAG does not launch tabs or authenticate identities.

Use `python dag.py pull --agent "$AGENT_ID"` (actual Event Lab identity required). The command atomically selects VERIFIER first from explicit review-request events, else IMPLEMENTER, else HOLD; see AGENTS.md and TAB_PROMPT.md. Owners publish `request-review` and reviewers finish with `review-result`. A Git record is not verifier authentication.

## Agent commands

```bash
cd /workspace/experiments/logos-kernel
python dag.py validate
python -m unittest -q test_dag
python -m unittest -q test_pull
test -n "$AGENT_ID" || exit 2
python dag.py pull --agent "$AGENT_ID"

# IMPLEMENTER after producing real, committed, pinned candidate evidence:
python dag.py request-review TASK_ID --owner "$AGENT_ID" --candidate-path artifacts/TASK_ID/CANDIDATE_MANIFEST.json

# VERIFIER after independent checks:
python dag.py review-result TASK_ID --reviewer "$AGENT_ID" --verdict HOLD --note "specific blocker"
# Or, with actual independently supported PASS and declared receipt:
python dag.py review-result TASK_ID --reviewer "$AGENT_ID" --verdict PASS --evidence artifacts/TASK_ID/DECLARED_RECEIPT.json
```

Replace placeholders with actual task IDs and existing artifact paths. Each tab follows the returned role; no manual `claim` first. A verified task's legitimate owner separately records final PASS. Admin `reopen`/`release` and `sync` are recovery operations; no forced release is authenticated by this script.

## Git commit rules and failure modes

- All ledger mutations (claim, release, pass, fail, block, reopen, add-task, priority-plan) automatically create a local Git commit for `instructions.jsonl` only. `python dag.py validate`, `ready`, and `list` are read-only and create no commits.
- Git user identity must be configured and `git` available. The command uses `git commit --only -- instructions.jsonl`, not `git add .` or an all-files commit.
- File locks serialize **DAG writers on this local filesystem**, including the commit. They do not serialize unrelated `git` processes or independent working copies. If Git fails, use `python dag.py sync` after resolving the cause.
- This mechanism commits the ledger only; implementation artifacts and receipts should be committed separately by their owners. There is no automatic push.
- The append is durable before its Git commit. There is no cross-filesystem transaction spanning both steps: a crash can leave a valid but uncommitted event. `sync` is the recovery command and adds no new event.

## Acceptance policy

Save full neural weights, connectivity/topology, configuration, checkpoint and source hashes. Separate neural reasoning, fixed ROM, host execution, verification oracle and program invention. Keep old champions unchanged until independent evidence, matched ablations, retention tests and rollback justify promotion.

Distinguish 320/320 local patterns, exhaustive four-bit MUL 256/256, sampled 8-bit MUL 4,217 cases and 8-bit exhaustive 65,536 pairs. Hash integrity is not scientific correctness or authentication.

## Existing research inputs and dependencies

- `/workspace/experiments/latticeforge_neural_isa_test/`
- `/workspace/experiments/rlt_split_algorithms_ablation_v1/`

This DAG repository requires **Python 3 standard library + Git**. Actual neural experiments require the dependencies of their original projects, typically PyTorch, architecture code and trained checkpoints. No files or models are copied here.
