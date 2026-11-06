# Logos Kernel — append-only task DAG

Canonical path: `/workspace/experiments/logos-kernel/`

## Source of truth and ownership

- `instructions.jsonl` is a SHA-256 hash-chained, append-only event log. Existing events are never rewritten.
- `dag.py` validates the task DAG and appends atomic `task`, `claim`, `release`, `status` and `reopen` events.
- `AGENTS.md` is the mandatory entry point and complete agent protocol.
- A claim records an **exclusive current owner**, not an alive worker or a job execution state.
- No scheduler, heartbeat, lease, auto-timeout, fake workers or automatic reassignment.
- The task DAG is acyclic; generated runtime instruction programs may contain CFG cycles.

## Agent commands

```bash
cd /workspace/experiments/logos-kernel
python dag.py validate
python -m unittest -q test_dag
python dag.py ready
python dag.py list

# Set AGENT_ID from your actual connected agent identity.
test -n "$AGENT_ID" || exit 2
python dag.py claim NC-000 --owner "$AGENT_ID"
# If leaving a task unfinished:
python dag.py release NC-000 --by "$AGENT_ID" --reason "handoff"

# Only after an independent verifier wrote the receipt:
python dag.py pass NC-000 --owner "$AGENT_ID" --verifier "$VERIFIER_ID" --evidence artifacts/NC-000/baseline_receipt.json
# Alternatives to PASS:
python dag.py fail NC-000 --owner "$AGENT_ID" --note "failed acceptance"
python dag.py block NC-000 --owner "$AGENT_ID" --note "external blocker"
# Operator-only correction of invalid ownership:
python dag.py release NC-000 --by "$OPERATOR_ID" --force --reason "confirmed abandoned claim"
# Retry failed/blocked tasks after review:
python dag.py reopen NC-000 --by "$OPERATOR_ID" --reason "blocker resolved"

# Expand DAG: new_task_event.json must be a full kind=task object.
python dag.py add-task --file /path/to/new_task_event.json
```

These are **command examples for alternative outcomes**, not one sequence to execute. Task `NC-000` must be ready; `AGENT_ID`, `VERIFIER_ID` and `OPERATOR_ID` must refer to real identities. Claim ownership cannot itself prove that the agent is running or that PASS evidence is correct.

## Acceptance policy

Save full neural weights, connectivity/topology, configuration, checkpoint and source hashes. Separate neural reasoning, fixed ROM, host execution, verification oracle and program invention. Keep old champions unchanged until independent evidence, matched ablations, retention tests and rollback justify promotion.

Distinguish 320/320 local patterns, exhaustive four-bit MUL 256/256, sampled 8-bit MUL 4,217 cases and 8-bit exhaustive 65,536 pairs. Hash integrity is not scientific correctness or authentication.

## Existing research inputs and dependencies

- `/workspace/experiments/latticeforge_neural_isa_test/`
- `/workspace/experiments/rlt_split_algorithms_ablation_v1/`

This DAG repository requires **Python 3 standard library + Git**. Actual neural experiments require the dependencies of their original projects, typically PyTorch, architecture code and trained checkpoints. No files or models are copied here.
